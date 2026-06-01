import os
import glob
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Import your text stripping helper from topic_model.py
from topic_model import _strip_html, _canonical_label, build_topic_model

def main():
    control_dir = "control"
    output_dir = "analysis_output"
    
    # 1. Gather all control files recursively (targeting only parquets to avoid duplicating CSV data)
    control_files = glob.glob(os.path.join(control_dir, "**", "*.parquet"), recursive=True)
    
    # Fallback to CSVs only if no Parquets are found
    if not control_files:
        control_files = glob.glob(os.path.join(control_dir, "**", "*.csv"), recursive=True)

    print(f"📋 Found {len(control_files)} control data file(s). Loading into an isolated dataframe...")
    
    # Load and combine temporarily to train a dedicated control topic model
    dfs = []
    for file_path in control_files:
        df_file = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_parquet(file_path)
        df_file["_source_file_path"] = file_path # Track file source to split them later
        dfs.append(df_file)
        
    combined_control = pd.concat(dfs, ignore_index=True)
    
    # 2. Extract and clean text for clustering using titles (matching your topic_model.py logic)
    title_col = "translated_title" if "translated_title" in combined_control.columns else "title"
    combined_control["text_temp"] = combined_control[title_col].fillna("").map(_strip_html)
    
    print(f"⚡ Loaded {len(combined_control):,} total isolated control articles.")
    print("⚡ Loading embedding model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 3. Train a completely independent BERTopic model ONLY on the control text
    print("\n🚀 Training a dedicated, non-overlapping topic model for CONTROL WEEKS only...")
    # Using your script's build helper with small adjustments for control data scale
    control_model, control_topics = build_topic_model(
        docs=combined_control["text_temp"].tolist(),
        embedding_model=embedding_model,
        min_topic_size=5,       # Adjusted for baseline control data scale
        top_n_words=10,
        reduce_topics=15,       # Keep it small and clean for clear donut pie charts
        reduce_outliers=True,
        label="isolated_control"
    )
    
    # Map the unique control topic indices to normalized friendly names
    topic_info = control_model.get_topic_info()
    label_map = dict(zip(topic_info["Topic"], topic_info["Name"].map(_canonical_label)))
    
    combined_control["topic_id"] = control_topics
    combined_control["topic_label"] = combined_control["topic_id"].map(label_map).fillna("outlier")
    
    # Drop temp column before saving
    combined_control = combined_control.drop(columns=["text_temp"])
    
    # 4. Save control topic info metadata so visualise.py knows what they are
    # We append/save it uniquely or merge it depending on visualizer expectations
    control_info_path = os.path.join(output_dir, "control_topic_info.csv")
    topic_info.to_csv(control_info_path, index=False)
    print(f"📁 Saved isolated control topic keys to {control_info_path}")

    # 5. Split back and overwrite the original files inside the control/ folder
    print("\n💾 Writing unique topic IDs back to your control/ folder...")
    for file_path, df_group in combined_control.groupby("_source_file_path"):
        clean_df = df_group.drop(columns=["_source_file_path"])
        
        if file_path.endswith(".csv"):
            clean_df.to_csv(file_path, index=False)
        else:
            clean_df.to_parquet(file_path, index=False)
        print(f"   ✅ Updated: {file_path}")

    print("\n🎉 Done! Control data is now fully topic-coded without ANY overlap with protest weeks.")

if __name__ == "__main__":
    main()