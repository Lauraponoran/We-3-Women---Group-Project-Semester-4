import os
import glob
import pandas as pd
import nltk
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Ensure you have the NLTK stopwords downloaded smoothly
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Import your text stripping helper from topic_model.py
from topic_model import _strip_html, _canonical_label, build_topic_model

def main():
    control_dir = "control"
    output_dir = "analysis_output"
    
    # 1. Gather all control files recursively, EXCLUDING the incremental folders
    all_parquets = glob.glob(os.path.join(control_dir, "**", "*.parquet"), recursive=True)
    control_files = [f for f in all_parquets if "incremental" not in f]
    
    # Fallback to CSVs only if no base Parquets are found
    if not control_files:
        all_csvs = glob.glob(os.path.join(control_dir, "**", "*.csv"), recursive=True)
        control_files = [f for f in all_csvs if "incremental" not in f]

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
    
    # 3. Build a Multi-Language Stopword Filter to remove noisy slices (image_eea0aa.png)
    print("🧹 Compiling multi-language stopword filter...")
    languages = ['english', 'spanish', 'french', 'german', 'russian']
    combined_stopwords = set()
    for lang in languages:
        combined_stopwords.update(stopwords.words(lang))
        
    # Inject core functional text fragments and case-sensitive variants
    custom_noise = {
        'the', 'in', 'to', 'of', 'and', 'de', 'en', 'la', 'el', 'un', 'una', 
        'los', 'les', 'for', 'with', 'on', 'at', 'by', 'from', 'as', 'an', 'is'
    }
    combined_stopwords.update(custom_noise)
    vectorizer_model = CountVectorizer(stop_words=list(combined_stopwords), min_df=2)
    
    # 4. Train a completely independent BERTopic model ONLY on the control text
    print("\n🚀 Training a dedicated, stopword-free topic model for CONTROL WEEKS only...")
    docs_list = combined_control["text_temp"].tolist()
    
    try:
        # Try running via your project's helper if it supports the vectorizer argument
        control_model, control_topics = build_topic_model(
            docs=docs_list,
            embedding_model=embedding_model,
            min_topic_size=5,       
            top_n_words=10,
            reduce_topics=15,       
            reduce_outliers=True,
            label="isolated_control",
            vectorizer_model=vectorizer_model
        )
    except TypeError:
        # Fallback: Initialize natively if your build_topic_model signature is rigid
        print("  ℹ️ Native initialization fallback (applying clean vectorizer directly)...")
        embeddings = embedding_model.encode(docs_list, show_progress_bar=True)
        control_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=5,
            nr_topics=15
        )
        control_topics, _ = control_model.fit_transform(docs_list, embeddings=embeddings)
    
    # Map the unique control topic indices to normalized friendly names
    topic_info = control_model.get_topic_info()
    label_map = dict(zip(topic_info["Topic"], topic_info["Name"].map(_canonical_label)))
    
    combined_control["topic_id"] = control_topics
    combined_control["topic_label"] = combined_control["topic_id"].map(label_map).fillna("outlier")
    
    # Drop temp column before saving
    combined_control = combined_control.drop(columns=["text_temp"])
    
    # 5. Save control topic info metadata so visualise.py knows what they are
    control_info_path = os.path.join(output_dir, "control_topic_info.csv")
    topic_info.to_csv(control_info_path, index=False)
    print(f"📁 Saved isolated control topic keys to {control_info_path}")

    # 6. Split back and overwrite BOTH original file formats inside the control/ folders
    print("\n💾 Writing unique stopword-free topic IDs back to your control/ folder...")
    for file_path, df_group in combined_control.groupby("_source_file_path"):
        clean_df = df_group.drop(columns=["_source_file_path"])
        
        # Determine the base path without extension
        base_path, _ = os.path.splitext(file_path)
        csv_target = base_path + ".csv"
        parquet_target = base_path + ".parquet"
        
        # Save to Parquet format
        clean_df.to_parquet(parquet_target, index=False)
        print(f"   ✅ Updated Parquet: {parquet_target}")
        
        # Save to CSV format explicitly so visualise.py reads the identical data
        clean_df.to_csv(csv_target, index=False)
        print(f"   ✅ Updated CSV:     {csv_target}")

    print("\n🎉 Done! Control CSV and Parquet files are clean and perfectly synchronized.")

if __name__ == "__main__":
    main()