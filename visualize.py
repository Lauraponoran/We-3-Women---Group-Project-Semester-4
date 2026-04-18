import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# ❶  CONFIGURATION & STYLING
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SENTIMENT = os.path.join("analysis_output", "articles_with_sentiment.csv")
DEFAULT_TOPIC_INFO = os.path.join("analysis_output", "topic_info.csv")
OUTPUT_DIR        = os.path.join("analysis_output", "figures")
DPI               = 300
TOP_N_TOPICS      = 10 

# Palette - Soft Feminist
DEEP_PURPLE    = "#7B2D8B"
RASPBERRY      = "#C2185B"
BLUSH          = "#F8BBD9"
LAVENDER_CREAM = "#F3EAF7"
DARK_PLUM      = "#2E1A3A"
OLIVE          = "#7B8B2D"
FEMINIST_COLOUR  = "#FF4081" 
FEMINIST_OUTLINE = "#C2185B"
PALETTE = [DEEP_PURPLE, RASPBERRY, "#A855B5", "#E91E8C", "#9C4D9E", "#E57399"]

def clean_topic_name(name: str) -> str:
    """
    Cleans BERTopic labels into human-readable strings.
    Example: '35_ich_ruth_frauen' -> 'Ich, Ruth, Frauen'
    """
    if not isinstance(name, str): return "Unknown"
    parts = name.split('_')
    # Skip the leading ID number (e.g. '35')
    start_idx = 1 if parts[0].replace('-', '').isdigit() else 0
    words = parts[start_idx:start_idx+3] # Take the top 3 descriptive words
    return ", ".join(w.capitalize() for w in words)

def is_feminist(name: str) -> bool:
    """Flags a topic as feminist based on keywords found in the representation."""
    keywords = ['women', 'feminist', 'frauen', 'mujeres', 'gender', 'equality', 'rights', 'peace', 'familia', 'domestic']
    name_lower = str(name).lower()
    return any(k in name_lower for k in keywords)

def _base_style(fig, ax):
    """Apply consistent aesthetic to all charts."""
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM)
    ax.tick_params(colors=DARK_PLUM, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(BLUSH)
    ax.grid(axis="x", color=BLUSH, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

# ══════════════════════════════════════════════════════════════════════════════
# ❷  CHART 1 — SENTIMENT BY OUTLET
# ══════════════════════════════════════════════════════════════════════════════

def chart_sentiment_by_outlet(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    agg = df.groupby("publisher")["sentiment_score"].mean().sort_values().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    _base_style(fig, ax)
    
    colors = [RASPBERRY if v < 0 else OLIVE for v in agg["sentiment_score"]]
    ax.barh(agg["publisher"], agg["sentiment_score"], color=colors)
    ax.axvline(0, color=DARK_PLUM, linewidth=1)
    ax.set_title("How Negative Is Each Outlet's Coverage?\nMean Sentiment Score by Publisher", fontweight="bold", fontsize=14)
    ax.set_xlabel("Negative ← Sentiment Score → Positive")
    
    fig.savefig(os.path.join(out_dir, "sentiment_by_outlet.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# ❸  CHART 2 — TOP TOPICS BY EVENT
# ══════════════════════════════════════════════════════════════════════════════

def chart_topics_by_event(df: pd.DataFrame, out_dir: str, dpi: int, top_n: int) -> None:
    events = sorted(df["event_label"].dropna().unique())
    n_events = len(events)
    if n_events == 0: return

    fig, axes = plt.subplots(1, n_events, figsize=(8 * n_events, 8), sharey=False)
    if n_events == 1: axes = [axes]

    for ax, event in zip(axes, events):
        _base_style(fig, ax)
        sub = df[df["event_label"] == event]
        counts = sub["friendly_name"].value_counts(normalize=True) * 100
        
        # Rescue logic: Top N by volume + all feminist topics present in this specific event
        top_labels = counts.head(top_n).index.tolist()
        fem_labels = [l for l in counts.index if is_feminist(l) and l not in top_labels]
        plot_labels = top_labels + fem_labels
        data = counts.loc[plot_labels].sort_values()

        colors = [FEMINIST_COLOUR if is_feminist(l) else DEEP_PURPLE for l in data.index]
        ax.barh(data.index, data.values, color=colors)
        ax.set_title(f"Event: {event.replace('_', ' ').title()}", fontweight="bold")
        ax.set_xlabel("% of Articles in Event")

    fig.suptitle("What Filled the News Cycle?\nTop Topics Per Protest Event (Excluding Outliers)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "topic_breakdown_by_event.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# ❹  CHART 3 — SENTIMENT BY TOPIC
# ══════════════════════════════════════════════════════════════════════════════

def chart_sentiment_by_topic(df: pd.DataFrame, out_dir: str, dpi: int, top_n: int) -> None:
    agg = df.groupby("friendly_name").agg(
        mean_sentiment=("sentiment_score", "mean"),
        count=("sentiment_score", "count")
    ).reset_index()

    # Get Top N by volume + all feminist topics to ensure they are represented
    top_vols = agg.nlargest(top_n, "count")["friendly_name"].tolist()
    fem_vols = agg[agg["friendly_name"].apply(is_feminist)]["friendly_name"].tolist()
    keep_labels = list(set(top_vols + fem_vols))
    
    plot_df = agg[agg["friendly_name"].isin(keep_labels)].sort_values("mean_sentiment")

    fig, ax = plt.subplots(figsize=(11, 8))
    _base_style(fig, ax)
    
    colors = []
    for _, row in plot_df.iterrows():
        if is_feminist(row['friendly_name']): colors.append(FEMINIST_COLOUR)
        else: colors.append(RASPBERRY if row['mean_sentiment'] < 0 else OLIVE)

    # Size dots based on article count for a visual 'weight'
    sizes = np.clip(plot_df["count"] * 5, 60, 1200)
    ax.scatter(plot_df["mean_sentiment"], plot_df["friendly_name"], s=sizes, c=colors, edgecolors=DARK_PLUM, alpha=0.8, zorder=3)
    ax.axvline(0, color=DARK_PLUM, linestyle="--", alpha=0.5)
    
    ax.set_title("Which Topics Get Framed Most Negatively?\nMean Sentiment by Topic (Size = Volume)", fontweight="bold", fontsize=14)
    ax.set_xlabel("Mean VADER Sentiment Score")

    fig.savefig(os.path.join(out_dir, "sentiment_by_topic.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# ❺  CHART 4 — OVERALL TOPIC PIE
# ══════════════════════════════════════════════════════════════════════════════

def chart_topic_pie(df: pd.DataFrame, out_dir: str, dpi: int, top_n: int) -> None:
    counts = df["friendly_name"].value_counts()
    
    top = counts.head(top_n)
    rest = counts.iloc[top_n:]
    
    # Rescue feminist topics so they are never hidden in 'Other'
    rescued = rest[rest.index.map(is_feminist)]
    other_count = rest[~rest.index.map(is_feminist)].sum()
    
    slices = pd.concat([top, rescued])
    if other_count > 0:
        slices["Other Topics"] = other_count

    pie_colors = []
    explode = []
    for lbl in slices.index:
        if is_feminist(lbl):
            pie_colors.append(FEMINIST_COLOUR)
            explode.append(0.1) 
        elif lbl == "Other Topics":
            pie_colors.append("#C9B8D4")
            explode.append(0)
        else:
            pie_colors.append(PALETTE[len(pie_colors) % len(PALETTE)])
            explode.append(0)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(LAVENDER_CREAM)
    
    wedges, texts, autotexts = ax.pie(
        slices.values, 
        labels=slices.index, 
        autopct='%1.1f%%',
        startangle=140, 
        colors=pie_colors, 
        explode=explode,
        textprops={'color': DARK_PLUM, 'fontsize': 9},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    
    # Highlight feminist labels
    for i, lbl in enumerate(slices.index):
        if is_feminist(lbl):
            texts[i].set_fontweight('bold')
            texts[i].set_color(FEMINIST_OUTLINE)

    ax.set_title("Overall News Coverage by Topic (Excluding Outliers)", fontsize=16, fontweight="bold")
    
    fig.savefig(os.path.join(out_dir, "topic_pie_overall.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# ❻  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", default=DEFAULT_SENTIMENT)
    parser.add_argument("--topic-info", default=DEFAULT_TOPIC_INFO)
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading data from {args.sentiment}...")
    df = pd.read_csv(args.sentiment)
    info_df = pd.read_csv(args.topic_info)

    # CRITICAL: Filter out Outliers (Topic -1) immediately
    df = df[df['topic_id'] != -1].copy()
    info_df = info_df[info_df['Topic'] != -1].copy()

    # Map Friendly Names using the remaining topics in topic_info.csv
    name_map = {row['Topic']: clean_topic_name(row['Name']) for _, row in info_df.iterrows()}
    df['friendly_name'] = df['topic_id'].map(name_map).fillna("Unknown")

    print(f"Generating charts (Analyzing {len(df)} clustered articles)...")
    chart_sentiment_by_outlet(df, args.out_dir, 300)
    chart_topics_by_event(df, args.out_dir, 300, 8)
    chart_sentiment_by_topic(df, args.out_dir, 300, 12)
    chart_topic_pie(df, args.out_dir, 300, 10)

    print(f"Success! All figures saved to {args.out_dir}")

if __name__ == "__main__":
    main()