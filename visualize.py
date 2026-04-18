"""
visualize.py — Poster Visualizations for Women's Protest News Corpus
=====================================================================
Generates 3 print-ready PNG charts for a feminist poster using the
outputs of topic_model.py and sentiment_analysis.py.

Charts produced
---------------
  1. sentiment_by_outlet.png
     Horizontal bar chart — mean VADER sentiment score per publisher,
     coloured by positive/negative, sorted most→least negative.
     Good for showing which outlets frame protest most negatively.

  2. topic_breakdown_by_event.png
     Grouped horizontal bar chart — top 8 global topics per event,
     showing % of articles in that event covering each topic.
     Good for showing how the news agenda shifted between events.

  3. protest_vs_control_sentiment.png
     Diverging bar / dot-plot — mean sentiment during protest weeks
     vs matched control weeks, per publisher.
     Good for showing how coverage tone changes when protests happen.

Colour palette — soft feminist
-------------------------------
  Primary:   #7B2D8B  (deep purple)
  Secondary: #C2185B  (raspberry)
  Accent:    #F8BBD9  (blush pink)
  Neutral:   #F3EAF7  (lavender cream)
  Text:      #2E1A3A  (dark plum)
  Negative:  #C2185B  (raspberry)
  Positive:  #7B8B2D  (muted olive green — legible against pink bg)

Dependencies
------------
    pip install matplotlib pandas pyarrow

Usage
-----
    python visualize.py                            # uses default input paths
    python visualize.py --sentiment path/to/articles_with_sentiment.csv
    python visualize.py --topics    path/to/articles_with_topics.parquet
    python visualize.py --out-dir   figures/
    python visualize.py --dpi 300                  # higher resolution for print
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# ❶  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SENTIMENT = os.path.join("analysis_output", "articles_with_sentiment.csv")
DEFAULT_TOPICS    = os.path.join("analysis_output", "articles_with_topics.parquet")
OUTPUT_DIR        = os.path.join("analysis_output", "figures")
DPI               = 300
TOP_N_TOPICS      = 8   # topics shown per event in chart 2

# ── Palette ──────────────────────────────────────────────────────────────────
DEEP_PURPLE   = "#7B2D8B"
RASPBERRY     = "#C2185B"
BLUSH         = "#F8BBD9"
LAVENDER_CREAM= "#F3EAF7"
DARK_PLUM     = "#2E1A3A"
OLIVE         = "#7B8B2D"
MID_PURPLE    = "#A855B5"
SOFT_PINK     = "#E91E8C"

PALETTE = [DEEP_PURPLE, RASPBERRY, MID_PURPLE, SOFT_PINK,
           "#9C4D9E", "#E57399", "#5C1A72", "#D81B60"]

def _base_style(fig, ax):
    """Apply shared poster styling to a figure/axes."""
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM)
    ax.tick_params(colors=DARK_PLUM, labelsize=11)
    ax.xaxis.label.set_color(DARK_PLUM)
    ax.yaxis.label.set_color(DARK_PLUM)
    ax.title.set_color(DARK_PLUM)
    for spine in ax.spines.values():
        spine.set_edgecolor(BLUSH)
    ax.grid(axis="x", color=BLUSH, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)


# ══════════════════════════════════════════════════════════════════════════════
# ❷  CHART 1 — SENTIMENT BY OUTLET
# ══════════════════════════════════════════════════════════════════════════════

def chart_sentiment_by_outlet(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    agg = (df.groupby("publisher")["sentiment_score"]
             .agg(mean="mean", n="count")
             .reset_index()
             .sort_values("mean"))

    fig, ax = plt.subplots(figsize=(10, max(5, len(agg) * 0.55)))
    _base_style(fig, ax)

    colors = [RASPBERRY if v < 0 else OLIVE for v in agg["mean"]]
    bars = ax.barh(agg["publisher"], agg["mean"], color=colors,
                   edgecolor="white", linewidth=0.5, height=0.65)

    # Value labels
    for bar, val in zip(bars, agg["mean"]):
        x = val + (0.01 if val >= 0 else -0.01)
        ha = "left" if val >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}", va="center", ha=ha,
                fontsize=9, color=DARK_PLUM, fontweight="bold")

    ax.axvline(0, color=DARK_PLUM, linewidth=1.2, linestyle="-")
    ax.set_xlabel("Mean VADER Sentiment Score  (−1 negative → +1 positive)", fontsize=11)
    ax.set_title("How Negative Is Each Outlet's Coverage?\nMean Sentiment Score by Publisher",
                 fontsize=14, fontweight="bold", pad=14)

    neg_patch = mpatches.Patch(color=RASPBERRY, label="Net negative")
    pos_patch = mpatches.Patch(color=OLIVE,     label="Net positive")
    ax.legend(handles=[neg_patch, pos_patch], loc="lower right",
              facecolor=LAVENDER_CREAM, edgecolor=BLUSH, labelcolor=DARK_PLUM)

    plt.tight_layout()
    path = os.path.join(out_dir, "sentiment_by_outlet.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ❸  CHART 2 — TOP TOPICS PER EVENT
# ══════════════════════════════════════════════════════════════════════════════

def chart_topics_by_event(df: pd.DataFrame, out_dir: str, dpi: int, top_n: int) -> None:
    events = sorted(df["event_label"].unique())
    n_events = len(events)

    # For each event, get the top N topics by share
    event_data = {}
    for event in events:
        sub = df[df["event_label"] == event]
        counts = (sub.groupby("topic_label").size()
                     .reset_index(name="n")
                     .assign(pct=lambda x: x["n"] / len(sub) * 100)
                     .sort_values("pct", ascending=False)
                     .head(top_n))
        event_data[event] = counts

    fig, axes = plt.subplots(1, n_events,
                             figsize=(9 * n_events, max(6, top_n * 0.6)),
                             sharey=False)
    if n_events == 1:
        axes = [axes]

    for ax, event in zip(axes, events):
        _base_style(fig, ax)
        data = event_data[event]
        # Truncate long topic labels
        labels = [l[:40] + "…" if len(l) > 40 else l for l in data["topic_label"]]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(data))]

        bars = ax.barh(labels[::-1], data["pct"].values[::-1],
                       color=colors[::-1], edgecolor="white",
                       linewidth=0.5, height=0.65)

        for bar, val in zip(bars, data["pct"].values[::-1]):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", ha="left",
                    fontsize=9, color=DARK_PLUM)

        # Clean up event label for title
        title = event.replace("_", " ")
        ax.set_title(f"{title}\nTop {top_n} Topics by Share of Coverage",
                     fontsize=13, fontweight="bold", pad=12, color=DARK_PLUM)
        ax.set_xlabel("% of articles in event window", fontsize=10)
        ax.set_xlim(0, data["pct"].max() * 1.25)

    fig.suptitle("What Filled the News Cycle?\nTop Topics Per Protest Event",
                 fontsize=16, fontweight="bold", color=DARK_PLUM, y=1.02)

    plt.tight_layout()
    path = os.path.join(out_dir, "topic_breakdown_by_event.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ❹  CHART 3 — SENTIMENT BY TOPIC
# ══════════════════════════════════════════════════════════════════════════════

def chart_sentiment_by_topic(df: pd.DataFrame, out_dir: str, dpi: int, top_n: int) -> None:
    """
    Horizontal bar chart of mean sentiment score per topic (top N by article count),
    coloured by valence. Shows which topics get framed most negatively across all outlets.
    Bubble size encodes number of articles so readers can weight the bars.
    """
    # Need both sentiment_score and topic_label in the same df
    if "topic_label" not in df.columns:
        print("    ⚠️  topic_label not found — skipping chart 3.")
        return

    agg = (df.groupby("topic_label")
             .agg(mean_sentiment=("sentiment_score", "mean"),
                  n_articles=("sentiment_score", "count"))
             .reset_index())

    # Filter out the outlier topic and topics with very few articles
    agg = agg[agg["topic_label"] != "outlier"]
    agg = agg[agg["n_articles"] >= 5]

    # Take top N topics by article count so the chart isn't overwhelming
    agg = (agg.nlargest(top_n, "n_articles")
              .sort_values("mean_sentiment"))

    fig, ax = plt.subplots(figsize=(11, max(6, len(agg) * 0.6)))
    _base_style(fig, ax)

    colors = [RASPBERRY if v < 0 else OLIVE for v in agg["mean_sentiment"]]

    # Normalise bubble sizes: smallest topic → 80pt², largest → 400pt²
    n     = agg["n_articles"].values
    sizes = 80 + (n - n.min()) / max(n.max() - n.min(), 1) * 320

    y = np.arange(len(agg))

    # Background bars (faint) to give a reference line feel
    ax.barh(y, agg["mean_sentiment"], color=colors, alpha=0.25,
            edgecolor="none", height=0.55)

    # Scatter dots sized by article count
    sc = ax.scatter(agg["mean_sentiment"], y, s=sizes,
                    c=colors, zorder=3, edgecolors=DARK_PLUM, linewidths=0.6)

    # Value + count labels
    for i, (val, n_art) in enumerate(zip(agg["mean_sentiment"], agg["n_articles"])):
        x_offset = 0.003 if val >= 0 else -0.003
        ha       = "left" if val >= 0 else "right"
        ax.text(val + x_offset, i,
                f"{val:+.3f}  (n={n_art})",
                va="center", ha=ha, fontsize=9,
                color=DARK_PLUM, fontweight="bold")

    # Clean up topic labels (BERTopic names can be long)
    labels = [l[:45] + "…" if len(l) > 45 else l for l in agg["topic_label"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color=DARK_PLUM, linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_xlabel("Mean VADER Sentiment Score  (−1 negative → +1 positive)", fontsize=11)
    ax.set_title(f"Which Topics Get Framed Most Negatively?\n"
                 f"Mean Sentiment by Topic (top {top_n} by article count)",
                 fontsize=14, fontweight="bold", pad=14)

    neg_patch = mpatches.Patch(color=RASPBERRY, alpha=0.7, label="Net negative")
    pos_patch = mpatches.Patch(color=OLIVE,     alpha=0.7, label="Net positive")
    ax.legend(handles=[neg_patch, pos_patch], loc="lower right",
              facecolor=LAVENDER_CREAM, edgecolor=BLUSH, labelcolor=DARK_PLUM)

    plt.tight_layout()
    path = os.path.join(out_dir, "sentiment_by_topic.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ❺  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", default=DEFAULT_SENTIMENT,
                        help="Path to articles_with_sentiment.csv (or .parquet)")
    parser.add_argument("--topics",    default=DEFAULT_TOPICS,
                        help="Path to articles_with_topics.parquet (or .csv)")
    parser.add_argument("--out-dir",   default=OUTPUT_DIR)
    parser.add_argument("--dpi",       type=int, default=DPI)
    parser.add_argument("--top-n-topics", type=int, default=TOP_N_TOPICS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load sentiment data
    print(f"Loading sentiment data from {args.sentiment} ...")
    if args.sentiment.endswith(".parquet"):
        sent_df = pd.read_parquet(args.sentiment)
    else:
        sent_df = pd.read_csv(args.sentiment)
    sent_df["is_control"] = sent_df["is_control"].astype(bool)
    print(f"  {len(sent_df):,} articles")

    # Load topic data (may be same file if sentiment was run on topic output)
    print(f"Loading topic data from {args.topics} ...")
    if args.topics.endswith(".parquet"):
        topic_df = pd.read_parquet(args.topics)
    else:
        topic_df = pd.read_csv(args.topics)
    topic_df["is_control"] = topic_df["is_control"].astype(bool)
    print(f"  {len(topic_df):,} articles")

    print(f"\nGenerating charts → {args.out_dir}/")

    print("\n[1/3] Sentiment by outlet ...")
    chart_sentiment_by_outlet(sent_df, args.out_dir, args.dpi)

    print("[2/3] Topic breakdown by event ...")
    chart_topics_by_event(topic_df, args.out_dir, args.dpi, args.top_n_topics)

    print("[3/3] Sentiment by topic ...")
    # Merge sentiment scores onto topic df (join on url if available, else index)
    if "sentiment_score" in topic_df.columns:
        merged_df = topic_df
    elif "url" in sent_df.columns and "url" in topic_df.columns:
        merged_df = topic_df.merge(
            sent_df[["url", "sentiment_score"]], on="url", how="left")
    else:
        merged_df = topic_df.copy()
        merged_df["sentiment_score"] = sent_df["sentiment_score"].values
    chart_sentiment_by_topic(merged_df, args.out_dir, args.dpi, args.top_n_topics)

    print(f"\n✅  All charts saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
