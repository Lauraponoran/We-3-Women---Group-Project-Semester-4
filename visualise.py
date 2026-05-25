"""
visualise.py — Charts for Women's Protest News Corpus Analysis
===============================================================
Generates publication-ready figures from the sentiment + topic outputs
produced by topic_model.py and sentiment_analysis.py.

Control week data is loaded from a separate directory (--control-dir) and
merged with the protest week data before plotting, so protest vs control
comparisons are always accurate.

Charts produced
---------------
  1.  sentiment_by_outlet.png              — mean sentiment per publisher
  2.  topic_breakdown_by_event.png         — top topics per protest event (bar)
  3.  topic_pie_{event}.png  (× N events)  — per-event topic donut charts
  4.  sentiment_by_topic.png               — mean sentiment × topic volume scatter
  5.  topic_pie_overall.png                — donut chart of global topic distribution
  6.  protest_vs_control.png               — protest vs control sentiment per publisher
  7.  language_breakdown.png               — EN/DE/FR/ES share per publisher

Usage
-----
    python visualise.py
    python visualise.py --sentiment path/to/articles_with_sentiment.csv
    python visualise.py --topic-info path/to/topic_info.csv
    python visualise.py --control-dir path/to/control          # folder with control CSVs
    python visualise.py --out-dir path/to/figures --top-n 12
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# ❶  CONFIGURATION & STYLING
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SENTIMENT  = os.path.join("analysis_output", "articles_with_sentiment.csv")
DEFAULT_TOPIC_INFO = os.path.join("analysis_output", "topic_info.csv")
DEFAULT_CONTROL    = os.path.join("control")   # sibling directory to news_output
OUTPUT_DIR         = os.path.join("analysis_output", "figures")
DPI                = 300

# ── Palette ───────────────────────────────────────────────────────────────────
DEEP_PURPLE      = "#7B2D8B"
RASPBERRY        = "#C2185B"
BLUSH            = "#F8BBD9"
LAVENDER_CREAM   = "#F3EAF7"
DARK_PLUM        = "#2E1A3A"
MUTED_TEAL       = "#4A8C8C"
# Feminist topics: distinct gold/amber — clearly different from raspberry in
# both colour and perceived brightness, distinguishable in greyscale too
FEMINIST_COLOUR  = "#E6A817"
FEMINIST_OUTLINE = "#B8860B"

PALETTE = [DEEP_PURPLE, "#A855B5", "#9C4D9E", "#6A1B9A", "#AB47BC", "#7E57C2"]


# ══════════════════════════════════════════════════════════════════════════════
# ❷  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

FEMINIST_KEYWORDS = [
    "women", "feminist", "frauen", "mujeres", "gender",
    "equality", "rights", "peace", "familia", "domestic",
]


def clean_topic_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return "Unknown"
    parts = name.split("_")
    start_idx = 1 if parts[0].replace("-", "").isdigit() else 0
    words = [w for w in parts[start_idx : start_idx + 3] if w]
    if not words:
        return name.capitalize()
    return ", ".join(w.capitalize() for w in words)


def is_feminist(name: str) -> bool:
    name_lower = str(name).lower()
    return any(k in name_lower for k in FEMINIST_KEYWORDS)


def _base_style(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM)
    ax.tick_params(colors=DARK_PLUM, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(BLUSH)
    ax.grid(axis="x", color=BLUSH, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: str, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _make_donut(
    ax: plt.Axes,
    slices: pd.Series,
    pie_colors: list[str],
    explode: list[float],
) -> list:
    """Draw a donut on ax; return wedges for legend building."""
    wedges, _, _ = ax.pie(
        slices.values,
        autopct="%1.1f%%",
        startangle=140,
        colors=pie_colors,
        explode=explode,
        pctdistance=0.82,
        textprops={"color": DARK_PLUM, "fontsize": 9, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5, "width": 0.5},
    )
    return wedges


def _slice_colors(slices: pd.Series) -> tuple[list[str], list[float]]:
    """Build pie_colors and explode lists for a slice series."""
    pie_colors  = []
    explode     = []
    non_fem_idx = 0
    for lbl in slices.index:
        if lbl == "All Other Topics":
            pie_colors.append("#D1C4E9")
            explode.append(0)
        elif is_feminist(lbl):
            pie_colors.append(FEMINIST_COLOUR)
            explode.append(0.08)
        else:
            pie_colors.append(PALETTE[non_fem_idx % len(PALETTE)])
            explode.append(0)
            non_fem_idx += 1
    return pie_colors, explode


def _build_slices(counts: pd.Series, top_n: int) -> pd.Series:
    """Return top_n slices + any feminist topics outside top_n + 'All Other Topics'."""
    top      = counts.head(top_n)
    rest     = counts.iloc[top_n:]
    rescued  = rest[rest.index.map(is_feminist)]
    other_n  = rest[~rest.index.map(is_feminist)].sum()
    slices   = pd.concat([top, rescued])
    if other_n > 0:
        slices["All Other Topics"] = other_n
    return slices


# ══════════════════════════════════════════════════════════════════════════════
# ❸  LOAD CONTROL DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_control_data(control_dir: str, name_map: dict) -> pd.DataFrame | None:
    """
    Load all CSVs from control_dir, apply friendly_name mapping, and return
    a combined dataframe tagged with is_control=True.

    Returns None if the directory doesn't exist or contains no CSV files.
    """
    if not os.path.isdir(control_dir):
        print(f"  ⚠️  Control dir not found: {control_dir} — protest vs control chart will be skipped.")
        return None

    csv_files = glob.glob(os.path.join(control_dir, "**", "*.csv"), recursive=True)
    csv_files = [f for f in csv_files if "incremental" not in f]

    if not csv_files:
        print(f"  ⚠️  No CSVs in {control_dir} — protest vs control chart will be skipped.")
        return None

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠️  Could not read {f}: {e}")

    if not dfs:
        return None

    ctrl = pd.concat(dfs, ignore_index=True)
    ctrl.drop_duplicates(subset=["url"] if "url" in ctrl.columns else None, inplace=True)
    ctrl["is_control"] = True

    # Normalise publisher column the same way topic_model.py does
    if "publisher" in ctrl.columns:
        ctrl["publisher"] = (
            ctrl["publisher"].astype(str)
            .str.extract(r"\.([A-Za-z0-9]+)")[0]
            .fillna(ctrl["publisher"].astype(str))
        )

    # Apply friendly_name if topic_id is present
    if "topic_id" in ctrl.columns and name_map:
        ctrl["friendly_name"] = ctrl["topic_id"].map(name_map).fillna("Unknown")

    print(f"  Loaded {len(ctrl):,} control articles from {control_dir}")
    return ctrl


# ══════════════════════════════════════════════════════════════════════════════
# ❹  CHART 1 — SENTIMENT BY OUTLET
# ══════════════════════════════════════════════════════════════════════════════

def chart_sentiment_by_outlet(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    agg = (
        df.groupby("publisher")["sentiment_score"]
        .mean()
        .sort_values()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, max(5, len(agg) * 0.45)))
    _base_style(fig, ax)

    colors = [RASPBERRY if v < 0 else MUTED_TEAL for v in agg["sentiment_score"]]
    ax.barh(agg["publisher"], agg["sentiment_score"], color=colors)
    ax.axvline(0, color=DARK_PLUM, linewidth=1)
    ax.set_title(
        "How Negative Is Each Outlet's Coverage?\nMean Sentiment Score by Publisher",
        fontweight="bold", fontsize=14, color=DARK_PLUM,
    )
    ax.set_xlabel("← More Negative    Sentiment Score    More Positive →", color=DARK_PLUM)

    neg_patch = mpatches.Patch(color=RASPBERRY,  label="Negative mean")
    pos_patch = mpatches.Patch(color=MUTED_TEAL, label="Positive mean")
    ax.legend(handles=[neg_patch, pos_patch], frameon=False, fontsize=9)

    _save(fig, os.path.join(out_dir, "sentiment_by_outlet.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❺  CHART 2 — TOP TOPICS BY EVENT  (bar chart)
# ══════════════════════════════════════════════════════════════════════════════

def chart_topics_by_event(
    df: pd.DataFrame, out_dir: str, dpi: int, top_n: int
) -> None:
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topics_by_event: 'friendly_name' missing — skipping.")
        return

    # Only protest weeks for this overview bar chart
    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    events = sorted(protest_df["event_label"].dropna().unique())
    if not events:
        return

    n_events = len(events)
    fig, axes = plt.subplots(1, n_events, figsize=(8 * n_events, 8), sharey=False)
    if n_events == 1:
        axes = [axes]

    for ax, event in zip(axes, events):
        _base_style(fig, ax)
        sub    = protest_df[protest_df["event_label"] == event]
        counts = sub["friendly_name"].value_counts(normalize=True) * 100

        top_labels = counts.head(top_n).index.tolist()
        fem_labels = [l for l in counts.index if is_feminist(l) and l not in top_labels]
        plot_labels = top_labels + fem_labels
        data   = counts.loc[plot_labels].sort_values()

        colors = [FEMINIST_COLOUR if is_feminist(l) else DEEP_PURPLE for l in data.index]
        ax.barh(data.index, data.values, color=colors)
        ax.set_title(event.replace("_", " ").title(), fontweight="bold", color=DARK_PLUM)
        ax.set_xlabel("% of Articles", color=DARK_PLUM)

    fig.suptitle(
        "Key Topics Per Protest Event (Protest Weeks Only)",
        fontsize=16, fontweight="bold", y=1.02, color=DARK_PLUM,
    )
    fem_patch = mpatches.Patch(color=FEMINIST_COLOUR, label="Feminist/protest topic")
    oth_patch = mpatches.Patch(color=DEEP_PURPLE,     label="Other topic")
    fig.legend(handles=[fem_patch, oth_patch], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=10)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "topic_breakdown_by_event.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❻  CHART 3 — PER-EVENT TOPIC PIE CHARTS  (one file per event)
# ══════════════════════════════════════════════════════════════════════════════

def chart_topic_pies_per_event(
    df: pd.DataFrame, out_dir: str, dpi: int, top_n: int
) -> None:
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topic_pies_per_event: 'friendly_name' missing — skipping.")
        return

    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    events = sorted(protest_df["event_label"].dropna().unique())

    for event in events:
        sub    = protest_df[protest_df["event_label"] == event]
        counts = sub["friendly_name"].value_counts()
        if counts.empty:
            continue

        slices              = _build_slices(counts, top_n)
        pie_colors, explode = _slice_colors(slices)

        fig, ax = plt.subplots(figsize=(13, 8))
        fig.patch.set_facecolor(LAVENDER_CREAM)

        wedges = _make_donut(ax, slices, pie_colors, explode)

        legend_labels = [
            f"{lbl} ({int(val):,} articles)"
            for lbl, val in zip(slices.index, slices.values)
        ]
        leg = ax.legend(
            wedges, legend_labels,
            title="Topics",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=9,
            frameon=False,
        )
        for i, text in enumerate(leg.get_texts()):
            if is_feminist(slices.index[i]):
                text.set_fontweight("bold")
                text.set_color(FEMINIST_OUTLINE)

        title = event.replace("_", " ").title()
        ax.set_title(
            f"Topic Distribution — {title}",
            fontsize=14, fontweight="bold", pad=20, color=DARK_PLUM,
        )

        safe = event.replace(" ", "_").replace("/", "-")
        _save(fig, os.path.join(out_dir, f"topic_pie_{safe}.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❼  CHART 4 — SENTIMENT BY TOPIC  (scatter)
# ══════════════════════════════════════════════════════════════════════════════

def chart_sentiment_by_topic(
    df: pd.DataFrame, out_dir: str, dpi: int, top_n: int
) -> None:
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_sentiment_by_topic: 'friendly_name' missing — skipping.")
        return

    agg = (
        df.groupby("friendly_name")["sentiment_score"]
        .agg(mean_sentiment="mean", count="count")
        .reset_index()
    )

    top_vol = agg.nlargest(top_n, "count")["friendly_name"].tolist()
    fem_vol = agg[agg["friendly_name"].apply(is_feminist)]["friendly_name"].tolist()
    keep    = list(set(top_vol + fem_vol))
    plot_df = agg[agg["friendly_name"].isin(keep)].sort_values("mean_sentiment")

    fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df) * 0.55)))
    _base_style(fig, ax)

    # Feminist/protest topics: gold diamond marker with dark outline
    # Negative other: raspberry circle
    # Positive other: teal circle
    # Shape + colour combination makes feminist topics unmistakable even in greyscale
    for _, row in plot_df.iterrows():
        if is_feminist(row["friendly_name"]):
            marker    = "D"   # diamond
            colour    = FEMINIST_COLOUR
            edge      = FEMINIST_OUTLINE
            edge_w    = 1.5
            z         = 4
        else:
            marker    = "o"
            colour    = RASPBERRY if row["mean_sentiment"] < 0 else MUTED_TEAL
            edge      = DARK_PLUM
            edge_w    = 0.5
            z         = 3
        size = np.clip(row["count"] * 5, 60, 1200)
        ax.scatter(
            row["mean_sentiment"], row["friendly_name"],
            s=size, c=colour, marker=marker,
            edgecolors=edge, linewidths=edge_w,
            alpha=0.88, zorder=z,
        )

    ax.axvline(0, color=DARK_PLUM, linestyle="--", alpha=0.5)
    ax.set_title(
        "Mean Sentiment by Topic\n(Dot size = article volume)",
        fontweight="bold", fontsize=14, color=DARK_PLUM,
    )
    ax.set_xlabel("← More Negative    Sentiment Score    More Positive →", color=DARK_PLUM)

    handles = [
        mpatches.Patch(color=FEMINIST_COLOUR, label="Feminist/protest topic (◆)"),
        mpatches.Patch(color=RASPBERRY,       label="Negative sentiment (other)"),
        mpatches.Patch(color=MUTED_TEAL,      label="Positive sentiment (other)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9)

    _save(fig, os.path.join(out_dir, "sentiment_by_topic.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❽  CHART 5 — OVERALL TOPIC PIE
# ══════════════════════════════════════════════════════════════════════════════

def chart_topic_pie(df: pd.DataFrame, out_dir: str, dpi: int, top_n: int) -> None:
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topic_pie: 'friendly_name' missing — skipping.")
        return

    counts              = df["friendly_name"].value_counts()
    slices              = _build_slices(counts, top_n)
    pie_colors, explode = _slice_colors(slices)

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(LAVENDER_CREAM)

    wedges = _make_donut(ax, slices, pie_colors, explode)

    legend_labels = [
        f"{lbl} ({int(val):,} articles)"
        for lbl, val in zip(slices.index, slices.values)
    ]
    leg = ax.legend(
        wedges, legend_labels,
        title="News Topics",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=10,
        frameon=False,
    )
    for i, text in enumerate(leg.get_texts()):
        if is_feminist(slices.index[i]):
            text.set_fontweight("bold")
            text.set_color(FEMINIST_OUTLINE)

    ax.set_title(
        "Distribution of Clustered News Topics (All Articles)",
        fontsize=16, fontweight="bold", pad=20, color=DARK_PLUM,
    )
    _save(fig, os.path.join(out_dir, "topic_pie_overall.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❾  CHART 6 — PROTEST vs CONTROL SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

def chart_protest_vs_control(
    df: pd.DataFrame, out_dir: str, dpi: int
) -> None:
    if "is_control" not in df.columns:
        print("  ⚠️  chart_protest_vs_control: 'is_control' column missing — skipping.")
        return
    if df["is_control"].nunique() < 2:
        print("  ⚠️  chart_protest_vs_control: only one value of is_control — "
              "check that control data was loaded. Skipping.")
        return

    agg = (
        df.groupby(["publisher", "is_control"])["sentiment_score"]
        .mean()
        .reset_index()
    )
    publishers = sorted(agg["publisher"].unique())
    x          = np.arange(len(publishers))
    width      = 0.35

    protest_vals = (
        agg[~agg["is_control"]]
        .set_index("publisher")["sentiment_score"]
        .reindex(publishers)
    )
    control_vals = (
        agg[agg["is_control"]]
        .set_index("publisher")["sentiment_score"]
        .reindex(publishers)
    )

    fig, ax = plt.subplots(figsize=(max(10, len(publishers) * 1.2), 6))
    _base_style(fig, ax)
    ax.grid(axis="y", color=BLUSH, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    ax.bar(x - width / 2, protest_vals, width,
           color=RASPBERRY, label="Protest week", alpha=0.9)
    ax.bar(x + width / 2, control_vals, width,
           color=DEEP_PURPLE, label="Control week", alpha=0.7)

    ax.axhline(0, color=DARK_PLUM, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(publishers, rotation=35, ha="right", color=DARK_PLUM)
    ax.set_ylabel("Mean Sentiment Score", color=DARK_PLUM)
    ax.set_title(
        "Sentiment During Protest Weeks vs Matched Control Weeks\nby Publisher",
        fontweight="bold", fontsize=14, color=DARK_PLUM,
    )
    ax.legend(frameon=False, fontsize=10)

    _save(fig, os.path.join(out_dir, "protest_vs_control.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❿  CHART 7 — LANGUAGE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

def chart_language_breakdown(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    if "language" not in df.columns:
        print("  ⚠️  chart_language_breakdown: 'language' column missing — skipping.")
        return

    lang_palette = {
        "EN":    DEEP_PURPLE,
        "DE":    RASPBERRY,
        "FR":    MUTED_TEAL,
        "ES":    "#E6A817",
        "OTHER": "#BDBDBD",
    }

    counts = df.groupby(["publisher", "language"]).size().reset_index(name="n")
    totals = counts.groupby("publisher")["n"].transform("sum")
    counts["pct"] = counts["n"] / totals * 100

    publishers = sorted(df["publisher"].unique())
    langs      = ["EN", "DE", "FR", "ES", "OTHER"]
    x          = np.arange(len(publishers))

    fig, ax = plt.subplots(figsize=(max(10, len(publishers) * 1.2), 6))
    _base_style(fig, ax)
    ax.grid(axis="y", color=BLUSH, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    bottom = np.zeros(len(publishers))
    for lang in langs:
        sub  = counts[counts["language"] == lang].set_index("publisher")
        vals = np.array([sub.loc[p, "pct"] if p in sub.index else 0.0 for p in publishers])
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom,
               color=lang_palette.get(lang, "#BDBDBD"), label=lang, alpha=0.9)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(publishers, rotation=35, ha="right", color=DARK_PLUM)
    ax.set_ylabel("% of Articles", color=DARK_PLUM)
    ax.set_ylim(0, 105)
    ax.set_title(
        "Article Language Composition by Publisher\n"
        "(non-English articles translated before sentiment scoring)",
        fontweight="bold", fontsize=13, color=DARK_PLUM,
    )
    ax.legend(title="Language", frameon=False, fontsize=9, bbox_to_anchor=(1, 1))

    _save(fig, os.path.join(out_dir, "language_breakdown.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓫  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment",   default=DEFAULT_SENTIMENT)
    parser.add_argument("--topic-info",  default=DEFAULT_TOPIC_INFO)
    parser.add_argument("--control-dir", default=DEFAULT_CONTROL,
                        help="Directory containing control week CSVs "
                             f"(default: {DEFAULT_CONTROL})")
    parser.add_argument("--out-dir",     default=OUTPUT_DIR)
    parser.add_argument("--top-n",       type=int, default=10,
                        help="Max topics in per-event and pie charts (default 10)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load protest data ─────────────────────────────────────────────────────
    print(f"Loading sentiment data from {args.sentiment} ...")
    df = pd.read_csv(args.sentiment)
    if "is_control" not in df.columns:
        df["is_control"] = False
    else:
        df["is_control"] = df["is_control"].astype(bool)
    print(f"  {len(df):,} articles")

    # ── Build friendly_name map ───────────────────────────────────────────────
    print(f"Loading topic info from {args.topic_info} ...")
    info_df  = pd.read_csv(args.topic_info)
    info_df  = info_df[info_df["Topic"] != -1].copy()
    name_map = {
        row["Topic"]: clean_topic_name(row["Name"])
        for _, row in info_df.iterrows()
    }

    # Filter outlier articles and attach friendly names
    df = df[df["topic_id"] != -1].copy()
    df["friendly_name"] = df["topic_id"].map(name_map).fillna("Unknown")

    # ── Load and merge control data ───────────────────────────────────────────
    ctrl = load_control_data(args.control_dir, name_map)
    if ctrl is not None:
        # Align columns — only keep cols present in both frames
        shared_cols = [c for c in df.columns if c in ctrl.columns]
        combined    = pd.concat([df[shared_cols], ctrl[shared_cols]], ignore_index=True)
    else:
        combined = df.copy()

    # ── Generate charts ───────────────────────────────────────────────────────
    print(f"\nGenerating charts → {args.out_dir}")

    # Charts that use protest weeks only
    chart_sentiment_by_outlet(df, args.out_dir, DPI)
    chart_topics_by_event(df, args.out_dir, DPI, args.top_n)
    chart_topic_pies_per_event(df, args.out_dir, DPI, args.top_n)
    chart_sentiment_by_topic(df, args.out_dir, DPI, args.top_n + 2)
    chart_topic_pie(df, args.out_dir, DPI, args.top_n)

    # Charts that need both protest + control
    chart_protest_vs_control(combined, args.out_dir, DPI)
    chart_language_breakdown(combined, args.out_dir, DPI)

    print(f"\n✅  All figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()