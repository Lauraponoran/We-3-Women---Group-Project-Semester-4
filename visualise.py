"""
visualise.py — Charts for Women's Protest News Corpus Analysis
===============================================================
Generates publication-ready figures from the sentiment + topic outputs
produced by topic_model.py and sentiment_analysis.py.

Charts produced
---------------
  1.  sentiment_by_outlet.png              — mean sentiment per publisher
  2.  topic_breakdown_by_event.png         — top topics per protest event (bar)
  3.  topic_pie_{event}.png  (× N events)  — per-event topic donut charts
  4.  topic_pie_overall.png                — donut: top N topics + feminist rescued
  5.  topic_pie_all_topics.png             — donut: every discovered topic, no collapsing
  6.  sentiment_by_topic.png               — mean sentiment × volume scatter
  7.  articles_per_event.png               — article counts per event with gap annotations
  8.  protest_vs_control.png               — protest vs control sentiment per publisher
  9.  sentiment_ideological_gap.png        — fairness: protest−control delta by ideology
  10. sentiment_by_event_type.png          — subgroup: single vs sustained events
  11. language_breakdown.png               — EN/DE/FR/ES share per publisher
  12. pipeline_diagram.png                 — workflow diagram for methods section
  13. bias_report.csv                      — representativeness statistics

Usage
-----
    python visualise.py
    python visualise.py --control-dir path/to/control
    python visualise.py --top-n 12
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# ❶  CONFIGURATION & STYLING
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SENTIMENT  = os.path.join("analysis_output", "articles_with_sentiment.csv")
DEFAULT_TOPIC_INFO = os.path.join("analysis_output", "topic_info.csv")
DEFAULT_CONTROL    = os.path.join("control")
OUTPUT_DIR         = os.path.join("analysis_output", "figures")
DPI                = 300

DEEP_PURPLE      = "#7B2D8B"
RASPBERRY        = "#C2185B"
BLUSH            = "#F8BBD9"
LAVENDER_CREAM   = "#F3EAF7"
DARK_PLUM        = "#2E1A3A"
MUTED_TEAL       = "#4A8C8C"
FEMINIST_COLOUR  = "#E6A817"
FEMINIST_OUTLINE = "#B8860B"
PALETTE = [DEEP_PURPLE, "#A855B5", "#9C4D9E", "#6A1B9A", "#AB47BC", "#7E57C2",
           "#5C6BC0", "#26A69A", "#66BB6A", "#FFA726", "#EC407A", "#8D6E63"]

# Ideological scores per publisher: -2 far-left … +2 far-right
# Based on AllSides / Ad Fontes Media indices where available
IDEOLOGY_SCORES: dict[str, float] = {
    "APNews": 0.0, "Reuters": 0.0, "VoiceOfAmerica": 0.2,
    "WashingtonPost": -0.8, "TheNewYorker": -1.0,
    "TheNation": -1.5, "TheIntercept": -1.5, "RollingStone": -1.0,
    "LATimes": -0.8, "BusinessInsider": -0.3,
    "CNBC": -0.2, "FoxNews": 1.5, "WashingtonTimes": 1.2,
    "FreeBeacon": 1.3, "TheGatewayPundit": 2.0,
    "BBC": -0.2, "TheGuardian": -1.0, "TheIndependent": -0.5,
    "EuronewsEN": 0.0, "iNews": -0.3, "DailyMail": 1.2,
    "TheTelegraph": 1.0, "TheSun": 1.3,
    "DW": 0.0, "SpiegelOnline": -0.5, "DieZeit": -0.5,
    "Tagesschau": 0.0, "FAZ": 0.5, "Taz": -1.5,
    "DerStandard": -0.5, "DiePresse": 0.5, "ORF": 0.0,
    "CBCNews": -0.3, "NationalPost": 0.8, "TheGlobeAndMail": 0.2,
    "LeMonde": -0.3, "LeFigaro": 0.8, "EuronewsFR": 0.0,
    "ElPais": -0.5, "ElMundo": 0.5, "ElDiario": -1.2,
    "LaVanguardia": 0.0, "ABC": 1.0, "Publico": -1.2,
    "IsraelNachrichten": 0.5,
}

# Human-readable notes explaining why certain events have fewer articles
EVENT_COVERAGE_NOTES: dict[str, str] = {
    "Womens_March_2017":              "CC-News sparse pre-2019",
    "International_Womens_Strike_2017": "CC-News sparse pre-2019",
    "MeToo_Protests_2017":            "CC-News sparse pre-2019",
    "IWD_2018_Global":                "CC-News sparse pre-2019",
    "Swiss_Womens_Strike_2019":       "Smaller national event",
    "Polish_Womens_Strike_2020":      "Strong EU coverage",
    "Sarah_Everard_Vigils_2021":      "Primarily UK coverage",
    "Roe_Leak_Protests_2022":         "High US publisher density",
    "Women_Life_Freedom_2022":        "Global coverage",
    "Israeli_Womens_Protests_2023":   "Primarily IL/EU coverage",
}

FEMINIST_KEYWORDS = [
    "women", "feminist", "frauen", "mujeres", "gender",
    "equality", "rights", "peace", "familia", "domestic",
]


# ══════════════════════════════════════════════════════════════════════════════
# ❷  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def clean_topic_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return "Unknown"
    parts = name.split("_")
    start_idx = 1 if parts[0].replace("-", "").isdigit() else 0
    words = [w for w in parts[start_idx : start_idx + 3] if w]
    return ", ".join(w.capitalize() for w in words) if words else name.capitalize()


def is_feminist(name: str) -> bool:
    return any(k in str(name).lower() for k in FEMINIST_KEYWORDS)


def _base_style(fig: plt.Figure, ax: plt.Axes, grid_axis: str = "x") -> None:
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM)
    ax.tick_params(colors=DARK_PLUM, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(BLUSH)
    ax.grid(axis=grid_axis, color=BLUSH, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: str, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _make_donut(ax, slices, pie_colors, explode):
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
    pie_colors, explode, non_fem_idx = [], [], 0
    for lbl in slices.index:
        if lbl == "All Other Topics":
            pie_colors.append("#D1C4E9"); explode.append(0)
        elif is_feminist(lbl):
            pie_colors.append(FEMINIST_COLOUR); explode.append(0.08)
        else:
            pie_colors.append(PALETTE[non_fem_idx % len(PALETTE)]); explode.append(0)
            non_fem_idx += 1
    return pie_colors, explode


def _build_slices(counts: pd.Series, top_n: int) -> pd.Series:
    top     = counts.head(top_n)
    rest    = counts.iloc[top_n:]
    rescued = rest[rest.index.map(is_feminist)]
    other_n = rest[~rest.index.map(is_feminist)].sum()
    slices  = pd.concat([top, rescued])
    if other_n > 0:
        slices["All Other Topics"] = other_n
    return slices


def _donut_with_legend(
    fig: plt.Figure, ax: plt.Axes,
    slices: pd.Series, pie_colors: list, explode: list,
    title: str,
) -> None:
    fig.patch.set_facecolor(LAVENDER_CREAM)
    wedges = _make_donut(ax, slices, pie_colors, explode)
    legend_labels = [f"{l} ({int(v):,})" for l, v in zip(slices.index, slices.values)]
    leg = ax.legend(wedges, legend_labels, title="Topics",
                    loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                    fontsize=9, frameon=False)
    for i, text in enumerate(leg.get_texts()):
        if is_feminist(slices.index[i]):
            text.set_fontweight("bold"); text.set_color(FEMINIST_OUTLINE)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=18, color=DARK_PLUM)


# ══════════════════════════════════════════════════════════════════════════════
# ❸  LOAD CONTROL DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_control_data(control_dir: str, name_map: dict) -> pd.DataFrame | None:
    if not os.path.isdir(control_dir):
        print(f"  ⚠️  Control dir not found: {control_dir} — protest vs control chart skipped.")
        return None
    csv_files = [f for f in glob.glob(os.path.join(control_dir, "**", "*.csv"), recursive=True)
                 if "incremental" not in f]
    if not csv_files:
        print(f"  ⚠️  No CSVs in {control_dir} — protest vs control chart skipped.")
        return None
    dfs = []
    for f in csv_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  ⚠️  Could not read {f}: {e}")
    if not dfs:
        return None
    ctrl = pd.concat(dfs, ignore_index=True)
    if "url" in ctrl.columns:
        ctrl.drop_duplicates(subset=["url"], inplace=True)
    ctrl["is_control"] = True
    if "publisher" in ctrl.columns:
        ctrl["publisher"] = (
            ctrl["publisher"].astype(str)
            .str.extract(r"\.([A-Za-z0-9]+)")[0]
            .fillna(ctrl["publisher"].astype(str))
        )
    if "topic_id" in ctrl.columns and name_map:
        ctrl["friendly_name"] = ctrl["topic_id"].map(name_map).fillna("Unknown")
    print(f"  Loaded {len(ctrl):,} control articles from {control_dir}")
    return ctrl


# ══════════════════════════════════════════════════════════════════════════════
# ❹  CHART 1 — SENTIMENT BY OUTLET
# ══════════════════════════════════════════════════════════════════════════════

def chart_sentiment_by_outlet(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    agg = df.groupby("publisher")["sentiment_score"].mean().sort_values().reset_index()
    fig, ax = plt.subplots(figsize=(10, max(5, len(agg) * 0.45)))
    _base_style(fig, ax)
    colors = [RASPBERRY if v < 0 else MUTED_TEAL for v in agg["sentiment_score"]]
    ax.barh(agg["publisher"], agg["sentiment_score"], color=colors)
    ax.axvline(0, color=DARK_PLUM, linewidth=1)
    ax.set_title("Mean Sentiment Score by Publisher",
                 fontweight="bold", fontsize=14, color=DARK_PLUM)
    ax.set_xlabel("← More Negative    Sentiment Score    More Positive →", color=DARK_PLUM)
    ax.legend(handles=[mpatches.Patch(color=RASPBERRY, label="Negative mean"),
                        mpatches.Patch(color=MUTED_TEAL, label="Positive mean")],
              frameon=False, fontsize=9)
    _save(fig, os.path.join(out_dir, "sentiment_by_outlet.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❺  CHART 2 — TOP TOPICS BY EVENT (bar)
# ══════════════════════════════════════════════════════════════════════════════

def chart_topics_by_event(df: pd.DataFrame, out_dir: str, dpi: int, top_n: int) -> None:
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topics_by_event: 'friendly_name' missing — skipping."); return
    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    events = sorted(protest_df["event_label"].dropna().unique())
    if not events:
        return
    fig, axes = plt.subplots(1, len(events), figsize=(8 * len(events), 8), sharey=False)
    if len(events) == 1:
        axes = [axes]
    for ax, event in zip(axes, events):
        _base_style(fig, ax)
        sub    = protest_df[protest_df["event_label"] == event]
        counts = sub["friendly_name"].value_counts(normalize=True) * 100
        top_l  = counts.head(top_n).index.tolist()
        fem_l  = [l for l in counts.index if is_feminist(l) and l not in top_l]
        data   = counts.loc[top_l + fem_l].sort_values()
        ax.barh(data.index, data.values,
                color=[FEMINIST_COLOUR if is_feminist(l) else DEEP_PURPLE for l in data.index])
        ax.set_title(event.replace("_", " ").title(), fontweight="bold", color=DARK_PLUM)
        ax.set_xlabel("% of Articles", color=DARK_PLUM)
    fig.suptitle("Key Topics Per Protest Event", fontsize=16, fontweight="bold",
                 y=1.02, color=DARK_PLUM)
    fig.legend(handles=[mpatches.Patch(color=FEMINIST_COLOUR, label="Feminist/protest"),
                         mpatches.Patch(color=DEEP_PURPLE, label="Other")],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=10)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "topic_breakdown_by_event.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❻  CHART 3 — PER-EVENT TOPIC PIES
# ══════════════════════════════════════════════════════════════════════════════

def chart_topic_pies_per_event(df: pd.DataFrame, out_dir: str, dpi: int, top_n: int) -> None:
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topic_pies_per_event: 'friendly_name' missing — skipping."); return
    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    for event in sorted(protest_df["event_label"].dropna().unique()):
        sub    = protest_df[protest_df["event_label"] == event]
        counts = sub["friendly_name"].value_counts()
        if counts.empty:
            continue
        slices = _build_slices(counts, top_n)
        colors, explode = _slice_colors(slices)
        fig, ax = plt.subplots(figsize=(13, 8))
        _donut_with_legend(fig, ax, slices, colors, explode,
                           f"Topic Distribution — {event.replace('_', ' ').title()}")
        safe = event.replace(" ", "_").replace("/", "-")
        _save(fig, os.path.join(out_dir, f"topic_pie_{safe}.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❼  CHART 4 — OVERALL TOPIC PIE (top N + rescued feminist)
# ══════════════════════════════════════════════════════════════════════════════

def chart_topic_pie(df: pd.DataFrame, out_dir: str, dpi: int, top_n: int) -> None:
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topic_pie: 'friendly_name' missing — skipping."); return
    counts  = df["friendly_name"].value_counts()
    slices  = _build_slices(counts, top_n)
    colors, explode = _slice_colors(slices)
    fig, ax = plt.subplots(figsize=(14, 9))
    _donut_with_legend(fig, ax, slices, colors, explode,
                       "Distribution of Clustered News Topics (All Articles)")
    _save(fig, os.path.join(out_dir, "topic_pie_overall.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❽  CHART 5 — ALL TOPICS PIE (no collapsing into "other")
# ══════════════════════════════════════════════════════════════════════════════

def chart_topic_pie_all(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    """Every discovered topic gets its own slice — nothing collapsed into 'other'.
    Labels are omitted from wedges to prevent overlap; the side legend carries all names."""
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topic_pie_all: 'friendly_name' missing — skipping."); return

    counts = df["friendly_name"].value_counts()
    # Assign colours: feminist topics in gold, others cycle through palette
    pie_colors, explode, non_fem_idx = [], [], 0
    for lbl in counts.index:
        if is_feminist(lbl):
            pie_colors.append(FEMINIST_COLOUR); explode.append(0.05)
        else:
            pie_colors.append(PALETTE[non_fem_idx % len(PALETTE)]); explode.append(0)
            non_fem_idx += 1

    # Wide figure to accommodate legend when there are many topics
    n = len(counts)
    fig_h = max(10, n * 0.28)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    fig.patch.set_facecolor(LAVENDER_CREAM)

    wedges, _, _ = ax.pie(
        counts.values,
        startangle=140,
        colors=pie_colors,
        explode=explode,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2, "width": 0.5},
        # No autopct labels — too crowded with many topics
    )
    legend_labels = [f"{l} ({int(v):,})" for l, v in zip(counts.index, counts.values)]
    leg = ax.legend(wedges, legend_labels,
                    title="All Topics",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1),
                    fontsize=8,
                    frameon=False)
    for i, text in enumerate(leg.get_texts()):
        if is_feminist(counts.index[i]):
            text.set_fontweight("bold"); text.set_color(FEMINIST_OUTLINE)

    ax.set_title("All Discovered Topics — Full Distribution\n(no collapsing; feminist topics in gold)",
                 fontsize=14, fontweight="bold", pad=20, color=DARK_PLUM)
    _save(fig, os.path.join(out_dir, "topic_pie_all_topics.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❾  CHART 6 — SENTIMENT BY TOPIC (scatter)
# ══════════════════════════════════════════════════════════════════════════════

def chart_sentiment_by_topic(df: pd.DataFrame, out_dir: str, dpi: int, top_n: int) -> None:
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_sentiment_by_topic: 'friendly_name' missing — skipping."); return
    agg = df.groupby("friendly_name")["sentiment_score"].agg(
        mean_sentiment="mean", count="count").reset_index()
    top_vol = agg.nlargest(top_n, "count")["friendly_name"].tolist()
    fem_vol = agg[agg["friendly_name"].apply(is_feminist)]["friendly_name"].tolist()
    plot_df = agg[agg["friendly_name"].isin(set(top_vol + fem_vol))].sort_values("mean_sentiment")

    fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df) * 0.55)))
    _base_style(fig, ax)
    for _, row in plot_df.iterrows():
        if is_feminist(row["friendly_name"]):
            m, c, ec, ew, z = "D", FEMINIST_COLOUR, FEMINIST_OUTLINE, 1.5, 4
        else:
            m = "o"
            c = RASPBERRY if row["mean_sentiment"] < 0 else MUTED_TEAL
            ec, ew, z = DARK_PLUM, 0.5, 3
        ax.scatter(row["mean_sentiment"], row["friendly_name"],
                   s=np.clip(row["count"] * 5, 60, 1200),
                   c=c, marker=m, edgecolors=ec, linewidths=ew, alpha=0.88, zorder=z)
    ax.axvline(0, color=DARK_PLUM, linestyle="--", alpha=0.5)
    ax.set_title("Mean Sentiment by Topic (dot size = volume)",
                 fontweight="bold", fontsize=14, color=DARK_PLUM)
    ax.set_xlabel("← More Negative    Sentiment Score    More Positive →", color=DARK_PLUM)
    ax.legend(handles=[
        mpatches.Patch(color=FEMINIST_COLOUR, label="Feminist/protest (◆)"),
        mpatches.Patch(color=RASPBERRY,       label="Negative (other)"),
        mpatches.Patch(color=MUTED_TEAL,      label="Positive (other)"),
    ], frameon=False, fontsize=9)
    _save(fig, os.path.join(out_dir, "sentiment_by_topic.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❿  CHART 7 — ARTICLES PER EVENT with coverage gap annotations
# ══════════════════════════════════════════════════════════════════════════════

def chart_articles_per_event(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    counts = protest_df.groupby("event_label").size().sort_values()

    fig, ax = plt.subplots(figsize=(11, max(5, len(counts) * 0.55)))
    _base_style(fig, ax)

    # Colour bars by volume: below median = raspberry, above = teal
    median_n = counts.median()
    colors   = [RASPBERRY if v < median_n else MUTED_TEAL for v in counts.values]
    bars     = ax.barh(counts.index, counts.values, color=colors, alpha=0.85)

    # Annotate each bar with article count + coverage note
    for bar, (event, n) in zip(bars, counts.items()):
        note = EVENT_COVERAGE_NOTES.get(event, "")
        label = f"  {n:,}"
        if note:
            label += f"  ·  {note}"
        ax.text(bar.get_width() + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=8.5, color=DARK_PLUM)

    ax.set_xlabel("Number of Articles", color=DARK_PLUM)
    ax.set_title("Articles Collected Per Protest Event\n"
                 "(bars below median in raspberry — see annotations for coverage gap reasons)",
                 fontweight="bold", fontsize=13, color=DARK_PLUM)
    ax.set_xlim(0, counts.max() * 1.55)

    ax.axvline(median_n, color=DARK_PLUM, linestyle=":", linewidth=1, alpha=0.6)
    ax.text(median_n + counts.max() * 0.01, -0.5, "median",
            fontsize=8, color=DARK_PLUM, alpha=0.7)

    _save(fig, os.path.join(out_dir, "articles_per_event.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓫  CHART 8 — PROTEST vs CONTROL SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

def chart_protest_vs_control(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    if "is_control" not in df.columns or df["is_control"].nunique() < 2:
        print("  ⚠️  chart_protest_vs_control: need both is_control values — skipping."); return
    agg = df.groupby(["publisher", "is_control"])["sentiment_score"].mean().reset_index()
    publishers = sorted(agg["publisher"].unique())
    x, w = np.arange(len(publishers)), 0.35
    pv = agg[~agg["is_control"]].set_index("publisher")["sentiment_score"].reindex(publishers)
    cv = agg[agg["is_control"]].set_index("publisher")["sentiment_score"].reindex(publishers)

    fig, ax = plt.subplots(figsize=(max(10, len(publishers) * 1.2), 6))
    _base_style(fig, ax, grid_axis="y")
    ax.bar(x - w/2, pv, w, color=RASPBERRY,     label="Protest week", alpha=0.9)
    ax.bar(x + w/2, cv, w, color=DEEP_PURPLE,   label="Control week", alpha=0.7)
    ax.axhline(0, color=DARK_PLUM, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(publishers, rotation=35, ha="right", color=DARK_PLUM)
    ax.set_ylabel("Mean Sentiment Score", color=DARK_PLUM)
    ax.set_title("Sentiment: Protest Weeks vs Matched Control Weeks by Publisher",
                 fontweight="bold", fontsize=14, color=DARK_PLUM)
    ax.legend(frameon=False, fontsize=10)
    _save(fig, os.path.join(out_dir, "protest_vs_control.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓬  CHART 9 — IDEOLOGICAL FAIRNESS GAP
# ══════════════════════════════════════════════════════════════════════════════

def chart_ideological_gap(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    """
    Fairness metric: (mean protest sentiment) − (mean control sentiment) per publisher,
    plotted against that publisher's ideological score.

    A negative gap means the outlet's coverage is more negative during protest weeks
    than during matched control weeks — a potential framing bias signal.
    Colouring by ideology score shows whether this gap correlates with political leaning.
    """
    if "is_control" not in df.columns or df["is_control"].nunique() < 2:
        print("  ⚠️  chart_ideological_gap: need both is_control values — skipping."); return

    agg = df.groupby(["publisher", "is_control"])["sentiment_score"].mean().unstack("is_control")
    agg.columns = ["protest", "control"]
    agg = agg.dropna()
    agg["gap"]      = agg["protest"] - agg["control"]
    agg["ideology"] = agg.index.map(lambda p: IDEOLOGY_SCORES.get(p, np.nan))
    agg = agg.dropna(subset=["ideology"])

    if agg.empty:
        print("  ⚠️  chart_ideological_gap: no publishers matched ideology scores — skipping.")
        return

    # Colour by ideology: left=teal, centre=purple, right=raspberry
    def _ideol_color(score: float) -> str:
        if score < -0.5:
            return MUTED_TEAL
        elif score > 0.5:
            return RASPBERRY
        else:
            return DEEP_PURPLE

    colors = [_ideol_color(s) for s in agg["ideology"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    _base_style(fig, ax, grid_axis="y")
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM)

    ax.scatter(agg["ideology"], agg["gap"], c=colors, s=100,
               edgecolors=DARK_PLUM, linewidths=0.6, alpha=0.88, zorder=3)

    # Label each point
    for pub, row in agg.iterrows():
        ax.annotate(pub, (row["ideology"], row["gap"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, color=DARK_PLUM, alpha=0.85)

    # Trend line
    if len(agg) >= 3:
        z = np.polyfit(agg["ideology"], agg["gap"], 1)
        p = np.poly1d(z)
        xs = np.linspace(agg["ideology"].min(), agg["ideology"].max(), 100)
        ax.plot(xs, p(xs), color=DARK_PLUM, linewidth=1.2,
                linestyle="--", alpha=0.5, label="Trend")

    ax.axhline(0, color=DARK_PLUM, linewidth=0.8, linestyle=":")
    ax.axvline(0, color=DARK_PLUM, linewidth=0.5, linestyle=":", alpha=0.4)
    ax.set_xlabel("← Left-leaning    Ideological Score    Right-leaning →", color=DARK_PLUM)
    ax.set_ylabel("Sentiment Gap (protest − control week)", color=DARK_PLUM)
    ax.set_title("Ideological Fairness Metric\n"
                 "Does political leaning predict more negative protest coverage?",
                 fontweight="bold", fontsize=13, color=DARK_PLUM)
    ax.legend(handles=[
        mpatches.Patch(color=MUTED_TEAL,   label="Left-leaning outlet"),
        mpatches.Patch(color=DEEP_PURPLE,  label="Centre outlet"),
        mpatches.Patch(color=RASPBERRY,    label="Right-leaning outlet"),
    ] + ([Line2D([0],[0], color=DARK_PLUM, linestyle="--", alpha=0.5, label="Trend")] if len(agg)>=3 else []),
    frameon=False, fontsize=9)

    _save(fig, os.path.join(out_dir, "sentiment_ideological_gap.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓭  CHART 10 — SENTIMENT BY EVENT TYPE (subgroup)
# ══════════════════════════════════════════════════════════════════════════════

def chart_sentiment_by_event_type(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    if "event_type" not in df.columns:
        print("  ⚠️  chart_sentiment_by_event_type: 'event_type' missing — skipping."); return

    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df

    agg = (protest_df.groupby(["event_type", "event_label"])["sentiment_score"]
           .mean().reset_index().sort_values(["event_type", "sentiment_score"]))

    event_types = sorted(agg["event_type"].unique())
    n = len(event_types)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    type_colors = {"single": DEEP_PURPLE, "sustained": RASPBERRY, "recurring": MUTED_TEAL}

    for ax, etype in zip(axes, event_types):
        _base_style(fig, ax)
        sub = agg[agg["event_type"] == etype].sort_values("sentiment_score")
        color = type_colors.get(etype, DEEP_PURPLE)
        ax.barh(sub["event_label"].str.replace("_", " "), sub["sentiment_score"],
                color=color, alpha=0.85)
        ax.axvline(0, color=DARK_PLUM, linewidth=0.8)
        ax.set_title(f"Event type: {etype.title()}", fontweight="bold", color=DARK_PLUM)
        ax.set_xlabel("Mean Sentiment", color=DARK_PLUM)

    fig.suptitle("Mean Sentiment by Event Type\n(single-day vs sustained protest events)",
                 fontsize=14, fontweight="bold", color=DARK_PLUM)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "sentiment_by_event_type.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓮  CHART 11 — LANGUAGE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

def chart_language_breakdown(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    if "language" not in df.columns:
        print("  ⚠️  chart_language_breakdown: 'language' missing — skipping."); return
    lang_palette = {"EN": DEEP_PURPLE, "DE": RASPBERRY, "FR": MUTED_TEAL,
                    "ES": "#E6A817", "OTHER": "#BDBDBD"}
    counts = df.groupby(["publisher","language"]).size().reset_index(name="n")
    counts["pct"] = counts["n"] / counts.groupby("publisher")["n"].transform("sum") * 100
    publishers = sorted(df["publisher"].unique())
    x = np.arange(len(publishers))
    fig, ax = plt.subplots(figsize=(max(10, len(publishers)*1.2), 6))
    _base_style(fig, ax, grid_axis="y")
    bottom = np.zeros(len(publishers))
    for lang in ["EN","DE","FR","ES","OTHER"]:
        sub  = counts[counts["language"]==lang].set_index("publisher")
        vals = np.array([sub.loc[p,"pct"] if p in sub.index else 0.0 for p in publishers])
        if vals.sum() == 0: continue
        ax.bar(x, vals, bottom=bottom, color=lang_palette.get(lang,"#BDBDBD"),
               label=lang, alpha=0.9)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(publishers, rotation=35, ha="right", color=DARK_PLUM)
    ax.set_ylabel("% of Articles", color=DARK_PLUM); ax.set_ylim(0, 105)
    ax.set_title("Article Language Composition by Publisher",
                 fontweight="bold", fontsize=13, color=DARK_PLUM)
    ax.legend(title="Language", frameon=False, fontsize=9, bbox_to_anchor=(1,1))
    _save(fig, os.path.join(out_dir, "language_breakdown.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓯  CHART 12 — PIPELINE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def chart_pipeline_diagram(out_dir: str, dpi: int) -> None:
    """Workflow diagram suitable for a paper methods section."""
    steps = [
        ("collect_articles\n_optimized.py", "Crawl CC-News\n45 publishers × 10 events\n+ matched control weeks"),
        ("build_corpus.py",                 "Merge per-event parquets\nDeduplicate by URL"),
        ("topic_model.py",                  "multilingual-e5-large embeddings\nBERTopic (HDBSCAN)\nOutlier reduction"),
        ("sentiment_analysis.py",           "Lingua language detection\nHelsinki opus-mt translation\nVADER scoring"),
        ("visualise.py",                    "12 publication figures\nbias_report.csv"),
    ]

    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM)
    ax.axis("off")

    n     = len(steps)
    xs    = np.linspace(0.08, 0.92, n)
    y_box = 0.5
    bw, bh = 0.14, 0.55

    for i, (script, desc) in enumerate(steps):
        x = xs[i]
        # Box
        rect = plt.Rectangle((x - bw/2, y_box - bh/2), bw, bh,
                              facecolor=DEEP_PURPLE, edgecolor=BLUSH,
                              linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        # Script name
        ax.text(x, y_box + 0.08, script, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white", zorder=3)
        # Description below
        ax.text(x, y_box - 0.18, desc, ha="center", va="center",
                fontsize=7, color=DARK_PLUM, zorder=3,
                bbox=dict(facecolor=LAVENDER_CREAM, edgecolor="none", pad=1))
        # Arrow to next
        if i < n - 1:
            ax.annotate("", xy=(xs[i+1] - bw/2 - 0.005, y_box),
                        xytext=(x + bw/2 + 0.005, y_box),
                        arrowprops=dict(arrowstyle="->", color=RASPBERRY,
                                        lw=1.8), zorder=4)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Analysis Pipeline", fontsize=14, fontweight="bold",
                 color=DARK_PLUM, pad=10)
    _save(fig, os.path.join(out_dir, "pipeline_diagram.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓰  BIAS REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_bias_report(df: pd.DataFrame, out_dir: str) -> None:
    """
    Outputs bias_report.csv with representativeness statistics:

    Section A — Coverage completeness
      Articles per publisher per event; publishers with < 5 articles in any
      window are flagged as potentially underrepresented.

    Section B — Language distribution
      Article counts and % per language overall and per publisher.
      Quantifies whether CC-News over-indexes on English.

    Section C — Protest / control balance
      Article counts in protest vs control windows overall and per publisher.
      Imbalance > 2:1 flagged.

    Section D — Outlier rate
      Raw (pre-reassignment) outlier % from topic_id_raw vs final topic_id.
      High raw outlier rate signals that the corpus may be too sparse for
      reliable topic modelling in some event subsets.

    Section E — Ideological balance
      Publisher counts per ideological bin (left / centre / right).
      An unbalanced set risks skewing aggregate sentiment findings.
    """
    os.makedirs(out_dir, exist_ok=True)
    sections = []

    # ── A: coverage completeness ──────────────────────────────────────────────
    cov = (df.groupby(["publisher","event_label"]).size()
             .reset_index(name="n_articles"))
    cov["flag_low_coverage"] = cov["n_articles"] < 5
    cov["section"] = "A_coverage_completeness"
    sections.append(cov)

    # ── B: language distribution ──────────────────────────────────────────────
    if "language" in df.columns:
        lang = df.groupby("language").size().reset_index(name="n_articles")
        lang["pct"] = (lang["n_articles"] / len(df) * 100).round(2)
        lang["section"] = "B_language_distribution"
        sections.append(lang)

        lang_pub = df.groupby(["publisher","language"]).size().reset_index(name="n_articles")
        lang_pub["section"] = "B_language_by_publisher"
        sections.append(lang_pub)

    # ── C: protest / control balance ──────────────────────────────────────────
    if "is_control" in df.columns:
        bal = df.groupby(["publisher","is_control"]).size().unstack("is_control").fillna(0)
        bal.columns = ["n_protest","n_control"] if False in bal.columns else bal.columns
        bal = bal.reset_index()
        if "True" in bal.columns or True in bal.columns:
            protest_col = False if False in bal.columns else "False"
            control_col = True  if True  in bal.columns else "True"
            bal.rename(columns={protest_col: "n_protest", control_col: "n_control"},
                       inplace=True, errors="ignore")
        if "n_protest" in bal.columns and "n_control" in bal.columns:
            bal["ratio_protest_to_control"] = (
                bal["n_protest"] / bal["n_control"].replace(0, np.nan)
            ).round(2)
            bal["flag_imbalanced"] = bal["ratio_protest_to_control"] > 2
        bal["section"] = "C_protest_control_balance"
        sections.append(bal)

    # ── D: outlier rate ───────────────────────────────────────────────────────
    if "topic_id_raw" in df.columns and "topic_id" in df.columns:
        n_total       = len(df)
        n_raw_outlier = (df["topic_id_raw"] == -1).sum()
        n_final_outlier = (df["topic_id"] == -1).sum()
        outlier_df = pd.DataFrame([{
            "n_total":              n_total,
            "n_raw_outliers":       int(n_raw_outlier),
            "pct_raw_outliers":     round(100 * n_raw_outlier / n_total, 2),
            "n_final_outliers":     int(n_final_outlier),
            "pct_final_outliers":   round(100 * n_final_outlier / n_total, 2),
            "n_reassigned":         int(n_raw_outlier - n_final_outlier),
            "section":              "D_outlier_rate",
        }])
        sections.append(outlier_df)

    # ── E: ideological balance ────────────────────────────────────────────────
    pub_counts = df["publisher"].value_counts().reset_index()
    pub_counts.columns = ["publisher", "n_articles"]
    pub_counts["ideology_score"] = pub_counts["publisher"].map(IDEOLOGY_SCORES)
    pub_counts["ideology_bin"] = pd.cut(
        pub_counts["ideology_score"],
        bins=[-3, -0.5, 0.5, 3],
        labels=["left", "centre", "right"],
    )
    pub_counts["section"] = "E_ideological_balance"
    sections.append(pub_counts)

    bin_summary = pub_counts.groupby("ideology_bin", observed=True).agg(
        n_publishers=("publisher","count"),
        total_articles=("n_articles","sum"),
    ).reset_index()
    bin_summary["section"] = "E_ideological_bin_summary"
    sections.append(bin_summary)

    # ── Combine and save ──────────────────────────────────────────────────────
    combined = pd.concat(sections, ignore_index=True)
    out_path = os.path.join(out_dir, "bias_report.csv")
    combined.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    # Print summary to stdout
    print("\n  ── Bias report summary ──────────────────────────────────────────")
    print(f"  Total articles        : {len(df):,}")
    if "language" in df.columns:
        lang_summary = df["language"].value_counts()
        print(f"  Language distribution : {lang_summary.to_dict()}")
    if "topic_id_raw" in df.columns:
        print(f"  Outlier rate (raw)    : {100*(df['topic_id_raw']==-1).mean():.1f}%")
        print(f"  Outlier rate (final)  : {100*(df['topic_id']==-1).mean():.1f}%")
    if "ideology_bin" in pub_counts.columns:
        ideol = pub_counts.groupby("ideology_bin", observed=True)["n_articles"].sum()
        print(f"  Articles by ideology  : {ideol.to_dict()}")
    low_cov = cov[cov["flag_low_coverage"]]
    if not low_cov.empty:
        print(f"  ⚠️  {len(low_cov)} publisher×event combinations have < 5 articles "
              f"(see Section A of bias_report.csv)")
    print("  ─────────────────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# ⓱  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment",   default=DEFAULT_SENTIMENT)
    parser.add_argument("--topic-info",  default=DEFAULT_TOPIC_INFO)
    parser.add_argument("--control-dir", default=DEFAULT_CONTROL)
    parser.add_argument("--out-dir",     default=OUTPUT_DIR)
    parser.add_argument("--top-n",       type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading sentiment data from {args.sentiment} ...")
    df = pd.read_csv(args.sentiment)
    df["is_control"] = df["is_control"].astype(bool) if "is_control" in df.columns else False
    print(f"  {len(df):,} articles")

    print(f"Loading topic info from {args.topic_info} ...")
    info_df  = pd.read_csv(args.topic_info)
    info_df  = info_df[info_df["Topic"] != -1].copy()
    name_map = {row["Topic"]: clean_topic_name(row["Name"]) for _, row in info_df.iterrows()}

    df = df[df["topic_id"] != -1].copy()
    df["friendly_name"] = df["topic_id"].map(name_map).fillna("Unknown")

    # Load and merge control data
    ctrl = load_control_data(args.control_dir, name_map)
    if ctrl is not None:
        shared  = [c for c in df.columns if c in ctrl.columns]
        combined = pd.concat([df[shared], ctrl[shared]], ignore_index=True)
    else:
        combined = df.copy()

    print(f"\nGenerating charts → {args.out_dir}")

    # Protest-only charts
    chart_sentiment_by_outlet(df, args.out_dir, DPI)
    chart_topics_by_event(df, args.out_dir, DPI, args.top_n)
    chart_topic_pies_per_event(df, args.out_dir, DPI, args.top_n)
    chart_topic_pie(df, args.out_dir, DPI, args.top_n)
    chart_topic_pie_all(df, args.out_dir, DPI)
    chart_sentiment_by_topic(df, args.out_dir, DPI, args.top_n + 2)
    chart_articles_per_event(df, args.out_dir, DPI)
    chart_sentiment_by_event_type(df, args.out_dir, DPI)

    # Charts requiring protest + control
    chart_protest_vs_control(combined, args.out_dir, DPI)
    chart_ideological_gap(combined, args.out_dir, DPI)
    chart_language_breakdown(combined, args.out_dir, DPI)

    # Static diagram
    chart_pipeline_diagram(args.out_dir, DPI)

    # Bias report
    print("\nGenerating bias report ...")
    generate_bias_report(combined, args.out_dir)

    print(f"\n✅  All outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
