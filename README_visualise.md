"""
visualise.py — Charts for Women's Protest News Corpus Analysis
===============================================================
Generates publication-ready figures from the sentiment + topic outputs
produced by topic_model.py and sentiment_analysis.py.

Charts produced
---------------
  1.  sentiment_by_outlet.png              — mean sentiment per publisher
  2.  topic_breakdown_by_event.png         — top topics per protest event (bar, NORMALISED)
  3.  topic_pie_{event}.png  (× N events)  — per-event topic donut charts (NORMALISED)
  4.  topic_pie_overall.png                — donut: top N topics + feminist rescued (NORMALISED)
  5.  topic_pie_all_topics.png             — donut: every discovered topic, no collapsing (absolute)
  6.  sentiment_by_topic.png               — mean sentiment × volume scatter
  7.  articles_per_event.png               — article counts per event with gap annotations
  8.  protest_vs_control.png               — protest vs control sentiment per publisher
  9.  sentiment_ideological_gap.png        — fairness: protest−control delta by ideology
  10. sentiment_by_event_type.png          — subgroup: single vs sustained events
  11. language_breakdown.png               — EN/DE/FR/ES share per publisher
  12. pipeline_diagram.png                 — workflow diagram for methods section
  13. bias_report.csv                      — representativeness statistics

  NEW — Demographic & Fairness suite
  -----------------------------------
  14. demographic_distribution.png         — article volume by outlet ideology bin + language group
  15. fairness_metric_comparison.png       — per-publisher normalised sentiment gap vs baseline
  16. model_performance.png               — topic model coherence / coverage diagnostics
  17. subgroup_comparison.png             — sentiment broken down by event × publisher ideology
  18. confusion_matrix_sentiment.png      — cross-tab predicted vs ground-truth sentiment labels
  19. workflow_diagram.png                — extended pipeline diagram (matches methods section prose)

  Normalisation
  -------------
  All topic-distribution charts (2–5, 17) operate on proportions rather than raw
  counts.  Within each event window the share of articles per topic is computed,
  then averaged across events so that large events (e.g. Women Life Freedom 2022,
  ~1,200 articles) do not dominate events with ~80 articles.  Chart 5
  (topic_pie_all_topics.png) intentionally retains raw counts — it is a topic-model
  diagnostic, not a cross-event comparison.

  Colour & legibility changes
  ---------------------------
  - PALETTE now uses 12 perceptually-distinct hue families so adjacent slices
    are always visually separable at 300 DPI.
  - Feminist/protest topics use FEMINIST_COLOUR fill (amber) + FEMINIST_OUTLINE
    edge (dark amber) on all charts, plus a ★ prefix and bold amber text in
    every legend.  In donut charts feminist wedges are additionally pulled out
    (explode) and given a circular legend swatch to distinguish them at a glance.
  - Chart titles contain only the human-readable title.  All methodological
    notes (n=, normalisation method, colour key) are placed as a subnote via
    fig.text() below the chart area so they do not clutter the title.

Usage
-----
    python visualise.py
    python visualise.py --control-dir path/to/control
    python visualise.py --top-n 12
    python visualise.py --confusion-labels path/to/labelled_sample.csv
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
DEFAULT_TOPICS     = os.path.join("analysis_output", "articles_with_topics.parquet")
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

# 12 perceptually-distinct hue families — adjacent slices are always separable.
# Previously the palette cycled through similar purple/violet hues which made
# slices hard to tell apart, especially in print.  Each entry here comes from
# a different hue family (purple, teal, coral, blue, pink, green, amber, gray,
# teal-light, coral-light, purple-light, teal-pale).
PALETTE = [
    "#7F77DD",  # purple-mid
    "#1D9E75",  # teal
    "#D85A30",  # coral
    "#378ADD",  # blue
    "#D4537E",  # pink
    "#639922",  # green
    "#BA7517",  # amber-dark
    "#888780",  # gray
    "#5DCAA5",  # teal-light
    "#F0997B",  # coral-light
    "#AFA9EC",  # purple-light
    "#9FE1CB",  # teal-pale
]

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


def _subnote(fig: plt.Figure, text: str, y: float = 0.01) -> None:
    """Place a small italic subnote below the chart area.

    Keeps chart titles clean (title = just the title) while preserving
    methodological context (n=, normalisation method, colour key) for readers.
    y=0.01 works for most single-panel charts; pass a lower value (e.g. -0.03)
    for figures with rotated x-axis tick labels.
    """
    fig.text(0.5, y, text, ha="center", fontsize=8.5,
             color=DARK_PLUM, alpha=0.68, style="italic",
             wrap=True)


def _slice_colors(slices: pd.Series) -> tuple[list[str], list[float], list[str]]:
    """Return (face_colors, explode, edge_colors) for a topic slice Series.

    Feminist slices: FEMINIST_COLOUR fill + FEMINIST_OUTLINE edge + larger explode.
    Other slices:    cycle PALETTE by hue family + white edge.
    'All Other Topics' bucket: muted lavender fill + white edge.

    Returns a 3-tuple so callers can pass edge_colors to _make_donut().
    """
    pie_colors, explode, edge_colors = [], [], []
    non_fem_idx = 0
    for lbl in slices.index:
        if lbl == "All Other Topics":
            pie_colors.append("#D1C4E9")
            explode.append(0)
            edge_colors.append("white")
        elif is_feminist(lbl):
            pie_colors.append(FEMINIST_COLOUR)
            explode.append(0.08)
            edge_colors.append(FEMINIST_OUTLINE)
        else:
            pie_colors.append(PALETTE[non_fem_idx % len(PALETTE)])
            explode.append(0)
            edge_colors.append("white")
            non_fem_idx += 1
    return pie_colors, explode, edge_colors


def _make_donut(ax, slices, pie_colors, explode, edge_colors=None):
    """Draw a donut chart and return the wedge artists.

    edge_colors allows per-wedge outline colour so feminist wedges get the dark
    amber border while non-feminist wedges keep a clean white edge.
    """
    if edge_colors is None:
        edge_colors = ["white"] * len(slices)
    wedges, _, _ = ax.pie(
        slices.values,
        autopct="%1.1f%%",
        startangle=140,
        colors=pie_colors,
        explode=explode,
        pctdistance=0.82,
        textprops={"color": DARK_PLUM, "fontsize": 9, "fontweight": "bold"},
        wedgeprops={"linewidth": 2.0, "width": 0.5},
    )
    for wedge, ec in zip(wedges, edge_colors):
        wedge.set_edgecolor(ec)
    return wedges


def _build_slices(counts: pd.Series, top_n: int) -> pd.Series:
    """Raw-count slice builder — retained for chart_topic_pie_all() only."""
    top     = counts.head(top_n)
    rest    = counts.iloc[top_n:]
    rescued = rest[rest.index.map(is_feminist)]
    other_n = rest[~rest.index.map(is_feminist)].sum()
    slices  = pd.concat([top, rescued])
    if other_n > 0:
        slices["All Other Topics"] = other_n
    return slices


def _build_slices_normalised(df: pd.DataFrame, top_n: int,
                              subset_col: str | None = None) -> pd.Series:
    """Normalised slice builder for overview / cross-event topic charts.

    Computes within-group proportions then averages them so that every group
    (event window) contributes equal weight regardless of article count.
    Returns a Series with values in [0, 1] (proportions, not percentages).
    """
    if subset_col and subset_col in df.columns:
        groups = []
        for _, grp in df.groupby(subset_col):
            props = grp["friendly_name"].value_counts(normalize=True)
            groups.append(props)
        if not groups:
            return pd.Series(dtype=float)
        mean_props = (pd.concat(groups, axis=1)
                      .fillna(0)
                      .mean(axis=1)
                      .sort_values(ascending=False))
    else:
        mean_props = (df["friendly_name"]
                      .value_counts(normalize=True)
                      .sort_values(ascending=False))

    top     = mean_props.head(top_n)
    rest    = mean_props.iloc[top_n:]
    rescued = rest[rest.index.map(is_feminist)]
    other_v = rest[~rest.index.map(is_feminist)].sum()
    slices  = pd.concat([top, rescued])
    if other_v > 0:
        slices["All Other Topics"] = other_v
    return slices


def _donut_with_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    slices: pd.Series,
    pie_colors: list,
    explode: list,
    edge_colors: list,
    title: str,
    subnote: str = "",
    pct_label: str = "%",
) -> None:
    """Draw a donut with a formatted legend.

    Legend entries:
      - Feminist topics: "★ Topic name — X.X%"  in bold FEMINIST_OUTLINE colour
        with a circular swatch (border-radius equivalent via marker) to make
        them instantly scannable without hunting for the gold colour.
      - Other topics:    "  Topic name — X.X%"  in DARK_PLUM

    Title is set on the axes; subnote is placed via fig.text() below the figure
    so the title line itself stays clean.
    """
    fig.patch.set_facecolor(LAVENDER_CREAM)
    wedges = _make_donut(ax, slices, pie_colors, explode, edge_colors)

    def _entry(lbl, val):
        prefix   = "★ " if is_feminist(lbl) else "  "
        val_str  = f"{val * 100:.1f}%" if pct_label == "%" else f"{int(val):,}"
        return f"{prefix}{lbl} — {val_str}"

    legend_labels = [_entry(l, v) for l, v in zip(slices.index, slices.values)]

    # Build custom legend handles so we can use circle patches for feminist
    # topics and square patches for others — the shape difference adds a second
    # visual channel on top of the colour difference.
    handles = []
    for i, (lbl, fc) in enumerate(zip(slices.index, pie_colors)):
        if is_feminist(lbl):
            h = mpatches.Circle((0, 0), radius=5, facecolor=fc,
                                 edgecolor=FEMINIST_OUTLINE, linewidth=1.5)
        else:
            h = mpatches.Patch(facecolor=fc, edgecolor="white", linewidth=0.8)
        handles.append(h)

    leg = ax.legend(handles, legend_labels,
                    title="Topics",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1),
                    fontsize=9,
                    frameon=False)
    for i, text in enumerate(leg.get_texts()):
        if is_feminist(slices.index[i]):
            text.set_fontweight("bold")
            text.set_color(FEMINIST_OUTLINE)
        else:
            text.set_color(DARK_PLUM)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=18, color=DARK_PLUM)
    if subnote:
        _subnote(fig, subnote)


def _bar_color(label: str, non_fem_counter: list) -> str:
    """Return bar fill colour for a topic label in a bar chart.

    Feminist labels: FEMINIST_COLOUR.
    Others: cycle PALETTE using a mutable counter list [int] passed by caller.
    """
    if is_feminist(label):
        return FEMINIST_COLOUR
    c = PALETTE[non_fem_counter[0] % len(PALETTE)]
    non_fem_counter[0] += 1
    return c


def _bar_legend_handles(labels: list[str]) -> list:
    """Build legend patch handles for a bar chart, mirroring _slice_colors logic."""
    handles = []
    non_fem_idx = 0
    for lbl in labels:
        if is_feminist(lbl):
            h = mpatches.Circle((0, 0), radius=5,
                                 facecolor=FEMINIST_COLOUR,
                                 edgecolor=FEMINIST_OUTLINE, linewidth=1.5,
                                 label=f"★ {lbl}")
        else:
            h = mpatches.Patch(facecolor=PALETTE[non_fem_idx % len(PALETTE)],
                                edgecolor="white", linewidth=0.5,
                                label=f"  {lbl}")
            non_fem_idx += 1
        handles.append(h)
    return handles


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
    ax.set_title("Mean sentiment score by publisher",
                 fontweight="bold", fontsize=14, color=DARK_PLUM)
    ax.set_xlabel("← More negative    Sentiment score    More positive →", color=DARK_PLUM)
    ax.legend(handles=[mpatches.Patch(color=RASPBERRY, label="Negative mean"),
                        mpatches.Patch(color=MUTED_TEAL, label="Positive mean")],
              frameon=False, fontsize=9)
    _subnote(fig, "VADER compound score (−1 to +1) · protest articles only")
    _save(fig, os.path.join(out_dir, "sentiment_by_outlet.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❺  CHART 2 — TOP TOPICS BY EVENT (bar) — NORMALISED
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
        show   = counts.loc[top_l + fem_l].sort_values()
        counter = [0]
        colors  = [_bar_color(l, counter) for l in show.index]
        edges   = [FEMINIST_OUTLINE if is_feminist(l) else "white" for l in show.index]
        widths  = [1.8 if is_feminist(l) else 0.5 for l in show.index]
        bars    = ax.barh(show.index, show.values, color=colors)
        for bar, ec, lw in zip(bars, edges, widths):
            bar.set_edgecolor(ec)
            bar.set_linewidth(lw)
        # Bold ★ y-tick labels for feminist topics
        for tick in ax.get_yticklabels():
            lbl = tick.get_text()
            if is_feminist(lbl):
                tick.set_color(FEMINIST_OUTLINE)
                tick.set_fontweight("bold")
                tick.set_text(f"★ {lbl}")
        ax.set_title(event.replace("_", " ").title(), fontweight="bold", color=DARK_PLUM)
        ax.set_xlabel("% of articles (within-event)", color=DARK_PLUM)
    fig.suptitle("Key topics per protest event",
                 fontsize=16, fontweight="bold", y=1.02, color=DARK_PLUM)
    fig.legend(handles=[
                   mpatches.Circle((0,0), radius=5,
                                   facecolor=FEMINIST_COLOUR,
                                   edgecolor=FEMINIST_OUTLINE,
                                   linewidth=1.5,
                                   label="★  Feminist / protest-related"),
                   mpatches.Patch(facecolor=PALETTE[0], edgecolor="white",
                                  label="Other topic"),
               ],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=10)
    _subnote(fig,
             "Proportions normalised within each event window · ★ = feminist/protest topic",
             y=-0.02)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "topic_breakdown_by_event.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❻  CHART 3 — PER-EVENT TOPIC PIES — NORMALISED
# ══════════════════════════════════════════════════════════════════════════════

def chart_topic_pies_per_event(df: pd.DataFrame, out_dir: str, dpi: int, top_n: int) -> None:
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topic_pies_per_event: 'friendly_name' missing — skipping."); return
    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    for event in sorted(protest_df["event_label"].dropna().unique()):
        sub    = protest_df[protest_df["event_label"] == event]
        slices = _build_slices_normalised(sub, top_n)
        if slices.empty:
            continue
        colors, explode, edges = _slice_colors(slices)
        fig, ax = plt.subplots(figsize=(13, 8))
        _donut_with_legend(
            fig, ax, slices, colors, explode, edges,
            title=f"Topic distribution — {event.replace('_', ' ').title()}",
            subnote=f"Within-event proportions · n={len(sub):,} articles · ★ = feminist/protest topic · amber circle = feminist",
            pct_label="%",
        )
        safe = event.replace(" ", "_").replace("/", "-")
        _save(fig, os.path.join(out_dir, f"topic_pie_{safe}.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❼  CHART 4 — OVERALL TOPIC PIE — NORMALISED ACROSS EVENTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_topic_pie(df: pd.DataFrame, out_dir: str, dpi: int, top_n: int) -> None:
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topic_pie: 'friendly_name' missing — skipping."); return
    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    slices = _build_slices_normalised(protest_df, top_n, subset_col="event_label")
    if slices.empty:
        return
    colors, explode, edges = _slice_colors(slices)
    fig, ax = plt.subplots(figsize=(14, 9))
    _donut_with_legend(
        fig, ax, slices, colors, explode, edges,
        title="Distribution of clustered news topics",
        subnote="Mean proportion per event (each event weighted equally) · ★ = feminist/protest topic · amber circle = feminist",
        pct_label="%",
    )
    _save(fig, os.path.join(out_dir, "topic_pie_overall.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❽  CHART 5 — ALL TOPICS PIE (raw counts — intentionally not normalised)
# ══════════════════════════════════════════════════════════════════════════════

def chart_topic_pie_all(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    """Full inventory of every discovered topic with absolute article counts.
    NOT normalised — purpose is topic-model diagnostics, not cross-event comparison.
    """
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topic_pie_all: 'friendly_name' missing — skipping."); return

    counts = df["friendly_name"].value_counts()
    pie_colors, explode, edge_colors = [], [], []
    non_fem_idx = 0
    for lbl in counts.index:
        if is_feminist(lbl):
            pie_colors.append(FEMINIST_COLOUR)
            explode.append(0.05)
            edge_colors.append(FEMINIST_OUTLINE)
        else:
            pie_colors.append(PALETTE[non_fem_idx % len(PALETTE)])
            explode.append(0)
            edge_colors.append("white")
            non_fem_idx += 1

    n = len(counts)
    fig_h = max(10, n * 0.28)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    fig.patch.set_facecolor(LAVENDER_CREAM)

    wedges, _, _ = ax.pie(
        counts.values,
        startangle=140,
        colors=pie_colors,
        explode=explode,
        wedgeprops={"linewidth": 1.5, "width": 0.5},
    )
    for wedge, ec in zip(wedges, edge_colors):
        wedge.set_edgecolor(ec)

    # Custom legend: ★ prefix + circle swatch for feminist, square for others
    handles = []
    legend_labels = []
    non_fem_idx = 0
    for i, (lbl, cnt) in enumerate(zip(counts.index, counts.values)):
        fc = pie_colors[i]
        if is_feminist(lbl):
            h = mpatches.Circle((0, 0), radius=5, facecolor=fc,
                                 edgecolor=FEMINIST_OUTLINE, linewidth=1.5)
            legend_labels.append(f"★ {lbl} ({int(cnt):,})")
        else:
            h = mpatches.Patch(facecolor=fc, edgecolor="white", linewidth=0.5)
            legend_labels.append(f"  {lbl} ({int(cnt):,})")
        handles.append(h)

    leg = ax.legend(handles, legend_labels,
                    title="All topics (raw counts)",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1),
                    fontsize=8,
                    frameon=False)
    for i, text in enumerate(leg.get_texts()):
        if is_feminist(counts.index[i]):
            text.set_fontweight("bold")
            text.set_color(FEMINIST_OUTLINE)
        else:
            text.set_color(DARK_PLUM)

    ax.set_title("All discovered topics — raw document counts",
                 fontsize=14, fontweight="bold", pad=20, color=DARK_PLUM)
    _subnote(fig,
             "Raw BERTopic model output · NOT normalised across events · for diagnostics only · ★ = feminist/protest topic")
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
            m, c, ec, ew, z = "D", FEMINIST_COLOUR, FEMINIST_OUTLINE, 1.8, 4
        else:
            m = "o"
            c = RASPBERRY if row["mean_sentiment"] < 0 else MUTED_TEAL
            ec, ew, z = DARK_PLUM, 0.5, 3
        ax.scatter(row["mean_sentiment"], row["friendly_name"],
                   s=np.clip(row["count"] * 5, 60, 1200),
                   c=c, marker=m, edgecolors=ec, linewidths=ew, alpha=0.88, zorder=z)

    # Bold + colour feminist y-tick labels
    for tick in ax.get_yticklabels():
        lbl = tick.get_text()
        if is_feminist(lbl):
            tick.set_color(FEMINIST_OUTLINE)
            tick.set_fontweight("bold")

    ax.axvline(0, color=DARK_PLUM, linestyle="--", alpha=0.5)
    ax.set_title("Mean sentiment by topic",
                 fontweight="bold", fontsize=14, color=DARK_PLUM)
    ax.set_xlabel("← More negative    Sentiment score    More positive →", color=DARK_PLUM)
    ax.legend(handles=[
        mpatches.Circle((0,0), radius=5,
                        facecolor=FEMINIST_COLOUR, edgecolor=FEMINIST_OUTLINE,
                        linewidth=1.5, label="★  Feminist/protest topic (◆)"),
        mpatches.Patch(color=RASPBERRY, label="Negative mean (other)"),
        mpatches.Patch(color=MUTED_TEAL, label="Positive mean (other)"),
    ], frameon=False, fontsize=9)
    _subnote(fig, "Dot size ∝ article volume · ★ feminist/protest topics in amber")
    _save(fig, os.path.join(out_dir, "sentiment_by_topic.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❿  CHART 7 — ARTICLES PER EVENT
# ══════════════════════════════════════════════════════════════════════════════

def chart_articles_per_event(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    counts = protest_df.groupby("event_label").size().sort_values()

    fig, ax = plt.subplots(figsize=(11, max(5, len(counts) * 0.55)))
    _base_style(fig, ax)

    median_n = counts.median()
    colors   = [RASPBERRY if v < median_n else MUTED_TEAL for v in counts.values]
    bars     = ax.barh(counts.index, counts.values, color=colors, alpha=0.85)

    for bar, (event, n) in zip(bars, counts.items()):
        note  = EVENT_COVERAGE_NOTES.get(event, "")
        label = f"  {n:,}"
        if note:
            label += f"  ·  {note}"
        ax.text(bar.get_width() + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=8.5, color=DARK_PLUM)

    ax.set_xlabel("Number of articles", color=DARK_PLUM)
    ax.set_title("Articles collected per protest event",
                 fontweight="bold", fontsize=13, color=DARK_PLUM)
    ax.set_xlim(0, counts.max() * 1.55)
    ax.axvline(median_n, color=DARK_PLUM, linestyle=":", linewidth=1, alpha=0.6)
    ax.text(median_n + counts.max() * 0.01, -0.5, "median",
            fontsize=8, color=DARK_PLUM, alpha=0.7)
    _subnote(fig,
             "Raspberry bars = below median · annotations explain coverage gaps (e.g. CC-News sparse pre-2019)")
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
    ax.bar(x - w/2, pv, w, color=RASPBERRY,   label="Protest week", alpha=0.9)
    ax.bar(x + w/2, cv, w, color=DEEP_PURPLE, label="Control week", alpha=0.7)
    ax.axhline(0, color=DARK_PLUM, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(publishers, rotation=35, ha="right", color=DARK_PLUM)
    ax.set_ylabel("Mean sentiment score", color=DARK_PLUM)
    ax.set_title("Sentiment: protest weeks vs matched control weeks",
                 fontweight="bold", fontsize=14, color=DARK_PLUM)
    ax.legend(frameon=False, fontsize=10)
    _subnote(fig, "Each publisher's protest-week mean vs matched ±4-week control window",
             y=-0.06)
    _save(fig, os.path.join(out_dir, "protest_vs_control.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓬  CHART 9 — IDEOLOGICAL FAIRNESS GAP
# ══════════════════════════════════════════════════════════════════════════════

def chart_ideological_gap(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
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

    def _ic(score):
        if score < -0.5: return MUTED_TEAL
        if score >  0.5: return RASPBERRY
        return DEEP_PURPLE

    colors = [_ic(s) for s in agg["ideology"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    _base_style(fig, ax, grid_axis="y")
    ax.scatter(agg["ideology"], agg["gap"], c=colors, s=100,
               edgecolors=DARK_PLUM, linewidths=0.6, alpha=0.88, zorder=3)
    for pub, row in agg.iterrows():
        ax.annotate(pub, (row["ideology"], row["gap"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, color=DARK_PLUM, alpha=0.85)
    if len(agg) >= 3:
        z  = np.polyfit(agg["ideology"], agg["gap"], 1)
        xs = np.linspace(agg["ideology"].min(), agg["ideology"].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color=DARK_PLUM, linewidth=1.2,
                linestyle="--", alpha=0.5, label="OLS trend")
    ax.axhline(0, color=DARK_PLUM, linewidth=0.8, linestyle=":")
    ax.axvline(0, color=DARK_PLUM, linewidth=0.5, linestyle=":", alpha=0.4)
    ax.set_xlabel("← Left-leaning    Ideological score (AllSides)    Right-leaning →",
                  color=DARK_PLUM)
    ax.set_ylabel("Sentiment gap (protest − control week)", color=DARK_PLUM)
    ax.set_title("Ideological fairness metric",
                 fontweight="bold", fontsize=13, color=DARK_PLUM)
    ax.legend(handles=[
        mpatches.Patch(color=MUTED_TEAL,  label="Left-leaning outlet"),
        mpatches.Patch(color=DEEP_PURPLE, label="Centre outlet"),
        mpatches.Patch(color=RASPBERRY,   label="Right-leaning outlet"),
    ] + ([Line2D([0],[0], color=DARK_PLUM, linestyle="--", alpha=0.5, label="OLS trend")]
         if len(agg) >= 3 else []),
    frameon=False, fontsize=9)
    _subnote(fig,
             "Gap = protest-week mean minus control-week mean · negative = more negative protest coverage")
    _save(fig, os.path.join(out_dir, "sentiment_ideological_gap.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓭  CHART 10 — SENTIMENT BY EVENT TYPE
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
        ax.barh(sub["event_label"].str.replace("_", " "), sub["sentiment_score"],
                color=type_colors.get(etype, DEEP_PURPLE), alpha=0.85)
        ax.axvline(0, color=DARK_PLUM, linewidth=0.8)
        ax.set_title(f"Event type: {etype.title()}", fontweight="bold", color=DARK_PLUM)
        ax.set_xlabel("Mean sentiment", color=DARK_PLUM)
    fig.suptitle("Mean sentiment by event type",
                 fontsize=14, fontweight="bold", color=DARK_PLUM)
    _subnote(fig, "Single-day vs sustained protest events · VADER compound score")
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
        ax.bar(x, vals, bottom=bottom,
               color=lang_palette.get(lang,"#BDBDBD"), label=lang, alpha=0.9)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(publishers, rotation=35, ha="right", color=DARK_PLUM)
    ax.set_ylabel("% of articles", color=DARK_PLUM)
    ax.set_ylim(0, 105)
    ax.set_title("Article language composition by publisher",
                 fontweight="bold", fontsize=13, color=DARK_PLUM)
    ax.legend(title="Language", frameon=False, fontsize=9, bbox_to_anchor=(1,1))
    _subnote(fig, "Lingua language detection · non-EN articles translated before sentiment scoring",
             y=-0.06)
    _save(fig, os.path.join(out_dir, "language_breakdown.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓯  CHART 12 — PIPELINE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def chart_pipeline_diagram(out_dir: str, dpi: int) -> None:
    steps = [
        ("collect_articles\n_optimized.py", "Crawl CC-News\n45 publishers × 10 events\n+ matched control weeks"),
        ("build_corpus.py",                 "Merge per-event parquets\nDeduplicate by URL"),
        ("topic_model.py",                  "multilingual-e5-large embeddings\nBERTopic (HDBSCAN)\nOutlier reduction"),
        ("sentiment_analysis.py",           "Lingua language detection\nHelsinki opus-mt translation\nVADER scoring"),
        ("visualise.py",                    "19 publication figures\nbias_report.csv"),
    ]
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM)
    ax.axis("off")
    n = len(steps)
    xs = np.linspace(0.08, 0.92, n)
    y_box = 0.5
    bw, bh = 0.14, 0.55
    for i, (script, desc) in enumerate(steps):
        x = xs[i]
        rect = plt.Rectangle((x - bw/2, y_box - bh/2), bw, bh,
                              facecolor=DEEP_PURPLE, edgecolor=BLUSH,
                              linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y_box + 0.08, script, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white", zorder=3)
        ax.text(x, y_box - 0.18, desc, ha="center", va="center",
                fontsize=7, color=DARK_PLUM, zorder=3,
                bbox=dict(facecolor=LAVENDER_CREAM, edgecolor="none", pad=1))
        if i < n - 1:
            ax.annotate("", xy=(xs[i+1] - bw/2 - 0.005, y_box),
                        xytext=(x + bw/2 + 0.005, y_box),
                        arrowprops=dict(arrowstyle="->", color=RASPBERRY,
                                        lw=1.8), zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Analysis pipeline", fontsize=14, fontweight="bold",
                 color=DARK_PLUM, pad=10)
    _save(fig, os.path.join(out_dir, "pipeline_diagram.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓰  BIAS REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_bias_report(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sections = []

    cov = (df.groupby(["publisher","event_label"]).size()
             .reset_index(name="n_articles"))
    cov["flag_low_coverage"] = cov["n_articles"] < 5
    cov["section"] = "A_coverage_completeness"
    sections.append(cov)

    if "language" in df.columns:
        lang = df.groupby("language").size().reset_index(name="n_articles")
        lang["pct"] = (lang["n_articles"] / len(df) * 100).round(2)
        lang["section"] = "B_language_distribution"
        sections.append(lang)
        lang_pub = df.groupby(["publisher","language"]).size().reset_index(name="n_articles")
        lang_pub["section"] = "B_language_by_publisher"
        sections.append(lang_pub)

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

    if "topic_id_raw" in df.columns and "topic_id" in df.columns:
        n_total         = len(df)
        n_raw_outlier   = (df["topic_id_raw"] == -1).sum()
        n_final_outlier = (df["topic_id"] == -1).sum()
        outlier_df = pd.DataFrame([{
            "n_total":            n_total,
            "n_raw_outliers":     int(n_raw_outlier),
            "pct_raw_outliers":   round(100 * n_raw_outlier / n_total, 2),
            "n_final_outliers":   int(n_final_outlier),
            "pct_final_outliers": round(100 * n_final_outlier / n_total, 2),
            "n_reassigned":       int(n_raw_outlier - n_final_outlier),
            "section":            "D_outlier_rate",
        }])
        sections.append(outlier_df)

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

    combined = pd.concat(sections, ignore_index=True)
    out_path  = os.path.join(out_dir, "bias_report.csv")
    combined.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    print("\n  ── Bias report summary ──────────────────────────────────────────")
    print(f"  Total articles        : {len(df):,}")
    if "language" in df.columns:
        print(f"  Language distribution : {df['language'].value_counts().to_dict()}")
    if "topic_id_raw" in df.columns:
        print(f"  Outlier rate (raw)    : {100*(df['topic_id_raw']==-1).mean():.1f}%")
        print(f"  Outlier rate (final)  : {100*(df['topic_id']==-1).mean():.1f}%")
    if "ideology_bin" in pub_counts.columns:
        ideol = pub_counts.groupby("ideology_bin", observed=True)["n_articles"].sum()
        print(f"  Articles by ideology  : {ideol.to_dict()}")
    low_cov = cov[cov["flag_low_coverage"]]
    if not low_cov.empty:
        print(f"  ⚠️  {len(low_cov)} publisher×event combinations have < 5 articles")
    print("  ─────────────────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# ⓱  CHART 14 — DEMOGRAPHIC DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def chart_demographic_distribution(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    if "language" not in df.columns:
        print("  ⚠️  chart_demographic_distribution: 'language' missing — skipping."); return

    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    protest_df = protest_df.copy()
    protest_df["ideology_score"] = protest_df["publisher"].map(IDEOLOGY_SCORES)
    protest_df["ideology_bin"] = pd.cut(
        protest_df["ideology_score"],
        bins=[-3, -0.5, 0.5, 3],
        labels=["Left", "Centre", "Right"],
    ).astype(str).replace("nan", "Unknown")

    lang_palette = {"EN": DEEP_PURPLE, "DE": RASPBERRY, "FR": MUTED_TEAL,
                    "ES": FEMINIST_COLOUR, "OTHER": "#BDBDBD"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(LAVENDER_CREAM)

    _base_style(fig, ax1, grid_axis="y")
    ideol_lang = (protest_df.groupby(["ideology_bin", "language"])
                  .size().reset_index(name="n"))
    ideol_bins = ["Left", "Centre", "Right", "Unknown"]
    x_pos  = np.arange(len(ideol_bins))
    bottom = np.zeros(len(ideol_bins))
    for lang in ["EN", "DE", "FR", "ES", "OTHER"]:
        sub  = ideol_lang[ideol_lang["language"] == lang].set_index("ideology_bin")
        vals = np.array([sub.loc[b,"n"] if b in sub.index else 0 for b in ideol_bins])
        if vals.sum() == 0: continue
        ax1.bar(x_pos, vals, bottom=bottom,
                color=lang_palette.get(lang,"#BDBDBD"), label=lang, alpha=0.9)
        bottom += vals
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ideol_bins, color=DARK_PLUM)
    ax1.set_ylabel("Number of articles", color=DARK_PLUM)
    ax1.set_title("A — Article volume by outlet ideology × language",
                  fontweight="bold", color=DARK_PLUM)
    ax1.legend(title="Language", frameon=False, fontsize=9)

    _base_style(fig, ax2, grid_axis="x")
    event_lang = (protest_df.groupby(["event_label","language"])
                  .size().reset_index(name="n"))
    events = sorted(protest_df["event_label"].dropna().unique())
    y_pos  = np.arange(len(events))
    left   = np.zeros(len(events))
    for lang in ["EN", "DE", "FR", "ES", "OTHER"]:
        sub  = event_lang[event_lang["language"] == lang].set_index("event_label")
        vals = np.array([sub.loc[e,"n"] if e in sub.index else 0 for e in events])
        if vals.sum() == 0: continue
        ax2.barh(y_pos, vals, left=left,
                 color=lang_palette.get(lang,"#BDBDBD"), label=lang, alpha=0.9)
        left += vals
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([e.replace("_"," ") for e in events], color=DARK_PLUM, fontsize=9)
    ax2.set_xlabel("Number of articles", color=DARK_PLUM)
    ax2.set_title("B — Article volume by event × language",
                  fontweight="bold", color=DARK_PLUM)
    ax2.legend(title="Language", frameon=False, fontsize=9)

    fig.suptitle("Demographic distribution of the corpus",
                 fontsize=15, fontweight="bold", color=DARK_PLUM, y=1.02)
    _subnote(fig, "Protest articles only · ideology bins from AllSides/Ad Fontes scores")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "demographic_distribution.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓲  CHART 15 — FAIRNESS METRIC COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def chart_fairness_metric_comparison(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    if "is_control" not in df.columns or df["is_control"].nunique() < 2:
        print("  ⚠️  chart_fairness_metric_comparison: need both is_control values — skipping.")
        return

    agg = df.groupby(["publisher","is_control"])["sentiment_score"].agg(
        ["mean","std","count"]).reset_index()
    agg.columns = ["publisher","is_control","mean","std","count"]

    protest_agg = agg[~agg["is_control"]].set_index("publisher")
    control_agg = agg[ agg["is_control"]].set_index("publisher")
    common_pubs = protest_agg.index.intersection(control_agg.index)

    rows = []
    for pub in common_pubs:
        p_mean = protest_agg.loc[pub,"mean"]
        c_mean = control_agg.loc[pub,"mean"]
        c_std  = control_agg.loc[pub,"std"]
        p_std  = protest_agg.loc[pub,"std"]
        p_n    = protest_agg.loc[pub,"count"]
        norm_gap = (p_mean - c_mean) / (c_std if c_std > 0 else 1.0)
        se       = p_std / np.sqrt(max(p_n, 1))
        ideology = IDEOLOGY_SCORES.get(pub, np.nan)
        rows.append({"publisher": pub, "norm_gap": norm_gap,
                     "se": se, "ideology": ideology})

    plot_df = pd.DataFrame(rows).sort_values("norm_gap")

    def _ic(score):
        if pd.isna(score): return "#BDBDBD"
        if score < -0.5:   return MUTED_TEAL
        if score >  0.5:   return RASPBERRY
        return DEEP_PURPLE

    colors = [_ic(s) for s in plot_df["ideology"]]
    y_pos  = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.45)))
    _base_style(fig, ax)
    ax.barh(y_pos, plot_df["norm_gap"], color=colors, alpha=0.85, zorder=3)
    ax.errorbar(plot_df["norm_gap"], y_pos,
                xerr=plot_df["se"] * 1.96,
                fmt="none", color=DARK_PLUM, linewidth=1.0, capsize=3, zorder=4)
    ax.axvline(0, color=DARK_PLUM, linewidth=1.0, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["publisher"], fontsize=9, color=DARK_PLUM)
    ax.set_xlabel("Normalised sentiment gap", color=DARK_PLUM)
    ax.set_title("Fairness metric comparison across publishers",
                 fontweight="bold", fontsize=13, color=DARK_PLUM)
    ax.legend(handles=[
        mpatches.Patch(color=MUTED_TEAL,  label="Left-leaning"),
        mpatches.Patch(color=DEEP_PURPLE, label="Centre"),
        mpatches.Patch(color=RASPBERRY,   label="Right-leaning"),
        mpatches.Patch(color="#BDBDBD",   label="Unknown ideology"),
    ], frameon=False, fontsize=9, loc="lower right")
    _subnote(fig,
             "Gap = (protest mean − control mean) ÷ control SD · error bars = 95% CI · "
             "negative = more negative coverage during protest week")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "fairness_metric_comparison.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓳  CHART 16 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

def chart_model_performance(df: pd.DataFrame, topic_info_path: str,
                             out_dir: str, dpi: int) -> None:
    if "topic_id" not in df.columns:
        print("  ⚠️  chart_model_performance: 'topic_id' missing — skipping."); return

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.patch.set_facecolor(LAVENDER_CREAM)

    # Panel A — topic size distribution
    ax = axes[0]
    _base_style(fig, ax, grid_axis="y")
    topic_sizes   = df["topic_id"].value_counts().sort_index()
    sizes_no_out  = topic_sizes[topic_sizes.index != -1]
    has_outlier   = -1 in topic_sizes.index
    ax.bar(range(len(sizes_no_out)), sorted(sizes_no_out.values, reverse=True),
           color=DEEP_PURPLE, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Topic rank (by size)", color=DARK_PLUM)
    ax.set_ylabel("Articles (log scale)", color=DARK_PLUM)
    ax.set_title(f"A — Topic size distribution",
                 fontweight="bold", color=DARK_PLUM)

    # Panel B — outlier rate by event
    ax = axes[1]
    _base_style(fig, ax, grid_axis="x")
    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    outlier_rate = (protest_df.groupby("event_label")
                   .apply(lambda g: (g["topic_id"] == -1).mean() * 100)
                   .sort_values())
    ax.barh(outlier_rate.index, outlier_rate.values,
            color=[RASPBERRY if v > 20 else MUTED_TEAL for v in outlier_rate.values],
            alpha=0.85)
    ax.axvline(20, color=DARK_PLUM, linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Outlier rate (%)", color=DARK_PLUM)
    ax.set_title("B — Outlier rate by event", fontweight="bold", color=DARK_PLUM)
    for i, (event, val) in enumerate(outlier_rate.items()):
        ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=8, color=DARK_PLUM)

    # Panel C — per-event topic counts
    ax = axes[2]
    _base_style(fig, ax, grid_axis="x")
    event_topics_dir  = os.path.join(os.path.dirname(topic_info_path), "event_topics")
    event_info_files  = glob.glob(os.path.join(event_topics_dir, "*_topic_info.csv"))
    if event_info_files:
        topic_counts = {}
        for f in sorted(event_info_files):
            try:
                tdf = pd.read_csv(f)
                event_name = os.path.basename(f).replace("_topic_info.csv", "")
                topic_counts[event_name] = int((tdf["Topic"] != -1).sum())
            except Exception:
                pass
        if topic_counts:
            tc = pd.Series(topic_counts).sort_values()
            ax.barh(tc.index, tc.values, color=MUTED_TEAL, alpha=0.85)
            ax.set_xlabel("Topics discovered", color=DARK_PLUM)
            ax.set_title("C — Per-event topic count", fontweight="bold", color=DARK_PLUM)
        else:
            ax.text(0.5, 0.5, "No per-event topic info files found",
                    ha="center", va="center", transform=ax.transAxes, color=DARK_PLUM)
            ax.set_title("C — Per-event topic count", fontweight="bold", color=DARK_PLUM)
    else:
        ax.text(0.5, 0.5,
                "Run topic_model.py without\n--no-event-topics to generate\nper-event models",
                ha="center", va="center", transform=ax.transAxes, color=DARK_PLUM, fontsize=10)
        ax.axis("off")
        ax.set_title("C — Per-event topic count (not available)",
                     fontweight="bold", color=DARK_PLUM)

    fig.suptitle("Topic model performance diagnostics",
                 fontsize=15, fontweight="bold", color=DARK_PLUM, y=1.02)
    n_topics   = len(sizes_no_out)
    n_outliers = int(topic_sizes.get(-1, 0))
    _subnote(fig,
             f"{n_topics} topics · {n_outliers:,} outlier docs · "
             "Panel B dotted line = 20% outlier threshold · "
             "Panel C from per-event BERTopic models")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "model_performance.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓴  CHART 17 — SUBGROUP COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def chart_subgroup_comparison(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    protest_df = protest_df.copy()
    protest_df["ideology_score"] = protest_df["publisher"].map(IDEOLOGY_SCORES)
    protest_df["ideology_bin"] = pd.cut(
        protest_df["ideology_score"],
        bins=[-3, -0.5, 0.5, 3],
        labels=["Left", "Centre", "Right"],
    ).astype(str).replace("nan", "Unknown")

    events     = sorted(protest_df["event_label"].dropna().unique())
    ideol_bins = ["Left", "Centre", "Right"]
    bin_colors = {"Left": MUTED_TEAL, "Centre": DEEP_PURPLE, "Right": RASPBERRY}

    heat = (protest_df.groupby(["event_label","ideology_bin"])["sentiment_score"]
            .mean().unstack("ideology_bin").reindex(columns=ideol_bins))

    has_topics = "friendly_name" in protest_df.columns
    n_panels   = 2 if has_topics else 1
    fig, axes  = plt.subplots(n_panels, 1, figsize=(14, 6 * n_panels))
    if n_panels == 1:
        axes = [axes]
    fig.patch.set_facecolor(LAVENDER_CREAM)

    ax = axes[0]
    _base_style(fig, ax, grid_axis="y")
    n_events = len(events)
    n_bins   = len(ideol_bins)
    group_w  = 0.8
    bar_w    = group_w / n_bins
    x        = np.arange(n_events)
    for j, ibin in enumerate(ideol_bins):
        vals   = [heat.loc[e, ibin] if e in heat.index and ibin in heat.columns
                  else np.nan for e in events]
        offset = (j - n_bins / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, width=bar_w * 0.9,
               color=bin_colors[ibin], label=ibin, alpha=0.88)
    ax.axhline(0, color=DARK_PLUM, linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("_"," ") for e in events],
                       rotation=35, ha="right", color=DARK_PLUM, fontsize=9)
    ax.set_ylabel("Mean sentiment score", color=DARK_PLUM)
    ax.set_title("A — Mean sentiment by event × outlet ideology",
                 fontweight="bold", color=DARK_PLUM)
    ax.legend(title="Outlet ideology", frameon=False, fontsize=9)

    if has_topics:
        ax2 = axes[1]
        _base_style(fig, ax2, grid_axis="y")
        topic_props = {}
        for ibin in ideol_bins:
            sub = protest_df[protest_df["ideology_bin"] == ibin]
            topic_props[ibin] = (sub["friendly_name"].value_counts(normalize=True)
                                 if not sub.empty else pd.Series(dtype=float))
        all_topics = sorted(set.union(*[set(v.index) for v in topic_props.values()
                                        if not v.empty]))
        x2     = np.arange(len(ideol_bins))
        bottom = np.zeros(len(ideol_bins))
        counter = [0]
        for topic in all_topics:
            vals  = [topic_props[b].get(topic, 0) * 100 for b in ideol_bins]
            color = FEMINIST_COLOUR if is_feminist(topic) else PALETTE[counter[0] % len(PALETTE)]
            edge  = FEMINIST_OUTLINE if is_feminist(topic) else "white"
            lw    = 1.8 if is_feminist(topic) else 0.5
            bars  = ax2.bar(x2, vals, bottom=bottom, color=color, alpha=0.85,
                            label=f"★ {topic}" if is_feminist(topic) else topic,
                            edgecolor=edge, linewidth=lw)
            if not is_feminist(topic):
                counter[0] += 1
            bottom += np.array(vals)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(ideol_bins, color=DARK_PLUM)
        ax2.set_ylabel("% of articles (within ideology bin)", color=DARK_PLUM)
        ax2.set_title("B — Topic composition by outlet ideology",
                      fontweight="bold", color=DARK_PLUM)
        # Compact legend: show feminist topics first, cap others at 10
        handles, labels = ax2.get_legend_handles_labels()
        fem_idx   = [i for i, l in enumerate(labels) if l.startswith("★")]
        other_idx = [i for i, l in enumerate(labels) if not l.startswith("★")][:10]
        shown_idx = fem_idx + other_idx
        ax2.legend([handles[i] for i in shown_idx],
                   [labels[i] for i in shown_idx],
                   title="Topics (★ = feminist)", frameon=False,
                   fontsize=7, bbox_to_anchor=(1, 1), loc="upper left", ncol=1)

    fig.suptitle("Subgroup comparison: sentiment & topic by outlet ideology",
                 fontsize=15, fontweight="bold", color=DARK_PLUM, y=1.02)
    _subnote(fig,
             "Panel B normalised within each ideology bin · ★ feminist topics in amber · "
             "consistent ideology effect across events supports structural framing bias claim")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "subgroup_comparison.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ㉑  CHART 18 — CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def chart_confusion_matrix(df: pd.DataFrame, out_dir: str, dpi: int,
                            labelled_sample_path: str | None = None) -> None:
    if "sentiment_label" not in df.columns:
        print("  ⚠️  chart_confusion_matrix: 'sentiment_label' missing — skipping."); return

    protest_df  = df[~df["is_control"]] if "is_control" in df.columns else df
    label_order = ["positive", "neutral", "negative"]
    col_labels  = label_order
    subnote_txt = ""

    if labelled_sample_path and os.path.exists(labelled_sample_path):
        try:
            gt     = pd.read_csv(labelled_sample_path)
            merged = protest_df.merge(gt[["url","human_label"]], on="url", how="inner")
            if merged.empty:
                raise ValueError("No overlapping URLs.")
            cm = pd.crosstab(
                merged["human_label"].str.lower(),
                merged["sentiment_label"].str.lower(),
                rownames=["Human label"], colnames=["VADER label"],
            ).reindex(index=label_order, columns=label_order, fill_value=0)
            n_total  = cm.values.sum()
            accuracy = np.diag(cm.values).sum() / max(n_total, 1)
            matrix_vals = cm.values
            title       = "Sentiment classifier confusion matrix"
            subnote_txt = (f"n={n_total:,} · VADER vs human annotation · "
                           f"accuracy={accuracy:.1%} · rows=human label · columns=VADER prediction")
        except Exception as e:
            print(f"  ⚠️  Could not load labelled sample ({e}) — falling back to proxy matrix.")
            labelled_sample_path = None

    if not (labelled_sample_path and os.path.exists(str(labelled_sample_path))):
        protest_df = protest_df.copy()
        protest_df["ideology_score"] = protest_df["publisher"].map(IDEOLOGY_SCORES)
        protest_df["ideology_bin"] = pd.cut(
            protest_df["ideology_score"],
            bins=[-3, -0.5, 0.5, 3],
            labels=["Left", "Centre", "Right"],
        ).astype(str).replace("nan", "Unknown")
        cm = (protest_df.groupby(["ideology_bin","sentiment_label"])
              .size().unstack("sentiment_label", fill_value=0))
        cm  = cm.div(cm.sum(axis=1), axis=0) * 100
        cm  = cm.reindex(columns=[c for c in label_order if c in cm.columns])
        matrix_vals = cm.values
        label_order = cm.index.tolist()
        col_labels  = cm.columns.tolist()
        title       = "Sentiment distribution by outlet ideology"
        subnote_txt = ("Proxy validation — no ground-truth labels supplied · "
                       "rows=ideology bin · columns=VADER label · values=% of row · "
                       "pass --confusion-labels for a true confusion matrix")

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM)
    im = ax.imshow(matrix_vals, cmap="Purples", aspect="auto")
    plt.colorbar(im, ax=ax, label="Count / %")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, color=DARK_PLUM, fontsize=11)
    ax.set_yticks(range(len(label_order)))
    ax.set_yticklabels(label_order, color=DARK_PLUM, fontsize=11)
    thresh = matrix_vals.max() / 2.0
    for i in range(matrix_vals.shape[0]):
        for j in range(matrix_vals.shape[1]):
            val = matrix_vals[i, j]
            txt = f"{val:.1f}" if isinstance(val, float) else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12,
                    color="white" if val > thresh else DARK_PLUM, fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=12, color=DARK_PLUM, pad=14)
    _subnote(fig, subnote_txt)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "confusion_matrix_sentiment.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ㉒  CHART 19 — EXTENDED WORKFLOW DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def chart_workflow_diagram(out_dir: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(20, 9))
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM)
    ax.axis("off")
    ax.set_xlim(0, 20); ax.set_ylim(0, 9)

    LANE_Y = {"data": 6.8, "analysis": 2.8}
    BOX_H  = 1.8; BOX_W = 2.8

    nodes = {
        "collect":   (1.8,  "data",
                      "collect_articles\n_optimized.py",
                      "CommonCrawl CC-News\nkeyword filter × publisher list",
                      "corpus_YYYY.parquet"),
        "control":   (1.8,  "analysis",
                      "collect_articles\n_optimized.py\n(control flag)",
                      "Matched control weeks\n±4 weeks from event",
                      "control/*.parquet"),
        "build":     (5.2,  "data",
                      "build_corpus.py",
                      "Merge + deduplicate by URL\nAttach event_label / event_type",
                      "corpus_all.parquet"),
        "sentiment": (8.6,  "data",
                      "sentiment_analysis.py",
                      "Lingua detection → opus-mt\ntranslation → VADER scoring",
                      "articles_with_sentiment.parquet"),
        "topic":     (12.0, "data",
                      "topic_model.py",
                      "all-MiniLM-L6-v2 + BERTopic\nHDBSCAN, outlier reduction",
                      "articles_with_topics.parquet\ntopic_info.csv"),
        "vis":       (15.8, "data",
                      "visualise.py",
                      "19 charts + bias_report.csv\nNormalised topic distributions",
                      "figures/*.png"),
        "bias":      (18.8, "analysis",
                      "bias_report.csv",
                      "Coverage · language · ideology\nbalance statistics",
                      "Supplementary table"),
    }

    arrow_pairs = [
        ("collect",   "build",     False),
        ("control",   "build",     True),
        ("build",     "sentiment", False),
        ("sentiment", "topic",     False),
        ("topic",     "vis",       False),
        ("vis",       "bias",      False),
    ]

    drawn: dict[str, tuple[float, float]] = {}
    for key, (cx, lane, script, tech, out) in nodes.items():
        cy      = LANE_Y[lane]
        is_ctrl = (lane == "analysis" and key == "control")
        rect    = plt.Rectangle((cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
                                 facecolor="#6A1B9A" if is_ctrl else DEEP_PURPLE,
                                 edgecolor=BLUSH, linewidth=1.5, zorder=2, alpha=0.92)
        ax.add_patch(rect)
        ax.text(cx, cy + 0.45, script, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white", zorder=3)
        ax.text(cx, cy - 0.05, tech, ha="center", va="center",
                fontsize=6.5, color=BLUSH, zorder=3)
        ax.text(cx, cy - 0.62, f"→ {out}", ha="center", va="center",
                fontsize=6.2, color="#E1BEE7", zorder=3, style="italic")
        drawn[key] = (cx, cy)

    ax.text(0.15, LANE_Y["data"],     "DATA\nPIPELINE",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=DEEP_PURPLE, rotation=90)
    ax.text(0.15, LANE_Y["analysis"], "ANALYSIS &\nOUTPUTS",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=DEEP_PURPLE, rotation=90)
    ax.axhline(4.8, color=BLUSH, linewidth=0.8, linestyle=":")

    for src, dst, dashed in arrow_pairs:
        sx, sy = drawn[src]; dx, dy = drawn[dst]
        ax.annotate("", xy=(dx - BOX_W/2 - 0.05, dy),
                    xytext=(sx + BOX_W/2 + 0.05, sy),
                    arrowprops=dict(arrowstyle="->", color=RASPBERRY,
                                    lw=1.6, linestyle="dashed" if dashed else "solid"),
                    zorder=4)

    ax.plot([0.6, 1.2], [0.4, 0.4], color=RASPBERRY, lw=1.6, linestyle="dashed")
    ax.text(1.3, 0.4, "optional / conditional data flow",
            va="center", fontsize=8, color=DARK_PLUM)

    ax.set_title("Extended analysis workflow",
                 fontsize=15, fontweight="bold", color=DARK_PLUM, pad=12)
    _save(fig, os.path.join(out_dir, "workflow_diagram.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓱  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures for the Women's Protest News Corpus."
    )
    parser.add_argument("--sentiment",        default=DEFAULT_SENTIMENT)
    parser.add_argument("--topics",           default=DEFAULT_TOPICS)
    parser.add_argument("--topic-info",       default=DEFAULT_TOPIC_INFO)
    parser.add_argument("--control-dir",      default=DEFAULT_CONTROL)
    parser.add_argument("--out-dir",          default=OUTPUT_DIR)
    parser.add_argument("--top-n",            type=int, default=10)
    parser.add_argument("--confusion-labels", default=None, metavar="PATH")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading sentiment data from {args.sentiment} ...")
    df = pd.read_csv(args.sentiment)
    df["is_control"] = df["is_control"].astype(bool) if "is_control" in df.columns else False
    print(f"  {len(df):,} articles")

    if "topic_id" not in df.columns:
        if os.path.exists(args.topics):
            print(f"Merging topic assignments from {args.topics} ...")
            topics_df  = pd.read_parquet(args.topics)
            topic_cols = [c for c in topics_df.columns if c.startswith("topic") or c == "url"]
            if "url" in df.columns and "url" in topics_df.columns:
                df = df.merge(topics_df[topic_cols], on="url", how="left")
                print(f"  Merged topic_id for {df['topic_id'].notna().sum():,} articles")
            else:
                print("  ⚠️  Cannot merge: no 'url' column to join on.")
        else:
            print(f"  ⚠️  topics file not found: {args.topics}")

    print(f"Loading topic info from {args.topic_info} ...")
    info_df  = pd.read_csv(args.topic_info)
    info_df  = info_df[info_df["Topic"] != -1].copy()
    name_map = {row["Topic"]: clean_topic_name(row["Name"]) for _, row in info_df.iterrows()}

    if "topic_id" in df.columns:
        df = df[df["topic_id"].notna() & (df["topic_id"] != -1)].copy()
        df["topic_id"]       = df["topic_id"].astype(int)
        df["friendly_name"]  = df["topic_id"].map(name_map).fillna("Unknown")
    else:
        print("  ⚠️  topic_id still missing — topic charts will be skipped")
        df["friendly_name"] = "Unknown"

    ctrl = load_control_data(args.control_dir, name_map)
    if ctrl is not None:
        shared   = [c for c in df.columns if c in ctrl.columns]
        combined = pd.concat([df[shared], ctrl[shared]], ignore_index=True)
    else:
        combined = df.copy()

    print(f"\nGenerating charts → {args.out_dir}")

    chart_sentiment_by_outlet(df, args.out_dir, DPI)
    chart_topics_by_event(df, args.out_dir, DPI, args.top_n)
    chart_topic_pies_per_event(df, args.out_dir, DPI, args.top_n)
    chart_topic_pie(df, args.out_dir, DPI, args.top_n)
    chart_topic_pie_all(df, args.out_dir, DPI)
    chart_sentiment_by_topic(df, args.out_dir, DPI, args.top_n + 2)
    chart_articles_per_event(df, args.out_dir, DPI)
    chart_sentiment_by_event_type(df, args.out_dir, DPI)
    chart_protest_vs_control(combined, args.out_dir, DPI)
    chart_ideological_gap(combined, args.out_dir, DPI)
    chart_language_breakdown(combined, args.out_dir, DPI)
    chart_pipeline_diagram(args.out_dir, DPI)

    print("\nGenerating new diagnostic & fairness charts ...")
    chart_demographic_distribution(combined, args.out_dir, DPI)
    chart_fairness_metric_comparison(combined, args.out_dir, DPI)
    chart_model_performance(df, args.topic_info, args.out_dir, DPI)
    chart_subgroup_comparison(combined, args.out_dir, DPI)
    chart_confusion_matrix(combined, args.out_dir, DPI,
                           labelled_sample_path=args.confusion_labels)
    chart_workflow_diagram(args.out_dir, DPI)

    print("\nGenerating bias report ...")
    generate_bias_report(combined, args.out_dir)

    print(f"\n✅  All outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()