"""
visualise.py — Charts for Women's Protest News Corpus Analysis
===============================================================
Generates publication-ready figures from the sentiment + topic outputs
produced by topic_model.py and sentiment_analysis.py.

Charts produced
---------------
  1.  sentiment_by_outlet.png              — mean sentiment per publisher
  2.  topic_breakdown_by_event.png         — top topics per protest event (bar, NORMALISED)
  3.  topic_pie_{event}.png  (× N events)  — per-event topic donuts using PER-EVENT topic models
  4.  topic_pie_protest_weeks.png          — overall protest weeks donut (NORMALISED)
  5.  topic_pie_control_weeks.png          — overall control weeks donut (NORMALISED)
  6.  topic_pie_all_topics.png             — every global topic, no collapsing (raw counts)
  7.  sentiment_by_topic.png               — mean sentiment × volume scatter
  8.  articles_per_event.png               — article counts per event with gap annotations
  9.  protest_vs_control.png               — protest vs control sentiment per publisher
  10. sentiment_ideological_gap.png        — fairness: protest−control delta by ideology
  11. sentiment_by_event_type.png          — subgroup: single vs sustained events
  12. language_breakdown.png               — EN/DE/FR/ES share per publisher
  13. pipeline_diagram.png                 — workflow diagram for methods section
  14. bias_report.csv                      — representativeness statistics
  15. demographic_distribution.png         — article volume by ideology bin + language group
  16. fairness_cohens_d_comparison.png     — Cohen's d effect size per publisher
  17. model_performance.png               — topic model coherence / coverage diagnostics
  18. fairness_subgroup_intersection.png   — sentiment heatmap: event × ideology
  19. confusion_matrix_sentiment.png       — sentiment label distribution / confusion matrix
  20. workflow_diagram.png                 — extended pipeline diagram
  21. topic_pie_control_{event}.png (× N)  — per-event CONTROL week topic donuts
  22. topic_pie_control_weeks_normalised.png — overall normalised control week topic breakdown

Per-event topic model note
--------------------------
Charts 2 and 3 now use PER-EVENT topic models (event_topic_label column) rather
than the global model. Each event has its own independent topic vocabulary discovered
from only that event's articles, so the pie for the Polish Women's Strike shows topics
specific to that news cycle, not global topics that happened to appear in that week.

The global model is still used for cross-event comparisons (charts 4–7) where a shared
topic vocabulary is needed.

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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# ❶  CONFIGURATION & STYLING
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SENTIMENT  = os.path.join("analysis_output", "articles_with_sentiment.csv")
DEFAULT_TOPIC_INFO = os.path.join("analysis_output", "topic_info.csv")
DEFAULT_CONTROL_TOPIC_INFO = os.path.join("analysis_output", "control_topic_info.csv")
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
PALETTE = [
    "#7F77DD", "#1D9E75", "#D85A30", "#378ADD", "#D4537E", "#639922",
    "#BA7517", "#888780", "#5DCAA5", "#F0997B", "#AFA9EC", "#9FE1CB",
]

_AS_LEFT       = -1.5
_AS_LEAN_LEFT  = -0.75
_AS_CENTER     =  0.0
_AS_LEAN_RIGHT = +0.75
_AS_RIGHT      = +1.5
_AS_FAR_LEFT   = -2.0
_AS_FAR_RIGHT  = +2.0

IDEOLOGY_SCORES: dict[str, float] = {
    "APNews":           _AS_CENTER,
    "Reuters":          _AS_CENTER,
    "VoiceOfAmerica":   _AS_LEAN_LEFT,
    "WashingtonPost":   _AS_LEAN_LEFT,
    "TheNewYorker":     _AS_LEFT,
    "TheNation":        _AS_FAR_LEFT,
    "TheIntercept":     _AS_LEFT,
    "RollingStone":     _AS_LEFT,
    "LATimes":          _AS_LEAN_LEFT,
    "BusinessInsider":  _AS_LEAN_LEFT,
    "CNBC":             _AS_CENTER,
    "FoxNews":          _AS_RIGHT,
    "WashingtonTimes":  _AS_LEAN_RIGHT,
    "FreeBeacon":       _AS_LEAN_RIGHT,
    "TheGatewayPundit": _AS_FAR_RIGHT,
    "BBC":              _AS_CENTER,
    "TheGuardian":      _AS_LEFT,
    "TheIndependent":   _AS_LEAN_LEFT,
    "iNews":            _AS_LEAN_LEFT,
    "DailyMail":        _AS_RIGHT,
    "TheTelegraph":     _AS_LEAN_RIGHT,
    "TheSun":           _AS_RIGHT,
    "EuronewsEN":       _AS_CENTER,
    "EuronewsFR":       _AS_CENTER,
    "DW":               _AS_CENTER,
    "SpiegelOnline":    _AS_LEAN_LEFT,
    "DieZeit":          _AS_LEAN_LEFT,
    "Tagesschau":       _AS_CENTER,
    "FAZ":              _AS_LEAN_RIGHT,
    "Taz":              _AS_FAR_LEFT,
    "DerStandard":      _AS_LEAN_LEFT,
    "DiePresse":        _AS_LEAN_RIGHT,
    "ORF":              _AS_CENTER,
    "CBCNews":          _AS_LEAN_LEFT,
    "NationalPost":     _AS_LEAN_RIGHT,
    "TheGlobeAndMail":  _AS_CENTER,
    "LeMonde":          _AS_LEAN_LEFT,
    "LeFigaro":         _AS_LEAN_RIGHT,
    "ElPais":           _AS_LEAN_LEFT,
    "ElMundo":          _AS_LEAN_RIGHT,
    "ElDiario":         _AS_LEFT,
    "LaVanguardia":     _AS_CENTER,
    "ABC":              _AS_RIGHT,
    "Publico":          _AS_LEFT,
}

_IDEOLOGY_LOWER: dict[str, float] = {k.lower(): v for k, v in IDEOLOGY_SCORES.items()}


def _ideology_score(publisher: str) -> float:
    """Case-insensitive ideology score lookup. Falls back to NaN if unknown."""
    s = str(publisher)
    return IDEOLOGY_SCORES.get(s, _IDEOLOGY_LOWER.get(s.lower(), np.nan))


EVENT_COVERAGE_NOTES: dict[str, str] = {
    "Womens_March_2017":               "CC-News sparse pre-2019",
    "International_Womens_Strike_2017": "CC-News sparse pre-2019",
    "MeToo_Protests_2017":             "CC-News sparse pre-2019",
    "IWD_2018_Global":                 "CC-News sparse pre-2019",
    "Swiss_Womens_Strike_2019":        "Smaller national event",
    "Polish_Womens_Strike_2020":       "Strong EU coverage",
    "Sarah_Everard_Vigils_2021":       "Primarily UK coverage",
    "Roe_Leak_Protests_2022":          "High US publisher density",
    "Women_Life_Freedom_2022":         "Global coverage",
    "Israeli_Womens_Protests_2023":    "Primarily IL/EU coverage",
}

FEMINIST_KEYWORDS = [
    "women", "feminist", "feminism", "frauen", "mujeres", "femmes", "gender",
    "equality", "sexism", "misogyn", "patriarch", "suffrag",
    "rights", "reproductive", "bodily", "autonomy",
    "abortion", "roe", "wade", "pro-choice", "prochoice", "pro choice",
    "pro-life", "prolife", "pro life",
    "choice", "planned parenthood", "contraception", "birth control",
    "domestic", "violence", "assault", "harassment", "rape", "metoo", "me too",
    "everard", "femicide",
    "familia", "peace", "strike", "greve", "huelga", "streik",
    "march", "rally", "uprising",
    "frauen", "gleichstellung", "igualdad", "egalite",
]


# ══════════════════════════════════════════════════════════════════════════════
# ❷  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_OTHER_LABELS = {"other", "outlier", "unknown", "all other topics"}


def _normalise_other(label: str) -> str:
    if str(label).strip().lower() in _OTHER_LABELS:
        return "All Other Topics"
    return label


def clean_topic_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return "Unknown"
    if name.strip().lower() in _OTHER_LABELS or name.strip() == "-1":
        return "All Other Topics"
    parts = name.split("_")
    start_idx = 1 if parts[0].replace("-", "").isdigit() else 0
    words = [w for w in parts[start_idx : start_idx + 3] if w]
    if not words:
        return name.capitalize()
    normalised = sorted(w.capitalize() for w in words)
    return ", ".join(normalised)


def is_feminist(name: str) -> bool:
    return any(k in str(name).lower() for k in FEMINIST_KEYWORDS)

def _subnote(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.01, text, ha="center", fontsize=9,
             color=DARK_PLUM, alpha=0.65, style="italic")

def _base_style(fig, ax, grid_axis: str = "both") -> None:
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM)
    if grid_axis == "none":
        ax.grid(visible=False)
    else:
        ax.grid(visible=True, axis=grid_axis, color=BLUSH,
                linewidth=0.8, linestyle="--", alpha=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(DARK_PLUM)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=DARK_PLUM, width=0.8)


def _save(fig: plt.Figure, path: str, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _make_donut(ax, slices, pie_colors, explode, edge_colors=None):
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


def _slice_colors(slices: pd.Series) -> tuple[list, list, list]:
    pie_colors, explode, edge_colors, non_fem_idx = [], [], [], 0
    for lbl in slices.index:
        if lbl == "All Other Topics":
            pie_colors.append("#D1C4E9"); explode.append(0)
            edge_colors.append("white")
        elif is_feminist(lbl):
            pie_colors.append(FEMINIST_COLOUR); explode.append(0.08)
            edge_colors.append(FEMINIST_OUTLINE)
        else:
            pie_colors.append(PALETTE[non_fem_idx % len(PALETTE)]); explode.append(0)
            edge_colors.append("white")
            non_fem_idx += 1
    return pie_colors, explode, edge_colors


def _build_slices(counts: pd.Series, top_n: int) -> pd.Series:
    pre_other = counts[counts.index.map(lambda l: l == "All Other Topics")].sum()
    counts = counts[counts.index.map(lambda l: l != "All Other Topics")]
    top     = counts.head(top_n)
    rest    = counts.iloc[top_n:]
    rescued = rest[rest.index.map(is_feminist)]
    other_n = rest[~rest.index.map(is_feminist)].sum() + pre_other
    slices  = pd.concat([top, rescued])
    if other_n > 0:
        slices["All Other Topics"] = other_n
    return slices


def _build_slices_normalised(df: pd.DataFrame, top_n: int,
                              subset_col: str | None = None,
                              label_col: str = "friendly_name") -> pd.Series:
    if label_col not in df.columns:
        return pd.Series(dtype=float)

    df = df.copy()
    df[label_col] = df[label_col].map(_normalise_other)

    if subset_col and subset_col in df.columns:
        groups = []
        for _, grp in df.groupby(subset_col):
            props = grp[label_col].value_counts(normalize=True)
            groups.append(props)
        if not groups:
            return pd.Series(dtype=float)
        mean_props = pd.concat(groups, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    else:
        mean_props = df[label_col].value_counts(normalize=True).sort_values(ascending=False)

    pre_other = mean_props.get("All Other Topics", 0.0)
    mean_props = mean_props[mean_props.index != "All Other Topics"]

    top     = mean_props.head(top_n)
    rest    = mean_props.iloc[top_n:]
    rescued = rest[rest.index.map(is_feminist)]
    other_v = rest[~rest.index.map(is_feminist)].sum() + pre_other
    slices  = pd.concat([top, rescued])
    if other_v > 0:
        slices["All Other Topics"] = other_v
    return slices


def _donut_with_legend(fig, ax, slices, pie_colors, explode, edge_colors,
                        title, subtitle="", pct_label="%") -> None:
    fig.patch.set_facecolor(LAVENDER_CREAM)
    wedges = _make_donut(ax, slices, pie_colors, explode, edge_colors)

    def _entry(lbl, val):
        star    = "★ " if is_feminist(lbl) else "   "
        val_str = f"{val*100:.1f}%" if pct_label == "%" else f"{int(val):,}"
        return f"{star}{lbl} — {val_str}"

    legend_labels = [_entry(l, v) for l, v in zip(slices.index, slices.values)]
    leg = ax.legend(wedges, legend_labels, title="Topics",
                    loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                    fontsize=9, frameon=False,
                    prop={"family": "monospace"})
    for i, text in enumerate(leg.get_texts()):
        if is_feminist(slices.index[i]):
            text.set_fontweight("bold")
            text.set_color(FEMINIST_OUTLINE)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=18, color=DARK_PLUM)
    if subtitle:
        fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9,
                 color=DARK_PLUM, alpha=0.65, style="italic")


def _load_event_model_notes(topic_info_dir: str) -> dict[str, dict]:
    path = os.path.join(topic_info_dir, "event_model_notes.csv")
    if not os.path.exists(path):
        return {}
    try:
        notes = pd.read_csv(path).set_index("event_label").to_dict(orient="index")
        sparse = [e for e, v in notes.items() if v.get("is_sparse")]
        if sparse:
            print(f"  ℹ️  Sparse events (fallback min_topic_size): {sparse}")
        return notes
    except Exception as e:
        print(f"  ⚠️  Could not load event_model_notes.csv: {e}")
        return {}


def _event_footnote(event: str, model_notes: dict, n_articles: int,
                     model_note: str = "") -> str:
    parts = [f"n={n_articles:,} articles", model_note]
    note  = model_notes.get(event, {})
    if note.get("is_sparse"):
        used = note.get("min_topic_size_used", "?")
        req  = note.get("requested_size", "?")
        parts.append(f"sparse corpus — fallback min_size={used} (requested {req})")
    return " · ".join(p for p in parts if p)


def _load_per_event_topic_names(topic_info_dir: str) -> dict[str, dict[int, str]]:
    maps: dict[str, dict[int, str]] = {}
    pattern = os.path.join(topic_info_dir, "*_topic_info.csv")
    for path in glob.glob(pattern):
        try:
            tdf = pd.read_csv(path)
            if "event_label" in tdf.columns:
                event = tdf["event_label"].iloc[0]
            else:
                event = os.path.basename(path).replace("_topic_info.csv", "")
            maps[event] = {
                int(row["Topic"]): clean_topic_name(row["Name"])
                for _, row in tdf.iterrows()
                if row["Topic"] != -1
            }
        except Exception as e:
            print(f"  ⚠️  Could not load {path}: {e}")
    return maps


# ══════════════════════════════════════════════════════════════════════════════
# ❸  LOAD CONTROL DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_control_data(control_dir: str, name_map: dict) -> "pd.DataFrame | None":
    if not os.path.isdir(control_dir):
        print(f"  ⚠️  Control dir not found: {control_dir} — protest vs control skipped.")
        return None
        
    csv_files = [f for f in glob.glob(os.path.join(control_dir, "**", "*.csv"), recursive=True)
                 if "incremental" not in f]
                 
    if not csv_files:
        print(f"  ⚠️  No CSVs in {control_dir} — protest vs control skipped.")
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
    
    # ── UPDATED MAPPING BLOCK INSIDE load_control_data ──
    control_info_path = os.path.join("analysis_output", "control_topic_info.csv")
    control_name_map = {}
    
    if os.path.exists(control_info_path):
        ctdf = pd.read_csv(control_info_path)
        # Standardize matching key columns
        topic_col = "Topic" if "Topic" in ctdf.columns else "topic"
        name_col = "Name" if "Name" in ctdf.columns else "name"
        
        control_name_map = {
            int(row[topic_col]): clean_topic_name(row[name_col]) 
            for _, row in ctdf.iterrows() 
            if row[topic_col] != -1 and pd.notna(row[topic_col])
        }
        print("  ℹ️  Loaded isolated control topic naming dictionary.")
        
    # Crucial Fix: Force the column population on the control data directly!
    if "topic_id" in ctrl.columns and control_name_map:
        ctrl["friendly_name"] = ctrl["topic_id"].map(control_name_map)
    elif "topic_label" in ctrl.columns:
        # Fallback if your processing script saved strings directly to topic_label
        ctrl["friendly_name"] = ctrl["topic_label"]

    # Fill remaining blanks so the data column structurally exists
    if "friendly_name" not in ctrl.columns:
        ctrl["friendly_name"] = "Unknown"
    else:
        ctrl["friendly_name"] = ctrl["friendly_name"].fillna("Unknown")
    # ──────────────────────────────────────────────────────────────────
        
    if "url" in ctrl.columns:
        ctrl.drop_duplicates(subset=["url"], inplace=True)
        
    ctrl["is_control"] = True
    
    if "publisher" in ctrl.columns:
        ctrl["publisher"] = (
            ctrl["publisher"].astype(str)
            .str.extract(r"\.([A-Za-z0-9]+)")[0]
            .fillna(ctrl["publisher"].astype(str))
        )
        
    if "sentiment_score" not in ctrl.columns:
        ctrl["sentiment_score"] = 0.0
        
    if "sentiment_label" not in ctrl.columns and "sentiment_score" in ctrl.columns:
        ctrl["sentiment_label"] = ctrl["sentiment_score"].apply(
            lambda s: "positive" if s >= 0.05 else ("negative" if s <= -0.05 else "neutral")
        )
        
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
# ❺  CHART 2 — TOP TOPICS BY EVENT (bar) — uses per-event topic labels
# ══════════════════════════════════════════════════════════════════════════════

def chart_topics_by_event(df: pd.DataFrame, out_dir: str, dpi: int,
                           top_n: int, event_topic_maps: dict,
                           model_notes: dict | None = None) -> None:
    model_notes = model_notes or {}
    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    events = sorted(protest_df["event_label"].dropna().unique())
    if not events:
        return

    fig, axes = plt.subplots(1, len(events), figsize=(8 * len(events), 8), sharey=False)
    if len(events) == 1:
        axes = [axes]

    for ax, event in zip(axes, events):
        _base_style(fig, ax)
        sub = protest_df[protest_df["event_label"] == event].copy()

        label_col = _attach_event_friendly_names(sub, event, event_topic_maps)

        counts = sub[label_col].value_counts(normalize=True) * 100
        top_l  = counts.head(top_n).index.tolist()
        fem_l  = [l for l in counts.index if is_feminist(l) and l not in top_l]
        data   = counts.loc[top_l + fem_l].sort_values()

        ax.barh(data.index, data.values,
                color=[FEMINIST_COLOUR if is_feminist(l) else DEEP_PURPLE for l in data.index])
        ax.set_title(event.replace("_", " ").title(), fontweight="bold", color=DARK_PLUM)
        ax.set_xlabel("% of Articles (within-event)", color=DARK_PLUM)

        note = model_notes.get(event, {})
        if note.get("is_sparse"):
            used = note.get("min_topic_size_used", "?")
            ax.set_xlabel(
                f"% of Articles (within-event)\n"
                f"* sparse corpus — fallback min_size={used}",
                color=DARK_PLUM, fontsize=8,
            )

    fig.suptitle("Key Topics Per Protest Event — Per-Event Topic Models\n"
                 "(each event's topics discovered independently)",
                 fontsize=16, fontweight="bold", y=1.02, color=DARK_PLUM)
    fig.legend(handles=[mpatches.Patch(color=FEMINIST_COLOUR, label="Feminist/protest"),
                         mpatches.Patch(color=DEEP_PURPLE, label="Other")],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=10)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "topic_breakdown_by_event.png"), dpi)


def _attach_event_friendly_names(sub: pd.DataFrame, event: str,
                                  event_topic_maps: dict) -> str:
    has_event_topics = False
    if event in event_topic_maps and "event_topic_id" in sub.columns:
        if (sub["event_topic_id"] != -1).any():
            has_event_topics = True
    elif "event_topic_label" in sub.columns:
        if sub["event_topic_label"].str.lower().ne("outlier").any():
            has_event_topics = True

    if has_event_topics:
        if event in event_topic_maps and "event_topic_id" in sub.columns:
            emap = event_topic_maps[event]
            sub["event_topic_friendly_name"] = sub["event_topic_id"].map(
                lambda tid: emap.get(int(tid), "All Other Topics")
                if pd.notna(tid) and tid != -1 else "All Other Topics"
            )
        else:
            sub["event_topic_friendly_name"] = sub["event_topic_label"].apply(clean_topic_name)
        return "event_topic_friendly_name"
    else:
        return "friendly_name"


# ══════════════════════════════════════════════════════════════════════════════
# ❻  CHART 3 — PER-EVENT TOPIC PIES — uses per-event topic models
# ══════════════════════════════════════════════════════════════════════════════

def chart_topic_pies_per_event(df: pd.DataFrame, out_dir: str, dpi: int,
                                top_n: int, event_topic_maps: dict,
                                model_notes: dict | None = None) -> None:
    model_notes = model_notes or {}
    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df

    for event in sorted(protest_df["event_label"].dropna().unique()):
        sub = protest_df[protest_df["event_label"] == event].copy()

        label_col = _attach_event_friendly_names(sub, event, event_topic_maps)

        if label_col == "friendly_name":
            note = model_notes.get(event, {})
            status = note.get("status", "unknown")
            if status == "failed":
                fallback_reason = "per-event model failed — showing global model topics"
            else:
                fallback_reason = "no per-event topics found — showing global model topics"
            print(f"  ℹ️  [{event}] {fallback_reason}")
            model_note = f"global topic model (fallback: {fallback_reason})"
        else:
            model_note = "per-event topic model"

        slices = _build_slices_normalised(sub, top_n, label_col=label_col)
        if slices.empty:
            continue

        colors, explode, edges = _slice_colors(slices)
        fig, ax = plt.subplots(figsize=(13, 8))
        subtitle   = _event_footnote(event, model_notes, len(sub), model_note)
        _donut_with_legend(
            fig, ax, slices, colors, explode, edges,
            title=f"Topic Distribution — {event.replace('_', ' ').title()}",
            subtitle=subtitle,
            pct_label="%",
        )
        safe = event.replace(" ", "_").replace("/", "-")
        _save(fig, os.path.join(out_dir, f"topic_pie_{safe}.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❼  CHART 4 — OVERALL TOPIC PIE — global model, normalised across events
# ══════════════════════════════════════════════════════════════════════════════

def chart_topic_pie(df: pd.DataFrame, out_dir: str, dpi: int, top_n: int) -> None:
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topic_pie: 'friendly_name' missing — skipping.")
        return

    if "is_control" in df.columns:
        protest_mask = df["is_control"] == False
        control_mask = df["is_control"] == True
    else:
        protest_mask = pd.Series([True] * len(df), index=df.index)
        control_mask = pd.Series([False] * len(df), index=df.index)

    subsets = [
        {"df": df[protest_mask], "filename": "topic_pie_protest_weeks.png",
         "title": "Topic Distribution: Protest Weeks (Global Model, Normalised)",
         "subtitle": "Mean proportion per event · global topic vocabulary"},
        {"df": df[control_mask], "filename": "topic_pie_control_weeks.png",
         "title": "Topic Distribution: Control Weeks (Global Model, Normalised)",
         "subtitle": "Mean proportion per event · global topic vocabulary"},
    ]

    for item in subsets:
        sub_df = item["df"]
        if sub_df.empty:
            continue
        slices = _build_slices_normalised(sub_df, top_n, subset_col="event_label")
        if slices.empty:
            continue
        colors, explode, edges = _slice_colors(slices)
        fig, ax = plt.subplots(figsize=(14, 9))
        _donut_with_legend(fig, ax, slices, colors, explode, edges,
                           title=item["title"], subtitle=item["subtitle"], pct_label="%")
        _save(fig, os.path.join(out_dir, item["filename"]), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❽  CHART 5 — ALL TOPICS PIE (no collapsing — absolute counts)
# ══════════════════════════════════════════════════════════════════════════════

def chart_topic_pie_all(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topic_pie_all: 'friendly_name' missing — skipping."); return

    df = df.copy()
    df["friendly_name"] = df["friendly_name"].map(_normalise_other)
    counts = df["friendly_name"].value_counts()
    pie_colors, explode, non_fem_idx = [], [], 0
    for lbl in counts.index:
        if is_feminist(lbl):
            pie_colors.append(FEMINIST_COLOUR); explode.append(0.05)
        else:
            pie_colors.append(PALETTE[non_fem_idx % len(PALETTE)]); explode.append(0)
            non_fem_idx += 1

    fig, ax = plt.subplots(figsize=(16, max(10, len(counts) * 0.28)))
    fig.patch.set_facecolor(LAVENDER_CREAM)

    wedges, _ = ax.pie(counts.values, startangle=140, colors=pie_colors, explode=explode,
                       wedgeprops={"edgecolor": "white", "linewidth": 1.2, "width": 0.5})
    leg = ax.legend(wedges, [f"{l} ({int(v):,})" for l, v in zip(counts.index, counts.values)],
                    title="All Topics (raw counts)", loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8, frameon=False)
    for i, text in enumerate(leg.get_texts()):
        if is_feminist(counts.index[i]):
            text.set_fontweight("bold"); text.set_color(FEMINIST_OUTLINE)

    ax.set_title("All Discovered Topics — Global Model, Raw Counts\n"
                 "(not normalised; for topic model inventory only)",
                 fontsize=14, fontweight="bold", pad=20, color=DARK_PLUM)
    _save(fig, os.path.join(out_dir, "topic_pie_all_topics.png"), dpi)


def chart_topic_pie_all_normalised(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topic_pie_all_normalised: 'friendly_name' missing — skipping."); return

    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    protest_df = protest_df.copy()
    protest_df["friendly_name"] = protest_df["friendly_name"].map(_normalise_other)

    if "event_label" in protest_df.columns:
        groups = []
        for _, grp in protest_df.groupby("event_label"):
            props = grp["friendly_name"].value_counts(normalize=True)
            groups.append(props)
        mean_props = (pd.concat(groups, axis=1).fillna(0)
                      .mean(axis=1).sort_values(ascending=False))
    else:
        mean_props = protest_df["friendly_name"].value_counts(normalize=True)

    pie_colors, explode, edge_colors, non_fem_idx = [], [], [], 0
    for lbl in mean_props.index:
        if lbl == "All Other Topics":
            pie_colors.append("#D1C4E9"); explode.append(0); edge_colors.append("white")
        elif is_feminist(lbl):
            pie_colors.append(FEMINIST_COLOUR); explode.append(0.05)
            edge_colors.append(FEMINIST_OUTLINE)
        else:
            pie_colors.append(PALETTE[non_fem_idx % len(PALETTE)]); explode.append(0)
            edge_colors.append("white"); non_fem_idx += 1

    fig, ax = plt.subplots(figsize=(16, max(10, len(mean_props) * 0.28)))
    fig.patch.set_facecolor(LAVENDER_CREAM)

    wedges, _, _ = ax.pie(
        mean_props.values, startangle=140, colors=pie_colors, explode=explode,
        autopct="%1.1f%%", pctdistance=0.82,
        textprops={"color": DARK_PLUM, "fontsize": 7, "fontweight": "bold"},
        wedgeprops={"linewidth": 1.2, "width": 0.5},
    )
    for wedge, ec in zip(wedges, edge_colors):
        wedge.set_edgecolor(ec)

    legend_labels = []
    for lbl, val in zip(mean_props.index, mean_props.values):
        prefix = "★ " if is_feminist(lbl) else "   "
        legend_labels.append(f"{prefix}{lbl} — {val*100:.1f}%")

    leg = ax.legend(wedges, legend_labels,
                    title="All Topics (normalised)", loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8, frameon=False,
                    prop={"family": "monospace"})
    for i, text in enumerate(leg.get_texts()):
        if is_feminist(mean_props.index[i]):
            text.set_fontweight("bold"); text.set_color(FEMINIST_OUTLINE)
        else:
            text.set_color(DARK_PLUM)

    ax.set_title("All Discovered Topics — Global Model, Normalised",
                 fontsize=14, fontweight="bold", pad=20, color=DARK_PLUM)
    n_events = protest_df["event_label"].nunique() if "event_label" in protest_df.columns else "?"
    _subnote(fig,
             f"Mean proportion per event ({n_events} events weighted equally) · "
             "directly comparable to raw-counts version · ★ = feminist/protest topic")
    _save(fig, os.path.join(out_dir, "topic_pie_all_topics_normalised.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ❾  CHART 6 — SENTIMENT BY TOPIC
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
            m, c, ec, ew, z = "o", (RASPBERRY if row["mean_sentiment"] < 0 else MUTED_TEAL), DARK_PLUM, 0.5, 3
        ax.scatter(row["mean_sentiment"], row["friendly_name"],
                   s=np.clip(row["count"] * 5, 60, 1200),
                   c=c, marker=m, edgecolors=ec, linewidths=ew, alpha=0.88, zorder=z)
    ax.axvline(0, color=DARK_PLUM, linestyle="--", alpha=0.5)
    ax.set_title("Mean Sentiment by Topic — Global Model (dot size = volume)",
                 fontweight="bold", fontsize=14, color=DARK_PLUM)
    ax.set_xlabel("← More Negative    Sentiment Score    More Positive →", color=DARK_PLUM)
    ax.legend(handles=[
        mpatches.Patch(color=FEMINIST_COLOUR, label="Feminist/protest (◆)"),
        mpatches.Patch(color=RASPBERRY,       label="Negative (other)"),
        mpatches.Patch(color=MUTED_TEAL,      label="Positive (other)"),
    ], frameon=False, fontsize=9)
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
        label = f"  {n:,}" + (f"  ·  {note}" if note else "")
        ax.text(bar.get_width() + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=8.5, color=DARK_PLUM)

    ax.set_xlabel("Number of Articles", color=DARK_PLUM)
    ax.set_title("Articles Collected Per Protest Event",
                 fontweight="bold", fontsize=13, color=DARK_PLUM)
    ax.set_xlim(0, counts.max() * 1.55)
    ax.axvline(median_n, color=DARK_PLUM, linestyle=":", linewidth=1, alpha=0.6)
    _save(fig, os.path.join(out_dir, "articles_per_event.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓫  CHART 8 — PROTEST vs CONTROL
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
    ax.set_ylabel("Mean Sentiment Score", color=DARK_PLUM)
    ax.set_title("Sentiment: Protest Weeks vs Matched Control Weeks by Publisher",
                 fontweight="bold", fontsize=14, color=DARK_PLUM)
    ax.legend(frameon=False, fontsize=10)
    _save(fig, os.path.join(out_dir, "protest_vs_control.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓬  CHART 9 — IDEOLOGICAL FAIRNESS GAP
# ══════════════════════════════════════════════════════════════════════════════

def chart_ideological_gap(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    if "is_control" not in df.columns or df["is_control"].nunique() < 2:
        print("  ⚠️  chart_ideological_gap: need both is_control values — skipping."); return
    agg = df.groupby(["publisher","is_control"])["sentiment_score"].mean().unstack("is_control")
    agg.columns = ["protest","control"]
    agg = agg.dropna()
    agg["gap"]      = agg["protest"] - agg["control"]
    agg["ideology"] = agg.index.map(_ideology_score)
    agg = agg.dropna(subset=["ideology"])
    if agg.empty:
        print("  ⚠️  chart_ideological_gap: no publishers matched — skipping."); return

    def _ic(s): return MUTED_TEAL if s < -0.5 else (RASPBERRY if s > 0.5 else DEEP_PURPLE)
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
                linestyle="--", alpha=0.5, label="Trend")
    ax.axhline(0, color=DARK_PLUM, linewidth=0.8, linestyle=":")
    ax.axvline(0, color=DARK_PLUM, linewidth=0.5, linestyle=":", alpha=0.4)
    ax.set_xlabel("← Left-leaning    Ideological Score    Right-leaning →", color=DARK_PLUM)
    ax.set_ylabel("Sentiment Gap (protest − control week)", color=DARK_PLUM)
    ax.set_title("Ideological Fairness Metric\n"
                 "Does political leaning predict more negative protest coverage?",
                 fontweight="bold", fontsize=13, color=DARK_PLUM)
    ax.legend(handles=[
        mpatches.Patch(color=MUTED_TEAL,  label="Left-leaning"),
        mpatches.Patch(color=DEEP_PURPLE, label="Centre"),
        mpatches.Patch(color=RASPBERRY,   label="Right-leaning"),
    ] + ([Line2D([0],[0], color=DARK_PLUM, linestyle="--", alpha=0.5, label="Trend")]
          if len(agg) >= 3 else []),
    frameon=False, fontsize=9)
    _save(fig, os.path.join(out_dir, "sentiment_ideological_gap.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓭  CHART 10 — SENTIMENT BY EVENT TYPE
# ══════════════════════════════════════════════════════════════════════════════

def chart_sentiment_by_event_type(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    if "event_type" not in df.columns:
        print("  ⚠️  chart_sentiment_by_event_type: 'event_type' missing — skipping."); return
    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    agg = (protest_df.groupby(["event_type","event_label"])["sentiment_score"]
           .mean().reset_index())
    event_types = sorted(agg["event_type"].unique())
    fig, axes = plt.subplots(1, len(event_types), figsize=(7 * len(event_types), 6))
    if len(event_types) == 1:
        axes = [axes]
    type_colors = {"single": DEEP_PURPLE, "sustained": RASPBERRY, "recurring": MUTED_TEAL}
    for ax, etype in zip(axes, event_types):
        _base_style(fig, ax)
        sub = agg[agg["event_type"] == etype].sort_values("sentiment_score")
        ax.barh(sub["event_label"].str.replace("_", " "), sub["sentiment_score"],
                color=type_colors.get(etype, DEEP_PURPLE), alpha=0.85)
        ax.axvline(0, color=DARK_PLUM, linewidth=0.8)
        ax.set_title(f"Event type: {etype.title()}", fontweight="bold", color=DARK_PLUM)
        ax.set_xlabel("Mean Sentiment", color=DARK_PLUM)
    fig.suptitle("Mean Sentiment by Event Type", fontsize=14, fontweight="bold", color=DARK_PLUM)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "sentiment_by_event_type.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓮  CHART 11 — LANGUAGE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

def chart_language_breakdown(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    if "language" not in df.columns:
        print("  ⚠️  chart_language_breakdown: 'language' missing — skipping."); return
    lang_pal = {"EN": DEEP_PURPLE, "DE": RASPBERRY, "FR": MUTED_TEAL,
                "ES": FEMINIST_COLOUR, "OTHER": "#BDBDBD"}
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
        ax.bar(x, vals, bottom=bottom, color=lang_pal.get(lang,"#BDBDBD"), label=lang, alpha=0.9)
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
    steps = [
        ("collect_articles\n_optimized.py", "Crawl CC-News\n45 publishers × 10 events\n+ matched control weeks"),
        ("build_corpus.py",                 "Merge per-event parquets\nDeduplicate by URL"),
        ("sentiment_analysis.py",           "Lingua detection\nopus-mt translation\nVADER scoring"),
        ("topic_model.py",                  "all-MiniLM-L6-v2\nBERTopic (HDBSCAN)\nOutlier reduction"),
        ("visualise.py",                    "22 publication figures\nbias_report.csv"),
    ]
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM); ax.axis("off")
    n, xs, y_box, bw, bh = len(steps), np.linspace(0.08, 0.92, len(steps)), 0.5, 0.14, 0.55
    for i, (script, desc) in enumerate(steps):
        x = xs[i]
        ax.add_patch(plt.Rectangle((x-bw/2, y_box-bh/2), bw, bh,
                                    facecolor=DEEP_PURPLE, edgecolor=BLUSH, linewidth=1.5, zorder=2))
        ax.text(x, y_box+0.08, script, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white", zorder=3)
        ax.text(x, y_box-0.18, desc, ha="center", va="center", fontsize=7, color=DARK_PLUM, zorder=3,
                bbox=dict(facecolor=LAVENDER_CREAM, edgecolor="none", pad=1))
        if i < n-1:
            ax.annotate("", xy=(xs[i+1]-bw/2-0.005, y_box), xytext=(x+bw/2+0.005, y_box),
                        arrowprops=dict(arrowstyle="->", color=RASPBERRY, lw=1.8), zorder=4)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title("Analysis Pipeline", fontsize=14, fontweight="bold", color=DARK_PLUM, pad=10)
    _save(fig, os.path.join(out_dir, "pipeline_diagram.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓰  BIAS REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_bias_report(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sections = []

    cov = df.groupby(["publisher","event_label"]).size().reset_index(name="n_articles")
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
        bal = bal.reset_index()
        if False in bal.columns and True in bal.columns:
            bal.rename(columns={False: "n_protest", True: "n_control"}, inplace=True)
        if "n_protest" in bal.columns and "n_control" in bal.columns:
            bal["ratio_protest_to_control"] = (
                bal["n_protest"] / bal["n_control"].replace(0, np.nan)).round(2)
            bal["flag_imbalanced"] = bal["ratio_protest_to_control"] > 2
        bal["section"] = "C_protest_control_balance"
        sections.append(bal)

    if "topic_id_raw" in df.columns and "topic_id" in df.columns:
        n = len(df)
        nr = (df["topic_id_raw"] == -1).sum()
        nf = (df["topic_id"] == -1).sum()
        sections.append(pd.DataFrame([{
            "n_total": n, "n_raw_outliers": int(nr),
            "pct_raw_outliers": round(100*nr/n, 2),
            "n_final_outliers": int(nf),
            "pct_final_outliers": round(100*nf/n, 2),
            "n_reassigned": int(nr-nf), "section": "D_outlier_rate",
        }]))

    pub_counts = df["publisher"].value_counts().reset_index()
    pub_counts.columns = ["publisher","n_articles"]
    pub_counts["ideology_score"] = pub_counts["publisher"].apply(_ideology_score)
    pub_counts["ideology_bin"] = pd.cut(
        pub_counts["ideology_score"], bins=[-3,-0.5,0.5,3],
        labels=["left","centre","right"])
    pub_counts["section"] = "E_ideological_balance"
    sections.append(pub_counts)

    combined = pd.concat(sections, ignore_index=True)
    out_path = os.path.join(out_dir, "bias_report.csv")
    combined.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    print("\n  ── Bias report summary ─────────────────────────────────")
    print(f"  Total articles        : {len(df):,}")
    if "language" in df.columns:
        print(f"  Language distribution : {df['language'].value_counts().to_dict()}")
    if "topic_id_raw" in df.columns:
        print(f"  Outlier rate (raw)    : {100*(df['topic_id_raw']==-1).mean():.1f}%")
        print(f"  Outlier rate (final)  : {100*(df['topic_id']==-1).mean():.1f}%")
    ideol = pub_counts.groupby("ideology_bin", observed=True)["n_articles"].sum()
    print(f"  Articles by ideology  : {ideol.to_dict()}")
    low_cov = cov[cov["flag_low_coverage"]]
    if not low_cov.empty:
        print(f"  ⚠️  {len(low_cov)} publisher×event combinations have < 5 articles")
    print("  ────────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# ⓱  CHART 13 — DEMOGRAPHIC DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def chart_demographic_distribution(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    if "language" not in df.columns:
        print("  ⚠️  chart_demographic_distribution: 'language' missing — skipping."); return
    protest_df = (df[~df["is_control"]] if "is_control" in df.columns else df).copy()

    lang_pal = {"EN": DEEP_PURPLE, "DE": RASPBERRY, "FR": MUTED_TEAL,
                "ES": FEMINIST_COLOUR, "OTHER": "#BDBDBD"}

    OUTLET_REGION: dict[str, str] = {
        "Associated Press News":        "USA",
        "CNBC":                         "USA",
        "Voice Of America":             "USA",
        "Washington Post":              "USA",
        "The New Yorker":               "USA",
        "The Nation":                   "USA",
        "The Intercept":                "USA",
        "Rolling Stone":                "USA",
        "Business Insider":             "USA",
        "Fox News":                     "USA",
        "The Washington Times":         "USA",
        "The Gateway Pundit":           "USA",
        "The BBC":                      "UK/International",
        "The Guardian":                 "UK/International",
        "The Independent":              "UK/International",
        "Daily Mail":                   "UK/International",
        "The Telegraph":                "UK/International",
        "The Sun":                      "UK/International",
        "i":                            "UK/International",
        "Reuters":                      "UK/International",
        "Deutsche Welle":               "Germany/DACH",
        "Spiegel Online":               "Germany/DACH",
        "Die Zeit":                     "Germany/DACH",
        "Tagesschau":                   "Germany/DACH",
        "Frankfurter Allgemeine Zeitung": "Germany/DACH",
        "Die Tageszeitung (taz)":       "Germany/DACH",
        "CBC News":                     "Canada",
        "The Globe and Mail":           "Canada",
        "Le Monde":                     "France",
        "Le Figaro":                    "France",
        "Euronews (EN)":                "EU/International",
        "Euronews (FR)":                "EU/International",
        "El País":                      "Spain/LatAm",
        "El Mundo":                     "Spain/LatAm",
        "elDiario.es":                  "Spain/LatAm",
        "La Vanguardia":                "Spain/LatAm",
    }

    REGION_COLOR = {
        "USA":              "#7F77DD",
        "UK/International": "#1D9E75",
        "Germany/DACH":     "#D85A30",
        "EU/International": "#378ADD",
        "France":           "#D4537E",
        "Spain/LatAm":      "#639922",
        "Canada":           "#BA7517",
        "Israel":           "#888780",
    }

    _OUTLET_REGION_LOWER = {k.lower(): v for k, v in OUTLET_REGION.items()}

    protest_df["region"] = protest_df["publisher"].map(
        lambda p: OUTLET_REGION.get(p, _OUTLET_REGION_LOWER.get(str(p).lower(), "Other"))
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    fig.patch.set_facecolor(LAVENDER_CREAM)

    _base_style(fig, ax1, grid_axis="x")
    event_lang = protest_df.groupby(["event_label", "language"]).size().reset_index(name="n")
    events = sorted(protest_df["event_label"].dropna().unique())
    y_pos  = np.arange(len(events)); left = np.zeros(len(events))
    for lang in ["EN", "DE", "FR", "ES", "OTHER"]:
        sub  = event_lang[event_lang["language"] == lang].set_index("event_label")
        vals = np.array([sub.loc[e, "n"] if e in sub.index else 0 for e in events])
        if vals.sum() == 0: continue
        ax1.barh(y_pos, vals, left=left, color=lang_pal[lang], label=lang, alpha=0.9)
        left += vals
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([e.replace("_", " ") for e in events], color=DARK_PLUM, fontsize=9)
    ax1.set_xlabel("Number of articles", color=DARK_PLUM)
    ax1.set_title("A — Articles by event × language", fontweight="bold", color=DARK_PLUM)
    ax1.legend(title="Language", frameon=False, fontsize=9)

    _base_style(fig, ax2, grid_axis="x")
    pub_counts = protest_df.groupby("publisher").size().sort_values()
    pub_regions = protest_df.groupby("publisher")["region"].first()
    colors = [REGION_COLOR.get(pub_regions.get(p, "Other"), "#BDBDBD") for p in pub_counts.index]
    ax2.barh(pub_counts.index, pub_counts.values, color=colors, alpha=0.88)
    ax2.set_xlabel("Number of articles", color=DARK_PLUM)
    ax2.set_title("B — Article volume by publisher", fontweight="bold", color=DARK_PLUM)
    ax2.tick_params(axis="y", labelsize=8)
    handles = [mpatches.Patch(color=c, label=r)
               for r, c in REGION_COLOR.items()
               if r in pub_regions.values]
    ax2.legend(handles=handles, title="Region", frameon=False, fontsize=8,
               bbox_to_anchor=(1, 1), loc="upper left")

    _base_style(fig, ax3, grid_axis="x")
    region_counts = protest_df.groupby("region").size().sort_values()
    rcolors = [REGION_COLOR.get(r, "#BDBDBD") for r in region_counts.index]
    bars = ax3.barh(region_counts.index, region_counts.values, color=rcolors, alpha=0.88)
    for bar, val in zip(bars, region_counts.values):
        ax3.text(bar.get_width() + region_counts.max() * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", fontsize=9, color=DARK_PLUM)
    ax3.set_xlabel("Number of articles", color=DARK_PLUM)
    ax3.set_title("C — Articles by outlet region", fontweight="bold", color=DARK_PLUM)
    ax3.set_xlim(0, region_counts.max() * 1.15)

    fig.suptitle("Corpus demographic distribution",
                 fontsize=15, fontweight="bold", color=DARK_PLUM, y=1.02)
    _subnote(fig, "Protest articles only · region assigned by outlet country of origin")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "demographic_distribution.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓲  CHART 14 — FAIRNESS METRIC (Cohen's d)
# ══════════════════════════════════════════════════════════════════════════════

def chart_fairness_metric_comparison(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    if "is_control" not in df.columns or df["is_control"].nunique() < 2:
        print("  ⚠️  chart_fairness_metric_comparison: need both is_control values — skipping."); return

    agg = df.groupby(["publisher","is_control"])["sentiment_score"].agg(
        mean="mean", std="std", count="count").unstack("is_control")
    agg.columns = ["_".join(str(c) for c in col) for col in agg.columns]

    col_map = {}
    for c in agg.columns:
        if "False" in c or "protest" in c.lower():
            if "mean" in c:   col_map[c] = "protest_mean"
            elif "std" in c:  col_map[c] = "protest_std"
            elif "count" in c:col_map[c] = "protest_count"
        if "True" in c or "control" in c.lower():
            if "mean" in c:   col_map[c] = "control_mean"
            elif "std" in c:  col_map[c] = "control_std"
            elif "count" in c:col_map[c] = "control_count"
    agg.rename(columns=col_map, inplace=True)

    needed = ["protest_mean","control_mean","protest_std","control_std","protest_count","control_count"]
    missing = [c for c in needed if c not in agg.columns]
    if missing:
        print(f"  ⚠️  chart_fairness_metric_comparison: missing columns {missing} — skipping."); return

    agg = agg.dropna(subset=["protest_mean","control_mean"])
    pooled = np.sqrt(
        ((agg["protest_count"]-1)*agg["protest_std"]**2 +
         (agg["control_count"]-1)*agg["control_std"]**2) /
        (agg["protest_count"]+agg["control_count"]-2)
    )
    agg["cohens_d"] = (agg["protest_mean"] - agg["control_mean"]) / pooled
    agg = agg.dropna(subset=["cohens_d"])
    if agg.empty:
        print("  ⚠️  chart_fairness_metric_comparison: empty after computing Cohen's d — skipping."); return

    y_pos = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(11, max(6, len(agg)*0.5)))
    _base_style(fig, ax, grid_axis="x")
    ax.barh(y_pos, agg["cohens_d"],
            color=[RASPBERRY if d < 0 else MUTED_TEAL for d in agg["cohens_d"]],
            edgecolor=DARK_PLUM, linewidth=0.5, alpha=0.85, height=0.55)
    ax.axvline(0, color=DARK_PLUM, linewidth=1.0)
    ax.axvspan(-0.2, 0.2, color=DARK_PLUM, alpha=0.04, label="Negligible effect")
    ax.set_yticks(y_pos); ax.set_yticklabels(agg.index, fontsize=9, color=DARK_PLUM)
    ax.set_xlabel("Cohen's d Effect Size (Protest vs Control)", color=DARK_PLUM)
    ax.set_title("Media Fairness Benchmark — Cohen's d\n"
                 "Negative = significantly harsher tone during protest weeks",
                 fontweight="bold", fontsize=13, color=DARK_PLUM)
    _save(fig, os.path.join(out_dir, "fairness_cohens_d_comparison.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓳  CHART 15 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

def chart_model_performance(df: pd.DataFrame, topic_info_path: str,
                             out_dir: str, dpi: int) -> None:
    if "topic_id" not in df.columns:
        print("  ⚠️  chart_model_performance: 'topic_id' missing — skipping."); return

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.patch.set_facecolor(LAVENDER_CREAM)

    ax = axes[0]; _base_style(fig, ax, grid_axis="y")
    topic_sizes = df["topic_id"].value_counts().sort_index()
    sizes_no_out = topic_sizes[topic_sizes.index != -1]
    ax.bar(range(len(sizes_no_out)), sorted(sizes_no_out.values, reverse=True),
           color=DEEP_PURPLE, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Topic rank (by size)", color=DARK_PLUM)
    ax.set_ylabel("Articles (log scale)", color=DARK_PLUM)
    ax.set_title(f"A — Topic Size Distribution\n({len(sizes_no_out)} topics)",
                 fontweight="bold", color=DARK_PLUM)

    ax = axes[1]; _base_style(fig, ax, grid_axis="x")
    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    outlier_rate = (protest_df.groupby("event_label")["topic_id"]
                   .apply(lambda g: (g == -1).mean() * 100)
                   .sort_values())
    ax.barh(outlier_rate.index, outlier_rate.values,
            color=[RASPBERRY if v > 20 else MUTED_TEAL for v in outlier_rate.values], alpha=0.85)
    ax.axvline(20, color=DARK_PLUM, linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Outlier rate (%)", color=DARK_PLUM)
    ax.set_title("B — Outlier Rate by Event", fontweight="bold", color=DARK_PLUM)
    for i, (event, val) in enumerate(outlier_rate.items()):
        ax.text(val+0.3, i, f"{val:.1f}%", va="center", fontsize=8, color=DARK_PLUM)

    ax = axes[2]; _base_style(fig, ax, grid_axis="x")
    event_topics_dir = os.path.join(os.path.dirname(topic_info_path), "event_topics")
    event_info_files = glob.glob(os.path.join(event_topics_dir, "*_topic_info.csv"))
    if event_info_files:
        tc = {}
        for f in sorted(event_info_files):
            try:
                tdf = pd.read_csv(f)
                ev  = tdf["event_label"].iloc[0] if "event_label" in tdf.columns else os.path.basename(f).replace("_topic_info.csv","")
                tc[ev] = int((tdf["Topic"] != -1).sum())
            except Exception:
                pass
        if tc:
            tc_s = pd.Series(tc).sort_values()
            ax.barh(tc_s.index, tc_s.values, color=MUTED_TEAL, alpha=0.85)
            ax.set_xlabel("Number of Topics Discovered", color=DARK_PLUM)
            ax.set_title("C — Per-Event Topic Count", fontweight="bold", color=DARK_PLUM)
    else:
        ax.text(0.5, 0.5, "Run topic_model.py without\n--no-event-topics",
                ha="center", va="center", transform=ax.transAxes, color=DARK_PLUM)
        ax.axis("off")
        ax.set_title("C — Per-Event Topic Count (unavailable)", fontweight="bold", color=DARK_PLUM)

    fig.suptitle("Topic Model Performance Diagnostics",
                 fontsize=15, fontweight="bold", color=DARK_PLUM, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "model_performance.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓴  CHART 16 — SUBGROUP COMPARISON (heatmap)
# ══════════════════════════════════════════════════════════════════════════════

def chart_subgroup_comparison(df: pd.DataFrame, out_dir: str, dpi: int) -> None:
    try:
        import seaborn as sns
    except ImportError:
        print("  ⚠️  chart_subgroup_comparison: seaborn not installed — skipping.")
        return

    protest_df = (df[df["is_control"] == False] if "is_control" in df.columns else df).copy()
    if protest_df.empty:
        print("  ⚠️  chart_subgroup_comparison: protest dataset empty — skipping."); return

    protest_df["ideology_bin"] = protest_df["publisher"].apply(_ideology_score).apply(
        lambda s: "Unknown" if pd.isna(s) else ("Left" if s < -0.3 else ("Right" if s > 0.3 else "Centre")))
    protest_df = protest_df[protest_df["ideology_bin"] != "Unknown"]

    heat = (protest_df.groupby(["event_label", "ideology_bin"])["sentiment_score"]
            .mean().unstack("ideology_bin").fillna(np.nan))
    cols = [c for c in ["Left", "Centre", "Right"] if c in heat.columns]
    heat = heat[cols]
    if heat.empty:
        print("  ⚠️  chart_subgroup_comparison: empty heatmap — skipping."); return

    vabs = max(abs(heat.values[~np.isnan(heat.values)].max()),
               abs(heat.values[~np.isnan(heat.values)].min()))
    vabs = max(vabs, 0.05)

    fig, ax = plt.subplots(figsize=(7, max(5, len(heat) * 0.65)))
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM)

    sns.heatmap(
        heat,
        cmap="RdYlGn",
        center=0,
        vmin=-vabs, vmax=vabs,
        annot=True, fmt=".2f",
        annot_kws={"fontsize": 12, "fontweight": "bold", "color": DARK_PLUM},
        linewidths=1.5, linecolor=LAVENDER_CREAM,
        cbar_kws={"label": "Mean sentiment score", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Sentiment across movements & ideologies",
                 fontweight="bold", fontsize=13, color=DARK_PLUM, pad=15)
    ax.set_xlabel("Publisher political leaning", color=DARK_PLUM, labelpad=10, fontsize=11)
    ax.set_ylabel("Protest event", color=DARK_PLUM, labelpad=10, fontsize=11)
    ax.tick_params(axis="x", labelsize=11, colors=DARK_PLUM, rotation=0)
    ax.tick_params(axis="y", labelsize=10, colors=DARK_PLUM, rotation=0)
    ax.set_yticklabels([l.get_text().replace("_", " ") for l in ax.get_yticklabels()])
    _subnote(fig, "Red = more negative · green = more positive · white = neutral (0) · "
             "unknown-ideology publishers excluded")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "fairness_subgroup_intersection.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ㉑  CHART 17 — CONFUSION MATRIX / SENTIMENT DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def chart_confusion_matrix(df: pd.DataFrame, out_dir: str, dpi: int,
                            labelled_sample_path: "str | None" = None) -> None:
    if "sentiment_label" not in df.columns:
        print("  ⚠️  chart_confusion_matrix: 'sentiment_label' missing — skipping."); return

    protest_df = df[~df["is_control"]] if "is_control" in df.columns else df
    label_order = ["positive", "neutral", "negative"]

    if labelled_sample_path and os.path.exists(labelled_sample_path):
        try:
            gt     = pd.read_csv(labelled_sample_path)
            merged = protest_df.merge(gt[["url", "human_label"]], on="url", how="inner")
            if merged.empty:
                raise ValueError("No overlapping URLs")
            cm = pd.crosstab(merged["human_label"].str.lower(),
                             merged["sentiment_label"].str.lower(),
                             rownames=["Human"], colnames=["VADER"]
                             ).reindex(index=label_order, columns=label_order, fill_value=0)
            n_total     = cm.values.sum()
            accuracy    = np.diag(cm.values).sum() / max(n_total, 1)
            title       = f"Sentiment confusion matrix  (acc={accuracy:.1%})"
            subtitle    = f"n={n_total:,} · rows=human label · columns=VADER prediction"
            matrix_vals = cm.values.astype(float)
            col_labels  = label_order
            row_labels  = label_order
            fmt_float   = False
        except Exception as e:
            print(f"  ⚠️  Could not load labelled sample ({e}) — falling back to proxy matrix.")
            labelled_sample_path = None

    if not (labelled_sample_path and os.path.exists(str(labelled_sample_path or ""))):
        protest_df = protest_df.copy()
        protest_df["ideology_bin"] = pd.cut(
            protest_df["publisher"].apply(_ideology_score),
            bins=[-3, -0.5, 0.5, 3], labels=["Left", "Centre", "Right"]
        ).astype(str).replace("nan", "Unknown")
        protest_df = protest_df[protest_df["ideology_bin"] != "Unknown"]
        cm = (protest_df.groupby(["ideology_bin", "sentiment_label"])
              .size().unstack("sentiment_label", fill_value=0))
        cm  = cm.div(cm.sum(axis=1), axis=0) * 100
        cm  = cm.reindex(columns=[c for c in label_order if c in cm.columns])
        matrix_vals = cm.values.astype(float)
        row_labels  = cm.index.tolist()
        col_labels  = cm.columns.tolist()
        title       = "Sentiment distribution by outlet ideology"
        subtitle    = ("Proxy validation — no ground-truth labels supplied · "
                       "rows=ideology bin · values=% of row")
        fmt_float   = True

    if fmt_float:
        cmap = "RdYlGn_r"
        vcenter = 33.3
        vabs    = max(abs(matrix_vals.max() - vcenter), abs(matrix_vals.min() - vcenter))
        vmin, vmax = vcenter - vabs, vcenter + vabs
        norm = None
    else:
        cmap = "Purples"
        vmin, vmax, vcenter, norm = None, None, None, None

    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 2.2),
                                    max(4, len(row_labels) * 1.4)))
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM)

    im = ax.imshow(matrix_vals, cmap=cmap, aspect="auto",
                   vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% of row" if fmt_float else "Count", color=DARK_PLUM, fontsize=10)
    cbar.ax.tick_params(colors=DARK_PLUM)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, color=DARK_PLUM, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, color=DARK_PLUM, fontsize=12, fontweight="bold")
    ax.tick_params(length=0)

    thresh = matrix_vals.max() * 0.6
    for i in range(matrix_vals.shape[0]):
        for j in range(matrix_vals.shape[1]):
            val  = matrix_vals[i, j]
            txt  = f"{val:.1f}%" if fmt_float else f"{int(val)}"
            dark = val < thresh
            ax.text(j, i, txt, ha="center", va="center", fontsize=14,
                    fontweight="bold",
                    color=DARK_PLUM if dark else "white")

    ax.set_title(title, fontweight="bold", fontsize=13, color=DARK_PLUM, pad=16)
    _subnote(fig, subtitle)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "confusion_matrix_sentiment.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ㉒  CHART 18 — EXTENDED WORKFLOW DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def chart_workflow_diagram(out_dir: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(20, 9))
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM); ax.axis("off")
    ax.set_xlim(0, 20); ax.set_ylim(0, 9)
    LANE_Y = {"data": 6.8, "analysis": 2.8}
    BOX_H, BOX_W = 1.8, 2.8
    nodes = {
        "collect":   (1.8,  "data",     "collect_articles\n_optimized.py",  "CC-News crawl\n45 publishers × 10 events",  "corpus_YYYY.parquet"),
        "control":   (1.8,  "analysis", "collect_articles\n(control flag)",  "Matched control weeks",                     "control/*.parquet"),
        "build":     (5.2,  "data",     "build_corpus.py",                   "Merge + deduplicate",                       "corpus_all.parquet"),
        "sentiment": (8.6,  "data",     "sentiment_analysis.py",             "Lingua → opus-mt → VADER",                  "articles_with_sentiment.parquet"),
        "topic":     (12.0, "data",     "topic_model.py",                    "all-MiniLM + BERTopic",                     "articles_with_topics.parquet"),
        "vis":       (15.8, "data",     "visualise.py",                      "22 charts + bias_report",                   "figures/*.png"),
        "bias":      (18.8, "analysis", "bias_report.csv",                   "Coverage · language\n· ideology stats",     "Supplementary table"),
    }
    arrows = [("collect","build",False),("control","build",True),
              ("build","sentiment",False),("sentiment","topic",False),
              ("topic","vis",False),("vis","bias",False)]
    pos = {}
    for key,(cx,lane,script,tech,out) in nodes.items():
        cy = LANE_Y[lane]
        ax.add_patch(plt.Rectangle((cx-BOX_W/2,cy-BOX_H/2),BOX_W,BOX_H,
                                    facecolor=DEEP_PURPLE,edgecolor=BLUSH,linewidth=1.5,zorder=2,alpha=0.92))
        ax.text(cx,cy+0.45,script,ha="center",va="center",fontsize=7.5,fontweight="bold",color="white",zorder=3)
        ax.text(cx,cy-0.05,tech,ha="center",va="center",fontsize=6.5,color=BLUSH,zorder=3)
        ax.text(cx,cy-0.62,f"→ {out}",ha="center",va="center",fontsize=6.2,color="#E1BEE7",zorder=3,style="italic")
        pos[key] = (cx,cy)
    ax.text(0.15,LANE_Y["data"],"DATA\nPIPELINE",ha="center",va="center",
            fontsize=9,fontweight="bold",color=DEEP_PURPLE,rotation=90)
    ax.text(0.15,LANE_Y["analysis"],"ANALYSIS &\nOUTPUTS",ha="center",va="center",
            fontsize=9,fontweight="bold",color=DEEP_PURPLE,rotation=90)
    ax.axhline(4.8,color=BLUSH,linewidth=0.8,linestyle=":")
    for src,dst,dashed in arrows:
        sx,sy = pos[src]; dx,dy = pos[dst]
        ax.annotate("",xy=(dx-BOX_W/2-0.05,dy),xytext=(sx+BOX_W/2+0.05,sy),
                    arrowprops=dict(arrowstyle="->",color=RASPBERRY,lw=1.6,
                                   linestyle="dashed" if dashed else "solid"),zorder=4)
    ax.plot([0.6,1.2],[0.4,0.4],color=RASPBERRY,lw=1.6,linestyle="dashed")
    ax.text(1.3,0.4,"optional / conditional",va="center",fontsize=8,color=DARK_PLUM)
    ax.set_title("Extended Analysis Workflow",fontsize=15,fontweight="bold",color=DARK_PLUM,pad=12)
    _save(fig, os.path.join(out_dir, "workflow_diagram.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ㉓  CHART 19 — PER-EVENT CONTROL WEEK TOPIC PIES  (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def chart_topic_pies_control_per_event(df: pd.DataFrame, out_dir: str, dpi: int,
                                        top_n: int) -> None:
    """
    One donut chart per matched control week, showing the global-model topic
    distribution for articles collected during that control period.

    Uses the global friendly_name column (same vocabulary as the protest-week
    overview pies) so protest and control topic distributions are directly
    comparable.  Per-event topic models are NOT used here because they were
    trained on protest-week articles only.

    Output filenames: topic_pie_control_{event_label}.png
    """
    if "is_control" not in df.columns:
        print("  ⚠️  chart_topic_pies_control_per_event: 'is_control' column missing — skipping.")
        return
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topic_pies_control_per_event: 'friendly_name' column missing — skipping.")
        return

    control_df = df[df["is_control"]].copy()
    if control_df.empty:
        print("  ⚠️  chart_topic_pies_control_per_event: no control articles found — skipping.")
        return
    if "event_label" not in control_df.columns:
        print("  ⚠️  chart_topic_pies_control_per_event: 'event_label' column missing — skipping.")
        return

    control_df["friendly_name"] = control_df["friendly_name"].map(_normalise_other)

    for event in sorted(control_df["event_label"].dropna().unique()):
        sub = control_df[control_df["event_label"] == event]
        if sub.empty:
            continue

        slices = _build_slices_normalised(sub, top_n, label_col="friendly_name")
        if slices.empty:
            print(f"  ⚠️  [{event}] control week: no topics after slicing — skipping.")
            continue

        colors, explode, edges = _slice_colors(slices)
        fig, ax = plt.subplots(figsize=(13, 8))
        subtitle = (
            f"n={len(sub):,} articles · global topic model · "
            f"matched control week for {event.replace('_', ' ').title()}"
        )
        _donut_with_legend(
            fig, ax, slices, colors, explode, edges,
            title=f"Control Week Topics — {event.replace('_', ' ').title()}",
            subtitle=subtitle,
            pct_label="%",
        )
        safe = event.replace(" ", "_").replace("/", "-")
        _save(fig, os.path.join(out_dir, f"topic_pie_control_{safe}.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ㉔  CHART 20 — OVERALL NORMALISED CONTROL WEEK TOPIC BREAKDOWN  (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def chart_topic_pie_control_normalised(df: pd.DataFrame, out_dir: str, dpi: int,
                                        top_n: int) -> None:
    """
    Single donut showing the mean topic proportion across ALL matched control
    weeks, weighted equally per event (same normalisation logic as the protest
    overview pie so the two charts are directly comparable).

    Output: topic_pie_control_weeks_normalised.png
    """
    if "is_control" not in df.columns:
        print("  ⚠️  chart_topic_pie_control_normalised: 'is_control' column missing — skipping.")
        return
    if "friendly_name" not in df.columns:
        print("  ⚠️  chart_topic_pie_control_normalised: 'friendly_name' column missing — skipping.")
        return

    control_df = df[df["is_control"]].copy()
    if control_df.empty:
        print("  ⚠️  chart_topic_pie_control_normalised: no control articles found — skipping.")
        return

    control_df["friendly_name"] = control_df["friendly_name"].map(_normalise_other)

    # Normalise per event then average — identical logic to _build_slices_normalised
    # with subset_col="event_label", but we call that helper directly.
    slices = _build_slices_normalised(
        control_df, top_n,
        subset_col="event_label" if "event_label" in control_df.columns else None,
        label_col="friendly_name",
    )
    if slices.empty:
        print("  ⚠️  chart_topic_pie_control_normalised: empty after slicing — skipping.")
        return

    n_events = (control_df["event_label"].nunique()
                if "event_label" in control_df.columns else "?")

    colors, explode, edges = _slice_colors(slices)
    fig, ax = plt.subplots(figsize=(14, 9))
    _donut_with_legend(
        fig, ax, slices, colors, explode, edges,
        title="Topic Distribution: All Control Weeks (Global Model, Normalised)",
        subtitle=(
            f"Mean proportion across {n_events} matched control events · "
            "global topic vocabulary · equal event weighting · ★ = feminist/protest topic"
        ),
        pct_label="%",
    )
    _save(fig, os.path.join(out_dir, "topic_pie_control_weeks_normalised.png"), dpi)


# ══════════════════════════════════════════════════════════════════════════════
# ⓱  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment",        default=DEFAULT_SENTIMENT)
    parser.add_argument("--topics",           default=DEFAULT_TOPICS)
    parser.add_argument("--topic-info",       default=DEFAULT_TOPIC_INFO)
    parser.add_argument("--control-dir",      default=DEFAULT_CONTROL)
    parser.add_argument("--out-dir",          default=OUTPUT_DIR)
    parser.add_argument("--top-n",            type=int, default=10)
    parser.add_argument("--confusion-labels", default=None, metavar="PATH")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load sentiment data ───────────────────────────────────────────────────
    print(f"Loading sentiment data from {args.sentiment} ...")
    df = pd.read_csv(args.sentiment)
    df["is_control"] = df["is_control"].astype(bool) if "is_control" in df.columns else False
    print(f"  {len(df):,} articles")

    if "sentiment_label" not in df.columns and "sentiment_score" in df.columns:
        df["sentiment_label"] = df["sentiment_score"].apply(
            lambda s: "positive" if s >= 0.05 else ("negative" if s <= -0.05 else "neutral")
        )
        print(f"  Derived sentiment_label: {df['sentiment_label'].value_counts().to_dict()}")

    # ── Merge topic columns ───────────────────────────────────────────────────
    if os.path.exists(args.topics):
        print(f"Merging topic assignments from {args.topics} ...")
        topics_df = pd.read_parquet(args.topics)
        topic_cols = [
            c for c in topics_df.columns
            if c.startswith("topic")
            or c.startswith("event_topic")
            or c in ("url", "language")
        ]
        if "url" in df.columns and "url" in topics_df.columns:
            stale = [c for c in topic_cols if c != "url" and c in df.columns]
            if stale:
                df = df.drop(columns=stale)
            df = df.merge(topics_df[topic_cols], on="url", how="left")
            if "language_x" in df.columns and "language_y" in df.columns:
                df["language"] = df["language_y"].fillna(df["language_x"])
                df.drop(columns=["language_x", "language_y"], inplace=True)
            print(f"  Merged topic columns for {df['topic_id'].notna().sum():,} articles")
            event_topic_cols = [c for c in df.columns if c.startswith("event_topic")]
            print(f"  Event topic columns present: {event_topic_cols}")
    else:
        print(f"  ⚠️  topics file not found: {args.topics}")

    # ── Publisher-based language fallback ─────────────────────────────────────
    corpus_lang_map = {
        "DW":"DE","SpiegelOnline":"DE","DieZeit":"DE","Tagesschau":"DE","FAZ":"DE","Taz":"DE",
        "DerStandard":"DE","DiePresse":"DE","ORF":"DE","IsraelNachrichten":"DE",
        "LeMonde":"FR","LeFigaro":"FR","EuronewsFR":"FR",
        "ElPais":"ES","ElMundo":"ES","ElDiario":"ES","LaVanguardia":"ES","ABC":"ES","Publico":"ES",
    }
    if "language" not in df.columns:
        df["language"] = df["publisher"].map(corpus_lang_map).fillna("EN")
    else:
        mask = df["language"].isna()
        if mask.any():
            df.loc[mask, "language"] = df.loc[mask, "publisher"].map(corpus_lang_map).fillna("EN")

    # ── Build global topic name map ───────────────────────────────────────────
    print(f"Loading topic info from {args.topic_info} ...")
    info_df  = pd.read_csv(args.topic_info)
    info_df  = info_df[info_df["Topic"] != -1].copy()
    name_map = {row["Topic"]: clean_topic_name(row["Name"]) for _, row in info_df.iterrows()}

    if "topic_id" in df.columns:
        df_unfiltered = df.copy()
        df = df[df["topic_id"].notna() & (df["topic_id"] != -1)].copy()
        df["topic_id"] = df["topic_id"].astype(int)
        df["friendly_name"] = df["topic_id"].map(name_map).fillna("Unknown")
    else:
        df_unfiltered = df.copy()
        df["friendly_name"] = "Unknown"

    # ── Load per-event topic label maps ───────────────────────────────────────
    event_topics_dir  = os.path.join(os.path.dirname(args.topic_info), "event_topics")
    event_topic_maps  = _load_per_event_topic_names(event_topics_dir)
    if event_topic_maps:
        print(f"  Loaded per-event topic maps for {len(event_topic_maps)} events")
    else:
        print("  ⚠️  No per-event topic maps found — per-event pies will use global labels")

    model_notes = _load_event_model_notes(event_topics_dir)

    # ── Load control data ─────────────────────────────────────────────────────
    ctrl = load_control_data(args.control_dir, name_map)
    if ctrl is not None:
        if "language" not in ctrl.columns:
            ctrl["language"] = ctrl["publisher"].map(corpus_lang_map).fillna("EN")
        
        # Ensure 'friendly_name' matches on the protest dataframe side too
        df_label_col = next((c for c in ["friendly_name", "topic_label", "topic_name"] if c in df.columns), None)
        if df_label_col and df_label_col != "friendly_name":
            df["friendly_name"] = df[df_label_col]
            
        # We copy the data over before wiping structural IDs for concatenation
        shared_cols = [c for c in df.columns if c in ctrl.columns]
        combined = pd.concat([df[shared_cols], ctrl[shared_cols]], ignore_index=True)
    else:
        combined = df.copy()

    print(f"\nGenerating charts → {args.out_dir}")

    # ── Protest-only charts ───────────────────────────────────────────────────
    chart_sentiment_by_outlet(df, args.out_dir, DPI)
    chart_topics_by_event(df, args.out_dir, DPI, args.top_n, event_topic_maps, model_notes)
    chart_topic_pies_per_event(df, args.out_dir, DPI, args.top_n, event_topic_maps, model_notes)
    chart_topic_pie(df, args.out_dir, DPI, args.top_n)
    chart_topic_pie_all(df, args.out_dir, DPI)
    chart_topic_pie_all_normalised(df, args.out_dir, DPI)
    chart_sentiment_by_topic(df, args.out_dir, DPI, args.top_n + 2)
    chart_articles_per_event(df, args.out_dir, DPI)
    chart_sentiment_by_event_type(df, args.out_dir, DPI)

    # ── Combined protest + control charts ─────────────────────────────────────
    chart_protest_vs_control(combined, args.out_dir, DPI)
    chart_ideological_gap(combined, args.out_dir, DPI)
    chart_language_breakdown(combined, args.out_dir, DPI)
    chart_demographic_distribution(combined, args.out_dir, DPI)
    chart_fairness_metric_comparison(combined, args.out_dir, DPI)
    chart_subgroup_comparison(combined, args.out_dir, DPI)
    chart_confusion_matrix(combined, args.out_dir, DPI, args.confusion_labels)

    # ── Control-week topic breakdowns (NEW) ───────────────────────────────────
    chart_topic_pies_control_per_event(combined, args.out_dir, DPI, args.top_n)
    chart_topic_pie_control_normalised(combined, args.out_dir, DPI, args.top_n)

    # ── Static diagrams ───────────────────────────────────────────────────────
    chart_pipeline_diagram(args.out_dir, DPI)
    chart_workflow_diagram(args.out_dir, DPI)
    chart_model_performance(df_unfiltered, args.topic_info, args.out_dir, DPI)

    # ── Bias report ───────────────────────────────────────────────────────────
    print("\nGenerating bias report ...")
    generate_bias_report(combined, args.out_dir)

    print(f"\n✅  All outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()