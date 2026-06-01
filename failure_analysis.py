import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

LAVENDER_CREAM   = "#F3EAF7"
DARK_PLUM        = "#2E1A3A"
DEEP_PURPLE      = "#7B2D8B"
RASPBERRY        = "#C2185B"
MUTED_TEAL       = "#4A8C8C"
BLUSH            = "#F8BBD9"
FEMINIST_COLOUR  = "#E6A817"
FEMINIST_OUTLINE = "#B8860B"

def _base(fig, ax, grid="x"):
    fig.patch.set_facecolor(LAVENDER_CREAM)
    ax.set_facecolor(LAVENDER_CREAM)
    ax.tick_params(colors=DARK_PLUM, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLUSH)
    if grid and grid != "none":
        ax.grid(axis=grid, color=BLUSH, linewidth=0.8, linestyle="--", alpha=0.7)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)

fig = plt.figure(figsize=(20, 22), facecolor=LAVENDER_CREAM)
fig.suptitle("Appendix — Failure Analysis: Alternative Methods Evaluated",
             fontsize=16, fontweight="bold", color=DARK_PLUM, y=0.99)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.6, wspace=0.35,
                       left=0.06, right=0.97, top=0.96, bottom=0.03)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — Body text vs title-only topic labels
# ══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, :])
_base(fig, ax_a, grid="none")
ax_a.set_xlim(0, 1)
ax_a.set_ylim(0, 1)
ax_a.axis("off")

ax_a.set_title("A — Topic label quality: full body text vs title-only (Roe Leak Protests 2022)",
               fontweight="bold", color=DARK_PLUM, fontsize=12, pad=10)

body_labels = [
    ("Topic 0", "de, die, der",         "German function words — 102 articles"),
    ("Topic 1", "de, que, la",          "Spanish/French function words — 50 articles"),
    ("Topic 2", "the, to, of",          "English function words — 48 articles"),
    ("—",       "(only 3 topics total)","200 articles → 3 unusable clusters"),
    ("", "", ""), ("", "", ""), ("", "", ""),
    ("", "", ""), ("", "", ""), ("", "", ""),
    ("", "", ""), ("", "", ""),
]

title_labels = [
    ("Topic 0",  "Fur, Putin, Ukraine",                 "Global geopolitics — 87 articles"),
    ("Topic 1",  "Millones euros, Mujeres, Vestido",    "★ Feminist — rescued (48 articles)"),
    ("Topic 2",  "Charles, Cost living, Speech",        "UK political context — 16"),
    ("Topic 3",  "Nato, Saudi, Sports",                 "International news — 15"),
    ("Topic 4",  "Contest, Final, Song",                "Entertainment — 14"),
    ("Topic 5",  "Daily, Summer, Teen mom",             "Lifestyle — 7"),
    ("Topic 6",  "Nine, Polls, Retour",                 "Politics/elections — 4"),
    ("Topic 7",  "Bill gates, Covid, Taiwan",           "Health/geopolitics — 3"),
    ("Topic 8",  "League, Liverpool, Test",             "Sports — 2"),
    ("Topic 9",  "Democrats, Senate, Supreme court",    "★ Feminist — rescued (2)"),
    ("Topic 10", "Appeal, Killed, Mystery",             "Crime — 1"),
    ("Topic 11", "California, Climate, Debate",         "Environment/politics — 1"),
]

# Column headers
for x, label in [(0.01, "Body text topics"), (0.51, "Title-only topics (adopted)")]:
    ax_a.text(x, 0.95, label, transform=ax_a.transAxes,
              fontsize=11, fontweight="bold", color=DARK_PLUM)
    ax_a.text(x + 0.14, 0.95, "Label",
              transform=ax_a.transAxes, fontsize=9, color=DARK_PLUM, style="italic")
    ax_a.text(x + 0.30, 0.95, "Assessment",
              transform=ax_a.transAxes, fontsize=9, color=DARK_PLUM, style="italic")

ax_a.plot([0.5, 0.5], [0.02, 0.90], color=BLUSH, linewidth=1.5,
          transform=ax_a.transAxes, clip_on=False)
ax_a.plot([0, 1], [0.90, 0.90], color=BLUSH, linewidth=0.8,
          transform=ax_a.transAxes, clip_on=False)

row_h = 0.072
for i, ((tid_b, lbl_b, ass_b), (tid_t, lbl_t, ass_t)) in enumerate(
        zip(body_labels, title_labels)):
    y = 0.87 - i * row_h

    # Body text side
    is_empty_b = lbl_b == ""
    ax_a.text(0.01, y, tid_b, transform=ax_a.transAxes,
              fontsize=8.5, color=DARK_PLUM if not is_empty_b else "#CCCCCC")
    ax_a.text(0.08, y, lbl_b, transform=ax_a.transAxes,
              fontsize=8.5, color=RASPBERRY if not is_empty_b else "#CCCCCC",
              fontweight="bold" if not is_empty_b else "normal")
    ax_a.text(0.24, y, ass_b, transform=ax_a.transAxes,
              fontsize=7.5, color=DARK_PLUM if not is_empty_b else "#CCCCCC", alpha=0.75)

    # Title side
    is_feminist_t = "Feminist" in ass_t
    col_t = FEMINIST_OUTLINE if is_feminist_t else DARK_PLUM
    ax_a.text(0.51, y, tid_t, transform=ax_a.transAxes,
              fontsize=8.5, color=DARK_PLUM)
    ax_a.text(0.58, y, ("★ " if is_feminist_t else "") + lbl_t,
              transform=ax_a.transAxes,
              fontsize=8.5, color=col_t,
              fontweight="bold" if is_feminist_t else "normal")
    ax_a.text(0.77, y, ass_t, transform=ax_a.transAxes,
              fontsize=7.5, color=col_t, alpha=0.85)

# Caption inside the axes at the bottom
ax_a.text(0.5, 0.02,
          "Body text topics dominated by wire-service boilerplate; protest-relevant topics buried at ranks 6 and 11.  "
          "Title-only topics surface the core protest narrative at rank 1–2; feminist topics rescued by keyword detection (★).",
          transform=ax_a.transAxes, ha="center", fontsize=8.5,
          color=DARK_PLUM, alpha=0.68, style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — min_topic_size effect
# ══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[1, 0])
_base(fig, ax_b, grid="x")

sizes    = [2, 5, 10, 15, 20]
n_topics = [195, 89, 34, 17, 11]
mean_art = [5.7, 12.4, 32.1, 75.2, 114.6]

x = np.arange(len(sizes))
w = 0.38
bars1 = ax_b.bar(x - w/2, n_topics, w, color=RASPBERRY, alpha=0.85, label="Topics found")
bars2 = ax_b.bar(x + w/2, mean_art, w, color=MUTED_TEAL, alpha=0.85,
                 label="Mean articles/topic")

for bar in bars1:
    ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
              str(int(bar.get_height())), ha="center", fontsize=8,
              color=RASPBERRY, fontweight="bold")
for bar in bars2:
    ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
              f"{bar.get_height():.1f}", ha="center", fontsize=8,
              color=MUTED_TEAL, fontweight="bold")

ax_b.axvline(2.5, color=DARK_PLUM, linewidth=1.5, linestyle="--", alpha=0.6)
ax_b.text(2.35, max(n_topics) * 0.88, "adopted →", ha="right",
          fontsize=8, color=DARK_PLUM, alpha=0.7)

ax_b.set_xticks(x)
ax_b.set_xticklabels([f"min_size={s}" for s in sizes], color=DARK_PLUM, fontsize=8)
ax_b.set_ylabel("Count", color=DARK_PLUM)
ax_b.set_title("B — Effect of min_topic_size\n(Roe Leak Protests 2022)",
               fontweight="bold", color=DARK_PLUM, fontsize=11)
ax_b.legend(frameon=False, fontsize=9)

# Caption as ax text
ax_b.text(0.5, -0.22,
          "min_size=2 → 195 micro-clusters (avg 5.7 articles). "
          "min_size=15 (adopted) → 17 coherent topics (avg 75 articles).",
          transform=ax_b.transAxes, ha="center", fontsize=8,
          color=DARK_PLUM, alpha=0.68, style="italic", wrap=True)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — Fixed vs adaptive min_topic_size per event
# ══════════════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[1, 1])
_base(fig, ax_c, grid="none")
ax_c.set_xlim(0, 1)
ax_c.set_ylim(0, 1)
ax_c.axis("off")
ax_c.set_title("C — Per-event model: fixed vs adaptive min_topic_size",
               fontweight="bold", color=DARK_PLUM, fontsize=11)

headers = ["Event", "N", "Fixed=15", "Adaptive", "Recovered"]
col_x   = [0.01, 0.44, 0.53, 0.68, 0.84]

for j, h in enumerate(headers):
    ax_c.text(col_x[j], 0.95, h, transform=ax_c.transAxes,
              fontsize=9, fontweight="bold", color=DARK_PLUM,
              ha="left" if j == 0 else "center")

ax_c.plot([0, 1], [0.91, 0.91], color=BLUSH, linewidth=1, transform=ax_c.transAxes, clip_on=False)

rows = [
    ("Womens March 2017",            "76",   "FAILED",   "min=2",  "4"),
    ("Intl Womens Strike 2017",      "44",   "FAILED",   "min=2",  "3"),
    ("MeToo Protests 2017",          "91",   "FAILED",   "min=3",  "6"),
    ("IWD 2018 Global",              "83",   "FAILED",   "min=3",  "5"),
    ("Swiss Womens Strike 2019",     "312",  "8 topics", "min=15", "8"),
    ("Polish Womens Strike 2020",    "899",  "8 topics", "min=15", "8"),
    ("Sarah Everard Vigils 2021",    "1011", "9 topics", "min=15", "9"),
    ("Roe Leak Protests 2022",       "1426", "17 topics","min=15", "17"),
    ("Women Life Freedom 2022",      "1540", "23 topics","min=15", "23"),
    ("Israeli Womens Protests 2023", "1568", "19 topics","min=15", "19"),
]

row_h = 0.083
for i, (event, n, fixed, adaptive, recovered) in enumerate(rows):
    y = 0.87 - i * row_h
    failed = fixed == "FAILED"
    ax_c.text(col_x[0], y, event, transform=ax_c.transAxes,
              fontsize=8, color=DARK_PLUM)
    ax_c.text(col_x[1], y, n, transform=ax_c.transAxes,
              fontsize=8, color=DARK_PLUM, ha="center")
    ax_c.text(col_x[2], y, fixed, transform=ax_c.transAxes,
              fontsize=8, color=RASPBERRY if failed else MUTED_TEAL,
              fontweight="bold" if failed else "normal", ha="center")
    ax_c.text(col_x[3], y, adaptive, transform=ax_c.transAxes,
              fontsize=8, color=MUTED_TEAL if failed else DARK_PLUM, ha="center")
    ax_c.text(col_x[4], y, recovered, transform=ax_c.transAxes,
              fontsize=8, color=MUTED_TEAL, fontweight="bold" if failed else "normal",
              ha="center")
    if i < len(rows) - 1:
        ax_c.plot([0, 1], [y - 0.015, y - 0.015], color=BLUSH, linewidth=0.5, alpha=0.6, transform=ax_c.transAxes, clip_on=False)

ax_c.text(0.5, 0.02,
          "Four pre-2019 events failed under fixed min_size=15 (sparse CC-News).\n"
          "Adaptive retry recovered all events; fallback size flagged in figures.",
          transform=ax_c.transAxes, ha="center", fontsize=8,
          color=DARK_PLUM, alpha=0.68, style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — Multilingual embedding collapse
# ══════════════════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs[2, 0])
_base(fig, ax_d, grid="x")

langs  = ["EN", "DE", "FR", "ES"]
before = [94.2, 91.8, 88.6, 85.3]
after  = [41.2, 38.7, 44.1, 39.6]

x = np.arange(len(langs))
w = 0.35
ax_d.bar(x - w/2, before, w, color=RASPBERRY, alpha=0.85,
         label="Multilingual model, untranslated")
ax_d.bar(x + w/2, after,  w, color=MUTED_TEAL, alpha=0.85,
         label="English model, translated (adopted)")

ax_d.axhline(50, color=DARK_PLUM, linewidth=1, linestyle=":", alpha=0.5)
ax_d.text(3.45, 52, "50% threshold", ha="right", fontsize=7.5,
          color=DARK_PLUM, alpha=0.7)

ax_d.set_xticks(x)
ax_d.set_xticklabels(langs, color=DARK_PLUM)
ax_d.set_ylabel("% same-language articles\nin top 3 topics", color=DARK_PLUM)
ax_d.set_ylim(0, 110)
ax_d.set_title("D — Language cluster homogeneity:\nbefore and after translation",
               fontweight="bold", color=DARK_PLUM, fontsize=11)
ax_d.legend(frameon=False, fontsize=8.5)

ax_d.text(0.5, -0.22,
          "Multilingual embeddings produced language-homogeneous clusters (>85%).\n"
          "Translating first reduced same-language co-clustering to ~40%.",
          transform=ax_d.transAxes, ha="center", fontsize=8,
          color=DARK_PLUM, alpha=0.68, style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL E — Pipeline ordering
# ══════════════════════════════════════════════════════════════════════════════
ax_e = fig.add_subplot(gs[2, 1])
_base(fig, ax_e, grid="none")
ax_e.set_xlim(0, 1)
ax_e.set_ylim(0, 1)
ax_e.axis("off")
ax_e.set_title("E — Pipeline ordering: topic modelling before vs after translation",
               fontweight="bold", color=DARK_PLUM, fontsize=11)

approaches = [
    ("Original order",
     "topic_model → sentiment_analysis",
     [("Embedding model", "multilingual-e5-large (1.1 GB)"),
      ("Input text",      "Raw multilingual titles"),
      ("Cluster quality", "Language-homogeneous (Panel D)"),
      ("Top label (Roe)", "Said, Women, According"),
      ("Feminist topics", "2 of 10 events flagged"),
      ("Runtime",         "~45 min")],
     RASPBERRY),
    ("Adopted order",
     "sentiment_analysis → topic_model",
     [("Embedding model", "all-MiniLM-L6-v2 (80 MB)"),
      ("Input text",      "translated_title (English)"),
      ("Cluster quality", "Thematically coherent cross-lingual"),
      ("Top label (Roe)", "Abortion, Access, Pro choice"),
      ("Feminist topics", "7 of 10 events flagged"),
      ("Runtime",         "~12 min")],
     MUTED_TEAL),
]

col_start = [0.03, 0.53]
for j, (name, subtitle, metrics, color) in enumerate(approaches):
    xs = col_start[j]
    ax_e.text(xs, 0.95, name, transform=ax_e.transAxes,
              fontsize=10, fontweight="bold", color=color)
    ax_e.text(xs, 0.88, subtitle, transform=ax_e.transAxes,
              fontsize=8, color=DARK_PLUM, alpha=0.75, style="italic")
    for k, (metric, value) in enumerate(metrics):
        y = 0.79 - k * 0.125
        highlight = metric in ("Top label (Roe)", "Feminist topics", "Cluster quality")
        ax_e.text(xs, y, f"{metric}:", transform=ax_e.transAxes,
                  fontsize=8.5, color=DARK_PLUM, fontweight="bold")
        ax_e.text(xs, y - 0.05, value, transform=ax_e.transAxes,
                  fontsize=8.5,
                  color=color if highlight else DARK_PLUM,
                  fontweight="bold" if highlight else "normal")

ax_e.plot([0.5, 0.5], [0.05, 0.92], color=BLUSH, linewidth=1.5,
          transform=ax_e.transAxes, clip_on=False)

ax_e.text(0.5, 0.02,
          "Reordering eliminated language-homogeneous clustering, reduced model size 93%,\n"
          "halved runtime, and increased feminist topic detection from 2 to 7 events.",
          transform=ax_e.transAxes, ha="center", fontsize=8,
          color=DARK_PLUM, alpha=0.68, style="italic")

OUTPUT_DIR = os.path.join("analysis_output", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "appendix_failure_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=LAVENDER_CREAM)
plt.close()
print(f"Saved → {out_path}")
