"""
collect_articles.py — Women's Protest Events: General News Corpus Collector
============================================================================
Collects a REPRESENTATIVE SAMPLE OF ALL NEWS published during each event
window — not just protest coverage. The goal is to measure what fraction
of general news coverage is about the protest, and how it's framed.

RESEARCH DESIGN
---------------
For each event window (and matched control week), we collect up to
MAX_PER_PUBLISHER articles from each publisher independently.
This gives a balanced cross-publisher sample suitable for:
  - Estimating proportion of protest coverage per outlet
  - Comparing sentiment/framing across ideological lines
  - Cross-event and protest vs. control comparisons

SAMPLE SIZE RATIONALE
---------------------
To estimate a proportion p with 95% confidence:
  n=97  → ±10% margin of error  (minimum acceptable)
  n=200 → ±7%  margin of error  (recommended)
  n=385 → ±5%  margin of error  (ideal if CC-News coverage allows)

We target 200 per publisher for protest event windows, and 100 per publisher
for control weeks (sufficient for baseline comparison while halving scrape time).

WHY PER-PUBLISHER CRAWLING
--------------------------
CCNewsCrawler streams WARCs sequentially. A global article cap fills up
with whichever publisher's WARCs appear first, giving you 297 Telegraph
articles and 3 Daily Mail. Crawling each publisher separately gives a
balanced sample.

IWD 2018 NOTE
-------------
The Spanish Feminist Strike (8-M) and the first Aurat March (Pakistan) both
occurred on March 8, 2018 — the same day as IWD 2018. Scraping them as
separate events would collect identical articles three times. Instead, they
are merged into a single window labelled IWD_2018_Global. Sub-event tagging
(which articles mention the Spanish strike vs. Aurat March vs. IWD generally)
is done at analysis time using keyword filters on title/body.

Dependencies:  pip install fundus pandas pyarrow tqdm
Usage:
    python collect_articles.py --event Womens_March_2017        # test one event
    python collect_articles.py                                   # full run
    python collect_articles.py --no-control                      # skip controls
    python collect_articles.py --reset-all                       # wipe & restart
    python collect_articles.py --per-publisher 200               # articles per pub
    python collect_articles.py --control-per-publisher 100       # control articles per pub
    python collect_articles.py --processes 2                     # parallelism
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta

import pandas as pd
from fundus import CCNewsCrawler, PublisherCollection
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# ❶  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR                  = "news_output"
PROGRESS_FILE               = os.path.join(OUTPUT_DIR, "progress.json")
MAX_PER_PUBLISHER           = 200   # target sample per publisher per protest event
CONTROL_PER_PUBLISHER       = 100   # reduced target for control weeks
DEFAULT_PROCESSES           = 2

OUTPUT_COLUMNS = [
    "url", "publisher", "title", "body", "authors", "topics",
    "publishing_date", "event_label", "event_type", "is_control",
]


# ══════════════════════════════════════════════════════════════════════════════
# ❷  PUBLISHERS — all names verified against your installed Fundus version
# ══════════════════════════════════════════════════════════════════════════════

PUBLISHERS = [
    # ── US ───────────────────────────────────────────────────────────────────
    PublisherCollection.us.APNews,
    PublisherCollection.us.Reuters,
    PublisherCollection.us.VoiceOfAmerica,
    PublisherCollection.us.WashingtonPost,
    PublisherCollection.us.TheNewYorker,
    PublisherCollection.us.TheNation,
    PublisherCollection.us.TheIntercept,
    PublisherCollection.us.RollingStone,
    PublisherCollection.us.LATimes,
    PublisherCollection.us.BusinessInsider,
    PublisherCollection.us.CNBC,
    PublisherCollection.us.FoxNews,
    PublisherCollection.us.WashingtonTimes,
    PublisherCollection.us.FreeBeacon,
    PublisherCollection.us.TheGatewayPundit,
    # ── UK ───────────────────────────────────────────────────────────────────
    PublisherCollection.uk.BBC,
    PublisherCollection.uk.TheGuardian,
    PublisherCollection.uk.TheIndependent,
    PublisherCollection.uk.EuronewsEN,
    PublisherCollection.uk.iNews,
    PublisherCollection.uk.DailyMail,
    PublisherCollection.uk.TheTelegraph,
    PublisherCollection.uk.TheSun,
    # ── DE ───────────────────────────────────────────────────────────────────
    PublisherCollection.de.DW,
    PublisherCollection.de.SpiegelOnline,
    PublisherCollection.de.DieZeit,
    PublisherCollection.de.Tagesschau,
    PublisherCollection.de.FAZ,
    PublisherCollection.de.Taz,
    # ── AT ───────────────────────────────────────────────────────────────────
    PublisherCollection.at.DerStandard,
    PublisherCollection.at.DiePresse,
    PublisherCollection.at.ORF,
    # ── CA ───────────────────────────────────────────────────────────────────
    PublisherCollection.ca.CBCNews,
    PublisherCollection.ca.NationalPost,
    PublisherCollection.ca.TheGlobeAndMail,
    # ── FR ───────────────────────────────────────────────────────────────────
    PublisherCollection.fr.LeMonde,
    PublisherCollection.fr.LeFigaro,
    PublisherCollection.fr.EuronewsFR,
    # ── ES ───────────────────────────────────────────────────────────────────
    PublisherCollection.es.ElPais,
    PublisherCollection.es.ElMundo,
    PublisherCollection.es.ElDiario,
    PublisherCollection.es.LaVanguardia,
    PublisherCollection.es.ABC,
    PublisherCollection.es.Publico,
    # ── IL ───────────────────────────────────────────────────────────────────
    PublisherCollection.il.IsraelNachrichten,
]


# ══════════════════════════════════════════════════════════════════════════════
# ❸  EVENTS
# ══════════════════════════════════════════════════════════════════════════════

EVENTS: list[dict] = [
    {"label": "Womens_March_2017",
     "event_type": "single",
     "window_start": date(2017, 1, 19), "window_end": date(2017, 1, 26),
     "note": "Jan 21; ~3-5M worldwide"},
    {"label": "International_Womens_Strike_2017",
     "event_type": "recurring",
     "window_start": date(2017, 3, 6),  "window_end": date(2017, 3, 13),
     "note": "First IWS; anchor Mar 8"},
    {"label": "MeToo_Protests_2017",
     "event_type": "sustained",
     "window_start": date(2017, 10, 15), "window_end": date(2017, 10, 22),
     "note": "Milano tweet Oct 15; Weinstein explosion"},
    # NOTE: Spanish Feminist Strike and Aurat March both occurred on IWD
    # (Mar 8, 2018) and are merged into this single window. Sub-event tagging
    # is handled at analysis time with keyword filters.
    {"label": "IWD_2018_Global",
     "event_type": "single",
     "window_start": date(2018, 3, 6),  "window_end": date(2018, 3, 13),
     "note": "IWD 2018; Spanish Feminist Strike ~6M on strike; first Aurat March Karachi"},
    {"label": "Swiss_Womens_Strike_2019",
     "event_type": "single",
     "window_start": date(2019, 6, 12), "window_end": date(2019, 6, 19),
     "note": "Jun 14; largest Swiss protest since 1991"},
    {"label": "Polish_Womens_Strike_2020",
     "event_type": "sustained",
     "window_start": date(2020, 10, 22), "window_end": date(2020, 10, 29),
     "note": "Constitutional Tribunal ruling Oct 22"},
    {"label": "Sarah_Everard_Vigils_2021",
     "event_type": "single",
     "window_start": date(2021, 3, 11), "window_end": date(2021, 3, 18),
     "note": "Clapham Common vigil Mar 13"},
    {"label": "Roe_Leak_Protests_2022",
     "event_type": "sustained",
     "window_start": date(2022, 5, 2),  "window_end": date(2022, 5, 9),
     "note": "Politico leak May 2; protests May 3-4"},
    {"label": "Women_Life_Freedom_2022",
     "event_type": "sustained",
     "window_start": date(2022, 9, 16), "window_end": date(2022, 9, 23),
     "note": "Mahsa Amini death Sep 16"},
    {"label": "Israeli_Womens_Protests_2023",
     "event_type": "single",
     "window_start": date(2023, 2, 27), "window_end": date(2023, 3, 6),
     "note": "Women-led protests against judicial overhaul"},
    {"label": "IWD_2019", "event_type": "recurring",
     "window_start": date(2019, 3, 6), "window_end": date(2019, 3, 13)},
    {"label": "IWD_2020", "event_type": "recurring",
     "window_start": date(2020, 3, 6), "window_end": date(2020, 3, 13)},
    {"label": "IWD_2021", "event_type": "recurring",
     "window_start": date(2021, 3, 6), "window_end": date(2021, 3, 13)},
    {"label": "IWD_2022", "event_type": "recurring",
     "window_start": date(2022, 3, 6), "window_end": date(2022, 3, 13)},
    {"label": "IWD_2023", "event_type": "recurring",
     "window_start": date(2023, 3, 6), "window_end": date(2023, 3, 13)},
]


# ══════════════════════════════════════════════════════════════════════════════
# ❹  CONTROL WEEKS
# ══════════════════════════════════════════════════════════════════════════════

_CONTROL_CANDIDATES = [2015, 2016, 2024]


def build_control_week(event: dict) -> dict | None:
    ws, we     = event["window_start"], event["window_end"]
    iso_week   = ws.isocalendar()[1]
    delta_days = (we - ws).days
    for cy in _CONTROL_CANDIDATES:
        jan4      = date(cy, 1, 4)
        w1_monday = jan4 - timedelta(days=jan4.weekday())
        cw_start  = w1_monday + timedelta(weeks=iso_week - 1)
        cw_end    = cw_start + timedelta(days=delta_days)
        if cw_start.year != cy:
            continue
        return {
            "label":        f"CONTROL_{event['label']}_{cy}",
            "event_type":   event["event_type"],
            "window_start": cw_start,
            "window_end":   cw_end,
            "is_control":   True,
            "matched_to":   event["label"],
        }
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ❺  DATE WINDOW
# ══════════════════════════════════════════════════════════════════════════════

def build_window(event: dict) -> tuple[datetime, datetime]:
    ws, we = event["window_start"], event["window_end"]
    et     = event.get("event_type", "single")
    from_d = ws - timedelta(days=2) if et in ("single", "recurring") else ws
    to_d   = we + timedelta(days=1)
    return (
        datetime(from_d.year, from_d.month, from_d.day, 0,  0,  0),
        datetime(to_d.year,   to_d.month,   to_d.day,   23, 59, 59),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ❻  PROGRESS
# ══════════════════════════════════════════════════════════════════════════════

def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(prog: dict) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump(prog, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# ❼  CRAWL ONE PUBLISHER FOR ONE EVENT
# ══════════════════════════════════════════════════════════════════════════════

def crawl_publisher(publisher, start_dt: datetime, end_dt: datetime,
                    max_articles: int, processes: int,
                    label: str, event_type: str,
                    is_control: bool) -> list[dict]:
    """Crawl a single publisher for the given window. Returns list of records."""
    crawler = CCNewsCrawler(
        publisher,
        start=start_dt,
        end=end_dt,
        processes=processes,
    )
    records = []
    try:
        for article in crawler.crawl(max_articles=max_articles):
            pub_date = article.publishing_date
            records.append({
                "url":             str(article.html.responded_url),
                "publisher":       article.publisher,
                "title":           article.title     or "",
                "body":            article.plaintext or "",
                "authors":         ", ".join(article.authors) if article.authors else "",
                "topics":          ", ".join(article.topics)  if article.topics  else "",
                "publishing_date": str(pub_date.date()) if pub_date else "",
                "event_label":     label,
                "event_type":      event_type,
                "is_control":      is_control,
            })
    except Exception as e:
        # Don't let one publisher crash the whole event
        print(f"\n      ⚠️  {publisher}: {e}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# ❽  PROCESS ONE EVENT
# ══════════════════════════════════════════════════════════════════════════════

def process_event(event: dict, progress: dict,
                  per_publisher: int, control_per_publisher: int,
                  processes: int) -> bool:
    label      = event["label"]
    is_control = event.get("is_control", False)
    event_type = event.get("event_type", "single")
    start_dt, end_dt = build_window(event)

    # Use reduced cap for control weeks
    cap = control_per_publisher if is_control else per_publisher

    tag = "CONTROL" if is_control else "EVENT  "
    print(f"\n{'─'*68}")
    print(f"[{tag}] {label}")
    print(f"Window  : {event['window_start']} → {event['window_end']}")
    print(f"CC-News : {start_dt.date()} – {end_dt.date()}")
    print(f"Cap     : {cap} articles/publisher")
    if event.get("note"):
        print(f"Note    : {event['note']}")

    # Resume: track which publishers are done for this event
    ep = progress.get(label, {"done_publishers": [], "n_articles": 0})
    done_pubs = set(ep.get("done_publishers", []))

    # Check if fully complete
    if done_pubs.issuperset({str(p) for p in PUBLISHERS}):
        print(f"  ✅  Already complete ({ep['n_articles']:,} articles) — skipping")
        return True

    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)
    inc_dir = os.path.join(OUTPUT_DIR, label, "incremental")
    os.makedirs(inc_dir, exist_ok=True)

    todos = [p for p in PUBLISHERS if str(p) not in done_pubs]
    print(f"\n  {len(todos)} publishers to crawl "
          f"({len(done_pubs)} already done), {cap} articles each\n")

    pbar = tqdm(todos, desc="  Publishers", unit="pub", leave=False)
    for publisher in pbar:
        pub_name = str(publisher).split(".")[-1]   # e.g. "TheTelegraph"
        pbar.set_postfix({"pub": pub_name})

        records = crawl_publisher(
            publisher, start_dt, end_dt,
            max_articles=cap,
            processes=processes,
            label=label, event_type=event_type, is_control=is_control,
        )
        n = len(records)
        pbar.write(f"    {pub_name:<30s}  {n:>4} articles")

        if records:
            df_batch = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
            safe     = pub_name.replace(".", "_")
            df_batch.to_parquet(os.path.join(inc_dir, f"{safe}.parquet"), index=False)

        ep["done_publishers"].append(str(publisher))
        ep["n_articles"] = ep.get("n_articles", 0) + n
        progress[label]  = ep
        save_progress(progress)

    pbar.close()

    # Merge incremental → single event parquet
    inc_files = [
        os.path.join(inc_dir, fn)
        for fn in os.listdir(inc_dir) if fn.endswith(".parquet")
    ]
    if not inc_files:
        print(f"\n  ⚠️   No articles retrieved for any publisher")
        return False

    combined = pd.concat([pd.read_parquet(p) for p in inc_files], ignore_index=True)
    combined = combined[[c for c in OUTPUT_COLUMNS if c in combined.columns]]
    combined.drop_duplicates(subset=["url"], inplace=True)

    out_path = os.path.join(OUTPUT_DIR, label, f"{label}.parquet")
    combined.to_parquet(out_path, index=False)
    combined.to_csv(out_path.replace(".parquet", ".csv"), index=False)

    n_total = len(combined)
    n_pubs  = combined["publisher"].nunique()
    avg_len = int(combined["body"].str.len().mean())
    pub_counts = combined["publisher"].value_counts()

    print(f"\n  ✅  {n_total:,} articles  |  {n_pubs} publishers  "
          f"|  avg body {avg_len:,} chars")
    print(f"  Publisher breakdown:")
    for pub, cnt in pub_counts.items():
        print(f"    {pub:<35s} {cnt:>4}")
    print(f"  Saved → {out_path}")

    progress[label]["n_articles"] = n_total
    save_progress(progress)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# ❾  BUILD COMBINED CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def build_corpus() -> None:
    parquet_files = [
        os.path.join(root, fn)
        for root, _, files in os.walk(OUTPUT_DIR)
        for fn in files
        if fn.endswith(".parquet")
        and "corpus" not in fn
        and "incremental" not in root
    ]
    if not parquet_files:
        print("  No parquet files found yet.")
        return

    corpus = pd.concat(
        [pd.read_parquet(p) for p in parquet_files], ignore_index=True
    )
    corpus.drop_duplicates(subset=["url"], inplace=True)

    out = os.path.join(OUTPUT_DIR, "corpus_all.parquet")
    corpus.to_parquet(out, index=False)

    n_protest = corpus[~corpus["is_control"]]["event_label"].nunique()
    n_control = corpus[ corpus["is_control"]]["event_label"].nunique()

    print(f"\n{'═'*68}")
    print(f"Corpus: {len(corpus):,} total articles")
    print(f"  Protest event windows : {n_protest}")
    print(f"  Control week windows  : {n_control}")
    print(f"  Publishers            : {corpus['publisher'].nunique()}")
    print(f"  Avg body length       : {corpus['body'].str.len().mean():,.0f} chars")
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# ❿  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect general news corpus for protest event windows."
    )
    parser.add_argument("--event",                   metavar="LABEL")
    parser.add_argument("--no-control",              action="store_true")
    parser.add_argument("--reset-all",               action="store_true")
    parser.add_argument("--per-publisher",           type=int, default=MAX_PER_PUBLISHER,
                        help=f"Articles per publisher per protest event (default {MAX_PER_PUBLISHER})")
    parser.add_argument("--control-per-publisher",   type=int, default=CONTROL_PER_PUBLISHER,
                        help=f"Articles per publisher per control week (default {CONTROL_PER_PUBLISHER})")
    parser.add_argument("--processes",               type=int, default=DEFAULT_PROCESSES)
    parser.add_argument("--corpus-only",             action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.corpus_only:
        build_corpus()
        return

    if args.reset_all and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("✓ Progress wiped.\n")

    all_events = list(EVENTS)
    if not args.no_control:
        for ev in EVENTS:
            ctrl = build_control_week(ev)
            if ctrl:
                all_events.append(ctrl)

    if args.event:
        all_events = [e for e in all_events if e["label"] == args.event]
        if not all_events:
            print(f"❌  '{args.event}' not found. Labels:")
            for e in EVENTS:
                print(f"    {e['label']}")
            return

    n_ev = sum(1 for e in all_events if not e.get("is_control"))
    n_ct = sum(1 for e in all_events if e.get("is_control"))

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Women's Protest Events — General News Corpus (Fundus)          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Publishers              : {len(PUBLISHERS)}")
    print(f"  Per publisher (events)  : {args.per_publisher} articles (target)")
    print(f"  Per publisher (control) : {args.control_per_publisher} articles (target)")
    print(f"  Protest events          : {n_ev}")
    print(f"  Control weeks           : {n_ct}")
    est = (args.per_publisher * len(PUBLISHERS) * n_ev +
           args.control_per_publisher * len(PUBLISHERS) * n_ct)
    print(f"  Max corpus est          : ~{est:,} articles")
    print(f"  Output                  : {OUTPUT_DIR}/")
    print()

    progress   = load_progress()
    successes, failures = [], []

    for event in all_events:
        ok = process_event(event, progress,
                           per_publisher=args.per_publisher,
                           control_per_publisher=args.control_per_publisher,
                           processes=args.processes)
        (successes if ok else failures).append(event["label"])

    build_corpus()

    if failures:
        print(f"\n⚠️  Empty/failed events (thin CC-News coverage for that period):")
        for lbl in failures:
            print(f"  ❌  {lbl}")


if __name__ == "__main__":
    main()
