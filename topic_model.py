"""
topic_model.py — BERTopic Topic Modelling for Women's Protest News Corpus
==========================================================================
Fits a BERTopic model on the full corpus, assigns a topic to every article,
and produces per-publisher and per-event topic distribution tables.

Two-pass topic modelling
-------------------------
❶  GLOBAL model  — fit on the full corpus to discover overarching themes that
   span all events and outlets. These topic_id / topic_label columns are what
   sentiment_analysis.py uses for cross-event sentiment comparisons.

❷  PER-EVENT models — a separate BERTopic is fit on each event's articles to
   discover the fine-grained topics specific to that event's news cycle.
   Results are saved to a dedicated folder (analysis_output/event_topics/) and
   also attached to the article dataframe as event_topic_id / event_topic_label.

Research design
---------------
We characterise the ENTIRE news environment during each protest event week —
not just protest-relevant articles. BERTopic is used to discover what topics
filled the news cycle, allowing us to:
  - Compare topic distributions across ideologically different outlets
  - Compare protest-week topic distributions against matched control weeks
  - Measure what fraction of coverage each topic (including protest-related)
    occupied per outlet per event

Why BERTopic over KMeans/Ward/DBSCAN
--------------------------------------
  - KMeans/Ward require specifying k in advance; topic count is not known
  - DBSCAN is sensitive to hyperparameters and struggles with high-dimensional text
  - BERTopic uses sentence embeddings + HDBSCAN to find k automatically,
    and produces human-readable topic labels via TF-IDF keyphrases
  - Returns a probability distribution over topics per document, which is
    exactly what we need for outlet-level comparison

Multilingual design
-------------------
  - Uses paraphrase-multilingual-MiniLM-L12-v2 so articles in EN/DE/FR/ES
    about the same event cluster by THEME rather than by language
  - min_topic_size is kept small (5) to allow fine-grained topic discovery;
    use --reduce-topics N to merge down to a target count after fitting
  - Post-hoc reduction via BERTopic's reduce_topics() merges the smallest
    topics first while preserving the most coherent ones

Dependencies
------------
    pip install bertopic sentence-transformers pandas pyarrow

Usage
-----
    python topic_model.py                            # full run, auto topics
    python topic_model.py --input path/to/corpus.parquet
    python topic_model.py --min-topic-size 5         # finer global topics (default)
    python topic_model.py --reduce-topics 50         # merge global topics down to ~50
    python topic_model.py --top-n-words 10           # words per topic label
    python topic_model.py --event-min-topic-size 3   # finer per-event topics (default)
    python topic_model.py --no-event-topics          # skip per-event modelling
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------------------
# Multilingual stopwords (EN / DE / FR / ES)
# ---------------------------------------------------------------------------
STOPWORDS = (
    # English
    ["the","and","to","of","in","is","it","that","for","he","his","was","with",
     "on","are","at","as","this","be","by","from","or","an","but","not","they",
     "we","have","had","were","been","has","their","which","who","its","said",
     "also","more","one","would","about","up","out","so","can","all","when","if",
     "into","than","do","there","over","what","no","just","some","two","will",
     "after","other","her","him","she","our","your","my","us","me","you","them",
     "did","how","may","while","then","such","these","those","before","between",
     "through","now","only","new","could","should","both","well","since","even",
     "most","many","any","very","same","because","time","first","way","each",
     "under","during","without"] +
    # German
    ["die","der","und","das","den","zu","von","mit","sich","ist","des","ein",
     "eine","auch","auf","an","dem","im","es","für","sind","nicht","wie","sie",
     "werden","hat","war","bei","wird","als","durch","einer","oder","um","nach",
     "noch","aber","aus","am","haben","wenn","mehr","nur","doch","beim","zur",
     "worden","wurde","wurden","sei","ob","da","wir","ihr","uns","man","kann",
     "muss","soll","wird","worden","hatte","haben","wäre","müssen","sollen"] +
    # French
    ["de","le","la","les","et","des","en","un","une","du","au","qui","que","pas",
     "sur","est","par","ou","dans","il","elle","ils","ce","se","si","mais","nous",
     "vous","leur","plus","tout","cette","ces","son","ses","ont","été","aussi",
     "après","avant","même","bien","peut","comme","avec","très","non","ni","lui",
     "leur","dont","car","donc","puis","cela","cet","cette","entre","sous",
     "vers","sans","lors","dès","chez","fait","tout","tous","toutes"] +
    # Spanish
    ["el","los","del","las","que","en","al","lo","con","una","por","para","su",
     "se","es","más","no","si","ha","son","sus","le","me","ya","pero","como",
     "cuando","bien","donde","ser","un","ante","así","sobre","tan","hasta",
     "desde","entre","sin","dos","este","esta","estos","estas","ese","esa",
     "esos","esas","siendo","sido","fue","eran","era","han","había","tienen",
     "tiene","hacer","hace","porque","aunque","además","durante","mediante"]
)


# ══════════════════════════════════════════════════════════════════════════════
# ❶  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_INPUT        = os.path.join("news_output", "corpus_all.parquet")
OUTPUT_DIR           = "analysis_output"
EVENT_TOPICS_DIR     = os.path.join(OUTPUT_DIR, "event_topics")

# Multilingual model — clusters by theme across EN/DE/FR/ES rather than by language
EMBEDDING_MODEL      = "paraphrase-multilingual-MiniLM-L12-v2"

# Global model defaults
MIN_TOPIC_SIZE       = 5
TOP_N_WORDS          = 10
REDUCE_TOPICS        = None  # None = keep all discovered topics

# Per-event model defaults — smaller because event subsets are much smaller
EVENT_MIN_TOPIC_SIZE = 3


# ══════════════════════════════════════════════════════════════════════════════
# ❷  LOAD CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus(path: str) -> pd.DataFrame:
    print(f"Loading corpus from {path} ...")
    df = pd.read_parquet(path)

    required = {"publisher", "title", "body", "event_label", "event_type", "is_control"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Corpus is missing expected columns: {missing}")

    df = df[df["body"].fillna("").str.strip().str.len() > 100].copy()
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
    df["is_control"] = df["is_control"].astype(bool)
    df["publisher"] = (
        df["publisher"].astype(str)
        .str.extract(r"\.([A-Za-z0-9]+)")[0]
        .fillna(df["publisher"].astype(str))
    )

    print(f"  {len(df):,} articles loaded "
          f"({df['publisher'].nunique()} publishers, "
          f"{df['event_label'].nunique()} event windows)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ❸  BUILD A BERTOPIC MODEL  (shared helper)
# ══════════════════════════════════════════════════════════════════════════════

def build_topic_model(
    docs: list[str],
    embedding_model: SentenceTransformer,
    min_topic_size: int,
    top_n_words: int,
    reduce_topics: int | None = None,
    label: str = "",
) -> tuple[BERTopic, list[int]]:
    """Fit a BERTopic model on `docs` and return (model, topic_assignments)."""

    vectorizer_model = CountVectorizer(
        stop_words=STOPWORDS,
        ngram_range=(1, 2),
        min_df=2,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        calculate_probabilities=False,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(docs)

    n_topics   = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    prefix     = f"  [{label}] " if label else "  "
    print(f"{prefix}Topics found: {n_topics}  |  "
          f"Outliers: {n_outliers} ({100 * n_outliers / len(topics):.1f}%)")

    if reduce_topics and n_topics > reduce_topics:
        print(f"{prefix}Reducing to ~{reduce_topics} topics ...")
        topic_model.reduce_topics(docs, nr_topics=reduce_topics)
        topics   = topic_model.topics_
        n_after  = len(set(topics)) - (1 if -1 in topics else 0)
        print(f"{prefix}Topics after reduction: {n_after}")

    return topic_model, list(topics)


# ══════════════════════════════════════════════════════════════════════════════
# ❹  GLOBAL MODEL
# ══════════════════════════════════════════════════════════════════════════════

def fit_global_model(
    df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    min_topic_size: int,
    top_n_words: int,
    reduce_topics: int | None,
) -> tuple[BERTopic, list[int]]:
    print(f"\nFitting GLOBAL BERTopic model on {len(df):,} articles ...")
    print(f"  min_topic_size={min_topic_size}, top_n_words={top_n_words}")
    if reduce_topics:
        print(f"  Will reduce to ~{reduce_topics} topics after fitting")

    return build_topic_model(
        docs=df["text"].tolist(),
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        reduce_topics=reduce_topics,
        label="global",
    )


def attach_global_topics(
    df: pd.DataFrame,
    topic_model: BERTopic,
    topics: list[int],
) -> pd.DataFrame:
    df = df.copy()
    df["topic_id"] = topics
    label_map = dict(zip(
        topic_model.get_topic_info()["Topic"],
        topic_model.get_topic_info()["Name"],
    ))
    df["topic_label"] = df["topic_id"].map(label_map).fillna("outlier")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ❺  PER-EVENT MODELS
# ══════════════════════════════════════════════════════════════════════════════

def fit_event_models(
    df: pd.DataFrame,
    embedding_model: SentenceTransformer,
    event_min_topic_size: int,
    top_n_words: int,
) -> pd.DataFrame:
    """
    Fit a separate BERTopic model for each event_label and attach
    event_topic_id / event_topic_label columns to df.
    Also saves per-event CSVs to EVENT_TOPICS_DIR.
    """
    os.makedirs(EVENT_TOPICS_DIR, exist_ok=True)

    events = sorted(df["event_label"].unique())
    print(f"\nFitting PER-EVENT BERTopic models for {len(events)} events ...")
    print(f"  event_min_topic_size={event_min_topic_size}, top_n_words={top_n_words}")

    df = df.copy()
    df["event_topic_id"]    = -1
    df["event_topic_label"] = "outlier"

    all_event_dists = []

    for event in events:
        mask  = df["event_label"] == event
        subset = df[mask].copy()

        if len(subset) < event_min_topic_size * 2:
            print(f"  [{event}] Skipping — too few articles ({len(subset)})")
            continue

        docs = subset["text"].tolist()

        try:
            event_model, event_topics = build_topic_model(
                docs=docs,
                embedding_model=embedding_model,
                min_topic_size=event_min_topic_size,
                top_n_words=top_n_words,
                label=event,
            )
        except Exception as e:
            print(f"  [{event}] ⚠️  Model failed: {e}")
            continue

        # Attach event-level topic assignments back to the main df
        topic_info = event_model.get_topic_info()
        label_map  = dict(zip(topic_info["Topic"], topic_info["Name"]))
        subset_idx = subset.index

        df.loc[subset_idx, "event_topic_id"]    = event_topics
        df.loc[subset_idx, "event_topic_label"] = [
            label_map.get(t, "outlier") for t in event_topics
        ]

        # ── Per-event topic info CSV ──────────────────────────────────────────
        safe_name = event.replace(" ", "_").replace("/", "-")

        topic_info["event_label"] = event
        topic_info.to_csv(
            os.path.join(EVENT_TOPICS_DIR, f"{safe_name}_topic_info.csv"),
            index=False,
        )

        # ── Per-event topic distribution (publisher × topic) ──────────────────
        subset = subset.copy()
        subset["event_topic_label"] = df.loc[subset_idx, "event_topic_label"].values

        dist = topic_distribution(subset, ["publisher", "is_control", "event_topic_label"])
        dist["event_label"] = event
        dist.to_csv(
            os.path.join(EVENT_TOPICS_DIR, f"{safe_name}_topic_dist.csv"),
            index=False,
        )

        all_event_dists.append(dist)
        print(f"  [{event}] Saved CSVs → {EVENT_TOPICS_DIR}/{safe_name}_*.csv")

    # Combined long-format table across all events
    if all_event_dists:
        combined = pd.concat(all_event_dists, ignore_index=True)
        combined.to_csv(
            os.path.join(EVENT_TOPICS_DIR, "all_events_topic_dist.csv"),
            index=False,
        )
        print(f"\n  Saved combined event topic dist → {EVENT_TOPICS_DIR}/all_events_topic_dist.csv")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ❻  DISTRIBUTION TABLES  (shared helper)
# ══════════════════════════════════════════════════════════════════════════════

def topic_distribution(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Topic share (%) within each group defined by group_cols.

    group_cols must include the topic column (topic_label or event_topic_label)
    as its last element. base_cols are everything except the topic column, and
    are used to compute per-group totals for the share calculation.
    """
    # The topic column is always the last entry in group_cols
    topic_col = group_cols[-1]
    base_cols = group_cols[:-1]

    counts = df.groupby(group_cols).size().reset_index(name="n_articles")
    totals = df.groupby(base_cols).size().reset_index(name="total_articles")
    merged = counts.merge(totals, on=base_cols)
    merged["topic_share_pct"] = (
        merged["n_articles"] / merged["total_articles"] * 100
    ).round(2)
    return merged.sort_values(
        base_cols + [topic_col, "topic_share_pct"],
        ascending=[True] * len(base_cols) + [True, False],
    )


# ══════════════════════════════════════════════════════════════════════════════
# ❼  SAVE GLOBAL OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_global_outputs(df: pd.DataFrame, topic_model: BERTopic) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Article-level results
    out_cols = [c for c in df.columns if c != "text"]
    df[out_cols].to_parquet(
        os.path.join(OUTPUT_DIR, "articles_with_topics.parquet"), index=False)
    print(f"\n  Saved article-level results → {OUTPUT_DIR}/articles_with_topics.parquet")

    # Global topic info
    topic_model.get_topic_info().to_csv(
        os.path.join(OUTPUT_DIR, "topic_info.csv"), index=False)
    print(f"  Saved global topic info     → {OUTPUT_DIR}/topic_info.csv")

    # Global distribution tables
    distribution_configs = [
        ("publisher_event",    ["publisher", "event_label", "is_control", "topic_label"]),
        ("event",              ["event_label", "is_control", "topic_label"]),
        ("protest_vs_control", ["is_control", "topic_label"]),
        ("event_type",         ["event_type", "is_control", "topic_label"]),
    ]

    for name, group_cols in distribution_configs:
        if not all(c in df.columns for c in group_cols):
            print(f"  ⚠️  Skipping {name}: missing columns {set(group_cols) - set(df.columns)}")
            continue
        out = os.path.join(OUTPUT_DIR, f"topic_dist_{name}.csv")
        topic_distribution(df, group_cols).to_csv(out, index=False)
        print(f"  Saved {name:<24s} → {out}")

    # Save model for potential reuse
    topic_model.save(os.path.join(OUTPUT_DIR, "bertopic_model"))
    print(f"  Saved BERTopic model        → {OUTPUT_DIR}/bertopic_model")


# ══════════════════════════════════════════════════════════════════════════════
# ❽  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",                default=DEFAULT_INPUT)
    parser.add_argument("--min-topic-size",       type=int,  default=MIN_TOPIC_SIZE)
    parser.add_argument("--top-n-words",          type=int,  default=TOP_N_WORDS)
    parser.add_argument("--event-min-topic-size", type=int,  default=EVENT_MIN_TOPIC_SIZE)
    parser.add_argument(
        "--reduce-topics",
        type=int,
        default=REDUCE_TOPICS,
        metavar="N",
        help="After fitting the global model, merge topics down to ~N. "
             "Omit to keep all discovered topics.",
    )
    parser.add_argument(
        "--no-event-topics",
        action="store_true",
        help="Skip per-event topic modelling and only run the global model.",
    )
    args = parser.parse_args()

    df = load_corpus(args.input)

    # Load embedding model once — shared by global and per-event passes
    print(f"\nLoading embedding model: {EMBEDDING_MODEL} ...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # ── Pass 1: global model ──────────────────────────────────────────────────
    global_model, global_topics = fit_global_model(
        df,
        embedding_model=embedding_model,
        min_topic_size=args.min_topic_size,
        top_n_words=args.top_n_words,
        reduce_topics=args.reduce_topics,
    )
    df = attach_global_topics(df, global_model, global_topics)

    # ── Pass 2: per-event models ──────────────────────────────────────────────
    if not args.no_event_topics:
        df = fit_event_models(
            df,
            embedding_model=embedding_model,
            event_min_topic_size=args.event_min_topic_size,
            top_n_words=args.top_n_words,
        )

    save_global_outputs(df, global_model)

    print("\n✅  Topic modelling complete.")
    print(f"    Global outputs  → {OUTPUT_DIR}/")
    if not args.no_event_topics:
        print(f"    Per-event CSVs  → {EVENT_TOPICS_DIR}/")
    print(f"    Next step: run  python sentiment_analysis.py")


if __name__ == "__main__":
    main()
