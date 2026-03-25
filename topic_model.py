"""
topic_model.py — BERTopic Topic Modelling for Women's Protest News Corpus
==========================================================================
Fits a BERTopic model on the full corpus, assigns a topic to every article,
and produces per-publisher and per-event topic distribution tables.

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

Dependencies
------------
    pip install bertopic sentence-transformers pandas pyarrow

Usage
-----
    python topic_model.py                          # full run
    python topic_model.py --input path/to/corpus.parquet
    python topic_model.py --min-topic-size 10      # smaller topics allowed
    python topic_model.py --top-n-words 8          # words per topic label
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


# ══════════════════════════════════════════════════════════════════════════════
# ❶  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_INPUT   = os.path.join("news_output", "corpus_all.parquet")
OUTPUT_DIR      = "analysis_output"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, good quality
MIN_TOPIC_SIZE  = 15                    # minimum articles per topic
TOP_N_WORDS     = 8                     # keywords shown per topic label


# ══════════════════════════════════════════════════════════════════════════════
# ❷  LOAD CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus(path: str) -> pd.DataFrame:
    print(f"Loading corpus from {path} ...")
    df = pd.read_parquet(path)
    df = df[df["body"].str.strip().str.len() > 100].copy()
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
    print(f"  {len(df):,} articles loaded ({df['publisher'].nunique()} publishers, "
          f"{df['event_label'].nunique()} event windows)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ❸  FIT BERTOPIC
# ══════════════════════════════════════════════════════════════════════════════

def fit_topic_model(df: pd.DataFrame,
                    min_topic_size: int,
                    top_n_words: int) -> tuple[BERTopic, list[int]]:
    print(f"\nFitting BERTopic (embedding model: {EMBEDDING_MODEL}) ...")
    print(f"  min_topic_size={min_topic_size}, top_n_words={top_n_words}")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        calculate_probabilities=False,
        verbose=True,
    )

    docs   = df["text"].tolist()
    topics, _ = topic_model.fit_transform(docs)

    n_topics   = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    print(f"\n  Topics found     : {n_topics}")
    print(f"  Outlier articles : {n_outliers} ({100*n_outliers/len(topics):.1f}%)")

    return topic_model, topics


# ══════════════════════════════════════════════════════════════════════════════
# ❹  ATTACH RESULTS TO DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════

def attach_topics(df: pd.DataFrame,
                  topic_model: BERTopic,
                  topics: list[int]) -> pd.DataFrame:
    df = df.copy()
    df["topic_id"] = topics
    topic_info = topic_model.get_topic_info()
    label_map  = dict(zip(topic_info["Topic"], topic_info["Name"]))
    df["topic_label"] = df["topic_id"].map(label_map).fillna("outlier")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ❺  DISTRIBUTION TABLES
# ══════════════════════════════════════════════════════════════════════════════

def topic_distribution(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Topic share (%) within each group defined by group_cols."""
    counts = (df.groupby(group_cols + ["topic_label"])
                .size().reset_index(name="n_articles"))
    totals = (df.groupby(group_cols)
                .size().reset_index(name="total_articles"))
    merged = counts.merge(totals, on=group_cols)
    merged["topic_share_pct"] = (merged["n_articles"] / merged["total_articles"] * 100).round(2)
    return merged.sort_values(
        group_cols + ["topic_share_pct"],
        ascending=[True] * len(group_cols) + [False]
    )


# ══════════════════════════════════════════════════════════════════════════════
# ❻  SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(df: pd.DataFrame, topic_model: BERTopic) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Article-level results
    df.drop(columns=["text"]).to_parquet(
        os.path.join(OUTPUT_DIR, "articles_with_topics.parquet"), index=False)
    print(f"\n  Saved article-level results → {OUTPUT_DIR}/articles_with_topics.parquet")

    # Topic info table
    topic_model.get_topic_info().to_csv(
        os.path.join(OUTPUT_DIR, "topic_info.csv"), index=False)
    print(f"  Saved topic info            → {OUTPUT_DIR}/topic_info.csv")

    # Distribution tables
    for name, group_cols in [
        ("publisher_event",    ["publisher", "event_label", "is_control"]),
        ("event",              ["event_label", "is_control"]),
        ("protest_vs_control", ["is_control"]),
    ]:
        out = os.path.join(OUTPUT_DIR, f"topic_dist_{name}.csv")
        topic_distribution(df, group_cols).to_csv(out, index=False)
        print(f"  Saved {name:<24s} → {out}")

    # Save model for reuse in sentiment step
    topic_model.save(os.path.join(OUTPUT_DIR, "bertopic_model"))
    print(f"  Saved BERTopic model        → {OUTPUT_DIR}/bertopic_model")


# ══════════════════════════════════════════════════════════════════════════════
# ❼  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          default=DEFAULT_INPUT)
    parser.add_argument("--min-topic-size", type=int, default=MIN_TOPIC_SIZE)
    parser.add_argument("--top-n-words",    type=int, default=TOP_N_WORDS)
    args = parser.parse_args()

    df = load_corpus(args.input)
    topic_model, topics = fit_topic_model(df, args.min_topic_size, args.top_n_words)
    df = attach_topics(df, topic_model, topics)
    save_outputs(df, topic_model)

    print("\n✅  Topic modelling complete.")
    print(f"    Next step: run  python sentiment_analysis.py")


if __name__ == "__main__":
    main()
