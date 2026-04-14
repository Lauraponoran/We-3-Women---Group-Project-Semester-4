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
    python topic_model.py --min-topic-size 5         # finer topics (default)
    python topic_model.py --reduce-topics 50         # merge down to ~50 topics
    python topic_model.py --top-n-words 10           # words per topic label
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

DEFAULT_INPUT   = os.path.join("news_output", "corpus_all.parquet")
OUTPUT_DIR      = "analysis_output"

# Multilingual model — clusters by theme across EN/DE/FR/ES rather than by language
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Keep small to allow fine-grained discovery; merge afterwards with --reduce-topics
MIN_TOPIC_SIZE  = 5

TOP_N_WORDS     = 10    # more keywords → easier to interpret each topic
REDUCE_TOPICS   = None  # set via --reduce-topics N; None = keep all discovered topics

# Actual columns from collect_articles.py:
# url, publisher, title, body, authors, topics, publishing_date,
# event_label, event_type, is_control


# ══════════════════════════════════════════════════════════════════════════════
# ❷  LOAD CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus(path: str) -> pd.DataFrame:
    print(f"Loading corpus from {path} ...")
    df = pd.read_parquet(path)

    # Ensure expected columns are present
    required = {"publisher", "title", "body", "event_label", "event_type", "is_control"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Corpus is missing expected columns: {missing}")

    # Drop articles with negligible body text
    df = df[df["body"].fillna("").str.strip().str.len() > 100].copy()

    # Combine title + body as the document text for BERTopic
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")

    # Normalise is_control to bool (it may be stored as bool or 0/1)
    df["is_control"] = df["is_control"].astype(bool)

    # publisher is stored as a repr string like "<PublisherEnum.APNews: ...>"
    # Extract a clean name for display
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
# ❸  FIT BERTOPIC
# ══════════════════════════════════════════════════════════════════════════════

def fit_topic_model(
    df: pd.DataFrame,
    min_topic_size: int,
    top_n_words: int,
    reduce_topics: int | None,
) -> tuple[BERTopic, list[int]]:

    print(f"\nFitting BERTopic (embedding model: {EMBEDDING_MODEL}) ...")
    print(f"  min_topic_size={min_topic_size}, top_n_words={top_n_words}")
    if reduce_topics:
        print(f"  Will reduce to ~{reduce_topics} topics after fitting")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # min_df=2 (not 3) because with small min_topic_size some topics have few docs
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
        verbose=True,
    )

    docs = df["text"].tolist()
    topics, _ = topic_model.fit_transform(docs)

    n_topics   = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    print(f"\n  Topics found (before reduction) : {n_topics}")
    print(f"  Outlier articles                : {n_outliers} ({100 * n_outliers / len(topics):.1f}%)")

    # --- optional post-hoc reduction ---
    if reduce_topics and n_topics > reduce_topics:
        print(f"\n  Reducing to ~{reduce_topics} topics ...")
        topic_model.reduce_topics(docs, nr_topics=reduce_topics)
        topics = topic_model.topics_  # updated assignments after reduction
        n_topics_after = len(set(topics)) - (1 if -1 in topics else 0)
        print(f"  Topics after reduction : {n_topics_after}")

    return topic_model, list(topics)


# ══════════════════════════════════════════════════════════════════════════════
# ❹  ATTACH RESULTS TO DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════

def attach_topics(
    df: pd.DataFrame,
    topic_model: BERTopic,
    topics: list[int],
) -> pd.DataFrame:
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
    merged["topic_share_pct"] = (
        merged["n_articles"] / merged["total_articles"] * 100
    ).round(2)
    return merged.sort_values(
        group_cols + ["topic_share_pct"],
        ascending=[True] * len(group_cols) + [False],
    )


# ══════════════════════════════════════════════════════════════════════════════
# ❻  SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(df: pd.DataFrame, topic_model: BERTopic) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Article-level results — drop the temporary text column
    out_cols = [c for c in df.columns if c != "text"]
    df[out_cols].to_parquet(
        os.path.join(OUTPUT_DIR, "articles_with_topics.parquet"), index=False)
    print(f"\n  Saved article-level results → {OUTPUT_DIR}/articles_with_topics.parquet")

    # Topic info table
    topic_model.get_topic_info().to_csv(
        os.path.join(OUTPUT_DIR, "topic_info.csv"), index=False)
    print(f"  Saved topic info            → {OUTPUT_DIR}/topic_info.csv")

    # Distribution tables
    distribution_configs = [
        # per publisher × event window × control flag
        ("publisher_event",    ["publisher", "event_label", "is_control"]),
        # per event window × control flag
        ("event",              ["event_label", "is_control"]),
        # protest weeks vs control weeks only
        ("protest_vs_control", ["is_control"]),
        # per event_type (single / sustained) × control flag
        ("event_type",         ["event_type", "is_control"]),
    ]

    for name, group_cols in distribution_configs:
        # Skip if any required column is missing (safety guard)
        if not all(c in df.columns for c in group_cols):
            print(f"  ⚠️  Skipping {name}: missing columns {set(group_cols) - set(df.columns)}")
            continue
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
    parser.add_argument(
        "--reduce-topics",
        type=int,
        default=REDUCE_TOPICS,
        metavar="N",
        help="After fitting, merge topics down to ~N using BERTopic's reduce_topics(). "
             "Useful for inspection: fit with small min-topic-size, then reduce. "
             "Omit to keep all discovered topics.",
    )
    args = parser.parse_args()

    df = load_corpus(args.input)
    topic_model, topics = fit_topic_model(
        df,
        min_topic_size=args.min_topic_size,
        top_n_words=args.top_n_words,
        reduce_topics=args.reduce_topics,
    )
    df = attach_topics(df, topic_model, topics)
    save_outputs(df, topic_model)

    print("\n✅  Topic modelling complete.")
    print(f"    Next step: run  python sentiment_analysis.py")


if __name__ == "__main__":
    main()
