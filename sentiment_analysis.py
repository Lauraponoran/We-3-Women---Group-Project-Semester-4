"""
sentiment_analysis.py — VADER Sentiment Analysis for Women's Protest News Corpus
==================================================================================
Runs sentiment analysis on every article in the corpus using VADER (Valence Aware
Dictionary and sEntiment Reasoner). Sentiment is scored on a -1 to +1 scale
(VADER's compound score) and saved back to the article-level parquet produced
by topic_model.py.

Research design
---------------
We run sentiment analysis on ALL articles — not just protest-relevant ones —
to characterise the overall emotional valence of the news environment during
protest weeks vs control weeks, and to compare this across ideologically
different outlets. This allows us to ask:
  - Is the overall news environment more negative during protest weeks?
  - Do right-leaning outlets have a more negative valence toward protest-adjacent
    topics even in general coverage?
  - Does sentiment toward protest-related topics differ from sentiment toward
    other topics in the same outlet during the same week?

Why VADER
----------
  - VADER is a lexicon- and rule-based sentiment analyser specifically designed
    for social media and news text
  - Its compound score maps directly to a -1 (most negative) to +1 (most positive)
    scale, making results easy to interpret and compare
  - Fully explainable: every score is driven by a transparent word-level lexicon
    (no black-box model weights)
  - Fast, runs entirely locally with no API costs
  - The pos/neu/neg component scores give additional interpretability per article

VADER compound score interpretation
-------------------------------------
  >= 0.05  → positive
  <= -0.05 → negative
  between  → neutral

Dependencies
------------
    pip install vaderSentiment pandas pyarrow tqdm

Usage
-----
    python sentiment_analysis.py                        # full run
    python sentiment_analysis.py --input path/to/articles_with_topics.parquet
    python sentiment_analysis.py --max-body-chars 1500  # truncate long articles
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ══════════════════════════════════════════════════════════════════════════════
# ❶  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_INPUT  = os.path.join("analysis_output", "articles_with_topics.parquet")
OUTPUT_DIR     = "analysis_output"
MAX_BODY_CHARS = 1500   # truncate article body for consistency with prior pipeline


# ══════════════════════════════════════════════════════════════════════════════
# ❷  SCORE ONE ARTICLE
# ══════════════════════════════════════════════════════════════════════════════

def score_article(title: str, body: str, analyzer: SentimentIntensityAnalyzer, max_body_chars: int) -> dict:
    """
    Concatenates title + truncated body, runs VADER, and returns a dict with:
      - sentiment_score     : compound score in [-1, 1]
      - sentiment_pos       : proportion of positive sentiment
      - sentiment_neu       : proportion of neutral sentiment
      - sentiment_neg       : proportion of negative sentiment
      - sentiment_label     : 'positive' | 'neutral' | 'negative'
    """
    body_snippet = body[:max_body_chars] if isinstance(body, str) else ""
    title_str    = title if isinstance(title, str) else ""
    text         = f"{title_str}. {body_snippet}".strip()

    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "sentiment_score": compound,
        "sentiment_pos":   scores["pos"],
        "sentiment_neu":   scores["neu"],
        "sentiment_neg":   scores["neg"],
        "sentiment_label": label,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ❸  RUN SENTIMENT ON FULL CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def run_sentiment(df: pd.DataFrame, max_body_chars: int) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()

    results = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Scoring", unit="art"):
        result = score_article(row.title, row.body, analyzer, max_body_chars)
        results.append(result)

    results_df = pd.DataFrame(results, index=df.index)
    df = pd.concat([df, results_df], axis=1)

    n_scored   = df["sentiment_score"].notna().sum()
    mean_sent  = df["sentiment_score"].mean()
    label_cts  = df["sentiment_label"].value_counts().to_dict()
    print(f"\n  Scored: {n_scored:,}  |  Mean compound: {mean_sent:.3f}")
    print(f"  Labels: {label_cts}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ❹  AGGREGATE TABLES
# ══════════════════════════════════════════════════════════════════════════════

def sentiment_aggregates(df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, group_cols in [
        ("publisher_event",    ["publisher", "event_label", "is_control"]),
        ("publisher",          ["publisher"]),
        ("event",              ["event_label", "is_control"]),
        ("protest_vs_control", ["is_control"]),
        ("topic",              ["topic_label"]),
        ("publisher_topic",    ["publisher", "topic_label"]),
    ]:
        agg = (df.groupby(group_cols)["sentiment_score"]
                 .agg(mean_sentiment="mean",
                      median_sentiment="median",
                      std_sentiment="std",
                      n_articles="count")
                 .reset_index()
                 .sort_values("mean_sentiment"))
        out = os.path.join(OUTPUT_DIR, f"sentiment_{name}.csv")
        agg.to_csv(out, index=False)
        print(f"  Saved sentiment_{name:<20s} → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# ❺  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          default=DEFAULT_INPUT)
    parser.add_argument("--max-body-chars", type=int, default=MAX_BODY_CHARS)
    args = parser.parse_args()

    print(f"Loading articles from {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} articles")

    print(f"\nRunning VADER sentiment analysis ...")
    df = run_sentiment(df, max_body_chars=args.max_body_chars)

    # Save full results
    out = os.path.join(OUTPUT_DIR, "articles_with_sentiment.parquet")
    df.to_parquet(out, index=False)
    df.to_csv(out.replace(".parquet", ".csv"), index=False)
    print(f"\n  Saved full results → {out}")

    print(f"\nBuilding aggregate tables ...")
    sentiment_aggregates(df)

    print("\n✅  Sentiment analysis complete.")


if __name__ == "__main__":
    main()
