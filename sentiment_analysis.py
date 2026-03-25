"""
sentiment_analysis.py — Ollama Sentiment Analysis for Women's Protest News Corpus
==================================================================================
Runs sentiment analysis on every article in the corpus using a locally hosted
Ollama model. Sentiment is scored on a -1 to +1 scale and saved back to the
article-level parquet produced by topic_model.py.

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

Why Ollama over VADER/TextBlob
--------------------------------
  - VADER and TextBlob use lexicon-based scoring that misses context, irony,
    and domain-specific language common in political news
  - Ollama uses a full LLM (e.g. llama3) that understands context and nuance
  - We can prompt it to score sentiment specifically toward the protest/event
    rather than general document sentiment, which is more meaningful for our RQ
  - Runs locally — no API costs, no data leaving the machine

Prerequisites
-------------
  1. Install Ollama: https://ollama.com
  2. Pull a model: ollama pull llama3
  3. Start the server: ollama serve   (runs on localhost:11434 by default)

Dependencies
------------
    pip install ollama pandas pyarrow tqdm

Usage
-----
    python sentiment_analysis.py                        # full run
    python sentiment_analysis.py --input path/to/articles_with_topics.parquet
    python sentiment_analysis.py --model llama3         # change model
    python sentiment_analysis.py --batch-size 50        # articles per checkpoint
    python sentiment_analysis.py --max-body-chars 1500  # truncate long articles
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import ollama
import pandas as pd
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# ❶  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_INPUT    = os.path.join("analysis_output", "articles_with_topics.parquet")
OUTPUT_DIR       = "analysis_output"
DEFAULT_MODEL    = "llama3"
BATCH_SIZE       = 50     # save progress every N articles
MAX_BODY_CHARS   = 1500   # truncate article body to keep prompts fast
CHECKPOINT_FILE  = os.path.join(OUTPUT_DIR, "sentiment_checkpoint.json")


# ══════════════════════════════════════════════════════════════════════════════
# ❷  PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a media analysis assistant. Your task is to score the
sentiment of a news article on a scale from -1.0 to +1.0, where:
  -1.0 = strongly negative (hostile, alarming, critical, condemning)
   0.0 = neutral or balanced
  +1.0 = strongly positive (supportive, celebratory, sympathetic)

You must respond with ONLY a JSON object in this exact format:
{"score": <float between -1.0 and 1.0>, "reasoning": "<one sentence>"}

Do not include any other text."""

def build_prompt(title: str, body: str) -> str:
    body_snippet = body[:MAX_BODY_CHARS] + ("..." if len(body) > MAX_BODY_CHARS else "")
    return f"Title: {title}\n\nBody: {body_snippet}"


# ══════════════════════════════════════════════════════════════════════════════
# ❸  SCORE ONE ARTICLE
# ══════════════════════════════════════════════════════════════════════════════

def score_article(title: str, body: str, model: str) -> tuple[float | None, str]:
    """Returns (score, reasoning). Score is None if parsing fails."""
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(title, body)},
            ],
        )
        text = response["message"]["content"].strip()

        # Extract JSON — handle models that wrap in ```json ... ```
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if not json_match:
            return None, "parse_error"

        parsed    = json.loads(json_match.group())
        score     = float(parsed["score"])
        reasoning = str(parsed.get("reasoning", ""))

        # Clamp to [-1, 1]
        score = max(-1.0, min(1.0, score))
        return score, reasoning

    except Exception as e:
        return None, f"error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# ❹  LOAD / SAVE CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(results: dict) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(results, f)


# ══════════════════════════════════════════════════════════════════════════════
# ❺  RUN SENTIMENT ON FULL CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def run_sentiment(df: pd.DataFrame, model: str) -> pd.DataFrame:
    # Load checkpoint: url → {score, reasoning}
    checkpoint = load_checkpoint()
    print(f"  Checkpoint: {len(checkpoint):,} articles already scored")

    scores     = {}
    reasonings = {}

    # Pre-populate from checkpoint
    for url, data in checkpoint.items():
        scores[url]     = data["score"]
        reasonings[url] = data["reasoning"]

    # Score remaining articles
    todo = df[~df["url"].isin(checkpoint)].copy()
    print(f"  Articles to score: {len(todo):,}")

    batch_buffer = {}

    pbar = tqdm(todo.itertuples(), total=len(todo), desc="Scoring", unit="art")
    for i, row in enumerate(pbar):
        score, reasoning = score_article(row.title, row.body, model)
        scores[row.url]     = score
        reasonings[row.url] = reasoning
        batch_buffer[row.url] = {"score": score, "reasoning": reasoning}

        if (i + 1) % BATCH_SIZE == 0:
            checkpoint.update(batch_buffer)
            save_checkpoint(checkpoint)
            batch_buffer = {}
            pbar.set_postfix({"saved": len(checkpoint)})

    # Save final batch
    if batch_buffer:
        checkpoint.update(batch_buffer)
        save_checkpoint(checkpoint)

    pbar.close()

    df = df.copy()
    df["sentiment_score"]     = df["url"].map(scores)
    df["sentiment_reasoning"] = df["url"].map(reasonings)

    n_scored  = df["sentiment_score"].notna().sum()
    n_failed  = df["sentiment_score"].isna().sum()
    mean_sent = df["sentiment_score"].mean()
    print(f"\n  Scored: {n_scored:,}  |  Failed: {n_failed:,}  |  Mean sentiment: {mean_sent:.3f}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ❻  AGGREGATE TABLES
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
# ❼  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          default=DEFAULT_INPUT)
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    parser.add_argument("--max-body-chars", type=int, default=MAX_BODY_CHARS)
    args = parser.parse_args()

    global BATCH_SIZE, MAX_BODY_CHARS
    BATCH_SIZE     = args.batch_size
    MAX_BODY_CHARS = args.max_body_chars

    print(f"Loading articles from {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} articles")

    # Check Ollama is running
    try:
        ollama.list()
        print(f"  Ollama running ✅  (model: {args.model})")
    except Exception:
        print("❌  Ollama is not running. Start it with:  ollama serve")
        print("    Then pull a model with:               ollama pull llama3")
        return

    print(f"\nRunning sentiment analysis ...")
    df = run_sentiment(df, args.model)

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
