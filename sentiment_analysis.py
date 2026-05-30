"""
sentiment_analysis.py — VADER Sentiment Analysis for Women's Protest News Corpus
==================================================================================
Runs sentiment analysis on every article in the corpus using VADER (Valence Aware
Dictionary and sEntiment Reasoner). Sentiment is scored on a -1 to +1 scale
(VADER's compound score) and saved back to the article-level parquet produced
by collect_articles_optimized.py / build_corpus.py.

The translated text columns (translated_title, translated_body) saved here
are used as input to topic_model.py so that topic clustering operates on
English text rather than mixed-language text.

Non-English articles (DE / FR / ES) are translated to English before scoring
using Helsinki-NLP's opus-mt models. Translation preserves emotional valence
well enough for sentiment purposes while keeping the full explainability of the
VADER lexicon. A `language` column records the detected language of each article
so that translated vs. native-English articles can be reported separately.

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

Why translate rather than use a multilingual model
---------------------------------------------------
  - Keeps the full VADER interpretability story: you can inspect any article's
    translation and see exactly which words drove the score
  - Translate-then-VADER is an established pattern in computational social
    science; reviewers expect it and it is easy to document
  - Helsinki opus-mt handles EN/DE/FR/ES well; sentiment-bearing vocabulary
    (condemn, celebrate, outrage, violence) translates reliably
  - Limitation: protest-specific slang or movement vocabulary may not translate
    idiomatically — documented in `translation_note` column per article

Language detection
------------------
  Uses `lingua-language-detector` (Python package: `lingua`), which is more
  accurate than `langdetect` on short news snippets and handles code-switching
  better. Detects EN / DE / FR / ES; anything else is flagged as "other" and
  scored on the raw (untranslated) text with a warning.

Translation models
------------------
  Helsinki-NLP/opus-mt-{src}-en loaded on demand per source language:
    - opus-mt-de-en  (German → English)
    - opus-mt-fr-en  (French → English)
    - opus-mt-es-en  (Spanish → English)
  Models are cached by HuggingFace after first download (~300 MB total).
  Translation is applied to title + first MAX_BODY_CHARS chars of body.

VADER compound score interpretation
-------------------------------------
  >= 0.05  → positive
  <= -0.05 → negative
  between  → neutral

Dependencies
------------
    pip install vaderSentiment pandas pyarrow tqdm lingua-language-detector transformers sentencepiece torch

Usage
-----
    python sentiment_analysis.py                        # full run
    python sentiment_analysis.py --input path/to/articles_with_topics.parquet
    python sentiment_analysis.py --max-body-chars 1500  # truncate long articles (default)
    python sentiment_analysis.py --no-translate         # skip translation, score raw text (EN only)
    python sentiment_analysis.py --device cuda          # use GPU for translation (default: cpu)
"""

from __future__ import annotations

import argparse
import os
import warnings
from functools import lru_cache

import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Language detection ────────────────────────────────────────────────────────
from lingua import Language, LanguageDetectorBuilder

# ── Translation ───────────────────────────────────────────────────────────────
from transformers import MarianMTModel, MarianTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# ❶  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_INPUT  = os.path.join("news_output", "corpus_all.parquet")
OUTPUT_DIR     = "analysis_output"
MAX_BODY_CHARS = 1500   # truncate article body before translation + scoring

# Languages we can translate; anything else scored on raw text with a note
SUPPORTED_LANGS = {"EN", "DE", "FR", "ES"}

# Helsinki opus-mt model names per source language
OPUS_MT_MODELS: dict[str, str] = {
    "DE": "Helsinki-NLP/opus-mt-de-en",
    "FR": "Helsinki-NLP/opus-mt-fr-en",
    "ES": "Helsinki-NLP/opus-mt-es-en",
}


# ══════════════════════════════════════════════════════════════════════════════
# ❷  LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def build_language_detector() -> object:
    """
    Build a Lingua detector restricted to EN / DE / FR / ES.
    Restricting the candidate set improves accuracy on short snippets
    versus detecting from all ~75 supported languages.
    """
    return (
        LanguageDetectorBuilder
        .from_languages(Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.SPANISH)
        .with_preloaded_language_models()
        .build()
    )


def detect_language(text: str, detector) -> str:
    """Return ISO 639-1 upper-case code: EN / DE / FR / ES, or 'OTHER'."""
    if not isinstance(text, str) or len(text.strip()) < 20:
        return "EN"   # too short to detect reliably; assume English
    result = detector.detect_language_of(text)
    if result is None:
        return "OTHER"
    # lingua Language enum names are full language names e.g. Language.GERMAN
    mapping = {
        Language.ENGLISH: "EN",
        Language.GERMAN:  "DE",
        Language.FRENCH:  "FR",
        Language.SPANISH: "ES",
    }
    return mapping.get(result, "OTHER")


# ══════════════════════════════════════════════════════════════════════════════
# ❸  TRANSLATION
# ══════════════════════════════════════════════════════════════════════════════

class TranslatorCache:
    """
    Loads Helsinki opus-mt models on demand and caches them in memory.
    Only the models actually needed for the corpus languages are loaded.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._models: dict[str, tuple[MarianTokenizer, MarianMTModel]] = {}

    def _load(self, lang: str) -> tuple[MarianTokenizer, MarianMTModel]:
        if lang not in self._models:
            model_name = OPUS_MT_MODELS[lang]
            print(f"  Loading translation model: {model_name} ...")
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model     = MarianMTModel.from_pretrained(model_name).to(self.device)
            model.eval()
            self._models[lang] = (tokenizer, model)
        return self._models[lang]

    def translate(self, text: str, src_lang: str) -> tuple[str, str]:
        """
        Translate `text` from `src_lang` to English.

        Returns (translated_text, note) where note is empty string on success
        or a short warning message if translation failed.
        """
        if src_lang not in OPUS_MT_MODELS:
            return text, f"no opus-mt model for lang={src_lang}; scored on raw text"

        try:
            tokenizer, model = self._load(src_lang)
            # MarianMT expects a list of strings
            inputs = tokenizer([text], return_tensors="pt",
                               padding=True, truncation=True,
                               max_length=512).to(self.device)
            import torch
            with torch.no_grad():
                translated_ids = model.generate(**inputs)
            translated = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            return translated, ""
        except Exception as exc:
            warnings.warn(f"Translation failed for lang={src_lang}: {exc}")
            return text, f"translation failed: {exc}; scored on raw text"


# ══════════════════════════════════════════════════════════════════════════════
# ❹  SCORE ONE ARTICLE
# ══════════════════════════════════════════════════════════════════════════════

def score_article(
    title: str,
    body: str,
    analyzer: SentimentIntensityAnalyzer,
    max_body_chars: int,
) -> dict:
    """
    Concatenates title + truncated body, runs VADER, returns score dict.

    Input text must already be in English (translate before calling if needed).

    Returns keys:
      sentiment_score   : compound score in [-1, 1]
      sentiment_pos     : proportion of positive sentiment
      sentiment_neu     : proportion of neutral sentiment
      sentiment_neg     : proportion of negative sentiment
      sentiment_label   : 'positive' | 'neutral' | 'negative'
    """
    body_snippet = body[:max_body_chars] if isinstance(body, str) else ""
    title_str    = title if isinstance(title, str) else ""
    text         = f"{title_str}. {body_snippet}".strip()

    scores   = analyzer.polarity_scores(text)
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
# ❺  RUN SENTIMENT ON FULL CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def run_sentiment(
    df: pd.DataFrame,
    max_body_chars: int,
    translate: bool,
    device: str,
) -> pd.DataFrame:
    analyzer   = SentimentIntensityAnalyzer()
    detector   = build_language_detector()
    translator = TranslatorCache(device=device) if translate else None

    records = []

    for row in tqdm(df.itertuples(), total=len(df), desc="Scoring", unit="art"):
        title = row.title if isinstance(row.title, str) else ""
        body  = row.body  if isinstance(row.body,  str) else ""

        # ── Language detection ────────────────────────────────────────────────
        # Detect on title + first 500 chars of body for speed; enough signal
        detect_text = f"{title} {body[:500]}"
        lang        = detect_language(detect_text, detector)

        # ── Translation if needed ─────────────────────────────────────────────
        translation_note = ""
        if translate and lang not in ("EN", "OTHER"):
            title_en, note_t = translator.translate(title, lang)
            body_en,  note_b = translator.translate(body[:max_body_chars], lang)
            translation_note = " | ".join(filter(None, [note_t, note_b]))
        else:
            title_en = title
            body_en  = body
            if lang == "OTHER" and translate:
                translation_note = "unsupported language; scored on raw text"

        # ── VADER scoring ─────────────────────────────────────────────────────
        sentiment = score_article(title_en, body_en, analyzer, max_body_chars)

        records.append({
            "language":          lang,
            "was_translated":    translate and lang not in ("EN", "OTHER"),
            "translation_note":  translation_note,
            # Translated text columns — consumed by topic_model.py so that
            # BERTopic clusters on English text rather than mixed-language text.
            # For English articles these are identical to the original columns.
            "translated_title":  title_en,
            "translated_body":   body_en,
            **sentiment,
        })

    results_df = pd.DataFrame(records, index=df.index)
    df         = pd.concat([df, results_df], axis=1)

    # ── Summary stats ─────────────────────────────────────────────────────────
    n_scored    = df["sentiment_score"].notna().sum()
    mean_sent   = df["sentiment_score"].mean()
    label_cts   = df["sentiment_label"].value_counts().to_dict()
    lang_cts    = df["language"].value_counts().to_dict()
    n_translated = df["was_translated"].sum() if "was_translated" in df.columns else 0

    print(f"\n  Scored:      {n_scored:,}")
    print(f"  Translated:  {n_translated:,} articles")
    print(f"  Languages:   {lang_cts}")
    print(f"  Mean compound score: {mean_sent:.3f}")
    print(f"  Labels:      {label_cts}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ❻  AGGREGATE TABLES
# ══════════════════════════════════════════════════════════════════════════════

def sentiment_aggregates(df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = [
        ("publisher_event",    ["publisher", "event_label", "is_control"]),
        ("publisher",          ["publisher"]),
        ("event",              ["event_label", "is_control"]),
        ("protest_vs_control", ["is_control"]),
        ("topic",              ["topic_label"]),
        ("publisher_topic",    ["publisher", "topic_label"]),
        # Language-stratified: lets you verify translated ≈ native-EN scores
        ("language",           ["language"]),
        ("publisher_language", ["publisher", "language"]),
    ]

    for name, group_cols in configs:
        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            print(f"  ⚠️  Skipping sentiment_{name}: missing columns {missing}")
            continue

        agg = (
            df.groupby(group_cols)["sentiment_score"]
            .agg(
                mean_sentiment="mean",
                median_sentiment="median",
                std_sentiment="std",
                n_articles="count",
            )
            .reset_index()
            .sort_values("mean_sentiment")
        )
        out = os.path.join(OUTPUT_DIR, f"sentiment_{name}.csv")
        agg.to_csv(out, index=False)
        print(f"  Saved sentiment_{name:<26s} → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# ❼  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Path to corpus parquet (default: corpus_all.parquet).")
    parser.add_argument("--max-body-chars", type=int, default=MAX_BODY_CHARS)
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Skip translation and score all articles on raw text. "
             "Non-English articles will score near-neutral; use only if corpus "
             "is overwhelmingly English.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for MarianMT translation models: 'cpu' or 'cuda' (default: cpu).",
    )
    args = parser.parse_args()

    print(f"Loading articles from {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} articles")

    if args.no_translate:
        print("\n⚠️  Translation disabled — non-English articles will score near-neutral.")

    print(f"\nRunning VADER sentiment analysis (translate={'yes' if not args.no_translate else 'no'}) ...")
    df = run_sentiment(
        df,
        max_body_chars=args.max_body_chars,
        translate=not args.no_translate,
        device=args.device,
    )

    # ── Save full results ──────────────────────────────────────────────────────
    out_parquet = os.path.join(OUTPUT_DIR, "articles_with_sentiment.parquet")
    out_csv     = out_parquet.replace(".parquet", ".csv")
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved full results → {out_parquet}")
    print(f"  Saved full results → {out_csv}")

    # ── Aggregate tables ───────────────────────────────────────────────────────
    print(f"\nBuilding aggregate tables ...")
    sentiment_aggregates(df)

    print("\n✅  Sentiment analysis complete.")
    print(f"    Outputs → {OUTPUT_DIR}/")
    print(f"    Next step: run  python topic_model.py")


if __name__ == "__main__":
    main()
