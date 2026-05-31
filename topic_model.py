"""
topic_model.py — BERTopic Topic Modelling for Women's Protest News Corpus
==========================================================================
Fits a BERTopic model on the full corpus, assigns a topic to every article,
and produces per-publisher and per-event topic distribution tables.

Pipeline order
--------------
topic_model.py now runs AFTER sentiment_analysis.py, not before it.

  sentiment_analysis.py  →  articles_with_sentiment.parquet
                                         ↓
                             topic_model.py  (default input)
                                         ↓
                             articles_with_topics.parquet

Reason: sentiment_analysis.py detects the language of each article and
translates DE/FR/ES bodies to English via Helsinki opus-mt before scoring.
We reuse that translated text for topic modelling so that BERTopic clusters
articles by THEME rather than by LANGUAGE. Without this step, the multilingual
embedding model still produces language-homogeneous clusters because the
embedding similarity between two English articles about abortion is higher
than between an English and a German article about the same topic — even with
a multilingual model.

The translated text is stored in the `translated_title` and `translated_body`
columns added by sentiment_analysis.py. `load_corpus()` builds the `text`
column from these when available, falling back to the original title/body for
articles that were already English.

Two-pass topic modelling
-------------------------
❶  GLOBAL model  — fit on the full corpus to discover overarching themes that
   span all events and outlets. topic_id / topic_label columns are used for
   cross-event comparisons in visualise.py.

❷  PER-EVENT models — a separate BERTopic is fit on each event's articles to
   discover fine-grained topics specific to that event's news cycle. Results
   are saved to analysis_output/event_topics/ and attached to the dataframe
   as event_topic_id / event_topic_label.

Why BERTopic over KMeans/Ward/DBSCAN
--------------------------------------
  - KMeans/Ward require specifying k in advance; topic count is not known
  - DBSCAN is sensitive to hyperparameters and struggles with high-dimensional text
  - BERTopic uses sentence embeddings + HDBSCAN to find k automatically,
    and produces human-readable topic labels via TF-IDF keyphrases
  - Returns per-document topic assignments suitable for outlet-level comparison

Outlier reduction (two-stage)
-------------------------------
  Stage 1 — lower min_topic_size (3 global, 2 per-event) so HDBSCAN creates
  more and smaller initial clusters.

  Stage 2 — reduce_outliers(strategy="embeddings") reassigns remaining outlier
  documents to their nearest topic by raw embedding distance. Works better than
  "c-tf-idf" for short articles. Original assignments preserved in topic_id_raw.

Topic reduction (default 30)
------------------------------
  min_topic_size=3 on ~7,500 articles discovers hundreds of micro-topics that
  all fall into "All Other Topics" in visualisations. REDUCE_TOPICS=30 merges
  them into ~30 coherent macro-themes via BERTopic's hierarchical reduction.
  Per-event models skip reduction — their small subsets benefit from granularity.
  Pass --reduce-topics 0 to disable.

Stopword lists + max_df
------------------------
  Stopword lists cover all function word forms in EN/DE/FR/ES, plus a dedicated
  news-domain boilerplate list (NEWS_STOPWORDS) covering wire-service names,
  journalistic hedges, generic protest vocabulary, and date/number words that
  would otherwise dominate every topic label.

  All stopword lists are merged into a deduplicated list via set() to remove
  the ~30 duplicates that existed across the sublists.

  max_df=0.85 (tightened from 0.95) excludes terms appearing in >85% of
  documents, catching high-frequency words that evade the stopword list because
  of spelling variants, casing, or tokenisation.

  Note: since we now cluster on English-translated text, the German/French/
  Spanish stopword lists are retained as a safety net for any untranslated
  fragments but are no longer the primary defence.

Label quality improvements
--------------------------
  Two additional components are injected into the BERTopic pipeline to produce
  cleaner, more distinctive topic labels:

  ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)
    Down-weights terms that are frequent *across topics* (not just across
    documents). This is BERTopic's built-in solution to stopword leakage in
    topic labels and has the highest single impact on label quality.

  KeyBERTInspired + MaximalMarginalRelevance representation models
    Re-ranks candidate label words by their embedding similarity to the topic
    centroid, so generic high-frequency words (which are not semantically
    central to any one cluster) score low. MMR additionally penalises near-
    duplicate label words for richer, more diverse labels.

Dependencies
------------
    pip install bertopic sentence-transformers pandas pyarrow

Usage
-----
    # Standard: run sentiment_analysis.py first, then:
    python topic_model.py

    # Explicit input:
    python topic_model.py --input analysis_output/articles_with_sentiment.parquet

    # Raw corpus (skips translation benefit — not recommended):
    python topic_model.py --input news_output/corpus_all.parquet

    python topic_model.py --reduce-topics 20   # fewer, broader topics
    python topic_model.py --reduce-topics 50   # more granular
    python topic_model.py --reduce-topics 0    # disable reduction
    python topic_model.py --no-event-topics    # global model only
    python topic_model.py --no-reduce-outliers # skip outlier reassignment
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from bertopic import BERTopic
import re

from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


# ──────────────────────────────────────────────────────────────────────────────
# Label normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────

_OTHER_RAW = {"other", "outlier", "unknown", "-1", ""}


def _canonical_label(name: str) -> str:
    """Normalise a BERTopic raw Name into a canonical, deduplicated label.

    Two changes vs raw BERTopic output:
      1. Catch-all labels (outlier, other, unknown) → "outlier" so they all
         merge into the single remainder bucket in visualise.py.
      2. The three keyword words are *sorted alphabetically* so that
         "0_warhol_monroe_marilyn" and "4_marilyn_monroe_warhol" produce the
         same "Marilyn, Monroe, Warhol" label and are counted together.
    """
    if not isinstance(name, str) or name.strip().lower() in _OTHER_RAW:
        return "outlier"
    parts = name.split("_")
    start_idx = 1 if parts[0].replace("-", "").isdigit() else 0
    words = [w for w in parts[start_idx: start_idx + 3] if w]
    if not words:
        return "outlier"
    return ", ".join(sorted(w.capitalize() for w in words))

# ──────────────────────────────────────────────────────────────────────────────
# Multilingual stopwords (EN / DE / FR / ES) + news-domain boilerplate
#
# Primary defence against function-word topic labels. Since input text is now
# English-translated, the EN list does most of the work. DE/FR/ES lists act as
# a safety net for untranslated fragments and proper nouns that survive
# translation.
#
# NEWS_STOPWORDS covers wire-service boilerplate, journalistic hedges, generic
# protest/event vocabulary, and date/number words that would otherwise dominate
# every topic label regardless of which event or publisher is being modelled.
#
# All lists are merged and deduplicated via set(), then cast to list because
# sklearn CountVectorizer requires stop_words to be a list (not a frozenset).
#
# Second layer of defence: max_df=0.85 in the CountVectorizer excludes any
# term appearing in >85% of documents regardless of whether it appears here.
#
# Third layer:  ClassTfidfTransformer(reduce_frequent_words=True) down-weights
# terms frequent *across topics* — the most effective single fix for stopword
# leakage in BERTopic labels.
# ──────────────────────────────────────────────────────────────────────────────

# ── English ───────────────────────────────────────────────────────────────────
_EN = [
    "a","about","above","after","again","against","all","also","am","an","and",
    "any","are","as","at","be","because","been","before","being","below",
    "between","both","but","by","can","could","did","do","does","doing","down",
    "during","each","few","for","from","further","get","got","had","has","have",
    "having","he","her","here","him","himself","his","how","i","if","in","into",
    "is","it","its","itself","just","me","more","most","my","myself","new","no",
    "nor","not","now","of","off","on","once","only","or","other","our","out",
    "over","own","re","s","said","same","she","should","so","some","such",
    "t","than","that","the","their","them","then","there","these","they",
    "this","those","through","time","to","too","under","until","up","us",
    "very","was","we","were","what","when","where","which","while","who",
    "whom","why","will","with","would","you","your","yourself","after","also",
    "back","even","first","into","just","may","much","new","now","one","only",
    "other","still","two","use","way","well","within","without","year","years",
    "said","says","say","told","tell","going","went","come","came","take",
    "made","make","know","good","people","man","men","woman","women","day",
    "week","month","report","news","according","told","like","want","see",
    "think","need","work","part","place","point","right","left","large",
    "long","high","since","ago","around","another","government","country",
    "city","world","state","mr","ms","mrs","dr","per","cent",
]

# ── German (safety net for untranslated fragments) ────────────────────────────
_DE = [
    "ab","aber","alle","allem","allen","aller","alles","als","also","am","an",
    "andere","anderen","anderer","anderes","auch","auf","aus","bei","beim",
    "da","damit","dann","das","dass","dem","den","denn","der","deren","des",
    "deshalb","die","dies","diese","diesem","diesen","dieser","dieses","doch",
    "du","durch","ein","eine","einem","einen","einer","eines","er","es","etwa",
    "etwas","für","gegen","hatte","haben","hat","ich","ihm","ihn","ihnen",
    "ihr","ihre","ihrem","ihren","ihrer","ihres","im","in","ins","ist","ja",
    "jede","jedem","jeden","jeder","jedes","kann","kein","keine","keinem",
    "keinen","keiner","keines","man","mehr","mein","meine","mit","muss","nach",
    "nicht","nichts","noch","nun","nur","ob","oder","ohne","schon","sehr",
    "sein","seine","seinem","seinen","seiner","seines","sich","sie","sind",
    "so","soll","sondern","über","um","und","uns","unter","vom","von","vor",
    "war","was","weil","wie","wieder","wir","wird","zu","zum","zur","zwischen",
    "wurde","wurden","worden","hatte","hatten","wäre","müssen","sollen",
]

# ── French (safety net) ───────────────────────────────────────────────────────
_FR = [
    "au","aux","avait","avec","avoir","c","ca","ce","cela","ces","cet","cette",
    "ceux","comme","dans","de","des","dont","du","elle","elles","en","entre",
    "et","eux","il","ils","je","l","la","le","les","leur","leurs","lui","ma",
    "mais","me","meme","mes","mon","ni","non","nous","on","ou","par","pas",
    "pour","qu","que","qui","quoi","sa","se","si","ses","son","sur","te","tes",
    "toi","ton","tout","toute","toutes","tous","tu","un","une","vous","y",
    "ete","etre","fait","faire","apres","avant","aussi","alors","ainsi","bien",
    "bon","bonne","car","donc","lors","meme","depuis","selon","quand","sous",
    "vers","notre","nos","chez","puis","donc","ceci","cela","celle","celui",
    "peu","plusieurs","souvent","toujours","tres","trop","ici","la","peut",
    "sera","serait","seront","sommes","suis","etait","etaient","etant",
]

# ── Spanish (safety net) ──────────────────────────────────────────────────────
_ES = [
    "al","algo","alguien","alguna","algunas","algunos","ante","antes","aquel",
    "aquella","aquellas","aquellos","aqui","asi","aunque","bajo","bien","cada",
    "como","con","contra","cual","cuales","cuando","cuanto","cuya","cuyas",
    "cuyo","cuyos","de","del","desde","donde","el","ella","ellas","ello",
    "ellos","en","entre","era","eran","eres","es","esa","esas","ese","eso",
    "esos","esta","estas","este","esto","estos","estoy","fue","fueron","gran",
    "ha","habia","habian","han","hasta","hay","he","hemos","hubo","la","las",
    "le","les","lo","los","me","mi","mis","mismo","misma","mas","mucho",
    "muchos","muy","nada","ni","no","nos","nosotras","nosotros","nuestra",
    "nuestras","nuestro","nuestros","o","otra","otras","otro","otros","para",
    "pero","poco","por","porque","que","quien","quienes","se","ser","si",
    "sin","sobre","su","sus","tambien","tan","tanto","te","tiene","tienen",
    "toda","todas","todo","todos","tu","tus","un","una","unas","uno","unos",
    "y","ya","yo","esta","estan","el","segun","tras","durante","mediante",
]

# ── News-domain boilerplate ───────────────────────────────────────────────────
# Words that are generic across ALL articles in a protest news corpus and
# therefore never discriminate between topics. Without this list they dominate
# TF-IDF labels regardless of the other defences.
_NEWS = [
    # wire-service / outlet names
    "reuters","associated","press","ap","afp","bbc","cnn","nyt","times",
    # journalistic attribution hedges
    "according","report","reports","reported","reporting",
    "said","say","says","saying","told","tells","tell",
    "added","noted","stated","confirmed","announced","described",
    "claimed","argued","suggested","indicated","explained",
    # generic protest / event vocabulary that spans every topic
    "protest","protests","protester","protesters","protesting","protested",
    "demonstration","demonstrations","demonstrator","demonstrators",
    "march","marches","marching","marched",
    "rally","rallies","rallied",
    "activist","activists","activism",
    "rights","movement","movements","campaign","campaigns",
    "crowd","crowds","group","groups","people","person",
    # calendar words (appear in datelines, never topic-discriminating)
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
    "january","february","april","june","july","august",
    "september","october","november","december",
    # generic numeric / quantifier words
    "hundred","hundreds","thousand","thousands","million","millions",
    "billion","number","numbers","many","several","few","more","less",
    # other near-universal journalism boilerplate
    "new","latest","update","updates","breaking","live",
    "official","officials","spokesperson","statement",
    "amid","following","ahead","despite","after","before",
]

# Merge all lists, deduplicate, and freeze for O(1) lookup
STOPWORDS: list[str] = list(set(_EN + _DE + _FR + _ES + _NEWS))


# ══════════════════════════════════════════════════════════════════════════════
# ❶  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Default input is now the sentiment output which contains translated text
DEFAULT_INPUT        = os.path.join("analysis_output", "articles_with_sentiment.parquet")
OUTPUT_DIR           = "analysis_output"
EVENT_TOPICS_DIR     = os.path.join(OUTPUT_DIR, "event_topics")

# English-only model — safe to use now that all text is translated to English.
# paraphrase-MiniLM-L6-v2 is fast and produces excellent topic clusters on
# English news text. If you want to keep the multilingual model for any reason,
# change this back to "intfloat/multilingual-e5-large" and restore _prefix_docs.
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"

MIN_TOPIC_SIZE       = 3
TOP_N_WORDS          = 10
REDUCE_TOPICS        = 30

# Starting min_topic_size for per-event models. If a model fails (e.g. sparse
# pre-2019 events with few articles), fit_event_models() retries with
# progressively halved values down to a floor of 2, and flags the event as
# sparse in event_topics/event_model_notes.csv.
EVENT_MIN_TOPIC_SIZE = 15


# ══════════════════════════════════════════════════════════════════════════════
# ❷  LOAD CORPUS
# ══════════════════════════════════════════════════════════════════════════════

# Compiled once at import time for performance across millions of characters.
_RE_STYLE_BLOCK  = re.compile(r"<(style|script)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_RE_HTML_TAG     = re.compile(r"<[^>]{0,500}>")
_RE_CSS_PROPERTY = re.compile(r"\b[\w-]+\s*:\s*[^;{}\n]{1,80};")
_RE_WHITESPACE   = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Remove HTML/CSS artifacts from article text.

    Some scraped articles contain raw HTML or inline CSS that produces
    nonsense topic labels such as "Css, Inlinelink, Yidnqd".  This function
    strips those artifacts before text reaches BERTopic.

    Steps (in order):
      1. Remove <style> and <script> blocks entirely (content + tags).
      2. Remove all remaining HTML tags.
      3. Remove CSS property declarations (e.g. "color: red;").
      4. Collapse whitespace.
    """
    text = _RE_STYLE_BLOCK.sub(" ", text)
    text = _RE_HTML_TAG.sub(" ", text)
    text = _RE_CSS_PROPERTY.sub(" ", text)
    return _RE_WHITESPACE.sub(" ", text).strip()


def load_corpus(path: str) -> pd.DataFrame:
    """Load corpus from parquet or CSV.

    Builds a `text` column for BERTopic from translated text when available
    (columns added by sentiment_analysis.py), falling back to original title
    and body for articles that were already English.

    Column priority for text construction:
      title : translated_title  > title
      body  : translated_body   > body
    """
    print(f"Loading corpus from {path} ...")
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    required = {"publisher", "title", "body", "event_label", "event_type", "is_control"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Corpus is missing expected columns: {missing}")

    df = df[df["body"].fillna("").str.strip().str.len() > 100].copy()
    df["is_control"] = df["is_control"].astype(bool)
    df["publisher"] = (
        df["publisher"].astype(str)
        .str.extract(r"\.([A-Za-z0-9]+)")[0]
        .fillna(df["publisher"].astype(str))
    )

    # Use translated text if available, otherwise fall back to original.
    # Topic modelling runs on TITLES ONLY — bodies contain too much boilerplate,
    # HTML noise, and repeated wire-service language that pollutes topic labels.
    # Titles are concise human-written summaries and produce much cleaner topics.
    # Change `title_col` to `title_col + " " + body_col` to revert to full text.
    has_translated = "translated_title" in df.columns
    if has_translated:
        title_col = df["translated_title"].fillna(df["title"].fillna(""))
        print("  Using translated_title for topic modelling ✓")
    else:
        title_col = df["title"].fillna("")
        print("  ⚠️  No translated_title column found — clustering on raw titles.")
        print("       Run sentiment_analysis.py first for best results.")

    df["text"] = title_col.map(_strip_html)

    print(f"  {len(df):,} articles loaded "
          f"({df['publisher'].nunique()} publishers, "
          f"{df['event_label'].nunique()} event windows)")
    if "language" in df.columns:
        print(f"  Language mix: {df['language'].value_counts().to_dict()}")

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
    reduce_outliers: bool = True,
    label: str = "",
) -> tuple[BERTopic, list[int]]:
    """Fit BERTopic on docs; return (model, topic_assignments).

    CountVectorizer settings:
      stop_words=STOPWORDS  — list; removes function/boilerplate words
                              from TF-IDF label candidates (O(1) lookup)
      max_df=0.85           — removes any term in >85% of docs; catches
                              high-frequency words that evade the stopword
                              list due to casing or tokenisation variants
      min_df=2              — requires a term in at least 2 docs; prevents
                              hapax legomena dominating labels
      ngram_range=(1,2)     — allows bigrams for richer topic labels

    ClassTfidfTransformer settings:
      reduce_frequent_words=True  — down-weights terms frequent *across*
                                    topics, not just across documents; the
                                    single highest-impact fix for stopword
                                    leakage in BERTopic labels
      bm25_weighting=True         — BM25 variant of c-TF-IDF; further
                                    penalises very common terms

    Representation models (applied after c-TF-IDF for final label ranking):
      KeyBERTInspired             — re-ranks candidates by embedding
                                    similarity to the topic centroid; generic
                                    high-frequency words score low because
                                    they are not semantically central to any
                                    single cluster
      MaximalMarginalRelevance    — penalises near-duplicate label words so
                                    the final top_n_words are diverse
    """
    vectorizer_model = CountVectorizer(
        stop_words=STOPWORDS,
        ngram_range=(1, 2),
        min_df=2,
        # Titles are shorter than bodies so a term appearing in >60% of title-
        # documents is almost certainly generic boilerplate, not a topic signal.
        # This is tighter than the 0.85 used for full-body text.
        max_df=0.60,
        # Require >=3 alphabetic chars. Silently drops 1-2 letter function
        # words ("de","la","du","le","il","au","zu","un","al" …) that are
        # nearly impossible to enumerate exhaustively across four languages.
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        # Normalise accents BEFORE stop_words matching so "uber" in the list
        # matches "über" in text. Without this accented tokens always leak
        # through as topic labels ("Die","Fur","Uber","Etre" …).
        strip_accents="unicode",
    )

    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True,
        bm25_weighting=True,
    )

    # KeyBERTInspired removed: on a protest corpus its centroid-similarity
    # re-ranking resurfaces high-frequency function words that appear in
    # every topic centroid, undoing CountVectorizer's filtering.
    # MMR alone is sufficient to diversify final label words.
    representation_model = MaximalMarginalRelevance(diversity=0.4)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
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
          f"Outliers before reassignment: {n_outliers} "
          f"({100 * n_outliers / max(len(topics), 1):.1f}%)")

    # ── Outlier reassignment ──────────────────────────────────────────────────
    if reduce_outliers and n_outliers > 0:
        topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings")
        topic_model.update_topics(docs, topics=topics)
        n_remaining = sum(1 for t in topics if t == -1)
        print(f"{prefix}Outliers after reassignment: {n_remaining} "
              f"({100 * n_remaining / max(len(topics), 1):.1f}%)")

    # ── Topic count reduction ─────────────────────────────────────────────────
    if reduce_topics and reduce_topics > 0:
        n_before = len(set(topics)) - (1 if -1 in topics else 0)
        if n_before > reduce_topics:
            print(f"{prefix}Reducing {n_before} topics → ~{reduce_topics} ...")
            topic_model.reduce_topics(docs, nr_topics=reduce_topics)
            topics  = topic_model.topics_
            n_after = len(set(topics)) - (1 if -1 in topics else 0)
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
    reduce_outliers: bool,
) -> tuple[BERTopic, list[int]]:
    print(f"\nFitting GLOBAL BERTopic model on {len(df):,} articles ...")
    print(f"  min_topic_size={min_topic_size}, top_n_words={top_n_words}, "
          f"reduce_outliers={reduce_outliers}, reduce_topics={reduce_topics}")

    return build_topic_model(
        docs=df["text"].tolist(),
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        reduce_topics=reduce_topics,
        reduce_outliers=reduce_outliers,
        label="global",
    )


def attach_global_topics(
    df: pd.DataFrame,
    topic_model: BERTopic,
    topics: list[int],
) -> pd.DataFrame:
    df = df.copy()
    df["topic_id_raw"] = topics   # pre-reassignment for auditability
    df["topic_id"]     = topics
    label_map = dict(zip(
        topic_model.get_topic_info()["Topic"],
        topic_model.get_topic_info()["Name"].map(_canonical_label),
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
    reduce_outliers: bool,
) -> pd.DataFrame:
    """Fit a separate BERTopic per event_label.

    Adaptive min_topic_size: if the requested size causes a CountVectorizer
    conflict (max_df < min_df — typical for small/sparse events), the model
    is retried with progressively halved min_topic_size values down to a floor
    of 2.  The size actually used is recorded in event_topic_min_size so
    visualise.py can add a footnote flagging sparse events.

    Per-event models intentionally do NOT apply reduce_topics — the subsets
    are small enough that micro-topic granularity is informative.
    """
    os.makedirs(EVENT_TOPICS_DIR, exist_ok=True)

    events = sorted(df["event_label"].unique())
    print(f"\nFitting PER-EVENT BERTopic models for {len(events)} events ...")
    print(f"  event_min_topic_size={event_min_topic_size}, "
          f"reduce_outliers={reduce_outliers}")

    df = df.copy()
    df["event_topic_id"]       = -1
    df["event_topic_label"]    = "outlier"
    df["event_topic_min_size"] = event_min_topic_size  # default; overwritten below

    # Mapping written to event_topics/event_model_notes.csv for visualise.py
    model_notes: dict[str, dict] = {}

    all_event_dists = []

    for event in events:
        mask   = df["event_label"] == event
        subset = df[mask].copy()

        if len(subset) < 4:  # absolute floor — can't cluster fewer than 4 docs
            print(f"  [{event}] Skipping — too few articles ({len(subset)})")
            model_notes[event] = {"status": "skipped", "n_articles": len(subset),
                                  "min_topic_size_used": None, "n_topics": 0}
            continue

        # ── Adaptive retry loop ───────────────────────────────────────────────
        # Start at the requested size; halve on each failure; floor at 2.
        size_to_try = event_min_topic_size
        event_model, event_topics, used_size = None, None, None
        last_error = ""
        while size_to_try >= 2:
            try:
                event_model, event_topics = build_topic_model(
                    docs=subset["text"].tolist(),
                    embedding_model=embedding_model,
                    min_topic_size=size_to_try,
                    top_n_words=top_n_words,
                    reduce_topics=None,
                    reduce_outliers=reduce_outliers,
                    label=event,
                )
                used_size = size_to_try
                break
            except Exception as e:
                last_error = str(e)
                next_size  = max(2, size_to_try // 2)
                if next_size == size_to_try:
                    break  # already at floor, no point retrying
                print(f"  [{event}] ⚠️  min_topic_size={size_to_try} failed "
                      f"({e}) — retrying with {next_size}")
                size_to_try = next_size

        if event_model is None:
            print(f"  [{event}] ⚠️  All retries failed: {last_error}")
            model_notes[event] = {"status": "failed", "n_articles": len(subset),
                                  "min_topic_size_used": None, "n_topics": 0,
                                  "error": last_error}
            continue
        # ─────────────────────────────────────────────────────────────────────

        topic_info = event_model.get_topic_info()
        n_topics   = int((topic_info["Topic"] != -1).sum())
        label_map  = dict(zip(topic_info["Topic"], topic_info["Name"].map(_canonical_label)))
        subset_idx = subset.index

        df.loc[subset_idx, "event_topic_id"]       = event_topics
        df.loc[subset_idx, "event_topic_min_size"] = used_size
        df.loc[subset_idx, "event_topic_label"]    = [
            label_map.get(t, "outlier") for t in event_topics
        ]

        is_fallback = used_size < event_min_topic_size
        model_notes[event] = {
            "status":             "fallback" if is_fallback else "ok",
            "n_articles":         len(subset),
            "min_topic_size_used": used_size,
            "requested_size":     event_min_topic_size,
            "n_topics":           n_topics,
            "is_sparse":          is_fallback,
        }
        if is_fallback:
            print(f"  [{event}] ℹ️  Used fallback min_topic_size={used_size} "
                  f"(requested {event_min_topic_size}) — flagged as sparse")

        safe_name = event.replace(" ", "_").replace("/", "-")
        topic_info["event_label"]       = event
        topic_info["is_sparse_event"]   = is_fallback
        topic_info["min_topic_size_used"] = used_size
        topic_info.to_csv(
            os.path.join(EVENT_TOPICS_DIR, f"{safe_name}_topic_info.csv"),
            index=False,
        )

        subset = subset.copy()
        subset["event_topic_label"] = df.loc[subset_idx, "event_topic_label"].values
        dist = topic_distribution(subset, ["publisher", "is_control", "event_topic_label"])
        dist["event_label"] = event
        dist.to_csv(
            os.path.join(EVENT_TOPICS_DIR, f"{safe_name}_topic_dist.csv"),
            index=False,
        )

        all_event_dists.append(dist)
        print(f"  [{event}] Saved → {EVENT_TOPICS_DIR}/{safe_name}_*.csv")

    # ── Save model notes (read by visualise.py for footnotes) ─────────────────
    notes_df = pd.DataFrame.from_dict(model_notes, orient="index")
    notes_df.index.name = "event_label"
    notes_df.reset_index(inplace=True)
    notes_path = os.path.join(EVENT_TOPICS_DIR, "event_model_notes.csv")
    notes_df.to_csv(notes_path, index=False)
    print(f"\n  Saved model notes           → {notes_path}")

    if all_event_dists:
        pd.concat(all_event_dists, ignore_index=True).to_csv(
            os.path.join(EVENT_TOPICS_DIR, "all_events_topic_dist.csv"),
            index=False,
        )
        print(f"  Saved combined              → {EVENT_TOPICS_DIR}/all_events_topic_dist.csv")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ❻  DISTRIBUTION TABLES
# ══════════════════════════════════════════════════════════════════════════════

def topic_distribution(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
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

    out_cols = [c for c in df.columns if c != "text"]
    df[out_cols].to_parquet(
        os.path.join(OUTPUT_DIR, "articles_with_topics.parquet"), index=False)
    print(f"\n  Saved article-level results → {OUTPUT_DIR}/articles_with_topics.parquet")

    topic_model.get_topic_info().to_csv(
        os.path.join(OUTPUT_DIR, "topic_info.csv"), index=False)
    print(f"  Saved global topic info     → {OUTPUT_DIR}/topic_info.csv")

    distribution_configs = [
        ("publisher_event",    ["publisher", "event_label", "is_control", "topic_label"]),
        ("event",              ["event_label", "is_control", "topic_label"]),
        ("protest_vs_control", ["is_control", "topic_label"]),
        ("event_type",         ["event_type", "is_control", "topic_label"]),
    ]

    for name, group_cols in distribution_configs:
        if not all(c in df.columns for c in group_cols):
            print(f"  ⚠️  Skipping {name}: missing {set(group_cols) - set(df.columns)}")
            continue
        out = os.path.join(OUTPUT_DIR, f"topic_dist_{name}.csv")
        topic_distribution(df, group_cols).to_csv(out, index=False)
        print(f"  Saved {name:<24s} → {out}")

    topic_model.save(os.path.join(OUTPUT_DIR, "bertopic_model"))
    print(f"  Saved BERTopic model        → {OUTPUT_DIR}/bertopic_model")


# ══════════════════════════════════════════════════════════════════════════════
# ❽  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help="Path to input parquet/CSV. Default: articles_with_sentiment.parquet "
             "(produced by sentiment_analysis.py). Pass corpus_all.parquet to run "
             "without pre-translation, but topic quality will be lower.",
    )
    parser.add_argument("--min-topic-size",       type=int, default=MIN_TOPIC_SIZE)
    parser.add_argument("--top-n-words",          type=int, default=TOP_N_WORDS)
    parser.add_argument("--event-min-topic-size", type=int, default=EVENT_MIN_TOPIC_SIZE)
    parser.add_argument(
        "--reduce-topics", type=int, default=REDUCE_TOPICS, metavar="N",
        help="Merge global topics down to ~N (default 30). Pass 0 to disable.",
    )
    parser.add_argument("--no-event-topics",    action="store_true")
    # Outlier reassignment is OFF by default when clustering on titles: short
    # titles mean ~50-60% of articles genuinely don't cluster thematically, and
    # force-assigning them to their nearest topic creates massive garbage clusters
    # labelled with function words.  Pass --reduce-outliers to re-enable.
    parser.add_argument(
        "--reduce-outliers", action="store_true",
        help="Reassign outlier articles to nearest topic (not recommended for title-only mode).",
    )
    args = parser.parse_args()

    df = load_corpus(args.input)

    print(f"\nLoading embedding model: {EMBEDDING_MODEL} ...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    reduce_outliers = args.reduce_outliers
    reduce_topics   = args.reduce_topics if args.reduce_topics > 0 else None

    global_model, global_topics = fit_global_model(
        df,
        embedding_model=embedding_model,
        min_topic_size=args.min_topic_size,
        top_n_words=args.top_n_words,
        reduce_topics=reduce_topics,
        reduce_outliers=reduce_outliers,
    )
    df = attach_global_topics(df, global_model, global_topics)

    if not args.no_event_topics:
        df = fit_event_models(
            df,
            embedding_model=embedding_model,
            event_min_topic_size=args.event_min_topic_size,
            top_n_words=args.top_n_words,
            reduce_outliers=reduce_outliers,
        )

    save_global_outputs(df, global_model)

    print("\n✅  Topic modelling complete.")
    print(f"    Global outputs → {OUTPUT_DIR}/")
    if not args.no_event_topics:
        print(f"    Per-event CSVs → {EVENT_TOPICS_DIR}/")
    print(f"    Next step: run  python visualise.py")


if __name__ == "__main__":
    main()
