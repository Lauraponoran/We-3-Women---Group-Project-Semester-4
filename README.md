# Women's Protest News Corpus — Analysis Pipeline

A computational analysis of how ideologically diverse news outlets cover women's protest events. The pipeline collects news articles around ten protest events across four languages, runs multilingual topic modelling and sentiment analysis, and produces publication-ready figures.

---

## Table of Contents

1. [Research Design](#research-design)
2. [Pipeline Overview](#pipeline-overview)
3. [Setup](#setup)
4. [Scripts](#scripts)
   - [collect\_articles\_optimized.py](#collect_articles_optimizedpy)
   - [build\_corpus.py](#build_corpuspy)
   - [topic\_model.py](#topic_modelpy)
   - [sentiment\_analysis.py](#sentiment_analysispy)
   - [visualise.py](#visualisepy)
5. [Bias Awareness & Mitigation](#bias-awareness--mitigation)
6. [Design Decisions](#design-decisions)
7. [Outputs](#outputs)
8. [Limitations](#limitations)

---

## Research Design

We characterise the **entire news environment** during each protest event week — not only protest-relevant articles. This allows us to ask:

- Is the overall news environment more negative during protest weeks than matched control weeks?
- Do ideologically different outlets assign systematically different topic distributions to protest-adjacent coverage?
- Does sentiment toward protest-related topics differ from sentiment toward other topics in the same outlet during the same week?

Each protest event is paired with a **matched control week**: the same ISO calendar week in a neutral year (2024 or 2016), ensuring seasonal and news-cycle confounds are minimised. Collecting the full news environment (not just filtered protest articles) means topic and sentiment differences are measured against a realistic baseline rather than a curated subset.

---

## Pipeline Overview

```
collect_articles_optimized.py   ← crawl CC-News per event + control week
         |
    news_output/
    +-- {event_label}/
        +-- incremental/        ← one parquet per publisher (resumable)
        +-- {event_label}.parquet

build_corpus.py                 ← merge all events into one file
         |
    news_output/corpus_all.parquet

sentiment_analysis.py           ← language detection -> translation -> VADER
         |                           also saves translated_title / translated_body
    analysis_output/
    +-- articles_with_sentiment.parquet / .csv   (contains translated text columns)
    +-- sentiment_*.csv

topic_model.py                  ← reads translated text -> BERTopic (global + per-event)
         |
    analysis_output/
    +-- articles_with_topics.parquet
    +-- topic_info.csv
    +-- topic_dist_*.csv
    +-- event_topics/

visualise.py                    ← publication figures + bias report
         |
    analysis_output/figures/
    +-- sentiment_by_outlet.png
    +-- topic_breakdown_by_event.png
    +-- topic_pie_{event}.png  (x N events)
    +-- topic_pie_overall.png
    +-- topic_pie_all_topics.png      <- every topic, nothing collapsed
    +-- sentiment_by_topic.png
    +-- articles_per_event.png        <- coverage gap documentation
    +-- protest_vs_control.png        <- primary research finding
    +-- sentiment_ideological_gap.png <- fairness metric
    +-- sentiment_by_event_type.png   <- subgroup comparison
    +-- language_breakdown.png
    +-- pipeline_diagram.png          <- methods figure
    +-- bias_report.csv
```

---

## Setup

```bash
pip install fundus bertopic sentence-transformers pandas pyarrow tqdm \
            lingua-language-detector transformers sentencepiece torch \
            vaderSentiment matplotlib numpy
```

> **Note:** `lingua-language-detector` is the Rust-backed language detection library by Martin Möller. Do **not** install the unrelated `lingua` package — they share a name but are incompatible.

Python 3.10+ recommended. GPU optional but speeds up translation significantly; pass `--device cuda` to `sentiment_analysis.py` if available.

---

## Scripts

### `collect_articles_optimized.py`

Crawls [CC-News](https://commoncrawl.org/the-data/get-started/) via the [Fundus](https://github.com/flairNLP/fundus) library to collect articles for each protest event and its matched control week.

**Events covered**

| Label | Event | Window |
|---|---|---|
| `Womens_March_2017` | Women's March | Jan 19–26 2017 |
| `International_Womens_Strike_2017` | International Women's Strike | Mar 6–13 2017 |
| `MeToo_Protests_2017` | #MeToo protests | Oct 15–22 2017 |
| `IWD_2018_Global` | IWD 2018 + Spanish Feminist Strike | Mar 6–13 2018 |
| `Swiss_Womens_Strike_2019` | Swiss Women's Strike | Jun 12–19 2019 |
| `Polish_Womens_Strike_2020` | Polish Constitutional Tribunal ruling | Oct 22–29 2020 |
| `Sarah_Everard_Vigils_2021` | Sarah Everard vigils | Mar 11–18 2021 |
| `Roe_Leak_Protests_2022` | Dobbs leak + protests | May 2–9 2022 |
| `Women_Life_Freedom_2022` | Mahsa Amini / Jin Jiyan Azadi | Sep 16–23 2022 |
| `Israeli_Womens_Protests_2023` | Israeli women's protests | Feb 27–Mar 6 2023 |

**Publishers** span 9 countries and cover a broad ideological range — from outlets like *The Nation* and *The Intercept* on the left through wire services (*AP*, *Reuters*) to *Fox News*, *The Daily Mail*, and *The Gateway Pundit* on the right, plus European and Spanish-language sources.

**Why article counts vary by event:** CC-News coverage is materially sparser before approximately mid-2019. Events from 2017 and 2018 (Women's March, #MeToo, IWD 2018, International Women's Strike) will return significantly fewer articles than 2022–2023 events regardless of per-publisher caps, because fewer crawl snapshots exist for those dates. The `articles_per_event.png` figure documents this with per-event counts and annotations; the `bias_report.csv` Section A flags any publisher × event combination with fewer than 5 articles. This asymmetry should be reported in the methods section and findings for pre-2019 events should be treated as indicative rather than definitive.

**Control weeks** are automatically computed as the same ISO calendar week in 2024 (falling back to 2016 if the week falls outside the target year). This matches seasonal news patterns while avoiding contamination from other feminist or protest events.

**Resumability:** progress is saved to `news_output/progress.json` after each publisher. Interrupted runs can be restarted without re-crawling completed publishers.

```bash
# Recommended first test (one event, small cap)
python collect_articles_optimized.py --event Roe_Leak_Protests_2022 --per-publisher 5 --reset-all

# Full run
python collect_articles_optimized.py --per-publisher 50

# Merge corpus only (if you already have per-event files)
python collect_articles_optimized.py --corpus-only
```

---

### `build_corpus.py`

Merges all per-event CSVs (excluding incremental files) into a single `corpus_all.parquet`. Run this if you assembled per-event files manually or want to rebuild the corpus without re-crawling.

```bash
python build_corpus.py
python build_corpus.py --input-dir path/to/news_output
```

Output: `news_output/corpus_all.parquet`

---

### `topic_model.py`

Runs BERTopic in two passes:

1. **Global model** — fit on the full corpus to discover overarching themes. Topic IDs and labels from this pass are what `sentiment_analysis.py` uses for cross-event comparisons.
2. **Per-event models** — a separate BERTopic fit on each event's articles to discover fine-grained topics specific to that event's news cycle. Saved to `analysis_output/event_topics/`.

```bash
# Standard: run sentiment_analysis.py first, then:
python topic_model.py

# Explicit input (same as default):
python topic_model.py --input analysis_output/articles_with_sentiment.parquet

# Raw corpus without translation benefit (not recommended):
python topic_model.py --input news_output/corpus_all.parquet

python topic_model.py --reduce-topics 20   # fewer, broader topics
python topic_model.py --reduce-topics 50   # more granular
python topic_model.py --reduce-topics 0    # disable reduction entirely
python topic_model.py --no-event-topics    # global model only
python topic_model.py --no-reduce-outliers # inspect raw outlier rate
```

**Pipeline order:** `topic_model.py` now runs *after* `sentiment_analysis.py`, not before it. `sentiment_analysis.py` saves `translated_title` and `translated_body` columns alongside the sentiment scores. `topic_model.py` reads these columns and uses the translated English text for embedding and clustering. This means BERTopic clusters articles by **topic** rather than by **language** — without this step, HDBSCAN produces language-homogeneous clusters (one large cluster of English articles, one of German, one of French) even with a multilingual embedding model, because within-language embedding similarity is higher than cross-language similarity on the same topic.

The embedding model has also been changed from `intfloat/multilingual-e5-large` to `all-MiniLM-L6-v2` — a fast, high-quality English sentence encoder (~80 MB). Since all input text is now English, the multilingual model's cross-language alignment is no longer needed and the smaller model is faster and produces tighter English topic clusters.

**Why BERTopic over LDA / KMeans / DBSCAN:**
- KMeans and hierarchical methods require specifying *k* in advance; the number of topics in a multilingual protest news corpus is not known a priori
- LDA assumes a bag-of-words model and does not capture semantic similarity across languages
- BERTopic uses sentence embeddings + HDBSCAN to find *k* automatically and produces human-readable TF-IDF keyphrases as labels
- Returns per-document topic assignments suitable for outlet-level distribution comparisons

**Outlier reduction (two-stage):** HDBSCAN assigns a large "outlier" bucket (topic `-1`) when `min_topic_size` is too large relative to corpus density. Two measures reduce this:

1. `min_topic_size` is set to 3 (global) and 2 (per-event), down from 5 and 3 respectively, so HDBSCAN creates more and smaller initial clusters.
2. `reduce_outliers(strategy="embeddings")` reassigns remaining outlier documents to their nearest topic by raw embedding cosine distance. Embedding-based reassignment is used rather than `"c-tf-idf"` because it works better for short articles and minority-language documents that have few term overlaps with any topic cluster. `min_df=1` in the CountVectorizer (previously 2) ensures rare protest-specific terms are available for TF-IDF labelling rather than being silently dropped.

The original HDBSCAN assignment is preserved in `topic_id_raw` for auditability. Pass `--no-reduce-outliers` to inspect the raw outlier rate. The `bias_report.csv` Section D reports both rates.

**Embedding model:** `intfloat/multilingual-e5-large` — encodes articles in EN/DE/FR/ES into a shared semantic space so that articles about the same theme cluster by content rather than by language. Requires the `"passage: "` prefix at inference time (applied automatically by `_prefix_docs()`).

> The model is ~1.1 GB. If download is slow, run:
> ```bash
> pip install hf-transfer
> HF_HUB_ENABLE_HF_TRANSFER=1 python topic_model.py
> ```
> Or substitute `intfloat/multilingual-e5-base` (~560 MB) for a lighter alternative.

---

### `sentiment_analysis.py`

Scores every article on VADER's compound sentiment scale (−1 to +1). Non-English articles are translated to English before scoring.

```bash
python sentiment_analysis.py
python sentiment_analysis.py --device cuda        # GPU translation
python sentiment_analysis.py --no-translate       # score raw text (EN only)
python sentiment_analysis.py --max-body-chars 1500
```

**Why VADER:**
VADER (Valence Aware Dictionary and sEntiment Reasoner) was designed specifically for news and social media text. Its compound score is fully explainable — every score is driven by a transparent word-level lexicon with no black-box model weights — making it straightforward to audit and report. This is preferable for academic research over transformer-based sentiment models, which trade interpretability for marginal accuracy gains.

**Why translate rather than use a multilingual sentiment model:**
Multilingual neural sentiment models (e.g. `twitter-xlm-roberta-base-sentiment`) are accurate but opaque. Translate-then-VADER preserves the full interpretability story: any article's score can be traced back to specific translated words in the VADER lexicon. This pattern is well-established in computational social science and is straightforward to document in a methods section.

**Translation models:** Helsinki-NLP opus-mt (`de->en`, `fr->en`, `es->en`), loaded on demand and cached. Models are ~300 MB total. Sentiment-bearing vocabulary (condemn, celebrate, outrage, violence) translates reliably across these language pairs; potential loss of idiomaticity is documented per article in the `translation_note` column.

**Language detection:** `lingua-language-detector` restricted to EN/DE/FR/ES. Restricting the candidate set improves accuracy on short news snippets compared to full-corpus detection across 75 languages.

**Output columns added:**

| Column | Description |
|---|---|
| `language` | Detected language code: EN / DE / FR / ES / OTHER |
| `was_translated` | Boolean — whether the article was translated before scoring |
| `translation_note` | Empty string on success; error message if translation failed |
| `sentiment_score` | VADER compound score in [−1, +1] |
| `sentiment_pos/neu/neg` | VADER component scores |
| `sentiment_label` | `positive` / `neutral` / `negative` |
| `translated_title` | English translation of title (= original for EN articles) |
| `translated_body` | English translation of body (= original for EN articles) |

---

### `visualise.py`

Generates 12 publication-ready figures and a bias report CSV from the sentiment and topic outputs.

```bash
python visualise.py
python visualise.py --top-n 12                     # topics in per-event and pie charts
python visualise.py --control-dir path/to/control  # point at your control week folder
```

Control week data lives in a separate directory (`control/` by default, sibling to `news_output/`). The script loads and merges it automatically before producing protest-vs-control charts; all other charts use protest weeks only.

| Figure | Description |
|---|---|
| `sentiment_by_outlet.png` | Mean sentiment score per publisher |
| `topic_breakdown_by_event.png` | Top topics per protest event, bar chart |
| `topic_pie_{event}.png` *(x N)* | Per-event topic donut, one file per event |
| `topic_pie_overall.png` | Top N topics + rescued feminist topics donut |
| `topic_pie_all_topics.png` | **Every** discovered topic, nothing collapsed into "other" |
| `sentiment_by_topic.png` | Mean sentiment x topic volume scatter |
| `articles_per_event.png` | Article counts per event with coverage gap annotations |
| `protest_vs_control.png` | **Primary finding** — protest vs control sentiment per publisher |
| `sentiment_ideological_gap.png` | Fairness metric: protest−control delta vs ideological score |
| `sentiment_by_event_type.png` | Subgroup: single-day vs sustained protest events |
| `language_breakdown.png` | EN/DE/FR/ES share per publisher |
| `pipeline_diagram.png` | Workflow diagram for methods section |
| `bias_report.csv` | Representativeness statistics (see Bias section) |

Feminist and protest-related topics are rendered in **amber/gold** (`#E6A817`) with a dark outline across all charts, and a diamond marker shape in the scatter. This distinguishes them from raspberry (negative) and teal (positive) by hue, brightness, and shape simultaneously — readable in greyscale.

---

## Bias Awareness & Mitigation

The pipeline addresses five categories of bias. Where bias cannot be fully eliminated, it is documented and quantified.

### Representation bias

**What it is:** the corpus is a subset of all possible news publishers; conclusions are only as representative as the publishers included.

**What we do:**
- 45 publishers across 9 countries, 4 languages, and the full left–right ideological spectrum (scored using AllSides / Ad Fontes Media indices)
- CC-News is an open-web crawl rather than a licensed feed, reducing the skew toward English-language anglophone sources common in proprietary datasets
- The `bias_report.csv` Section E reports article counts per ideological bin (left / centre / right) so imbalance is visible and reportable

**Why the sample is broadly representative:** CC-News crawls are not systematically biased toward any political orientation — they reflect what publishers actually publish online. The 50-article-per-publisher cap is applied uniformly, preventing high-volume publishers from dominating the corpus. The `articles_per_event.png` chart and `bias_report.csv` Section A document any publisher × event combinations with fewer than 5 articles, allowing you to flag underrepresented outlets explicitly.

**Remaining caveat:** CC-News over-indexes on publishers with frequent crawl snapshots (typically larger, more established outlets). Smaller or newer publishers may be systematically underrepresented. The `language_breakdown.png` chart quantifies the EN/DE/FR/ES split per publisher so this can be reported.

### Methodological bias

**What it is:** analytical choices — stopword lists, topic model hyperparameters, sentiment lexicon — embed assumptions that shape results.

**What we do:**
- Stopword lists are fully documented in `topic_model.py` and cover all four corpus languages; they are deterministic and inspectable. `max_df=0.95` in the CountVectorizer provides a second layer of defence against function-word labels by excluding any term appearing in more than 95% of documents. The root cause of language-cluster topics ("The, To, And"; "Die, Der, Und") is now solved upstream: topic modelling runs on pre-translated English text produced by `sentiment_analysis.py`, so HDBSCAN clusters by topic rather than by language
- BERTopic's TF-IDF labels are reproducible; the full topic info is saved to `topic_info.csv` for manual review
- `topic_id_raw` preserves pre-reassignment topic assignments; `bias_report.csv` Section D reports the raw and final outlier rates side by side, so the impact of the reassignment heuristic is transparent
- `min_df=1` ensures no terms are silently dropped from topic labelling
- VADER's complete lexicon is publicly available at https://github.com/cjhutto/vaderSentiment

### Framing bias

**What it is:** the research design assumes protest weeks differ from control weeks; the choice of which events to include and how to window them shapes what is measurable.

**What we do:**
- Matched control weeks (same ISO calendar week, neutral year) directly test the assumption rather than taking it as given — if no sentiment gap appears, that is a finding
- The full news environment is collected rather than protest-filtered articles, so "coverage share" is a meaningful metric rather than a presupposition
- Event windows are ±1 day around the core protest dates, documented in the events table; window choice is a design parameter and should be reported as such

### Model bias (dominant-narrative assumption)

**What it is:** the research assumes that existing news coverage reflects and reinforces dominant cultural narratives about women's protest. This is a theoretical prior, not an empirical finding.

**What we do:**
- The `sentiment_ideological_gap.png` figure directly tests whether ideological leaning predicts more negative protest-week coverage, making the assumption empirically falsifiable
- The `sentiment_by_event_type.png` subgroup comparison tests whether sustained vs single-day protest events are covered differently, which probes narrative assumptions about protest legitimacy
- Reporting the ideological gap with a trend line and per-publisher labels allows readers to evaluate the claim rather than accepting it

### Annotation bias

Not applicable — the pipeline contains no human annotation step. Sentiment scoring is fully automated via VADER (lexicon-based) and topic labelling is automated via BERTopic TF-IDF. Both methods are deterministic and reproducible.

---

## Design Decisions

### Why characterise the full news environment, not just protest articles?

Filtering to protest-relevant articles introduces a selection bias: we would only see how outlets frame protests when they choose to cover them. Collecting all articles in the event window lets us measure what proportion of the news cycle each outlet devoted to protest-adjacent topics, and whether the overall emotional register of coverage shifts during protest weeks — questions that cannot be answered from a filtered sample.

### Why matched control weeks?

Protest events are not randomly distributed across the calendar — many cluster around International Women's Day in March, or follow major news triggers. A simple pre/post comparison would conflate protest-week effects with seasonal patterns (e.g. March news is structurally different from October news). Matching on ISO calendar week in a neutral year controls for this.

### Why multilingual rather than English-only?

Several key outlets in the corpus publish in German, French, and Spanish. Restricting to English would remove European left and right-wing perspectives that are central to a cross-national comparison. The multilingual embedding model clusters articles by theme across languages, so a German article about the Polish Women's Strike and an English article about the same event will end up in the same topic cluster.

### Why VADER + translation rather than a multilingual sentiment model?

Interpretability. VADER's lexicon is public and human-auditable; any score can be explained by pointing to specific words. Multilingual transformer models produce better raw accuracy on benchmark datasets but provide no mechanism for explaining individual scores, which makes them harder to defend in a social science paper. Translate-then-VADER is an established pattern in the field and is straightforward to document.

---

## Outputs

```
news_output/
+-- progress.json
+-- corpus_all.parquet
+-- {event_label}/
    +-- {event_label}.parquet / .csv
    +-- incremental/

control/
+-- {control_label}/
    +-- {control_label}.parquet / .csv

analysis_output/
+-- articles_with_topics.parquet      <- topic_id / topic_id_raw / topic_label
+-- articles_with_sentiment.parquet / .csv
+-- topic_info.csv
+-- topic_dist_publisher_event.csv
+-- topic_dist_event.csv
+-- topic_dist_protest_vs_control.csv
+-- topic_dist_event_type.csv
+-- sentiment_publisher_event.csv
+-- sentiment_publisher.csv
+-- sentiment_event.csv
+-- sentiment_protest_vs_control.csv
+-- sentiment_topic.csv
+-- sentiment_publisher_topic.csv
+-- sentiment_language.csv
+-- sentiment_publisher_language.csv
+-- bertopic_model/
+-- event_topics/
|   +-- all_events_topic_dist.csv
|   +-- {event_label}_topic_info.csv / _topic_dist.csv
+-- figures/
    +-- sentiment_by_outlet.png
    +-- topic_breakdown_by_event.png
    +-- topic_pie_{event_label}.png    <- one per event
    +-- topic_pie_overall.png
    +-- topic_pie_all_topics.png       <- every topic, no collapsing
    +-- sentiment_by_topic.png
    +-- articles_per_event.png
    +-- protest_vs_control.png
    +-- sentiment_ideological_gap.png
    +-- sentiment_by_event_type.png
    +-- language_breakdown.png
    +-- pipeline_diagram.png
    +-- bias_report.csv
```

---

## Limitations

- **CC-News pre-2019 sparsity:** events from 2017–2018 have substantially fewer articles than post-2019 events regardless of per-publisher caps, because fewer crawl snapshots exist. The `articles_per_event.png` chart documents this; findings for those events should be treated as indicative.
- **Translation quality:** protest-specific vocabulary (chants, movement-specific terms, political slang) may not translate idiomatically. The `translation_note` column flags failures; the `sentiment_language` aggregate table allows verification that translated articles do not score systematically differently from native English ones.
- **VADER lexicon coverage:** VADER was built on English social media text. Some formal journalistic register (particularly in German and Austrian broadsheets) may not match its lexicon well even after translation.
- **Topic labelling:** BERTopic topic names are auto-generated TF-IDF keyphrases and should be manually reviewed before use in figures or tables. The `friendly_name` column takes the first three keywords; treat these as working labels.
- **Outlier reassignment:** `reduce_outliers(strategy="embeddings")` is a post-hoc heuristic — some reassignments will be imprecise for articles that are genuinely topically ambiguous. The raw outlier rate is preserved in `topic_id_raw` and reported in `bias_report.csv` Section D.
- **Ideological scores:** publisher ideology scores in `visualise.py` are hand-assigned based on AllSides and Ad Fontes Media ratings and should be treated as approximate. They are not updated dynamically.
- **Control week neutrality:** the control years (2024, 2016) were chosen to avoid major women's rights events, but cannot be fully neutral — 2016 in particular overlaps with the US presidential election cycle.
