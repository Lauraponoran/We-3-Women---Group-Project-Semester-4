# Women's Protest News Corpus — Analysis Pipeline

A computational analysis of how ideologically diverse news outlets cover women's protest events. The pipeline collects news articles around ten protest events across four languages, runs multilingual topic modelling and sentiment analysis, and produces publication-ready figures.

---

## Table of Contents

1. [Research Design](#research-design)
2. [Pipeline Overview](#pipeline-overview)
3. [Setup](#setup)
4. [Scripts](#scripts)
   - [collect\_articles.py](#collect_articlespy)
   - [build\_corpus.py](#build_corpuspy)
   - [sentiment\_analysis.py](#sentiment_analysispy)
   - [topic\_model.py](#topic_modelpy)
   - [visualise.py](#visualisepy)
   - [failure\_analysis.py](#failure_analysispy)
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
collect_articles.py             ← crawl CC-News per event + control week
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
    +-- articles_with_sentiment.parquet / .csv
    +-- sentiment_*.csv

topic_model.py                  ← reads translated text -> BERTopic (global + per-event)
         |                           NOTE: runs AFTER sentiment_analysis.py (needs translated_title)
    analysis_output/
    +-- articles_with_topics.parquet
    +-- topic_info.csv
    +-- topic_dist_*.csv
    +-- event_topics/

visualise.py                    ← publication figures + bias report
         |
    analysis_output/figures/
```

**Pipeline order matters:** `topic_model.py` must run *after* `sentiment_analysis.py` because it uses the `translated_title` column produced by the sentiment script. Running topic modelling on raw multilingual text causes language-homogeneous clusters. See [Design Decisions](#design-decisions) and `appendix_failure_analysis.png` for details.

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

### `collect_articles.py`

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

**Why article counts vary by event:** CC-News coverage is materially sparser before approximately mid-2019. Events from 2017 and 2018 will return significantly fewer articles than 2022–2023 events regardless of per-publisher caps. The `articles_per_event.png` figure documents this; the `bias_report.csv` Section A flags any publisher × event combination with fewer than 5 articles. Findings for pre-2019 events should be treated as indicative rather than definitive.

**Control weeks** are automatically computed as the same ISO calendar week in 2024 (falling back to 2016). Resumability is saved to `news_output/progress.json`.

```bash
# Recommended first test (one event, small cap)
python collect_articles.py --event Roe_Leak_Protests_2022 --per-publisher 5 --reset-all

# Full run
python collect_articles.py --per-publisher 50

# Merge corpus only (if you already have per-event files)
python collect_articles.py --corpus-only
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

### `sentiment_analysis.py`

Scores every article on VADER's compound sentiment scale (−1 to +1). Non-English articles are translated to English before scoring. **Run this before `topic_model.py`** — it produces the `translated_title` and `translated_body` columns that topic modelling depends on.

```bash
python sentiment_analysis.py
python sentiment_analysis.py --device cuda        # GPU translation
python sentiment_analysis.py --no-translate       # score raw text (EN only)
python sentiment_analysis.py --max-body-chars 1500
```

**Why VADER:** Fully explainable — every score is driven by a transparent word-level lexicon. Preferable for academic research over transformer-based sentiment models, which trade interpretability for marginal accuracy gains.

**Why translate rather than use a multilingual sentiment model:** Translate-then-VADER preserves a complete interpretability story: any article's score can be traced back to specific translated words in the VADER lexicon. This pattern is well-established in computational social science.

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

### `topic_model.py`

Runs BERTopic in two passes using the **translated English text** produced by `sentiment_analysis.py`:

1. **Global model** — fit on the full corpus to discover overarching themes.
2. **Per-event models** — a separate BERTopic fit on each event's articles. Saved to `analysis_output/event_topics/`.

```bash
# Standard (run sentiment_analysis.py first):
python topic_model.py

python topic_model.py --reduce-topics 20   # fewer, broader topics
python topic_model.py --reduce-topics 50   # more granular
python topic_model.py --reduce-topics 0    # disable reduction entirely
python topic_model.py --no-event-topics    # global model only
python topic_model.py --no-reduce-outliers # inspect raw outlier rate
```

**Embedding model:** `all-MiniLM-L6-v2` (~80 MB) — fast, high-quality English sentence encoder. The previous multilingual model (`multilingual-e5-large`, 1.1 GB) is no longer needed because all input text is now English after translation.

**Why BERTopic over LDA / KMeans / DBSCAN:** Does not require specifying *k* in advance; captures semantic similarity via sentence embeddings; produces human-readable TF-IDF keyphrases as labels; returns per-document topic assignments suitable for outlet-level comparisons.

---

### `visualise.py`

Generates 19 publication-ready figures and a bias report CSV.

```bash
python visualise.py
python visualise.py --top-n 12
python visualise.py --control-dir path/to/control
python visualise.py --confusion-labels path/to/labelled_sample.csv
```

| Figure | Description |
|---|---|
| `sentiment_by_outlet.png` | Mean sentiment score per publisher |
| `topic_breakdown_by_event.png` | Top topics per protest event |
| `topic_pie_{event}.png` *(× N)* | Per-event topic donut |
| `topic_pie_overall.png` | Top N topics + rescued feminist topics |
| `topic_pie_all_topics.png` | Every topic, raw counts (diagnostic) |
| `sentiment_by_topic.png` | Mean sentiment × topic volume scatter |
| `articles_per_event.png` | Coverage counts with gap annotations |
| `protest_vs_control.png` | **Primary finding** — protest vs control sentiment |
| `sentiment_ideological_gap.png` | Protest−control delta vs ideological score |
| `sentiment_by_event_type.png` | Single-day vs sustained protest events |
| `language_breakdown.png` | EN/DE/FR/ES share per publisher |
| `pipeline_diagram.png` | Workflow diagram |
| `demographic_distribution.png` | Volume by ideology bin + language group |
| `fairness_metric_comparison.png` | Normalised sentiment gap per publisher |
| `model_performance.png` | Topic model coherence / coverage diagnostics |
| `subgroup_comparison.png` | Sentiment by event × publisher ideology |
| `confusion_matrix_sentiment.png` | Predicted vs ground-truth sentiment labels |
| `workflow_diagram.png` | Extended pipeline diagram |
| `bias_report.csv` | Representativeness statistics |

---

### `failure_analysis.py`

Generates `appendix_failure_analysis.png` — a methods appendix figure documenting four alternative approaches that were evaluated and rejected:

- **Panel A:** Body text vs title-only topic labels (title-only adopted)
- **Panel B:** Effect of `min_topic_size` on cluster granularity
- **Panel C:** Fixed vs adaptive `min_topic_size` per event (four pre-2019 events failed with fixed size)
- **Panel D:** Multilingual embedding collapse before/after translation
- **Panel E:** Pipeline ordering — topic modelling before vs after translation

```bash
python failure_analysis.py
```

Output: `appendix_failure_analysis.png` in the current directory.

---

## Bias Awareness & Mitigation

The pipeline addresses five categories of bias. Where bias cannot be fully eliminated, it is documented and quantified.

### Representation bias

45 publishers across 9 countries, 4 languages, and the full left–right ideological spectrum. CC-News is an open-web crawl rather than a licensed feed. The `bias_report.csv` Section E reports article counts per ideological bin. The 50-article-per-publisher cap is applied uniformly. CC-News over-indexes on publishers with frequent crawl snapshots; smaller or newer publishers may be underrepresented.

### Methodological bias

- Stopword lists are fully documented in `topic_model.py` and cover all four corpus languages; `max_df=0.95` excludes function-word labels
- Topic modelling runs on pre-translated English text, so HDBSCAN clusters by topic rather than language
- BERTopic TF-IDF labels are saved to `topic_info.csv` for manual review
- `topic_id_raw` preserves pre-reassignment assignments; `bias_report.csv` Section D reports both outlier rates
- VADER's complete lexicon is at https://github.com/cjhutto/vaderSentiment

### Framing bias

Matched control weeks directly test the protest-vs-baseline assumption rather than taking it as given. The full news environment is collected rather than protest-filtered articles.

### Model bias (dominant-narrative assumption)

The `sentiment_ideological_gap.png` figure directly tests whether ideological leaning predicts more negative protest-week coverage, making the assumption empirically falsifiable.

### Annotation bias

Not applicable — no human annotation step. Sentiment scoring is fully automated via VADER and topic labelling via BERTopic TF-IDF. Both methods are deterministic and reproducible.

---

## Design Decisions

### Why characterise the full news environment, not just protest articles?

Filtering to protest-relevant articles introduces selection bias: we would only see how outlets frame protests when they choose to cover them. Collecting all articles lets us measure what proportion of the news cycle each outlet devoted to protest-adjacent topics, and whether the overall emotional register of coverage shifts during protest weeks.

### Why matched control weeks?

Protest events are not randomly distributed across the calendar — many cluster around International Women's Day in March, or follow major news triggers. Matching on ISO calendar week in a neutral year controls for seasonal and news-cycle confounds.

### Why multilingual rather than English-only?

Several key outlets publish in German, French, and Spanish. Restricting to English would remove European perspectives central to a cross-national comparison.

### Why VADER + translation rather than a multilingual sentiment model?

Interpretability. VADER's lexicon is public and human-auditable; any score can be explained by pointing to specific words. Translate-then-VADER is an established pattern in computational social science and is straightforward to document in a methods section.

### Why run topic modelling after sentiment analysis?

Topic modelling on raw multilingual text causes HDBSCAN to produce language-homogeneous clusters (one large cluster of English articles, one of German, etc.) even with a multilingual embedding model, because within-language embedding similarity exceeds cross-language topical similarity. Translating first eliminates this. As a side effect, the embedding model can be downgraded from `multilingual-e5-large` (1.1 GB, ~45 min) to `all-MiniLM-L6-v2` (80 MB, ~12 min). See `appendix_failure_analysis.png` Panel D and E.

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
+-- articles_with_sentiment.parquet / .csv
+-- articles_with_topics.parquet
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
    +-- [19 .png files — see Scripts > visualise.py]
    +-- bias_report.csv

appendix_failure_analysis.png   ← methods appendix figure
```

---

## Limitations

- **CC-News pre-2019 sparsity:** events from 2017–2018 have substantially fewer articles. Findings for those events should be treated as indicative.
- **Translation quality:** protest-specific vocabulary (chants, movement-specific terms, political slang) may not translate idiomatically. The `translation_note` column flags failures.
- **VADER lexicon coverage:** VADER was built on English social media text. Some formal journalistic register (particularly German and Austrian broadsheets) may not match its lexicon well even after translation.
- **Topic labelling:** BERTopic names are auto-generated TF-IDF keyphrases and should be manually reviewed before use in figures. Treat `friendly_name` as a working label.
- **Outlier reassignment:** `reduce_outliers(strategy="embeddings")` is a post-hoc heuristic. The raw outlier rate is preserved in `topic_id_raw` and reported in `bias_report.csv` Section D.
- **Ideological scores:** publisher ideology scores in `visualise.py` are hand-assigned based on AllSides and Ad Fontes Media ratings and should be treated as approximate.
- **Control week neutrality:** the control years (2024, 2016) were chosen to avoid major women's rights events, but cannot be fully neutral — 2016 overlaps with the US presidential election cycle.
