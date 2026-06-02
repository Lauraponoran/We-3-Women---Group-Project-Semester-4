# Women's Protest News Corpus — Analysis Pipeline

A computational analysis of how ideologically diverse news outlets cover women's protest events. The pipeline collects news articles around ten protest events across four languages, runs multilingual topic modelling and sentiment analysis, and produces publication-ready figures.

---

## Table of Contents

1. [Research Design](#research-design)
2. [Pipeline Overview](#pipeline-overview)
3. [Setup](#setup)
4. [Scripts](#scripts)
   - [collect_articles.py](#collect_articlespy)
   - [build_corpus.py](#build_corpuspy)
   - [sentiment_analysis.py](#sentiment_analysispy)
   - [topic_model.py](#topic_modelpy)
   - [visualise.py](#visualisepy)
   - [process_control_isolated.py](#process_control_isolatedpy)
   - [failure_analysis.py](#failure_analysispy)
   - [run_single.sh](#run_singlesh)
5. [Reproduction Commands](#reproduction-commands)
6. [Bias Awareness & Mitigation](#bias-awareness--mitigation)
7. [Design Decisions](#design-decisions)
8. [Outputs](#outputs)
9. [Limitations](#limitations)

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

process_control_isolated.py     ← optional control-only BERTopic model
         |                          trained only on control-week articles
    control/* updated with
    topic_id / topic_label

failure_analysis.py            ← generates methods appendix figure
         |
    analysis_output/figures/
    +-- appendix_failure_analysis.png

run_single.sh                  ← SLURM batch script for large-scale
                                  article collection on Snellius HPC

requirements.txt               ← pinned project dependencies
```

**Pipeline order matters:** `topic_model.py` must run *after* `sentiment_analysis.py` because it uses the `translated_title` column produced by the sentiment script. Running topic modelling on raw multilingual text causes language-homogeneous clusters. See [Design Decisions](#design-decisions) and `appendix_failure_analysis.png` for details.

---

## Setup

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

### Main dependencies

- Fundus (CC-News article collection)
- BERTopic (topic modelling)
- Sentence Transformers (semantic embeddings)
- HDBSCAN + UMAP (topic clustering)
- VADER Sentiment
- Pandas / PyArrow (data processing)
- Ollama (optional local LLM support)

> **Note:** `lingua-language-detector` is the Rust-backed language detection library by Martin Möller. Do **not** install the unrelated `lingua` package — they share a name but are incompatible.

> **Note:** GPU acceleration is optional but significantly speeds up translation during sentiment analysis.

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

1. **Global model** — fit on the full corpus to discover overarching themes. `topic_id` / `topic_label` columns are used for cross-event comparisons in `visualise.py`.
2. **Per-event models** — a separate BERTopic is fit on each event's articles to discover fine-grained topics specific to that event's news cycle. Results are saved to `analysis_output/event_topics/` and attached to the dataframe as `event_topic_id` / `event_topic_label`.

**Key design choices:**

- **Title-only clustering:** topic modelling runs on `translated_title` only, not body text. Titles are concise human-written summaries that produce much cleaner topic labels; bodies add boilerplate wire-service language that pollutes labels.
- **Embedding model:** `all-MiniLM-L6-v2` (~80 MB) — fast, high-quality English sentence encoder. The previous multilingual model (`multilingual-e5-large`, 1.1 GB) is no longer needed because all input text is now English after translation.
- **HTML stripping:** a preprocessing step removes `<style>`, `<script>`, and inline CSS from scraped article titles before text reaches BERTopic, preventing nonsense labels like "Css, Inlinelink, Yidnqd".
- **Stopwords:** four-language stopword lists (EN/DE/FR/ES) plus a `NEWS_STOPWORDS` list covering wire-service names, journalistic hedges, and date/number words — all merged and deduplicated. CountVectorizer also applies `max_df=0.60`, `min_df=2`, and `token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"` (requires ≥3 alphabetic characters, stripping accents first) as complementary defences.
- **Label quality:** `ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)` down-weights terms frequent across topics; `MaximalMarginalRelevance(diversity=0.4)` diversifies final label words. KeyBERTInspired was removed because centroid-similarity re-ranking resurfaces high-frequency function words on protest corpora.
- **Outlier reassignment:** disabled by default in title-only mode (~50–60% of short article titles do not cluster thematically; force-reassigning them creates garbage clusters). Pass `--reduce-outliers` to re-enable.
- **Adaptive per-event `min_topic_size`:** if a sparse event causes a CountVectorizer conflict, the model retries with progressively halved values down to a floor of 2, and flags the event in `event_topics/event_model_notes.csv`.
- **Canonical label deduplication:** raw BERTopic topic names have their three keyword words sorted alphabetically before display, so `"0_warhol_monroe_marilyn"` and `"4_marilyn_monroe_warhol"` produce the same `"Marilyn, Monroe, Warhol"` label and are counted together.

```bash
# Standard (run sentiment_analysis.py first):
python topic_model.py

python topic_model.py --reduce-topics 20   # fewer, broader topics
python topic_model.py --reduce-topics 50   # more granular
python topic_model.py --reduce-topics 0    # disable reduction entirely
python topic_model.py --no-event-topics    # global model only
python topic_model.py --no-reduce-outliers # inspect raw outlier rate
python topic_model.py --reduce-outliers    # re-enable outlier reassignment
```

**Why BERTopic over LDA / KMeans / DBSCAN:** Does not require specifying *k* in advance; captures semantic similarity via sentence embeddings; produces human-readable TF-IDF keyphrases as labels; returns per-document topic assignments suitable for outlet-level comparisons.

---

### `visualise.py`

Generates 22 publication-ready figures and a bias report CSV. Charts 2 and 3 use **per-event topic models** (`event_topic_label` column) so each event's pie shows topics discovered independently from only that event's articles. The global model is used for cross-event comparisons (charts 4–7) where a shared topic vocabulary is needed.

```bash
python visualise.py
python visualise.py --top-n 12
python visualise.py --control-dir path/to/control
python visualise.py --confusion-labels path/to/labelled_sample.csv
```

| Figure | Description |
|---|---|
| `sentiment_by_outlet.png` | Mean sentiment score per publisher |
| `topic_breakdown_by_event.png` | Top topics per protest event (per-event models, normalised) |
| `topic_pie_{event}.png` *(× N)* | Per-event topic donuts using per-event topic models |
| `topic_pie_protest_weeks.png` | Overall protest weeks donut (global model, normalised) |
| `topic_pie_control_weeks.png` | Overall control weeks donut (global model, normalised) |
| `topic_pie_all_topics.png` | Every global topic, raw counts (diagnostic) |
| `topic_pie_all_topics_normalised.png` | Every global topic, normalised across events |
| `sentiment_by_topic.png` | Mean sentiment × topic volume scatter |
| `articles_per_event.png` | Coverage counts with gap annotations |
| `protest_vs_control.png` | **Primary finding** — protest vs control sentiment |
| `sentiment_ideological_gap.png` | Protest−control delta vs ideological score |
| `sentiment_by_event_type.png` | Single-day vs sustained protest events |
| `language_breakdown.png` | EN/DE/FR/ES share per publisher |
| `pipeline_diagram.png` | Workflow diagram |
| `demographic_distribution.png` | Volume by ideology bin + language group |
| `fairness_cohens_d_comparison.png` | Cohen's d effect size per publisher |
| `model_performance.png` | Topic model coherence / coverage diagnostics |
| `fairness_subgroup_intersection.png` | Sentiment heatmap: event × ideology |
| `confusion_matrix_sentiment.png` | Predicted vs ground-truth sentiment labels |
| `workflow_diagram.png` | Extended pipeline diagram |
| `topic_pie_control_{event}.png` *(× N)* | Per-event control week topic donuts (global model) |
| `topic_pie_control_weeks_normalised.png` | Overall normalised control week topic breakdown |
| `bias_report.csv` | Representativeness statistics |

**Control week topic charts** (`topic_pie_control_{event}.png`, `topic_pie_control_weeks_normalised.png`) use the global topic vocabulary — not per-event models — so protest and control distributions are directly comparable on a shared axis.

**Feminist/protest topic highlighting:** topics whose labels match a keyword list (`FEMINIST_KEYWORDS`) are rendered in gold with a star (★) in legend entries and exploded slightly in donut charts, making it easy to identify protest-relevant topics regardless of their ranked position.

**Ideological scoring:** publisher ideology scores are hand-assigned based on AllSides and Ad Fontes Media ratings. The `sentiment_ideological_gap.png` figure directly tests whether leaning predicts more negative protest-week coverage by plotting protest−control sentiment delta against ideology score and fitting a trend line.

---
### `process_control_isolated.py`

Optional utility script that trains a completely separate BERTopic model using only control-week articles.

Unlike the main pipeline, which applies a shared global topic vocabulary across protest and control weeks, this script creates an independent topic space for the control corpus. This can be useful for diagnostic analyses, robustness checks, or exploratory comparisons.

The script:

1. Loads all files in `control/`
2. Removes HTML artefacts from titles
3. Builds a multilingual stopword list
4. Trains a dedicated BERTopic model on control articles only
5. Generates topic labels and topic IDs
6. Writes the results back into the original control CSV and Parquet files

```bash
python process_control_isolated.py
```

Outputs:

```text
analysis_output/
+-- control_topic_info.csv

control/
+-- */*.csv
+-- */*.parquet
    (updated with topic_id and topic_label columns)
```

**Note:** This script is not required for reproducing the main results reported in the paper. It was developed as an exploratory robustness analysis.
---

### `failure_analysis.py`

Generates a publication-ready methods appendix figure (`appendix_failure_analysis.png`) documenting alternative modelling strategies that were evaluated and ultimately rejected during pipeline development.

The figure provides transparency about major methodological decisions and demonstrates why the final pipeline configuration was adopted.

Panels include:

- **A:** Full-body topic modelling versus title-only topic modelling
- **B:** Sensitivity analysis of `min_topic_size`
- **C:** Fixed versus adaptive per-event topic modelling
- **D:** Multilingual clustering before and after translation
- **E:** Pipeline ordering (topic modelling before vs after translation)

```bash
python failure_analysis.py
```

Output:

```text
analysis_output/figures/
+-- appendix_failure_analysis.png
```

This figure is intended for supplementary materials and reproducibility documentation rather than the main analysis pipeline.

---
### `run_single.sh`

SLURM batch script used to run article collection on the SURF Snellius HPC cluster.

The script:

- activates the project virtual environment
- launches `collect_articles.py`
- stores job logs in a dedicated logs directory
- allows long-running collection jobs to execute without interruption

Submit a job:

```bash
sbatch run_single.sh
```

Monitor jobs:

```bash
squeue -u <username>
```

Cancel a job:

```bash
scancel <jobid>
```

View logs:

```bash
tail -f logs/<jobid>.out
```

This script is only required when running the collection stage on a SLURM-managed computing cluster and is not needed for local execution.
---

## Reproduction Commands

The tables and figures in this project were generated with the following commands, in order after `sentiment_analysis.py` had completed:

```bash
python topic_model.py --event-min-topic-size 15 --reduce-topics 30
python visualise.py --top-n 20
```

**Why these flags:**

`--event-min-topic-size 15` sets the starting `min_topic_size` for per-event BERTopic models. A value of 15 requires at least 15 articles to form a topic cluster, which filters out noise from one-off articles while still being small enough to capture meaningful topics in events with moderate coverage (200–500 articles). For the four sparse pre-2019 events (fewer than ~100 articles), the adaptive retry logic automatically halves this value until a model succeeds, flagging those events as sparse in `event_model_notes.csv`.

`--reduce-topics 30` merges the global model's initially discovered micro-topics down to approximately 30 macro-themes via BERTopic's hierarchical topic reduction. Without reduction, `min_topic_size=3` on ~7,500 articles discovers hundreds of micro-topics that collapse into "All Other Topics" in every visualisation. 30 topics produces a balance between granularity and readability across the 22 output figures.

`--top-n 20` in `visualise.py` shows the top 20 topics by article share in bar charts and donut charts before the remainder is collapsed into "All Other Topics". 20 was chosen because the global model produces ~30 topics after reduction; showing 20 captures the substantive majority of coverage while keeping figures legible. Feminist/protest topics below rank 20 are rescued from the "Other" bucket regardless of this threshold.

---

## Bias Awareness & Mitigation

The pipeline addresses five categories of bias. Where bias cannot be fully eliminated, it is documented and quantified.

### Representation bias

45 publishers across 9 countries, 4 languages, and the full left–right ideological spectrum. CC-News is an open-web crawl rather than a licensed feed. The `bias_report.csv` Section E reports article counts per ideological bin. The 50-article-per-publisher cap is applied uniformly. CC-News over-indexes on publishers with frequent crawl snapshots; smaller or newer publishers may be underrepresented.

### Methodological bias

- Stopword lists are fully documented in `topic_model.py` and cover all four corpus languages; `max_df=0.60` excludes high-frequency terms that evade stopwords due to casing or tokenisation variants
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

### Why title-only topic modelling?

Body text contains repeated wire-service boilerplate, dateline language, HTML artefacts, and generic journalistic hedges that dominate TF-IDF labels regardless of topical content. Titles are concise, human-written summaries with a much higher signal-to-noise ratio. Panel A of `appendix_failure_analysis.png` documents the label quality comparison.

### Why are per-event topic models used for charts 2 and 3, but the global model for the overview pies?

Per-event models allow each event's donut to reflect the specific vocabulary of that news cycle — the Polish Women's Strike topic pie should not be forced to use topic labels coined from the global Women's March vocabulary. For cross-event comparisons (the overview pies and the protest-vs-control topic breakdowns) a shared vocabulary is necessary, so the global model is used there.

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
+-- control_topic_info.csv
+-- bertopic_model/
+-- event_topics/
|   +-- all_events_topic_dist.csv
|   +-- event_model_notes.csv
|   +-- {event_label}_topic_info.csv / _topic_dist.csv
+-- figures/
|   +-- [22 .png files — see Scripts > visualise.py]
|   +-- appendix_failure_analysis.png
|   +-- bias_report.csv

```

---

## Limitations

- **CC-News pre-2019 sparsity:** events from 2017–2018 have substantially fewer articles. Findings for those events should be treated as indicative.
- **Translation quality:** protest-specific vocabulary (chants, movement-specific terms, political slang) may not translate idiomatically. The `translation_note` column flags failures.
- **VADER lexicon coverage:** VADER was built on English social media text. Some formal journalistic register (particularly German and Austrian broadsheets) may not match its lexicon well even after translation.
- **Topic labelling:** BERTopic names are auto-generated TF-IDF keyphrases and should be manually reviewed before use in figures. Treat `friendly_name` as a working label.
- **Outlier rate in title-only mode:** because article titles are short and topically ambiguous, a high proportion of articles (~50–60%) are not assigned to any topic cluster. This is expected behaviour and does not indicate a modelling failure. Outlier documents are excluded from topic distribution charts but are included in all sentiment analyses.
- **Outlier reassignment disabled by default:** pass `--reduce-outliers` to re-enable. The raw outlier rate is preserved in `topic_id_raw` and reported in `bias_report.csv` Section D.
- **Ideological scores:** publisher ideology scores in `visualise.py` are hand-assigned based on AllSides and Ad Fontes Media ratings and should be treated as approximate.
- **Control week neutrality:** the control years (2024, 2016) were chosen to avoid major women's rights events, but cannot be fully neutral — 2016 overlaps with the US presidential election cycle.
- **`seaborn` dependency:** the `fairness_subgroup_intersection.png` heatmap requires `seaborn`. If not installed, that chart is silently skipped and all others still generate.

Note: The .gitignore contains several files excluded from this repository as the model files exceed standard Git file size limits.

## Repository Exclusions (.gitignore)

The following files and directories are excluded from version control to protect sensitive keys, preserve local environment configurations, and keep the repository lightweight:

- **analysis_output/bertopic_model/** – Contains the heavy, trained global BERTopic model weights and embeddings. These files exceed standard Git file size limits.  
- **.env** – Stores local environment variables and API credentials.
- **venv-agent/ & bin/** – Local Python virtual environments and executive binaries.
- **.DS_Store** – Metadata files automatically generated by macOS.
- **output & output** – Generic local output caches. 

Note on analysis_output/: While the main analysis_output/ folder and its underlying subdirectories/CSVs are tracked to preserve reproducibility, the specific bertopic_model/ binary directory remains strictly ignored.  