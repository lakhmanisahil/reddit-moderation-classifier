# Reddit Post Analysis & Moderation Classifier

![CI](https://github.com/<your-username>/reddit-moderation-classifier/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Reddit gets huge volumes of posts, and some of them are spam, toxic or low-effort. Manual moderation is slow, inconsistent and doesn't scale.
This project analyses what drives deletions and flags, then builds two automated moderators and compares them **on the same posts with the same metrics**:

1. **EDA:** pandas, Pearson correlations (with p-values), plots and word clouds of deleted vs kept posts.
2. **Classical NLP:** tokenization, stop-word removal, stemming and **TF-IDF**, then **Decision Tree, SVM and Random Forest**, tuned with grid search and F1 threshold tuning, to predict whether a post gets deleted.
3. **LLM moderation:** **Gemini** with a **response schema and enums**. Every answer is strict JSON, `{"category": SAFE|SPAM|TOXIC|LOW_EFFORT|MISINFORMATION, "action": KEEP|DELETE}`. Three prompting strategies are compared: zero-shot, visible chain-of-thought, and internal reasoning (thinking) with a JSON-only output.

## Results

> Replace with **your** numbers from `reports/RESULTS.md`. That file is rewritten on every run.

| Model | Deletion F1 | Deletion accuracy | Category accuracy |
|---|---|---|---|
| Random baseline | 0.32 | 0.59 | n/a |
| Decision Tree | 0.66 | 0.77 | n/a |
| **Linear SVM** (best classical) | **0.70** | 0.80 | n/a |
| Random Forest | 0.69 | 0.80 | n/a |
| Gemini, zero-shot | _run_ | _run_ | _run_ |
| Gemini, CoT (visible reasoning) | _run_ | _run_ | _run_ |
| Gemini, CoT (internal reasoning, JSON only) | _run_ | _run_ | _run_ |

Every structured output parsed successfully, with no invalid JSON or out-of-enum labels (`invalid_outputs = 0`).

<p align="center">
  <img src="reports/figures/model_comparison.png" width="85%"><br>
  <img src="reports/figures/wordclouds.png" width="85%">
</p>

## How it works

```
 Raw posts (title, body, score, upvote ratio, reports, flagged?, deleted?)
        │
        ▼  clean: fill empty bodies · blank "[removed]" placeholders · drop duplicate ids
        │
        ├──▶ EDA ─▶ Pearson r + p-values · deletion rate by subreddit/category · distributions · word clouds
        │
        ▼  tokenize → stop words (keep negations) → Porter stemming → TF-IDF (1–2-grams)
        │
        ├──▶ Decision Tree ┐  GridSearchCV (5-fold, F1)
        ├──▶ Linear SVM    ├─ + decision-threshold tuning (F1)
        └──▶ Random Forest ┘  → test set (once)
        
 Same unseen test posts ─▶ Gemini · response schema + enums · 3 prompt strategies
                         ─▶ head-to-head table: deletion F1/accuracy (all) + category accuracy (Gemini)
```

### Design decisions

- **F1, not accuracy, for deletion.** Only about 30% of posts are deleted, so always answering "kept" already scores about 70% accuracy.
- **Text-only features.** Votes and comments correlate with deletion, but partly *because* the post was deleted. Using them would leak the answer.
- **TF-IDF inside the pipeline.** The vocabulary is learned from the training folds only.
- **Category and action are separate.** `DELETE` is a decision, not a kind of content. Keeping the `action` field separate lets the LLM be scored on exactly the same `is_deleted` target as the classical models.
- **Structured outputs.** The JSON schema plus enums guarantee machine-readable, valid labels. Consistency comes from the schema. With Gemini 3, temperature stays at its default, because Google warns that lowering it can degrade reasoning.
- **Internal vs visible reasoning.** In `cot_visible`, the reasoning is written into the JSON *before* the label. In `cot_internal`, the model reasons in hidden thinking tokens and returns only `{category, action}`. That is what "reason internally, output only JSON" actually requires. Telling a non-thinking model to "reason internally" does nothing, because a model can only reason through tokens it generates.
- **Cache every LLM call.** Each answer is appended to a JSONL cache, so re-runs are free and a crashed run resumes where it stopped.

## Quick start

```bash
git clone https://github.com/<your-username>/reddit-moderation-classifier.git
cd reddit-moderation-classifier
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/make_synthetic_data.py   # or a real dataset, see Data below
python run_eda.py                       # charts → reports/figures/
python train.py                         # 3 models, about 3–5 minutes
cp .env.example .env                    # paste your key from aistudio.google.com/apikey
python run_gemini.py --dry-run          # test the pipeline without spending quota
python run_gemini.py                    # Gemini vs classical models on 150 test posts
```

## Data

| Column | Meaning |
|---|---|
| `title`, `selftext` | post title and body |
| `subreddit`, `author`, `created_utc` | where, who, when |
| `score`, `upvote_ratio`, `num_comments` | the post's rating and engagement |
| `num_reports`, `is_flagged` | user reports / AutoModerator flags |
| `author_karma`, `account_age_days` | author history |
| `is_deleted` | **target:** removed by moderators (1) or not (0) |
| `category` | ground-truth moderation category (used to score Gemini) |

**Synthetic data (default).** `scripts/make_synthetic_data.py` generates 8,000 posts with realistic patterns and deliberate noise.

**Real data.** Use [HumanMOD](https://huggingface.co/datasets/SamW/HumanMOD) (8.5k moderated and 8.5k unmoderated posts) and run `python scripts/prepare_real_data.py <file>`.

## Project structure

```
├── run_eda.py · train.py · run_gemini.py   # the three pipeline steps
├── src/
│   ├── config.py            # all paths and settings
│   ├── data_loader.py       # load + clean
│   ├── text_preprocessing.py# tokenize, stop words, stemming
│   ├── eda.py               # Pearson tables, plots, word clouds
│   ├── models.py            # TF-IDF pipelines, grid search, threshold tuning
│   ├── llm.py               # Gemini schemas, enums, prompts, client
│   └── report.py            # writes reports/RESULTS.md
├── scripts/                 # synthetic data generator, real-data converter
├── notebooks/01_walkthrough.ipynb
├── tests/                   # pytest, run by GitHub Actions on every push
└── docs/STEP_BY_STEP.md     # how it was built, why each step
```

## Limitations and next steps

- Subreddit rules differ. Next step: train per-subreddit models or add the subreddit as a feature.
- Some removals have reasons that aren't in the text (wrong flair, reposts), which caps every text-only model.
- Calling an LLM on every post is slow and costs money at scale. Next step: a cascade where the SVM handles confident cases and Gemini handles uncertain ones and explains its decision.

## Author

**Your Name**: [LinkedIn](https://linkedin.com/in/<your-profile>) · [GitHub](https://github.com/<your-username>)
