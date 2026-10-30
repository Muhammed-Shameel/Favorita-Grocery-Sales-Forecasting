# Retail Demand Intelligence — Favorita Grocery Sales Forecasting

An end-to-end time-series / tabular machine-learning project that forecasts **daily grocery
sales for 54 Corporación Favorita stores across 3 product-family panels**, using a modular
data pipeline, leakage-aware feature engineering, a **CatBoost** regressor, and a **recursive
16-day-ahead forecasting** strategy validated against an honest time-based holdout.

> **Scope note:** this is a portfolio / learning project built on the publicly circulated
> Favorita grocery-sales dataset. It is **not** an official Kaggle competition submission and
> no leaderboard score is claimed. See [Honesty & Limitations](#honesty--limitations).

---

## 1. The Problem

Grocery retailers must decide how much stock to order for every product category in every
store, days before the shelves are empty. Under-forecasting causes stockouts and lost revenue;
over-forecasting causes waste (especially for perishables).

This project treats that as a **multi-series, multi-step forecasting** problem:

- **Grain:** one row per `(date, store_nbr, family)`.
- **Target:** `sales` (units sold per day).
- **Horizon:** forecast the next **16 days** recursively — each day's prediction feeds the next
  day's lag features, the way a real forecast would be produced in production.
- **Training window:** `2013-01-01 → 2017-08-15`
- **Forecast window (local):** `2017-08-16 → 2017-08-31` → **28,512 rows**

---

## 2. Highlights

- **Leakage-aware feature engineering** — every sales lag / rolling statistic is built on
  `shift(1)` history so the current-day target is never used as an input.
- **Direct vs. recursive evaluation** — the project deliberately reports the *recursive*
  score (the realistic one) instead of the flattering *direct* score, and explains why.
- **Recursive multi-step forecasting** — predictions are written back into the rolling history
  and the history buffer is trimmed to the longest required lag for memory efficiency.
- **Controlled model tuning** — an 8-configuration CatBoost screen with early stopping plus a
  feature-importance ablation study; the champion is selected by **recursive MAE**.
- **Modular, reusable code** — ingestion, cleaning, validation, and feature engineering live in
  an importable `src/` package; notebooks orchestrate the pipeline.
- **Reproducible artifacts** — the final CatBoost model and its feature list are persisted and
  loaded by a dedicated inference notebook that regenerates the forecast from scratch.

---

## 3. Dataset

The pipeline consumes the standard Favorita tables (placed in `data/raw/`, **not committed** —
see [Setup](#7-setup--how-to-run)):

| Table | Shape | Description |
|---|---|---|
| `train.csv` | 3,000,888 × 6 | Historical `sales` & `onpromotion` per store/family/day |
| `test.csv` | 28,512 × 5 | 16-day forecast panel (known covariates, no `sales`) |
| `stores.csv` | 54 × 5 | Store metadata: `city`, `state`, `type`, `cluster` |
| `transactions.csv` | 83,488 × 3 | Daily store footfall (merged for exploration; dropped before modeling) |
| `oil.csv` | 1,218 × 2 | `dcoilwtico` crude-oil price (Ecuador is a net oil exporter) |
| `holidays_events.csv` | 350 × 6 | National / regional / local holidays, events, transfers |

Key facts:

- **54 stores**, **33 product families**, daily grain, **~4.5 years** of history.
- The local `test.csv` panel (28,512 rows) is the **16-day horizon only**. The full official
  competition test set is much larger, so this project's output is a *local forecast*, not a
  competition submission.
- `oil` had 43 missing prices (weekends/holidays) → forward-filled, then a single bounded
  back-fill for the leading gap.

---

## 4. Project Structure

```text
retail-demand-intelligence/
├── data/
│   ├── raw/                 # source CSVs (git-ignored, provide your own)
│   └── processed/           # features.parquet (generated, git-ignored)
├── models/                  # saved model artifacts + feature/metadata JSON
│   ├── catboost_final.cbm
│   ├── catboost_final_features.json
│   ├── catboost_baseline.cbm
│   └── favorita_lightgbm_best.joblib   # early baseline (training code not included)
├── notebooks/
│   ├── 01_data_exploration.ipynb        # EDA + schema/quality validation
│   ├── 02_data_cleaning.ipynb           # cleaning walkthrough
│   ├── 03_exploratary_data_analysis.ipynb  # relationships / seasonality
│   ├── 04_feature_engineering.ipynb     # FE + training + tuning + recursive evaluation
│   ├── 05_model_training.ipynb          # loads engineered features (stub/loader)
│   └── 06_testing.ipynb                 # inference: loads final model, recursive forecast
├── outputs/
│   └── submission_catboost_final.csv    # generated 28,512-row forecast (git-ignored)
├── src/
│   ├── ingestion/           # CSV loading
│   ├── cleaning/            # date parsing, oil fill, promo cap, holiday agg, master merge
│   ├── validations/         # schema / missing / duplicate / business-key / coverage checks
│   └── features/            # calendar, lag/rolling, promotion, recursive forecast features
├── requirements.txt
└── README.md
```

> Heavy data and model binaries are intentionally **excluded from Git** (`.gitignore`) because
> some files exceed GitHub's 100 MB limit. The directory structure is preserved with `.gitkeep`.

---

## 5. Pipeline & Methodology

### 5.1 Ingestion → Validation → Cleaning

1. **Ingestion** (`src/ingestion`) loads the raw CSVs.
2. **Validation** (`src/validations`) runs schema checks, missing-value and duplicate counts,
   business-key uniqueness (`date/store_nbr/family`), store referential integrity, and
   chronological train/test coverage.
3. **Cleaning** (`src/cleaning`):
   - Parse `date` → datetime across tables.
   - Oil: `ffill` + one bounded `bfill`.
   - Capping promotion outliers (`onpromotion` clipped to 169) to tame data-entry errors.
   - Holidays/events aggregated to a daily, national-level summary (`event_count`, type flags,
     locale flags).
   - Merge train/test with `oil`, `holidays_events`, and `stores`; attach `transactions`
     for the historical frame only.

### 5.2 Feature Engineering (`src/features`)

| Group | Features |
|---|---|
| Calendar | `year`, `month`, `day`, `day_of_week`, `week_of_year`, `is_weekend`, `is_month_start/end`, `is_payday`, cyclical `month_sin/cos`, `weekday_sin/cos` |
| Sales history (leakage-safe) | Lags `1, 7, 14, 21, 28, 35, 56`; rolling `mean/std` over `7/14/28`; `zero_sales_pct_28`; `sales_trend_7_28` — **all computed on `shift(1)` history** |
| Promotion | `has_promotion`, `promotion_days_7`, `promotion_days_28` (previous-day windows) |
| Exogenous / store | `dcoilwtico` (oil), holiday/event flags, `city`, `state`, `type`, `cluster` |

**Why `shift(1)` matters:** rolling/lag features use only days *before* the target date. The
naive implementation (rolling that includes the current day) leaks the label into the features
and inflates validation scores. `create_forecast_lag_features()` reproduces the same
historical-only logic at prediction time so training and inference stay consistent.

### 5.3 Modeling

- **Algorithm:** CatBoost regression with **native categorical handling** (no manual one-hot),
  which suits the high-cardinality store/family/location columns.
- **Loss / metric:** `RMSE` objective, `MAE` evaluation metric (MAE is the decision metric;
  sales errors are best read on the absolute-unit scale).
- **Categoricals:** `family`, `city`, `state`, `type`.
- **Training sample:** a fixed-seed **500,000-row** sample of the training window, chosen for
  memory. (See limitations.)
- **Final configuration:** `depth=9`, `learning_rate=0.04`, `iterations=850`, `l2_leaf_reg=5`.

### 5.4 Time-based Validation & Recursive Forecasting

- **Split:** train `date < 2017-07-01`, validate `2017-07-01 → 2017-07-16` (16 days, 28,512 rows) —
  a strict chronological holdout, never shuffled.
- **Direct evaluation:** predict the block using actual historical lags. Optimistic, because
  late-window days would not really have real prior-day sales in production.
- **Recursive evaluation (the honest one):** loop day-by-day; after each day, feed the **model's
  own predictions** (not actuals) back into the lag/rolling history and re-forecast. This is what
  the deployed forecast actually does, so it's the number that matters.

```text
Direct validation  →  screen CatBoost configs
        →  select top candidate(s)
        →  recursive July evaluation  →  champion
        →  retrain / recursive forecast for the 16-day horizon
```

---

## 6. Results

**Primary (realistic) metric — recursive 16-day evaluation, 2017-07-01 → 2017-07-16 (28,512 rows):**

| Metric | Value |
|---|---|
| **MAE** | **63.34** |
| **RMSE** | 226.29 |
| **Bias** | +10.29 (slight over-prediction) |

**Direct vs. recursive for the same champion model** (illustrates why recursive is reported):

| Evaluation | MAE | Note |
|---|---|---|
| Direct (actual in-window lags) | ≈ 55 | optimistic — uses information unavailable at true forecast time |
| **Recursive (predicted lags)** | **≈ 63** | **honest — mirrors real 16-day forecasting** |

**Baselines / comparison:**
- An 8-configuration CatBoost hyperparameter screen (depth / learning-rate / regularization) plus
  a feature-importance ablation drove the final selection; removing the five weakest features made
  validation slightly worse, so the full feature set was retained.
- A **LightGBM** model was explored as an earlier benchmark. Its trained artifact is in `models/`,
  but its **training notebook is not included in this repository**, so those figures should be
  treated as historical context rather than a re-runnable comparison here.

**Final forecast:** the loaded `catboost_final.cbm` (850 trees, 49 features) produced all
**28,512** predictions for `2017-08-16 → 2017-08-31` — **non-negative** (clipped at 0) and
**no missing values** — saved to `outputs/submission_catboost_final.csv`.

---

## 7. Setup & How to Run

### Requirements
Python 3.10+. The pipeline needs these packages:

```bash
pip install pandas numpy catboost lightgbm scikit-learn pyarrow joblib
```

### Steps
1. **Create a virtual environment** (`.venv/`) and install the packages above.
2. **Add the raw data** to `data/raw/` (the Favorita tables: `train`, `test`, `stores`,
   `transactions`, `oil`, `holidays_events`). These files are git-ignored for size reasons.
3. **Run the notebooks in order:**

   | # | Notebook | What it does |
   |---|---|---|
   | 1 | `01_data_exploration` | EDA + schema/quality validation |
   | 2 | `02_data_cleaning` | Cleaning walkthrough |
   | 3 | `03_exploratary_data_analysis` | Seasonality & relationships |
   | 4 | `04_feature_engineering` | Build features → train → tune → recursive evaluation → fit & save final model |
   | 5 | `06_testing` | Load `catboost_final.cbm` → recursive 16-day forecast → write output CSV |

   The pipeline is reproducible from the CatBoost path (feature build → train → evaluate →
   forecast). `05_model_training.ipynb` is a lightweight loader for the engineered feature frame.

> Notebooks use paths relative to the project root; run them from within the repository so the
> `src/` package imports resolve.

---

## 8. Tech Stack

`Python` · `pandas` · `NumPy` · `CatBoost` · `scikit-learn` (metrics) · `LightGBM` (baseline
artifact) · `PyArrow` (Parquet) · `Jupyter`

---

## 9. Honesty & Limitations

Stated plainly so the results are interpreted correctly:

- **Subsampled training.** The final model is trained on a **500k-row** sample (of ~2.9M rows)
  for memory, not the full training set. A full-data retrain is the natural next step.
- **Validation-window reuse.** The July 1–16 window is used for early stopping, model selection,
  **and** reporting, so the headline MAE is mildly selection-optimistic. The Aug 16–31 forecast has
  no local ground truth and is therefore **unevaluated**.
- **Not a Kaggle submission.** The 28,512-row output is the local 16-day panel, not the official
  competition test set; **no leaderboard rank or score is claimed**.
- **LightGBM baseline not reproducible here** (code not included).
- **No deployment / business-impact claims.** This is a modeling and pipeline project — there is
  no serving API, monitoring, or measured revenue impact.
- **Known feature inconsistency (flagged).** Rolling standard deviation is computed with `ddof=1`
  during training (pandas) but `ddof=0` at forecast time (NumPy); the effect on the reported
  metric is small but real and documented here rather than hidden.

---

## 10. Possible Next Steps

- Retrain the final model on the full training window.
- Add a truly held-out evaluation fold (separate from the tuning window).
- Unify the recursive loop and promotion features into `src/` to remove notebook duplication.
- Add a lightweight `run_pipeline.py` entry point and pin exact dependency versions.

---

*Built as an end-to-end learning/portfolio project demonstrating time-series feature engineering,
leakage-aware validation, recursive forecasting, and reproducible ML pipelines.*
