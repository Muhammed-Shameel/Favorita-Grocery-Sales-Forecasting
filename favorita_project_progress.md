# Favorita Ecuador Sales Forecasting — Project Progress

## 1. Project Goal

Build an end-to-end sales forecasting system for the Favorita Ecuador dataset, with:
- data validation and feature engineering
- time-aware model evaluation
- recursive multi-step forecasting
- model comparison
- eventual production prediction/dashboard pipeline

The main dataframe is `df`.

---

## 2. Dataset

Current engineered dataframe:

- Shape: 3,000,888 rows × 53 columns
- Training period: 2013-01-01 → 2017-08-15
- Competition test period: 2017-08-16 → 2017-08-31
- Historical validation period: 2017-07-01 → 2017-07-16
- Validation rows: 28,512

Important original tables:
- train: 3,000,888 × 6
- stores: 54 × 5
- oil: 1,218 × 2
- holidays_events: 350 × 6
- transactions: 83,488 × 3
- test: 28,512 × 5

Oil had 43 missing values. These were filled using forward-fill followed by backward-fill.

---

## 3. Current Engineered Features

Current dataframe columns:

```text
['id', 'date', 'store_nbr', 'family', 'sales', 'onpromotion',
 'dcoilwtico', 'event_count', 'has_holiday', 'has_event',
 'has_additional', 'has_transfer', 'has_bridge', 'has_workday',
 'is_national', 'is_regional', 'is_local', 'city', 'state', 'type',
 'cluster', 'transactions', 'year', 'month', 'day', 'day_of_week',
 'week_of_year', 'is_weekend', 'is_month_start', 'is_month_end',
 'is_payday', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
 'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_21',
 'sales_lag_28', 'sales_lag_35', 'sales_lag_56', 'sales_mean_7',
 'sales_std_7', 'sales_mean_14', 'sales_std_14', 'sales_mean_28',
 'sales_std_28', 'zero_sales_pct_28', 'sales_trend_7_28',
 'has_promotion', 'promotion_days_7', 'promotion_days_28']
```

### Calendar features
- year
- month
- day
- day_of_week
- week_of_year
- is_weekend
- is_month_start
- is_month_end
- is_payday
- month_sin / month_cos
- weekday_sin / weekday_cos

### Sales history features
- lags: 1, 7, 14, 21, 28, 35, 56
- rolling mean/std: 7, 14, 28 days
- zero-sales percentage over 28 days
- 7-day vs 28-day sales trend

### Promotion features
- has_promotion
- promotion_days_7
- promotion_days_28

### Event/store features
- event_count
- holiday/event/additional/transfer/bridge/workday flags
- national/regional/local flags
- city, state, type, cluster
- transactions
- oil price

---

## 4. Feature Engineering Implementation

The project uses:

`src/features/build_features.py`

Main functions:

```python
create_calendar_features()
sales_lag_features()
promotion_features()
create_forecast_lag_features()
```

The recursive forecasting function keeps the latest 56 observations per `(store_nbr, family)` because the largest required lag/window is 56.

Important note:
`create_forecast_lag_features()` currently updates sales-derived lag/rolling features recursively. Promotion rolling features are carried from the forecast rows rather than recomputed recursively.

---

## 5. Memory Optimization

The original 3M+ row dataframe caused memory pressure.

The dataframe was optimized by:
- converting float64 values to float32 where appropriate
- downcasting integer columns
- converting repeated string columns such as `family`, `city`, `state`, and `type` to categorical dtype
- avoiding unnecessary full dataframe copies
- selecting only modelling columns when possible

This allowed CatBoost training on a 500,000-row sample.

---

## 6. Modelling Setup

Dropped from the model:

```python
DROP_COLS = [
    "id",
    "sales",
    "transactions",
    "date"
]
```

Categorical features:

```python
categorical_features = [
    "family",
    "city",
    "state",
    "type"
]
```

CatBoost handles categorical features natively, so no manual one-hot encoding was used.

CatBoost documentation recommends native categorical handling rather than preprocessing categorical variables with one-hot encoding. citeturn0search0

CatBoost also supports supplying a validation dataset through `eval_set`, which is used for monitoring and best-iteration selection. citeturn0search1

---

## 7. LightGBM Results

Several LightGBM models were tested.

Best relevant results:

### Direct validation
- MAE ≈ 76–80 depending on configuration
- RMSE ≈ 255–267

### Recursive evaluation
Best LightGBM configuration:

- MAE: 80.25
- RMSE: 354.10
- Bias: -53.00

Another recursive LightGBM result:

- MAE: 80.84
- RMSE: 360.37
- Bias: -53.09

LightGBM therefore became the previous benchmark.

---

## 8. CatBoost Baseline

CatBoost was selected as the next model because it can naturally handle the categorical store/product information.

Baseline configuration:

```python
CatBoostRegressor(
    iterations=600,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    eval_metric="MAE",
    random_seed=42,
    verbose=100,
    allow_writing_files=False,
    thread_count=4
)
```

Training used a 500,000-row random sample from the training period.

The full 53-feature model achieved direct validation:

- MAE: 56.6753
- Best iteration: 593

A reduced 48-feature version was tested by removing:

```text
has_workday
has_bridge
is_payday
is_local
is_regional
```

The reduced model was slightly worse:

- MAE: 56.7725

Therefore the full 53-feature feature set was retained.

---

## 9. CatBoost Recursive Evaluation

The important evaluation is recursive because future predictions should use previous predictions rather than actual future sales.

Recursive evaluation over:

**2017-07-01 → 2017-07-16**

produced:

- Rows: 28,512
- MAE: 65.2177
- RMSE: 230.0473
- Bias: +14.3121

This is currently the project champion.

### Comparison

| Model | Recursive MAE | Recursive RMSE | Bias |
|---|---:|---:|---:|
| LightGBM | 80.25 | 354.10 | -53.00 |
| CatBoost | **65.22** | **230.05** | **+14.31** |

Approximate improvement of CatBoost over LightGBM:

- MAE: ~19% lower
- RMSE: ~35% lower
- Bias moved much closer to zero

Therefore the current champion is:

**CatBoost — Recursive MAE 65.22 / RMSE 230.05**

---

## 10. Feature Importance Observation

The weakest features in the first CatBoost model included:

```text
has_additional       0.145871
year                 0.140874
city                 0.117361
zero_sales_pct_28    0.100364
promotion_days_28    0.088386
has_transfer         0.072558
is_month_end         0.063713
has_promotion        0.043963
state                0.043860
promotion_days_7     0.038644
is_regional          0.018294
is_local             0.011380
is_payday            0.003162
has_bridge           0.001482
has_workday          0.000000
```

Removing the five weakest features did not improve validation, so they were retained.

CatBoost also provides built-in feature selection methods if more formal feature selection is needed later. citeturn0search3

---

## 11. Current Next Step — CatBoost Hyperparameter Tuning

The next task is controlled hyperparameter tuning.

We are NOT doing a huge automated search because each full model takes several minutes and the dataset is large.

Current candidates:

### A — Baseline
```text
depth = 8
learning_rate = 0.05
iterations = 700
l2_leaf_reg = 3
random_strength = 1.0
bagging_temperature = 1.0
```

### B — Deeper
```text
depth = 9
learning_rate = 0.04
iterations = 850
l2_leaf_reg = 5
random_strength = 1.0
bagging_temperature = 1.0
```

### C — Shallower
```text
depth = 7
learning_rate = 0.04
iterations = 900
l2_leaf_reg = 5
random_strength = 0.8
bagging_temperature = 1.0
```

### D — More regularization
```text
depth = 8
learning_rate = 0.04
iterations = 900
l2_leaf_reg = 10
random_strength = 1.0
bagging_temperature = 1.0
```

All candidates should use:
- the same 500,000-row training sample
- the same July 1–16 validation period
- the same categorical features
- the same random seed
- thread_count=4

The tuning screen will identify the strongest candidate using direct validation first.

After that, the best 1–2 candidates must be evaluated with the existing recursive forecasting pipeline.

The recursive score remains the final model-selection metric.

---

## 12. Important Evaluation Rule

Do NOT choose the final model solely from direct validation.

Direct validation tells us how well the model predicts a known validation block using historical features.

Recursive evaluation is more realistic because predictions for later days depend on previous predictions.

Therefore:

```text
Direct validation
        ↓
Screen CatBoost configurations
        ↓
Select top 1–2
        ↓
Recursive July evaluation
        ↓
Choose final champion
        ↓
Retrain / forecast competition test
```

---

## 13. Current Project Status

Completed:

- [x] Load Favorita datasets
- [x] Validate data
- [x] Merge tables
- [x] Handle oil missing values
- [x] Calendar features
- [x] Event features
- [x] Store metadata
- [x] Sales lag features
- [x] Rolling statistics
- [x] Promotion features
- [x] Memory optimization
- [x] LightGBM baseline
- [x] LightGBM recursive evaluation
- [x] CatBoost baseline
- [x] CatBoost feature comparison
- [x] CatBoost recursive evaluation
- [x] Establish current champion

Current stage:

**CatBoost hyperparameter tuning → recursive evaluation → final forecasting**

---

## 14. Current Champion

```text
MODEL: CatBoost

Recursive validation:
MAE  = 65.2177
RMSE = 230.0473
Bias = +14.3121

Validation period:
2017-07-01 → 2017-07-16

Rows:
28,512
```

This is the benchmark that every new model must beat.
