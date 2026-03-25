# Wine Quality (Multiclass Classification + Tests)

This repo builds and evaluates models to **predict wine `quality`** from physicochemical measurements, and runs a basic multivariate hypothesis test comparing acidity profiles between red and white wines.

All modeling scripts use the same dataset format and (unless otherwise noted) the same **train/test split**:
- **Split**: 80/20 hold-out
- **Seed**: `set.seed(42)`
- **Metric**: **accuracy** = fraction of test rows with exactly-correct `quality` class

## Data

### Input dataset used for models
- **File**: `train.csv`
- **Delimiter**: `;`
- **Target**: `quality` (treated as a factor for classification)
- **Wine type indicator**: `is_red` (`1` = red, `0` = white)
- **Predictors**: the remaining physicochemical columns (acidity, sugar, sulfur, etc.)

### Raw sources (optional)
- `winequality-red.csv`
- `winequality-white.csv`

### Generated / derived datasets
- `data/processed/winequality_combined.csv` (created by `scripts/combine_wine_datasets.R`)

## Scripts (what each one does)

All scripts are runnable from the project root with `Rscript ...`.

### `scripts/combine_wine_datasets.R`
Combines `winequality-red.csv` and `winequality-white.csv` into a single dataset and adds `is_red`.

- **Inputs**:
  - default: `winequality-red.csv`, `winequality-white.csv`
  - fallback: if raw sources are missing, it can reuse `train.csv` (must contain `is_red`)
- **Output**: `data/processed/winequality_combined.csv`
- **Run**:

```bash
Rscript scripts/combine_wine_datasets.R
```

### `scripts/train_multinomial_logit.R`
Trains and evaluates three **multinomial** classification models for `quality`:
- **Multinomial logistic regression** (`nnet::multinom`)
- **Ridge** multinomial regression (`glmnet`, `alpha = 0`)
- **Lasso** multinomial regression (`glmnet`, `alpha = 1`)

- **Input**: `train.csv`
- **Outputs** (written to `models/`):
  - `multinomial_logit_model.rds`, `multinomial_logit_metrics.csv`, `multinomial_logit_predictions.csv`
  - `multinomial_ridge_model.rds`, `multinomial_ridge_metrics.csv`, `multinomial_ridge_predictions.csv`
  - `multinomial_lasso_model.rds`, `multinomial_lasso_metrics.csv`, `multinomial_lasso_predictions.csv`
- **Run**:

```bash
Rscript scripts/train_multinomial_logit.R
```

### `scripts/tune_multinomial_glmnet.R`
Runs a small tuning/benchmark for **glmnet multinomial** models comparing:
- main effects vs **all pairwise interactions** (`model.matrix(~(.)^2, ...)`)
- ridge vs lasso vs elastic net (includes `alpha = 0`, `0.5`, `1`)

- **Input**: `train.csv`
- **Outputs**:
  - `models/multinomial_glmnet_tuning_results.csv` (leaderboard)
  - `models/best_multinomial_glmnet_model.rds`
- **Run**:

```bash
Rscript scripts/tune_multinomial_glmnet.R
```

### `scripts/quick_improvements.R`
Fast “what helps accuracy?” benchmark that stays close to the original evaluation approach, but tries:
- a small **alpha grid** for glmnet
- **targeted interaction terms** (a small hand-picked set, not all pairwise)
- an **ordinal logistic regression** baseline (`MASS::polr`) as a comparison model

- **Input**: `train.csv`
- **Output**: `models/quick_improvements_results.csv`
- **Run**:

```bash
Rscript scripts/quick_improvements.R
```

### `scripts/run_manova_acidity.R`
Runs a **MANOVA** to test whether wine type (red vs white) differs across the multivariate set of acidity variables found in the dataset (columns containing `"acidity"`).

- **Input**: `train.csv`
- **Outputs**:
  - `models/manova_acidity_test.csv` (Pillai + Wilks stats)
  - `models/manova_acidity_summary.txt` (full printed summary)
- **Run**:

```bash
Rscript scripts/run_manova_acidity.R
```

### `scripts/advanced_multinomial_pipeline.R` (experimental)
More extensive experimentation script (feature engineering, class weighting, outlier identification, repeated splits). It exists for exploration and writes outputs to `models/`, but it can take longer to run than the other scripts.

## Reproducing the full set of results

```bash
Rscript scripts/train_multinomial_logit.R
Rscript scripts/tune_multinomial_glmnet.R
Rscript scripts/quick_improvements.R
Rscript scripts/run_manova_acidity.R
```

## Outputs

Most outputs are written to `models/` as:
- `*_metrics.csv`: accuracy + macro metrics (where applicable)
- `*_predictions.csv`: row-level actual vs predicted classes
- `*.rds`: saved R model objects

## Notes on “accuracy”

All reported accuracies in the modeling scripts are **hold-out test accuracies** on the 20% test split:
\[
\text{accuracy}=\frac{\#\{\text{predicted quality}=\text{actual quality}\}}{\#\{\text{test rows}\}}
\]

## Keeping this README updated

As we add or change scripts/models, I’ll keep this README in sync by updating:
- the script list (purpose + inputs/outputs)
- the recommended “run order”
- any changes to evaluation metrics or dataset requirements