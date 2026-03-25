#!/usr/bin/env Rscript

# Purpose:
# - Train a multinomial glmnet model for `quality` that is more robust to outliers by:
#   - winsorizing numeric predictors (train-derived quantile caps)
#   - robust scaling (median/MAD computed on train)
# - Evaluates on the same 80/20 hold-out split (set.seed(42)).
#
# Inputs:
# - `train.csv` (semicolon-delimited)
#
# Outputs (written to `models/`):
# - `robust_glmnet_model.rds` (glmnet fit + preprocessing params)
# - `robust_glmnet_metrics.csv`
# - `robust_glmnet_predictions.csv`
#
# Run:
# - Rscript scripts/train_outlier_robust_glmnet.R [train_csv_path]

suppressWarnings(suppressMessages({
  args <- commandArgs(trailingOnly = TRUE)
}))

default_input <- "train.csv"
default_out_dir <- "models"
default_model_path <- file.path(default_out_dir, "robust_glmnet_model.rds")
default_metrics_path <- file.path(default_out_dir, "robust_glmnet_metrics.csv")
default_pred_path <- file.path(default_out_dir, "robust_glmnet_predictions.csv")

if (!requireNamespace("glmnet", quietly = TRUE)) {
  stop(
    "Package 'glmnet' is required but not installed. Install with: install.packages('glmnet')",
    call. = FALSE
  )
}

if (length(args) == 0) {
  input_path <- default_input
} else if (length(args) == 1) {
  input_path <- args[[1]]
} else {
  stop(
    "Usage: Rscript scripts/train_outlier_robust_glmnet.R [train_csv_path]",
    call. = FALSE
  )
}

if (!file.exists(input_path)) {
  stop(sprintf("Input file not found: %s", input_path), call. = FALSE)
}

df <- read.csv(input_path, sep = ";", header = TRUE, check.names = FALSE)

required_cols <- c(
  "fixed.acidity", "volatile.acidity", "citric.acid", "residual.sugar",
  "chlorides", "free.sulfur.dioxide", "total.sulfur.dioxide", "density",
  "pH", "sulphates", "alcohol", "quality", "is_red"
)

missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(
    sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")),
    call. = FALSE
  )
}

feature_cols <- setdiff(required_cols, "quality")
df <- df[complete.cases(df[, required_cols]), required_cols]
df$quality <- as.factor(df$quality)

set.seed(42)
idx <- sample.int(nrow(df), size = floor(0.8 * nrow(df)))
train_df <- df[idx, ]
test_df <- df[-idx, ]

metric_frame <- function(actual, predicted) {
  labels <- levels(actual)
  cm <- table(
    actual = factor(actual, levels = labels),
    predicted = factor(predicted, levels = labels)
  )

  precision <- numeric(length(labels))
  recall <- numeric(length(labels))
  f1 <- numeric(length(labels))

  for (i in seq_along(labels)) {
    tp <- cm[i, i]
    fp <- sum(cm[, i]) - tp
    fn <- sum(cm[i, ]) - tp
    p <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    r <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    precision[i] <- p
    recall[i] <- r
    f1[i] <- if ((p + r) == 0) 0 else (2 * p * r) / (p + r)
  }

  data.frame(
    metric = c("accuracy", "macro_precision", "macro_recall", "macro_f1"),
    value = c(
      mean(predicted == actual),
      mean(precision),
      mean(recall),
      mean(f1)
    )
  )
}

winsorize_fit <- function(train_mat, probs = c(0.01, 0.99)) {
  lo <- apply(train_mat, 2, quantile, probs = probs[[1]], na.rm = TRUE)
  hi <- apply(train_mat, 2, quantile, probs = probs[[2]], na.rm = TRUE)
  list(lo = lo, hi = hi, probs = probs)
}

winsorize_apply <- function(mat, fit) {
  out <- mat
  for (j in seq_len(ncol(out))) {
    out[, j] <- pmin(pmax(out[, j], fit$lo[[j]]), fit$hi[[j]])
  }
  out
}

robust_scale_fit <- function(train_mat) {
  center <- apply(train_mat, 2, median, na.rm = TRUE)
  scale <- apply(train_mat, 2, mad, na.rm = TRUE)
  scale[scale == 0] <- 1
  list(center = center, scale = scale)
}

robust_scale_apply <- function(mat, fit) {
  sweep(sweep(mat, 2, fit$center, "-"), 2, fit$scale, "/")
}

train_x_raw <- as.matrix(train_df[, feature_cols])
test_x_raw <- as.matrix(test_df[, feature_cols])

win_fit <- winsorize_fit(train_x_raw, probs = c(0.01, 0.99))
train_x_win <- winsorize_apply(train_x_raw, win_fit)
test_x_win <- winsorize_apply(test_x_raw, win_fit)

scale_fit <- robust_scale_fit(train_x_win)
train_x <- robust_scale_apply(train_x_win, scale_fit)
test_x <- robust_scale_apply(test_x_win, scale_fit)

y_train <- train_df$quality
y_test <- test_df$quality

# Elastic net with interactions performed best earlier; keep it as the robust variant too.
train_features <- as.data.frame(train_x)
test_features <- as.data.frame(test_x)
names(train_features) <- feature_cols
names(test_features) <- feature_cols

x_train <- model.matrix(~ (. )^2, data = train_features)[, -1, drop = FALSE]
x_test <- model.matrix(~ (. )^2, data = test_features)[, -1, drop = FALSE]

min_class_count <- min(as.numeric(table(y_train)))
nfolds <- max(3, min(5, min_class_count))

set.seed(42)
cv_fit <- glmnet::cv.glmnet(
  x = x_train,
  y = y_train,
  family = "multinomial",
  alpha = 0.5,
  type.measure = "class",
  nfolds = nfolds
)

pred <- predict(cv_fit$glmnet.fit, newx = x_test, s = cv_fit$lambda.min, type = "class")
pred <- factor(as.vector(pred), levels = levels(y_test))

metrics <- metric_frame(y_test, pred)
predictions <- data.frame(
  actual_quality = as.character(y_test),
  predicted_quality = as.character(pred)
)

if (!dir.exists(default_out_dir)) {
  dir.create(default_out_dir, recursive = TRUE)
}

saveRDS(
  list(
    cv_fit = cv_fit,
    alpha = 0.5,
    lambda_min = cv_fit$lambda.min,
    winsor = win_fit,
    robust_scale = scale_fit,
    feature_cols = feature_cols,
    design = "pairwise_interactions",
    seed = 42
  ),
  file = default_model_path
)
write.csv(metrics, file = default_metrics_path, row.names = FALSE)
write.csv(predictions, file = default_pred_path, row.names = FALSE)

cat(sprintf("Rows used: %d (train=%d, test=%d)\n", nrow(df), nrow(train_df), nrow(test_df)))
cat(sprintf("Saved model: %s\n", default_model_path))
cat(sprintf("Saved metrics: %s\n", default_metrics_path))
cat(sprintf("Saved predictions: %s\n", default_pred_path))
cat("Metrics:\n")
print(metrics)

