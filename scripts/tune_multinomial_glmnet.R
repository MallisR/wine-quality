#!/usr/bin/env Rscript

# Purpose:
# - Benchmark/tune multinomial glmnet models for predicting `quality`, comparing:
#   - main effects vs all pairwise interactions (~(.)^2)
#   - ridge (alpha=0), lasso (alpha=1), elastic net (alpha=0.5)
#
# Inputs:
# - `train.csv` (semicolon-delimited)
#
# Outputs (written to `models/`):
# - `multinomial_glmnet_tuning_results.csv` (leaderboard)
# - `best_multinomial_glmnet_model.rds` (best cv.glmnet object + metadata)
#
# Run:
# - Rscript scripts/tune_multinomial_glmnet.R [train_csv_path]

suppressWarnings(suppressMessages({
  args <- commandArgs(trailingOnly = TRUE)
}))

default_input <- "train.csv"
default_out_dir <- "models"
default_results_path <- file.path(default_out_dir, "multinomial_glmnet_tuning_results.csv")
default_best_model_path <- file.path(default_out_dir, "best_multinomial_glmnet_model.rds")

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
    "Usage: Rscript scripts/tune_multinomial_glmnet.R [train_csv_path]",
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
    accuracy = mean(predicted == actual),
    macro_precision = mean(precision),
    macro_recall = mean(recall),
    macro_f1 = mean(f1)
  )
}

fit_and_score <- function(model_name, x_train, x_test, y_train, y_test, alpha_value) {
  set.seed(42)
  cv_fit <- glmnet::cv.glmnet(
    x = x_train,
    y = y_train,
    family = "multinomial",
    alpha = alpha_value,
    type.measure = "class"
  )
  pred <- predict(cv_fit$glmnet.fit, newx = x_test, s = cv_fit$lambda.min, type = "class")
  pred <- factor(as.vector(pred), levels = levels(y_test))
  metrics <- metric_frame(y_test, pred)

  list(
    model_name = model_name,
    alpha = alpha_value,
    lambda_min = cv_fit$lambda.min,
    metrics = metrics,
    cv_fit = cv_fit
  )
}

train_features <- train_df[, feature_cols]
test_features <- test_df[, feature_cols]
y_train <- train_df$quality
y_test <- test_df$quality

# Main effects design matrix
x_train_main <- model.matrix(~ ., data = train_features)[, -1, drop = FALSE]
x_test_main <- model.matrix(~ ., data = test_features)[, -1, drop = FALSE]

# Pairwise interaction design matrix (includes main effects + interactions)
x_train_inter <- model.matrix(~ (. )^2, data = train_features)[, -1, drop = FALSE]
x_test_inter <- model.matrix(~ (. )^2, data = test_features)[, -1, drop = FALSE]

candidate_specs <- list(
  list(name = "lasso_main", alpha = 1, x_train = x_train_main, x_test = x_test_main),
  list(name = "ridge_main", alpha = 0, x_train = x_train_main, x_test = x_test_main),
  list(name = "elastic_main_alpha_0.5", alpha = 0.5, x_train = x_train_main, x_test = x_test_main),
  list(name = "lasso_interactions", alpha = 1, x_train = x_train_inter, x_test = x_test_inter),
  list(name = "ridge_interactions", alpha = 0, x_train = x_train_inter, x_test = x_test_inter),
  list(name = "elastic_interactions_alpha_0.5", alpha = 0.5, x_train = x_train_inter, x_test = x_test_inter)
)

fits <- lapply(
  candidate_specs,
  function(spec) {
    fit_and_score(
      model_name = spec$name,
      x_train = spec$x_train,
      x_test = spec$x_test,
      y_train = y_train,
      y_test = y_test,
      alpha_value = spec$alpha
    )
  }
)

results <- do.call(
  rbind,
  lapply(
    fits,
    function(f) {
      cbind(
        data.frame(
          model_name = f$model_name,
          alpha = f$alpha,
          lambda_min = f$lambda_min
        ),
        f$metrics
      )
    }
  )
)

results <- results[order(-results$accuracy, -results$macro_f1), ]
best_name <- results$model_name[1]
best_fit <- fits[[which(vapply(fits, function(f) f$model_name == best_name, logical(1)))[1]]]

if (!dir.exists(default_out_dir)) {
  dir.create(default_out_dir, recursive = TRUE)
}

write.csv(results, file = default_results_path, row.names = FALSE)
saveRDS(
  list(
    best_model_name = best_fit$model_name,
    alpha = best_fit$alpha,
    lambda_min = best_fit$lambda_min,
    cv_fit = best_fit$cv_fit,
    feature_set = if (grepl("interactions", best_fit$model_name)) "interactions" else "main_effects"
  ),
  file = default_best_model_path
)

cat(sprintf("Rows used: %d (train=%d, test=%d)\n", nrow(df), nrow(train_df), nrow(test_df)))
cat(sprintf("Saved tuning results: %s\n", default_results_path))
cat(sprintf("Saved best model: %s\n", default_best_model_path))
cat("Leaderboard (sorted by accuracy then macro F1):\n")
print(results)
