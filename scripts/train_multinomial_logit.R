#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  args <- commandArgs(trailingOnly = TRUE)
}))

default_input <- "train.csv"
default_model_path <- file.path("models", "multinomial_logit_model.rds")
default_metrics_path <- file.path("models", "multinomial_logit_metrics.csv")
default_pred_path <- file.path("models", "multinomial_logit_predictions.csv")
default_ridge_model_path <- file.path("models", "multinomial_ridge_model.rds")
default_ridge_metrics_path <- file.path("models", "multinomial_ridge_metrics.csv")
default_ridge_pred_path <- file.path("models", "multinomial_ridge_predictions.csv")
default_lasso_model_path <- file.path("models", "multinomial_lasso_model.rds")
default_lasso_metrics_path <- file.path("models", "multinomial_lasso_metrics.csv")
default_lasso_pred_path <- file.path("models", "multinomial_lasso_predictions.csv")

if (!requireNamespace("nnet", quietly = TRUE)) {
  stop(
    "Package 'nnet' is required but not installed. Install with: install.packages('nnet')",
    call. = FALSE
  )
}
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
    "Usage: Rscript scripts/train_multinomial_logit.R [train_csv_path]",
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

build_metrics <- function(actual, predicted) {
  labels <- levels(actual)
  cm <- table(
    actual = factor(actual, levels = labels),
    predicted = factor(predicted, levels = labels)
  )

  class_precision <- numeric(length(labels))
  class_recall <- numeric(length(labels))
  class_f1 <- numeric(length(labels))

  for (i in seq_along(labels)) {
    tp <- cm[i, i]
    fp <- sum(cm[, i]) - tp
    fn <- sum(cm[i, ]) - tp

    precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    recall <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    f1 <- if ((precision + recall) == 0) 0 else 2 * precision * recall / (precision + recall)

    class_precision[i] <- precision
    class_recall[i] <- recall
    class_f1[i] <- f1
  }

  data.frame(
    metric = c("accuracy", "macro_precision", "macro_recall", "macro_f1"),
    value = c(
      mean(predicted == actual),
      mean(class_precision),
      mean(class_recall),
      mean(class_f1)
    )
  )
}

build_predictions_df <- function(actual, predicted) {
  data.frame(
    actual_quality = as.character(actual),
    predicted_quality = as.character(predicted)
  )
}

test_actual <- test_df$quality

model_formula <- as.formula(paste("quality ~", paste(feature_cols, collapse = " + ")))
multinom_fit <- nnet::multinom(model_formula, data = train_df, trace = FALSE)
multinom_pred <- predict(multinom_fit, newdata = test_df, type = "class")
multinom_metrics <- build_metrics(test_actual, multinom_pred)
multinom_predictions <- build_predictions_df(test_actual, multinom_pred)

x_train <- as.matrix(train_df[, feature_cols])
x_test <- as.matrix(test_df[, feature_cols])
y_train <- train_df$quality

set.seed(42)
ridge_cv <- glmnet::cv.glmnet(
  x = x_train,
  y = y_train,
  family = "multinomial",
  alpha = 0,
  type.measure = "class"
)
ridge_fit <- ridge_cv$glmnet.fit
ridge_pred <- predict(ridge_fit, newx = x_test, s = ridge_cv$lambda.min, type = "class")
ridge_pred <- factor(as.vector(ridge_pred), levels = levels(test_actual))
ridge_metrics <- build_metrics(test_actual, ridge_pred)
ridge_predictions <- build_predictions_df(test_actual, ridge_pred)

set.seed(42)
lasso_cv <- glmnet::cv.glmnet(
  x = x_train,
  y = y_train,
  family = "multinomial",
  alpha = 1,
  type.measure = "class"
)
lasso_fit <- lasso_cv$glmnet.fit
lasso_pred <- predict(lasso_fit, newx = x_test, s = lasso_cv$lambda.min, type = "class")
lasso_pred <- factor(as.vector(lasso_pred), levels = levels(test_actual))
lasso_metrics <- build_metrics(test_actual, lasso_pred)
lasso_predictions <- build_predictions_df(test_actual, lasso_pred)

model_dir <- dirname(default_model_path)
if (!dir.exists(model_dir)) {
  dir.create(model_dir, recursive = TRUE)
}

saveRDS(multinom_fit, file = default_model_path)
write.csv(multinom_metrics, file = default_metrics_path, row.names = FALSE)
write.csv(multinom_predictions, file = default_pred_path, row.names = FALSE)

saveRDS(
  list(model = ridge_fit, lambda_min = ridge_cv$lambda.min, cv_fit = ridge_cv),
  file = default_ridge_model_path
)
write.csv(ridge_metrics, file = default_ridge_metrics_path, row.names = FALSE)
write.csv(ridge_predictions, file = default_ridge_pred_path, row.names = FALSE)

saveRDS(
  list(model = lasso_fit, lambda_min = lasso_cv$lambda.min, cv_fit = lasso_cv),
  file = default_lasso_model_path
)
write.csv(lasso_metrics, file = default_lasso_metrics_path, row.names = FALSE)
write.csv(lasso_predictions, file = default_lasso_pred_path, row.names = FALSE)

cat(sprintf("Rows used: %d (train=%d, test=%d)\n", nrow(df), nrow(train_df), nrow(test_df)))
cat("Saved artifacts:\n")
cat(sprintf("- Multinomial logit model: %s\n", default_model_path))
cat(sprintf("- Multinomial logit metrics: %s\n", default_metrics_path))
cat(sprintf("- Multinomial logit predictions: %s\n", default_pred_path))
cat(sprintf("- Ridge model: %s\n", default_ridge_model_path))
cat(sprintf("- Ridge metrics: %s\n", default_ridge_metrics_path))
cat(sprintf("- Ridge predictions: %s\n", default_ridge_pred_path))
cat(sprintf("- Lasso model: %s\n", default_lasso_model_path))
cat(sprintf("- Lasso metrics: %s\n", default_lasso_metrics_path))
cat(sprintf("- Lasso predictions: %s\n", default_lasso_pred_path))

cat("\nMultinomial logit metrics:\n")
print(multinom_metrics)
cat("\nRidge metrics:\n")
print(ridge_metrics)
cat("\nLasso metrics:\n")
print(lasso_metrics)
