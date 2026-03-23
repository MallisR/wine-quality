#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  args <- commandArgs(trailingOnly = TRUE)
}))

default_input <- "train.csv"
default_model_path <- file.path("models", "multinomial_logit_model.rds")
default_metrics_path <- file.path("models", "multinomial_logit_metrics.csv")
default_pred_path <- file.path("models", "multinomial_logit_predictions.csv")

if (!requireNamespace("nnet", quietly = TRUE)) {
  stop(
    "Package 'nnet' is required but not installed. Install with: install.packages('nnet')",
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

model_formula <- as.formula(paste("quality ~", paste(feature_cols, collapse = " + ")))
fit <- nnet::multinom(model_formula, data = train_df, trace = FALSE)

pred <- predict(fit, newdata = test_df, type = "class")

test_actual <- test_df$quality
acc <- mean(pred == test_actual)

labels <- levels(test_actual)
cm <- table(
  actual = factor(test_actual, levels = labels),
  predicted = factor(pred, levels = labels)
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

metrics <- data.frame(
  metric = c("accuracy", "macro_precision", "macro_recall", "macro_f1"),
  value = c(
    acc,
    mean(class_precision),
    mean(class_recall),
    mean(class_f1)
  )
)

predictions <- data.frame(
  actual_quality = as.character(test_actual),
  predicted_quality = as.character(pred)
)

model_dir <- dirname(default_model_path)
if (!dir.exists(model_dir)) {
  dir.create(model_dir, recursive = TRUE)
}

saveRDS(fit, file = default_model_path)
write.csv(metrics, file = default_metrics_path, row.names = FALSE)
write.csv(predictions, file = default_pred_path, row.names = FALSE)

cat(sprintf("Rows used: %d (train=%d, test=%d)\n", nrow(df), nrow(train_df), nrow(test_df)))
cat(sprintf("Saved model: %s\n", default_model_path))
cat(sprintf("Saved metrics: %s\n", default_metrics_path))
cat(sprintf("Saved predictions: %s\n", default_pred_path))
cat("Metrics:\n")
print(metrics)
