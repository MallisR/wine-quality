#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  args <- commandArgs(trailingOnly = TRUE)
}))

default_input <- "train.csv"
default_out_dir <- "models"
default_results_path <- file.path(default_out_dir, "advanced_multinomial_results.csv")
default_outlier_rows_path <- file.path(default_out_dir, "identified_outliers.csv")
default_outlier_summary_path <- file.path(default_out_dir, "identified_outliers_summary.csv")

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
    "Usage: Rscript scripts/advanced_multinomial_pipeline.R [train_csv_path]",
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

df <- df[complete.cases(df[, required_cols]), required_cols]
df$quality <- as.factor(df$quality)

add_engineered_features <- function(data) {
  eps <- 1e-6
  data$free_to_total_sulfur_ratio <- data$free.sulfur.dioxide / (data$total.sulfur.dioxide + eps)
  data$sulphates_to_chlorides_ratio <- data$sulphates / (data$chlorides + eps)
  data$acidity_balance <- data$fixed.acidity - data$volatile.acidity
  data$alcohol_density_interaction <- data$alcohol * data$density
  data
}

detect_outliers_by_iqr <- function(data, feature_names, iqr_mult = 1.5, max_allowed_flags = 2) {
  q1 <- vapply(data[, feature_names, drop = FALSE], quantile, numeric(1), probs = 0.25)
  q3 <- vapply(data[, feature_names, drop = FALSE], quantile, numeric(1), probs = 0.75)
  iqr <- q3 - q1

  lower <- q1 - iqr_mult * iqr
  upper <- q3 + iqr_mult * iqr

  flags <- sapply(
    feature_names,
    function(f) {
      data[[f]] < lower[[f]] | data[[f]] > upper[[f]]
    }
  )

  if (is.vector(flags)) {
    flags <- matrix(flags, ncol = 1)
  }

  flag_count <- rowSums(flags)
  is_outlier <- flag_count > max_allowed_flags

  list(
    is_outlier = is_outlier,
    flag_count = flag_count,
    lower = lower,
    upper = upper
  )
}

apply_outlier_bounds <- function(data, feature_names, lower, upper, max_allowed_flags = 2) {
  flags <- sapply(
    feature_names,
    function(f) {
      data[[f]] < lower[[f]] | data[[f]] > upper[[f]]
    }
  )

  if (is.vector(flags)) {
    flags <- matrix(flags, ncol = 1)
  }

  rowSums(flags) > max_allowed_flags
}

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

class_weights <- function(y) {
  tab <- table(y)
  inv_freq <- 1 / as.numeric(tab)
  names(inv_freq) <- names(tab)
  as.numeric(inv_freq[as.character(y)])
}

make_matrix <- function(df_features, include_interactions = FALSE) {
  if (include_interactions) {
    model.matrix(~ (. )^2, data = df_features)[, -1, drop = FALSE]
  } else {
    model.matrix(~ ., data = df_features)[, -1, drop = FALSE]
  }
}

stratified_train_indices <- function(y, train_frac = 0.8) {
  idx <- integer(0)
  for (cls in levels(y)) {
    cls_idx <- which(y == cls)
    n_train_cls <- max(1, floor(length(cls_idx) * train_frac))
    idx <- c(idx, sample(cls_idx, size = n_train_cls))
  }
  sort(unique(idx))
}

run_one_model <- function(
  model_name,
  train_df,
  test_df,
  feature_cols,
  alpha_value,
  use_interactions = FALSE,
  use_class_weights = FALSE,
  remove_outliers = FALSE
) {
  train_original <- train_df
  train_local <- train_df
  test_local <- test_df
  outlier_filter_applied <- FALSE

  if (remove_outliers) {
    det <- detect_outliers_by_iqr(train_local, feature_cols, iqr_mult = 1.5, max_allowed_flags = 2)
    keep <- !det$is_outlier
    train_local <- train_local[keep, , drop = FALSE]

    # If filtering removes too many rare classes, keep the original training rows.
    class_counts <- table(train_local$quality)
    if (length(class_counts) < 2 || any(class_counts < 2)) {
      train_local <- train_original
    } else {
      outlier_filter_applied <- TRUE
    }
  }

  x_train <- make_matrix(train_local[, feature_cols, drop = FALSE], include_interactions = use_interactions)
  x_test <- make_matrix(test_local[, feature_cols, drop = FALSE], include_interactions = use_interactions)
  y_train <- train_local$quality
  y_test <- test_local$quality

  w <- NULL
  if (use_class_weights) {
    w <- class_weights(y_train)
  }

  min_class_count <- min(as.numeric(table(y_train)))
  nfolds <- max(3, min(5, min_class_count))

  cv_fit <- tryCatch(
    glmnet::cv.glmnet(
      x = x_train,
      y = y_train,
      family = "multinomial",
      alpha = alpha_value,
      type.measure = "class",
      weights = w,
      nfolds = nfolds
    ),
    error = function(e) NULL
  )

  if (is.null(cv_fit)) {
    return(data.frame(
      model_name = model_name,
      alpha = alpha_value,
      interactions = use_interactions,
      class_weights = use_class_weights,
      outlier_filtered = outlier_filter_applied,
      lambda_min = NA_real_,
      train_rows = nrow(train_local),
      test_rows = nrow(test_local),
      accuracy = NA_real_,
      macro_precision = NA_real_,
      macro_recall = NA_real_,
      macro_f1 = NA_real_
    ))
  }

  pred <- predict(cv_fit$glmnet.fit, newx = x_test, s = cv_fit$lambda.min, type = "class")
  pred <- factor(as.vector(pred), levels = levels(y_test))
  metrics <- metric_frame(y_test, pred)

  data.frame(
    model_name = model_name,
    alpha = alpha_value,
    interactions = use_interactions,
    class_weights = use_class_weights,
    outlier_filtered = outlier_filter_applied,
    lambda_min = cv_fit$lambda.min,
    train_rows = nrow(train_local),
    test_rows = nrow(test_local),
    accuracy = metrics$accuracy,
    macro_precision = metrics$macro_precision,
    macro_recall = metrics$macro_recall,
    macro_f1 = metrics$macro_f1
  )
}

df_eng <- add_engineered_features(df)
base_features <- setdiff(required_cols, "quality")
engineered_features <- c(
  base_features,
  "free_to_total_sulfur_ratio",
  "sulphates_to_chlorides_ratio",
  "acidity_balance",
  "alcohol_density_interaction"
)

global_outliers <- detect_outliers_by_iqr(df_eng, engineered_features, iqr_mult = 1.5, max_allowed_flags = 2)
outlier_rows <- df_eng[global_outliers$is_outlier, ]
outlier_rows$outlier_flag_count <- global_outliers$flag_count[global_outliers$is_outlier]

outlier_summary <- data.frame(
  total_rows = nrow(df_eng),
  outlier_rows = sum(global_outliers$is_outlier),
  outlier_pct = mean(global_outliers$is_outlier) * 100
)

seeds <- c(11, 22, 33)
model_specs <- list(
  list(name = "lasso_main", alpha = 1, interactions = FALSE, weights = FALSE, outliers = FALSE, engineered = FALSE),
  list(name = "lasso_interactions", alpha = 1, interactions = TRUE, weights = FALSE, outliers = FALSE, engineered = FALSE),
  list(name = "elastic_interactions_alpha_0.5", alpha = 0.5, interactions = TRUE, weights = FALSE, outliers = FALSE, engineered = FALSE),
  list(name = "weighted_elastic_interactions", alpha = 0.5, interactions = TRUE, weights = TRUE, outliers = FALSE, engineered = FALSE),
  list(name = "weighted_elastic_engineered_interactions", alpha = 0.5, interactions = TRUE, weights = TRUE, outliers = FALSE, engineered = TRUE),
  list(name = "weighted_elastic_engineered_interactions_no_outliers", alpha = 0.5, interactions = TRUE, weights = TRUE, outliers = TRUE, engineered = TRUE)
)

all_results <- list()
row_id <- 1

for (seed in seeds) {
  set.seed(seed)
  idx <- stratified_train_indices(df_eng$quality, train_frac = 0.8)
  train_df <- df_eng[idx, ]
  test_df <- df_eng[-idx, ]

  for (spec in model_specs) {
    features <- if (spec$engineered) engineered_features else base_features
    res <- run_one_model(
      model_name = spec$name,
      train_df = train_df,
      test_df = test_df,
      feature_cols = features,
      alpha_value = spec$alpha,
      use_interactions = spec$interactions,
      use_class_weights = spec$weights,
      remove_outliers = spec$outliers
    )
    res$split_seed <- seed
    all_results[[row_id]] <- res
    row_id <- row_id + 1
  }
}

results_by_split <- do.call(rbind, all_results)

summary_results <- aggregate(
  cbind(accuracy, macro_precision, macro_recall, macro_f1, train_rows) ~ model_name,
  data = results_by_split,
  FUN = mean
)

sd_results <- aggregate(
  cbind(accuracy, macro_precision, macro_recall, macro_f1) ~ model_name,
  data = results_by_split,
  FUN = sd
)
names(sd_results)[2:5] <- c("accuracy_sd", "macro_precision_sd", "macro_recall_sd", "macro_f1_sd")

summary_results <- merge(summary_results, sd_results, by = "model_name", all.x = TRUE)
summary_results <- summary_results[order(-summary_results$accuracy, -summary_results$macro_f1), ]

if (!dir.exists(default_out_dir)) {
  dir.create(default_out_dir, recursive = TRUE)
}

write.csv(summary_results, file = default_results_path, row.names = FALSE)
write.csv(outlier_rows, file = default_outlier_rows_path, row.names = FALSE)
write.csv(outlier_summary, file = default_outlier_summary_path, row.names = FALSE)

cat(sprintf("Rows used: %d\n", nrow(df_eng)))
cat(sprintf("Saved model comparison: %s\n", default_results_path))
cat(sprintf("Saved identified outlier rows: %s\n", default_outlier_rows_path))
cat(sprintf("Saved outlier summary: %s\n", default_outlier_summary_path))
cat("\nAverage performance across repeated splits:\n")
print(summary_results)
cat("\nOutlier summary:\n")
print(outlier_summary)
