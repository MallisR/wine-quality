#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  args <- commandArgs(trailingOnly = TRUE)
}))

default_input <- "train.csv"
default_out_dir <- "models"
default_results_path <- file.path(default_out_dir, "quick_improvements_results.csv")

if (!requireNamespace("glmnet", quietly = TRUE)) {
  stop(
    "Package 'glmnet' is required but not installed. Install with: install.packages('glmnet')",
    call. = FALSE
  )
}
if (!requireNamespace("MASS", quietly = TRUE)) {
  stop(
    "Package 'MASS' is required but not installed. Install with: install.packages('MASS')",
    call. = FALSE
  )
}

if (length(args) == 0) {
  input_path <- default_input
} else if (length(args) == 1) {
  input_path <- args[[1]]
} else {
  stop(
    "Usage: Rscript scripts/quick_improvements.R [train_csv_path]",
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

feature_cols <- setdiff(required_cols, "quality")
train_features <- train_df[, feature_cols, drop = FALSE]
test_features <- test_df[, feature_cols, drop = FALSE]

# Main effects (no interactions)
x_train_main <- model.matrix(~ ., data = train_features)[, -1, drop = FALSE]
x_test_main <- model.matrix(~ ., data = test_features)[, -1, drop = FALSE]

# Targeted interactions only (small set to keep it fast)
interaction_formula <- ~ . +
  alcohol:density +
  alcohol:sulphates +
  alcohol:fixed.acidity +
  density:pH +
  is_red:alcohol

x_train_target_inter <- model.matrix(interaction_formula, data = train_features)[, -1, drop = FALSE]
x_test_target_inter <- model.matrix(interaction_formula, data = test_features)[, -1, drop = FALSE]

fit_glmnet_variant <- function(name, x_train, x_test, alpha_value, train_y, test_y) {
  y_train <- train_y
  y_test <- test_y

  min_class_count <- min(as.numeric(table(y_train)))
  nfolds <- max(3, min(5, min_class_count))

  set.seed(42)
  cv_fit <- glmnet::cv.glmnet(
    x = x_train,
    y = y_train,
    family = "multinomial",
    alpha = alpha_value,
    type.measure = "class",
    nfolds = nfolds
  )

  pred <- predict(cv_fit$glmnet.fit, newx = x_test, s = cv_fit$lambda.min, type = "class")
  pred <- factor(as.vector(pred), levels = levels(y_test))
  metrics <- metric_frame(y_test, pred)

  cbind(
    data.frame(
      model_name = name,
      alpha = alpha_value,
      lambda_min = cv_fit$lambda.min
    ),
    metrics
  )
}

results <- list()
r <- 1

alpha_grid <- c(0, 0.25, 0.5, 0.75, 1)

for (a in alpha_grid) {
  results[[r]] <- fit_glmnet_variant(
    name = sprintf("glmnet_main_alpha_%s", a),
    x_train = x_train_main,
    x_test = x_test_main,
    alpha_value = a,
    train_y = train_df$quality,
    test_y = test_df$quality
  )
  r <- r + 1

  results[[r]] <- fit_glmnet_variant(
    name = sprintf("glmnet_target_inter_alpha_%s", a),
    x_train = x_train_target_inter,
    x_test = x_test_target_inter,
    alpha_value = a,
    train_y = train_df$quality,
    test_y = test_df$quality
  )
  r <- r + 1
}

# Ordinal logistic regression (fast baseline that respects ordering)
ord_train <- train_df
ord_test <- test_df
ord_train$quality_ord <- ordered(as.integer(as.character(ord_train$quality)))
ord_test$quality_ord <- ordered(as.integer(as.character(ord_test$quality)))

ord_formula <- as.formula(paste("quality_ord ~", paste(feature_cols, collapse = " + ")))
ord_fit <- MASS::polr(ord_formula, data = ord_train, Hess = FALSE)
ord_pred <- predict(ord_fit, newdata = ord_test, type = "class")
ord_pred <- factor(as.character(ord_pred), levels = levels(test_df$quality))
ord_metrics <- metric_frame(test_df$quality, ord_pred)

results[[r]] <- cbind(
  data.frame(model_name = "ordinal_polr_main", alpha = NA_real_, lambda_min = NA_real_),
  ord_metrics
)

results_df <- do.call(rbind, results)
results_df <- results_df[order(-results_df$accuracy, -results_df$macro_f1), ]

if (!dir.exists(default_out_dir)) {
  dir.create(default_out_dir, recursive = TRUE)
}
write.csv(results_df, file = default_results_path, row.names = FALSE)

cat(sprintf("Rows used: %d (train=%d, test=%d)\n", nrow(df), nrow(train_df), nrow(test_df)))
cat(sprintf("Saved results: %s\n", default_results_path))
cat("Leaderboard (sorted by accuracy then macro F1):\n")
print(results_df)
