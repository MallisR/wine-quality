# Shared utilities used by scripts in this repo.
#
# Keeping these helpers in one place makes the modeling scripts shorter, easier to read,
# and less error-prone (one consistent definition of: data loading, splitting, metrics).

load_train_data <- function(path, required_cols) {
  if (!file.exists(path)) {
    stop(sprintf("Input file not found: %s", path), call. = FALSE)
  }

  df <- read.csv(path, sep = ";", header = TRUE, check.names = FALSE)

  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop(
      sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")),
      call. = FALSE
    )
  }

  # Drop rows with missing values in required columns to keep downstream models simple
  # and comparable across scripts.
  df <- df[complete.cases(df[, required_cols]), required_cols, drop = FALSE]

  df
}

split_train_test <- function(df, train_frac = 0.8, seed = 42) {
  set.seed(seed)
  idx <- sample.int(nrow(df), size = floor(train_frac * nrow(df)))
  list(train = df[idx, , drop = FALSE], test = df[-idx, , drop = FALSE])
}

metric_multiclass <- function(actual, predicted) {
  # Accuracy is "exact class match" on the hold-out test split.
  accuracy <- mean(predicted == actual)

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
    value = c(accuracy, mean(precision), mean(recall), mean(f1))
  )
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
}

