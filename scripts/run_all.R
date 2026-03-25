#!/usr/bin/env Rscript

# Purpose:
# - Run all analysis/model scripts in a consistent order and produce a single summary table.
# - Consolidates results from the output files written under `models/`.
#
# Notes on comparability:
# - Most modeling scripts report a single 80/20 hold-out split accuracy (set.seed(42)).
# - The advanced pipeline reports an average across repeated splits (not directly comparable).
#
# Run:
# - Rscript scripts/run_all.R
# - Rscript scripts/run_all.R --skip-advanced

suppressWarnings(suppressMessages({
  args <- commandArgs(trailingOnly = TRUE)
}))

skip_advanced <- any(args %in% c("--skip-advanced", "--no-advanced"))

run_cmd <- function(cmd) {
  cat(sprintf("\n==> %s\n", cmd))
  status <- system(cmd)
  if (!identical(status, 0L)) {
    stop(sprintf("Command failed (exit code %s): %s", status, cmd), call. = FALSE)
  }
}

read_metric_csv_accuracy <- function(path) {
  if (!file.exists(path)) return(NA_real_)
  df <- read.csv(path, stringsAsFactors = FALSE)
  # Expected format: metric,value
  acc_row <- df[df$metric == "accuracy", , drop = FALSE]
  if (nrow(acc_row) == 0) return(NA_real_)
  as.numeric(acc_row$value[[1]])
}

best_accuracy_from_leaderboard <- function(path) {
  if (!file.exists(path)) return(NA_real_)
  df <- read.csv(path, stringsAsFactors = FALSE)
  if (!("accuracy" %in% names(df))) return(NA_real_)
  suppressWarnings(max(as.numeric(df$accuracy), na.rm = TRUE))
}

best_model_from_leaderboard <- function(path) {
  if (!file.exists(path)) return(NA_character_)
  df <- read.csv(path, stringsAsFactors = FALSE)
  if (!all(c("accuracy", "model_name") %in% names(df))) return(NA_character_)
  df <- df[order(-df$accuracy), , drop = FALSE]
  as.character(df$model_name[[1]])
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
}

ensure_dir("models")

# 1) Core multinomial models (multinom + ridge + lasso)
run_cmd("Rscript scripts/train_multinomial_logit.R")

# 2) glmnet interaction tuning benchmark (pairwise interactions)
run_cmd("Rscript scripts/tune_multinomial_glmnet.R")

# 3) quick improvements benchmark (targeted interactions)
run_cmd("Rscript scripts/quick_improvements.R")

# 4) outlier-robust preprocessing + glmnet
if (file.exists("scripts/train_outlier_robust_glmnet.R")) {
  run_cmd("Rscript scripts/train_outlier_robust_glmnet.R")
}

# 5) MANOVA test (not an accuracy metric, but include results)
run_cmd("Rscript scripts/run_manova_acidity.R")

# 6) Advanced pipeline (optional; can take longer)
if (!skip_advanced && file.exists("scripts/advanced_multinomial_pipeline.R")) {
  run_cmd("Rscript scripts/advanced_multinomial_pipeline.R")
}

summary_rows <- list(
  data.frame(
    artifact = "multinomial_logit",
    evaluation = "single_split_seed42",
    accuracy = read_metric_csv_accuracy("models/multinomial_logit_metrics.csv"),
    details = "nnet::multinom"
  ),
  data.frame(
    artifact = "multinomial_ridge",
    evaluation = "single_split_seed42",
    accuracy = read_metric_csv_accuracy("models/multinomial_ridge_metrics.csv"),
    details = "glmnet alpha=0"
  ),
  data.frame(
    artifact = "multinomial_lasso",
    evaluation = "single_split_seed42",
    accuracy = read_metric_csv_accuracy("models/multinomial_lasso_metrics.csv"),
    details = "glmnet alpha=1"
  ),
  data.frame(
    artifact = "glmnet_tuning_best",
    evaluation = "single_split_seed42",
    accuracy = best_accuracy_from_leaderboard("models/multinomial_glmnet_tuning_results.csv"),
    details = best_model_from_leaderboard("models/multinomial_glmnet_tuning_results.csv")
  ),
  data.frame(
    artifact = "quick_improvements_best",
    evaluation = "single_split_seed42",
    accuracy = best_accuracy_from_leaderboard("models/quick_improvements_results.csv"),
    details = best_model_from_leaderboard("models/quick_improvements_results.csv")
  ),
  data.frame(
    artifact = "robust_glmnet",
    evaluation = "single_split_seed42",
    accuracy = read_metric_csv_accuracy("models/robust_glmnet_metrics.csv"),
    details = "winsorize(1/99%) + median/MAD scale + glmnet alpha=0.5 + interactions"
  )
)

if (file.exists("models/advanced_multinomial_results.csv")) {
  adv <- read.csv("models/advanced_multinomial_results.csv", stringsAsFactors = FALSE)
  if ("accuracy" %in% names(adv) && "model_name" %in% names(adv)) {
    adv <- adv[order(-adv$accuracy), , drop = FALSE]
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      artifact = "advanced_pipeline_best",
      evaluation = "multi_split_average",
      accuracy = as.numeric(adv$accuracy[[1]]),
      details = as.character(adv$model_name[[1]])
    )
  }
}

if (file.exists("models/manova_acidity_test.csv")) {
  man <- read.csv("models/manova_acidity_test.csv", stringsAsFactors = FALSE)
  # Keep only a short summary in the run-all table.
  pillai_p <- man$p_value[man$test == "Pillai"]
  wilks_p <- man$p_value[man$test == "Wilks"]
  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    artifact = "manova_acidity",
    evaluation = "hypothesis_test",
    accuracy = NA_real_,
    details = sprintf("Pillai p=%s; Wilks p=%s", pillai_p, wilks_p)
  )
}

summary_df <- do.call(rbind, summary_rows)
summary_df <- summary_df[order(is.na(summary_df$accuracy), -summary_df$accuracy), , drop = FALSE]

out_csv <- file.path("models", "run_all_summary.csv")
write.csv(summary_df, out_csv, row.names = FALSE)

cat("\n====================\n")
cat("Run-all summary:\n")
print(summary_df)
cat(sprintf("\nSaved: %s\n", out_csv))

