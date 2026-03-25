#!/usr/bin/env Rscript

# Purpose:
# - MANOVA test: do red vs white wines differ across the multivariate acidity profile?
# - Uses all columns containing "acidity" as the multivariate response and tests group
#   differences by `is_red` (converted to `wine_type` factor).
#
# Inputs:
# - `train.csv` (semicolon-delimited), must include `is_red` and acidity columns
#
# Outputs (written to `models/`):
# - `manova_acidity_test.csv` (Pillai + Wilks stats)
# - `manova_acidity_summary.txt` (full printed MANOVA summaries)
#
# Run:
# - Rscript scripts/run_manova_acidity.R [train_csv_path]

suppressWarnings(suppressMessages({
  args <- commandArgs(trailingOnly = TRUE)
}))

default_input <- "train.csv"
default_out_dir <- "models"
default_stats_path <- file.path(default_out_dir, "manova_acidity_test.csv")
default_summary_path <- file.path(default_out_dir, "manova_acidity_summary.txt")

if (length(args) == 0) {
  input_path <- default_input
} else if (length(args) == 1) {
  input_path <- args[[1]]
} else {
  stop(
    "Usage: Rscript scripts/run_manova_acidity.R [train_csv_path]",
    call. = FALSE
  )
}

if (!file.exists(input_path)) {
  stop(sprintf("Input file not found: %s", input_path), call. = FALSE)
}

df <- read.csv(input_path, sep = ";", header = TRUE, check.names = FALSE)

acidity_vars <- grep("acidity", names(df), value = TRUE)
required_cols <- c(acidity_vars, "is_red")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(
    sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")),
    call. = FALSE
  )
}

if (length(acidity_vars) < 2) {
  stop("MANOVA needs at least two acidity variables.", call. = FALSE)
}

df <- df[complete.cases(df[, required_cols]), required_cols]
df$wine_type <- factor(df$is_red, levels = c(0, 1), labels = c("white", "red"))

response_expr <- paste(acidity_vars, collapse = ", ")
manova_formula <- as.formula(paste0("cbind(", response_expr, ") ~ wine_type"))

fit <- manova(manova_formula, data = df)
summary_pillai <- summary(fit, test = "Pillai")
summary_wilks <- summary(fit, test = "Wilks")

pillai_stats <- summary_pillai$stats
wilks_stats <- summary_wilks$stats

stats_out <- data.frame(
  test = c("Pillai", "Wilks"),
  statistic = c(pillai_stats[1, 1], wilks_stats[1, 1]),
  approx_F = c(pillai_stats[1, 4], wilks_stats[1, 4]),
  num_df = c(pillai_stats[1, 2], wilks_stats[1, 2]),
  den_df = c(pillai_stats[1, 3], wilks_stats[1, 3]),
  p_value = c(pillai_stats[1, 6], wilks_stats[1, 6])
)

if (!dir.exists(default_out_dir)) {
  dir.create(default_out_dir, recursive = TRUE)
}

write.csv(stats_out, file = default_stats_path, row.names = FALSE)

capture.output(
  {
    cat("MANOVA formula:\n")
    print(manova_formula)
    cat("\nAcidity variables used:\n")
    print(acidity_vars)
    cat("\nPillai test summary:\n")
    print(summary_pillai)
    cat("\nWilks test summary:\n")
    print(summary_wilks)
  },
  file = default_summary_path
)

cat(sprintf("Rows used: %d\n", nrow(df)))
cat(sprintf("Saved MANOVA stats: %s\n", default_stats_path))
cat(sprintf("Saved MANOVA summary: %s\n", default_summary_path))
cat("Key results:\n")
print(stats_out)
