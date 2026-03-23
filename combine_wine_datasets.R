#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  args <- commandArgs(trailingOnly = TRUE)
}))

default_red <- "winequality-red.csv"
default_white <- "winequality-white.csv"
default_output <- file.path("data", "processed", "winequality_combined.csv")

if (length(args) == 0) {
  red_path <- default_red
  white_path <- default_white
  output_path <- default_output
} else if (length(args) == 3) {
  red_path <- args[[1]]
  white_path <- args[[2]]
  output_path <- args[[3]]
} else {
  stop(
    paste(
      "Usage:",
      "Rscript combine_wine_datasets.R [red_csv white_csv output_csv]"
    ),
    call. = FALSE
  )
}

if (!file.exists(red_path)) {
  stop(paste("Red dataset not found:", red_path), call. = FALSE)
}
if (!file.exists(white_path)) {
  stop(paste("White dataset not found:", white_path), call. = FALSE)
}

red <- read.csv(red_path, sep = ";", header = TRUE, check.names = FALSE)
white <- read.csv(white_path, sep = ";", header = TRUE, check.names = FALSE)

if (!identical(names(red), names(white))) {
  stop(
    paste(
      "Input CSV schemas differ.",
      "Red columns:",
      paste(names(red), collapse = ","),
      "| White columns:",
      paste(names(white), collapse = ",")
    ),
    call. = FALSE
  )
}

red$is_red <- 1L
white$is_red <- 0L

combined <- rbind(red, white)

out_dir <- dirname(output_path)
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

write.csv(combined, file = output_path, row.names = FALSE)
cat(sprintf("Wrote %d rows to %s\n", nrow(combined), output_path))
