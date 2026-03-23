#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  args <- commandArgs(trailingOnly = TRUE)
}))

default_red <- "winequality-red.csv"
default_white <- "winequality-white.csv"
default_output <- file.path("data", "processed", "winequality_combined.csv")

resolve_input <- function(path_value, expected_name) {
  if (file.exists(path_value)) {
    return(path_value)
  }

  # If only a filename is provided, try to find it anywhere under project root.
  if (!grepl("[/\\\\]", path_value)) {
    escaped_name <- gsub("\\.", "\\\\.", expected_name)
    matches <- list.files(
      path = ".",
      pattern = paste0("^", escaped_name, "$"),
      recursive = TRUE,
      full.names = TRUE
    )
    existing <- matches[file.exists(matches)]
    if (length(existing) > 0) {
      return(existing[[1]])
    }
  }

  return(path_value)
}

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

red_path <- resolve_input(red_path, "winequality-red.csv")
white_path <- resolve_input(white_path, "winequality-white.csv")

if (!file.exists(red_path) || !file.exists(white_path)) {
  # Repo fallback: if sources are unavailable but a combined train.csv exists, use it.
  if (file.exists("train.csv")) {
    combined <- read.csv("train.csv", sep = ";", header = TRUE, check.names = FALSE)
    if (!("is_red" %in% names(combined))) {
      stop(
        "train.csv exists but does not include required 'is_red' column.",
        call. = FALSE
      )
    }

    out_dir <- dirname(output_path)
    if (!dir.exists(out_dir)) {
      dir.create(out_dir, recursive = TRUE)
    }
    write.csv(combined, file = output_path, row.names = FALSE)
    cat(
      sprintf(
        "Wrote %d rows to %s (using existing combined dataset: train.csv)\n",
        nrow(combined),
        output_path
      )
    )
    quit(save = "no", status = 0)
  }

  stop(
    paste(
      "Could not find source files.",
      "Red path checked:", red_path,
      "| White path checked:", white_path
    ),
    call. = FALSE
  )
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
