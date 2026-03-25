# Wine Acidity LDA Analysis — Red vs White (Vinho Verde)
# =========================================================
# Run in terminal: Rscript wine_lda.R
#
# Required packages (run once if not installed):
# install.packages(c("tidyverse", "ggplot2", "MASS", "patchwork"))

library(tidyverse)
library(ggplot2)
library(MASS)
library(patchwork)

RED_COLOR   <- "#A32D2D"
WHITE_COLOR <- "#D4A017"
PALETTE     <- c("Red" = RED_COLOR, "White" = WHITE_COLOR)

# ── 1. Load data ──────────────────────────────────────────────────────────────
red   <- read.csv("winequality-red.csv",   sep = ";") |> mutate(type = "Red")
white <- read.csv("winequality-white.csv", sep = ";") |> mutate(type = "White")
df    <- bind_rows(red, white) |> mutate(type = factor(type))

# ── 2. Run LDA ────────────────────────────────────────────────────────────────
lda_model <- lda(type ~ fixed.acidity + volatile.acidity + citric.acid + pH,
                 data = df)

cat("\n══════════════════════════════════════════\n")
cat("  LDA: Red vs White Wine Acidity\n")
cat("══════════════════════════════════════════\n\n")

cat("── Group means ─────────────────────────\n")
print(lda_model$means)

cat("\n── Coefficients of linear discriminants ─\n")
print(lda_model$scaling)

# ── 3. Predictions & accuracy ─────────────────────────────────────────────────
lda_pred <- predict(lda_model, df)

confusion <- table(Predicted = lda_pred$class, Actual = df$type)
accuracy  <- round(sum(diag(confusion)) / sum(confusion) * 100, 2)

cat("\n── Confusion matrix ────────────────────\n")
print(confusion)
cat(sprintf("\nOverall accuracy: %.2f%%\n", accuracy))

# ── 4. Save results ───────────────────────────────────────────────────────────
results <- data.frame(
  Actual    = df$type,
  Predicted = lda_pred$class,
  LD1       = lda_pred$x[, 1]
)
write.csv(results, "wine_lda_results.csv", row.names = FALSE)
cat("Saved → wine_lda_results.csv\n")

# ── 5. Plots ──────────────────────────────────────────────────────────────────
theme_wine <- theme_minimal(base_size = 12) +
  theme(
    plot.title      = element_text(face = "bold", size = 12),
    legend.position = "bottom",
    legend.title    = element_blank()
  )

# 5a. LD1 density plot
p1 <- ggplot(results, aes(x = LD1, fill = Actual, color = Actual)) +
  geom_density(alpha = 0.3, linewidth = 1) +
  geom_vline(data = results |> group_by(Actual) |> summarise(m = mean(LD1)),
             aes(xintercept = m, color = Actual), linetype = "dashed", linewidth = 1) +
  scale_fill_manual(values  = PALETTE) +
  scale_color_manual(values = PALETTE) +
  labs(title = "LD1 score distribution",
       subtitle = "Higher separation = better classification",
       x = "Linear Discriminant 1", y = "Density") +
  theme_wine

# 5b. Coefficient importance bar chart
coef_df <- data.frame(
  Variable    = rownames(lda_model$scaling),
  Coefficient = abs(lda_model$scaling[, 1])
) |> arrange(desc(Coefficient))

p2 <- ggplot(coef_df, aes(x = reorder(Variable, Coefficient), y = Coefficient)) +
  geom_col(fill = RED_COLOR, width = 0.5) +
  geom_text(aes(label = round(Coefficient, 3)), hjust = -0.2, size = 3.5) +
  coord_flip() +
  labs(title = "Variable importance in LDA",
       subtitle = "Absolute LD1 coefficients",
       x = "", y = "Absolute coefficient") +
  theme_wine +
  theme(legend.position = "none")

# 5c. Scatter: actual vs predicted
results$Correct <- results$Actual == results$Predicted
p3 <- ggplot(results, aes(x = LD1, y = as.numeric(Actual),
                           color = Actual, shape = Correct)) +
  geom_jitter(alpha = 0.3, size = 1.2, height = 0.15) +
  scale_color_manual(values = PALETTE) +
  scale_shape_manual(values = c("TRUE" = 16, "FALSE" = 4),
                     labels = c("TRUE" = "Correct", "FALSE" = "Misclassified")) +
  scale_y_continuous(breaks = c(1, 2), labels = c("Red", "White")) +
  labs(title = "Classification results",
       subtitle = sprintf("Accuracy: %.2f%%", accuracy),
       x = "LD1 Score", y = "", shape = "") +
  theme_wine

# 5d. Confusion matrix heatmap
conf_df <- as.data.frame(confusion)
p4 <- ggplot(conf_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6, fontface = "bold",
            color = "white") +
  scale_fill_gradient(low = "#D4A017", high = "#A32D2D") +
  labs(title = "Confusion matrix",
       subtitle = "Rows = predicted, Cols = actual") +
  theme_wine +
  theme(legend.position = "none")

# ── 6. Combine and save ───────────────────────────────────────────────────────
combined <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    title    = "LDA — Red vs White Wine Classification by Acidity",
    subtitle = sprintf("Overall classification accuracy: %.2f%%", accuracy),
    theme    = theme(
      plot.title    = element_text(face = "bold", size = 14, hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray40")
    )
  )

ggsave("wine_lda_plots.png", combined, width = 14, height = 10, dpi = 150)
cat("Saved → wine_lda_plots.png\n")

print(combined)
cat("\nDone!\n")