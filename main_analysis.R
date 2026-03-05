# main_analysis.R — PROCOVA Digital Twin Framework
# Prognostic covariate adjustment for clinical trial efficiency

library(randomForestSRC)
library(survival)
library(pec)
library(timeROC)
library(data.table)
library(ggplot2)
library(future.apply)

source("config.R")

set.seed(SEED)

# ============================================================
# 0. Load data
# ============================================================
# Assumes df is a data.frame with columns listed in config.R
# Required columns: covariates + outcome + follow-up time + latent_event_time + censor_time
# For real data: latent_event_time = event time for events, Inf for censored
#                censor_time = potential censoring time (end of enrollment - index date)
# Uncomment and modify the line below for your data source:
# df <- readRDS("data/cohort.rds")

# Rename outcome columns for convenience
df$time    <- df[[TIME_COL]]
df$outcome <- df[[OUTCOME_COL]]

# Drop rows with missing time or outcome
df <- df[!is.na(df$time) & !is.na(df$outcome) & df$time > 0, ]

# ============================================================
# 1. Split: 60% training / 40% simulation
# ============================================================
n <- nrow(df)
train_idx <- sample(seq_len(n), size = floor(TRAIN_FRACTION * n))

train_df <- df[train_idx, ]
simul_df <- df[-train_idx, ]

cat("Training set:", nrow(train_df), "patients\n")
cat("Simulation set:", nrow(simul_df), "patients\n")

# ============================================================
# 2. RSF: 5-fold cross-validation on training set
# ============================================================
folds <- sample(rep(1:CV_FOLDS, length.out = nrow(train_df)))

cv_results <- data.frame(
  fold    = integer(),
  c_index = numeric(),
  stringsAsFactors = FALSE
)

rsf_formula <- as.formula(
  paste("Surv(time, outcome) ~", paste(COVARIATES, collapse = " + "))
)

cat("\n--- Cross-validation ---\n")
for (k in 1:CV_FOLDS) {
  cv_train <- train_df[folds != k, ]
  cv_test  <- train_df[folds == k, ]

  rsf_k <- rfsrc(rsf_formula, data = cv_train,
                  ntree = RSF_NTREE, nodesize = RSF_NODESIZE,
                  nsplit = RSF_NSPLIT, importance = FALSE)

  pred_k <- predict(rsf_k, newdata = cv_test)

  # C-index: higher ensemble mortality should correspond to higher risk
  conc <- concordance(Surv(time, outcome) ~ I(-pred_k$predicted),
                          data = cv_test)
  c_idx <- conc$concordance

  cv_results <- rbind(cv_results, data.frame(fold = k, c_index = c_idx))
  cat(sprintf("  Fold %d: C-index = %.4f\n", k, c_idx))
}

cat(sprintf("\nMean CV C-index: %.4f (SD: %.4f)\n",
            mean(cv_results$c_index), sd(cv_results$c_index)))

# ============================================================
# 3. Refit RSF on full training set
# ============================================================
cat("\n--- Fitting final RSF on full training set ---\n")
rsf_final <- rfsrc(rsf_formula, data = train_df,
                   ntree = RSF_NTREE, nodesize = RSF_NODESIZE,
                   nsplit = RSF_NSPLIT, importance = TRUE)

# Variable importance
vimp <- sort(rsf_final$importance, decreasing = TRUE)
cat("\nTop 10 variable importance:\n")
print(round(head(vimp, 10), 4))

# ============================================================
# 4. Generate prognostic scores for simulation set
# ============================================================
pred_simul <- predict(rsf_final, newdata = simul_df)
simul_df$prog_score <- pred_simul$predicted

# ============================================================
# 5. Trial simulation
# ============================================================
cat("\n--- Trial simulation ---\n")

plan(multisession)  # parallel backend

run_one_replicate <- function(simul_df, n_per_arm, hr) {
  pool_size <- nrow(simul_df)
  total_n   <- 2 * n_per_arm

  # Sample without replacement
  idx <- sample(seq_len(pool_size), size = total_n, replace = FALSE)
  trial <- simul_df[idx, ]

  # Assign arms: first half = control, second half = treatment
  trial$arm <- rep(c(0, 1), each = n_per_arm)

  # Rescale treatment arm: divide latent event time by HR, then re-censor
  # Requires columns: latent_event_time (original event time), censor_time
  is_trt <- trial$arm == 1
  new_event_time <- trial$latent_event_time
  new_event_time[is_trt] <- trial$latent_event_time[is_trt] / hr

  trial$time[is_trt]    <- pmin(new_event_time[is_trt], trial$censor_time[is_trt])
  trial$outcome[is_trt] <- as.integer(new_event_time[is_trt] <= trial$censor_time[is_trt])

  # Cox unadjusted
  cox_unadj <- coxph(Surv(time, outcome) ~ arm, data = trial, robust = TRUE)
  su <- summary(cox_unadj)$coefficients

  # Cox adjusted with prognostic score
  cox_adj <- coxph(Surv(time, outcome) ~ arm + prog_score,
                   data = trial, robust = TRUE)
  sa <- summary(cox_adj)$coefficients

  data.frame(
    loghr_unadj = su["arm", "coef"],
    se_unadj    = su["arm", "robust se"],
    pval_unadj  = su["arm", "Pr(>|z|)"],
    loghr_adj   = sa["arm", "coef"],
    se_adj      = sa["arm", "robust se"],
    pval_adj    = sa["arm", "Pr(>|z|)"]
  )
}

# Run simulation grid
results_list <- list()

for (hr in HR_SCENARIOS) {
  for (n_arm in N_PER_ARM) {
    if (2 * n_arm > nrow(simul_df)) {
      cat(sprintf("  Skipping HR=%.2f, n=%d (not enough patients)\n", hr, n_arm))
      next
    }

    cat(sprintf("  HR=%.2f, n_per_arm=%d ...\n", hr, n_arm))

    reps <- future_lapply(1:B_REPS, function(b) {
      run_one_replicate(simul_df, n_arm, hr)
    }, future.seed = TRUE)

    reps_df <- do.call(rbind, reps)

    # Aggregate
    var_unadj <- var(reps_df$loghr_unadj)
    var_adj   <- var(reps_df$loghr_adj)

    results_list[[length(results_list) + 1]] <- data.frame(
      hr          = hr,
      n_per_arm   = n_arm,
      true_loghr  = log(hr),
      mean_loghr_unadj = mean(reps_df$loghr_unadj),
      mean_loghr_adj   = mean(reps_df$loghr_adj),
      var_unadj   = var_unadj,
      var_adj     = var_adj,
      var_ratio   = var_adj / var_unadj,
      pct_reduction = (1 - var_adj / var_unadj) * 100,
      power_unadj = mean(reps_df$loghr_unadj < 0 & reps_df$pval_unadj / 2 < ALPHA),
      power_adj   = mean(reps_df$loghr_adj < 0 & reps_df$pval_adj / 2 < ALPHA),
      bias_unadj  = mean(reps_df$loghr_unadj) - log(hr),
      bias_adj    = mean(reps_df$loghr_adj) - log(hr)
    )
  }
}

results <- do.call(rbind, results_list)

cat("\n--- Results ---\n")
print(results, digits = 3)

# ============================================================
# 6. Save results
# ============================================================
dir.create("results", showWarnings = FALSE)
saveRDS(results, "results/simulation_results.rds")
saveRDS(rsf_final, "results/rsf_final_model.rds")
write.csv(results, "results/simulation_results.csv", row.names = FALSE)

# ============================================================
# 7. Plots
# ============================================================

# --- Power curves ---
results_plot <- results[results$hr != 1.0, ]

p_power <- ggplot(results_plot, aes(x = n_per_arm)) +
  geom_line(aes(y = power_unadj, color = "Unadjusted")) +
  geom_line(aes(y = power_adj, color = "PROCOVA-adjusted")) +
  geom_point(aes(y = power_unadj, color = "Unadjusted")) +
  geom_point(aes(y = power_adj, color = "PROCOVA-adjusted")) +
  geom_hline(yintercept = 0.80, linetype = "dashed", color = "gray50") +
  facet_wrap(~ paste("HR =", hr)) +
  scale_color_manual(values = c("Unadjusted" = "#1C1917",
                                "PROCOVA-adjusted" = "#DC2626")) +
  labs(x = "Sample size per arm", y = "Power",
       title = "Power: Unadjusted vs PROCOVA-adjusted Cox model",
       color = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("results/power_curves.png", p_power, width = 10, height = 6, dpi = 300)

# --- Variance reduction ---
p_var <- ggplot(results_plot, aes(x = n_per_arm, y = pct_reduction)) +
  geom_line() +
  geom_point() +
  facet_wrap(~ paste("HR =", hr)) +
  labs(x = "Sample size per arm", y = "Variance reduction (%)",
       title = "Variance reduction from prognostic adjustment") +
  theme_minimal()

ggsave("results/variance_reduction.png", p_var, width = 10, height = 6, dpi = 300)

# --- Type I error (HR = 1.0) ---
type1 <- results[results$hr == 1.0, ]
if (nrow(type1) > 0) {
  cat("\n--- Type I Error (HR = 1.0) ---\n")
  print(type1[, c("n_per_arm", "power_unadj", "power_adj")], digits = 3)
}

cat("\nDone. Results saved to results/\n")
