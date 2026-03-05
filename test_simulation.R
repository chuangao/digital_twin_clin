# test_simulation.R — Test the pipeline with simulated data
# Uses a single HR and sample size to verify the algorithm works

library(randomForestSRC)
library(survival)
library(ggplot2)

set.seed(42)

# ============================================================
# 1. Simulate a cohort (mimicking apixaban AF patients)
# ============================================================
n_total <- 10000

df <- data.frame(
  age_index          = round(rnorm(n_total, mean = 72, sd = 10)),
  gdr_cd             = factor(sample(c("M", "F"), n_total, replace = TRUE)),
  race               = factor(sample(c("W", "B", "A", "H"), n_total,
                               replace = TRUE, prob = c(0.7, 0.15, 0.08, 0.07))),
  ethnicity          = factor(sample(c("NH", "H"), n_total, replace = TRUE, prob = c(0.9, 0.1))),
  bl_alc             = rbinom(n_total, 1, 0.05),
  bl_anybleed        = rbinom(n_total, 1, 0.15),
  bl_chf             = rbinom(n_total, 1, 0.25),
  bl_cad             = rbinom(n_total, 1, 0.20),
  bl_diab            = rbinom(n_total, 1, 0.30),
  bl_hyp             = rbinom(n_total, 1, 0.70),
  bl_liver           = rbinom(n_total, 1, 0.05),
  bl_majorbleed      = rbinom(n_total, 1, 0.08),
  bl_mi              = rbinom(n_total, 1, 0.10),
  bl_pad             = rbinom(n_total, 1, 0.08),
  bl_pvd             = rbinom(n_total, 1, 0.10),
  bl_renal           = rbinom(n_total, 1, 0.15),
  bl_stomach         = rbinom(n_total, 1, 0.10),
  bl_stroke          = rbinom(n_total, 1, 0.12),
  bl_tia             = rbinom(n_total, 1, 0.06),
  bl_cci_score       = rpois(n_total, 3),
  bl_chads_vasc_score = rpois(n_total, 3),
  bl_hasbled_score   = rpois(n_total, 2)
)

# Generate survival outcome driven by some covariates
lp <- 0.03 * (df$age_index - 72) +
  0.4 * df$bl_chf +
  0.3 * df$bl_diab +
  0.5 * df$bl_majorbleed +
  0.2 * df$bl_renal +
  0.15 * df$bl_cci_score +
  0.1 * df$bl_chads_vasc_score

# Exponential survival times — target ~15-20% event rate
baseline_rate <- 0.0001
event_time <- rexp(n_total, rate = baseline_rate * exp(lp))

# Administrative censoring at ~3 years
censor_time <- runif(n_total, min = 300, max = 1100)

df$followup_time_mb <- pmin(event_time, censor_time)
df$outcome_mb       <- as.integer(event_time <= censor_time)

# Store latent event time and censor time for re-censoring during simulation
df$latent_event_time <- event_time
df$censor_time       <- censor_time

cat("Event rate:", mean(df$outcome_mb), "\n")
cat("Median follow-up:", median(df$followup_time_mb), "days\n")

# ============================================================
# 2. Run the pipeline with reduced parameters
# ============================================================
source("config.R")

# Override for speed
RSF_NTREE  <- 200
B_REPS     <- 200
HR_TEST    <- 0.80
N_ARM_TEST <- 1000

# Rename columns
df$time    <- df[[TIME_COL]]
df$outcome <- df[[OUTCOME_COL]]

# Split 60/40
n <- nrow(df)
train_idx <- sample(seq_len(n), size = floor(TRAIN_FRACTION * n))
train_df  <- df[train_idx, ]
simul_df  <- df[-train_idx, ]

cat("\nTraining:", nrow(train_df), "| Simulation:", nrow(simul_df), "\n")

# ============================================================
# 3. Cross-validation
# ============================================================
rsf_formula <- as.formula(
  paste("Surv(time, outcome) ~", paste(COVARIATES, collapse = " + "))
)

folds <- sample(rep(1:CV_FOLDS, length.out = nrow(train_df)))

cat("\n--- Cross-validation ---\n")
cv_cindex <- numeric(CV_FOLDS)
for (k in 1:CV_FOLDS) {
  rsf_k  <- rfsrc(rsf_formula, data = train_df[folds != k, ],
                   ntree = RSF_NTREE, nodesize = RSF_NODESIZE, nsplit = RSF_NSPLIT)
  pred_k <- predict(rsf_k, newdata = train_df[folds == k, ])
  conc   <- concordance(Surv(time, outcome) ~ I(-pred_k$predicted),
                        data = train_df[folds == k, ])
  cv_cindex[k] <- conc$concordance
  cat(sprintf("  Fold %d: C-index = %.4f\n", k, cv_cindex[k]))
}
cat(sprintf("Mean C-index: %.4f\n", mean(cv_cindex)))

# ============================================================
# 4. Final RSF + prognostic scores
# ============================================================
rsf_final  <- rfsrc(rsf_formula, data = train_df,
                    ntree = RSF_NTREE, nodesize = RSF_NODESIZE, nsplit = RSF_NSPLIT,
                    importance = TRUE)

pred_simul <- predict(rsf_final, newdata = simul_df)
simul_df$prog_score <- pred_simul$predicted

cat("\nTop 5 variable importance:\n")
print(round(sort(rsf_final$importance, decreasing = TRUE)[1:5], 4))

# ============================================================
# 5. Simulation: single (HR, n) scenario
# ============================================================
cat(sprintf("\n--- Simulation: HR=%.2f, n_per_arm=%d, B=%d ---\n",
            HR_TEST, N_ARM_TEST, B_REPS))

run_one <- function(simul_df, n_per_arm, hr) {
  idx   <- sample(nrow(simul_df), 2 * n_per_arm, replace = FALSE)
  trial <- simul_df[idx, ]
  trial$arm <- rep(c(0, 1), each = n_per_arm)

  # Rescale treatment arm: divide latent event time by HR, then re-censor
  is_trt <- trial$arm == 1
  new_event_time <- trial$latent_event_time
  new_event_time[is_trt] <- trial$latent_event_time[is_trt] / hr

  trial$time[is_trt]    <- pmin(new_event_time[is_trt], trial$censor_time[is_trt])
  trial$outcome[is_trt] <- as.integer(new_event_time[is_trt] <= trial$censor_time[is_trt])

  su <- summary(coxph(Surv(time, outcome) ~ arm, data = trial, robust = TRUE))$coefficients
  sa <- summary(coxph(Surv(time, outcome) ~ arm + prog_score, data = trial, robust = TRUE))$coefficients

  c(loghr_unadj = su["arm", "coef"],
    se_unadj    = su["arm", "robust se"],
    pval_unadj  = su["arm", "Pr(>|z|)"],
    loghr_adj   = sa["arm", "coef"],
    se_adj      = sa["arm", "robust se"],
    pval_adj    = sa["arm", "Pr(>|z|)"])
}

reps <- t(sapply(1:B_REPS, function(b) run_one(simul_df, N_ARM_TEST, HR_TEST)))
reps <- as.data.frame(reps)

# ============================================================
# 6. Results
# ============================================================
var_unadj <- var(reps$loghr_unadj)
var_adj   <- var(reps$loghr_adj)

cat("\n========== RESULTS ==========\n")
cat(sprintf("True log(HR):       %.4f\n", log(HR_TEST)))
cat(sprintf("Mean log(HR) unadj: %.4f  (bias: %.4f)\n",
            mean(reps$loghr_unadj), mean(reps$loghr_unadj) - log(HR_TEST)))
cat(sprintf("Mean log(HR) adj:   %.4f  (bias: %.4f)\n",
            mean(reps$loghr_adj), mean(reps$loghr_adj) - log(HR_TEST)))
cat(sprintf("\nVar(unadjusted): %.6f\n", var_unadj))
cat(sprintf("Var(adjusted):   %.6f\n", var_adj))
cat(sprintf("Variance ratio:  %.4f\n", var_adj / var_unadj))
cat(sprintf("Variance reduction: %.1f%%\n", (1 - var_adj / var_unadj) * 100))
cat(sprintf("\nPower (one-sided, alpha=0.025):\n"))
cat(sprintf("  Unadjusted: %.1f%%\n",
            100 * mean(reps$loghr_unadj < 0 & reps$pval_unadj / 2 < 0.025)))
cat(sprintf("  Adjusted:   %.1f%%\n",
            100 * mean(reps$loghr_adj < 0 & reps$pval_adj / 2 < 0.025)))

# Quick density plot
p <- ggplot() +
  geom_density(aes(x = reps$loghr_unadj, fill = "Unadjusted"), alpha = 0.4) +
  geom_density(aes(x = reps$loghr_adj, fill = "Adjusted"), alpha = 0.4) +
  geom_vline(xintercept = log(HR_TEST), linetype = "dashed") +
  scale_fill_manual(values = c("Unadjusted" = "#1C1917", "Adjusted" = "#DC2626")) +
  labs(x = "Estimated log(HR)", y = "Density",
       title = "Distribution of treatment effect estimates",
       subtitle = sprintf("HR=%.2f, n=%d/arm, B=%d", HR_TEST, N_ARM_TEST, B_REPS),
       fill = NULL) +
  theme_minimal()

ggsave("results/test_loghr_distribution.png", p, width = 8, height = 5, dpi = 150)
cat("\nPlot saved to results/test_loghr_distribution.png\n")
cat("Done.\n")
