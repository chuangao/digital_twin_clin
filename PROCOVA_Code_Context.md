# Digital Twin PROCOVA Project — Technical Specification
## Date: March 4, 2026

---

## 1. Project Overview

Build a **prognostic digital twin framework** using real-world evidence (RWE) to prospectively assess whether prognostic covariate adjustment in a Cox model can reduce variance (and therefore sample size) for a future clinical trial comparing a novel treatment against apixaban for stroke prevention in atrial fibrillation.

**Key innovation:** Unlike prior PROCOVA work that retrospectively re-analyzes completed trials, this framework **simulates trials from RWE** to prospectively quantify variance reduction before any trial data exist. Both the prognostic model training and the efficiency assessment use only RWE.

**Downstream application:** Librexia clinical program (milvexian Phase III), but the compound name is confidential in publications.

---

## 2. Data Source & Structure

**Database:** Optum Clinformatics Data Mart (CDM) — US administrative claims

### 2.1 Observed Data Columns (from screenshot of actual data)

```
patid              — patient ID (numeric, e.g., 33003293402)
index_dt           — index date (date, e.g., 2022-08-28)
age_index          — age at index (integer, e.g., 77, 86, 78)
gdr_cd             — gender code (F/M)
race               — race (W = White, etc.)
ethnicity          — ethnicity
bus                — insurance type (MCR = Medicare, COM = Commercial)

# Binary baseline comorbidity flags (0/1, measured in pre-index period):
bl_alc             — alcohol use disorder
bl_anybleed        — any bleeding history
bl_chf             — congestive heart failure
bl_cad             — coronary artery disease
bl_diab            — diabetes
bl_hyp             — hypertension
bl_liver           — liver disease
bl_majorbleed      — major bleeding history
bl_mi              — prior myocardial infarction
bl_pad             — peripheral artery disease
bl_pvd             — peripheral vascular disease
bl_renal           — renal disease
bl_stomach         — stomach/GI issues
bl_stroke          — prior stroke
bl_tia             — prior TIA

# Composite scores (integer):
bl_cci_score       — Charlson Comorbidity Index
bl_chads_vasc_score — CHA₂DS₂-VASc score
bl_hasbled_score   — HAS-BLED score

# Outcome (major bleeding endpoint shown in screenshot):
outcome_mb         — event indicator (0/1) for major bleeding
followup_time_mb   — follow-up time in days (e.g., 1129, 319, 405)
censor_reason      — reason for censoring (Disenroll, Discont, Major_bleed)
```

### 2.2 Notes on Endpoints

- The screenshot shows **major bleeding** as the outcome (outcome_mb, followup_time_mb)
- The abstract and slides reference **time-to-first-stroke** as the primary endpoint
- The framework should work for either endpoint — the code should be modular to handle different outcome columns
- There may be separate outcome columns for stroke (e.g., outcome_stroke, followup_time_stroke)

---

## 3. Study Design: 60/40 Split

### Part 1: Training Set (60%)
- Train RSF model using **cross-validation** within this partition
- Outcome (time-to-event) IS used — supervised learning
- Output: fitted RSF model that predicts Ŝ(t* | X) for any patient

### Part 2: Simulation Set (40%)
- Split further into two arms for trial simulation:
  - **Control arm (Group A):** Real apixaban outcomes unchanged
  - **Treatment arm (Group B):** Event times rescaled by 1/HR to simulate novel treatment effect
- Apply trained RSF from Part 1 to generate prognostic scores for Part 2 patients

---

## 4. Prognostic Model: Random Survival Forest

### 4.1 Training (on 60% partition, with cross-validation)

```r
library(randomForestSRC)

# Covariates to include (matching actual data columns):
covariates <- c("age_index", "gdr_cd", "race", "bus",
                "bl_alc", "bl_anybleed", "bl_chf", "bl_cad", "bl_diab",
                "bl_hyp", "bl_liver", "bl_majorbleed", "bl_mi", "bl_pad",
                "bl_pvd", "bl_renal", "bl_stomach", "bl_stroke", "bl_tia",
                "bl_cci_score", "bl_chads_vasc_score", "bl_hasbled_score")

# Formula construction
formula <- as.formula(paste("Surv(followup_time, outcome) ~",
                            paste(covariates, collapse = " + ")))

rsf_model <- rfsrc(formula, data = training_set,
                   ntree = 1000, nodesize = 15, nsplit = 10,
                   importance = TRUE)
```

### 4.2 Cross-Validation
- Use k-fold CV (e.g., 5-fold) within the 60% training partition
- Evaluate at each fold: C-index, time-dependent AUC, IBS
- Report average metrics across folds

### 4.3 Evaluation Metrics
- **Harrell's C-index:** Expected 0.65–0.75
- **Time-dependent AUC:** At landmark times (e.g., 12, 18, 24 months)
- **Integrated Brier Score (IBS)**
- **R² (explained variation):** Directly determines variance/sample size reduction. R² ≈ 0.20 → ~20% reduction
- **Variable importance:** Permutation-based from RSF

### 4.4 Prognostic Score Generation (applied to 40% partition)

```r
# Predict survival probability at landmark time t*
predictions <- predict(rsf_model, newdata = simulation_set)
prog_score <- predictions$survival[, which(predictions$time.interest == t_star)]
```

---

## 5. Trial Simulation (on 40% partition)

### 5.1 Creating Two Arms

```r
# Control arm: real outcomes, unchanged
control <- simulation_subset_A  # random half of 40% partition

# Treatment arm: rescale event times
treatment <- simulation_subset_B  # other half
# For patients with events: T_new = T_observed / HR
# For censored patients: keep censoring time unchanged
treatment$followup_time[treatment$outcome == 1] <-
  treatment$followup_time[treatment$outcome == 1] / HR
```

### 5.2 HR Scenarios
- **HR = 0.70** (optimistic) — 30% hazard reduction
- **HR = 0.80** (expected) — 20% hazard reduction
- **HR = 0.90** (conservative) — 10% hazard reduction

### 5.3 Sample Size Grid
- n per arm: 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000
- B = 1,000 bootstrap replications per (HR, n) combination

---

## 6. Analysis: Cox Model Comparison

### 6.1 Two Models Per Replicate

```r
# Unadjusted
cox_unadj <- coxph(Surv(time, event) ~ treatment,
                   data = trial_data, robust = TRUE)

# Prognostic-adjusted (PROCOVA)
cox_adj <- coxph(Surv(time, event) ~ treatment + prog_score,
                 data = trial_data, robust = TRUE)
```

**Key:** `robust = TRUE` uses the Lin & Wei (1989) sandwich variance estimator.

### 6.2 Metrics to Collect Per Replicate
- Estimated log(HR) for treatment (both models)
- SE of log(HR) (both models)
- p-value (one-sided)
- Whether p < 0.025 (rejection indicator)
- 95% CI for HR

### 6.3 Aggregated Metrics Across B Replicates

For each (HR, n) combination:
- **Power:** Proportion of replicates where p < 0.025 (one-sided α = 0.025)
- **Variance ratio:** Var(log HR_adjusted) / Var(log HR_unadjusted) — **this is the fundamental quantity**
  - The variance ratio directly equals the sample size reduction ratio
  - It is independent of the target power level
  - R² ≈ 1 − variance_ratio
- **Type I error (HR = 1.0):** Must verify rejection rate ≤ 0.025
- **Bias:** Mean of estimated log(HR) vs true log(HR)
- **MSE:** Mean squared error of log(HR) estimates

### 6.4 Key Conceptual Point: Variance Reduction vs Power

The sample size reduction is fundamentally about **variance reduction**, not power:
- If Var(adjusted) / Var(unadjusted) = 0.80, then 20% fewer patients needed for **any** target power
- Power is one way to operationalize it, but the variance ratio is the core quantity
- Do NOT frame results as "X% reduction at 80% power" — frame as "X% variance reduction, enabling equivalent sample size reduction"

---

## 7. Key Outputs / Deliverables

1. **Cohort characteristics table** — demographics and clinical features
2. **RSF model performance** — C-index, AUC, IBS, R², variable importance plot
3. **Simulation results table** — variance ratio at each (HR, n) for adjusted vs unadjusted
4. **Power curves figure** — power vs sample size, adjusted vs unadjusted, faceted by HR
5. **Type I error table** — rejection rates under HR = 1.0
6. **Bias plot** — distribution of estimated HR vs true HR

---

## 8. R Package Dependencies

```r
library(randomForestSRC)   # Random survival forest
library(survival)          # Cox PH models, Surv objects
library(pec)               # Prediction error curves, Brier score
library(timeROC)           # Time-dependent AUC
library(riskRegression)    # C-index, calibration
library(data.table)        # Fast data ops
library(dplyr)             # Tidyverse
library(ggplot2)           # Plots
library(patchwork)         # Multi-panel figures
library(future)            # Parallel computing
library(future.apply)      # Parallel apply
library(parallel)          # mclapply etc.
```

---

## 9. Regulatory & Methodological References

- **FDA (2023):** Guidance on covariate adjustment in RCTs — endorses pre-specified adjustment
- **EMA (2015):** Similar guidance (CPMP/EWP/2863/99)
- **Schuler et al. (2022):** Foundational PROCOVA paper — continuous outcomes only, used clinical trial data
- **Lin & Wei (1989):** Sandwich SE for Cox models
- **Ishwaran et al. (2008):** RSF methodology
- **Hansen (2008):** Prognostic analogue of the propensity score

**Our novelty vs prior PROCOVA work:**
1. **Survival endpoints** — extending from continuous/binary to time-to-event via Cox + sandwich SE
2. **RWE as training data** — using claims data rather than prior clinical trial data
3. **Prospective simulation** — simulating future trials from RWE to assess efficiency, not retrospectively re-analyzing completed trials

---

## 10. Presentation & Publication

### 10.1 Concept Slide (completed)
- **File:** Digital_Twin_Flow_Final.pptx (13.3" × 7.5" wide format)
- **Layout:** Left-to-right flow: Optum cohort → split up (training 60% → RSF with tree diagrams + predicted scores) and down (simulation 40% → apixaban controls + simulated treatment arm) → RSF drops prognostic score down to Cox model with survival curves → upward arrow to 15–25% variance reduction
- **Color scheme:** Red/white/neutral brand palette (#DC2626 red, #F5F5F4 light gray panels, #1C1917 black text)
- **Bottom strip:** Timeline — Feb 2026 cohort attribution → Mar 15 submit IEEE ICHI → Mar 20 simulation complete → Post-trial apply to Librexia

### 10.2 Workshop Abstract (completed)
- **Venue:** HDT 2026 Workshop at IEEE ICHI 2026, June 1, 2026, Minneapolis, MN
- **Format:** 1-page abstract (including references), IEEE Proceedings Format, double-blinded
- **Submission:** EasyChair (https://easychair.org/conferences/?conf=hdt2026)
- **Deadline:** March 15, 2026 | Notification: March 21 | Camera-ready: March 28
- **Title:** "A Real-World Evidence Framework for Building Prognostic Digital Twins to Optimize Clinical Trial Efficiency"
- **Key framing:** General framework with apixaban AF as example; no confidential compound names; prospective vs retrospective distinction; variance reduction (not power-specific)

---

## 11. Code Architecture Recommendations

```
project/
├── 001_cohort_construction.R     # Build apixaban AF cohort from Optum
├── 002_data_split.R              # 60/40 stratified split
├── 003_rsf_training.R            # Train RSF with cross-validation
├── 004_model_evaluation.R        # C-index, AUC, IBS, R², variable importance
├── 005_prognostic_scoring.R      # Apply RSF to simulation set
├── 006_trial_simulation.R        # Simulate two-arm trials, rescale events
├── 007_cox_analysis.R            # Fit unadjusted vs adjusted Cox models
├── 008_aggregate_results.R       # Power, variance ratio, bias, Type I error
├── 009_visualization.R           # Power curves, forest plots, etc.
├── config.R                      # Parameters: HR scenarios, n grid, B reps, t*
├── utils.R                       # Helper functions
└── results/                      # Output tables and figures
```

### Key Parameters (config.R)
```r
HR_SCENARIOS <- c(0.70, 0.80, 0.90, 1.00)  # 1.00 for Type I error check
N_PER_ARM <- c(500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000)
B_REPS <- 1000
ALPHA <- 0.025  # one-sided
T_STAR <- 365   # landmark time in days (12 months) — adjust as needed
TRAIN_FRACTION <- 0.60
RSF_NTREE <- 1000
RSF_NODESIZE <- 15
CV_FOLDS <- 5
```
