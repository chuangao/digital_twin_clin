# config.R — Parameters for PROCOVA Digital Twin Framework

# --- Data columns ---
OUTCOME_COL <- "outcome_mb"
TIME_COL    <- "followup_time_mb"

COVARIATES <- c(
  "age_index", "gdr_cd", "race", "ethnicity",
  "bl_alc", "bl_anybleed", "bl_chf", "bl_cad", "bl_diab",
  "bl_hyp", "bl_liver", "bl_majorbleed", "bl_mi", "bl_pad",
  "bl_pvd", "bl_renal", "bl_stomach", "bl_stroke", "bl_tia",
  "bl_cci_score", "bl_chads_vasc_score", "bl_hasbled_score"
)

# --- Train/simulation split ---
TRAIN_FRACTION <- 0.60
SEED           <- 42

# --- RSF parameters ---
RSF_NTREE    <- 1000
RSF_NODESIZE <- 15
RSF_NSPLIT   <- 10
CV_FOLDS     <- 5

# --- Trial simulation ---
HR_SCENARIOS <- c(0.70, 0.80, 0.90, 1.00)  # 1.00 for Type I error
N_PER_ARM    <- c(500, 750, 1000, 1250, 1500, 2000, 2500)
B_REPS       <- 1000
ALPHA        <- 0.025  # one-sided
