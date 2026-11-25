"""Constants and default hyper-parameters (same values as in the notebooks)."""

VERSION = "mt3"
REG = ["hr", "rmssd"]
DISORDERS = ["Healthy", "NFLE", "RBD", "PLM", "Insomnia", "Narcolepsy"]     # SDB (4) and Bruxism (2): too few subjects
STAGES = ["N1", "N2", "N3", "REM"]
IGNORE = -100

# ---- training protocol ----
N_FOLDS = 5
SEQ_LEN = 12                      # 12 consecutive 5-s windows = 60 s context
EPOCHS, PATIENCE, MIN_EPOCHS = 60, 12, 15
BATCH_SEQ, LR, WD = 24, 5e-4, 0.05
PER_SUBJECT = 1500
USE_AUG = True
USE_STAGE_HEAD = True
EMA_DECAY = 0.999
SEED = 42
EVAL_BATCH_SEQ = 64
