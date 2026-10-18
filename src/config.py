"""Central configuration for Federated ScamShield.

Every hyperparameter, path and constant used anywhere in the pipeline lives
here. Other modules import from this file and must not hard-code numbers.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility / hardware
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
DEVICE = "cpu"                      # CPU only - LinuxONE / s390x has no GPU
TORCH_NUM_THREADS = os.cpu_count() or 1

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATA_FILE = DATA_DIR / "PaySim.csv"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
MODEL_LOCAL_TEMPLATE = "model_local_bank{bank}.pt"
MODEL_CENTRAL_PATH = ARTIFACTS_DIR / "model_central.pt"
MODEL_FED_PATH = ARTIFACTS_DIR / "model_federated.pt"
MODEL_FED_DP_PATH = ARTIFACTS_DIR / "model_federated_dp.pt"
ONNX_PATH = ARTIFACTS_DIR / "scamshield_model.onnx"
WATCHLIST_DB_PATH = ARTIFACTS_DIR / "watchlist.db"
WATCHLIST_DP_DB_PATH = ARTIFACTS_DIR / "watchlist_dp.db"

METRICS_PATH = RESULTS_DIR / "metrics.json"
FED_ROUNDS_PATH = RESULTS_DIR / "fed_rounds.json"
FED_DP_ROUNDS_PATH = RESULTS_DIR / "fed_dp_rounds.json"
RING_GROUND_TRUTH_PATH = RESULTS_DIR / "ring_ground_truth.json"
PLOT_PR_AUC_PATH = RESULTS_DIR / "pr_auc_comparison.png"
PLOT_CONVERGENCE_PATH = RESULTS_DIR / "fed_convergence.png"
PLOT_CONFUSION_B_PATH = RESULTS_DIR / "confusion_matrix_B.png"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
]
LABEL_COLUMN = "isFraud"
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]   # the only types that contain fraud
TYPE_ENCODING = {"TRANSFER": 0, "CASH_OUT": 1}
STEPS_PER_DAY = 24
TOTAL_STEPS = 744

# ---------------------------------------------------------------------------
# Banks and time split
# ---------------------------------------------------------------------------
NUM_BANKS = 4
BANK_RATIOS = [0.4, 0.3, 0.2, 0.1]
HASH_BUCKETS = 100                  # MD5(nameOrig) -> int % 100
HASH_HEX_CHARS = 8                  # first 8 hex chars of the digest
TRAIN_TEST_SPLIT_STEP = 600         # step < 600 = train, step >= 600 = test

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
AMOUNT_RATIO_CLIP = (0.0, 10.0)
BALANCE_RATIO_DEST_CLIP = (0.0, 100.0)
WHOLE_AMOUNT_MODULUS = 1000
AMOUNT_MATCH_TOLERANCE = 0.01       # currency units for "amount == balance"
CASHOUT_WINDOW_STEPS = 24           # TRANSFER dest cashes out within N hours
AMOUNT_MATCH_WINDOW_STEPS = 1       # TRANSFER/CASH_OUT of same amount within N hours
DEST_COUNT_WINDOW_STEPS = 24        # trailing window for dest_txn_count_24h

# Monetary columns are heavy-tailed (up to ~1e8). A signed log1p before the
# StandardScaler keeps the MLP well conditioned. Set False to scale raw values.
LOG_TRANSFORM_MONETARY = True
MONETARY_COLUMNS = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
    "newbalanceDest", "balance_delta_org", "balance_delta_dest",
    "error_balance_org", "error_balance_dest",
]
SCALED_CLIP = 10.0                  # clip standardised values to +/- this

# day_of_month is always computed, but by default it is NOT fed to the model:
# train covers days 0-24 and test days 25-30, so the model would only ever see
# out-of-range values at test time, and PaySim's fraud rate drifts strongly
# with time (legit volume falls off, fraud does not). Set True to include it.
USE_DAY_OF_MONTH = False

ALL_FEATURE_COLUMNS = [
    # 1. basic
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    # 2-16. engineered
    "amount_ratio",
    "balance_delta_org",
    "balance_delta_dest",
    "orig_balance_zero",
    "dest_balance_zero",
    "amount_equals_balance",
    "is_whole_amount",
    "hour_of_day",
    "day_of_month",
    "amount_log",
    "balance_ratio_dest",
    "transfer_to_cashout_flag",
    "error_balance_org",
    "error_balance_dest",
    "type_encoded",
    # extras
    "hour_sin",
    "hour_cos",
    "amount_matched_pair",
    "dest_txn_count_24h",
]
FEATURE_COLUMNS = [
    c for c in ALL_FEATURE_COLUMNS if USE_DAY_OF_MONTH or c != "day_of_month"
]

# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------
HIDDEN_1 = 64
HIDDEN_2 = 32
DROPOUT = 0.3
BATCH_SIZE = 2048
EVAL_BATCH_SIZE = 16384
LR = 1e-3
WEIGHT_DECAY = 1e-5
POS_WEIGHT_MODE = "ratio"           # "ratio" = neg/pos, "sqrt" = sqrt(neg/pos)

LOCAL_TRAIN_EPOCHS = 10             # Setup A
CENTRAL_EPOCHS = 10                 # Setup C

# ---------------------------------------------------------------------------
# Federated learning (Setup B)
# ---------------------------------------------------------------------------
FED_ROUNDS = 5
LOCAL_EPOCHS = 2
FEDPROX_MU = 0.0                    # > 0 enables FedProx (e.g. 0.01)
FED_LR_WARMUP_ROUNDS = 0            # > 0 ramps LR linearly over the first rounds

# ---------------------------------------------------------------------------
# Differential privacy (Setup E)
# ---------------------------------------------------------------------------
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 0.5
DP_DELTA = 1e-5
DP_RDP_ORDERS = list(range(2, 128))
# Per-sample gradients are undefined under BatchNorm (it mixes samples in a
# batch), so the DP model swaps BatchNorm for LayerNorm.
DP_NORM = "layer"

# ---------------------------------------------------------------------------
# Watchlist / mule rings (Setup D)
# ---------------------------------------------------------------------------
HMAC_KEY = "scamshield-datathon-2026"   # demo only, never use real keys
WATCHLIST_SCORE_THRESHOLD = 0.7
WATCHLIST_TRACE_HOPS = 5                # follow money out of flagged accounts this
                                        # many hops (0 = high-risk edges only)
SIM_START_DATE = "2026-01-01"           # step 0 maps to this date for flag_date
RING_MATCH_FRACTION = 0.5               # share of a ring's accounts that must
                                        # land in one detected component

NUM_INJECTED_RINGS = 20
RING_MIN_SIZE = 3
RING_MAX_SIZE = 6
RING_AMOUNT_RANGE = (50_000.0, 500_000.0)
RING_COMMISSION_RANGE = (0.01, 0.05)    # each mule keeps 1-5% before forwarding
RING_MAX_HOP_DELAY_STEPS = 2            # hours between consecutive hops
RING_DEST_UNREPORTED_PROB = 0.5         # P(dest balances recorded as 0/0)
RING_ACCOUNT_PREFIX = "C9"
RING_ACCOUNT_DIGITS = 9

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
TARGET_FPR = 0.01
