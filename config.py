"""
config.py
=========
Central configuration for the RV-Crypto forecasting pipeline (Framework points 9-51).

Edit the paths / dates / flags here rather than inside the module code.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"
for d in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Wide-format merged daily file produced by the earlier steps (1-8) of the framework.
DAILY_WIDE_CSV = DATA_DIR / "_Wide_format__MERGED_daily_RV_5m_BTC_ETH_XRP_LTC_common_sample.csv"

# OPTIONAL: raw 5-minute OHLCV panel, long format with columns
#   timestamp, coin, open, high, low, close, volume
# If this file exists, true Bipower Variation / jump-continuous decomposition
# (Framework point 15) is computed from intraday returns.
# If it does NOT exist, the pipeline falls back to a daily-RV-only jump proxy
# (see jumps.py) and prints a loud warning. Point this at your intraday file
# once you have it and re-run features.py / build_daily_dataset().
FIVE_MIN_LONG_CSV = DATA_DIR / "clean_5min_long.csv"  # may not exist

# ---------------------------------------------------------------------------
# Assets (suffixes must match the column suffixes in DAILY_WIDE_CSV)
# ---------------------------------------------------------------------------
COINS = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT"]
COIN_DISPLAY = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "LTCUSDT": "LTC", "XRPUSDT": "XRP"}

# ---------------------------------------------------------------------------
# Sample / forecasting design (Framework points 3, 28)
# ---------------------------------------------------------------------------
SAMPLE_START = "2018-01-01"
SAMPLE_END = "2026-06-20"

TRAIN_START = "2018-01-01"
INITIAL_TRAIN_END = "2019-06-30"      # initial training window end
VALID_START = "2019-07-01"
VALID_END = "2019-12-31"
FULL_TRAIN_END_BEFORE_OOS = "2019-12-31"  # first model is trained on 2018-01-01..2019-12-31
OOS_START = "2020-01-01"
OOS_END = "2026-06-20"

EPS_LOGRV = 1e-12          # epsilon for LogRV when RV = 0
RV_FLOOR = 1e-12           # positivity floor for forecasts

# ---------------------------------------------------------------------------
# HAR / lag structure (points 13-14)
# ---------------------------------------------------------------------------
LAGS = [1, 2, 3, 5, 10, 22]
WEEKLY_WINDOW = 5
MONTHLY_WINDOW = 22

# ---------------------------------------------------------------------------
# DL sequence lengths (point 26-27) and retrain cadence
# ---------------------------------------------------------------------------
DL_SEQ_LENGTHS = [22, 44, 66]
DL_RETRAIN_FREQ = "M"     # retrain monthly, forecast daily in between (point 28)
ML_RETRAIN_FREQ = "M"     # RF/XGB/LGBM: monthly retrain is the practical default;
                           # set to "D" for true daily retraining (much slower)

# ---------------------------------------------------------------------------
# Rolling windows (points 32, 37, 38)
# ---------------------------------------------------------------------------
HYBRID_LAMBDA_WINDOW = 180
HYBRID_LAMBDA_WINDOWS_ROBUST = [90, 180, 365]

ROLL_QLIKE_WINDOW = 180
ROLL_QLIKE_WINDOWS_ROBUST = [90, 180, 365]

MCS_WINDOW = 180
MCS_ALPHA = 0.10
MCS_UPDATE_FREQ = "W"     # "D" for daily rolling MCS, "W" for weekly (point 38 fallback)
MCS_N_BOOT = 500          # bootstrap replications for MCS p-values
MCS_BLOCK_LEN = 10        # stationary bootstrap expected block length

# ---------------------------------------------------------------------------
# DMCE weighting scheme (point 40): "inverse_qlike", "equal", "softmax"
# ---------------------------------------------------------------------------
DMCE_WEIGHT_SCHEME = "inverse_qlike"
SOFTMAX_ETA_GRID = [1, 5, 10]

# ---------------------------------------------------------------------------
# Model families (point 17, 31, 43)
# ---------------------------------------------------------------------------
ECON_MODELS = ["HAR_RV", "HAR_RV_J", "HAR_RV_CJ", "Realized_GARCH"]
AI_MODELS = ["RF", "XGBoost", "LightGBM", "LSTM", "TCN"]
STANDALONE_MODELS = ["Naive"] + ECON_MODELS + AI_MODELS
HYBRID_PAIRS = [(e, a) for e in ECON_MODELS for a in AI_MODELS]  # 20 hybrids

# ---------------------------------------------------------------------------
# Predefined crisis regimes (point 9)
# ---------------------------------------------------------------------------
CRISIS_REGIMES = {
    "CryptoCrash2018":   ("2018-01-01", "2018-12-31"),
    "COVID":             ("2020-03-01", "2020-05-31"),
    "ChinaBan":          ("2021-05-01", "2021-07-31"),
    "TerraLuna":         ("2022-05-01", "2022-06-30"),
    "CryptoWinter":      ("2022-01-01", "2022-12-31"),
    "FTX":               ("2022-11-01", "2022-12-31"),
    "ETF":               ("2024-01-01", "2024-03-31"),
    "Liquidation2025":   ("2025-10-01", "2025-10-31"),
    "Correction2026":    ("2026-02-01", "2026-06-20"),
}
# Windows used for the OOS "crisis evaluation" set (excludes the 2018 training-stress
# period, which largely precedes the OOS window and is descriptive only per point 9).
OOS_CRISIS_REGIMES = {k: v for k, v in CRISIS_REGIMES.items() if k != "CryptoCrash2018"}

RANDOM_SEED = 42
