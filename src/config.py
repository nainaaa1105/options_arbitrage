from pathlib import Path
import os

from dotenv import load_dotenv



# PROJECT PATHS


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"

UPSTOX_RAW_DIR = RAW_DIR / "upstox"

PROCESSED_DIR = DATA_DIR / "processed"


# Create directories automatically
UPSTOX_RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)



# ENVIRONMENT


load_dotenv(BASE_DIR / ".env")

UPSTOX_ACCESS_TOKEN = os.getenv(
    "UPSTOX_ACCESS_TOKEN"
)

if not UPSTOX_ACCESS_TOKEN:
    raise RuntimeError(
        "UPSTOX_ACCESS_TOKEN not found in .env"
    )



# UPSTOX


BASE_URL = "https://api.upstox.com/v2"

NIFTY_KEY = "NSE_INDEX|Nifty 50"



# DATA COLLECTION SETTINGS


# Number of strikes on EACH side of ATM
STRIKES_EACH_SIDE = 10

# Therefore:
# 10 below + ATM + 10 above = 21 strikes
EXPECTED_STRIKES = (
    STRIKES_EACH_SIDE * 2
) + 1

# CE + PE for every strike
EXPECTED_CONTRACTS = (
    EXPECTED_STRIKES * 2
)


# Option candles
OPTION_TIMEFRAME = "5minute"

# NIFTY candles
NIFTY_TIMEFRAME = "1minute"



# HISTORICAL WINDOW


# Exactly 5 trading sessions ending on expiry
HISTORICAL_TRADING_SESSIONS = 5



# API SETTINGS


REQUEST_DELAY_SECONDS = 0.15

MAX_RETRIES = 3



# TEST EXPIRY


TEST_EXPIRY = "2025-09-23"
# Initial research assumption.
# This will later be replaceable by a historical
# risk-free-rate series.
RISK_FREE_RATE = 0.06

# Expiry is treated as 15:30 IST on the expiry date.
EXPIRY_HOUR = 15
EXPIRY_MINUTE = 30

# IV solver bounds.
IV_MIN = 1e-6
IV_MAX = 5.0

# Numerical tolerance for IV solver.
IV_SOLVER_XTOL = 1e-8

# Minimum option price considered usable for IV.
MIN_OPTION_PRICE = 1e-6

# Maximum reasonable IV used for the first research dataset.
# 500% annualized volatility is deliberately generous.
MAX_REASONABLE_IV = 5.0

# Output files
FEATURE_DATASET_FILE = (
    PROCESSED_DIR / "options_features.parquet"
)