import os
from pathlib import Path
from dotenv import load_dotenv



# PROJECT PATHS


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
RAW_UPSTOX_DIR = RAW_DIR / "upstox"
RAW_NSE_DIR = RAW_DIR / "nse"

PROCESSED_DIR = DATA_DIR / "processed"


# Create directories automatically
RAW_UPSTOX_DIR.mkdir(parents=True, exist_ok=True)
RAW_NSE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)



# ENVIRONMENT


load_dotenv(PROJECT_ROOT / ".env")

UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")



# UPSTOX


BASE_URL = "https://api.upstox.com/v2"

NIFTY_KEY = "NSE_INDEX|Nifty 50"



# DATA COLLECTION SETTINGS


# Number of strikes on EACH side of ATM.
#
# 10 means:
#
# ATM - 10 strikes
# ATM
# ATM + 10 strikes
#
# Total = 21 strikes
#
STRIKES_EACH_SIDE = 10


# Option candle timeframe
OPTION_TIMEFRAME = "5minute"

# NIFTY is collected at 1 minute because
# Upstox historical NIFTY data does not support 5minute
# through the endpoint we are currently using.
NIFTY_TIMEFRAME = "1minute"



# TEST SETTINGS


TEST_EXPIRY = "2025-09-23"

# Five trading-day test window
TEST_FROM_DATE = "2025-09-18"
TEST_TO_DATE = "2025-09-23"



# LARGE COLLECTION SETTINGS


# Number of trading days before expiry to collect.
#
# 5 is a good starting point.
#
# We can later change this to 10.
HISTORICAL_DAYS_PER_EXPIRY = 5


# Small delay between API requests.
#
# Helps avoid hammering the API.
REQUEST_DELAY_SECONDS = 0.15


# Number of retries for temporary API errors.
MAX_RETRIES = 3



# VALIDATION


# We expect:
# 21 strikes × 2 option types = 42 contracts
EXPECTED_STRIKES = (STRIKES_EACH_SIDE * 2) + 1

EXPECTED_CONTRACTS = EXPECTED_STRIKES * 2