"""
Configuration module for the trading system.
Contains all environment variables, constants, and configuration settings.
"""

import os
import argparse
from datetime import time
import pytz
from dotenv import load_dotenv

load_dotenv()

# Parse command line arguments early
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--simulate", action="store_true")
parser.add_argument("--mode", choices=["realtime", "simulate"])
args, _ = parser.parse_known_args()

# Simulation mode configuration
SIMULATION_MODE = args.simulate or (args.mode and args.mode == "simulate")

# Timezone
IST = pytz.timezone("Asia/Kolkata")

# Directory configurations
COMBINED_DATA_DIR = os.path.join(os.getcwd(), "combined_data")

# Environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "1000000003")

# Market configuration
MARKET_OPEN_TIME = time(9, 15)  # From CONFIG
MARKET_CLOSE_TIME = time(15, 30)  # From CONFIG
TRADING_END_TIME = time.fromisoformat(os.getenv("TRADING_END", "15:20:00"))
FORCE_CLOSE_TIME = time.fromisoformat(os.getenv("FORCE_CLOSE", "15:15:00"))
SQUARE_OFF_TIME = time.fromisoformat(os.getenv("SQUARE_OFF_TIME", "15:16:00"))

# Trading configuration
MIN_VOTES = int(os.getenv("MIN_VOTES", 1))
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", 100000))
MAX_QUANTITY = int(os.getenv("MAX_QUANTITY", 2))
MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS", 0.02))
VOLATILITY_THRESHOLD = float(os.getenv("VOLATILITY_THRESHOLD", 0.02))
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", 100))
BID_ASK_THRESHOLD = int(os.getenv("BID_ASK_THRESHOLD", 500))
RELATIVE_VOLUME_THRESHOLD = float(os.getenv("RELATIVE_VOLUME_THRESHOLD", 1.2))
MIN_PRICE_THRESHOLD = float(os.getenv("MIN_PRICE_THRESHOLD", 50.0))
MAX_PRICE_THRESHOLD = float(os.getenv("MAX_PRICE_THRESHOLD", 5000.0))
QUOTE_API_RATE_LIMIT = int(os.getenv("QUOTE_API_RATE_LIMIT", 60))

# Default values
DEFAULT_TICK_SIZE = 0.05  # Default tick size for NSE equities
