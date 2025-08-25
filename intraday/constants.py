"""
Core constants and configuration used across the intraday trading system.

This module centralizes environment-dependent configuration, simulation mode
detection, common paths, and global thresholds. By consolidating these values,
other modules can import from a single place, reducing duplication and risk of
configuration drift.

Key responsibilities:
- Detect simulation mode from CLI args (without consuming user args).
- Define timezone objects and commonly used paths.
- Read environment variables for API credentials and tunable parameters.
- Provide default values where reasonable to ensure the app can initialize.
- Preload symbol maps that are used widely (e.g., ticker -> security_id).

Usage:
    from intraday.constants import (
        IST, SIMULATION_MODE, DB_PATH, COMBINED_DATA_DIR,
        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
        DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID,
        DEFAULT_TICK_SIZE,
        MIN_VOTES, ACCOUNT_SIZE, MAX_QUANTITY, MAX_DAILY_LOSS_PERCENT,
        VOLATILITY_THRESHOLD, API_RATE_LIMIT, BID_ASK_THRESHOLD,
        RELATIVE_VOLUME_THRESHOLD, MIN_PRICE_THRESHOLD, MAX_PRICE_THRESHOLD,
        QUOTE_API_RATE_LIMIT,
        TICKER_TO_ID_MAP,
    )

Notes:
- This module should not import internal modules that depend on it to avoid
  circular dependencies. Keep it focused on OS/env I/O and basic Python types.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import pandas as pd
import pytz
from dotenv import load_dotenv

load_dotenv()
# -----------------------------------------------------------------------------
# Timezone
# -----------------------------------------------------------------------------
IST = pytz.timezone("Asia/Kolkata")

# -----------------------------------------------------------------------------
# CLI and simulation mode configuration
# -----------------------------------------------------------------------------
# Detect simulation mode early based on argv to gate env checks and client init.
# We only parse known args and leave the rest for the main program.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--simulate", action="store_true")
_parser.add_argument("--mode", choices=["realtime", "simulate"])
_args, _ = _parser.parse_known_args()

SIMULATION_MODE: bool = bool(
    _args.simulate or (_args.mode and _args.mode == "simulate")
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
DB_PATH: str = os.getenv("ORDERS_DB_PATH", "orders.db")
COMBINED_DATA_DIR: str = os.path.join(os.getcwd(), "combined_data")

# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: str | None = os.getenv("TELEGRAM_CHAT_ID")
DHAN_ACCESS_TOKEN: str | None = os.getenv("DHAN_ACCESS_TOKEN")
DHAN_CLIENT_ID: str = os.getenv("DHAN_CLIENT_ID", "1000000003")

# Trading / quoting defaults
DEFAULT_TICK_SIZE: float = float(os.getenv("DEFAULT_TICK_SIZE", 0.05))

# -----------------------------------------------------------------------------
# Trading configuration (tunable via env)
# -----------------------------------------------------------------------------
MIN_VOTES: int = int(os.getenv("MIN_VOTES", 3))
ACCOUNT_SIZE: float = float(os.getenv("ACCOUNT_SIZE", 100000))
MAX_QUANTITY: int = int(os.getenv("MAX_QUANTITY", 2))
MAX_DAILY_LOSS_PERCENT: float = float(os.getenv("MAX_DAILY_LOSS", 0.02))
VOLATILITY_THRESHOLD: float = float(os.getenv("VOLATILITY_THRESHOLD", 0.02))
API_RATE_LIMIT: int = int(os.getenv("API_RATE_LIMIT", 100))
BID_ASK_THRESHOLD: int = int(os.getenv("BID_ASK_THRESHOLD", 500))
RELATIVE_VOLUME_THRESHOLD: float = float(os.getenv("RELATIVE_VOLUME_THRESHOLD", 1.2))
MIN_PRICE_THRESHOLD: float = float(os.getenv("MIN_PRICE_THRESHOLD", 50.0))
MAX_PRICE_THRESHOLD: float = float(os.getenv("MAX_PRICE_THRESHOLD", 5000.0))
QUOTE_API_RATE_LIMIT: int = int(
    os.getenv("QUOTE_API_RATE_LIMIT", 60)
)  # Default: 60/min

# -----------------------------------------------------------------------------
# Symbol map (optional convenience)
# -----------------------------------------------------------------------------
TICKER_TO_ID_MAP: Dict[str, int] = {}
try:
    nifty500_df = pd.read_csv("csv/ind_nifty500list.csv")
    TICKER_TO_ID_MAP = nifty500_df.set_index("ticker")["security_id"].to_dict()
except Exception:
    # Avoid hard failure; downstream modules should handle empty maps gracefully.
    TICKER_TO_ID_MAP = {}
