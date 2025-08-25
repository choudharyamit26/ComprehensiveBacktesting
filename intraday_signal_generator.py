import asyncio
import pickle
import pandas as pd
import os
import logging
from datetime import date, datetime, timedelta, time, timezone
import pytz
from retrying import retry
import ast
import pandas_ta as ta
import aiohttp
import sys
import traceback
from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from cachetools import TTLCache
from comprehensive_backtesting.data import init_dhan_client
from get_llm_signal import get_llm_signal, get_openrouter_llm_signal
from intraday.utils import get_index_signal_dhan_api, get_sector_security_id
from live_data import (
    get_combined_data_with_persistent_live,
    read_live_data_from_csv,
    initialize_live_data_from_config,
    CONFIG,
)

from historical_cache import clear_cache_for_today, get_cache_info
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import sqlite3
import asyncio
import aiofiles
import json
from datetime import datetime
from typing import Optional, Dict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import aiohttp

# Database setup
DB_PATH = "orders.db"

# CLI and simulation mode configuration
import argparse

# Detect simulation mode early based on argv to gate env checks and client init
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--simulate", action="store_true")
parser.add_argument("--mode", choices=["realtime", "simulate"])
args, _ = parser.parse_known_args()

SIMULATION_MODE = args.simulate or (args.mode and args.mode == "simulate")

# Directory for offline combined data
COMBINED_DATA_DIR = os.path.join(os.getcwd(), "combined_data")
# Cache for loaded simulation dataframes keyed by security_id
SIM_DATA_CACHE: Dict[int, pd.DataFrame] = {}

# Initialize Dhan client
if not SIMULATION_MODE:
    dhan = init_dhan_client()
else:

    class _MockDhan:
        def get_fund_limits(self):
            return {"data": {"availabelBalance": 1e9}}

        def get_positions(self):
            return {"status": "success", "data": []}

    dhan = _MockDhan()

IST = pytz.timezone("Asia/Kolkata")
import asyncio
import json
import struct
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime
import logging
import pandas as pd
import websocket
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Global WebSocket client instance

IST = pytz.timezone("Asia/Kolkata")


class DateTimeFormatter(logging.Formatter):
    """Custom formatter with IST datetime"""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=IST)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " IST"


def setup_logging_inline():
    """Improved logging setup with proper initialization checks"""

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    formatter = DateTimeFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    root_logger = logging.getLogger()

    # Clear existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler with proper error handling
    try:
        file_handler = logging.FileHandler(
            os.path.join("logs", "trading_system.log"), mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
    except PermissionError:
        logger.error("Permission denied for logs directory, using current directory")
        file_handler = logging.FileHandler(
            "trading_system.log", mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
    except Exception as e:
        logger.error(f"File handler creation failed: {e}")
        file_handler = None

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)

    # Create specialized loggers
    logger = logging.getLogger("quant_trader")
    logger.setLevel(logging.INFO)

    trade_logger = logging.getLogger("trade_execution")
    trade_logger.setLevel(logging.INFO)

    # Clear existing trade logger handlers
    for handler in trade_logger.handlers[:]:
        trade_logger.removeHandler(handler)

    # Trade-specific file handler
    try:
        trade_file_handler = logging.FileHandler(
            os.path.join("logs", "trades.log"), mode="a", encoding="utf-8"
        )
        trade_file_handler.setFormatter(formatter)
        trade_logger.addHandler(trade_file_handler)
        trade_logger.propagate = False
    except Exception as e:
        logger.error(f"Trade log handler failed: {e}")
        trade_logger.propagate = True

    return logger, trade_logger


# Environment configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "1000000003")
DEFAULT_TICK_SIZE = 0.05  # Default tick size for NSE equities

logger, trade_logger = setup_logging_inline()

if (
    not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_ACCESS_TOKEN])
    and not SIMULATION_MODE
):
    logger.critical("Missing required environment variables")
    raise EnvironmentError("Required environment variables not set")
elif SIMULATION_MODE and not all(
    [TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_ACCESS_TOKEN]
):
    logger.warning(
        "Simulation mode: skipping checks for TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_ACCESS_TOKEN"
    )

# Market configuration
MARKET_OPEN_TIME = CONFIG["MARKET_OPEN"]
MARKET_CLOSE_TIME = CONFIG["MARKET_CLOSE"]
TRADING_END_TIME = time.fromisoformat(os.getenv("TRADING_END", "15:20:00"))
FORCE_CLOSE_TIME = time.fromisoformat(os.getenv("FORCE_CLOSE", "15:15:00"))
SQUARE_OFF_TIME = time.fromisoformat(os.getenv("SQUARE_OFF_TIME", "15:16:00"))

# Trading configuration
MIN_VOTES = int(os.getenv("MIN_VOTES", 3))
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", 100000))
MAX_QUANTITY = int(os.getenv("MAX_QUANTITY", 2))
MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS", 0.02))
VOLATILITY_THRESHOLD = float(os.getenv("VOLATILITY_THRESHOLD", 0.02))
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", 100))
BID_ASK_THRESHOLD = int(os.getenv("BID_ASK_THRESHOLD", 500))
RELATIVE_VOLUME_THRESHOLD = float(os.getenv("RELATIVE_VOLUME_THRESHOLD", 1.2))
MIN_PRICE_THRESHOLD = float(os.getenv("MIN_PRICE_THRESHOLD", 50.0))
MAX_PRICE_THRESHOLD = float(os.getenv("MAX_PRICE_THRESHOLD", 5000.0))
QUOTE_API_RATE_LIMIT = int(os.getenv("QUOTE_API_RATE_LIMIT", 60))  # Default: 60/min

# Initialize thread pool and locks
dhan_lock = asyncio.Lock()
thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dhan_worker")

# Preload symbol map
try:
    nifty500_df = pd.read_csv("csv/ind_nifty500list.csv")
    TICKER_TO_ID_MAP = nifty500_df.set_index("ticker")["security_id"].to_dict()
except Exception as e:
    logger.error(f"Failed to load symbol map: {e}")
    TICKER_TO_ID_MAP = {}

# Initialize caches and managers
HOLIDAY_CACHE = {}
CANDLE_BUILDERS = {}

STRATEGY_REGISTRY = {}


def register_strategy(name: str, strategy_class):
    """Register a new strategy in the framework with multiple aliases."""
    import re

    if not isinstance(name, str) or not name:
        raise ValueError("Strategy name must be a non-empty string")

    aliases = set()
    aliases.add(name)
    aliases.add(strategy_class.__name__)

    for alias in aliases:
        if alias in STRATEGY_REGISTRY:
            logger.warning(f"Strategy alias '{alias}' already registered, overwriting")
        STRATEGY_REGISTRY[alias] = strategy_class
        logger.info(f"Registered strategy alias: {alias}")


def get_strategy(strategy_name: str):
    """Retrieve a strategy class by name."""
    strategy = STRATEGY_REGISTRY.get(strategy_name)
    if strategy is None:
        logger.error(f"Strategy '{strategy_name}' not found in registry")
        raise KeyError(f"Strategy '{strategy_name}' not found")
    return strategy


def register_available_strategies():
    """Register strategies with correct module paths and class names"""
    strategies_to_load = [
        (
            "live_strategies.vwap_bounce_rejection",
            "VWAPBounceRejection",
            "VWAPBounceRejection",
        ),
        ("live_strategies.pivot_cci", "PivotCCI", "PivotCCI"),
        (
            "live_strategies.BB_PivotPoints_Strategy",
            "BBPivotPointsStrategy",
            "BBPivotPointsStrategy",
        ),
        ("live_strategies.BB_VWAP_Strategy", "BBVWAPStrategy", "BBVWAPStrategy"),
        ("live_strategies.rsi_bb", "RSIBB", "RSIBB"),
        (
            "live_strategies.head_shoulders_confirmation",
            "HeadShouldersConfirmation",
            "HeadShouldersConfirmation",
        ),
        (
            "live_strategies.RSI_Supertrend_Intraday",
            "RSISupertrendIntraday",
            "RSISupertrendIntraday",
        ),
        ("live_strategies.rsi_cci", "RSICCI", "RSICCI"),
        ("live_strategies.sr_rsi", "SRRSI", "SRRSI"),
        ("live_strategies.sr_rsi_volume", "SRRSIVolume", "SRRSIVolume"),
        ("live_strategies.rsi_adx", "RSIADX", "RSIADX"),
        (
            "live_strategies.EMAStochasticPullback",
            "EMAStochasticPullback",
            "EMAStochasticPullback",
        ),
        (
            "live_strategies.BB_Supertrend_Strategy",
            "BBSupertrendStrategy",
            "BBSupertrendStrategy",
        ),
        (
            "live_strategies.ATR_Volume_expansion",
            "ATRVolumeExpansion",
            "ATRVolumeExpansion",
        ),
        ("live_strategies.supertrend_cci_cmf", "SupertrendCCICMF", "SupertrendCCICMF"),
        ("live_strategies.bsav_intraday", "BSAV", "BSAV"),
        ("live_strategies.ema_adx", "EMAADXTrend", "EMAADXTrend"),
        ("live_strategies.emamulti_strategy", "EMAMultiStrategy", "EMAMultiStrategy"),
        ("live_strategies.macd_volume", "MACDVolume", "MACDVolume"),
        ("live_strategies.rmbev_intraday", "RMBEV", "RMBEV"),
        ("live_strategies.verv", "VERV", "VERV"),
        (
            "live_strategies.trendline_williams",
            "TrendlineWilliams",
            "TrendlineWilliams",
        ),
    ]

    loaded = 0
    failed_strategies = []

    for module_path, class_name, alias in strategies_to_load:
        try:
            module = __import__(module_path, fromlist=[class_name])
            strategy_class = getattr(module, class_name)
            register_strategy(alias, strategy_class)
            loaded += 1
            logger.debug(f"✅ Loaded strategy: {alias} from {module_path}")
        except ImportError as e:
            failed_strategies.append(f"{module_path} (ImportError: {e})")
            logger.warning(
                f"❌ Strategy import failed: {module_path}.{class_name} -> {e}"
            )
        except AttributeError as e:
            failed_strategies.append(
                f"{module_path}.{class_name} (AttributeError: {e})"
            )
            logger.warning(
                f"❌ Strategy class not found: {module_path}.{class_name} -> {e}"
            )
        except Exception as e:
            failed_strategies.append(f"{module_path}.{class_name} (Error: {e})")
            logger.warning(
                f"❌ Strategy registration failed: {module_path}.{class_name} -> {e}"
            )

    logger.info(
        f"Strategy registry initialized with {loaded}/{len(strategies_to_load)} strategies"
    )
    logger.info(f"Loaded strategies: {list(STRATEGY_REGISTRY.keys())}")

    if failed_strategies:
        logger.warning(f"Failed to load {len(failed_strategies)} strategies:")
        for failed in failed_strategies:
            logger.warning(f"  - {failed}")


# Initialize strategy registry
register_available_strategies()


class CacheManager:
    def __init__(self, max_size=1000, ttl=3600):
        self.depth_cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.historical_cache = TTLCache(maxsize=max_size // 2, ttl=ttl * 2)
        self.volatility_cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.volume_cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.cache_hits = defaultdict(int)
        self.cache_misses = defaultdict(int)

    def log_cache_stats(self, cache_name: str):
        hits = self.cache_hits[cache_name]
        misses = self.cache_misses[cache_name]
        if hits + misses > 0:
            logger.debug(
                f"Cache {cache_name} - Hits: {hits}, Misses: {misses}, Hit Rate: {hits / (hits + misses):.2%}"
            )


cache_manager = CacheManager(max_size=1000, ttl=3600)


class PositionManager:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.position_lock = asyncio.Lock()
        self.max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", 10))

        # Track daily traded securities to prevent multiple trades per day
        self.daily_traded_securities = set()
        self.last_reset_date = datetime.now().date()

        # Simulation-specific attributes
        self.simulated_pnl = 0.0
        self.simulated_trades = []

        # Initialize database tables if needed
        self._init_db()

    def _init_db(self):
        """Initialize position tracking tables in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create positions table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS active_positions (
                    order_id TEXT PRIMARY KEY,
                    security_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    original_quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    current_stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    strategy_name TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    breakeven_moved INTEGER DEFAULT 0,
                    partial_profit_taken INTEGER DEFAULT 0,
                    last_updated TEXT NOT NULL,
                    status TEXT DEFAULT 'ACTIVE'
                )
            """
            )
            # Main super orders table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS super_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE NOT NULL,
                    dhan_client_id TEXT,
                    correlation_id TEXT,
                    order_status TEXT,
                    transaction_type TEXT,
                    exchange_segment TEXT,
                    product_type TEXT,
                    order_type TEXT,
                    validity TEXT,
                    trading_symbol TEXT,
                    security_id TEXT,
                    quantity INTEGER,
                    remaining_quantity INTEGER,
                    ltp REAL,
                    price REAL,
                    after_market_order BOOLEAN,
                    leg_name TEXT,
                    exchange_order_id TEXT,
                    create_time TEXT,
                    update_time TEXT,
                    exchange_time TEXT,
                    oms_error_description TEXT,
                    average_traded_price REAL,
                    filled_qty INTEGER,
                    target_price REAL,
                    stop_loss_price REAL,
                    trailing_jump REAL,
                    request_payload TEXT,
                    response_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Order legs table for target and stop loss details
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS order_legs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_order_id TEXT,
                    order_id TEXT,
                    leg_name TEXT,
                    transaction_type TEXT,
                    total_quantity INTEGER,
                    remaining_quantity INTEGER,
                    triggered_quantity INTEGER,
                    price REAL,
                    order_status TEXT,
                    trailing_jump REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_order_id) REFERENCES super_orders (order_id)
                )
            """
            )

            # Create indexes for better performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_order_id ON super_orders (order_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_security_id ON super_orders (security_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_order_status ON super_orders (order_status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_parent_order ON order_legs (parent_order_id)"
            )

            conn.commit()
            conn.close()
            logger.info("Database initialized")

        except Exception as e:
            logger.error(f"Failed to initialize position database: {e}")

    async def get_order_id_by_security_id(self, security_id: int) -> str:
        """Fetch order_id using security_id from super_orders table"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT order_id FROM super_orders WHERE security_id = ? ORDER BY created_at DESC LIMIT 1",
                (str(security_id),),
            )
            result = cursor.fetchone()
            conn.close()
            if result:
                return result[0]
            return None
        except Exception as e:
            trade_logger.error(
                f"Failed to fetch order_id for security_id {security_id}: {str(e)}"
            )
            return None

    async def get_order_from_db(self, order_id: str) -> Optional[Dict]:
        """Fetch order details from super_orders table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT order_id, security_id, quantity, price, target_price, 
                       stop_loss_price, transaction_type, order_status
                FROM super_orders 
                WHERE order_id = ?
            """,
                (order_id,),
            )

            result = cursor.fetchone()
            conn.close()

            if result:
                return {
                    "order_id": result[0],
                    "security_id": int(result[1]),
                    "quantity": result[2],
                    "entry_price": result[3],
                    "take_profit": result[4],
                    "stop_loss": result[5],
                    "direction": result[6],
                    "status": result[7],
                }
            return None

        except Exception as e:
            logger.error(f"Error fetching order from DB: {e}")
            return None

    async def add_position(
        self, order_id: str, ticker: str, strategy_name: str, **kwargs
    ) -> bool:
        """Add position to tracking using order details from database"""
        async with self.position_lock:
            try:
                # Get order details from database
                order_data = await self.get_order_from_db(order_id)
                if not order_data:
                    logger.error(f"Order {order_id} not found in database")
                    return False

                # Add to active positions table
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO active_positions (
                        order_id, security_id, ticker, direction, quantity, 
                        original_quantity, entry_price, current_stop_loss, 
                        take_profit, strategy_name, entry_time, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        order_id,
                        order_data["security_id"],
                        ticker,
                        order_data["direction"],
                        order_data["quantity"],
                        order_data["quantity"],
                        order_data["entry_price"],
                        order_data["stop_loss"],
                        order_data["take_profit"],
                        strategy_name,
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                    ),
                )

                conn.commit()
                conn.close()

                # Mark as traded today
                # await self.mark_as_traded_today(order_data["security_id"])

                trade_logger.info(
                    f"NEW POSITION | {ticker} | {order_data['direction']} | "
                    f"Qty: {order_data['quantity']} | Entry: ₹{order_data['entry_price']:.2f} | "
                    f"SL: ₹{order_data['stop_loss']:.2f} | TP: ₹{order_data['take_profit']:.2f}"
                )

                return True

            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(
                    f"Error adding position: {e}. Tracebak:{str(traceback.print_exc())}"
                )
                return False

    async def get_active_positions(self) -> List[Dict]:
        """Get all active positions from database"""
        try:
            results = dhan.get_positions()
            positions = []
            for pos in results:
                # If response is dict-like and has 'positionType', filter out CLOSED
                if isinstance(pos, dict):
                    if pos.get("positionType") != "CLOSED":
                        positions.append(pos)
                # If response is row/tuple, fallback to old logic (for backward compatibility)
                elif isinstance(pos, (list, tuple)):
                    # If positionType is present and not CLOSED, add
                    position_type = None
                    if len(pos) > 3 and isinstance(pos[3], str):
                        position_type = pos[3]
                    if position_type != "CLOSED":
                        positions.append(
                            {
                                "order_id": self.get_order_id_by_security_id(pos[1]),
                                "security_id": pos[1],
                                "ticker": pos[2],
                                "direction": pos[3],
                                "quantity": pos[4],
                                "original_quantity": pos[5],
                                "entry_price": pos[6],
                                "current_stop_loss": pos[7],
                                "take_profit": pos[8],
                                "strategy_name": pos[9],
                                "entry_time": datetime.fromisoformat(pos[10]),
                                "breakeven_moved": bool(pos[11]),
                                "partial_profit_taken": bool(pos[12]),
                                "last_updated": datetime.fromisoformat(pos[13]),
                            }
                        )
            return positions
        except Exception as e:
            logger.error(f"Error getting active positions: {e}")
            return []

    async def calculate_profit_percentage(
        self, position: Dict, current_price: float
    ) -> float:
        """Calculate profit percentage for a position"""
        logger.info(f"Calculating profit percentage for {position,current_price}")
        entry_price = position["entry_price"]
        if position["direction"] == "BUY":
            return ((current_price - entry_price) / entry_price) * 100
        else:  # SHORT position
            return ((entry_price - current_price) / entry_price) * 100

    async def update_position_to_breakeven(self, order_id: str, position: Dict) -> bool:
        """Update stop loss to breakeven (entry price)"""
        try:
            new_stop_loss = position["entry_price"]

            # Update in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE active_positions 
                SET current_stop_loss = ?, breakeven_moved = 1, last_updated = ?
                WHERE order_id = ?
            """,
                (new_stop_loss, datetime.now().isoformat(), order_id),
            )

            conn.commit()
            conn.close()

            # Update actual order via Dhan API (if not simulation)
            if not SIMULATION_MODE:
                try:
                    modify_response = dhan.modify_order(
                        order_id=order_id,
                        order_type="STOP_LOSS",
                        price=new_stop_loss,
                        quantity=position["quantity"],
                    )
                    logger.info(
                        f"Modified order {order_id} stop loss to breakeven: ₹{new_stop_loss:.2f}. Response:{modify_response}"
                    )
                except Exception as e:
                    logger.error(f"Failed to modify order {order_id}: {e}")
                    return False

            # Send Telegram notification
            await send_telegram_alert(
                f"*{position['ticker']} BREAKEVEN MOVED* 🛡️\n"
                f"Stop Loss moved to Entry Price: ₹{new_stop_loss:.2f}\n"
                f"Position is now risk-free!\n"
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            )

            logger.info(
                f"{position['ticker']} moved to breakeven @ ₹{new_stop_loss:.2f}"
            )
            return True

        except Exception as e:
            logger.error(f"Error updating position to breakeven: {e}")
            return False

    async def take_partial_profit(
        self, order_id: str, position: Dict, current_price: float
    ) -> bool:
        """Sell half the quantity and lock in partial profit"""
        try:
            # Calculate half quantity (minimum 1 share)
            half_quantity = max(1, position["quantity"] // 2)
            remaining_quantity = position["quantity"] - half_quantity

            if remaining_quantity <= 0:
                logger.warning(
                    f"Cannot take partial profit for {position['ticker']}: insufficient quantity"
                )
                return False

            # Determine exit direction
            exit_direction = "SELL" if position["direction"] == "BUY" else "BUY"

            # Place market order for partial exit
            exit_order = await place_market_order(
                position["security_id"], exit_direction, half_quantity
            )

            if exit_order and exit_order.get("orderId"):
                # Calculate partial profit
                entry_price = position["entry_price"]
                if position["direction"] == "BUY":
                    partial_pnl = (current_price - entry_price) * half_quantity
                else:
                    partial_pnl = (entry_price - current_price) * half_quantity

                # Update position in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE active_positions 
                    SET quantity = ?, partial_profit_taken = 1, last_updated = ?
                    WHERE order_id = ?
                """,
                    (remaining_quantity, datetime.now().isoformat(), order_id),
                )

                conn.commit()
                conn.close()

                # Track for simulation
                if SIMULATION_MODE:
                    self.simulated_pnl += partial_pnl
                    self.simulated_trades.append(
                        {
                            "ticker": position["ticker"],
                            "direction": exit_direction,
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "quantity": half_quantity,
                            "pnl": partial_pnl,
                            "reason": "Partial Profit",
                            "timestamp": datetime.now(),
                        }
                    )

                # Send Telegram notification
                await send_telegram_alert(
                    f"*{position['ticker']} PARTIAL PROFIT TAKEN* 💰\n"
                    f"Sold {half_quantity} shares @ ₹{current_price:.2f}\n"
                    f"Profit: ₹{partial_pnl:.2f}\n"
                    f"Remaining: {remaining_quantity} shares\n"
                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                )

                trade_logger.info(
                    f"PARTIAL PROFIT | {position['ticker']} | {exit_direction} | "
                    f"Qty: {half_quantity} | Price: ₹{current_price:.2f} | P&L: ₹{partial_pnl:.2f}"
                )

                logger.info(
                    f"{position['ticker']} partial profit: {half_quantity} shares @ ₹{current_price:.2f}, "
                    f"P&L: ₹{partial_pnl:.2f}, Remaining: {remaining_quantity}"
                )

                return True

            else:
                logger.error(f"Partial profit order failed for {position['ticker']}")
                return False

        except Exception as e:
            logger.error(f"Error taking partial profit: {e}")
            return False

    async def close_position(
        self, order_id: str, exit_price: float = None, reason: str = "Manual"
    ) -> bool:
        """Close position and remove from tracking"""
        try:
            # Get position details
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT ticker, direction, quantity, entry_price, strategy_name
                FROM active_positions WHERE order_id = ?
            """,
                (order_id,),
            )

            result = cursor.fetchone()
            if not result:
                logger.warning(f"Position {order_id} not found")
                return False

            ticker, direction, quantity, entry_price, strategy_name = result

            # Mark as closed in database
            cursor.execute(
                """
                UPDATE active_positions 
                SET status = 'CLOSED', last_updated = ?
                WHERE order_id = ?
            """,
                (datetime.now().isoformat(), order_id),
            )

            conn.commit()
            conn.close()

            # Calculate P&L if exit price provided
            pnl = 0.0
            if exit_price:
                if direction == "BUY":
                    pnl = (exit_price - entry_price) * quantity
                else:
                    pnl = (entry_price - exit_price) * quantity

                if SIMULATION_MODE:
                    self.simulated_pnl += pnl
                    self.simulated_trades.append(
                        {
                            "ticker": ticker,
                            "direction": direction,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "quantity": quantity,
                            "pnl": pnl,
                            "reason": reason,
                            "timestamp": datetime.now(),
                        }
                    )

            trade_logger.info(
                f"CLOSED POSITION | {ticker} | {direction} | Qty: {quantity} | "
                f"Entry: ₹{entry_price:.2f} | Exit: ₹{exit_price:.2f} | "
                f"P&L: ₹{pnl:.2f} | Reason: {reason}"
            )

            logger.info(f"Closed position {order_id} for {ticker}, P&L: ₹{pnl:.2f}")
            return True

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    async def monitor_positions(self):
        """Main position monitoring loop with breakeven and partial profit logic"""
        logger.info("Starting position monitoring with profit management")

        while True:
            try:
                positions = await self.get_active_positions()
                if not positions:
                    await asyncio.sleep(30)
                    continue

                for position in positions:
                    print("FROM monitor position:", position)
                    logger.info(f"Monitoring position: {position}")
                    try:

                        current_price = fetch_realtime_quote(position["security_id"])[
                            position["security_id"]
                        ]["price"]

                        # Calculate profit percentage
                        profit_pct = await self.calculate_profit_percentage(
                            position, current_price
                        )

                        # 1. Move to breakeven at 0.5% profit
                        if (
                            profit_pct >= 0.15
                            and not position["breakeven_moved"]
                            and not position["partial_profit_taken"]
                        ):
                            await self.update_position_to_breakeven(
                                position["order_id"], position
                            )
                            await self.take_partial_profit(
                                position["order_id"], position, current_price
                            )
                            continue

                        # 3. Check stop-loss/take-profit triggers
                        exit_triggered = False
                        reason = ""

                        if position["direction"] == "BUY":
                            if current_price <= position["current_stop_loss"]:
                                exit_triggered = True
                                reason = "Stop-loss hit"
                            elif current_price >= position["take_profit"]:
                                exit_triggered = True
                                reason = "Take-profit hit"
                        else:  # SHORT
                            if current_price >= position["current_stop_loss"]:
                                exit_triggered = True
                                reason = "Stop-loss hit"
                            elif current_price <= position["take_profit"]:
                                exit_triggered = True
                                reason = "Take-profit hit"

                        if exit_triggered:
                            # Place exit order
                            exit_direction = (
                                "SELL" if position["direction"] == "BUY" else "BUY"
                            )
                            exit_order = await place_market_order(
                                position["security_id"],
                                exit_direction,
                                position["quantity"],
                            )

                            if exit_order:
                                await self.close_position(
                                    position["order_id"], current_price, reason
                                )

                    except Exception as e:
                        logger.error(
                            f"Error monitoring position {position['ticker']}: {e}"
                        )

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(60)

    async def get_simulation_report(self) -> Dict:
        """Generate simulation performance report"""
        if not self.simulated_trades:
            return {"message": "No trades recorded"}

        winning_trades = [t for t in self.simulated_trades if t["pnl"] > 0]
        losing_trades = [t for t in self.simulated_trades if t["pnl"] <= 0]

        return {
            "total_pnl": self.simulated_pnl,
            "total_trades": len(self.simulated_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (
                len(winning_trades) / len(self.simulated_trades)
                if self.simulated_trades
                else 0
            ),
            "average_pnl_per_trade": (
                self.simulated_pnl / len(self.simulated_trades)
                if self.simulated_trades
                else 0
            ),
            "largest_win": (
                max([t["pnl"] for t in winning_trades]) if winning_trades else 0
            ),
            "largest_loss": (
                min([t["pnl"] for t in losing_trades]) if losing_trades else 0
            ),
            "trades": self.simulated_trades,
        }


position_manager = PositionManager(DB_PATH)


def round_to_tick_size(price: float, tick_size: float) -> float:
    """Round a price to the nearest multiple of the tick size."""
    return round(price / tick_size) * tick_size


async def check_ticker_traded_today(security_id: int) -> dict:
    """
    Check if ticker has been traded today using SQLite database
    Returns dict with trade status and details
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get today's date in the same format as stored in DB
        today = date.today().isoformat()

        # Check for any orders placed today for this security
        cursor.execute(
            """
            SELECT 
                order_id, 
                order_status, 
                transaction_type, 
                quantity, 
                price,
                create_time,
                trading_symbol
            FROM super_orders 
            WHERE security_id = ? 
            AND DATE(created_at) = DATE(?)
            ORDER BY created_at DESC
        """,
            (str(security_id), today),
        )

        orders_today = cursor.fetchall()

        if orders_today:
            # Convert to list of dicts for easier handling
            orders = [dict(order) for order in orders_today]

            # Check if any order is still active (not cancelled/rejected)
            active_orders = [
                order
                for order in orders
                if order["order_status"] not in ["CANCELLED", "REJECTED"]
            ]

            conn.close()

            return {
                "traded_today": True,
                "total_orders": len(orders),
                "active_orders": len(active_orders),
                "orders": orders,
                "latest_order": orders[0] if orders else None,
            }
        else:
            conn.close()
            return {
                "traded_today": False,
                "total_orders": 0,
                "active_orders": 0,
                "orders": [],
                "latest_order": None,
            }

    except Exception as e:
        trade_logger.error(f"Failed to check ticker trading status from DB: {str(e)}")
        return {
            "traded_today": False,
            "total_orders": 0,
            "active_orders": 0,
            "orders": [],
            "latest_order": None,
            "error": str(e),
        }


def is_security_id_in_positions(security_id: int, positions: list[dict]) -> bool:
    """Check if the given security_id exists in the list of positions."""
    positions = (pos for pos in positions["data"])
    is_today_trade = False
    active_orders = 0
    total_orders = 0
    for pos in positions:
        for k, v in pos.items():
            if k == "securityId" and v == str(security_id):
                is_today_trade = True
            if k == "positionType" and v != "CLOSED":
                active_orders += 1
        total_orders += 1
    logger.info(f"Today's positions :{[positions]}")

    return is_today_trade, total_orders, active_orders


async def execute_strategy_signal(
    ticker: str,
    security_id: int,
    signal: str,
    regime: str,
    adx_value: float,
    atr_value: float,
    hist_data: pd.DataFrame,
    strategy_name: str,
    strategy_instance=None,
    **params,
) -> bool:
    """Enhanced signal execution with SQLite-based trading check and proper position direction handling."""
    current_price = None
    try:
        todays_positions = dhan.get_positions()
        is_today_trade, total_orders, active_orders = is_security_id_in_positions(
            security_id, todays_positions
        )

        if is_today_trade:
            logger.info(
                f"{ticker} - Already traded today. "
                f"Total orders: {total_orders}, "
                f"Active orders: {active_orders}"
            )
        if total_orders >= 10:
            dhan.kill_switch()

            # Skip if we already have active orders for this security today
            if is_today_trade:
                logger.info(
                    f"{ticker} - Skipping new {signal} signal due to existing active orders"
                )
                return False

            return False  # No new action needed if already traded today
        # if current is before 9:30 AM and after 15:00PM do not place a new trade
        if datetime.now(IST).time() < time(9, 30) or datetime.now(IST).time() > time(
            15, 0
        ):
            logger.info(
                f"{ticker} - Current time {datetime.now(IST).time()} is outside suitable trading hours"
            )
            return False
        # Add volatility filter
        volatility = await calculate_stock_volatility(security_id)
        if volatility > VOLATILITY_THRESHOLD:
            logger.warning(
                f"Skipping {ticker} due to high volatility: {volatility:.4f}"
            )
            return False

        # Check daily loss limit
        pnl_data = await pnl_tracker.update_daily_pnl()
        if isinstance(pnl_data, dict):
            current_pnl = pnl_data.get("total", 0)
        else:
            current_pnl = pnl_data or 0

        if current_pnl <= -MAX_DAILY_LOSS_PERCENT * ACCOUNT_SIZE:
            message = (
                f"🛑 TRADING HALTED: Daily loss limit reached\n"
                f"Current P&L: ₹{current_pnl:.2f}\n"
                f"Limit: ₹{-MAX_DAILY_LOSS_PERCENT * ACCOUNT_SIZE:.2f}"
            )
            await send_telegram_alert(message)
            logger.critical("Daily loss limit reached - trading halted")
            return False

        # Get current quote
        quotes = await fetch_realtime_quote([security_id])
        quote = quotes.get(security_id)
        if not quote and not hist_data.iloc[-1]["close"]:
            logger.warning(f"Price unavailable for {ticker}")
            return False
        current_price = quote or hist_data.iloc[-1]["close"]
        vwap = await calculate_vwap(hist_data)
        logger.info(f"Current price for {ticker}: ₹{current_price}")

        # Improved entry price logic
        entry_price = (
            min(
                (
                    current_price["price"]
                    if isinstance(current_price, dict)
                    else current_price
                ),
                vwap * 0.998,
            )
            if signal == "BUY"
            else max(
                (
                    current_price["price"]
                    if isinstance(current_price, dict)
                    else current_price
                ),
                vwap * 1.002,
            )
        )

        # Calculate risk parameters
        risk_params = calculate_risk_params(regime, atr_value, entry_price, signal)
        now = datetime.now(IST)

        tick_size = DEFAULT_TICK_SIZE

        # Round prices to the nearest tick size
        rounded_entry_price = round_to_tick_size(entry_price, tick_size)
        rounded_stop_loss = round_to_tick_size(risk_params["stop_loss"], tick_size)
        rounded_take_profit = round_to_tick_size(risk_params["take_profit"], tick_size)

        # Check available funds
        funds = dhan.get_fund_limits().get("data", {}).get("availabelBalance", 0)
        required_margin = (rounded_entry_price * risk_params["position_size"]) / 5

        if funds < required_margin:
            logger.warning(
                f"{ticker} - Insufficient funds: ₹{funds:.2f} < ₹{required_margin:.2f}"
            )

            await send_telegram_alert(
                f"*{ticker} Order Failed* ❌\n"
                f"Signal: {signal} at ₹{rounded_entry_price:.2f}\n"
                f"Insufficient funds: ₹{funds:.2f} < ₹{required_margin:.2f}\n"
            )
            return False

        # Proceed with order placement since ticker hasn't been traded today
        # Prepare entry notification
        direction_emoji = "🟢" if signal == "BUY" else "🔴"
        position_size = risk_params["position_size"]
        position_type = "Long" if signal == "BUY" else "Short"

        # Get today's trading summary for context

        message = (
            f"*{ticker} ENTRY SIGNAL* {direction_emoji}\n"
            f"Strategies: `{params.get('strategy_names', 'N/A')}`\n"
            f"Direction: {position_type}\n"
            f"Entry: ₹{rounded_entry_price:.2f} | VWAP: ₹{vwap:.2f}\n"
            f"Current: ₹{current_price["price"] if isinstance(current_price, dict) else current_price:.2f}\n"
            f"Regime: {regime} (ADX: {adx_value:.2f})\n"
            f"Volatility: {volatility:.4f}\n"
            f"Size: {position_size} | SL: ₹{rounded_stop_loss:.2f}\n"
            f"TP: ₹{rounded_take_profit:.2f}\n"
            f"Risk: ₹{abs(rounded_entry_price - rounded_stop_loss) * position_size:.2f}\n"
            f"Time: {now.strftime('%H:%M:%S')}\n"
        )

        logger.info(f"Executing {signal} signal for {ticker}")
        await send_telegram_alert(message)

        # Place order
        try:
            order_response = await place_super_order(
                security_id,
                signal,
                rounded_entry_price,
                rounded_stop_loss,
                rounded_take_profit,
                position_size,
            )

        except Exception as e:
            logger.error(f"Order placement failed for {ticker}: {str(e)}")
            await send_telegram_alert(
                f"*{ticker} Order Failed* ❌\n"
                f"Signal: {signal} at ₹{rounded_entry_price:.2f}\n"
                f"Error: {str(e)}"
            )
            return False

        if order_response and order_response.get("orderId"):
            # Add position to manager
            success = await position_manager.add_position(
                order_response["orderId"],
                security_id,
                # ticker,
                # rounded_entry_price,
                # position_size,
                # rounded_stop_loss,
                # rounded_take_profit,
                # signal,  # This is the position direction (BUY = LONG, SELL = SHORT)
                strategy_name,
                # strategy_instance,
            )
            if success:
                trade_logger.info(
                    f"{'[SIM] ' if SIMULATION_MODE else ''}ORDER EXECUTED | {ticker} | "
                    f"{signal} | Qty: {position_size} | Price: ₹{rounded_entry_price:.2f} | "
                    f"OrderID: {order_response['orderId']}"
                )

            return success
        else:
            logger.error(f"Order failed for {ticker}: {order_response}")
            await send_telegram_alert(
                f"*{ticker} Order Failed* ❌\n"
                f"Signal: {signal} at ₹{rounded_entry_price:.2f}\n"
                f"Order response: {order_response}"
            )
            return False

    except Exception as e:
        logger.error(f"Signal execution failed for {ticker}: {str(e)}")
        logger.error(traceback.format_exc())
        await send_telegram_alert(
            f"*{ticker} Execution Failed* ❌\n"
            f"Error: {str(e)}\n"
            f"Current Price: ₹{current_price if current_price else 'N/A'}"
        )
        return False


# Function to clean up old database entries (optional)
async def cleanup_old_orders(days_to_keep: int = 30):
    """Clean up old order entries from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Delete orders older than specified days
        cursor.execute(
            """
            DELETE FROM order_legs 
            WHERE parent_order_id IN (
                SELECT order_id FROM super_orders 
                WHERE DATE(created_at) < DATE('now', '-{} days')
            )
        """.format(
                days_to_keep
            )
        )

        cursor.execute(
            """
            DELETE FROM super_orders 
            WHERE DATE(created_at) < DATE('now', '-{} days')
        """.format(
                days_to_keep
            )
        )

        deleted_orders = cursor.rowcount
        conn.commit()
        conn.close()

        if deleted_orders > 0:
            logger.info(f"Cleaned up {deleted_orders} old order records")

    except Exception as e:
        logger.error(f"Failed to cleanup old orders: {str(e)}")


async def process_stock_with_exit_monitoring(
    ticker: str, security_id: int, strategies: List[Dict]
) -> None:
    """Separate entry and exit signal processing with LLM confirmation"""
    async with adaptive_semaphore:
        try:
            logger.debug(f"Processing {ticker} (ID: {security_id})")
            current_time = datetime.now(IST)

            # Get current combined data
            combined_data = await get_combined_data(security_id)
            if combined_data is None:
                logger.warning(f"{ticker} - No data available")
                return

            # Check minimum data requirements
            data_length = len(combined_data)
            min_bars = max(
                [
                    get_strategy(s["Strategy"]).get_min_data_points(
                        ast.literal_eval(s["Best_Parameters"])
                        if isinstance(s["Best_Parameters"], str)
                        else s["Best_Parameters"]
                    )
                    for s in strategies
                    if s["Strategy"] in STRATEGY_REGISTRY
                ],
                default=30,
            )

            if data_length < min_bars:
                logger.warning(
                    f"{ticker} - Insufficient data ({data_length} < {min_bars})"
                )
                return

            # Calculate market regime
            regime, adx_value, atr_value = calculate_regime(combined_data)
            logger.debug(
                f"{ticker} - Regime: {regime} (ADX: {adx_value:.2f}, ATR: {atr_value:.2f})"
            )

            # Process entry signals from strategies
            signals = []
            strategy_instances = []
            strategy_names = []  # Collect all strategy names

            for strat in strategies:
                strategy_name = strat["Strategy"]
                try:
                    strategy_class = get_strategy(strategy_name)
                    params = strat.get("Best_Parameters", {})
                    if isinstance(params, str) and params.strip():
                        try:
                            params = ast.literal_eval(params)
                        except (ValueError, SyntaxError):
                            params = {}

                    strategy_instance = strategy_class(combined_data, **params)
                    signal = strategy_instance.run()

                    if signal in ["BUY", "SELL", "buy", "sell"]:
                        signals.append(signal.upper())
                        strategy_instances.append(
                            {
                                "instance": strategy_instance,
                                "name": strategy_name,
                                "signal": signal,
                                "params": params,
                            }
                        )
                        strategy_names.append(strategy_name)
                        logger.debug(
                            f"{ticker} - {strategy_name} generated {signal} signal"
                        )

                except Exception as e:
                    logger.error(f"{ticker} - {strategy_name} failed: {e}")

            if not signals:
                return

            # Execute strongest signal based on votes
            buy_votes = signals.count("BUY")
            sell_votes = signals.count("SELL")
            min_vote_diff = int(os.getenv("MIN_VOTE_DIFF", 1))
            logger.info(
                f"{ticker} - Buy votes: {buy_votes}, Sell votes: {sell_votes}, Stratgies:{strategy_names}"
            )

            strategy_signal = None
            primary_strategy = None

            # Determine the strategy consensus signal
            if buy_votes >= MIN_VOTES and (buy_votes - sell_votes) >= min_vote_diff:
                strategy_signal = "BUY"
                # primary_strategy = next(
                #     s for s in strategy_instances if s["signal"] == "BUY"
                # )
            elif sell_votes >= MIN_VOTES and (sell_votes - buy_votes) >= min_vote_diff:
                strategy_signal = "SELL"
                # primary_strategy = next(
                #     s for s in strategy_instances if s["signal"] == "SELL"
                # )

            # If we have a strategy signal, get LLM confirmation
            nifty_signal = get_index_signal_dhan_api("13", "Nifty 50", 0.6)
            sector_security_id, index_name = get_sector_security_id(security_id)
            sector_signal = get_index_signal_dhan_api(
                sector_security_id, index_name, 0.6
            )
            if strategy_signal and (
                (
                    strategy_signal.upper() == "BUY"
                    and (
                        nifty_signal["signal"].upper() in ["BUY", "BOTH"]
                        or sector_signal["signal"].upper() in ["BUY", "BOTH"]
                    )
                )
                or (
                    strategy_signal.upper() == "SELL"
                    and (
                        nifty_signal["signal"].upper() in ["BUY", "BOTH"]
                        or sector_signal["signal"].upper() in ["BUY", "BOTH"]
                    )
                )
            ):
                logger.info(
                    f"{ticker} - Strategy consensus: {strategy_signal} Nifty signal: {nifty_signal["signal"].upper()} Sector signal: {sector_signal["signal"].upper()}"
                )

                # Get LLM signal (only BUY/SELL, no HOLD)
                # llm_signal = await get_openrouter_llm_signal(ticker, combined_data)
                # logger.info(f"{ticker} - LLM signal: {llm_signal}")

                # Only proceed if LLM gives a clear BUY/SELL signal
                # if llm_signal in ["BUY", "SELL"]:
                # Compare signals and execute only if they match
                # if strategy_signal == llm_signal:
                #     logger.info(
                #         f"{ticker} - SIGNAL CONFIRMATION: Both strategy and LLM agree on {strategy_signal}"
                #     )
                executed = await execute_strategy_signal(
                    ticker,
                    security_id,
                    strategy_signal,
                    regime,
                    adx_value,
                    atr_value,
                    combined_data,
                    strategy_names[0],
                    # primary_strategy["instance"],
                    strategy_names=strategy_names,  # Pass the list of strategy names
                    # **primary_strategy["params"],
                )

                if executed:
                    # await position_manager.update_last_trade_time(ticker, current_time)
                    logger.info(f"{ticker} - Order executed with dual confirmation")
                #     else:
                #         logger.info(
                #             f"{ticker} - SIGNAL MISMATCH: Strategy={strategy_signal}, LLM={llm_signal}. No order placed."
                #         )
                # else:
                #     logger.info(
                #         f"{ticker} - LLM provided no actionable signal ({llm_signal}). No order placed."
                # )
            else:
                logger.debug(f"{ticker} - No qualifying strategy signal generated")
        except Exception as e:
            logger.error(f"{ticker} - Processing failed: {str(e)}")
            logger.error(traceback.format_exc())


class EnhancedCandle:
    __slots__ = (
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bid_qty",
        "ask_qty",
        "bid_depth",
        "ask_depth",
    )

    def __init__(self):
        self.datetime = None
        self.open = 0.0
        self.high = 0.0
        self.low = float("inf")
        self.close = 0.0
        self.volume = 0
        self.bid_qty = 0
        self.ask_qty = 0
        self.bid_depth = 0
        self.ask_depth = 0


class OptimizedRateLimiter:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.last_refill = datetime.now(IST)
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = datetime.now(IST)
            elapsed = (now - self.last_refill).total_seconds()
            if elapsed >= 1.0:
                tokens_to_add = int(elapsed * (self.rate_limit / 60.0))
                self.tokens = min(self.rate_limit, self.tokens + tokens_to_add)
                self.last_refill = now
            while self.tokens < 1:
                await asyncio.sleep(0.1)
                now = datetime.now(IST)
                elapsed = (now - self.last_refill).total_seconds()
                if elapsed >= 1.0:
                    tokens_to_add = int(elapsed * (self.rate_limit / 60.0))
                    self.tokens = min(self.rate_limit, self.tokens + tokens_to_add)
                    self.last_refill = now
            self.tokens -= 1


rate_limiter = OptimizedRateLimiter(rate_limit=QUOTE_API_RATE_LIMIT)


def is_high_volume_period() -> bool:
    now = datetime.now(IST).time()
    high_volume_periods = [
        (time(9, 15), time(10, 30)),
        (time(14, 30), time(15, 30)),
    ]
    return any(start <= now <= end for start, end in high_volume_periods)


def calculate_relative_volume(current_volume: float, avg_volume: float) -> float:
    return current_volume / avg_volume if avg_volume > 0 else 0.0


class TelegramQueue:
    def __init__(self):
        self.message_queue = asyncio.Queue(maxsize=100)
        self.is_running = False
        self._worker_task = None
        self.rate_limiter = OptimizedRateLimiter(
            rate_limit=30
        )  # 30 messages per minute

    async def start(self):
        if not self.is_running:
            self.is_running = True
            self._worker_task = asyncio.create_task(self._process_messages())
            logger.info("Telegram queue started")

    async def stop(self):
        self.is_running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Telegram queue stopped")

    async def _process_messages(self):
        while self.is_running:
            try:
                queue_size = self.message_queue.qsize()
                if queue_size > 80:
                    logger.warning(f"Telegram queue size high: {queue_size}/100")
                await self.rate_limiter.acquire()
                message = await asyncio.wait_for(self.message_queue.get(), timeout=5.0)
                success = await self._send_message_with_retry(message)
                if success:
                    logger.debug(f"Telegram message sent successfully")
                else:
                    logger.warning(f"Failed to send telegram message after retries")
                self.message_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Telegram queue error: {e}")
                await asyncio.sleep(5)

    async def _send_message_with_retry(
        self, message: str, max_retries: int = 3
    ) -> bool:
        for attempt in range(max_retries):
            try:
                success = await self._send_message(message)
                if success:
                    logger.debug(f"Telegram message sent on attempt {attempt + 1}")
                    return True
                logger.warning(f"Telegram send attempt {attempt + 1} failed")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
            except asyncio.TimeoutError as e:
                logger.error(f"Telegram send attempt {attempt + 1} timed out: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
            except Exception as e:
                logger.error(
                    f"Telegram send attempt {attempt + 1} failed with error: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)

        # Fallback: Log to file
        try:
            with open("failed_telegram_messages.log", "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} - {message}\n"
                )
            logger.info("Logged failed Telegram message to file")
        except Exception as e:
            logger.error(f"Failed to log message to file: {e}")
        return False

    async def _send_message(self, message: str) -> bool:
        try:
            if SIMULATION_MODE and not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
                logger.info(f"[SIM] Telegram message: {message[:100]}...")
                return True

            if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
                logger.error("Missing Telegram credentials")
                return False

            if len(message) > 4000:
                message = message[:3900] + "...\n[TRUNCATED]"

            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message.replace("*", r"\*"),
                "parse_mode": "Markdown",
            }

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return True
                    else:
                        text = await response.text()
                        logger.error(f"Telegram API error {response.status}: {text}")
                        return False
        except Exception as e:
            import traceback

            stack_trace = traceback.format_exc()
            logger.error(f"Telegram send failed: {stack_trace}")
            return False

    async def send_alert(self, message: str):
        try:
            if SIMULATION_MODE:
                message = f"[SIM] {message}"
            await self.message_queue.put(message)
            logger.debug(f"Queued telegram message: {message[:50]}...")
        except asyncio.QueueFull:
            logger.warning("Telegram queue full, dropping message")
        except Exception as e:
            logger.error(f"Failed to queue telegram message: {e}")


# Initialize telegram queue
telegram_queue = TelegramQueue()


async def send_telegram_alert(message: str):
    """Send telegram alert via queue"""
    await telegram_queue.send_alert(message)


class APIClient:
    def __init__(self):
        self.connector = None
        self.timeout = aiohttp.ClientTimeout(total=10, connect=5)
        self.session = None

    async def get_session(self):
        if self.session is None or self.session.closed:
            if self.connector is None:
                self.connector = aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=20,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True,
                )
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout,
                headers={"access-token": DHAN_ACCESS_TOKEN},
            )
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
        if self.connector and not self.connector.closed:
            await self.connector.close()


api_client = APIClient()


async def save_super_order_to_db(
    request_payload: dict,
    response_data: dict,
    security_id: int,
    transaction_type: str,
    current_price: float,
    stop_loss: float,
    take_profit: float,
    quantity: int,
):
    """Save super order response to SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Extract data from response (for simple place order response)
        order_id = response_data.get("orderId", "")
        order_status = response_data.get("orderStatus", "UNKNOWN")

        # Insert main order record
        cursor.execute(
            """
            INSERT INTO super_orders (
                order_id, dhan_client_id, correlation_id, order_status,
                transaction_type, exchange_segment, product_type, order_type,
                security_id, quantity, price, target_price, stop_loss_price,
                trailing_jump, request_payload, response_data, update_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                order_id,
                request_payload.get("dhanClientId", ""),
                request_payload.get("correlationId", ""),
                order_status,
                transaction_type,
                request_payload.get("exchangeSegment", ""),
                request_payload.get("productType", ""),
                request_payload.get("orderType", ""),
                str(security_id),
                quantity,
                current_price,
                take_profit,
                stop_loss,
                request_payload.get("trailingJump", 0),
                json.dumps(request_payload),
                json.dumps(response_data),
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

        trade_logger.info(f"Super order saved to DB | OrderID: {order_id}")

    except Exception as e:
        trade_logger.error(f"Failed to save super order to DB: {str(e)}")


async def update_super_order_from_list_response(order_data: dict):
    """Update super order with detailed data from list API response"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        order_id = order_data.get("orderId", "")

        # Update main order with complete details
        cursor.execute(
            """
            UPDATE super_orders SET
                dhan_client_id = ?, correlation_id = ?, order_status = ?,
                transaction_type = ?, exchange_segment = ?, product_type = ?,
                order_type = ?, validity = ?, trading_symbol = ?, security_id = ?,
                quantity = ?, remaining_quantity = ?, ltp = ?, price = ?,
                after_market_order = ?, leg_name = ?, exchange_order_id = ?,
                create_time = ?, update_time = ?, exchange_time = ?,
                oms_error_description = ?, average_traded_price = ?, filled_qty = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE order_id = ?
        """,
            (
                order_data.get("dhanClientId", ""),
                order_data.get("correlationId", ""),
                order_data.get("orderStatus", ""),
                order_data.get("transactionType", ""),
                order_data.get("exchangeSegment", ""),
                order_data.get("productType", ""),
                order_data.get("orderType", ""),
                order_data.get("validity", ""),
                order_data.get("tradingSymbol", ""),
                order_data.get("securityId", ""),
                order_data.get("quantity", 0),
                order_data.get("remainingQuantity", 0),
                order_data.get("ltp", 0.0),
                order_data.get("price", 0.0),
                order_data.get("afterMarketOrder", False),
                order_data.get("legName", ""),
                order_data.get("exchangeOrderId", ""),
                order_data.get("createTime", ""),
                order_data.get("updateTime", ""),
                order_data.get("exchangeTime", ""),
                order_data.get("omsErrorDescription", ""),
                order_data.get("averageTradedPrice", 0.0),
                order_data.get("filledQty", 0),
                order_id,
            ),
        )

        # Delete existing legs for this order
        cursor.execute("DELETE FROM order_legs WHERE parent_order_id = ?", (order_id,))

        # Insert leg details
        leg_details = order_data.get("legDetails", [])
        for leg in leg_details:
            cursor.execute(
                """
                INSERT INTO order_legs (
                    parent_order_id, order_id, leg_name, transaction_type,
                    total_quantity, remaining_quantity, triggered_quantity,
                    price, order_status, trailing_jump
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    order_id,
                    leg.get("orderId", ""),
                    leg.get("legName", ""),
                    leg.get("transactionType", ""),
                    leg.get("totalQuatity", 0),  # Note: API has typo "Quatity"
                    leg.get("remainingQuantity", 0),
                    leg.get("triggeredQuantity", 0),
                    leg.get("price", 0.0),
                    leg.get("orderStatus", ""),
                    leg.get("trailingJump", 0.0),
                ),
            )

        conn.commit()
        conn.close()

        trade_logger.info(f"Super order updated in DB | OrderID: {order_id}")

    except Exception as e:
        trade_logger.error(f"Failed to update super order in DB: {str(e)}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (asyncio.TimeoutError, aiohttp.ClientConnectionError)
    ),
)
async def place_super_order(
    security_id: int,
    transaction_type: str,
    current_price: float,
    stop_loss: float,
    take_profit: float,
    quantity: int = MAX_QUANTITY,
) -> Optional[Dict]:
    """Enhanced order placement with proper logging and SQLite storage"""

    if SIMULATION_MODE:
        # Simulate successful order placement
        order_id = f"SIM-{security_id}-{int(datetime.now(IST).timestamp())}"

        # Create simulated request and response
        simulated_request = {
            "dhanClientId": DHAN_CLIENT_ID,
            "correlationId": f"{security_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "transactionType": transaction_type,
            "exchangeSegment": "NSE_EQ",
            "productType": "INTRADAY",
            "orderType": "LIMIT",
            "securityId": str(security_id),
            "quantity": quantity,
            "price": round(current_price, 2),
            "targetPrice": round(take_profit, 2),
            "stopLossPrice": round(stop_loss, 2),
            "trailingJump": 0.1,
        }

        simulated_response = {"orderId": order_id, "orderStatus": "PENDING"}

        # Save to database
        await save_super_order_to_db(
            simulated_request,
            simulated_response,
            security_id,
            transaction_type,
            current_price,
            stop_loss,
            take_profit,
            quantity,
        )

        trade_logger.info(
            f"[SIM] SUPER ORDER | {security_id} | {transaction_type} | "
            f"Qty: {quantity} | Price: ₹{current_price:.2f} | "
            f"SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f} | OrderID: {order_id}"
        )
        return simulated_response

    try:
        # await rate_limiter.acquire()
        url = "https://api.dhan.co/v2/super/orders"
        payload = {
            "dhanClientId": DHAN_CLIENT_ID,
            "correlationId": f"{security_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "transactionType": transaction_type,
            "exchangeSegment": "NSE_EQ",
            "productType": "INTRADAY",
            "orderType": "LIMIT",
            "securityId": str(security_id),
            "quantity": quantity,
            "price": round(current_price, 2),
            "targetPrice": round(take_profit, 2),
            "stopLossPrice": round(stop_loss, 2),
            "trailingJump": 1 if current_price > 1000 else 0.1,
        }

        session = await api_client.get_session()
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                order_id = data.get("orderId")
                await send_telegram_alert(
                    f"*{security_id} Order Placed\n"
                    f"Signal: {transaction_type} at ₹{current_price:.2f}\n"
                    f"Reposne: {response}"
                )
                if order_id:
                    # Save to database
                    await save_super_order_to_db(
                        payload,
                        data,
                        security_id,
                        transaction_type,
                        current_price,
                        stop_loss,
                        take_profit,
                        quantity,
                    )

                    trade_logger.info(
                        f"SUPER ORDER PLACED | {security_id} | {transaction_type} | "
                        f"Qty: {quantity} | Price: ₹{current_price:.2f} | "
                        f"SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f} | OrderID: {order_id}"
                    )
                    return data
                else:
                    trade_logger.error(f"Super order failed - no order ID: {data}")
            else:
                text = await response.text()
                trade_logger.error(f"Super order HTTP error {response.status}: {text}")
        return None
    except Exception as e:
        trade_logger.error(f"Super order exception: {str(e)}")
        return None


# Helper functions for database operations
async def get_super_order_by_id(order_id: str) -> Optional[Dict]:
    """Retrieve super order by ID from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM super_orders WHERE order_id = ?
        """,
            (order_id,),
        )

        row = cursor.fetchone()
        if row:
            order_data = dict(row)

            # Get leg details
            cursor.execute(
                """
                SELECT * FROM order_legs WHERE parent_order_id = ?
            """,
                (order_id,),
            )

            legs = cursor.fetchall()
            order_data["leg_details"] = [dict(leg) for leg in legs]

            conn.close()
            return order_data

        conn.close()
        return None

    except Exception as e:
        trade_logger.error(f"Failed to get super order from DB: {str(e)}")
        return None


async def get_pending_super_orders() -> list:
    """Get all pending super orders from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM super_orders 
            WHERE order_status IN ('PENDING', 'TRANSIT', 'PART_TRADED')
            ORDER BY create_time DESC
        """
        )

        rows = cursor.fetchall()
        orders = []

        for row in rows:
            order_data = dict(row)

            # Get leg details for each order
            cursor.execute(
                """
                SELECT * FROM order_legs WHERE parent_order_id = ?
            """,
                (order_data["order_id"],),
            )

            legs = cursor.fetchall()
            order_data["leg_details"] = [dict(leg) for leg in legs]
            orders.append(order_data)

        conn.close()
        return orders

    except Exception as e:
        trade_logger.error(f"Failed to get pending orders from DB: {str(e)}")
        return []


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (asyncio.TimeoutError, aiohttp.ClientConnectionError)
    ),
)
async def place_market_order(
    security_id: int, transaction_type: str, quantity: int
) -> Optional[Dict]:
    """Enhanced market order placement for exits"""
    if SIMULATION_MODE:
        order_id = f"SIM-EXIT-{security_id}-{int(datetime.now(IST).timestamp())}"
        trade_logger.info(
            f"[SIM] MARKET ORDER | {security_id} | {transaction_type} | Qty: {quantity}"
        )
        return {"orderId": order_id, "status": "success"}

    try:
        await rate_limiter.acquire()
        url = "https://api.dhan.co/v2/orders"
        payload = {
            "dhanClientId": DHAN_CLIENT_ID,
            "exchangeSegment": "NSE_EQ",
            "securityId": str(security_id),
            "transactionType": transaction_type,
            "orderType": "MARKET",
            "productType": "INTRADAY",
            "quantity": quantity,
        }

        session = await api_client.get_session()
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                order_id = data.get("orderId")
                if order_id:
                    trade_logger.info(
                        f"MARKET ORDER PLACED | {security_id} | {transaction_type} | "
                        f"Qty: {quantity} | OrderID: {order_id}"
                    )
                    return data
                else:
                    trade_logger.error(f"Market order failed - no order ID: {data}")
            else:
                text = await response.text()
                trade_logger.error(f"Market order HTTP error {response.status}: {text}")
        return None
    except Exception as e:
        trade_logger.error(f"Market order exception: {str(e)}")
        return None


async def fetch_realtime_quote(security_ids: List[int]) -> Dict[int, Optional[Dict]]:
    """Enhanced quote fetching with proper error handling"""
    if not security_ids:
        return {}

    # Simulation mode: read the latest close from combined_data files
    if SIMULATION_MODE:
        results: Dict[int, Optional[Dict]] = {}
        for sid in security_ids:
            try:
                df = SIM_DATA_CACHE.get(sid)
                if df is None:
                    df = await get_simulated_combined_data(sid)
                if df is not None and not df.empty:
                    last_row = df.iloc[-1]
                    ts = last_row.get("datetime", datetime.now(IST))
                    try:
                        if isinstance(ts, pd.Timestamp):
                            ts = ts.to_pydatetime()
                        if ts.tzinfo is None:
                            ts = IST.localize(ts)
                    except Exception:
                        ts = datetime.now(IST)
                    results[sid] = {"price": float(last_row["close"]), "timestamp": ts}
                else:
                    results[sid] = None
            except Exception as e:
                logger.error(f"Simulated quote error for {sid}: {e}")
                results[sid] = None
        return results

    # Real-time mode
    batch_size = 5
    results = {}

    for i in range(0, len(security_ids), batch_size):
        batch = security_ids[int(i) : i + batch_size]
        await rate_limiter.acquire()
        batch_results = await _fetch_quote_batch(batch)
        results.update(batch_results)

    return results


async def _fetch_quote_batch(security_ids: List[int]) -> Dict[int, Optional[Dict]]:
    """Fetch quotes for a batch of securities"""
    try:
        await asyncio.sleep(5)  # Rate limiting
        payload = {"NSE_EQ": [int(sid) for sid in security_ids]}

        # Use thread pool for synchronous dhan call
        response = await asyncio.to_thread(dhan.quote_data, payload)

        if response.get("status") == "success":
            result = {}
            for sec_id in security_ids:
                sec_id_str = str(sec_id)
                quote_data = (
                    response.get("data", {})
                    .get("data", {})
                    .get("NSE_EQ", {})
                    .get(sec_id_str)
                )
                if quote_data:
                    try:
                        trade_time = datetime.strptime(
                            quote_data["last_trade_time"], "%d/%m/%Y %H:%M:%S"
                        ).replace(tzinfo=IST)
                        result[sec_id] = {
                            "price": float(quote_data["last_price"]),
                            "timestamp": trade_time,
                        }
                    except KeyError as e:
                        logger.warning(f"Missing quote data for {sec_id}: {e}")
                        result[sec_id] = None
                else:
                    result[sec_id] = None
            return result

        elif response.get("status") == "failure":
            logger.warning(f"Quote API failed for {security_ids}: {response}")
            return {sec_id: None for sec_id in security_ids}
        else:
            logger.error(f"Unexpected quote API response: {response}")
            return {sec_id: None for sec_id in security_ids}

    except Exception as e:
        logger.error(f"Batch quote error for {security_ids}: {str(e)}")
        return {sec_id: None for sec_id in security_ids}


def build_enhanced_candles(
    security_id: int, interval_minutes: int = 5
) -> Optional[List[EnhancedCandle]]:
    depth_cache = cache_manager.depth_cache.get(security_id)
    if not depth_cache:
        return None

    if security_id not in CANDLE_BUILDERS:
        CANDLE_BUILDERS[security_id] = {
            "current_candle": EnhancedCandle(),
            "last_candle_time": None,
        }

    builder = CANDLE_BUILDERS[security_id]
    candles = []

    for data_point in list(depth_cache):
        timestamp = data_point["timestamp"]
        candle_time = timestamp.replace(second=0, microsecond=0)
        minute_group = (timestamp.minute // interval_minutes) * interval_minutes
        candle_time = candle_time.replace(minute=minute_group)

        if builder["last_candle_time"] != candle_time:
            if builder["current_candle"].volume > 0:
                candles.append(builder["current_candle"])
            builder["current_candle"] = EnhancedCandle()
            builder["current_candle"].open = data_point["ltp"]
            builder["current_candle"].datetime = candle_time
            builder["last_candle_time"] = candle_time

        candle = builder["current_candle"]
        price = data_point["ltp"]
        candle.high = max(candle.high, price) if candle.high != 0 else price
        candle.low = min(candle.low, price) if candle.low != float("inf") else price
        candle.close = price
        candle.volume += data_point["volume"]
        candle.bid_qty = data_point["bid_qty"]
        candle.ask_qty = data_point["ask_qty"]
        candle.bid_depth = data_point["bid_depth"]
        candle.ask_depth = data_point["ask_depth"]

    depth_cache.clear()
    return candles


async def get_simulated_combined_data(security_id: int) -> Optional[pd.DataFrame]:
    """Async version to load simulated data"""
    try:
        # First check memory cache
        if security_id in SIM_DATA_CACHE:
            return SIM_DATA_CACHE[security_id]

        # Build file mapping on first run
        if not hasattr(get_simulated_combined_data, "file_map"):
            file_map = {}
            for fname in os.listdir(COMBINED_DATA_DIR):
                if fname.lower().endswith(".csv") and fname.startswith(
                    "combined_data_"
                ):
                    try:
                        # Extract security ID from filename pattern: combined_data_<ID>.csv
                        sec_id_str = fname.split("_")[-1].split(".")[0]
                        if sec_id_str.isdigit():
                            sec_id = int(sec_id_str)
                            file_map[sec_id] = os.path.join(COMBINED_DATA_DIR, fname)
                    except Exception as e:
                        logger.debug(f"File mapping error for {fname}: {e}")
            get_simulated_combined_data.file_map = file_map

        file_path = get_simulated_combined_data.file_map.get(security_id)

        if file_path and os.path.exists(file_path):
            # Async file reading using thread pool
            df = await run_in_thread(pd.read_csv, file_path)
            df.columns = [c.lower() for c in df.columns]

            # Handle datetime conversion
            if "datetime" in df.columns:
                try:
                    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                except Exception:
                    pass
            elif "timestamp" in df.columns:
                try:
                    df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    df = df.drop("timestamp", axis=1)
                except Exception:
                    pass

            # Filter required columns
            required_cols = ["datetime", "open", "high", "low", "close", "volume"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logger.error(f"Missing columns in {file_path}: {missing}")
                return None

            # Sort and deduplicate
            df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")

            # Cache in memory
            SIM_DATA_CACHE[security_id] = df
            return df

        logger.warning(f"No simulated data found for security {security_id}")
        return None

    except Exception as e:
        logger.error(f"Failed to load simulated data: {e}")
        return None


# Helper function for async file operations
async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(thread_pool, func, *args, **kwargs)


# Helper function for async file operations
async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(thread_pool, func, *args, **kwargs)


async def get_combined_data(security_id: int) -> Optional[pd.DataFrame]:
    """Corrected async data retrieval with proper await handling"""
    if SIMULATION_MODE:
        # Ensure we await the async simulation function
        return await get_simulated_combined_data(security_id)

    # Rest of real-time implementation
    cache_key = f"{security_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    if cache_key in cache_manager.historical_cache:
        cache_manager.cache_hits["historical"] += 1
        return cache_manager.historical_cache[cache_key]

    try:
        # Use the enhanced cached version from live_data
        combined_data = await get_combined_data_with_persistent_live(
            security_id=int(security_id),
            exchange_segment="NSE_EQ",
            auto_start_live_collection=True,
        )

        if combined_data is None:
            cache_manager.cache_misses["historical"] += 1
            return None

        # Build enhanced candles
        enhanced_candles = build_enhanced_candles(security_id)
        if enhanced_candles:
            enhanced_data = [
                {
                    "datetime": candle.datetime,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "bid_qty": candle.bid_qty,
                    "ask_qty": candle.ask_qty,
                    "bid_depth": candle.bid_depth,
                    "ask_depth": candle.ask_depth,
                }
                for candle in enhanced_candles
            ]
            enhanced_df = pd.DataFrame(enhanced_data)
            if not enhanced_df.empty:
                combined_data = pd.concat(
                    [combined_data, enhanced_df], ignore_index=True
                )
                combined_data = (
                    combined_data.sort_values("datetime")
                    .drop_duplicates("datetime", keep="last")
                    .reset_index(drop=True)
                )

        # Cache in short-term cache
        cache_manager.historical_cache[cache_key] = combined_data
        return combined_data

    except Exception as e:
        logger.error(f"Error in get_combined_data for {security_id}: {e}")
        return None


async def cache_warmup_for_trading():
    """
    Warm up the cache before trading starts.
    This should be called during pre-market hours.
    """
    try:
        logger.info("🔥 Starting cache warmup for trading session")

        # Get all securities from strategies
        strategies_df = pd.read_csv("csv/selected_stocks_strategies.csv")
        nifty500 = pd.read_csv("csv/ind_nifty500list.csv")

        ticker_to_security = nifty500.set_index("ticker")["security_id"].to_dict()
        unique_tickers = strategies_df["Ticker"].unique()

        security_ids = [
            ticker_to_security[ticker]
            for ticker in unique_tickers
            if ticker in ticker_to_security
        ]

        logger.info(f"Warming cache for {len(security_ids)} securities")

        # Batch process to avoid overwhelming the system
        batch_size = 5
        successful_cache = 0

        for i in range(0, len(security_ids), batch_size):
            batch = security_ids[i : i + batch_size]
            batch_tasks = []

            for security_id in batch:
                task = asyncio.create_task(get_combined_data(security_id))
                batch_tasks.append(task)

            try:
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for j, result in enumerate(results):
                    if not isinstance(result, Exception) and result is not None:
                        successful_cache += 1
                        logger.debug(f"✅ Cached data for security {batch[j]}")
                    else:
                        logger.warning(
                            f"⚠️ Failed to cache data for security {batch[j]}"
                        )

            except Exception as e:
                logger.error(f"Error in cache warmup batch: {e}")

            # Small delay between batches
            await asyncio.sleep(2)

        # Get final cache statistics
        cache_stats = get_cache_info()
        logger.info(
            f"✅ Cache warmup complete: {successful_cache}/{len(security_ids)} securities cached"
        )
        logger.info(f"📊 Historical cache stats: {cache_stats}")

        return successful_cache, len(security_ids)

    except Exception as e:
        logger.error(f"Error during cache warmup: {e}")
        return 0, 0


async def fetch_dynamic_holidays(year: int) -> List[str]:
    if year in HOLIDAY_CACHE:
        return HOLIDAY_CACHE[year]

    try:
        url = "https://www.nseindia.com/api/holiday-master?type=trading"
        session = await api_client.get_session()
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                holidays = [
                    d["tradingDate"]
                    for d in data.get("data", [])
                    if d.get("trading") == "N"
                ]
                logger.info(f"Fetched {len(holidays)} trading holidays for {year}")
                HOLIDAY_CACHE[year] = holidays
                return holidays
    except Exception as e:
        logger.warning(f"Failed to fetch holidays: {e}")

    holidays = [
        "2025-01-26",
        "2025-03-14",
        "2025-03-29",
        "2025-04-11",
        "2025-04-17",
        "2025-05-01",
        "2025-06-17",
        "2025-07-17",
        "2025-08-15",
        "2025-09-05",
        "2025-10-02",
        "2025-10-23",
        "2025-11-12",
        "2025-12-25",
    ]
    HOLIDAY_CACHE[year] = holidays
    return holidays


@lru_cache(maxsize=100)
def calculate_vwap_cached(
    hist_data_hash: int, volume_sum: float, typical_price_volume_sum: float
) -> float:
    return typical_price_volume_sum / volume_sum if volume_sum != 0 else 0


async def calculate_vwap(hist_data: pd.DataFrame) -> float:
    try:
        if hist_data is None or len(hist_data) == 0:
            return 0
        typical_price = (hist_data["high"] + hist_data["low"] + hist_data["close"]) / 3
        volume_sum = hist_data["volume"].sum()
        typical_price_volume_sum = (typical_price * hist_data["volume"]).sum()
        data_hash = hash(str(hist_data.shape) + str(volume_sum))
        return calculate_vwap_cached(data_hash, volume_sum, typical_price_volume_sum)
    except Exception as e:
        logger.error(f"VWAP calculation error: {e}")
        return 0


@lru_cache(maxsize=200)
def calculate_regime_cached(adx_val: float, atr_val: float) -> Tuple[str, float, float]:
    if adx_val > 25:
        return "trending", adx_val, atr_val
    elif adx_val < 20:
        return "range_bound", adx_val, atr_val
    return "transitional", adx_val, atr_val


def calculate_regime(
    data: pd.DataFrame, adx_period: int = 14
) -> Tuple[str, float, float]:
    if len(data) < adx_period:
        return "unknown", 0.0, 0.0
    try:
        adx = ta.adx(data["high"], data["low"], data["close"], length=adx_period)
        atr = ta.atr(data["high"], data["low"], data["close"], length=adx_period)
        if isinstance(adx, pd.Series):
            latest_adx = (
                adx.iloc[-1] if len(adx) > 0 and not pd.isna(adx.iloc[-1]) else 0.0
            )
        elif isinstance(adx, pd.DataFrame):
            adx_col = adx.columns[0] if len(adx.columns) > 0 else adx.columns[-1]
            latest_adx = (
                adx[adx_col].iloc[-1]
                if len(adx) > 0 and not pd.isna(adx[adx_col].iloc[-1])
                else 0.0
            )
        else:
            latest_adx = float(adx) if not pd.isna(adx) else 0.0
        if isinstance(atr, pd.Series):
            latest_atr = (
                atr.iloc[-1] if len(atr) > 0 and not pd.isna(atr.iloc[-1]) else 0.0
            )
        elif isinstance(atr, pd.DataFrame):
            atr_col = atr.columns[0] if len(atr.columns) > 0 else atr.columns[-1]
            latest_atr = (
                atr[atr_col].iloc[-1]
                if len(atr) > 0 and not pd.isna(atr[atr_col].iloc[-1])
                else 0.0
            )
        else:
            latest_atr = float(atr) if not pd.isna(atr) else 0.0
        return calculate_regime_cached(latest_adx, latest_atr)
    except Exception as e:
        logger.error(f"Regime calculation error: {str(e)}")
        return "unknown", 0.0, 0.0


@lru_cache(maxsize=500)
def calculate_risk_params_cached(
    regime: str, atr: float, current_price: float, direction: str, account_size: float
) -> Dict[str, float]:
    risk_per_trade = 0.01 * account_size
    if atr <= 0:
        atr = current_price * 0.01
    params_map = {
        "trending": {"sl_mult": 2.0, "tp_mult": 3.0, "risk_factor": 0.8},
        "range_bound": {"sl_mult": 1.5, "tp_mult": 2.0, "risk_factor": 1.0},
        "transitional": {"sl_mult": 1.8, "tp_mult": 2.5, "risk_factor": 0.9},
        "unknown": {"sl_mult": 1.8, "tp_mult": 2.5, "risk_factor": 0.9},
    }
    cfg = params_map.get(regime, params_map["unknown"])
    stop_loss_distance = atr * cfg["sl_mult"]
    position_size = min(
        MAX_QUANTITY,
        max(1, int((risk_per_trade / stop_loss_distance) * cfg["risk_factor"])),
    )
    if direction == "BUY":
        stop_loss = current_price - stop_loss_distance
        take_profit = current_price + (atr * cfg["tp_mult"] * 1.75)
    else:
        stop_loss = current_price + stop_loss_distance
        take_profit = current_price - (atr * cfg["tp_mult"] * 1.75)
    return {
        "position_size": position_size,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }


def calculate_risk_params(
    regime: str, atr: float, current_price: float, direction: str
) -> Dict[str, float]:
    return calculate_risk_params_cached(
        regime, atr, current_price, direction, ACCOUNT_SIZE
    )


class PnLTracker:
    def __init__(self):
        self.cache = {
            "realized": 0.0,
            "unrealized": 0.0,
            "total": 0.0,
            "last_updated": None,
        }
        self.cache_ttl = 300  # Cache TTL in seconds (5 minutes)

    async def update_daily_pnl(self) -> dict:
        """Update daily P&L using dhan.get_positions()."""
        now = datetime.now(IST)
        if (
            self.cache["last_updated"]
            and (now - self.cache["last_updated"]).total_seconds() < self.cache_ttl
        ):
            logger.debug(f"Returning cached P&L: Total ₹{self.cache['total']:.2f}")
            return {
                "realized": self.cache["realized"],
                "unrealized": self.cache["unrealized"],
                "total": self.cache["total"],
            }

        # Retry logic: attempt up to 3 times with exponential backoff
        for attempt in range(3):
            try:
                await rate_limiter.acquire()
                # Run synchronous dhan.get_positions() in a separate thread
                response = await asyncio.to_thread(dhan.get_positions)
                if response.get("status") == "success" and response.get("data"):
                    realized_pnl = 0.0
                    unrealized_pnl = 0.0

                    # Process each position
                    for position in response["data"]:
                        try:
                            realized_pnl += float(position.get("realizedProfit", 0.0))
                            unrealized_pnl += float(
                                position.get("unrealizedProfit", 0.0)
                            )
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Error parsing P&L values for {position.get('tradingSymbol', 'Unknown')}: {e}"
                            )
                            continue

                    total_pnl = realized_pnl + unrealized_pnl

                    # Update cache
                    self.cache["realized"] = realized_pnl
                    self.cache["unrealized"] = unrealized_pnl
                    self.cache["total"] = total_pnl
                    self.cache["last_updated"] = now

                    logger.info(
                        f"Updated P&L - Realized: ₹{realized_pnl:.2f}, Unrealized: ₹{unrealized_pnl:.2f}, Total: ₹{total_pnl:.2f}"
                    )

                    return {
                        "realized": realized_pnl,
                        "unrealized": unrealized_pnl,
                        "total": total_pnl,
                    }

                elif response.get("status") == "success" and not response.get("data"):
                    # No positions found - this is valid
                    logger.info("No open positions found")
                    self.cache["realized"] = 0.0
                    self.cache["unrealized"] = 0.0
                    self.cache["total"] = 0.0
                    self.cache["last_updated"] = now

                    return {"realized": 0.0, "unrealized": 0.0, "total": 0.0}

                else:
                    logger.warning(
                        f"API call failed: {response.get('remarks', 'No remarks')}"
                    )
                    # Don't return 0 on first attempt, try again
                    if attempt == 2:  # Last attempt
                        return {
                            "realized": self.cache.get("realized", 0.0),
                            "unrealized": self.cache.get("unrealized", 0.0),
                            "total": self.cache.get("total", 0.0),
                        }

            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < 2:  # Not the last attempt
                    await asyncio.sleep(2**attempt)  # Exponential backoff: 1s, 2s, 4s
                else:
                    logger.error("Failed to update P&L after 3 attempts")
                    # Return cached values or zeros
                    return {
                        "realized": self.cache.get("realized", 0.0),
                        "unrealized": self.cache.get("unrealized", 0.0),
                        "total": self.cache.get("total", 0.0),
                    }

        # This shouldn't be reached, but just in case
        return {
            "realized": self.cache.get("realized", 0.0),
            "unrealized": self.cache.get("unrealized", 0.0),
            "total": self.cache.get("total", 0.0),
        }

    def get_current_pnl(self) -> dict:
        """Get current P&L without making API call."""
        return {
            "realized": self.cache.get("realized", 0.0),
            "unrealized": self.cache.get("unrealized", 0.0),
            "total": self.cache.get("total", 0.0),
            "last_updated": self.cache.get("last_updated"),
        }

    def get_position_summary(self, response_data: list) -> str:
        """Generate a summary of current positions."""
        if not response_data:
            return "No open positions"

        summary = []
        for pos in response_data:
            symbol = pos.get("tradingSymbol", "Unknown")
            position_type = pos.get("positionType", "Unknown")
            net_qty = pos.get("netQty", 0)
            unrealized = float(pos.get("unrealizedProfit", 0.0))

            summary.append(
                f"{symbol}: {position_type} {abs(net_qty)} (₹{unrealized:+.2f})"
            )

        return " | ".join(summary)


pnl_tracker = PnLTracker()


async def calculate_stock_volatility(security_id: int) -> float:
    cache_key = f"vol_{security_id}_{date.today()}"
    if cache_key in cache_manager.volatility_cache:
        cache_manager.cache_hits["volatility"] += 1
        return cache_manager.volatility_cache[cache_key]

    try:
        live_data = read_live_data_from_csv(security_id=security_id)
        if live_data is None or len(live_data) < 20:
            cache_manager.cache_misses["volatility"] += 1
            return 0

        returns = live_data["close"].tail(20).pct_change().dropna()
        volatility = returns.std()
        cache_manager.volatility_cache[cache_key] = volatility
        cache_manager.cache_hits["volatility"] += 1
        cache_manager.log_cache_stats("volatility")
        return volatility
    except Exception as e:
        logger.error(f"Volatility calculation error for {security_id}: {e}")
        cache_manager.cache_misses["volatility"] += 1
        return 0


async def calculate_average_volume(security_id: int) -> float:
    cache_key = f"vol_{security_id}_{date.today()}"
    try:
        live_data = read_live_data_from_csv(security_id=security_id)
        if live_data is None or len(live_data) < 20:
            cache_manager.cache_misses["volume"] += 1
            return 0
        avg_volume = live_data["volume"].tail(20).mean()
        cache_manager.volume_cache[cache_key] = avg_volume
        cache_manager.cache_hits["volume"] += 1
        cache_manager.log_cache_stats("volume")
        return avg_volume
    except Exception as e:
        logger.error(f"Volume calculation error for {security_id}: {e}")
        cache_manager.cache_misses["volume"] += 1
        return 0


class AdaptiveSemaphore:
    def __init__(self, initial_limit: int = 25):
        self.semaphore = asyncio.Semaphore(initial_limit)

    async def acquire(self):
        return await self.semaphore.acquire()

    def release(self):
        self.semaphore.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


adaptive_semaphore = AdaptiveSemaphore(25)


async def schedule_square_off():
    while True:
        try:
            now = datetime.now(IST)
            target_time = IST.localize(datetime.combine(now.date(), SQUARE_OFF_TIME))
            if now > target_time:
                target_time += timedelta(days=1)
            sleep_seconds = (target_time - now).total_seconds()
            if sleep_seconds > 0:
                await asyncio.sleep(min(sleep_seconds, 3600))
                if await market_hours_check():
                    async with position_manager.position_lock:
                        for order_id in list(position_manager.open_positions.keys()):
                            pos = position_manager.open_positions[order_id]
                            direction = "SELL" if pos["direction"] == "BUY" else "BUY"
                            await place_market_order(
                                pos["security_id"], direction, pos["quantity"]
                            )
                            await position_manager.close_position(
                                order_id, pos["entry_price"]
                            )
                            await send_telegram_alert(
                                f"*{pos['ticker']} SQUARED OFF* 🛑\nPrice: Market"
                            )
            else:
                await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Square off scheduler error: {e}")
            await asyncio.sleep(300)


async def send_enhanced_heartbeat():
    """Enhanced heartbeat with cache information."""
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour

            active_tasks = len([t for t in asyncio.all_tasks() if not t.done()])

            # Get cache sizes from both caching systems
            short_term_cache_sizes = {
                "depth": len(cache_manager.depth_cache),
                "historical": len(cache_manager.historical_cache),
                "volatility": len(cache_manager.volatility_cache),
                "volume": len(cache_manager.volume_cache),
            }

            # Get historical cache stats
            hist_cache_stats = get_cache_info()

            open_positions = len(position_manager.open_positions)

            message = (
                "💖 *SYSTEM HEARTBEAT*\n"
                f"Status: Operational\n"
                f"Active Tasks: {active_tasks}\n"
                f"Open Positions: {open_positions}\n"
                f"Short-term Cache: {short_term_cache_sizes}\n"
                f"Historical Cache: {hist_cache_stats['disk_cache_entries']} entries\n"
                f"Cached Securities: {hist_cache_stats['total_unique_securities']}\n"
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            )
            await send_telegram_alert(message)

        except Exception as e:
            logger.error(f"Enhanced heartbeat error: {e}")


async def cleanup_resources():
    while True:
        try:
            await asyncio.sleep(1800)
            cache_manager.log_cache_stats("depth")
            cache_manager.log_cache_stats("historical")
            cache_manager.log_cache_stats("volatility")
            cache_manager.log_cache_stats("volume")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def cache_maintenance():
    """
    Periodic cache maintenance to keep the system optimized.
    """
    while True:
        try:
            await asyncio.sleep(7200)  # Every 2 hours

            logger.info("🧹 Running cache maintenance")

            # Log cache statistics
            cache_manager.log_cache_stats("depth")
            cache_manager.log_cache_stats("historical")
            cache_manager.log_cache_stats("volatility")
            cache_manager.log_cache_stats("volume")

            # Get historical cache stats
            hist_stats = get_cache_info()
            logger.info(f"Historical cache stats: {hist_stats}")

            # Optional: Clear cache for current day if we're past market hours
            now = datetime.now(IST)
            market_close = time(15, 30)

            if now.time() > market_close:
                # Market is closed, we can clear today's intraday cache
                # But keep historical cache as it's still valid
                logger.info("Market closed - clearing intraday caches")

                # Clear short-term caches
                cache_manager.depth_cache.clear()
                # Keep historical cache as it contains valuable data
                clear_cache_for_today()  # Add this call

        except Exception as e:
            logger.error(f"Cache maintenance error: {e}")


# Cache for market times to avoid repeated calculations
market_times_cache = {}


def get_market_times_cached(day_date=None):
    """Get market times with caching - FIXED timezone handling."""
    if day_date is None:
        day_date = datetime.now(IST).date()

    # Check cache first
    if day_date in market_times_cache:
        return market_times_cache[day_date]

    try:
        # FIXED: Use replace(tzinfo=IST) instead of IST.localize()
        market_open_dt = datetime.combine(day_date, MARKET_OPEN_TIME).replace(
            tzinfo=IST
        )
        market_close_dt = datetime.combine(day_date, MARKET_CLOSE_TIME).replace(
            tzinfo=IST
        )
        trading_end_dt = datetime.combine(day_date, TRADING_END_TIME).replace(
            tzinfo=IST
        )

        # Cache the results
        market_times_cache[day_date] = (market_open_dt, market_close_dt, trading_end_dt)

        return market_open_dt, market_close_dt, trading_end_dt

    except Exception as e:
        logger.error(f"Error calculating market times for {day_date}: {str(e)}")
        # Return default times for today as fallback
        now = datetime.now(IST)
        return (
            datetime.combine(now.date(), MARKET_OPEN_TIME).replace(tzinfo=IST),
            datetime.combine(now.date(), MARKET_CLOSE_TIME).replace(tzinfo=IST),
            datetime.combine(now.date(), TRADING_END_TIME).replace(tzinfo=IST),
        )


async def market_hours_check():
    """Enhanced market hours check with proper timezone handling."""
    try:
        now = datetime.now(IST)
        today = now.date()

        # Skip weekends (Saturday=5, Sunday=6)
        if now.weekday() >= 5:
            logger.debug(f"Market closed - Weekend ({now.strftime('%A')})")
            return False

        # Get today's market times
        market_open_dt, market_close_dt, trading_end_dt = get_market_times_cached(today)

        # CORRECTED: Use market_close_dt instead of trading_end_dt for session check
        if market_open_dt <= now <= market_close_dt:
            return True

        # Log status for debugging
        if now < market_open_dt:
            time_to_open = (market_open_dt - now).total_seconds() / 60
            logger.debug(f"Market opens in {time_to_open:.1f} minutes")
        # CORRECTED: Check against market_close_dt instead of trading_end_dt
        elif now > market_close_dt:
            logger.debug("Market closed for the day")

        return False

    except Exception as e:
        logger.error(f"Error in market hours check: {str(e)}")
        return False  # Conservative fallback


def is_trading_day(date_obj=None):
    """Check if given date is a trading day (not weekend, could be extended for holidays)."""
    if date_obj is None:
        date_obj = datetime.now(IST).date()

    # Skip weekends
    return date_obj.weekday() < 5


def get_next_trading_day(date_obj=None):
    """Get the next trading day from given date."""
    if date_obj is None:
        date_obj = datetime.now(IST).date()

    next_day = date_obj + timedelta(days=1)

    # Skip weekends
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)

    return next_day


def time_until_market_open():
    """Get seconds until market opens (today or next trading day)."""
    now = datetime.now(IST)
    today = now.date()

    # Get today's market open time
    market_open_dt = datetime.combine(today, MARKET_OPEN_TIME).replace(tzinfo=IST)

    # If market hasn't opened today and it's a trading day
    if now < market_open_dt and is_trading_day(today):
        return (market_open_dt - now).total_seconds()

    # Otherwise, get next trading day's market open
    next_trading_day = get_next_trading_day(today)
    next_market_open = datetime.combine(next_trading_day, MARKET_OPEN_TIME).replace(
        tzinfo=IST
    )

    return (next_market_open - now).total_seconds()


def time_until_market_close():
    """Get seconds until market closes today (returns None if market is closed)."""
    now = datetime.now(IST)
    today = now.date()

    if not is_trading_day(today):
        return None

    market_open_dt, market_close_dt, trading_end_dt = get_market_times_cached(today)

    # If we're before market open or after market close
    if now < market_open_dt or now > trading_end_dt:
        return None

    # Return time until trading end
    return (trading_end_dt - now).total_seconds()


# Clean up cache periodically (call this daily)
def cleanup_market_times_cache():
    """Clean up old entries from market times cache."""
    global market_times_cache
    cutoff_date = datetime.now(IST).date() - timedelta(days=7)  # Keep last 7 days

    old_keys = [
        date_key for date_key in market_times_cache.keys() if date_key < cutoff_date
    ]
    for key in old_keys:
        del market_times_cache[key]

    if old_keys:
        logger.debug(f"Cleaned up {len(old_keys)} old market time cache entries")


# Also fix the thread pool shutdown issue
def shutdown_thread_pool_safely(pool, timeout=30):
    """Safely shutdown thread pool with timeout."""
    try:
        # Python 3.9+ has timeout parameter
        import sys

        if sys.version_info >= (3, 9):
            pool.shutdown(wait=True, timeout=timeout)
        else:
            # For older Python versions
            pool.shutdown(wait=True)
        logger.info("Thread pool shutdown successfully")
    except TypeError:
        # Fallback for versions without timeout parameter
        pool.shutdown(wait=True)
        logger.info("Thread pool shutdown (no timeout support)")
    except Exception as e:
        logger.error(f"Error shutting down thread pool: {e}")


async def main_trading_loop_with_cache():
    """Enhanced main trading loop with intelligent caching."""
    background_tasks = []  # Initialize early to avoid UnboundLocalError

    try:
        await telegram_queue.start()
        await send_telegram_alert("🚀 Bot started with enhanced caching")
        asyncio.create_task(position_manager.monitor_positions())

        # Initialize live data system
        await initialize_live_data_from_config()

        # Pre-market cache warmup - Fixed timezone handling
        now = datetime.now(IST)
        # Use replace() instead of localize() for timezone objects
        market_open = datetime.combine(now.date(), MARKET_OPEN_TIME).replace(tzinfo=IST)

        if now < market_open:
            time_to_market = (market_open - now).total_seconds()
            if time_to_market > 1800:  # More than 30 minutes before market
                logger.info("🔥 Starting pre-market cache warmup")
                cached_count, total_count = await cache_warmup_for_trading()
                await send_telegram_alert(
                    f"📊 Cache Warmup Complete\n"
                    f"Cached: {cached_count}/{total_count} securities\n"
                    f"Ready for market open!"
                )

        try:
            strategies_df = pd.read_csv("csv/selected_stocks_strategies.csv")
            nifty500 = pd.read_csv("csv/ind_nifty500list.csv")
        except Exception as e:
            logger.critical(f"Data load failed: {str(e)}")
            await send_telegram_alert(f"❌ Data load failed: {str(e)}")
            return

        # Prepare stock universe (existing logic)
        stock_universe = []
        ticker_to_security = nifty500.set_index("ticker")["security_id"].to_dict()

        for ticker in strategies_df["Ticker"].unique():
            if ticker in ticker_to_security:
                stock_data = strategies_df[strategies_df["Ticker"] == ticker]
                stock_universe.append(
                    {
                        "ticker": ticker,
                        "security_id": ticker_to_security[ticker],
                        "strategies": stock_data.to_dict("records"),
                    }
                )

        logger.info(f"Prepared {len(stock_universe)} stocks for trading")

        # Enhanced background tasks with cache maintenance
        background_tasks = [
            asyncio.create_task(schedule_square_off()),
            asyncio.create_task(send_enhanced_heartbeat()),  # Enhanced version
            asyncio.create_task(cache_maintenance()),  # New task
        ]

        logger.info(f"Started {len(background_tasks)} background tasks")

        batch_size = 3
        loop_count = 0
        INDIVIDUAL_TASK_TIMEOUT = 180  # Seconds per stock
        BATCH_TIMEOUT = 300  # Seconds per batch (up from 50)
        while await market_hours_check():
            loop_count += 1
            start_time = datetime.now(IST)

            logger.debug(f"Starting trading loop iteration {loop_count}")

            for i in range(0, len(stock_universe), batch_size):
                batch = stock_universe[i : i + batch_size]
                batch_tasks = []

                # Create tasks with individual timeouts
                for s in batch:
                    task = asyncio.create_task(
                        asyncio.wait_for(
                            process_stock_with_exit_monitoring(
                                s["ticker"], s["security_id"], s["strategies"]
                            ),
                            timeout=INDIVIDUAL_TASK_TIMEOUT,
                        )
                    )
                    batch_tasks.append(task)

                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=BATCH_TIMEOUT,
                    )

                    # Enhanced timeout handling
                    for j, result in enumerate(results):
                        ticker = batch[j]["ticker"]
                        if isinstance(result, asyncio.TimeoutError):
                            logger.warning(f"Timeout processing {ticker}")
                        elif isinstance(result, Exception):
                            logger.error(f"Error processing {ticker}: {result}")

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Batch timeout - cancelling {len(batch_tasks)} tasks"
                    )
                    for task in batch_tasks:
                        if not task.done():
                            task.cancel()
                    # Log specific tickers causing delay
                    for s in batch:
                        logger.debug(f"Pending: {s['ticker']}")
                except Exception as e:
                    logger.error(f"Error in batch processing: {str(e)}")

                # Small delay between batches
                await asyncio.sleep(1)

            elapsed = (datetime.now(IST) - start_time).total_seconds()
            sleep_time = max(30 - elapsed, 5)

            if loop_count % 10 == 0:
                logger.info(
                    f"Completed loop {loop_count}, elapsed: {elapsed:.2f}s, sleeping: {sleep_time:.1f}s"
                )

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        logger.info("Market hours ended - exiting trading loop")

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt - shutting down gracefully")
        await send_telegram_alert("⏹️ Bot stopped by user")

    except Exception as e:
        logger.critical(f"Enhanced main loop failure: {str(e)}")
        logger.error(traceback.format_exc())
        await send_telegram_alert(f"*CRITICAL ERROR*\nTrading stopped: {str(e)}")

    finally:
        logger.info("Starting cleanup...")

        # Close API client
        try:
            await api_client.close()
            logger.info("API client closed")
        except Exception as e:
            logger.error(f"Error closing API client: {e}")

        # Shutdown thread pool
        try:
            shutdown_thread_pool_safely(thread_pool, timeout=30)
        except Exception as e:
            logger.error(f"Error shutting down thread pool: {e}")

        # Cancel background tasks
        if background_tasks:
            logger.info(f"Cancelling {len(background_tasks)} background tasks...")
            for task in background_tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to finish cancellation
            try:
                await asyncio.wait(background_tasks, timeout=10.0)
                logger.info("Background tasks cancelled")
            except asyncio.TimeoutError:
                logger.warning("Some background tasks didn't cancel in time")

        # Final telegram notification
        try:
            await send_telegram_alert("🛑 Bot shutdown complete")
        except Exception as e:
            logger.error(f"Failed to send shutdown alert: {e}")

        logger.info("Cleanup complete")


async def main_simulation_loop():
    """Continuous simulation run using offline data files"""
    try:
        await telegram_queue.start()
        logger.info("Starting continuous simulation mode")

        # Initialize position manager
        asyncio.create_task(position_manager.monitor_positions())

        try:
            strategies_df = pd.read_csv("csv/selected_stocks_strategies.csv")
            nifty500 = pd.read_csv("csv/ind_nifty500list.csv")
        except Exception as e:
            logger.critical(f"Data load failed: {str(e)}")
            return

        stock_universe = []
        ticker_to_security = nifty500.set_index("ticker")["security_id"].to_dict()
        for ticker in strategies_df["Ticker"].unique():
            if ticker in ticker_to_security:
                stock_data = strategies_df[strategies_df["Ticker"] == ticker]
                stock_universe.append(
                    {
                        "ticker": ticker,
                        "security_id": int(ticker_to_security[ticker]),
                        "strategies": stock_data.to_dict("records"),
                    }
                )

        logger.info(f"Prepared {len(stock_universe)} stocks for simulation")

        # Start background tasks

        asyncio.create_task(position_manager.monitor_positions()),
        asyncio.create_task(send_enhanced_heartbeat()),

        # Continuous processing loop
        batch_size = 5
        while True:
            start_time = datetime.now(IST)
            opened_positions = 0

            # Process stocks in batches
            for i in range(0, len(stock_universe), batch_size):
                batch = stock_universe[i : i + batch_size]
                tasks = [
                    asyncio.create_task(
                        process_stock_with_exit_monitoring(
                            s["ticker"], s["security_id"], s["strategies"]
                        )
                    )
                    for s in batch
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(0)  # Yield control

                # Count new positions
                for s in batch:
                    if await position_manager.has_position(s["security_id"]):
                        opened_positions += 1

            # Log progress
            logger.info(f"Processed batch. Positions opened: {opened_positions}")

            # Add position monitoring
            await position_manager.monitor_positions()

            # Throttle processing
            elapsed = (datetime.now(IST) - start_time).total_seconds()
            sleep_time = max(30 - elapsed, 5)
            await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        logger.info("Simulation cancelled by user")
    except Exception as e:
        logger.critical(f"Simulation failure: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    try:
        # Re-parse arguments for main execution
        parser = argparse.ArgumentParser(description="Intraday Signal Generator")
        parser.add_argument(
            "--mode",
            choices=["realtime", "simulate"],
            default="realtime",
            help="Run mode: realtime (default) or simulate (use combined_data)",
        )
        parser.add_argument(
            "--simulate", action="store_true", help="Shortcut for --mode simulate"
        )
        args = parser.parse_args()

        # Update SIMULATION_MODE based on final args
        SIMULATION_MODE = args.simulate or args.mode == "simulate"

        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        if SIMULATION_MODE:
            logger.info("Starting SIMULATION MODE")
            asyncio.run(main_simulation_loop())
        else:
            logger.info("Starting LIVE TRADING MODE")
            # logger.info(f"Starting main trading loop with cache: Monitoring positions:{monitor_task}")
            asyncio.run(main_trading_loop_with_cache())

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.critical(f"System failure: {str(e)}")
        logger.error(traceback.format_exc())
