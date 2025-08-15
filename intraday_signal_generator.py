import asyncio
import pickle
import pandas as pd
import os
import logging
from datetime import date, datetime, timedelta, time
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
from live_data import (
    get_combined_data_with_persistent_live,
    read_live_data_from_csv,
    initialize_live_data_from_config,
    CONFIG,
)


# Initialize Dhan client
dhan = init_dhan_client()

IST = pytz.timezone("Asia/Kolkata")


class DateTimeFormatter(logging.Formatter):
    """Custom formatter with IST datetime"""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=IST)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " IST"


# Setup logging
formatter = DateTimeFormatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger().handlers.clear()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
file_handler = logging.FileHandler("trading_system.log", mode="a")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)
logger = logging.getLogger("quant_trader")
logger.setLevel(logging.INFO)
trade_logger = logging.getLogger("trade_execution")
trade_logger.setLevel(logging.INFO)
trade_file_handler = logging.FileHandler("trades.log", mode="a")
trade_file_handler.setFormatter(formatter)
trade_logger.addHandler(trade_file_handler)
trade_logger.propagate = False
logger.info("Logging system initialized with IST timestamps")
trade_logger.info("Trade logging system initialized")

# Environment configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "1000000003")

if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_ACCESS_TOKEN]):
    logger.critical("Missing required environment variables")
    raise EnvironmentError("Required environment variables not set")

# Market configuration
MARKET_OPEN_TIME = CONFIG["MARKET_OPEN"]
MARKET_CLOSE_TIME = CONFIG["MARKET_CLOSE"]
TRADING_END_TIME = time.fromisoformat(os.getenv("TRADING_END", "15:20:00"))
FORCE_CLOSE_TIME = time.fromisoformat(os.getenv("FORCE_CLOSE", "15:15:00"))
SQUARE_OFF_TIME = time.fromisoformat(os.getenv("SQUARE_OFF_TIME", "15:16:00"))

# Trading configuration
MIN_VOTES = int(os.getenv("MIN_VOTES", 2))
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", 100000))
MAX_QUANTITY = int(os.getenv("MAX_QUANTITY", 2))
MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS", 0.02))
VOLATILITY_THRESHOLD = float(os.getenv("VOLATILITY_THRESHOLD", 0.012))
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", 100))
BID_ASK_THRESHOLD = int(os.getenv("BID_ASK_THRESHOLD", 500))
RELATIVE_VOLUME_THRESHOLD = float(os.getenv("RELATIVE_VOLUME_THRESHOLD", 1.2))
MIN_PRICE_THRESHOLD = float(os.getenv("MIN_PRICE_THRESHOLD", 50.0))
MAX_PRICE_THRESHOLD = float(os.getenv("MAX_PRICE_THRESHOLD", 5000.0))
# Add new environment variable
QUOTE_API_RATE_LIMIT = int(os.getenv("QUOTE_API_RATE_LIMIT", 60))  # Default: 60/min

# Initialize quote-specific rate limiter
# Initialize thread pool
dhan_lock = asyncio.Lock()
thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dhan_worker")

# Preload symbol map
try:
    nifty500_df = pd.read_csv("ind_nifty500list.csv")
    TICKER_TO_ID_MAP = nifty500_df.set_index("ticker")["security_id"].to_dict()
except Exception as e:
    logger.error(f"Failed to load symbol map: {e}")
    TICKER_TO_ID_MAP = {}

# Initialize caches and managers
HOLIDAY_CACHE = {}
CANDLE_BUILDERS = {}

STRATEGY_REGISTRY = {}


def register_strategy(name: str, strategy_class):
    """Register a new strategy in the framework with multiple aliases.

    Args:
        name (str): Unique name for the strategy.
        strategy_class: Backtrader strategy class to register.

    Raises:
        ValueError: If the name is already registered or strategy_class is invalid.

    Example:
        >>> from ema_rsi import EMARSI
        >>> register_strategy("EMARSI", EMARSI)
    """
    import re

    if not isinstance(name, str) or not name:
        raise ValueError("Strategy name must be a non-empty string")

    # Only register the provided name and the class name as aliases
    aliases = set()
    aliases.add(name)
    aliases.add(strategy_class.__name__)

    for alias in aliases:
        if alias in STRATEGY_REGISTRY:
            logger.warning(f"Strategy alias '{alias}' already registered, overwriting")
        STRATEGY_REGISTRY[alias] = strategy_class
        logger.info(f"Registered strategy alias: {alias}")


def get_strategy(strategy_name: str):
    """Retrieve a strategy class by name.

    Args:
        strategy_name (str): Name of the strategy to retrieve.

    Returns:
        The strategy class or None if not found.

    Raises:
        KeyError: If the strategy_name is not registered.

    Example:
        >>> strategy = get_strategy("EMARSI")
    """
    strategy = STRATEGY_REGISTRY.get(strategy_name)
    if strategy is None:
        logger.error(f"Strategy '{strategy_name}' not found in registry")
        raise KeyError(f"Strategy '{strategy_name}' not found")
    return strategy


try:
    from live_strategies.trendline_williams import TrendlineWilliams
    from live_strategies.vwap_bounce_rejection import VWAPBounceRejection
    from live_strategies.BB_PivotPoints_Strategy import BBPivotPointsStrategy
    from live_strategies.BB_VWAP_Strategy import BBVWAPStrategy
    from live_strategies.rsi_bb import RSIBB
    from live_strategies.sr_rsi import SRRSI
    from live_strategies.EMAStochasticPullback import EMAStochasticPullback
    from live_strategies.rsi_adx import RSIADX
    from live_strategies.RSI_Supertrend_Intraday import RSISupertrendIntraday
    from live_strategies.sr_rsi_volume import SRRSIVolume
    from live_strategies.supertrend_cci_cmf import SupertrendCCICMF
    from live_strategies.volume_atr_price_action import Volume_ATR_PriceAction
    from live_strategies.rsi_cci import RSICCI
    from live_strategies.head_shoulders_confirmation import HeadShouldersConfirmation
    from live_strategies.ema_adx import EMAADXTrend
    from live_strategies.rmbev_intraday import RMBEV
    from live_strategies.pivot_cci import PivotCCI
    from live_strategies.BB_Supertrend_Strategy import BBSupertrendStrategy

    register_strategy("TrendlineWilliams", TrendlineWilliams)
    register_strategy("VWAPBounceRejection", VWAPBounceRejection)
    register_strategy("BBPivotPointsStrategy", BBPivotPointsStrategy)
    register_strategy("BBVWAPStrategy", BBVWAPStrategy)
    register_strategy("RSIBB", RSIBB)
    register_strategy("SRRSI", SRRSI)
    register_strategy("EMAStochasticPullback", EMAStochasticPullback)
    register_strategy("RSIADX", RSIADX)
    register_strategy("RSISupertrendIntraday", RSISupertrendIntraday)
    register_strategy("SRRSIVolume", SRRSIVolume)
    register_strategy("SupertrendCCICMF", SupertrendCCICMF)
    register_strategy("Volume_ATR_PriceAction", Volume_ATR_PriceAction)
    register_strategy("RSICCI", RSICCI)
    register_strategy("HeadShouldersConfirmation", HeadShouldersConfirmation)
    register_strategy("EMAADXTrend", EMAADXTrend)
    register_strategy("RMBevIntraday", RMBEV)
    register_strategy("PivotCCI", PivotCCI)
    register_strategy("BBSupertrendStrategy", BBSupertrendStrategy)
except ImportError as e:
    logger.error(f"Failed to register EMARSI: {str(e)}")


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
        logger.debug(
            f"Cache {cache_name} - Hits: {hits}, Misses: {misses}, Hit Rate: {hits / (hits + misses):.2%}"
        )


cache_manager = CacheManager(max_size=1000, ttl=3600)


# class PositionManager:
#     def __init__(self):
#         self.open_positions = {}
#         self.positions_by_security = {}
#         self.strategy_instances = {}
#         self.max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", 10))
#         self.position_lock = asyncio.Lock()
#         self.last_trade_times = {}
#         self.last_trade_lock = asyncio.Lock()

#     def get_last_trade_time(self, ticker: str) -> Optional[datetime]:
#         return self.last_trade_times.get(ticker)

#     async def update_last_trade_time(self, ticker: str, trade_time: datetime):
#         async with self.last_trade_lock:
#             self.last_trade_times[ticker] = trade_time
#         logger.debug(f"Updated last trade time for {ticker}: {trade_time}")

#     async def add_position(
#         self,
#         order_id: str,
#         security_id: int,
#         ticker: str,
#         entry_price: float,
#         quantity: int,
#         stop_loss: float,
#         take_profit: float,
#         direction: str,
#         strategy_name: str,
#         strategy_instance=None,
#     ):
#         async with self.position_lock:
#             # Check for existing position for this security
#             if security_id in self.positions_by_security:
#                 logger.warning(
#                     f"Position already exists for {ticker} (security_id: {security_id})"
#                 )
#                 return False

#             # Enforce max positions limit
#             if len(self.open_positions) >= self.max_open_positions:
#                 logger.warning(
#                     f"Max open positions ({self.max_open_positions}) reached, cannot add {ticker}"
#                 )
#                 return False

#             # Create new position
#             self.open_positions[order_id] = {
#                 "security_id": security_id,
#                 "ticker": ticker,
#                 "entry_price": entry_price,
#                 "quantity": quantity,
#                 "stop_loss": stop_loss,
#                 "take_profit": take_profit,
#                 "direction": direction,
#                 "strategy_name": strategy_name,
#                 "entry_time": datetime.now(IST),
#                 "last_updated": datetime.now(IST),
#             }
#             self.positions_by_security[security_id] = order_id

#             # Store strategy instance for exit signal monitoring
#             if strategy_instance:
#                 self.strategy_instances[order_id] = strategy_instance

#             logger.info(
#                 f"Added position {order_id} for {ticker}: {direction} @ ₹{entry_price:.2f} using {strategy_name}"
#             )
#             return True

#     async def check_strategy_exit_signals(
#         self, security_id: int, current_data: pd.DataFrame
#     ):
#         """Check if any strategy is generating exit signals for open positions"""
#         async with self.position_lock:
#             order_id = self.positions_by_security.get(security_id)
#             if not order_id or order_id not in self.open_positions:
#                 return None

#             position = self.open_positions[order_id]
#             strategy_instance = self.strategy_instances.get(order_id)

#             if not strategy_instance:
#                 return None

#             try:
#                 # Update strategy instance with new data
#                 strategy_instance.data = current_data

#                 # Check if strategy wants to exit
#                 # For strategies with open positions, we need to check their exit conditions
#                 if hasattr(strategy_instance, "should_exit"):
#                     exit_signal = strategy_instance.should_exit()
#                     if exit_signal:
#                         return {
#                             "action": "exit",
#                             "reason": exit_signal.get("reason", "Strategy exit signal"),
#                             "price": current_data.iloc[-1]["close"],
#                         }

#                 # Alternative: Run strategy and check if it closed the position
#                 last_signal = strategy_instance.run()
#                 if (
#                     strategy_instance.open_positions == []
#                     and len(strategy_instance.completed_trades) > 0
#                 ):
#                     # Position was closed by strategy
#                     last_trade = strategy_instance.completed_trades[-1]
#                     return {
#                         "action": "exit",
#                         "reason": "Strategy completed trade",
#                         "price": last_trade["exit_price"],
#                         "pnl": last_trade["pnl"],
#                     }

#             except Exception as e:
#                 logger.error(
#                     f"Error checking exit signal for {position['ticker']}: {e}"
#                 )

#             return None

#     async def execute_strategy_exit(self, order_id: str, exit_info: dict):
#         """Execute exit based on strategy signal"""
#         async with self.position_lock:
#             if order_id not in self.open_positions:
#                 return False

#             position = self.open_positions[order_id]
#             exit_direction = "SELL" if position["direction"] == "BUY" else "BUY"

#             # Place market order for exit
#             exit_order = await place_market_order(
#                 position["security_id"], exit_direction, position["quantity"]
#             )

#             if exit_order and exit_order.get("orderId"):
#                 # Calculate P&L
#                 current_price = exit_info["price"]
#                 entry_price = position["entry_price"]
#                 quantity = position["quantity"]

#                 if position["direction"] == "BUY":
#                     pnl = (current_price - entry_price) * quantity
#                 else:
#                     pnl = (entry_price - current_price) * quantity

#                 # Send exit notification
#                 await self.send_exit_notification(
#                     position, exit_info, pnl, current_price
#                 )

#                 # Remove position
#                 await self.close_position(order_id)

#                 return True

#             return False

#     async def send_exit_notification(
#         self, position: dict, exit_info: dict, pnl: float, exit_price: float
#     ):
#         """Send detailed exit notification"""
#         pnl_emoji = "📈" if pnl > 0 else "📉"
#         status_emoji = "✅" if pnl > 0 else "❌"

#         hold_time = datetime.now(IST) - position["entry_time"]
#         hold_minutes = int(hold_time.total_seconds() / 60)

#         message = (
#             f"*{position['ticker']} EXIT SIGNAL* {status_emoji}\n"
#             f"Strategy: {position['strategy_name']}\n"
#             f"Direction: {position['direction']}\n"
#             f"Entry: ₹{position['entry_price']:.2f} → Exit: ₹{exit_price:.2f}\n"
#             f"Quantity: {position['quantity']}\n"
#             f"P&L: ₹{pnl:.2f} {pnl_emoji}\n"
#             f"Hold Time: {hold_minutes} minutes\n"
#             f"Reason: {exit_info['reason']}\n"
#             f"Time: {datetime.now(IST).strftime('%H:%M:%S')}"
#         )

#         await send_telegram_alert(message)

#         # Log the exit
#         trade_logger.info(
#             f"EXIT EXECUTED | {position['ticker']} | {position['direction']} | "
#             f"Entry: ₹{position['entry_price']:.2f} | Exit: ₹{exit_price:.2f} | "
#             f"P&L: ₹{pnl:.2f} | Reason: {exit_info['reason']}"
#         )

#     async def monitor_positions(self):
#         """Enhanced position monitoring with strategy exit signals"""
#         while True:
#             try:
#                 async with self.position_lock:
#                     now = datetime.now(IST)

#                     if not self.open_positions:
#                         await asyncio.sleep(60)
#                         continue

#                     # Check each position for exit signals
#                     for order_id, position in list(self.open_positions.items()):
#                         security_id = position["security_id"]
#                         ticker = position["ticker"]

#                         # Get current market data
#                         combined_data = await get_combined_data(security_id)
#                         if combined_data is None or len(combined_data) < 10:
#                             continue

#                         # Get current quote for price monitoring
#                         quotes = await fetch_realtime_quote([security_id])
#                         quote = quotes.get(security_id)
#                         if not quote:
#                             continue

#                         current_price = quote["price"]

#                         # 1. Check strategy exit signals first
#                         strategy_exit = await self.check_strategy_exit_signals(
#                             security_id, combined_data
#                         )
#                         if strategy_exit:
#                             logger.info(
#                                 f"Strategy exit signal for {ticker}: {strategy_exit['reason']}"
#                             )
#                             await self.execute_strategy_exit(order_id, strategy_exit)
#                             continue

#                         # 2. Check traditional stop-loss/take-profit
#                         exit_triggered = False
#                         exit_reason = ""

#                         if position["direction"] == "BUY":
#                             if current_price <= position["stop_loss"]:
#                                 exit_triggered = True
#                                 exit_reason = "Stop-loss hit"
#                             elif current_price >= position["take_profit"]:
#                                 exit_triggered = True
#                                 exit_reason = "Take-profit hit"
#                         else:  # SELL
#                             if current_price >= position["stop_loss"]:
#                                 exit_triggered = True
#                                 exit_reason = "Stop-loss hit"
#                             elif current_price <= position["take_profit"]:
#                                 exit_triggered = True
#                                 exit_reason = "Take-profit hit"

#                         if exit_triggered:
#                             exit_info = {
#                                 "action": "exit",
#                                 "reason": exit_reason,
#                                 "price": current_price,
#                             }
#                             await self.execute_strategy_exit(order_id, exit_info)
#                             continue

#                         # 3. Update trailing stops
#                         if (
#                             now - position["last_updated"]
#                         ).total_seconds() > 300:  # Every 5 minutes
#                             regime, adx_value, atr_value = calculate_regime(
#                                 combined_data
#                             )
#                             if atr_value > 0:
#                                 if position["direction"] == "BUY":
#                                     new_sl = max(
#                                         position["stop_loss"],
#                                         current_price - atr_value * 1.5,
#                                     )
#                                     if new_sl > position["stop_loss"]:
#                                         await self.update_position(
#                                             order_id, stop_loss=new_sl
#                                         )
#                                         logger.info(
#                                             f"Updated trailing stop for {ticker}: ₹{new_sl:.2f}"
#                                         )
#                                 else:
#                                     new_sl = min(
#                                         position["stop_loss"],
#                                         current_price + atr_value * 1.5,
#                                     )
#                                     if new_sl < position["stop_loss"]:
#                                         await self.update_position(
#                                             order_id, stop_loss=new_sl
#                                         )
#                                         logger.info(
#                                             f"Updated trailing stop for {ticker}: ₹{new_sl:.2f}"
#                                         )

#                 await asyncio.sleep(
#                     30
#                 )  # Check every 30 seconds for faster exit signals

#             except Exception as e:
#                 logger.error(f"Enhanced position monitor error: {e}", exc_info=True)
#                 await asyncio.sleep(300)


class PositionManager:
    def __init__(self):
        self.open_positions = {}
        self.positions_by_security = {}
        self.strategy_instances = {}
        self.max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", 10))
        self.position_lock = asyncio.Lock()
        self.last_trade_times = {}
        self.last_trade_lock = asyncio.Lock()
        self.cooldown_minutes = int(os.getenv("COOLDOWN_MINUTES", 30))

    async def get_last_trade_time(self, ticker: str) -> Optional[datetime]:
        """Get last trade time for a ticker with thread safety"""
        async with self.last_trade_lock:
            return self.last_trade_times.get(ticker)

    async def update_last_trade_time(self, ticker: str, trade_time: datetime):
        """Update last trade time for a ticker with thread safety"""
        async with self.last_trade_lock:
            self.last_trade_times[ticker] = trade_time
            logger.debug(f"Updated last trade time for {ticker}: {trade_time}")
            await self.save_trade_times()

    async def save_trade_times(self):
        """Persist trade times to disk for restart resilience"""
        try:
            with open("last_trades.pkl", "wb") as f:
                pickle.dump(self.last_trade_times, f)
        except Exception as e:
            logger.error(f"Failed to save trade times: {e}")

    async def load_trade_times(self):
        """Load trade times from disk on startup"""
        try:
            if os.path.exists("last_trades.pkl"):
                with open("last_trades.pkl", "rb") as f:
                    self.last_trade_times = pickle.load(f)
                    logger.info(
                        f"Loaded {len(self.last_trade_times)} trade times from disk"
                    )
        except Exception as e:
            logger.error(f"Failed to load trade times: {e}")

    async def add_position(
        self,
        order_id: str,
        security_id: int,
        ticker: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        take_profit: float,
        direction: str,
        strategy_name: str,
        strategy_instance=None,
    ):
        """Add a new position to the manager"""
        async with self.position_lock:
            # Check for existing position
            if security_id in self.positions_by_security:
                logger.warning(f"Position already exists for {ticker}")
                return False

            # Enforce max positions limit
            if len(self.open_positions) >= self.max_open_positions:
                logger.warning(
                    f"Max open positions reached ({self.max_open_positions})"
                )
                return False

            # Create new position
            self.open_positions[order_id] = {
                "security_id": security_id,
                "ticker": ticker,
                "entry_price": entry_price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "direction": direction,
                "strategy_name": strategy_name,
                "entry_time": datetime.now(IST),
                "last_updated": datetime.now(IST),
            }
            self.positions_by_security[security_id] = order_id

            # Store strategy instance for exit monitoring
            if strategy_instance:
                self.strategy_instances[order_id] = strategy_instance

            logger.info(
                f"Added position {order_id} for {ticker}: {direction} @ ₹{entry_price:.2f}"
            )
            return True

    async def update_position(self, order_id: str, **updates):
        """Update position parameters (e.g., trailing stop)"""
        async with self.position_lock:
            if order_id in self.open_positions:
                position = self.open_positions[order_id]
                position.update(updates)
                position["last_updated"] = datetime.now(IST)
                logger.info(f"Updated position {order_id}: {updates}")

    async def close_position(self, order_id: str):
        """Remove a position from the manager"""
        async with self.position_lock:
            if order_id in self.open_positions:
                position = self.open_positions[order_id]
                security_id = position["security_id"]

                # Remove from security index
                if security_id in self.positions_by_security:
                    del self.positions_by_security[security_id]

                # Remove strategy instance
                if order_id in self.strategy_instances:
                    del self.strategy_instances[order_id]

                # Remove position
                del self.open_positions[order_id]

                logger.info(f"Closed position {order_id} for {position['ticker']}")

    async def check_strategy_exit_signals(
        self, security_id: int, current_data: pd.DataFrame
    ):
        """Check for exit signals from strategy instances"""
        async with self.position_lock:
            order_id = self.positions_by_security.get(security_id)
            if not order_id or order_id not in self.open_positions:
                return None

            position = self.open_positions[order_id]
            strategy_instance = self.strategy_instances.get(order_id)

            if not strategy_instance:
                return None

            try:
                # Update strategy with new data
                strategy_instance.data = current_data

                # Check strategy-specific exit conditions
                if hasattr(strategy_instance, "should_exit"):
                    exit_signal = strategy_instance.should_exit()
                    if exit_signal:
                        return {
                            "action": "exit",
                            "reason": exit_signal.get("reason", "Strategy exit"),
                            "price": current_data.iloc[-1]["close"],
                        }

                # Check if strategy automatically closed position
                if hasattr(strategy_instance, "open_positions") and hasattr(
                    strategy_instance, "completed_trades"
                ):
                    if (
                        not strategy_instance.open_positions
                        and strategy_instance.completed_trades
                    ):
                        last_trade = strategy_instance.completed_trades[-1]
                        return {
                            "action": "exit",
                            "reason": "Strategy closed position",
                            "price": last_trade.get(
                                "exit_price", current_data.iloc[-1]["close"]
                            ),
                        }

            except Exception as e:
                logger.error(f"Exit signal check error for {position['ticker']}: {e}")

            return None

    async def execute_strategy_exit(self, order_id: str, exit_info: dict):
        """Execute exit based on strategy signal"""
        async with self.position_lock:
            if order_id not in self.open_positions:
                return False

            position = self.open_positions[order_id]
            exit_direction = "SELL" if position["direction"] == "BUY" else "BUY"

            # Place market order for exit
            exit_order = await place_market_order(
                position["security_id"], exit_direction, position["quantity"]
            )

            if exit_order and exit_order.get("orderId"):
                # Calculate P&L
                current_price = exit_info["price"]
                entry_price = position["entry_price"]
                quantity = position["quantity"]
                pnl = (
                    (current_price - entry_price) * quantity
                    if position["direction"] == "BUY"
                    else (entry_price - current_price) * quantity
                )

                # Send exit notification
                await self.send_exit_notification(
                    position, exit_info, pnl, current_price
                )

                # Remove position
                await self.close_position(order_id)
                return True
            return False

    async def send_exit_notification(
        self, position: dict, exit_info: dict, pnl: float, exit_price: float
    ):
        """Send detailed exit notification"""
        pnl_emoji = "📈" if pnl > 0 else "📉"
        status_emoji = "✅" if pnl > 0 else "❌"
        hold_time = datetime.now(IST) - position["entry_time"]
        hold_minutes = int(hold_time.total_seconds() / 60)

        message = (
            f"*{position['ticker']} EXIT SIGNAL* {status_emoji}\n"
            f"Strategy: {position['strategy_name']}\n"
            f"Direction: {position['direction']}\n"
            f"Entry: ₹{position['entry_price']:.2f} → Exit: ₹{exit_price:.2f}\n"
            f"Quantity: {position['quantity']}\n"
            f"P&L: ₹{pnl:.2f} {pnl_emoji}\n"
            f"Hold Time: {hold_minutes} minutes\n"
            f"Reason: {exit_info['reason']}\n"
            f"Time: {datetime.now(IST).strftime('%H:%M:%S')}"
        )

        await send_telegram_alert(message)
        trade_logger.info(
            f"EXIT | {position['ticker']} | {position['direction']} | "
            f"Entry: ₹{position['entry_price']:.2f} | Exit: ₹{exit_price:.2f} | "
            f"P&L: ₹{pnl:.2f} | Reason: {exit_info['reason']}"
        )

    async def monitor_positions(self):
        """Continuous position monitoring with exit checks"""
        while True:
            try:
                async with self.position_lock:
                    if not self.open_positions:
                        await asyncio.sleep(30)
                        continue

                    # Process each position
                    for order_id, position in list(self.open_positions.items()):
                        security_id = position["security_id"]
                        ticker = position["ticker"]

                        # Get current market data
                        combined_data = await get_combined_data(security_id)
                        if combined_data is None or len(combined_data) < 10:
                            continue

                        # Get current quote
                        quotes = await fetch_realtime_quote([security_id])
                        quote = quotes.get(security_id)
                        if not quote:
                            continue
                        current_price = quote["price"]

                        # 1. Check strategy exit signals
                        exit_signal = await self.check_strategy_exit_signals(
                            security_id, combined_data
                        )
                        if exit_signal:
                            logger.info(
                                f"{ticker} exit signal: {exit_signal['reason']}"
                            )
                            await self.execute_strategy_exit(order_id, exit_signal)
                            continue

                        # 2. Check stop-loss/take-profit
                        exit_triggered = False
                        if position["direction"] == "BUY":
                            if current_price <= position["stop_loss"]:
                                exit_info = {
                                    "reason": "Stop-loss hit",
                                    "price": current_price,
                                }
                                exit_triggered = True
                            elif current_price >= position["take_profit"]:
                                exit_info = {
                                    "reason": "Take-profit hit",
                                    "price": current_price,
                                }
                                exit_triggered = True
                        else:  # SELL
                            if current_price >= position["stop_loss"]:
                                exit_info = {
                                    "reason": "Stop-loss hit",
                                    "price": current_price,
                                }
                                exit_triggered = True
                            elif current_price <= position["take_profit"]:
                                exit_info = {
                                    "reason": "Take-profit hit",
                                    "price": current_price,
                                }
                                exit_triggered = True

                        if exit_triggered:
                            await self.execute_strategy_exit(order_id, exit_info)
                            continue

                        # 3. Update trailing stops every 5 minutes
                        now = datetime.now(IST)
                        if (now - position["last_updated"]).total_seconds() > 300:
                            regime, adx_value, atr_value = calculate_regime(
                                combined_data
                            )
                            if atr_value > 0:
                                if position["direction"] == "BUY":
                                    new_sl = max(
                                        position["stop_loss"],
                                        current_price - atr_value * 1.5,
                                    )
                                    if new_sl > position["stop_loss"]:
                                        await self.update_position(
                                            order_id, stop_loss=new_sl
                                        )
                                else:
                                    new_sl = min(
                                        position["stop_loss"],
                                        current_price + atr_value * 1.5,
                                    )
                                    if new_sl < position["stop_loss"]:
                                        await self.update_position(
                                            order_id, stop_loss=new_sl
                                        )

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(60)


position_manager = PositionManager()


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
    """Enhanced signal execution with success tracking"""
    try:
        # Add volatility filter
        volatility = await calculate_stock_volatility(security_id)
        if volatility > VOLATILITY_THRESHOLD:
            logger.warning(
                f"Skipping {ticker} due to high volatility: {volatility:.4f}"
            )
            return False

        current_pnl = await pnl_tracker.update_daily_pnl()
        if current_pnl <= -MAX_DAILY_LOSS_PERCENT * ACCOUNT_SIZE:
            message = (
                f"🛑 TRADING HALTED: Daily loss limit reached\n"
                f"Current P&L: ₹{current_pnl:.2f}\n"
                f"Limit: ₹{-MAX_DAILY_LOSS_PERCENT * ACCOUNT_SIZE:.2f}"
            )
            await send_telegram_alert(message)
            logger.critical("Daily loss limit reached - trading halted")
            return False

        quotes = await fetch_realtime_quote([security_id])
        quote = quotes.get(security_id)
        if not quote:
            logger.warning(f"Price unavailable for {ticker}")
            return False

        current_price = quote["price"]
        vwap = await calculate_vwap(hist_data)
        entry_price = (
            min(current_price, vwap * 0.998)
            if signal == "BUY"
            else max(current_price, vwap * 1.002)
        )

        risk_params = calculate_risk_params(regime, atr_value, entry_price, signal)
        now = datetime.now(IST)

        # Add emoji for direction
        direction_emoji = "[BUY]" if signal == "BUY" else "[SELL]"
        funds = dhan.get_fund_limits().get("data", {}).get("availabelBalance", 0)
        if funds > current_price / 5:
            position_size = (
                risk_params["position_size"] if funds > (entry_price / 5) * 2 else 1
            )
            message = (
                f"*{ticker} ENTRY SIGNAL* {direction_emoji}\n"
                f"Strategy: {strategy_name}\n"
                f"Direction: {signal}\n"
                f"Entry: ₹{entry_price:.2f} | VWAP: ₹{vwap:.2f}\n"
                f"Current: ₹{current_price:.2f}\n"
                f"Regime: {regime} (ADX: {adx_value:.2f})\n"
                f"Volatility: {volatility:.4f}\n"
                f"Size: {risk_params['position_size']} | SL: ₹{risk_params['stop_loss']:.2f}\n"
                f"TP: ₹{risk_params['take_profit']:.2f}\n"
                f"Risk: ₹{abs(entry_price - risk_params['stop_loss']) * risk_params['position_size']:.2f}\n"
                f"Params: {params}\n"
                f"Time: {now.strftime('%H:%M:%S')}"
            )
            logger.info(
                f"Executing {signal} signal for {ticker}: {message.replace('₹', 'Rs')}"
            )
            order_response = await place_super_order(
                security_id,
                signal,
                entry_price,
                risk_params["stop_loss"],
                risk_params["take_profit"],
                position_size,
            )

            if order_response and order_response.get("orderId"):
                # Use enhanced position manager
                await position_manager.add_position(
                    order_response["orderId"],
                    security_id,
                    ticker,
                    entry_price,
                    position_size,
                    risk_params["stop_loss"],
                    risk_params["take_profit"],
                    signal,
                    strategy_name,
                    strategy_instance,
                )
                await send_telegram_alert(message)
                return True  # Success!
            else:
                await send_telegram_alert(
                    f"*{ticker} Order Failed* ❌\nSignal: {signal} at ₹{entry_price:.2f}"
                )
                return False
        else:
            await send_telegram_alert(
                f"*{ticker} Order Failed* ❌\nSignal: {signal} at ₹{entry_price:.2f} due to insufficient funds"
            )
            return False

    except Exception as e:
        logger.error(f"Enhanced signal execution failed for {ticker}: {str(e)}")
        return False


# async def process_stock_with_exit_monitoring(
#     ticker: str, security_id: int, strategies: List[Dict]
# ) -> None:
#     """Enhanced stock processing with exit signal monitoring"""
#     async with adaptive_semaphore:
#         try:
#             logger.info(f"Processing {ticker} (ID: {security_id})")

#             combined_data = await get_combined_data(security_id)
#             if combined_data is None:
#                 logger.warning(f"{ticker} - No data available")
#                 return

#             # Check if we have existing position for this stock
#             existing_position = position_manager.positions_by_security.get(security_id)

#             if existing_position:
#                 # For existing positions, focus on exit signal monitoring
#                 exit_signal = await position_manager.check_strategy_exit_signals(
#                     security_id, combined_data
#                 )
#                 if exit_signal:
#                     logger.info(
#                         f"{ticker} - Exit signal detected: {exit_signal['reason']}"
#                     )
#                 return

#             # Regular entry signal processing (existing code)
#             data_length = len(combined_data)
#             min_bars_list = []

#             for strat in strategies:
#                 try:
#                     strategy_class = get_strategy(strat["Strategy"])
#                     params = strat.get("Best_Parameters", {})
#                     if isinstance(params, str) and params.strip():
#                         try:
#                             params = ast.literal_eval(params)
#                         except (ValueError, SyntaxError) as e:
#                             logger.warning(f"{ticker} - Failed to parse params: {e}")
#                             params = {}
#                     elif not isinstance(params, dict):
#                         params = {}

#                     min_data_points = strategy_class.get_min_data_points(params)
#                     min_bars_list.append(min_data_points)
#                 except Exception as e:
#                     logger.warning(f"{ticker} - Error calculating min bars: {e}")
#                     min_bars_list.append(30)

#             min_bars = max(min_bars_list) if min_bars_list else 30
#             if data_length < min_bars:
#                 logger.warning(
#                     f"{ticker} - Insufficient data ({data_length} < {min_bars})"
#                 )
#                 return

#             regime, adx_value, atr_value = calculate_regime(combined_data)
#             logger.info(
#                 f"{ticker} - Regime: {regime} (ADX: {adx_value:.2f}, ATR: {atr_value:.2f})"
#             )

#             signals = []
#             strategy_instances = []

#             for strat in strategies:
#                 strategy_name = strat["Strategy"]
#                 try:
#                     strategy_class = get_strategy(strategy_name)
#                 except KeyError:
#                     logger.warning(f"{ticker} - Strategy {strategy_name} not found")
#                     continue

#                 params = strat.get("Best_Parameters", {})
#                 if isinstance(params, str) and params.strip():
#                     try:
#                         params = ast.literal_eval(params)
#                     except (ValueError, SyntaxError) as e:
#                         logger.warning(f"{ticker} - Failed to parse params: {e}")
#                         params = {}
#                 elif not isinstance(params, dict):
#                     params = {}

#                 try:
#                     strategy_instance = strategy_class(combined_data, **params)
#                     signal = strategy_instance.run()

#                     if signal in ["BUY", "SELL"]:
#                         signals.append(signal)
#                         strategy_instances.append(
#                             {
#                                 "instance": strategy_instance,
#                                 "name": strategy_name,
#                                 "signal": signal,
#                                 "params": params,
#                             }
#                         )
#                         logger.info(f"{ticker} - {strategy_name} signal: {signal}")

#                 except Exception as e:
#                     logger.error(
#                         f"{ticker} - {strategy_name} execution failed: {str(e)}"
#                     )

#             if not signals:
#                 return

#             signal_counts = {"BUY": signals.count("BUY"), "SELL": signals.count("SELL")}
#             buy_votes = signal_counts["BUY"]
#             sell_votes = signal_counts["SELL"]
#             logger.info(
#                 f"{ticker} - Signal votes - BUY: {buy_votes}, SELL: {sell_votes}, Strategy: {strategy_name}"
#             )
#             # Execute signal with strategy instance for exit monitoring
#             if buy_votes >= MIN_VOTES and buy_votes > sell_votes:
#                 # Find the strongest BUY signal strategy instance
#                 buy_strategies = [s for s in strategy_instances if s["signal"] == "BUY"]
#                 primary_strategy = buy_strategies[
#                     0
#                 ]  # Use first BUY strategy for monitoring

#                 await execute_strategy_signal(
#                     ticker,
#                     security_id,
#                     "BUY",
#                     regime,
#                     adx_value,
#                     atr_value,
#                     combined_data,
#                     primary_strategy["name"],
#                     primary_strategy["instance"],
#                     **primary_strategy["params"],
#                 )

#             elif sell_votes >= MIN_VOTES and sell_votes > buy_votes:
#                 # Find the strongest SELL signal strategy instance
#                 sell_strategies = [
#                     s for s in strategy_instances if s["signal"] == "SELL"
#                 ]
#                 primary_strategy = sell_strategies[
#                     0
#                 ]  # Use first SELL strategy for monitoring

#                 await execute_strategy_signal(
#                     ticker,
#                     security_id,
#                     "SELL",
#                     regime,
#                     adx_value,
#                     atr_value,
#                     combined_data,
#                     primary_strategy["name"],
#                     primary_strategy["instance"],
#                     **primary_strategy["params"],
#                 )

#         except Exception as e:
#             logger.error(f"{ticker} - Enhanced processing failed: {str(e)}")


async def process_stock_with_exit_monitoring(
    ticker: str, security_id: int, strategies: List[Dict]
) -> None:
    """Enhanced stock processing with exit signal monitoring and cooldown period"""
    async with adaptive_semaphore:
        try:
            logger.info(f"Processing {ticker} (ID: {security_id})")

            # Get current time with timezone
            current_time = datetime.now(IST)

            # Check if we have existing position for this stock (bypass cooldown for exits)
            existing_position = position_manager.positions_by_security.get(security_id)

            if existing_position:
                # For existing positions, focus on exit signal monitoring (bypass cooldown)
                combined_data = await get_combined_data(security_id)
                if combined_data is None:
                    logger.warning(f"{ticker} - No data available for exit check")
                    return

                exit_signal = await position_manager.check_strategy_exit_signals(
                    security_id, combined_data
                )
                if exit_signal:
                    logger.info(
                        f"{ticker} - Exit signal detected: {exit_signal['reason']}"
                    )
                    await position_manager.execute_strategy_exit(
                        existing_position, exit_signal
                    )
                return

            # Check cooldown period for new entries
            last_trade_time = await position_manager.get_last_trade_time(ticker)

            if last_trade_time and current_time < last_trade_time + timedelta(
                minutes=position_manager.cooldown_minutes
            ):
                logger.info(
                    f"{ticker} - Skipping due to {position_manager.cooldown_minutes}-minute cooldown period"
                )
                return

            combined_data = await get_combined_data(security_id)
            if combined_data is None:
                logger.warning(f"{ticker} - No data available")
                return

            # Regular entry signal processing
            data_length = len(combined_data)
            min_bars_list = []

            for strat in strategies:
                try:
                    strategy_class = get_strategy(strat["Strategy"])
                    params = strat.get("Best_Parameters", {})
                    if isinstance(params, str) and params.strip():
                        try:
                            params = ast.literal_eval(params)
                        except (ValueError, SyntaxError) as e:
                            logger.warning(f"{ticker} - Failed to parse params: {e}")
                            params = {}
                    elif not isinstance(params, dict):
                        params = {}

                    min_data_points = strategy_class.get_min_data_points(params)
                    min_bars_list.append(min_data_points)
                except Exception as e:
                    logger.warning(f"{ticker} - Error calculating min bars: {e}")
                    min_bars_list.append(30)

            min_bars = max(min_bars_list) if min_bars_list else 30
            if data_length < min_bars:
                logger.warning(
                    f"{ticker} - Insufficient data ({data_length} < {min_bars})"
                )
                return

            regime, adx_value, atr_value = calculate_regime(combined_data)
            logger.info(
                f"{ticker} - Regime: {regime} (ADX: {adx_value:.2f}, ATR: {atr_value:.2f})"
            )

            signals = []
            strategy_instances = []

            for strat in strategies:
                strategy_name = strat["Strategy"]
                try:
                    strategy_class = get_strategy(strategy_name)
                except KeyError:
                    logger.warning(f"{ticker} - Strategy {strategy_name} not found")
                    continue

                params = strat.get("Best_Parameters", {})
                if isinstance(params, str) and params.strip():
                    try:
                        params = ast.literal_eval(params)
                    except (ValueError, SyntaxError) as e:
                        logger.warning(f"{ticker} - Failed to parse params: {e}")
                        params = {}
                elif not isinstance(params, dict):
                    params = {}

                try:
                    strategy_instance = strategy_class(combined_data, **params)
                    signal = strategy_instance.run()

                    if signal in ["BUY", "SELL"]:
                        signals.append(signal)
                        strategy_instances.append(
                            {
                                "instance": strategy_instance,
                                "name": strategy_name,
                                "signal": signal,
                                "params": params,
                            }
                        )
                        logger.info(f"{ticker} - {strategy_name} signal: {signal}")

                except Exception as e:
                    logger.error(
                        f"{ticker} - {strategy_name} execution failed: {str(e)}"
                    )

            if not signals:
                return

            signal_counts = {"BUY": signals.count("BUY"), "SELL": signals.count("SELL")}
            buy_votes = signal_counts["BUY"]
            sell_votes = signal_counts["SELL"]
            min_vote_diff = int(os.getenv("MIN_VOTE_DIFF", 1))

            logger.info(
                f"{ticker} - Signal votes - BUY: {buy_votes}, SELL: {sell_votes}"
            )

            # Execute signal with strategy instance for exit monitoring
            if buy_votes >= MIN_VOTES and (buy_votes - sell_votes) >= min_vote_diff:
                # Find the strongest BUY signal strategy instance
                buy_strategies = [s for s in strategy_instances if s["signal"] == "BUY"]
                primary_strategy = buy_strategies[0]

                # Execute and update trade time only if successful
                executed = await execute_strategy_signal(
                    ticker,
                    security_id,
                    "BUY",
                    regime,
                    adx_value,
                    atr_value,
                    combined_data,
                    primary_strategy["name"],
                    primary_strategy["instance"],
                    **primary_strategy["params"],
                )

                if executed:
                    await position_manager.update_last_trade_time(ticker, current_time)

            elif sell_votes >= MIN_VOTES and (sell_votes - buy_votes) >= min_vote_diff:
                # Find the strongest SELL signal strategy instance
                sell_strategies = [
                    s for s in strategy_instances if s["signal"] == "SELL"
                ]
                primary_strategy = sell_strategies[0]

                # Execute and update trade time only if successful
                executed = await execute_strategy_signal(
                    ticker,
                    security_id,
                    "SELL",
                    regime,
                    adx_value,
                    atr_value,
                    combined_data,
                    primary_strategy["name"],
                    primary_strategy["instance"],
                    **primary_strategy["params"],
                )

                if executed:
                    await position_manager.update_last_trade_time(ticker, current_time)

        except Exception as e:
            logger.error(f"{ticker} - Enhanced processing failed: {str(e)}")


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
        self.message_queue = asyncio.Queue()
        self.is_running = False

    async def start(self):
        if not self.is_running:
            self.is_running = True
            asyncio.create_task(self._process_messages())

    async def _process_messages(self):
        while self.is_running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=5.0)
                await self._send_message(message)
                await asyncio.sleep(1)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Telegram queue error: {e}")

    async def _send_message(self, message: str):
        try:
            if len(message) > 4000:
                message = message[:3900] + "...\n[TRUNCATED]"
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown",
            }
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.debug(f"Telegram sent: {message[:50]}...")
                    else:
                        text = await response.text()
                        logger.error(f"Telegram error {response.status}: {text}")
        except Exception as e:
            logger.error(f"Telegram send failed: {str(e)}")

    def send_alert(self, message: str):
        try:
            self.message_queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning("Telegram queue full, dropping message")


telegram_queue = TelegramQueue()


async def send_telegram_alert(message: str):
    await telegram_queue._send_message(message)


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


@retry(
    stop_max_attempt_number=3,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
)
async def place_super_order(
    security_id: int,
    transaction_type: str,
    current_price: float,
    stop_loss: float,
    take_profit: float,
    quantity: int = MAX_QUANTITY,
) -> Optional[Dict]:
    try:
        await rate_limiter.acquire()
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
            "trailingJump": 0.1,
        }
        session = await api_client.get_session()
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                order_id = data.get("orderId")
                if order_id:
                    trade_logger.info(
                        f"Order placed | {security_id} | {transaction_type} | "
                        f"Qty: {quantity} | Price: {current_price:.2f} | "
                        f"SL: {stop_loss:.2f} | TP: {take_profit:.2f}"
                    )
                    return data
                else:
                    trade_logger.error(f"Order placement failed: {data}")
            else:
                text = await response.text()
                trade_logger.error(f"HTTP error {response.status}: {text}")
        return None
    except Exception as e:
        trade_logger.error(f"Order placement exception: {str(e)}")
        return None


@retry(
    stop_max_attempt_number=3,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
)
async def place_market_order(
    security_id: int, transaction_type: str, quantity: int
) -> Optional[Dict]:
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
                        f"Market order placed | {security_id} | {transaction_type} | Qty: {quantity}"
                    )
                    return data
            else:
                text = await response.text()
                logger.error(f"Market order failed: {text}")
        return None
    except Exception as e:
        logger.error(f"Market order exception: {str(e)}")
        return None


async def fetch_realtime_quote(security_ids: List[int]) -> Dict[int, Optional[Dict]]:
    if not security_ids:
        return {}

    batch_size = 5  # Keep batch size small to avoid 429
    results = {}

    for i in range(0, len(security_ids), batch_size):
        batch = security_ids[i : i + batch_size]
        # Acquire from quote-specific rate limiter
        await rate_limiter.acquire()
        batch_results = await _fetch_quote_batch(batch)
        results.update(batch_results)

    return results


async def _fetch_quote_batch(security_ids: List[int]) -> Dict[int, Optional[Dict]]:
    import time

    time.sleep(2)
    try:
        payload = {"NSE_EQ": [int(sid) for sid in security_ids]}
        response = dhan.quote_data(payload)
        if response.get("status") == "success":
            result = {}
            for sec_id in security_ids:
                sec_id_str = str(sec_id)
                quote = response.get("data")["data"]["NSE_EQ"].get(sec_id_str)
                if quote:
                    try:
                        trade_time = datetime.strptime(
                            quote["last_trade_time"], "%d/%m/%Y %H:%M:%S"
                        ).replace(tzinfo=IST)
                        result[sec_id] = {
                            "price": float(quote["last_price"]),
                            "timestamp": trade_time,
                        }
                    except KeyError:
                        logger.warning(f"Missing keys in quote for {sec_id}: {quote}")
                        result[sec_id] = None
                else:
                    result[sec_id] = None
            return result
        elif response.get("status") == "failure":
            backoff = 5.0
            logger.warning(
                f"Quote API rate limit exceeded. Backing off for {backoff:.2f}s"
            )
            await asyncio.sleep(backoff)
            return await _fetch_quote_batch(security_ids)
        else:
            logger.error(
                f"Quote API status {response.get("status")} for {security_ids}: {response}"
            )
            return {sec_id: None for sec_id in security_ids}
    except Exception as e:
        import traceback

        traceback.print_exc()
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


async def get_combined_data(security_id: int) -> Optional[pd.DataFrame]:
    cache_key = f"{security_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    if cache_key in cache_manager.historical_cache:
        cache_manager.cache_hits["historical"] += 1
        return cache_manager.historical_cache[cache_key]

    try:
        combined_data = await get_combined_data_with_persistent_live(
            security_id=int(security_id),
            exchange_segment="NSE_EQ",
            auto_start_live_collection=True,
        )
        if combined_data is None:
            cache_manager.cache_misses["historical"] += 1
            return None

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

        cache_manager.historical_cache[cache_key] = combined_data
        cache_manager.cache_hits["historical"] += 1
        if len(cache_manager.historical_cache) > cache_manager.historical_cache.maxsize:
            logger.debug("Evicting old historical cache entries")
            cache_manager.log_cache_stats("historical")

        return combined_data
    except Exception as e:
        logger.error(f"Error in get_combined_data for {security_id}: {e}")
        cache_manager.cache_misses["historical"] += 1
        return None


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
        take_profit = current_price + (atr * cfg["tp_mult"])
    else:
        stop_loss = current_price + stop_loss_distance
        take_profit = current_price - (atr * cfg["tp_mult"])
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
        self.cache = {"realized": 0.0, "last_updated": None}
        self.cache_ttl = 300

    async def update_daily_pnl(self) -> float:
        now = datetime.now(IST)
        if (
            self.cache["last_updated"]
            and (now - self.cache["last_updated"]).total_seconds() < self.cache_ttl
        ):
            return self.cache["realized"]
        try:
            await rate_limiter.acquire()
            url = "https://api.dhan.co/v2/reports/pnl"
            params = {
                "type": "INTRADAY",
                "fromDate": date.today().strftime("%Y-%m-%d"),
                "toDate": date.today().strftime("%Y-%m-%d"),
            }
            session = await api_client.get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("data"):
                        total_pnl = sum(float(item["netPnl"]) for item in data["data"])
                        self.cache["realized"] = total_pnl
                        self.cache["last_updated"] = now
                        logger.info(f"Updated daily P&L: ₹{total_pnl:.2f}")
                        return total_pnl
            return 0.0
        except Exception as e:
            logger.error(f"Failed to update P&L: {str(e)}")
            return 0.0


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


async def process_stock(ticker: str, security_id: int, strategies: List[Dict]) -> None:
    async with adaptive_semaphore:
        try:
            logger.info(f"Processing {ticker} (ID: {security_id})")
            data_task = asyncio.create_task(get_combined_data(security_id))
            combined_data = await data_task
            if combined_data is None:
                logger.warning(f"{ticker} - No data available")
                return

            data_length = len(combined_data)
            min_bars_list = []
            for strat in strategies:
                try:
                    strategy_class = get_strategy(strat["Strategy"])
                    params = strat.get("Best_Parameters", {})
                    if isinstance(params, str) and params.strip():
                        try:
                            params = ast.literal_eval(params)
                        except (ValueError, SyntaxError) as e:
                            logger.warning(
                                f"{ticker} - Failed to parse params for {strat['Strategy']}: {e}"
                            )
                            params = {}
                    elif not isinstance(params, dict):
                        params = {}
                    min_data_points = strategy_class.get_min_data_points(params)
                    min_bars_list.append(min_data_points)
                except Exception as e:
                    logger.warning(
                        f"{ticker} - Error calculating min bars for {strat.get('Strategy', 'unknown')}: {e}"
                    )
                    min_bars_list.append(30)

            min_bars = max(min_bars_list) if min_bars_list else 30
            if data_length < min_bars:
                logger.warning(
                    f"{ticker} - Insufficient data ({data_length} < {min_bars})"
                )
                return

            regime, adx_value, atr_value = calculate_regime(combined_data)
            logger.info(
                f"{ticker} - Regime: {regime} (ADX: {adx_value:.2f}, ATR: {atr_value:.2f})"
            )

            logger.info(f"{ticker} - Processing {len(strategies)} strategies")
            signals = []

            for i, strat in enumerate(strategies):
                strategy_name = strat["Strategy"]
                try:
                    strategy_class = get_strategy(strategy_name)
                except KeyError:
                    logger.warning(f"{ticker} - Strategy {strategy_name} not found")
                    continue

                params = strat.get("Best_Parameters", {})
                if isinstance(params, str) and params.strip():
                    try:
                        params = ast.literal_eval(params)
                    except (ValueError, SyntaxError) as e:
                        logger.warning(f"{ticker} - Failed to parse params: {e}")
                        params = {}
                elif not isinstance(params, dict):
                    params = {}

                try:
                    min_bars_needed = strategy_class.get_min_data_points(params)
                    if data_length < min_bars_needed:
                        continue
                except Exception as e:
                    logger.error(
                        f"{ticker} - Error checking data requirements for {strategy_name}: {e}"
                    )
                    continue

                try:
                    strategy_instance = strategy_class(combined_data, **params)
                    signal = strategy_instance.run()
                    if signal in ["BUY", "SELL"]:
                        signals.append(signal)
                        logger.info(f"{ticker} - {strategy_name} signal: {signal}")
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    logger.error(
                        f"{ticker} - {strategy_name} execution failed: {str(e)}"
                    )

            if not signals:
                logger.warning(f"{ticker} - No signals generated")
                return

            signal_counts = {"BUY": signals.count("BUY"), "SELL": signals.count("SELL")}
            buy_votes = signal_counts["BUY"]
            sell_votes = signal_counts["SELL"]

            if buy_votes >= MIN_VOTES and buy_votes > sell_votes:
                logger.info(f"{ticker} - Executing BUY signal")
                await execute_strategy_signal(
                    ticker,
                    security_id,
                    "BUY",
                    regime,
                    adx_value,
                    atr_value,
                    combined_data,
                    strategy_name,
                    **params,
                )
            elif sell_votes >= MIN_VOTES and sell_votes > buy_votes:
                logger.info(f"{ticker} - Executing SELL signal")
                await execute_strategy_signal(
                    ticker,
                    security_id,
                    "SELL",
                    regime,
                    adx_value,
                    atr_value,
                    combined_data,
                    strategy_name,
                    **params,
                )
            else:
                logger.info(f"{ticker} - No final signal")
        except Exception as e:
            logger.error(f"{ticker} - Processing failed: {str(e)}")
            logger.error(traceback.format_exc())


@lru_cache(maxsize=10)
def get_market_times_cached(date_str: str) -> Tuple[datetime, datetime, datetime]:
    day_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    market_open_dt = IST.localize(datetime.combine(day_date, MARKET_OPEN_TIME))
    market_close_dt = IST.localize(datetime.combine(day_date, MARKET_CLOSE_TIME))
    trading_end_dt = IST.localize(datetime.combine(day_date, TRADING_END_TIME))
    return market_open_dt, market_close_dt, trading_end_dt


async def market_hours_check() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    holidays = await fetch_dynamic_holidays(now.year)
    if now.strftime("%Y-%m-%d") in holidays:
        logger.info("Market holiday - no trading")
        return False
    market_open_dt, market_close_dt, trading_end_dt = get_market_times_cached(
        now.strftime("%Y-%m-%d")
    )
    if now < market_open_dt:
        sleep_time = (market_open_dt - now).total_seconds()
        if sleep_time > 0:
            await asyncio.sleep(min(sleep_time, 300))
        return True
    elif now > market_close_dt:
        return False
    elif now > trading_end_dt:
        return False
    return True


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
                            await position_manager.close_position(order_id)
                            await send_telegram_alert(
                                f"*{pos['ticker']} SQUARED OFF* 🛑\nPrice: Market"
                            )
            else:
                await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Square off scheduler error: {e}")
            await asyncio.sleep(300)


async def send_heartbeat():
    while True:
        try:
            await asyncio.sleep(3600)
            active_tasks = len([t for t in asyncio.all_tasks() if not t.done()])
            cache_sizes = {
                "depth": len(cache_manager.depth_cache),
                "historical": len(cache_manager.historical_cache),
                "volatility": len(cache_manager.volatility_cache),
                "volume": len(cache_manager.volume_cache),
            }
            open_positions = len(position_manager.open_positions)
            message = (
                "💓 *SYSTEM HEARTBEAT*\n"
                f"Status: Operational\n"
                f"Active Tasks: {active_tasks}\n"
                f"Open Positions: {open_positions}\n"
                f"Cache Sizes: {cache_sizes}\n"
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            )
            await send_telegram_alert(message)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")


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


async def main_trading_loop():
    try:
        await telegram_queue.start()
        await send_telegram_alert("Bot started successfully")
        await initialize_live_data_from_config()

        try:
            strategies_df = pd.read_csv("selected_stocks_strategies.csv")
            nifty500 = pd.read_csv("ind_nifty500list.csv")
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
                        "security_id": ticker_to_security[ticker],
                        "strategies": stock_data.to_dict("records"),
                    }
                )
        logger.info(f"Prepared {len(stock_universe)} stocks for trading")

        background_tasks = [
            asyncio.create_task(schedule_square_off()),
            asyncio.create_task(send_heartbeat()),
            asyncio.create_task(cleanup_resources()),
            asyncio.create_task(position_manager.monitor_positions()),
            asyncio.create_task(position_manager.load_trade_times()),
            asyncio.create_task(position_manager.monitor_positions()),
        ]

        batch_size = 3
        while await market_hours_check():
            start_time = datetime.now(IST)
            for i in range(0, len(stock_universe), batch_size):
                batch = stock_universe[i : i + batch_size]
                batch_tasks = [
                    asyncio.create_task(
                        process_stock_with_exit_monitoring(
                            s["ticker"], s["security_id"], s["strategies"]
                        )
                    )
                    for s in batch
                ]
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=25.0,
                    )
                except asyncio.TimeoutError:
                    for task in batch_tasks:
                        if not task.done():
                            task.cancel()
                await asyncio.sleep(1)
            elapsed = (datetime.now(IST) - start_time).total_seconds()
            sleep_time = max(30 - elapsed, 5)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    except Exception as e:
        logger.critical(f"Main loop failure: {str(e)}")
        logger.error(traceback.format_exc())
        await send_telegram_alert(f"*CRITICAL ERROR*\nTrading stopped: {str(e)}")
    finally:
        await api_client.close()
        thread_pool.shutdown(wait=True)
        for task in background_tasks:
            task.cancel()


if __name__ == "__main__":
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        asyncio.run(main_trading_loop())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.critical(f"System failure: {str(e)}")
        logger.error(traceback.format_exc())
