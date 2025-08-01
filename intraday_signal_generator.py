import asyncio
import pandas as pd
import backtrader as bt
import os
import logging
from datetime import date, datetime, timedelta, time
import pytz
from retrying import retry
import ast
import pandas_ta as ta
import aiohttp
import math
import sys
import traceback
from functools import lru_cache
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from cachetools import TTLCache  # NEW: For TTL-based caching
from comprehensive_backtesting.data import init_dhan_client

# Import registry functions
from comprehensive_backtesting.registry import get_strategy


# NEW: CacheManager class for centralized cache management
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


# NEW: PositionManager class for active position management
class PositionManager:
    def __init__(self):
        self.open_positions = (
            {}
        )  # {order_id: {security_id, ticker, entry_price, quantity, sl, tp, direction}}
        self.max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", 10))
        self.position_lock = asyncio.Lock()

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
    ):
        async with self.position_lock:
            if len(self.open_positions) >= self.max_open_positions:
                logger.warning(
                    f"Max open positions ({self.max_open_positions}) reached, cannot add {ticker}"
                )
                return False
            self.open_positions[order_id] = {
                "security_id": security_id,
                "ticker": ticker,
                "entry_price": entry_price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "direction": direction,
                "last_updated": datetime.now(IST),
            }
            logger.info(
                f"Added position {order_id} for {ticker}: {direction} @ ₹{entry_price:.2f}"
            )
            return True

    async def update_position(
        self, order_id: str, stop_loss: float = None, take_profit: float = None
    ):
        async with self.position_lock:
            if order_id not in self.open_positions:
                return False
            pos = self.open_positions[order_id]
            if stop_loss:
                pos["stop_loss"] = stop_loss
            if take_profit:
                pos["take_profit"] = take_profit
            pos["last_updated"] = datetime.now(IST)
            logger.info(
                f"Updated position {order_id} for {pos['ticker']}: SL=₹{pos['stop_loss']:.2f}, TP=₹{pos['take_profit']:.2f}"
            )
            return True

    async def close_position(self, order_id: str):
        async with self.position_lock:
            if order_id in self.open_positions:
                pos = self.open_positions.pop(order_id)
                logger.info(f"Closed position {order_id} for {pos['ticker']}")
                return True
            return False

    async def monitor_positions(self):
        """Background task to monitor and update positions"""
        while True:
            try:
                async with self.position_lock:
                    now = datetime.now(IST)
                    for order_id, pos in list(self.open_positions.items()):
                        # Fetch current price
                        quote = await fetch_realtime_quote(pos["security_id"])
                        if not quote:
                            continue
                        current_price = quote["price"]

                        # Check stop-loss and take-profit
                        if pos["direction"] == "BUY":
                            if current_price <= pos["stop_loss"]:
                                await place_market_order(
                                    pos["security_id"], "SELL", pos["quantity"]
                                )
                                await self.close_position(order_id)
                                await send_telegram_alert(
                                    f"*{pos['ticker']} STOP-LOSS HIT* 🛑\nPrice: ₹{current_price:.2f}"
                                )
                            elif current_price >= pos["take_profit"]:
                                await place_market_order(
                                    pos["security_id"], "SELL", pos["quantity"]
                                )
                                await self.close_position(order_id)
                                await send_telegram_alert(
                                    f"*{pos['ticker']} TAKE-PROFIT HIT* 🎯\nPrice: ₹{current_price:.2f}"
                                )
                        else:  # SELL
                            if current_price >= pos["stop_loss"]:
                                await place_market_order(
                                    pos["security_id"], "BUY", pos["quantity"]
                                )
                                await self.close_position(order_id)
                                await send_telegram_alert(
                                    f"*{pos['ticker']} STOP-LOSS HIT* 🛑\nPrice: ₹{current_price:.2f}"
                                )
                            elif current_price <= pos["take_profit"]:
                                await place_market_order(
                                    pos["security_id"], "BUY", pos["quantity"]
                                )
                                await self.close_position(order_id)
                                await send_telegram_alert(
                                    f"*{pos['ticker']} TAKE-PROFIT HIT* 🎯\nPrice: ₹{current_price:.2f}"
                                )

                        # Update trailing stop if applicable
                        if (
                            now - pos["last_updated"]
                        ).total_seconds() > 300:  # Update every 5 minutes
                            hist_data = await get_combined_data(pos["security_id"])
                            if hist_data is not None:
                                _, _, atr = calculate_regime(hist_data)
                                if pos["direction"] == "BUY":
                                    new_sl = max(
                                        pos["stop_loss"], current_price - atr * 1.5
                                    )
                                    await self.update_position(
                                        order_id, stop_loss=new_sl
                                    )
                                else:
                                    new_sl = min(
                                        pos["stop_loss"], current_price + atr * 1.5
                                    )
                                    await self.update_position(
                                        order_id, stop_loss=new_sl
                                    )

                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(300)  # Retry after 5 minutes


# NEW: BacktraderRunner for async execution
class BacktraderRunner:
    def __init__(self):
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="bt_runner"
        )

    async def run_strategy(
        self, strategy_class, data: pd.DataFrame, params: Dict
    ) -> Optional[str]:
        try:

            def run_cerebro():
                cerebro = bt.Cerebro(stdstats=False, runonce=True, optdatas=True)
                data_feed = bt.feeds.PandasData(
                    dataname=data,
                    datetime="datetime",
                    fromdate=data.iloc[0]["datetime"],
                    todate=data.iloc[-1]["datetime"],
                )
                cerebro.adddata(data_feed)
                cerebro.addstrategy(
                    OptimizedSignalGeneratorWrapper,
                    strategy_class=strategy_class,
                    **params,
                )
                results = cerebro.run()
                return results[0].signal if results else None

            loop = asyncio.get_event_loop()
            signal = await loop.run_in_executor(self.executor, run_cerebro)
            return signal
        except Exception as e:
            logger.error(f"Backtrader run failed: {e}")
            return None

    def shutdown(self):
        self.executor.shutdown(wait=True)


class DateTimeFormatter(logging.Formatter):
    """Custom formatter with IST datetime"""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=pytz.timezone("Asia/Kolkata"))
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " IST"


# Create the custom formatter
formatter = DateTimeFormatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Clear any existing handlers to avoid duplication
logging.getLogger().handlers.clear()

# Setup console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Setup file handler
file_handler = logging.FileHandler("trading_system.log", mode="a")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Configure main logger (inherits from root)
logger = logging.getLogger("quant_trader")
logger.setLevel(logging.INFO)

# Configure trade logger with its own file
trade_logger = logging.getLogger("trade_execution")
trade_logger.setLevel(logging.INFO)
trade_file_handler = logging.FileHandler("trades.log", mode="a")
trade_file_handler.setFormatter(formatter)
trade_logger.addHandler(trade_file_handler)
trade_logger.propagate = False

# Test the logging
logger.info("Logging system initialized with IST timestamps")
trade_logger.info("Trade logging system initialized")

# Environment configuration - Cache parsed values
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "1000000003")

# Validate environment variables
if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_ACCESS_TOKEN]):
    logger.critical("Missing required environment variables")
    raise EnvironmentError("Required environment variables not set")

# Market configuration - Pre-parse time objects
MARKET_OPEN_STR = os.getenv("MARKET_OPEN", "09:15:00")
MARKET_CLOSE_STR = os.getenv("MARKET_CLOSE", "15:30:00")
TRADING_END_STR = os.getenv("TRADING_END", "15:20:00")
FORCE_CLOSE_STR = os.getenv("FORCE_CLOSE", "15:15:00")
SQUARE_OFF_TIME_STR = os.getenv("SQUARE_OFF_TIME", "15:16:00")

# Pre-parse time objects for performance
MARKET_OPEN_TIME = time.fromisoformat(MARKET_OPEN_STR)
MARKET_CLOSE_TIME = time.fromisoformat(MARKET_CLOSE_STR)
TRADING_END_TIME = time.fromisoformat(TRADING_END_STR)
FORCE_CLOSE_TIME = time.fromisoformat(FORCE_CLOSE_STR)
SQUARE_OFF_TIME_TIME = time.fromisoformat(SQUARE_OFF_TIME_STR)

# Updated configuration constants - Optimized for Large Cap stocks
MIN_VOTES = int(os.getenv("MIN_VOTES", 2))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 6))
MORNING_WINDOW_SIZE = int(os.getenv("MORNING_WINDOW_SIZE", 3))
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", 100000))
MAX_QUANTITY = int(os.getenv("MAX_QUANTITY", 2))
MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS", 0.02))

# Updated thresholds for Large Cap stocks
VOLATILITY_THRESHOLD = float(
    os.getenv("VOLATILITY_THRESHOLD", 0.012)
)  # Reduced from 0.03 to 1.2%
LIQUIDITY_THRESHOLD = int(
    os.getenv("LIQUIDITY_THRESHOLD", 75000)
)  # Reduced from 100000
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", 180))
BID_ASK_THRESHOLD = int(os.getenv("BID_ASK_THRESHOLD", 500))  # Reduced from 1000

# New parameters for enhanced filtering
RELATIVE_VOLUME_THRESHOLD = float(
    os.getenv("RELATIVE_VOLUME_THRESHOLD", 1.2)
)  # 1.2x average volume
MIN_PRICE_THRESHOLD = float(
    os.getenv("MIN_PRICE_THRESHOLD", 50.0)
)  # Minimum price for large caps
MAX_PRICE_THRESHOLD = float(
    os.getenv("MAX_PRICE_THRESHOLD", 5000.0)
)  # Maximum price filter

# Initialize Dhan client and thread pool
dhan = init_dhan_client()
dhan_lock = asyncio.Lock()
thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dhan_worker")

# Preload and cache symbol map
try:
    nifty500_df = pd.read_csv("ind_nifty500list.csv")
    SYMBOL_MAP = nifty500_df.set_index("security_id")["ticker"].to_dict()
    TICKER_TO_ID_MAP = nifty500_df.set_index("ticker")["security_id"].to_dict()
except Exception as e:
    logger.error(f"Failed to load symbol map: {e}")
    SYMBOL_MAP = {}
    TICKER_TO_ID_MAP = {}

# Initialize caches with CacheManager
cache_manager = CacheManager(max_size=1000, ttl=3600)  # 1 hour TTL
HOLIDAY_CACHE = {}
CANDLE_BUILDERS = {}
daily_pnl_tracker = {"realized": 0.0, "unrealized": 0.0, "last_updated": None}

# Pre-compile timezone object
IST = pytz.timezone("Asia/Kolkata")

# NEW: Initialize BacktraderRunner and PositionManager
backtrader_runner = BacktraderRunner()
position_manager = PositionManager()


class EnhancedCandle:
    """Memory-optimized candle structure using __slots__"""

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
    """Optimized rate limiter with token bucket algorithm"""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.last_refill = datetime.now()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = datetime.now()
            elapsed = (now - self.last_refill).total_seconds()
            if elapsed > 60:
                self.tokens = self.rate_limit
                self.last_refill = now
            elif elapsed > 0:
                tokens_to_add = int(elapsed * (self.rate_limit / 60))
                self.tokens = min(self.rate_limit, self.tokens + tokens_to_add)
                if tokens_to_add > 0:
                    self.last_refill = now

            while self.tokens <= 0:
                await asyncio.sleep(0.1)
                elapsed = (datetime.now() - self.last_refill).total_seconds()
                if elapsed > 60:
                    self.tokens = self.rate_limit
                    self.last_refill = datetime.now()

            self.tokens -= 1


rate_limiter = OptimizedRateLimiter(API_RATE_LIMIT)


def is_high_volume_period() -> bool:
    """Check if current time is high-volume trading period"""
    now = datetime.now(IST).time()
    high_volume_periods = [
        (time(9, 15), time(10, 30)),
        (time(14, 30), time(15, 30)),
    ]
    return any(start <= now <= end for start, end in high_volume_periods)


def get_volatility_threshold_for_time() -> float:
    """Dynamic volatility threshold based on time of day"""
    base_threshold = VOLATILITY_THRESHOLD
    return base_threshold * 0.8 if is_high_volume_period() else base_threshold * 1.1


def adjust_thresholds_for_regime(regime: str, base_threshold: float) -> float:
    """Adjust filtering thresholds based on market regime"""
    adjustments = {
        "trending": 0.85,
        "range_bound": 1.15,
        "transitional": 1.0,
        "unknown": 1.0,
    }
    return base_threshold * adjustments.get(regime, 1.0)


def calculate_relative_volume(current_volume: float, avg_volume: float) -> float:
    """Calculate relative volume ratio"""
    return current_volume / avg_volume if avg_volume > 0 else 0.0


class TelegramQueue:
    """Queue-based Telegram message sender to batch messages"""

    def __init__(self):
        self.message_queue = None
        self.is_running = False
        self._initialized = False

    async def _initialize(self):
        if not self._initialized:
            self.message_queue = asyncio.Queue()
            self._initialized = True

    async def start(self):
        await self._initialize()
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
            if self.message_queue is not None:
                self.message_queue.put_nowait(message)
            else:
                logger.warning("Telegram queue not initialized, dropping message")
        except asyncio.QueueFull:
            logger.warning("Telegram queue full, dropping message")


telegram_queue = TelegramQueue()


async def send_telegram_alert(message: str):
    telegram_queue.send_alert(message)


class APIClient:
    """Optimized API client with connection pooling"""

    def __init__(self):
        self.connector = None
        self.timeout = None
        self.session = None
        self._initialized = False

    async def _initialize(self):
        if not self._initialized:
            self.connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
            )
            self.timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self._initialized = True

    async def get_session(self):
        await self._initialize()
        if self.session is None or self.session.closed:
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
                    logger.error(f"Order placement failed: {data}")
            else:
                text = await response.text()
                logger.error(f"HTTP error {response.status}: {text}")
        return None
    except Exception as e:
        logger.error(f"Order placement exception: {str(e)}")
        return None


@retry(
    stop_max_attempt_number=3,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
)
async def modify_super_order(order_id: str, leg_name: str, **params) -> Optional[Dict]:
    try:
        await rate_limiter.acquire()
        url = f"https://api.dhan.co/v2/super/orders/{order_id}"
        payload = {
            "dhanClientId": DHAN_CLIENT_ID,
            "orderId": order_id,
            "legName": leg_name,
        }
        payload.update(params)
        session = await api_client.get_session()
        async with session.put(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                trade_logger.info(
                    f"Order modified | {order_id} | {leg_name} | Params: {params}"
                )
                return data
            else:
                text = await response.text()
                logger.error(f"Modify order failed: {text}")
        return None
    except Exception as e:
        logger.error(f"Order modify exception: {str(e)}")
        return None


@retry(
    stop_max_attempt_number=3,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
)
async def cancel_super_order(order_id: str, leg_name: str) -> Optional[Dict]:
    try:
        await rate_limiter.acquire()
        url = f"https://api.dhan.co/v2/super/orders/{order_id}/{leg_name}"
        session = await api_client.get_session()
        async with session.delete(url) as response:
            if response.status == 202:
                data = await response.json()
                trade_logger.info(f"Order canceled | {order_id} | {leg_name}")
                return data
            else:
                text = await response.text()
                logger.error(f"Cancel order failed: {text}")
        return None
    except Exception as e:
        logger.error(f"Order cancel exception: {str(e)}")
        return None


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


async def verify_order_execution(order_id: str) -> bool:
    try:
        await rate_limiter.acquire()
        url = f"https://api.dhan.co/v2/orders/{order_id}"
        session = await api_client.get_session()
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("status") == "EXECUTED"
        return False
    except Exception:
        return False


async def fetch_and_store_market_depth(security_id: int) -> bool:
    try:
        await rate_limiter.acquire()
        url = "https://api.dhan.co/v2/marketDepth"
        params = {"securityId": str(security_id), "exchangeSegment": "NSE_EQ"}
        session = await api_client.get_session()
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if data.get("data") and "depth" in data["data"]:
                    depth = data["data"]["depth"]
                    timestamp = datetime.now()
                    buy_levels = depth.get("buy", [])
                    sell_levels = depth.get("sell", [])
                    bid_qty = buy_levels[0]["quantity"] if buy_levels else 0
                    ask_qty = sell_levels[0]["quantity"] if sell_levels else 0
                    bid_depth = sum(level["quantity"] for level in buy_levels[:5])
                    ask_depth = sum(level["quantity"] for level in sell_levels[:5])
                    cache_data = {
                        "timestamp": timestamp,
                        "bid_qty": bid_qty,
                        "ask_qty": ask_qty,
                        "bid_depth": bid_depth,
                        "ask_depth": ask_depth,
                        "ltp": float(depth["ltp"]),
                        "volume": int(depth["volume"]),
                    }
                    cache_manager.depth_cache[security_id] = deque(
                        [cache_data], maxlen=100
                    )
                    cache_manager.cache_hits["depth"] += 1
                    return True
                cache_manager.cache_misses["depth"] += 1
        return False
    except Exception as e:
        logger.error(f"Market depth storage error: {str(e)}")
        cache_manager.cache_misses["depth"] += 1
        return False


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

    hist_data = await fetch_historical_data(security_id, days_back=5)
    if hist_data is None:
        cache_manager.cache_misses["historical"] += 1
        return None

    cache_manager.historical_cache[cache_key] = hist_data
    cache_manager.cache_hits["historical"] += 1
    if len(cache_manager.historical_cache) > cache_manager.historical_cache.maxsize:
        logger.debug("Evicting old historical cache entries")
        cache_manager.log_cache_stats("historical")

    enhanced_candles = build_enhanced_candles(security_id)
    if not enhanced_candles:
        return hist_data

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

    if enhanced_data:
        enhanced_df = pd.DataFrame(enhanced_data)
        combined = pd.concat([hist_data, enhanced_df], ignore_index=True)
        combined = (
            combined.sort_values("datetime")
            .drop_duplicates("datetime", keep="last")
            .reset_index(drop=True)
        )
        return combined
    return hist_data


@lru_cache(maxsize=1000)
def get_symbol_from_id(security_id: int) -> str:
    return SYMBOL_MAP.get(security_id, f"UNKNOWN_{security_id}")


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


async def fetch_historical_data(
    security_id: int, days_back: int = 5, interval: int = 5
) -> Optional[pd.DataFrame]:
    try:
        now = datetime.now(IST)
        today = now.date()
        holidays = await fetch_dynamic_holidays(today.year)
        holiday_set = set(holidays)
        valid_days = []
        current_day = today
        days_checked = 0
        max_days_to_check = min(days_back * 2, 60)

        while len(valid_days) < days_back and days_checked < max_days_to_check:
            if (
                current_day.weekday() < 5
                and current_day.strftime("%Y-%m-%d") not in holiday_set
            ):
                valid_days.append(current_day)
            current_day -= timedelta(days=1)
            days_checked += 1

        if not valid_days:
            logger.error(f"No valid trading days found for {security_id}")
            return None

        logger.debug(
            f"Fetching data for {security_id} across {len(valid_days)} trading days"
        )

        async def fetch_day_data(day: date) -> Optional[pd.DataFrame]:
            try:
                from_date = IST.localize(datetime.combine(day, MARKET_OPEN_TIME))
                to_date = (
                    IST.localize(datetime.combine(day, MARKET_CLOSE_TIME))
                    if day != today
                    else now
                )
                to_date = min(to_date, now)
                async with dhan_lock:
                    data = await asyncio.get_event_loop().run_in_executor(
                        thread_pool,
                        lambda: dhan.intraday_minute_data(
                            security_id=security_id,
                            exchange_segment="NSE_EQ",
                            instrument_type="EQUITY",
                            interval=interval,
                            from_date=from_date.strftime("%Y-%m-%d %H:%M:%S"),
                            to_date=to_date.strftime("%Y-%m-%d %H:%M:%S"),
                        ),
                    )
                if data.get("status") == "success" and data.get("data"):
                    df = pd.DataFrame(data["data"])
                    if not df.empty:
                        df["datetime"] = pd.to_datetime(
                            df["timestamp"], unit="s", utc=True
                        ).dt.tz_convert(IST)
                        df = df[
                            ["datetime", "open", "high", "low", "close", "volume"]
                        ].copy()
                        logger.debug(
                            f"Fetched {len(df)} bars for {security_id} on {day}"
                        )
                        return df
                return None
            except Exception as e:
                logger.error(
                    f"Error fetching data for {security_id} on {day}: {str(e)}"
                )
                return None

        semaphore = asyncio.Semaphore(2)

        async def fetch_with_semaphore(day: date) -> Optional[pd.DataFrame]:
            async with semaphore:
                await asyncio.sleep(0.1)
                return await fetch_day_data(day)

        tasks = [fetch_with_semaphore(day) for day in valid_days[:10]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_dataframes = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Task failed for day {valid_days[i]}: {result}")
            elif result is not None and not result.empty:
                valid_dataframes.append(result)

        if not valid_dataframes:
            logger.error(f"No data fetched for {security_id}")
            return None

        full_df = pd.concat(valid_dataframes, ignore_index=True)
        full_df = (
            full_df.sort_values("datetime")
            .drop_duplicates("datetime", keep="last")
            .reset_index(drop=True)
        )
        logger.info(f"Successfully fetched {len(full_df)} total bars for {security_id}")
        return full_df
    except Exception as e:
        logger.error(f"Historical data error for {security_id}: {str(e)}")
        return None


async def fetch_realtime_quote(security_id: int) -> Optional[Dict]:
    try:
        await rate_limiter.acquire()
        url = "https://api.dhan.co/v2/quotes"
        params = {"securityId": str(security_id), "exchangeSegment": "NSE_EQ"}
        session = await api_client.get_session()
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if data.get("data"):
                    quote = data["data"][0]
                    return {
                        "price": float(quote["last_price"]),
                        "timestamp": datetime.strptime(
                            quote["last_trade_time"], "%d/%m/%Y %H:%M:%S"
                        ).replace(tzinfo=IST),
                    }
        return None
    except Exception as e:
        logger.error(f"Realtime quote error: {str(e)}")
        return None


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


class OptimizedSignalGeneratorWrapper(bt.Strategy):
    __slots__ = ("signal", "strategy")

    def __init__(self, strategy_class, **params):
        self.signal = None
        self.strategy = strategy_class(self.datas[0], **params)

    def next(self):
        self.signal = None
        self.strategy.next()

    def buy(self, *args, **kwargs):
        self.signal = "BUY"

    def sell(self, *args, **kwargs):
        self.signal = "SELL"


class PnLTracker:
    def __init__(self):
        self.cache = {"realized": 0.0, "last_updated": None}
        self.cache_ttl = 300

    async def update_daily_pnl(self) -> float:
        now = datetime.now()
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


async def enhanced_stock_filtering(
    ticker: str, security_id: int, combined_data: pd.DataFrame, regime: str, atr: float
) -> Tuple[bool, str]:
    try:
        if len(combined_data) < 30:
            return False, "Insufficient data"

        latest_price = combined_data["close"].iloc[-1]
        if latest_price < MIN_PRICE_THRESHOLD or latest_price > MAX_PRICE_THRESHOLD:
            return False, f"Price out of range: ₹{latest_price:.2f}"

        # MODIFIED: Use ATR-based volatility threshold for intraday
        volatility = await calculate_stock_volatility(security_id)
        volatility_threshold = atr * 0.5  # Dynamic threshold based on ATR
        volatility_threshold = adjust_thresholds_for_regime(
            regime, volatility_threshold
        )

        if volatility < volatility_threshold:
            return (
                False,
                f"Low volatility: {volatility:.3f} < {volatility_threshold:.3f}",
            )

        # MODIFIED: Use shorter-term volume average
        avg_volume = await calculate_average_volume(security_id)
        recent_volume = combined_data["volume"].tail(5).mean()
        relative_vol = calculate_relative_volume(recent_volume, avg_volume)

        min_relative_vol = RELATIVE_VOLUME_THRESHOLD * (
            0.8 if is_high_volume_period() else 1.0
        )

        if relative_vol < min_relative_vol:
            return (
                False,
                f"Low relative volume: {relative_vol:.2f}x (threshold: {min_relative_vol:.2f}x)",
            )

        if "bid_qty" in combined_data.columns and "ask_qty" in combined_data.columns:
            latest = combined_data.iloc[-1]
            bid_qty = latest.get("bid_qty", 0)
            ask_qty = latest.get("ask_qty", 0)
            min_qty = BID_ASK_THRESHOLD * (0.7 if avg_volume < 100000 else 1.0)

            if bid_qty < min_qty or ask_qty < min_qty:
                return (
                    False,
                    f"Poor liquidity: Bid={bid_qty}, Ask={ask_qty} (min: {min_qty})",
                )

        if len(combined_data) >= 10:
            recent_vol_avg = combined_data["volume"].tail(5).mean()
            earlier_vol_avg = combined_data["volume"].tail(10).head(5).mean()
            if earlier_vol_avg > 0:
                volume_trend = recent_vol_avg / earlier_vol_avg
                if volume_trend < 0.8:
                    return False, f"Declining volume trend: {volume_trend:.2f}"

        return True, "Passed all filters"
    except Exception as e:
        logger.error(f"Enhanced filtering error for {ticker}: {str(e)}")
        return False, f"Filtering error: {str(e)}"


pnl_tracker = PnLTracker()


async def execute_strategy_signal(
    ticker: str,
    security_id: int,
    signal: str,
    regime: str,
    adx_value: float,
    atr_value: float,
    hist_data: pd.DataFrame,
):
    try:
        current_pnl = await pnl_tracker.update_daily_pnl()
        if current_pnl <= -MAX_DAILY_LOSS_PERCENT * ACCOUNT_SIZE:
            message = (
                f"🛑 TRADING HALTED: Daily loss limit reached\n"
                f"Current P&L: ₹{current_pnl:.2f}\n"
                f"Limit: ₹{-MAX_DAILY_LOSS_PERCENT * ACCOUNT_SIZE:.2f}"
            )
            await send_telegram_alert(message)
            logger.critical("Daily loss limit reached - trading halted")
            return

        if signal not in ["BUY", "SELL"]:
            logger.warning(f"Invalid signal for {ticker}: {signal}")
            return

        quote = await fetch_realtime_quote(security_id)
        if not quote:
            logger.warning(f"Price unavailable for {ticker}")
            return

        current_price = quote["price"]
        vwap = await calculate_vwap(hist_data)
        entry_price = (
            min(current_price, vwap * 0.998)
            if signal == "BUY"
            else max(current_price, vwap * 1.002)
        )

        risk_params = calculate_risk_params(regime, atr_value, entry_price, signal)
        now = datetime.now(IST)
        volatility = await calculate_stock_volatility(security_id)

        message = (
            f"*{ticker} Signal* 📈\n"
            f"Direction: {signal}\n"
            f"Entry: ₹{entry_price:.2f} | VWAP: ₹{vwap:.2f}\n"
            f"Current: ₹{current_price:.2f}\n"
            f"Regime: {regime} (ADX: {adx_value:.2f})\n"
            f"Volatility: {volatility:.3f}\n"
            f"Size: {risk_params['position_size']} | SL: ₹{risk_params['stop_loss']:.2f}\n"
            f"TP: ₹{risk_params['take_profit']:.2f}\n"
            f"Risk: ₹{abs(entry_price - risk_params['stop_loss']) * risk_params['position_size']:.2f}\n"
            f"Time: {now.strftime('%H:%M:%S')}"
        )

        order_response = await place_super_order(
            security_id,
            signal,
            entry_price,
            risk_params["stop_loss"],
            risk_params["take_profit"],
            risk_params["position_size"],
        )

        if order_response and order_response.get("orderId"):
            await position_manager.add_position(
                order_response["orderId"],
                security_id,
                ticker,
                entry_price,
                risk_params["position_size"],
                risk_params["stop_loss"],
                risk_params["take_profit"],
                signal,
            )
            await send_telegram_alert(message)
        else:
            await send_telegram_alert(
                f"*{ticker} Order Failed* ❌\nSignal: {signal} at ₹{entry_price:.2f}"
            )
    except Exception as e:
        logger.error(f"Signal execution failed for {ticker}: {str(e)}")


async def calculate_stock_volatility(security_id: int) -> float:
    cache_key = f"vol_{security_id}_{date.today()}"
    if cache_key in cache_manager.volatility_cache:
        cache_manager.cache_hits["volatility"] += 1
        return cache_manager.volatility_cache[cache_key]

    try:
        hist_data = await fetch_historical_data(
            security_id, days_back=1, interval=5
        )  # MODIFIED: 1 day for intraday
        if hist_data is None or len(hist_data) < 20:
            cache_manager.cache_misses["volatility"] += 1
            return 0

        returns = (
            hist_data["close"].tail(20).pct_change().dropna()
        )  # MODIFIED: Last 20 bars (~100 minutes)
        volatility = returns.std()  # MODIFIED: No annualization for intraday
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
        hist_data = await fetch_historical_data(
            security_id, days_back=1, interval=5
        )  # MODIFIED: 1 day for intraday
        if hist_data is None or len(hist_data) < 20:
            cache_manager.cache_misses["volume"] += 1
            return 0
        avg_volume = hist_data["volume"].tail(20).mean()  # MODIFIED: Last 20 bars
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
        self.current_limit = initial_limit
        self.last_adjustment = datetime.now()

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
            logger.info(f"=== PROCESSING {ticker} (ID: {security_id}) ===")
            logger.info(f"{ticker} - Step 1: Fetching market depth and data")
            depth_task = asyncio.create_task(fetch_and_store_market_depth(security_id))
            data_task = asyncio.create_task(get_combined_data(security_id))
            await depth_task
            combined_data = await data_task
            if combined_data is None:
                logger.warning(f"{ticker} - FAILED: No combined data available")
                return

            data_length = len(combined_data)
            logger.info(f"{ticker} - Data length: {data_length} bars")
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
                    logger.debug(
                        f"{ticker} - {strat['Strategy']} needs {min_data_points} bars"
                    )
                except Exception as e:
                    logger.warning(
                        f"{ticker} - Error calculating min bars for {strat.get('Strategy', 'unknown')}: {e}"
                    )
                    min_bars_list.append(30)

            min_bars = max(min_bars_list) if min_bars_list else 30
            logger.info(f"{ticker} - Maximum min_bars required: {min_bars}")

            if data_length < min_bars:
                logger.warning(
                    f"{ticker} - FAILED: Insufficient data ({data_length} < {min_bars})"
                )
                return

            regime, adx_value, atr_value = calculate_regime(combined_data)
            logger.info(
                f"{ticker} - Regime: {regime} (ADX: {adx_value:.2f}, ATR: {atr_value:.2f})"
            )

            logger.info(f"{ticker} - Step 3: Enhanced large cap filtering")
            # filter_passed, filter_reason = await enhanced_stock_filtering(
            #     ticker, security_id, combined_data, regime, atr_value  # MODIFIED: Pass ATR
            # )

            # if not filter_passed:
            #     logger.warning(f"{ticker} - FILTERED OUT: {filter_reason}")
            #     return

            # logger.info(f"{ticker} - PASSED FILTERING: {filter_reason}")

            logger.info(f"{ticker} - Step 4: Processing {len(strategies)} strategies")
            signals = []

            for i, strat in enumerate(strategies):
                strategy_name = strat["Strategy"]
                logger.info(f"{ticker} - Strategy {i+1}: {strategy_name}")
                try:
                    strategy_class = get_strategy(strategy_name)
                except KeyError:
                    logger.warning(f"{ticker} - Strategy {strategy_name} not found")
                    continue

                params = strat.get("Best_Parameters", {})
                if isinstance(params, str) and params.strip():
                    try:
                        params = ast.literal_eval(params)
                        logger.info(f"{ticker} - {strategy_name} params: {params}")
                    except (ValueError, SyntaxError) as e:
                        logger.warning(f"{ticker} - Failed to parse params: {e}")
                        params = {}
                elif not isinstance(params, dict):
                    logger.warning(
                        f"{ticker} - Invalid params type: {type(params)}, using default"
                    )
                    params = {}

                try:
                    min_bars_needed = strategy_class.get_min_data_points(params)
                    logger.info(
                        f"{ticker} - {strategy_name} needs {min_bars_needed} bars, have {data_length}"
                    )
                    if data_length < min_bars_needed:
                        logger.warning(
                            f"{ticker} - {strategy_name} SKIPPED: Not enough data"
                        )
                        continue
                except Exception as e:
                    logger.error(
                        f"{ticker} - Error checking data requirements for {strategy_name}: {e}"
                    )
                    continue

                # MODIFIED: Use BacktraderRunner for async execution
                signal = await backtrader_runner.run_strategy(
                    strategy_class, combined_data, params
                )
                if signal:
                    signals.append(signal)
                    logger.info(
                        f"{ticker} - {strategy_name} GENERATED SIGNAL: {signal}"
                    )
                else:
                    logger.info(f"{ticker} - {strategy_name} generated no signal")

            logger.info(
                f"{ticker} - Step 5: Signal voting from {len(signals)} signals: {signals}"
            )

            if not signals:
                logger.warning(f"{ticker} - FINAL: No signals generated")
                return

            signal_counts = {"BUY": signals.count("BUY"), "SELL": signals.count("SELL")}
            buy_votes = signal_counts["BUY"]
            sell_votes = signal_counts["SELL"]

            logger.info(
                f"{ticker} - Vote count - BUY: {buy_votes}, SELL: {sell_votes} (MIN_VOTES: {MIN_VOTES})"
            )

            if buy_votes >= MIN_VOTES and buy_votes > sell_votes:
                logger.info(f"{ticker} - EXECUTING BUY SIGNAL")
                await execute_strategy_signal(
                    ticker,
                    security_id,
                    "BUY",
                    regime,
                    adx_value,
                    atr_value,
                    combined_data,
                )
            elif sell_votes >= MIN_VOTES and sell_votes > buy_votes:
                logger.info(f"{ticker} - EXECUTING SELL SIGNAL")
                await execute_strategy_signal(
                    ticker,
                    security_id,
                    "SELL",
                    regime,
                    adx_value,
                    atr_value,
                    combined_data,
                )
            else:
                logger.info(f"{ticker} - NO FINAL SIGNAL: Insufficient votes or tie")
        except Exception as e:
            logger.error(f"{ticker} - PROCESS FAILED: {str(e)}")
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
        logger.debug("Weekend - no trading")
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
            logger.info(
                f"Pre-market: waiting {sleep_time/60:.1f} minutes until {MARKET_OPEN_STR}"
            )
            await asyncio.sleep(min(sleep_time, 300))
        return True
    elif now > market_close_dt:
        logger.info("Market closed")
        return False
    elif now > trading_end_dt:
        logger.info("Post trading end time")
        return False
    return True


async def schedule_square_off():
    while True:
        try:
            now = datetime.now(IST)
            target_time = IST.localize(
                datetime.combine(now.date(), SQUARE_OFF_TIME_TIME)
            )
            if now > target_time:
                target_time += timedelta(days=1)
            sleep_seconds = (target_time - now).total_seconds()
            if sleep_seconds > 0:
                logger.info(f"Square off scheduled in {sleep_seconds/60:.1f} minutes")
                await asyncio.sleep(min(sleep_seconds, 3600))
                if await market_hours_check():
                    logger.info("Executing scheduled square off")
                    # NEW: Close all open positions
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
                    logger.info("Skipping square off on non-trading day")
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
            open_positions = len(
                position_manager.open_positions
            )  # NEW: Include position count
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
            logger.debug("Completed resource cleanup")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def main_trading_loop():
    try:
        await telegram_queue.start()
        try:
            strategies_df = pd.read_csv("selected_stocks_strategies.csv")
            nifty500 = pd.read_csv("ind_nifty500list.csv")
            logger.info(f"Loaded {len(strategies_df)} strategy configurations")
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

        # NEW: Start position manager monitoring
        background_tasks = [
            asyncio.create_task(schedule_square_off()),
            asyncio.create_task(send_heartbeat()),
            asyncio.create_task(cleanup_resources()),
            asyncio.create_task(position_manager.monitor_positions()),
        ]

        batch_size = 10
        while await market_hours_check():
            start_time = datetime.now()
            for i in range(0, len(stock_universe), batch_size):
                batch = stock_universe[i : i + batch_size]
                batch_tasks = [
                    asyncio.create_task(
                        process_stock(s["ticker"], s["security_id"], s["strategies"])
                    )
                    for s in batch
                ]
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=25.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Batch {i//batch_size + 1} timed out")
                    for task in batch_tasks:
                        if not task.done():
                            task.cancel()
                await asyncio.sleep(1)
            elapsed = (datetime.now() - start_time).total_seconds()
            sleep_time = max(30 - elapsed, 5)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        logger.info("Trading session completed")
    except Exception as e:
        logger.critical(f"Main loop failure: {str(e)}")
        logger.error(traceback.format_exc())
        await send_telegram_alert(f"*CRITICAL ERROR*\nTrading stopped: {str(e)}")
    finally:
        await api_client.close()
        thread_pool.shutdown(wait=True)
        backtrader_runner.shutdown()  # NEW: Shutdown Backtrader executor
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
