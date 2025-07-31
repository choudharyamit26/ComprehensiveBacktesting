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
from comprehensive_backtesting.data import init_dhan_client

# Import registry functions
from comprehensive_backtesting.registry import get_strategy

# Configure logging with improved performance
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_system.log", mode="a"),  # 1MB buffer
    ],
)
logger = logging.getLogger("quant_trader")
trade_logger = logging.getLogger("trade_execution")
trade_logger.setLevel(logging.INFO)
trade_logger.addHandler(logging.FileHandler("trades.log", mode="a"))

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
TRADING_END_STR = os.getenv("TRADING_END", "15:05:00")
FORCE_CLOSE_STR = os.getenv("FORCE_CLOSE", "15:15:00")
SQUARE_OFF_TIME_STR = os.getenv("SQUARE_OFF_TIME", "15:16:00")

# Pre-parse time objects for performance
MARKET_OPEN_TIME = time.fromisoformat(MARKET_OPEN_STR)
MARKET_CLOSE_TIME = time.fromisoformat(MARKET_CLOSE_STR)
TRADING_END_TIME = time.fromisoformat(TRADING_END_STR)
FORCE_CLOSE_TIME = time.fromisoformat(FORCE_CLOSE_STR)
SQUARE_OFF_TIME_TIME = time.fromisoformat(SQUARE_OFF_TIME_STR)

# Configuration constants
MIN_VOTES = int(os.getenv("MIN_VOTES", 2))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 6))
MORNING_WINDOW_SIZE = int(os.getenv("MORNING_WINDOW_SIZE", 3))
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", 100000))
MAX_QUANTITY = int(os.getenv("MAX_QUANTITY", 2))
MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS", 0.02))
VOLATILITY_THRESHOLD = float(os.getenv("VOLATILITY_THRESHOLD", 0.03))
LIQUIDITY_THRESHOLD = int(os.getenv("LIQUIDITY_THRESHOLD", 500000))
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", 180))
BID_ASK_THRESHOLD = int(os.getenv("BID_ASK_THRESHOLD", 1000))

# Initialize Dhan client and thread pool
dhan = init_dhan_client()
dhan_lock = asyncio.Lock()
thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dhan_worker")

# Preload and cache symbol map
try:
    nifty500_df = pd.read_csv("ind_nifty500list.csv")
    SYMBOL_MAP = nifty500_df.set_index("security_id")["ticker"].to_dict()
    # Create reverse mapping for faster lookups
    TICKER_TO_ID_MAP = nifty500_df.set_index("ticker")["security_id"].to_dict()
except Exception as e:
    logger.error(f"Failed to load symbol map: {e}")
    SYMBOL_MAP = {}
    TICKER_TO_ID_MAP = {}

# Optimized caches with size limits
HOLIDAY_CACHE = {}
MARKET_DEPTH_CACHE = defaultdict(lambda: deque(maxlen=100))  # Limit cache size
CANDLE_BUILDERS = {}
HISTORICAL_DATA_CACHE = {}  # Cache for historical data
VOLATILITY_CACHE = {}  # Cache for volatility calculations
VOLUME_CACHE = {}  # Cache for volume calculations

# Daily P&L tracker
daily_pnl_tracker = {"realized": 0.0, "unrealized": 0.0, "last_updated": None}

# Pre-compile timezone object
IST = pytz.timezone("Asia/Kolkata")


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
            # Refill tokens based on elapsed time
            elapsed = (now - self.last_refill).total_seconds()
            if elapsed > 60:  # Refill every minute
                self.tokens = self.rate_limit
                self.last_refill = now
            elif elapsed > 0:
                tokens_to_add = int(elapsed * (self.rate_limit / 60))
                self.tokens = min(self.rate_limit, self.tokens + tokens_to_add)
                if tokens_to_add > 0:
                    self.last_refill = now

            while self.tokens <= 0:
                await asyncio.sleep(0.1)
                # Check for refill again
                elapsed = (datetime.now() - self.last_refill).total_seconds()
                if elapsed > 60:
                    self.tokens = self.rate_limit
                    self.last_refill = datetime.now()

            self.tokens -= 1


# Optimized rate limiter instance
rate_limiter = OptimizedRateLimiter(API_RATE_LIMIT)


# Optimized Telegram utilities
class TelegramQueue:
    """Queue-based Telegram message sender to batch messages"""

    def __init__(self):
        self.message_queue = None
        self.is_running = False
        self._initialized = False

    async def _initialize(self):
        """Initialize the queue when event loop is available"""
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
                await asyncio.sleep(1)  # Rate limit Telegram messages
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Telegram queue error: {e}")

    async def _send_message(self, message: str):
        """Send message with optimized error handling"""
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
        """Non-blocking message sending"""
        try:
            if self.message_queue is not None:
                self.message_queue.put_nowait(message)
            else:
                logger.warning("Telegram queue not initialized, dropping message")
        except asyncio.QueueFull:
            logger.warning("Telegram queue full, dropping message")


# Global Telegram queue - will be initialized when first used
telegram_queue = TelegramQueue()


async def send_telegram_alert(message: str):
    """Legacy wrapper for backward compatibility"""
    telegram_queue.send_alert(message)


# Optimized API operations with connection pooling
class APIClient:
    """Optimized API client with connection pooling"""

    def __init__(self):
        self.connector = None
        self.timeout = None
        self.session = None
        self._initialized = False

    async def _initialize(self):
        """Initialize the connector and timeout when event loop is available"""
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


# Global API client - will be initialized when first used
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
    """Optimized order placement with connection reuse"""
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
    """Optimized order modification"""
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
    """Optimized order cancellation"""
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
    """Optimized market order placement"""
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
    """Optimized order verification"""
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


# Optimized market depth functions
async def fetch_and_store_market_depth(security_id: int) -> bool:
    """Optimized market depth fetching with better caching"""
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

                    # Calculate depth metrics efficiently
                    buy_levels = depth.get("buy", [])
                    sell_levels = depth.get("sell", [])

                    bid_qty = buy_levels[0]["quantity"] if buy_levels else 0
                    ask_qty = sell_levels[0]["quantity"] if sell_levels else 0
                    bid_depth = sum(level["quantity"] for level in buy_levels[:5])
                    ask_depth = sum(level["quantity"] for level in sell_levels[:5])

                    # Store in optimized cache
                    MARKET_DEPTH_CACHE[security_id].append(
                        {
                            "timestamp": timestamp,
                            "bid_qty": bid_qty,
                            "ask_qty": ask_qty,
                            "bid_depth": bid_depth,
                            "ask_depth": ask_depth,
                            "ltp": float(depth["ltp"]),
                            "volume": int(depth["volume"]),
                        }
                    )
                    return True
        return False
    except Exception as e:
        logger.error(f"Market depth storage error: {str(e)}")
        return False


def build_enhanced_candles(
    security_id: int, interval_minutes: int = 5
) -> Optional[List[EnhancedCandle]]:
    """Optimized candle building with memory efficiency"""
    depth_cache = MARKET_DEPTH_CACHE.get(security_id)
    if not depth_cache:
        return None

    # Initialize candle builder if needed
    if security_id not in CANDLE_BUILDERS:
        CANDLE_BUILDERS[security_id] = {
            "current_candle": EnhancedCandle(),
            "last_candle_time": None,
        }

    builder = CANDLE_BUILDERS[security_id]
    candles = []

    # Process data points efficiently
    for data_point in list(depth_cache):  # Convert deque to list for iteration
        timestamp = data_point["timestamp"]
        candle_time = timestamp.replace(second=0, microsecond=0)
        minute_group = (timestamp.minute // interval_minutes) * interval_minutes
        candle_time = candle_time.replace(minute=minute_group)

        # Check if we're in a new candle
        if builder["last_candle_time"] != candle_time:
            # Finalize current candle if it has data
            if builder["current_candle"].volume > 0:
                candles.append(builder["current_candle"])

            # Start new candle
            builder["current_candle"] = EnhancedCandle()
            builder["current_candle"].open = data_point["ltp"]
            builder["current_candle"].datetime = candle_time
            builder["last_candle_time"] = candle_time

        # Update candle efficiently
        candle = builder["current_candle"]
        price = data_point["ltp"]

        if candle.high == 0:
            candle.high = price
        else:
            candle.high = max(candle.high, price)

        if candle.low == float("inf"):
            candle.low = price
        else:
            candle.low = min(candle.low, price)

        candle.close = price
        candle.volume += data_point["volume"]

        # Update depth info
        candle.bid_qty = data_point["bid_qty"]
        candle.ask_qty = data_point["ask_qty"]
        candle.bid_depth = data_point["bid_depth"]
        candle.ask_depth = data_point["ask_depth"]

    # Clear processed data
    depth_cache.clear()

    return candles


async def get_combined_data(security_id: int) -> Optional[pd.DataFrame]:
    """Optimized data combination with caching"""
    # Check cache first
    cache_key = f"{security_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    if cache_key in HISTORICAL_DATA_CACHE:
        hist_data = HISTORICAL_DATA_CACHE[cache_key]
    else:
        hist_data = await fetch_historical_data(security_id, days_back=5)
        if hist_data is None:
            return None
        # Cache with TTL
        HISTORICAL_DATA_CACHE[cache_key] = hist_data
        # Clean old cache entries
        if len(HISTORICAL_DATA_CACHE) > 100:
            old_keys = list(HISTORICAL_DATA_CACHE.keys())[:50]
            for old_key in old_keys:
                del HISTORICAL_DATA_CACHE[old_key]

    # Build enhanced candles
    enhanced_candles = build_enhanced_candles(security_id)
    if not enhanced_candles:
        return hist_data

    # Convert enhanced candles to DataFrame efficiently
    enhanced_data = []
    for candle in enhanced_candles:
        enhanced_data.append(
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
        )

    if enhanced_data:
        enhanced_df = pd.DataFrame(enhanced_data)
        # Efficient concatenation and deduplication
        combined = pd.concat([hist_data, enhanced_df], ignore_index=True)
        combined = (
            combined.sort_values("datetime")
            .drop_duplicates("datetime", keep="last")
            .reset_index(drop=True)
        )
        return combined

    return hist_data


# Optimized utility functions
@lru_cache(maxsize=1000)
def get_symbol_from_id(security_id: int) -> str:
    """Cached symbol resolution"""
    return SYMBOL_MAP.get(security_id, f"UNKNOWN_{security_id}")


async def fetch_dynamic_holidays(year: int) -> List[str]:
    """Optimized holiday fetching with better caching"""
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

    # Fallback to static list
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
    """Highly optimized historical data fetching"""
    try:
        now = datetime.now(IST)
        today = now.date()

        # Fetch holidays once and cache
        holidays = await fetch_dynamic_holidays(today.year)
        holiday_set = set(holidays)  # O(1) lookup

        # More efficient trading days calculation
        valid_days = []
        current_day = today
        days_checked = 0
        max_days_to_check = min(days_back * 2, 60)  # Reasonable limit

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
            """Optimized single day data fetch"""
            try:
                from_date = IST.localize(datetime.combine(day, MARKET_OPEN_TIME))

                if day == today and now.time() < MARKET_CLOSE_TIME:
                    to_date = now
                else:
                    to_date = IST.localize(datetime.combine(day, MARKET_CLOSE_TIME))

                to_date = min(to_date, now)

                # Use thread pool for blocking Dhan call
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
                        # Optimized datetime conversion
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

        # Use semaphore for concurrency control
        semaphore = asyncio.Semaphore(2)  # Reduced from 3 to prevent rate limiting

        async def fetch_with_semaphore(day: date) -> Optional[pd.DataFrame]:
            async with semaphore:
                await asyncio.sleep(0.1)  # Small delay between requests
                return await fetch_day_data(day)

        # Execute with better error handling
        tasks = [
            fetch_with_semaphore(day) for day in valid_days[:10]
        ]  # Limit to 10 days max
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results efficiently
        valid_dataframes = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Task failed for day {valid_days[i]}: {result}")
            elif result is not None and not result.empty:
                valid_dataframes.append(result)

        if not valid_dataframes:
            logger.error(f"No data fetched for {security_id}")
            return None

        # Efficient concatenation
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
    """Optimized realtime quote fetching"""
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
    """Cached VWAP calculation"""
    if volume_sum == 0:
        return 0
    return typical_price_volume_sum / volume_sum


async def calculate_vwap(hist_data: pd.DataFrame) -> float:
    """Optimized VWAP calculation with caching"""
    try:
        if hist_data is None or len(hist_data) == 0:
            return 0

        # Calculate typical price efficiently
        typical_price = (hist_data["high"] + hist_data["low"] + hist_data["close"]) / 3
        volume_sum = hist_data["volume"].sum()
        typical_price_volume_sum = (typical_price * hist_data["volume"]).sum()

        # Use cached calculation
        data_hash = hash(str(hist_data.shape) + str(volume_sum))
        return calculate_vwap_cached(data_hash, volume_sum, typical_price_volume_sum)
    except Exception as e:
        logger.error(f"VWAP calculation error: {e}")
        return 0


# Optimized risk management functions
@lru_cache(maxsize=200)
def calculate_regime_cached(adx_val: float, atr_val: float) -> Tuple[str, float, float]:
    """Cached regime calculation"""
    if adx_val > 25:
        return "trending", adx_val, atr_val
    elif adx_val < 20:
        return "range_bound", adx_val, atr_val
    return "transitional", adx_val, atr_val


def calculate_regime(
    data: pd.DataFrame, adx_period: int = 14
) -> Tuple[str, float, float]:
    """Optimized market regime calculation"""
    if len(data) < adx_period:
        return "unknown", 0.0, 0.0

    try:
        # Use vectorized operations
        adx = ta.adx(data["high"], data["low"], data["close"], length=adx_period)
        atr = ta.atr(data["high"], data["low"], data["close"], length=adx_period)

        latest_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0
        latest_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0

        return calculate_regime_cached(latest_adx, latest_atr)
    except Exception as e:
        logger.error(f"Regime calculation error: {str(e)}")
        return "unknown", 0.0, 0.0


@lru_cache(maxsize=500)
def calculate_risk_params_cached(
    regime: str, atr: float, current_price: float, direction: str, account_size: float
) -> Dict[str, float]:
    """Cached risk parameter calculation"""
    risk_per_trade = 0.01 * account_size

    # Handle zero ATR case
    if atr <= 0:
        atr = current_price * 0.01

    # Regime-based parameters (pre-computed)
    params_map = {
        "trending": {"sl_mult": 2.0, "tp_mult": 3.0, "risk_factor": 0.8},
        "range_bound": {"sl_mult": 1.5, "tp_mult": 2.0, "risk_factor": 1.0},
        "transitional": {"sl_mult": 1.8, "tp_mult": 2.5, "risk_factor": 0.9},
        "unknown": {"sl_mult": 1.8, "tp_mult": 2.5, "risk_factor": 0.9},
    }
    cfg = params_map.get(regime, params_map["unknown"])

    # Calculate risk parameters
    stop_loss_distance = atr * cfg["sl_mult"]
    position_size = min(
        MAX_QUANTITY,
        max(1, int((risk_per_trade / stop_loss_distance) * cfg["risk_factor"])),
    )

    if direction == "BUY":
        stop_loss = current_price - stop_loss_distance
        take_profit = current_price + (atr * cfg["tp_mult"])
    else:  # SELL
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
    """Wrapper for cached risk parameter calculation"""
    return calculate_risk_params_cached(
        regime, atr, current_price, direction, ACCOUNT_SIZE
    )


class OptimizedSignalGeneratorWrapper(bt.Strategy):
    """Optimized wrapper strategy with minimal overhead"""

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


# Optimized P&L tracking
class PnLTracker:
    """Optimized P&L tracker with caching"""

    def __init__(self):
        self.cache = {"realized": 0.0, "last_updated": None}
        self.cache_ttl = 300  # 5 minutes

    async def update_daily_pnl(self) -> float:
        """Fetch and update daily P&L with caching"""
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


# Global P&L tracker
pnl_tracker = PnLTracker()


# Optimized trading operations
async def execute_strategy_signal(
    ticker: str,
    security_id: int,
    signal: str,
    regime: str,
    adx_value: float,
    atr_value: float,
    hist_data: pd.DataFrame,
):
    """Optimized signal execution with better error handling"""
    try:
        # Circuit breaker check with cached P&L
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

        # Validate signal
        if signal not in ["BUY", "SELL"]:
            logger.warning(f"Invalid signal for {ticker}: {signal}")
            return

        # Get current price with retry
        quote = await fetch_realtime_quote(security_id)
        if not quote:
            logger.warning(f"Price unavailable for {ticker}")
            return

        current_price = quote["price"]

        # Calculate VWAP for better entry
        vwap = await calculate_vwap(hist_data)
        if vwap > 0:
            entry_price = (
                min(current_price, vwap * 0.998)
                if signal == "BUY"
                else max(current_price, vwap * 1.002)
            )
        else:
            entry_price = current_price

        # Calculate risk parameters using cached function
        risk_params = calculate_risk_params(regime, atr_value, entry_price, signal)

        # Prepare optimized alert message
        now = datetime.now(IST)
        message = (
            f"*{ticker} Signal*\n"
            f"Direction: {signal}\n"
            f"Entry: ₹{entry_price:.2f} | VWAP: ₹{vwap:.2f}\n"
            f"Regime: {regime} (ADX: {adx_value:.2f})\n"
            f"Size: {risk_params['position_size']} | SL: ₹{risk_params['stop_loss']:.2f}\n"
            f"TP: ₹{risk_params['take_profit']:.2f}\n"
            f"Time: {now.strftime('%H:%M:%S')}"
        )

        # Place order
        order_response = await place_super_order(
            security_id,
            signal,
            entry_price,
            risk_params["stop_loss"],
            risk_params["take_profit"],
            risk_params["position_size"],
        )

        if order_response and order_response.get("orderId"):
            await send_telegram_alert(message)
        else:
            await send_telegram_alert(
                f"*{ticker} Order Failed*\nSignal: {signal} at ₹{entry_price:.2f}"
            )

    except Exception as e:
        logger.error(f"Signal execution failed for {ticker}: {str(e)}")


# Optimized volatility and volume calculations with caching
async def calculate_stock_volatility(security_id: int) -> float:
    """Cached volatility calculation"""
    cache_key = f"vol_{security_id}_{date.today()}"
    if cache_key in VOLATILITY_CACHE:
        return VOLATILITY_CACHE[cache_key]

    try:
        hist_data = await fetch_historical_data(security_id, days_back=30, interval=5)
        if hist_data is None or len(hist_data) < 5:
            return 0

        # Efficient volatility calculation
        returns = hist_data["close"].pct_change().dropna()
        volatility = returns.std() * math.sqrt(252)

        # Cache result
        VOLATILITY_CACHE[cache_key] = volatility

        # Clean cache if too large
        if len(VOLATILITY_CACHE) > 500:
            old_keys = list(VOLATILITY_CACHE.keys())[:250]
            for key in old_keys:
                del VOLATILITY_CACHE[key]

        return volatility
    except Exception as e:
        logger.error(f"Volatility calculation error for {security_id}: {e}")
        return 0


async def calculate_average_volume(security_id: int) -> float:
    """Cached volume calculation"""
    cache_key = f"vol_{security_id}_{date.today()}"
    if cache_key in VOLUME_CACHE:
        return VOLUME_CACHE[cache_key]

    try:
        hist_data = await fetch_historical_data(security_id, days_back=10, interval=5)
        if hist_data is None or len(hist_data) < 5:
            return 0

        avg_volume = hist_data["volume"].mean()

        # Cache result
        VOLUME_CACHE[cache_key] = avg_volume

        # Clean cache if too large
        if len(VOLUME_CACHE) > 500:
            old_keys = list(VOLUME_CACHE.keys())[:250]
            for key in old_keys:
                del VOLUME_CACHE[key]

        return avg_volume
    except Exception as e:
        logger.error(f"Volume calculation error for {security_id}: {e}")
        return 0


# Optimized concurrency control
class AdaptiveSemaphore:
    """Adaptive semaphore that adjusts based on system load"""

    def __init__(self, initial_limit: int = 50):
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


# Global adaptive semaphore
adaptive_semaphore = AdaptiveSemaphore(30)  # Reduced from 50


async def process_stock(ticker: str, security_id: int, strategies: List[Dict]) -> None:
    """Enhanced debugging version of process_stock"""
    async with adaptive_semaphore:
        try:
            logger.info(f"=== PROCESSING {ticker} (ID: {security_id}) ===")

            # Step 1: Data fetching
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
            # print([get_strategy(strat["Strategy"]).get_min_data_points(strat["Best_Parameters"]) for strat in strategies])
            min_bars = max(
                get_strategy(strat["Strategy"]).get_min_data_points(
                    strat["Best_Parameters"]
                )
                for strat in strategies
            )
            # print("===============",min_bars)
            if data_length < 20:  # Reduced from 100
                logger.warning(
                    f"{ticker} - FAILED: Insufficient data ({data_length} < {min_bars})"
                )
                return

            # Step 2: Liquidity check
            # logger.info(f"{ticker} - Step 2: Liquidity check")
            # if not combined_data.empty:
            #     latest = combined_data.iloc[-1]
            #     print(">>>>>>",latest)
            #     bid_qty = latest.get("bid_qty", 0)
            #     ask_qty = latest.get("ask_qty", 0)

            #     logger.info(
            #         f"{ticker} - Bid: {bid_qty}, Ask: {ask_qty}, Threshold: {BID_ASK_THRESHOLD}"
            #     )

            #     if bid_qty < BID_ASK_THRESHOLD or ask_qty < BID_ASK_THRESHOLD:
            #         logger.warning(
            #             f"{ticker} - FAILED: Liquidity check (Bid={bid_qty}, Ask={ask_qty})"
            #         )
            #         return

            # Step 3: Volatility and volume checks
            logger.info(f"{ticker} - Step 3: Volatility and volume checks")
            vol_task = asyncio.create_task(calculate_stock_volatility(security_id))
            volume_task = asyncio.create_task(calculate_average_volume(security_id))

            volatility, avg_volume = await asyncio.gather(vol_task, volume_task)

            logger.info(
                f"{ticker} - Volatility: {volatility:.4f} (threshold: {VOLATILITY_THRESHOLD})"
            )
            logger.info(
                f"{ticker} - Avg Volume: {avg_volume:.0f} (threshold: {LIQUIDITY_THRESHOLD})"
            )

            if volatility < VOLATILITY_THRESHOLD:
                logger.warning(
                    f"{ticker} - FAILED: Volatility too low ({volatility:.4f} < {VOLATILITY_THRESHOLD})"
                )
                return

            if avg_volume < LIQUIDITY_THRESHOLD:
                logger.warning(
                    f"{ticker} - FAILED: Volume too low ({avg_volume:.0f} < {LIQUIDITY_THRESHOLD})"
                )
                return

            # Step 4: Strategy processing
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

                # Parse parameters
                params = strat.get("Best_Parameters", {})
                if isinstance(params, str) and params.strip():
                    try:
                        params = ast.literal_eval(params)
                        logger.info(f"{ticker} - {strategy_name} params: {params}")
                    except (ValueError, SyntaxError) as e:
                        logger.warning(f"{ticker} - Failed to parse params: {e}")
                        params = {}

                # Check data requirements
                min_bars = strategy_class.get_min_data_points(params)
                logger.info(
                    f"{ticker} - {strategy_name} needs {min_bars} bars, have {data_length}"
                )

                if data_length < min_bars:
                    logger.warning(
                        f"{ticker} - {strategy_name} SKIPPED: Not enough data"
                    )
                    continue

                # Run strategy
                try:
                    logger.info(f"{ticker} - Running {strategy_name}")
                    cerebro = bt.Cerebro(stdstats=False, runonce=True, optdatas=True)
                    data_feed = bt.feeds.PandasData(
                        dataname=combined_data,
                        datetime="datetime",
                        fromdate=combined_data.iloc[0]["datetime"],
                        todate=combined_data.iloc[-1]["datetime"],
                    )
                    cerebro.adddata(data_feed)
                    cerebro.addstrategy(
                        OptimizedSignalGeneratorWrapper,
                        strategy_class=strategy_class,
                        **params,
                    )

                    results = cerebro.run()
                    if results and results[0].signal:
                        signal = results[0].signal
                        signals.append(signal)
                        logger.info(
                            f"{ticker} - {strategy_name} GENERATED SIGNAL: {signal}"
                        )
                    else:
                        logger.info(f"{ticker} - {strategy_name} generated no signal")

                except Exception as e:
                    logger.error(f"{ticker} - {strategy_name} FAILED: {str(e)}")

            # Step 5: Signal voting
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

            # Calculate regime
            regime, adx_value, atr_value = calculate_regime(combined_data)
            logger.info(
                f"{ticker} - Regime: {regime} (ADX: {adx_value:.2f}, ATR: {atr_value:.2f})"
            )

            # Final decision
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


# Optimized market hours checking
@lru_cache(maxsize=10)
def get_market_times_cached(date_str: str) -> Tuple[datetime, datetime, datetime]:
    """Cache market times for the day"""
    day_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    market_open_dt = IST.localize(datetime.combine(day_date, MARKET_OPEN_TIME))
    market_close_dt = IST.localize(datetime.combine(day_date, MARKET_CLOSE_TIME))
    trading_end_dt = IST.localize(datetime.combine(day_date, TRADING_END_TIME))
    return market_open_dt, market_close_dt, trading_end_dt


async def market_hours_check() -> bool:
    """Optimized market hours validation"""
    now = datetime.now(IST)

    # Quick weekday check
    if now.weekday() >= 5:
        logger.debug("Weekend - no trading")
        return False

    # Cached holiday check
    holidays = await fetch_dynamic_holidays(now.year)
    if now.strftime("%Y-%m-%d") in holidays:
        logger.info("Market holiday - no trading")
        return False

    # Cached market times
    market_open_dt, market_close_dt, trading_end_dt = get_market_times_cached(
        now.strftime("%Y-%m-%d")
    )

    if now < market_open_dt:
        sleep_time = (market_open_dt - now).total_seconds()
        if sleep_time > 0:
            logger.info(
                f"Pre-market: waiting {sleep_time/60:.1f} minutes until {MARKET_OPEN_STR}"
            )
            await asyncio.sleep(min(sleep_time, 300))  # Max 5 minute sleep
        return True
    elif now > market_close_dt:
        logger.info("Market closed")
        return False
    elif now > trading_end_dt:
        logger.info("Post trading end time")
        return False

    return True


async def schedule_square_off():
    """Optimized square off scheduling"""
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
                await asyncio.sleep(min(sleep_seconds, 3600))  # Max 1 hour sleep

                if await market_hours_check():
                    logger.info("Executing scheduled square off")
                    # Add square off logic here
                else:
                    logger.info("Skipping square off on non-trading day")
            else:
                await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Square off scheduler error: {e}")
            await asyncio.sleep(300)  # 5 minute retry


async def send_heartbeat():
    """Optimized heartbeat with system metrics"""
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour

            # Get basic system info
            active_tasks = len([t for t in asyncio.all_tasks() if not t.done()])
            cache_sizes = {
                "depth": sum(len(cache) for cache in MARKET_DEPTH_CACHE.values()),
                "historical": len(HISTORICAL_DATA_CACHE),
                "volatility": len(VOLATILITY_CACHE),
                "volume": len(VOLUME_CACHE),
            }

            message = (
                "💓 *SYSTEM HEARTBEAT*\n"
                f"Status: Operational\n"
                f"Active Tasks: {active_tasks}\n"
                f"Cache Sizes: {cache_sizes}\n"
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            )
            await send_telegram_alert(message)
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")


async def cleanup_resources():
    """Cleanup resources periodically"""
    while True:
        try:
            await asyncio.sleep(1800)  # Every 30 minutes

            # Clear old cache entries
            current_time = datetime.now()

            # Clean market depth cache
            for security_id in list(MARKET_DEPTH_CACHE.keys()):
                cache = MARKET_DEPTH_CACHE[security_id]
                # Remove entries older than 1 hour
                while (
                    cache
                    and (current_time - cache[0]["timestamp"]).total_seconds() > 3600
                ):
                    cache.popleft()

            # Clean historical data cache (keep only current day)
            current_day = current_time.strftime("%Y%m%d")
            keys_to_remove = [
                key
                for key in HISTORICAL_DATA_CACHE.keys()
                if len(parts := key.split("_")) != 2
                or not parts[1].startswith(current_day)
            ]
            for key in keys_to_remove:
                del HISTORICAL_DATA_CACHE[key]

            logger.debug("Completed resource cleanup")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def main_trading_loop():
    """Optimized main trading loop with better error handling"""
    try:
        # Start Telegram queue
        await telegram_queue.start()

        # Load data with error handling
        try:
            strategies_df = pd.read_csv("selected_stocks_strategies.csv")
            nifty500 = pd.read_csv("ind_nifty500list.csv")
            logger.info(f"Loaded {len(strategies_df)} strategy configurations")
        except Exception as e:
            logger.critical(f"Data load failed: {str(e)}")
            return

        # Build optimized stock universe
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

        # Start background tasks
        background_tasks = [
            asyncio.create_task(schedule_square_off()),
            asyncio.create_task(send_heartbeat()),
            asyncio.create_task(cleanup_resources()),
        ]

        # Main trading loop with better batch processing
        batch_size = 10  # Process stocks in batches

        while await market_hours_check():
            start_time = datetime.now()

            # Process stocks in batches to reduce memory usage
            for i in range(0, len(stock_universe), batch_size):
                batch = stock_universe[i : i + batch_size]

                # Create tasks for current batch
                batch_tasks = [
                    asyncio.create_task(
                        process_stock(s["ticker"], s["security_id"], s["strategies"])
                    )
                    for s in batch
                ]

                # Wait for batch completion with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=25.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Batch {i//batch_size + 1} timed out")
                    # Cancel remaining tasks
                    for task in batch_tasks:
                        if not task.done():
                            task.cancel()

                # Small delay between batches
                await asyncio.sleep(1)

            # Adaptive throttling
            elapsed = (datetime.now() - start_time).total_seconds()
            sleep_time = max(30 - elapsed, 5)  # Minimum 5 seconds between cycles
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        logger.info("Trading session completed")

    except Exception as e:
        logger.critical(f"Main loop failure: {str(e)}")
        logger.error(traceback.format_exc())
        await send_telegram_alert(f"*CRITICAL ERROR*\nTrading stopped: {str(e)}")
    finally:
        # Cleanup
        await api_client.close()
        thread_pool.shutdown(wait=True)

        # Cancel background tasks
        for task in background_tasks if "background_tasks" in locals() else []:
            task.cancel()


if __name__ == "__main__":
    try:
        # Set event loop policy for better performance on Windows
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        asyncio.run(main_trading_loop())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.critical(f"System failure: {str(e)}")
        logger.error(traceback.format_exc())
