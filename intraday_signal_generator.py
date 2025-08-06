import asyncio
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
    get_security_symbol_map,
    CONFIG,
)

# Import registry functions
from comprehensive_backtesting.registry import get_strategy
from live_strategies.trendline_williams import TrendlineWilliams

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


class PositionManager:
    def __init__(self):
        self.open_positions = {}
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
                    security_ids = [
                        pos["security_id"] for pos in self.open_positions.values()
                    ]

                    if not security_ids:
                        await asyncio.sleep(60)
                        continue

                    quotes = await fetch_realtime_quote(security_ids)

                    for order_id, pos in list(self.open_positions.items()):
                        quote = quotes.get(pos["security_id"])
                        if not quote:
                            continue
                        current_price = quote["price"]

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

                        if (now - pos["last_updated"]).total_seconds() > 300:
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

                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(300)


position_manager = PositionManager()


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

        quotes = await fetch_realtime_quote([security_id])
        quote = quotes.get(security_id)
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
        print(
            ">>>>>>>>>>>>>>>>>>",
            dhan.get_fund_limits().get("data").get("availabelBalance"),
        )
        if (
            dhan.get_fund_limits().get("data").get("availabelBalance")
            > current_price / 5
        ):
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
                    strategy_instance = TrendlineWilliams(combined_data, **params)
                    signal = strategy_instance.run()
                    if signal in ["BUY", "SELL"]:
                        signals.append(signal)
                        logger.info(f"{ticker} - {strategy_name} signal: {signal}")
                except Exception as e:
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
        ]

        batch_size = 3
        while await market_hours_check():
            start_time = datetime.now(IST)
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
