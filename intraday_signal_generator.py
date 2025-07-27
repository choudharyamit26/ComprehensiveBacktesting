import asyncio
import pandas as pd
import backtrader as bt
import numpy as np
import os
import logging
from datetime import date, datetime, timedelta, time
import pytz
from retrying import retry
import requests
import ast
import pandas_ta as ta
import aiohttp
import json
import math
import sys
import traceback
from comprehensive_backtesting.data import init_dhan_client

# Import registry functions
from comprehensive_backtesting.registry import get_strategy, STRATEGY_REGISTRY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("trading_system.log")],
)
logger = logging.getLogger("quant_trader")
trade_logger = logging.getLogger("trade_execution")
trade_logger.setLevel(logging.INFO)
trade_logger.addHandler(logging.FileHandler("trades.log"))

# Environment configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "1000000003")  # Default test ID

# Validate environment variables
if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_ACCESS_TOKEN]):
    logger.critical("Missing required environment variables")
    raise EnvironmentError("Required environment variables not set")

# Market configuration
MARKET_OPEN = os.getenv("MARKET_OPEN", "09:15:00")
MARKET_CLOSE = os.getenv("MARKET_CLOSE", "15:30:00")
TRADING_END = os.getenv("TRADING_END", "15:05:00")
FORCE_CLOSE = os.getenv("FORCE_CLOSE", "15:15:00")
SQUARE_OFF_TIME = os.getenv("SQUARE_OFF_TIME", "15:16:00")  # Square off at 3:16 PM
MIN_VOTES = int(os.getenv("MIN_VOTES", 2))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 6))
MORNING_WINDOW_SIZE = int(os.getenv("MORNING_WINDOW_SIZE", 3))
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", 100000))
MAX_QUANTITY = int(os.getenv("MAX_QUANTITY", 2))
MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS", 0.02))  # 2% of account
VOLATILITY_THRESHOLD = float(os.getenv("VOLATILITY_THRESHOLD", 0.03))  # 3% daily move
LIQUIDITY_THRESHOLD = int(os.getenv("LIQUIDITY_THRESHOLD", 500000))  # 5L shares/day
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", 180))  # 3 requests/sec
BID_ASK_THRESHOLD = int(os.getenv("BID_ASK_THRESHOLD", 1000))  # Default 1000 shares

# Initialize Dhan client
dhan = init_dhan_client()
dhan_lock = asyncio.Lock()  # Thread safety for Dhan client

# Preload symbol map
try:
    nifty500_df = pd.read_csv("ind_nifty500list.csv")
    SYMBOL_MAP = nifty500_df.set_index("security_id")["ticker"].to_dict()
except Exception as e:
    logger.error(f"Failed to load symbol map: {e}")
    SYMBOL_MAP = {}

# Holiday cache
HOLIDAY_CACHE = {}

# Market depth cache and candle builders
MARKET_DEPTH_CACHE = {}
CANDLE_BUILDERS = {}

# Daily P&L tracker
daily_pnl_tracker = {"realized": 0.0, "unrealized": 0.0, "last_updated": None}


# Enhanced candle structure with depth information
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
        self.bid_qty = 0  # Best bid quantity
        self.ask_qty = 0  # Best ask quantity
        self.bid_depth = 0  # Total bid depth (sum of top 5 levels)
        self.ask_depth = 0  # Total ask depth (sum of top 5 levels)


# Telegram utilities
async def send_telegram_alert(message):
    """Send alert with truncation handling and markdown support"""
    try:
        # Truncate long messages
        if len(message) > 4000:
            message = message[:3900] + "...\n[TRUNCATED]"

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=5) as response:
                if response.status == 200:
                    logger.info(f"Telegram alert sent: {message[:100]}...")
                else:
                    text = await response.text()
                    logger.error(f"Telegram error {response.status}: {text}")
    except Exception as e:
        logger.error(f"Telegram send failed: {str(e)}")


# API rate limiting
api_rate_bucket = API_RATE_LIMIT
last_rate_reset = datetime.now()


async def consume_api_token():
    global api_rate_bucket, last_rate_reset

    now = datetime.now()
    if (now - last_rate_reset).total_seconds() > 60:
        api_rate_bucket = API_RATE_LIMIT
        last_rate_reset = now

    while api_rate_bucket <= 0:
        await asyncio.sleep(0.1)
        if (datetime.now() - last_rate_reset).total_seconds() > 60:
            api_rate_bucket = API_RATE_LIMIT
            last_rate_reset = datetime.now()

    api_rate_bucket -= 1


# DhanHQ API operations
@retry(
    stop_max_attempt_number=3,
    wait_exponential_multiplier=2000,
    wait_exponential_max=30000,
)
async def place_super_order(
    security_id,
    transaction_type,
    current_price,
    stop_loss,
    take_profit,
    quantity=MAX_QUANTITY,
):
    """Place Super Order with proper risk management"""
    try:
        await consume_api_token()

        url = "https://api.dhan.co/v2/super/orders"
        headers = {
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
        }

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

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=headers, json=payload, timeout=10
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    order_id = data.get("orderId")
                    if order_id:
                        # Log and return
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
    wait_exponential_multiplier=2000,
    wait_exponential_max=30000,
)
async def modify_super_order(order_id, leg_name, **params):
    """Modify existing Super Order"""
    try:
        await consume_api_token()

        url = f"https://api.dhan.co/v2/super/orders/{order_id}"
        headers = {
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
        }

        payload = {
            "dhanClientId": DHAN_CLIENT_ID,
            "orderId": order_id,
            "legName": leg_name,
        }
        payload.update(params)

        async with aiohttp.ClientSession() as session:
            async with session.put(
                url, headers=headers, json=payload, timeout=10
            ) as response:
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
    wait_exponential_multiplier=2000,
    wait_exponential_max=30000,
)
async def cancel_super_order(order_id, leg_name):
    """Cancel Super Order leg"""
    try:
        await consume_api_token()

        url = f"https://api.dhan.co/v2/super/orders/{order_id}/{leg_name}"
        headers = {
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
        }

        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=headers, timeout=10) as response:
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


async def place_market_order(security_id, transaction_type, quantity):
    """Place market order for square off"""
    try:
        await consume_api_token()

        url = "https://api.dhan.co/v2/orders"
        headers = {
            "Content-Type": "application/json",
            "access-token": DHAN_ACCESS_TOKEN,
        }

        payload = {
            "dhanClientId": DHAN_CLIENT_ID,
            "exchangeSegment": "NSE_EQ",
            "securityId": str(security_id),
            "transactionType": transaction_type,
            "orderType": "MARKET",
            "productType": "INTRADAY",
            "quantity": quantity,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=headers, json=payload, timeout=10
            ) as response:
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


async def verify_order_execution(order_id):
    """Verify if order was executed"""
    try:
        await consume_api_token()

        url = f"https://api.dhan.co/v2/orders/{order_id}"
        headers = {"access-token": DHAN_ACCESS_TOKEN}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "EXECUTED":
                        return True
        return False
    except Exception:
        return False


# Market depth functions
async def fetch_and_store_market_depth(security_id):
    """Fetch and store market depth data for candle building"""
    try:
        await consume_api_token()
        url = "https://api.dhan.co/v2/marketDepth"
        headers = {"access-token": DHAN_ACCESS_TOKEN}
        params = {"securityId": str(security_id), "exchangeSegment": "NSE_EQ"}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, params=params, timeout=3
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("data") and "depth" in data["data"]:
                        depth = data["data"]["depth"]
                        timestamp = datetime.now()

                        # Store in cache
                        if security_id not in MARKET_DEPTH_CACHE:
                            MARKET_DEPTH_CACHE[security_id] = []

                        # Calculate depth metrics
                        bid_qty = (
                            depth["buy"][0]["quantity"]
                            if depth["buy"] and len(depth["buy"]) > 0
                            else 0
                        )
                        ask_qty = (
                            depth["sell"][0]["quantity"]
                            if depth["sell"] and len(depth["sell"]) > 0
                            else 0
                        )
                        bid_depth = (
                            sum(level["quantity"] for level in depth["buy"][:5])
                            if depth["buy"]
                            else 0
                        )
                        ask_depth = (
                            sum(level["quantity"] for level in depth["sell"][:5])
                            if depth["sell"]
                            else 0
                        )

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


def build_enhanced_candles(security_id, interval_minutes=5):
    """Build 5-minute candles with depth information"""
    if security_id not in MARKET_DEPTH_CACHE or not MARKET_DEPTH_CACHE[security_id]:
        return None

    # Initialize candle builder if needed
    if security_id not in CANDLE_BUILDERS:
        CANDLE_BUILDERS[security_id] = {
            "current_candle": EnhancedCandle(),
            "last_candle_time": None,
        }

    builder = CANDLE_BUILDERS[security_id]
    depth_data = MARKET_DEPTH_CACHE[security_id]
    candles = []

    for data_point in depth_data:
        timestamp = data_point["timestamp"]
        candle_time = timestamp.replace(second=0, microsecond=0)
        minute_group = (timestamp.minute // interval_minutes) * interval_minutes
        candle_time = candle_time.replace(minute=minute_group)

        # Check if we're in a new candle
        if builder["last_candle_time"] is None:
            builder["last_candle_time"] = candle_time
            builder["current_candle"].open = data_point["ltp"]
            builder["current_candle"].datetime = candle_time
        elif candle_time != builder["last_candle_time"]:
            # Finalize current candle
            if builder["current_candle"].volume > 0:
                candles.append(builder["current_candle"])

            # Start new candle
            builder["current_candle"] = EnhancedCandle()
            builder["current_candle"].open = data_point["ltp"]
            builder["current_candle"].datetime = candle_time
            builder["last_candle_time"] = candle_time

        # Update candle with new data
        candle = builder["current_candle"]
        price = data_point["ltp"]

        candle.high = max(candle.high, price) if candle.high != 0 else price
        candle.low = min(candle.low, price) if candle.low != float("inf") else price
        candle.close = price
        candle.volume += data_point["volume"]

        # Update depth info (use last values in the interval)
        candle.bid_qty = data_point["bid_qty"]
        candle.ask_qty = data_point["ask_qty"]
        candle.bid_depth = data_point["bid_depth"]
        candle.ask_depth = data_point["ask_depth"]

    # Clear processed data
    MARKET_DEPTH_CACHE[security_id] = []

    return candles


async def get_combined_data(security_id):
    """Combine historical and real-time enhanced candles"""
    # Fetch historical data
    hist_data = await fetch_historical_data(security_id, days_back=5)
    if hist_data is None:
        return None

    # Build enhanced candles from depth data
    enhanced_candles = build_enhanced_candles(security_id)

    if not enhanced_candles:
        return hist_data

    # Convert enhanced candles to DataFrame
    enhanced_list = []
    for candle in enhanced_candles:
        enhanced_list.append(
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

    enhanced_df = pd.DataFrame(enhanced_list)

    # Combine with historical data
    combined = pd.concat([hist_data, enhanced_df]).sort_values("datetime")
    combined = combined.drop_duplicates("datetime").reset_index(drop=True)

    return combined


# Data utilities
def get_symbol_from_id(security_id):
    """Resolve symbol from security ID"""
    return SYMBOL_MAP.get(security_id, f"UNKNOWN_{security_id}")


async def fetch_dynamic_holidays(year):
    """Fetch NSE holidays dynamically"""
    if year in HOLIDAY_CACHE:
        return HOLIDAY_CACHE[year]

    try:
        url = f"https://www.nseindia.com/api/holiday-master?type=trading"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    holidays = [
                        d["tradingDate"] for d in data["data"] if d["trading"] == "N"
                    ]
                    logger.info(f"Fetched {len(holidays)} trading holidays for {year}")
                    HOLIDAY_CACHE[year] = holidays
                    return holidays
        return []
    except:
        logger.warning("Failed to fetch holidays, using static list")
        holidays = [  # Fallback to 2025 holidays
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


async def fetch_historical_data(security_id, days_back=20, interval="5min"):
    """Fetch historical data with dynamic holiday handling"""
    try:
        ist = pytz.timezone("Asia/Kolkata")
        today = datetime.now(ist).date()
        holidays = await fetch_dynamic_holidays(today.year)

        # Calculate valid trading days
        valid_days = []
        current_day = today - timedelta(days=1)
        while len(valid_days) < days_back:
            if (
                current_day.weekday() < 5
                and current_day.strftime("%Y-%m-%d") not in holidays
            ):
                valid_days.append(current_day)
            current_day -= timedelta(days=1)

        # Fetch data for valid days
        all_data = []
        interval_min = int(interval.replace("m", ""))

        for day in valid_days:
            from_date = datetime.combine(day, time(9, 15), tzinfo=ist)
            to_date = datetime.combine(day, time(15, 30), tzinfo=ist)

            # Use thread for synchronous Dhan call
            async with dhan_lock:
                data = await asyncio.to_thread(
                    dhan.intraday_minute_data,
                    security_id=security_id,
                    exchange_segment="NSE_EQ",
                    instrument_type="EQUITY",
                    interval=interval_min,
                    from_date=from_date.strftime("%Y-%m-%d %H:%M:%S"),
                    to_date=to_date.strftime("%Y-%m-%d %H:%M:%S"),
                )

            if data.get("status") == "success" and data.get("data"):
                df = pd.DataFrame(data["data"])
                df["datetime"] = pd.to_datetime(
                    df["timestamp"], unit="s", utc=True
                ).dt.tz_convert(ist)
                df = df[["datetime", "open", "high", "low", "close", "volume"]]
                all_data.append(df)

        if not all_data:
            return None

        full_df = pd.concat(all_data).sort_values("datetime")
        logger.info(f"Fetched {len(full_df)} bars for {security_id}")
        return full_df
    except Exception as e:
        logger.error(f"Historical data error: {str(e)}")
        return None


async def fetch_realtime_quote(security_id):
    """Fetch realtime quote for a single security"""
    try:
        await consume_api_token()

        url = "https://api.dhan.co/v2/quotes"
        headers = {"access-token": DHAN_ACCESS_TOKEN}
        params = {"securityId": str(security_id), "exchangeSegment": "NSE_EQ"}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, params=params, timeout=5
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("data"):
                        quote = data["data"][0]
                        return {
                            "price": float(quote["last_price"]),
                            "timestamp": datetime.strptime(
                                quote["last_trade_time"], "%d/%m/%Y %H:%M:%S"
                            ).replace(tzinfo=pytz.timezone("Asia/Kolkata")),
                        }
        return None
    except Exception as e:
        logger.error(f"Realtime quote error: {str(e)}")
        return None


async def calculate_vwap(hist_data):
    """Calculate Volume Weighted Average Price"""
    try:
        if hist_data is None or len(hist_data) == 0:
            return 0
        hist_data["typical_price"] = (
            hist_data["high"] + hist_data["low"] + hist_data["close"]
        ) / 3
        vwap = (hist_data["typical_price"] * hist_data["volume"]).sum() / hist_data[
            "volume"
        ].sum()
        return vwap
    except:
        return 0


# Risk management
def calculate_regime(data, adx_period=14):
    """Determine market regime with ADX"""
    if len(data) < adx_period:
        return "unknown", 0.0, 0.0

    try:
        adx = ta.adx(data["high"], data["low"], data["close"], length=adx_period)
        atr = ta.atr(data["high"], data["low"], data["close"], length=adx_period)
        latest_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0
        latest_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0

        if latest_adx > 25:
            return "trending", latest_adx, latest_atr
        elif latest_adx < 20:
            return "range_bound", latest_adx, latest_atr
        return "transitional", latest_adx, latest_atr
    except Exception as e:
        logger.error(f"Regime calculation error: {str(e)}")
        return "unknown", 0.0, 0.0


def calculate_risk_params(regime, atr, current_price, direction):
    """Calculate position sizing and risk levels"""
    risk_per_trade = 0.01 * ACCOUNT_SIZE

    # Handle zero ATR case
    if atr <= 0:
        atr = current_price * 0.01  # Default to 1% of price

    # Regime-based parameters
    params = {
        "trending": {"sl_mult": 2.0, "tp_mult": 3.0, "risk_factor": 0.8},
        "range_bound": {"sl_mult": 1.5, "tp_mult": 2.0, "risk_factor": 1.0},
        "transitional": {"sl_mult": 1.8, "tp_mult": 2.5, "risk_factor": 0.9},
        "unknown": {"sl_mult": 1.8, "tp_mult": 2.5, "risk_factor": 0.9},
    }
    cfg = params.get(regime, params["unknown"])

    # Calculate risk parameters
    stop_loss_distance = atr * cfg["sl_mult"]
    position_size = min(
        MAX_QUANTITY, int((risk_per_trade / stop_loss_distance) * cfg["risk_factor"])
    )
    position_size = max(1, position_size)

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


# Strategy wrapper for signal generation
class SignalGeneratorWrapper(bt.Strategy):
    """
    Wrapper strategy to capture signals without executing trades
    """

    def __init__(self, strategy_class, **params):
        self.signal = None
        self.strategy = strategy_class(self, **params)

    def next(self):
        # Reset signal at each bar
        self.signal = None

        # Run the original strategy
        self.strategy.next()

    def buy(self, *args, **kwargs):
        self.signal = "BUY"

    def sell(self, *args, **kwargs):
        self.signal = "SELL"


# P&L Tracking
async def update_daily_pnl():
    """Fetch and update daily P&L from broker"""
    try:
        await consume_api_token()
        url = "https://api.dhan.co/v2/reports/pnl"
        headers = {"access-token": DHAN_ACCESS_TOKEN}
        params = {
            "type": "INTRADAY",
            "fromDate": date.today().strftime("%Y-%m-%d"),
            "toDate": date.today().strftime("%Y-%m-%d"),
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, params=params, timeout=5
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("data"):
                        total_pnl = 0.0
                        for item in data["data"]:
                            total_pnl += float(item["netPnl"])

                        daily_pnl_tracker["realized"] = total_pnl
                        daily_pnl_tracker["last_updated"] = datetime.now()
                        logger.info(f"Updated daily P&L: ₹{total_pnl:.2f}")
                    return total_pnl
        return 0.0
    except Exception as e:
        logger.error(f"Failed to update P&L: {str(e)}")
        return 0.0


# Trading operations
async def execute_strategy_signal(
    ticker, security_id, signal, regime, adx_value, atr_value, hist_data
):
    """Execute trading signal with proper risk management"""
    # Circuit breaker check
    current_pnl = await update_daily_pnl()
    if current_pnl <= -MAX_DAILY_LOSS_PERCENT * ACCOUNT_SIZE:
        await send_telegram_alert(
            f"🛑 TRADING HALTED: Daily loss limit reached\n"
            f"Current P&L: ₹{current_pnl:.2f}\n"
            f"Limit: ₹{-MAX_DAILY_LOSS_PERCENT * ACCOUNT_SIZE:.2f}"
        )
        logger.critical("Daily loss limit reached - trading halted")
        return  # Skip trade execution

    # Validate signal
    if signal not in ["BUY", "SELL"]:
        logger.warning(f"Invalid signal for {ticker}: {signal}")
        return

    # Get current price
    quote = await fetch_realtime_quote(security_id)
    if not quote:
        logger.warning(f"Price unavailable for {ticker}")
        return

    current_price = quote["price"]

    # Calculate VWAP for better entry
    vwap = await calculate_vwap(hist_data)
    if vwap > 0:
        if signal == "BUY":
            entry_price = min(current_price, vwap * 0.998)
        else:  # SELL
            entry_price = max(current_price, vwap * 1.002)
    else:
        entry_price = current_price

    # Calculate risk parameters
    risk_params = calculate_risk_params(regime, atr_value, entry_price, signal)

    # Prepare alert message
    message = (
        f"*{ticker} Signal*\n"
        f"Direction: {signal}\n"
        f"Entry Price: ₹{entry_price:.2f}\n"
        f"VWAP: ₹{vwap:.2f}\n"
        f"Regime: {regime} (ADX: {adx_value:.2f})\n"
        f"Size: {risk_params['position_size']} shares\n"
        f"Stop Loss: ₹{risk_params['stop_loss']:.2f}\n"
        f"Take Profit: ₹{risk_params['take_profit']:.2f}\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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


async def calculate_stock_volatility(security_id):
    """Calculate historical volatility"""
    try:
        hist_data = await fetch_historical_data(
            security_id, days_back=30, interval="1d"
        )
        if hist_data is None or len(hist_data) < 5:
            return 0

        returns = hist_data["close"].pct_change().dropna()
        volatility = returns.std() * math.sqrt(252)  # Annualized volatility
        return volatility
    except:
        return 0


async def calculate_average_volume(security_id):
    """Calculate average daily volume"""
    try:
        hist_data = await fetch_historical_data(
            security_id, days_back=10, interval="1d"
        )
        if hist_data is None or len(hist_data) < 5:
            return 0

        avg_volume = hist_data["volume"].mean()
        return avg_volume
    except:
        return 0


# Concurrency limiter
CONCURRENCY_LIMIT = 50
concurrency_semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)


async def process_stock(ticker, security_id, strategies):
    """Process stock with multiple strategies using enhanced data"""
    async with concurrency_semaphore:
        try:
            logger.info(f"Processing {ticker} with {len(strategies)} strategies")

            # Update market depth cache
            await fetch_and_store_market_depth(security_id)

            # Get combined data (historical + real-time enhanced)
            combined_data = await get_combined_data(security_id)
            if combined_data is None or len(combined_data) < 100:
                logger.warning(f"Insufficient data for {ticker}")
                return

            # Liquidity check using latest depth data
            if not combined_data.empty:
                latest = combined_data.iloc[-1]
                if "bid_qty" in latest and "ask_qty" in latest:
                    if (
                        latest["bid_qty"] < BID_ASK_THRESHOLD
                        or latest["ask_qty"] < BID_ASK_THRESHOLD
                    ):
                        logger.info(
                            f"Skipping {ticker} due to low liquidity: "
                            f"Bid={latest['bid_qty']}, Ask={latest['ask_qty']} "
                            f"(Threshold={BID_ASK_THRESHOLD})"
                        )
                        return

            # Filter by volatility and liquidity
            volatility = await calculate_stock_volatility(security_id)
            avg_volume = await calculate_average_volume(security_id)

            if volatility < VOLATILITY_THRESHOLD:
                logger.info(
                    f"Skipping {ticker} due to low volatility: {volatility:.4f}"
                )
                return
            if avg_volume < LIQUIDITY_THRESHOLD:
                logger.info(f"Skipping {ticker} due to low liquidity: {avg_volume:.0f}")
                return

            # Initialize strategies
            signals = []
            for strat in strategies:
                strategy_name = strat["Strategy"]
                try:
                    # Use registry to get strategy class
                    strategy_class = get_strategy(strategy_name)
                except KeyError:
                    logger.warning(
                        f"Strategy {strategy_name} not found in registry for {ticker}"
                    )
                    continue

                # Parse parameters
                params = strat.get("Best_Parameters", {})
                if isinstance(params, str) and params.strip():
                    try:
                        params = ast.literal_eval(params)
                    except:
                        params = {}
                elif not params:
                    params = {}

                # Create cerebro instance
                cerebro = bt.Cerebro(stdstats=False, runonce=True)
                data_feed = bt.feeds.PandasData(
                    dataname=combined_data, datetime="datetime"  # Use enhanced data
                )
                cerebro.adddata(data_feed)

                # Add wrapper strategy
                cerebro.addstrategy(
                    SignalGeneratorWrapper, strategy_class=strategy_class, **params
                )

                # Get min data points
                min_bars = strategy_class.get_min_data_points(params)
                if len(combined_data) < min_bars:
                    logger.warning(
                        f"Skipping {strategy_name} for {ticker}: need {min_bars} bars, have {len(combined_data)}"
                    )
                    continue

                # Run strategy
                try:
                    results = cerebro.run()
                    if results and results[0].signal:
                        signals.append(results[0].signal)
                        logger.info(
                            f"Strategy {strategy_name} generated {results[0].signal} signal for {ticker}"
                        )
                except Exception as e:
                    logger.error(
                        f"Strategy {strategy_name} failed for {ticker}: {str(e)}"
                    )
                    logger.error(traceback.format_exc())

            # Process signals
            if not signals:
                return

            # Count votes
            buy_votes = signals.count("BUY")
            sell_votes = signals.count("SELL")

            # Determine market regime
            regime, adx_value, atr_value = calculate_regime(combined_data)

            # Execute based on voting
            if buy_votes >= MIN_VOTES and buy_votes > sell_votes:
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
                await execute_strategy_signal(
                    ticker,
                    security_id,
                    "SELL",
                    regime,
                    adx_value,
                    atr_value,
                    combined_data,
                )

        except Exception as e:
            logger.error(f"Stock processing failed for {ticker}: {str(e)}")
            logger.error(traceback.format_exc())


async def market_hours_check():
    """Validate we're within market hours"""
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    # Check weekday
    if now.weekday() >= 5:
        logger.info("Weekend - no trading")
        return False

    # Check holidays
    holidays = await fetch_dynamic_holidays(now.year)
    if now.strftime("%Y-%m-%d") in holidays:
        logger.info("Market holiday - no trading")
        return False

    # Check trading hours
    market_open = datetime.strptime(MARKET_OPEN, "%H:%M:%S").time()
    market_close = datetime.strptime(MARKET_CLOSE, "%H:%M:%S").time()
    trading_end = datetime.strptime(TRADING_END, "%H:%M:%S").time()

    if now.time() < market_open:
        logger.info(f"Pre-market: waiting until {MARKET_OPEN}")
        await asyncio.sleep(
            (datetime.combine(now.date(), market_open) - now).total_seconds()
        )
        return True
    elif now.time() > market_close:
        logger.info("Market closed")
        return False
    elif now.time() > trading_end:
        logger.info("Post trading end time")
        return False
    return True


async def schedule_square_off():
    """Schedule daily square off at 3:16 PM IST"""
    ist = pytz.timezone("Asia/Kolkata")
    while True:
        now = datetime.now(ist)
        target_time = datetime.combine(
            now.date(), datetime.strptime(SQUARE_OFF_TIME, "%H:%M:%S").time()
        )

        if now > target_time:
            # Schedule for next day
            target_time += timedelta(days=1)

        sleep_seconds = (target_time - now).total_seconds()
        if sleep_seconds > 0:
            logger.info(
                f"Scheduled square off at {target_time} (in {sleep_seconds/60:.1f} minutes)"
            )
            await asyncio.sleep(sleep_seconds)

            # Verify it's a trading day
            if await market_hours_check():
                logger.info("Executing scheduled square off")
            else:
                logger.info("Skipping square off on non-trading day")
        else:
            await asyncio.sleep(60)  # Prevent tight loop


async def send_heartbeat():
    """Send periodic heartbeat to Telegram"""
    while True:
        await asyncio.sleep(3600)  # Every hour
        message = (
            "💓 *SYSTEM HEARTBEAT*\n"
            f"Status: Operational\n"
            f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        await send_telegram_alert(message)


async def main_trading_loop():
    """Main trading execution loop"""
    try:
        # Load strategy configurations
        try:
            strategies_df = pd.read_csv("selected_stocks_strategies.csv")
            nifty500 = pd.read_csv("ind_nifty500list.csv")
        except Exception as e:
            logger.critical(f"Data load failed: {str(e)}")
            return

        # Prepare stock universe
        stock_universe = []
        for ticker in strategies_df["Ticker"].unique():
            stock_data = strategies_df[strategies_df["Ticker"] == ticker]
            security_id = nifty500[nifty500["ticker"] == ticker]["security_id"].values
            if len(security_id) > 0:
                stock_universe.append(
                    {
                        "ticker": ticker,
                        "security_id": security_id[0],
                        "strategies": stock_data.to_dict("records"),
                    }
                )

        logger.info(f"Loaded {len(stock_universe)} stocks for trading")

        # Start background tasks
        asyncio.create_task(schedule_square_off())
        asyncio.create_task(send_heartbeat())

        # Main trading loop
        while await market_hours_check():
            start_time = datetime.now()

            # Process all stocks concurrently
            tasks = [
                process_stock(s["ticker"], s["security_id"], s["strategies"])
                for s in stock_universe
            ]
            await asyncio.gather(*tasks)

            # Throttle to 30-second intervals
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed < 30:
                await asyncio.sleep(30 - elapsed)

        logger.info("Trading session completed")

    except Exception as e:
        logger.critical(f"Main loop failure: {str(e)}")
        logger.error(traceback.format_exc())
        await send_telegram_alert(f"*CRITICAL ERROR*\nTrading stopped: {str(e)}")


if __name__ == "__main__":
    try:
        asyncio.run(main_trading_loop())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.critical(f"System failure: {str(e)}")
        logger.error(traceback.format_exc())
