"""
Utility functions for time handling, market hours, and price helpers.

This module groups small pure functions to avoid duplication and ease testing.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, time, timedelta
import os
from typing import Dict, Tuple

import aiohttp
from cachetools import TTLCache
import pytz

from .logging_setup import setup_logging
from .constants import DHAN_ACCESS_TOKEN, IST
from dotenv import load_dotenv
import pandas as pd
from pytz import timezone
import requests
import logging
from dotenv import load_dotenv

load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def round_to_tick_size(price: float, tick_size: float) -> float:
    """Round a price to the nearest multiple of the tick size."""
    return round(price / tick_size) * tick_size


# Cache for market times to avoid repeated calculations
_market_times_cache = {}


def get_market_times_cached(day_date=None):
    """Get market times with caching with IST timezone aware datetimes.

    Returns a tuple (open_dt, close_dt, trading_end_dt), each timezone-aware.
    """
    from live_data import CONFIG  # Local import to avoid cycles

    if day_date is None:
        day_date = datetime.now(IST).date()

    if day_date in _market_times_cache:
        return _market_times_cache[day_date]

    try:
        market_open_dt = datetime.combine(day_date, CONFIG["MARKET_OPEN"]).replace(
            tzinfo=IST
        )
        market_close_dt = datetime.combine(day_date, CONFIG["MARKET_CLOSE"]).replace(
            tzinfo=IST
        )
        trading_end_dt = datetime.combine(
            day_date, time.fromisoformat("15:20:00")
        ).replace(tzinfo=IST)
        _market_times_cache[day_date] = (
            market_open_dt,
            market_close_dt,
            trading_end_dt,
        )
        return market_open_dt, market_close_dt, trading_end_dt
    except Exception:
        now = datetime.now(IST)
        return (
            datetime.combine(now.date(), time(9, 15)).replace(tzinfo=IST),
            datetime.combine(now.date(), time(15, 30)).replace(tzinfo=IST),
            datetime.combine(now.date(), time(15, 20)).replace(tzinfo=IST),
        )


def is_trading_day(date_obj=None) -> bool:
    if date_obj is None:
        date_obj = datetime.now(IST).date()
    return date_obj.weekday() < 5


def market_hours_check() -> bool:
    """Return True if current time is within market open and close times."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    market_open_dt, market_close_dt, _ = get_market_times_cached(now.date())
    return market_open_dt <= now <= market_close_dt


def time_until_market_open() -> float:
    now = datetime.now(IST)
    today = now.date()
    market_open_dt, _, _ = get_market_times_cached(today)
    if now < market_open_dt and is_trading_day(today):
        return (market_open_dt - now).total_seconds()
    # compute next trading day
    next_day = today
    while True:
        next_day = next_day + timedelta(days=1)
        if is_trading_day(next_day):
            break
    next_open, _, _ = get_market_times_cached(next_day)
    return (next_open - now).total_seconds()


def time_until_market_close() -> float | None:
    now = datetime.now(IST)
    today = now.date()
    if not is_trading_day(today):
        return None
    _, market_close_dt, trading_end_dt = get_market_times_cached(today)
    if now < market_open_dt or now > trading_end_dt:  # type: ignore[name-defined]
        return None
    return (trading_end_dt - now).total_seconds()


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
        logger, _ = setup_logging()
        if hits + misses > 0:
            logger.debug(
                f"Cache {cache_name} - Hits: {hits}, Misses: {misses}, Hit Rate: {hits / (hits + misses):.2%}"
            )


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


# def get_index_signal_dhan_api(
#     security_id: str, index_name: str = "Index", threshold: float = 0.6
# ) -> dict:
#     """
#     Returns BUY, SELL, or BOTH recommendation for any index using Dhan API's intraday charts.

#     Signal Logic (Percentage-based):
#     - BUY: if Index moves >= +threshold% (from first candle open to most recent close)
#     - SELL: if Index moves <= -threshold% (from first candle open to most recent close)
#     - BOTH: if movement is between -threshold% and +threshold%

#     Parameters:
#     security_id: str, Dhan API security ID for the index (e.g., "13" for Nifty50, "21" for Nifty Auto)
#     index_name: str, Name of the index for logging/display purposes
#     threshold: float, Percentage threshold for signals (default: 0.6%)

#     Returns:
#     dict: Contains signal, change, and additional info
#     """
#     CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
#     ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")

#     try:
#         ist = timezone("Asia/Kolkata")
#         current_date = datetime.now(ist)
#         date_str = current_date.strftime("%Y-%m-%d")

#         logger.info(f"Fetching {index_name} data for date: {date_str}")

#         # API endpoint and payload
#         url = "https://api.dhan.co/v2/charts/intraday"
#         headers = {
#             "client-id": CLIENT_ID,
#             "access-token": ACCESS_TOKEN,
#             "Content-Type": "application/json",
#         }
#         payload = {
#             "securityId": security_id,
#             "exchangeSegment": "IDX_I",
#             "instrument": "INDEX",
#             "interval": "15",  # 15-minute intervals
#             "oi": False,
#             "fromDate": date_str,
#             "toDate": date_str,
#         }

#         # Make API request
#         response = requests.post(url, headers=headers, json=payload, timeout=30)
#         response.raise_for_status()
#         data = response.json()
#         # print(data)
#         # Validate API response
#         required_fields = ["open", "high", "low", "close", "volume", "timestamp"]
#         if not all(field in data for field in required_fields):
#             return {
#                 "signal": "ERROR",
#                 "message": "Invalid API response structure",
#                 "change": None,
#                 "data": data,
#             }

#         # Check if data arrays are not empty
#         if not data["open"] or len(data["open"]) == 0:
#             return {
#                 "signal": "NO_DATA",
#                 "message": f"No trading data available for {index_name} on {date_str}",
#                 "change": None,
#                 "data": None,
#             }

#         # Convert to DataFrame for easier manipulation
#         df = pd.DataFrame(
#             {
#                 "open": pd.to_numeric(data["open"], errors="coerce"),
#                 "high": pd.to_numeric(data["high"], errors="coerce"),
#                 "low": pd.to_numeric(data["low"], errors="coerce"),
#                 "close": pd.to_numeric(data["close"], errors="coerce"),
#                 "volume": pd.to_numeric(data["volume"], errors="coerce"),
#                 "timestamp": data["timestamp"],
#             }
#         )

#         # Remove any rows with NaN values
#         df = df.dropna()

#         if df.empty:
#             return {
#                 "signal": "ERROR",
#                 "message": "No valid price data after cleaning",
#                 "change": None,
#                 "data": None,
#             }

#         # Calculate day's change (first open to most recent close)
#         first_candle_open = df["open"].iloc[0]
#         most_recent_close = df["close"].iloc[-1]
#         day_high = df["high"].max()
#         day_low = df["low"].min()

#         # Calculate the main change (first candle open to most recent close)
#         change_points = round(most_recent_close - first_candle_open, 2)
#         change_pct = round((change_points / first_candle_open) * 100, 2)

#         # Calculate threshold in points (percentage-based)
#         threshold_points = first_candle_open * threshold / 100

#         # Generate signal based on percentage change
#         if change_pct >= threshold:
#             signal = "BUY"
#         elif change_pct <= -threshold:
#             signal = "SELL"
#         # else:
#         #     signal = "BOTH"

#         logger.info(
#             f"{index_name} change: {change_points} points ({change_pct}%) - Signal: {signal}"
#         )

#         return {
#             "index_name": index_name,
#             "security_id": security_id,
#             "signal": signal,
#             "change_points": change_points,
#             "change_percent": change_pct,
#             "first_candle_open": first_candle_open,
#             "most_recent_close": most_recent_close,
#             "day_high": day_high,
#             "day_low": day_low,
#             "threshold_percent": threshold,
#             "threshold_points": threshold_points,
#             "date": date_str,
#             "data_points": len(df),
#             "first_timestamp": df["timestamp"].iloc[0] if len(df) > 0 else None,
#             "last_timestamp": df["timestamp"].iloc[-1] if len(df) > 0 else None,
#             "message": f"{index_name} moved {change_points:+.2f} points ({change_pct:+.2f}%) from first candle open",
#         }

#     except requests.exceptions.Timeout:
#         return {"signal": "ERROR", "message": "API request timeout", "change": None}
#     except requests.exceptions.RequestException as e:
#         return {
#             "signal": "ERROR",
#             "message": f"API request failed: {str(e)}",
#             "change": None,
#         }
#     except Exception as e:
#         logger.error(f"Unexpected error for {index_name}: {str(e)}")
#         return {
#             "signal": "ERROR",
#             "message": f"Unexpected error: {str(e)}",
#             "change": None,
#         }


def get_index_signal_dhan_api(
    security_id: str, index_name: str = "Index", threshold: float = 0.4
) -> dict:
    """
    Returns BUY, SELL, or BOTH recommendation for any index using Dhan API's intraday charts.

    Signal Logic (Percentage-based):
    - BUY: if Index moves >= +0.4% (from first candle open to most recent close)
    - SELL: if Index moves <= -0.4% (from first candle open to most recent close)
    - BOTH: if movement is between -0.4% and +0.4%

    Parameters:
    security_id: str, Dhan API security ID for the index (e.g., "13" for Nifty50)
    index_name: str, Name of the index for logging/display purposes
    threshold: float, Percentage threshold for signals (default: 0.4%)

    Returns:
    dict: Contains signal, change, and additional info
    """
    CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
    ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")

    try:
        ist = pytz.timezone(
            "Asia/Kolkata"
        )  # Fixed: Use pytz.timezone instead of datetime.timezone
        current_date = datetime.now(ist)
        date_str = current_date.strftime("%Y-%m-%d")

        logger.info(f"Fetching {index_name} data for date: {date_str}")

        # API endpoint and payload
        url = "https://api.dhan.co/v2/charts/intraday"
        headers = {
            "client-id": CLIENT_ID,
            "access-token": ACCESS_TOKEN,
            "Content-Type": "application/json",
        }
        payload = {
            "securityId": security_id,
            "exchangeSegment": "IDX_I",
            "instrument": "INDEX",
            "interval": "15",  # 15-minute intervals
            "oi": False,
            "fromDate": date_str,
            "toDate": date_str,
        }

        # Make API request
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        logger.info(
            f"API response data from nifty signal generator: {data}"
        )  # Debug log for API response
        # Validate API response
        required_fields = ["open", "high", "low", "close", "volume", "timestamp"]
        if not all(field in data for field in required_fields):
            return {
                "signal": "ERROR",
                "message": "Invalid API response structure",
                "change": None,
                "data": data,
            }

        # Check if data arrays are not empty
        if not data["open"] or len(data["open"]) == 0:
            return {
                "signal": "NO_DATA",
                "message": f"No trading data available for {index_name} on {date_str}",
                "change": None,
                "data": None,
            }

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(
            {
                "open": pd.to_numeric(data["open"], errors="coerce"),
                "high": pd.to_numeric(data["high"], errors="coerce"),
                "low": pd.to_numeric(data["low"], errors="coerce"),
                "close": pd.to_numeric(data["close"], errors="coerce"),
                "volume": pd.to_numeric(data["volume"], errors="coerce"),
                "timestamp": data["timestamp"],
            }
        )

        # Remove any rows with NaN values
        df = df.dropna()

        if df.empty:
            return {
                "signal": "ERROR",
                "message": "No valid price data after cleaning",
                "change": None,
                "data": None,
            }

        # Fetch first candle (open price of first 15-minute candle, typically 9:15 AM IST)
        first_candle_open = df["open"].iloc[0]
        first_candle_timestamp = df["timestamp"].iloc[0]

        # Fetch most recent candle (close price of last 15-minute candle, e.g., 10:00 AM IST)
        most_recent_close = df["close"].iloc[-1]
        most_recent_timestamp = df["timestamp"].iloc[-1]

        # Calculate day's change (first open to most recent close)
        change_points = round(most_recent_close - first_candle_open, 2)
        change_pct = round((change_points / first_candle_open) * 100, 2)

        # Calculate threshold in points (percentage-based)
        threshold_points = first_candle_open * threshold / 100

        # Generate signal based on percentage change
        if change_points > 80:
            signal = "BUY"
        elif change_points < -80:
            signal = "SELL"
        else:
            signal = "BOTH"

        logger.info(
            f"{index_name} change: {change_points} points ({change_pct}%) - Signal: {signal}"
        )

        return {
            "index_name": index_name,
            "security_id": security_id,
            "signal": signal,
            "change_points": change_points,
            "change_percent": change_pct,
            "first_candle_open": first_candle_open,
            "most_recent_close": most_recent_close,
            "day_high": df["high"].max(),
            "day_low": df["low"].min(),
            "threshold_percent": 0.4,
            "threshold_points": threshold_points,
            "date": date_str,
            "data_points": len(df),
            "first_timestamp": first_candle_timestamp,
            "last_timestamp": most_recent_timestamp,
            "message": f"{index_name} moved {change_points:+.2f} points ({change_pct:+.2f}%) from first candle open",
        }

    except requests.exceptions.Timeout:
        return {"signal": "ERROR", "message": "API request timeout", "change": None}
    except requests.exceptions.RequestException as e:
        return {
            "signal": "ERROR",
            "message": f"API request failed: {str(e)}",
            "change": None,
        }
    except Exception as e:
        logger.error(f"Unexpected error for {index_name}: {str(e)}")
        return {
            "signal": "ERROR",
            "message": f"Unexpected error: {str(e)}",
            "change": None,
        }


# Load the Nifty 50 data from CSV
def load_nifty50_data(csv_path: str = "csv/ind_nifty50list.csv") -> pd.DataFrame:
    """Load the Nifty 50 constituents data from CSV"""
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Warning: Nifty 50 CSV file not found at {csv_path}")
        return pd.DataFrame()


# Create a comprehensive sector mapping
def create_sector_mapping(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """Create a mapping from security_id to (sector_security_id, sector_index_name)"""
    sector_mapping = {}

    # Industry to sector index mapping
    industry_to_sector = {
        "Automobile and Auto Components": ("14", "Nifty Auto"),
        "Financial Services": ("27", "Nifty Financial Services"),
        "Information Technology": ("29", "Nifty IT"),
        "Healthcare": ("32", "Nifty Pharma"),
        "Fast Moving Consumer Goods": ("28", "Nifty FMCG"),
        "Metals & Mining": ("31", "Nifty Metal"),
        "Oil Gas & Consumable Fuels": (
            "30",
            "Nifty Energy",
        ),  # Assuming Energy index ID is 30
        "Consumer Durables": (
            "13",
            "Nifty 50",
        ),  # No specific index, default to Nifty 50
        "Services": ("13", "Nifty 50"),  # No specific index, default to Nifty 50
        "Telecommunication": ("30", "Nifty Media"),  # Media index is closest
        "Capital Goods": ("13", "Nifty 50"),  # No specific index, default to Nifty 50
        "Construction Materials": (
            "13",
            "Nifty 50",
        ),  # No specific index, default to Nifty 50
        "Power": ("13", "Nifty 50"),  # No specific index, default to Nifty 50
        "Consumer Services": (
            "13",
            "Nifty 50",
        ),  # No specific index, default to Nifty 50
        "Construction": ("13", "Nifty 50"),  # No specific index, default to Nifty 50
    }

    # Special cases for specific companies
    special_cases = {
        # Banks should map to Nifty Bank instead of Financial Services
        "AXISBANK": ("25", "Nifty Bank"),
        "HDFCBANK": ("25", "Nifty Bank"),
        "ICICIBANK": ("25", "Nifty Bank"),
        "INDUSINDBK": ("25", "Nifty Bank"),
        "KOTAKBANK": ("25", "Nifty Bank"),
        "SBIN": ("25", "Nifty Bank"),
        # Insurance companies should stay with Financial Services
        "HDFCLIFE": ("27", "Nifty Financial Services"),
        "SBILIFE": ("27", "Nifty Financial Services"),
        # Jio Financial is a special case
        "JIOFIN": ("27", "Nifty Financial Services"),
    }

    # Create mapping for each security
    for _, row in df.iterrows():
        security_id = str(row["security_id"])
        ticker = row["ticker"]
        industry = row["Industry"]

        # Check if this is a special case
        if ticker in special_cases:
            sector_mapping[security_id] = special_cases[ticker]
        elif industry in industry_to_sector:
            sector_mapping[security_id] = industry_to_sector[industry]
        else:
            # Default to Nifty 50 if industry not mapped
            sector_mapping[security_id] = ("13", "Nifty 50")

    return sector_mapping


# Global variable to store the sector mapping
SECTOR_MAPPING = {}


# Initialize the sector mapping
def init_sector_mapping(csv_path: str = "csv/ind_nifty50list.csv"):
    """Initialize the sector mapping from CSV data"""
    global SECTOR_MAPPING
    df = load_nifty50_data(csv_path)
    if not df.empty:
        SECTOR_MAPPING = create_sector_mapping(df)
    else:
        # Fallback mapping if CSV is not available
        SECTOR_MAPPING = {
            # Auto Sector
            "3456": ("14", "Nifty Auto"),  # Tata Motors
            "16669": ("14", "Nifty Auto"),  # Bajaj Auto
            "2031": ("14", "Nifty Auto"),  # Mahindra & Mahindra
            "1348": ("14", "Nifty Auto"),  # Hero MotoCorp
            "910": ("14", "Nifty Auto"),  # Eicher Motors
            "10999": ("14", "Nifty Auto"),  # Maruti Suzuki
            # Bank Sector
            "5900": ("25", "Nifty Bank"),  # Axis Bank
            "1333": ("25", "Nifty Bank"),  # HDFC Bank
            "4963": ("25", "Nifty Bank"),  # ICICI Bank
            "5258": ("25", "Nifty Bank"),  # IndusInd Bank
            "1922": ("25", "Nifty Bank"),  # Kotak Mahindra Bank
            "3045": ("25", "Nifty Bank"),  # State Bank of India
            # IT Sector
            "7229": ("29", "Nifty IT"),  # HCL Technologies
            "1594": ("29", "Nifty IT"),  # Infosys
            "11536": ("29", "Nifty IT"),  # TCS
            "13538": ("29", "Nifty IT"),  # Tech Mahindra
            "3787": ("29", "Nifty IT"),  # Wipro
            # Pharma Sector
            "694": ("32", "Nifty Pharma"),  # Cipla
            "881": ("32", "Nifty Pharma"),  # Dr. Reddy's
            "3351": ("32", "Nifty Pharma"),  # Sun Pharma
            # FMCG Sector
            "236": ("28", "Nifty FMCG"),  # Asian Paints
            "1394": ("28", "Nifty FMCG"),  # Hindustan Unilever
            "1660": ("28", "Nifty FMCG"),  # ITC
            "17963": ("28", "Nifty FMCG"),  # Nestle India
            "3432": ("28", "Nifty FMCG"),  # Tata Consumer Products
            "3506": ("28", "Nifty FMCG"),  # Titan Company
            # Metal Sector
            "25": ("31", "Nifty Metal"),  # Adani Enterprises
            "1363": ("31", "Nifty Metal"),  # Hindalco Industries
            "11723": ("31", "Nifty Metal"),  # JSW Steel
            "3499": ("31", "Nifty Metal"),  # Tata Steel
            # Financial Services (Non-Bank)
            "317": ("27", "Nifty Financial Services"),  # Bajaj Finance
            "16675": ("27", "Nifty Financial Services"),  # Bajaj Finserv
            "467": ("27", "Nifty Financial Services"),  # HDFC Life Insurance
            "18143": ("27", "Nifty Financial Services"),  # Jio Financial Services
            "21808": ("27", "Nifty Financial Services"),  # SBI Life Insurance
            "4306": ("27", "Nifty Financial Services"),  # Shriram Finance
            # Energy services
            "20374": ("30", "Nifty Energy"),  # Coal India
            "14977": ("30", "Nifty Energy"),  # Power Grid
            "11630": ("30", "Nifty Energy"),  # NTPC
            # Oil and Gas
            "2475": ("470", "Nifty Oil and Gas"),  # ONGC
            "2885": ("470", "Nifty Oil and Gas"),  # Reliance Industries
            # Others (default to Nifty 50)
            "15083": ("13", "Nifty 50"),  # Adani Ports
            "157": ("13", "Nifty 50"),  # Apollo Hospitals
            "383": ("13", "Nifty 50"),  # Bharat Electronics
            "10604": ("13", "Nifty 50"),  # Bharti Airtel
            "5097": ("13", "Nifty 50"),  # Eternal
            "1232": ("13", "Nifty 50"),  # Grasim Industries
            "11483": ("13", "Nifty 50"),  # Larsen & Toubro
            "1964": ("13", "Nifty 50"),  # Trent
            "11532": ("13", "Nifty 50"),  # UltraTech Cement
        }


# Initialize the sector mapping when the module is imported
init_sector_mapping()


def get_sector_security_id(equity_security_id: str) -> Tuple[str, str]:
    """
    Maps an equity security ID to its corresponding sector index security ID and name.

    Parameters:
    equity_security_id: str, Dhan API security ID for the equity

    Returns:
    tuple: (sector_security_id, sector_index_name)
    """
    # Convert to string to ensure consistent type
    equity_security_id = str(equity_security_id)

    # Return the sector mapping if available, otherwise default to Nifty 50
    if equity_security_id in SECTOR_MAPPING:
        return SECTOR_MAPPING[equity_security_id]
    else:
        # Default to Nifty 50 if no specific sector mapping found
        return ("13", "Nifty 50")


def print_index_signal_summary(result):
    """Helper function to print formatted signal summary for any index"""
    if result["signal"] == "ERROR" or result["signal"] == "NO_DATA":
        print(f"❌ {result['message']}")
        return

    # Signal emoji mapping
    emoji_map = {"BUY": "🟢", "SELL": "🔴", "BOTH": "🟡"}

    print(f"\n{'='*50}")
    print(f"📊 {result['index_name']} TRADING SIGNAL")
    print(f"{'='*50}")
    print(f"Date: {result['date']}")
    print(f"Signal: {emoji_map.get(result['signal'], '⚪')} {result['signal']}")
    print(
        f"Change: {result['change_points']:+.2f} points ({result['change_percent']:+.2f}%)"
    )
    print(f"First Candle Open: {result['first_candle_open']:.2f}")
    print(f"Most Recent Close: {result['most_recent_close']:.2f}")
    print(f"Day High: {result['day_high']:.2f}")
    print(f"Day Low: {result['day_low']:.2f}")
    print(
        f"Threshold: ±{result['threshold_percent']}% (±{result['threshold_points']:.2f} points)"
    )
    print(f"Data Points: {result['data_points']} intervals")
    if result.get("first_timestamp"):
        print(f"Data Period: {result['first_timestamp']} to {result['last_timestamp']}")
    print(f"{'='*50}")
    print(
        f"📝 Note: Change calculated from first candle open to most recent candle close"
    )


def get_current_time_ist():
    ist_tz = pytz.timezone("Asia/Kolkata")
    now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
    return now_utc.astimezone(ist_tz)


# # Example usage
# if __name__ == "__main__":
#     # Get Nifty 50 signal (security ID: 13)
# nifty_result = get_index_signal_dhan_api("13", "Nifty 50", 0.6)
# print_index_signal_summary(nifty_result)

#     # Get Nifty Auto signal (security ID: 21)
# auto_result = get_index_signal_dhan_api("14", "Nifty Auto", 0.6)
# print(auto_result)
# print_index_signal_summary(auto_result)

#     # Get Bank Nifty signal (security ID: 23) with custom threshold
#     bank_nifty_result = get_index_signal_dhan_api("23", "Bank Nifty", 0.8)
#     print_index_signal_summary(bank_nifty_result)


#     # Test with known equity security IDs from the Nifty 50
#     test_cases = [
#         "3456",  # Tata Motors -> Nifty Auto
#         "1333",  # HDFC Bank -> Nifty Bank
#         "1594",  # Infosys -> Nifty IT
#         "881",   # Dr. Reddy's -> Nifty Pharma
#         "1394",  # Hindustan Unilever -> Nifty FMCG
#         "3499",  # Tata Steel -> Nifty Metal
#         "317",   # Bajaj Finance -> Nifty Financial Services
#         "999999",  # Unknown -> Nifty 50 (default)
#     ]

#     print("Testing Sector Mapping:")
#     print("=" * 60)
#     for security_id in test_cases:
#         sector_id, sector_name = get_sector_security_id(security_id)
#         print(f"Equity Security ID: {security_id} -> Sector: {sector_name} (ID: {sector_id})")

#     # Display the complete mapping
#     print("\nComplete Sector Mapping:")
#     print("=" * 60)
#     for equity_id, (sector_id, sector_name) in sorted(SECTOR_MAPPING.items(), key=lambda x: x[1][1]):
#         print(f"{equity_id}: {sector_name} (ID: {sector_id})")
