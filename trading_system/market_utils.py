"""
Market utilities for time checks, holidays, and market status.
"""

import os
import logging
from datetime import datetime, date, time, timedelta
from typing import List
from .config import IST, MARKET_OPEN_TIME, MARKET_CLOSE_TIME, TRADING_END_TIME
from .order_management import api_client

logger = logging.getLogger("quant_trader")

# Cache for market times to avoid repeated calculations
market_times_cache = {}
HOLIDAY_CACHE = {}


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


def is_high_volume_period() -> bool:
    now = datetime.now(IST).time()
    high_volume_periods = [
        (time(9, 15), time(10, 30)),
        (time(14, 30), time(15, 30)),
    ]
    return any(start <= now <= end for start, end in high_volume_periods)


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
