import os
import pickle
import pandas as pd
import hashlib
from datetime import datetime, date, timedelta, time
from typing import Dict, List, Optional, Tuple
import logging
import pytz

from comprehensive_backtesting.data import init_dhan_client

logger = logging.getLogger(__name__)
dhan = init_dhan_client()
CONFIG = {
    "CLIENT_ID": os.getenv("DHAN_CLIENT_ID"),
    "ACCESS_TOKEN": os.getenv("DHAN_ACCESS_TOKEN"),
    "EXCHANGE_SEGMENT": "NSE_EQ",
    "TIMEFRAME": 5,
    "MARKET_OPEN": time(9, 15),
    "MARKET_CLOSE": time(15, 30),
    "HISTORICAL_DATA_END": time(15, 55),
    "EXIT_BUFFER_MINUTES": 15,
    "CSV_FILE": "csv/trading_signals.csv",
    "LIVE_DATA_CSV": "csv/live_data.csv",
    "COMBINED_DATA_CSV": "combined_data.csv",
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID"),
    "DAYS_TO_FETCH": 2,
    "SIMULATE_MODE": True,
    "INITIAL_CAPITAL": 150000,
}


class HistoricalDataCache:
    """
    Persistent cache for historical data with intelligent invalidation.
    """

    def __init__(self, cache_dir: str = "cache", max_days_cache: int = 30):
        self.cache_dir = cache_dir
        self.max_days_cache = max_days_cache
        self.memory_cache = {}  # In-memory cache for current session
        self.cache_metadata = {}  # Track cache creation times and validity

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Load metadata on initialization
        self._load_cache_metadata()

        # Clean old cache files on startup
        self._cleanup_old_cache_files()

        logger.info(f"Historical data cache initialized in {cache_dir}")

    def _load_cache_metadata(self):
        """Load cache metadata from disk."""
        metadata_file = os.path.join(self.cache_dir, "cache_metadata.pkl")
        try:
            if os.path.exists(metadata_file):
                with open(metadata_file, "rb") as f:
                    self.cache_metadata = pickle.load(f)
                logger.info(
                    f"Loaded cache metadata for {len(self.cache_metadata)} entries"
                )
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            self.cache_metadata = {}

    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        metadata_file = os.path.join(self.cache_dir, "cache_metadata.pkl")
        try:
            with open(metadata_file, "wb") as f:
                pickle.dump(self.cache_metadata, f)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _cleanup_old_cache_files(self):
        """Remove cache files older than max_days_cache."""
        try:
            cutoff_date = date.today() - timedelta(days=self.max_days_cache)
            removed_count = 0

            for cache_key in list(self.cache_metadata.keys()):
                metadata = self.cache_metadata[cache_key]
                cache_date = datetime.fromisoformat(metadata["date"]).date()

                if cache_date < cutoff_date:
                    # Remove from metadata
                    del self.cache_metadata[cache_date]

                    # Remove cache file
                    cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                        removed_count += 1

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old cache files")
                self._save_cache_metadata()

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

    def _generate_cache_key(
        self, security_ids: List[int], from_date: str, to_date: str
    ) -> str:
        """Generate a unique cache key for the request."""
        # Sort security_ids to ensure consistent keys
        sorted_ids = sorted(security_ids)
        key_string = f"{sorted_ids}_{from_date}_{to_date}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _is_cache_valid(self, cache_key: str, request_date: date) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_metadata:
            return False

        metadata = self.cache_metadata[cache_key]
        cache_date = datetime.fromisoformat(metadata["date"]).date()

        # Historical data for past dates should never change, so it's always valid
        # For current/recent dates, we might want to refresh if market is still active
        ist_tz = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist_tz)

        # If the request is for a date before today, cache is always valid
        if request_date < date.today():
            return True

        # For today's data, check if it's after market hours
        market_close = time(15, 30)
        if request_date == date.today() and now.time() > market_close:
            # Market is closed, today's data is complete and valid
            return True

        # For ongoing trading day, cache is valid for 5 minutes
        cache_age = (
            now - datetime.fromisoformat(metadata["timestamp"])
        ).total_seconds()
        return cache_age < 300  # 5 minutes

    def get_cached_data(
        self, security_ids: List[int], from_date: str, to_date: str
    ) -> Optional[Dict[int, pd.DataFrame]]:
        """Retrieve cached historical data if valid."""
        cache_key = self._generate_cache_key(security_ids, from_date, to_date)
        request_date = datetime.strptime(to_date[:10], "%Y-%m-%d").date()

        # Check memory cache first
        if cache_key in self.memory_cache and self._is_cache_valid(
            cache_key, request_date
        ):
            logger.debug(f"Cache HIT (memory): {cache_key}")
            return self.memory_cache[cache_key]

        # Check disk cache
        if self._is_cache_valid(cache_key, request_date):
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            try:
                if os.path.exists(cache_file):
                    with open(cache_file, "rb") as f:
                        cached_data = pickle.load(f)

                    # Load into memory cache for faster subsequent access
                    self.memory_cache[cache_key] = cached_data

                    logger.info(
                        f"Cache HIT (disk): {cache_key} - {len(cached_data)} securities"
                    )
                    return cached_data
            except Exception as e:
                logger.error(f"Failed to load cached data: {e}")

        logger.debug(f"Cache MISS: {cache_key}")
        return None

    def cache_data(
        self,
        security_ids: List[int],
        from_date: str,
        to_date: str,
        data: Dict[int, pd.DataFrame],
    ):
        """Cache historical data to both memory and disk."""
        cache_key = self._generate_cache_key(security_ids, from_date, to_date)

        try:
            # Save to memory cache
            self.memory_cache[cache_key] = data

            # Save to disk cache
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)

            # Update metadata
            self.cache_metadata[cache_key] = {
                "security_ids": security_ids,
                "from_date": from_date,
                "to_date": to_date,
                "date": to_date[:10],  # Store date for cleanup
                "timestamp": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
                "size": len(data),
            }

            # Save metadata
            self._save_cache_metadata()

            logger.info(f"Cached historical data: {cache_key} - {len(data)} securities")

        except Exception as e:
            logger.error(f"Failed to cache data: {e}")

    def invalidate_cache(self, security_ids: List[int] = None, date_str: str = None):
        """Invalidate specific cache entries."""
        keys_to_remove = []

        for cache_key, metadata in self.cache_metadata.items():
            should_remove = False

            if security_ids and any(
                sid in metadata["security_ids"] for sid in security_ids
            ):
                should_remove = True

            if date_str and metadata["date"] == date_str:
                should_remove = True

            if should_remove:
                keys_to_remove.append(cache_key)

        for cache_key in keys_to_remove:
            # Remove from memory
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]

            # Remove from disk
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)

            # Remove from metadata
            if cache_key in self.cache_metadata:
                del self.cache_metadata[cache_key]

        if keys_to_remove:
            self._save_cache_metadata()
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        memory_size = len(self.memory_cache)
        disk_size = len(self.cache_metadata)

        # Calculate total cached securities
        total_securities = set()
        for metadata in self.cache_metadata.values():
            total_securities.update(metadata["security_ids"])

        return {
            "memory_cache_entries": memory_size,
            "disk_cache_entries": disk_size,
            "total_unique_securities": len(total_securities),
            "cache_directory": self.cache_dir,
            "max_days_retention": self.max_days_cache,
        }


# Global cache instance
historical_cache = HistoricalDataCache()


async def fetch_historical_data_with_cache(tickers, exchange_segment):
    """
    Enhanced version of fetch_historical_data with intelligent caching.
    """
    if not dhan:
        logger.error("Cannot fetch historical data: Dhan client not initialized")
        return None

    try:
        ist_tz = pytz.timezone("Asia/Kolkata")
        today = datetime.now(ist_tz)

        # Prepare date range (same logic as original)
        current_date = today - timedelta(days=1)
        trading_sessions = {sec_id: [] for sec_id in tickers[exchange_segment]}
        days_checked = 0
        max_days_to_check = 10
        NSE_HOLIDAYS_2025 = ["2025-01-26", "2025-03-14"]

        security_ids = tickers[exchange_segment]

        # Check cache first for recent data
        # Generate cache request for the last 3 trading days
        cache_from_date = (today - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
        cache_to_date = today.strftime("%Y-%m-%d %H:%M:%S")

        cached_data = historical_cache.get_cached_data(
            security_ids, cache_from_date, cache_to_date
        )

        if cached_data:
            logger.info(
                f"Using cached historical data for {len(cached_data)} securities"
            )
            # Verify cached data meets the 2-session requirement
            valid_cached_data = {}
            for sec_id, df in cached_data.items():
                if (
                    df is not None and len(df) >= 100
                ):  # Reasonable minimum for 2 sessions
                    valid_cached_data[sec_id] = df

            if len(valid_cached_data) == len(security_ids):
                return valid_cached_data
            else:
                logger.info(f"Cached data incomplete, fetching fresh data")

        # Original fetching logic (unchanged)
        while (
            any(len(sessions) < 2 for sessions in trading_sessions.values())
            and days_checked < max_days_to_check
        ):
            if (
                current_date.weekday() >= 5
                or current_date.strftime("%Y-%m-%d") in NSE_HOLIDAYS_2025
            ):
                logger.info(f"Skipping non-trading day {current_date.date()}")
                current_date -= timedelta(days=1)
                days_checked += 1
                continue

            from_date = datetime.combine(
                current_date.date(), CONFIG["MARKET_OPEN"]
            ).replace(tzinfo=ist_tz)
            to_date = datetime.combine(
                current_date.date(), CONFIG["HISTORICAL_DATA_END"]
            ).replace(tzinfo=ist_tz)
            from_date_str = from_date.strftime("%Y-%m-%d %H:%M:%S")
            to_date_str = to_date.strftime("%Y-%m-%d %H:%M:%S")

            for security_id in tickers[exchange_segment]:
                if len(trading_sessions[security_id]) >= 2:
                    continue

                logger.info(
                    f"Fetching historical data for {security_id} from {from_date_str} to {to_date_str}"
                )

                import time

                time.sleep(1)  # Rate limiting

                data = dhan.intraday_minute_data(
                    security_id=security_id,
                    exchange_segment=exchange_segment,
                    instrument_type="EQUITY",
                    interval=CONFIG["TIMEFRAME"],
                    from_date=from_date_str,
                    to_date=to_date_str,
                )

                if (
                    data
                    and data.get("status") == "success"
                    and "data" in data
                    and data["data"]
                ):
                    df_chunk = pd.DataFrame(data["data"])
                    required_fields = [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "timestamp",
                    ]
                    missing_fields = [
                        field
                        for field in required_fields
                        if field not in df_chunk.columns
                    ]

                    if missing_fields:
                        logger.warning(
                            f"Missing fields {missing_fields} for {security_id}"
                        )
                        continue

                    df_chunk["datetime"] = pd.to_datetime(
                        df_chunk["timestamp"], unit="s", utc=True, errors="coerce"
                    )
                    df_chunk["datetime"] = df_chunk["datetime"].dt.tz_convert(ist_tz)
                    df_chunk = df_chunk.dropna(subset=["datetime"])

                    valid_date_range = (df_chunk["datetime"] >= from_date) & (
                        df_chunk["datetime"] <= to_date
                    )
                    df_chunk = df_chunk[valid_date_range]

                    if len(df_chunk) < 50:
                        logger.info(
                            f"Insufficient data ({len(df_chunk)} rows) for {security_id}"
                        )
                        continue

                    df_chunk = df_chunk[
                        ["datetime", "open", "high", "low", "close", "volume"]
                    ]
                    trading_sessions[security_id].append(df_chunk)
                    logger.info(f"Fetched {len(df_chunk)} rows for {security_id}")

                else:
                    logger.warning(f"No data for {security_id} from {from_date_str}")

            current_date -= timedelta(days=1)
            days_checked += 1

        # Process and combine data
        result = {}
        for security_id in tickers[exchange_segment]:
            sessions = trading_sessions[security_id]
            if len(sessions) >= 2:
                df = pd.concat(sessions, ignore_index=True)
                df = df.drop_duplicates(subset=["datetime"], keep="last")
                df = df.sort_values("datetime").reset_index(drop=True)
                logger.info(
                    f"Combined {len(df)} rows for {security_id} across {len(sessions)} sessions"
                )
                result[security_id] = df
            else:
                logger.error(f"Failed to fetch 2 sessions for {security_id}")
                result[security_id] = None

        # Cache the successful results
        if result and any(df is not None for df in result.values()):
            # Cache with current timestamp range
            actual_from_date = min(
                df["datetime"].min() for df in result.values() if df is not None
            ).strftime("%Y-%m-%d %H:%M:%S")
            actual_to_date = max(
                df["datetime"].max() for df in result.values() if df is not None
            ).strftime("%Y-%m-%d %H:%M:%S")

            historical_cache.cache_data(
                security_ids, actual_from_date, actual_to_date, result
            )

        return result

    except Exception as e:
        logger.error(f"Error fetching historical data with cache: {e}")
        raise


def get_cache_info():
    """Get information about the current cache state."""
    return historical_cache.get_cache_stats()


def clear_cache_for_today():
    """Clear cache entries for today (useful for development/testing)."""
    today_str = date.today().strftime("%Y-%m-%d")
    historical_cache.invalidate_cache(date_str=today_str)
    logger.info(f"Cleared cache for {today_str}")


def clear_all_cache():
    """Clear all cached data."""
    import shutil

    cache_dir = historical_cache.cache_dir

    # Clear memory cache
    historical_cache.memory_cache.clear()
    historical_cache.cache_metadata.clear()

    # Remove cache directory
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    logger.info("Cleared all cache data")


# Example usage and testing functions
async def test_cache_performance():
    """Test the performance improvement from caching."""
    import time

    # Mock tickers for testing
    test_tickers = {"NSE_EQ": [11536, 1333]}  # Replace with actual security IDs

    print("=== Testing Historical Data Caching Performance ===")

    # First fetch (should hit API)
    start_time = time.time()
    print("First fetch (no cache)...")
    data1 = await fetch_historical_data_with_cache(test_tickers, "NSE_EQ")
    first_fetch_time = time.time() - start_time
    print(f"First fetch took: {first_fetch_time:.2f} seconds")

    # Second fetch (should hit cache)
    start_time = time.time()
    print("Second fetch (cached)...")
    data2 = await fetch_historical_data_with_cache(test_tickers, "NSE_EQ")
    second_fetch_time = time.time() - start_time
    print(f"Second fetch took: {second_fetch_time:.2f} seconds")

    # Performance improvement
    if second_fetch_time > 0:
        improvement = (first_fetch_time - second_fetch_time) / first_fetch_time * 100
        print(f"Performance improvement: {improvement:.1f}%")

    # Cache stats
    stats = get_cache_info()
    print(f"Cache stats: {stats}")

    return data1, data2
