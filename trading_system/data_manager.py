"""
Data management system for fetching and caching market data.
"""

import os
import asyncio
import pandas as pd
import logging
from datetime import datetime, date
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from .config import SIMULATION_MODE, COMBINED_DATA_DIR, IST
from .cache_manager import cache_manager
from .rate_limiter import rate_limiter
from live_data import get_combined_data_with_persistent_live, read_live_data_from_csv

logger = logging.getLogger("quant_trader")

# Thread pool for async operations
thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dhan_worker")

# Cache for loaded simulation dataframes keyed by security_id
SIM_DATA_CACHE: Dict[int, pd.DataFrame] = {}

# Initialize Dhan client
if not SIMULATION_MODE:
    from comprehensive_backtesting.data import init_dhan_client

    dhan = init_dhan_client()
else:

    class _MockDhan:
        def get_fund_limits(self):
            return {"data": {"availabelBalance": 1e9}}

        def quote_data(self, payload):
            # Not used in simulation; fetch_realtime_quote will read from files
            return {"status": "failure", "message": "simulation mode"}

        def get_positions(self):
            return {"status": "success", "data": []}

    dhan = _MockDhan()


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


# Initialize caches and managers
CANDLE_BUILDERS = {}


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


async def run_in_thread(func, *args, **kwargs):
    """Helper function for async file operations"""
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
            use_cache=True,
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
        batch = security_ids[i : i + batch_size]
        await rate_limiter.acquire()
        batch_results = await _fetch_quote_batch(batch)
        results.update(batch_results)

    return results


async def _fetch_quote_batch(security_ids: List[int]) -> Dict[int, Optional[Dict]]:
    """Fetch quotes for a batch of securities"""
    try:
        await asyncio.sleep(1)  # Rate limiting
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
            logger.warning(
                f"Quote API failed for {security_ids}: {response.get('remarks', 'Unknown error')}"
            )
            return {sec_id: None for sec_id in security_ids}
        else:
            logger.error(f"Unexpected quote API response: {response}")
            return {sec_id: None for sec_id in security_ids}

    except Exception as e:
        logger.error(f"Batch quote error for {security_ids}: {str(e)}")
        return {sec_id: None for sec_id in security_ids}


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
        from historical_cache import get_cache_info

        cache_stats = get_cache_info()
        logger.info(
            f"✅ Cache warmup complete: {successful_cache}/{len(security_ids)} securities cached"
        )
        logger.info(f"📊 Historical cache stats: {cache_stats}")

        return successful_cache, len(security_ids)

    except Exception as e:
        logger.error(f"Error during cache warmup: {e}")
        return 0, 0
