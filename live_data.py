from datetime import datetime, time, timedelta
import pandas as pd
import logging
import os
import pytz
import asyncio
import logging
from typing import Dict, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
import threading
import queue
from historical_cache import (
    historical_cache,
    fetch_historical_data_with_cache,
    get_cache_info,
    clear_cache_for_today,
)

logger = logging.getLogger(__name__)
from comprehensive_backtesting.data import init_dhan_client

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

dhan = init_dhan_client()


def fetch_tickers_from_csv(csv_path="csv/ind_nifty50list.csv"):
    """Fetch ticker security IDs from a CSV file."""
    if not os.path.exists(csv_path):
        logger.error(f"CSV file {csv_path} not found")
        return []
    try:
        df = pd.read_csv(csv_path)
        if "security_id" not in df.columns:
            logger.error(f"CSV file {csv_path} does not contain 'security_id' column")
            return []
        return df["security_id"].tolist()
    except Exception as e:
        logger.error(f"Failed to read tickers from {csv_path}: {e}")
        return []


def get_security_symbol_map(security_id, csv_path="csv/ind_nifty50list.csv"):
    """
    Create a mapping of security IDs to tickers from the CSV file.
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV file {csv_path} not found")
        return {}
    try:
        df = pd.read_csv(csv_path)
        symbol_map = {}
        security_id = int(security_id)
        match = df[df["security_id"] == security_id]
        if not match.empty:
            symbol_map[security_id] = match["ticker"].iloc[0]
        else:
            logger.warning(f"No ticker for security ID {security_id} in {csv_path}")
            symbol_map[security_id] = f"UNKNOWN_{security_id}"
        return symbol_map
    except Exception as e:
        logger.error(f"Failed to read symbol map from {csv_path}: {e}")
        return {}


CONFIG = {
    "CLIENT_ID": os.getenv("DHAN_CLIENT_ID"),
    "ACCESS_TOKEN": os.getenv("DHAN_ACCESS_TOKEN"),
    "TICKERS": {"NSE_EQ": fetch_tickers_from_csv()},
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


async def fetch_historical_data(tickers, exchange_segment):
    """
    Enhanced fetch_historical_data with intelligent caching.
    This replaces the original function in live_data.py
    """
    return await fetch_historical_data_with_cache(tickers, exchange_segment)


def get_historical_cache_stats():
    """Get statistics about the historical data cache."""
    return get_cache_info()


def clear_today_cache():
    """Clear cache for today's data (useful during active trading)."""
    clear_cache_for_today()


def warm_cache_for_securities(security_ids: List[int]) -> Dict:
    """
    Pre-warm the cache for a list of securities.
    This is useful to run before market open.
    """
    import asyncio

    async def _warm_cache():
        results = {}
        for security_id in security_ids:
            try:
                logger.info(f"Warming cache for security {security_id}")
                data = await get_combined_data_with_persistent_live(
                    security_id=security_id, auto_start_live_collection=False
                )
                results[security_id] = data is not None
            except Exception as e:
                logger.error(f"Failed to warm cache for {security_id}: {e}")
                results[security_id] = False
        return results

    return asyncio.run(_warm_cache())


# Enhanced startup function with cache warming:


async def startup_live_data_system_with_cache(warm_cache: bool = True):
    """
    Enhanced startup script with optional cache warming.
    """
    try:
        logger.info("🚀 Starting trading system with live data collection and caching")

        # Show initial cache state
        cache_stats = get_historical_cache_stats()
        logger.info(f"📊 Initial cache state: {cache_stats}")

        # Start live data system
        success = await initialize_live_data_from_config()

        if success:
            # Optionally warm cache for all securities
            if warm_cache:
                logger.info("🔥 Warming cache for configured securities...")
                security_ids = CONFIG["TICKERS"]["NSE_EQ"][
                    :5
                ]  # Limit to first 5 for demo
                warm_results = warm_cache_for_securities(security_ids)
                successful_warm = sum(1 for result in warm_results.values() if result)
                logger.info(
                    f"✅ Cache warmed for {successful_warm}/{len(warm_results)} securities"
                )

            status = get_live_data_status()
            final_cache_stats = get_historical_cache_stats()

            logger.info(f"✅ Live data system ready: {status}")
            logger.info(f"📊 Final cache state: {final_cache_stats}")

            import atexit

            atexit.register(stop_persistent_live_data)

            return True
        else:
            logger.error("❌ Failed to initialize live data system")
            return False

    except Exception as e:
        logger.error(f"❌ Error during live data system startup: {e}")
        return False


@dataclass
class TickData:
    """Represents a single tick of market data."""

    security_id: int
    timestamp: datetime
    last_price: float
    volume: int
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_qty: int = 0
    ask_qty: int = 0


class CandleAggregator:
    """Aggregates tick data into OHLCV candles."""

    def __init__(self, interval_minutes: int = 5):
        self.interval_minutes = interval_minutes
        self.current_candles: Dict[int, Dict] = {}  # security_id -> current candle data
        self.completed_candles: Dict[int, List[Dict]] = defaultdict(
            list
        )  # security_id -> list of completed candles
        self.tick_buffer: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )  # Buffer recent ticks
        self.lock = threading.Lock()

    def add_tick(self, tick: TickData) -> Optional[Dict]:
        """
        Add a tick and return a completed candle if interval is finished.
        """
        with self.lock:
            security_id = tick.security_id

            # Add to tick buffer
            self.tick_buffer[security_id].append(tick)

            # Get candle start time (round down to interval boundary)
            candle_start = self._get_candle_start_time(tick.timestamp)
            # Initialize or update current candle
            if security_id not in self.current_candles:
                self.current_candles[security_id] = self._create_new_candle(
                    tick, candle_start
                )
            else:
                current_candle = self.current_candles[security_id]

                # Check if we need to complete the current candle and start a new one
                if candle_start > current_candle["candle_start"]:
                    # Complete the current candle
                    completed_candle = current_candle.copy()
                    completed_candle["datetime"] = current_candle["candle_start"]
                    completed_candle["security_id"] = (
                        security_id  # Add security_id to completed candle
                    )
                    self.completed_candles[security_id].append(completed_candle)

                    # Start new candle
                    self.current_candles[security_id] = self._create_new_candle(
                        tick, candle_start
                    )

                    logger.debug(
                        f"Completed 5-min candle for {security_id}: {completed_candle}"
                    )
                    return completed_candle
                else:
                    # Update current candle
                    self._update_candle(current_candle, tick)

            return None

    def _get_candle_start_time(self, timestamp: datetime) -> datetime:
        """Get the start time of the candle for given timestamp."""
        minutes = (timestamp.minute // self.interval_minutes) * self.interval_minutes
        return timestamp.replace(minute=minutes, second=0, microsecond=0)

    def _create_new_candle(self, tick: TickData, candle_start: datetime) -> Dict:
        """Create a new candle from the first tick."""
        return {
            "candle_start": candle_start,
            "open": tick.last_price,
            "high": tick.last_price,
            "low": tick.last_price,
            "close": tick.last_price,
            "volume": tick.volume,
            "tick_count": 1,
            "vwap": tick.last_price,
            "total_value": (
                tick.last_price * tick.volume if tick.volume > 0 else tick.last_price
            ),
        }

    def _update_candle(self, candle: Dict, tick: TickData) -> None:
        """Update existing candle with new tick."""
        candle["high"] = max(candle["high"], tick.last_price)
        candle["low"] = min(candle["low"], tick.last_price)
        candle["close"] = tick.last_price
        candle["volume"] += tick.volume
        candle["tick_count"] += 1

        if tick.volume > 0:
            candle["total_value"] += tick.last_price * tick.volume
            candle["vwap"] = (
                candle["total_value"] / candle["volume"]
                if candle["volume"] > 0
                else tick.last_price
            )

    def get_completed_candles_df(self, security_id: int) -> pd.DataFrame:
        """Get completed candles as DataFrame."""
        with self.lock:
            if (
                security_id not in self.completed_candles
                or not self.completed_candles[security_id]
            ):
                return pd.DataFrame()

            candles = self.completed_candles[security_id].copy()

        df = pd.DataFrame(candles)
        if not df.empty:
            df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def get_current_candle_df(self, security_id: int) -> pd.DataFrame:
        """Get current incomplete candle as DataFrame."""
        with self.lock:
            if security_id not in self.current_candles:
                return pd.DataFrame()

            candle = self.current_candles[security_id].copy()

        candle["datetime"] = candle["candle_start"]
        df = pd.DataFrame([candle])
        df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
        df["datetime"] = pd.to_datetime(df["datetime"])

        return df


def save_candle_to_csv(completed_candle: Dict, csv_file: str):
    """
    Save a completed candle to CSV file with proper append handling.
    """
    try:
        # Create DataFrame from completed candle
        candle_df = pd.DataFrame([completed_candle])
        candle_df["datetime"] = pd.to_datetime(candle_df["datetime"])

        # Select and order columns properly
        columns_to_save = [
            "security_id",
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        candle_df = candle_df[columns_to_save]

        # Check if file exists and handle header
        file_exists = os.path.exists(csv_file)

        if file_exists:
            # Read existing data to check for duplicates
            try:
                existing_df = pd.read_csv(csv_file)
                existing_df["datetime"] = pd.to_datetime(existing_df["datetime"])

                # Check if this exact candle already exists
                duplicate_check = existing_df[
                    (existing_df["security_id"] == completed_candle["security_id"])
                    & (existing_df["datetime"] == completed_candle["datetime"])
                ]

                if not duplicate_check.empty:
                    logger.debug(
                        f"Candle already exists for security {completed_candle['security_id']} at {completed_candle['datetime']}, skipping save"
                    )
                    return

            except Exception as e:
                logger.warning(f"Could not read existing CSV for duplicate check: {e}")

        # Append to CSV
        candle_df.to_csv(csv_file, mode="a", index=False, header=not file_exists)
        logger.info(
            f"Saved candle for security {completed_candle['security_id']} at {completed_candle['datetime']} to {csv_file}"
        )

    except Exception as e:
        logger.error(f"Failed to save candle to CSV {csv_file}: {e}")


class LiveDataCollector:
    """Collects live tick data from Dhan API."""

    def __init__(
        self, dhan_client, security_ids: List[int], exchange_segment: str = "NSE_EQ"
    ):
        self.dhan = dhan_client
        self.security_ids = security_ids
        self.exchange_segment = exchange_segment
        self.aggregator = CandleAggregator(interval_minutes=5)
        self.is_running = False
        self.collection_thread = None
        self.data_queue = queue.Queue()
        self.last_quotes = {}

    async def start_collection(self):
        """Start collecting live data."""
        self.is_running = True
        logger.info(
            f"Starting live data collection for securities: {self.security_ids}"
        )

        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

    def stop_collection(self):
        """Stop collecting live data."""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Stopped live data collection")

    def _collection_loop(self):
        """Main collection loop running in separate thread."""
        import time

        ist_tz = pytz.timezone("Asia/Kolkata")
        consecutive_errors = 0
        max_consecutive_errors = 10
        # ADDED: Rate limiting configuration
        request_delay = 1.0  # Delay between requests in seconds

        logger.info("Live data collection loop started")

        while self.is_running:
            try:
                current_time = datetime.now(ist_tz)
                if not self._is_market_open(current_time):
                    if consecutive_errors == 0:
                        logger.info("Market is closed, continuing to monitor...")
                    time.sleep(60)
                    continue

                if consecutive_errors > 0:
                    logger.info("Market is open, resuming active data collection")
                    consecutive_errors = 0

                # FIX: Convert security IDs to strings before sending
                securities = {
                    self.exchange_segment: [int(sec_id) for sec_id in self.security_ids]
                }
                response = dhan.quote_data(securities)
                # logger.info(
                #     f"Fetched live data for securities: {self.security_ids}. Data: {response}"
                # )
                if response and response.get("status") == "success":
                    self._process_quotes(response, current_time)
                    consecutive_errors = 0
                else:
                    logger.warning(f"Failed to fetch quotes: {response}")
                    consecutive_errors += 1

                # ADDED: Rate limiting delay
                time.sleep(request_delay)

            except Exception as e:
                logger.error(f"Error in live data collection loop: {e}")
                consecutive_errors += 1
                import traceback

                traceback.print_exc()

    def _is_market_open(self, current_time: datetime) -> bool:
        """Check if market is currently open."""
        if current_time.weekday() >= 5:
            return False

        market_open = time(9, 15)
        market_close = time(15, 25)
        current_time_only = current_time.time()

        return market_open <= current_time_only <= market_close

    def _process_quotes(self, response: Dict, timestamp: datetime):
        """Process quote response and create tick data."""
        try:
            data = response.get("data", {}).get("data", {})
            exchange_data = data.get(self.exchange_segment, {})

            for security_id_str, quote_data in exchange_data.items():
                security_id = int(security_id_str)
                if security_id not in self.security_ids:
                    continue

                last_price = float(quote_data.get("last_price", 0))
                volume = int(quote_data.get("volume", 0))

                depth = quote_data.get("depth", {})
                buy_depth = depth.get("buy", [])
                sell_depth = depth.get("sell", [])

                bid_price = float(buy_depth[0]["price"]) if buy_depth else 0.0
                ask_price = float(sell_depth[0]["price"]) if sell_depth else 0.0
                bid_qty = int(buy_depth[0]["quantity"]) if buy_depth else 0
                ask_qty = int(sell_depth[0]["quantity"]) if sell_depth else 0

                last_quote = self.last_quotes.get(security_id, {})
                if (
                    last_quote.get("last_price") != last_price
                    or last_quote.get("volume") != volume
                ):

                    tick = TickData(
                        security_id=security_id,
                        timestamp=timestamp,
                        last_price=last_price,
                        volume=volume - last_quote.get("volume", 0),
                        bid_price=bid_price,
                        ask_price=ask_price,
                        bid_qty=bid_qty,
                        ask_qty=ask_qty,
                    )
                    completed_candle = self.aggregator.add_tick(tick)
                    if completed_candle:
                        logger.info(f"New 5-min candle completed for {security_id}")
                        # Save completed candle to CSV using the new function
                        save_candle_to_csv(completed_candle, CONFIG["LIVE_DATA_CSV"])

                    self.last_quotes[security_id] = {
                        "last_price": last_price,
                        "volume": volume,
                        "timestamp": timestamp,
                    }

        except Exception as e:
            logger.error(f"Error processing quotes: {e}")

    def get_live_candles(
        self, security_id: int, include_current: bool = True
    ) -> pd.DataFrame:
        """Get live 5-minute candles for a security."""
        completed_df = self.aggregator.get_completed_candles_df(security_id)
        if not include_current:
            return completed_df

        current_df = self.aggregator.get_current_candle_df(security_id)

        if completed_df.empty:
            return current_df
        elif current_df.empty:
            return completed_df
        else:
            combined = pd.concat([completed_df, current_df], ignore_index=True)
            return combined.sort_values("datetime").reset_index(drop=True)


async def get_combined_data_with_persistent_live(
    security_id: int,
    exchange_segment: str = "NSE_EQ",
    auto_start_live_collection: bool = True,
    use_cache: bool = True,  # New parameter to control caching
) -> Optional[pd.DataFrame]:
    """
    Get combined historical and live tick-aggregated 5-minute data using persistent collector.
    Now with enhanced historical data caching.
    """
    try:
        tickers = {exchange_segment: [security_id]}

        if use_cache:
            cache_stats = get_cache_info()
            logger.info(f"Cache stats before fetch: {cache_stats}")

        logger.info(f"Fetching historical 5-min data for security {security_id}")

        # This now uses the cached version
        hist_data_dict = await fetch_historical_data(
            tickers=tickers, exchange_segment=exchange_segment
        )

        if (
            not hist_data_dict
            or security_id not in hist_data_dict
            or hist_data_dict[security_id] is None
        ):
            logger.error(f"No historical data found for security {security_id}")
            return None

        historical_data = hist_data_dict[security_id].copy()
        logger.info(
            f"Historical data: {len(historical_data)} records from {historical_data['datetime'].min()} to {historical_data['datetime'].max()}"
        )

        # Rest of the function remains the same...
        logger.info("Initializing live data collection...")
        live_manager = await get_live_data_manager(
            dhan, [security_id], exchange_segment
        )
        if live_manager is not None:
            live_manager.add_security(security_id)

        if live_manager:
            logger.info(f"Fetching live 5-min candles for security {security_id}")
            live_candles = live_manager.get_live_candles(
                security_id, include_current=True
            )
            if not live_candles.empty:
                logger.info(
                    f"Live candles: {len(live_candles)} records from {live_candles['datetime'].min()} to {live_candles['datetime'].max()}"
                )

                combined_data = pd.concat(
                    [historical_data, live_candles], ignore_index=True
                )

                combined_data = (
                    combined_data.sort_values("datetime")
                    .drop_duplicates("datetime", keep="last")
                    .reset_index(drop=True)
                )

                # Save combined data to CSV with security_id
                try:
                    combined_data_with_id = combined_data.copy()
                    combined_data_with_id["security_id"] = security_id
                    combined_data_with_id = combined_data_with_id[
                        [
                            "security_id",
                            "datetime",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                        ]
                    ]

                    # Create unique filename for each security
                    os.makedirs("combined_data", exist_ok=True)
                    combined_csv = f"combined_data/combined_data_{security_id}.csv"
                    combined_data_with_id.to_csv(combined_csv, index=False)
                    logger.info(
                        f"Saved combined data for {security_id} to {combined_csv}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to save combined data to CSV for {security_id}: {e}"
                    )

                logger.info(f"Combined data: {len(combined_data)} total records")
                return combined_data
            else:
                logger.info(
                    "No live candles available yet, returning historical data only"
                )
        else:
            logger.info(
                "Live data manager not available, returning historical data only"
            )

        return historical_data

    except Exception as e:
        logger.error(
            f"Error in get_combined_data_with_persistent_live for security {security_id}: {e}"
        )
        return None


async def start_persistent_live_data(
    security_ids: List[int], exchange_segment: str = "NSE_EQ"
):
    """Start persistent live data collection for given securities."""
    try:
        manager = await get_live_data_manager(dhan, security_ids, exchange_segment)
        status = manager.get_status()
        logger.info(f"Persistent live data started: {status}")
        return manager
    except Exception as e:
        logger.error(f"Failed to start persistent live data: {e}")
        return None


def stop_persistent_live_data():
    """Stop persistent live data collection."""
    global _live_data_manager
    if _live_data_manager:
        _live_data_manager.shutdown()
        _live_data_manager = None
        logger.info("Persistent live data collection stopped")


def get_live_data_status() -> Dict:
    """Get status of persistent live data collection."""
    manager = get_existing_live_data_manager()
    if manager:
        return manager.get_status()
    return {"status": "not_running"}


async def initialize_live_data_from_config():
    """Initialize live data collection using your existing CONFIG."""
    try:
        security_ids = CONFIG["TICKERS"]["NSE_EQ"]
        exchange_segment = CONFIG["EXCHANGE_SEGMENT"]

        logger.info(
            f"Initializing live data for {len(security_ids)} securities from CONFIG"
        )
        manager = await start_persistent_live_data(security_ids, exchange_segment)

        if manager:
            logger.info("Live data collection initialized successfully from CONFIG")
            return True
        else:
            logger.error("Failed to initialize live data collection from CONFIG")
            return False

    except Exception as e:
        logger.error(f"Error initializing live data from CONFIG: {e}")
        return False


class PersistentLiveDataManager:
    """Manages persistent live data collection across trading sessions."""

    def __init__(
        self, dhan_client, security_ids: List[int], exchange_segment: str = "NSE_EQ"
    ):
        self.dhan = dhan_client
        self.security_ids = security_ids
        self.exchange_segment = exchange_segment
        self.collector = None
        self.is_initialized = False
        self.startup_time = None

    async def initialize(self):
        """Initialize the persistent data collection system."""
        if self.is_initialized:
            logger.warning("Live data manager already initialized")
            return

        try:
            self.collector = LiveDataCollector(
                self.dhan, self.security_ids, self.exchange_segment
            )

            await self.collector.start_collection()
            self.startup_time = datetime.now()
            self.is_initialized = True

            logger.info(
                f"Persistent live data manager initialized for {len(self.security_ids)} securities"
            )
            logger.info(f"Collection started at {self.startup_time}")

        except Exception as e:
            logger.error(f"Failed to initialize live data manager: {e}")
            raise

    def get_status(self) -> Dict:
        """Get status of the live data collection system."""
        if not self.is_initialized or not self.collector:
            return {"status": "not_initialized"}

        uptime = datetime.now() - self.startup_time if self.startup_time else None

        return {
            "status": "running" if self.collector.is_running else "stopped",
            "startup_time": self.startup_time,
            "uptime": str(uptime) if uptime else None,
            "securities_count": len(self.security_ids),
            "securities": self.security_ids,
            "exchange_segment": self.exchange_segment,
            "candles_collected": {
                sec_id: len(self.collector.aggregator.completed_candles.get(sec_id, []))
                for sec_id in self.security_ids
            },
        }

    def get_live_candles(
        self, security_id: int, include_current: bool = True
    ) -> pd.DataFrame:
        """Get live candles for a security."""
        if not self.is_initialized or not self.collector:
            logger.warning("Live data manager not initialized")
            return pd.DataFrame()

        return self.collector.get_live_candles(security_id, include_current)

    def add_security(self, security_id: int):
        """Add a new security to live collection."""
        print("------------", security_id)
        if security_id not in self.security_ids:
            self.security_ids.append(security_id)
            if self.collector:
                self.collector.security_ids = self.security_ids.copy()
            logger.info(f"Added security {security_id} to live collection")

    def remove_security(self, security_id: int):
        """Remove a security from live collection."""
        if security_id in self.security_ids:
            self.security_ids.remove(security_id)
            if self.collector:
                self.collector.security_ids = self.security_ids.copy()
            logger.info(f"Removed security {security_id} from live collection")

    def shutdown(self):
        """Gracefully shutdown the live data collection."""
        if self.collector:
            self.collector.stop_collection()
            logger.info("Live data collection shutdown completed")
        self.is_initialized = False


_live_data_manager: Optional[PersistentLiveDataManager] = None


async def get_live_data_manager(
    dhan_client, security_ids: List[int], exchange_segment: str = "NSE_EQ"
) -> PersistentLiveDataManager:
    """Get or create the global live data manager."""
    global _live_data_manager

    if _live_data_manager is None:
        _live_data_manager = PersistentLiveDataManager(
            dhan_client, security_ids, exchange_segment
        )
        await _live_data_manager.initialize()

    return _live_data_manager


def get_existing_live_data_manager() -> Optional[PersistentLiveDataManager]:
    """Get the existing live data manager without creating a new one."""
    return _live_data_manager


def read_live_data_from_csv(
    csv_file: str = None, security_id: int = None
) -> pd.DataFrame:
    """
    Read live data from CSV file with optional filtering by security_id.
    """
    if csv_file is None:
        csv_file = CONFIG["LIVE_DATA_CSV"]

    try:
        if not os.path.exists(csv_file):
            logger.warning(f"CSV file {csv_file} does not exist")
            return pd.DataFrame()

        df = pd.read_csv(csv_file)

        if df.empty:
            logger.info(f"CSV file {csv_file} is empty")
            return pd.DataFrame()

        # Convert datetime column
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Filter by security_id if provided
        if security_id is not None:
            df = df[df["security_id"] == security_id].copy()

        # Sort by datetime
        df = df.sort_values("datetime").reset_index(drop=True)

        logger.info(
            f"Read {len(df)} live data records from {csv_file}"
            + (f" for security {security_id}" if security_id else "")
        )

        return df

    except Exception as e:
        logger.error(f"Error reading live data from CSV {csv_file}: {e}")
        return pd.DataFrame()


def get_all_securities_from_live_csv(csv_file: str = None) -> List[int]:
    """
    Get list of all unique security IDs from live data CSV.
    """
    if csv_file is None:
        csv_file = CONFIG["LIVE_DATA_CSV"]

    try:
        df = read_live_data_from_csv(csv_file)
        if df.empty or "security_id" not in df.columns:
            return []

        unique_securities = df["security_id"].unique().tolist()
        logger.info(
            f"Found {len(unique_securities)} unique securities in {csv_file}: {unique_securities}"
        )
        return unique_securities

    except Exception as e:
        logger.error(f"Error getting securities from CSV {csv_file}: {e}")
        return []


async def example_persistent_usage():
    """Example of how to use the persistent live data system."""

    print("=== Initializing Live Data from CONFIG ===")
    success = await initialize_live_data_from_config()
    if success:
        print("✅ Live data collection started successfully")
    else:
        print("❌ Failed to start live data collection")
        return

    status = get_live_data_status()
    print(f"📊 Live Data Status: {status}")

    print("⏳ Waiting 30 seconds for data collection...")
    await asyncio.sleep(30)

    security_id = 11536
    print(f"\n=== Getting Combined Data for Security {security_id} ===")

    combined_data = await get_combined_data_with_persistent_live(
        security_id=security_id, exchange_segment="NSE_EQ"
    )

    if combined_data is not None:
        print(f"✅ Combined data retrieved:")
        print(f"   📈 Shape: {combined_data.shape}")
        print(
            f"   📅 Date range: {combined_data['datetime'].min()} to {combined_data['datetime'].max()}"
        )
        print(f"   🕐 Last 5 records:")
        print(combined_data.tail().to_string())
    else:
        print("❌ Failed to get combined data")

    # Test reading from CSV
    print(f"\n=== Reading Live Data from CSV ===")
    live_csv_data = read_live_data_from_csv(security_id=security_id)
    if not live_csv_data.empty:
        print(
            f"✅ Read {len(live_csv_data)} records from CSV for security {security_id}"
        )
        print(
            f"   📅 Date range: {live_csv_data['datetime'].min()} to {live_csv_data['datetime'].max()}"
        )
    else:
        print("ℹ️ No live data in CSV yet")

    print(f"\n=== Manual Control Example ===")

    manager = get_existing_live_data_manager()
    if manager:
        manager.add_security(12345)
        print("✅ Added security 12345 to live collection")

        status = manager.get_status()
        print(f"📊 Updated Status: {status}")

    print(f"\n🔄 Live data collection will continue running in background...")
    print(f"💡 Call stop_persistent_live_data() to stop when needed")


async def startup_live_data_system():
    """
    Startup script to initialize live data collection.
    """
    try:
        logger.info("🚀 Starting trading system with live data collection")

        success = await initialize_live_data_from_config()

        if success:
            status = get_live_data_status()
            logger.info(f"✅ Live data system ready: {status}")

            import atexit

            atexit.register(stop_persistent_live_data)

            return True
        else:
            logger.error("❌ Failed to initialize live data system")
            return False

    except Exception as e:
        logger.error(f"❌ Error during live data system startup: {e}")
        return False


async def main_with_persistent_live():
    """Main function demonstrating persistent live data collection."""

    print("🚀 Starting persistent live data collection system...")
    await startup_live_data_system()

    print("💼 Starting main trading logic...")

    try:
        for i in range(10):
            print(f"\n--- Trading Iteration {i+1} ---")

            combined_data = await get_combined_data_with_persistent_live(
                security_id=11536
            )

            if combined_data is not None:
                latest_price = combined_data.iloc[-1]["close"]
                latest_time = combined_data.iloc[-1]["datetime"]
                print(f"📊 Latest price: ₹{latest_price:.2f} at {latest_time}")

                # Also check CSV data
                csv_data = read_live_data_from_csv(security_id=11536)
                if not csv_data.empty:
                    csv_count = len(csv_data)
                    csv_latest = csv_data.iloc[-1]["datetime"]
                    print(
                        f"💾 CSV contains {csv_count} live candles, latest: {csv_latest}"
                    )

            else:
                print("⚠️  No data available")

            await asyncio.sleep(60)

    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")

    finally:
        print("🧹 Shutting down live data collection...")
        stop_persistent_live_data()
        print("✅ Shutdown complete")


if __name__ == "__main__":
    # asyncio.run(example_persistent_usage())
    asyncio.run(main_with_persistent_live())
