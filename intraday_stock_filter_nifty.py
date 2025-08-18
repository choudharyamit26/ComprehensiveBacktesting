import pandas as pd
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from comprehensive_backtesting.data import get_security_id, init_dhan_client

warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Dhan client
dhan = init_dhan_client()


@dataclass
class FilterCriteria:
    """Configuration for stock filtering criteria"""

    min_daily_range: float = 1.5
    min_atr_pct: float = 1.5
    min_avg_volume: int = 1_000_000
    min_rel_volume: float = 0.6
    min_price: float = 50
    max_price: float = 3000
    min_momentum: float = -10.0  # Allow negative momentum for SELL signals
    required_signals: List[str] = None

    def __post_init__(self):
        if self.required_signals is None:
            self.required_signals = ["BUY", "SELL"]


@dataclass
class StockMetrics:
    """Container for stock metrics"""

    daily_range_pct: Optional[float] = None
    atr_pct: Optional[float] = None
    avg_volume: Optional[int] = None
    relative_volume: Optional[float] = None
    momentum_pct: Optional[float] = None
    current_price: Optional[float] = None
    signal: str = "HOLD"
    signal_strength: float = 0.0
    rsi: Optional[float] = None
    ma5: Optional[float] = None
    ma20: Optional[float] = None
    recommendation_score: float = 0.0
    recommendation: str = "HOLD"


class DataManager:
    """Manage data retrieval and caching"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def read_stocks_from_csv(
        self, csv_file: str = "csv/ind_nifty50list.csv"
    ) -> List[str]:
        """Read and validate stock symbols from CSV"""
        try:
            if not os.path.exists(csv_file):
                logger.error(f"CSV file '{csv_file}' not found.")
                return []

            df = pd.read_csv(csv_file)
            if "ticker" not in df.columns:
                logger.error("No 'ticker' column found in CSV.")
                return []

            symbols = df["ticker"].dropna().astype(str).tolist()
            cleaned_symbols = self._clean_symbols(symbols)
            logger.info(f"Read {len(cleaned_symbols)} stock symbols from '{csv_file}'")

            # Check for cached validated symbols
            cache_file = f"{self.cache_dir}/validated_nifty50_tickers.csv"
            if os.path.exists(cache_file):
                cached_df = pd.read_csv(cache_file)
                logger.info(f"Using cached validated tickers from '{cache_file}'")
                return cached_df["ticker"].tolist()

            # Validate symbols if no cache exists
            valid_symbols = self._validate_symbols(cleaned_symbols)
            if valid_symbols:
                pd.DataFrame(valid_symbols, columns=["ticker"]).to_csv(
                    cache_file, index=False
                )
                logger.info(f"Validated tickers saved to '{cache_file}'")

            return valid_symbols

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Error reading CSV file '{csv_file}': {e}")
            return []

    def _clean_symbols(self, symbols: List[str]) -> List[str]:
        """Clean stock symbols"""
        cleaned = []
        for symbol in symbols:
            cleaned_symbol = symbol.strip().replace(".NS", "").replace(".BO", "")
            if (
                cleaned_symbol
                and len(cleaned_symbol) <= 15
                and cleaned_symbol.replace(".", "").replace("-", "").isalnum()
            ):
                cleaned.append(cleaned_symbol)
        return cleaned

    def _validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate symbols by checking data availability"""
        valid_symbols = []
        total = len(symbols)
        logger.info(f"Validating {total} tickers...")

        for i, symbol in enumerate(symbols, 1):
            if i % 20 == 0:
                logger.info(f"Validating ticker {i}/{total}: {symbol}")
            if self._is_valid_symbol(symbol):
                valid_symbols.append(symbol)

        logger.info(f"Validated {len(valid_symbols)}/{total} tickers")
        return valid_symbols

    def _is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol has valid data"""
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=10)
            from_date_str = from_date.strftime("%Y-%m-%d")
            to_date_str = to_date.strftime("%Y-%m-%d")

            test_data = dhan.historical_daily_data(
                security_id=get_security_id(symbol),
                exchange_segment="NSE_EQ",
                instrument_type="EQUITY",
                from_date=from_date_str,
                to_date=to_date_str,
            )

            if test_data.get("status") == "success" and test_data.get("data"):
                data_dict = test_data["data"]
                # Check if data structure is correct and has minimum required fields
                required_fields = [
                    "high",
                    "low",
                    "close",
                    "volume",
                    "open",
                    "timestamp",
                ]
                has_required_fields = all(
                    field in data_dict for field in required_fields
                )

                if has_required_fields:
                    # Check if arrays have data
                    has_data = all(
                        isinstance(data_dict[field], list) and len(data_dict[field]) > 0
                        for field in required_fields
                    )
                    return has_data
            return False
        except Exception as e:
            logger.debug(f"Validation failed for {symbol}: {e}")
            return False

    def get_historical_data(
        self,
        ticker: str,
        period: str = "3mo",
        retries: int = 3,
        delay: float = 2.0,
        timeout: int = 10,
    ) -> Optional[pd.DataFrame]:
        """Get historical data with retries and caching"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        cache_file = f"{self.cache_dir}/{ticker}_{period}_{current_date}.csv"

        # Check cache first - look for today's cache file
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
                if not df.empty and len(df) >= 20:
                    logger.debug(f"Loaded cached data for {ticker} ({len(df)} records)")
                    return df
            except Exception as e:
                logger.debug(f"Failed to load cached data for {ticker}: {e}")

        # Clean up old cache files for this ticker
        self._cleanup_old_cache(ticker, period)

        # Fetch fresh data
        for attempt in range(retries):
            try:
                to_date = datetime.now()
                from_date = to_date - timedelta(days=90)  # 3 months
                from_date_str = from_date.strftime("%Y-%m-%d")
                to_date_str = to_date.strftime("%Y-%m-%d")

                logger.debug(
                    f"Fetching data for {ticker}: {from_date_str} to {to_date_str}"
                )

                data = dhan.historical_daily_data(
                    security_id=get_security_id(ticker),
                    exchange_segment="NSE_EQ",
                    instrument_type="EQUITY",
                    from_date=from_date_str,
                    to_date=to_date_str,
                )

                if data.get("status") != "success" or not data.get("data"):
                    logger.debug(
                        f"No valid data for {ticker}: Status={data.get('status')}"
                    )
                    if attempt < retries - 1:
                        time.sleep(delay)
                        continue
                    return None

                # Process the new data structure
                df = self._process_api_response(data["data"], ticker)
                if df is None or df.empty:
                    logger.debug(f"Failed to process data for {ticker}")
                    if attempt < retries - 1:
                        time.sleep(delay)
                        continue
                    return None

                # Ensure minimum data points
                if len(df) < 20:
                    logger.debug(f"Insufficient data points for {ticker}: {len(df)}")
                    if attempt < retries - 1:
                        time.sleep(delay)
                        continue
                    return None

                # Save to cache
                try:
                    df.to_csv(cache_file)
                    logger.debug(
                        f"Saved data for {ticker} to cache ({len(df)} records)"
                    )
                except Exception as e:
                    logger.debug(f"Failed to save cache for {ticker}: {e}")

                return df

            except Exception as e:
                logger.debug(
                    f"Error fetching data for {ticker} (attempt {attempt + 1}/{retries}): {str(e)[:100]}"
                )
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue

        logger.warning(f"Failed to get data for {ticker} after {retries} attempts")
        return None

    def _process_api_response(
        self, data_dict: dict, ticker: str
    ) -> Optional[pd.DataFrame]:
        """Process the API response into a DataFrame"""
        try:
            # Validate data structure
            required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(field in data_dict for field in required_fields):
                logger.debug(
                    f"Missing required fields for {ticker}. Available: {list(data_dict.keys())}"
                )
                return None

            # Check if all arrays have the same length
            lengths = [len(data_dict[field]) for field in required_fields]
            if not all(length == lengths[0] for length in lengths):
                logger.debug(f"Inconsistent array lengths for {ticker}: {lengths}")
                return None

            if lengths[0] == 0:
                logger.debug(f"Empty data arrays for {ticker}")
                return None

            # Create DataFrame from arrays
            df_data = {
                "Open": data_dict["open"],
                "High": data_dict["high"],
                "Low": data_dict["low"],
                "Close": data_dict["close"],
                "Volume": data_dict["volume"],
            }

            df = pd.DataFrame(df_data)

            # Convert timestamps to dates
            timestamps = data_dict["timestamp"]
            dates = [datetime.fromtimestamp(ts) for ts in timestamps]
            df["Date"] = dates
            df.set_index("Date", inplace=True)

            # Sort by date and remove duplicates
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            # Validate and clean data
            df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

            # Remove rows with invalid data (zeros or negative values)
            df = df[
                (df["Open"] > 0)
                & (df["High"] > 0)
                & (df["Low"] > 0)
                & (df["Close"] > 0)
                & (df["Volume"] >= 0)
            ]

            # Validate OHLC relationships
            df = df[
                (df["High"] >= df["Low"])
                & (df["High"] >= df["Open"])
                & (df["High"] >= df["Close"])
                & (df["Low"] <= df["Open"])
                & (df["Low"] <= df["Close"])
            ]

            logger.debug(f"Processed {len(df)} valid records for {ticker}")
            logger.debug(
                f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
            )

            return df

        except Exception as e:
            logger.debug(f"Error processing API response for {ticker}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _cleanup_old_cache(self, ticker: str, period: str):
        """Clean up old cache files for a ticker"""
        try:
            pattern = f"{ticker}_{period}_"
            for filename in os.listdir(self.cache_dir):
                if filename.startswith(pattern) and filename.endswith(".csv"):
                    file_path = os.path.join(self.cache_dir, filename)
                    # Remove files older than 1 day
                    if time.time() - os.path.getmtime(file_path) > 86400:  # 24 hours
                        os.remove(file_path)
                        logger.debug(f"Removed old cache file: {filename}")
        except Exception as e:
            logger.debug(f"Error cleaning up cache for {ticker}: {e}")


class TechnicalIndicators:
    """Optimized technical indicator calculations"""

    @staticmethod
    def calculate_atr_from_data(
        data: pd.DataFrame, period: int = 14
    ) -> Optional[float]:
        """Calculate Average True Range"""
        if not TechnicalIndicators._validate_data(data, ["High", "Low", "Close"]):
            return None
        try:
            high, low, close = data["High"], data["Low"], data["Close"]
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period, min_periods=period).mean()
            return atr.iloc[-1] if not atr.empty and pd.notna(atr.iloc[-1]) else None
        except Exception as e:
            logger.debug(f"Error calculating ATR: {e}")
            return None

    @staticmethod
    def calculate_daily_range(data: pd.DataFrame, periods: int = 5) -> Optional[float]:
        """Calculate average daily range percentage"""
        if not TechnicalIndicators._validate_data(data, ["High", "Low", "Close"]):
            return None
        try:
            recent_data = data.tail(periods).dropna(subset=["High", "Low", "Close"])
            if recent_data.empty:
                return None
            ranges = (
                (recent_data["High"] - recent_data["Low"]) / recent_data["Close"] * 100
            )
            result = ranges.mean()
            return result if pd.notna(result) else None
        except Exception as e:
            logger.debug(f"Error calculating daily range: {e}")
            return None

    @staticmethod
    def calculate_relative_volume(
        data: pd.DataFrame, period: int = 20
    ) -> Optional[float]:
        """Calculate relative volume"""
        if (
            not TechnicalIndicators._validate_data(data, ["Volume"])
            or len(data) < period
        ):
            return None
        try:
            recent_volume = data["Volume"].iloc[-1]
            avg_volume = data["Volume"].iloc[-period:-1].mean()
            if pd.isna(recent_volume) or pd.isna(avg_volume) or avg_volume <= 0:
                return None
            return float(recent_volume / avg_volume)
        except Exception as e:
            logger.debug(f"Error calculating relative volume: {e}")
            return None

    @staticmethod
    def calculate_momentum(data: pd.DataFrame, period: int = 5) -> Optional[float]:
        """Calculate price momentum"""
        if (
            not TechnicalIndicators._validate_data(data, ["Close"])
            or len(data) < period + 1
        ):
            return None
        try:
            current_close = data["Close"].iloc[-1]
            past_close = data["Close"].iloc[-(period + 1)]
            if pd.isna(current_close) or pd.isna(past_close) or past_close <= 0:
                return None
            return (current_close - past_close) / past_close * 100
        except Exception as e:
            logger.debug(f"Error calculating momentum: {e}")
            return None

    @staticmethod
    def calculate_moving_averages(
        data: pd.DataFrame, short: int = 5, long: int = 20
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate moving averages"""
        if not TechnicalIndicators._validate_data(data, ["Close"]) or len(data) < long:
            return None, None
        try:
            short_ma = (
                data["Close"].rolling(window=short, min_periods=short).mean().iloc[-1]
            )
            long_ma = (
                data["Close"].rolling(window=long, min_periods=long).mean().iloc[-1]
            )
            short_ma = short_ma if pd.notna(short_ma) else None
            long_ma = long_ma if pd.notna(long_ma) else None
            return short_ma, long_ma
        except Exception as e:
            logger.debug(f"Error calculating moving averages: {e}")
            return None, None

    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate RSI"""
        if (
            not TechnicalIndicators._validate_data(data, ["Close"])
            or len(data) < period + 1
        ):
            return None
        try:
            closes = data["Close"]
            delta = closes.diff()
            gain = (
                (delta.where(delta > 0, 0))
                .rolling(window=period, min_periods=period)
                .mean()
            )
            loss = (
                (-delta.where(delta < 0, 0))
                .rolling(window=period, min_periods=period)
                .mean()
            )
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            result = rsi.iloc[-1] if not rsi.empty and pd.notna(rsi.iloc[-1]) else None
            return result
        except Exception as e:
            logger.debug(f"Error calculating RSI: {e}")
            return None

    @staticmethod
    def calculate_macd(
        data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate MACD"""
        if (
            not TechnicalIndicators._validate_data(data, ["Close"])
            or len(data) < slow + signal
        ):
            return None, None, None
        try:
            closes = data["Close"]
            ema_fast = closes.ewm(span=fast, min_periods=fast).mean()
            ema_slow = closes.ewm(span=slow, min_periods=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
            histogram = macd_line - signal_line

            macd_val = (
                macd_line.iloc[-1]
                if not macd_line.empty and pd.notna(macd_line.iloc[-1])
                else None
            )
            signal_val = (
                signal_line.iloc[-1]
                if not signal_line.empty and pd.notna(signal_line.iloc[-1])
                else None
            )
            hist_val = (
                histogram.iloc[-1]
                if not histogram.empty and pd.notna(histogram.iloc[-1])
                else None
            )

            return macd_val, signal_val, hist_val
        except Exception as e:
            logger.debug(f"Error calculating MACD: {e}")
            return None, None, None

    @staticmethod
    def _validate_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate data has required columns and is not empty"""
        return (
            data is not None
            and isinstance(data, pd.DataFrame)
            and not data.empty
            and all(col in data.columns for col in required_columns)
            and not all(data[col].isna().all() for col in required_columns)
        )


class SignalGenerator:
    """Generate trading signals based on technical indicators"""

    def __init__(self):
        self.indicators = TechnicalIndicators()

    def generate_signal(self, data: pd.DataFrame) -> Tuple[str, float]:
        """Generate trading signal with strength score"""
        if data is None or data.empty:
            return "HOLD", 0.0

        signals = []
        strengths = []

        # Moving Average Signal
        self._add_ma_signal(data, signals, strengths)

        # RSI Signal
        self._add_rsi_signal(data, signals, strengths)

        # MACD Signal
        self._add_macd_signal(data, signals, strengths)

        # Momentum Signal
        self._add_momentum_signal(data, signals, strengths)

        # Volume Confirmation
        self._add_volume_confirmation(data, signals, strengths)

        return self._calculate_final_signal(signals, strengths)

    def _add_ma_signal(
        self, data: pd.DataFrame, signals: List[str], strengths: List[float]
    ):
        """Add moving average crossover signal"""
        short_ma, long_ma = self.indicators.calculate_moving_averages(data)
        if short_ma is not None and long_ma is not None and long_ma > 0:
            ma_diff_pct = ((short_ma - long_ma) / long_ma) * 100
            if short_ma > long_ma:
                signals.append("BUY")
                strengths.append(min(abs(ma_diff_pct) * 2, 10))
            else:
                signals.append("SELL")
                strengths.append(min(abs(ma_diff_pct) * 2, 10))

    def _add_rsi_signal(
        self, data: pd.DataFrame, signals: List[str], strengths: List[float]
    ):
        """Add RSI signal"""
        rsi = self.indicators.calculate_rsi(data)
        if rsi is not None:
            if rsi < 30:
                signals.append("BUY")
                strengths.append((30 - rsi) * 0.3)
            elif rsi > 70:
                signals.append("SELL")
                strengths.append((rsi - 70) * 0.3)

    def _add_macd_signal(
        self, data: pd.DataFrame, signals: List[str], strengths: List[float]
    ):
        """Add MACD signal"""
        macd_line, signal_line, histogram = self.indicators.calculate_macd(data)
        if macd_line is not None and signal_line is not None:
            macd_strength = abs(histogram) if histogram is not None else 0
            if macd_line > signal_line:
                signals.append("BUY")
                strengths.append(min(macd_strength * 50, 10))
            else:
                signals.append("SELL")
                strengths.append(min(macd_strength * 50, 10))

    def _add_momentum_signal(
        self, data: pd.DataFrame, signals: List[str], strengths: List[float]
    ):
        """Add momentum signal"""
        momentum = self.indicators.calculate_momentum(data, period=5)
        if momentum is not None:
            if momentum > 2:
                signals.append("BUY")
                strengths.append(min(momentum * 0.5, 10))
            elif momentum < -2:
                signals.append("SELL")
                strengths.append(min(abs(momentum) * 0.5, 10))

    def _add_volume_confirmation(
        self, data: pd.DataFrame, signals: List[str], strengths: List[float]
    ):
        """Add volume confirmation"""
        rel_volume = self.indicators.calculate_relative_volume(data)
        if rel_volume is not None and rel_volume > 1.5:
            buy_count = signals.count("BUY")
            sell_count = signals.count("SELL")
            volume_boost = min((rel_volume - 1.5) * 2, 5)
            if buy_count > sell_count:
                signals.append("BUY")
                strengths.append(volume_boost)
            elif sell_count > buy_count:
                signals.append("SELL")
                strengths.append(volume_boost)

    def _calculate_final_signal(
        self, signals: List[str], strengths: List[float]
    ) -> Tuple[str, float]:
        """Calculate final signal and average strength"""
        if not signals:
            return "HOLD", 0.0

        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        hold_count = signals.count("HOLD")
        avg_strength = sum(strengths) / len(strengths) if strengths else 0.0

        if buy_count > sell_count and buy_count > hold_count:
            return "BUY", round(avg_strength, 2)
        elif sell_count > buy_count and sell_count > hold_count:
            return "SELL", round(avg_strength, 2)
        else:
            return "HOLD", round(avg_strength, 2)


class RecommendationEngine:
    """Generate recommendation scores and labels"""

    @staticmethod
    def calculate_score(metrics: StockMetrics) -> float:
        """Calculate recommendation score (0-100)"""
        score = 0.0

        # Signal Strength (0-20 points)
        if metrics.signal_strength is not None:
            score += min(metrics.signal_strength * 2, 20)

        # Daily Range (0-20 points)
        if metrics.daily_range_pct is not None:
            if metrics.daily_range_pct >= 3:
                score += 20
            elif metrics.daily_range_pct >= 2:
                score += 15
            elif metrics.daily_range_pct >= 1.5:
                score += 10

        # ATR (0-15 points)
        if metrics.atr_pct is not None:
            if metrics.atr_pct >= 3:
                score += 15
            elif metrics.atr_pct >= 2:
                score += 10
            elif metrics.atr_pct >= 1.5:
                score += 5

        # Volume (0-15 points)
        if metrics.relative_volume is not None:
            if metrics.relative_volume >= 2:
                score += 15
            elif metrics.relative_volume >= 1.5:
                score += 10
            elif metrics.relative_volume >= 1:
                score += 5

        # Momentum (0-10 points)
        if metrics.momentum_pct is not None:
            momentum_score = min(abs(metrics.momentum_pct) * 0.5, 10)
            score += momentum_score

        # RSI (0-8 points)
        if metrics.rsi is not None:
            if metrics.rsi < 30 or metrics.rsi > 70:
                score += 8
            elif metrics.rsi < 40 or metrics.rsi > 60:
                score += 4

        # Price Range (0-5 points)
        if metrics.current_price is not None:
            if 100 <= metrics.current_price <= 2000:
                score += 5
            elif 50 <= metrics.current_price <= 3000:
                score += 3

        return min(round(score, 2), 100)

    @staticmethod
    def get_label(score: float, signal: str) -> str:
        """Get recommendation label"""
        if signal not in ["BUY", "SELL"]:
            return signal
        if score >= 50:
            return f"STRONG {signal}"
        elif score >= 35:
            return f"GOOD {signal}"
        elif score >= 20:
            return f"MODERATE {signal}"
        else:
            return f"WEAK {signal}"


class StockAnalyzer:
    """Analyze stocks for intraday trading suitability"""

    def __init__(self, criteria: FilterCriteria = None):
        self.criteria = criteria or FilterCriteria()
        self.indicators = TechnicalIndicators()
        self.signal_generator = SignalGenerator()
        self.recommendation_engine = RecommendationEngine()

    def analyze_stock(self, ticker: str, data: pd.DataFrame) -> Optional[Dict]:
        """Analyze a single stock"""
        if not self._validate_data_requirements(data):
            logger.debug(f"{ticker}: Failed data validation")
            return None

        try:
            metrics = self._calculate_metrics(data, ticker)
            if not self._meets_criteria(metrics):
                logger.debug(f"{ticker}: Did not meet filtering criteria")
                return None

            metrics.recommendation_score = self.recommendation_engine.calculate_score(
                metrics
            )
            metrics.recommendation = self.recommendation_engine.get_label(
                metrics.recommendation_score, metrics.signal
            )

            return self._format_output(ticker, metrics)

        except Exception as e:
            logger.debug(f"Error analyzing {ticker}: {e}")
            return None

    def _validate_data_requirements(self, data: pd.DataFrame) -> bool:
        """Validate data meets minimum requirements"""
        return (
            data is not None
            and isinstance(data, pd.DataFrame)
            and not data.empty
            and len(data) >= 20
        )

    def _calculate_metrics(self, data: pd.DataFrame, ticker: str) -> StockMetrics:
        """Calculate all metrics for a stock"""
        metrics = StockMetrics()

        # Calculate basic metrics
        metrics.daily_range_pct = self.indicators.calculate_daily_range(data)
        atr = self.indicators.calculate_atr_from_data(data)
        current_price = data["Close"].iloc[-1] if "Close" in data.columns else None

        if atr is not None and current_price is not None and current_price > 0:
            metrics.atr_pct = atr / current_price * 100

        metrics.current_price = current_price
        metrics.avg_volume = (
            int(data["Volume"].tail(20).mean()) if "Volume" in data.columns else None
        )
        metrics.relative_volume = self.indicators.calculate_relative_volume(data)
        metrics.momentum_pct = self.indicators.calculate_momentum(data)
        metrics.rsi = self.indicators.calculate_rsi(data)
        metrics.ma5, metrics.ma20 = self.indicators.calculate_moving_averages(data)
        metrics.signal, metrics.signal_strength = self.signal_generator.generate_signal(
            data
        )

        return metrics

    def _meets_criteria(self, metrics: StockMetrics) -> bool:
        """Check if metrics meet filtering criteria"""
        try:
            # Debug logging
            logger.debug(f"Checking criteria for metrics: {metrics}")

            # Basic criteria that apply to all signals
            basic_conditions = [
                metrics.daily_range_pct is not None
                and metrics.daily_range_pct >= self.criteria.min_daily_range,
                metrics.atr_pct is not None
                and metrics.atr_pct >= self.criteria.min_atr_pct,
                metrics.avg_volume is not None
                and metrics.avg_volume >= self.criteria.min_avg_volume,
                metrics.relative_volume is not None
                and metrics.relative_volume >= self.criteria.min_rel_volume,
                metrics.current_price is not None
                and self.criteria.min_price
                <= metrics.current_price
                <= self.criteria.max_price,
                metrics.signal in self.criteria.required_signals,
            ]

            # Momentum criteria - more intelligent based on signal type
            momentum_condition = True  # Default to True
            if metrics.momentum_pct is not None:
                if metrics.signal == "BUY":
                    # For BUY signals, allow slightly negative momentum but prefer positive
                    momentum_condition = metrics.momentum_pct >= -2.0
                elif metrics.signal == "SELL":
                    # For SELL signals, allow negative momentum (downward movement is good for sell)
                    momentum_condition = (
                        metrics.momentum_pct >= -15.0
                    )  # Allow strong negative momentum
                else:
                    # For HOLD signals, use the original criteria
                    momentum_condition = (
                        metrics.momentum_pct >= self.criteria.min_momentum
                    )

            all_conditions = basic_conditions + [momentum_condition]
            result = all(all_conditions)

            logger.debug(
                f"Criteria check result: {result}, conditions: {all_conditions}"
            )
            return result

        except Exception as e:
            logger.debug(f"Error in criteria check: {e}")
            return False

    def _format_output(self, ticker: str, metrics: StockMetrics) -> Dict:
        """Format metrics for output"""
        return {
            "Stock": ticker,
            "Signal": metrics.signal,
            "Recommendation": metrics.recommendation,
            "Recommendation Score": metrics.recommendation_score,
            "Current Price": (
                round(metrics.current_price, 2) if metrics.current_price else None
            ),
            "Daily Range %": (
                round(metrics.daily_range_pct, 2) if metrics.daily_range_pct else None
            ),
            "ATR %": round(metrics.atr_pct, 2) if metrics.atr_pct else None,
            "Avg Volume": metrics.avg_volume,
            "Relative Volume": (
                round(metrics.relative_volume, 2) if metrics.relative_volume else None
            ),
            "Momentum %": (
                round(metrics.momentum_pct, 2) if metrics.momentum_pct else None
            ),
            "RSI": round(metrics.rsi, 2) if metrics.rsi else None,
            "MA5": round(metrics.ma5, 2) if metrics.ma5 else None,
            "MA20": round(metrics.ma20, 2) if metrics.ma20 else None,
            "Signal Strength": metrics.signal_strength,
        }


class IntradayStockFilter:
    """Main class for intraday stock filtering"""

    def __init__(self, criteria: FilterCriteria = None, max_workers: int = 5):
        self.analyzer = StockAnalyzer(criteria)
        self.data_manager = DataManager()
        self.max_workers = max_workers
        self.signal_priority = {"BUY": 1, "SELL": 2, "HOLD": 3}

    def select_stocks(self, csv_file: str = "csv/ind_nifty50list.csv") -> List[Dict]:
        """Main method to select stocks for intraday trading"""
        logger.info("Starting stock selection for intraday trading...")
        logger.info(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Use cached validated tickers if available
        validated_csv = "cache/validated_nifty50_tickers.csv"
        if os.path.exists(validated_csv):
            csv_file = validated_csv
            logger.info(f"Using validated tickers from '{csv_file}'")

        tickers = self.data_manager.read_stocks_from_csv(csv_file)
        print(tickers)
        if not tickers:
            logger.error("No stock tickers found. Exiting...")
            return []

        logger.info(f"Analyzing {len(tickers)} stocks with parallel processing...")
        selected_stocks = self._process_stocks_parallel(tickers)

        # Sort by recommendation score and signal priority
        selected_stocks.sort(
            key=lambda x: (
                -x["Recommendation Score"],
                self.signal_priority.get(x["Signal"], 3),
            )
        )

        self._display_results(selected_stocks, len(tickers))
        self._save_results(selected_stocks)
        return selected_stocks

    def _process_stocks_parallel(self, tickers: List[str]) -> List[Dict]:
        """Process stocks in parallel"""
        selected_stocks = []
        stats = {"processed": 0, "selected": 0, "buy_signals": 0, "sell_signals": 0}

        def progress_callback(ticker: str, selected: bool, status: str):
            stats["processed"] += 1
            if selected:
                stats["selected"] += 1
                if status == "BUY":
                    stats["buy_signals"] += 1
                elif status == "SELL":
                    stats["sell_signals"] += 1

            if stats["processed"] % 10 == 0 or stats["processed"] == len(tickers):
                progress_pct = (stats["processed"] / len(tickers)) * 100
                logger.info(
                    f"Progress: {stats['processed']}/{len(tickers)} ({progress_pct:.1f}%) | "
                    f"Selected: {stats['selected']} | BUY: {stats['buy_signals']} | SELL: {stats['sell_signals']}"
                )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_ticker, ticker, progress_callback
                ): ticker
                for ticker in tickers
            }

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result:
                        selected_stocks.append(result)
                except Exception as e:
                    logger.warning(f"Error processing {ticker}: {e}")

        return selected_stocks

    def _process_single_ticker(self, ticker: str, progress_callback) -> Optional[Dict]:
        """Process a single ticker"""
        try:
            data = self.data_manager.get_historical_data(ticker)
            if data is None:
                progress_callback(ticker, False, "No data")
                return None

            result = self.analyzer.analyze_stock(ticker, data)
            if result:
                progress_callback(ticker, True, result["Signal"])
                return result
            else:
                progress_callback(ticker, False, "Criteria not met")
                return None

        except Exception as e:
            progress_callback(ticker, False, f"Error: {str(e)[:20]}")
            return None

    def _display_results(self, selected_stocks: List[Dict], total_tickers: int):
        """Display analysis results"""
        if not selected_stocks:
            print("\n" + "=" * 80)
            print("🎯 SELECTED STOCKS FOR INTRADAY TRADING")
            print("=" * 80)
            print("❌ No stocks with BUY/SELL signals meet the criteria.")
            print("\n💡 Try adjusting the filtering criteria:")
            print("- Lower the minimum daily range requirement (currently 1.5%)")
            print("- Reduce the minimum volume requirement (currently 1,000,000)")
            print("- Lower the relative volume threshold (currently 0.6)")
            print("- Adjust the price range filters (currently 50-3000)")
            print("- Check Dhan API connectivity and data availability")
            return

        print("\n" + "=" * 140)
        print("🎯 SELECTED STOCKS FOR INTRADAY TRADING (BUY/SELL SIGNALS ONLY)")
        print("=" * 140)
        print(
            f"{'No':<3} {'Stock':<10} {'Signal':<6} {'Recommendation':<16} {'Score':<5} "
            f"{'Price':<8} {'Range%':<7} {'ATR%':<6} {'Volume':<10} {'RelVol':<6} {'RSI':<5}"
        )
        print("-" * 140)

        for i, stock in enumerate(selected_stocks, 1):
            signal_color = "🟢" if stock["Signal"] == "BUY" else "🔴"
            print(
                f"{i:<3} {stock['Stock']:<10} {signal_color}{stock['Signal']:<5} "
                f"{stock['Recommendation']:<16} {stock['Recommendation Score']:<5} "
                f"₹{stock['Current Price'] or 'N/A':<7} "
                f"{stock['Daily Range %'] or 'N/A':<6} "
                f"{stock['ATR %'] or 'N/A':<5} "
                f"{stock['Avg Volume'] or 'N/A':>9} "
                f"{stock['Relative Volume'] or 'N/A':<5} "
                f"{stock['RSI'] or 'N/A':<5}"
            )

        # Calculate summary statistics
        buy_count = sum(1 for stock in selected_stocks if stock["Signal"] == "BUY")
        sell_count = sum(1 for stock in selected_stocks if stock["Signal"] == "SELL")
        strong_buy = sum(
            1 for stock in selected_stocks if "STRONG BUY" in stock["Recommendation"]
        )
        strong_sell = sum(
            1 for stock in selected_stocks if "STRONG SELL" in stock["Recommendation"]
        )
        good_buy = sum(
            1 for stock in selected_stocks if "GOOD BUY" in stock["Recommendation"]
        )
        good_sell = sum(
            1 for stock in selected_stocks if "GOOD SELL" in stock["Recommendation"]
        )

        print("\n" + "=" * 140)
        print("📈 SIGNAL SUMMARY:")
        print(
            f"🟢 BUY Signals: {buy_count} (🔥 Strong: {strong_buy}, ⭐ Good: {good_buy})"
        )
        print(
            f"🔴 SELL Signals: {sell_count} (🔥 Strong: {strong_sell}, ⭐ Good: {good_sell})"
        )
        print(f"📊 Total Selected: {len(selected_stocks)} stocks")
        print(f"🎯 Success Rate: {(len(selected_stocks)/total_tickers*100):.1f}%")

        if selected_stocks:
            print(f"\n🏆 TOP 5 RECOMMENDATIONS:")
            for i, stock in enumerate(selected_stocks[:5], 1):
                signal_emoji = "🟢" if stock["Signal"] == "BUY" else "🔴"
                print(
                    f"{i}. {stock['Stock']} - {stock['Recommendation']} "
                    f"(Score: {stock['Recommendation Score']}) {signal_emoji}"
                )
        print("=" * 140)

    def _save_results(self, selected_stocks: List[Dict]):
        """Save results to CSV"""
        if selected_stocks:
            df = pd.DataFrame(selected_stocks)
            column_order = [
                "Stock",
                "Signal",
                "Recommendation",
                "Recommendation Score",
                "Current Price",
                "Daily Range %",
                "ATR %",
                "Avg Volume",
                "Relative Volume",
                "Momentum %",
                "RSI",
                "MA5",
                "MA20",
                "Signal Strength",
            ]
            df = df[column_order]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"csv/selected_stocks_with_recommendations_{timestamp}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Results saved to '{filename}'")


def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "=" * 80)
    print("🚀 ENHANCED STOCK FILTER SCRIPT - USAGE INSTRUCTIONS")
    print("=" * 80)
    print("\n✨ FEATURES:")
    print("✅ Uses Dhan API for historical data")
    print("✅ Only BUY/SELL signals (HOLD signals filtered out)")
    print("✅ Recommendation scoring system (0-100 scale)")
    print("✅ Recommendation labels (🔥 STRONG, ⭐ GOOD, ✅ MODERATE, 🔸 WEAK)")
    print("✅ Stocks sorted by recommendation score")
    print("✅ Top 5 recommendations highlight")
    print("✅ Real-time progress tracking with counts")
    print("✅ Enhanced visual feedback with emojis")
    print("✅ Success rate calculation")
    print("✅ Robust data validation and error handling")
    print("✅ Improved caching mechanism with proper date handling")
    print("✅ Fixed API data processing for array-based responses")
    print("\n🔧 USAGE:")
    print("filter = IntradayStockFilter()")
    print("filter.select_stocks(csv_file='ind_nifty500list.csv')")
    print("\n📊 RECOMMENDATION SCORING:")
    print("- Signal Strength: Technical indicator consensus (0-20 points)")
    print("- Daily Range: Higher volatility = better for intraday (0-20 points)")
    print("- ATR: Average True Range percentage (0-15 points)")
    print("- Volume: Relative volume vs average (0-15 points)")
    print("- Momentum: Price momentum strength (0-10 points)")
    print("- RSI: Extreme values get bonus (0-8 points)")
    print("- Price Range: Bonus for optimal price range (0-5 points)")
    print("- Total Score: 0-100 (Higher = Better recommendation)")
    print("\n🏆 RECOMMENDATION LABELS:")
    print("🔥 STRONG BUY/SELL: Score ≥ 50")
    print("⭐ GOOD BUY/SELL: Score ≥ 35")
    print("✅ MODERATE BUY/SELL: Score ≥ 20")
    print("🔸 WEAK BUY/SELL: Score < 20")
    print("\n📄 CSV FILE FORMAT:")
    print("- Must contain a 'ticker' column with stock symbols")
    print("- Symbols can be with or without .NS suffix")
    print("- Example CSV content:")
    print("  ticker")
    print("  RELIANCE")
    print("  TCS")
    print("  HDFCBANK")
    print("\n🔧 DEBUGGING:")
    print("- Set logging level to DEBUG for detailed logs:")
    print("  logging.getLogger().setLevel(logging.DEBUG)")
    print("- Check cache directory for saved data")
    print("- Verify Dhan API connectivity")
    print("=" * 80)


# Additional debugging function
def debug_single_stock(ticker: str, criteria: FilterCriteria = None):
    """Debug analysis for a single stock"""
    logger.setLevel(logging.DEBUG)

    analyzer = StockAnalyzer(criteria)
    data_manager = DataManager()

    print(f"\n🔍 DEBUGGING ANALYSIS FOR {ticker}")
    print("=" * 50)

    # Get data
    data = data_manager.get_historical_data(ticker)
    if data is None:
        print(f"❌ No data available for {ticker}")
        return

    print(f"✅ Data retrieved: {len(data)} records")
    print(
        f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
    )
    print(f"💰 Current price: ₹{data['Close'].iloc[-1]:.2f}")

    # Show recent data sample
    print(f"\n📊 RECENT DATA SAMPLE (Last 5 days):")
    recent_data = data.tail(5)
    for idx, row in recent_data.iterrows():
        print(
            f"{idx.strftime('%Y-%m-%d')}: O={row['Open']:.2f}, H={row['High']:.2f}, L={row['Low']:.2f}, C={row['Close']:.2f}, V={int(row['Volume']):,}"
        )

    # Analyze
    result = analyzer.analyze_stock(ticker, data)
    if result:
        print(f"\n✅ Stock passed all criteria!")
        print(f"📊 Result: {result}")
    else:
        print(f"\n❌ Stock did not meet criteria")

        # Calculate individual metrics for debugging
        metrics = analyzer._calculate_metrics(data, ticker)
        print(f"\n📈 CALCULATED METRICS:")
        print(f"- Daily Range %: {metrics.daily_range_pct}")
        print(f"- ATR %: {metrics.atr_pct}")
        print(
            f"- Avg Volume: {metrics.avg_volume:,}"
            if metrics.avg_volume
            else "- Avg Volume: None"
        )
        print(f"- Relative Volume: {metrics.relative_volume}")
        print(f"- Momentum %: {metrics.momentum_pct}")
        print(f"- Signal: {metrics.signal}")
        print(f"- RSI: {metrics.rsi}")

        print(f"\n🎯 CRITERIA CHECK:")
        criteria = analyzer.criteria
        print(
            f"- Daily Range >= {criteria.min_daily_range}%: {metrics.daily_range_pct and metrics.daily_range_pct >= criteria.min_daily_range}"
        )
        print(
            f"- ATR >= {criteria.min_atr_pct}%: {metrics.atr_pct and metrics.atr_pct >= criteria.min_atr_pct}"
        )
        print(
            f"- Avg Volume >= {criteria.min_avg_volume:,}: {metrics.avg_volume and metrics.avg_volume >= criteria.min_avg_volume}"
        )
        print(
            f"- Relative Volume >= {criteria.min_rel_volume}: {metrics.relative_volume and metrics.relative_volume >= criteria.min_rel_volume}"
        )
        print(
            f"- Price in range {criteria.min_price}-{criteria.max_price}: {metrics.current_price and criteria.min_price <= metrics.current_price <= criteria.max_price}"
        )

        # Smart momentum check based on signal
        momentum_ok = False
        momentum_desc = "N/A"
        if metrics.momentum_pct is not None:
            if metrics.signal == "BUY":
                momentum_ok = metrics.momentum_pct >= -2.0
                momentum_desc = f">= -2.0% (BUY signal allows slight negative)"
            elif metrics.signal == "SELL":
                momentum_ok = metrics.momentum_pct >= -15.0
                momentum_desc = f">= -15.0% (SELL signal allows strong negative)"
            else:
                momentum_ok = metrics.momentum_pct >= criteria.min_momentum
                momentum_desc = f">= {criteria.min_momentum}% (HOLD signal)"

        print(f"- Momentum {momentum_desc}: {momentum_ok}")
        print(
            f"- Signal in {criteria.required_signals}: {metrics.signal in criteria.required_signals}"
        )


if __name__ == "__main__":
    print_usage_instructions()

    # Example usage
    filter = IntradayStockFilter()
    selected_stocks = filter.select_stocks(csv_file="csv/ind_nifty50list.csv")
