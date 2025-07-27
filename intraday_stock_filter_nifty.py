import yfinance as yf
import pandas as pd
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import logging

from comprehensive_backtesting.registry import STRATEGY_REGISTRY


warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class FilterCriteria:
    """Configuration for stock filtering criteria"""

    min_daily_range: float = 1.5
    min_atr_pct: float = 1.5
    min_avg_volume: int = 1_000_000
    min_rel_volume: float = 0.6
    min_price: float = 50
    max_price: float = 3000
    min_momentum: float = 0.0
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


class TechnicalIndicators:
    """Optimized technical indicator calculations with caching"""

    @staticmethod
    @lru_cache(maxsize=128)
    def calculate_atr(data_hash: int, period: int = 14) -> Optional[float]:
        """Placeholder for cached ATR calculation"""
        # Implementation note: Actual caching would require serializing DataFrame
        # This is a simplified version using calculate_atr_from_data
        pass

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
            logger.warning(f"Error calculating ATR: {e}")
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
            return ranges.mean()
        except Exception as e:
            logger.warning(f"Error calculating daily range: {e}")
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

            return float(recent_volume / avg_volume) if avg_volume > 0 else None
        except Exception as e:
            logger.warning(f"Error calculating relative volume: {e}")
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

            return (
                (current_close - past_close) / past_close * 100
                if past_close > 0
                else None
            )
        except Exception as e:
            logger.warning(f"Error calculating momentum: {e}")
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
            return short_ma, long_ma
        except Exception as e:
            logger.warning(f"Error calculating moving averages: {e}")
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
                delta.where(delta > 0, 0)
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

            return rsi.iloc[-1] if not rsi.empty and pd.notna(rsi.iloc[-1]) else None
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
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

            return (
                (
                    macd_line.iloc[-1]
                    if not macd_line.empty and pd.notna(macd_line.iloc[-1])
                    else None
                ),
                (
                    signal_line.iloc[-1]
                    if not signal_line.empty and pd.notna(signal_line.iloc[-1])
                    else None
                ),
                (
                    histogram.iloc[-1]
                    if not histogram.empty and pd.notna(histogram.iloc[-1])
                    else None
                ),
            )
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
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


class DataManager:
    """Manage data retrieval and caching"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def read_stocks_from_csv(self, csv_file: str = "ind_nifty50list.csv") -> List[str]:
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

            cache_file = f"{self.cache_dir}/validated_symbols.csv"
            if os.path.exists(cache_file):
                cached_df = pd.read_csv(cache_file)
                return cached_df["ticker"].tolist()

            valid_symbols = self._validate_symbols(cleaned_symbols)

            if valid_symbols:
                pd.DataFrame(valid_symbols, columns=["ticker"]).to_csv(
                    cache_file, index=False
                )
                logger.info(f"Validated tickers saved to cache")

            return valid_symbols

        except Exception as e:
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
                logger.info(f"Validating ticker {i}/{total}")

            if self._is_valid_symbol(symbol):
                valid_symbols.append(symbol)

        logger.info(f"Validated {len(valid_symbols)}/{len(symbols)} tickers")
        return valid_symbols

    def _is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol has valid data"""
        nse_ticker = symbol + ".NS"
        try:
            test_data = yf.download(
                nse_ticker,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=False,
                timeout=5,
            )

            if test_data.empty:
                return False

            if isinstance(test_data.columns, pd.MultiIndex):
                if nse_ticker not in [
                    col[1] for col in test_data.columns if isinstance(col, tuple)
                ]:
                    return False
                test_data = test_data.xs(
                    nse_ticker, level="Ticker", axis=1, drop_level=True
                )

            return all(
                col in test_data.columns for col in ["High", "Low", "Close", "Volume"]
            )

        except Exception:
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
        cache_file = f"{self.cache_dir}/{ticker}_{period}.csv"

        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 3600:
                try:
                    return pd.read_csv(cache_file, index_col=0, parse_dates=True)
                except Exception:
                    pass

        nse_ticker = ticker + ".NS"
        yf_ticker = yf.Ticker(nse_ticker)

        for attempt in range(retries):
            try:
                data = yf_ticker.history(
                    period=period, interval="1d", auto_adjust=False, timeout=timeout
                )

                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        if nse_ticker in [
                            col[1] for col in data.columns if isinstance(col, tuple)
                        ]:
                            data = data.xs(
                                nse_ticker, level="Ticker", axis=1, drop_level=True
                            )
                        else:
                            return None

                    required_columns = ["High", "Low", "Close", "Volume"]
                    if all(col in data.columns for col in required_columns):
                        try:
                            data.to_csv(cache_file)
                        except Exception:
                            pass
                        return data

                return None

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                logger.warning(f"Failed to get data for {ticker}: {e}")

        return None


class StockAnalyzer:
    """Analyze stocks for intraday trading suitability"""

    def __init__(self, criteria: FilterCriteria = None):
        self.criteria = criteria or FilterCriteria()
        self.indicators = TechnicalIndicators()
        self.signal_generator = SignalGenerator()
        self.recommendation_engine = RecommendationEngine()
        self.data_manager = DataManager()

    def analyze_stock(self, ticker: str, data: pd.DataFrame) -> Optional[Dict]:
        """Analyze a single stock"""
        if not self._validate_data_requirements(data):
            return None

        try:
            metrics = self._calculate_metrics(data, ticker)

            if not self._meets_criteria(metrics):
                return None

            metrics.recommendation_score = self.recommendation_engine.calculate_score(
                metrics
            )
            metrics.recommendation = self.recommendation_engine.get_label(
                metrics.recommendation_score, metrics.signal
            )

            return self._format_output(ticker, metrics)

        except Exception as e:
            logger.warning(f"Error analyzing {ticker}: {e}")
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
            return (
                metrics.daily_range_pct is not None
                and metrics.daily_range_pct >= self.criteria.min_daily_range
                and metrics.atr_pct is not None
                and metrics.atr_pct >= self.criteria.min_atr_pct
                and metrics.avg_volume is not None
                and metrics.avg_volume >= self.criteria.min_avg_volume
                and metrics.relative_volume is not None
                and metrics.relative_volume >= self.criteria.min_rel_volume
                and metrics.current_price is not None
                and self.criteria.min_price
                <= metrics.current_price
                <= self.criteria.max_price
                and metrics.momentum_pct is not None
                and metrics.momentum_pct >= self.criteria.min_momentum
                and metrics.signal in self.criteria.required_signals
            )
        except Exception:
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

    def select_stocks(self, csv_file: str = "ind_nifty50list.csv") -> List[Dict]:
        """Main method to select stocks for intraday trading"""
        logger.info("Starting stock selection for intraday trading...")
        logger.info(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        validated_csv = "validated_nifty50_tickers.csv"
        if os.path.exists(validated_csv):
            csv_file = validated_csv
            logger.info(f"Using validated tickers from '{csv_file}'")

        tickers = self.data_manager.read_stocks_from_csv(csv_file)
        if not tickers:
            logger.error("No stock tickers found. Exiting...")
            return []

        logger.info(f"Analyzing {len(tickers)} stocks with parallel processing...")

        selected_stocks = self._process_stocks_parallel(tickers)

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
            print("- Lower the minimum daily range requirement")
            print("- Reduce the minimum volume requirement")
            print("- Adjust the price range filters")
            print("- Check market conditions (trending vs sideways)")
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
            df.to_csv("selected_stocks_with_recommendations.csv", index=False)
            logger.info("Results saved to 'selected_stocks_with_recommendations.csv'")


def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "=" * 80)
    print("🚀 ENHANCED STOCK FILTER SCRIPT - USAGE INSTRUCTIONS")
    print("=" * 80)
    print("\n✨ NEW FEATURES:")
    print("✅ Only BUY/SELL signals (HOLD signals filtered out)")
    print("✅ Recommendation scoring system (0-100 scale)")
    print("✅ Recommendation labels (🔥 STRONG, ⭐ GOOD, ✅ MODERATE, 🔸 WEAK)")
    print("✅ Stocks sorted by recommendation score (best first)")
    print("✅ Top 5 recommendations highlight")
    print("✅ Enhanced signal strength calculation")
    print("✅ Real-time progress tracking with counts")
    print("✅ Enhanced visual feedback with emojis")
    print("✅ Success rate calculation")
    print("✅ Improved CSV output with recommendations")
    print("✅ Better error handling and timeouts")
    print("\n🔧 USAGE:")
    print("select_stocks_for_intraday(csv_file='ind_nifty50list.csv')")
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
    print("  ticker")
    print("  RELIANCE")
    print("  TCS")
    print("  HDFCBANK")
    print("=" * 80)


if __name__ == "__main__":
    print_usage_instructions()
    filter = IntradayStockFilter()
    filter.select_stocks(csv_file="ind_nifty50list.csv")
