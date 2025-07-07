import json
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List, Optional
import warnings
import logging
import matplotlib.pyplot as plt
from io import BytesIO
import concurrent.futures
from scipy.stats import spearmanr

from comprehensive_backtesting.utils import run_backtest
from .data import get_data_sync
from .reports import PerformanceAnalyzer
from .parameter_optimization import optimize_strategy

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationAnalyzer:
    """Comprehensive validation analysis for trading strategies with benchmark comparison."""

    def __init__(
        self,
        strategy_name: str,
        ticker: str,
        initial_cash: float = 100000.0,
        commission: float = 0.05,
        analyzers: Optional[List] = None,
        benchmark: str = "^NSEI",  # Default to Nifty 50
    ):
        """Initialize the validation analyzer with benchmark.

        Args:
            strategy_name (str): Name of the registered strategy
            ticker (str): Stock ticker symbol.
            initial_cash (float): Initial portfolio cash.
            commission (float): Broker commission rate.
            benchmark (str): Benchmark ticker
            analyzers (List, optional): List of analyzers to use.
        """
        from .registry import get_strategy  # Import here to avoid circular dependencies

        # Get strategy class from registry
        self.strategy_class = get_strategy(strategy_name)
        if not self.strategy_class:
            raise ValueError(f"Strategy '{strategy_name}' not found in registry")

        self.strategy_name = strategy_name
        self.ticker = ticker
        self.benchmark = benchmark
        self.initial_cash = initial_cash
        self.commission = commission
        self.analyzers = analyzers if analyzers is not None else []
        self.default_params = self._get_default_strategy_params()
        self.max_param_values = {  # Define maximum parameter values for optimization
            "fast_ema_period": 20,
            "slow_ema_period": 50,
            "rsi_period": 20,
            "rsi_upper": 75,
            "rsi_lower": 25,
        }
        logger.info(
            f"Initialized ValidationAnalyzer for {ticker} with strategy {strategy_name} (Benchmark: {benchmark})"
        )

    def _get_default_strategy_params(self) -> Dict:
        """Get default parameters from strategy class if available"""
        if hasattr(self.strategy_class, "get_default_params"):
            return self.strategy_class.get_default_params()
        return {}

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def run_validation_analysis(
        self,
        start_date: str,
        end_date: str,
        interval: str,
        n_trials: int,
        analysis_type: str = "walkforward",
        in_sample_days: int = 60,
        out_sample_days: int = 20,
        step_days: int = 20,
        min_trades: int = 1,
        split_ratio: float = 0.7,
        optimize_in_sample: bool = True,
        gap_days: int = 1,
    ) -> Dict:
        """Run validation analysis with benchmark comparison."""
        # Validate date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        total_days = (end_dt - start_dt).days
        logger.info(f"Total date range: {start_date} to {end_date} ({total_days} days)")

        # Fetch initial data to verify availability
        data = get_data_sync(self.ticker, start_date, end_date, interval=interval)
        if data.empty:
            logger.error(
                f"No data available for {self.ticker} from {start_date} to {end_date}"
            )
            raise ValueError(
                f"No data available for {self.ticker} from {start_date} to {end_date}"
            )
        logger.info(
            f"Loaded {len(data)} bars for {self.ticker} from {start_date} to {end_date}"
        )

        max_required = self.strategy_class.get_min_data_points(self.max_param_values)
        if len(data) < max_required:
            logger.error(
                f"Insufficient data: {len(data)} bars < {max_required} required"
            )
            raise ValueError(
                f"Insufficient data: {len(data)} bars < {max_required} required"
            )

        logger.info(
            f"Running {analysis_type} analysis for {self.ticker} from {start_date} to {end_date}"
        )
        print("=" * 60)
        print(
            f"{analysis_type.upper()} VALIDATION ANALYSIS (BENCHMARK: {self.benchmark})"
        )
        print("=" * 60)

        try:
            if analysis_type.lower() == "walkforward":
                min_required_days = self.strategy_class.get_min_data_points(
                    self.max_param_values
                )
                if in_sample_days < min_required_days:
                    logger.warning(
                        f"In-sample period {in_sample_days} days too short for max params, adjusting to {min_required_days}"
                    )
                    in_sample_days = min_required_days
                results = self.walk_forward_analysis(
                    start_date=start_date,
                    end_date=end_date,
                    in_sample_days=in_sample_days,
                    out_sample_days=out_sample_days,
                    step_days=step_days,
                    n_trials=n_trials,
                    min_trades=min_trades,
                    interval=interval,
                    gap_days=gap_days,
                )
            elif analysis_type.lower() == "inout":
                results = self.in_sample_out_sample_analysis(
                    start_date=start_date,
                    end_date=end_date,
                    split_ratio=split_ratio,
                    optimize_in_sample=optimize_in_sample,
                    n_trials=n_trials,
                    interval=interval,
                    gap_days=gap_days,
                )
            else:
                raise ValueError(
                    f"Invalid analysis_type: {analysis_type}. Choose 'walkforward' or 'inout'."
                )
            return results
        except Exception as e:
            logger.error(f"Validation analysis failed: {str(e)}")
            raise

    def _generate_day_based_windows(
        self,
        start_date: str,
        end_date: str,
        in_sample_days: int,
        out_sample_days: int,
        step_days: int,
        gap_days: int = 1,
    ) -> List[Dict]:
        """Generate walk-forward windows with dynamic date range extension for insufficient data."""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        windows = []
        window_id = 1
        min_required = self.strategy_class.get_min_data_points(self.max_param_values)

        # Calculate total days needed for one complete window
        total_days_needed = in_sample_days + gap_days + out_sample_days
        available_days = (end_dt - start_dt).days

        # Check if sufficient data is available
        if available_days < total_days_needed:
            logger.warning(
                f"Insufficient total data: {available_days} days < {total_days_needed} required"
            )
            return windows

        # Adjust in-sample days if too short
        trading_days_estimate = in_sample_days * 5 // 7
        if trading_days_estimate < min_required:
            new_in_sample_days = int(min_required * 7 / 5) + 10
            logger.warning(
                f"Initial in-sample days ({in_sample_days}) too short, adjusting to {new_in_sample_days}"
            )
            in_sample_days = new_in_sample_days

        current_start = start_dt

        while True:
            in_sample_end = current_start + timedelta(days=in_sample_days)
            out_sample_start = in_sample_end + timedelta(days=gap_days)
            out_sample_end = out_sample_start + timedelta(days=out_sample_days)

            if out_sample_end > end_dt:
                break

            # Fetch in-sample data
            in_sample_data = get_data_sync(
                self.ticker,
                current_start.strftime("%Y-%m-%d"),
                in_sample_end.strftime("%Y-%m-%d"),
                interval="1d",
            )

            if in_sample_data.empty:
                logger.warning(f"Window {window_id}: No in-sample data available")
                current_start = current_start + timedelta(days=step_days)
                continue

            # Check if sufficient data points
            if len(in_sample_data) < min_required:
                logger.warning(
                    f"Window {window_id}: Insufficient in-sample data ({len(in_sample_data)} < {min_required})"
                )
                current_start = current_start + timedelta(days=step_days)
                continue

            windows.append(
                {
                    "window_id": window_id,
                    "in_sample_start": current_start.strftime("%Y-%m-%d"),
                    "in_sample_end": in_sample_end.strftime("%Y-%m-%d"),
                    "out_sample_start": out_sample_start.strftime("%Y-%m-%d"),
                    "out_sample_end": out_sample_end.strftime("%Y-%m-%d"),
                }
            )

            window_id += 1
            current_start = current_start + timedelta(days=step_days)

            if window_id > 1000:
                logger.warning("Generated maximum number of windows (1000)")
                break

        print(f"Generated {len(windows)} windows with parameters:")
        print(f"  In-sample: {in_sample_days} days")
        print(f"  Out-sample: {out_sample_days} days")
        print(f"  Gap: {gap_days} days")
        print(f"  Step: {step_days} days")

        return windows

    def in_sample_out_sample_analysis(
        self,
        start_date: str,
        end_date: str,
        interval: str,
        n_trials: int,
        split_ratio: float = 0.7,
        optimize_in_sample: bool = True,
        gap_days: int = 1,
    ) -> Dict:
        """Perform in-sample and out-of-sample analysis with benchmark."""
        logger.info(f"Starting in-sample/out-of-sample analysis for {self.ticker}")
        print("=" * 60)
        print(f"IN-SAMPLE / OUT-OF-SAMPLE ANALYSIS (BENCHMARK: {self.benchmark})")
        print("=" * 60)

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        total_days = (end_dt - start_dt).days
        min_required_days = self.strategy_class.get_min_data_points(
            self.max_param_values
        )
        if total_days < min_required_days:
            logger.error(
                f"Insufficient data period: {total_days} days < {min_required_days} required"
            )
            raise ValueError(
                f"Insufficient data period: {total_days} days < {min_required_days} required"
            )

        split_days = int(total_days * split_ratio)
        split_date = start_dt + timedelta(days=split_days)
        gap_start_date = split_date + timedelta(days=gap_days)
        split_date_str = split_date.strftime("%Y-%m-%d")
        gap_start_str = gap_start_date.strftime("%Y-%m-%d")

        print(f"Data Period: {start_date} to {end_date}")
        print(f"In-Sample Period: {start_date} to {split_date_str}")
        print(
            f"Out-of-Sample Period: {gap_start_str} to {end_date} (with {gap_days} day gap)"
        )
        print(
            f"Split Ratio: {split_ratio:.1%} in-sample, {1-split_ratio:.1%} out-of-sample"
        )

        results = {
            "split_date": split_date_str,
            "gap_start": gap_start_str,
            "in_sample_period": (start_date, split_date_str),
            "out_sample_period": (gap_start_str, end_date),
            "split_ratio": split_ratio,
            "gap_days": gap_days,
        }

        try:
            # Get benchmark returns
            benchmark_in_return = self._get_benchmark_return(start_date, split_date_str)
            benchmark_out_return = self._get_benchmark_return(gap_start_str, end_date)

            results["benchmark_returns"] = {
                "in_sample": benchmark_in_return,
                "out_sample": benchmark_out_return,
            }

            if optimize_in_sample:
                logger.info(
                    f"Optimizing parameters on in-sample data ({n_trials} trials)"
                )
                optimization_results = optimize_strategy(
                    strategy_class=self.strategy_name,
                    ticker=self.ticker,
                    start_date=start_date,
                    end_date=split_date_str,
                    n_trials=n_trials,
                    initial_cash=self.initial_cash,
                    commission=self.commission,
                    interval=interval,
                )
                best_params = optimization_results["best_params"]
                print(f"Best parameters found: {best_params}")
                results["optimized_params"] = best_params
                results["optimization_results"] = optimization_results
            else:
                logger.info("Using default parameters")
                best_params = self.default_params
                results["optimized_params"] = None

            print("\nTesting on in-sample data...")
            in_sample_results = self._run_backtest(
                start_date, split_date_str, interval=interval, **best_params
            )
            print("Testing on out-of-sample data...")
            out_sample_results = self._run_backtest(
                gap_start_str, end_date, interval=interval, **best_params
            )

            in_sample_analyzer = PerformanceAnalyzer(in_sample_results)
            out_sample_analyzer = PerformanceAnalyzer(out_sample_results)

            in_sample_report = in_sample_analyzer.generate_full_report()
            out_sample_report = out_sample_analyzer.generate_full_report()

            # Add benchmark alpha
            in_sample_report["summary"]["alpha"] = (
                in_sample_report["summary"].get("total_return_pct", 0)
                - benchmark_in_return
            )
            out_sample_report["summary"]["alpha"] = (
                out_sample_report["summary"].get("total_return_pct", 0)
                - benchmark_out_return
            )

            results["in_sample_performance"] = in_sample_report
            results["out_sample_performance"] = out_sample_report
            results["performance_degradation"] = self._calculate_degradation(
                in_sample_report, out_sample_report
            )

            # Add stability metrics
            results["parameter_stability"] = self._assess_parameter_stability(
                in_sample_report, out_sample_report
            )

            self._print_in_out_comparison(results)
            return results

        except Exception as e:
            logger.error(
                f"In-sample/out-of-sample analysis failed: {str(e)}. Check data or strategy compatibility."
            )
            raise

    def _get_benchmark_return(self, start_date: str, end_date: str) -> float:
        """Calculate benchmark return for a given period"""
        try:
            benchmark_data = get_data_sync(
                self.benchmark,
                start_date,
                end_date,
                interval="1d",  # Daily data for benchmark
            )
            if benchmark_data.empty:
                logger.warning(
                    f"No benchmark data found for {self.benchmark} from {start_date} to {end_date}"
                )
                return 0.0

            start_price = benchmark_data.iloc[0]["close"]
            end_price = benchmark_data.iloc[-1]["close"]
            return (end_price - start_price) / start_price * 100
        except Exception as e:
            logger.error(f"Error calculating benchmark return: {str(e)}")
            return 0.0

    def _assess_parameter_stability(
        self, in_sample_perf: Dict, out_sample_perf: Dict
    ) -> Dict:
        """Calculate stability metrics between in-sample and out-of-sample results"""
        stability = {}

        # Key metrics to compare
        metrics = [
            "total_return_pct",
            "annual_return_pct",
            "max_drawdown_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "win_rate_percent",
            "profit_factor",
            "alpha",  # Added alpha to stability metrics
        ]

        for metric in metrics:
            in_val = in_sample_perf.get("summary", {}).get(
                metric
            ) or in_sample_perf.get("trade_analysis", {}).get(metric)
            out_val = out_sample_perf.get("summary", {}).get(
                metric
            ) or out_sample_perf.get("trade_analysis", {}).get(metric)

            if in_val is None or out_val is None:
                stability[metric] = "N/A"
                continue

            try:
                # Calculate percentage change
                pct_change = (
                    (out_val - in_val) / abs(in_val) * 100
                    if in_val != 0
                    else float("inf")
                )
                stability[f"{metric}_change"] = pct_change

                # Stability rating
                if abs(pct_change) < 20:
                    stability[f"{metric}_stability"] = "High"
                elif abs(pct_change) < 50:
                    stability[f"{metric}_stability"] = "Medium"
                else:
                    stability[f"{metric}_stability"] = "Low"
            except TypeError:
                stability[metric] = "N/A"

        return stability

    def walk_forward_analysis(
        self,
        start_date: str,
        end_date: str,
        interval: str,
        n_trials: int,
        in_sample_days: int = 30,
        out_sample_days: int = 15,
        step_days: int = 15,
        min_trades: int = 1,
        gap_days: int = 1,
        parallel: bool = True,
    ) -> Dict:
        """Robust walk-forward validation with benchmark comparison."""

        logger.info(f"Starting walk-forward analysis for {self.ticker}")
        print("=" * 60)
        print(f"WALK-FORWARD VALIDATION WITH BENCHMARK {self.benchmark}")
        print("=" * 60)
        print(f"Parameters:")
        print(f"  Strategy: {self.strategy_name}")
        print(f"  In-sample period: {in_sample_days} days")
        print(f"  Out-of-sample period: {out_sample_days} days")
        print(f"  Gap period: {gap_days} days")
        print(f"  Step size: {step_days} days")
        print(f"  Optimization trials per window: {n_trials}")
        print(f"  Parallel processing: {parallel}")

        # Validate date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        total_days = (end_dt - start_dt).days
        min_required_days = self.strategy_class.get_min_data_points(
            self.max_param_values
        )
        print(f"  Date range: {start_date} to {end_date} ({total_days} days)")
        print(f"  Minimum required: {min_required_days} days")

        if total_days < min_required_days:
            raise ValueError(
                f"Insufficient data period for walk-forward analysis. "
                f"Need at least {min_required_days} days, but only have {total_days} days. "
                f"Consider reducing parameter ranges or extending the date range."
            )

        # Generate day-based windows with gap
        print(f"\nGenerating walk-forward windows...")
        windows = self._generate_day_based_windows(
            start_date, end_date, in_sample_days, out_sample_days, step_days, gap_days
        )

        if not windows:
            print("ERROR: No windows could be generated!")
            print("This could be due to:")
            print("- Date range too short for the specified window parameters")
            print("- Insufficient trading days in the specified period")
            print("- Window parameters are too large relative to available data")
            print("\nRecommendations:")
            print("- Extend the date range (start_date to end_date)")
            print("- Reduce in_sample_days and/or out_sample_days")
            print("- Reduce step_days to create more overlapping windows")
            print("- Check if the date range includes sufficient trading days")

            return {
                "parameters": {
                    "strategy": self.strategy_name,
                    "in_sample_days": in_sample_days,
                    "out_sample_days": out_sample_days,
                    "gap_days": gap_days,
                    "step_days": step_days,
                    "n_trials": n_trials,
                    "min_trades": min_trades,
                },
                "windows": [],
                "summary_stats": {},
                "error": "No windows generated",
            }

        print(f"Successfully generated {len(windows)} walk-forward windows")

        # Display first few windows for verification
        print(f"\nFirst few windows:")
        for i, window in enumerate(windows[:3]):
            print(
                f"  Window {i+1}: {window['in_sample_start']} to {window['in_sample_end']} "
                f"-> {window['out_sample_start']} to {window['out_sample_end']}"
            )
        if len(windows) > 3:
            print(f"  ... and {len(windows) - 3} more windows")

        results = {
            "parameters": {
                "strategy": self.strategy_name,
                "in_sample_days": in_sample_days,
                "out_sample_days": out_sample_days,
                "gap_days": gap_days,
                "step_days": step_days,
                "n_trials": n_trials,
                "min_trades": min_trades,
            },
            "windows": [],
            "summary_stats": {},
        }

        # Process windows
        print(f"\nProcessing {len(windows)} windows sequentially...")
        for i, window in enumerate(windows):
            try:
                print(
                    f"Processing window {i+1}/{len(windows)}: "
                    f"{window['in_sample_start']} to {window['out_sample_end']}"
                )
                window_result = self._process_walk_forward_window(
                    window, i, n_trials, min_trades, interval, len(windows)
                )
                results["windows"].append(window_result)
            except Exception as e:
                logger.error(f"Error processing window {i+1}: {e}")
                # Add failed window info
                results["windows"].append(
                    {
                        "window_id": window.get("window_id", i + 1),
                        "valid": False,
                        "error": str(e),
                        "periods": window,
                    }
                )

        # Calculate summary statistics
        valid_windows = [w for w in results["windows"] if w.get("valid", False)]
        processed_windows = len(results["windows"])

        print(f"\nWindow Processing Summary:")
        print(f"  Total windows: {len(windows)}")
        print(f"  Processed windows: {processed_windows}")
        print(f"  Valid windows: {len(valid_windows)}")
        print(f"  Failed windows: {processed_windows - len(valid_windows)}")

        if valid_windows:
            all_in_sample_returns = [
                w["in_sample_performance"]["summary"].get("total_return_pct", 0)
                for w in valid_windows
            ]
            all_out_sample_returns = [
                w["out_sample_performance"]["summary"].get("total_return_pct", 0)
                for w in valid_windows
            ]
            all_alphas = [
                w["out_sample_performance"]["summary"].get("alpha", 0)
                for w in valid_windows
            ]

            # Calculate correlation coefficient
            if len(all_out_sample_returns) > 1:
                correlation, p_value = spearmanr(
                    all_in_sample_returns, all_out_sample_returns
                )
            else:
                correlation, p_value = 0, 1.0

            # Calculate benchmark stats
            benchmark_returns = [
                self._get_benchmark_return(
                    w["periods"]["out_sample_start"], w["periods"]["out_sample_end"]
                )
                for w in valid_windows
            ]
            avg_benchmark_return = (
                np.mean(benchmark_returns) if benchmark_returns else 0
            )
            avg_alpha = np.mean(all_alphas) if all_alphas else 0

            results["summary_stats"] = {
                "total_windows": len(windows),
                "processed_windows": processed_windows,
                "valid_windows": len(valid_windows),
                "success_rate": len(valid_windows) / len(windows) * 100,
                "avg_in_sample_return": (
                    np.mean(all_in_sample_returns) if all_in_sample_returns else 0
                ),
                "avg_out_sample_return": (
                    np.mean(all_out_sample_returns) if all_out_sample_returns else 0
                ),
                "std_in_sample_return": (
                    np.std(all_in_sample_returns) if all_in_sample_returns else 0
                ),
                "std_out_sample_return": (
                    np.std(all_out_sample_returns) if all_out_sample_returns else 0
                ),
                "correlation": correlation,
                "correlation_pvalue": p_value,
                "avg_degradation": (
                    np.mean(
                        [
                            w["degradation"]["return_degradation"]
                            for w in valid_windows
                            if "degradation" in w
                        ]
                    )
                    if valid_windows
                    else 0
                ),
                "win_rate_out_sample": (
                    sum(1 for r in all_out_sample_returns if r > 0)
                    / len(all_out_sample_returns)
                    * 100
                    if all_out_sample_returns
                    else 0
                ),
                "consistency_ratio": (
                    (
                        sum(
                            1
                            for w in valid_windows
                            if w["out_sample_performance"]["summary"].get(
                                "total_return_pct", 0
                            )
                            > 0
                        )
                        / len(valid_windows)
                    )
                    * 100
                    if valid_windows
                    else 0
                ),
                "avg_alpha": avg_alpha,
                "avg_benchmark_return": avg_benchmark_return,
                "alpha_consistency": (
                    sum(1 for a in all_alphas if a > 0) / len(all_alphas) * 100
                    if all_alphas
                    else 0
                ),
            }

            print(f"\n" + "=" * 40)
            print("WALK-FORWARD SUMMARY")
            print("=" * 40)
            print(
                f"Valid windows: {len(valid_windows)}/{len(windows)} ({results['summary_stats']['success_rate']:.1f}%)"
            )
            print(
                f"Average in-sample return: {results['summary_stats']['avg_in_sample_return']:.2f}%"
            )
            print(
                f"Average out-of-sample return: {results['summary_stats']['avg_out_sample_return']:.2f}%"
            )
            print(
                f"Out-of-sample win rate: {results['summary_stats']['win_rate_out_sample']:.1f}%"
            )
            print(
                f"Consistency ratio: {results['summary_stats']['consistency_ratio']:.1f}%"
            )
            print(
                f"Return correlation: {results['summary_stats']['correlation']:.3f} (p={results['summary_stats']['correlation_pvalue']:.4f})"
            )
            print(
                f"Average degradation: {results['summary_stats']['avg_degradation']:.2f}%"
            )
            print(
                f"Average {self.benchmark} return: {results['summary_stats']['avg_benchmark_return']:.2f}%"
            )
            print(f"Average Alpha: {results['summary_stats']['avg_alpha']:.2f}%")
            print(
                f"Alpha consistency: {results['summary_stats']['alpha_consistency']:.1f}%"
            )

            # Generate plot
            plot_data = self.plot_walk_forward_results(results)
            if plot_data:
                results["plot"] = plot_data

        else:
            print(f"\n" + "=" * 40)
            print("WALK-FORWARD SUMMARY")
            print("=" * 40)
            print(f"No valid windows found out of {len(windows)} total windows")
            print("This could be due to:")
            print("- Insufficient data in the individual window date ranges")
            print("- Not enough trades generated by the strategy")
            print("- Technical indicator calculation issues")
            print("- Strategy parameter optimization failures")
            print("\nRecommendations:")
            print(
                "- Use longer in-sample periods (try 150-200 days minimum for large parameters)"
            )
            print("- Reduce the minimum trade requirement")
            print("- Check your strategy parameters and ensure they're reasonable")
            print("- Verify the data quality for the specified date range")
            print("- Consider using a different interval (e.g., 1h instead of 1d)")

            results["summary_stats"] = {
                "total_windows": len(windows),
                "processed_windows": processed_windows,
                "valid_windows": 0,
                "success_rate": 0,
                "error": "No valid windows generated",
            }

        return results

    def _run_backtest(
        self, start_date: str, end_date: str, interval: str, **strategy_params
    ):
        """Run a backtest with given parameters and benchmark."""
        logger.info(
            f"Running backtest from {start_date} to {end_date} with params {strategy_params}"
        )
        try:
            # Proceed with backtest
            results, cerebro = run_backtest(
                strategy_class=self.strategy_class,
                interval=interval,
                ticker=self.ticker,
                start_date=start_date,
                end_date=end_date,
                initial_cash=self.initial_cash,
                commission=self.commission,
                **strategy_params,
            )

            if not results:
                logger.warning("Backtest returned no results")
                return None

            return results[0]

        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            return None

    def _process_walk_forward_window(
        self,
        window: Dict,
        window_idx: int,
        n_trials: int,
        min_trades: int,
        interval: str,
        total_windows: int,
    ) -> Dict:
        """Process a single walk-forward window with benchmark."""
        logger.info(f"Processing window {window_idx+1}/{total_windows}")
        print(f"\nProcessing window {window_idx+1}/{total_windows}")
        print(f"  In-sample: {window['in_sample_start']} to {window['in_sample_end']}")
        print(
            f"  Out-sample: {window['out_sample_start']} to {window['out_sample_end']}"
        )

        window_result = {
            "window_id": window_idx + 1,
            "periods": window,
            "valid": False,
            "error": None,
        }

        try:
            # Get benchmark returns for this window
            benchmark_in_return = self._get_benchmark_return(
                window["in_sample_start"], window["in_sample_end"]
            )
            benchmark_out_return = self._get_benchmark_return(
                window["out_sample_start"], window["out_sample_end"]
            )
            window_result["benchmark_returns"] = {
                "in_sample": benchmark_in_return,
                "out_sample": benchmark_out_return,
            }

            # Optimize parameters on in-sample data
            print("  Optimizing parameters...")
            try:
                optimization_results = optimize_strategy(
                    strategy_class=self.strategy_name,
                    ticker=self.ticker,
                    start_date=window["in_sample_start"],
                    end_date=window["in_sample_end"],
                    n_trials=n_trials,
                    initial_cash=self.initial_cash,
                    commission=self.commission,
                    interval=interval,
                )
                best_params = optimization_results["best_params"]
                print(f"Best parameters: {best_params}")
            except Exception as e:
                logger.warning(
                    f"Optimization failed: {str(e)}. Using default parameters."
                )
                best_params = self.default_params
                print(f"Using default parameters: {best_params}")

            # Run backtests and handle insufficient data
            print("  Running in-sample backtest...")
            in_sample_results = self._run_backtest(
                window["in_sample_start"],
                window["in_sample_end"],
                interval=interval,
                **best_params,
            )

            if in_sample_results is None:
                window_result["valid"] = False
                window_result["error"] = "Insufficient data in in-sample backtest"
                return window_result

            print("  Running out-of-sample backtest...")
            out_sample_results = self._run_backtest(
                window["out_sample_start"],
                window["out_sample_end"],
                interval=interval,
                **best_params,
            )

            if out_sample_results is None:
                window_result["valid"] = False
                window_result["error"] = "Insufficient data in out-sample backtest"
                return window_result

            # Generate performance reports
            in_sample_analyzer = PerformanceAnalyzer(in_sample_results)
            out_sample_analyzer = PerformanceAnalyzer(out_sample_results)

            in_sample_perf = in_sample_analyzer.generate_full_report()
            out_sample_perf = out_sample_analyzer.generate_full_report()

            # --- PATCH: Add trades and timereturn to out_sample_perf ---
            try:
                out_sample_perf["trades"] = (
                    out_sample_analyzer.get_trades()
                )  # Should be a list of dicts
            except Exception as e:
                out_sample_perf["trades"] = []
                logger.warning(f"Could not extract trades: {e}")

            try:
                out_sample_perf["timereturn"] = (
                    out_sample_analyzer.get_timereturn()
                )  # Should be a dict or pd.Series
            except Exception as e:
                out_sample_perf["timereturn"] = {}
                logger.warning(f"Could not extract timereturn: {e}")

            # Optionally, do the same for in_sample_perf if needed
            try:
                in_sample_perf["trades"] = in_sample_analyzer.get_trades()
            except Exception as e:
                in_sample_perf["trades"] = []
            try:
                in_sample_perf["timereturn"] = in_sample_analyzer.get_timereturn()
            except Exception as e:
                in_sample_perf["timereturn"] = {}

            # Add benchmark alpha
            in_sample_perf["summary"]["alpha"] = (
                in_sample_perf["summary"].get("total_return_pct", 0)
                - benchmark_in_return
            )
            out_sample_perf["summary"]["alpha"] = (
                out_sample_perf["summary"].get("total_return_pct", 0)
                - benchmark_out_return
            )

            # Get trade counts
            in_trade_analysis = in_sample_perf.get("trade_analysis", {})
            out_trade_analysis = out_sample_perf.get("trade_analysis", {})
            in_sample_trades = in_trade_analysis.get("total_trades", 0)
            out_sample_trades = out_trade_analysis.get("total_trades", 0)

            # Check if window is valid
            valid = in_sample_trades >= min_trades and out_sample_trades >= min_trades

            # Prepare results
            window_result.update(
                {
                    "best_params": best_params,
                    "in_sample_performance": in_sample_perf,
                    "out_sample_performance": out_sample_perf,
                    "valid": valid,
                    "degradation": self._calculate_degradation(
                        in_sample_perf, out_sample_perf
                    ),
                }
            )

            if valid:
                in_sample_return = in_sample_perf["summary"].get("total_return_pct", 0)
                out_sample_return = out_sample_perf["summary"].get(
                    "total_return_pct", 0
                )
                print(f"  In-sample return: {in_sample_return:.2f}%")
                print(f"  Out-sample return: {out_sample_return:.2f}%")
                print(f"  {self.benchmark} in-sample: {benchmark_in_return:.2f}%")
                print(f"  {self.benchmark} out-sample: {benchmark_out_return:.2f}%")
                print(f"  Alpha: {out_sample_perf['summary']['alpha']:.2f}%")
                print(f"  Trades: {in_sample_trades} (in) / {out_sample_trades} (out)")
            else:
                print(
                    f"  Skipped: Insufficient trades ({in_sample_trades}/{out_sample_trades})"
                )

        except Exception as e:
            logger.error(f"Error processing window {window_idx+1}: {str(e)}")
            window_result["error"] = str(e)

        return window_result

    def _calculate_degradation(
        self, in_sample_perf: Dict, out_sample_perf: Dict
    ) -> Dict:
        """Calculate comprehensive performance degradation metrics."""
        try:
            # Get metrics with safe defaults
            in_return = in_sample_perf["summary"].get("total_return_pct", 0)
            out_return = out_sample_perf["summary"].get("total_return_pct", 0)
            in_alpha = in_sample_perf["summary"].get("alpha", 0)
            out_alpha = out_sample_perf["summary"].get("alpha", 0)
            in_sharpe = in_sample_perf["summary"].get("sharpe_ratio", 0)
            out_sharpe = out_sample_perf["summary"].get("sharpe_ratio", 0)
            in_profit_factor = in_sample_perf.get("trade_analysis", {}).get(
                "profit_factor", 0
            )
            out_profit_factor = out_sample_perf.get("trade_analysis", {}).get(
                "profit_factor", 0
            )
            in_max_dd = in_sample_perf["summary"].get("max_drawdown_pct", 0)
            out_max_dd = out_sample_perf["summary"].get("max_drawdown_pct", 0)

            # Calculate relative metrics
            return_degradation = in_return - out_return
            alpha_degradation = in_alpha - out_alpha
            sharpe_degradation = in_sharpe - out_sharpe
            profit_factor_degradation = in_profit_factor - out_profit_factor
            dd_increase = out_max_dd - in_max_dd

            # Calculate stability ratios
            return_ratio = out_return / in_return if in_return != 0 else float("inf")
            alpha_ratio = out_alpha / in_alpha if in_alpha != 0 else float("inf")
            profit_factor_ratio = (
                out_profit_factor / in_profit_factor
                if in_profit_factor != 0
                else float("inf")
            )
            sharpe_ratio = out_sharpe / in_sharpe if in_sharpe != 0 else float("inf")
            return {
                "return_degradation": return_degradation,
                "alpha_degradation": alpha_degradation,
                "sharpe_degradation": sharpe_degradation,
                "profit_factor_degradation": profit_factor_degradation,
                "drawdown_increase": dd_increase,
                "return_ratio": return_ratio,
                "alpha_ratio": alpha_ratio,
                "sharpe_ratio": sharpe_ratio,
                "profit_factor_ratio": profit_factor_ratio,
                "stability_score": self._calculate_stability_score(
                    return_ratio, alpha_ratio, sharpe_ratio, profit_factor_ratio
                ),
            }
        except Exception as e:
            logger.error(f"Error calculating degradation: {str(e)}")
            return {
                "return_degradation": 0,
                "alpha_degradation": 0,
                "sharpe_degradation": 0,
                "profit_factor_degradation": 0,
                "drawdown_increase": 0,
                "return_ratio": 0,
                "alpha_ratio": 0,
                "sharpe_ratio": 0,
                "profit_factor_ratio": 0,
                "stability_score": 0,
            }

    def _calculate_stability_score(
        self,
        return_ratio: float,
        alpha_ratio: float,
        sharpe_ratio: float,
        profit_factor_ratio: float,
    ) -> float:
        """Calculate composite stability score from key metrics"""
        # Normalize ratios to be between 0 and 2 (1 is perfect stability)
        normalized_ratios = []

        for ratio in [return_ratio, alpha_ratio, sharpe_ratio, profit_factor_ratio]:
            if ratio == float("inf"):
                normalized = 0  # Penalize infinite ratios
            elif ratio > 1:
                normalized = 2 - ratio if ratio < 2 else 0
            else:
                normalized = ratio
            normalized_ratios.append(normalized)

        # Average normalized scores and scale to 0-100
        avg_score = sum(normalized_ratios) / len(normalized_ratios)
        return max(0, min(100, avg_score * 100))

    def generate_walk_forward_report(self, wf_results, filename=None):
        """Generate a comprehensive report for walk-forward analysis."""
        report = {
            "overview": {
                "total_windows": wf_results["summary_stats"].get("total_windows", 0),
                "valid_windows": wf_results["summary_stats"].get("valid_windows", 0),
                "avg_in_sample_return": wf_results["summary_stats"].get(
                    "avg_in_sample_return", 0
                ),
                "avg_out_sample_return": wf_results["summary_stats"].get(
                    "avg_out_sample_return", 0
                ),
                "out_sample_win_rate": wf_results["summary_stats"].get(
                    "win_rate_out_sample", 0
                ),
                "return_correlation": wf_results["summary_stats"].get("correlation", 0),
                "avg_degradation": wf_results["summary_stats"].get(
                    "avg_degradation", 0
                ),
                "avg_benchmark_return": wf_results["summary_stats"].get(
                    "avg_benchmark_return", 0
                ),
                "avg_alpha": wf_results["summary_stats"].get("avg_alpha", 0),
                "alpha_consistency": wf_results["summary_stats"].get(
                    "alpha_consistency", 0
                ),
            },
            "windows": [],
        }

        for window in wf_results["windows"]:
            if window.get("valid", False):
                window_report = {
                    "window_id": window["window_id"],
                    "in_sample_period": f"{window['periods']['in_sample_start']} to {window['periods']['in_sample_end']}",
                    "out_sample_period": f"{window['periods']['out_sample_start']} to {window['periods']['out_sample_end']}",
                    "best_params": window.get("best_params", {}),
                    "in_sample_performance": window.get(
                        "in_sample_performance", {}
                    ).get("summary", {}),
                    "out_sample_performance": window.get(
                        "out_sample_performance", {}
                    ).get("summary", {}),
                    "benchmark_in_sample": window.get("benchmark_returns", {}).get(
                        "in_sample", 0
                    ),
                    "benchmark_out_sample": window.get("benchmark_returns", {}).get(
                        "out_sample", 0
                    ),
                    "degradation": window.get("degradation", {}),
                }
                report["windows"].append(window_report)

        if filename:
            try:
                with open(filename, "w") as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Walk-forward report saved to {filename}")
            except Exception as e:
                logger.error(f"Error saving report: {str(e)}")

        return report

    def generate_best_params_report(self, wf_results, filename=None):
        """Generate a report summarizing the best parameters and their performance metrics."""
        report = {
            "ticker": self.ticker,
            "strategy": self.strategy_name,
            "benchmark": self.benchmark,
            "total_windows": wf_results["summary_stats"].get("total_windows", 0),
            "valid_windows": wf_results["summary_stats"].get("valid_windows", 0),
            "parameter_summary": {},
            "performance_summary": {
                "avg_out_sample_return": wf_results["summary_stats"].get(
                    "avg_out_sample_return", 0
                ),
                "avg_out_sample_drawdown": 0,
                "avg_win_percentage": 0,
                "avg_total_trades": 0,
                "avg_winning_trades": 0,
                "avg_losing_trades": 0,
                "avg_win_loss_ratio": 0,
                "avg_profit_factor": 0,
                "avg_alpha": wf_results["summary_stats"].get("avg_alpha", 0),
                "avg_benchmark_return": wf_results["summary_stats"].get(
                    "avg_benchmark_return", 0
                ),
            },
            "windows": [],
        }

        # Aggregate parameters and performance metrics
        param_counts = {}
        returns = []
        alphas = []
        benchmark_returns = []
        drawdowns = []
        win_percentages = []
        total_trades = []
        winning_trades = []
        losing_trades = []
        win_loss_ratios = []
        profit_factors = []

        for window in wf_results["windows"]:
            if window.get("valid", False):
                params = tuple(
                    sorted(window["best_params"].items())
                )  # Convert params to tuple for hashing
                param_counts[params] = param_counts.get(params, 0) + 1

                out_sample_perf = window.get("out_sample_performance", {}).get(
                    "summary", {}
                )
                trade_analysis = window.get("out_sample_performance", {}).get(
                    "trade_analysis", {}
                )

                window_report = {
                    "window_id": window["window_id"],
                    "out_sample_period": f"{window['periods']['out_sample_start']} to {window['periods']['out_sample_end']}",
                    "best_params": window.get("best_params", {}),
                    "total_return_pct": out_sample_perf.get("total_return_pct", 0),
                    "alpha": out_sample_perf.get("alpha", 0),
                    "benchmark_return": window.get("benchmark_returns", {}).get(
                        "out_sample", 0
                    ),
                    "max_drawdown_pct": out_sample_perf.get("max_drawdown_pct", 0),
                    "win_percentage": (
                        trade_analysis.get("win_rate_percent", 0)
                        if isinstance(trade_analysis, dict)
                        else 0
                    ),
                    "total_trades": (
                        trade_analysis.get("total_trades", 0)
                        if isinstance(trade_analysis, dict)
                        else 0
                    ),
                    "winning_trades": (
                        trade_analysis.get("winning_trades", 0)
                        if isinstance(trade_analysis, dict)
                        else 0
                    ),
                    "losing_trades": (
                        trade_analysis.get("losing_trades", 0)
                        if isinstance(trade_analysis, dict)
                        else 0
                    ),
                    "win_loss_ratio": (
                        trade_analysis.get("win_rate_percent", 0)
                        / (100 - trade_analysis.get("win_rate_percent", 0))
                        if trade_analysis.get("win_rate_percent", 0) not in [0, 100]
                        else 0
                    ),
                    "profit_factor": (
                        trade_analysis.get("profit_factor", 0)
                        if isinstance(trade_analysis, dict)
                        else 0
                    ),
                }
                report["windows"].append(window_report)

                returns.append(out_sample_perf.get("total_return_pct", 0))
                alphas.append(out_sample_perf.get("alpha", 0))
                benchmark_returns.append(
                    window.get("benchmark_returns", {}).get("out_sample", 0)
                )
                drawdowns.append(out_sample_perf.get("max_drawdown_pct", 0))
                win_percentages.append(
                    trade_analysis.get("win_rate_percent", 0)
                    if isinstance(trade_analysis, dict)
                    else 0
                )
                total_trades.append(
                    trade_analysis.get("total_trades", 0)
                    if isinstance(trade_analysis, dict)
                    else 0
                )
                winning_trades.append(
                    trade_analysis.get("winning_trades", 0)
                    if isinstance(trade_analysis, dict)
                    else 0
                )
                losing_trades.append(
                    trade_analysis.get("losing_trades", 0)
                    if isinstance(trade_analysis, dict)
                    else 0
                )
                win_loss_ratios.append(window_report["win_loss_ratio"])
                profit_factors.append(
                    trade_analysis.get("profit_factor", 0)
                    if isinstance(trade_analysis, dict)
                    else 0
                )

        # Summarize parameter frequency
        report["parameter_summary"]

        # Calculate performance averages
        if report["windows"]:
            report["performance_summary"] = {
                "avg_out_sample_return": np.mean(returns),
                "avg_alpha": np.mean(alphas),
                "avg_benchmark_return": np.mean(benchmark_returns),
                "avg_out_sample_drawdown": np.mean(drawdowns),
                "avg_win_percentage": np.mean(win_percentages),
                "avg_total_trades": np.mean(total_trades),
                "avg_winning_trades": np.mean(winning_trades),
                "avg_losing_trades": np.mean(losing_trades),
                "avg_win_loss_ratio": (
                    np.mean([r for r in win_loss_ratios if r != float("inf")])
                    if any(r != float("inf") for r in win_loss_ratios)
                    else 0
                ),
                "avg_profit_factor": (
                    np.mean([pf for pf in profit_factors if pf != float("inf")])
                    if any(pf != float("inf") for pf in profit_factors)
                    else 0
                ),
            }

        # Print the report
        print("=" * 60)
        print(f"BEST PARAMETERS PERFORMANCE REPORT (VS {self.benchmark})")
        print("=" * 60)
        print(f"\nTicker: {report['ticker']}")
        print(f"Strategy: {report['strategy']}")
        print(f"Benchmark: {report['benchmark']}")
        print(f"Total Windows: {report['total_windows']}")
        print(f"Valid Windows: {report['valid_windows']}")
        print("\nParameter Frequency:")
        print("-" * 30)
        for param_set in report["parameter_summary"]:
            print(f"Parameters: {param_set['parameters']}")
            print(f"Frequency: {param_set['frequency']} windows")
            print()

        print("\nPerformance Summary (Out-of-Sample):")
        print("-" * 30)
        print(
            f"Average Return: {report['performance_summary']['avg_out_sample_return']:.2f}%"
        )
        print(f"Average Alpha: {report['performance_summary']['avg_alpha']:.2f}%")
        print(
            f"Average {self.benchmark} Return: {report['performance_summary']['avg_benchmark_return']:.2f}%"
        )
        print(
            f"Average Max Drawdown: {report['performance_summary']['avg_out_sample_drawdown']:.2f}%"
        )
        print(
            f"Average Win Percentage: {report['performance_summary']['avg_win_percentage']:.2f}%"
        )
        print(
            f"Average Total Trades: {report['performance_summary']['avg_total_trades']:.2f}"
        )
        print(
            f"Average Winning Trades: {report['performance_summary']['avg_winning_trades']:.2f}"
        )
        print(
            f"Average Losing Trades: {report['performance_summary']['avg_losing_trades']:.2f}"
        )
        print(
            f"Average Win/Loss Ratio: {report['performance_summary']['avg_win_loss_ratio']:.2f}"
        )
        print(
            f"Average Profit Factor: {report['performance_summary']['avg_profit_factor']:.2f}"
        )

        print("\nPer Window Details:")
        print("-" * 30)
        for window in report["windows"]:
            print(f"Window {window['window_id']}:")
            print(f"  Out-of-Sample Period: {window['out_sample_period']}")
            print(f"  Best Parameters: {window['best_params']}")
            print(f"  Total Return: {window['total_return_pct']:.2f}%")
            print(f"  Alpha: {window['alpha']:.2f}%")
            print(f"  {self.benchmark} Return: {window['benchmark_return']:.2f}%")
            print(f"  Max Drawdown: {window['max_drawdown_pct']:.2f}%")
            print(f"  Win Percentage: {window['win_percentage']:.2f}%")
            print(f"  Total Trades: {window['total_trades']}")
            print(f"  Winning Trades: {window['winning_trades']}")
            print(f"  Losing Trades: {window['losing_trades']}")
            print(f"  Win/Loss Ratio: {window['win_loss_ratio']:.2f}")
            print(f"  Profit Factor: {window['profit_factor']:.2f}")
            print()

        if filename:
            try:
                with open(filename, "w") as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Best parameters report saved to {filename}")
            except Exception as e:
                logger.error(f"Error saving best parameters report: {str(e)}")

        return report

    def print_walk_forward_report(self, report):
        """Print a formatted walk-forward analysis report with benchmark comparison."""
        print("=" * 60)
        print(f"WALK-FORWARD ANALYSIS REPORT (VS {self.benchmark})")
        print("=" * 60)
        overview = report["overview"]
        print("\nOVERVIEW")
        print("-" * 30)
        print(f"Total windows: {overview['total_windows']}")
        print(f"Valid windows: {overview['valid_windows']}")
        print(f"Average in-sample return: {overview['avg_in_sample_return']:.2f}%")
        print(f"Average out-sample return: {overview['avg_out_sample_return']:.2f}%")
        print(
            f"Average {self.benchmark} return: {overview['avg_benchmark_return']:.2f}%"
        )
        print(f"Average Alpha: {overview['avg_alpha']:.2f}%")
        print(f"Out-sample win rate: {overview['out_sample_win_rate']:.1f}%")
        print(f"Return correlation: {overview['return_correlation']:.3f}")
        print(f"Average degradation: {overview['avg_degradation']:.2f}%")

        print("\nPER WINDOW DETAILS")
        print("-" * 30)
        for window in report["windows"]:
            print(f"Window {window['window_id']}:")
            print(f"  In-sample period: {window['in_sample_period']}")

            print(f"  Out-sample period: {window['out_sample_period']}")
            print(f"  Best parameters: {window['best_params']}")
            print(
                f"  In-sample return: {window['in_sample_performance'].get('total_return_pct', 0):.2f}%"
            )
            print(
                f"  Out-sample return: {window['out_sample_performance'].get('total_return_pct', 0):.2f}%"
            )
            print(f"  {self.benchmark} in-sample: {window['benchmark_in_sample']:.2f}%")
            print(
                f"  {self.benchmark} out-sample: {window['benchmark_out_sample']:.2f}%"
            )
            print(f"  Alpha: {window['out_sample_performance'].get('alpha', 0):.2f}%")
            print(
                f"  Degradation: {window['degradation'].get('return_degradation', 0):.2f}%"
            )
            print()

    def plot_walk_forward_results(self, wf_results: Dict):
        """Generate walk-forward analysis plots with benchmark comparison."""
        if not wf_results["windows"]:
            print("No walk-forward results to plot")
            return None

        # Extract data for plotting
        valid_windows = [w for w in wf_results["windows"] if w.get("valid", False)]

        if not valid_windows:
            print("No valid windows to plot")
            return None

        window_ids = [w["window_id"] for w in valid_windows]
        in_sample_returns = [
            w["in_sample_performance"]["summary"].get("total_return_pct", 0)
            for w in valid_windows
        ]
        out_sample_returns = [
            w["out_sample_performance"]["summary"].get("total_return_pct", 0)
            for w in valid_windows
        ]
        benchmark_returns = [
            w["benchmark_returns"].get("out_sample", 0) for w in valid_windows
        ]
        alphas = [
            w["out_sample_performance"]["summary"].get("alpha", 0)
            for w in valid_windows
        ]
        degradations = [
            w["degradation"].get("return_degradation", 0) for w in valid_windows
        ]

        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Walk-Forward Analysis vs {self.benchmark}", fontsize=16, fontweight="bold"
        )

        # Plot 1: Strategy vs benchmark returns
        axes[0, 0].plot(
            window_ids, out_sample_returns, "b-o", label="Strategy", linewidth=2
        )
        axes[0, 0].plot(
            window_ids, benchmark_returns, "g-s", label=self.benchmark, linewidth=2
        )
        axes[0, 0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[0, 0].set_xlabel("Window ID")
        axes[0, 0].set_ylabel("Return (%)")
        axes[0, 0].set_title(f"Strategy vs {self.benchmark} Returns (Out-of-Sample)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Alphas
        axes[0, 1].bar(window_ids, alphas, color="orange", alpha=0.7)
        axes[0, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[0, 1].set_xlabel("Window ID")
        axes[0, 1].set_ylabel("Alpha (%)")
        axes[0, 1].set_title(f"Alpha vs {self.benchmark} per Window")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Cumulative returns
        cumulative_strategy = np.cumprod(1 + np.array(out_sample_returns) / 100) - 1
        cumulative_benchmark = np.cumprod(1 + np.array(benchmark_returns) / 100) - 1
        axes[1, 0].plot(
            window_ids,
            cumulative_strategy * 100,
            "b-o",
            linewidth=2,
            markersize=4,
            label="Strategy",
        )
        axes[1, 0].plot(
            window_ids,
            cumulative_benchmark * 100,
            "g-s",
            linewidth=2,
            markersize=4,
            label=self.benchmark,
        )
        axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[1, 0].set_xlabel("Window ID")
        axes[1, 0].set_ylabel("Cumulative Return (%)")
        axes[1, 0].set_title("Cumulative Out-of-Sample Returns")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Performance degradation
        axes[1, 1].bar(window_ids, degradations, color="red", alpha=0.7)
        axes[1, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[1, 1].set_xlabel("Window ID")
        axes[1, 1].set_ylabel("Degradation (%)")
        axes[1, 1].set_title("Performance Degradation per Window")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

        # Save plot to bytes for Streamlit
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf

    def _print_in_out_comparison(self, results: Dict):
        """Print a comparison of in-sample and out-of-sample performance with benchmark."""
        print("\n" + "=" * 40)
        print(f"PERFORMANCE COMPARISON (VS {self.benchmark})")
        print("=" * 40)

        # Extract summary dictionaries safely
        in_summary = results["in_sample_performance"].get("summary", {})
        out_summary = results["out_sample_performance"].get("summary", {})
        degradation = results.get("performance_degradation", {})
        bench_returns = results.get("benchmark_returns", {})

        # Convert values to float safely
        in_return = self._safe_float(in_summary.get("total_return_pct", 0))
        in_alpha = self._safe_float(in_summary.get("alpha", 0))
        in_annual = self._safe_float(in_summary.get("annual_return_pct", 0))
        in_drawdown = self._safe_float(in_summary.get("max_drawdown_pct", 0))
        in_sharpe = self._safe_float(in_summary.get("sharpe_ratio", 0))

        out_return = self._safe_float(out_summary.get("total_return_pct", 0))
        out_alpha = self._safe_float(out_summary.get("alpha", 0))
        out_annual = self._safe_float(out_summary.get("annual_return_pct", 0))
        out_drawdown = self._safe_float(out_summary.get("max_drawdown_pct", 0))
        out_sharpe = self._safe_float(out_summary.get("sharpe_ratio", 0))

        benchmark_in = self._safe_float(bench_returns.get("in_sample", 0))
        benchmark_out = self._safe_float(bench_returns.get("out_sample", 0))

        return_degradation = self._safe_float(degradation.get("return_degradation", 0))
        alpha_degradation = self._safe_float(degradation.get("alpha_degradation", 0))
        sharpe_degradation = self._safe_float(degradation.get("sharpe_degradation", 0))

        print(
            f"\nIn-sample ({results['in_sample_period'][0]} to {results['in_sample_period'][1]}):"
        )
        print(f"  Strategy Return: {in_return:.2f}%")
        print(f"  {self.benchmark} Return: {benchmark_in:.2f}%")
        print(f"  Alpha: {in_alpha:.2f}%")
        print(f"  Annualized Return: {in_annual:.2f}%")
        print(f"  Max Drawdown: {in_drawdown:.2f}%")
        print(f"  Sharpe Ratio: {in_sharpe:.2f}")

        print(
            f"\nOut-of-sample ({(results['out_sample_period'][0])} to {results['out_sample_period'][1]})"
        )
        print(f"  Strategy Return: {out_return:.2f}%")
        print(f"  {self.benchmark} Return: {benchmark_out:.2f}%")
        print(f"  Alpha: {out_alpha:.2f}%")
        print(f"  Annualized Return: {out_annual:.2f}%")
        print(f"  Max Drawdown: {out_drawdown:.2f}%")
        print(f"  Sharpe Ratio: {out_sharpe:.2f}")

        print(f"\nPerformance Degradation:")
        print(f"  Return Degradation: {return_degradation:.2f}%")
        print(f"  Alpha Degradation: {alpha_degradation:.2f}%")
        print(f"  Sharpe Degradation: {sharpe_degradation:.2f}")
        print("=" * 40)
