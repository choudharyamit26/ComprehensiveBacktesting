import json
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List, Optional
import warnings
import logging
import matplotlib.pyplot as plt
from io import BytesIO

from .data import get_data_sync
from .reports import PerformanceAnalyzer
from .parameter_optimization import optimize_strategy, SortinoRatio

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationAnalyzer:
    """Comprehensive validation analysis for trading strategies, including in-sample, out-of-sample, and walk-forward analysis."""

    def __init__(
        self,
        strategy_name: str,  # Changed from strategy_class to strategy_name
        ticker: str,
        initial_cash: float = 100000.0,
        commission: float = 0.05,
        analyzers: Optional[List] = None,
    ):
        """Initialize the validation analyzer.

        Args:
            strategy_name (str): Name of the registered strategy
            ticker (str): Stock ticker symbol.
            initial_cash (float): Initial portfolio cash.
            commission (float): Broker commission rate.
        """
        from .registry import get_strategy  # Import here to avoid circular dependencies

        # Get strategy class from registry
        self.strategy_class = get_strategy(strategy_name)
        if not self.strategy_class:
            raise ValueError(f"Strategy '{strategy_name}' not found in registry")

        self.strategy_name = strategy_name
        self.ticker = ticker
        self.initial_cash = initial_cash
        self.commission = commission
        self.analyzers = analyzers if analyzers is not None else []
        logger.info(
            f"Initialized ValidationAnalyzer for {ticker} with strategy {strategy_name}"
        )

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def run_validation_analysis(
        self,
        start_date: str,
        end_date: str,
        analysis_type: str = "walkforward",
        in_sample_days: int = 30,
        out_sample_days: int = 15,
        step_days: int = 15,
        n_trials: int = 20,
        min_trades: int = 1,
        split_ratio: float = 0.7,
        optimize_in_sample: bool = True,
        interval: str = "5m",
    ) -> Dict:
        """Run validation analysis (in-sample/out-of-sample or walk-forward).

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            analysis_type (str): Type of analysis ('walkforward' or 'inout').
            in_sample_days (int): Length of in-sample period in days (for walk-forward).
            out_sample_days (int): Length of out-of-sample period in days (for walk-forward).
            step_days (int): Step size between windows in days (for walk-forward).
            n_trials (int): Number of optimization trials.
            min_trades (int): Minimum trades required for valid results (for walk-forward).
            split_ratio (float): Ratio for in-sample period (for in-sample/out-of-sample).
            optimize_in_sample (bool): Whether to optimize parameters on in-sample data.

        Returns:
            Dict: Analysis results.
        """
        logger.info(
            f"Running {analysis_type} analysis for {self.ticker} from {start_date} to {end_date}"
        )
        print("=" * 60)
        print(f"{analysis_type.upper()} VALIDATION ANALYSIS")
        print("=" * 60)

        try:
            if analysis_type.lower() == "walkforward":
                results = self.walk_forward_analysis(
                    start_date=start_date,
                    end_date=end_date,
                    in_sample_days=in_sample_days,
                    out_sample_days=out_sample_days,
                    step_days=step_days,
                    n_trials=n_trials,
                    min_trades=min_trades,
                )
            elif analysis_type.lower() == "inout":
                results = self.in_sample_out_sample_analysis(
                    start_date=start_date,
                    end_date=end_date,
                    split_ratio=split_ratio,
                    optimize_in_sample=optimize_in_sample,
                    n_trials=n_trials,
                )
            else:
                raise ValueError(
                    f"Invalid analysis_type: {analysis_type}. Choose 'walkforward' or 'inout'."
                )
            return results
        except Exception as e:
            logger.error(f"Validation analysis failed: {str(e)}")
            raise

    def in_sample_out_sample_analysis(
        self,
        start_date: str,
        end_date: str,
        split_ratio: float = 0.7,
        optimize_in_sample: bool = True,
        n_trials: int = 50,
        interval: str = "5m",
    ) -> Dict:
        """Perform in-sample and out-of-sample analysis.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            split_ratio (float): Ratio for in-sample period (default: 0.7).
            optimize_in_sample (bool): Whether to optimize parameters on in-sample data.
            n_trials (int): Number of optimization trials.

        Returns:
            Dict: Analysis results including performance metrics and parameters.
        """
        logger.info(f"Starting in-sample/out-of-sample analysis for {self.ticker}")
        print("=" * 60)
        print("IN-SAMPLE / OUT-OF-SAMPLE ANALYSIS")
        print("=" * 60)

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        total_days = (end_dt - start_dt).days
        split_days = int(total_days * split_ratio)
        split_date = start_dt + timedelta(days=split_days)
        split_date_str = split_date.strftime("%Y-%m-%d")

        print(f"Data Period: {start_date} to {end_date}")
        print(f"In-Sample Period: {start_date} to {split_date_str}")
        print(f"Out-of-Sample Period: {split_date_str} to {end_date}")
        print(
            f"Split Ratio: {split_ratio:.1%} in-sample, {1-split_ratio:.1%} out-of-sample"
        )

        results = {
            "split_date": split_date_str,
            "in_sample_period": (start_date, split_date_str),
            "out_sample_period": (split_date_str, end_date),
            "split_ratio": split_ratio,
        }

        try:
            if optimize_in_sample:
                logger.info(
                    f"Optimizing parameters on in-sample data ({n_trials} trials)"
                )
                print(f"Strategy class type: {type(self.strategy_class)}")
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
                best_params = {}
                results["optimized_params"] = None

            print("\nTesting on in-sample data...")
            in_sample_results = self._run_backtest(
                start_date, split_date_str, interval=interval, **best_params
            )
            print("Testing on out-of-sample data...")
            out_sample_results = self._run_backtest(
                split_date_str, end_date, interval=interval, **best_params
            )

            in_sample_analyzer = PerformanceAnalyzer(in_sample_results)
            out_sample_analyzer = PerformanceAnalyzer(out_sample_results)
            results["in_sample_performance"] = in_sample_analyzer.generate_full_report()
            results["out_sample_performance"] = (
                out_sample_analyzer.generate_full_report()
            )
            results["performance_degradation"] = self._calculate_degradation(
                results["in_sample_performance"], results["out_sample_performance"]
            )

            self._print_in_out_comparison(results)
            return results

        except Exception as e:
            logger.error(
                f"In-sample/out-of-sample analysis failed: {str(e)}. Check data or strategy compatibility."
            )
            raise

    def generate_walk_forward_report(self, wf_results, filename=None):
        """Generate a comprehensive report for walk-forward analysis.

        Args:
            wf_results (Dict): Walk-forward analysis results.
            filename (str, optional): File to save the report.

        Returns:
            Dict: Structured report with overview and per-window details.
        """
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
        """Generate a report summarizing the best parameters and their performance metrics across all windows.

        Args:
            wf_results (Dict): Walk-forward analysis results.
            filename (str, optional): File to save the report.

        Returns:
            Dict: Structured report with best parameters and performance metrics.
        """
        report = {
            "ticker": self.ticker,
            "strategy": self.strategy_class.__name__,
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
            },
            "windows": [],
        }

        # Aggregate parameters and performance metrics
        param_counts = {}
        returns = []
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
        report["parameter_summary"] = [
            {"parameters": dict(params), "frequency": count}
            for params, count in param_counts.items()
        ]
        report["parameter_summary"].sort(key=lambda x: x["frequency"], reverse=True)

        # Calculate performance averages
        if report["windows"]:
            report["performance_summary"] = {
                "avg_out_sample_return": np.mean(returns),
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
        print("BEST PARAMETERS PERFORMANCE REPORT")
        print("=" * 60)
        print(f"\nTicker: {report['ticker']}")
        print(f"Strategy: {report['strategy']}")
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
        """Print a formatted walk-forward analysis report.

        Args:
            report (Dict): Report from generate_walk_forward_report.
        """
        print("=" * 60)
        print("WALK-FORWARD ANALYSIS REPORT")
        print("=" * 60)
        overview = report["overview"]
        print("\nOVERVIEW")
        print("-" * 30)
        print(f"Total windows: {overview['total_windows']}")
        print(f"Valid windows: {overview['valid_windows']}")
        print(f"Average in-sample return: {overview['avg_in_sample_return']:.2f}%")
        print(f"Average out-sample return: {overview['avg_out_sample_return']:.2f}%")
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
            print(
                f"  Degradation: {window['degradation'].get('return_degradation', 0):.2f}%"
            )
            print()

        print("=" * 60)

    def plot_walk_forward_results(self, wf_results: Dict):
        """Generate walk-forward analysis plots suitable for Streamlit."""
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
        degradations = [
            w["degradation"].get("return_degradation", 0) for w in valid_windows
        ]

        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Walk-Forward Analysis Results", fontsize=16, fontweight="bold")

        # Plot 1: In-sample vs Out-of-sample returns
        axes[0, 0].plot(
            window_ids, in_sample_returns, "b-o", label="In-Sample", linewidth=2
        )
        axes[0, 0].plot(
            window_ids, out_sample_returns, "r-s", label="Out-of-Sample", linewidth=2
        )
        axes[0, 0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[0, 0].set_xlabel("Window ID")
        axes[0, 0].set_ylabel("Return (%)")
        axes[0, 0].set_title("In-Sample vs Out-of-Sample Returns")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Performance degradation
        axes[0, 1].bar(window_ids, degradations, color="orange", alpha=0.7)
        axes[0, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[0, 1].set_xlabel("Window ID")
        axes[0, 1].set_ylabel("Degradation (%)")
        axes[0, 1].set_title("Performance Degradation per Window")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Scatter plot of in-sample vs out-of-sample
        axes[1, 0].scatter(in_sample_returns, out_sample_returns, alpha=0.7, s=60)
        min_val = min(min(in_sample_returns), min(out_sample_returns)) - 5
        max_val = max(max(in_sample_returns), max(out_sample_returns)) + 5
        axes[1, 0].plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            alpha=0.5,
            label="Perfect Correlation",
        )
        axes[1, 0].set_xlabel("In-Sample Return (%)")
        axes[1, 0].set_ylabel("Out-of-Sample Return (%)")
        axes[1, 0].set_title("In-Sample vs Out-of-Sample Correlation")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Cumulative returns
        cumulative_out_sample = np.cumprod(1 + np.array(out_sample_returns) / 100) - 1
        axes[1, 1].plot(
            window_ids, cumulative_out_sample * 100, "g-o", linewidth=2, markersize=4
        )
        axes[1, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[1, 1].set_xlabel("Window ID")
        axes[1, 1].set_ylabel("Cumulative Return (%)")
        axes[1, 1].set_title("Cumulative Out-of-Sample Returns")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

        # Save plot to bytes for Streamlit
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf

    def walk_forward_analysis(
        self,
        start_date: str,
        end_date: str,
        in_sample_days: int = 30,
        out_sample_days: int = 15,
        step_days: int = 15,
        n_trials: int = 20,
        min_trades: int = 1,
        interval: str = "5m",
    ) -> Dict:
        """Perform walk-forward analysis with day-based windows."""
        logger.info(f"Starting walk-forward analysis for {self.ticker}")
        print("=" * 60)
        print("WALK-FORWARD ANALYSIS")
        print("=" * 60)
        print(f"Parameters:")
        print(f"  Strategy: {self.strategy_name}")
        print(f"  In-sample period: {in_sample_days} days")
        print(f"  Out-of-sample period: {out_sample_days} days")
        print(f"  Step size: {step_days} days")
        print(f"  Optimization trials per window: {n_trials}")

        # Generate day-based windows
        windows = self._generate_day_based_windows(
            start_date, end_date, in_sample_days, out_sample_days, step_days
        )
        print(f"\nGenerated {len(windows)} walk-forward windows")

        results = {
            "parameters": {
                "strategy": self.strategy_name,
                "in_sample_days": in_sample_days,
                "out_sample_days": out_sample_days,
                "step_days": step_days,
                "n_trials": n_trials,
                "min_trades": min_trades,
            },
            "windows": [],
            "summary_stats": {},
        }

        valid_windows = 0
        all_out_sample_returns = []
        all_in_sample_returns = []

        for i, window in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")
            print(f"\nProcessing window {i+1}/{len(windows)}")
            print(
                f"  In-sample: {window['in_sample_start']} to {window['in_sample_end']}"
            )
            print(
                f"  Out-sample: {window['out_sample_start']} to {window['out_sample_end']}"
            )

            try:
                print("  Optimizing parameters...")
                try:
                    # Use strategy name instead of class
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
                    best_params = {
                        "fast_ema_period": 12,
                        "slow_ema_period": 26,
                        "rsi_period": 14,
                        "rsi_upper": 70,
                        "rsi_lower": 30,
                    }
                    print(f"Using default parameters: {best_params}")

                # Check data sufficiency for in-sample and out-of-sample
                in_sample_data = get_data_sync(
                    self.ticker,
                    window["in_sample_start"],
                    window["in_sample_end"],
                    interval=interval,
                )
                out_sample_data = get_data_sync(
                    self.ticker,
                    window["out_sample_start"],
                    window["out_sample_end"],
                    interval=interval,
                )
                min_data_points = (
                    self.strategy_class.get_min_data_points(best_params)
                    if hasattr(self.strategy_class, "get_min_data_points")
                    else 50
                )

                if (
                    len(in_sample_data) < min_data_points
                    or len(out_sample_data) < min_data_points
                ):
                    logger.warning(
                        f"Insufficient data in window {i+1}: in-sample {len(in_sample_data)} rows, out-sample {len(out_sample_data)} rows, required {min_data_points}"
                    )
                    window_result = {
                        "window_id": i + 1,
                        "periods": window,
                        "error": f"Insufficient data: in-sample {len(in_sample_data)} rows, out-sample {len(out_sample_data)} rows, required {min_data_points}",
                        "valid": False,
                    }
                    results["windows"].append(window_result)
                    continue

                print("  Running in-sample backtest...")
                in_sample_results = self._run_backtest(
                    window["in_sample_start"],
                    window["in_sample_end"],
                    interval=interval,
                    **best_params,
                )
                print("  Running out-of-sample backtest...")
                out_sample_results = self._run_backtest(
                    window["out_sample_start"],
                    window["out_sample_end"],
                    interval=interval,
                    **best_params,
                )

                # Log the structure of results for debugging
                logger.debug(
                    f"in_sample_results type: {type(in_sample_results)}, instance: {isinstance(in_sample_results, bt.Strategy)}"
                )
                logger.debug(
                    f"out_sample_results type: {type(out_sample_results)}, instance: {isinstance(out_sample_results, bt.Strategy)}"
                )

                # Extract analyzer names if available
                try:
                    logger.debug(
                        f"in_sample_results analyzers: {[name for name in in_sample_results.analyzers.getnames()]}"
                    )
                    logger.debug(
                        f"out_sample_results analyzers: {[name for name in out_sample_results.analyzers.getnames()]}"
                    )
                except Exception:
                    pass

                try:
                    in_sample_analyzer = PerformanceAnalyzer(in_sample_results)
                    out_sample_analyzer = PerformanceAnalyzer(out_sample_results)
                    in_sample_perf = in_sample_analyzer.generate_full_report()
                    out_sample_perf = out_sample_analyzer.generate_full_report()

                    # Check for errors in performance reports
                    if "error" in in_sample_perf or "error" in out_sample_perf:
                        error_msg = (
                            in_sample_perf.get("error", "Unknown error in in-sample")
                            if "error" in in_sample_perf
                            else out_sample_perf.get(
                                "error", "Unknown error in out-sample"
                            )
                        )
                        logger.warning(
                            f"Performance analysis failed for window {i+1}: {error_msg}"
                        )
                        window_result = {
                            "window_id": i + 1,
                            "periods": window,
                            "error": error_msg,
                            "valid": False,
                        }
                        results["windows"].append(window_result)
                        continue

                    in_trade_analysis = in_sample_perf.get("trade_analysis", {})
                    out_trade_analysis = out_sample_perf.get("trade_analysis", {})
                    in_sample_trades = (
                        in_trade_analysis.get("total_trades", 0)
                        if isinstance(in_trade_analysis, dict)
                        else 0
                    )
                    out_sample_trades = (
                        out_trade_analysis.get("total_trades", 0)
                        if isinstance(out_trade_analysis, dict)
                        else 0
                    )

                    # Store both the summary report and the actual strategy instance for out-of-sample
                    window_result = {
                        "window_id": i + 1,
                        "periods": window,
                        "best_params": best_params,
                        "in_sample_performance": in_sample_perf,
                        "out_sample_performance": out_sample_perf,
                        "out_sample_strategy": out_sample_results,
                        "valid": in_sample_trades >= min_trades
                        and out_sample_trades >= min_trades,
                        "degradation": self._calculate_degradation(
                            in_sample_perf, out_sample_perf
                        ),
                    }

                    if window_result["valid"]:
                        valid_windows += 1
                        in_sample_return = in_sample_perf["summary"].get(
                            "total_return_pct", 0
                        )
                        out_sample_return = out_sample_perf["summary"].get(
                            "total_return_pct", 0
                        )
                        all_in_sample_returns.append(in_sample_return)
                        all_out_sample_returns.append(out_sample_return)
                        print(f"  In-sample return: {in_sample_return:.2f}%")
                        print(f"  Out-sample return: {out_sample_return:.2f}%")
                        print(
                            f"  Trades: {in_sample_trades} (in) / {out_sample_trades} (out)"
                        )
                    else:
                        print(
                            f"  Skipped: Insufficient trades ({in_sample_trades}/{out_sample_trades})"
                        )

                    results["windows"].append(window_result)

                except ValueError as e:
                    logger.warning(
                        f"Performance analysis failed for window {i+1}: {str(e)}"
                    )
                    window_result = {
                        "window_id": i + 1,
                        "periods": window,
                        "error": f"Performance analysis failed: {str(e)}",
                        "valid": False,
                    }
                    results["windows"].append(window_result)
                    continue

            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"Error processing window {i+1}: {str(e)}")
                window_result = {
                    "window_id": i + 1,
                    "periods": window,
                    "error": str(e),
                    "valid": False,
                }
                results["windows"].append(window_result)

        if valid_windows > 0:
            results["summary_stats"] = {
                "total_windows": len(windows),
                "valid_windows": valid_windows,
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
                "correlation": (
                    np.corrcoef(all_in_sample_returns, all_out_sample_returns)[0, 1]
                    if len(all_out_sample_returns) > 1
                    else 0
                ),
                "avg_degradation": (
                    np.mean(
                        [
                            w["degradation"]["return_degradation"]
                            for w in results["windows"]
                            if w["valid"] and "degradation" in w
                        ]
                    )
                    if any(
                        w["valid"] and "degradation" in w for w in results["windows"]
                    )
                    else 0
                ),
                "win_rate_out_sample": (
                    sum(1 for r in all_out_sample_returns if r > 0)
                    / len(all_out_sample_returns)
                    * 100
                    if all_out_sample_returns
                    else 0
                ),
            }

            print(f"\n" + "=" * 40)
            print("WALK-FORWARD SUMMARY")
            print("=" * 40)
            print(f"Valid windows: {valid_windows}/{len(windows)}")
            print(
                f"Average in-sample return: {results['summary_stats']['avg_in_sample_return']:.2f}%"
            )
            print(
                f"Average out-of-sample return: {results['summary_stats']['avg_out_sample_return']:.2f}%"
            )
            print(
                f"Out-of-sample win rate: {results['summary_stats']['win_rate_out_sample']:.1f}%"
            )
            print(f"Return correlation: {results['summary_stats']['correlation']:.3f}")
            print(
                f"Average degradation: {results['summary_stats']['avg_degradation']:.2f}%"
            )

            print("\nGenerating walk-forward report...")
            report = self.generate_walk_forward_report(results)
            self.print_walk_forward_report(report)

            print("\nGenerating best parameters report...")
            self.generate_best_params_report(results)

            # Generate plot
            plot_data = self.plot_walk_forward_results(results)
            if plot_data:
                results["plot"] = plot_data

            self.generate_walk_forward_report(
                results, filename=f"{self.ticker}_wf_report.json"
            )
            self.generate_best_params_report(
                results, filename=f"{self.ticker}_bp_report.json"
            )
        else:
            print(f"\n" + "=" * 40)
            print("WALK-FORWARD SUMMARY")
            print("=" * 40)
            print(f"No valid windows found out of {len(windows)} total windows")
            print("This could be due to:")
            print("- Insufficient data in the date ranges")
            print("- Not enough trades generated by the strategy")
            print("- Technical indicator calculation errors")
            print("\nRecommendations:")
            print("- Use longer date ranges")
            print("- Reduce the minimum trade requirement")
            print("- Check your strategy parameters")

        # NEW: Prepare results for Streamlit UI
        # Simplify the structure for easier rendering in Streamlit
        for window in results["windows"]:
            if "out_sample_performance" in window:
                # Flatten summary metrics
                window["summary"] = window["out_sample_performance"].get("summary", {})
                window["trade_analysis"] = window["out_sample_performance"].get(
                    "trade_analysis", {}
                )
                window["risk_metrics"] = window["out_sample_performance"].get(
                    "risk_metrics", {}
                )

                # Remove full performance report to reduce size
                window.pop("in_sample_performance", None)
                window.pop("out_sample_performance", None)

                # Convert dates to strings for JSON serialization
                for date_key in [
                    "in_sample_start",
                    "in_sample_end",
                    "out_sample_start",
                    "out_sample_end",
                ]:
                    if date_key in window.get("periods", {}):
                        window["periods"][date_key] = str(window["periods"][date_key])

        # Add cumulative equity curve data
        cumulative_equity = 100
        equity_curve = []
        dates = []

        for window in results["windows"]:
            if window.get("valid", False) and "timereturn" in window.get(
                "out_sample_performance", {}
            ):
                equity_data = window["out_sample_performance"]["timereturn"]
                sorted_dates = sorted(equity_data.keys())
                for date in sorted_dates:
                    return_pct = equity_data[date]
                    cumulative_equity *= 1 + return_pct / 100
                    dates.append(date)
                    equity_curve.append(cumulative_equity)

        if equity_curve:
            results["cumulative_equity"] = {"dates": dates, "values": equity_curve}

        return results

    def _generate_day_based_windows(
        self,
        start_date: str,
        end_date: str,
        in_sample_days: int,
        out_sample_days: int,
        step_days: int,
    ) -> List[Dict]:
        """Generate day-based windows for walk-forward analysis."""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        windows = []
        window_id = 1

        current_start = start_dt
        while current_start < end_dt:
            # Calculate window boundaries
            in_sample_end = current_start + timedelta(days=in_sample_days)
            out_sample_end = in_sample_end + timedelta(days=out_sample_days)

            # Ensure windows don't exceed data range
            if out_sample_end > end_dt:
                break

            windows.append(
                {
                    "window_id": window_id,
                    "in_sample_start": current_start.strftime("%Y-%m-%d"),
                    "in_sample_end": in_sample_end.strftime("%Y-%m-%d"),
                    "out_sample_start": in_sample_end.strftime("%Y-%m-%d"),
                    "out_sample_end": out_sample_end.strftime("%Y-%m-%d"),
                }
            )

            window_id += 1
            current_start += timedelta(days=step_days)

        return windows

    def _run_backtest(
        self, start_date: str, end_date: str, interval: str = "5m", **strategy_params
    ):
        """Run a backtest with given parameters.

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            **strategy_params: Strategy-specific parameters.

        Returns:
            Backtrader strategy instance.

        Raises:
            ValueError: If data is insufficient or backtest fails.
        """
        logger.info(
            f"Running backtest from {start_date} to {end_date} with params {strategy_params}"
        )
        try:
            data_df = get_data_sync(
                self.ticker, start_date, end_date, interval=interval
            )
            if data_df is None or data_df.empty:
                raise ValueError(
                    f"No data available for {self.ticker} from {start_date} to {end_date}"
                )

            min_data_points = (
                self.strategy_class.get_min_data_points(strategy_params)
                if hasattr(self.strategy_class, "get_min_data_points")
                else 50
            )
            if len(data_df) < min_data_points:
                raise ValueError(
                    f"Insufficient data: {len(data_df)} rows available, {min_data_points} required"
                )

            data = bt.feeds.PandasData(
                dataname=data_df,
                datetime=None,
                open="Open",
                high="High",
                low="Low",
                close="Close",
                volume="Volume",
                openinterest=None,
            )

            cerebro = bt.Cerebro()
            cerebro.addstrategy(self.strategy_class, **strategy_params)
            # Add 5-minute data
            cerebro.adddata(data, name="5m")
            # Add 15-minute resampled data
            cerebro.resampledata(
                data, timeframe=bt.TimeFrame.Minutes, compression=15, name="15m"
            )
            cerebro.broker.setcash(self.initial_cash)
            cerebro.broker.setcommission(commission=self.commission)
            # Add analyzers from self.analyzers if provided, else use defaults
            if self.analyzers:
                for analyzer, kwargs in self.analyzers:
                    cerebro.addanalyzer(analyzer, **kwargs)
            else:
                cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
                cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
                cerebro.addanalyzer(
                    bt.analyzers.Returns,
                    _name="returns",
                    timeframe=bt.TimeFrame.Minutes,
                    compression=5,
                )
                cerebro.addanalyzer(
                    bt.analyzers.SharpeRatio,
                    _name="sharpe",
                    timeframe=bt.TimeFrame.Minutes,
                    compression=5,
                )
                cerebro.addanalyzer(
                    bt.analyzers.TimeReturn,
                    _name="timereturn",
                    timeframe=bt.TimeFrame.Minutes,
                    compression=5,
                )
                cerebro.addanalyzer(SortinoRatio, _name="sortino")
                try:
                    cerebro.addanalyzer(
                        bt.analyzers.Calmar,
                        _name="calmar",
                        timeframe=bt.TimeFrame.Minutes,
                        compression=5,
                    )
                except Exception:
                    pass

            print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
            results = cerebro.run()
            print(f"Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}")

            # Defensive check for results
            if not isinstance(results, list) or not results:
                logger.error("Backtest returned invalid results: empty or not a list")
                raise ValueError("Backtest failed: no valid strategy instance returned")

            strategy_instance = results[0]
            logger.debug(
                f"Strategy instance type: {type(strategy_instance)}, analyzers: {[name for name in strategy_instance.analyzers.getnames()]}"
            )
            return strategy_instance

        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            raise

    def _calculate_degradation(
        self, in_sample_perf: Dict, out_sample_perf: Dict
    ) -> Dict:
        """Calculate performance degradation between in-sample and out-of-sample results.

        Args:
            in_sample_perf (Dict): In-sample performance metrics.
            out_sample_perf (Dict): Out-of-sample performance metrics.

        Returns:
            Dict: Degradation metrics.
        """
        try:
            in_return = in_sample_perf["summary"].get("total_return_pct", 0)
            out_return = out_sample_perf["summary"].get("total_return_pct", 0)
            in_sharpe = in_sample_perf["summary"].get("sharpe_ratio", 0)
            out_sharpe = out_sample_perf["summary"].get("sharpe_ratio", 0)

            # Ensure values are floats for calculation
            try:
                in_return = float(in_return)
            except Exception:
                in_return = 0.0
            try:
                out_return = float(out_return)
            except Exception:
                out_return = 0.0
            try:
                in_sharpe = float(in_sharpe)
            except Exception:
                in_sharpe = 0.0
            try:
                out_sharpe = float(out_sharpe)
            except Exception:
                out_sharpe = 0.0

            return {
                "return_degradation": in_return - out_return,
                "sharpe_degradation": (
                    in_sharpe - out_sharpe
                    if in_sharpe is not None and out_sharpe is not None
                    else None
                ),
            }
        except Exception as e:
            logger.error(f"Error calculating degradation: {str(e)}")
            return {"return_degradation": 0, "sharpe_degradation": None}

    def _print_in_out_comparison(self, results: Dict):
        """Print a comparison of in-sample and out-of-sample performance."""
        print("\n" + "=" * 40)
        print("PERFORMANCE COMPARISON")
        print("=" * 40)

        # Extract summary dictionaries safely
        in_summary = results["in_sample_performance"].get("summary", {})
        out_summary = results["out_sample_performance"].get("summary", {})
        degradation = results.get("performance_degradation", {})

        # Convert values to float safely
        in_return = self._safe_float(in_summary.get("total_return_pct", 0))
        in_annual = self._safe_float(in_summary.get("annual_return_pct", 0))
        in_drawdown = self._safe_float(in_summary.get("max_drawdown_pct", 0))
        in_sharpe = self._safe_float(in_summary.get("sharpe_ratio", 0))

        out_return = self._safe_float(out_summary.get("total_return_pct", 0))
        out_annual = self._safe_float(out_summary.get("annual_return_pct", 0))
        out_drawdown = self._safe_float(out_summary.get("max_drawdown_pct", 0))
        out_sharpe = self._safe_float(out_summary.get("sharpe_ratio", 0))

        return_degradation = self._safe_float(degradation.get("return_degradation", 0))
        sharpe_degradation = self._safe_float(degradation.get("sharpe_degradation", 0))

        print(
            f"\nIn-sample ({results['in_sample_period'][0]} to {results['in_sample_period'][1]}):"
        )
        print(f"  Total Return: {in_return:.2f}%")
        print(f"  Annualized Return: {in_annual:.2f}%")
        print(f"  Max Drawdown: {in_drawdown:.2f}%")
        print(f"  Sharpe Ratio: {in_sharpe:.2f}")

        print(
            f"\nOut-of-sample ({results['out_sample_period'][0]} to {results['out_sample_period'][1]}):"
        )
        print(f"  Total Return: {out_return:.2f}%")
        print(f"  Annualized Return: {out_annual:.2f}%")
        print(f"  Max Drawdown: {out_drawdown:.2f}%")
        print(f"  Sharpe Ratio: {out_sharpe:.2f}")

        print(f"\nPerformance Degradation:")
        print(f"  Return Degradation: {return_degradation:.2f}%")
        print(f"  Sharpe Degradation: {sharpe_degradation:.2f}")
        print("=" * 40)
