import json
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import warnings

from .data import get_data
from .reports import PerformanceAnalyzer
from .parameter_optimization import optimize_strategy

warnings.filterwarnings("ignore")


class ValidationAnalyzer:
    """
    Comprehensive validation analysis including in-sample, out-of-sample,
    and walk-forward analysis for trading strategies.
    """

    def __init__(
        self, strategy_class, ticker="AAPL", initial_cash=100000.0, commission=0.00
    ):
        self.strategy_class = strategy_class
        self.ticker = ticker
        self.initial_cash = initial_cash
        self.commission = commission

        # Minimum data requirements for indicators (will be set dynamically)
        self.min_data_points = 100  # Will be determined per strategy/params

    def _get_required_min_data_points(self, **strategy_params):
        try:
            # Create dummy data with 1000 rows of non-zero values
            dummy_data = pd.DataFrame(
                {
                    "Open": 100.0 + np.random.normal(0, 0.1, 1000),
                    "High": 101.0 + np.random.normal(0, 0.1, 1000),
                    "Low": 99.0 + np.random.normal(0, 0.1, 1000),
                    "Close": 100.0 + np.random.normal(0, 0.1, 1000),
                    "Volume": 10000.0 + np.random.normal(0, 100, 1000),
                },
                index=pd.date_range(start="2000-01-01", periods=1000, freq="D"),
            )

            data = bt.feeds.PandasData(dataname=dummy_data)
            cerebro = bt.Cerebro()
            cerebro.adddata(data)
            cerebro.addstrategy(self.strategy_class, **strategy_params)

            # Run with minimal preloading to avoid indicator calculations
            strategies = cerebro.run(
                preload=False, runonce=False, stdstats=False, maxcpus=1
            )

            if not strategies:
                print("No strategies were run. Using fallback value.")
                return 50

            strat_instance = strategies[0]

            # Collect all indicators
            indicators = []
            for attr_name in dir(strat_instance):
                attr = getattr(strat_instance, attr_name)
                if isinstance(attr, bt.Indicator):
                    indicators.append(attr)

            if not indicators:
                print("No indicators found in the strategy. Using fallback value.")
                return 50

            max_period = 0
            for ind in indicators:
                try:
                    # Get period from various possible locations
                    period_val = None
                    if hasattr(ind, "period"):
                        period_val = ind.period
                    elif hasattr(ind, "params") and hasattr(ind.params, "period"):
                        period_val = ind.params.period
                    elif hasattr(ind, "p") and hasattr(ind.p, "period"):
                        period_val = ind.p.period

                    if period_val is not None:
                        period_val = int(period_val)
                        max_period = max(max_period, period_val)
                except Exception as e:
                    print(f"Error accessing period for indicator {ind}: {e}")
                    continue

            if max_period == 0:
                print("No valid periods found for indicators. Using fallback value.")
                return 50

            return max(50, max_period + 20)  # Buffer of 20, minimum 50

        except Exception as e:
            print(f"Dynamic period detection failed: {e}. Using fallback 50")
            return 50

    def _validate_data_length(self, start_date, end_date, **strategy_params):
        min_data_points = self._get_required_min_data_points(**strategy_params)
        data_df = get_data(self.ticker, start_date, end_date)
        data_length = len(data_df)

        if data_length < min_data_points:
            print(f"  Required: {min_data_points} bars, Available: {data_length} bars")
            print("  Solution: Increase period length or reduce indicator periods")
            return False
        return True

    def in_sample_out_sample_analysis(
        self,
        start_date: str,
        end_date: str,
        split_ratio: float = 0.7,
        optimize_in_sample: bool = True,
        n_trials: int = 50,
    ) -> Dict:
        """
        Perform in-sample and out-of-sample analysis.

        Parameters:
        start_date (str): Start date for analysis
        end_date (str): End date for analysis
        split_ratio (float): Ratio for in-sample period (0.7 = 70% in-sample, 30% out-of-sample)
        optimize_in_sample (bool): Whether to optimize parameters on in-sample data
        n_trials (int): Number of optimization trials

        Returns:
        Dict: Analysis results
        """

        print("=" * 60)
        print("IN-SAMPLE / OUT-OF-SAMPLE ANALYSIS")
        print("=" * 60)

        # Calculate split date
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

        # Validate data lengths
        if not self._validate_data_length(start_date, split_date_str):
            raise ValueError("Insufficient data for in-sample period")

        if not self._validate_data_length(split_date_str, end_date):
            raise ValueError("Insufficient data for out-of-sample period")

        results = {
            "split_date": split_date_str,
            "in_sample_period": (start_date, split_date_str),
            "out_sample_period": (split_date_str, end_date),
            "split_ratio": split_ratio,
        }

        if optimize_in_sample:
            print(f"\nOptimizing parameters on in-sample data ({n_trials} trials)...")

            # Optimize on in-sample data
            optimization_results = optimize_strategy(
                strategy_class=self.strategy_class,
                ticker=self.ticker,
                start_date=start_date,
                end_date=split_date_str,
                n_trials=n_trials,
                initial_cash=self.initial_cash,
                commission=self.commission,
            )

            best_params = optimization_results["best_params"]
            print(f"Best parameters found: {best_params}")

            # Test on in-sample data with optimized parameters
            print("\nTesting optimized parameters on in-sample data...")
            in_sample_results = self._run_backtest(
                start_date, split_date_str, **best_params
            )

            # Test on out-of-sample data with same parameters
            print("Testing optimized parameters on out-of-sample data...")
            out_sample_results = self._run_backtest(
                split_date_str, end_date, **best_params
            )

            results["optimized_params"] = best_params
            results["optimization_results"] = optimization_results

        else:
            print("\nUsing default parameters for both periods...")

            # Use default parameters
            in_sample_results = self._run_backtest(start_date, split_date_str)
            out_sample_results = self._run_backtest(split_date_str, end_date)

            results["optimized_params"] = None

        # Analyze results
        in_sample_analyzer = PerformanceAnalyzer(in_sample_results)
        out_sample_analyzer = PerformanceAnalyzer(out_sample_results)

        results["in_sample_performance"] = in_sample_analyzer.generate_full_report()
        results["out_sample_performance"] = out_sample_analyzer.generate_full_report()

        # Calculate degradation metrics
        results["performance_degradation"] = self._calculate_degradation(
            results["in_sample_performance"], results["out_sample_performance"]
        )

        # Print comparison
        self._print_in_out_comparison(results)

        return results

    def generate_walk_forward_report(self, wf_results, filename=None):
        """
        Generate a comprehensive report for walk-forward analysis.

        Parameters:
        wf_results (Dict): Walk-forward analysis results dictionary
        filename (str, optional): If provided, save the report to this file

        Returns:
        Dict: Structured report containing overview and per-window details
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
                print(f"Walk-forward report saved to {filename}")
            except Exception as e:
                print(f"Error saving report: {str(e)}")

        return report

    def print_walk_forward_report(self, report):
        """
        Print a formatted walk-forward analysis report to the console.

        Parameters:
        report (Dict): The report generated by generate_walk_forward_report
        """
        print("=" * 60)
        print("WALK-FORWARD ANALYSIS REPORT")
        print("=" * 60)

        # Overview Section
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

        # Per-Window Details
        print("\nPER WINDOW DETAILS")
        print("-" * 30)
        for window in report["windows"]:
            print(f"Window {window['window_id']}:")
            print(f"  In-sample period: {window['in_sample_period']}")
            print(f"  Out-sample period: {window['out_sample_period']}")
            print(f"  Best parameters: {window['best_params']}")
            in_perf = window["in_sample_performance"]
            out_perf = window["out_sample_performance"]
            print(f"  In-sample return: {in_perf.get('total_return_pct', 0):.2f}%")
            print(f"  Out-sample return: {out_perf.get('total_return_pct', 0):.2f}%")
            print(
                f"  Degradation: {window['degradation'].get('return_degradation', 0):.2f}%"
            )
            print()

        print("=" * 60)

    def walk_forward_analysis(
        self,
        start_date: str,
        end_date: str,
        in_sample_months: int = 12,
        out_sample_months: int = 3,
        step_months: int = 1,
        n_trials: int = 20,
        min_trades: int = 5,
    ) -> Dict:
        """
        Perform walk-forward analysis.

        Parameters:
        start_date (str): Start date for analysis
        end_date (str): End date for analysis
        in_sample_months (int): Length of in-sample period in months
        out_sample_months (int): Length of out-of-sample period in months
        step_months (int): Step size between windows in months
        n_trials (int): Number of optimization trials per window
        min_trades (int): Minimum trades required for valid results

        Returns:
        Dict: Walk-forward analysis results
        """

        print("=" * 60)
        print("WALK-FORWARD ANALYSIS")
        print("=" * 60)

        print(f"Parameters:")
        print(f"  In-sample period: {in_sample_months} months")
        print(f"  Out-of-sample period: {out_sample_months} months")
        print(f"  Step size: {step_months} months")
        print(f"  Optimization trials per window: {n_trials}")

        # Generate windows
        windows = self._generate_walk_forward_windows(
            start_date, end_date, in_sample_months, out_sample_months, step_months
        )

        print(f"\nGenerated {len(windows)} walk-forward windows")

        results = {
            "parameters": {
                "in_sample_months": in_sample_months,
                "out_sample_months": out_sample_months,
                "step_months": step_months,
                "n_trials": n_trials,
                "min_trades": min_trades,
                "min_data_points": self.min_data_points,
            },
            "windows": [],
            "summary_stats": {},
        }

        valid_windows = 0
        all_out_sample_returns = []
        all_in_sample_returns = []

        for i, window in enumerate(windows):
            print(f"\nProcessing window {i+1}/{len(windows)}")
            print(
                f"  In-sample: {window['in_sample_start']} to {window['in_sample_end']}"
            )
            print(
                f"  Out-sample: {window['out_sample_start']} to {window['out_sample_end']}"
            )

            try:
                # Validate in-sample data length first (no params yet)
                if not self._validate_data_length(
                    window["in_sample_start"], window["in_sample_end"]
                ):
                    print("  Skipped: Insufficient in-sample data")
                    window_result = {
                        "window_id": i + 1,
                        "periods": window,
                        "error": "Insufficient in-sample data",
                        "valid": False,
                    }
                    results["windows"].append(window_result)
                    continue

                # Optimize on in-sample data
                print("  Optimizing parameters...")
                optimization_results = optimize_strategy(
                    strategy_class=self.strategy_class,
                    ticker=self.ticker,
                    start_date=window["in_sample_start"],
                    end_date=window["in_sample_end"],
                    n_trials=n_trials,
                    initial_cash=self.initial_cash,
                    commission=self.commission,
                )

                best_params = optimization_results["best_params"]

                # Validate out-of-sample data length with best_params
                if not self._validate_data_length(
                    window["out_sample_start"], window["out_sample_end"], **best_params
                ):
                    print("  Skipped: Insufficient out-of-sample data")
                    window_result = {
                        "window_id": i + 1,
                        "periods": window,
                        "error": "Insufficient out-of-sample data",
                        "valid": False,
                    }
                    results["windows"].append(window_result)
                    continue

                # Test on in-sample data
                print("  Running in-sample backtest...")
                in_sample_results = self._run_backtest(
                    window["in_sample_start"], window["in_sample_end"], **best_params
                )

                # Test on out-of-sample data
                print("  Running out-of-sample backtest...")
                out_sample_results = self._run_backtest(
                    window["out_sample_start"], window["out_sample_end"], **best_params
                )

                # Analyze results
                in_sample_analyzer = PerformanceAnalyzer(in_sample_results)
                out_sample_analyzer = PerformanceAnalyzer(out_sample_results)

                in_sample_perf = in_sample_analyzer.generate_full_report()
                out_sample_perf = out_sample_analyzer.generate_full_report()

                # Check if results are valid (enough trades)
                # Defensive: check if performance reports are dicts
                if not isinstance(in_sample_perf, dict):
                    print(
                        "  Warning: in_sample_perf is not a dict, treating as 0 trades."
                    )
                    in_sample_trades = 0
                else:
                    in_trade_analysis = in_sample_perf.get("trade_analysis", {})
                    if not isinstance(in_trade_analysis, dict):
                        print(
                            "  Warning: in_sample trade_analysis is not a dict, treating as 0 trades."
                        )
                        in_sample_trades = 0
                    else:
                        in_sample_trades = in_trade_analysis.get("total_trades", 0)
                if not isinstance(out_sample_perf, dict):
                    print(
                        "  Warning: out_sample_perf is not a dict, treating as 0 trades."
                    )
                    out_sample_trades = 0
                else:
                    out_trade_analysis = out_sample_perf.get("trade_analysis", {})
                    if not isinstance(out_trade_analysis, dict):
                        print(
                            "  Warning: out_sample trade_analysis is not a dict, treating as 0 trades."
                        )
                        out_sample_trades = 0
                    else:
                        out_sample_trades = out_trade_analysis.get("total_trades", 0)

                if in_sample_trades >= min_trades and out_sample_trades >= min_trades:
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

                # Store window results
                window_result = {
                    "window_id": i + 1,
                    "periods": window,
                    "best_params": best_params,
                    "in_sample_performance": in_sample_perf,
                    "out_sample_performance": out_sample_perf,
                    "valid": in_sample_trades >= min_trades
                    and out_sample_trades >= min_trades,
                    "degradation": self._calculate_degradation(
                        in_sample_perf, out_sample_perf
                    ),
                }

                results["windows"].append(window_result)

            except Exception as e:
                import traceback

                traceback.print_exc()  # Print full traceback for debugging
                print(f"  Error processing window: {str(e)}")
                # Don't print full traceback for each window to avoid spam

                window_result = {
                    "window_id": i + 1,
                    "periods": window,
                    "error": str(e),
                    "valid": False,
                }
                results["windows"].append(window_result)

        # Calculate summary statistics
        if valid_windows > 0:
            results["summary_stats"] = {
                "total_windows": len(windows),
                "valid_windows": valid_windows,
                "avg_in_sample_return": np.mean(all_in_sample_returns),
                "avg_out_sample_return": np.mean(all_out_sample_returns),
                "std_in_sample_return": np.std(all_in_sample_returns),
                "std_out_sample_return": np.std(all_out_sample_returns),
                "correlation": (
                    np.corrcoef(all_in_sample_returns, all_out_sample_returns)[0, 1]
                    if len(all_in_sample_returns) > 1
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

            # Existing summary print (can be kept or replaced)
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

            # Generate and print the detailed report
            print("\nGenerating walk-forward report...")
            report = self.generate_walk_forward_report(results)
            self.print_walk_forward_report(report)

            # Prompt user to save the report
            save_report = (
                input("Save walk-forward report to file? (y/n): ").lower().strip()
            )
            if save_report == "y":
                filename = (
                    input("Enter filename (default: ticker_wf_report.json): ").strip()
                    or f"{self.ticker}_wf_report.json"
                )
                self.generate_walk_forward_report(results, filename=filename)
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

        return results

    def _run_backtest(self, start_date: str, end_date: str, **strategy_params):
        """Run a backtest with given parameters."""

        # Get data with validation
        min_data_points = self._get_required_min_data_points(**strategy_params)
        data_df = get_data(self.ticker, start_date, end_date)
        if len(data_df) < min_data_points:
            raise ValueError(
                f"Insufficient data: {len(data_df)} points (need {min_data_points})"
            )

        # Create Backtrader data feed with proper column mapping
        data = bt.feeds.PandasData(
            dataname=data_df,
            datetime=None,  # Use index as datetime
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            openinterest=None,
        )

        # Create Cerebro engine
        cerebro = bt.Cerebro()
        cerebro.addstrategy(self.strategy_class, **strategy_params)
        cerebro.adddata(data)
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.Calmar, _name="calmar")
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
        cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")

        # Run backtest with robust error handling for indicator/data length issues
        try:
            results = cerebro.run()
            return results
        except IndexError as e:
            print(
                f"  Error: {str(e)}\n  Likely cause: Not enough data for one or more indicators in this window.\n  Suggestion: Increase min_data_points or check your indicator periods."
            )
            return None
        except Exception as e:
            print(f"  Unexpected error during backtest: {str(e)}")
            return None

    def _generate_walk_forward_windows(
        self,
        start_date: str,
        end_date: str,
        in_sample_months: int,
        out_sample_months: int,
        step_months: int,
    ) -> List[Dict]:
        """Generate walk-forward analysis windows."""

        windows = []
        current_start = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        while True:
            # Calculate in-sample period
            in_sample_end = current_start + relativedelta(months=in_sample_months)

            # Calculate out-of-sample period
            out_sample_start = in_sample_end
            out_sample_end = out_sample_start + relativedelta(months=out_sample_months)

            # Check if we have enough data
            if out_sample_end > end_dt:
                break

            window = {
                "in_sample_start": current_start.strftime("%Y-%m-%d"),
                "in_sample_end": in_sample_end.strftime("%Y-%m-%d"),
                "out_sample_start": out_sample_start.strftime("%Y-%m-%d"),
                "out_sample_end": out_sample_end.strftime("%Y-%m-%d"),
            }

            windows.append(window)

            # Move to next window
            current_start = current_start + relativedelta(months=step_months)

        return windows

    def _calculate_degradation(
        self, in_sample_perf: Dict, out_sample_perf: Dict
    ) -> Dict:
        """Calculate performance degradation from in-sample to out-of-sample."""

        try:
            in_return = in_sample_perf["summary"].get("total_return_pct", 0)
            out_return = out_sample_perf["summary"].get("total_return_pct", 0)

            in_sharpe = in_sample_perf["summary"].get("sharpe_ratio", 0)
            out_sharpe = out_sample_perf["summary"].get("sharpe_ratio", 0)

            in_drawdown = in_sample_perf["summary"].get("max_drawdown_pct", 0)
            out_drawdown = out_sample_perf["summary"].get("max_drawdown_pct", 0)

            degradation = {
                "return_degradation": in_return - out_return,
                "sharpe_degradation": in_sharpe - out_sharpe,
                "drawdown_increase": out_drawdown - in_drawdown,
                "return_ratio": out_return / in_return if in_return != 0 else 0,
                "sharpe_ratio": out_sharpe / in_sharpe if in_sharpe != 0 else 0,
            }

        except Exception as e:
            degradation = {"error": f"Error calculating degradation: {str(e)}"}

        return degradation

    def _print_in_out_comparison(self, results: Dict):
        """Print comparison between in-sample and out-of-sample results."""

        print("\n" + "=" * 50)
        print("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
        print("=" * 50)

        in_perf = results["in_sample_performance"]["summary"]
        out_perf = results["out_sample_performance"]["summary"]
        degradation = results["performance_degradation"]

        if (
            "error" not in in_perf
            and "error" not in out_perf
            and "error" not in degradation
        ):
            print(
                f"{'Metric':<25} {'In-Sample':<15} {'Out-Sample':<15} {'Degradation':<15}"
            )
            print("-" * 70)
            print(
                f"{'Total Return (%)':<25} {in_perf['total_return_pct']:<15.2f} {out_perf['total_return_pct']:<15.2f} {degradation['return_degradation']:<15.2f}"
            )
            print(
                f"{'Sharpe Ratio':<25} {in_perf['sharpe_ratio']:<15.3f} {out_perf['sharpe_ratio']:<15.3f} {degradation['sharpe_degradation']:<15.3f}"
            )
            print(
                f"{'Max Drawdown (%)':<25} {in_perf['max_drawdown_pct']:<15.2f} {out_perf['max_drawdown_pct']:<15.2f} {degradation['drawdown_increase']:<15.2f}"
            )
            print(
                f"{'Final Value':<25} {in_perf['final_value']:<15,.0f} {out_perf['final_value']:<15,.0f} {out_perf['final_value']-in_perf['final_value']:<15,.0f}"
            )

            print(f"\nPerformance Ratios:")
            print(f"  Return Ratio (Out/In): {degradation['return_ratio']:.3f}")
            print(f"  Sharpe Ratio (Out/In): {degradation['sharpe_ratio']:.3f}")
        else:
            print(
                "Error in performance comparison - unable to display detailed metrics"
            )

    def plot_walk_forward_results(
        self, wf_results: Dict, save_path: Optional[str] = None
    ):
        """Plot walk-forward analysis results."""

        if not wf_results["windows"]:
            print("No walk-forward results to plot")
            return

        # Extract data for plotting
        valid_windows = [w for w in wf_results["windows"] if w["valid"]]

        if not valid_windows:
            print("No valid windows to plot")
            return

        window_ids = [w["window_id"] for w in valid_windows]
        in_sample_returns = [
            w["in_sample_performance"]["summary"]["total_return_pct"]
            for w in valid_windows
        ]
        out_sample_returns = [
            w["out_sample_performance"]["summary"]["total_return_pct"]
            for w in valid_windows
        ]
        degradations = [
            w["degradation"]["return_degradation"]
            for w in valid_windows
            if "degradation" in w
        ]

        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Walk-Forward Analysis Results - {self.ticker}",
            fontsize=16,
            fontweight="bold",
        )

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
        if degradations:
            axes[0, 1].bar(
                window_ids[: len(degradations)], degradations, color="orange", alpha=0.7
            )
            axes[0, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
            axes[0, 1].set_xlabel("Window ID")
            axes[0, 1].set_ylabel("Degradation (%)")
            axes[0, 1].set_title("Performance Degradation per Window")
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "No degradation data available",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )

        # Plot 3: Scatter plot of in-sample vs out-of-sample
        axes[1, 0].scatter(in_sample_returns, out_sample_returns, alpha=0.7, s=60)

        # Add diagonal line (perfect correlation)
        if in_sample_returns and out_sample_returns:
            min_val = min(min(in_sample_returns), min(out_sample_returns))
            max_val = max(max(in_sample_returns), max(out_sample_returns))
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
        if out_sample_returns:
            cumulative_out_sample = (
                np.cumprod(1 + np.array(out_sample_returns) / 100) - 1
            )
            axes[1, 1].plot(
                window_ids,
                cumulative_out_sample * 100,
                "g-o",
                linewidth=2,
                markersize=4,
            )
            axes[1, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
            axes[1, 1].set_xlabel("Window ID")
            axes[1, 1].set_ylabel("Cumulative Return (%)")
            axes[1, 1].set_title("Cumulative Out-of-Sample Returns")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No return data available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()


# Example usage functions
def run_validation_analysis(
    strategy_class, ticker="AAPL", start_date="2020-01-01", end_date="2025-06-01"
):
    """
    Run a comprehensive validation analysis including both in-sample/out-of-sample
    and walk-forward analysis.
    """

    validator = ValidationAnalyzer(strategy_class, ticker)

    print("Starting comprehensive validation analysis...")

    # 1. In-sample / Out-of-sample analysis
    print("\n" + "=" * 80)
    print("PHASE 1: IN-SAMPLE / OUT-OF-SAMPLE ANALYSIS")
    print("=" * 80)

    is_oos_results = validator.in_sample_out_sample_analysis(
        start_date=start_date,
        end_date=end_date,
        split_ratio=0.7,
        optimize_in_sample=True,
        n_trials=30,
    )

    # 2. Walk-forward analysis
    print("\n" + "=" * 80)
    print("PHASE 2: WALK-FORWARD ANALYSIS")
    print("=" * 80)

    wf_results = validator.walk_forward_analysis(
        start_date=start_date,
        end_date=end_date,
        in_sample_months=18,
        out_sample_months=6,
        step_months=3,
        n_trials=20,
        min_trades=3,
    )

    # 3. Plot results
    print("\n" + "=" * 80)
    print("PHASE 3: VISUALIZATION")
    print("=" * 80)

    validator.plot_walk_forward_results(wf_results)

    return {
        "in_sample_out_sample": is_oos_results,
        "walk_forward": wf_results,
        "validator": validator,
    }


print("\nValidation analysis completed!")
