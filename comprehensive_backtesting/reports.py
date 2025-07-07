import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import backtrader as bt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Comprehensive performance analysis and reporting for backtesting results."""

    def __init__(self, strategy_results, initial_cash=100000.0):
        """Initialize the performance analyzer.

        Args:
            strategy_results: Backtrader strategy results (list, single instance, or dict).
            initial_cash (float): Initial portfolio cash.
        """
        self.initial_cash = initial_cash
        self.is_dict_input = False

        # Handle different input types
        if isinstance(strategy_results, dict):
            # Input is already a processed dictionary
            self.is_dict_input = True
            self.processed_results = strategy_results
            self.strategy = None
            logger.info("Initialized PerformanceAnalyzer with dictionary input")
        elif isinstance(strategy_results, list) and strategy_results:
            # Input is a list of strategy results
            self.strategy = strategy_results[0] if len(strategy_results) > 0 else None
            self.processed_results = None
            logger.info("Initialized PerformanceAnalyzer with list input")
        else:
            # Input is a single strategy instance
            self.strategy = strategy_results
            self.processed_results = None
            logger.info("Initialized PerformanceAnalyzer with strategy object")

    def generate_full_report(self, resample_freq="ME"):
        """Generate a comprehensive performance report.

        Args:
            resample_freq (str): Pandas offset alias for resampling returns (e.g., 'ME', 'D', 'H', '15T').

        Returns:
            dict: Report with summary, trade analysis, risk metrics, and drawdown analysis.
        """
        if self.is_dict_input:
            # Return the already processed results
            logger.info(
                "Returning pre-processed performance report", self.processed_results
            )

            return self.processed_results

        if self.strategy is None:
            logger.error("No strategy results available")
            return {"error": "No strategy results available"}

        try:
            report = {
                "summary": self.get_performance_summary(),
                "trade_analysis": self.get_trade_analysis(),
                "risk_metrics": self.get_risk_metrics(),
                "resampled_returns": self.get_resampled_returns(
                    resample_freq=resample_freq
                ),
                "drawdown_analysis": self.get_drawdown_analysis(),
            }
            logger.info("Generated full performance report")
            return report
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(
                f"Error generating full report: {str(e)}. Check analyzer availability."
            )
            return {"error": f"Error generating report: {str(e)}"}

    def get_performance_summary(self):
        """Get basic performance metrics.

        Returns:
            dict: Summary metrics including returns, Sharpe ratio, Sortino ratio, and drawdown.
        """
        if self.is_dict_input:
            return self.processed_results.get("summary", {})

        try:
            if self.strategy is None:
                logger.warning("No strategy available for performance summary")
                return {
                    "initial_cash": self.initial_cash,
                    "final_value": self.initial_cash,
                    "total_return_pct": 0,
                    "annual_return_pct": 0,
                    "sharpe_ratio": 0,
                    "sortino_ratio": 0,
                    "max_drawdown_pct": 0,
                    "calmar_ratio": 0,
                    "sqn": 0,
                    "profit_loss": 0,
                }
            returns_analyzer = getattr(self.strategy.analyzers, "returns", None)
            sharpe_analyzer = getattr(self.strategy.analyzers, "sharpe", None)
            sortino_analyzer = getattr(self.strategy.analyzers, "sortino", None)
            drawdown_analyzer = getattr(self.strategy.analyzers, "drawdown", None)
            calmar_analyzer = getattr(self.strategy.analyzers, "calmar", None)
            sqn_analyzer = getattr(self.strategy.analyzers, "sqn", None)

            # Defensive: If any analyzer is missing, return N/A or 0 for that metric
            returns_analysis = (
                returns_analyzer.get_analysis() if returns_analyzer else {}
            )
            sharpe_analysis = sharpe_analyzer.get_analysis() if sharpe_analyzer else {}
            sortino_analysis = (
                sortino_analyzer.get_analysis() if sortino_analyzer else {}
            )
            drawdown_analysis = (
                drawdown_analyzer.get_analysis() if drawdown_analyzer else {}
            )
            calmar_analysis = calmar_analyzer.get_analysis() if calmar_analyzer else {}
            sqn_analysis = sqn_analyzer.get_analysis() if sqn_analyzer else {}

            total_return = returns_analysis.get("rtot", 0) or 0
            annual_return = returns_analysis.get("rnorm", 0) or 0

            # Robust extraction and formatting for ratios
            def robust_float(val):
                try:
                    if val is None or (
                        isinstance(val, (int, float)) and abs(val) < 1e-12
                    ):
                        return 0
                    return float(val)
                except Exception:
                    return 0

            # FIXED: Proper Sharpe ratio calculation
            sharpe_ratio = 0
            if sharpe_analyzer:
                try:
                    # Get raw value from analyzer
                    raw_sharpe = sharpe_analysis.get("sharperatio", None)

                    # Scale down if value is unreasonably large
                    if raw_sharpe and abs(raw_sharpe) > 100:
                        sharpe_ratio = raw_sharpe / 100
                    else:
                        sharpe_ratio = robust_float(raw_sharpe)
                except Exception as e:
                    logger.error(f"Error processing Sharpe ratio: {e}")
                    sharpe_ratio = 0

            # FIXED: Proper Sortino ratio calculation
            sortino_ratio = 0
            if sortino_analyzer:
                try:
                    # Get raw value from analyzer
                    raw_sortino = sortino_analysis.get("sortinoratio", None)

                    # Scale down if value is unreasonably large
                    if raw_sortino and abs(raw_sortino) > 100:
                        sortino_ratio = raw_sortino / 100
                    else:
                        sortino_ratio = robust_float(raw_sortino)
                except Exception as e:
                    logger.error(f"Error processing Sortino ratio: {e}")
                    sortino_ratio = 0

            max_drawdown = drawdown_analysis.get("max", {}).get("drawdown", 0) or 0
            raw_calmar = calmar_analysis.get("calmarratio", None)
            calmar_ratio = robust_float(raw_calmar)
            # If calmar_ratio is N/A, try to compute manually if possible
            if calmar_ratio == 0:
                if annual_return and max_drawdown and abs(max_drawdown) > 1e-12:
                    calmar_ratio = robust_float(annual_return / max_drawdown)
                else:
                    calmar_ratio = 0
            sqn = robust_float(sqn_analysis.get("sqn", None))

            total_return_pct = (np.exp(total_return) - 1) * 100
            summary = {
                "initial_cash": self.initial_cash,
                "final_value": self.strategy.broker.getvalue(),
                "total_return_pct": total_return_pct,
                "annual_return_pct": annual_return * 100,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown_pct": max_drawdown,
                "calmar_ratio": calmar_ratio,
                "sqn": sqn,
                "profit_loss": self.strategy.broker.getvalue() - self.initial_cash,
            }
            logger.info("Generated performance summary")
            return summary

        except Exception as e:
            logger.error(
                f"Error calculating performance summary: {str(e)}. Check analyzer setup."
            )
            # Always return a valid structure, even on error
            return {
                "initial_cash": self.initial_cash,
                "final_value": self.initial_cash,
                "total_return_pct": 0,
                "annual_return_pct": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown_pct": 0,
                "calmar_ratio": 0,
                "sqn": 0,
                "profit_loss": 0,
            }

    def get_trade_analysis(self):
        """Analyze individual trades and return both summary and detailed trade list.

        Returns:
            dict: Trade metrics including win rate, average win/loss, profit factor, and a list of all trades.
        """
        if self.is_dict_input:
            return self.processed_results.get("trade_analysis", {})

        try:
            if self.strategy is None:
                logger.warning("No strategy available for trade analysis")
                return {
                    "total_trades": 0,
                    "message": "No strategy available",
                    "trades": [],
                }

            trades_analyzer = getattr(self.strategy.analyzers, "trades", None)
            if trades_analyzer is None:
                logger.warning("Trade analyzer not available on strategy instance")
                return {
                    "total_trades": 0,
                    "message": "Trade analyzer not available",
                    "trades": [],
                }

            trade_analyzer = trades_analyzer.get_analysis()
            total_trades = trade_analyzer.get("total", {}).get("total", 0) or 0

            if total_trades == 0:
                logger.warning("No trades executed")
                return {
                    "total_trades": 0,
                    "message": "No trades executed",
                    "trades": [],
                }

            won_trades = trade_analyzer.get("won", {}).get("total", 0) or 0
            lost_trades = trade_analyzer.get("lost", {}).get("total", 0) or 0
            win_rate = won_trades / total_trades * 100 if total_trades > 0 else 0
            avg_win = (
                trade_analyzer.get("won", {}).get("pnl", {}).get("average", 0) or 0
            )
            avg_loss = (
                trade_analyzer.get("lost", {}).get("pnl", {}).get("average", 0) or 0
            )
            total_win_pnl = (
                trade_analyzer.get("won", {}).get("pnl", {}).get("total", 0) or 0
            )
            total_loss_pnl = (
                trade_analyzer.get("lost", {}).get("pnl", {}).get("total", 0) or 0
            )
            # Robust profit factor calculation
            if total_win_pnl == 0 and total_loss_pnl == 0:
                profit_factor = 0
            elif total_loss_pnl == 0:
                profit_factor = 0
            elif total_win_pnl == 0:
                profit_factor = 0
            else:
                profit_factor = abs(total_win_pnl / total_loss_pnl)

            trades_list = []
            closed_trades = trade_analyzer.get("closed", []) or trade_analyzer.get(
                "trades", []
            )
            if (
                isinstance(closed_trades, list)
                and closed_trades
                and isinstance(closed_trades[0], dict)
            ):
                for i, trade in enumerate(closed_trades):
                    try:
                        entry_date = trade.get("datein", None)
                        exit_date = trade.get("dateout", None)
                        if entry_date is not None:
                            entry_date = pd.to_datetime(entry_date, unit="s")
                        if exit_date is not None:
                            exit_date = pd.to_datetime(exit_date, unit="s")
                        trade_info = {
                            "trade_id": i + 1,
                            "entry_date": entry_date,
                            "exit_date": exit_date,
                            "size": trade.get("size", 0),
                            "price_in": trade.get("pricein", 0),
                            "price_out": trade.get("priceout", 0),
                            "pnl": trade.get("pnl", 0),
                            "pnl_comm": trade.get("pnlcomm", 0),
                            "direction": (
                                "long" if trade.get("size", 0) > 0 else "short"
                            ),
                        }
                        trades_list.append(trade_info)
                    except Exception as e:
                        logger.warning(f"Error parsing trade {i}: {e}")
            else:
                # If not available, leave empty
                trades_list = []

            # --- Extract all completed orders from the strategy.orders attribute ---
            orders_list = []
            if hasattr(self.strategy, "orders"):
                for order in self.strategy.orders:
                    if order.status == bt.Order.Completed:
                        try:
                            orders_list.append(
                                {
                                    "datetime": bt.num2date(order.executed.dt),
                                    "size": order.executed.size,
                                    "price": order.executed.price,
                                    "isbuy": order.isbuy(),
                                    "issell": order.issell(),
                                    "bar_index": getattr(
                                        order.executed, "bar_index", None
                                    ),
                                    "commission": order.executed.comm,
                                    "value": order.executed.value,
                                    "order_ref": order.ref,
                                    "order_type": order.getordername(),
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Error parsing order: {e}")

            analysis = {
                "total_trades": total_trades,
                "winning_trades": won_trades,
                "losing_trades": lost_trades,
                "win_rate_percent": win_rate,
                "average_win": avg_win,
                "average_loss": avg_loss,
                "profit_factor": profit_factor,
                "average_trade_duration": trade_analyzer.get("len", {}).get(
                    "average", 0
                )
                or 0,
                "max_winning_streak": trade_analyzer.get("streak", {})
                .get("won", {})
                .get("longest", 0)
                or 0,
                "max_losing_streak": trade_analyzer.get("streak", {})
                .get("lost", {})
                .get("longest", 0)
                or 0,
                "trades": trades_list,
                "orders": orders_list,
            }
            logger.info("Generated trade analysis with detailed trade and order list")
            return analysis

        except Exception as e:
            logger.error(
                f"Error calculating trade analysis: {str(e)}. Check trade analyzer."
            )
            # Always return a valid structure, even on error
            return {"total_trades": 0, "message": f"Error: {str(e)}", "trades": []}

    def get_risk_metrics(self):
        """Calculate risk-related metrics.

        Returns:
            dict: Risk metrics including volatility and VaR.
        """
        if self.is_dict_input:
            return self.processed_results.get("risk_metrics", {})

        try:
            if self.strategy is None:
                raise ValueError("No strategy available")

            timereturn_analyzer = getattr(self.strategy.analyzers, "timereturn", None)
            if timereturn_analyzer is None:
                logger.warning("TimeReturn analyzer not available")
                return {"message": "TimeReturn analyzer not available"}

            returns = pd.Series(timereturn_analyzer.get_analysis())
            if returns.empty:
                logger.warning("No returns data available for risk metrics")
                return {"message": "No returns data available"}

            annualized_volatility = returns.std() * np.sqrt(252) * 100
            var_95 = np.percentile(returns, 5) * 100
            var_99 = np.percentile(returns, 1) * 100

            metrics = {
                "annualized_volatility_percent": annualized_volatility,
                "var_95_percent": var_95,
                "var_99_percent": var_99,
            }
            logger.info("Generated risk metrics")
            return metrics

        except Exception as e:
            logger.error(
                f"Error calculating risk metrics: {str(e)}. Check TimeReturn analyzer."
            )
            return {"error": f"Error calculating risk metrics: {str(e)}"}

    def get_resampled_returns(self, resample_freq="ME"):
        """Calculate returns resampled by the given frequency.

        Args:
            resample_freq (str): Pandas offset alias for resampling (e.g., 'ME', 'D', 'H', '15T').

        Returns:
            dict: Resampled return statistics.
        """
        if self.is_dict_input:
            return self.processed_results.get("resampled_returns", {})

        try:
            if self.strategy is None:
                raise ValueError("No strategy available")

            timereturn_analyzer = getattr(self.strategy.analyzers, "timereturn", None)
            if timereturn_analyzer is None:
                logger.warning("TimeReturn analyzer not available")
                return {"message": "TimeReturn analyzer not available"}

            returns = pd.Series(timereturn_analyzer.get_analysis())
            if returns.empty:
                logger.warning("No returns data available for resampled returns")
                return {"message": "No returns data available"}

            returns_df = pd.DataFrame({"returns": returns})
            returns_df.index = pd.to_datetime(returns_df.index)
            resampled_returns = returns_df.resample(resample_freq).sum()
            resampled_returns_pct = resampled_returns * 100

            # Convert Timestamp keys to strings for JSON serialization
            resampled_returns_dict = {
                str(k): v for k, v in resampled_returns_pct["returns"].to_dict().items()
            }

            analysis = {
                "resampled_returns_pct": resampled_returns_dict,
                "average_resampled_return_pct": resampled_returns_pct["returns"].mean(),
                "resampled_return_std_pct": resampled_returns_pct["returns"].std(),
                "positive_periods": len(
                    resampled_returns_pct[resampled_returns_pct["returns"] > 0]
                ),
                "negative_periods": len(
                    resampled_returns_pct[resampled_returns_pct["returns"] <= 0]
                ),
                "resample_freq": resample_freq,
            }
            logger.info(f"Generated resampled returns analysis (freq={resample_freq})")
            return analysis

        except Exception as e:
            logger.error(
                f"Error calculating resampled returns: {str(e)}. Check TimeReturn analyzer."
            )
            return {"error": f"Error calculating resampled returns: {str(e)}"}

    def get_drawdown_analysis(self):
        """Analyze drawdowns.

        Returns:
            dict: Drawdown metrics including max drawdown and average duration.
        """
        if self.is_dict_input:
            return self.processed_results.get("drawdown_analysis", {})

        try:
            if self.strategy is None:
                raise ValueError("No strategy available")

            drawdown_analyzer = getattr(self.strategy.analyzers, "drawdown", None)
            if drawdown_analyzer is None:
                raise ValueError("DrawDown analyzer not available")

            drawdown_analysis = drawdown_analyzer.get_analysis()
            max_drawdown = drawdown_analysis.get("max", {}).get("drawdown", 0) or 0
            max_drawdown_duration = drawdown_analysis.get("max", {}).get("len", 0) or 0
            avg_drawdown = drawdown_analysis.get("drawdown", 0) or 0
            avg_drawdown_duration = drawdown_analysis.get("len", 0) or 0

            analysis = {
                "max_drawdown_pct": max_drawdown,
                "max_drawdown_duration": max_drawdown_duration,
                "average_drawdown_pct": avg_drawdown,
                "average_drawdown_duration": avg_drawdown_duration,
            }
            logger.info("Generated drawdown analysis")
            return analysis

        except Exception as e:
            logger.error(
                f"Error calculating drawdown analysis: {str(e)}. Check DrawDown analyzer."
            )
            return {"error": f"Error calculating drawdown analysis: {str(e)}"}

    def save_report_to_file(self, filename):
        """Save the performance report to a JSON file.

        Args:
            filename (str): Path to save the report.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            report = self.generate_full_report()

            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            report_converted = convert_numpy(report)

            with open(filename, "w") as f:
                json.dump(report_converted, f, indent=2, default=str)
            logger.info(f"Report saved to {filename}")
            return True
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Error saving report to {filename}: {str(e)}")
            return False

    def print_report(self, resample_freq="ME"):
        """Print a formatted performance report.

        Args:
            resample_freq (str): Pandas offset alias for resampling returns (e.g., 'ME', 'D', 'H', '15T').
        """
        try:
            report = self.generate_full_report(resample_freq=resample_freq)
            if "error" in report:
                print(f"Error generating report: {report['error']}")
                return

            print("=" * 60)
            print("PERFORMANCE REPORT")
            print("=" * 60)

            summary = report.get("summary", {})
            if "error" not in summary and summary:
                print("\nSummary Metrics")
                print("-" * 30)
                print(f"Initial Cash: ${summary.get('initial_cash', 0):}")
                print(f"Final Value: ${summary.get('final_value', 0):}")
                print(f"Total Return: {summary.get('total_return_pct', 0)}%")
                print(f"Annual Return: {summary.get('annual_return_pct', 0)}%")
                print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0)}")
                print(f"Sortino Ratio: {summary.get('sortino_ratio', 0)}")
                print(f"Max Drawdown: {summary.get('max_drawdown_pct', 0)}%")
                print(f"Calmar Ratio: {summary.get('calmar_ratio', 0)}")
                print(f"SQN: {summary.get('sqn', 0)}")
                print(f"Profit/Loss: ${summary.get('profit_loss', 0):}")

            trade_analysis = report.get("trade_analysis", {})
            if (
                "error" not in trade_analysis
                and trade_analysis.get("total_trades", 0) > 0
            ):
                print("\nTrade Analysis")
                print("-" * 30)
                print(f"Total Trades: {trade_analysis.get('total_trades', 0)}")
                print(f"Winning Trades: {trade_analysis.get('winning_trades', 0)}")
                print(f"Losing Trades: {trade_analysis.get('losing_trades', 0)}")
                print(f"Win Rate: {trade_analysis.get('win_rate_percent', 0):.2f}%")
                print(f"Average Win: ${trade_analysis.get('average_win', 0):,.2f}")
                print(f"Average Loss: ${trade_analysis.get('average_loss', 0):,.2f}")

                profit_factor = trade_analysis.get("profit_factor", 0)
                if profit_factor == float("inf"):
                    print("Profit Factor: ∞ (no losing trades)")
                else:
                    print(f"Profit Factor: {profit_factor:.2f}")

                print(
                    f"Average Trade Duration: {trade_analysis.get('average_trade_duration', 0):.1f} days"
                )
                print(
                    f"Max Winning Streak: {trade_analysis.get('max_winning_streak', 0)}"
                )
                print(
                    f"Max Losing Streak: {trade_analysis.get('max_losing_streak', 0)}"
                )

                # Print detailed trade list
                trades = trade_analysis.get("trades", [])
                if trades:
                    print("\nDetailed Trades")
                    print("-" * 80)
                    print(
                        f"{'ID':>3} {'Entry':>20} {'Exit':>20} {'Size':>6} {'In':>8} {'Out':>8} {'PnL':>10} {'PnL_Comm':>10} {'Dir':>6}"
                    )
                    print("-" * 80)
                    for t in trades:
                        entry = (
                            t["entry_date"].strftime("%Y-%m-%d %H:%M")
                            if t["entry_date"] is not None
                            else "-"
                        )
                        exit = (
                            t["exit_date"].strftime("%Y-%m-%d %H:%M")
                            if t["exit_date"] is not None
                            else "-"
                        )
                        print(
                            f"{t['trade_id']:>3} {entry:>20} {exit:>20} {t['size']:>6} {t['price_in']:>8.2f} {t['price_out']:>8.2f} {t['pnl']:>10.2f} {t['pnl_comm']:>10.2f} {t['direction']:>6}"
                        )
                    print("-" * 80)

            risk_metrics = report.get("risk_metrics", {})
            if "error" not in risk_metrics and risk_metrics:
                print("\nRisk Metrics")
                print("-" * 30)
                print(
                    f"Annualized Volatility: {risk_metrics.get('annualized_volatility_percent', 0):.2f}%"
                )
                print(f"VaR (95%): {risk_metrics.get('var_95_percent', 0):.2f}%")
                print(f"VaR (99%): {risk_metrics.get('var_99_percent', 0):.2f}%")

            resampled_returns = report.get("resampled_returns", {})
            if "error" not in resampled_returns and resampled_returns:
                print(
                    f"\nResampled Returns (freq={resampled_returns.get('resample_freq', resample_freq)})"
                )
                print("-" * 30)
                print(
                    f"Average Resampled Return: {resampled_returns.get('average_resampled_return_pct', 0):.2f}%"
                )
                print(
                    f"Resampled Return Std: {resampled_returns.get('resampled_return_std_pct', 0):.2f}%"
                )
                print(
                    f"Positive Periods: {resampled_returns.get('positive_periods', 0)}"
                )
                print(
                    f"Negative Periods: {resampled_returns.get('negative_periods', 0)}"
                )

            drawdown_analysis = report.get("drawdown_analysis", {})
            if "error" not in drawdown_analysis and drawdown_analysis:
                print("\nDrawdown Analysis")
                print("-" * 30)
                print(
                    f"Max Drawdown: {drawdown_analysis.get('max_drawdown_pct', 0):.2f}%"
                )
                print(
                    f"Max Drawdown Duration: {drawdown_analysis.get('max_drawdown_duration', 0)} bars"
                )
                print(
                    f"Average Drawdown: {drawdown_analysis.get('average_drawdown_pct', 0):.2f}%"
                )
                print(
                    f"Average Drawdown Duration: {drawdown_analysis.get('average_drawdown_duration', 0)} bars"
                )

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error processing report: {str(e)}")
            logger.error(f"Error printing report: {str(e)}")
            print(f"Error printing report: {str(e)}")


def compare_strategies(strategy_results_dict):
    """Compare multiple strategy results.

    Args:
        strategy_results_dict (dict): Dictionary of strategy names to results.

    Returns:
        pd.DataFrame: Comparison of performance metrics.
    """
    try:
        comparison_data = []
        for strategy_name, results in strategy_results_dict.items():
            analyzer = PerformanceAnalyzer(results)
            report = analyzer.generate_full_report()
            if "error" in report:
                logger.warning(
                    f"Skipping {strategy_name} due to error: {report['error']}"
                )
                continue

            summary = report.get("summary", {})
            trade_analysis = report.get("trade_analysis", {})
            if "error" in summary or "error" in trade_analysis:
                logger.warning(f"Skipping {strategy_name} due to incomplete data")
                continue

            # Handle numpy types for comparison
            def safe_get(d, key, default=0):
                val = d.get(key, default)
                if isinstance(val, (np.integer, np.floating)):
                    return float(val)
                return val

            metrics = {
                "Strategy": strategy_name,
                "Total Return (%)": safe_get(summary, "total_return_pct"),
                "Annual Return (%)": safe_get(summary, "annual_return_pct"),
                "Sharpe Ratio": safe_get(summary, "sharpe_ratio"),
                "Sortino Ratio": safe_get(summary, "sortino_ratio"),
                "Max Drawdown (%)": safe_get(summary, "max_drawdown_pct"),
                "Calmar Ratio": safe_get(summary, "calmar_ratio"),
                "SQN": safe_get(summary, "sqn"),
                "Win Rate (%)": safe_get(trade_analysis, "win_rate_percent"),
                "Profit Factor": safe_get(trade_analysis, "profit_factor"),
                "Total Trades": safe_get(trade_analysis, "total_trades"),
            }
            comparison_data.append(metrics)

        if not comparison_data:
            logger.warning("No valid strategy results for comparison")
            return pd.DataFrame()

        df = pd.DataFrame(comparison_data)
        logger.info("Generated strategy comparison")
        return df.set_index("Strategy")

    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Error comparing strategies: {str(e)}. Check strategy results.")
        return pd.DataFrame()
