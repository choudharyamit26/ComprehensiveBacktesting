import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import backtrader as bt
from typing import Dict, List, Any, Union, Optional


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

    def get_trades(self) -> List[Dict[str, Any]]:
        """Return the list of trades as a list of dicts (for walk-forward saving/UI)."""
        if self.is_dict_input:
            return self.processed_results.get("trade_analysis", {}).get("trades", [])

        if self.strategy is None:
            return []

        trades_list = []
        if not hasattr(self.strategy, "completed_trades"):
            logger.warning("Strategy does not have completed_trades attribute")
            return []

        for i, trade in enumerate(self.strategy.completed_trades):
            try:
                # Calculate PnL with commission
                pnl = trade.get("pnl", 0)
                commission = trade.get("commission", 0)
                pnl_comm = (
                    pnl - commission
                    if pnl is not None and commission is not None
                    else pnl
                )

                trade_dict = {
                    "trade_id": trade.get("ref", i),
                    "entry_date": trade.get("entry_time"),
                    "exit_date": trade.get("exit_time"),
                    "size": trade.get("size", 0),
                    "price_in": trade.get("entry_price", 0),
                    "price_out": trade.get("exit_price", 0),
                    "pnl": pnl,
                    "pnl_comm": pnl_comm,
                    "direction": trade.get("direction", "unknown"),
                    "commission": commission,
                    "status": trade.get("status", "unknown"),
                    "bar_held": trade.get("bar_held", None),
                }
                trades_list.append(trade_dict)
            except Exception as e:
                logger.warning(f"Error processing trade {i}: {e}")
                continue

        return trades_list

    def get_timereturn(self) -> Dict[str, float]:
        """Return the timereturn series as a dict (date: return) for equity curve plotting."""
        if self.is_dict_input:
            # Try to get from processed results if available
            risk_metrics = self.processed_results.get("risk_metrics", {})
            timereturn = self.processed_results.get("timereturn", None)
            if timereturn is not None:
                return timereturn
            # Try to extract from resampled_returns if present
            resampled = self.processed_results.get("resampled_returns", {})
            if "resampled_returns_pct" in resampled:
                return resampled["resampled_returns_pct"]
            return {}

        try:
            if self.strategy is None:
                return {}

            timereturn_analyzer = getattr(self.strategy.analyzers, "timereturn", None)
            if timereturn_analyzer is None:
                return {}

            returns = pd.Series(timereturn_analyzer.get_analysis())
            if returns.empty:
                return {}

            # Convert index to string for JSON serialization
            returns.index = returns.index.map(str)
            return returns.to_dict()
        except Exception as e:
            logger.error(f"Error getting timereturn: {e}")
            return {}

    def generate_full_report(self, resample_freq="ME") -> Dict[str, Any]:
        """Generate a comprehensive performance report.

        Args:
            resample_freq (str): Pandas offset alias for resampling returns (e.g., 'ME', 'D', 'H', '15T').

        Returns:
            dict: Report with summary, trade analysis, risk metrics, and drawdown analysis.
        """
        if self.is_dict_input:
            # Return the already processed results
            logger.info("Returning pre-processed performance report")
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
                "timereturn": self.get_timereturn(),
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

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get basic performance metrics.

        Returns:
            dict: Summary metrics including returns, Sharpe ratio, Sortino ratio, and drawdown.
        """
        if self.is_dict_input:
            return self.processed_results.get("summary", {})

        default_summary = {
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

        try:
            if self.strategy is None:
                logger.warning("No strategy available for performance summary")
                return default_summary

            # Get analyzers with error handling
            analyzers = getattr(self.strategy, "analyzers", None)
            if analyzers is None:
                logger.warning("No analyzers found on strategy")
                return default_summary

            returns_analyzer = getattr(analyzers, "returns", None)
            sharpe_analyzer = getattr(analyzers, "sharpe", None)
            sortino_analyzer = getattr(analyzers, "sortino", None)
            drawdown_analyzer = getattr(analyzers, "drawdown", None)
            calmar_analyzer = getattr(analyzers, "calmar", None)
            sqn_analyzer = getattr(analyzers, "sqn", None)

            # Get analysis results with error handling
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

            # Extract returns
            total_return = returns_analysis.get("rtot", 0) or 0
            annual_return = returns_analysis.get("rnorm", 0) or 0

            # Robust extraction and formatting for ratios
            def robust_float(val, default=0):
                try:
                    if val is None or (
                        isinstance(val, (int, float)) and abs(val) < 1e-12
                    ):
                        return default
                    if isinstance(val, (int, float)) and not np.isfinite(val):
                        return default
                    return float(val)
                except (ValueError, TypeError):
                    return default

            # Process Sharpe ratio
            sharpe_ratio = 0
            if sharpe_analyzer:
                try:
                    raw_sharpe = sharpe_analysis.get("sharperatio", None)
                    if raw_sharpe is not None:
                        # Scale down if value is unreasonably large (likely wrong scaling)
                        if abs(raw_sharpe) > 100:
                            sharpe_ratio = raw_sharpe / 100
                        else:
                            sharpe_ratio = robust_float(raw_sharpe)
                except Exception as e:
                    logger.error(f"Error processing Sharpe ratio: {e}")
                    sharpe_ratio = 0

            # Process Sortino ratio
            sortino_ratio = 0
            if sortino_analyzer:
                try:
                    raw_sortino = sortino_analysis.get("sortinoratio", None)
                    if raw_sortino is not None:
                        # Scale down if value is unreasonably large (likely wrong scaling)
                        if abs(raw_sortino) > 100:
                            sortino_ratio = raw_sortino / 100
                        else:
                            sortino_ratio = robust_float(raw_sortino)
                except Exception as e:
                    logger.error(f"Error processing Sortino ratio: {e}")
                    sortino_ratio = 0

            # Process other metrics
            max_drawdown = drawdown_analysis.get("max", {}).get("drawdown", 0) or 0
            max_drawdown = robust_float(max_drawdown)

            calmar_ratio = robust_float(calmar_analysis.get("calmarratio", None))
            # Manual Calmar calculation if needed
            if calmar_ratio == 0 and annual_return != 0 and max_drawdown != 0:
                calmar_ratio = robust_float(annual_return / abs(max_drawdown))

            sqn = robust_float(sqn_analysis.get("sqn", None))

            # Calculate final values
            final_value = getattr(
                self.strategy.broker, "getvalue", lambda: self.initial_cash
            )()
            total_return_pct = (
                (np.exp(total_return) - 1) * 100 if total_return != 0 else 0
            )
            profit_loss = final_value - self.initial_cash

            summary = {
                "initial_cash": self.initial_cash,
                "final_value": final_value,
                "total_return_pct": total_return_pct,
                "annual_return_pct": annual_return * 100,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown_pct": max_drawdown,
                "calmar_ratio": calmar_ratio,
                "sqn": sqn,
                "profit_loss": profit_loss,
            }
            logger.info("Generated performance summary")
            return summary

        except Exception as e:
            logger.error(
                f"Error calculating performance summary: {str(e)}. Check analyzer setup."
            )
            return default_summary

    def get_trade_analysis(self) -> Dict[str, Any]:
        """Analyze individual trades and return both summary and detailed trade list.

        Returns:
            dict: Trade metrics including win rate, average win/loss, profit factor, and a list of all trades.
        """
        if self.is_dict_input:
            return self.processed_results.get("trade_analysis", {})

        default_analysis = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_percent": 0,
            "average_win": 0,
            "average_loss": 0,
            "profit_factor": 0,
            "average_trade_duration": 0,
            "max_winning_streak": 0,
            "max_losing_streak": 0,
            "trades": [],
            "orders": [],
            "message": "No strategy available",
        }

        try:
            if self.strategy is None:
                logger.warning("No strategy available for trade analysis")
                return default_analysis

            trades_list = self.get_trades()
            total_trades = len(trades_list)
            if total_trades == 0:
                logger.warning("No trades executed")
                return {**default_analysis, "message": "No trades executed"}

            # Calculate trade statistics from trades_list
            winning_trades = [t for t in trades_list if t.get("pnl", 0) > 0]
            losing_trades = [t for t in trades_list if t.get("pnl", 0) < 0]
            won_trades = len(winning_trades)
            lost_trades = len(losing_trades)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            avg_win = (
                np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
            )
            avg_loss = (
                np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0
            )
            total_win_pnl = (
                np.sum([t["pnl"] for t in winning_trades]) if winning_trades else 0
            )
            total_loss_pnl = (
                np.sum([t["pnl"] for t in losing_trades]) if losing_trades else 0
            )
            # Profit factor
            profit_factor = 0
            if total_win_pnl > 0 and total_loss_pnl < 0:
                profit_factor = abs(total_win_pnl / total_loss_pnl)
            elif total_win_pnl > 0 and total_loss_pnl == 0:
                profit_factor = float("inf")

            # Average trade duration (in bars, if available)
            durations = [
                t["bar_held"] for t in trades_list if t.get("bar_held") is not None
            ]
            avg_trade_duration = np.mean(durations) if durations else 0

            # Max winning/losing streaks (by consecutive wins/losses)
            max_winning_streak = 0
            max_losing_streak = 0
            current_win_streak = 0
            current_loss_streak = 0
            for t in trades_list:
                if t.get("pnl", 0) > 0:
                    current_win_streak += 1
                    max_winning_streak = max(max_winning_streak, current_win_streak)
                    current_loss_streak = 0
                elif t.get("pnl", 0) < 0:
                    current_loss_streak += 1
                    max_losing_streak = max(max_losing_streak, current_loss_streak)
                    current_win_streak = 0
                else:
                    current_win_streak = 0
                    current_loss_streak = 0

            # Process orders if available
            orders_list = []
            if hasattr(self.strategy, "orders"):
                for order in self.strategy.orders:
                    try:
                        if order.status == bt.Order.Completed:
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
                        continue

            analysis = {
                "total_trades": total_trades,
                "winning_trades": won_trades,
                "losing_trades": lost_trades,
                "win_rate_percent": win_rate,
                "average_win": avg_win,
                "average_loss": avg_loss,
                "profit_factor": profit_factor,
                "average_trade_duration": avg_trade_duration,
                "max_winning_streak": max_winning_streak,
                "max_losing_streak": max_losing_streak,
                "trades": trades_list,
                "orders": orders_list,
            }
            logger.info(
                "Generated trade analysis with detailed trade and order list (from trades_list)"
            )
            return analysis

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(
                f"Error calculating trade analysis: {str(e)}. Check trade analyzer."
            )
            return {**default_analysis, "message": f"Error: {str(e)}"}

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Calculate risk-related metrics.

        Returns:
            dict: Risk metrics including volatility and VaR.
        """
        if self.is_dict_input:
            return self.processed_results.get("risk_metrics", {})

        try:
            if self.strategy is None:
                raise ValueError("No strategy available")

            analyzers = getattr(self.strategy, "analyzers", None)
            if analyzers is None:
                raise ValueError("No analyzers found on strategy")

            timereturn_analyzer = getattr(analyzers, "timereturn", None)
            if timereturn_analyzer is None:
                logger.warning("TimeReturn analyzer not available")
                return {"message": "TimeReturn analyzer not available"}

            returns = pd.Series(timereturn_analyzer.get_analysis())
            if returns.empty:
                logger.warning("No returns data available for risk metrics")
                return {"message": "No returns data available"}

            # Calculate risk metrics
            annualized_volatility = returns.std() * np.sqrt(252) * 100
            var_95 = np.percentile(returns, 5) * 100
            var_99 = np.percentile(returns, 1) * 100

            # Calculate additional risk metrics
            skewness = returns.skew()
            kurtosis = returns.kurtosis()

            metrics = {
                "annualized_volatility_percent": annualized_volatility,
                "var_95_percent": var_95,
                "var_99_percent": var_99,
                "skewness": skewness,
                "kurtosis": kurtosis,
            }
            logger.info("Generated risk metrics")
            return metrics

        except Exception as e:
            logger.error(
                f"Error calculating risk metrics: {str(e)}. Check TimeReturn analyzer."
            )
            return {"error": f"Error calculating risk metrics: {str(e)}"}

    def get_resampled_returns(self, resample_freq="ME") -> Dict[str, Any]:
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

            analyzers = getattr(self.strategy, "analyzers", None)
            if analyzers is None:
                raise ValueError("No analyzers found on strategy")

            timereturn_analyzer = getattr(analyzers, "timereturn", None)
            if timereturn_analyzer is None:
                logger.warning("TimeReturn analyzer not available")
                return {"message": "TimeReturn analyzer not available"}

            returns = pd.Series(timereturn_analyzer.get_analysis())
            if returns.empty:
                logger.warning("No returns data available for resampled returns")
                return {"message": "No returns data available"}

            # Convert to DataFrame and resample
            returns_df = pd.DataFrame({"returns": returns})
            returns_df.index = pd.to_datetime(returns_df.index)

            try:
                resampled_returns = returns_df.resample(resample_freq).sum()
            except Exception as e:
                logger.error(f"Error resampling with frequency {resample_freq}: {e}")
                # Fallback to monthly if the frequency is invalid
                resampled_returns = returns_df.resample("ME").sum()
                resample_freq = "ME"

            resampled_returns_pct = resampled_returns * 100

            # Convert Timestamp keys to strings for JSON serialization
            resampled_returns_dict = {
                str(k): v for k, v in resampled_returns_pct["returns"].to_dict().items()
            }

            positive_periods = len(
                resampled_returns_pct[resampled_returns_pct["returns"] > 0]
            )
            negative_periods = len(
                resampled_returns_pct[resampled_returns_pct["returns"] <= 0]
            )

            analysis = {
                "resampled_returns_pct": resampled_returns_dict,
                "average_resampled_return_pct": resampled_returns_pct["returns"].mean(),
                "resampled_return_std_pct": resampled_returns_pct["returns"].std(),
                "positive_periods": positive_periods,
                "negative_periods": negative_periods,
                "resample_freq": resample_freq,
            }
            logger.info(f"Generated resampled returns analysis (freq={resample_freq})")
            return analysis

        except Exception as e:
            logger.error(
                f"Error calculating resampled returns: {str(e)}. Check TimeReturn analyzer."
            )
            return {"error": f"Error calculating resampled returns: {str(e)}"}

    def get_drawdown_analysis(self) -> Dict[str, Any]:
        """Analyze drawdowns.

        Returns:
            dict: Drawdown metrics including max drawdown and average duration.
        """
        if self.is_dict_input:
            return self.processed_results.get("drawdown_analysis", {})

        try:
            if self.strategy is None:
                raise ValueError("No strategy available")

            analyzers = getattr(self.strategy, "analyzers", None)
            if analyzers is None:
                raise ValueError("No analyzers found on strategy")

            drawdown_analyzer = getattr(analyzers, "drawdown", None)
            if drawdown_analyzer is None:
                raise ValueError("DrawDown analyzer not available")

            drawdown_analysis = drawdown_analyzer.get_analysis()

            # Extract drawdown metrics with error handling
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

    def save_report_to_file(self, filename: str) -> bool:
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
                    if np.isnan(obj) or np.isinf(obj):
                        return None
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, datetime):
                    return obj.isoformat()
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
                print(f"Initial Cash: ${summary.get('initial_cash', 0):,.2f}")
                print(f"Final Value: ${summary.get('final_value', 0):,.2f}")
                print(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
                print(f"Annual Return: {summary.get('annual_return_pct', 0):.2f}%")
                print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
                print(f"Sortino Ratio: {summary.get('sortino_ratio', 0):.2f}")
                print(f"Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
                print(f"Calmar Ratio: {summary.get('calmar_ratio', 0):.2f}")
                print(f"SQN: {summary.get('sqn', 0):.2f}")
                print(f"Profit/Loss: ${summary.get('profit_loss', 0):,.2f}")

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
                    print("-" * 90)
                    print(
                        f"{'ID':>3} {'Entry':>12} {'Exit':>12} {'Size':>6} {'In':>8} {'Out':>8} {'PnL':>10} {'PnL+Comm':>10} {'Dir':>6}"
                    )
                    print("-" * 90)
                    for t in trades:
                        entry = (
                            t["entry_date"].strftime("%Y-%m-%d")
                            if t["entry_date"]
                            else "-"
                        )
                        exit = (
                            t["exit_date"].strftime("%Y-%m-%d")
                            if t["exit_date"]
                            else "-"
                        )
                        pnl_comm = t.get("pnl_comm", t.get("pnl", 0))
                        print(
                            f"{t['trade_id']:>3} {entry:>12} {exit:>12} {t['size']:>6} {t['price_in']:>8.2f} {t['price_out']:>8.2f} {t['pnl']:>10.2f} {pnl_comm:>10.2f} {t['direction']:>6}"
                        )
                    print("-" * 90)

            risk_metrics = report.get("risk_metrics", {})
            if (
                "error" not in risk_metrics
                and risk_metrics
                and "message" not in risk_metrics
            ):
                print("\nRisk Metrics")
                print("-" * 30)
                print(
                    f"Annualized Volatility: {risk_metrics.get('annualized_volatility_percent', 0):.2f}%"
                )
                print(f"VaR (95%): {risk_metrics.get('var_95_percent', 0):.2f}%")
                print(f"VaR (99%): {risk_metrics.get('var_99_percent', 0):.2f}%")
                if "skewness" in risk_metrics:
                    print(f"Skewness: {risk_metrics.get('skewness', 0):.2f}")
                if "kurtosis" in risk_metrics:
                    print(f"Kurtosis: {risk_metrics.get('kurtosis', 0):.2f}")

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
