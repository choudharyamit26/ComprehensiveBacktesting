import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Comprehensive performance analysis and reporting for backtesting results."""

    def __init__(self, strategy_results, initial_cash=100000.0):
        """Initialize the performance analyzer.

        Args:
            strategy_results: Backtrader strategy results (list or single instance).
            initial_cash (float): Initial portfolio cash.
        """
        self.results = strategy_results
        self.initial_cash = initial_cash
        # Accept both a list of results or a single strategy instance
        if isinstance(strategy_results, list) and strategy_results:
            self.strategy = (
                strategy_results[0]
                if strategy_results and len(strategy_results) > 0
                else None
            )
        else:
            self.strategy = strategy_results
        logger.info("Initialized PerformanceAnalyzer")

    def generate_full_report(self, resample_freq="ME"):
        """Generate a comprehensive performance report.

        Args:
            resample_freq (str): Pandas offset alias for resampling returns (e.g., 'ME', 'D', 'H', '15T').

        Returns:
            dict: Report with summary, trade analysis, risk metrics, and drawdown analysis.

        Example:
            >>> analyzer = PerformanceAnalyzer(results)
            >>> report = analyzer.generate_full_report(resample_freq="D")
        """
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
            logger.error(
                f"Error generating full report: {str(e)}. Check analyzer availability."
            )
            return {"error": f"Error generating report: {str(e)}"}

    def get_performance_summary(self):
        """Get basic performance metrics.

        Returns:
            dict: Summary metrics including returns, Sharpe ratio, Sortino ratio, and drawdown.
        """
        try:
            if self.strategy is None:
                raise ValueError("No strategy available")

            returns_analyzer = getattr(self.strategy.analyzers, "returns", None)
            sharpe_analyzer = getattr(self.strategy.analyzers, "sharpe", None)
            sortino_analyzer = getattr(self.strategy.analyzers, "sortino", None)
            drawdown_analyzer = getattr(self.strategy.analyzers, "drawdown", None)
            calmar_analyzer = getattr(self.strategy.analyzers, "calmar", None)
            sqn_analyzer = getattr(self.strategy.analyzers, "sqn", None)

            if returns_analyzer is None:
                raise ValueError("Returns analyzer not available")

            returns_analysis = returns_analyzer.get_analysis()
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
            sharpe_ratio = sharpe_analysis.get("sharperatio", 0) or 0
            sortino_ratio = sortino_analysis.get("sortinoratio", 0) or 0
            max_drawdown = drawdown_analysis.get("max", {}).get("drawdown", 0) or 0
            calmar_ratio = calmar_analysis.get("calmarratio", 0) or 0
            sqn = sqn_analysis.get("sqn", 0) or 0

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
            return {"error": f"Error calculating performance summary: {str(e)}"}

    def get_trade_analysis(self):
        """Analyze individual trades.

        Returns:
            dict: Trade metrics including win rate, average win/loss, and profit factor.
        """
        try:
            if self.strategy is None:
                raise ValueError("No strategy available")

            trades_analyzer = getattr(self.strategy.analyzers, "trades", None)
            if trades_analyzer is None:
                raise ValueError("Trade analyzer not available")

            trade_analyzer = trades_analyzer.get_analysis()
            total_trades = trade_analyzer.get("total", {}).get("total", 0) or 0

            if total_trades == 0:
                logger.warning("No trades executed")
                return {"total_trades": 0, "message": "No trades executed"}

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
            profit_factor = (
                abs(total_win_pnl / total_loss_pnl)
                if total_loss_pnl != 0
                else float("inf")
            )

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
            }
            logger.info("Generated trade analysis")
            return analysis

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(
                f"Error calculating trade analysis: {str(e)}. Check trade analyzer."
            )
            return {"error": f"Error calculating trade analysis: {str(e)}"}

    def get_risk_metrics(self):
        """Calculate risk-related metrics.

        Returns:
            dict: Risk metrics including volatility and VaR.
        """
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
            with open(filename, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {filename}")
            return True
        except Exception as e:
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
            if "error" not in summary:
                print("\nSummary Metrics")
                print("-" * 30)
                print(f"Initial Cash: ${summary.get('initial_cash', 0):,.2f}")
                print(f"Final Value: ${summary.get('final_value', 0):,.2f}")
                print(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
                print(f"Annual Return: {summary.get('annual_return_pct', 0):.2f}%")
                print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.3f}")
                print(f"Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
                print(f"Calmar Ratio: {summary.get('calmar_ratio', 0):.3f}")
                print(f"SQN: {summary.get('sqn', 0):.3f}")

            trade_analysis = report.get("trade_analysis", {})
            if (
                "error" not in trade_analysis
                and trade_analysis.get("total_trades", 0) > 0
            ):
                print("\nTrade Analysis")
                print("-" * 30)
                print(f"Total Trades: {trade_analysis.get('total_trades', 0)}")
                print(f"Win Rate: {trade_analysis.get('win_rate_percent', 0):.2f}%")
                print(f"Average Win: ${trade_analysis.get('average_win', 0):,.2f}")
                print(f"Average Loss: ${trade_analysis.get('average_loss', 0):,.2f}")
                print(f"Profit Factor: {trade_analysis.get('profit_factor', 0):.2f}")
                print(
                    f"Average Trade Duration: {trade_analysis.get('average_trade_duration', 0):.1f} days"
                )
                print(
                    f"Max Winning Streak: {trade_analysis.get('max_winning_streak', 0)}"
                )
                print(
                    f"Max Losing Streak: {trade_analysis.get('max_losing_streak', 0)}"
                )

            risk_metrics = report.get("risk_metrics", {})
            if "error" not in risk_metrics:
                print("\nRisk Metrics")
                print("-" * 30)
                print(
                    f"Annualized Volatility: {risk_metrics.get('annualized_volatility_percent', 0):.2f}%"
                )
                print(f"VaR (95%): {risk_metrics.get('var_95_percent', 0):.2f}%")
                print(f"VaR (99%): {risk_metrics.get('var_99_percent', 0):.2f}%")

            resampled_returns = report.get("resampled_returns", {})
            if "error" not in resampled_returns:
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
            if "error" not in drawdown_analysis:
                print("\nDrawdown Analysis")
                print("-" * 30)
                print(
                    f"Max Drawdown: {drawdown_analysis.get('max_drawdown_pct', 0):.2f}%"
                )
                print(
                    f"Max Drawdown Duration: {drawdown_analysis.get('max_drawdown_duration', 0)} days"
                )
                print(
                    f"Average Drawdown: {drawdown_analysis.get('average_drawdown_pct', 0):.2f}%"
                )
                print(
                    f"Average Drawdown Duration: {drawdown_analysis.get('average_drawdown_duration', 0)} days"
                )

        except Exception as e:
            logger.error(f"Error printing report: {str(e)}")
            print(f"Error printing report: {str(e)}")

    def plot_performance(self, save_path=None):
        """Plot performance metrics.

        Args:
            save_path (str, optional): Path to save the plot.
        """
        try:
            if self.strategy is None:
                raise ValueError("No strategy available")

            timereturn_analyzer = getattr(self.strategy.analyzers, "timereturn", None)
            drawdown_analyzer = getattr(self.strategy.analyzers, "drawdown", None)
            if timereturn_analyzer is None or drawdown_analyzer is None:
                raise ValueError("Required analyzers not available")

            returns = pd.Series(timereturn_analyzer.get_analysis())
            drawdowns = pd.Series(drawdown_analyzer.get_analysis().get("drawdown", []))

            if returns.empty:
                logger.warning("No returns data available for plotting")
                print("No returns data available for plotting")
                return

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            fig.suptitle("Performance Analysis", fontsize=16)

            returns_df = pd.DataFrame({"returns": returns})
            returns_df.index = pd.to_datetime(returns_df.index)
            cumulative_returns = (1 + returns).cumprod() - 1
            ax1.plot(
                returns_df.index,
                cumulative_returns * 100,
                label="Cumulative Return (%)",
                color="blue",
            )
            ax1.set_ylabel("Cumulative Return (%)")
            ax1.legend()
            ax1.grid(True)

            ax2.plot(returns_df.index, drawdowns, label="Drawdown (%)", color="red")
            ax2.set_ylabel("Drawdown (%)")
            ax2.set_xlabel("Date")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300)
                logger.info(f"Performance plot saved to {save_path}")
            plt.show()

        except Exception as e:
            logger.error(f"Error plotting performance: {str(e)}. Check analyzer data.")
            print(f"Error plotting performance: {str(e)}")


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

            metrics = {
                "Strategy": strategy_name,
                "Total Return (%)": summary.get("total_return_pct", 0),
                "Annual Return (%)": summary.get("annual_return_pct", 0),
                "Sharpe Ratio": summary.get("sharpe_ratio", 0),
                "Max Drawdown (%)": summary.get("max_drawdown_pct", 0),
                "Calmar Ratio": summary.get("calmar_ratio", 0),
                "SQN": summary.get("sqn", 0),
                "Win Rate (%)": trade_analysis.get("win_rate_percent", 0),
                "Profit Factor": trade_analysis.get("profit_factor", 0),
                "Total Trades": trade_analysis.get("total_trades", 0),
            }
            comparison_data.append(metrics)

        if not comparison_data:
            logger.warning("No valid strategy results for comparison")
            return pd.DataFrame()

        df = pd.DataFrame(comparison_data)
        logger.info("Generated strategy comparison")
        return df.set_index("Strategy")

    except Exception as e:
        logger.error(f"Error comparing strategies: {str(e)}. Check strategy results.")
        return pd.DataFrame()
