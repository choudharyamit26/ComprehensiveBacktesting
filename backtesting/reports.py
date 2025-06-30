import pandas as pd
import numpy as np
from datetime import datetime
import json


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis and reporting for backtesting results.
    """

    def __init__(self, strategy_results, initial_cash=100000.0):
        self.results = strategy_results
        self.initial_cash = initial_cash
        print("Initializing PerformanceAnalyzer with results:", self.results)
        # Fix: Use proper None check for Backtrader objects
        self.strategy = (
            strategy_results[0]
            if strategy_results and len(strategy_results) > 0
            else None
        )

    def generate_full_report(self):
        """Generate a comprehensive performance report."""

        # Fix: Use is None instead of not self.strategy
        if self.strategy is None:
            return "No strategy results available"

        report = {
            "summary": self.get_performance_summary(),
            "trade_analysis": self.get_trade_analysis(),
            "risk_metrics": self.get_risk_metrics(),
            "monthly_returns": self.get_monthly_returns(),
            "drawdown_analysis": self.get_drawdown_analysis(),
        }

        return report

    def get_performance_summary(self):
        """Get basic performance metrics."""

        try:
            # Fix: Add None check before accessing analyzers
            if self.strategy is None:
                return {"error": "No strategy available"}

            # Get analyzers with safe access
            print(
                "Analyzing performance returns summary...",
                self.strategy.analyzers.returns.get_analysis(),
            )
            print(
                "Analyzing performance sharpe summary...",
                self.strategy.analyzers.sharpe.get_analysis(),
            )
            print(
                "Analyzing performance drawdown summary...",
                self.strategy.analyzers.drawdown.get_analysis(),
            )

            returns_analyzer = getattr(self.strategy.analyzers, "returns", None)
            sharpe_analyzer = getattr(self.strategy.analyzers, "sharpe", None)
            drawdown_analyzer = getattr(self.strategy.analyzers, "drawdown", None)

            if returns_analyzer is None:
                return {"error": "Returns analyzer not available"}

            # Calculate metrics with safe access
            returns_analysis = returns_analyzer.get_analysis()
            sharpe_analysis = sharpe_analyzer.get_analysis() if sharpe_analyzer else {}
            drawdown_analysis = (
                drawdown_analyzer.get_analysis() if drawdown_analyzer else {}
            )

            total_return = returns_analysis.get("rtot", 0) or 0
            annual_return = returns_analysis.get("rnorm", 0) or 0
            sharpe_ratio = (
                sharpe_analysis.get("sharperatio", 0) if sharpe_analysis else 0
            )
            max_drawdown = (
                drawdown_analysis.get("max", {}).get("drawdown", 0)
                if drawdown_analysis
                else 0
            )

            # Ensure no None values
            total_return = (total_return or 0) * 100
            annual_return = (annual_return or 0) * 100
            sharpe_ratio = sharpe_ratio or 0
            max_drawdown = max_drawdown or 0

            # Final portfolio value
            final_value = self.strategy.broker.getvalue()
            total_return_pct = (np.exp(total_return) - 1) * 100
            summary = {
                "initial_cash": self.initial_cash,
                "final_value": final_value,
                "total_return_pct": total_return_pct,
                "annual_return_pct": annual_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown_pct": max_drawdown,
                "profit_loss": final_value - self.initial_cash,
            }

        except Exception as e:
            import traceback

            traceback.print_exc()
            summary = {"error": f"Error calculating performance summary: {str(e)}"}

        return summary

    def get_trade_analysis(self):
        """Analyze individual trades."""

        try:
            # Fix: Add None check
            if self.strategy is None:
                return {"error": "No strategy available"}

            trades_analyzer = getattr(self.strategy.analyzers, "trades", None)

            if trades_analyzer is None:
                return {"error": "Trade analyzer not available"}

            trade_analyzer = trades_analyzer.get_analysis()

            total_trades = trade_analyzer.get("total", {}).get("total", 0)

            if total_trades == 0:
                return {"total_trades": 0, "message": "No trades executed"}
            won_trades = trade_analyzer.get("won", {}).get("total", 0) or 0
            lost_trades = trade_analyzer.get("lost", {}).get("total", 0) or 0

            analysis = {
                "total_trades": total_trades,
                "winning_trades": won_trades,
                "losing_trades": lost_trades,
                "win_rate_pct": (
                    (won_trades / total_trades) * 100 if total_trades > 0 else 0
                ),
                "avg_win": trade_analyzer.get("won", {})
                .get("pnl", {})
                .get("average", 0)
                or 0,
                "avg_loss": trade_analyzer.get("lost", {})
                .get("pnl", {})
                .get("average", 0)
                or 0,
                "largest_win": trade_analyzer.get("won", {})
                .get("pnl", {})
                .get("max", 0)
                or 0,
                "largest_loss": trade_analyzer.get("lost", {})
                .get("pnl", {})
                .get("max", 0)
                or 0,
                "total_pnl": trade_analyzer.get("pnl", {})
                .get("net", {})
                .get("total", 0)
                or 0,
            }

            # Calculate profit factor
            total_wins = (
                trade_analyzer.get("won", {}).get("pnl", {}).get("total", 0) or 0
            )
            total_losses = abs(
                trade_analyzer.get("lost", {}).get("pnl", {}).get("total", 0) or 0
            )
            analysis["profit_factor"] = (
                total_wins / total_losses if total_losses > 0 else float("inf")
            )

        except Exception as e:
            analysis = {"error": f"Error calculating trade analysis: {str(e)}"}

        return analysis

    def get_risk_metrics(self):
        """Calculate additional risk metrics."""

        try:
            # Fix: Add None check
            if self.strategy is None:
                return {"error": "No strategy available"}

            # Get portfolio values over time
            portfolio_values = []
            data_len = len(self.strategy.data)

            if data_len < 2:
                return {"error": "Insufficient data for risk calculations"}

            # Simple approximation - in reality you'd track daily values
            initial_value = self.initial_cash
            final_value = self.strategy.broker.getvalue()

            # Create a simple linear progression for demonstration
            # In practice, you'd want to track actual daily portfolio values
            for i in range(data_len):
                progress = i / (data_len - 1) if data_len > 1 else 0
                value = initial_value + (final_value - initial_value) * progress
                portfolio_values.append(value)

            # Calculate returns
            returns = np.diff(portfolio_values) / portfolio_values[:-1]

            # Risk metrics
            volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized
            downside_returns = returns[returns < 0]
            downside_volatility = (
                np.std(downside_returns) * np.sqrt(252) * 100
                if len(downside_returns) > 0
                else 0
            )

            # Value at Risk (95% confidence)
            var_95 = np.percentile(returns, 5) * 100 if len(returns) > 0 else 0

            # Maximum consecutive losses
            max_consec_losses = self._calculate_max_consecutive_losses(returns)

            risk_metrics = {
                "volatility_pct": volatility,
                "downside_volatility_pct": downside_volatility,
                "var_95_pct": var_95,
                "max_consecutive_losses": max_consec_losses,
                "calmar_ratio": self._calculate_calmar_ratio(),
            }

        except Exception as e:
            risk_metrics = {"error": f"Error calculating risk metrics: {str(e)}"}

        return risk_metrics

    def get_monthly_returns(self):
        """Calculate monthly returns."""

        try:
            # Fix: Add None check
            if self.strategy is None:
                return {"error": "No strategy available"}

            returns_analyzer = getattr(self.strategy.analyzers, "returns", None)
            if returns_analyzer is None:
                return {"error": "Returns analyzer not available"}

            returns_analysis = returns_analyzer.get_analysis()

            # For now, return basic info
            monthly_data = {
                "total_return": returns_analysis.get("rtot", 0) * 100,
                "note": "Monthly breakdown requires daily portfolio tracking",
            }

        except Exception as e:
            monthly_data = {"error": f"Error calculating monthly returns: {str(e)}"}

        return monthly_data

    def get_drawdown_analysis(self):
        """Analyze drawdown periods."""

        try:
            # Fix: Add None check
            if self.strategy is None:
                return {"error": "No strategy available"}

            drawdown_analyzer = getattr(self.strategy.analyzers, "drawdown", None)
            if drawdown_analyzer is None:
                return {"error": "Drawdown analyzer not available"}

            drawdown_analysis = drawdown_analyzer.get_analysis()

            analysis = {
                "max_drawdown_pct": drawdown_analysis.get("max", {}).get("drawdown", 0),
                "max_drawdown_duration": drawdown_analysis.get("max", {}).get("len", 0),
                "avg_drawdown_pct": drawdown_analysis.get("drawdown", 0),
                "avg_drawdown_duration": drawdown_analysis.get("len", 0),
            }

        except Exception as e:
            analysis = {"error": f"Error calculating drawdown analysis: {str(e)}"}

        return analysis

    def _calculate_max_consecutive_losses(self, returns):
        """Calculate maximum consecutive losing periods."""

        consecutive_losses = 0
        max_consecutive = 0

        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0

        return max_consecutive

    def _calculate_calmar_ratio(self):
        """Calculate Calmar ratio (Annual return / Max drawdown)."""

        try:
            # Fix: Add None check
            if self.strategy is None:
                return 0

            returns_analyzer = getattr(self.strategy.analyzers, "returns", None)
            drawdown_analyzer = getattr(self.strategy.analyzers, "drawdown", None)

            if returns_analyzer is None or drawdown_analyzer is None:
                return 0

            returns_analysis = returns_analyzer.get_analysis()
            drawdown_analysis = drawdown_analyzer.get_analysis()

            annual_return = returns_analysis.get("rnorm", 0) * 100
            max_drawdown = drawdown_analysis.get("max", {}).get("drawdown", 0)

            if max_drawdown > 0:
                return annual_return / max_drawdown
            else:
                return float("inf") if annual_return > 0 else 0

        except Exception:
            return 0

    def print_report(self):
        """Print a formatted performance report."""

        report = self.generate_full_report()

        # Handle case where report is just an error string
        if isinstance(report, str):
            print("=" * 60)
            print("PERFORMANCE REPORT")
            print("=" * 60)
            print(f"Error: {report}")
            return

        print("=" * 60)
        print("PERFORMANCE REPORT")
        print("=" * 60)

        # Performance Summary
        print("\nPERFORMANCE SUMMARY")
        print("-" * 30)
        summary = report["summary"]
        if "error" not in summary:
            print(f"Initial Cash:        ${summary.get('initial_cash', 0):,.2f}")
            print(f"Final Value:         ${summary.get('final_value', 0):,.2f}")
            print(f"Total Return:        {summary.get('total_return_pct', 0):.2f}%")
            print(f"Annual Return:       {summary.get('annual_return_pct', 0):.2f}%")
            print(f"Sharpe Ratio:        {summary.get('sharpe_ratio', 0):.3f}")
            print(f"Max Drawdown:        {summary.get('max_drawdown_pct', 0):.2f}%")
            print(f"Profit/Loss:         ${summary.get('profit_loss', 0):,.2f}")
        else:
            print(f"Error: {summary['error']}")

        # Trade Analysis
        print("\nTRADE ANALYSIS")
        print("-" * 30)
        trades = report["trade_analysis"]
        if "error" not in trades:
            if trades.get("total_trades", 0) == 0:
                print(trades.get("message", "No trade data available"))
            else:
                print(f"Total Trades:        {trades.get('total_trades', 0)}")
                print(f"Winning Trades:      {trades.get('winning_trades', 0)}")
                print(f"Losing Trades:       {trades.get('losing_trades', 0)}")
                print(f"Win Rate:            {trades.get('win_rate_pct', 0):.2f}%")
                print(f"Average Win:         ${trades.get('avg_win', 0):.2f}")
                print(f"Average Loss:        ${trades.get('avg_loss', 0):.2f}")
                print(f"Largest Win:         ${trades.get('largest_win', 0):.2f}")
                print(f"Largest Loss:        ${trades.get('largest_loss', 0):.2f}")
                profit_factor = trades.get("profit_factor", 0)
                if profit_factor == float("inf"):
                    print(f"Profit Factor:       ∞")
                else:
                    print(f"Profit Factor:       {profit_factor:.2f}")
        else:
            print(f"Error: {trades['error']}")

        # Risk Metrics
        print("\nRISK METRICS")
        print("-" * 30)
        risk = report["risk_metrics"]
        if "error" not in risk:
            print(f"Volatility:          {risk.get('volatility_pct', 0):.2f}%")
            print(f"Downside Volatility: {risk.get('downside_volatility_pct', 0):.2f}%")
            print(f"VaR (95%):           {risk.get('var_95_pct', 0):.2f}%")
            print(f"Max Consec. Losses:  {risk.get('max_consecutive_losses', 0)}")
            calmar_ratio = risk.get("calmar_ratio", 0)
            if calmar_ratio == float("inf"):
                print(f"Calmar Ratio:        ∞")
            else:
                print(f"Calmar Ratio:        {calmar_ratio:.3f}")
        else:
            print(f"Error: {risk['error']}")

        print("=" * 60)

    def save_report_to_file(self, filename=None):
        """Save report to JSON file."""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{timestamp}.json"

        report = self.generate_full_report()

        try:
            with open(filename, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved to {filename}")
        except Exception as e:
            print(f"Error saving report: {str(e)}")


def compare_strategies(results_dict):
    """
    Compare multiple strategy results.

    Parameters:
    results_dict (dict): Dictionary with strategy names as keys and results as values

    Returns:
    pd.DataFrame: Comparison table
    """

    comparison_data = []

    for strategy_name, results in results_dict.items():
        try:
            analyzer = PerformanceAnalyzer(results)
            summary = analyzer.get_performance_summary()

            if "error" not in summary:
                comparison_data.append(
                    {
                        "Strategy": strategy_name,
                        "Total Return (%)": summary["total_return_pct"],
                        "Annual Return (%)": summary["annual_return_pct"],
                        "Sharpe Ratio": summary["sharpe_ratio"],
                        "Max Drawdown (%)": summary["max_drawdown_pct"],
                        "Final Value": summary["final_value"],
                    }
                )
        except Exception as e:
            print(f"Error analyzing {strategy_name}: {str(e)}")
            continue

    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df = df.set_index("Strategy")
        return df
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage would require actual backtest results
    print("Reports module loaded successfully!")
    print(
        "Use PerformanceAnalyzer class with your backtest results to generate reports."
    )
