import backtrader as bt
import numpy as np
import optuna
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool

from comprehensive_backtesting.data import get_data_sync
from comprehensive_backtesting.registry import get_strategy
from comprehensive_backtesting.sma_bollinger_band import SMABollinger
from backtrader.analyzers import TimeReturn


class StrategyTradeAnalyzer(bt.Analyzer):
    def __init__(self):
        self.trades = []

    def stop(self):
        """Called at the end of the backtest"""
        # Get trades from the strategy if it has them
        if hasattr(self.strategy, "get_completed_trades"):
            strategy_trades = self.strategy.get_completed_trades()

            # Convert to the format expected by your code
            for trade in strategy_trades:
                trade_info = {
                    "trade_id": trade.get("ref", len(self.trades) + 1),
                    "entry_date": trade.get("entry_time"),
                    "entry_time": trade.get("entry_time"),  # Added
                    "exit_date": trade.get("exit_time"),
                    "exit_time": trade.get("exit_time"),  # Added
                    "entry_price": trade.get("entry_price", 0),
                    "exit_price": trade.get("exit_price", 0),
                    "pnl": trade.get("pnl", 0),
                    "pnl_net": trade.get("pnl_net", 0),
                    "direction": trade.get("direction", "Unknown"),
                    "size": trade.get("size", 0),
                    "duration": trade.get("bars_held", 0),
                    "commission": trade.get("commission", 0),
                    "status": trade.get("status", "Unknown"),
                }
                self.trades.append(trade_info)

        print(f"StrategyTradeAnalyzer collected {len(self.trades)} trades")

    def get_analysis(self):
        return self.trades.copy()


# [Modify create_cerebro function]
def create_cerebro(data, strategy_class, params, initial_cash, commission):
    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(strategy_class, **params)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Add TimeReturn analyzer for equity curve
    cerebro.addanalyzer(TimeReturn, timeframe=bt.TimeFrame.Days, _name="timereturn")

    # Use the strategy's own trade tracking
    cerebro.addanalyzer(StrategyTradeAnalyzer, _name="tradehistory")

    return cerebro


def extract_trades(strat):
    """Extract trades from strategy"""
    trades = []

    # First try the analyzer
    if hasattr(strat.analyzers, "tradehistory"):
        try:
            trades = strat.analyzers.tradehistory.get_analysis()
            print(f"Extracted {len(trades)} trades from StrategyTradeAnalyzer")
        except Exception as e:
            print(f"Error extracting from analyzer: {e}")

    # If no trades from analyzer, try strategy directly
    if not trades and hasattr(strat, "get_completed_trades"):
        try:
            strategy_trades = strat.get_completed_trades()
            print(f"Found {len(strategy_trades)} trades in strategy")

            # Convert to expected format
            for trade in strategy_trades:
                trade_info = {
                    "trade_id": trade.get("ref", len(trades) + 1),
                    "entry_date": trade.get("entry_time"),
                    "entry_time": trade.get("entry_time"),  # Added
                    "exit_date": trade.get("exit_time"),
                    "exit_time": trade.get("exit_time"),  # Added
                    "entry_price": trade.get("entry_price", 0),
                    "exit_price": trade.get("exit_price", 0),
                    "pnl": trade.get("pnl", 0),
                    "pnl_net": trade.get("pnl_net", 0),
                    "direction": trade.get("direction", "Unknown"),
                    "size": trade.get("size", 0),
                    "duration": trade.get("bars_held", 0),
                    "commission": trade.get("commission", 0),
                    "status": trade.get("status", "Unknown"),
                }
                trades.append(trade_info)

        except Exception as e:
            print(f"Error extracting from strategy: {e}")

    if not trades:
        print("No trades found in analyzer or strategy")

    return trades


def process_window(args):
    (
        window_num,
        train_start,
        train_end,
        test_start,
        test_end,
        data,
        strategy_name,  # changed from strategy_class
        optimization_params,
        optimization_metric,
        initial_cash,
        commission,
        n_trials,
    ) = args

    # Resolve strategy class from name (multiprocessing safe)
    strategy_class = get_strategy(strategy_name)

    print(f"Processing window {window_num}")

    train_data = data[(data.index >= train_start) & (data.index <= train_end)]

    def objective(trial):
        params = {}
        for param_name, param_config in optimization_params.items():
            if param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"]
                )

        # Ensure slow_sma_period > fast_sma_period
        if "fast_sma_period" in params and "slow_sma_period" in params:
            if params["slow_sma_period"] <= params["fast_sma_period"]:
                params["slow_sma_period"] = params["fast_sma_period"] + 1

        cerebro = create_cerebro(
            train_data, strategy_class, params, initial_cash, commission
        )
        results = cerebro.run()
        strat = results[0]
        metrics = extract_metrics(strat)
        return metrics[optimization_metric]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"Window {window_num} best params: {best_params}")

    # In-sample evaluation
    cerebro_in = create_cerebro(
        train_data, strategy_class, best_params, initial_cash, commission
    )
    results_in = cerebro_in.run()
    strat_in = results_in[0]
    in_sample_metrics = extract_metrics(strat_in)
    in_sample_trades = extract_trades(strat_in)

    # Save in-sample trades to CSV
    if in_sample_trades:
        in_sample_trades_df = pd.DataFrame(in_sample_trades)
        in_sample_trades_df.to_csv(
            f"window_{window_num}_in_sample_trades.csv", index=False
        )
        print(
            f"Saved {len(in_sample_trades)} in-sample trades to window_{window_num}_in_sample_trades.csv"
        )
    else:
        print(f"No in-sample trades to save for window {window_num}")

    # Out-of-sample evaluation
    test_data = data[(data.index >= test_start) & (data.index <= test_end)]
    cerebro_out = create_cerebro(
        test_data, strategy_class, best_params, initial_cash, commission
    )
    results_out = cerebro_out.run()
    strat_out = results_out[0]
    out_sample_metrics = extract_metrics(strat_out)
    out_sample_trades = extract_trades(strat_out)

    # Save out-of-sample trades to CSV
    if out_sample_trades:
        out_sample_trades_df = pd.DataFrame(out_sample_trades)
        out_sample_trades_df.to_csv(
            f"window_{window_num}_out_sample_trades.csv", index=False
        )
        print(
            f"Saved {len(out_sample_trades)} out-of-sample trades to window_{window_num}_out_sample_trades.csv"
        )
    else:
        print(f"No out-of-sample trades to save for window {window_num}")

    return {
        "walk_forward": window_num,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "best_params": best_params,
        "in_sample_metrics": in_sample_metrics,
        "out_sample_metrics": out_sample_metrics,
        "in_sample_trades": in_sample_trades,
        "out_sample_trades": out_sample_trades,
    }


def extract_metrics(strat):
    metrics = {}

    # Get returns analysis
    returns_analysis = strat.analyzers.returns.get_analysis()
    metrics["total_return"] = returns_analysis["rtot"]

    # Handle Sharpe ratio calculation more robustly
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()

    # Try to get Sharpe ratio from analyzer first
    if (
        sharpe_analysis
        and "sharperatio" in sharpe_analysis
        and sharpe_analysis["sharperatio"] is not None
    ):
        # Check if it's a valid number (not NaN or infinite)
        sharpe_value = sharpe_analysis["sharperatio"]
        if not (pd.isna(sharpe_value) or np.isinf(sharpe_value)):
            metrics["sharpe_ratio"] = sharpe_value
        else:
            metrics["sharpe_ratio"] = None
    else:
        # Fallback: Calculate Sharpe ratio manually if possible
        try:
            if "rnorm" in returns_analysis:
                # Get normalized returns
                returns_value = returns_analysis.get("rnorm")
                # Check if returns_value is a dictionary or a single float
                if isinstance(returns_value, dict) and len(returns_value) > 1:
                    # Convert to pandas Series for easier calculation
                    returns_series = pd.Series(list(returns_value.values()))

                    # Calculate Sharpe ratio manually
                    if len(returns_series) > 1 and returns_series.std() > 0:
                        # Assuming risk-free rate is 0 for simplicity
                        sharpe_manual = (
                            returns_series.mean() / returns_series.std() * (252**0.5)
                        )  # Annualized
                        if not (pd.isna(sharpe_manual) or np.isinf(sharpe_manual)):
                            metrics["sharpe_ratio"] = sharpe_manual
                        else:
                            metrics["sharpe_ratio"] = None
                    else:
                        metrics["sharpe_ratio"] = None
                else:
                    # Handle case where rnorm is a single float or empty
                    metrics["sharpe_ratio"] = None
            else:
                metrics["sharpe_ratio"] = None
        except Exception as e:
            print(f"Error calculating Sharpe ratio manually: {e}")
            metrics["sharpe_ratio"] = None

    # Get drawdown
    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    metrics["max_drawdown"] = drawdown_analysis["max"]["drawdown"]

    timereturn = strat.analyzers.timereturn.get_analysis()
    equity_curve = pd.Series(timereturn)
    equity_curve = (1 + equity_curve).cumprod() * strat.cerebro.broker.startingcash
    metrics["equity_curve"] = equity_curve
    return metrics


def calculate_trade_statistics(trades):
    """Calculate comprehensive trade statistics"""
    if not trades:
        return {}

    df = pd.DataFrame(trades)

    # Basic statistics
    total_trades = len(df)
    winning_trades = df[df["pnl_net"] > 0]
    losing_trades = df[df["pnl_net"] < 0]
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

    # Profit metrics
    gross_profit = winning_trades["pnl_net"].sum() if not winning_trades.empty else 0
    gross_loss = losing_trades["pnl_net"].sum() if not losing_trades.empty else 0
    net_profit = df["pnl_net"].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss < 0 else float("inf")

    # Average values
    avg_win = winning_trades["pnl_net"].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades["pnl_net"].mean() if not losing_trades.empty else 0
    avg_trade = df["pnl_net"].mean()
    avg_duration = df["duration"].mean() if "duration" in df.columns else 0

    # Max values
    max_win = winning_trades["pnl_net"].max() if not winning_trades.empty else 0
    max_loss = losing_trades["pnl_net"].min() if not losing_trades.empty else 0
    max_duration = df["duration"].max() if "duration" in df.columns else 0

    # Directional statistics
    long_trades = df[df["direction"] == "Long"]
    short_trades = df[df["direction"] == "Short"]
    long_win_rate = (
        len(long_trades[long_trades["pnl_net"] > 0]) / len(long_trades)
        if len(long_trades) > 0
        else 0
    )
    short_win_rate = (
        len(short_trades[short_trades["pnl_net"] > 0]) / len(short_trades)
        if len(short_trades) > 0
        else 0
    )

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_trade": avg_trade,
        "max_win": max_win,
        "max_loss": max_loss,
        "avg_duration": avg_duration,
        "max_duration": max_duration,
        "num_winning_trades": len(winning_trades),
        "num_losing_trades": len(losing_trades),
        "num_long_trades": len(long_trades),
        "num_short_trades": len(short_trades),
    }


class WalkForwardAnalysis:
    def __init__(
        self,
        data,
        strategy_class,
        optimization_params,
        optimization_metric="total_return",
        training_ratio=0.8,
        testing_ratio=0.2,
        step_ratio=0.25,
        initial_cash=10000,
        commission=0.001,
        n_trials=50,
        verbose=False,
    ):
        self.data = data
        self.strategy_class = strategy_class
        self.optimization_params = optimization_params
        self.optimization_metric = optimization_metric
        self.training_ratio = training_ratio
        self.testing_ratio = testing_ratio
        self.step_ratio = step_ratio
        self.initial_cash = initial_cash
        self.commission = commission
        self.n_trials = n_trials
        self.verbose = verbose
        self.results = []

        # Calculate dynamic training and testing periods
        total_days = (data.index.max() - data.index.min()).days
        self.window_length = int(total_days * (training_ratio + testing_ratio))
        self.training_period = int(self.window_length * training_ratio)
        self.testing_period = self.window_length - self.training_period
        self.step_period = int(self.testing_period * step_ratio)

        # Debug print for period calculations
        print(f"Total days: {total_days}")
        print(f"Window length: {self.window_length} days")
        print(f"Training period: {self.training_period} days")
        print(f"Testing period: {self.testing_period} days")
        print(f"Step period: {self.step_period} days")

    def run_analysis(self):
        windows = []
        current_date = self.data.index.min()
        walk_forward_num = 1
        print(f"Starting window generation at {current_date}")

        # Corrected window generation logic
        while (
            current_date + timedelta(days=self.window_length) <= self.data.index.max()
        ):
            train_start = current_date
            train_end = current_date + timedelta(days=self.training_period - 1)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.testing_period - 1)

            windows.append(
                (walk_forward_num, train_start, train_end, test_start, test_end)
            )
            print(
                f"Window {walk_forward_num}: Train {train_start} to {train_end}, Test {test_start} to {test_end}"
            )

            current_date = current_date + timedelta(days=self.step_period)
            walk_forward_num += 1

        print(f"Generated {len(windows)} windows")
        with Pool(processes=4) as pool:
            args = [
                (
                    w[0],
                    w[1],
                    w[2],
                    w[3],
                    w[4],
                    self.data,
                    self.strategy_class,
                    self.optimization_params,
                    self.optimization_metric,
                    self.initial_cash,
                    self.commission,
                    self.n_trials,
                )
                for w in windows
            ]
            self.results = pool.map(process_window, args)

        # Aggregate all equity curves
        self.all_in_sample_equity = pd.DataFrame()
        self.all_out_sample_equity = pd.DataFrame()

        for result in self.results:
            window_id = result["walk_forward"]
            in_sample_curve = result["in_sample_metrics"].get(
                "equity_curve", pd.Series()
            )
            out_sample_curve = result["out_sample_metrics"].get(
                "equity_curve", pd.Series()
            )

            if not in_sample_curve.empty:
                self.all_in_sample_equity[f"Window {window_id}"] = in_sample_curve
            if not out_sample_curve.empty:
                self.all_out_sample_equity[f"Window {window_id}"] = out_sample_curve

    def get_overall_metrics(self):
        in_sample_returns = [
            r["in_sample_metrics"]["total_return"] for r in self.results
        ]
        out_sample_returns = [
            r["out_sample_metrics"]["total_return"] for r in self.results
        ]

        # Filter out None values and convert to float for Sharpe ratios
        in_sample_sharpes = []
        out_sample_sharpes = []

        for r in self.results:
            # In-sample Sharpe ratios
            in_sharpe = r["in_sample_metrics"]["sharpe_ratio"]
            if in_sharpe is not None and not (
                pd.isna(in_sharpe) or np.isinf(in_sharpe)
            ):
                in_sample_sharpes.append(float(in_sharpe))

            # Out-of-sample Sharpe ratios
            out_sharpe = r["out_sample_metrics"]["sharpe_ratio"]
            if out_sharpe is not None and not (
                pd.isna(out_sharpe) or np.isinf(out_sharpe)
            ):
                out_sample_sharpes.append(float(out_sharpe))

        overall = {
            "in_sample_avg_return": (
                sum(in_sample_returns) / len(in_sample_returns)
                if in_sample_returns
                else 0
            ),
            "out_sample_avg_return": (
                sum(out_sample_returns) / len(out_sample_returns)
                if out_sample_returns
                else 0
            ),
            "in_sample_avg_sharpe": (
                sum(in_sample_sharpes) / len(in_sample_sharpes)
                if in_sample_sharpes
                else None
            ),
            "out_sample_avg_sharpe": (
                sum(out_sample_sharpes) / len(out_sample_sharpes)
                if out_sample_sharpes
                else None
            ),
            "in_sample_sharpe_count": len(in_sample_sharpes),
            "out_sample_sharpe_count": len(out_sample_sharpes),
            "total_windows": len(self.results),
        }
        return overall

    def get_window_summary(self):
        """Return a DataFrame summarizing each walk-forward window"""
        rows = []
        for result in self.results:
            row = {
                "window": result["walk_forward"],
                "train_start": result["train_start"],
                "train_end": result["train_end"],
                "test_start": result["test_start"],
                "test_end": result["test_end"],
                "best_params": str(result["best_params"]),  # Convert dict to string
                # In-sample metrics
                "in_sample_total_return": result["in_sample_metrics"].get(
                    "total_return", 0
                ),
                "in_sample_sharpe_ratio": result["in_sample_metrics"].get(
                    "sharpe_ratio", None
                ),
                "in_sample_max_drawdown": result["in_sample_metrics"].get(
                    "max_drawdown", 0
                ),
                # Out-of-sample metrics
                "out_sample_total_return": result["out_sample_metrics"].get(
                    "total_return", 0
                ),
                "out_sample_sharpe_ratio": result["out_sample_metrics"].get(
                    "sharpe_ratio", None
                ),
                "out_sample_max_drawdown": result["out_sample_metrics"].get(
                    "max_drawdown", 0
                ),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def generate_trade_statistics(self):
        """Generate comprehensive trade statistics for all windows"""
        # Aggregate all trades
        all_in_sample_trades = []
        all_out_sample_trades = []

        for result in self.results:
            all_in_sample_trades.extend(result["in_sample_trades"])
            all_out_sample_trades.extend(result["out_sample_trades"])

        # Calculate statistics
        in_sample_stats = calculate_trade_statistics(all_in_sample_trades)
        out_sample_stats = calculate_trade_statistics(all_out_sample_trades)

        # Create summary DataFrame
        stats_summary = pd.DataFrame(
            {
                "Metric": [
                    "Total Trades",
                    "Win Rate",
                    "Long Win Rate",
                    "Short Win Rate",
                    "Gross Profit",
                    "Gross Loss",
                    "Net Profit",
                    "Profit Factor",
                    "Avg Win",
                    "Avg Loss",
                    "Avg Trade",
                    "Max Win",
                    "Max Loss",
                    "Avg Duration",
                    "Max Duration",
                    "Winning Trades",
                    "Losing Trades",
                    "Long Trades",
                    "Short Trades",
                ],
                "In-Sample": [
                    in_sample_stats.get("total_trades", 0),
                    in_sample_stats.get("win_rate", 0),
                    in_sample_stats.get("long_win_rate", 0),
                    in_sample_stats.get("short_win_rate", 0),
                    in_sample_stats.get("gross_profit", 0),
                    in_sample_stats.get("gross_loss", 0),
                    in_sample_stats.get("net_profit", 0),
                    in_sample_stats.get("profit_factor", 0),
                    in_sample_stats.get("avg_win", 0),
                    in_sample_stats.get("avg_loss", 0),
                    in_sample_stats.get("avg_trade", 0),
                    in_sample_stats.get("max_win", 0),
                    in_sample_stats.get("max_loss", 0),
                    in_sample_stats.get("avg_duration", 0),
                    in_sample_stats.get("max_duration", 0),
                    in_sample_stats.get("num_winning_trades", 0),
                    in_sample_stats.get("num_losing_trades", 0),
                    in_sample_stats.get("num_long_trades", 0),
                    in_sample_stats.get("num_short_trades", 0),
                ],
                "Out-of-Sample": [
                    out_sample_stats.get("total_trades", 0),
                    out_sample_stats.get("win_rate", 0),
                    out_sample_stats.get("long_win_rate", 0),
                    out_sample_stats.get("short_win_rate", 0),
                    out_sample_stats.get("gross_profit", 0),
                    out_sample_stats.get("gross_loss", 0),
                    out_sample_stats.get("net_profit", 0),
                    out_sample_stats.get("profit_factor", 0),
                    out_sample_stats.get("avg_win", 0),
                    out_sample_stats.get("avg_loss", 0),
                    out_sample_stats.get("avg_trade", 0),
                    out_sample_stats.get("max_win", 0),
                    out_sample_stats.get("max_loss", 0),
                    out_sample_stats.get("avg_duration", 0),
                    out_sample_stats.get("max_duration", 0),
                    out_sample_stats.get("num_winning_trades", 0),
                    out_sample_stats.get("num_losing_trades", 0),
                    out_sample_stats.get("num_long_trades", 0),
                    out_sample_stats.get("num_short_trades", 0),
                ],
            }
        )

        return stats_summary, all_in_sample_trades, all_out_sample_trades


# def run_walkforward_example():
#     # Example data and strategy
#     data = get_data_sync(
#         ticker="SBIN.NS", start_date="2020-01-01", end_date="2025-07-01", interval="1d"
#     )

#     # Debug print for data range
#     print(f"Data range: {data.index.min()} to {data.index.max()}")
#     print(f"Data columns: {data.columns}")
#     print(f"Data head:\n{data.head()}")
#     strategy_class = get_strategy("SMABollinger")

#     wf = WalkForwardAnalysis(
#         data=data,
#         strategy_class=strategy_class.__name__,
#         optimization_params=strategy_class.optimization_params,
#         optimization_metric="total_return",
#         training_ratio=0.6,
#         testing_ratio=0.15,
#         step_ratio=0.2,
#         n_trials=50,
#         verbose=False,
#     )
#     wf.run_analysis()

#     # Print results
#     print("\nWalk-Forward Analysis Results:")
#     for result in wf.results:
#         print(f"\nWindow {result['walk_forward']}:")
#         print(f"Training Period: {result['train_start']} to {result['train_end']}")
#         print(f"Testing Period: {result['test_start']} to {result['test_end']}")
#         print(f"Best Parameters: {result['best_params']}")
#         print(f"In-Sample Metrics: {result['in_sample_metrics']}")
#         print(f"Out-of-Sample Metrics: {result['out_sample_metrics']}")

#     # Generate trade statistics summary
#     stats_summary, all_in_sample, all_out_sample = wf.generate_trade_statistics()

#     # Save window summary with parameters
#     window_summary = wf.get_window_summary()
#     window_summary.to_csv("window_parameters_summary.csv", index=False)
#     print("\nSaved window parameters summary to 'window_parameters_summary.csv'")

#     # Save aggregated trades
#     if all_in_sample:
#         pd.DataFrame(all_in_sample).to_csv("all_in_sample_trades.csv", index=False)
#     if all_out_sample:
#         pd.DataFrame(all_out_sample).to_csv("all_out_sample_trades.csv", index=False)

#     # Print and save trade statistics
#     print("\nTrade Statistics Summary:")
#     print(stats_summary.to_string(index=False))
#     stats_summary.to_csv("trade_statistics_summary.csv", index=False)
#     print("\nSaved trade statistics to 'trade_statistics_summary.csv'")

#     # Print overall metrics
#     overall = wf.get_overall_metrics()
#     print("\nOverall Performance:")
#     print(f"In-Sample Avg Return: {overall['in_sample_avg_return']:.4f}")
#     print(f"Out-of-Sample Avg Return: {overall['out_sample_avg_return']:.4f}")

#     if overall["in_sample_avg_sharpe"] is not None:
#         print(
#             f"In-Sample Avg Sharpe: {overall['in_sample_avg_sharpe']:.4f} (based on {overall['in_sample_sharpe_count']} valid values)"
#         )
#     else:
#         print(f"In-Sample Avg Sharpe: N/A (no valid values found)")

#     if overall["out_sample_avg_sharpe"] is not None:
#         print(
#             f"Out-of-Sample Avg Sharpe: {overall['out_sample_avg_sharpe']:.4f} (based on {overall['out_sample_sharpe_count']} valid values)"
#         )
#     else:
#         print(f"Out-of-Sample Avg Sharpe: N/A (no valid values found)")

#     print(f"Total Windows: {overall['total_windows']}")


# if __name__ == "__main__":
#     run_walkforward_example()
