import backtrader as bt
import numpy as np
import optuna
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool

from comprehensive_backtesting.data import get_data_sync
from comprehensive_backtesting.sma_bollinger_band import SMABollinger


class StrategyTradeAnalyzer(bt.Analyzer):
    def __init__(self):
        self.trades = []

    def stop(self):
        """Called at the end of the backtest"""
        if hasattr(self.strategy, "get_completed_trades"):
            strategy_trades = self.strategy.get_completed_trades()
            for trade in strategy_trades:
                trade_info = {
                    "trade_id": trade.get("ref", len(self.trades) + 1),
                    "entry_time": trade.get("entry_time"),
                    "entry_price": trade.get("entry_price"),
                    "exit_time": trade.get("exit_time"),
                    "exit_price": trade.get("exit_price"),
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
    cerebro.addanalyzer(StrategyTradeAnalyzer, _name="tradehistory")
    return cerebro


def extract_trades(strat):
    """Extract trades from strategy"""
    trades = []
    if hasattr(strat.analyzers, "tradehistory"):
        try:
            trades = strat.analyzers.tradehistory.get_analysis()
            print(f"Extracted {len(trades)} trades from StrategyTradeAnalyzer")
        except Exception as e:
            print(f"Error extracting from analyzer: {e}")
    if not trades and hasattr(strat, "get_completed_trades"):
        try:
            strategy_trades = strat.get_completed_trades()
            print(f"Found {len(strategy_trades)} trades in strategy")
            for trade in strategy_trades:
                trade_info = {
                    "trade_id": trade.get("ref", len(trades) + 1),
                    "entry_time": trade.get("entry_time"),
                    "entry_price": trade.get("entry_price"),
                    "exit_time": trade.get("exit_time"),
                    "exit_price": trade.get("exit_price"),
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


def compute_trade_stats(trades):
    """Compute trade statistics from a list of trades"""
    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "avg_profit_per_trade": 0.0}

    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade["pnl_net"] > 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    avg_profit_per_trade = (
        sum(trade["pnl_net"] for trade in trades) / total_trades
        if total_trades > 0
        else 0.0
    )

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_profit_per_trade": avg_profit_per_trade,
    }


def process_window(args):
    (
        window_num,
        train_start,
        train_end,
        test_start,
        test_end,
        data,
        strategy_class,
        optimization_params,
        optimization_metric,
        initial_cash,
        commission,
        n_trials,
    ) = args
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
    cerebro_in = create_cerebro(
        train_data, strategy_class, best_params, initial_cash, commission
    )
    results_in = cerebro_in.run()
    strat_in = results_in[0]
    in_sample_metrics = extract_metrics(strat_in)
    in_sample_trades = extract_trades(strat_in)
    in_sample_trade_stats = compute_trade_stats(in_sample_trades)
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
    test_data = data[(data.index >= test_start) & (data.index <= test_end)]
    cerebro_out = create_cerebro(
        test_data, strategy_class, best_params, initial_cash, commission
    )
    results_out = cerebro_out.run()
    strat_out = results_out[0]
    out_sample_metrics = extract_metrics(strat_out)
    out_sample_trades = extract_trades(strat_out)
    out_sample_trade_stats = compute_trade_stats(out_sample_trades)
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
        "in_sample_trade_stats": in_sample_trade_stats,
        "out_sample_trade_stats": out_sample_trade_stats,
    }


def extract_metrics(strat):
    metrics = {}
    returns_analysis = strat.analyzers.returns.get_analysis()
    metrics["total_return"] = returns_analysis["rtot"]
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    if (
        sharpe_analysis
        and "sharperatio" in sharpe_analysis
        and sharpe_analysis["sharperatio"] is not None
    ):
        sharpe_value = sharpe_analysis["sharperatio"]
        if not (pd.isna(sharpe_value) or np.isinf(sharpe_value)):
            metrics["sharpe_ratio"] = sharpe_value
        else:
            metrics["sharpe_ratio"] = None
    else:
        try:
            if "rnorm" in returns_analysis:
                returns_value = returns_analysis.get("rnorm")
                if isinstance(returns_value, dict) and len(returns_value) > 1:
                    returns_series = pd.Series(list(returns_value.values()))
                    if len(returns_series) > 1 and returns_series.std() > 0:
                        sharpe_manual = (
                            returns_series.mean() / returns_series.std() * (252**0.5)
                        )
                        if not (pd.isna(sharpe_manual) or np.isinf(sharpe_manual)):
                            metrics["sharpe_ratio"] = sharpe_manual
                        else:
                            metrics["sharpe_ratio"] = None
                    else:
                        metrics["sharpe_ratio"] = None
                else:
                    metrics["sharpe_ratio"] = None
            else:
                metrics["sharpe_ratio"] = None
        except Exception as e:
            print(f"Error calculating Sharpe ratio manually: {e}")
            metrics["sharpe_ratio"] = None
    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    metrics["max_drawdown"] = drawdown_analysis["max"]["drawdown"]
    return metrics


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
        total_days = (data.index.max() - data.index.min()).days
        self.training_period = int(total_days * self.training_ratio)
        self.testing_period = int(total_days * self.testing_ratio)
        self.step_period = int(self.testing_period * self.step_ratio)
        print(f"Total days: {total_days}")
        print(f"Training period: {self.training_period} days")
        print(f"Testing period: {self.testing_period} days")
        print(f"Step period: {self.step_period} days")

    def run_analysis(self):
        windows = []
        current_date = self.data.index.min()
        walk_forward_num = 1
        print(f"Starting window generation at {current_date}")

        # Fixed condition: ensure we have enough data for both training and testing
        while (
            current_date + timedelta(days=self.training_period + self.testing_period)
            <= self.data.index.max()
        ):
            train_start = current_date
            train_end = current_date + timedelta(days=self.training_period - 1)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.testing_period - 1)

            # Ensure test_end doesn't exceed data range
            if test_end > self.data.index.max():
                test_end = self.data.index.max()

            # Only add window if we have sufficient test data
            if test_end > test_start:
                windows.append(
                    (walk_forward_num, train_start, train_end, test_start, test_end)
                )
                print(
                    f"Window {walk_forward_num}: Train {train_start} to {train_end}, Test {test_start} to {test_end}"
                )
                walk_forward_num += 1

            # Move to next window
            current_date = current_date + timedelta(days=self.step_period)

            # Additional safety check to prevent infinite loop
            if current_date >= self.data.index.max():
                break

        print(f"Generated {len(windows)} windows")

        # Debug: Show expected vs actual window count
        expected_windows = (
            max(
                1,
                (self.data.index.max() - self.data.index.min()).days
                - self.training_period
                - self.testing_period
                + 1,
            )
            // self.step_period
            + 1
        )
        print(f"Expected approximately {expected_windows} windows based on step size")

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

    def get_overall_metrics(self):
        in_sample_returns = [
            r["in_sample_metrics"]["total_return"] for r in self.results
        ]
        out_sample_returns = [
            r["out_sample_metrics"]["total_return"] for r in self.results
        ]
        in_sample_sharpes = []
        out_sample_sharpes = []
        in_sample_max_drawdowns = [
            r["in_sample_metrics"]["max_drawdown"] for r in self.results
        ]
        out_sample_max_drawdowns = [
            r["out_sample_metrics"]["max_drawdown"] for r in self.results
        ]

        # Aggregate trade statistics
        all_in_sample_trades = [
            trade for r in self.results for trade in r["in_sample_trades"]
        ]
        all_out_sample_trades = [
            trade for r in self.results for trade in r["out_sample_trades"]
        ]
        in_sample_trade_stats = compute_trade_stats(all_in_sample_trades)
        out_sample_trade_stats = compute_trade_stats(all_out_sample_trades)

        for r in self.results:
            in_sharpe = r["in_sample_metrics"]["sharpe_ratio"]
            if in_sharpe is not None and not (
                pd.isna(in_sharpe) or np.isinf(in_sharpe)
            ):
                in_sample_sharpes.append(float(in_sharpe))
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
            "in_sample_max_drawdown": (
                max(in_sample_max_drawdowns) if in_sample_max_drawdowns else 0
            ),
            "out_sample_max_drawdown": (
                max(out_sample_max_drawdowns) if out_sample_max_drawdowns else 0
            ),
            "in_sample_trade_stats": in_sample_trade_stats,
            "out_sample_trade_stats": out_sample_trade_stats,
            "in_sample_sharpe_count": len(in_sample_sharpes),
            "out_sample_sharpe_count": len(out_sample_sharpes),
            "total_windows": len(self.results),
        }
        return overall


def run_walkforward_example():
    data = get_data_sync(
        ticker="SBIN.NS", start_date="2020-01-01", end_date="2025-07-01", interval="1d"
    )
    print(f"Data range: {data.index.min()} to {data.index.max()}")
    print(f"Data columns: {data.columns}")
    print(f"Data head:\n{data.head()}")

    wf = WalkForwardAnalysis(
        data=data,
        strategy_class=SMABollinger,
        optimization_params=SMABollinger.optimization_params,
        optimization_metric="total_return",
        training_ratio=0.6,
        testing_ratio=0.15,
        step_ratio=0.2,
        n_trials=50,
        verbose=False,
    )
    wf.run_analysis()
    print("Walk-Forward Analysis Results:")
    for result in wf.results:
        print(f"\nWindow {result['walk_forward']}:")
        print(f"Training Period: {result['train_start']} to {result['train_end']}")
        print(f"Testing Period: {result['test_start']} to {result['test_end']}")
        print(f"Best Parameters: {result['best_params']}")
        print(f"In-Sample Metrics: {result['in_sample_metrics']}")
        print(f"In-Sample Trade Stats: {result['in_sample_trade_stats']}")
        print(f"Out-of-Sample Metrics: {result['out_sample_metrics']}")
        print(f"Out-of-Sample Trade Stats: {result['out_sample_trade_stats']}")

    overall = wf.get_overall_metrics()
    print("\nOverall Performance:")
    print(f"In-Sample Avg Return: {overall['in_sample_avg_return']:.4f}")
    print(f"Out-of-Sample Avg Return: {overall['out_sample_avg_return']:.4f}")
    print(f"In-Sample Max Drawdown: {overall['in_sample_max_drawdown']:.2f}%")
    print(f"Out-of-Sample Max Drawdown: {overall['out_sample_max_drawdown']:.2f}%")

    if overall["in_sample_avg_sharpe"] is not None:
        print(
            f"In-Sample Avg Sharpe: {overall['in_sample_avg_sharpe']:.4f} (based on {overall['in_sample_sharpe_count']} valid values)"
        )
    else:
        print(f"In-Sample Avg Sharpe: N/A (no valid values found)")

    if overall["out_sample_avg_sharpe"] is not None:
        print(
            f"Out-of-Sample Avg Sharpe: {overall['out_sample_avg_sharpe']:.4f} (based on {overall['out_sample_sharpe_count']} valid values)"
        )
    else:
        print(f"Out-of-Sample Avg Sharpe: N/A (no valid values found)")

    print("\nIn-Sample Trade Statistics:")
    print(f"Total Trades: {overall['in_sample_trade_stats']['total_trades']}")
    print(f"Win Rate: {overall['in_sample_trade_stats']['win_rate']:.2f}%")
    print(
        f"Average Profit per Trade: ${overall['in_sample_trade_stats']['avg_profit_per_trade']:.2f}"
    )

    print("\nOut-of-Sample Trade Statistics:")
    print(f"Total Trades: {overall['out_sample_trade_stats']['total_trades']}")
    print(f"Win Rate: {overall['out_sample_trade_stats']['win_rate']:.2f}%")
    print(
        f"Average Profit per Trade: ${overall['out_sample_trade_stats']['avg_profit_per_trade']:.2f}"
    )

    print(f"\nTotal Windows: {overall['total_windows']}")


if __name__ == "__main__":
    run_walkforward_example()
