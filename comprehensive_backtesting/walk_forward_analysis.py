import backtrader as bt
import numpy as np
import optuna
import pandas as pd
from datetime import timedelta
from multiprocessing import Pool, get_context

from comprehensive_backtesting.registry import get_strategy


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


def create_cerebro(data, strategy_class, params, initial_cash, commission):
    cerebro = bt.Cerebro()

    # Add data feed
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # Add strategy
    cerebro.addstrategy(strategy_class, **params)

    # Set broker parameters
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0, annualize=True
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Add TimeReturn analyzer for daily returns (crucial for Sharpe calculation)
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        timeframe=bt.TimeFrame.Days,
        compression=1,
        _name="timereturn",
    )

    # Add our custom trade analyzer
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
        strategy_name,  # Strategy name to resolve the class
        optimization_params,
        optimization_metric,
        initial_cash,
        commission,
        n_trials,
        _,
    ) = args

    # Resolve strategy class from name (multiprocessing safe)
    strategy_class = get_strategy(strategy_name)

    print(f"Processing window {window_num}")

    train_data = data[(data.index >= train_start) & (data.index <= train_end)]

    def objective(trial):
        params = {}
        print(f"Optimization params: {optimization_params}")
        for param_name, param_config in optimization_params.items():
            param_type = param_config.get("type")

            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config.get("low", 1),  # Default to 1 if "low" is missing
                    param_config.get(
                        "high", 100
                    ),  # Default to 100 if "high" is missing
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config.get("low", 0.0),  # Default to 0.0 if "low" is missing
                    param_config.get(
                        "high", 1.0
                    ),  # Default to 1.0 if "high" is missing
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config.get(
                        "choices", []
                    ),  # Default to empty list if "choices" is missing
                )
            elif param_type == "loguniform":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config.get("low", 1e-5),  # Default to small value
                    param_config.get("high", 1.0),  # Default to 1.0
                    log=True,
                )
            else:
                print(
                    f"Warning: Unknown parameter type '{param_type}' for {param_name}, skipping."
                )
                continue

        # Ensure slow_sma_period > fast_sma_period for strategies that require it
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

    # Out-of-sample evaluation
    test_data = data[(data.index >= test_start) & (data.index <= test_end)]
    cerebro_out = create_cerebro(
        test_data, strategy_class, best_params, initial_cash, commission
    )
    results_out = cerebro_out.run()
    strat_out = results_out[0]
    out_sample_metrics = extract_metrics(strat_out)
    out_sample_trades = extract_trades(strat_out)

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
    metrics["total_return"] = returns_analysis.get("rtot", 0)

    # Get TimeReturn data for proper Sharpe calculation
    timereturn_analysis = strat.analyzers.timereturn.get_analysis()

    # Calculate Sharpe ratio using TimeReturn data
    if timereturn_analysis and len(timereturn_analysis) > 1:
        # Convert to pandas Series for easier calculation
        returns_series = pd.Series(timereturn_analysis)

        # Remove any NaN or infinite values
        returns_series = returns_series.replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns_series) > 1 and returns_series.std() > 0:
            # Calculate annualized Sharpe ratio (assuming daily returns)
            # Risk-free rate assumed to be 0 for simplicity
            daily_mean = returns_series.mean()
            daily_std = returns_series.std()

            # Annualize (252 trading days per year)
            annualized_return = (1 + daily_mean) ** 252 - 1
            annualized_volatility = daily_std * np.sqrt(252)

            if annualized_volatility > 0:
                sharpe_ratio = annualized_return / annualized_volatility

                # Validate the Sharpe ratio
                if not (pd.isna(sharpe_ratio) or np.isinf(sharpe_ratio)):
                    metrics["sharpe_ratio"] = sharpe_ratio
                else:
                    metrics["sharpe_ratio"] = 0.0
            else:
                metrics["sharpe_ratio"] = 0.0
        else:
            metrics["sharpe_ratio"] = 0.0
    else:
        # Fallback: try the built-in Sharpe analyzer
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        if (
            sharpe_analysis
            and "sharperatio" in sharpe_analysis
            and sharpe_analysis["sharperatio"] is not None
            and not (
                pd.isna(sharpe_analysis["sharperatio"])
                or np.isinf(sharpe_analysis["sharperatio"])
            )
        ):
            metrics["sharpe_ratio"] = sharpe_analysis["sharperatio"]
        else:
            metrics["sharpe_ratio"] = 0.0

    # Get drawdown
    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    metrics["max_drawdown"] = drawdown_analysis.get("max", {}).get("drawdown", 0)

    # Create equity curve from TimeReturn data
    if timereturn_analysis:
        equity_curve = pd.Series(timereturn_analysis)
        # Convert to cumulative returns starting from initial cash
        equity_curve = (1 + equity_curve).cumprod() * strat.cerebro.broker.startingcash
        metrics["equity_curve"] = equity_curve
    else:
        # Fallback: create simple equity curve
        metrics["equity_curve"] = pd.Series([strat.cerebro.broker.startingcash])

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
        training_ratio=0.5,
        gap_ratio=0.2,
        testing_ratio=0.3,
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
        self.gap_ratio = gap_ratio
        self.testing_ratio = testing_ratio
        self.step_ratio = step_ratio
        self.initial_cash = initial_cash
        self.commission = commission
        self.n_trials = n_trials
        self.verbose = verbose
        self.results = []

        # Calculate dynamic training, gap, and testing periods
        total_days = (data.index.max() - data.index.min()).days
        self.window_length = int(
            total_days * (training_ratio + gap_ratio + testing_ratio)
        )
        self.training_period = int(self.window_length * training_ratio)
        self.gap_period = int(self.window_length * gap_ratio)
        self.testing_period = int(self.window_length * testing_ratio)
        self.step_period = int(self.testing_period * step_ratio)
        self.module_name = __name__

        # Debug print for period calculations
        print(f"Total days: {total_days}")
        print(f"Window length: {self.window_length} days")
        print(f"Training period: {self.training_period} days")
        print(f"Gap period: {self.gap_period} days")
        print(f"Testing period: {self.testing_period} days")
        print(f"Step period: {self.step_period} days")
        print(f"Module name: {self.module_name}")

    def run_analysis(self):
        windows = []
        current_date = self.data.index.min()
        walk_forward_num = 1
        print(f"Starting window generation at {current_date}")

        # Window generation logic with gap period
        while (
            current_date + timedelta(days=self.window_length) <= self.data.index.max()
        ):
            train_start = current_date
            train_end = current_date + timedelta(days=self.training_period - 1)
            gap_start = train_end + timedelta(days=1)
            gap_end = gap_start + timedelta(days=self.gap_period - 1)
            test_start = gap_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.testing_period - 1)

            windows.append(
                (walk_forward_num, train_start, train_end, test_start, test_end)
            )
            print(
                f"Window {walk_forward_num}: Train {train_start} to {train_end}, "
                f"Gap {gap_start} to {gap_end}, Test {test_start} to {test_end}"
            )

            current_date = current_date + timedelta(days=self.step_period)
            walk_forward_num += 1

        print(f"Generated {len(windows)} windows")
        ctx = get_context("spawn")
        import multiprocessing as mp

        with ctx.Pool(processes=min(4, mp.cpu_count())) as pool:
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
                    self.module_name,  # Pass module name
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
        in_sample_returns = []
        out_sample_returns = []
        in_sample_sharpes = []
        out_sample_sharpes = []

        for r in self.results:
            # Collect returns
            in_return = r["in_sample_metrics"].get("total_return", 0)
            out_return = r["out_sample_metrics"].get("total_return", 0)

            if in_return is not None:
                in_sample_returns.append(in_return)
            if out_return is not None:
                out_sample_returns.append(out_return)

            # Collect Sharpe ratios (now should be valid numbers)
            in_sharpe = r["in_sample_metrics"].get("sharpe_ratio", 0)
            out_sharpe = r["out_sample_metrics"].get("sharpe_ratio", 0)

            # Since we now ensure sharpe_ratio is always a valid number (0.0 if invalid)
            if in_sharpe is not None and not (
                pd.isna(in_sharpe) or np.isinf(in_sharpe)
            ):
                in_sample_sharpes.append(float(in_sharpe))

            if out_sharpe is not None and not (
                pd.isna(out_sharpe) or np.isinf(out_sharpe)
            ):
                out_sample_sharpes.append(float(out_sharpe))

        # Calculate averages
        in_sample_avg_return = (
            sum(in_sample_returns) / len(in_sample_returns) if in_sample_returns else 0
        )
        out_sample_avg_return = (
            sum(out_sample_returns) / len(out_sample_returns)
            if out_sample_returns
            else 0
        )
        in_sample_avg_sharpe = (
            sum(in_sample_sharpes) / len(in_sample_sharpes) if in_sample_sharpes else 0
        )
        out_sample_avg_sharpe = (
            sum(out_sample_sharpes) / len(out_sample_sharpes)
            if out_sample_sharpes
            else 0
        )

        # Calculate win rates
        positive_out_sample = sum(1 for r in out_sample_returns if r > 0)
        win_rate_out_sample = (
            (positive_out_sample / len(out_sample_returns) * 100)
            if out_sample_returns
            else 0
        )

        # Calculate correlation between in-sample and out-sample returns
        correlation = 0
        if (
            len(in_sample_returns) == len(out_sample_returns)
            and len(in_sample_returns) > 1
        ):
            try:
                correlation = np.corrcoef(in_sample_returns, out_sample_returns)[0, 1]
                if pd.isna(correlation):
                    correlation = 0
            except:
                correlation = 0

        # Calculate average degradation
        degradations = []
        for i in range(min(len(in_sample_returns), len(out_sample_returns))):
            degradation = in_sample_returns[i] - out_sample_returns[i]
            degradations.append(degradation)

        avg_degradation = sum(degradations) / len(degradations) if degradations else 0

        overall = {
            "total_windows": len(self.results),
            "in_sample_return_avg_return": in_sample_avg_return
            * 100,  # Convert to percentage
            "out_sample_avg_return": out_sample_avg_return
            * 100,  # Convert to percentage
            "in_sample_avg_sharpe": in_sample_avg_sharpe,
            "out_sample_avg_sharpe": out_sample_avg_sharpe,
            "win_rate_out_sample": win_rate_out_sample,
            "correlation": correlation,
            "avg_degradation": avg_degradation * 100,  # Convert to percentage
            "in_sample_sharpe_count": len(in_sample_sharpes),
            "out_sample_sharpe_count": len(out_sample_sharpes),
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
