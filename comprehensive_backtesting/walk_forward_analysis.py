import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
import warnings
import ast
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pytz

from comprehensive_backtesting.data import get_data_sync
from comprehensive_backtesting.ema_rsi import EMARSI
from comprehensive_backtesting.sma_bollinger_band import SMABollinger

warnings.filterwarnings("ignore")


def extract_indicator_values_from_strategy(
    strategy: bt.Strategy, dt: datetime
) -> Dict[str, float]:
    """Extract indicator values from strategy at a given datetime."""
    indicators = {}
    try:
        # Ensure datetime is timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        else:
            dt = dt.astimezone(pytz.UTC)

        # Find the closest data index
        data_dt = data_dt = [
            (
                pytz.utc.localize(bt.num2date(d))
                if bt.num2date(d).tzinfo is None
                else bt.num2date(d).astimezone(pytz.utc)
            )
            for d in strategy.data.datetime.array
        ]
        closest_idx = min(range(len(data_dt)), key=lambda i: abs(data_dt[i] - dt))

        # Extract SMABollinger indicators
        for attr_name in ["fast_sma", "slow_sma", "bb_upper", "bb_lower"]:
            if hasattr(strategy, attr_name):
                indicator = getattr(strategy, attr_name)
                if hasattr(indicator, "lines"):
                    value = indicator.lines[0][closest_idx]
                    indicators[attr_name] = (
                        float(value) if not np.isnan(value) else None
                    )
                if strategy.params.verbose:
                    print(
                        f"Extracted {attr_name} at {dt}: {indicators.get(attr_name, 'None')}"
                    )
    except Exception as e:
        if strategy.params.verbose:
            print(f"Error extracting indicators at {dt}: {e}")
    return indicators


def extract_trades(
    strategy_result: bt.Strategy, data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Improved trade extraction that prioritizes strategy's own trade tracking.
    This should be much more reliable than the original version.
    """
    trades = []

    # Method 1: Use strategy's own completed trades (if available)
    if hasattr(strategy_result, "get_completed_trades"):
        try:
            completed_trades = strategy_result.get_completed_trades()
            if completed_trades:
                if strategy_result.params.verbose:
                    print(
                        f"Found {len(completed_trades)} trades from strategy.get_completed_trades()"
                    )
                return pd.DataFrame(completed_trades)
        except Exception as e:
            if strategy_result.params.verbose:
                print(f"Error getting completed trades from strategy: {e}")

    # Method 2: Use BackTrader's built-in trade analyzer
    if hasattr(strategy_result, "analyzers") and hasattr(
        strategy_result.analyzers, "trades"
    ):
        try:
            trades_analyzer = strategy_result.analyzers.trades
            trades_analysis = trades_analyzer.get_analysis()

            # Extract individual trades from the analyzer
            if "total" in trades_analysis and "trades" in trades_analysis:
                bt_trades = trades_analysis["trades"]
                for trade in bt_trades:
                    trade_info = {
                        "ref": getattr(trade, "ref", 0),
                        "entry_time": (
                            bt.num2date(trade.dtopen)
                            if hasattr(trade, "dtopen")
                            else None
                        ),
                        "exit_time": (
                            bt.num2date(trade.dtclose)
                            if hasattr(trade, "dtclose")
                            else None
                        ),
                        "entry_price": getattr(trade, "price", 0),
                        "exit_price": getattr(trade, "price", 0)
                        + (getattr(trade, "pnl", 0) / getattr(trade, "size", 1)),
                        "size": abs(getattr(trade, "size", 0)),
                        "pnl": getattr(trade, "pnl", 0),
                        "pnl_net": getattr(trade, "pnlcomm", getattr(trade, "pnl", 0)),
                        "commission": getattr(trade, "commission", 0),
                        "status": "Won" if getattr(trade, "pnl", 0) > 0 else "Lost",
                        "direction": (
                            "Long" if getattr(trade, "size", 0) > 0 else "Short"
                        ),
                        "bars_held": getattr(trade, "barlen", 0),
                    }
                    trades.append(trade_info)

                if trades and strategy_result.params.verbose:
                    print(f"Found {len(trades)} trades from TradeAnalyzer")

        except Exception as e:
            if strategy_result.params.verbose:
                print(f"Error extracting from TradeAnalyzer: {e}")

    # Method 3: Access BackTrader's internal trade list
    if not trades and hasattr(strategy_result, "_trades"):
        try:
            if strategy_result.params.verbose:
                print(
                    f"Found {len(strategy_result._trades)} trades in strategy._trades"
                )

            for trade_obj in strategy_result._trades:
                if hasattr(trade_obj, "isclosed") and trade_obj.isclosed:
                    try:
                        entry_dt = (
                            bt.num2date(trade_obj.dtopen)
                            if hasattr(trade_obj, "dtopen")
                            else None
                        )
                        exit_dt = (
                            bt.num2date(trade_obj.dtclose)
                            if hasattr(trade_obj, "dtclose")
                            else None
                        )

                        if entry_dt and entry_dt.tzinfo is None:
                            entry_dt = entry_dt.replace(tzinfo=pytz.UTC)
                        if exit_dt and exit_dt.tzinfo is None:
                            exit_dt = exit_dt.replace(tzinfo=pytz.UTC)

                        trade_info = {
                            "ref": getattr(trade_obj, "ref", 0),
                            "entry_time": entry_dt,
                            "exit_time": exit_dt,
                            "entry_price": getattr(trade_obj, "price", 0),
                            "exit_price": getattr(trade_obj, "price", 0)
                            + (
                                getattr(trade_obj, "pnl", 0)
                                / getattr(trade_obj, "size", 1)
                            ),
                            "size": abs(getattr(trade_obj, "size", 0)),
                            "pnl": getattr(trade_obj, "pnl", 0),
                            "pnl_net": getattr(
                                trade_obj, "pnlcomm", getattr(trade_obj, "pnl", 0)
                            ),
                            "commission": getattr(trade_obj, "commission", 0),
                            "status": (
                                "Won" if getattr(trade_obj, "pnl", 0) > 0 else "Lost"
                            ),
                            "direction": (
                                "Long" if getattr(trade_obj, "size", 0) > 0 else "Short"
                            ),
                            "bars_held": getattr(trade_obj, "barlen", 0),
                        }
                        trades.append(trade_info)

                    except Exception as e:
                        if strategy_result.params.verbose:
                            print(f"Error processing trade: {e}")
                        continue

        except Exception as e:
            if strategy_result.params.verbose:
                print(f"Error accessing _trades: {e}")

    # Method 4: Fallback to order-based extraction (your original method, but improved)
    if (
        not trades
        and hasattr(strategy_result, "broker")
        and hasattr(strategy_result.broker, "orders")
    ):
        try:
            orders = strategy_result.broker.orders
            completed_orders = [
                order for order in orders if order.status == 4
            ]  # Status 4 = Completed

            # Group orders by pairs (buy followed by sell)
            buy_orders = [o for o in completed_orders if o.isbuy()]
            sell_orders = [o for o in completed_orders if o.issell()]

            # Match buy and sell orders
            min_pairs = min(len(buy_orders), len(sell_orders))

            for i in range(min_pairs):
                try:
                    buy_order = buy_orders[i]
                    sell_order = sell_orders[i]

                    entry_dt = bt.num2date(buy_order.executed.dt)
                    exit_dt = bt.num2date(sell_order.executed.dt)

                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=pytz.UTC)
                    if exit_dt.tzinfo is None:
                        exit_dt = exit_dt.replace(tzinfo=pytz.UTC)

                    pnl = (sell_order.executed.price - buy_order.executed.price) * abs(
                        buy_order.executed.size
                    )
                    total_commission = buy_order.executed.comm + abs(
                        sell_order.executed.comm
                    )
                    pnl_net = pnl - total_commission

                    trade_info = {
                        "ref": buy_order.ref,
                        "entry_time": entry_dt,
                        "exit_time": exit_dt,
                        "entry_price": buy_order.executed.price,
                        "exit_price": sell_order.executed.price,
                        "size": abs(buy_order.executed.size),
                        "pnl": pnl,
                        "pnl_net": pnl_net,
                        "commission": total_commission,
                        "status": "Won" if pnl > 0 else "Lost",
                        "direction": "Long",
                        "bars_held": (exit_dt - entry_dt).days,
                    }
                    trades.append(trade_info)

                except Exception as e:
                    if strategy_result.params.verbose:
                        print(f"Error processing order pair {i}: {e}")
                    continue

            if trades and strategy_result.params.verbose:
                print(f"Found {len(trades)} trades from order matching")

        except Exception as e:
            if strategy_result.params.verbose:
                print(f"Error in order-based extraction: {e}")

    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        trades_df = trades_df.drop_duplicates(subset=["ref"])
        if strategy_result.params.verbose:
            print(f"Final result: {len(trades_df)} unique trades extracted")
    else:
        if strategy_result.params.verbose:
            print("No trades extracted by any method")

    return trades_df


class WalkForwardAnalysis:
    """
    A comprehensive walk-forward analysis class for backtrader strategies using Optuna optimization.
    """

    def __init__(
        self,
        strategy_class: bt.Strategy,
        data: pd.DataFrame,
        training_period: int = 252,
        testing_period: int = 63,
        optimization_params: Dict[str, Dict] = None,
        optimization_metric: str = "sharpe_ratio",
        initial_cash: float = 100000,
        commission: float = 0.09,  # Flat commission per log
        n_trials: int = 50,
        timeout: Optional[int] = None,
        verbose: bool = True,
        optuna_sampler: Optional[optuna.samplers.BaseSampler] = None,
        optuna_pruner: Optional[optuna.pruners.BasePruner] = None,
    ):
        """
        Initialize the walk-forward analysis with Optuna optimization.
        """
        self.strategy_class = strategy_class
        self.data = data
        self.training_period = training_period
        self.testing_period = testing_period
        self.optimization_params = optimization_params or getattr(
            strategy_class, "optimization_params", {}
        )
        self.optimization_metric = optimization_metric
        self.initial_cash = initial_cash
        self.commission = commission
        self.n_trials = n_trials
        self.timeout = timeout
        self.verbose = verbose
        self.results = []
        self.all_trades = []

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")

        if not self.optimization_params:
            if self.verbose:
                print(
                    "Warning: No optimization parameters provided. Running without optimization."
                )

        self.sampler = optuna_sampler or TPESampler(seed=42)
        self.pruner = optuna_pruner or MedianPruner(
            n_startup_trials=5, n_warmup_steps=10
        )

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _create_cerebro(
        self, start_date: datetime, end_date: datetime, params: Dict = None
    ) -> bt.Cerebro:
        """Create a cerebro instance with specified parameters."""
        cerebro = bt.Cerebro()

        mask = (self.data.index >= start_date) & (
            self.data.index <= end_date + timedelta(days=1)
        )
        filtered_df = self.data.loc[mask].copy()

        if len(filtered_df) == 0:
            if self.verbose:
                print(f"No data found for period {start_date} to {end_date}")
            raise ValueError(f"No data found for period {start_date} to {end_date}")

        if self.verbose:
            print(
                f"Data filtered: {len(filtered_df)} bars from {filtered_df.index.min()} to {filtered_df.index.max()}"
            )
            print(
                f"Price range: Close min={filtered_df['Close'].min():.2f}, max={filtered_df['Close'].max():.2f}"
            )

        data_feed = bt.feeds.PandasData(dataname=filtered_df)
        cerebro.adddata(data_feed)

        if params:
            params["verbose"] = self.verbose
            if self.verbose:
                print(f"Initializing strategy with params: {params}")
            cerebro.addstrategy(self.strategy_class, **params)
        else:
            cerebro.addstrategy(self.strategy_class, verbose=self.verbose)

        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(
            commission=self.commission, commtype=bt.CommInfoBase.COMM_FIXED
        )

        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.VWR, _name="vwr")

        return cerebro

    def _extract_performance_metrics(self, strat) -> Dict[str, float]:
        """Extract performance metrics from strategy with analyzers - IMPROVED VERSION."""
        metrics = {}

        try:
            analyzers = strat.analyzers

            returns_analyzer = getattr(analyzers, "returns", None)
            if returns_analyzer:
                returns = returns_analyzer.get_analysis()
                metrics["total_return"] = returns.get("rtot", 0.0)
                metrics["avg_return"] = returns.get("ravg", 0.0)
            else:
                metrics["total_return"] = 0.0
                metrics["avg_return"] = 0.0

            sharpe_analyzer = getattr(analyzers, "sharpe", None)
            if sharpe_analyzer:
                sharpe = sharpe_analyzer.get_analysis()
                metrics["sharpe_ratio"] = sharpe.get("sharperatio", 0.0) or 0.0
            else:
                metrics["sharpe_ratio"] = 0.0

            drawdown_analyzer = getattr(analyzers, "drawdown", None)
            if drawdown_analyzer:
                drawdown = drawdown_analyzer.get_analysis()
                metrics["max_drawdown"] = drawdown.get("max", {}).get("drawdown", 0.0)
            else:
                metrics["max_drawdown"] = 0.0

            vwr_analyzer = getattr(analyzers, "vwr", None)
            if vwr_analyzer:
                vwr = vwr_analyzer.get_analysis()
                metrics["vwr"] = vwr.get("vwr", 0.0) or 0.0
            else:
                metrics["vwr"] = 0.0

            trades_analyzer = getattr(analyzers, "trades", None)
            if trades_analyzer:
                trades = trades_analyzer.get_analysis()
                metrics["total_trades"] = trades.get("total", {}).get("total", 0)
                metrics["winning_trades"] = trades.get("won", {}).get("total", 0)
                metrics["losing_trades"] = trades.get("lost", {}).get("total", 0)

                win_rate = 0.0
                if metrics["total_trades"] > 0:
                    win_rate = metrics["winning_trades"] / metrics["total_trades"]
                metrics["win_rate"] = win_rate

                gross_profit = trades.get("won", {}).get("pnl", {}).get("total", 0.0)
                gross_loss = abs(
                    trades.get("lost", {}).get("pnl", {}).get("total", 0.0)
                )
                metrics["profit_factor"] = (
                    gross_profit / gross_loss if gross_loss > 0 else 0.0
                )

                if metrics["total_trades"] > 0:
                    metrics["avg_trade_pnl"] = trades.get("pnl", {}).get("average", 0.0)
                else:
                    metrics["avg_trade_pnl"] = 0.0
            else:
                metrics.update(
                    {
                        "total_trades": 0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "win_rate": 0.0,
                        "profit_factor": 0.0,
                        "avg_trade_pnl": 0.0,
                    }
                )

            # IMPROVED: Use the better trade extraction method
            trades_df = extract_trades(strat, self.data)
            metrics["trades_list"] = trades_df.to_dict("records")

            # Cross-check the numbers
            extracted_trade_count = len(trades_df)
            analyzer_trade_count = metrics["total_trades"]

            if self.verbose:
                print(f"TradeAnalyzer reports: {analyzer_trade_count} trades")
                print(f"extract_trades_improved found: {extracted_trade_count} trades")

                if extracted_trade_count != analyzer_trade_count:
                    print(
                        f"WARNING: Trade count mismatch! Using extract_trades_improved count."
                    )
                    # Update metrics to match extracted trades
                    if extracted_trade_count > 0:
                        won_trades = len(
                            [
                                t
                                for t in trades_df.to_dict("records")
                                if t.get("status") == "Won"
                            ]
                        )
                        lost_trades = extracted_trade_count - won_trades

                        metrics["total_trades"] = extracted_trade_count
                        metrics["winning_trades"] = won_trades
                        metrics["losing_trades"] = lost_trades
                        metrics["win_rate"] = (
                            won_trades / extracted_trade_count
                            if extracted_trade_count > 0
                            else 0.0
                        )

                        # Recalculate profit factor
                        total_profit = sum(
                            [
                                t.get("pnl", 0)
                                for t in trades_df.to_dict("records")
                                if t.get("pnl", 0) > 0
                            ]
                        )
                        total_loss = abs(
                            sum(
                                [
                                    t.get("pnl", 0)
                                    for t in trades_df.to_dict("records")
                                    if t.get("pnl", 0) < 0
                                ]
                            )
                        )
                        metrics["profit_factor"] = (
                            total_profit / total_loss if total_loss > 0 else 0.0
                        )

                        avg_pnl = (
                            sum([t.get("pnl", 0) for t in trades_df.to_dict("records")])
                            / extracted_trade_count
                        )
                        metrics["avg_trade_pnl"] = avg_pnl

        except Exception as e:
            if self.verbose:
                print(f"Error extracting metrics: {e}")
            metrics = {
                "total_return": 0.0,
                "avg_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "vwr": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_trade_pnl": 0.0,
                "trades_list": [],
            }

        return metrics

    def _create_objective_function(
        self, start_date: datetime, end_date: datetime
    ) -> Callable:
        """Create an objective function for Optuna optimization."""

        def objective(trial):
            params = {}
            for param_name, param_config in self.optimization_params.items():
                param_type = param_config["type"]

                if param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", 1),
                    )
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", None),
                        log=param_config.get("log", False),
                    )
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")

            try:
                cerebro = self._create_cerebro(start_date, end_date, params)
                results = cerebro.run()

                if results and len(results) > 0:
                    strat = results[0]
                    metrics = self._extract_performance_metrics(strat)

                    score = metrics.get(self.optimization_metric, 0.0)

                    if self.optimization_metric == "max_drawdown":
                        score = -score

                    if metrics["total_trades"] < 5:
                        score *= 0.1
                    elif metrics["total_trades"] == 0:
                        if self.verbose:
                            print(f"No trades executed for params: {params}")

                    return score
                else:
                    if self.verbose:
                        print("No results from cerebro.run()")
                    return float("-inf")

            except Exception as e:
                if self.verbose:
                    print(f"Error in objective function: {e}")
                return float("-inf")

        return objective

    def _optimize_parameters(
        self, start_date: datetime, end_date: datetime
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize strategy parameters using Optuna."""
        if not self.optimization_params:
            return {}, 0.0

        study_name = (
            f"walkforward_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=study_name,
        )

        objective_func = self._create_objective_function(start_date, end_date)

        if self.verbose:
            print(f"Optimizing with Optuna ({self.n_trials} trials)...")

        study.optimize(
            objective_func,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=self.verbose,
        )

        best_params = study.best_params
        best_score = study.best_value

        if self.verbose:
            print(f"Best score: {best_score:.4f}")
            print(f"Best params: {best_params}")

        return best_params, best_score

    def run_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete walk-forward analysis.

        Returns:
            Tuple of (results DataFrame, trades DataFrame)
        """
        data_start = self.data.index.min().to_pydatetime()
        data_end = self.data.index.max().to_pydatetime()

        if self.verbose:
            print(f"Starting walk-forward analysis from {data_start} to {data_end}")
            print(f"Training period: {self.training_period} days")
            print(f"Testing period: {self.testing_period} days")
            print(f"Optimization metric: {self.optimization_metric}")
            print(f"Optuna trials per period: {self.n_trials}")
            print(f"Optimization parameters: {self.optimization_params}")

        current_date = data_start
        walk_forward_num = 1

        while (
            current_date + timedelta(days=self.training_period + self.testing_period)
            <= data_end
        ):
            train_start = current_date
            train_end = current_date + timedelta(days=self.training_period)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.testing_period)

            if self.verbose:
                print(f"\nWalk-forward {walk_forward_num}:")
                print(f"Training: {train_start.date()} to {train_end.date()}")
                print(f"Testing: {test_start.date()} to {test_end.date()}")

            best_params, best_score = self._optimize_parameters(train_start, train_end)

            try:
                cerebro = self._create_cerebro(test_start, test_end, best_params)
                results = cerebro.run()

                if results and len(results) > 0:
                    strat = results[0]
                    metrics = self._extract_performance_metrics(strat)

                    result_dict = {
                        "walk_forward": walk_forward_num,
                        "train_start": train_start.date(),
                        "train_end": train_end.date(),
                        "test_start": test_start.date(),
                        "test_end": test_end.date(),
                        "best_params": str(best_params),
                        "optimization_score": best_score,
                        **metrics,
                    }

                    self.results.append(result_dict)

                    for trade in metrics["trades_list"]:
                        trade["walk_forward"] = walk_forward_num
                        trade["test_start"] = test_start.date()
                        trade["test_end"] = test_end.date()
                        self.all_trades.append(trade)

                    if self.verbose:
                        print(
                            f"Test results: Return={metrics['total_return']:.2%}, "
                            f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                            f"Drawdown={metrics['max_drawdown']:.2%}, "
                            f"Trades={metrics['total_trades']}"
                        )

            except Exception as e:
                if self.verbose:
                    print(f"Error in testing phase: {e}")

            current_date = test_start
            walk_forward_num += 1

        return pd.DataFrame(self.results), pd.DataFrame(self.all_trades)

    def get_parameter_analysis(self) -> pd.DataFrame:
        """
        Analyze the parameter stability across walk-forward periods.

        Returns:
            DataFrame with parameter values for each period
        """
        if not self.results:
            return pd.DataFrame()

        param_data = []
        for result in self.results:
            try:
                params_str = result["best_params"]
                if params_str and params_str != "{}":
                    params = ast.literal_eval(params_str)
                    param_row = {
                        "walk_forward": result["walk_forward"],
                        "test_start": result["test_start"],
                        "test_end": result["test_end"],
                        "total_return": result["total_return"],
                        "sharpe_ratio": result["sharpe_ratio"],
                        **params,
                    }
                    param_data.append(param_row)
            except Exception as e:
                if self.verbose:
                    print(
                        f"Error parsing parameters for period {result['walk_forward']}: {e}"
                    )
                continue

        return pd.DataFrame(param_data) if param_data else pd.DataFrame()

    def get_trade_details(self) -> pd.DataFrame:
        """
        Get detailed information about all trades.

        Returns:
            DataFrame with trade details
        """
        trades_df = pd.DataFrame(self.all_trades)
        if len(trades_df) > 0:
            trades_df = trades_df.drop_duplicates(subset=["ref"])
            # Rename columns to match log format
            trades_df = trades_df.rename(
                columns={
                    "pnl_net": "net_profit",
                    "entry_time": "entry_date",
                    "exit_time": "exit_date",
                    "size": "qty",
                }
            )
            # Add bars_held to match log's "Bars Held"
            trades_df["bars_held"] = (
                trades_df["exit_date"] - trades_df["entry_date"]
            ).dt.days
        else:
            # Create empty DataFrame with expected columns
            trades_df = pd.DataFrame(
                columns=[
                    "ref",
                    "entry_date",
                    "exit_date",
                    "entry_price",
                    "exit_price",
                    "qty",
                    "pnl",
                    "net_profit",
                    "commission",
                    "status",
                    "direction",
                    "walk_forward",
                    "test_start",
                    "test_end",
                    "bars_held",
                ]
            )
        return trades_df

    def get_summary_statistics(self) -> Dict[str, float]:
        """Get summary statistics across all walk-forward periods."""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        summary = {
            "total_periods": len(df),
            "avg_return": df["total_return"].mean(),
            "std_return": df["total_return"].std(),
            "sharpe_ratio": (
                df["total_return"].mean() / df["total_return"].std()
                if df["total_return"].std() > 0
                else 0.0
            ),
            "avg_period_sharpe": df["sharpe_ratio"].mean(),
            "avg_max_drawdown": df["max_drawdown"].mean(),
            "max_drawdown_overall": df["max_drawdown"].max(),
            "positive_periods": (df["total_return"] > 0).sum(),
            "positive_period_ratio": (df["total_return"] > 0).mean(),
            "avg_trades_per_period": df["total_trades"].mean(),
            "avg_win_rate": df["win_rate"].mean(),
            "avg_profit_factor": df["profit_factor"].mean(),
            "avg_vwr": df["vwr"].mean(),
            "consistency_score": (df["total_return"] > 0).mean()
            * df["sharpe_ratio"].mean(),
        }

        return summary

    def plot_results(self, save_path: str = None):
        """Plot walk-forward analysis results."""
        if not self.results:
            print("No results to plot")
            return

        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            df = pd.DataFrame(self.results)
            trades_df = pd.DataFrame(self.all_trades)

            fig, axes = plt.subplots(3, 2, figsize=(15, 15))
            fig.suptitle("Walk-Forward Analysis Results", fontsize=16)

            ax1 = axes[0, 0]
            test_dates = pd.to_datetime(df["test_start"])
            ax1.plot(
                test_dates, df["total_return"] * 100, "o-", linewidth=2, markersize=6
            )
            ax1.set_title("Returns by Period")
            ax1.set_ylabel("Return (%)")
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5)

            ax2 = axes[0, 1]
            ax2.plot(
                test_dates,
                df["sharpe_ratio"],
                "s-",
                linewidth=2,
                markersize=6,
                color="green",
            )
            ax2.set_title("Sharpe Ratio by Period")
            ax2.set_ylabel("Sharpe Ratio")
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)

            ax3 = axes[1, 0]
            ax3.plot(
                test_dates,
                df["max_drawdown"] * 100,
                "^-",
                linewidth=2,
                markersize=6,
                color="red",
            )
            ax3.set_title("Max Drawdown by Period")
            ax3.set_ylabel("Max Drawdown (%)")
            ax3.grid(True, alpha=0.3)

            ax4 = axes[1, 1]
            ax4.bar(test_dates, df["total_trades"], alpha=0.7, color="orange")
            ax4.set_title("Total Trades by Period")
            ax4.set_ylabel("Number of Trades")
            ax4.grid(True, alpha=0.3)

            ax5 = axes[2, 0]
            if not trades_df.empty:
                ax5.hist(trades_df["pnl"], bins=30, alpha=0.7, color="purple")
                ax5.set_title("Trade PNL Distribution")
                ax5.set_xlabel("PNL")
                ax5.set_ylabel("Frequency")
                ax5.grid(True, alpha=0.3)

            ax6 = axes[2, 1]
            if not trades_df.empty:
                ax6.hist(trades_df["bars_held"], bins=30, alpha=0.7, color="blue")
                ax6.set_title("Trade Duration Distribution")
                ax6.set_xlabel("Duration (bars)")
                ax6.set_ylabel("Frequency")
                ax6.grid(True, alpha=0.3)

            for ax in [ax1, ax2, ax3, ax4]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Plot saved to {save_path}")

            plt.show()

        except ImportError:
            print("matplotlib not available for plotting")
        except Exception as e:
            print(f"Error creating plots: {e}")

    def export_results(self, filepath: str, trades_filepath: str = None):
        """Export results and trade details to CSV files."""
        if not self.results:
            print("No results to export")
            return

        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")

        if trades_filepath and self.all_trades:
            trades_df = pd.DataFrame(self.all_trades)
            trades_df = trades_df.drop_duplicates(subset=["ref"])
            trades_df.to_csv(trades_filepath, index=False)
            print(f"Trade details exported to {trades_filepath}")

    def get_detailed_summary(self) -> str:
        """Get a detailed text summary of the analysis."""
        if not self.results:
            return "No results available"

        df = pd.DataFrame(self.results)
        trades_df = pd.DataFrame(self.all_trades)
        summary = self.get_summary_statistics()

        report = f"""
Walk-Forward Analysis Summary Report
===================================

Analysis Period: {df['test_start'].iloc[0]} to {df['test_end'].iloc[-1]}
Total Walk-Forward Periods: {len(df)}

Performance Metrics:
- Average Return per Period: {summary['avg_return']:.2%}
- Standard Deviation of Returns: {summary['std_return']:.2%}
- Overall Sharpe Ratio: {summary['sharpe_ratio']:.3f}
- Average Period Sharpe Ratio: {summary['avg_period_sharpe']:.3f}
- Maximum Drawdown: {summary['max_drawdown_overall']:.2%}
- Average Drawdown: {summary['avg_max_drawdown']:.2%}

Trading Activity:
- Average Trades per Period: {summary['avg_trades_per_period']:.1f}
- Average Win Rate: {summary['avg_win_rate']:.1%}
- Average Profit Factor: {summary['avg_profit_factor']:.2f}
- Total Trades: {len(trades_df) if not trades_df.empty else 0}

Consistency Metrics:
- Positive Periods: {summary['positive_periods']:.0f} out of {summary['total_periods']:.0f}
- Positive Period Ratio: {summary['positive_period_ratio']:.1%}
- Consistency Score: {summary['consistency_score']:.3f}

Period-by-Period Results:
"""

        for _, row in df.iterrows():
            period_trades = (
                trades_df[trades_df["walk_forward"] == row["walk_forward"]]
                if not trades_df.empty
                else pd.DataFrame()
            )
            report += f"""
Period {row['walk_forward']}: {row['test_start']} to {row['test_end']}
Return: {row['total_return']:.2%}, Sharpe: {row['sharpe_ratio']:.2f}
Drawdown: {row['max_drawdown']:.2%}, Trades: {row['total_trades']}
Best Parameters: {row['best_params']}
Number of Trades: {len(period_trades)}
"""

        return report


def run_walkforward_example():
    """Example of how to use the walk-forward analysis with Optuna."""
    df = get_data_sync(
        ticker="SBIN.NS", start_date="2020-01-01", end_date="2025-12-31", interval="1d"
    )

    if df.empty:
        print("Error: No data retrieved from get_data_sync")
        return None, None, None, None

    required_columns = ["Close", "High", "Low", "Open", "Volume"]
    if not all(col in df.columns for col in required_columns):
        print(
            f"Error: DataFrame missing required columns. Found: {list(df.columns)}, Required: {required_columns}"
        )
        return None, None, None, None

    if not isinstance(df.index, pd.DatetimeIndex):
        print("Error: DataFrame index is not a DatetimeIndex")
        return None, None, None, None

    # Ensure timezone consistency
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    print(f"Data retrieved: {len(df)} bars from {df.index.min()} to {df.index.max()}")
    print(
        f"Price range: Close min={df['Close'].min():.2f}, max={df['Close'].max():.2f}"
    )
    print(f"Sample data:\n{df.head(3)}\n{df.tail(3)}")
    print(f"Data statistics:\n{df.describe()}")

    wfa = WalkForwardAnalysis(
        strategy_class=SMABollinger,
        data=df,
        training_period=252,
        testing_period=63,
        optimization_metric="sharpe_ratio",
        initial_cash=100000,
        commission=0.09,  # Flat commission per log
        n_trials=50,
        verbose=True,
    )

    results_df, trades_df = wfa.run_analysis()
    summary = wfa.get_summary_statistics()
    param_analysis = wfa.get_parameter_analysis()
    trades_df = wfa.get_trade_details()
    print("\nWalk-Forward Analysis Results:")
    if len(results_df) > 0:
        print(
            results_df[
                [
                    "walk_forward",
                    "test_start",
                    "test_end",
                    "total_return",
                    "sharpe_ratio",
                    "max_drawdown",
                    "total_trades",
                ]
            ].to_string(index=False)
        )
    else:
        print(
            "No walk-forward results generated. Check data and strategy configuration."
        )

    print("\nSummary Statistics:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    print("\nParameter Analysis:")
    if len(param_analysis) > 0:
        print(param_analysis.to_string(index=False))
    else:
        print("No parameter analysis available.")

    print("\nTrade Details:")
    if len(trades_df) > 0:
        print(
            trades_df[
                [
                    "walk_forward",
                    "ref",
                    "entry_date",
                    "exit_date",
                    "entry_price",
                    "exit_price",
                    "qty",
                    "net_profit",
                    "commission",
                    "bars_held",
                ]
            ].to_string(index=False)
        )
    else:
        print("No trades executed. Possible issues:")
        print(
            "- Data range mismatch with trade log (check sample data and price range above)"
        )
        print(
            "- Strategy parameters too restrictive (check optimization_params in Parameter Analysis)"
        )
        print("- Trade extraction failure (check 'Trade captured' logs)")
        print(
            "- Orders not completing (check 'Initializing strategy' and 'Trade captured' logs)"
        )
        print("- Data issues (check data statistics and continuity)")

    if len(results_df) > 0:
        print("\nAdditional Insights:")
        print(
            f"Best performing period: {results_df.loc[results_df['total_return'].idxmax()]['walk_forward']} "
            f"with return of {results_df['total_return'].max():.2%}"
        )
        print(
            f"Worst performing period: {results_df.loc[results_df['total_return'].idxmin()]['walk_forward']} "
            f"with return of {results_df['total_return'].min():.2%}"
        )

        if len(param_analysis) > 0:
            print("\nParameter Stability:")
            for param in param_analysis.columns:
                if param not in [
                    "walk_forward",
                    "test_start",
                    "test_end",
                    "total_return",
                    "sharpe_ratio",
                ]:
                    print(
                        f"{param}: mean={param_analysis[param].mean():.1f}, "
                        f"std={param_analysis[param].std():.1f}"
                    )

    return results_df, summary, param_analysis, trades_df


if __name__ == "__main__":
    results, summary, param_analysis, trades = run_walkforward_example()
