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
import traceback

# Assuming these imports work with your existing code
from comprehensive_backtesting.data import get_data_sync
from comprehensive_backtesting.ema_rsi import EMARSI
from comprehensive_backtesting.sma_bollinger_band import SMABollinger

warnings.filterwarnings("ignore")


def safe_extract_indicator_values(
    strategy: bt.Strategy, dt: datetime
) -> Dict[str, float]:
    """Safely extract indicator values from strategy at a given datetime."""
    indicators = {}
    try:
        # Ensure datetime is timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        else:
            dt = dt.astimezone(pytz.UTC)

        # Get data length to avoid index errors
        data_len = len(strategy.data)
        if data_len == 0:
            return indicators

        # Convert datetime to backtrader's internal format
        target_num = bt.date2num(dt)

        # Find the closest data index
        idx = min(
            range(data_len), key=lambda i: abs(strategy.data.datetime[i] - target_num)
        )

        # Safely extract indicators with bounds checking
        for attr_name in ["fast_sma", "slow_sma", "bb_upper", "bb_lower", "ema", "rsi"]:
            if hasattr(strategy, attr_name):
                indicator = getattr(strategy, attr_name)
                if 0 <= idx < len(indicator):
                    value = indicator[idx]
                    indicators[attr_name] = (
                        float(value) if not np.isnan(value) else None
                    )

        if (
            hasattr(strategy, "params")
            and hasattr(strategy.params, "verbose")
            and strategy.params.verbose
        ):
            print(f"Extracted indicators at {dt}: {indicators}")

    except Exception as e:
        if (
            hasattr(strategy, "params")
            and hasattr(strategy.params, "verbose")
            and strategy.params.verbose
        ):
            print(f"Error extracting indicators at {dt}: {e}")

    return indicators


def safe_extract_trades(strategy_result: bt.Strategy) -> pd.DataFrame:
    """
    Safely extract trades with comprehensive error handling and correct price logic.
    """
    trades = []

    try:
        # Method 1: Access BackTrader's internal trade list
        if hasattr(strategy_result, "_trades"):
            for trade_obj in strategy_result._trades:
                if hasattr(trade_obj, "isclosed") and trade_obj.isclosed:
                    try:
                        entry_dt = bt.num2date(trade_obj.dtopen)
                        if entry_dt.tzinfo is None:
                            entry_dt = entry_dt.replace(tzinfo=pytz.UTC)
                        else:
                            entry_dt = entry_dt.astimezone(pytz.UTC)

                        exit_dt = bt.num2date(trade_obj.dtclose)
                        if exit_dt.tzinfo is None:
                            exit_dt = exit_dt.replace(tzinfo=pytz.UTC)
                        else:
                            exit_dt = exit_dt.astimezone(pytz.UTC)

                        # Use pclose for exit price instead of calculation
                        exit_price = trade_obj.pclose
                        entry_price = trade_obj.price
                        size = trade_obj.size
                        pnl = trade_obj.pnl
                        pnl_net = trade_obj.pnlcomm
                        commission = trade_obj.commission
                        status = "Won" if pnl_net > 0 else "Lost"
                        direction = "Long" if size > 0 else "Short"
                        bars_held = trade_obj.barlen

                        trade_info = {
                            "ref": trade_obj.ref,
                            "entry_time": entry_dt,
                            "exit_time": exit_dt,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "size": abs(size),
                            "pnl": pnl,
                            "pnl_net": pnl_net,
                            "commission": commission,
                            "status": status,
                            "direction": direction,
                            "bars_held": bars_held,
                        }
                        trades.append(trade_info)

                    except Exception as e:
                        if (
                            hasattr(strategy_result, "params")
                            and hasattr(strategy_result.params, "verbose")
                            and strategy_result.params.verbose
                        ):
                            print(f"Error processing trade: {e}")

    except Exception as e:
        if (
            hasattr(strategy_result, "params")
            and hasattr(strategy_result.params, "verbose")
            and strategy_result.params.verbose
        ):
            print(f"Error in trade extraction: {e}")

    # Convert to DataFrame
    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        trades_df = trades_df.drop_duplicates(subset=["ref"])
        if (
            hasattr(strategy_result, "params")
            and hasattr(strategy_result.params, "verbose")
            and strategy_result.params.verbose
        ):
            print(f"Extracted {len(trades_df)} trades")
    else:
        if (
            hasattr(strategy_result, "params")
            and hasattr(strategy_result.params, "verbose")
            and strategy_result.params.verbose
        ):
            print("No trades extracted")

    return trades_df


class WalkForwardAnalysis:
    """
    Walk-forward analysis with better error handling and data validation.
    """

    def __init__(
        self,
        strategy_class: bt.Strategy,
        data: pd.DataFrame,
        training_period: int = 252,
        testing_period: int = 21,
        step_size: int = 21,
        window_type: str = "rolling",
        min_training_samples: int = 100,
        optimization_params: Dict[str, Dict] = None,
        optimization_metric: str = "sharpe_ratio",
        initial_cash: float = 100000,
        commission: float = 0.001,
        n_trials: int = 20,
        timeout: Optional[int] = None,
        verbose: bool = True,
        purge_gap: int = 0,
        embargo_period: int = 0,
        min_data_points: int = 50,
        buffer_days: int = 5,  # New parameter for indicator warm-up
    ):
        """
        Initialize improved walk-forward analysis with better validation.
        """
        self.strategy_class = strategy_class
        self.data = data.copy()
        self.training_period = training_period
        self.testing_period = testing_period
        self.step_size = step_size
        self.window_type = window_type.lower()
        self.min_training_samples = min_training_samples
        self.optimization_params = optimization_params or {}
        self.optimization_metric = optimization_metric
        self.initial_cash = initial_cash
        self.commission = commission
        self.n_trials = n_trials
        self.timeout = timeout
        self.verbose = verbose
        self.purge_gap = purge_gap
        self.embargo_period = embargo_period
        self.min_data_points = min_data_points
        self.buffer_days = buffer_days  # For indicator warm-up

        # Validation
        if self.window_type not in ["rolling", "anchored"]:
            raise ValueError("window_type must be 'rolling' or 'anchored'")

        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

        if self.testing_period <= 0:
            raise ValueError("testing_period must be positive")

        if len(self.data) < self.min_training_samples + self.testing_period:
            raise ValueError(
                f"Insufficient data: need at least {self.min_training_samples + self.testing_period} points"
            )

        # Prepare data
        self._prepare_data()

        self.results = []
        self.all_trades = []

        # Setup optuna
        self.sampler = TPESampler(seed=42)
        self.pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=5)

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _prepare_data(self):
        """Prepare and validate data for analysis."""
        # Ensure data is sorted by index
        self.data = self.data.sort_index()

        # Remove any duplicate indices
        self.data = self.data[~self.data.index.duplicated(keep="first")]

        # Fill missing values with forward fill then backward fill
        self.data = self.data.fillna(method="ffill").fillna(method="bfill")

        # Ensure we have the required OHLCV columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [
            col for col in required_columns if col not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Remove rows with any NaN values in OHLCV columns
        self.data = self.data.dropna(subset=required_columns)

        if len(self.data) < self.min_training_samples + self.testing_period:
            raise ValueError(
                f"After cleaning, insufficient data: {len(self.data)} points available"
            )

        if self.verbose:
            print(
                f"Data prepared: {len(self.data)} valid data points from {self.data.index.min()} to {self.data.index.max()}"
            )

    def _get_walk_forward_splits(
        self,
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate walk-forward splits using trading days instead of calendar days.
        """
        splits = []
        dates = self.data.index.to_pydatetime()
        total_days = len(dates)
        start_idx = 0

        # Safety counter to prevent infinite loops
        iteration = 0

        while start_idx < total_days - self.min_training_samples - self.testing_period:
            iteration += 1
            if iteration > 1000:  # Safety break
                break

            # Training window
            train_start_idx = start_idx
            train_end_idx = start_idx + self.training_period - 1

            # Ensure training end doesn't exceed data bounds
            if train_end_idx >= total_days:
                break

            # Testing window
            test_start_idx = train_end_idx + self.purge_gap + 1
            if test_start_idx >= total_days:
                break

            test_end_idx = test_start_idx + self.testing_period - 1
            if test_end_idx >= total_days:
                test_end_idx = total_days - 1

            # Get actual dates
            train_start = dates[train_start_idx]
            train_end = dates[train_end_idx]
            test_start = dates[test_start_idx]
            test_end = dates[test_end_idx]

            # Validate training period has sufficient bars
            training_bars = train_end_idx - train_start_idx + 1
            if training_bars < self.min_training_samples:
                start_idx += self.step_size
                continue

            # Validate testing period has sufficient bars
            testing_bars = test_end_idx - test_start_idx + 1
            if testing_bars < self.min_data_points:
                start_idx += self.step_size
                continue

            splits.append((train_start, train_end, test_start, test_end))

            # Advance by step size
            start_idx += self.step_size

        return splits

    def _validate_split_data(self, start_date: datetime, end_date: datetime) -> bool:
        """Validate that we have sufficient and quality data for the given period."""
        mask = (self.data.index >= start_date) & (self.data.index <= end_date)
        filtered_data = self.data.loc[mask]

        if len(filtered_data) < self.min_data_points:
            return False

        # Check for data quality issues
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_columns:
            if col in filtered_data.columns:
                if (
                    filtered_data[col].isna().sum() > len(filtered_data) * 0.1
                ):  # >10% missing
                    return False
                if (filtered_data[col] <= 0).sum() > len(
                    filtered_data
                ) * 0.1:  # >10% zero/negative
                    return False

        return True

    def run_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the standardized walk-forward analysis.

        Returns:
            Tuple of (results DataFrame, trades DataFrame)
        """
        splits = self._get_walk_forward_splits()
        if not splits:
            raise ValueError(
                "No valid walk-forward splits generated. Check data and parameters."
            )

        if self.verbose:
            print(f"Generated {len(splits)} walk-forward splits")
            print(f"Window type: {self.window_type}")
            print(f"Training period: {self.training_period} days")
            print(f"Testing period: {self.testing_period} days")
            print(f"Step size: {self.step_size} days")
            print(f"Purge gap: {self.purge_gap} days")
            print(f"Embargo period: {self.embargo_period} days")
            print(f"Buffer days: {self.buffer_days} days")

        for i, (train_start, train_end, test_start, test_end) in enumerate(splits, 1):

            if self.verbose:
                print(f"\nWalk-forward {i}/{len(splits)}:")
                print(
                    f"Training: {train_start.date()} to {train_end.date()} ({len(self.data[(self.data.index >= train_start) & (self.data.index <= train_end)])} bars)"
                )
                print(
                    f"Testing: {test_start.date()} to {test_end.date()} ({len(self.data[(self.data.index >= test_start) & (self.data.index <= test_end)])} bars)"
                )

            # Validate data availability
            if not self._validate_split_data(train_start, train_end):
                if self.verbose:
                    print(f"Insufficient training data, skipping period {i}")
                continue

            if not self._validate_split_data(test_start, test_end):
                if self.verbose:
                    print(f"Insufficient testing data, skipping period {i}")
                continue

            # Optimize parameters on training data
            try:
                best_params, best_score = self._optimize_parameters(
                    train_start, train_end
                )
            except Exception as e:
                if self.verbose:
                    print(f"Optimization failed for period {i}: {e}")
                continue

            # Test on out-of-sample data with buffer for indicator warm-up
            try:
                # Create cerebro with buffer around test period
                cerebro = self._create_cerebro(
                    test_start - timedelta(days=self.buffer_days),
                    test_end + timedelta(days=self.buffer_days),
                    best_params,
                )

                # Run backtest but only analyze the test period
                results = cerebro.run()

                if results and len(results) > 0:
                    strat = results[0]

                    # Extract metrics only for the test period
                    metrics = self._extract_performance_metrics(
                        strat, test_start, test_end
                    )

                    result_dict = {
                        "walk_forward": i,
                        "train_start": train_start.date(),
                        "train_end": train_end.date(),
                        "test_start": test_start.date(),
                        "test_end": test_end.date(),
                        "train_days": (train_end - train_start).days,
                        "test_days": (test_end - test_start).days,
                        "best_params": str(best_params),
                        "optimization_score": best_score,
                        **metrics,
                    }

                    self.results.append(result_dict)

                    # Add trade details
                    for trade in metrics["trades_list"]:
                        trade["walk_forward"] = i
                        trade["test_start"] = test_start.date()
                        trade["test_end"] = test_end.date()
                        self.all_trades.append(trade)

                    if self.verbose:
                        print(
                            f"Results: Return={metrics['total_return']:.2%}, "
                            f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                            f"Trades={metrics['total_trades']}"
                        )

            except Exception as e:
                if self.verbose:
                    print(f"Testing failed for period {i}: {e}")
                    traceback.print_exc()
                continue

        return pd.DataFrame(self.results), pd.DataFrame(self.all_trades)

    def get_statistical_significance(self) -> Dict[str, float]:
        """
        Calculate statistical significance of results.

        Returns:
            Dictionary with significance tests
        """
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)
        returns = df["total_return"]

        # Basic statistics
        stats = {
            "mean_return": returns.mean(),
            "std_return": returns.std(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "sharpe_ratio": returns.mean() / returns.std() if returns.std() > 0 else 0,
        }

        # T-test for mean return significantly different from zero
        if len(returns) > 1:
            t_stat = returns.mean() / (returns.std() / np.sqrt(len(returns)))
            stats["t_statistic"] = t_stat
            # Critical t-value for 95% confidence (approximation)
            critical_t = 1.96  # For large samples
            stats["is_significant_95pct"] = abs(t_stat) > critical_t

        return stats

    def get_robustness_metrics(self) -> Dict[str, float]:
        """
        Calculate robustness metrics for the walk-forward analysis.

        Returns:
            Dictionary with robustness metrics
        """
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        # Parameter stability (if optimization was used)
        param_analysis = self.get_parameter_analysis()
        param_stability = {}

        if not param_analysis.empty:
            numeric_params = param_analysis.select_dtypes(include=[np.number]).columns
            numeric_params = [
                col
                for col in numeric_params
                if col not in ["walk_forward", "total_return", "sharpe_ratio"]
            ]

            for param in numeric_params:
                if param in param_analysis.columns:
                    param_stability[f"{param}_cv"] = (
                        param_analysis[param].std() / param_analysis[param].mean()
                        if param_analysis[param].mean() != 0
                        else np.inf
                    )

        robustness = {
            "return_consistency": (
                1 - (df["total_return"].std() / abs(df["total_return"].mean()))
                if df["total_return"].mean() != 0
                else 0
            ),
            "positive_periods_ratio": (df["total_return"] > 0).mean(),
            "max_consecutive_losses": self._calculate_max_consecutive_losses(
                df["total_return"]
            ),
            "performance_stability": df["sharpe_ratio"].std(),
            **param_stability,
        }

        return robustness

    def _calculate_max_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive losing periods."""
        losing_periods = (returns < 0).astype(int)
        max_consecutive = 0
        current_consecutive = 0

        for is_loss in losing_periods:
            if is_loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _create_cerebro(
        self, start_date: datetime, end_date: datetime, params: Dict = None
    ) -> bt.Cerebro:
        """Create a cerebro instance with improved error handling."""
        try:
            cerebro = bt.Cerebro(runonce=False, exactbars=True)

            # Filter data for the given date range
            mask = (self.data.index >= start_date) & (self.data.index <= end_date)
            filtered_df = self.data.loc[mask].copy()

            if len(filtered_df) < self.min_data_points:
                raise ValueError(
                    f"Insufficient data: only {len(filtered_df)} points available"
                )

            # Ensure data integrity
            filtered_df = filtered_df.dropna()
            if len(filtered_df) < self.min_data_points:
                raise ValueError(
                    f"Insufficient data after removing NaNs: only {len(filtered_df)} points available"
                )

            if self.verbose:
                print(
                    f"Creating cerebro with {len(filtered_df)} data points from {filtered_df.index.min()} to {filtered_df.index.max()}"
                )

            # Create data feed
            data_feed = bt.feeds.PandasData(
                dataname=filtered_df,
                datetime=None,  # Use index as datetime
                open=0 if "Open" in filtered_df.columns else -1,
                high=1 if "High" in filtered_df.columns else -1,
                low=2 if "Low" in filtered_df.columns else -1,
                close=3 if "Close" in filtered_df.columns else -1,
                volume=4 if "Volume" in filtered_df.columns else -1,
                openinterest=None,
            )
            cerebro.adddata(data_feed)

            # Add strategy with parameters
            strategy_params = {"verbose": self.verbose}
            if params:
                strategy_params.update(params)

            cerebro.addstrategy(self.strategy_class, **strategy_params)

            # Set broker parameters
            cerebro.broker.setcash(self.initial_cash)
            cerebro.broker.setcommission(commission=self.commission)

            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
            cerebro.addanalyzer(
                bt.analyzers.SharpeRatio,
                _name="sharpe",
                riskfreerate=0.0,
                annualize=True,
            )
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            cerebro.addanalyzer(bt.analyzers.VWR, _name="vwr")

            return cerebro

        except Exception as e:
            if self.verbose:
                print(f"Error creating cerebro: {e}")
            raise

    def _extract_performance_metrics(
        self, strat, test_start: datetime, test_end: datetime
    ) -> Dict[str, float]:
        """Extract performance metrics with improved error handling and trade filtering."""
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

        try:
            if not hasattr(strat, "analyzers"):
                return metrics

            analyzers = strat.analyzers

            # Returns analysis
            if hasattr(analyzers, "returns") and analyzers.returns:
                returns = analyzers.returns.get_analysis()
                metrics["total_return"] = returns.get("rtot", 0.0) or 0.0
                metrics["avg_return"] = returns.get("ravg", 0.0) or 0.0

            # Sharpe ratio
            if hasattr(analyzers, "sharpe") and analyzers.sharpe:
                sharpe = analyzers.sharpe.get_analysis()
                sharpe_value = sharpe.get("sharperatio", 0.0)
                metrics["sharpe_ratio"] = (
                    sharpe_value if sharpe_value and not np.isnan(sharpe_value) else 0.0
                )

            # Drawdown
            if hasattr(analyzers, "drawdown") and analyzers.drawdown:
                drawdown = analyzers.drawdown.get_analysis()
                dd_value = drawdown.get("max", {}).get("drawdown", 0.0)
                metrics["max_drawdown"] = (
                    dd_value if dd_value and not np.isnan(dd_value) else 0.0
                )

            # VWR
            if hasattr(analyzers, "vwr") and analyzers.vwr:
                vwr = analyzers.vwr.get_analysis()
                vwr_value = vwr.get("vwr", 0.0)
                metrics["vwr"] = (
                    vwr_value if vwr_value and not np.isnan(vwr_value) else 0.0
                )

            # Extract individual trades and filter by test period
            trades_df = safe_extract_trades(strat)
            if not trades_df.empty:
                # Filter trades to only include those entered during the test period
                trades_df = trades_df[
                    (trades_df["entry_time"] >= test_start)
                    & (trades_df["entry_time"] <= test_end)
                ]
                metrics["trades_list"] = trades_df.to_dict("records")
                metrics["total_trades"] = len(trades_df)

                # Recalculate metrics based on actual trades
                if metrics["total_trades"] > 0:
                    winning_trades = trades_df[trades_df["pnl_net"] > 0]
                    losing_trades = trades_df[trades_df["pnl_net"] < 0]

                    metrics["winning_trades"] = len(winning_trades)
                    metrics["losing_trades"] = len(losing_trades)
                    metrics["win_rate"] = (
                        metrics["winning_trades"] / metrics["total_trades"]
                    )

                    gross_profit = winning_trades["pnl_net"].sum()
                    gross_loss = abs(losing_trades["pnl_net"].sum())

                    if gross_loss > 0:
                        metrics["profit_factor"] = gross_profit / gross_loss
                    else:
                        metrics["profit_factor"] = 0.0

                    metrics["avg_trade_pnl"] = trades_df["pnl_net"].mean()

            # If no trades extracted, try to get summary from analyzer
            if (
                metrics["total_trades"] == 0
                and hasattr(analyzers, "trades")
                and analyzers.trades
            ):
                trades = analyzers.trades.get_analysis()

                # Total trades
                total_dict = trades.get("total", {})
                metrics["total_trades"] = (
                    total_dict.get("total", 0) if isinstance(total_dict, dict) else 0
                )

                # Winning trades
                won_dict = trades.get("won", {})
                metrics["winning_trades"] = (
                    won_dict.get("total", 0) if isinstance(won_dict, dict) else 0
                )

                # Losing trades
                lost_dict = trades.get("lost", {})
                metrics["losing_trades"] = (
                    lost_dict.get("total", 0) if isinstance(lost_dict, dict) else 0
                )

                # Win rate
                if metrics["total_trades"] > 0:
                    metrics["win_rate"] = (
                        metrics["winning_trades"] / metrics["total_trades"]
                    )

                # Profit factor
                gross_profit = 0.0
                gross_loss = 0.0

                if isinstance(won_dict, dict) and "pnl" in won_dict:
                    won_pnl = won_dict["pnl"]
                    if isinstance(won_pnl, dict):
                        gross_profit = won_pnl.get("total", 0.0) or 0.0

                if isinstance(lost_dict, dict) and "pnl" in lost_dict:
                    lost_pnl = lost_dict["pnl"]
                    if isinstance(lost_pnl, dict):
                        gross_loss = abs(lost_pnl.get("total", 0.0) or 0.0)

                metrics["profit_factor"] = (
                    gross_profit / gross_loss if gross_loss > 0 else 0.0
                )

                # Average trade PNL
                pnl_dict = trades.get("pnl", {})
                if isinstance(pnl_dict, dict):
                    avg_pnl = pnl_dict.get("average", 0.0)
                    metrics["avg_trade_pnl"] = (
                        avg_pnl if avg_pnl and not np.isnan(avg_pnl) else 0.0
                    )

        except Exception as e:
            if self.verbose:
                print(f"Error extracting performance metrics: {e}")
            return metrics

        return metrics

    def _create_objective_function(
        self, start_date: datetime, end_date: datetime
    ) -> Callable:
        """Create an objective function for Optuna optimization with better error handling."""

        def objective(trial):
            params = {}

            try:
                # Build parameters from optimization config
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
                        if self.verbose:
                            print(f"Warning: Unknown parameter type: {param_type}")
                        continue

                # Validate parameter combinations
                if "fast_sma" in params and "slow_sma" in params:
                    if params["fast_sma"] >= params["slow_sma"]:
                        # Skip invalid parameter combination
                        return float("-inf")

                # Run backtest
                cerebro = self._create_cerebro(start_date, end_date, params)
                results = cerebro.run()

                if results and len(results) > 0:
                    strat = results[0]
                    metrics = self._extract_performance_metrics(
                        strat, start_date, end_date
                    )

                    # Get the optimization metric
                    score = metrics.get(self.optimization_metric, 0.0)

                    # Handle special cases
                    if self.optimization_metric == "max_drawdown":
                        score = -score  # Minimize drawdown

                    # Penalize strategies with very few trades
                    min_expected_trades = max(
                        3, (end_date - start_date).days // 10
                    )  # Dynamic threshold
                    if metrics["total_trades"] < min_expected_trades:
                        penalty_factor = max(
                            0.1, metrics["total_trades"] / min_expected_trades
                        )
                        score *= penalty_factor

                    # Penalize strategies with very poor win rates
                    if metrics["win_rate"] < 0.3 and metrics["total_trades"] > 5:
                        score *= 0.8

                    return score if not np.isnan(score) else -1.0
                else:
                    return -1.0

            except Exception as e:
                if self.verbose:
                    print(f"Error in objective function: {e}")
                return -1.0

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


def run_standardized_walkforward_example():
    """Example showing standardized walk-forward analysis."""

    # Get data
    df = get_data_sync(
        ticker="SBIN.NS", start_date="2020-01-01", end_date="2024-12-31", interval="1d"
    )

    # Initialize with standardized parameters
    wfa = WalkForwardAnalysis(
        strategy_class=SMABollinger,
        data=df,
        training_period=252,  # 1 year training
        testing_period=21,  # 1 month testing
        step_size=21,  # Advance by 1 month each time
        window_type="rolling",  # Rolling window
        min_training_samples=200,  # Minimum training samples
        min_data_points=10,  # Set to 10% of testing period
        purge_gap=1,  # 1 day gap to prevent look-ahead
        embargo_period=0,  # No embargo
        optimization_metric="sharpe_ratio",
        initial_cash=100000,
        commission=0.001,  # 0.1% commission
        n_trials=30,
        verbose=True,
        buffer_days=10,  # Extra buffer for indicators
    )

    # Run analysis
    results_df, trades_df = wfa.run_analysis()

    # Get additional metrics
    significance = wfa.get_statistical_significance()
    robustness = wfa.get_robustness_metrics()

    print("\nStatistical Significance:")
    for key, value in significance.items():
        print(f"{key}: {value}")

    print("\nRobustness Metrics:")
    for key, value in robustness.items():
        print(f"{key}: {value}")

    # Export results
    wfa.export_results("wfa_results.csv", "wfa_trades.csv")

    return results_df, trades_df, significance, robustness


if __name__ == "__main__":
    results, trades, significance, robustness = run_standardized_walkforward_example()
