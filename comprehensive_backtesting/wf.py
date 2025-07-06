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
import logging
from concurrent.futures import ThreadPoolExecutor

# Assuming these imports work with your existing code
from comprehensive_backtesting.data import get_data_sync
from comprehensive_backtesting.ema_rsi import EMARSI
from comprehensive_backtesting.sma_bollinger_band import SMABollinger

# Configure logging with DEBUG level for detailed diagnostics
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


def safe_extract_indicator_values(
    strategy: bt.Strategy, dt: datetime
) -> Dict[str, float]:
    """Safely extract indicator values from strategy at a given datetime."""
    indicators = {}
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        else:
            dt = dt.astimezone(pytz.UTC)

        data_len = len(strategy.data)
        if data_len == 0:
            logger.warning("No data available for indicator extraction")
            return indicators

        data_dt = [
            bt.num2date(strategy.data.datetime[i]).astimezone(pytz.UTC)
            for i in range(data_len)
            if strategy.data.datetime[i] is not None
        ]

        if not data_dt:
            logger.warning("No valid datetime data found")
            return indicators

        closest_idx = min(range(len(data_dt)), key=lambda i: abs(data_dt[i] - dt))

        for attr_name in ["fast_sma", "slow_sma", "bb_upper", "bb_lower", "ema", "rsi"]:
            if hasattr(strategy, attr_name):
                indicator = getattr(strategy, attr_name)
                if hasattr(indicator, "lines") and len(indicator.lines) > 0:
                    try:
                        value = indicator.lines[0][closest_idx]
                        indicators[attr_name] = (
                            float(value) if not np.isnan(value) else None
                        )
                    except (IndexError, ValueError) as e:
                        logger.debug(
                            f"Error extracting {attr_name} at index {closest_idx}: {e}"
                        )
                        indicators[attr_name] = None

        if getattr(strategy.params, "verbose", False):
            logger.debug(f"Extracted indicators at {dt}: {indicators}")

    except Exception as e:
        logger.error(f"Error extracting indicators at {dt}: {e}")

    return indicators


def safe_extract_trades(
    strategy_result: bt.Strategy, data: pd.DataFrame = None
) -> pd.DataFrame:
    """Extract trades with improved efficiency and error handling."""
    trades = []

    try:
        if hasattr(strategy_result, "_trades"):
            for trade_obj in strategy_result._trades:
                if hasattr(trade_obj, "isclosed") and trade_obj.isclosed:
                    try:
                        entry_dt = (
                            bt.num2date(trade_obj.dtopen).astimezone(pytz.UTC)
                            if trade_obj.dtopen
                            else None
                        )
                        exit_dt = (
                            bt.num2date(trade_obj.dtclose).astimezone(pytz.UTC)
                            if trade_obj.dtclose
                            else None
                        )
                        size = getattr(trade_obj, "size", 0)
                        pnl = getattr(trade_obj, "pnl", 0)

                        trades.append(
                            {
                                "ref": getattr(trade_obj, "ref", 0),
                                "entry_time": entry_dt,
                                "exit_time": exit_dt,
                                "entry_price": getattr(trade_obj, "price", 0),
                                "exit_price": getattr(trade_obj, "price", 0)
                                + (pnl / size if size != 0 else 0),
                                "size": abs(size),
                                "pnl": pnl,
                                "pnl_net": getattr(trade_obj, "pnlcomm", pnl),
                                "commission": getattr(trade_obj, "commission", 0),
                                "status": "Won" if pnl > 0 else "Lost",
                                "direction": "Long" if size > 0 else "Short",
                                "bars_held": getattr(trade_obj, "barlen", 0),
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error processing trade: {e}")
                        continue

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df = trades_df.drop_duplicates(subset=["ref"])
            logger.debug(f"Extracted {len(trades_df)} unique trades")
        else:
            logger.debug("No trades extracted")

        return trades_df

    except Exception as e:
        logger.error(f"Error in trade extraction: {e}")
        return pd.DataFrame()


class WalkForwardAnalysis:
    """Improved walk-forward analysis with enhanced error handling and diagnostics."""

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
        min_data_points: int = 10,
        max_workers: int = 4,
    ):
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
        self.max_workers = max_workers
        self.results = []
        self.all_trades = []

        self.sampler = TPESampler(seed=42)
        self.pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=5)

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            logger.setLevel(logging.WARNING)

        self._validate_inputs()
        self._prepare_data()

    def _validate_inputs(self):
        """Validate initialization parameters."""
        if self.window_type not in ["rolling", "anchored"]:
            raise ValueError("window_type must be 'rolling' or 'anchored'")

        if self.step_size <= 0 or self.testing_period <= 0:
            raise ValueError("step_size and testing_period must be positive")

        if len(self.data) < self.min_training_samples + self.testing_period:
            logger.warning(
                f"Data length ({len(self.data)}) is less than required ({self.min_training_samples + self.testing_period}). Adjusting min_training_samples."
            )
            self.min_training_samples = max(50, len(self.data) - self.testing_period)

        if self.optimization_params:
            for param_name, config in self.optimization_params.items():
                if config.get("type") not in ["int", "float", "categorical"]:
                    raise ValueError(f"Invalid parameter type for {param_name}")
                if config.get("type") == "categorical" and not config.get("choices"):
                    raise ValueError(
                        f"Categorical parameter {param_name} requires choices"
                    )

    def _prepare_data(self):
        """Prepare and validate data with enhanced checks."""
        self.data = self.data.sort_index()
        self.data = self.data[~self.data.index.duplicated(keep="first")]

        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex")

        if self.data.index.tz is None:
            self.data.index = self.data.index.tz_localize(pytz.UTC)
        else:
            self.data.index = self.data.index.tz_convert(pytz.UTC)

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [
            col for col in required_columns if col not in self.data.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        initial_len = len(self.data)
        for col in ["Open", "High", "Low", "Close"]:
            if (self.data[col] <= 0).any():
                logger.warning(f"Removing {col} rows with negative or zero values")
                self.data = self.data[self.data[col] > 0]

        self.data = self.data.fillna(method="ffill").fillna(method="bfill")
        self.data = self.data.dropna(subset=required_columns)

        if len(self.data) < self.min_training_samples + self.testing_period:
            logger.warning(
                f"After cleaning, data length ({len(self.data)}) is less than required. Adjusting min_training_samples to {max(50, len(self.data) - self.testing_period)}"
            )
            self.min_training_samples = max(50, len(self.data) - self.testing_period)

        logger.info(
            f"Data prepared: {len(self.data)} valid points from {self.data.index.min()} to {self.data.index.max()} (removed {initial_len - len(self.data)} rows)"
        )

    def diagnose_data(self):
        """Diagnose data issues and provide detailed statistics."""
        diag = {
            "total_rows": len(self.data),
            "start_date": self.data.index.min(),
            "end_date": self.data.index.max(),
            "missing_values": {
                col: self.data[col].isna().sum()
                for col in ["Open", "High", "Low", "Close", "Volume"]
            },
            "negative_values": {
                col: (self.data[col] <= 0).sum()
                for col in ["Open", "High", "Low", "Close"]
            },
            "data_gaps": [],
            "price_volatility": self.data["Close"].pct_change().describe().to_dict(),
        }

        date_diff = self.data.index.to_series().diff().dt.total_seconds() / (24 * 3600)
        gaps = date_diff[date_diff > 1].index
        if not gaps.empty:
            diag["data_gaps"] = [(g, date_diff[g]) for g in gaps]

        logger.info("Data Diagnostics:")
        for key, value in diag.items():
            logger.info(f"{key}: {value}")

        return diag

    def _get_walk_forward_splits(
        self,
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate walk-forward splits with detailed logging."""
        splits = []
        data_start = self.data.index.min().to_pydatetime()
        data_end = self.data.index.max().to_pydatetime()
        current_start = data_start
        iteration = 0
        max_iterations = 1000

        while current_start < data_end and iteration < max_iterations:
            iteration += 1
            train_start = current_start if self.window_type == "rolling" else data_start
            train_end = train_start + timedelta(days=self.training_period)

            train_mask = (self.data.index >= train_start) & (
                self.data.index <= train_end
            )
            train_data = self.data.loc[train_mask]

            if len(train_data) < self.min_training_samples:
                logger.debug(
                    f"Split {iteration} at {current_start.date()}: Insufficient training data ({len(train_data)} < {self.min_training_samples})"
                )
                current_start += timedelta(days=self.step_size)
                continue

            purge_end = train_end + timedelta(days=self.purge_gap)
            test_start = purge_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.testing_period)

            test_mask = (self.data.index >= test_start) & (self.data.index <= test_end)
            test_data = self.data.loc[test_mask]

            if len(test_data) < self.min_data_points:
                logger.debug(
                    f"Split {iteration} at {current_start.date()}: Insufficient test data ({len(test_data)} < {self.min_data_points})"
                )
                current_start += timedelta(days=self.step_size)
                continue

            if not self._validate_split_data(train_start, train_end):
                logger.debug(
                    f"Split {iteration} at {current_start.date()}: Training data failed validation"
                )
                current_start += timedelta(days=self.step_size)
                continue

            if not self._validate_split_data(test_start, test_end):
                logger.debug(
                    f"Split {iteration} at {current_start.date()}: Test data failed validation"
                )
                current_start += timedelta(days=self.step_size)
                continue

            splits.append((train_start, train_end, test_start, test_end))
            logger.debug(
                f"Split {iteration} accepted: Train {train_start.date()} to {train_end.date()}, Test {test_start.date()} to {test_end.date()}, Test points: {len(test_data)}"
            )
            current_start += timedelta(days=self.step_size)

            if test_end >= data_end:
                break

        if not splits:
            logger.error(
                "No valid splits generated. Possible causes: insufficient data points, strict validation criteria, or parameter misconfiguration."
            )
            logger.error(
                f"Config: training_period={self.training_period}, testing_period={self.testing_period}, step_size={self.step_size}, min_training_samples={self.min_training_samples}, min_data_points={self.min_data_points}"
            )
        else:
            logger.info(f"Generated {len(splits)} valid splits")

        return splits

    def _validate_split_data(self, start_date: datetime, end_date: datetime) -> bool:
        """Validate split data with relaxed criteria."""
        mask = (self.data.index >= start_date) & (self.data.index <= end_date)
        filtered_data = self.data.loc[mask]

        if len(filtered_data) < self.min_data_points:
            logger.debug(
                f"Validation failed for {start_date.date()} to {end_date.date()}: Insufficient data points ({len(filtered_data)} < {self.min_data_points})"
            )
            return False

        for col in ["Open", "High", "Low", "Close"]:
            missing_ratio = filtered_data[col].isna().sum() / len(filtered_data)
            if missing_ratio > 0.1:
                logger.debug(
                    f"Validation failed for {start_date.date()} to {end_date.date()}: Too many missing values in {col} ({missing_ratio:.2%})"
                )
                return False
            if (filtered_data[col] <= 0).sum() > 0:
                logger.debug(
                    f"Validation failed for {start_date.date()} to {end_date.date()}: Negative/zero values in {col}"
                )
                return False
            if len(filtered_data) > 1:
                returns = filtered_data["Close"].pct_change()
                extreme_moves = (returns.abs() > 1.0).sum()
                if extreme_moves > 0:
                    logger.debug(
                        f"Validation failed for {start_date.date()} to {end_date.date()}: Extreme price movements in {col} ({extreme_moves} occurrences)"
                    )
                    return False

        return True

    def run_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run walk-forward analysis with diagnostics."""
        self.diagnose_data()

        splits = self._get_walk_forward_splits()
        if not splits:
            raise ValueError("No valid walk-forward splits generated")

        logger.info(f"Generated {len(splits)} walk-forward splits")

        def process_split(split, idx):
            train_start, train_end, test_start, test_end = split

            try:
                logger.debug(
                    f"Processing period {idx+1}: Train {train_start.date()} to {train_end.date()}, Test {test_start.date()} to {test_end.date()}"
                )
                best_params, best_score = self._optimize_parameters(
                    train_start, train_end
                )
                logger.debug(
                    f"Period {idx+1}: Best params: {best_params}, Best score: {best_score}"
                )

                if best_score == 0 or not best_params:
                    logger.warning(
                        f"Period {idx+1}: Optimization failed or produced no valid parameters. Skipping period."
                    )
                    return None

                cerebro = self._create_cerebro(test_start, test_end, best_params)
                results = cerebro.run()

                if results and len(results) > 0:
                    strat = results[0]
                    metrics = self._extract_performance_metrics(strat)

                    result_dict = {
                        "walk_forward": idx + 1,
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

                    trades = metrics["trades_list"]
                    for trade in trades:
                        trade["walk_forward"] = idx + 1
                        trade["test_start"] = test_start.date()
                        trade["test_end"] = test_end.date()

                    logger.info(
                        f"Period {idx+1}: Return={metrics['total_return']:.2%}, "
                        f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                        f"Trades={metrics['total_trades']}"
                    )

                    return result_dict, trades
                else:
                    logger.warning(f"Period {idx+1}: No results from cerebro.run()")
                    return None
            except Exception as e:
                logger.error(f"Error processing period {idx+1}: {e}")
                return None

        results = []
        all_trades = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(process_split, split, i)
                for i, split in enumerate(splits)
            ]
            for future in futures:
                result = future.result()
                if result:
                    result_dict, trades = result
                    results.append(result_dict)
                    all_trades.extend(trades)

        self.results = results
        self.all_trades = all_trades
        return pd.DataFrame(results), pd.DataFrame(all_trades)

    def _create_cerebro(
        self, start_date: datetime, end_date: datetime, params: Dict = None
    ) -> bt.Cerebro:
        """Create cerebro instance with optimized data handling."""
        cerebro = bt.Cerebro()

        buffer_days = 50  # Increased buffer to ensure sufficient data for indicators
        buffer_start = start_date - timedelta(days=buffer_days)
        buffer_end = end_date + timedelta(days=buffer_days)

        mask = (self.data.index >= buffer_start) & (self.data.index <= buffer_end)
        filtered_df = self.data.loc[mask].copy()

        logger.debug(
            f"Creating cerebro for {start_date.date()} to {end_date.date()}: {len(filtered_df)} data points before cleaning"
        )

        if len(filtered_df) < self.min_data_points:
            raise ValueError(f"Insufficient data: only {len(filtered_df)} points")

        filtered_df = filtered_df.dropna()
        logger.debug(f"After cleaning: {len(filtered_df)} data points")

        if len(filtered_df) < self.min_data_points:
            raise ValueError(
                f"Insufficient data after cleaning: {len(filtered_df)} points"
            )

        # Ensure sufficient data for indicators (e.g., EMA, RSI)
        min_lookback = (
            max(
                params.get("fast_ema", 20),
                params.get("slow_ema", 50),
                params.get("rsi_period", 14),
            )
            if params
            else 50
        )
        if len(filtered_df) < min_lookback:
            raise ValueError(
                f"Insufficient data for indicators: {len(filtered_df)} points, need at least {min_lookback}"
            )

        data_feed = bt.feeds.PandasData(dataname=filtered_df)
        cerebro.adddata(data_feed)

        strategy_params = {"verbose": self.verbose}
        if params:
            strategy_params.update(params)

        cerebro.addstrategy(self.strategy_class, **strategy_params)
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.VWR, _name="vwr")

        return cerebro

    def _extract_performance_metrics(self, strat) -> Dict[str, float]:
        """Extract performance metrics with enhanced error handling."""
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
            if hasattr(strat, "analyzers"):
                if hasattr(strat.analyzers, "returns"):
                    returns = strat.analyzers.returns.get_analysis()
                    metrics["total_return"] = returns.get("rtot", 0.0)
                    metrics["avg_return"] = returns.get("ravg", 0.0)

                if hasattr(strat.analyzers, "sharpe"):
                    sharpe = strat.analyzers.sharpe.get_analysis()
                    metrics["sharpe_ratio"] = sharpe.get("sharperatio", 0.0) or 0.0

                if hasattr(strat.analyzers, "drawdown"):
                    drawdown = strat.analyzers.drawdown.get_analysis()
                    metrics["max_drawdown"] = (
                        drawdown.get("max", {}).get("drawdown", 0.0) or 0.0
                    )

                if hasattr(strat.analyzers, "vwr"):
                    vwr = strat.analyzers.vwr.get_analysis()
                    metrics["vwr"] = vwr.get("vwr", 0.0) or 0.0

                if hasattr(strat.analyzers, "trades"):
                    trades = strat.analyzers.trades.get_analysis()
                    metrics["total_trades"] = trades.get("total", {}).get("total", 0)
                    metrics["winning_trades"] = trades.get("won", {}).get("total", 0)
                    metrics["losing_trades"] = trades.get("lost", {}).get("total", 0)
                    metrics["win_rate"] = (
                        metrics["winning_trades"] / metrics["total_trades"]
                        if metrics["total_trades"] > 0
                        else 0.0
                    )

                    gross_profit = (
                        trades.get("won", {}).get("pnl", {}).get("total", 0.0) or 0.0
                    )
                    gross_loss = abs(
                        trades.get("lost", {}).get("pnl", {}).get("total", 0.0) or 0.0
                    )
                    metrics["profit_factor"] = (
                        gross_profit / gross_loss if gross_loss > 0 else 0.0
                    )
                    metrics["avg_trade_pnl"] = (
                        trades.get("pnl", {}).get("average", 0.0) or 0.0
                    )

            trades_df = safe_extract_trades(strat, self.data)
            metrics["trades_list"] = trades_df.to_dict("records")

            if len(trades_df) > 0:
                metrics["total_trades"] = len(trades_df)
                metrics["winning_trades"] = len(trades_df[trades_df["pnl"] > 0])
                metrics["losing_trades"] = len(trades_df[trades_df["pnl"] <= 0])
                metrics["win_rate"] = (
                    metrics["winning_trades"] / metrics["total_trades"]
                    if metrics["total_trades"] > 0
                    else 0.0
                )
                metrics["profit_factor"] = (
                    (
                        trades_df[trades_df["pnl"] > 0]["pnl"].sum()
                        / abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
                    )
                    if trades_df[trades_df["pnl"] < 0]["pnl"].sum() != 0
                    else 0.0
                )
                metrics["avg_trade_pnl"] = (
                    trades_df["pnl"].mean() if not trades_df.empty else 0.0
                )

        except Exception as e:
            logger.error(f"Error extracting performance metrics: {e}")

        return metrics

    def _create_objective_function(
        self, start_date: datetime, end_date: datetime
    ) -> Callable:
        """Create optimized objective function for Optuna."""

        def objective(trial):
            params = {}
            try:
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

                cerebro = self._create_cerebro(start_date, end_date, params)
                results = cerebro.run()

                if results and len(results) > 0:
                    metrics = self._extract_performance_metrics(results[0])
                    score = metrics.get(self.optimization_metric, 0.0)

                    if self.optimization_metric == "max_drawdown":
                        score = -score
                    if metrics["total_trades"] < 3:
                        score *= 0.5
                    if metrics["win_rate"] < 0.3 and metrics["total_trades"] > 5:
                        score *= 0.8

                    logger.debug(f"Trial params: {params}, Score: {score}")
                    return score if not np.isnan(score) else -1.0
                logger.debug(f"No results from cerebro.run() for params: {params}")
                return -1.0
            except Exception as e:
                logger.error(f"Error in objective function with params {params}: {e}")
                return -1.0

        return objective

    def _optimize_parameters(
        self, start_date: datetime, end_date: datetime
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize parameters with improved stability checks."""
        if not self.optimization_params:
            logger.warning(
                "No optimization parameters provided. Returning default parameters."
            )
            return {}, 0.0

        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=f"walkforward_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
        )

        objective_func = self._create_objective_function(start_date, end_date)
        study.optimize(
            objective_func,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=self.verbose,
        )

        if study.best_value <= -1.0:
            logger.warning(
                f"Optimization failed for {start_date.date()} to {end_date.date()}. No valid parameters found."
            )
            return {}, 0.0

        return study.best_params, study.best_value

    def get_statistical_significance(self) -> Dict[str, float]:
        """Calculate statistical significance with enhanced metrics."""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)
        returns = df["total_return"]

        stats = {
            "mean_return": returns.mean(),
            "std_return": returns.std(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "sharpe_ratio": returns.mean() / returns.std() if returns.std() > 0 else 0,
        }

        if len(returns) > 1:
            t_stat = returns.mean() / (returns.std() / np.sqrt(len(returns)))
            stats["t_statistic"] = t_stat
            stats["is_significant_95pct"] = abs(t_stat) > 1.96

        return stats

    def get_robustness_metrics(self) -> Dict[str, float]:
        """Calculate robustness metrics with parameter stability."""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)
        param_analysis = self.get_parameter_analysis()

        robustness = {
            "return_consistency": 1
            - (df["total_return"].std() / abs(df["total_return"].mean() or 1)),
            "positive_periods_ratio": (df["total_return"] > 0).mean(),
            "max_consecutive_losses": self._calculate_max_consecutive_losses(
                df["total_return"]
            ),
            "performance_stability": df["sharpe_ratio"].std(),
        }

        if not param_analysis.empty:
            for param in param_analysis.select_dtypes(include=[np.number]).columns:
                if param not in ["walk_forward", "total_return", "sharpe_ratio"]:
                    robustness[f"{param}_cv"] = (
                        param_analysis[param].std() / param_analysis[param].mean()
                        if param_analysis[param].mean() != 0
                        else np.inf
                    )

        return robustness

    def _calculate_max_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive losing periods."""
        losing_periods = (returns < 0).astype(int)
        max_consecutive = current_consecutive = 0

        for is_loss in losing_periods:
            if is_loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        return max_consecutive

    def get_parameter_analysis(self) -> pd.DataFrame:
        """Analyze parameter stability across periods."""
        param_data = []
        for result in self.results:
            try:
                params = (
                    ast.literal_eval(result["best_params"])
                    if result["best_params"]
                    else {}
                )
                param_data.append(
                    {
                        "walk_forward": result["walk_forward"],
                        "test_start": result["test_start"],
                        "test_end": result["test_end"],
                        "total_return": result["total_return"],
                        "sharpe_ratio": result["sharpe_ratio"],
                        **params,
                    }
                )
            except Exception as e:
                logger.error(
                    f"Error parsing parameters for period {result['walk_forward']}: {e}"
                )

        return pd.DataFrame(param_data)

    def get_trade_details(self) -> pd.DataFrame:
        """Get detailed trade information."""
        trades_df = pd.DataFrame(self.all_trades)
        if not trades_df.empty:
            trades_df = trades_df.drop_duplicates(subset=["ref"])
            trades_df = trades_df.rename(
                columns={
                    "pnl_net": "net_profit",
                    "entry_time": "entry_date",
                    "exit_time": "exit_date",
                    "size": "qty",
                }
            )
            trades_df["bars_held"] = (
                trades_df["exit_date"] - trades_df["entry_date"]
            ).dt.days
        return trades_df

    def get_summary_statistics(self) -> Dict[str, float]:
        """Get comprehensive summary statistics."""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)
        trades_df = self.get_trade_details()

        return {
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

    def plot_results(self, save_path: str = None):
        """Enhanced result visualization."""
        if not self.results:
            logger.warning("No results to plot")
            return

        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import seaborn as sns

            df = pd.DataFrame(self.results)
            trades_df = pd.DataFrame(self.all_trades)

            sns.set_style("whitegrid")
            fig, axes = plt.subplots(3, 2, figsize=(15, 15))
            fig.suptitle("Walk-Forward Analysis Results", fontsize=16)

            ax1 = axes[0, 0]
            test_dates = pd.to_datetime(df["test_start"])
            sns.lineplot(x=test_dates, y=df["total_return"] * 100, ax=ax1, marker="o")
            ax1.set_title("Returns by Period")
            ax1.set_ylabel("Return (%)")
            ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5)

            ax2 = axes[0, 1]
            sns.lineplot(
                x=test_dates, y=df["sharpe_ratio"], ax=ax2, marker="s", color="green"
            )
            ax2.set_title("Sharpe Ratio by Period")
            ax2.set_ylabel("Sharpe Ratio")
            ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)

            ax3 = axes[1, 0]
            sns.lineplot(
                x=test_dates,
                y=df["max_drawdown"] * 100,
                ax=ax3,
                marker="^",
                color="red",
            )
            ax3.set_title("Max Drawdown by Period")
            ax3.set_ylabel("Max Drawdown (%)")

            ax4 = axes[1, 1]
            ax4.bar(test_dates, df["total_trades"], color="orange", alpha=0.7)
            ax4.set_title("Total Trades by Period")
            ax4.set_ylabel("Number of Trades")

            ax5 = axes[2, 0]
            if not trades_df.empty:
                sns.histplot(trades_df["pnl"], bins=30, ax=ax5, color="purple")
                ax5.set_title("Trade PNL Distribution")
                ax5.set_xlabel("PNL")
                ax5.set_ylabel("Frequency")

            ax6 = axes[2, 1]
            if not trades_df.empty:
                sns.histplot(trades_df["bars_held"], bins=30, ax=ax6, color="blue")
                ax6.set_title("Trade Duration Distribution")
                ax6.set_xlabel("Duration (bars)")
                ax6.set_ylabel("Frequency")

            for ax in [ax1, ax2, ax3, ax4]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Plot saved to {save_path}")

            plt.show()

        except ImportError:
            logger.error("matplotlib or seaborn not available for plotting")
        except Exception as e:
            logger.error(f"Error creating plots: {e}")

    def export_results(self, filepath: str, trades_filepath: str = None):
        """Export results with error handling."""
        if not self.results:
            logger.warning("No results to export")
            return

        try:
            df = pd.DataFrame(self.results)
            df.to_csv(filepath, index=False)
            logger.info(f"Results exported to {filepath}")

            if trades_filepath and self.all_trades:
                trades_df = pd.DataFrame(self.all_trades)
                trades_df = trades_df.drop_duplicates(subset=["ref"])
                trades_df.to_csv(trades_filepath, index=False)
                logger.info(f"Trade details exported to {trades_filepath}")
        except Exception as e:
            logger.error(f"Error exporting results: {e}")

    def get_detailed_summary(self) -> str:
        """Generate detailed summary report."""
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
- Average trades per Period: {summary['avg_trades_per_period']:.1f}
- Average Win Rate: {summary['avg_win_rate']:.1%}
- Average Profit Factor: {summary['avg_profit_factor']:.2f}
- Total Trades: {len(trades_df) if not trades_df.empty else 0}

Consistency Metrics:
- Positive Periods: {summary['positive_periods']:.0f}/{summary['total_periods']:.0f}
- Positive Period Ratio: {summary['positive_period_ratio']:.1%}
- Consistency Score: {summary['consistency_score']:.3f}

Period-by-Period Results:
"""
        for _, row in df.iterrows():
            period_trades = trades_df[trades_df["walk_forward"] == row["walk_forward"]]
            report += f"""
Period {row['walk_forward']}: {row['test_start']} to {row['test_end']}
Return: {row['total_return']:.2%}, Sharpe: {row['sharpe_ratio']:.2f}
Drawdown: {row['max_drawdown']:.2%}, Trades: {row['total_trades']}
Best Parameters: {row['best_params']}
Number of Trades: {len(period_trades)}
"""
        return report


def run_standardized_walkforward_example(strategy_name: str = "EMARSI"):
    """Run walk-forward analysis with dynamic strategy and parameter handling."""
    try:
        df = get_data_sync(
            ticker="SBIN.NS",
            start_date="2020-01-01",
            end_date="2024-12-31",
            interval="1d",
        )

        if df.empty:
            raise ValueError("No data retrieved from get_data_sync")

        default_optimization_params = {
            "EMARSI": {
                "fast_ema": {"type": "int", "low": 5, "high": 20, "step": 1},
                "slow_ema": {"type": "int", "low": 20, "high": 50, "step": 1},
                "rsi_period": {"type": "int", "low": 10, "high": 30, "step": 1},
                "rsi_low": {"type": "int", "low": 20, "high": 40, "step": 1},
                "rsi_high": {"type": "int", "low": 60, "high": 80, "step": 1},
            },
            "SMABollinger": {
                "sma_period": {"type": "int", "low": 10, "high": 50, "step": 1},
                "bb_period": {"type": "int", "low": 10, "high": 30, "step": 1},
                "bb_dev": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.1},
            },
        }

        strategy_map = {"EMARSI": EMARSI, "SMABollinger": SMABollinger}

        if strategy_name not in strategy_map:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. Choose from {list(strategy_map.keys())}"
            )

        strategy_class = strategy_map[strategy_name]

        optimization_params = getattr(strategy_class, "optimization_params", None)
        if optimization_params is None or not isinstance(optimization_params, dict):
            logger.warning(
                f"No valid optimization_params found in {strategy_name}. Using default parameters."
            )
            optimization_params = default_optimization_params.get(strategy_name, {})
        else:
            logger.info(
                f"Using optimization_params from {strategy_name}: {optimization_params}"
            )

        for param_name, config in optimization_params.items():
            if config.get("type") not in ["int", "float", "categorical"]:
                raise ValueError(
                    f"Invalid parameter type for {param_name} in {strategy_name}"
                )
            if config.get("type") == "categorical" and not config.get("choices"):
                raise ValueError(
                    f"Categorical parameter {param_name} in {strategy_name} requires choices"
                )
            if config.get("type") in ["int", "float"] and (
                "low" not in config or "high" not in config
            ):
                raise ValueError(
                    f"Parameter {param_name} in {strategy_name} missing low or high"
                )

        if not optimization_params:
            logger.warning(
                f"No optimization parameters for {strategy_name}. Optimization will be skipped."
            )

        wfa = WalkForwardAnalysis(
            strategy_class=strategy_class,
            data=df,
            training_period=252,
            testing_period=21,
            step_size=21,
            window_type="rolling",
            min_training_samples=100,
            purge_gap=1,
            embargo_period=0,
            optimization_metric="sharpe_ratio",
            optimization_params=optimization_params,
            initial_cash=100000,
            commission=0.001,
            n_trials=50,
            verbose=True,
            max_workers=4,
        )

        results_df, trades_df = wfa.run_analysis()

        significance = wfa.get_statistical_significance()
        robustness = wfa.get_robustness_metrics()

        logger.info("\nStatistical Significance:")
        for key, value in significance.items():
            logger.info(f"{key}: {value}")

        logger.info("\nRobustness Metrics:")
        for key, value in robustness.items():
            logger.info(f"{key}: {value}")

        return results_df, trades_df, significance, robustness

    except Exception as e:
        logger.error(f"Error running walkforward example: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}, {}


if __name__ == "__main__":
    results, trades, significance, robustness = run_standardized_walkforward_example()
