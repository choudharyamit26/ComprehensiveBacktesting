from typing import Any, Dict
import backtrader as bt
import optuna

from comprehensive_backtesting.registry import get_strategy
from .data import get_data_sync
import logging
import numpy as np
import pandas as pd
from .utils import run_backtest
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SortinoRatio(bt.Analyzer):
    """Custom Backtrader analyzer to calculate the Sortino ratio."""

    params = (
        ("riskfreerate", 0.0),
        ("factor", 252),
    )

    def __init__(self):
        self.rets = self.strategy.analyzers.timereturn.get_analysis()

    def stop(self):
        returns = pd.Series(self.rets)
        if returns.empty:
            self.result = None
            self.analysis = {"sortinoratio": None}
            return

        # Calculate downside deviation (standard deviation of negative returns)
        negative_returns = returns[returns < 0]
        if negative_returns.empty:
            downside_deviation = None
        else:
            downside_deviation = np.sqrt(np.mean(negative_returns**2)) * np.sqrt(
                self.params.factor
            )

        # Calculate annualized return
        total_return = self.strategy.analyzers.returns.get_analysis().get("rtot", 0.0)
        annualized_return = (np.exp(total_return) - 1) * self.params.factor

        # Sortino ratio: (annualized return - risk-free rate) / downside deviation
        if downside_deviation is None or downside_deviation == 0:
            self.result = None
        else:
            self.result = (
                annualized_return - self.params.riskfreerate
            ) / downside_deviation
        self.analysis = {"sortinoratio": self.result}

    def get_analysis(self):
        return self.analysis


class OptimizationObjective:
    def __init__(
        self,
        data,
        strategy_class,
        ticker: str,
        start_date: str,
        end_date: str,
        initial_cash: float,
        commission: float,
        interval,
    ):
        """Initialize optimization objective.

        Args:
            strategy_class: Backtrader strategy class.
            ticker (str): Stock ticker symbol.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            initial_cash (float): Initial portfolio cash.
            commission (float): Broker commission rate.
            interval (str): Data interval (e.g., '5m', '15m').
        """
        if not isinstance(strategy_class, type):
            raise TypeError(
                f"strategy_class must be a class, got {type(strategy_class)}"
            )
        self.strategy_class = strategy_class
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.commission = commission
        self.interval = interval
        self.data = data
        if self.data is None or self.data.empty:
            raise ValueError(
                f"No data available for {ticker} from {start_date} to {end_date}"
            )
        logger.info(
            f"Initialized optimization for {ticker} from {start_date} to {end_date} with strategy {strategy_class}"
        )

    def __call__(self, trial):
        """Objective function for Optuna optimization with robust error handling."""
        try:
            logger.info(f"Starting trial {trial.number} for {self.ticker}")

            # Get parameters from strategy class
            if hasattr(self.strategy_class, "get_param_space"):
                params = self.strategy_class.get_param_space(trial)
            else:
                logger.warning("Using default parameter space")
                params = {
                    "fast_ema_period": trial.suggest_int("fast_ema_period", 5, 20),
                    "slow_ema_period": trial.suggest_int("slow_ema_period", 21, 50),
                    "rsi_period": trial.suggest_int("rsi_period", 10, 20),
                    "rsi_upper": trial.suggest_int("rsi_upper", 60, 75),
                    "rsi_lower": trial.suggest_int("rsi_lower", 25, 40),
                }

            logger.info(f"Trial parameters: {params}")

            # Run backtest with current parameters
            results, _ = run_backtest(
                data=self.data,
                strategy_class=self.strategy_class,
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_cash=self.initial_cash,
                commission=self.commission,
                interval=self.interval,
                **params,
            )

            # Validate results structure
            if not results or not isinstance(results, list) or len(results) == 0:
                logger.error(
                    "Backtest returned empty results (possibly due to insufficient data)"
                )
                return -999

            strategy = results[0]
            if not isinstance(strategy, bt.Strategy):
                logger.error("Backtest returned invalid strategy object")
                return -999

            # Extract performance metrics with robust fallbacks
            try:
                # Access analyzers with defensive checks
                returns_analyzer = getattr(strategy.analyzers, "returns", None)
                trades_analyzer = getattr(strategy.analyzers, "trades", None)
                sharpe_analyzer = getattr(strategy.analyzers, "sharpe", None)
                sortino_analyzer = getattr(strategy.analyzers, "sortino", None)

                # Get total return
                total_return = 0
                if returns_analyzer:
                    returns_analysis = returns_analyzer.get_analysis()
                    total_return = returns_analysis.get("rtot", 0)

                # Get trade metrics
                total_trades = 0
                if trades_analyzer:
                    trade_analysis = trades_analyzer.get_analysis()
                    total_trades = trade_analysis.get("total", {}).get("total", 0)

                # Get risk-adjusted metrics
                sharpe_ratio = 0
                if sharpe_analyzer:
                    sharpe_analysis = sharpe_analyzer.get_analysis()
                    sharpe_ratio = sharpe_analysis.get("sharperatio", 0)
                    if sharpe_ratio is None or not isinstance(
                        sharpe_ratio, (int, float)
                    ):
                        sharpe_ratio = 0
                sortino_ratio = 0
                if sortino_analyzer:
                    sortino_analysis = sortino_analyzer.get_analysis()
                    sortino_ratio = sortino_analysis.get("sortinoratio", 0)
                    if sortino_ratio is None or not isinstance(
                        sortino_ratio, (int, float)
                    ):
                        sortino_ratio = 0

                # Calculate composite objective
                objective_value = (
                    total_return * 10
                )  # Prioritize returns over risk metrics

                # Apply trade-based adjustments
                if total_trades < 3:
                    # Penalize but don't eliminate low-trade configs
                    adjustment = 0.5 + (total_trades * 0.2)
                    if not isinstance(objective_value, (int, float)):
                        objective_value = -999
                    else:
                        objective_value *= adjustment
                    logger.info(f"Low trade adjustment: {adjustment:.2f}x")

                # Final validation
                if not np.isfinite(objective_value):
                    logger.warning("Non-finite objective value, returning -999")
                    return -999

                logger.info(
                    f"Trial {trial.number} results: "
                    f"Trades={total_trades}, "
                    f"Return={total_return:.4f}, "
                    f"Sharpe={sharpe_ratio:.4f}, "
                    f"Sortino={sortino_ratio:.4f}, "
                    f"Objective={objective_value:.4f}"
                )

                return objective_value

            except Exception as e:
                logger.error(f"Metric extraction failed: {str(e)}")
                return -999

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"Trial {trial.number} failed completely: {str(e)}")
            return -999


def optimize_strategy(
    data,
    strategy_class,
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    n_trials: int,
    initial_cash: float = 100000.0,
    commission: float = 0.00,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Optimize strategy parameters using Optuna.

    Args:
        strategy_class: Backtrader strategy class.
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        n_trials (int): Number of optimization trials.
        initial_cash (float): Initial portfolio cash.
        commission (float): Broker commission rate.

    Returns:
        Dict: Optimization results including best parameters and value.
    """
    logger.info(
        f"Starting parameter optimization for {ticker} with {strategy_class} on interval {interval}"
    )
    print(
        f"Optimizing strategy {strategy_class} for {ticker} from {start_date} to {end_date}"
    )
    print(f"Number of trials: {n_trials}")
    print(f"Strategy class type: {type(strategy_class)}")

    try:
        study = optuna.create_study(direction="maximize")
        objective = OptimizationObjective(
            data=data,
            strategy_class=get_strategy(strategy_class),
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            commission=commission,
            interval=interval,
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"Optimization completed. Best value: {best_value:.2f}%")
        print(f"Best value: {best_value:.2f}%")
        print(f"Best parameters: {best_params}")

        # Run final backtest with best parameters
        print("Running final backtest with best parameters...")
        try:
            results, cerebro = run_backtest(
                data=data,
                strategy_class=strategy_class,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                initial_cash=initial_cash,
                commission=commission,
                interval=interval,
                **best_params,
            )
        except Exception as e:
            logger.error(f"Final backtest failed: {str(e)}")
            results = None
            cerebro = None

        return {
            "best_params": best_params,
            "best_value": best_value,
            "study": study,  # Always return study object
            "cerebro": cerebro,
            "results": results,
        }

    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        traceback.print_exc()
        print(f"Optimization failed: {str(e)}. Returning default parameters.")
        return {
            "best_params": {
                "fast_ema_period": 12,
                "slow_ema_period": 26,
                "rsi_period": 14,
                "rsi_upper": 70,
                "rsi_lower": 30,
            },
            "best_value": None,
            "study": None,
            "cerebro": None,
            "results": None,
        }


def analyze_optimization_results(study, save_plots=False):
    """Analyze and visualize optimization results.

    Args:
        study: Optuna study object.
        save_plots (bool): Whether to save plots to files.

    Returns:
        dict: Analysis results.
    """
    if study is None:
        return {"error": "No study available for analysis"}

    logger.info("Analyzing optimization results...")
    try:
        trials_df = study.trials_dataframe()
        if trials_df.empty:
            logger.warning("No completed trials found")
            return {"error": "No completed trials found"}

        analysis = {
            "n_trials": len(trials_df),
            "n_complete": len(trials_df[trials_df["state"] == "COMPLETE"]),
            "n_failed": len(trials_df[trials_df["state"] == "FAIL"]),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "mean_objective": trials_df["value"].mean(),
            "std_objective": trials_df["value"].std(),
            "median_objective": trials_df["value"].median(),
        }

        # Add parameter importance if possible
        try:
            analysis["parameter_importance"] = optuna.importance.get_param_importances(
                study
            )
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {str(e)}")
            analysis["parameter_importance"] = {}

        param_stats = {}
        for param in study.best_params.keys():
            param_col = f"params_{param}"
            if param_col in trials_df.columns:
                param_stats[param] = {
                    "best": study.best_params[param],
                    "mean": trials_df[param_col].mean(),
                    "std": trials_df[param_col].std(),
                    "min": trials_df[param_col].min(),
                    "max": trials_df[param_col].max(),
                }

        analysis["parameter_statistics"] = param_stats

        print("=== OPTIMIZATION ANALYSIS ===")
        print(f"Total trials: {analysis['n_trials']}")
        print(f"Completed trials: {analysis['n_complete']}")
        print(f"Failed trials: {analysis['n_failed']}")
        print(f"Best objective value: {analysis['best_value']:.4f}")
        print(f"Mean objective value: {analysis['mean_objective']:.4f}")
        print(f"Median objective value: {analysis['median_objective']:.4f}")
        print(f"Std objective value: {analysis['std_objective']:.4f}")
        print("\nBest parameters:")
        for param, value in analysis["best_params"].items():
            print(f"  {param}: {value}")
        print("\nParameter importance:")
        for param, importance in analysis["parameter_importance"].items():
            print(f"  {param}: {importance:.4f}")
        print("\nParameter statistics:")
        for param, stats in param_stats.items():
            print(f"  {param}:")
            print(
                f"    Best: {stats['best']}, Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}"
            )
            print(f"    Range: [{stats['min']}, {stats['max']}]")

        if save_plots:
            try:
                import matplotlib.pyplot as plt

                fig = optuna.visualization.matplotlib.plot_optimization_history(study)
                plt.savefig("optimization_history.png")
                print("Saved optimization_history.png")
                fig = optuna.visualization.matplotlib.plot_param_importances(study)
                plt.savefig("parameter_importance.png")
                print("Saved parameter_importance.png")
                plt.close("all")
            except ImportError:
                logger.warning("Matplotlib not available for plotting")
            except Exception as e:
                logger.error(f"Error generating plots: {str(e)}")

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing results: {str(e)}")
        traceback.print_exc()
        return {"error": f"Error analyzing results: {str(e)}"}


def quick_optimize(strategy_class, ticker, n_trials, interval):
    """Quick optimization for testing purposes.

    Args:
        strategy_class: Backtrader strategy class.
        ticker (str): Stock ticker symbol.
        n_trials (int): Number of optimization trials.

    Returns:
        dict: Optimization results including best parameters and performance.
    """
    try:
        logger.info(
            f"Starting quick optimization for {ticker} with {n_trials} trials..."
        )
        print(f"\nQuick optimization for {ticker} with {n_trials} trials...")

        results = optimize_strategy(
            strategy_class=strategy_class,
            ticker=ticker,
            start_date="2022-01-01",
            end_date="2025-06-01",
            n_trials=n_trials,
            initial_cash=100000.0,
            commission=0.001,
            interval=interval,
        )

        print("\nQuick Optimization Results:")
        print(f"Best objective value: {results['best_value']:.4f}")
        print("Best parameters:")
        for param, value in results["best_params"].items():
            print(f"  {param}: {value}")

        save_analysis = (
            input("\nSave optimization analysis plots? (y/n): ").lower().strip()
        )
        if save_analysis == "y":
            analyze_optimization_results(results["study"], save_plots=True)
        else:
            analyze_optimization_results(results["study"], save_plots=False)

        return results

    except Exception as e:
        logger.error(f"Quick optimization failed: {str(e)}")
        traceback.print_exc()
        print(f"Quick optimization failed: {str(e)}")
        return {"error": f"Quick optimization failed: {str(e)}"}
