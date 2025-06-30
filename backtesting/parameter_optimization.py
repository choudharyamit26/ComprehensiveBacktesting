from typing import Any, Dict
import backtrader as bt
import optuna
from .data import get_data
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
            self.result = 0.0
            return

        # Calculate downside deviation (standard deviation of negative returns)
        negative_returns = returns[returns < 0]
        downside_deviation = (
            np.sqrt(np.mean(negative_returns**2)) * np.sqrt(self.params.factor)
            if not negative_returns.empty
            else 0.0
        )

        # Calculate annualized return
        total_return = self.strategy.analyzers.returns.get_analysis().get("rtot", 0.0)
        annualized_return = (np.exp(total_return) - 1) * self.params.factor

        # Sortino ratio: (annualized return - risk-free rate) / downside deviation
        self.result = (
            (annualized_return - self.params.riskfreerate) / downside_deviation
            if downside_deviation != 0
            else 0.0
        )
        self.analysis = {"sortinoratio": self.result}

    def get_analysis(self):
        return self.analysis


class OptimizationObjective:
    def __init__(
        self,
        strategy_class,
        ticker: str,
        start_date: str,
        end_date: str,
        initial_cash: float,
        commission: float,
    ):
        """Initialize optimization objective.

        Args:
            strategy_class: Backtrader strategy class.
            ticker (str): Stock ticker symbol.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            initial_cash (float): Initial portfolio cash.
            commission (float): Broker commission rate.
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
        self.data = get_data(ticker, start_date, end_date)
        if self.data is None or self.data.empty:
            raise ValueError(
                f"No data available for {ticker} from {start_date} to {end_date}"
            )
        logger.info(
            f"Initialized optimization for {ticker} from {start_date} to {end_date} with strategy {strategy_class.__name__}"
        )

    def __call__(self, trial):
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object.

        Returns:
            float: Objective value (negative total return for maximization).
        """
        try:
            logger.debug(f"Running trial {trial.number} for {self.ticker}")
            params = (
                self.strategy_class.get_param_space(trial)
                if hasattr(self.strategy_class, "get_param_space")
                else {}
            )
            logger.debug(f"Trial parameters: {params}")

            min_data_points = (
                self.strategy_class.get_min_data_points(params)
                if hasattr(self.strategy_class, "get_min_data_points")
                else 100
            )
            if len(self.data) < min_data_points:
                logger.warning(
                    f"Insufficient data: {len(self.data)} rows available, {min_data_points} required"
                )
                return -999

            results, _ = run_backtest(
                strategy_class=self.strategy_class,
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_cash=self.initial_cash,
                commission=self.commission,
                **params,
            )

            # Defensive check for results
            if not isinstance(results, list) or not results:
                logger.error(
                    f"Invalid results from run_backtest: type {type(results)}, value {results}"
                )
                return -999
            if not isinstance(results[0], bt.Strategy):
                logger.error(
                    f"Invalid strategy instance: type {type(results[0])}, value {results[0]}"
                )
                return -999

            # Access trade analyzer
            trade_analyzer = results[0].analyzers.trades.get_analysis()
            total_trades = trade_analyzer.get("total", {}).get("total", 0)
            total_return = trade_analyzer.get("pnl", {}).get("net", {}).get("total", 0)

            if total_trades < 5:  # Minimum trade threshold
                logger.warning(f"Insufficient trades: {total_trades}")
                return -999

            objective_value = (
                total_return / self.initial_cash * 100
            )  # Return as percentage
            logger.debug(
                f"Trial {trial.number} objective value: {objective_value:.2f}%"
            )
            return objective_value

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {str(e)}")
            traceback.print_exc()
            return -999


def optimize_strategy(
    strategy_class,
    ticker: str,
    start_date: str,
    end_date: str,
    n_trials: int = 50,
    initial_cash: float = 100000.0,
    commission: float = 0.00,
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
        f"Starting parameter optimization for {ticker} with {strategy_class.__name__}"
    )
    print(
        f"Optimizing strategy {strategy_class.__name__} for {ticker} from {start_date} to {end_date}"
    )
    print(f"Number of trials: {n_trials}")
    print(f"Strategy class type: {type(strategy_class)}")

    try:
        study = optuna.create_study(direction="maximize")
        objective = OptimizationObjective(
            strategy_class=strategy_class,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            commission=commission,
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"Optimization completed. Best value: {best_value:.2f}%")
        print(f"Best value: {best_value:.2f}%")
        print(f"Best parameters: {best_params}")

        # Run final backtest with best parameters
        print("Running final backtest with best parameters...")
        results, cerebro = run_backtest(
            strategy_class=strategy_class,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            commission=commission,
            **best_params,
        )

        return {
            "best_params": best_params,
            "best_value": best_value,
            "study": study,
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
            "parameter_importance": optuna.importance.get_param_importances(study),
        }

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


def quick_optimize(strategy_class, ticker="AAPL", n_trials=20):
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
