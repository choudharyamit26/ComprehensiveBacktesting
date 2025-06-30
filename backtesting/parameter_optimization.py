import backtrader as bt
import optuna
from optuna.samplers import TPESampler
from .data import get_data
from stratgies.ema_rsi import EMARSI

# from stratgies.ema_rsi import EMARSI


class OptimizationObjective:
    def __init__(self, strategy_class, data, initial_cash=100000.0, commission=0.001):
        self.strategy_class = strategy_class
        self.data = data
        self.initial_cash = initial_cash
        self.commission = commission

    def __call__(self, trial):
        # Suggest parameter values for optimization
        if self.strategy_class == EMARSI:
            # params = {
            #     'fast_ema_period': trial.suggest_int('fast_ema_period', 5, 30),
            #     'slow_ema_period': trial.suggest_int('slow_ema_period', 31, 60),
            #     'rsi_period': trial.suggest_int('rsi_period', 10, 20),
            #     'rsi_upper': trial.suggest_int('rsi_upper', 65, 85),
            #     'rsi_lower': trial.suggest_int('rsi_lower', 15, 35),
            # }
            params = {
                "fast_ema_period": trial.suggest_int("fast_ema_period", 5, 30),
                "slow_ema_period": trial.suggest_int(
                    "slow_ema_period", 31, 75
                ),  # Reduced max
                "rsi_period": trial.suggest_int("rsi_period", 10, 20),
                "rsi_upper": trial.suggest_int("rsi_upper", 60, 75),  # More reasonable
                "rsi_lower": trial.suggest_int("rsi_lower", 25, 40),  # More reasonable
            }
            # Ensure fast EMA > slow EMA
            if params["fast_ema_period"] <= params["slow_ema_period"]:
                params["fast_ema_period"] = params["slow_ema_period"] + 5

        else:
            # Default parameters for other strategies
            params = {}

        try:
            # Run backtest with suggested parameters
            cerebro = bt.Cerebro()
            cerebro.addstrategy(self.strategy_class, **params)
            cerebro.adddata(self.data)
            cerebro.broker.setcash(self.initial_cash)
            cerebro.broker.setcommission(commission=self.commission)

            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

            # Run the backtest
            results = cerebro.run()
            strategy_result = results[0]

            # Extract performance metrics
            sharpe_ratio = strategy_result.analyzers.sharpe.get_analysis().get(
                "sharperatio", 0
            )
            total_return = strategy_result.analyzers.returns.get_analysis().get(
                "rtot", 0
            )
            max_drawdown = (
                strategy_result.analyzers.drawdown.get_analysis()
                .get("max", {})
                .get("drawdown", 0)
            )

            # Get trade statistics
            trade_analysis = strategy_result.analyzers.trades.get_analysis()
            total_trades = trade_analysis.get("total", {}).get("total", 0)
            # win_rate = 0
            # if total_trades > 0:
            #     won_trades = trade_analysis.get('won', {}).get('total', 0)
            #     win_rate = won_trades / total_trades
            won_trades = trade_analysis.get("won", {}).get("total", 0) or 0
            if total_trades > 0:
                win_rate = won_trades / total_trades
            else:
                win_rate = 0
                return -999  # Add heavy penalty for no trades

            # Handle None values
            if sharpe_ratio is None:
                sharpe_ratio = 0
            if total_return is None:
                total_return = 0
            if max_drawdown is None:
                max_drawdown = 100

            # Enhanced optimization objective
            # Prioritize: Sharpe ratio, penalize high drawdown, reward consistent trading
            objective = (
                sharpe_ratio * 1.0  # Primary: Sharpe ratio
                + (total_return / 100) * 0.3  # Secondary: Total return
                + win_rate * 0.2  # Bonus: Win rate
                - (max_drawdown / 100) * 0.5  # Penalty: Drawdown
                - (1 / (total_trades + 1)) * 0.1  # Penalty: Too few trades
            )

            return objective

        except Exception as e:
            # Return poor score for failed optimizations
            print(f"Optimization trial failed: {str(e)}")
            return -999


def optimize_strategy(
    strategy_class,
    ticker="AAPL",
    start_date="2022-01-01",
    end_date="2025-06-01",
    n_trials=100,
    initial_cash=100000.0,
    commission=0.001,
    timeout=None,
):
    """
    Optimize strategy parameters using Optuna.

    Parameters:
    strategy_class: The strategy class to optimize
    ticker (str): Stock ticker symbol
    start_date (str): Start date for optimization
    end_date (str): End date for optimization
    n_trials (int): Number of optimization trials
    initial_cash (float): Initial cash amount
    commission (float): Commission rate
    timeout (int): Timeout in seconds (optional)

    Returns:
    dict: Best parameters and performance metrics
    """

    try:
        # Get data
        print(f"Loading data for {ticker}...")
        data_df = get_data(ticker, start_date, end_date)

        if data_df is None or data_df.empty:
            raise ValueError(f"No data available for {ticker}")

        data = bt.feeds.PandasData(dataname=data_df)

        # Create optimization objective
        objective = OptimizationObjective(
            strategy_class, data, initial_cash, commission
        )

        # Create study with better configuration
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42, n_startup_trials=min(20, n_trials // 5)),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        # Run optimization
        print(f"Starting optimization with {n_trials} trials...")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            n_jobs=1,  # Keep single-threaded for stability
        )

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        print(f"\nOptimization completed!")
        print(f"Best objective value: {best_value:.4f}")
        print(f"Best parameters: {best_params}")

        # Run final backtest with best parameters
        print("Running final validation with best parameters...")
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_class, **best_params)
        cerebro.adddata(data)
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)

        # Add comprehensive analyzers for final run
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.Calmar, _name="calmar")
        cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")

        results = cerebro.run()
        strategy_result = results[0]

        # Extract comprehensive performance metrics
        performance = {
            "sharpe_ratio": strategy_result.analyzers.sharpe.get_analysis().get(
                "sharperatio", 0
            ),
            "total_return": strategy_result.analyzers.returns.get_analysis().get(
                "rtot", 0
            ),
            "annual_return": strategy_result.analyzers.returns.get_analysis().get(
                "rnorm", 0
            ),
            "max_drawdown": strategy_result.analyzers.drawdown.get_analysis()
            .get("max", {})
            .get("drawdown", 0),
            "calmar_ratio": strategy_result.analyzers.calmar.get_analysis().get(
                "calmarratio", 0
            ),
            "sqn": strategy_result.analyzers.sqn.get_analysis().get("sqn", 0),
            "trades": strategy_result.analyzers.trades.get_analysis(),
            "final_value": cerebro.broker.getvalue(),
        }

        # Handle None values in performance metrics
        for key, value in performance.items():
            if value is None and key != "trades":
                performance[key] = 0

        return {
            "best_params": best_params,
            "best_value": best_value,
            "performance": performance,
            "study": study,
            "cerebro": cerebro,
            "n_trials_completed": len(study.trials),
        }

    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        raise


def analyze_optimization_results(study, save_plots=False):
    """
    Analyze and visualize optimization results.

    Parameters:
    study: Optuna study object
    save_plots (bool): Whether to save plots to files

    Returns:
    dict: Analysis results
    """

    try:
        # Get trials dataframe
        trials_df = study.trials_dataframe()

        if trials_df.empty:
            return {"error": "No completed trials found"}

        # Basic statistics
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

        # Parameter statistics
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

        # Parameter statistics
        print("\nParameter statistics:")
        for param, stats in param_stats.items():
            print(f"  {param}:")
            print(
                f"    Best: {stats['best']}, Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}"
            )
            print(f"    Range: [{stats['min']}, {stats['max']}]")

        # Try to generate plots if requested
        if save_plots:
            try:
                import matplotlib.pyplot as plt

                # Optimization history
                fig = optuna.visualization.matplotlib.plot_optimization_history(study)
                fig.savefig("optimization_history.png")
                print("Saved optimization_history.png")

                # Parameter importance
                fig = optuna.visualization.matplotlib.plot_param_importances(study)
                fig.savefig("parameter_importance.png")
                print("Saved parameter_importance.png")

                plt.close("all")

            except ImportError:
                print("Matplotlib not available for plotting")
            except Exception as e:
                print(f"Error generating plots: {str(e)}")

        return analysis

    except Exception as e:
        return {"error": f"Error analyzing results: {str(e)}"}


def quick_optimize(strategy_class, ticker="AAPL", n_trials=20):
    """Quick optimization for testing purposes."""

    return optimize_strategy(
        strategy_class=strategy_class,
        ticker=ticker,
        start_date="2023-01-01",
        end_date="2024-12-31",
        n_trials=n_trials,
    )


if __name__ == "__main__":
    # Example usage
    print("=== PARAMETER OPTIMIZATION DEMO ===")

    # Quick test
    quick_test = input("Run quick test? (y/n): ").lower().strip()

    if quick_test == "y":
        results = quick_optimize(EMARSI, "AAPL", 10)
    else:
        # Full optimization
        ticker = input("Enter ticker (default AAPL): ").strip() or "AAPL"
        n_trials = int(input("Enter number of trials (default 50): ").strip() or "50")

        results = optimize_strategy(
            strategy_class=EMARSI,
            ticker=ticker,
            start_date="2022-01-01",
            end_date="2025-06-01",
            n_trials=n_trials,
        )

    # Analyze results
    analysis = analyze_optimization_results(results["study"], save_plots=True)

    # Plot results if requested
    plot_choice = input("\nPlot backtest results? (y/n): ").lower().strip()
    if plot_choice == "y":
        results["cerebro"].plot(style="candlestick", barup="green", bardown="red")
