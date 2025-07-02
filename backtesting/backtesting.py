import datetime
import asyncio
import backtrader as bt
import pandas as pd

from backtesting.utils import run_backtest
from .data import get_data_sync, validate_data, preview_data_sync
from strategies.registry import get_strategy
from .parameter_optimization import optimize_strategy
from .reports import PerformanceAnalyzer, compare_strategies
from .validation import ValidationAnalyzer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_basic_backtest(ticker, start_date, end_date):
    """Run a basic backtest with default parameters."""
    logger.info(f"Running basic backtest for {ticker}")
    print(f"\n=== BASIC BACKTEST: {ticker} ===")
    try:
        results, cerebro = run_backtest(
            strategy_class="EMARSI",
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        analyzer = PerformanceAnalyzer(results)
        analyzer.print_report()
        return results, cerebro
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Basic backtest failed: {str(e)}")
        raise


def run_parameter_optimization(ticker, start_date, end_date, n_trials=50):
    """Run parameter optimization analysis."""
    logger.info(f"Running parameter optimization for {ticker}")
    print(f"\n=== PARAMETER OPTIMIZATION: {ticker} ===")
    print(f"Running optimization with {n_trials} trials...")
    try:
        optimization_results = optimize_strategy(
            strategy_class=get_strategy("EMARSI"),
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            n_trials=n_trials,
        )
        print("\nOptimized Strategy Performance:")
        analyzer_optimized = PerformanceAnalyzer(optimization_results["results"])
        analyzer_optimized.print_report()
        return optimization_results
    except Exception as e:
        logger.error(f"Parameter optimization failed: {str(e)}")
        raise


def run_insample_outsample_analysis(ticker, start_date, end_date):
    """Run in-sample/out-of-sample validation analysis."""
    logger.info(f"Running in-sample/out-of-sample analysis for {ticker}")
    print(f"\n=== IN-SAMPLE / OUT-OF-SAMPLE ANALYSIS: {ticker} ===")
    try:
        validation_analyzer = ValidationAnalyzer(
            strategy_class=get_strategy("EMARSI"), ticker=ticker
        )
        results = validation_analyzer.in_sample_out_sample_analysis(
            start_date=start_date, end_date=end_date
        )
        print("\nIn-Sample Performance:")
        print("-" * 60)
        summary = results["in_sample_performance"].get("summary", {})
        print(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
        print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
        print("\nOut-of-Sample Performance:")
        print("-" * 60)
        summary = results["out_sample_performance"].get("summary", {})
        print(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
        print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
        print("\nValidation Summary:")
        degradation = results.get("performance_degradation", {})
        print(f"Return degradation: {degradation.get('return_degradation', 0):.2f}%")
        print(f"Sharpe degradation: {degradation.get('sharpe_degradation', 0):.3f}")
        return results
    except Exception as e:
        logger.error(f"In-sample/out-of-sample analysis failed: {str(e)}")
        return run_basic_comparison_analysis(ticker, start_date, end_date)


def run_walkforward_analysis(
    ticker, start_date, end_date, window_days=None, out_days=None, step_days=None, n_trials=20, min_trades=1
):
    """Run walk-forward analysis."""
    logger.info(f"Running walk-forward analysis for {ticker}")
    print(f"\n=== WALK-FORWARD ANALYSIS: {ticker} ===")
    try:
        validation_analyzer = ValidationAnalyzer(
            strategy_class=get_strategy("EMARSI"), ticker=ticker
        )

        # Calculate maximum available days (60 days for intraday)
        max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        max_days = min(max_days, 60)  # Enforce 60-day limit

        # Set defaults based on available data
        default_window = min(30, max_days // 2)
        default_out = min(15, max_days // 4)
        default_step = min(15, max_days // 4)

        # Use provided values or defaults
        window_days = window_days if window_days is not None else default_window
        out_days = out_days if out_days is not None else default_out
        step_days = step_days if step_days is not None else default_step

        results = validation_analyzer.walk_forward_analysis(
            start_date=start_date,
            end_date=end_date,
            in_sample_days=window_days,
            out_sample_days=out_days,
            step_days=step_days,
            n_trials=n_trials,
            min_trades=min_trades,
        )
        summary = results.get("summary_stats", {})
        print(f"\nWalk-Forward Analysis Summary:")
        print(
            f"Valid windows: {summary.get('valid_windows', 0)}/{summary.get('total_windows', 0)}"
        )
        print(
            f"Average in-sample return: {summary.get('avg_in_sample_return', 0):.2f}%"
        )
        print(
            f"Average out-of-sample return: {summary.get('avg_out_sample_return', 0):.2f}%"
        )
        print(f"Out-of-sample win rate: {summary.get('win_rate_out_sample', 0):.1f}%")
        print(f"Return correlation: {summary.get('correlation', 0):.3f}")
        print(f"Average degradation: {summary.get('avg_degradation', 0):.2f}%")

        validation_analyzer.plot_walk_forward_results(results)
        return results
    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {str(e)}")
        return run_parameter_optimization(ticker, start_date, end_date)


def run_comprehensive_validation(ticker, start_date, end_date):
    """Run comprehensive validation analysis."""
    logger.info(f"Running comprehensive validation for {ticker}")
    print(f"\n=== COMPREHENSIVE VALIDATION ANALYSIS: {ticker} ===")
    try:
        results = ValidationAnalyzer.run_validation_analysis(
            strategy_class=get_strategy("EMARSI"),
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        print("\nComprehensive Validation Complete!")
        return results
    except Exception as e:
        logger.error(f"Comprehensive validation failed: {str(e)}")
        return run_basic_backtest(ticker, start_date, end_date)


def run_basic_comparison_analysis(ticker, start_date, end_date):
    """Run basic strategy comparison."""
    logger.info(f"Running strategy comparison for {ticker}")
    print(f"\n=== STRATEGY COMPARISON ANALYSIS: {ticker} ===")
    parameter_sets = {
        "Conservative": {
            "slow_ema_period": 21,
            "fast_ema_period": 50,
            "rsi_period": 14,
            "rsi_upper": 70,
            "rsi_lower": 30,
        },
        "Aggressive": {
            "slow_ema_period": 8,
            "fast_ema_period": 21,
            "rsi_period": 10,
            "rsi_upper": 80,
            "rsi_lower": 20,
        },
        "Balanced": {
            "slow_ema_period": 12,
            "fast_ema_period": 26,
            "rsi_period": 14,
            "rsi_upper": 75,
            "rsi_lower": 25,
        },
    }

    results_comparison = {}
    for name, params in parameter_sets.items():
        print(f"\nRunning {name} strategy...")
        try:
            results, _ = run_backtest(
                strategy_class=get_strategy("EMARSI"),  # Use get_strategy
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                **params,
            )
            results_comparison[name] = results
        except Exception as e:
            logger.error(f"Failed to run {name} strategy: {str(e)}")
            continue

    if results_comparison:
        try:
            comparison_df = compare_strategies(results_comparison)
            if not comparison_df.empty:
                print("\nStrategy Comparison:")
                print(comparison_df.round(3))
        except Exception as e:
            logger.error(f"Strategy comparison failed: {str(e)}")
    return results_comparison


def run_complete_backtest(ticker, start_date, end_date):
    """Run a complete demonstration of all analyses."""
    logger.info(f"Running full demo for {ticker}")
    print(f"\n=== FULL DEMO: {ticker} ===")
    results = {}

    print("\n" + "=" * 60)
    try:
        basic_results, basic_cerebro = run_basic_backtest(ticker, start_date, end_date)
        results["basic"] = basic_results
        results["basic_cerebro"] = basic_cerebro
    except Exception as e:
        logger.error(f"Basic backtest failed: {str(e)}")
        return results

    print("\n" + "=" * 60)
    try:
        opt_results = run_parameter_optimization(ticker, start_date, end_date)
        results["optimization"] = opt_results
    except Exception as e:
        logger.error(f"Parameter optimization failed: {str(e)}")

    print("\n" + "=" * 60)
    try:
        validation_results = run_insample_outsample_analysis(
            ticker, start_date, end_date
        )
        results["validation"] = validation_results
    except Exception as e:
        logger.error(f"Validation analysis failed: {str(e)}")

    print("\n" + "=" * 60)
    try:
        comparison_results = run_basic_comparison_analysis(ticker, start_date, end_date)
        results["comparison"] = comparison_results
    except Exception as e:
        logger.error(f"Strategy comparison failed: {str(e)}")

    print("\n" + "=" * 60)
    try:
        validation_analyzer = ValidationAnalyzer(
            strategy_class=get_strategy("EMARSI"), ticker=ticker
        )

        # Calculate maximum available days
        max_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        max_days = min(max_days, 60)

        # Set defaults based on available data
        in_sample = min(30, max_days // 2)
        out_sample = min(15, max_days // 4)
        step = min(15, max_days // 4)

        wf_results = validation_analyzer.walk_forward_analysis(
            start_date=start_date,
            end_date=end_date,
            in_sample_days=in_sample,
            out_sample_days=out_sample,
            step_days=step,
            n_trials=20,
            min_trades=1,
        )
        results["walk_forward"] = wf_results
    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {str(e)}")

    print("\n" + "=" * 60)
    print("Generating comprehensive reports...")
    try:
        basic_analyzer = PerformanceAnalyzer(basic_results)
        basic_analyzer.save_report_to_file(f"{ticker}_basic_report.json")
        print(f"Basic report saved: {ticker}_basic_report.json")
        if (
            "optimization" in results
            and results["optimization"]
            and "cerebro" in results["optimization"]
        ):
            opt_analyzer = PerformanceAnalyzer(results["optimization"]["results"])
            opt_analyzer.save_report_to_file(f"{ticker}_optimized_report.json")
            print(f"Optimized report saved: {ticker}_optimized_report.json")
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")

    try:

        basic_cerebro.plot(style="candlestick", barup="green", bardown="red")
        if (
            "optimization" in results
            and results["optimization"]
            and "cerebro" in results["optimization"]
        ):
            print("Plotting optimized strategy results...")
            results["optimization"]["cerebro"].plot(
                style="candlestick", barup="green", bardown="red"
            )
    except Exception as e:
        logger.error(f"Plotting failed: {str(e)}")

    return results


def main():
    """Main function for interactive backtesting framework."""
    logger.info("Starting Enhanced Backtesting Framework")
    print("=== ENHANCED BACKTESTING FRAMEWORK ===\n")
    print("Select analysis type:")
    print("1. Basic backtest")
    print("2. Parameter optimization")
    print("3. In-sample / Out-of-sample analysis")
    print("4. Walk-forward analysis")
    print("5. Comprehensive validation analysis")
    print("6. Full demo (all analyses)")

    choice = input("\nEnter your choice (1-6): ").strip()
    import datetime

    ticker = input("Enter ticker symbol (default: SBIN.NS): ").strip() or "SBIN.NS"
    interval = "5m"  # Default to intraday for Indian equities
    end_date = datetime.date.today() - datetime.timedelta(days=2)
    start_date = end_date - datetime.timedelta(days=58)
    print(f"Using date range: {start_date} to {end_date} (last 60 days for intraday)")

    try:
        if choice == "1":
            run_basic_backtest(ticker, start_date, end_date)
        elif choice == "2":
            run_parameter_optimization(ticker, start_date, end_date)
        elif choice == "3":
            run_insample_outsample_analysis(ticker, start_date, end_date)
        elif choice == "4":
            run_walkforward_analysis(ticker, start_date, end_date)
        elif choice == "5":
            run_comprehensive_validation(ticker, start_date, end_date)
        elif choice == "6":
            run_complete_backtest(ticker, start_date, end_date)
        else:
            print("Invalid choice. Running basic backtest...")
            run_basic_backtest(ticker, start_date, end_date)
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        logger.error(
            f"Error during analysis: {str(e)}. Check data, strategy, or dependencies."
        )
        print(f"\nError during analysis: {str(e)}")
        print("Trying basic fallback...")
        try:
            test_data = get_data_sync(ticker, start_date, end_date, interval="5m")
            if validate_data(test_data):
                print("Data is valid. The issue might be with the strategy module.")
            else:
                print("Data validation failed.")
        except Exception as data_error:
            logger.error(f"Data fetch failed: {str(data_error)}")
            print(f"Data fetch also failed: {str(data_error)}")

    print("\n=== ANALYSIS COMPLETE ===")


def quick_test():
    """Quick test function for rapid development."""
    logger.info("Running quick test")
    print("=== QUICK TEST ===")
    ticker = input("Enter ticker (default: AAPL): ").strip() or "AAPL"

    end_date = datetime.date.today() - datetime.timedelta(days=2)
    start_date = end_date - datetime.timedelta(days=58)
    try:
        results, cerebro = run_backtest(
            strategy_class="EMARSI",
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        analyzer = PerformanceAnalyzer(results)
        analyzer.print_report()

        cerebro.plot(style="candlestick", barup="green", bardown="red")
    except Exception as e:
        logger.error(f"Quick test failed: {str(e)}")
        try:
            print("Testing data loading...")
            end_date = datetime.date.today() - datetime.timedelta(days=2)
            start_date = end_date - datetime.timedelta(days=58)
            df = preview_data_sync(ticker, start_date, end_date, rows=3)
            if df is not None:
                is_valid = validate_data(df)
                print(f"Data validation: {'PASSED' if is_valid else 'FAILED'}")
        except Exception as data_error:
            logger.error(f"Data test failed: {str(data_error)}")
            print(f"Data test also failed: {str(data_error)}")


if __name__ == "__main__":
    print("Welcome to the Enhanced Backtesting Framework!")
    mode_choice = input(
        "\nChoose mode:\n1. Full interactive menu\n2. Quick test\n3. Basic example\nEnter choice (1-3): "
    ).strip()

    try:
        if mode_choice == "1":
            main()
        elif mode_choice == "2":
            quick_test()
        elif mode_choice == "3":
            logger.info("Running basic example")
            print("Running basic example...")
            cerebro = bt.Cerebro()
            end_date = datetime.date.today() - datetime.timedelta(days=2)
            start_date = end_date - datetime.timedelta(days=58)
            data_df = get_data_sync("AAPL", start_date, end_date, interval="5m")
            data_5m = bt.feeds.PandasData(
                dataname=data_df,
                datetime=None,
                open="Open",
                high="High",
                low="Low",
                close="Close",
                volume="Volume",
                openinterest=None,
            )
            cerebro.addstrategy(get_strategy("EMARSI"))
            cerebro.broker.setcash(100000.0)
            cerebro.broker.setcommission(commission=0.001)
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            cerebro.addanalyzer(bt.analyzers.Calmar, _name="calmar")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
            cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")
            cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
            cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annualreturn")
            cerebro.addanalyzer(bt.analyzers.Transactions, _name="transactions")
            cerebro.addanalyzer(bt.analyzers.PositionsValue, _name="positionsvalue")
            # Add 5-minute data
            cerebro.adddata(data_5m, name="5m")
            # Add 15-minute resampled data
            cerebro.resampledata(
                data_5m, timeframe=bt.TimeFrame.Minutes, compression=15, name="15m"
            )
            print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
            results = cerebro.run()
            print(f"Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
            analyzer = PerformanceAnalyzer(results)
            analyzer.print_report()
            plot_choice = input("\nPlot results? (y/n): ").lower().strip()
            cerebro.plot(style="candlestick", barup="green", bardown="red")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\nUnexpected error: {str(e)}")
        print("Please check your dependencies and try again.")
    print("\nThank you for using the Enhanced Backtesting Framework!")
