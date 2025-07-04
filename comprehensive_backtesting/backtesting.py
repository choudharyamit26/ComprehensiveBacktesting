import datetime
import asyncio
import backtrader as bt
import pandas as pd

from comprehensive_backtesting.utils import run_backtest
from .data import get_data_sync, validate_data, preview_data_sync
from .registry import get_strategy
from .parameter_optimization import optimize_strategy
from .reports import PerformanceAnalyzer, compare_strategies
from .validation import ValidationAnalyzer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_basic_backtest(
    strategy_class, ticker, start_date, end_date, interval, analyzers=None
):
    """Run a basic backtest with default parameters."""
    logger.info(f"Running basic backtest for {ticker}")
    print(f"\n=== BASIC BACKTEST: {ticker} ===")
    try:
        results, cerebro = run_backtest(
            strategy_class=strategy_class,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            analyzer=analyzers,
        )
        analyzer = PerformanceAnalyzer(results)
        analyzer.print_report()
        return results, cerebro
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Basic backtest failed: {str(e)}")
        raise


def run_parameter_optimization(
    strategy_class,
    ticker,
    start_date,
    end_date,
    interval,
    n_trials,
    analyzers=None,
):
    """Run parameter optimization analysis."""
    logger.info(f"Running parameter optimization for {ticker}")
    print(f"\n=== PARAMETER OPTIMIZATION: {ticker} ===")
    print(f"Running optimization with {n_trials} trials...")
    try:
        optimization_results = optimize_strategy(
            strategy_class=strategy_class,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            n_trials=n_trials,
            interval=interval,
            analyzers=analyzers,
        )
        print("\nOptimized Strategy Performance:")
        analyzer_optimized = PerformanceAnalyzer(optimization_results["results"])
        analyzer_optimized.print_report()
        return optimization_results
    except Exception as e:
        logger.error(f"Parameter optimization failed: {str(e)}")
        raise


def run_insample_outsample_analysis(
    strategy_name, ticker, start_date, end_date, interval
):
    """Run in-sample/out-of-sample validation analysis."""
    logger.info(f"Running in-sample/out-of-sample analysis for {ticker}")
    print(f"\n=== IN-SAMPLE / OUT-OF-SAMPLE ANALYSIS: {ticker} ===")
    try:
        validation_analyzer = ValidationAnalyzer(
            strategy_name=strategy_name, ticker=ticker
        )
        results = validation_analyzer.in_sample_out_sample_analysis(
            start_date=start_date, end_date=end_date, interval=interval
        )
        print("\nIn-Sample Performance:")
        print("-" * 60)
        summary = results["in_sample_performance"].get("summary", {})
        print(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
        print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0)}")
        print(f"Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
        print("\nOut-of-Sample Performance:")
        print("-" * 60)
        summary = results["out_sample_performance"].get("summary", {})
        print(f"Total Return: {summary.get('total_return_pct', 0)}%")
        print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0)}")
        print(f"Max Drawdown: {summary.get('max_drawdown_pct', 0)}%")
        print("\nValidation Summary:")
        degradation = results.get("performance_degradation", {})
        print(f"Return degradation: {degradation.get('return_degradation', 0)}%")
        print(f"Sharpe degradation: {degradation.get('sharpe_degradation', 0)}")
        return results
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"In-sample/out-of-sample analysis failed: {str(e)}")
        return run_basic_comparison_analysis(
            strategy_name, ticker, start_date, end_date, interval=interval
        )


def run_walkforward_analysis(
    ticker,
    start_date,
    end_date,
    window_days,
    out_days,
    step_days,
    n_trials,
    min_trades,
    strategy_name,
    interval,  # Default to 5-minute interval for Indian equities
):
    """Run walk-forward analysis."""
    logger.info(f"Running walk-forward analysis for {ticker}")
    print(f"\n=== WALK-FORWARD ANALYSIS: {ticker} ===")
    try:
        validation_analyzer = ValidationAnalyzer(
            strategy_name=strategy_name, ticker=ticker
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
            interval=interval,
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
        import traceback

        traceback.print_exc()
        logger.error(f"Walk-forward analysis failed: {str(e)}")
        return run_parameter_optimization(
            strategy_name, ticker, start_date, end_date, n_trials=10, interval=interval
        )


def run_comprehensive_validation(strategy_class, ticker, start_date, end_date):
    """Run comprehensive validation analysis."""
    logger.info(f"Running comprehensive validation for {ticker}")
    print(f"\n=== COMPREHENSIVE VALIDATION ANALYSIS: {ticker} ===")
    try:
        results = ValidationAnalyzer.run_validation_analysis(
            strategy_name=strategy_class,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        print("\nComprehensive Validation Complete!")
        return results
    except Exception as e:
        logger.error(f"Comprehensive validation failed: {str(e)}")
        return run_basic_backtest(strategy_class, ticker, start_date, end_date)


def run_basic_comparison_analysis(
    strategy_name, ticker, start_date, end_date, interval
):
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
                strategy_class=strategy_name,  # Use get_strategy
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                **params,
            )
            results_comparison[name] = results
        except Exception as e:
            logger.error(f"Failed to run {name} strategy: {str(e)}")
            continue

    # if results_comparison:
    #     try:
    comparison_df = compare_strategies(results_comparison)
    # if not comparison_df.empty:
    # print("\nStrategy Comparison:")
    print(comparison_df.round(3))
    # except Exception as e:
    #     logger.error(f"Strategy comparison failed: {str(e)}")
    return results_comparison


def run_complete_backtest(
    ticker, start_date, end_date, strategy_class, analyzers, interval
):
    """Run a complete demonstration of all analyses."""
    logger.info(f"Running complete backtest for {ticker}")
    print(f"\n=== COMPLETE BACKTEST: {ticker} ===")
    results = {}

    print("\n" + "=" * 60)
    try:
        basic_results, basic_cerebro = run_basic_backtest(
            strategy_class,
            ticker,
            start_date,
            end_date,
            interval=interval,
            analyzers=analyzers,
        )
        results["basic"] = basic_results
        results["basic_cerebro"] = basic_cerebro
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Basic backtest failed: {str(e)}")
        return results

    print("\n" + "=" * 60)
    try:
        opt_results = run_parameter_optimization(
            strategy_class,
            ticker,
            start_date,
            end_date,
            n_trials=10,
            interval=interval,
            analyzers=analyzers,
        )
        results["optimization"] = opt_results
    except Exception as e:
        logger.error(f"Parameter optimization failed: {str(e)}")

    print("\n" + "=" * 60)
    try:
        validation_results = run_insample_outsample_analysis(
            strategy_class, ticker, start_date, end_date, interval=interval
        )
        results["validation"] = validation_results
    except Exception as e:
        logger.error(f"Validation analysis failed: {str(e)}")

    print("\n" + "=" * 60)
    try:
        comparison_results = run_basic_comparison_analysis(
            strategy_class, ticker, start_date, end_date, interval=interval
        )
        results["comparison"] = comparison_results
    except Exception as e:
        logger.error(f"Strategy comparison failed: {str(e)}")

    print("\n" + "=" * 60)
    try:
        validation_analyzer = ValidationAnalyzer(
            strategy_name=strategy_class, ticker=ticker
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
            interval=interval,
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

    return results
