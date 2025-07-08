import pandas as pd

from comprehensive_backtesting.utils import run_backtest
from .parameter_optimization import optimize_strategy
from .reports import PerformanceAnalyzer, compare_strategies
from .validation import ValidationAnalyzer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_basic_backtest(strategy_class, ticker, start_date, end_date, interval):
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
        )
        print("\nOptimized Strategy Performance:")
        analyzer_optimized = PerformanceAnalyzer(optimization_results["results"])
        analyzer_optimized.print_report()
        return optimization_results
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Parameter optimization failed: {str(e)}")
        raise


def run_insample_outsample_analysis(
    strategy_name, ticker, start_date, end_date, interval, n_trials
):
    """Run in-sample/out-of-sample validation analysis."""
    logger.info(f"Running in-sample/out-of-sample analysis for {ticker}")
    print(f"\n=== IN-SAMPLE / OUT-OF-SAMPLE ANALYSIS: {ticker} ===")
    try:
        validation_analyzer = ValidationAnalyzer(
            strategy_name=strategy_name, ticker=ticker
        )
        results = validation_analyzer.in_sample_out_sample_analysis(
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            n_trials=n_trials,
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
    interval,
):
    """Run walk-forward analysis."""
    logger.info(f"Running walk-forward analysis for {ticker}")
    print(f"\n=== WALK-FORWARD ANALYSIS: {ticker} ===")
    try:
        validation_analyzer = ValidationAnalyzer(
            strategy_name=strategy_name, ticker=ticker
        )

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
        # Confirm trades extraction for each window if present
        if "windows" in results:
            for idx, window in enumerate(results["windows"]):
                # Prefer completed_trades if available, else fallback to trades
                completed_trades = None
                out_perf = window.get("out_sample_performance", {})
                if isinstance(out_perf, dict):
                    completed_trades = out_perf.get("completed_trades")
                    trades = (
                        completed_trades
                        if completed_trades is not None
                        else out_perf.get("trades", [])
                    )
                else:
                    trades = []
                logger.info(
                    f"[Walk-Forward] Window {idx+1}: {len(trades)} trades extracted."
                )
                # Print trade details for UI (or you can format as needed)
                if trades:
                    print(f"\nWindow {idx+1} Completed Trades:")
                    print("-" * 120)
                    print(
                        f"{'Ref':>4} {'Entry Time':>20} {'Exit Time':>20} {'Entry Px':>10} {'Exit Px':>10} {'Size':>5} {'PnL':>10} {'PnL Net':>10} {'Comm':>8} {'Status':>8} {'Dir':>6} {'Bars':>5}"
                    )
                    print("-" * 120)
                    for t in trades:
                        print(
                            f"{t.get('ref', '-')!s:>4} {str(t.get('entry_time', '-')):>20} {str(t.get('exit_time', '-')):>20} "
                            f"{t.get('entry_price', '-'):>10.2f} {t.get('exit_price', '-'):>10.2f} {t.get('size', '-'):>5} "
                            f"{t.get('pnl', '-'):>10.2f} {t.get('pnl_net', '-'):>10.2f} {t.get('commission', '-'):>8.2f} "
                            f"{t.get('status', '-'):>8} {t.get('direction', '-'):>6} {t.get('bars_held', '-'):>5}"
                        )
                    print("-" * 120)
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
            strategy_class=strategy_name,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            n_trials=n_trials,
            interval=interval,
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

    comparison_df = compare_strategies(results_comparison)
    print(comparison_df.round(3))
    return results_comparison


def run_complete_backtest(
    ticker,
    start_date,
    end_date,
    window_days,
    out_days,
    step_days,
    strategy_class,
    analyzers,
    interval,
    n_trials,
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
            strategy_class=strategy_class,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            n_trials=n_trials,
            interval=interval,
        )
        results["optimization"] = opt_results
    except Exception as e:
        logger.error(f"Parameter optimization failed: {str(e)}")

    print("\n" + "=" * 60)
    try:
        validation_results = run_insample_outsample_analysis(
            strategy_class,
            ticker,
            start_date,
            end_date,
            interval=interval,
            n_trials=n_trials,
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

        wf_results = validation_analyzer.walk_forward_analysis(
            start_date=start_date,
            end_date=end_date,
            in_sample_days=window_days,
            out_sample_days=out_days,
            step_days=step_days,
            n_trials=n_trials,
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
