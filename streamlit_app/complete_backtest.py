"""
Complete backtest functionality combining all analysis types.
"""

import streamlit as st
import logging
from comprehensive_backtesting.backtesting import run_complete_backtest
from .data_processing import generate_strategy_report
from .visualization import display_best_strategies_report

logger = logging.getLogger(__name__)


def complete_backtest(data, progress_bar, params, ticker):
    """Run a full demonstration of backtest, optimization, and walk-forward analysis."""
    strategy_reports = []
    print("Starting complete backtest...", params)

    for idx, strategy in enumerate(params["selected_strategy"]):
        data_copy = data.copy()
        params_copy = params.copy()
        strategy_length = len(params["selected_strategy"])

        progress_bar.progress(int((idx / strategy_length) * 60))

        # Run complete backtest
        results = run_complete_backtest(
            data=data_copy,
            ticker=ticker,
            start_date=params_copy["start_date"],
            end_date=params_copy["end_date"],
            strategy_class=strategy,
            interval=params_copy["timeframe"],
            n_trials=params_copy["n_trials"],
        )

        if not results:
            st.error(f"Complete backtest failed for {strategy.__name__}")
            continue

        # Display results (existing visualizations)
        from .visualization import plot_composite_backtest_results
        from .table_generators import create_parameter_evolution_table

        display_composite_results(results, data_copy, ticker, params_copy["timeframe"])
        display_parameter_evolution(results, ticker)
        display_strategy_comparison(results, ticker)
        progress_bar.progress(int((idx / len(params["selected_strategy"])) * 70 + 10))
        display_complete_backtest_summary(results, ticker, params_copy["timeframe"])
        display_basic_results(results, data_copy, ticker)
        progress_bar.progress(int((idx / len(params["selected_strategy"])) * 80 + 10))
        display_optimized_results(results, data_copy, ticker, params_copy["timeframe"])
        display_walkforward_results(
            results, ticker, params_copy["timeframe"], params_copy, progress_bar
        )

        # Generate reports for all strategy types
        # 1. Basic strategy
        if "basic" in results:
            report = generate_strategy_report(
                results["basic"],
                f"Basic {strategy.__name__}",
                ticker,
                params_copy["timeframe"],
            )
            if report:
                strategy_reports.append(report)

        # 2. Optimized strategy
        if "optimization" in results and "results" in results["optimization"]:
            report = generate_strategy_report(
                results["optimization"]["results"],
                f"Optimized {strategy.__name__}",
                ticker,
                params_copy["timeframe"],
            )
            if report:
                strategy_reports.append(report)

        # 3. Walk-forward windows
        if "walk_forward" in results and "windows" in results["walk_forward"]:
            for i, window in enumerate(results["walk_forward"]["windows"]):
                if window.get("valid") and "out_sample_performance" in window:
                    report = generate_strategy_report(
                        window["out_sample_performance"],
                        f"WF {strategy.__name__} Window {i+1}",
                        ticker,
                        params_copy["timeframe"],
                    )
                    if report:
                        strategy_reports.append(report)

    # Update progress to 100% and show completion
    progress_bar.progress(100)
    st.toast("Complete backtest finished")

    # Display consolidated report for all strategies
    st.subheader("📊 Complete Backtest - Best Strategies Report")
    display_best_strategies_report(strategy_reports, ticker, params["timeframe"])


def display_composite_results(results, data, ticker, timeframe):
    """Display composite backtest results visualization."""
    st.subheader("Strategy Comparison" + " " + ticker)
    from .visualization import plot_composite_backtest_results

    composite_fig = plot_composite_backtest_results(results, data)
    if composite_fig:
        st.plotly_chart(
            composite_fig,
            use_container_width=True,
            key=f"composite_fig_{ticker}_{timeframe}",
        )
    else:
        st.warning("Could not generate composite strategy comparison")


def display_parameter_evolution(results, ticker):
    """Display parameter evolution table from walk-forward analysis."""
    if "walk_forward" in results:
        st.subheader("Walk-Forward Parameter Evolution" + " " + ticker)
        from .table_generators import create_parameter_evolution_table

        param_evolution_df = create_parameter_evolution_table(results["walk_forward"])
        if not param_evolution_df.empty:
            st.dataframe(param_evolution_df, use_container_width=True)
        else:
            st.info("No parameter evolution data available for walk-forward")


def display_strategy_comparison(results, ticker):
    """Display comparison between basic, optimized, and walk-forward strategies"""
    st.subheader("Strategy Comparison" + " " + ticker)
    # Implementation would go here - this is a complex function that would need
    # to be adapted from the original code
    st.info("Strategy comparison functionality to be implemented")


def display_complete_backtest_summary(results, ticker, timeframe):
    """Display enhanced summary for complete backtest with trade statistics and best times."""
    st.subheader("Complete Backtest Summary" + " " + ticker)
    # Implementation would go here - this is a complex function that would need
    # to be adapted from the original code
    st.info("Complete backtest summary functionality to be implemented")


def display_basic_results(results, data, ticker):
    """Display results from basic backtest."""
    st.subheader("Basic Backtest Results" + " " + ticker)
    # Implementation would go here - this is a complex function that would need
    # to be adapted from the original code
    st.info("Basic results display functionality to be implemented")


def display_optimized_results(results, data, ticker, timeframe):
    """Display results from optimized backtest."""
    st.subheader("Optimization Results" + " " + ticker)
    # Implementation would go here - this is a complex function that would need
    # to be adapted from the original code
    st.info("Optimized results display functionality to be implemented")


def display_walkforward_results(results, ticker, timeframe, params, progress_bar):
    """Display comprehensive results from walk-forward analysis."""
    st.subheader("Walk-Forward Analysis Results" + " " + ticker)
    # Implementation would go here - this is a complex function that would need
    # to be adapted from the original code
    st.info("Walk-forward results display functionality to be implemented")
