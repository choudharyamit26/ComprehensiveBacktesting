"""
Analysis runner functions for different types of backtesting analysis.
"""

import pandas as pd
import streamlit as st
import logging
import uuid
from comprehensive_backtesting.data import get_data_sync
from comprehensive_backtesting.backtesting import (
    run_basic_backtest,
    run_parameter_optimization,
)
from comprehensive_backtesting.reports import PerformanceAnalyzer
from comprehensive_backtesting.walk_forward_analysis import WalkForwardAnalysis
from comprehensive_backtesting.registry import get_strategy
from .data_processing import generate_strategy_report, analyze_best_time_ranges
from .table_generators import (
    create_summary_table,
    create_parameters_table,
    create_trades_table,
    create_best_times_table,
)
from .visualization import (
    plot_contour,
    plot_time_analysis,
    display_best_strategies_report,
)
from .indicators import detect_strategy_indicators, create_dynamic_indicators_table
from .optimization import display_best_parameters
from .utils import get_strategy
import ast

logger = logging.getLogger(__name__)


def validate_data(data, ticker):
    """Validate fetched data for analysis."""
    if data.empty:
        st.error("No data available for the selected ticker and date range.")
        return False

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in data.columns]
        st.error(f"Missing required columns: {', '.join(missing)}")
        return False

    return True


def run_backtest_analysis(
    params, data, analyzer_config, progress_bar, status_text, ticker
):
    """Run backtest analysis and display results."""
    status_text.text("Running backtest...")
    data = data.copy()  # Ensure we don't modify the original data
    start_date = params["start_date"].strftime("%Y-%m-%d")
    end_date = params["end_date"].strftime("%Y-%m-%d")
    interval = params["timeframe"]
    strategies_length = len(params["selected_strategy"])

    # Collect all strategy reports
    strategy_reports = []

    for idx, strategy in enumerate(params["selected_strategy"]):
        progress_bar.progress(int((idx / strategies_length) * 100))

        results, cerebro = run_basic_backtest(
            data=data,
            strategy_class=strategy,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

        # Initialize PerformanceAnalyzer with results
        analyzer = PerformanceAnalyzer(results[0])
        report = analyzer.generate_full_report()

        # Generate strategy report
        strategy_report = generate_strategy_report(results, strategy, ticker, interval)
        if strategy_report:
            strategy_reports.append(strategy_report)

        # Display individual strategy results
        st.write(
            f"### Backtest Results Summary for {ticker} Using Strategy: {strategy}"
        )
        summary_table = create_summary_table(report)
        st.table(summary_table)

        # Dynamic Technical Indicators Table
        st.write("### 📊 Technical Indicators - Latest Values")
        strategy_instance = get_strategy(results)
        indicators_df = create_dynamic_indicators_table(data, strategy_instance)
        if not indicators_df.empty:
            st.dataframe(indicators_df, use_container_width=True)

            # Show detected indicators summary
            detected_indicators = detect_strategy_indicators(strategy_instance)
            if detected_indicators:
                st.write("**🔍 Detected Strategy Indicators:**")
                indicator_summary = []
                for name, info in detected_indicators.items():
                    params_dict = info.get("params", {})
                    user_params = {
                        k: v
                        for k, v in params_dict.items()
                        if not (
                            k.startswith("_")
                            or callable(v)
                            or isinstance(v, type)
                            or "method" in str(type(v)).lower()
                        )
                    }
                    params_str = (
                        ", ".join([f"{k} = {v}" for k, v in user_params.items()])
                        if user_params
                        else "No parameters"
                    )
                    indicator_summary.append(
                        f"- **{info['type']}** (`{name}`): {params_str}"
                    )
                st.markdown("\n".join(indicator_summary))
            else:
                st.info("No technical indicators detected in this strategy.")
        else:
            st.info("No indicator data available")

        # Comprehensive Trades Table
        st.write("### 📊 Detailed Trades Table")
        try:
            trades_df, trades_error = create_trades_table(results, data)
            if trades_error:
                st.warning(f"Could not create trades table: {trades_error}")
            elif not trades_df.empty:

                def highlight_columns(col):
                    if "Entry" in col:
                        return ["background-color: #e8f5e9"] * len(col)
                    elif "Exit" in col:
                        return ["background-color: #ffebee"] * len(col)
                    return [""] * len(col)

                styled_trades = trades_df.style.apply(highlight_columns, axis=0)
                st.dataframe(styled_trades, use_container_width=True)
            else:
                st.info("No trades executed during the backtest period.")
        except Exception as e:
            st.error(f"Error creating trades table: {str(e)}")
            logger.exception("Error creating trades table")

        # Trade Statistics Summary
        st.write("### 📈 Trade Statistics Summary")
        from .data_processing import analyze_best_trades

        best_trades_analysis = analyze_best_trades(results)
        if "error" not in best_trades_analysis:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trades", best_trades_analysis["total_trades"])
                st.metric("Winning Trades", best_trades_analysis["winning_trades"])
            with col2:
                win_rate = (
                    (
                        best_trades_analysis["winning_trades"]
                        / best_trades_analysis["total_trades"]
                        * 100
                    )
                    if best_trades_analysis["total_trades"] > 0
                    else 0
                )
                st.metric("Win Rate", f"{win_rate:.1f}%")
                st.metric("Losing Trades", best_trades_analysis["losing_trades"])
            with col3:
                st.metric("Total P&L", f"{best_trades_analysis['total_pnl']:.2f}")
                st.metric("Best Trade", f"{best_trades_analysis['best_trade_pnl']:.2f}")
            with col4:
                st.metric(
                    "Avg Winning Trade",
                    f"{best_trades_analysis['avg_winning_trade']:.2f}",
                )
                st.metric(
                    "Avg Losing Trade",
                    f"{best_trades_analysis['avg_losing_trade']:.2f}",
                )

        # Best Trading Times Analysis
        st.write("### ⏰ Best Trading Times Analysis")
        time_analysis = analyze_best_time_ranges(results)
        if "error" not in time_analysis:
            hours_df, days_df, months_df = create_best_times_table(time_analysis)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**🕐 Best Hours to Trade**")
                if hours_df is not None and not hours_df.empty:
                    st.dataframe(hours_df, use_container_width=True)
                else:
                    st.info("No hourly data available")
            with col2:
                st.write("**📅 Best Days to Trade**")
                if days_df is not None and not days_df.empty:
                    st.dataframe(days_df, use_container_width=True)
                else:
                    st.info("No daily data available")
            with col3:
                st.write("**📆 Best Months to Trade**")
                if months_df is not None and not months_df.empty:
                    st.dataframe(months_df, use_container_width=True)
                else:
                    st.info("No monthly data available")

            st.write("### 📊 Trading Time Analysis")
            time_chart = plot_time_analysis(time_analysis)
            if time_chart:
                st.plotly_chart(time_chart, use_container_width=True, key=uuid.uuid4())
        else:
            st.warning(
                f"Could not analyze best times: {time_analysis.get('error', 'Unknown error')}"
            )

    # Update progress to 100% and show completion
    progress_bar.progress(100)
    status_text.text("Backtest complete!")
    st.toast("Backtesting complete")

    # Display best strategies report after all strategies are processed
    display_best_strategies_report(strategy_reports, ticker, interval)
    return strategy_reports


def run_optimization_analysis(
    params, data, analyzer_config, progress_bar, status_text, ticker
):
    """Run optimization analysis and display results."""
    status_text.text("Starting optimization...")
    data = data.copy()  # Ensure we don't modify the original data
    start_date = params["start_date"].strftime("%Y-%m-%d")
    end_date = params["end_date"].strftime("%Y-%m-%d")
    interval = params["timeframe"]
    n_trials = params["n_trials"]
    strategy_reports = []
    strategy_length = len(params["selected_strategy"])

    for idx, strategy in enumerate(params["selected_strategy"]):
        progress_bar.progress(int((idx / strategy_length) * 100))

        results = run_parameter_optimization(
            data=data,
            strategy_class=strategy,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            n_trials=n_trials,
            interval=interval,
        )

        # Check for errors in optimization results
        if results.get("results") is None:
            error_msg = results.get("error", "Unknown error during optimization.")
            st.error(f"Optimization failed: {error_msg}")
            continue

        # Initialize PerformanceAnalyzer with results
        analyzer = PerformanceAnalyzer(results["results"][0])
        report = analyzer.generate_full_report()

        # Generate strategy report
        strategy_report = generate_strategy_report(
            results["results"], strategy, ticker, interval
        )
        if strategy_report:
            strategy_reports.append(strategy_report)

        # Display report as table instead of JSON
        st.write(
            f"### Optimization Results Summary for {ticker} Using Strategy: {strategy}"
        )
        summary_table = create_summary_table(report)
        st.table(summary_table)

        # Optimization Contour Plot
        st.write("### 🗺️ Parameter Optimization Landscape")
        contour_fig = plot_contour(results["study"])
        if contour_fig:
            st.plotly_chart(contour_fig, use_container_width=True, key=uuid.uuid4())
        else:
            st.warning("Could not generate contour plot")

        # Dynamic Technical Indicators for Optimized Strategy
        st.write("### 📊 Optimized Strategy - Technical Indicators")
        opt_strategy = get_strategy(results["results"])
        opt_indicators_df = create_dynamic_indicators_table(data, opt_strategy)
        if not opt_indicators_df.empty:
            st.dataframe(opt_indicators_df, use_container_width=True)

            # Show optimized parameters
            opt_detected_indicators = detect_strategy_indicators(opt_strategy)
            if opt_detected_indicators:
                st.write("**🎯 Optimized Strategy Indicators:**")
                opt_indicator_summary = []
                for name, info in opt_detected_indicators.items():
                    params_dict = info.get("params", {})
                    user_params = {
                        k: v
                        for k, v in params_dict.items()
                        if not (
                            k.startswith("_")
                            or callable(v)
                            or isinstance(v, type)
                            or "method" in str(type(v)).lower()
                        )
                    }
                    if user_params:
                        params_str = ", ".join(
                            [f"{k}={v}" for k, v in user_params.items()]
                        )
                    else:
                        params_str = "No parameters"
                    opt_indicator_summary.append(
                        f"- **{info['type']}** (`{name}`): {params_str}"
                    )
                st.markdown("\n".join(opt_indicator_summary))
        else:
            st.info("No technical indicators detected in optimized strategy.")

        # Best Parameters Analysis
        st.write("### 🎯 Best Parameters Analysis")
        best_params_info = display_best_parameters(results)
        if "error" not in best_params_info:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**📊 Optimization Summary**")
                st.metric(
                    "Best Objective Value",
                    (
                        f"{best_params_info['best_objective_value']:.4f}"
                        if best_params_info["best_objective_value"]
                        else 0
                    ),
                )
                st.metric("Success Rate", f"{best_params_info['success_rate']:.1f}%")
                st.metric(
                    "Completed Trials",
                    f"{best_params_info['completed_trials']}/{best_params_info['total_trials']}",
                )

            with col2:
                st.write("**⚙️ Parameter Details**")
                params_df = create_parameters_table(best_params_info)
                if not params_df.empty:
                    st.dataframe(params_df, use_container_width=True)
                else:
                    st.info("No parameter data available")
        else:
            logger.info("No best parameters found", best_params_info)
            st.warning(
                "Could not display best parameters: "
                + best_params_info.get("error", "Unknown error")
            )

        # Comprehensive Trades Table for Optimized Strategy
        st.write("### 📊 Optimized Strategy - Detailed Trades Table")
        try:
            opt_trades_df, opt_trades_error = create_trades_table(
                results["results"], data
            )
            if opt_trades_error:
                st.warning(
                    f"Could not create optimized trades table: {opt_trades_error}"
                )
            elif not opt_trades_df.empty:
                # Apply styling to highlight entry and exit indicators
                def highlight_columns(col):
                    if "Entry" in col:
                        return ["background-color: #e8f5e9"] * len(col)
                    elif "Exit" in col:
                        return ["background-color: #ffebee"] * len(col)
                    return [""] * len(col)

                styled_trades = opt_trades_df.style.apply(highlight_columns, axis=0)
                st.dataframe(styled_trades, use_container_width=True)
            else:
                st.info("No trades executed by the optimized strategy.")
        except Exception as e:
            st.error(f"Error creating trades table: {str(e)}")
            logger.exception("Error creating trades table")

        # Trade Statistics Summary for Optimized Strategy
        st.write("### 📈 Optimized Strategy - Trade Statistics")
        from .data_processing import analyze_best_trades

        best_trades_analysis = analyze_best_trades(results["results"])
        if "error" not in best_trades_analysis:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Trades", best_trades_analysis["total_trades"])
                st.metric("Winning Trades", best_trades_analysis["winning_trades"])

            with col2:
                win_rate = (
                    (
                        best_trades_analysis["winning_trades"]
                        / best_trades_analysis["total_trades"]
                        * 100
                    )
                    if best_trades_analysis["total_trades"] > 0
                    else 0
                )
                st.metric("Win Rate", f"{win_rate:.1f}%")
                st.metric("Losing Trades", best_trades_analysis["losing_trades"])

            with col3:
                st.metric("Total P&L", f"{best_trades_analysis['total_pnl']:.2f}")
                st.metric(
                    "Best Trade",
                    f"{best_trades_analysis['best_trade_pnl']:.2f}",
                )

            with col4:
                st.metric(
                    "Avg Winning Trade",
                    f"{best_trades_analysis['avg_winning_trade']:.2f}",
                )
                st.metric(
                    "Avg Losing Trade",
                    f"{best_trades_analysis['avg_losing_trade']:.2f}",
                )

        # Best Trading Times Analysis for Optimized Strategy
        st.write("### ⏰ Optimized Strategy - Best Trading Times")
        time_analysis = analyze_best_time_ranges(results["results"])
        if "error" not in time_analysis:
            hours_df, days_df, months_df = create_best_times_table(time_analysis)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**🕐 Best Hours to Trade**")
                if hours_df is not None and not hours_df.empty:
                    st.dataframe(hours_df, use_container_width=True)
                else:
                    st.info("No hourly data available")

            with col2:
                st.write("**📅 Best Days to Trade**")
                if days_df is not None and not days_df.empty:
                    st.dataframe(days_df, use_container_width=True)
                else:
                    st.info("No daily data available")

            with col3:
                st.write("**📆 Best Months to Trade**")
                if months_df is not None and not months_df.empty:
                    st.dataframe(months_df, use_container_width=True)
                else:
                    st.info("No monthly data available")

            # Time Analysis Chart for Optimized Strategy
            st.write("### 📊 Optimized Strategy - Trading Time Analysis")
            time_chart = plot_time_analysis(time_analysis)
            if time_chart:
                st.plotly_chart(time_chart, use_container_width=True, key=uuid.uuid4())
        else:
            st.warning(
                "Could not analyze time ranges: "
                + time_analysis.get("error", "Unknown error")
            )

    # Update progress to 100% and show completion
    progress_bar.progress(100)
    status_text.text("Optimization complete!")
    st.toast("Optimization complete")

    # Display best strategies report after all strategies are processed
    st.subheader("📊 Optimization - Best Strategies Report")
    display_best_strategies_report(strategy_reports, ticker, interval)
    return strategy_reports


def run_walkforward_analysis(
    params, data, analyzer_config, progress_bar, status_text, ticker
):
    """Run walk-forward analysis and display results."""
    status_text.text("Starting walk-forward analysis...")
    data = data.copy()
    strategy_reports = []
    strategy_length = len(params["selected_strategy"])

    for idx, strategy in enumerate(params["selected_strategy"]):
        progress_bar.progress(int((idx / strategy_length) * 100))
        from comprehensive_backtesting.registry import get_strategy

        strategy_class = get_strategy(strategy)

        wf = WalkForwardAnalysis(
            data=data,
            strategy_name=strategy_class.__name__,
            optimization_params=strategy_class.optimization_params,
            optimization_metric="sharpe_ratio",
            training_ratio=0.6,
            testing_ratio=0.15,
            step_ratio=0.2,
            n_trials=params["n_trials"],
            verbose=False,
        )
        wf.run_analysis()

        # Generate trade statistics summary
        from comprehensive_backtesting.walk_forward_analysis import (
            calculate_trade_statistics,
        )

        stats_summary, all_in_sample, all_out_sample = wf.generate_trade_statistics()

        # Save window summary with parameters
        window_summary = wf.get_window_summary()

        # FIX: Ensure best_params are properly extracted
        if not window_summary.empty and "best_params" in window_summary.columns:
            # Convert string representation of dict to actual dict
            try:
                window_summary["best_params"] = window_summary["best_params"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
            except:
                print("Warning: Could not convert best_params string to dict")

        # Format percentage columns
        percent_cols = [
            col
            for col in window_summary.columns
            if "return" in col.lower()
            or "drawdown" in col.lower()
            or "in-sample" in col.lower()
            or "out-sample" in col.lower()
        ]
        for col in percent_cols:
            window_summary[col] = (
                window_summary[col].astype(str).str.replace("%", "", regex=False)
            )
            window_summary[col] = pd.to_numeric(window_summary[col], errors="coerce")

        # Print overall metrics
        overall = wf.get_overall_metrics()

        # NEW: Check if we have results
        if not wf.results:
            st.error(
                "Walk-forward analysis produced no results. Please check your parameters."
            )
            continue

        # Generate strategy report for each valid window
        for i, window in enumerate(wf.results):
            if window.get("valid") and "out_sample_performance" in window:
                report = generate_strategy_report(
                    window["out_sample_performance"],
                    f"Walk-Forward Window {i+1}",
                    ticker,
                    params["timeframe"],
                )
                if report:
                    strategy_reports.append(report)

        # Display walk-forward results
        from .display_results import display_walkforward_results_detailed

        display_walkforward_results_detailed(wf, ticker, params, strategy)

    # Update progress to 100% and show completion
    progress_bar.progress(100)
    status_text.text("Walk-forward analysis complete!")
    st.toast("Walk-forward analysis complete")

    # Display best strategies report after all strategies are processed
    st.subheader("📊 Walk-Forward - Best Strategies Report")
    display_best_strategies_report(strategy_reports, ticker, params["timeframe"])
    return strategy_reports


def run_analysis(params):
    """Run the selected analysis based on user parameters."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Validate inputs
        from .ui_components import validate_inputs

        errors = validate_inputs(params)
        if errors:
            for error in errors:
                st.sidebar.error(error)
            return

        progress_bar.progress(10)
        status_text.text("Fetching data...")

        for ticker in params["ticker"]:
            if not ticker:
                st.sidebar.error("Ticker cannot be empty.")
                return

            # Fetch data
            data = get_data_sync(
                ticker,
                params["start_date"],
                params["end_date"],
                interval=params["timeframe"],
            )

            # Validate data
            if not validate_data(data, ticker):
                progress_bar.progress(0)
                status_text.text("Analysis failed - no data")
                return

            progress_bar.progress(30)
            status_text.text("Preparing analyzers...")

            # Prepare analyzer configuration
            analyzer_config = [
                (params["available_analyzers"][name], {"_name": name.lower()})
                for name in params["selected_analyzers"]
            ]

            # Run analysis based on type
            if params["analysis_type"] == "Backtest":
                progress_bar.progress(40)
                status_text.text("Running Backtest...")
                run_backtest_analysis(
                    params, data, analyzer_config, progress_bar, status_text, ticker
                )
            elif params["analysis_type"] == "Optimization":
                progress_bar.progress(40)
                status_text.text("Running Optimization...")
                run_optimization_analysis(
                    params, data, analyzer_config, progress_bar, status_text, ticker
                )
            elif params["analysis_type"] == "Walk-Forward":
                progress_bar.progress(40)
                status_text.text("Running Walk-Forward Analysis...")
                run_walkforward_analysis(
                    params, data, analyzer_config, progress_bar, status_text, ticker
                )
            elif params["analysis_type"] == "Complete Backtest":
                progress_bar.progress(40)
                status_text.text("Running Complete Backtest...")
                from .complete_backtest import complete_backtest

                complete_backtest(data, progress_bar, params, ticker)

    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Error running analysis: {e}", exc_info=True)
        st.error(f"Analysis failed: {str(e)}")
        progress_bar.progress(0)
        status_text.text("Analysis failed.")
