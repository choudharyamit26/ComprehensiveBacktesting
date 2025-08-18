"""
Result display functions for different types of analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import uuid
import logging
from comprehensive_backtesting.reports import PerformanceAnalyzer
from comprehensive_backtesting.walk_forward_analysis import calculate_trade_statistics
from .table_generators import create_summary_table, create_trades_table
from .data_processing import analyze_best_trades, analyze_best_time_ranges
from .table_generators import create_best_times_table
from .visualization import plot_time_analysis
from .utils import calculate_return_metrics

logger = logging.getLogger(__name__)


def display_walkforward_results_detailed(wf, ticker, params, strategy):
    """Display comprehensive results from walk-forward analysis with enhanced trade analysis and time return analysis."""

    summary_stats = wf.get_overall_metrics()

    # Handle both old and new window formats
    windows = wf.results
    if not windows:
        st.error("No windows were generated in walk-forward analysis.")
        st.info("This might be due to insufficient data or incompatible parameters.")
        return

    # Walk-Forward Summary Visualization
    st.subheader(
        f"📊 Walk-Forward Analysis Summary for {ticker} Using strategy: {strategy}"
    )

    # Display overall summary statistics
    if summary_stats:
        st.write("### 📈 Overall Performance Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(
            "Valid Windows",
            f"{summary_stats.get('total_windows', 0)}",
        )
        col2.metric(
            "Avg In-Sample Return",
            f"{summary_stats.get('in_sample_return_avg_return', 0):.4f}%",
        )
        col3.metric(
            "Avg Out-Sample Return",
            f"{summary_stats.get('out_sample_avg_return', 0):.4f}%",
        )
        col4.metric(
            "Avg In-Sample Sharpe",
            f"{summary_stats.get('in_sample_avg_sharpe', 0):.4f}",
        )
        col5.metric(
            "Avg Out-Sample Sharpe",
            f"{summary_stats.get('out_sample_avg_sharpe', 0):.4f}",
        )
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Out-Sample Win Rate", f"{summary_stats.get('win_rate_out_sample', 0)}%"
        )
        col2.metric("Return Correlation", f"{summary_stats.get('correlation', 0):.3f}")
        col3.metric(
            "Avg Degradation", f"{summary_stats.get('avg_degradation', 0):.2f}%"
        )

    # 1. Parameter evolution table
    st.write("### ⚙️ Parameter Evolution Across Windows")
    param_evolution = []

    for i, window in enumerate(windows):
        if not window.get("valid", True):
            continue

        periods = {
            "in_sample_start": window.get("train_start", ""),
            "in_sample_end": window.get("train_end", ""),
            "out_sample_start": window.get("test_start", ""),
            "out_sample_end": window.get("test_end", ""),
        }
        best_params = window.get("best_params", {})

        # Get performance metrics
        in_perf = window.get("in_sample_metrics", {})
        out_perf = window.get("out_sample_metrics", {})

        param_evolution.append(
            {
                "Window": i + 1,
                "In-Sample Period": f"{periods.get('in_sample_start', '')} to {periods.get('in_sample_end', '')}",
                "Out-Sample Period": f"{periods.get('out_sample_start', '')} to {periods.get('out_sample_end', '')}",
                **best_params,
                "In Return (%)": in_perf.get("total_return", 0),
                "In Sharpe": in_perf.get("sharpe_ratio", 0),
                "Out Return (%)": out_perf.get("total_return", 0),
                "Out Sharpe": out_perf.get("sharpe_ratio", 0),
            }
        )

    if param_evolution:
        param_df = pd.DataFrame(param_evolution)
        # Fill None/NaN with 0.0 for columns to be formatted
        for col in ["In Return (%)", "Out Return (%)", "In Sharpe", "Out Sharpe"]:
            if col in param_df.columns:
                param_df[col] = param_df[col].fillna(0.0)
        st.dataframe(
            param_df.style.format(
                {
                    "In Return (%)": "{:.2f}%",
                    "Out Return (%)": "{:.2f}%",
                    "In Sharpe": "{:.2f}",
                    "Out Sharpe": "{:.2f}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("No parameter evolution data available")

    # Time Return Analysis Section
    st.subheader("⏱️ Time Return Analysis")
    st.write("### Monthly Return Distribution")

    # Aggregate all trades for monthly analysis
    all_trades = []
    for window in windows:
        if not window.get("valid", True):
            continue

        # Get all trades (in-sample and out-sample)
        in_trades = window.get("in_sample_trades", [])
        out_trades = window.get("out_sample_trades", [])

        all_trades.extend(in_trades)
        all_trades.extend(out_trades)

    if all_trades:
        # Create DataFrame from trades
        trade_df = pd.DataFrame(all_trades)

        # Convert to datetime and extract month
        if "entry_date" in trade_df.columns:
            trade_df["entry_date"] = pd.to_datetime(trade_df["entry_date"])
            trade_df["month"] = trade_df["entry_date"].dt.to_period("M")

            # Calculate monthly P&L
            monthly_pnl = trade_df.groupby("month")["pnl_net"].sum().reset_index()
            monthly_pnl["month"] = monthly_pnl["month"].dt.to_timestamp()

            # Calculate monthly return percentage
            initial_cash = params.get("initial_cash", 10000)
            monthly_pnl["return_pct"] = (monthly_pnl["pnl_net"] / initial_cash) * 100

            # Create visualization
            fig = px.bar(
                monthly_pnl,
                x="month",
                y="return_pct",
                labels={"return_pct": "Return (%)", "month": "Month"},
                title="Monthly Returns Across All Periods",
                color_discrete_sequence=["#1f77b4"],
            )
            fig.update_layout(xaxis_title="Month", yaxis_title="Return (%)", height=500)
            st.plotly_chart(fig, use_container_width=True, key=uuid.uuid4())

            # Monthly return metrics
            st.write("### Monthly Return Metrics")

            metrics = calculate_return_metrics(monthly_pnl["return_pct"])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Month", metrics.get("Best Month", "N/A"))
                st.metric("Worst Month", metrics.get("Worst Month", "N/A"))

            with col2:
                st.metric(
                    "Avg Positive Month", metrics.get("Avg Positive Month", "N/A")
                )
                st.metric(
                    "Avg Negative Month", metrics.get("Avg Negative Month", "N/A")
                )

            with col3:
                st.metric("Win Rate", metrics.get("Win Rate", "N/A"))
                st.metric("Std Dev", metrics.get("Std Dev", "N/A"))

            # Display monthly returns table
            st.write("### Monthly Return Details")
            monthly_pnl["return_pct"] = monthly_pnl["return_pct"].apply(
                lambda x: f"{x:.2f}%"
            )
            st.dataframe(
                monthly_pnl[["month", "return_pct"]].rename(
                    columns={"month": "Month", "return_pct": "Return (%)"}
                ),
                use_container_width=True,
            )
        else:
            st.warning("Trade data missing 'entry_date' field for time analysis")
    else:
        st.info("No trade data available for time return analysis")

    # Detailed Window Analysis
    st.write("### 📅 Detailed Window Analysis")

    for i, window in enumerate(windows):
        if not window.get("valid", True):
            continue

        periods = {
            "in_sample_start": window.get("train_start", ""),
            "in_sample_end": window.get("train_end", ""),
            "out_sample_start": window.get("test_start", ""),
            "out_sample_end": window.get("test_end", ""),
        }
        in_perf = window.get("in_sample_metrics", {})
        out_perf = window.get("out_sample_metrics", {})

        in_trades = window.get("in_sample_trades", [])
        out_trades = window.get("out_sample_trades", [])

        with st.expander(
            f"Window {i+1}: Train {periods.get('in_sample_start', '')} to {periods.get('in_sample_end', '')} | Test {periods.get('out_sample_start', '')} to {periods.get('out_sample_end', '')}",
            expanded=False,
        ):
            # Window summary
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Training Period**")
                st.write(f"Start: {periods.get('in_sample_start', '')}")
                st.write(f"End: {periods.get('in_sample_end', '')}")
                st.write(f"Optimization Trials: {params.get('n_trials', 0)}")

                st.write("**Best Parameters**")
                if window.get("best_params"):
                    for param, value in window["best_params"].items():
                        st.code(f"{param}: {value}")
                else:
                    st.info("No best parameters available")

            with col2:
                st.write("**Performance Summary**")

                # Handle None values for metrics
                in_total_return = in_perf.get("total_return", 0)
                out_total_return = out_perf.get("total_return", 0)

                in_sharpe = in_perf.get("sharpe_ratio", 0)
                out_sharpe = out_perf.get("sharpe_ratio", 0)

                in_max_dd = in_perf.get("max_drawdown", 0)
                out_max_dd = out_perf.get("max_drawdown", 0)

                st.metric(
                    "In-Sample Return",
                    f"{in_total_return:.2f}%" if in_total_return is not None else "N/A",
                )
                st.metric(
                    "Out-Sample Return",
                    (
                        f"{out_total_return:.2f}%"
                        if out_total_return is not None
                        else "N/A"
                    ),
                )

                st.metric(
                    "In-Sample Sharpe",
                    f"{in_sharpe:.2f}" if in_sharpe is not None else "N/A",
                )
                st.metric(
                    "Out-Sample Sharpe",
                    f"{out_sharpe:.2f}" if out_sharpe is not None else "N/A",
                )

                st.metric(
                    "In Max Drawdown",
                    f"{in_max_dd:.2f}%" if in_max_dd is not None else "N/A",
                )
                st.metric(
                    "Out Max Drawdown",
                    f"{out_max_dd:.2f}%" if out_max_dd is not None else "N/A",
                )

            # Create tabs for detailed analysis
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Equity Curves", "In-Sample", "Out-Sample", "Time Analysis"]
            )

            with tab1:  # Equity Curves
                st.write("### 📈 Equity Curves Comparison")

                # Get equity curves
                in_equity = in_perf.get("equity_curve", pd.Series())
                out_equity = out_perf.get("equity_curve", pd.Series())

                if not in_equity.empty or not out_equity.empty:
                    fig = go.Figure()

                    if not in_equity.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=in_equity.index,
                                y=in_equity.values,
                                mode="lines",
                                name="In-Sample",
                                line=dict(color="#1f77b4", width=3),
                            )
                        )

                    if not out_equity.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=out_equity.index,
                                y=out_equity.values,
                                mode="lines",
                                name="Out-Sample",
                                line=dict(color="#ff7f0e", width=3),
                            )
                        )

                    # Add vertical line at test start
                    test_start = periods.get("out_sample_start", "")
                    if test_start:
                        try:
                            test_start_dt = pd.to_datetime(test_start)
                            fig.add_vline(
                                x=test_start_dt,
                                line_dash="dash",
                                line_color="green",
                                annotation_text="Test Start",
                            )
                        except Exception as e:
                            logger.warning(f"Could not add test start line: {e}")

                    fig.update_layout(
                        title="Portfolio Value",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        showlegend=True,
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True, key=uuid.uuid4())
                else:
                    st.info("No equity curve data available")

            with tab2:  # In-Sample
                st.write("### 📊 In-Sample Analysis")

                # Trade Statistics
                if in_trades:
                    st.write("#### Trade Statistics")
                    win_trades = [t for t in in_trades if t.get("pnl_net", 0) > 0]
                    loss_trades = [t for t in in_trades if t.get("pnl_net", 0) <= 0]

                    avg_win = (
                        sum(t["pnl_net"] for t in win_trades) / len(win_trades)
                        if win_trades
                        else 0
                    )
                    avg_loss = (
                        sum(t["pnl_net"] for t in loss_trades) / len(loss_trades)
                        if loss_trades
                        else 0
                    )
                    win_rate = (
                        len(win_trades) / len(in_trades) * 100 if in_trades else 0
                    )

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Trades", len(in_trades))
                        st.metric("Win Rate", f"{win_rate:.1f}%")

                    with col2:
                        st.metric("Avg Win", f"{avg_win:.2f}")
                        st.metric("Avg Loss", f"{avg_loss:.2f}")

                    with col3:
                        profit_factor = abs(avg_win / avg_loss) if avg_loss else 0
                        st.metric("Profit Factor", f"{profit_factor:.2f}")
                        st.metric(
                            "Win/Loss Ratio",
                            f"{abs(avg_win/avg_loss):.2f}" if avg_loss else "N/A",
                        )

                    with col4:
                        st.metric(
                            "Max Win",
                            (
                                f"{max(t['pnl_net'] for t in win_trades):.2f}"
                                if win_trades
                                else "N/A"
                            ),
                        )
                        st.metric(
                            "Max Loss",
                            (
                                f"{min(t['pnl_net'] for t in loss_trades):.2f}"
                                if loss_trades
                                else "N/A"
                            ),
                        )

                    # Detailed Trades Table
                    st.write("#### Detailed Trades")
                    in_trade_df = pd.DataFrame(in_trades)

                    # Format datetime columns
                    for col in ["entry_date", "exit_date"]:
                        if col in in_trade_df.columns:
                            in_trade_df[col] = pd.to_datetime(in_trade_df[col])

                    # Display formatted table
                    st.dataframe(in_trade_df, use_container_width=True)
                else:
                    st.info("No in-sample trades executed")

            with tab3:  # Out-Sample
                st.write("### 📊 Out-Sample Analysis")

                # Trade Statistics
                if out_trades:
                    st.write("#### Trade Statistics")
                    win_trades = [t for t in out_trades if t.get("pnl_net", 0) > 0]
                    loss_trades = [t for t in out_trades if t.get("pnl_net", 0) <= 0]

                    avg_win = (
                        sum(t["pnl_net"] for t in win_trades) / len(win_trades)
                        if win_trades
                        else 0
                    )
                    avg_loss = (
                        sum(t["pnl_net"] for t in loss_trades) / len(loss_trades)
                        if loss_trades
                        else 0
                    )
                    win_rate = (
                        len(win_trades) / len(out_trades) * 100 if out_trades else 0
                    )

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Trades", len(out_trades))
                        st.metric("Win Rate", f"{win_rate:.1f}%")

                    with col2:
                        st.metric("Avg Win", f"{avg_win:.2f}")
                        st.metric("Avg Loss", f"{avg_loss:.2f}")

                    with col3:
                        profit_factor = abs(avg_win / avg_loss) if avg_loss else 0
                        st.metric("Profit Factor", f"{profit_factor:.2f}")
                        st.metric(
                            "Win/Loss Ratio",
                            f"{abs(avg_win/avg_loss):.2f}" if avg_loss else "N/A",
                        )

                    with col4:
                        st.metric(
                            "Max Win",
                            (
                                f"{max(t['pnl_net'] for t in win_trades):.2f}"
                                if win_trades
                                else "N/A"
                            ),
                        )
                        st.metric(
                            "Max Loss",
                            (
                                f"{min(t['pnl_net'] for t in loss_trades):.2f}"
                                if loss_trades
                                else "N/A"
                            ),
                        )

                    # Detailed Trades Table
                    st.write("#### Detailed Trades")
                    out_trade_df = pd.DataFrame(out_trades)

                    # Format datetime columns
                    for col in ["entry_date", "exit_date"]:
                        if col in out_trade_df.columns:
                            out_trade_df[col] = pd.to_datetime(out_trade_df[col])

                    # Display formatted table
                    st.dataframe(out_trade_df, use_container_width=True)
                else:
                    st.info("No out-sample trades executed")

            with tab4:  # Time Analysis
                st.write("### ⏱️ Time Return Analysis")

                # Combine in-sample and out-sample trades
                all_window_trades = in_trades + out_trades
                if all_window_trades:
                    trade_df = pd.DataFrame(all_window_trades)

                    if "entry_date" in trade_df.columns:
                        # Hourly analysis
                        st.write("#### Hourly Returns")
                        trade_df["entry_date"] = pd.to_datetime(trade_df["entry_date"])
                        trade_df["hour"] = trade_df["entry_date"].dt.hour

                        hourly_pnl = (
                            trade_df.groupby("hour")["pnl_net"].sum().reset_index()
                        )

                        if not hourly_pnl.empty:
                            fig = px.bar(
                                hourly_pnl,
                                x="hour",
                                y="pnl_net",
                                labels={"pnl_net": "P&L", "hour": "Hour of Day"},
                                title="Hourly Returns",
                                color_discrete_sequence=["#2ca02c"],
                            )
                            fig.update_layout(
                                xaxis_title="Hour (24h)",
                                yaxis_title="Profit/Loss",
                                height=400,
                            )
                            st.plotly_chart(
                                fig, use_container_width=True, key=uuid.uuid4()
                            )

                        # Day of week analysis
                        st.write("#### Day of Week Returns")
                        trade_df["day_of_week"] = trade_df["entry_date"].dt.day_name()
                        day_order = [
                            "Monday",
                            "Tuesday",
                            "Wednesday",
                            "Thursday",
                            "Friday",
                            "Saturday",
                            "Sunday",
                        ]
                        trade_df["day_of_week"] = pd.Categorical(
                            trade_df["day_of_week"], categories=day_order, ordered=True
                        )

                        dow_pnl = (
                            trade_df.groupby("day_of_week")["pnl_net"]
                            .sum()
                            .reset_index()
                        )

                        if not dow_pnl.empty:
                            fig = px.bar(
                                dow_pnl,
                                x="day_of_week",
                                y="pnl_net",
                                labels={"pnl_net": "P&L", "day_of_week": "Day of Week"},
                                title="Day of Week Returns",
                                color_discrete_sequence=["#d62728"],
                            )
                            fig.update_layout(
                                xaxis_title="Day of Week",
                                yaxis_title="Profit/Loss",
                                height=400,
                            )
                            st.plotly_chart(
                                fig, use_container_width=True, key=uuid.uuid4()
                            )

                        # Monthly analysis
                        st.write("#### Monthly Returns")
                        trade_df["month"] = trade_df["entry_date"].dt.to_period("M")
                        monthly_pnl = (
                            trade_df.groupby("month")["pnl_net"].sum().reset_index()
                        )
                        monthly_pnl["month"] = monthly_pnl["month"].dt.to_timestamp()

                        if not monthly_pnl.empty:
                            fig = px.bar(
                                monthly_pnl,
                                x="month",
                                y="pnl_net",
                                labels={"pnl_net": "P&L", "month": "Month"},
                                title="Monthly Returns",
                                color_discrete_sequence=["#9467bd"],
                            )
                            fig.update_layout(
                                xaxis_title="Month",
                                yaxis_title="Profit/Loss",
                                height=400,
                            )
                            st.plotly_chart(
                                fig, use_container_width=True, key=uuid.uuid4()
                            )
                    else:
                        st.warning(
                            "Trade data missing 'entry_date' field for time analysis"
                        )
                else:
                    st.info("No trade data available for time analysis")

    # Performance degradation metrics
    st.write("### 📉 Performance Degradation Summary")
    degradation_data = []

    for i, window in enumerate(windows):
        if not window.get("valid", True):
            continue

        in_perf = window.get("in_sample_metrics", {})
        out_perf = window.get("out_sample_metrics", {})

        in_return = in_perf.get("total_return", 0)
        out_return = out_perf.get("total_return", 0)
        degradation = out_return - in_return

        degradation_data.append(
            {
                "Window": i + 1,
                "In-Sample Return": in_return,
                "Out-Sample Return": out_return,
                "Degradation (%)": degradation,
            }
        )

    if degradation_data:
        deg_df = pd.DataFrame(degradation_data)
        st.bar_chart(
            deg_df.set_index("Window")[["In-Sample Return", "Out-Sample Return"]]
        )
        st.dataframe(
            deg_df.style.format(
                {
                    "In-Sample Return": "{:.2f}%",
                    "Out-Sample Return": "{:.2f}%",
                    "Degradation (%)": "{:.2f}%",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("No degradation data available")

    # 5. Overall metrics
    st.subheader("📈 Overall Performance Summary")
    overall = wf.get_overall_metrics()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "In-Sample Avg Return", f"{overall['in_sample_return_avg_return']:.4f}"
        )
        st.metric("Out-Sample Avg Return", f"{overall['out_sample_avg_return']:.4f}")
    with col2:
        if overall["in_sample_avg_sharpe"] is not None:
            st.metric("In-Sample Avg Sharpe", f"{overall['in_sample_avg_sharpe']:.4f}")
        if overall["out_sample_avg_sharpe"] is not None:
            st.metric(
                "Out-Sample Avg Sharpe", f"{overall['out_sample_avg_sharpe']:.4f}"
            )
    with col3:
        st.metric(
            "Valid Windows",
            f"{overall['total_windows']}",
        )

    # 6. Trade statistics summary
    st.write("### 📋 Aggregate Trade Statistics")
    stats_summary, all_in_sample, all_out_sample = wf.generate_trade_statistics()
    st.dataframe(stats_summary, use_container_width=True)

    st.success("Walk-forward analysis display complete!")
