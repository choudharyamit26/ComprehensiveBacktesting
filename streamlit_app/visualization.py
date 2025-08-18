"""
Visualization and plotting functions for the Streamlit application.
"""

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import optuna
import optuna.visualization
import logging
import uuid
from datetime import datetime
from comprehensive_backtesting.reports import PerformanceAnalyzer
from .utils import get_strategy
from .config import IST

logger = logging.getLogger(__name__)


def plot_contour(study):
    """Generate a contour plot for Optuna optimization results."""
    try:
        if not study or not study.trials:
            logger.warning("No trials available for contour plot")
            return None

        fig = optuna.visualization.plot_contour(study)
        return fig
    except Exception as e:
        logger.error(f"Error generating contour plot: {e}")
        return None


def plot_equity_curve(equity_data):
    """Plot equity curve from time return analyzer."""
    try:
        if not equity_data:
            return None

        if isinstance(equity_data, pd.Series):
            # Already a series, use directly
            equity_series = equity_data
        elif isinstance(equity_data, dict):
            # Handle different dictionary formats
            if all(isinstance(k, (pd.Timestamp, datetime)) for k in equity_data.keys()):
                # Dictionary with datetime keys
                dates = list(equity_data.keys())
                values = list(equity_data.values())
            elif all(isinstance(k, str) for k in equity_data.keys()):
                # Dictionary with string dates
                dates = pd.to_datetime(list(equity_data.keys()))
                values = list(equity_data.values())
            else:
                # Fallback to numeric index
                dates = np.arange(len(equity_data))
                values = list(equity_data.values())

            equity_series = pd.Series(values, index=dates)
        else:
            return None

        # Calculate cumulative returns
        cumulative = (1 + equity_series).cumprod()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                mode="lines",
                name="Equity Curve",
                line=dict(color="royalblue", width=3),
            )
        )

        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode="x",
            showlegend=True,
        )

        return fig
    except Exception as e:
        logger.error(f"Error plotting equity curve: {e}")
        return None


def plot_composite_backtest_results(results, data):
    """Visualize all backtest phases in a single chart.

    Args:
        results (dict): Backtest results containing basic, optimized, and walk-forward data
        data (pd.DataFrame): Historical price data

    Returns:
        go.Figure: Plotly figure showing all backtest phases
    """
    try:
        fig = go.Figure()

        # Add price data
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                name="Price",
                line=dict(color="rgba(100,100,100,0.5)"),
                visible="legendonly",
            )
        )

        # Basic strategy equity
        if "basic" in results and results["basic"]:
            strategy = get_strategy(results["basic"])
            if hasattr(strategy, "analyzers") and hasattr(
                strategy.analyzers, "timereturn"
            ):
                equity_data = strategy.analyzers.timereturn.get_analysis()
                if equity_data:
                    dates = sorted(equity_data.keys())
                    values = [equity_data[d] for d in dates]
                    cumulative = np.cumprod([1 + v / 100 for v in values]) * 100
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=cumulative,
                            mode="lines",
                            name="Basic Strategy",
                            line=dict(color="blue", width=2),
                        )
                    )

        # Optimized strategy equity
        if (
            "optimization" in results
            and "results" in results["optimization"]
            and results["optimization"]["results"]
        ):
            strategy = get_strategy(results["optimization"]["results"])
            if hasattr(strategy, "analyzers") and hasattr(
                strategy.analyzers, "timereturn"
            ):
                equity_data = strategy.analyzers.timereturn.get_analysis()
                if equity_data:
                    dates = sorted(equity_data.keys())
                    values = [equity_data[d] for d in dates]
                    cumulative = np.cumprod([1 + v / 100 for v in values]) * 100
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=cumulative,
                            mode="lines",
                            name="Optimized Strategy",
                            line=dict(color="green", width=2),
                        )
                    )

        # Walk-forward equity
        if "walk_forward" in results and "windows" in results["walk_forward"]:
            windows = results["walk_forward"]["windows"]
            valid_windows = [w for w in windows if w.get("valid", False)]

            for i, window in enumerate(valid_windows):
                if (
                    "out_sample_performance" in window
                    and "timereturn" in window["out_sample_performance"]
                ):
                    equity_data = window["out_sample_performance"]["timereturn"]
                    if equity_data:
                        dates = sorted(equity_data.keys())
                        values = [equity_data[d] for d in dates]
                        cumulative = np.cumprod([1 + v / 100 for v in values]) * 100
                        fig.add_trace(
                            go.Scatter(
                                x=dates,
                                y=cumulative,
                                mode="lines",
                                name=f"WF Window {i+1}",
                                line=dict(width=1.5, dash="dot"),
                            )
                        )

        fig.update_layout(
            title="Complete Backtest - Strategy Comparison",
            xaxis_title="Date",
            yaxis_title="Equity Value (Indexed to 100)",
            legend_title="Strategies",
            hovermode="x unified",
            height=600,
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating composite plot: {e}")
        return None


def plot_walkforward_summary(results):
    """Create a comprehensive plot showing walk-forward performance and parameter evolution."""
    try:
        windows = results["walk_forward"].get("windows", [])
        if not windows:
            return None

        # Create subplots with secondary y-axis for parameters
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            specs=[[{"type": "scatter"}], [{"type": "table"}]],
        )

        # 1. Equity Curve and Returns
        cumulative_return = 100
        equity_points = [cumulative_return]
        dates = []
        window_returns = []
        window_labels = []

        for i, window in enumerate(windows):
            if window.get("valid", True):
                equity = window["out_sample_performance"].get("timereturn", {})
                if equity:
                    # Calculate cumulative returns
                    returns = list(equity.values())
                    for ret in returns:
                        cumulative_return *= 1 + ret / 100
                        equity_points.append(cumulative_return)

                    # Add dates for plotting
                    dates.extend(sorted(equity.keys()))

                    # Store window return
                    summary = window["out_sample_performance"].get("summary", {})
                    return_pct = summary.get("total_return_pct", 0)
                    window_returns.append(return_pct)
                    window_labels.append(f"Window {i+1}")

        # Add equity curve
        if len(equity_points) > 1:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=equity_points,
                    mode="lines",
                    name="Cumulative Equity",
                    line=dict(color="#636EFA", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(99, 110, 250, 0.1)",
                ),
                row=1,
                col=1,
            )

        # Add window returns as bars
        if window_returns:
            fig.add_trace(
                go.Bar(
                    x=window_labels,
                    y=window_returns,
                    name="Window Return",
                    marker_color=[
                        "#00CC96" if r >= 0 else "#EF553B" for r in window_returns
                    ],
                    text=[f"{r:.2f}%" for r in window_returns],
                    textposition="auto",
                ),
                row=2,
                col=1,
            )

        # 2. Parameter Evolution Table
        param_data = []
        param_columns = set()

        for i, window in enumerate(windows):
            if window.get("valid", True):
                param_row = {"Window": f"Window {i+1}"}
                best_params = window.get("best_params", {})

                # Get summary metrics
                summary = window["out_sample_performance"].get("summary", {})
                param_row["Return (%)"] = summary.get("total_return_pct", 0)
                param_row["Max Drawdown (%)"] = summary.get("max_drawdown_pct", 0)
                param_row["Sharpe Ratio"] = summary.get("sharpe_ratio", "N/A")

                # Add parameters
                for param, value in best_params.items():
                    param_row[param] = value
                    param_columns.add(param)

                param_data.append(param_row)

        # Create table data
        if param_data:
            # Define column order
            table_columns = [
                "Window",
                "Return (%)",
                "Sharpe Ratio",
                "Max Drawdown (%)",
            ] + sorted(param_columns)
            table_data = []

            for row in param_data:
                table_row = [row.get(col, "") for col in table_columns]
                table_data.append(table_row)

            # Add table to plot
            fig.add_trace(
                go.Table(
                    header=dict(values=table_columns, font=dict(size=10), align="left"),
                    cells=dict(values=list(zip(*table_data)), align="left"),
                ),
                row=2,
                col=1,
            )
        else:
            # Fallback if no parameter data
            fig.add_trace(go.Scatter(x=[], y=[], showlegend=False), row=2, col=1)

        # Update layout
        fig.update_layout(
            title="Walk-Forward Analysis Summary",
            height=800,
            showlegend=True,
            hovermode="x unified",
        )

        fig.update_yaxes(title_text="Equity Value", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=1)

        return fig

    except Exception as e:
        logger.error(f"Error creating walk-forward summary plot: {str(e)}")
        return None


def plot_time_analysis(time_analysis):
    """Create visualizations for time analysis data."""
    try:
        if "error" in time_analysis:
            return None

        # Create subplots for different time analyses
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Winning Trades by Hour",
                "Winning Trades by Day",
                "P&L by Hour",
                "P&L by Day",
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        # Hourly trade count
        if time_analysis.get("hourly_distribution"):
            hours = list(time_analysis["hourly_distribution"].keys())
            counts = [
                stats["count"]
                for stats in time_analysis["hourly_distribution"].values()
            ]
            fig.add_trace(
                go.Bar(
                    x=[f"{h}:00" for h in hours],
                    y=counts,
                    name="Trades by Hour",
                    marker_color="lightblue",
                ),
                row=1,
                col=1,
            )

        # Daily trade count
        if time_analysis.get("daily_distribution"):
            days = list(time_analysis["daily_distribution"].keys())
            counts = [
                stats["count"] for stats in time_analysis["daily_distribution"].values()
            ]
            fig.add_trace(
                go.Bar(
                    x=days, y=counts, name="Trades by Day", marker_color="lightgreen"
                ),
                row=1,
                col=2,
            )

        # Hourly P&L
        if time_analysis.get("hourly_distribution"):
            hours = list(time_analysis["hourly_distribution"].keys())
            pnls = [
                stats["total_pnl"]
                for stats in time_analysis["hourly_distribution"].values()
            ]
            fig.add_trace(
                go.Bar(
                    x=[f"{h}:00" for h in hours],
                    y=pnls,
                    name="P&L by Hour",
                    marker_color="orange",
                ),
                row=2,
                col=1,
            )

        # Daily P&L
        if time_analysis.get("daily_distribution"):
            days = list(time_analysis["daily_distribution"].keys())
            pnls = [
                stats["total_pnl"]
                for stats in time_analysis["daily_distribution"].values()
            ]
            fig.add_trace(
                go.Bar(x=days, y=pnls, name="P&L by Day", marker_color="salmon"),
                row=2,
                col=2,
            )

        fig.update_layout(title="Trading Time Analysis", height=600, showlegend=False)

        return fig
    except Exception as e:
        logger.error(f"Error creating time analysis plot: {e}")
        return None


def display_best_strategies_report(strategy_reports, ticker, timeframe):
    """Display and export best strategies report"""
    import streamlit as st

    # Shortlist strategies with strong risk-adjusted returns (Sharpe >1.0),
    # low drawdowns (<15%), and a profit factor >1.5
    if strategy_reports:
        # Create DataFrame for display (without raw params dict)
        display_df = pd.DataFrame(
            [
                {k: v for k, v in report.items() if k != "Params"}
                for report in strategy_reports
                if report.get("Win Rate (%)", 0) > 50
                and report.get("Total Trades", 0) > 10
                and abs(report.get("Avg Winning Trade", 0))
                > abs(report.get("Avg Losing Trade", 0))
            ]
        )

        if display_df.empty:
            st.info(
                "No strategies met the criteria (Win Rate > 50%, > 10 trades, Avg Winning Trade > Avg Losing Trade)"
            )
            return pd.DataFrame()

        # Sort by Win Rate and Total Return for better visualization
        display_df = display_df.sort_values(
            by=["Win Rate (%)", "Total Return (%)"], ascending=False
        )

        # Create detailed DataFrame for export
        export_df = pd.DataFrame(strategy_reports)
        if "Params" in export_df.columns:
            params_df = pd.json_normalize(export_df["Params"])
            if not params_df.empty:
                export_df = pd.concat(
                    [export_df.drop(["Params"], axis=1), params_df], axis=1
                )
            else:
                export_df = export_df.drop(["Params"], axis=1)

        st.subheader(f"📊 Best Performing Strategies for {ticker} ({timeframe})")
        st.write(
            "Strategies with Win Rate > 50%, > 10 trades, and Avg Winning Trade > Avg Losing Trade, sorted by Win Rate and Total Return"
        )
        st.dataframe(display_df, use_container_width=True)

        # Create CSV and download button
        uuid_str = str(uuid.uuid4())
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Full Report (CSV)",
            data=csv,
            file_name=f"{ticker}_{timeframe}_best_strategies.csv",
            mime="text/csv",
            key=f"{uuid_str}_export_best_strategies",
        )
        # Display a chart of key metrics for top strategies using Plotly
        if not display_df.empty:
            st.write("### 📊 Strategy Performance Comparison")
            chart_data = display_df[
                [
                    "Strategy",
                    "Win Rate (%)",
                    "Total Return (%)",
                    "Sharpe Ratio",
                    "Total P&L",
                ]
            ]

            # Create a grouped bar chart with Plotly
            fig = go.Figure(
                data=[
                    go.Bar(
                        name="Win Rate (%)",
                        x=chart_data["Strategy"],
                        y=chart_data["Win Rate (%)"],
                        marker_color="rgba(75, 192, 192, 0.6)",
                        marker_line_color="rgba(75, 192, 192, 1)",
                        marker_line_width=1,
                    ),
                    go.Bar(
                        name="Total Return (%)",
                        x=chart_data["Strategy"],
                        y=chart_data["Total Return (%)"],
                        marker_color="rgba(54, 162, 235, 0.6)",
                        marker_line_color="rgba(54, 162, 235, 1)",
                        marker_line_width=1,
                    ),
                    go.Bar(
                        name="Sharpe Ratio",
                        x=chart_data["Strategy"],
                        y=chart_data["Sharpe Ratio"],
                        marker_color="rgba(255, 99, 132, 0.6)",
                        marker_line_color="rgba(255, 99, 132, 1)",
                        marker_line_width=1,
                    ),
                    go.Bar(
                        name="Total P&L",
                        x=chart_data["Strategy"],
                        y=chart_data["Total P&L"],
                        marker_color="rgba(153, 102, 255, 0.6)",
                        marker_line_color="rgba(153, 102, 255, 1)",
                        marker_line_width=1,
                    ),
                ]
            )

            fig.update_layout(
                barmode="group",
                title=f"Performance Metrics for {ticker} ({timeframe})",
                xaxis_title="Strategy",
                yaxis_title="Value",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
                margin=dict(t=100),
                template=(
                    "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly"
                ),
            )

            st.plotly_chart(fig, use_container_width=True, key=uuid.uuid4())
        return display_df
    else:
        st.info(
            "No strategies met the criteria (Win Rate > 50%, > 10 trades, Avg Winning Trade > Avg Losing Trade)"
        )
        return pd.DataFrame()
