"""
Table generation functions for the Streamlit application.
"""

import pandas as pd
import numpy as np
import logging
import pytz
import backtrader as bt
from datetime import datetime
from comprehensive_backtesting.reports import PerformanceAnalyzer
from .config import IST
from .utils import clean_value, clean_numeric, get_strategy, is_possible_number

logger = logging.getLogger(__name__)


def create_summary_table(report):
    """Convert JSON report into a structured table for display with consistent data types."""
    try:
        # Extract summary and trade analysis
        summary = report.get("summary", {})
        trade_analysis = report.get("trade_analysis", {})
        risk_metrics = report.get("risk_metrics", {})

        # Prepare table data
        table_data = []

        # Add summary metrics
        for key, value in summary.items():
            # Handle date formatting
            if key in ["start_date", "end_date"]:
                value = str(value)  # Convert dates to string
            cleaned_value = clean_value(value)
            table_data.append(
                {
                    "Category": "Summary",
                    "Metric": key.replace("_", " ").title(),
                    "Value": cleaned_value,
                }
            )

        # Add trade analysis metrics
        for key, value in trade_analysis.items():
            # Skip nested dicts and lists
            if isinstance(value, (dict, list)):
                continue

            cleaned_value = clean_value(value)
            table_data.append(
                {
                    "Category": "Trade Analysis",
                    "Metric": key.replace("_", " ").title(),
                    "Value": cleaned_value,
                }
            )

        # Add risk metrics
        for key, value in risk_metrics.items():
            cleaned_value = clean_value(value)
            table_data.append(
                {
                    "Category": "Risk Metrics",
                    "Metric": key.replace("_", " ").title(),
                    "Value": cleaned_value,
                }
            )

        # Create DataFrame
        df = pd.DataFrame(table_data)

        # Ensure Arrow compatibility: if df is empty, return as is
        if df.empty:
            return df

        # For the 'Value' column, if it is supposed to be numeric, coerce errors to NaN
        # Only coerce if the column is not all strings (dates/labels)
        if "Value" in df.columns:
            # Only convert if at least one value is numeric
            if df["Value"].apply(is_possible_number).any():
                df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        return df

    except Exception as e:
        import streamlit as st

        st.info("No trades executed during the backtest period.")
        return pd.DataFrame()


def create_parameters_table(best_params_info):
    """Create a table for best parameters analysis with proper data types."""
    if "error" in best_params_info:
        return pd.DataFrame()

    try:
        params_data = []
        for param, value in best_params_info["best_parameters"].items():
            importance = best_params_info["parameter_importance"].get(param, 0)

            cleaned_value = clean_numeric(value)
            cleaned_importance = clean_numeric(importance)

            params_data.append(
                {
                    "Parameter": param,
                    "Best Value": cleaned_value,
                    "Importance": cleaned_importance,
                }
            )

        df = pd.DataFrame(params_data)

        # Sort by importance, treating None as 0 for sorting
        return df.sort_values("Importance", ascending=False, key=lambda x: x.fillna(0))

    except Exception as e:
        logger.error(f"Error creating parameters table: {e}")
        return pd.DataFrame()


def extract_indicator_values_from_strategy(
    strategy_result: bt.Strategy, target_datetime: datetime
):
    """Extract indicator values from strategy at a specific datetime."""
    indicator_values = {}

    try:
        # Convert target datetime to backtrader's internal format
        target_dt_num = bt.date2num(target_datetime)

        # Get the data feed from the strategy
        data_feed = strategy_result.data0 if hasattr(strategy_result, "data0") else None
        if data_feed is None:
            return indicator_values

        # Find the closest bar index to the target datetime
        closest_bar_idx = None
        min_diff = float("inf")

        # Search through all bars to find the closest match
        for i in range(len(data_feed)):
            try:
                bar_dt_num = data_feed.datetime[i]
                diff = abs(bar_dt_num - target_dt_num)
                if diff < min_diff:
                    min_diff = diff
                    closest_bar_idx = i
            except IndexError:
                continue

        if closest_bar_idx is None:
            return indicator_values

        logger.debug(
            f"Target datetime: {target_datetime}, closest bar index: {closest_bar_idx}"
        )

        # Extract indicator values at the closest bar
        for attr_name in dir(strategy_result):
            if attr_name.startswith("_") or attr_name in [
                "data",
                "data0",
                "data1",
                "broker",
            ]:
                continue

            try:
                attr = getattr(strategy_result, attr_name)

                # Check if it's an indicator by looking for lines attribute
                if (
                    hasattr(attr, "lines")
                    and hasattr(attr, "_clock")
                    and hasattr(attr, "_owner")
                    and callable(getattr(attr, "__call__", None))
                ):

                    indicator_name = attr_name

                    # Get the indicator value at the closest bar
                    try:
                        # BackTrader indicators use negative indexing from current position
                        # 0 is current bar, -1 is previous bar, etc.
                        # We need to calculate how many bars back from the current position

                        # Get the total length of the indicator
                        indicator_length = len(attr)

                        # Calculate the offset from the current position
                        # If closest_bar_idx is 0, we want the most recent value (index 0)
                        # If closest_bar_idx is 1, we want the previous value (index -1)
                        bars_back = indicator_length - 1 - closest_bar_idx

                        # Access the indicator value
                        if bars_back >= 0 and bars_back < indicator_length:
                            if bars_back == 0:
                                value = attr[0]  # Current value
                            else:
                                value = attr[-bars_back]  # Historical value

                            if value is not None and not (
                                isinstance(value, float) and np.isnan(value)
                            ):
                                indicator_values[indicator_name] = float(value)
                                logger.debug(
                                    f"Extracted {indicator_name}: {value} at bars_back={bars_back}"
                                )

                    except (IndexError, ValueError, AttributeError) as e:
                        logger.debug(f"Error accessing indicator {indicator_name}: {e}")
                        continue

            except Exception as e:
                logger.debug(f"Error accessing attribute {attr_name}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Error extracting indicator values: {e}")

    return indicator_values


def extract_trades(
    strategy_result: bt.Strategy, data: pd.DataFrame = None
) -> pd.DataFrame:
    """Extract trades from BackTrader strategy result with indicator values."""
    try:
        trades = []

        # Ensure data index is a DatetimeIndex and in UTC if provided
        if data is not None and not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        if data is not None and data.index.tz is None:
            data_index = data.index.tz_localize("UTC")
        elif data is not None:
            data_index = data.index.tz_convert("UTC")
        else:
            data_index = None

        # Get all indicators from the strategy for reference
        strategy_indicators = {}
        for attr_name in dir(strategy_result):
            if not attr_name.startswith("_"):
                try:
                    attr = getattr(strategy_result, attr_name)
                    if hasattr(attr, "__class__") and "backtrader.indicators" in str(
                        type(attr)
                    ):
                        strategy_indicators[attr_name] = attr
                        logger.debug(
                            f"Found indicator: {attr_name} - {attr.__class__.__name__}"
                        )
                except Exception as e:
                    continue

        # Method 1: Extract from completed orders (most reliable)
        if hasattr(strategy_result, "broker") and hasattr(
            strategy_result.broker, "orders"
        ):
            orders = strategy_result.broker.orders
            completed_orders = [
                order for order in orders if order.status == 4
            ]  # Status 4 = Completed

            open_positions = []  # Stack of open long positions
            open_short_positions = []  # Stack of open short positions

            for order in completed_orders:
                if hasattr(order, "executed") and order.executed.size != 0:
                    exec_dt = bt.num2date(order.executed.dt)
                    exec_price = order.executed.price
                    exec_size = order.executed.size
                    commission = order.executed.comm

                    # Convert execution datetime to UTC if needed
                    if exec_dt.tzinfo is None:
                        exec_dt = exec_dt.replace(tzinfo=pytz.UTC)
                    else:
                        exec_dt = exec_dt.astimezone(pytz.UTC)

                    # Initialize trade dictionary
                    trade_info = {
                        "entry_time": None,
                        "exit_time": exec_dt,
                        "entry_price": None,
                        "exit_price": exec_price,
                        "size": abs(exec_size),
                        "pnl": 0,
                        "pnl_net": 0,
                        "commission": commission,
                        "status": None,
                        "direction": "Long" if exec_size > 0 else "Short",
                    }

                    if exec_size > 0:  # Buy order (Long entry or Short exit)
                        if open_short_positions:  # Closing short position
                            short_entry = open_short_positions.pop(0)
                            size = min(abs(short_entry["size"]), exec_size)
                            pnl = (short_entry["price"] - exec_price) * size
                            total_comm = short_entry["commission"] + commission

                            trade_info.update(
                                {
                                    "entry_time": short_entry["datetime"],
                                    "entry_price": short_entry["price"],
                                    "pnl": pnl,
                                    "pnl_net": pnl - total_comm,
                                    "commission": total_comm,
                                    "status": "Won" if pnl > 0 else "Lost",
                                    "direction": "Short",
                                }
                            )

                            # Extract indicator values for entry and exit times
                            if trade_info["entry_time"]:
                                entry_indicators = (
                                    extract_indicator_values_from_strategy(
                                        strategy_result, trade_info["entry_time"]
                                    )
                                )
                                for ind_name, ind_value in entry_indicators.items():
                                    trade_info[f"{ind_name}_entry"] = ind_value

                            if trade_info["exit_time"]:
                                exit_indicators = (
                                    extract_indicator_values_from_strategy(
                                        strategy_result, trade_info["exit_time"]
                                    )
                                )
                                for ind_name, ind_value in exit_indicators.items():
                                    trade_info[f"{ind_name}_exit"] = ind_value

                            trades.append(trade_info)
                        else:  # Opening long position
                            open_positions.append(
                                {
                                    "datetime": exec_dt,
                                    "price": exec_price,
                                    "size": exec_size,
                                    "commission": commission,
                                }
                            )

                    else:  # Sell order (Long exit or Short entry)
                        if open_positions:  # Closing long position
                            long_entry = open_positions.pop(0)
                            size = min(long_entry["size"], abs(exec_size))
                            pnl = (exec_price - long_entry["price"]) * size
                            total_comm = long_entry["commission"] + abs(commission)

                            trade_info.update(
                                {
                                    "entry_time": long_entry["datetime"],
                                    "entry_price": long_entry["price"],
                                    "pnl": pnl,
                                    "pnl_net": pnl - total_comm,
                                    "commission": total_comm,
                                    "status": "Won" if pnl > 0 else "Lost",
                                    "direction": "Long",
                                }
                            )

                            # Extract indicator values for entry and exit times
                            if trade_info["entry_time"]:
                                entry_indicators = (
                                    extract_indicator_values_from_strategy(
                                        strategy_result, trade_info["entry_time"]
                                    )
                                )
                                for ind_name, ind_value in entry_indicators.items():
                                    trade_info[f"{ind_name}_entry"] = ind_value

                            if trade_info["exit_time"]:
                                exit_indicators = (
                                    extract_indicator_values_from_strategy(
                                        strategy_result, trade_info["exit_time"]
                                    )
                                )
                                for ind_name, ind_value in exit_indicators.items():
                                    trade_info[f"{ind_name}_exit"] = ind_value

                            trades.append(trade_info)
                        else:  # Opening short position
                            open_short_positions.append(
                                {
                                    "datetime": exec_dt,
                                    "price": exec_price,
                                    "size": exec_size,
                                    "commission": abs(commission),
                                }
                            )

        # Method 2: Fallback - try to access _trades directly from strategy
        if not trades and hasattr(strategy_result, "_trades"):
            logger.info(
                f"Found {len(strategy_result._trades)} trades in strategy._trades"
            )

            for trade_obj in strategy_result._trades:
                if hasattr(trade_obj, "isclosed") and trade_obj.isclosed:
                    try:
                        entry_dt = (
                            bt.num2date(trade_obj.dtopen)
                            if hasattr(trade_obj, "dtopen")
                            else None
                        )
                        exit_dt = (
                            bt.num2date(trade_obj.dtclose)
                            if hasattr(trade_obj, "dtclose")
                            else None
                        )

                        # Ensure timezone consistency
                        if entry_dt and entry_dt.tzinfo is None:
                            entry_dt = entry_dt.replace(tzinfo=pytz.UTC)
                        if exit_dt and exit_dt.tzinfo is None:
                            exit_dt = exit_dt.replace(tzinfo=pytz.UTC)

                        trade_info = {
                            "entry_time": entry_dt,
                            "exit_time": exit_dt,
                            "entry_price": getattr(trade_obj, "price", 0),
                            "exit_price": (
                                getattr(trade_obj, "price", 0)
                                + (
                                    getattr(trade_obj, "pnl", 0)
                                    / getattr(trade_obj, "size", 1)
                                )
                                if getattr(trade_obj, "size", 1) != 0
                                else getattr(trade_obj, "price", 0)
                            ),
                            "size": abs(getattr(trade_obj, "size", 0)),
                            "pnl": getattr(trade_obj, "pnl", 0),
                            "pnl_net": getattr(
                                trade_obj, "pnlcomm", getattr(trade_obj, "pnl", 0)
                            ),
                            "commission": getattr(trade_obj, "commission", 0),
                            "status": (
                                "Won" if getattr(trade_obj, "pnl", 0) > 0 else "Lost"
                            ),
                            "direction": (
                                "Long" if getattr(trade_obj, "size", 0) > 0 else "Short"
                            ),
                        }

                        # Extract indicator values for entry and exit
                        if entry_dt:
                            entry_indicators = extract_indicator_values_from_strategy(
                                strategy_result, entry_dt
                            )
                            for ind_name, ind_value in entry_indicators.items():
                                trade_info[f"{ind_name}_entry"] = ind_value

                        if exit_dt:
                            exit_indicators = extract_indicator_values_from_strategy(
                                strategy_result, exit_dt
                            )
                            for ind_name, ind_value in exit_indicators.items():
                                trade_info[f"{ind_name}_exit"] = ind_value

                        trades.append(trade_info)
                    except Exception as e:
                        logger.warning(f"Error processing trade: {e}")
                        continue

        # Method 3: Alternative approach using strategy's trade analyzer if available
        if not trades and hasattr(strategy_result, "analyzers"):
            try:
                # Look for trade analyzer results
                for analyzer_name in dir(strategy_result.analyzers):
                    if not analyzer_name.startswith("_"):
                        analyzer = getattr(strategy_result.analyzers, analyzer_name)
                        if hasattr(analyzer, "get_analysis"):
                            analysis = analyzer.get_analysis()
                            logger.debug(f"Found analyzer {analyzer_name}: {analysis}")
            except Exception as e:
                logger.debug(f"Error accessing analyzers: {e}")

        # Method 4: Fallback to summary trade analysis if available
        if not trades and strategy_result.get("trade_analysis"):
            trades = strategy_result.get("trade_analysis").get("trades", [])
            for i, trade in enumerate(trades):
                pnl = trade.get("pnl", 0)
                commission = trade.get("commission", 0)
                pnl_comm = (
                    pnl - commission
                    if pnl is not None and commission is not None
                    else pnl
                )
                trade_info = {
                    "trade_id": trade.get("ref", i),
                    "entry_time": trade.get("entry_time"),
                    "exit_time": trade.get("exit_time"),
                    "size": trade.get("size", 0),
                    "entry_price": trade.get("entry_price", 0),
                    "exit_price": trade.get("exit_price", 0),
                    "pnl": pnl,
                    "pnl_net": pnl_comm,
                    "direction": trade.get("direction"),
                    "commission": commission,
                    "status": trade.get("status"),
                    "bar_held": trade.get("bar_held", None),
                }
                if len(trades) < 100:
                    trades.append(trade_info)

        # Add regime information with robust timezone handling
        if data is not None and "vol_regime" in data.columns and trades:
            for trade in trades:
                try:
                    entry_time = trade["entry_time"]

                    if not isinstance(entry_time, pd.Timestamp):
                        entry_time = pd.Timestamp(entry_time)
                    if entry_time.tzinfo is None:
                        entry_time = entry_time.tz_localize("UTC")
                    else:
                        entry_time = entry_time.tz_convert("UTC")

                    time_diff = data_index - entry_time
                    abs_time_diff = np.abs(time_diff.total_seconds())
                    min_diff_seconds = abs_time_diff.min()

                    if min_diff_seconds <= 60:  # Within 1 minute
                        closest_idx = abs_time_diff.argmin()
                except Exception as e:
                    logger.warning(f"Regime assignment failed: {str(e)}")

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            logger.info(
                f"Extracted {len(trades_df)} trades with columns: {trades_df.columns.tolist()}"
            )

            # Log indicator columns found
            indicator_columns = [
                col
                for col in trades_df.columns
                if col.endswith("_entry") or col.endswith("_exit")
            ]
            if indicator_columns:
                logger.info(f"Indicator columns: {indicator_columns}")
            else:
                logger.warning("No indicator columns found in trades")
        else:
            logger.warning("No trades extracted")

        return trades_df

    except Exception as e:
        logger.error(f"Trade extraction failed: {e}")
        return pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "size",
                "pnl",
                "pnl_net",
                "commission",
                "status",
                "direction",
            ]
        )


def create_trades_table(results, data=None):
    """Create a comprehensive trades table with all trade details including indicator values."""
    try:
        strategy = get_strategy(results)
        logger.info(
            f"[create_trades_table] Using strategy: {getattr(strategy, '__class__', type(strategy)).__name__}"
        )

        # Extract trades using the robust method
        trades_df = extract_trades(strategy, data)
        if trades_df.empty:
            logger.warning(
                "[create_trades_table] No trades extracted using extract_trades"
            )
            return pd.DataFrame(), "No trades executed during the period"

        numeric_cols = [
            "entry_price",
            "exit_price",
            "size",
            "pnl",
            "pnl_net",
            "commission",
        ]
        for col in numeric_cols:
            if col in trades_df.columns:
                trades_df[col] = pd.to_numeric(trades_df[col], errors="coerce")

        # Only convert if not already tz-aware and in IST
        for col in ["entry_time", "exit_time"]:
            if col in trades_df.columns:
                trades_df[col] = pd.to_datetime(trades_df[col], errors="coerce")
                if trades_df[col].dt.tz is None:
                    trades_df[col] = (
                        trades_df[col].dt.tz_localize("UTC").dt.tz_convert(IST)
                    )
                else:
                    trades_df[col] = trades_df[col].dt.tz_convert(IST)

        trades_df["duration"] = trades_df["exit_time"] - trades_df["entry_time"]
        trades_df["duration_hours"] = trades_df["duration"].dt.total_seconds() / 3600
        trades_df["duration_days"] = trades_df["duration"].dt.days
        # Avoid division by zero
        with pd.option_context("mode.use_inf_as_na", True):
            trades_df["return_pct"] = np.where(
                (trades_df["entry_price"] * trades_df["size"] != 0),
                (trades_df["pnl_net"] / (trades_df["entry_price"] * trades_df["size"]))
                * 100,
                0,
            )

        # --- Vectorized formatting for trade details ---
        base_info = pd.DataFrame(
            {
                "Trade #": np.arange(1, len(trades_df) + 1),
                "Entry Date": trades_df["entry_time"].dt.strftime("%Y-%m-%d"),
                "Entry Time": trades_df["entry_time"].dt.strftime("%H:%M:%S"),
                "Entry Price": trades_df["entry_price"].round(2).astype(str),
                "Exit Date": trades_df["exit_time"].dt.strftime("%Y-%m-%d"),
                "Exit Time": trades_df["exit_time"].dt.strftime("%H:%M:%S"),
                "Exit Price": trades_df["exit_price"].round(2).astype(str),
                "Size": trades_df["size"].astype(int),
                "Direction": trades_df["direction"].astype(str),
                "P&L": trades_df["pnl_net"].round(2).astype(str),
                "Return %": trades_df["return_pct"].round(2).astype(str) + "%",
                "Duration (Hours)": trades_df["duration_hours"].round(1).astype(str),
                "Duration (Days)": trades_df["duration"].dt.days,
                "Status": trades_df["status"].astype(str),
            }
        )

        # --- Indicator columns (entry/exit) ---
        indicator_cols = [
            col
            for col in trades_df.columns
            if col.endswith("_entry") or col.endswith("_exit")
        ]
        indicator_info = {}
        for col in indicator_cols:
            indicator_name = col.rsplit("_", 1)[0]
            context = "Entry" if col.endswith("_entry") else "Exit"
            display_name = f"{indicator_name} ({context})"
            # Format as string with 2 decimals if numeric
            vals = trades_df[col]
            if np.issubdtype(vals.dtype, np.number):
                indicator_info[display_name] = vals.round(2).astype(str)
            else:
                indicator_info[display_name] = vals.astype(str)

        # --- Combine all columns ---
        df = pd.concat(
            [base_info] + ([pd.DataFrame(indicator_info)] if indicator_info else []),
            axis=1,
        )

        # Order columns: base first, then entry indicators, then exit indicators
        base_columns = [
            "Trade #",
            "Entry Date",
            "Entry Time",
            "Entry Price",
            "Exit Date",
            "Exit Time",
            "Exit Price",
            "Size",
            "Direction",
            "P&L",
            "Return %",
            "Duration (Hours)",
            "Duration (Days)",
            "Status",
        ]
        indicator_columns = [col for col in df.columns if col not in base_columns]
        entry_indicators = sorted([col for col in indicator_columns if "Entry" in col])
        exit_indicators = sorted([col for col in indicator_columns if "Exit" in col])
        ordered_columns = base_columns + entry_indicators + exit_indicators

        return df[ordered_columns], None
    except Exception as e:
        logger.error(f"Error creating trades table: {e}", exc_info=True)
        return pd.DataFrame(), f"Error creating trades table: {str(e)}"


def create_best_times_table(time_analysis):
    """Create tables for best trading times analysis."""
    if "error" in time_analysis:
        return None, None, None

    try:
        # Best Hours Table
        hours_data = []
        if time_analysis.get("hourly_distribution"):
            for hour, stats in time_analysis["hourly_distribution"].items():
                hours_data.append(
                    {
                        "Hour": f"{hour:02d}:00",
                        "Winning Trades": stats["count"],
                        "Total P&L": f"{stats['total_pnl']:.2f}",
                        "Avg P&L per Trade": (
                            f"{stats['total_pnl']/stats['count']:.2f}"
                            if stats["count"] > 0
                            else "0.00"
                        ),
                    }
                )
        hours_df = (
            pd.DataFrame(hours_data).sort_values("Winning Trades", ascending=False)
            if hours_data
            else pd.DataFrame()
        )

        # Best Days Table
        days_data = []
        if time_analysis.get("daily_distribution"):
            for day, stats in time_analysis["daily_distribution"].items():
                days_data.append(
                    {
                        "Day of Week": day,
                        "Winning Trades": stats["count"],
                        "Total P&L": f"{stats['total_pnl']:.2f}",
                        "Avg P&L per Trade": (
                            f"{stats['total_pnl']/stats['count']:.2f}"
                            if stats["count"] > 0
                            else "0.00"
                        ),
                    }
                )
        days_df = (
            pd.DataFrame(days_data).sort_values("Winning Trades", ascending=False)
            if days_data
            else pd.DataFrame()
        )

        # Best Months Table
        months_data = []
        if time_analysis.get("monthly_distribution"):
            for month, stats in time_analysis["monthly_distribution"].items():
                months_data.append(
                    {
                        "Month": month,
                        "Winning Trades": stats["count"],
                        "Total P&L": f"{stats['total_pnl']:.2f}",
                        "Avg P&L per Trade": (
                            f"{stats['total_pnl']/stats['count']:.2f}"
                            if stats["count"] > 0
                            else "N/A"
                        ),
                    }
                )
        months_df = (
            pd.DataFrame(months_data).sort_values("Winning Trades", ascending=False)
            if months_data
            else pd.DataFrame()
        )

        return hours_df, days_df, months_df

    except Exception as e:
        logger.error(f"Error creating best times tables: {e}")
        return None, None, None


def create_parameter_evolution_table(wf_results):
    """Create a table showing parameter evolution across walk-forward windows.

    Args:
        wf_results (dict): Walk-forward analysis results containing windows data

    Returns:
        pd.DataFrame: Table showing parameter evolution across windows
    """
    if not wf_results or "windows" not in wf_results:
        return pd.DataFrame()

    windows = wf_results["windows"]
    valid_windows = [w for w in windows if w.get("valid", False)]

    if not valid_windows:
        return pd.DataFrame()

    # Collect all parameter names across windows
    all_params = set()
    for window in valid_windows:
        if "best_params" in window:
            all_params |= set(window["best_params"].keys())

    # Create a row for each window
    rows = []
    for i, window in enumerate(valid_windows):
        row = {"Window": i + 1}
        # Add out-sample performance if available
        if "out_sample_performance" in window:
            perf = window["out_sample_performance"].get("summary", {})
            row["Return (%)"] = perf.get("total_return_pct", 0)
            row["Sharpe Ratio"] = perf.get("sharpe_ratio", 0)
            row["Max Drawdown (%)"] = perf.get("max_drawdown_pct", 0)

        # Add parameters
        params = window.get("best_params", {})
        for param in all_params:
            row[param] = params.get(param, None)

        rows.append(row)

    return pd.DataFrame(rows)


def create_strategy_comparison_table(basic_report, opt_report):
    """Create a comparison table between basic and optimized strategies."""
    try:
        comparison_data = []

        # Basic Strategy Data
        basic_summary = basic_report.get("summary", {})
        basic_trades = basic_report.get("trade_analysis", {})

        # Optimized Strategy Data
        opt_summary = opt_report.get("summary", {})
        opt_trades = opt_report.get("trade_analysis", {})

        metrics = [
            (
                "Total Return (%)",
                basic_summary.get("total_return_pct", 0),
                opt_summary.get("total_return_pct", 0),
            ),
            (
                "Sharpe Ratio",
                basic_summary.get("sharpe_ratio", 0),
                opt_summary.get("sharpe_ratio", 0),
            ),
            (
                "Max Drawdown (%)",
                basic_summary.get("max_drawdown_pct", 0),
                opt_summary.get("max_drawdown_pct", 0),
            ),
            (
                "Total Trades",
                basic_trades.get("total_trades", 0),
                opt_trades.get("total_trades", 0),
            ),
            (
                "Win Rate (%)",
                basic_trades.get("win_rate_percent", 0),
                opt_trades.get("win_rate_percent", 0),
            ),
            (
                "Profit Factor",
                basic_trades.get("profit_factor", 0),
                opt_trades.get("profit_factor", 0),
            ),
            (
                "Final Value ()",
                basic_summary.get("final_value", 0),
                opt_summary.get("final_value", 0),
            ),
        ]

        for metric, basic_val, opt_val in metrics:
            improvement = ""
            if (
                isinstance(basic_val, (int, float))
                and isinstance(opt_val, (int, float))
                and basic_val != 0
            ):
                if metric == "Max Drawdown (%)":  # Lower is better
                    pct_change = ((basic_val - opt_val) / abs(basic_val)) * 100
                else:  # Higher is better
                    pct_change = ((opt_val - basic_val) / abs(basic_val)) * 100
                improvement = f"{pct_change:+.1f}%"

            comparison_data.append(
                {
                    "Metric": metric,
                    "Basic Strategy": (
                        f"{basic_val:.2f}"
                        if isinstance(basic_val, float)
                        else str(basic_val)
                    ),
                    "Optimized Strategy": (
                        f"{opt_val:.2f}" if isinstance(opt_val, float) else str(opt_val)
                    ),
                    "Improvement": improvement,
                }
            )

        return pd.DataFrame(comparison_data)

    except Exception as e:
        logger.error(f"Error creating comparison table: {e}")
        return pd.DataFrame()
