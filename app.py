"""
Comprehensive Backtesting Framework - Streamlit Application

This application provides a dynamic backtesting framework with the following features:

DYNAMIC TICKER MANAGEMENT:
- Load tickers from external file (tickers.txt) or use defaults
- Add/remove tickers through the UI
- Import/export ticker lists
- Custom ticker input with validation
- Automatic ticker format validation

TICKER CONFIGURATION:
- Default tickers are defined in DEFAULT_TICKERS list
- External ticker file: tickers.txt (one ticker per line)
- Supports various ticker formats (e.g., AAPL, RELIANCE.NS, BRK-A)

USAGE:
1. Select from predefined ticker list or enter custom ticker
2. Manage ticker lists through the sidebar expander
3. Import/export ticker lists as needed
4. All ticker operations are validated for format correctness
"""

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import optuna
import optuna.visualization
import backtrader as bt
from io import BytesIO
import matplotlib
from comprehensive_backtesting.parameter_optimization import SortinoRatio
from comprehensive_backtesting.utils import DEFAULT_TICKERS
from comprehensive_backtesting.validation import ValidationAnalyzer

matplotlib.use("Agg")
import logging
from datetime import datetime, timedelta
import pytz
from comprehensive_backtesting.registry import STRATEGY_REGISTRY, get_strategy
from comprehensive_backtesting.data import get_data_sync
from comprehensive_backtesting.reports import PerformanceAnalyzer, compare_strategies
import json
import numpy as np
from comprehensive_backtesting.parameter_optimization import optimize_strategy
from comprehensive_backtesting.backtesting import (
    run_basic_backtest,
    run_complete_backtest,
    run_parameter_optimization,
    run_walkforward_analysis,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set timezone for IST
IST = pytz.timezone("Asia/Kolkata")


def load_tickers_from_file(file_path="tickers.txt"):
    """Load tickers from a text file, one ticker per line."""
    try:
        with open(file_path, "r") as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        return tickers
    except FileNotFoundError:
        logger.info(f"Ticker file {file_path} not found, using default tickers")
        return DEFAULT_TICKERS
    except Exception as e:
        logger.error(f"Error loading tickers from file: {e}")
        return DEFAULT_TICKERS


def save_tickers_to_file(tickers, file_path="tickers.txt"):
    """Save tickers to a text file, one ticker per line."""
    try:
        with open(file_path, "w") as f:
            for ticker in tickers:
                f.write(f"{ticker.strip().upper()}\n")
        return True
    except Exception as e:
        logger.error(f"Error saving tickers to file: {e}")
        return False


def get_available_tickers():
    """Get list of available tickers. Loads from file if available, otherwise uses defaults."""
    return load_tickers_from_file()


def validate_ticker_format(ticker):
    """Validate ticker symbol format."""
    if not ticker:
        return False, "Ticker cannot be empty"

    ticker = ticker.strip()
    if len(ticker) < 1:
        return False, "Ticker too short"

    if len(ticker) > 20:
        return False, "Ticker too long (max 20 characters)"

    # Allow alphanumeric characters, dots, hyphens, and underscores
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
    if not all(c in allowed_chars for c in ticker.upper()):
        return (
            False,
            "Invalid characters in ticker. Use only letters, numbers, dots, hyphens, and underscores",
        )

    return True, "Valid ticker format"


def display_best_parameters(study_results):
    """Display best parameters from optimization study."""
    try:
        if not study_results or "study" not in study_results:
            return {"error": "No optimization study available"}
        study = study_results["study"]
        if not hasattr(study, "best_params") or not study.best_params:
            return {"error": "No best parameters found"}
        best_params = study.best_params
        best_value = getattr(study, "best_value", None)
        # Get parameter importance if available
        param_importance = {}
        try:
            param_importance = optuna.importance.get_param_importances(study)
        except Exception:
            param_importance = {}
        # Get trial statistics
        completed_trials = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        return {
            "best_parameters": best_params,
            "best_objective_value": best_value,
            "parameter_importance": param_importance,
            "total_trials": len(study.trials),
            "completed_trials": len(completed_trials),
            "success_rate": (
                (len(completed_trials) / len(study.trials)) * 100 if study.trials else 0
            ),
        }
    except Exception as e:
        logger.error(f"Error displaying best parameters: {e}")
        return {"error": str(e)}


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


def get_first_strategy(results):
    # Accepts results as [strategy], [[strategy]], or strategy
    if isinstance(results, list):
        if len(results) > 0 and isinstance(results[0], list):
            return results[0][0]
        elif len(results) > 0:
            return results[0]
    return results


def analyze_best_trades(results):
    """Analyze and extract best performing trades."""
    try:
        strategy = get_first_strategy(results)
        # Remove demo-related code and strings from the codebase
        # Robust trade extraction from PerformanceAnalyzer if available
        from comprehensive_backtesting.reports import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer(results)
        trade_analysis = analyzer.get_trade_analysis()
        trades = trade_analysis.get("trades", [])
        if trades:
            trade_list = []
            for t in trades:
                try:
                    entry_date = t["entry_date"]
                    exit_date = t["exit_date"]
                    pnl = t.get("pnl", 0)
                    size = t.get("size", 1)
                    trade_info = {
                        "trade_id": t.get("trade_id", None),
                        "entry_date": (
                            str(entry_date) if entry_date is not None else "-"
                        ),
                        "exit_date": str(exit_date) if exit_date is not None else "-",
                        "entry_price": t.get("price_in", 0),
                        "exit_price": t.get("price_out", 0),
                        "pnl": pnl,
                        "size": size,
                        "direction": t.get("direction", ""),
                        "return_pct": (
                            (pnl / (t.get("price_in", 1) * abs(size))) * 100
                            if t.get("price_in", 0) > 0 and size != 0
                            else 0
                        ),
                    }
                    trade_list.append(trade_info)
                except Exception as e:
                    logger.warning(f"Invalid trade data format: {e}")
                    continue
            if trade_list:
                # Sort trades by PnL to find best trades
                trade_list.sort(key=lambda x: x["pnl"], reverse=True)
                best_trades = trade_list[:5]
                total_pnl = sum(trade["pnl"] for trade in trade_list)
                winning_trades = [trade for trade in trade_list if trade["pnl"] > 0]
                losing_trades = [trade for trade in trade_list if trade["pnl"] < 0]
                return {
                    "best_trades": best_trades,
                    "total_trades": len(trade_list),
                    "winning_trades": len(winning_trades),
                    "losing_trades": len(losing_trades),
                    "total_pnl": total_pnl,
                    "avg_winning_trade": (
                        sum(trade["pnl"] for trade in winning_trades)
                        / len(winning_trades)
                        if winning_trades
                        else 0
                    ),
                    "avg_losing_trade": (
                        sum(trade["pnl"] for trade in losing_trades)
                        / len(losing_trades)
                        if losing_trades
                        else 0
                    ),
                    "best_trade_pnl": best_trades[0]["pnl"] if best_trades else 0,
                    "worst_trade_pnl": trade_list[-1]["pnl"] if trade_list else 0,
                }
        # fallback to legacy logic if no trades found
        trades = None
        if hasattr(strategy, "analyzers") and hasattr(
            strategy.analyzers, "tradeanalyzer"
        ):
            trades = strategy.analyzers.tradeanalyzer.get_analysis()
        elif hasattr(strategy, "analyzers") and hasattr(strategy.analyzers, "trades"):
            trades = strategy.analyzers.trades.get_analysis()

        if not trades:
            return {"error": "No trade data available"}

        # Try to extract per-trade list (preferred)
        trade_list = []
        closed_trades = trades.get("closed", []) or trades.get("trades", [])
        if (
            isinstance(closed_trades, list)
            and closed_trades
            and isinstance(closed_trades[0], dict)
        ):
            for i, trade in enumerate(closed_trades):
                try:
                    entry_date = (
                        pd.to_datetime(trade["datein"], unit="s")
                        .tz_localize("UTC")
                        .tz_convert(IST)
                    )
                    exit_date = (
                        pd.to_datetime(trade["dateout"], unit="s")
                        .tz_localize("UTC")
                        .tz_convert(IST)
                    )
                    pnl = trade.get("pnl", 0)

                    trade_info = {
                        "trade_id": i + 1,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "entry_price": trade["pricein"],
                        "exit_price": trade["priceout"],
                        "pnl": pnl,
                        "duration_hours": (exit_date - entry_date).total_seconds()
                        / 3600,
                        "return_pct": (
                            (pnl / (trade["pricein"] * trade.get("size", 1))) * 100
                            if trade["pricein"] > 0
                            else 0
                        ),
                    }
                    trade_list.append(trade_info)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Invalid trade data format: {e}")
                    continue
            if trade_list:
                # Sort trades by PnL to find best trades
                trade_list.sort(key=lambda x: x["pnl"], reverse=True)
                best_trades = trade_list[:5]
                total_pnl = sum(trade["pnl"] for trade in trade_list)
                winning_trades = [trade for trade in trade_list if trade["pnl"] > 0]
                losing_trades = [trade for trade in trade_list if trade["pnl"] < 0]
                return {
                    "best_trades": best_trades,
                    "total_trades": len(trade_list),
                    "winning_trades": len(winning_trades),
                    "losing_trades": len(losing_trades),
                    "total_pnl": total_pnl,
                    "avg_winning_trade": (
                        sum(trade["pnl"] for trade in winning_trades)
                        / len(winning_trades)
                        if winning_trades
                        else 0
                    ),
                    "avg_losing_trade": (
                        sum(trade["pnl"] for trade in losing_trades)
                        / len(losing_trades)
                        if losing_trades
                        else 0
                    ),
                    "best_trade_pnl": best_trades[0]["pnl"] if best_trades else 0,
                    "worst_trade_pnl": trade_list[-1]["pnl"] if trade_list else 0,
                }
        # If no per-trade list, use summary stats from TradeAnalyzer
        if "total" in trades and "won" in trades and "lost" in trades:
            total_trades = trades.get("total", {}).get("total", 0)
            winning_trades = trades.get("won", {}).get("total", 0)
            losing_trades = trades.get("lost", {}).get("total", 0)
            total_pnl = trades.get("pnl", {}).get("net", {}).get("total", 0)
            avg_winning_trade = trades.get("won", {}).get("pnl", {}).get("average", 0)
            avg_losing_trade = trades.get("lost", {}).get("pnl", {}).get("average", 0)
            best_trade_pnl = trades.get("won", {}).get("pnl", {}).get("max", 0)
            worst_trade_pnl = trades.get("lost", {}).get("pnl", {}).get("max", 0)
            # Holding period (bars)
            avg_holding_bars = trades.get("len", {}).get("average", 0)
            avg_holding_won = trades.get("len", {}).get("won", {}).get("average", 0)
            avg_holding_lost = trades.get("len", {}).get("lost", {}).get("average", 0)
            # Compose a pseudo-trade for UI
            best_trades = [
                {
                    "trade_id": 1,
                    "entry_date": None,
                    "exit_date": None,
                    "entry_price": None,
                    "exit_price": None,
                    "pnl": best_trade_pnl,
                    "duration_hours": avg_holding_bars,  # bars, not hours
                    "return_pct": None,
                }
            ]
            return {
                "best_trades": best_trades,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "total_pnl": total_pnl,
                "avg_winning_trade": avg_winning_trade,
                "avg_losing_trade": avg_losing_trade,
                "best_trade_pnl": best_trade_pnl,
                "worst_trade_pnl": worst_trade_pnl,
                "avg_holding_bars": avg_holding_bars,
                "avg_holding_won": avg_holding_won,
                "avg_holding_lost": avg_holding_lost,
            }
        return {"error": "No valid trades found"}
    except Exception as e:
        logger.error(f"Error analyzing best trades: {e}")
        return {"error": str(e)}


def detect_strategy_indicators(strategy):
    """Dynamically detect indicators used by a strategy."""
    indicators = {}
    try:
        for attr_name in dir(strategy):
            if attr_name.startswith("_"):
                continue

            attr = getattr(strategy, attr_name)

            # Improved indicator detection
            if isinstance(attr, bt.Indicator):
                indicators[attr_name] = {
                    "indicator": attr,
                    "type": attr.__class__.__name__,
                    "name": attr_name,
                }

                # Extract parameters
                if hasattr(attr, "params"):
                    params = {}
                    for pname in attr.params._getkeys():
                        try:
                            params[pname] = getattr(attr.params, pname)
                        except AttributeError:
                            continue
                    indicators[attr_name]["params"] = params

        logger.info(f"Detected indicators: {list(indicators.keys())}")
        return indicators

    except Exception as e:
        logger.error(f"Error detecting indicators: {e}")
        return {}


def calculate_indicator_values(data, indicator_info):
    """Calculate indicator values based on detected indicator info."""
    calculated_indicators = {}

    try:
        for name, info in indicator_info.items():
            indicator_type = info["type"]
            params = info.get("params", {})

            if indicator_type == "EMA":
                period = params.get("period", 20)
                calculated_indicators[name] = {
                    "values": data["Close"].ewm(span=period).mean(),
                    "type": "line",
                    "subplot": "price",
                    "color": "#ff6b35" if "fast" in name.lower() else "#004e89",
                    "name": f"{name.upper()} ({period})",
                }

            elif indicator_type == "SMA":
                period = params.get("period", 20)
                calculated_indicators[name] = {
                    "values": data["Close"].rolling(window=period).mean(),
                    "type": "line",
                    "subplot": "price",
                    "color": "#2e8b57",
                    "name": f"{name.upper()} ({period})",
                }

            elif indicator_type == "RSI":
                period = params.get("period", 14)
                rsi_values = calculate_rsi(data["Close"], period)
                calculated_indicators[name] = {
                    "values": rsi_values,
                    "type": "line",
                    "subplot": "oscillator",
                    "color": "#9d4edd",
                    "name": f"{name.upper()} ({period})",
                    "y_range": [0, 100],
                    "levels": {
                        "overbought": params.get("upperband", 70),
                        "oversold": params.get("lowerband", 30),
                        "neutral": 50,
                    },
                }

            elif indicator_type == "MACD":
                # MACD calculation
                fast_period = params.get("period_me1", 12)
                slow_period = params.get("period_me2", 26)
                signal_period = params.get("period_signal", 9)

                ema_fast = data["Close"].ewm(span=fast_period).mean()
                ema_slow = data["Close"].ewm(span=slow_period).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=signal_period).mean()
                histogram = macd_line - signal_line

                calculated_indicators[f"{name}_line"] = {
                    "values": macd_line,
                    "type": "line",
                    "subplot": "oscillator",
                    "color": "#1f77b4",
                    "name": f"MACD Line",
                }
                calculated_indicators[f"{name}_signal"] = {
                    "values": signal_line,
                    "type": "line",
                    "subplot": "oscillator",
                    "color": "#ff7f0e",
                    "name": f"Signal Line",
                }
                calculated_indicators[f"{name}_histogram"] = {
                    "values": histogram,
                    "type": "bar",
                    "subplot": "oscillator",
                    "color": "#2ca02c",
                    "name": f"MACD Histogram",
                }

            elif indicator_type == "BollingerBands":
                period = params.get("period", 20)
                std_dev = params.get("devfactor", 2)

                sma = data["Close"].rolling(window=period).mean()
                std = data["Close"].rolling(window=period).std()
                upper_band = sma + (std * std_dev)
                lower_band = sma - (std * std_dev)

                calculated_indicators[f"{name}_upper"] = {
                    "values": upper_band,
                    "type": "line",
                    "subplot": "price",
                    "color": "#ff0000",
                    "name": f"BB Upper ({period}, {std_dev})",
                    "line_style": "dash",
                }
                calculated_indicators[f"{name}_middle"] = {
                    "values": sma,
                    "type": "line",
                    "subplot": "price",
                    "color": "#0000ff",
                    "name": f"BB Middle ({period})",
                }
                calculated_indicators[f"{name}_lower"] = {
                    "values": lower_band,
                    "type": "line",
                    "subplot": "price",
                    "color": "#ff0000",
                    "name": f"BB Lower ({period}, {std_dev})",
                    "line_style": "dash",
                }

            elif indicator_type == "Stochastic":
                k_period = params.get("period_k", 14)
                d_period = params.get("period_d", 3)

                lowest_low = data["Low"].rolling(window=k_period).min()
                highest_high = data["High"].rolling(window=k_period).max()
                k_percent = 100 * (
                    (data["Close"] - lowest_low) / (highest_high - lowest_low)
                )
                d_percent = k_percent.rolling(window=d_period).mean()

                calculated_indicators[f"{name}_k"] = {
                    "values": k_percent,
                    "type": "line",
                    "subplot": "oscillator",
                    "color": "#ff6b35",
                    "name": f"Stoch %K ({k_period})",
                    "y_range": [0, 100],
                    "levels": {"overbought": 80, "oversold": 20},
                }
                calculated_indicators[f"{name}_d"] = {
                    "values": d_percent,
                    "type": "line",
                    "subplot": "oscillator",
                    "color": "#004e89",
                    "name": f"Stoch %D ({d_period})",
                    "y_range": [0, 100],
                }

        return calculated_indicators

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def create_candlestick_chart_with_trades(
    data, results, title="Candlestick Chart with Trades"
):
    """Create a dynamic candlestick chart with strategy-specific indicators."""
    try:
        strategy = get_first_strategy(results)

        # Dynamically detect indicators used by the strategy
        detected_indicators = detect_strategy_indicators(strategy)
        calculated_indicators = calculate_indicator_values(data, detected_indicators)

        # Determine subplot structure based on detected indicators
        price_indicators = [
            k for k, v in calculated_indicators.items() if v["subplot"] == "price"
        ]
        oscillator_indicators = [
            k for k, v in calculated_indicators.items() if v["subplot"] == "oscillator"
        ]

        # Dynamic subplot configuration
        num_rows = 3  # Base: Price + Volume + Equity
        subplot_titles = ["Price Action & Trades", "Volume", "Equity Curve"]
        row_heights = [0.6, 0.2, 0.2]

        if oscillator_indicators:
            num_rows = 4
            subplot_titles = [
                "Price Action & Trades",
                "Technical Indicators",
                "Volume",
                "Equity Curve",
            ]
            row_heights = [0.5, 0.2, 0.15, 0.15]

        # Create dynamic subplot structure
        fig = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=subplot_titles,
            row_heights=row_heights,
            specs=[[{"secondary_y": False}] for _ in range(num_rows)],
        )

        # Ensure index is DatetimeIndex and convert timezone for display
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        # Filter out rows with missing OHLCV data
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        filtered_data = data.dropna(subset=required_cols)
        dates = (
            filtered_data.index.tz_localize("UTC").tz_convert(IST)
            if filtered_data.index.tz is None
            else filtered_data.index.tz_convert(IST)
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=filtered_data["Open"],
                high=filtered_data["High"],
                low=filtered_data["Low"],
                close=filtered_data["Close"],
                name="OHLC",
                increasing_line_color="#00ff88",
                decreasing_line_color="#ff4444",
                increasing_fillcolor="#00ff88",
                decreasing_fillcolor="#ff4444",
            ),
            row=1,
            col=1,
        )
        # Explicitly set x-axis type to date
        fig.update_xaxes(type="date")
        # Add price-based indicators to the price chart
        for indicator_name, indicator_data in calculated_indicators.items():
            if indicator_data["subplot"] == "price":
                line_style = indicator_data.get("line_style", "solid")
                line_dict = {"color": indicator_data["color"], "width": 2}
                if line_style == "dash":
                    line_dict["dash"] = "dash"

                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=indicator_data["values"],
                        mode="lines",
                        name=indicator_data["name"],
                        line=line_dict,
                        hovertemplate=f'{indicator_data["name"]}: %{{y:.2f}}<extra></extra>',
                    ),
                    row=1,
                    col=1,
                )

        # Add oscillator indicators to separate subplot (if any)
        oscillator_row = 2 if oscillator_indicators else None
        if oscillator_row:
            for indicator_name, indicator_data in calculated_indicators.items():
                if indicator_data["subplot"] == "oscillator":
                    if indicator_data["type"] == "line":
                        fig.add_trace(
                            go.Scatter(
                                x=dates,
                                y=indicator_data["values"],
                                mode="lines",
                                name=indicator_data["name"],
                                line=dict(color=indicator_data["color"], width=2),
                                hovertemplate=f'{indicator_data["name"]}: %{{y:.2f}}<extra></extra>',
                            ),
                            row=oscillator_row,
                            col=1,
                        )
                    elif indicator_data["type"] == "bar":
                        fig.add_trace(
                            go.Bar(
                                x=dates,
                                y=indicator_data["values"],
                                name=indicator_data["name"],
                                marker_color=indicator_data["color"],
                                opacity=0.7,
                            ),
                            row=oscillator_row,
                            col=1,
                        )

                    # Add levels for oscillators
                    if "levels" in indicator_data:
                        levels = indicator_data["levels"]
                        if "overbought" in levels:
                            fig.add_hline(
                                y=levels["overbought"],
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"Overbought ({levels['overbought']})",
                                row=oscillator_row,
                                col=1,
                            )
                        if "oversold" in levels:
                            fig.add_hline(
                                y=levels["oversold"],
                                line_dash="dash",
                                line_color="green",
                                annotation_text=f"Oversold ({levels['oversold']})",
                                row=oscillator_row,
                                col=1,
                            )
                        if "neutral" in levels:
                            fig.add_hline(
                                y=levels["neutral"],
                                line_dash="dot",
                                line_color="gray",
                                annotation_text=f"Neutral ({levels['neutral']})",
                                row=oscillator_row,
                                col=1,
                            )

                    # Set y-axis range for oscillators
                    if "y_range" in indicator_data:
                        fig.update_yaxes(
                            range=indicator_data["y_range"], row=oscillator_row, col=1
                        )

        # Extract and plot trades
        trades = None
        if hasattr(strategy, "analyzers") and hasattr(
            strategy.analyzers, "tradeanalyzer"
        ):
            trades = strategy.analyzers.tradeanalyzer.get_analysis()
        elif hasattr(strategy, "analyzers") and hasattr(strategy.analyzers, "trades"):
            trades = strategy.analyzers.trades.get_analysis()

        buy_dates, buy_prices, sell_dates, sell_prices = [], [], [], []
        trade_lines = []

        if trades:
            closed_trades = trades.get("closed", []) or trades.get("trades", [])
            if isinstance(closed_trades, list):
                for i, trade in enumerate(closed_trades):
                    try:
                        entry_date = (
                            pd.to_datetime(trade["datein"], unit="s")
                            .tz_localize("UTC")
                            .tz_convert(IST)
                        )
                        exit_date = (
                            pd.to_datetime(trade["dateout"], unit="s")
                            .tz_localize("UTC")
                            .tz_convert(IST)
                        )
                        entry_price = trade["pricein"]
                        exit_price = trade["priceout"]
                        pnl = trade.get("pnl", 0)

                        buy_dates.append(entry_date)
                        buy_prices.append(entry_price)
                        sell_dates.append(exit_date)
                        sell_prices.append(exit_price)

                        # Add trade connection line
                        line_color = "green" if pnl > 0 else "red"
                        fig.add_trace(
                            go.Scatter(
                                x=[entry_date, exit_date],
                                y=[entry_price, exit_price],
                                mode="lines",
                                line=dict(color=line_color, width=2, dash="dot"),
                                name=f"Trade {i+1}",
                                showlegend=False,
                                hovertemplate=f"Trade {i+1}<br>P&L: {pnl:.2f}<extra></extra>",
                            ),
                            row=1,
                            col=1,
                        )

                    except (KeyError, TypeError) as e:
                        logger.warning(f"Invalid trade data format: {e}")
                        continue

        # Add buy signals
        if buy_dates:
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="lime",
                        line=dict(color="darkgreen", width=2),
                    ),
                    name="Buy Signal",
                    hovertemplate="Buy<br>Price: %{y:.2f}<br>Date: %{x}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Add sell signals
        if sell_dates:
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="red",
                        line=dict(color="darkred", width=2),
                    ),
                    name="Sell Signal",
                    hovertemplate="Sell<br>Price: %{y:.2f}<br>Date: %{x}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Determine volume and equity rows based on layout
        volume_row = num_rows - 1  # Second to last row
        equity_row = num_rows  # Last row

        # Add volume bars
        colors = [
            "#00ff88" if close >= open else "#ff4444"
            for close, open in zip(data["Close"], data["Open"])
        ]

        fig.add_trace(
            go.Bar(
                x=dates,
                y=data["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
                hovertemplate="Volume: %{y:,.0f}<extra></extra>",
            ),
            row=volume_row,
            col=1,
        )

        # Add equity curve
        equity = None
        if hasattr(strategy, "analyzers") and hasattr(strategy.analyzers, "timereturn"):
            equity = strategy.analyzers.timereturn.get_analysis()
        elif hasattr(strategy, "analyzers") and hasattr(
            strategy.analyzers, "time_return"
        ):
            equity = strategy.analyzers.time_return.get_analysis()

        if equity:
            try:
                if isinstance(equity, dict):
                    equity_keys = list(equity.keys())
                    if equity_keys:
                        first_key = equity_keys[0]
                        if isinstance(first_key, str):
                            equity_dates = pd.to_datetime(
                                equity_keys, utc=True
                            ).tz_convert(IST)
                        elif isinstance(first_key, (pd.Timestamp, datetime)):
                            if first_key.tzinfo is None:
                                equity_dates = (
                                    pd.to_datetime(equity_keys)
                                    .tz_localize("UTC")
                                    .tz_convert(IST)
                                )
                            else:
                                equity_dates = pd.to_datetime(equity_keys).tz_convert(
                                    IST
                                )
                        else:
                            equity_dates = pd.to_datetime(
                                equity_keys, unit="s", utc=True
                            ).tz_convert(IST)
                        equity_values = list(equity.values())

                        # Convert to cumulative returns for better visualization
                        cumulative_returns = [(1 + val) for val in equity_values]

                        fig.add_trace(
                            go.Scatter(
                                x=equity_dates,
                                y=cumulative_returns,
                                mode="lines",
                                name="Equity Curve",
                                line=dict(color="#1f77b4", width=2),
                                hovertemplate="Equity: %{y:.4f}<br>Date: %{x}<extra></extra>",
                            ),
                            row=equity_row,
                            col=1,
                        )
            except Exception as e:
                logger.warning(f"Could not plot equity curve: {e}")

        # Dynamic layout updates based on number of rows
        layout_updates = {
            "title": title,
            f"xaxis{num_rows}_title": "Date",
            "yaxis_title": "Price ()",
            f"yaxis{volume_row}_title": "Volume",
            f"yaxis{equity_row}_title": "Equity",
            "height": 800 + (num_rows - 3) * 200,  # Dynamic height based on subplots
            "showlegend": True,
            "hovermode": "x unified",
        }

        # Add oscillator y-axis title if present
        if oscillator_indicators:
            layout_updates["yaxis2_title"] = "Indicators"

        fig.update_layout(**layout_updates)

        # Remove gaps for non-trading hours and weekends
        fig.update_xaxes(
            rangebreaks=[
                dict(
                    bounds=[6, 0], pattern="day of week"
                ),  # Hide weekends (Saturday=5, Sunday=6)
                dict(
                    bounds=["15:30", "09:15"], pattern="hour"
                ),  # Hide non-trading hours
            ]
        )
        # Remove rangeslider for cleaner look
        fig.update_layout(xaxis_rangeslider_visible=False)

        return fig

    except Exception as e:
        logger.error(f"Error creating candlestick chart: {e}", exc_info=True)
        st.error(f"Failed to create candlestick chart: {str(e)}")
        return None


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


def get_indicator_value_at_datetime(indicator, target_datetime: datetime, data_feed):
    """Get indicator value at a specific datetime."""
    target_dt_num = bt.date2num(target_datetime)

    # Find the bar closest to the target datetime
    closest_idx = None
    min_diff = float("inf")

    # Search for the closest bar
    for i in range(len(data_feed)):
        bar_dt_num = data_feed.datetime[i]
        diff = abs(bar_dt_num - target_dt_num)
        if diff < min_diff:
            min_diff = diff
            closest_idx = i

    if closest_idx is not None:
        # Calculate bars back from current position
        indicator_length = len(indicator)
        bars_back = indicator_length - 1 - closest_idx

        # Access the indicator value using BackTrader's indexing
        if bars_back >= 0 and bars_back < indicator_length:
            if bars_back == 0:
                value = indicator[0]  # Current value
            else:
                value = indicator[-bars_back]  # Historical value

            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                return float(value)


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
                        "regime": "unknown",
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

                            # ADDED: Extract indicator values for entry and exit times
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

                            # ADDED: Extract indicator values for entry and exit times
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

                        # ADDED: Extract indicator values for entry and exit
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

        # Add regime information with robust timezone handling
        if data is not None and "vol_regime" in data.columns and trades:
            for trade in trades:
                try:
                    entry_time = trade["entry_time"]
                    if entry_time is None:
                        trade["regime"] = "unknown"
                        continue

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
                        trade["regime"] = data.iloc[closest_idx]["vol_regime"]
                    else:
                        trade["regime"] = "unknown"
                except Exception as e:
                    logger.warning(f"Regime assignment failed: {str(e)}")
                    trade["regime"] = "unknown"

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
                "regime",
            ]
        )


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
            # Handle numeric values consistently
            elif isinstance(value, (int, float)):
                value = f"{value:.4f}" if abs(value) < 0.01 else f"{value:.2f}"
            table_data.append(
                {
                    "Category": "Summary",
                    "Metric": key.replace("_", " ").title(),
                    "Value": value,
                }
            )

        # Add trade analysis metrics
        for key, value in trade_analysis.items():
            # Skip nested dicts and lists
            if isinstance(value, (dict, list)):
                continue
            # Format numeric values consistently
            if isinstance(value, (int, float)):
                value = f"{value:.4f}" if abs(value) < 0.01 else f"{value:.2f}"
            table_data.append(
                {
                    "Category": "Trade Analysis",
                    "Metric": key.replace("_", " ").title(),
                    "Value": value,
                }
            )

        # Add risk metrics
        for key, value in risk_metrics.items():
            # Format numeric values consistently
            if isinstance(value, (int, float)):
                value = f"{value:.4f}" if abs(value) < 0.01 else f"{value:.2f}"
            table_data.append(
                {
                    "Category": "Risk Metrics",
                    "Metric": key.replace("_", " ").title(),
                    "Value": value,
                }
            )

        # Create DataFrame and ensure consistent string types
        df = pd.DataFrame(table_data)
        if "Value" in df.columns:
            df["Value"] = df["Value"].astype(str)  # Convert all values to string

        return df

    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Error creating summary table: {e}")
        return pd.DataFrame()


def create_trades_table(results, data=None):
    """Create a comprehensive trades table with all trade details including indicator values."""
    try:
        strategy = get_first_strategy(results)
        logger.info(
            f"[create_trades_table] Using strategy: {strategy.__class__.__name__}"
        )

        # Extract trades using the robust method
        trades_df = extract_trades(strategy, data)

        if trades_df.empty:
            logger.warning(
                "[create_trades_table] No trades extracted using extract_trades"
            )
            return pd.DataFrame(), "No trades executed during the period"

        # Convert to IST timezone for display
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"]).dt.tz_convert(
            IST
        )
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"]).dt.tz_convert(
            IST
        )

        # Format trade details
        formatted_trades = []
        for i, trade in trades_df.iterrows():
            duration = trade["exit_time"] - trade["entry_time"]
            duration_hours = duration.total_seconds() / 3600
            duration_days = duration.days
            return_pct = (
                (trade["pnl_net"] / (trade["entry_price"] * trade["size"])) * 100
                if trade["entry_price"] * trade["size"] != 0
                else 0
            )

            # Create base trade info
            trade_info = {
                "Trade #": i + 1,
                "Entry Date": trade["entry_time"].strftime("%Y-%m-%d"),
                "Entry Time": trade["entry_time"].strftime("%H:%M:%S"),
                "Entry Price": f"{trade['entry_price']:.2f}",
                "Exit Date": trade["exit_time"].strftime("%Y-%m-%d"),
                "Exit Time": trade["exit_time"].strftime("%H:%M:%S"),
                "Exit Price": f"{trade['exit_price']:.2f}",
                "Size": int(trade["size"]),
                "Direction": trade["direction"],
                "P&L": f"{trade['pnl_net']:.2f}",
                "Return %": f"{return_pct:.2f}%",
                "Duration (Hours)": f"{duration_hours:.1f}",
                "Duration (Days)": duration_days,
                "Status": trade["status"],
                "Regime": trade.get("regime", "N/A"),
            }

            # Add indicator values
            for col in trades_df.columns:
                if col.endswith("_entry") or col.endswith("_exit"):
                    indicator_name = col.rsplit("_", 1)[
                        0
                    ]  # Extract base indicator name
                    value = trade[col]
                    # Add both the indicator name and its value to the table
                    if pd.notnull(value):
                        # Format indicator name and context
                        context = "Entry" if col.endswith("_entry") else "Exit"
                        display_name = f"{indicator_name} ({context})"
                        trade_info[display_name] = (
                            f"{value:.2f}"
                            if isinstance(value, (int, float))
                            else str(value)
                        )

            formatted_trades.append(trade_info)

        # Create DataFrame and sort columns
        df = pd.DataFrame(formatted_trades)

        # Order columns: Trade details first, then entry indicators, then exit indicators
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
            "Regime",
        ]

        # Separate indicator columns
        indicator_columns = [col for col in df.columns if col not in base_columns]

        # Sort indicators: entry first, then exit
        entry_indicators = sorted([col for col in indicator_columns if "Entry" in col])
        exit_indicators = sorted([col for col in indicator_columns if "Exit" in col])

        # Combine all columns in desired order
        ordered_columns = base_columns + entry_indicators + exit_indicators

        return df[ordered_columns], None

    except Exception as e:
        logger.error(f"Error creating trades table: {e}", exc_info=True)
        return pd.DataFrame(), f"Error creating trades table: {str(e)}"


def analyze_best_time_ranges(results, data=None):
    """Analyze time ranges when most winning trades occurred using extract_trades."""
    try:
        strategy = get_first_strategy(results)
        # Use extract_trades for robust trade extraction
        trades_df = extract_trades(strategy, data)
        if trades_df.empty:
            logger.warning(
                "[analyze_best_time_ranges] No trade data available (trades_df is empty)"
            )
            return {"error": "No trade data available"}
        # Filter for winning trades
        winning_trades = trades_df[trades_df["pnl"] > 0].copy()
        if winning_trades.empty:
            logger.warning("[analyze_best_time_ranges] No winning trades found")
            return {"error": "No winning trades found"}
        # Ensure entry_time is datetime and in IST
        if not pd.api.types.is_datetime64_any_dtype(winning_trades["entry_time"]):
            winning_trades["entry_time"] = pd.to_datetime(winning_trades["entry_time"])
        if winning_trades["entry_time"].dt.tz is None:
            winning_trades["entry_time"] = (
                winning_trades["entry_time"].dt.tz_localize("UTC").dt.tz_convert(IST)
            )
        else:
            winning_trades["entry_time"] = winning_trades["entry_time"].dt.tz_convert(
                IST
            )
        # Add hour, day_of_week, month columns
        winning_trades["hour"] = winning_trades["entry_time"].dt.hour
        winning_trades["day_of_week"] = winning_trades["entry_time"].dt.day_name()
        winning_trades["month"] = winning_trades["entry_time"].dt.month_name()
        # Analyze by hour of day
        hourly_stats = (
            winning_trades.groupby("hour")
            .agg(count=("pnl", "count"), total_pnl=("pnl", "sum"))
            .to_dict("index")
        )
        best_hours = sorted(
            hourly_stats.items(), key=lambda x: x[1]["count"], reverse=True
        )[:3]
        # Analyze by day of week
        daily_stats = (
            winning_trades.groupby("day_of_week")
            .agg(count=("pnl", "count"), total_pnl=("pnl", "sum"))
            .to_dict("index")
        )
        best_days = sorted(
            daily_stats.items(), key=lambda x: x[1]["count"], reverse=True
        )[:3]
        # Analyze by month
        monthly_stats = (
            winning_trades.groupby("month")
            .agg(count=("pnl", "count"), total_pnl=("pnl", "sum"))
            .to_dict("index")
        )
        best_months = sorted(
            monthly_stats.items(), key=lambda x: x[1]["count"], reverse=True
        )[:3]
        return {
            "total_winning_trades": len(winning_trades),
            "best_hours": [
                {
                    "hour": f"{hour}:00",
                    "trades": stats["count"],
                    "total_pnl": stats["total_pnl"],
                }
                for hour, stats in best_hours
            ],
            "best_days": [
                {"day": day, "trades": stats["count"], "total_pnl": stats["total_pnl"]}
                for day, stats in best_days
            ],
            "best_months": [
                {
                    "month": month,
                    "trades": stats["count"],
                    "total_pnl": stats["total_pnl"],
                }
                for month, stats in best_months
            ],
            "hourly_distribution": hourly_stats,
            "daily_distribution": daily_stats,
            "monthly_distribution": monthly_stats,
        }
    except Exception as e:
        logger.error(f"Error analyzing best time ranges: {e}")
        return {"error": str(e)}


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
                            else "0.00"
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


def create_parameters_table(best_params_info):
    """Create a table for best parameters analysis."""
    if "error" in best_params_info:
        return pd.DataFrame()

    try:
        params_data = []
        for param, value in best_params_info["best_parameters"].items():
            importance = best_params_info["parameter_importance"].get(param, 0)
            params_data.append(
                {
                    "Parameter": param,
                    "Best Value": (
                        f"{value:.4f}" if isinstance(value, float) else str(value)
                    ),
                    "Importance": f"{importance:.3f}" if importance else "N/A",
                }
            )

        df = pd.DataFrame(params_data).sort_values(
            "Importance",
            ascending=False,
            key=lambda x: pd.to_numeric(x.replace("N/A", "0"), errors="coerce"),
        )
        return df

    except Exception as e:
        logger.error(f"Error creating parameters table: {e}")
        return pd.DataFrame()


def create_dynamic_indicators_table(data, strategy):
    """Create a dynamic table showing current technical indicator values based on strategy."""
    try:
        # Detect indicators dynamically
        detected_indicators = detect_strategy_indicators(strategy)
        calculated_indicators = calculate_indicator_values(data, detected_indicators)

        if not calculated_indicators:
            return pd.DataFrame()

        # Get latest values (last 5 periods)
        latest_data = []
        for i in range(min(5, len(data))):
            idx = -(i + 1)  # Start from last and go backwards
            date = data.index[idx].strftime("%Y-%m-%d %H:%M")

            row_data = {"Date/Time": date, "Close Price": data["Close"].iloc[idx]}

            # Add all detected indicators
            for indicator_name, indicator_data in calculated_indicators.items():
                if not indicator_data["values"].empty and idx < len(
                    indicator_data["values"]
                ):
                    value = indicator_data["values"].iloc[idx]
                    if pd.notna(value):
                        row_data[indicator_data["name"]] = value

            # Add signal analysis for common indicators
            signals = []

            # EMA signals
            ema_indicators = [
                k for k in calculated_indicators.keys() if "ema" in k.lower()
            ]
            if len(ema_indicators) >= 2:
                fast_ema = None
                slow_ema = None
                for ema_name in ema_indicators:
                    if "fast" in ema_name.lower():
                        fast_ema = calculated_indicators[ema_name]["values"].iloc[idx]
                    elif "slow" in ema_name.lower():
                        slow_ema = calculated_indicators[ema_name]["values"].iloc[idx]

                if fast_ema is not None and slow_ema is not None:
                    signals.append(
                        "EMA: Bullish" if fast_ema > slow_ema else "EMA: Bearish"
                    )

            # RSI signals
            rsi_indicators = [
                k for k in calculated_indicators.keys() if "rsi" in k.lower()
            ]
            for rsi_name in rsi_indicators:
                rsi_data = calculated_indicators[rsi_name]
                if "levels" in rsi_data:
                    rsi_value = rsi_data["values"].iloc[idx]
                    levels = rsi_data["levels"]
                    if rsi_value > levels.get("overbought", 70):
                        signals.append("RSI: Overbought")
                    elif rsi_value < levels.get("oversold", 30):
                        signals.append("RSI: Oversold")
                    else:
                        signals.append("RSI: Neutral")

            # MACD signals
            macd_line_indicators = [
                k
                for k in calculated_indicators.keys()
                if "macd" in k.lower() and "line" in k.lower()
            ]
            macd_signal_indicators = [
                k
                for k in calculated_indicators.keys()
                if "macd" in k.lower() and "signal" in k.lower()
            ]

            if macd_line_indicators and macd_signal_indicators:
                macd_line = calculated_indicators[macd_line_indicators[0]][
                    "values"
                ].iloc[idx]
                macd_signal = calculated_indicators[macd_signal_indicators[0]][
                    "values"
                ].iloc[idx]
                signals.append(
                    "MACD: Bullish" if macd_line > macd_signal else "MACD: Bearish"
                )

            row_data["Signals"] = " | ".join(signals) if signals else "No signals"
            latest_data.append(row_data)

        return pd.DataFrame(latest_data)

    except Exception as e:
        logger.error(f"Error creating dynamic indicators table: {e}")
        return pd.DataFrame()


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


def complete_backtest(
    user_ticker=None,
    user_start_date=None,
    user_end_date=None,
    user_strategy=None,
    user_timeframe=None,
    user_analyzers=None,
):
    """Run a full demonstration of backtest, optimization, and walk-forward analysis."""
    st.header("Complete Backtest Demonstration")
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        # Use user-supplied parameters or fall back to defaults
        ticker = user_ticker if user_ticker else get_available_tickers()[0]
        start_date = (
            user_start_date
            if user_start_date
            else (datetime.today().date() - timedelta(days=60)).strftime("%Y-%m-%d")
        )
        end_date = (
            user_end_date
            if user_end_date
            else (datetime.today().date() - timedelta(days=2)).strftime("%Y-%m-%d")
        )
        timeframe = user_timeframe if user_timeframe else "5m"
        strategy_name = (
            user_strategy if user_strategy else list(STRATEGY_REGISTRY.keys())[0]
        )

        # Prepare analyzer configuration
        available_analyzers = {
            "SharpeRatio": bt.analyzers.SharpeRatio,
            "DrawDown": bt.analyzers.DrawDown,
            "Returns": bt.analyzers.Returns,
            "TradeAnalyzer": bt.analyzers.TradeAnalyzer,
            "TimeReturn": bt.analyzers.TimeReturn,
            "SortinoRatio": SortinoRatio,
            "Calmar": bt.analyzers.Calmar,
            "SQN": bt.analyzers.SQN,
        }
        selected_analyzers = (
            user_analyzers
            if user_analyzers
            else [
                "SharpeRatio",
                "DrawDown",
                "Returns",
                "TradeAnalyzer",
                "TimeReturn",
                "SortinoRatio",
            ]
        )
        analyzer_config = [
            (available_analyzers[name], {"_name": name.lower()})
            for name in selected_analyzers
        ]

        # Validate data
        status_text.text("Fetching data for demo...")
        data = get_data_sync(ticker, start_date, end_date, interval=timeframe)
        if data.empty:
            st.error("No data available for the demo ticker and date range.")
            status_text.text("No data available for the demo ticker and date range.")
            return

        # Run complete backtest using the function from backtesting.py
        status_text.text("Running complete backtest...")
        results = run_complete_backtest(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            strategy_class=strategy_name,
            analyzers=analyzer_config,
            interval=timeframe,
        )
        progress_bar.progress(100)
        status_text.text("Complete backtest finished!")
        st.toast("Backtesting complete")

        # Display results
        if "basic" in results:
            st.subheader("Basic Backtest Results")
            basic_analyzer = PerformanceAnalyzer(results["basic"])
            basic_report = basic_analyzer.generate_full_report()

            # Show summary as table
            st.write("### Performance Summary")
            summary_table = create_summary_table(basic_report)
            st.dataframe(summary_table, use_container_width=True)

            # Plot basic backtest chart
            st.write("### Basic Strategy Performance")
            fig = create_candlestick_chart_with_trades(
                data, results["basic"], "Basic Strategy Results"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if "optimization" in results:
            st.subheader("Optimization Results")
            opt_analyzer = PerformanceAnalyzer(results["optimization"]["results"])
            opt_report = opt_analyzer.generate_full_report()

            # Show summary as table
            st.write("### Performance Summary")
            summary_table = create_summary_table(opt_report)
            st.dataframe(summary_table, use_container_width=True)

            # Plot optimization contour
            st.write("### Parameter Optimization Landscape")
            contour_fig = plot_contour(results["optimization"]["study"])
            if contour_fig:
                st.plotly_chart(contour_fig, use_container_width=True)

            # Plot optimized strategy chart
            st.write("### Optimized Strategy Performance")
            fig = create_candlestick_chart_with_trades(
                data, results["optimization"]["results"], "Optimized Strategy Results"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if "walk_forward" in results:
            st.subheader("Walk-Forward Analysis Results")
            print("WFFFFFFF====", results)
            wf_analyzer = PerformanceAnalyzer(
                results["walk_forward"]["windows"][-1]["summary"]
            )
            wf_report = wf_analyzer.generate_full_report()

            # Show summary as table
            st.write("### Performance Summary")
            summary_table = create_summary_table(wf_report)
            st.dataframe(summary_table, use_container_width=True)

            # Plot equity curve
            st.write("### Walk-Forward Equity Curve")
            equity_data = wf_report.get("timereturn", {})
            equity_fig = plot_equity_curve(equity_data)
            if equity_fig:
                st.plotly_chart(equity_fig, use_container_width=True)

        # Show comparison if available
        if "comparison" in results:
            st.subheader("Strategy Comparison")
            comparison_df = compare_strategies(results["comparison"])
            st.dataframe(comparison_df)

        # Show summary metrics
        st.subheader("Summary Metrics")
        col1, col2 = st.columns(2)
        with col1:
            if "basic" in results:
                st.metric(
                    "Basic Return",
                    f"{basic_report.get('summary', {}).get('total_return_pct', 0):.2f}%",
                )
            if "optimization" in results:
                st.metric(
                    "Optimized Return",
                    f"{opt_report.get('summary', {}).get('total_return_pct', 0):.2f}%",
                )
        with col2:
            if "walk_forward" in results:
                st.metric(
                    "Walk-Forward Return",
                    f"{wf_report.get('summary', {}).get('total_return_pct', 0):.2f}%",
                )

    except Exception as e:
        logger.error(f"Complete backtest failed: {e}", exc_info=True)
        st.error(f"Complete backtest failed: {str(e)}")
        progress_bar.progress(0)
        status_text.text("Complete backtest failed.")


def main():
    st.set_page_config(
        page_title="Comprehensive Backtesting Framework",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📈 Comprehensive Backtesting Framework")
    st.markdown(
        """
    **Advanced Trading Strategy Analysis Platform**
    
    This framework provides comprehensive backtesting capabilities with:
    - 📊 **Enhanced Candlestick Charts** with trade markers and volume analysis
    - 📋 **Detailed Trade Tables** with comprehensive trade information
    - ⏰ **Best Trading Times Analysis** with hourly, daily, and monthly breakdowns
    - 🎯 **Parameter Optimization** with visual parameter importance analysis
    - 🗺️ **Optimization Landscape** visualization with contour plots
    - 📈 **Strategy Comparison** tables showing improvement metrics
    """
    )

    # Add custom CSS for better styling
    st.markdown(
        """
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
    }
    .trade-table {
        font-size: 0.9rem;
    }
    .improvement-positive {
        color: #00C851;
        font-weight: bold;
    }
    .improvement-negative {
        color: #ff4444;
        font-weight: bold;
    }
    .indicator-entry {
        background-color: #e8f5e9;
    }
    .indicator-exit {
        background-color: #ffebee;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Initialize progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Sidebar for inputs
    st.sidebar.header("Backtest Configuration")

    # Ticker Management Section
    with st.sidebar.expander("Ticker Management", expanded=False):
        st.write("**Manage Ticker List**")

        # Show current tickers
        current_tickers = get_available_tickers()
        st.write(f"Current tickers ({len(current_tickers)}):")
        st.write(
            ", ".join(current_tickers[:10])
            + ("..." if len(current_tickers) > 10 else "")
        )

        # Add new ticker
        new_ticker = st.text_input(
            "Add New Ticker", help="Enter ticker symbol to add to the list"
        )
        if st.button("Add Ticker") and new_ticker:
            new_ticker = new_ticker.strip().upper()
            is_valid, message = validate_ticker_format(new_ticker)
            if not is_valid:
                st.error(message)
            elif new_ticker not in current_tickers:
                updated_tickers = current_tickers + [new_ticker]
                if save_tickers_to_file(updated_tickers):
                    st.success(f"Added {new_ticker} to ticker list")
                    st.experimental_rerun()
                else:
                    st.error("Failed to save ticker list")
            else:
                st.warning(f"{new_ticker} already exists in the list")

        # Remove ticker
        if current_tickers:
            ticker_to_remove = st.selectbox("Remove Ticker", [""] + current_tickers)
            if st.button("Remove Ticker") and ticker_to_remove:
                updated_tickers = [t for t in current_tickers if t != ticker_to_remove]
                if save_tickers_to_file(updated_tickers):
                    st.success(f"Removed {ticker_to_remove} from ticker list")
                    st.experimental_rerun()
                else:
                    st.error("Failed to save ticker list")

        # Reset to defaults
        if st.button("Reset to Default Tickers"):
            if save_tickers_to_file(DEFAULT_TICKERS):
                st.success("Reset to default ticker list")
                st.experimental_rerun()
            else:
                st.error("Failed to reset ticker list")

        # Export ticker list
        if st.button("Export Ticker List"):
            ticker_text = "\n".join(current_tickers)
            st.download_button(
                label="Download tickers.txt",
                data=ticker_text,
                file_name="tickers.txt",
                mime="text/plain",
            )

        # Import ticker list
        uploaded_file = st.file_uploader("Import Ticker List", type=["txt"])
        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode("utf-8")
                imported_tickers = [
                    line.strip().upper() for line in content.split("\n") if line.strip()
                ]

                # Validate all tickers
                valid_tickers = []
                invalid_tickers = []
                for ticker in imported_tickers:
                    is_valid, _ = validate_ticker_format(ticker)
                    if is_valid:
                        valid_tickers.append(ticker)
                    else:
                        invalid_tickers.append(ticker)

                if valid_tickers:
                    if save_tickers_to_file(valid_tickers):
                        st.success(f"Imported {len(valid_tickers)} valid tickers")
                        if invalid_tickers:
                            st.warning(
                                f"Skipped {len(invalid_tickers)} invalid tickers: {', '.join(invalid_tickers[:5])}"
                            )
                        st.experimental_rerun()
                    else:
                        st.error("Failed to save imported tickers")
                else:
                    st.error("No valid tickers found in the uploaded file")
            except Exception as e:
                st.error(f"Error importing ticker list: {str(e)}")

    # Strategy selection
    strategy_names = list(STRATEGY_REGISTRY.keys())
    selected_strategy = st.sidebar.selectbox("Select Strategy", strategy_names)

    # Date inputs with validation
    end_date_default = datetime.today().date() - timedelta(days=2)
    start_date_default = end_date_default - timedelta(days=58)
    start_date = st.sidebar.date_input(
        "Start Date",
        value=start_date_default,
        max_value=end_date_default - timedelta(days=1),
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=end_date_default,
        min_value=start_date + timedelta(days=1),
        max_value=datetime.today().date(),
    )

    if start_date >= end_date:
        st.sidebar.error("End date must be after start date.")
        status_text.text("Invalid date range selected.")
        return

    # Ticker input with validation - dynamic ticker selection
    ticker_input_method = st.sidebar.radio(
        "Ticker Input Method", ["Select from List", "Enter Custom Ticker"]
    )

    if ticker_input_method == "Select from List":
        available_tickers = get_available_tickers()
        ticker = st.sidebar.selectbox("Ticker Symbol", available_tickers)
    else:
        ticker = (
            st.sidebar.text_input(
                "Custom Ticker Symbol",
                value="",
                help="Enter ticker symbol (e.g., AAPL, GOOGL, RELIANCE.NS)",
            )
            .strip()
            .upper()
        )

        if ticker:
            # Add validation for custom ticker format
            is_valid, message = validate_ticker_format(ticker)
            if not is_valid:
                st.sidebar.error(message)
                status_text.text("Invalid ticker format.")
                return

    if not ticker:
        st.sidebar.error("Ticker symbol cannot be empty.")
        status_text.text("Ticker symbol cannot be empty.")
        return

    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Backtest", "Optimization", "Walk-Forward", "Complete Backtest"],
    )
    if analysis_type in ["Optimization", "Walk-Forward", "Complete Backtest"]:
        n_trials = st.sidebar.slider(
            "Number of Trials",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Number of optimization trials to run (increases in steps of 10)",
        )
    else:
        n_trials = 10

    # Timeframe selection
    timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m"])

    # Analyzer selection
    available_analyzers = {
        "SharpeRatio": bt.analyzers.SharpeRatio,
        "DrawDown": bt.analyzers.DrawDown,
        "Returns": bt.analyzers.Returns,
        "TradeAnalyzer": bt.analyzers.TradeAnalyzer,
        "TimeReturn": bt.analyzers.TimeReturn,
        "SortinoRatio": SortinoRatio,
        "Calmar": bt.analyzers.Calmar,
        "SQN": bt.analyzers.SQN,
    }
    selected_analyzers = st.sidebar.multiselect(
        "Select Analyzers",
        list(available_analyzers.keys()),
        default=[
            "SharpeRatio",
            "DrawDown",
            "Returns",
            "TradeAnalyzer",
            "TimeReturn",
            "SortinoRatio",
        ],
    )

    # Validate analyzer dependencies
    if "SortinoRatio" in selected_analyzers and "TimeReturn" not in selected_analyzers:
        st.sidebar.error("SortinoRatio requires TimeReturn analyzer.")
        status_text.text("Invalid analyzer configuration.")
        return

    # Run button
    if st.sidebar.button("Run Analysis"):
        try:
            # Fetch data with validation
            start_dt = IST.localize(datetime.combine(start_date, datetime.min.time()))
            end_dt = IST.localize(datetime.combine(end_date, datetime.min.time()))
            data = get_data_sync(ticker, start_date, end_date, interval=timeframe)

            if data.empty:
                st.error("No data available for the selected ticker and date range.")
                status_text.text(
                    "No data available for the selected ticker and date range."
                )
                return

            # Validate data
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in data.columns for col in required_columns):
                st.error(f"Missing required columns: {required_columns}")
                status_text.text("Missing required data columns.")
                return
            if data[required_columns].isnull().any().any():
                st.error("Data contains missing values.")
                status_text.text("Data contains missing values.")
                return
            if (
                not data[required_columns]
                .apply(lambda x: x.apply(lambda y: isinstance(y, (int, float))))
                .all()
                .all()
            ):
                st.error("Data contains non-numeric values.")
                status_text.text("Data contains non-numeric values.")
                return

            # Prepare analyzer configuration
            analyzer_config = [
                (available_analyzers[name], {"_name": name.lower()})
                for name in selected_analyzers
            ]

            # Run analysis based on type
            if analysis_type == "Backtest":
                status_text.text("Running backtest...")
                results, cerebro = run_basic_backtest(
                    strategy_class=selected_strategy,
                    ticker=ticker,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    interval=timeframe,
                    analyzers=analyzer_config,
                )
                progress_bar.progress(100)
                status_text.text("Backtest complete!")
                st.toast("Backtesting complete")

                # Initialize PerformanceAnalyzer with results
                analyzer = PerformanceAnalyzer(results[0])
                report = analyzer.generate_full_report()

                # Display report as table instead of JSON
                st.write("### Backtest Results Summary")
                summary_table = create_summary_table(report)
                st.dataframe(summary_table, use_container_width=True)

                # Export summary table
                if st.button("Export Summary Table"):
                    csv_data = summary_table.to_csv(index=False)
                    st.download_button(
                        label="Download Summary as CSV",
                        data=csv_data,
                        file_name=f"{ticker}_backtest_summary.csv",
                        mime="text/csv",
                    )

                # Enhanced Candlestick Chart with Technical Indicators
                st.write("### 📈 Enhanced Chart with Technical Indicators & Trades")
                plotly_fig = create_candlestick_chart_with_trades(data, results)
                if plotly_fig:
                    st.plotly_chart(plotly_fig, use_container_width=True)
                    # Export plot
                    if st.button("Export Enhanced Chart"):
                        buf = BytesIO()
                        plotly_fig.write_image(buf, format="png")
                        st.download_button(
                            label="Download Enhanced Chart as PNG",
                            data=buf.getvalue(),
                            file_name=f"{ticker}_enhanced_backtest_chart.png",
                            mime="image/png",
                        )
                else:
                    st.warning("Could not generate chart")

                # Dynamic Technical Indicators Table
                st.write("### 📊 Technical Indicators - Latest Values")
                strategy = get_first_strategy(results)
                indicators_df = create_dynamic_indicators_table(data, strategy)
                if not indicators_df.empty:
                    st.dataframe(indicators_df, use_container_width=True)

                    # Export indicators table
                    csv_data = indicators_df.to_csv(index=False)
                    st.download_button(
                        label="Download Indicators Table as CSV",
                        data=csv_data,
                        file_name=f"{ticker}_indicators_table.csv",
                        mime="text/csv",
                    )

                    # Show detected indicators summary
                    detected_indicators = detect_strategy_indicators(strategy)
                    if detected_indicators:
                        st.write("**🔍 Detected Strategy Indicators:**")
                        indicator_summary = []
                        for name, info in detected_indicators.items():
                            # Filter out non-user-friendly params
                            params = info.get("params", {})
                            user_params = {
                                k: v
                                for k, v in params.items()
                                if not (
                                    k.startswith("_")
                                    or callable(v)
                                    or isinstance(v, type)
                                    or "method" in str(type(v)).lower()
                                )
                            }
                            if user_params:
                                params_str = ", ".join(
                                    [f"{k} = {v}" for k, v in user_params.items()]
                                )
                            else:
                                params_str = "No parameters"
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
                    trades_df, trades_error = create_trades_table(results)
                    if trades_error:
                        st.warning(f"Could not create trades table: {trades_error}")
                    elif not trades_df.empty:
                        # Apply styling to highlight entry and exit indicators
                        def highlight_columns(col):
                            if "Entry" in col:
                                return ["background-color: #e8f5e9"] * len(col)
                            elif "Exit" in col:
                                return ["background-color: #ffebee"] * len(col)
                            return [""] * len(col)

                        styled_trades = trades_df.style.apply(highlight_columns, axis=0)
                        st.dataframe(styled_trades, use_container_width=True)

                        # Export trades table
                        csv_data = trades_df.to_csv(index=False)
                        st.download_button(
                            label="Download Trades Table as CSV",
                            data=csv_data,
                            file_name=f"{ticker}_trades_table.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("No trades executed during the backtest period.")
                except Exception as e:
                    st.error(f"Error creating trades table: {str(e)}")
                    logger.exception("Error creating trades table")

                # Trade Statistics Summary
                st.write("### 📈 Trade Statistics Summary")
                best_trades_analysis = analyze_best_trades(results)
                if "error" not in best_trades_analysis:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Trades", best_trades_analysis["total_trades"])
                        st.metric(
                            "Winning Trades", best_trades_analysis["winning_trades"]
                        )

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
                        st.metric(
                            "Losing Trades", best_trades_analysis["losing_trades"]
                        )

                    with col3:
                        st.metric(
                            "Total P&L", f"{best_trades_analysis['total_pnl']:.2f}"
                        )
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

                # Best Trading Times Analysis
                st.write("### ⏰ Best Trading Times Analysis")
                time_analysis = analyze_best_time_ranges(results)
                if "error" not in time_analysis:
                    hours_df, days_df, months_df = create_best_times_table(
                        time_analysis
                    )

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

                    # Time Analysis Chart
                    st.write("### 📊 Trading Time Analysis")
                    time_chart = plot_time_analysis(time_analysis)
                    if time_chart:
                        st.plotly_chart(time_chart, use_container_width=True)
                else:
                    st.warning(
                        f"Could not analyze best times: {time_analysis.get('error', 'Unknown error')}"
                    )

                # Export option
                if st.button("Export Full Backtest Report"):
                    report_json = json.dumps(report, indent=2, default=str)
                    st.download_button(
                        label="Download Full Report as JSON",
                        data=report_json,
                        file_name=f"{ticker}_backtest_report.json",
                        mime="application/json",
                    )

            elif analysis_type == "Optimization":
                status_text.text("Starting optimization...")
                results = run_parameter_optimization(
                    strategy_class=selected_strategy,
                    ticker=ticker,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    n_trials=n_trials,
                    interval=timeframe,
                    analyzers=analyzer_config,
                )
                progress_bar.progress(100)
                status_text.text("Optimization complete!")
                st.toast("Backtesting complete")

                # Check for errors in optimization results
                if results.get("results") is None:
                    error_msg = results.get(
                        "error", "Unknown error during optimization."
                    )
                    st.error(f"Optimization failed: {error_msg}")
                    status_text.text("Optimization failed.")
                    return

                # Initialize PerformanceAnalyzer with results
                analyzer = PerformanceAnalyzer(results["results"][0])
                report = analyzer.generate_full_report()

                # Display report as table instead of JSON
                st.write("### Optimization Results Summary")
                summary_table = create_summary_table(report)
                st.dataframe(summary_table, use_container_width=True)

                # Export summary table
                if st.button("Export Optimization Summary"):
                    csv_data = summary_table.to_csv(index=False)
                    st.download_button(
                        label="Download Summary as CSV",
                        data=csv_data,
                        file_name=f"{ticker}_optimization_summary.csv",
                        mime="text/csv",
                    )

                # Optimization Contour Plot
                st.write("### 🗺️ Parameter Optimization Landscape")
                contour_fig = plot_contour(results["study"])
                if contour_fig:
                    st.plotly_chart(contour_fig, use_container_width=True)
                    if st.button("Export Contour Plot"):
                        buf = BytesIO()
                        contour_fig.write_image(buf, format="png")
                        st.download_button(
                            label="Download Contour Plot as PNG",
                            data=buf.getvalue(),
                            file_name=f"{ticker}_contour_plot.png",
                            mime="image/png",
                        )
                else:
                    st.warning("Could not generate contour plot")

                # Enhanced Candlestick Chart for Optimized Strategy
                st.write("### 📈 Optimized Strategy - Enhanced Chart with Indicators")
                plotly_fig = create_candlestick_chart_with_trades(
                    data, results["results"], "Optimized Strategy Results"
                )
                if plotly_fig:
                    st.plotly_chart(plotly_fig, use_container_width=True)
                    if st.button("Export Optimized Strategy Chart"):
                        buf = BytesIO()
                        plotly_fig.write_image(buf, format="png")
                        st.download_button(
                            label="Download Optimized Strategy Chart as PNG",
                            data=buf.getvalue(),
                            file_name=f"{ticker}_optimized_strategy_chart.png",
                            mime="image/png",
                        )
                else:
                    st.warning("Could not generate optimized strategy chart")

                # Dynamic Technical Indicators for Optimized Strategy
                st.write("### 📊 Optimized Strategy - Technical Indicators")
                opt_strategy = get_first_strategy(results["results"])
                opt_indicators_df = create_dynamic_indicators_table(data, opt_strategy)
                if not opt_indicators_df.empty:
                    st.dataframe(opt_indicators_df, use_container_width=True)

                    # Show optimized parameters
                    opt_detected_indicators = detect_strategy_indicators(opt_strategy)
                    if opt_detected_indicators:
                        st.write("**🎯 Optimized Strategy Indicators:**")
                        opt_indicator_summary = []
                        for name, info in opt_detected_indicators.items():
                            params = info.get("params", {})
                            user_params = {
                                k: v
                                for k, v in params.items()
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
                                else "N/A"
                            ),
                        )
                        st.metric(
                            "Success Rate", f"{best_params_info['success_rate']:.1f}%"
                        )
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
                    st.warning(
                        "Could not display best parameters: "
                        + best_params_info.get("error", "Unknown error")
                    )

                # Comprehensive Trades Table for Optimized Strategy
                st.write("### 📊 Optimized Strategy - Detailed Trades Table")
                try:
                    opt_trades_df, opt_trades_error = create_trades_table(
                        results["results"]
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

                        styled_trades = opt_trades_df.style.apply(
                            highlight_columns, axis=0
                        )
                        st.dataframe(styled_trades, use_container_width=True)

                        # Export optimized trades table
                        csv_data = opt_trades_df.to_csv(index=False)
                        st.download_button(
                            label="Download Optimized Trades Table as CSV",
                            data=csv_data,
                            file_name=f"{ticker}_optimized_trades_table.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("No trades executed by the optimized strategy.")
                except Exception as e:
                    st.error(f"Error creating trades table: {str(e)}")
                    logger.exception("Error creating trades table")

                # Trade Statistics Summary for Optimized Strategy
                st.write("### 📈 Optimized Strategy - Trade Statistics")
                best_trades_analysis = analyze_best_trades(results["results"])
                if "error" not in best_trades_analysis:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Trades", best_trades_analysis["total_trades"])
                        st.metric(
                            "Winning Trades", best_trades_analysis["winning_trades"]
                        )

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
                        st.metric(
                            "Losing Trades", best_trades_analysis["losing_trades"]
                        )

                    with col3:
                        st.metric(
                            "Total P&L", f"{best_trades_analysis['total_pnl']:.2f}"
                        )
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
                    hours_df, days_df, months_df = create_best_times_table(
                        time_analysis
                    )

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
                        st.plotly_chart(time_chart, use_container_width=True)
                else:
                    st.warning(
                        "Could not analyze time ranges: "
                        + time_analysis.get("error", "Unknown error")
                    )

                # Export option
                if st.button("Export Full Optimization Report"):
                    report_json = json.dumps(report, indent=2, default=str)
                    st.download_button(
                        label="Download Full Report as JSON",
                        data=report_json,
                        file_name=f"{ticker}_optimization_report.json",
                        mime="application/json",
                    )
            elif analysis_type == "Walk-Forward":
                status_text.text("Starting walk-forward analysis...")
                # Use strategy name instead of class
                results = run_walkforward_analysis(
                    ticker=ticker,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    window_days=30,
                    out_days=7,
                    step_days=7,
                    n_trials=n_trials,
                    interval=timeframe,
                    strategy_name=selected_strategy,  # Changed from strategy_class
                )
                progress_bar.progress(100)
                status_text.text("Walk-forward analysis complete!")
                st.toast("Backtesting complete")

                # NEW: Comprehensive Walk-Forward Results Display
                if results and "windows" in results and results["windows"]:
                    st.subheader("📊 Walk-Forward Analysis Results")

                    # 1. Display overall summary metrics
                    st.write("### 🔍 Overall Walk-Forward Metrics")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Windows", len(results["windows"]))
                        st.metric(
                            "Valid Windows",
                            sum(1 for w in results["windows"] if w.get("valid", False)),
                        )

                    with col2:
                        avg_return = np.mean(
                            [
                                w["out_sample_performance"]["summary"][
                                    "total_return_pct"
                                ]
                                for w in results["windows"]
                                if w.get("valid", False)
                            ]
                        )
                        st.metric("Average Return", f"{avg_return:.2f}%")

                    with col3:
                        valid_windows = [
                            w for w in results["windows"] if w.get("valid", False)
                        ]
                        if valid_windows:
                            best_window = max(
                                valid_windows,
                                key=lambda w: w["out_sample_performance"]["summary"][
                                    "total_return_pct"
                                ],
                            )
                            st.metric(
                                "Best Window Return",
                                f"{best_window['out_sample_performance']['summary']['total_return_pct']:.2f}%",
                            )
                        else:
                            st.metric("Best Window Return", "N/A")

                    # 2. Show parameter evolution
                    st.write("### ⚙️ Parameter Evolution Across Windows")
                    param_data = []
                    for i, window in enumerate(results["windows"]):
                        if window.get("valid", False):
                            params = window.get("best_params", {})
                            param_data.append(
                                {
                                    "Window": i + 1,
                                    **params,
                                    "Return": window["out_sample_performance"][
                                        "summary"
                                    ]["total_return_pct"],
                                }
                            )
                    param_df = pd.DataFrame(param_data)
                    if not param_df.empty:
                        st.dataframe(
                            param_df.style.background_gradient(
                                cmap="RdYlGn", subset=["Return"]
                            ),
                            use_container_width=True,
                        )

                    # 3. Display each window in expanders
                    st.write("### 📅 Detailed Window Analysis")
                    for i, window in enumerate(results["windows"]):
                        if window.get("valid", False):
                            with st.expander(
                                f"Window {i+1} ({window['out_sample_start']} to {window['out_sample_end']})",
                                expanded=False,
                            ):
                                # Window summary
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write("**In-Sample Period**")
                                    st.write(f"Start: {window['in_sample_start']}")
                                    st.write(f"End: {window['in_sample_end']}")
                                    st.write(f"Optimization Trials: {n_trials}")

                                with col2:
                                    st.write("**Out-Sample Performance**")
                                    st.metric(
                                        "Return",
                                        f"{window['out_sample_performance']['summary']['total_return_pct']:.2f}%",
                                    )
                                    st.metric(
                                        "Sharpe Ratio",
                                        f"{window['out_sample_performance']['risk_metrics']['sharpe_ratio']:.2f}",
                                    )

                                # Show best parameters
                                st.write("**🎯 Best Parameters**")
                                if window.get("best_params"):
                                    for param, value in window["best_params"].items():
                                        st.code(f"{param}: {value}")

                                # Plot out-sample equity curve
                                st.write("### 📈 Out-Sample Equity Curve")
                                equity_data = window["out_sample_performance"].get(
                                    "timereturn", {}
                                )
                                equity_fig = plot_equity_curve(equity_data)
                                if equity_fig:
                                    st.plotly_chart(
                                        equity_fig, use_container_width=True
                                    )

                                # Show trades table for out-sample period
                                st.write("### 💰 Out-Sample Trades")
                                try:
                                    # Get strategy and data for this window
                                    strategy_obj = window.get("out_sample_strategy")
                                    out_sample_data = get_data_sync(
                                        ticker,
                                        window["out_sample_start"],
                                        window["out_sample_end"],
                                        interval=timeframe,
                                    )

                                    if strategy_obj and not out_sample_data.empty:
                                        trades_df, trades_error = create_trades_table(
                                            strategy_obj, out_sample_data
                                        )
                                        if trades_error:
                                            st.warning(
                                                f"Could not create trades table: {trades_error}"
                                            )
                                        elif not trades_df.empty:
                                            st.dataframe(
                                                trades_df, use_container_width=True
                                            )
                                        else:
                                            st.info(
                                                "No trades executed in this out-sample period"
                                            )
                                    else:
                                        st.warning("Missing data for trades table")
                                except Exception as e:
                                    st.error(f"Error creating trades table: {str(e)}")

                                # Trade analysis summary
                                st.write("### 📊 Trade Statistics")
                                if "trade_analysis" in window["out_sample_performance"]:
                                    ta = window["out_sample_performance"][
                                        "trade_analysis"
                                    ]
                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        st.metric(
                                            "Total Trades", ta.get("total_trades", 0)
                                        )
                                        st.metric(
                                            "Win Rate",
                                            f"{ta.get('win_rate_percent', 0):.1f}%",
                                        )

                                    with col2:
                                        st.metric(
                                            "Profit Factor",
                                            f"{ta.get('profit_factor', 0):.2f}",
                                        )
                                        st.metric(
                                            "Avg Win/Avg Loss",
                                            f"{ta.get('avg_win_avg_loss_ratio', 0):.2f}",
                                        )

                                    with col3:
                                        st.metric(
                                            "Max Consecutive Wins",
                                            ta.get("consecutive_wins", 0),
                                        )
                                        st.metric(
                                            "Max Consecutive Losses",
                                            ta.get("consecutive_losses", 0),
                                        )

                    # 4. Combined equity curve visualization
                    st.write("### 📈 Combined Equity Curve")
                    cumulative_return = 100
                    equity_points = [cumulative_return]
                    dates = []

                    for window in results["windows"]:
                        if window.get("valid", False):
                            equity = window["out_sample_performance"].get(
                                "timereturn", {}
                            )
                            if equity:
                                # Calculate cumulative returns
                                returns = list(equity.values())
                                for ret in returns:
                                    cumulative_return *= 1 + ret / 100
                                    equity_points.append(cumulative_return)

                                # Add dates for plotting
                                dates.extend(sorted(equity.keys()))

                    if len(equity_points) > 1:
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=dates,
                                y=equity_points,
                                mode="lines",
                                name="Cumulative Equity",
                            )
                        )
                        fig.update_layout(
                            title="Walk-Forward Equity Curve",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Value",
                            showlegend=True,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # 5. Export option
                    if st.button("Export Full Walk-Forward Report", key="export_wf"):
                        report_json = json.dumps(results, indent=2, default=str)
                        st.download_button(
                            label="Download Full Report as JSON",
                            data=report_json,
                            file_name=f"{ticker}_walkforward_report.json",
                            mime="application/json",
                        )
                else:
                    st.warning("No valid walk-forward windows available")

            elif analysis_type == "Complete Backtest":
                complete_backtest()

        except Exception as e:
            logger.error(f"Error running analysis: {e}", exc_info=True)
            st.error(f"Analysis failed: {str(e)}")
            progress_bar.progress(0)
            status_text.text("Analysis failed.")


if __name__ == "__main__":
    main()
