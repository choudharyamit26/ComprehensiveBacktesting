"""
Utility functions for the Streamlit application.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from comprehensive_backtesting.utils import DEFAULT_TICKERS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global strategy results storage
STRATEGY_RESULTS = defaultdict(
    lambda: {"backtest": None, "optimization": None, "walkforward": None}
)


def safe_float(value, default=0):
    """Safely convert various value types to float"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Clean percentage signs and commas
        cleaned = value.replace("%", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def safe_percentage(value, default=0):
    """Safely convert percentage values to decimal format (0-1 range)"""
    if isinstance(value, (int, float)):
        # If value is > 1, assume it's in percentage format (65.5 -> 0.655)
        return float(value) / 100 if value > 1 else float(value)
    if isinstance(value, str):
        # Clean percentage signs and commas
        cleaned = value.replace("%", "").replace(",", "").strip()
        try:
            val = float(cleaned)
            # Convert to decimal if it appears to be in percentage format
            return val / 100 if val > 1 else val
        except ValueError:
            return default
    return default


def clean_value(value):
    """Clean value by handling NaN, infinite, and None values."""
    if isinstance(value, float):
        # Handle NaN and infinite values
        if pd.isna(value) or not np.isfinite(value):
            return None
        return value
    elif isinstance(value, (int, np.integer)):
        return value
    elif value is None:
        return None
    elif isinstance(value, str):
        # Convert "N/A" to None for numeric fields
        if value.lower() == "n/a":
            return None
        return value
    else:
        return str(value)


def clean_numeric(val):
    """Clean numeric values for proper handling."""
    if isinstance(val, (int, float)):
        if pd.isna(val) or not np.isfinite(val):
            return None
        return val
    return val


def load_tickers_from_file(file_path="csv/nifty50_highbeta_stocks.csv"):
    """
    Load tickers from a CSV file with a 'ticker' column, validate them using yfinance,
    and save validated tickers to a new CSV file.

    Args:
        file_path (str): Path to the CSV file containing tickers

    Returns:
        list: List of validated ticker symbols
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.info(f"CSV file '{file_path}' not found, using default tickers")
            return DEFAULT_TICKERS

        # Read CSV file
        df = pd.read_csv(file_path)
        if "ticker" not in df.columns:
            logger.info(
                f"No 'ticker' column found in CSV file '{file_path}', using default tickers"
            )
            return DEFAULT_TICKERS

        # Extract and clean tickers
        symbols = df["ticker"].dropna().astype(str).tolist()
        cleaned_symbols = []
        for symbol in symbols:
            cleaned_symbol = symbol.strip().replace(".NS", "").replace(".BO", "")
            if (
                cleaned_symbol
                and len(cleaned_symbol) <= 15
                and cleaned_symbol.replace(".", "").replace("-", "").isalnum()
            ):
                cleaned_symbols.append(cleaned_symbol)

        logger.info(f"Read {len(cleaned_symbols)} stock symbols from '{file_path}'")

        valid_symbols = []
        total_symbols = len(cleaned_symbols)
        logger.info(f"Validating {total_symbols} tickers...")

        for i, symbol in enumerate(cleaned_symbols, 1):
            logger.debug(f"Validating ticker {i}/{total_symbols}: {symbol}")
            nse_ticker = symbol
            valid_symbols.append(nse_ticker)

        logger.info(f"Validated {len(valid_symbols)}/{len(cleaned_symbols)} tickers")

        return valid_symbols

    except Exception as e:
        logger.error(f"Error reading CSV file '{file_path}': {e}")
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


def get_strategy(results):
    """Extract strategy from results - handles different result formats."""
    if isinstance(results, list):
        if len(results) > 0 and isinstance(results[0], list):
            return results[0][0]
        elif len(results) > 0:
            return results[0]
    return results


def extract_report_metrics(report):
    """Extract key metrics from a strategy report with robust handling"""
    return {
        "win_rate": safe_percentage(report.get("Win Rate (%)", 0)),
        "sharpe_ratio": safe_float(report.get("Sharpe Ratio", 0)),
        "total_return_pct": safe_float(report.get("Total Return (%)", 0)),
        "max_drawdown_pct": safe_float(report.get("Max Drawdown (%)", 0)),
        "profit_factor": safe_float(report.get("Profit Factor", 0)),
        "total_pnl": safe_float(report.get("Total P&L", 0)),
        "total_trades": safe_float(report.get("Total Trades", 0), default=0),
        "avg_win": safe_float(report.get("Avg Winning Trade", 0)),
        "avg_loss": safe_float(report.get("Avg Losing Trade", 0)),
        "win_loss_ratio": safe_float(report.get("Win/Loss Ratio", 0)),
        "winning_trades": safe_float(report.get("Winning Trades", 0)),
        "losing_trades": safe_float(report.get("Losing Trades", 0)),
        "best_trade_pnl": safe_float(report.get("Best Trade P&L", 0)),
        "worst_trade_pnl": safe_float(report.get("Worst Trade P&L", 0)),
        "strategy_name": report.get("Strategy", ""),
    }


def get_strategy_params(strategy_instance):
    """Extract parameters from a strategy instance"""
    try:
        # For Backtrader strategies
        if hasattr(strategy_instance, "params") and hasattr(
            strategy_instance.params, "_getkwargs"
        ):
            return strategy_instance.params._getkwargs()
        # For our custom strategies
        elif hasattr(strategy_instance, "params"):
            return strategy_instance.params
        # For dictionary-based results (like walk-forward)
        elif isinstance(strategy_instance, dict) and "params" in strategy_instance:
            return strategy_instance["params"]
        else:
            return {}
    except Exception as e:
        logger.error(f"Error getting strategy params: {str(e)}")
        return {}


def is_possible_number(x):
    """Check if a value can be converted to a number."""
    if isinstance(x, (int, float, np.integer, np.floating)):
        return True
    try:
        float(x)
        return True
    except:
        return False


def calculate_return_metrics(returns_series):
    """Calculate return metrics from a returns series."""
    if returns_series.empty:
        return {}

    metrics = {
        "Best Month": f"{returns_series.max():.2f}%",
        "Worst Month": f"{returns_series.min():.2f}%",
        "Avg Positive Month": f"{returns_series[returns_series > 0].mean():.2f}%",
        "Avg Negative Month": f"{returns_series[returns_series < 0].mean():.2f}%",
        "Win Rate": f"{len(returns_series[returns_series > 0]) / len(returns_series) * 100:.1f}%",
        "Std Dev": f"{returns_series.std():.2f}%",
    }
    return metrics
