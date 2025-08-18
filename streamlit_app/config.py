"""
Configuration settings for the Streamlit application.
"""

import pytz
from datetime import datetime, timedelta
import backtrader as bt
from comprehensive_backtesting.parameter_optimization import SortinoRatio

# Set timezone for IST
IST = pytz.timezone("Asia/Kolkata")


# Default date settings
def get_default_dates():
    """Get default start and end dates for backtesting."""
    end_date_default = datetime.today().date() - timedelta(days=2)
    start_date_default = end_date_default - timedelta(days=365)
    return start_date_default, end_date_default


# Available analyzers configuration
AVAILABLE_ANALYZERS = {
    "SharpeRatio": bt.analyzers.SharpeRatio,
    "DrawDown": bt.analyzers.DrawDown,
    "Returns": bt.analyzers.Returns,
    "TradeAnalyzer": bt.analyzers.TradeAnalyzer,
    "TimeReturn": bt.analyzers.TimeReturn,
    "SortinoRatio": SortinoRatio,
    "Calmar": bt.analyzers.Calmar,
    "SQN": bt.analyzers.SQN,
}

# Default analyzer selection
DEFAULT_ANALYZERS = [
    "SharpeRatio",
    "DrawDown",
    "Returns",
    "TradeAnalyzer",
    "TimeReturn",
    "SortinoRatio",
]

# Optimization parameters
OPTIMIZATION_PARAMETERS = [
    "total_return",
    "sharpe_ratio",
    "max_drawdown",
    "sortino_ratio",
    "calmar",
    "time_return",
]

# Timeframe options
TIMEFRAMES = ["5m", "1m", "2m", "3m", "4m", "15m", "1h", "4h", "1d"]

# Analysis types
ANALYSIS_TYPES = ["Backtest", "Optimization", "Walk-Forward", "Complete Backtest"]

# Ticker input methods
TICKER_INPUT_METHODS = ["Select from List", "Enter Custom Ticker"]

# Page configuration
PAGE_CONFIG = {
    "page_title": "Comprehensive Backtesting Framework",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Custom CSS styles
CUSTOM_CSS = """
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
"""
