from datetime import timedelta, datetime
import backtrader as bt
import logging

from .data import get_data_sync, validate_data
from .registry import get_strategy


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEFAULT_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "SBIN.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "HCLTECH.NS",
    "AXISBANK.NS",
]


def run_backtest(
    data,
    strategy_class,
    interval,
    ticker,
    start_date,
    end_date,
    initial_cash=100000.0,
    commission=0.001,
    **strategy_params,
):
    """Run a backtest with specified parameters."""
    logger.info(
        f"Running backtest for {ticker} with strategy {strategy_class} and type {type(strategy_class)}"
    )
    if isinstance(strategy_class, str):
        strategy_class = get_strategy(strategy_class)

    logger.info(
        f"Running backtest for {ticker} with strategy {strategy_class.__name__}"
    )

    # Get data with proper timezone handling
    data_df = data.copy()

    if not validate_data(data_df, strict=False):  # Added strict mode
        logger.warning(f"Data validation warnings for {ticker}. Proceeding anyway.")

    # Calculate min data points
    min_data_points = (
        strategy_class.get_min_data_points(strategy_params)
        if hasattr(strategy_class, "get_min_data_points")
        else 50
    )

    # Convert start_date to datetime if it's a string
    if isinstance(start_date, str):
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start_dt = start_date

    # Extend date range if insufficient data
    max_attempts = 10  # Prevent infinite loop
    attempts = 0

    while len(data_df) < min_data_points and attempts < max_attempts:
        attempts += 1
        # Extend the start date backwards to get more historical data
        start_dt -= timedelta(days=30)  # Go back 30 days each attempt (was 5 days)
        extended_start = start_dt.strftime("%Y-%m-%d")

        logger.info(
            f"Insufficient data ({len(data_df)} < {min_data_points}). "
            f"Extending start date to {extended_start}"
        )

        data_df = get_data_sync(ticker, extended_start, end_date, interval=interval)

    if len(data_df) < min_data_points:
        logger.error(
            f"Insufficient data: {len(data_df)} rows available, "
            f"{min_data_points} required. Skipping backtest."
        )
        # Return empty results and None cerebro to signal failure upstream
        return [], None

    # Check if data_df is empty or has insufficient data
    if data_df.empty:
        logger.error(f"No data available for {ticker} in the specified date range.")
        return [], None

    data = bt.feeds.PandasData(
        dataname=data_df,
        datetime=None,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        openinterest=None,
    )

    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class, **strategy_params)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    # Add analyzer classes instead of instantiated objects
    from comprehensive_backtesting.parameter_optimization import SortinoRatio

    print(bt.analyzers.Transactions)
    analyzer_classes = [
        (
            bt.analyzers.SharpeRatio,
            {"_name": "sharpe", "timeframe": bt.TimeFrame.Minutes, "compression": 5},
        ),
        (bt.analyzers.DrawDown, {"_name": "drawdown"}),
        (
            bt.analyzers.Returns,
            {"_name": "returns", "timeframe": bt.TimeFrame.Minutes, "compression": 5},
        ),
        (bt.analyzers.TradeAnalyzer, {"_name": "trades"}),
        (
            bt.analyzers.Calmar,
            {"_name": "calmar", "timeframe": bt.TimeFrame.Minutes, "compression": 5},
        ),
        (
            bt.analyzers.TimeReturn,
            {
                "_name": "timereturn",
                "timeframe": bt.TimeFrame.Minutes,
                "compression": 5,
            },
        ),
        (bt.analyzers.SQN, {"_name": "sqn"}),
        (SortinoRatio, {"_name": "sortino"}),
        (bt.analyzers.Transactions, {"_name": "trade_history"}),
    ]

    for analyzer_class, params in analyzer_classes:
        cerebro.addanalyzer(analyzer_class, **params)

    # Add data to cerebro
    cerebro.adddata(data, name=interval)

    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
    print(f"Data range: {data_df.index.min()} to {data_df.index.max()}")
    print(f"Total bars: {len(data_df)}")

    try:
        results = cerebro.run()
        print(f"Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
        return results, cerebro
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise


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
