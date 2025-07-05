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
    data_df = get_data_sync(ticker, start_date, end_date, interval=interval)

    if not validate_data(data_df, strict=False):
        logger.warning(f"Data validation warnings for {ticker}. Proceeding anyway.")

    # Calculate min data points
    min_data_points = (
        strategy_class.get_min_data_points(strategy_params, interval)
        if hasattr(strategy_class, "get_min_data_points")
        else 20
    )

    # DEBUG: Show data details
    logger.info(f"Data range: {data_df.index.min()} to {data_df.index.max()}")
    logger.info(f"Total bars: {len(data_df)}")
    logger.info(f"Required min bars: {min_data_points}")

    # Add strict check to prevent EMA calculation errors
    max_period = max(
        strategy_params.get("fast_ema_period", 12),
        strategy_params.get("slow_ema_period", 26),
        strategy_params.get("rsi_period", 14),
    )

    if len(data_df) < max_period:
        logger.error(
            f"Insufficient data for EMA calculation: {len(data_df)} rows available, "
            f"maximum period required {max_period}. Skipping backtest."
        )
        return [], None
    elif len(data_df) < min_data_points:
        logger.warning(
            f"Data might be insufficient: {len(data_df)} < {min_data_points} "
            f"but proceeding with backtest"
        )

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

    # Adjust timeframe and compression based on interval
    if interval == "1d":
        timeframe = bt.TimeFrame.Days
        compression = 1
    else:
        timeframe = bt.TimeFrame.Minutes
        compression = 5  # Default to 5-minute compression

    # CORRECTED: Add analyzer classes instead of instantiated objects
    from comprehensive_backtesting.parameter_optimization import SortinoRatio

    analyzer_classes = [
        (
            bt.analyzers.SharpeRatio,
            {"_name": "sharpe", "timeframe": timeframe, "compression": compression},
        ),
        (bt.analyzers.DrawDown, {"_name": "drawdown"}),
        (
            bt.analyzers.Returns,
            {"_name": "returns", "timeframe": timeframe, "compression": compression},
        ),
        (bt.analyzers.TradeAnalyzer, {"_name": "trades"}),
        (
            bt.analyzers.Calmar,
            {"_name": "calmar", "timeframe": timeframe, "compression": compression},
        ),
        (
            bt.analyzers.TimeReturn,
            {
                "_name": "timereturn",
                "timeframe": timeframe,
                "compression": compression,
            },
        ),
        (bt.analyzers.SQN, {"_name": "sqn"}),
        (SortinoRatio, {"_name": "sortino"}),
    ]

    for analyzer_class, params in analyzer_classes:
        cerebro.addanalyzer(analyzer_class, **params)

    # Add main data
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
