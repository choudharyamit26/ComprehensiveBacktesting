import backtrader as bt
import logging
import pandas as pd

from .data import get_data_sync, validate_data
from stratgies.registry import get_strategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_backtest(
    strategy_class,
    ticker="AAPL",
    start_date="2022-01-01",
    end_date="2025-06-01",
    initial_cash=100000.0,
    commission=0.001,
    interval="5m",
    **strategy_params,
):
    """Run a backtest with specified parameters."""
    if isinstance(strategy_class, str):
        strategy_class = get_strategy(strategy_class)

    logger.info(
        f"Running backtest for {ticker} with strategy {strategy_class.__name__}"
    )

    # Get data with proper timezone handling
    data_df = get_data_sync(ticker, start_date, end_date, interval=interval)

    if not validate_data(data_df, strict=False):  # Added strict mode
        logger.warning(f"Data validation warnings for {ticker}. Proceeding anyway.")

    # Calculate min data points
    min_data_points = (
        strategy_class.get_min_data_points(strategy_params)
        if hasattr(strategy_class, "get_min_data_points")
        else 50
    )

    if len(data_df) < min_data_points:
        logger.warning(
            f"Insufficient data: {len(data_df)} rows available, "
            f"{min_data_points} required. Continuing but trades may be limited."
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

    # CORRECTED: Add analyzer classes instead of instantiated objects
    analyzer_classes = [
        (bt.analyzers.SharpeRatio, {"_name": "sharpe"}),
        (bt.analyzers.DrawDown, {"_name": "drawdown"}),
        (bt.analyzers.Returns, {"_name": "returns"}),
        (bt.analyzers.TradeAnalyzer, {"_name": "trades"}),
        (bt.analyzers.Calmar, {"_name": "calmar"}),
        (bt.analyzers.TimeReturn, {"_name": "timereturn"}),
        (bt.analyzers.SQN, {"_name": "sqn"}),
    ]

    for analyzer_class, params in analyzer_classes:
        cerebro.addanalyzer(analyzer_class, **params)

    # Add 5-minute data
    cerebro.adddata(data, name="5m")

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
