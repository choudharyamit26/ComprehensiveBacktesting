import backtrader as bt
import logging

from stratgies.registry import get_strategy
from .data import get_data, validate_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_backtest(
    strategy_class,
    ticker="AAPL",
    start_date="2022-01-01",
    end_date="2025-06-01",
    initial_cash=100000.0,
    commission=0.001,
    **strategy_params,
):
    """Run a backtest with specified parameters.

    Args:
        strategy_class: Backtrader strategy class or string name in registry.
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        initial_cash (float): Initial portfolio cash.
        commission (float): Broker commission rate.
        **strategy_params: Strategy-specific parameters.

    Returns:
        tuple: (results, cerebro) - Backtrader results and cerebro instance.

    Raises:
        ValueError: If data or strategy is invalid.
    """
    if isinstance(strategy_class, str):
        strategy_class = get_strategy(strategy_class)

    logger.info(
        f"Running backtest for {ticker} with strategy {strategy_class.__name__}"
    )
    cerebro = bt.Cerebro()
    data_df = get_data(ticker, start_date, end_date)
    if not validate_data(data_df):
        raise ValueError(f"Invalid data for {ticker}. Check data source or date range.")

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

    cerebro.addstrategy(strategy_class, **strategy_params)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Calmar, _name="calmar")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annualreturn")
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="transactions")
    cerebro.addanalyzer(bt.analyzers.PositionsValue, _name="positionsvalue")
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")

    cerebro.adddata(data)
    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}")

    try:
        results = cerebro.run()
        print(f"Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
        return results, cerebro
    except Exception as e:
        logger.error(
            f"Backtest failed: {str(e)}. Check strategy parameters or data compatibility."
        )
        raise


# def run_backtest(
#     strategy_class,
#     ticker: str,
#     start_date: str,
#     end_date: str,
#     initial_cash: float = 100000.0,
#     commission: float = 0.00,
#     **strategy_params
# ):
#     """Run a backtest for a given strategy and data.

#     Args:
#         strategy_class: Backtrader strategy class.
#         ticker (str): Stock ticker symbol.
#         start_date (str): Start date in 'YYYY-MM-DD' format.
#         end_date (str): End date in 'YYYY-MM-DD' format.
#         initial_cash (float): Initial portfolio cash.
#         commission (float): Broker commission rate.
#         **strategy_params: Strategy-specific parameters.

#     Returns:
#         Tuple: (results, cerebro) from Backtrader.
#     """
#     logger.info(f"Running backtest for {ticker} from {start_date} to {end_date}")
#     try:
#         data_df = get_data(ticker, start_date, end_date)
#         if data_df is None or data_df.empty:
#             raise ValueError(f"No data available for {ticker} from {start_date} to {end_date}")

#         data = bt.feeds.PandasData(
#             dataname=data_df,
#             datetime=None,
#             open="Open",
#             high="High",
#             low="Low",
#             close="Close",
#             volume="Volume",
#             openinterest=None,
#         )

#         cerebro = bt.Cerebro()
#         cerebro.addstrategy(strategy_class, **strategy_params)
#         cerebro.adddata(data)
#         cerebro.broker.setcash(initial_cash)
#         cerebro.broker.setcommission(commission=commission)
#         cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
#         cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
#         results = cerebro.run()
#         return results[0], cerebro

#     except Exception as e:
#         logger.error(f"Backtest failed: {str(e)}")
#         raise
