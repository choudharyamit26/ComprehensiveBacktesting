import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class ADXEMAMACDStrategy(bt.Strategy):
    """
    ADX + EMA + MACD Trend Strategy

    This strategy combines ADX, EMA, and MACD to identify strong trending markets with momentum.
    It enters trades when all indicators confirm a trend and exits when trend strength weakens.

    Strategy Type: TREND + MOMENTUM
    =============================
    This strategy uses ADX for trend strength, EMA for trend direction, and MACD for momentum.
    Exits occur when trend strength or momentum deteriorates.

    Strategy Logic:
    ==============
    Long Position Rules:
    - Entry: ADX > threshold AND price above EMA AND MACD line > signal line
    - Exit: ADX < threshold OR price below EMA OR MACD line < signal line

    Short Position Rules:
    - Entry: ADX > threshold AND price below EMA AND MACD line < signal line
    - Exit: ADX < threshold OR price above EMA OR MACD line > signal line

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST
    - Uses warmup period for indicator stability
    - Prevents order overlap

    Indicators Used:
    ===============
    - ADX: Average Directional Index for trend strength
    - EMA: Exponential Moving Average for trend direction
    - MACD: Moving Average Convergence Divergence for momentum

    Parameters:
    ==========
    - adx_period (int): ADX period (default: 14)
    - adx_threshold (float): ADX threshold for trend strength (default: 25.0)
    - ema_period (int): EMA period (default: 20)
    - macd_fast (int): MACD fast EMA period (default: 12)
    - macd_slow (int): MACD slow EMA period (default: 26)
    - macd_signal (int): MACD signal line period (default: 9)
    - verbose (bool): Enable detailed logging (default: False)

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(ADXEMAMACDStrategy, adx_period=14, ema_period=20)
    cerebro.run()

    Best market conditions:
    ======================
    - Strong trending markets with high momentum
    - Avoid range-bound or low-momentum markets
    """

    params = (
        ("adx_period", 14),
        ("adx_threshold", 25.0),
        ("ema_period", 20),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("verbose", False),
    )

    optimization_params = {
        "adx_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "adx_threshold": {"type": "float", "low": 20.0, "high": 30.0, "step": 1.0},
        "ema_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "macd_fast": {"type": "int", "low": 8, "high": 16, "step": 1},
        "macd_slow": {"type": "int", "low": 20, "high": 30, "step": 1},
        "macd_signal": {"type": "int", "low": 5, "high": 12, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.adx = btind.ADX(self.data, period=self.params.adx_period)
        self.ema = btind.EMA(self.data.close, period=self.params.ema_period)
        self.macd = btind.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal,
        )

        self.bullish_entry = bt.And(
            self.adx > self.params.adx_threshold,
            self.data.close > self.ema,
            self.macd.macd > self.macd.signal,
        )
        self.bearish_entry = bt.And(
            self.adx > self.params.adx_threshold,
            self.data.close < self.ema,
            self.macd.macd < self.macd.signal,
        )
        self.bullish_exit = bt.Or(
            self.adx < self.params.adx_threshold,
            self.data.close < self.ema,
            self.macd.macd < self.macd.signal,
        )
        self.bearish_exit = bt.Or(
            self.adx < self.params.adx_threshold,
            self.data.close > self.ema,
            self.macd.macd > self.macd.signal,
        )

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(self.params.adx_period, self.params.ema_period, self.params.macd_slow)
            + 5
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized ADXEMAMACDStrategy with params: {self.params}")

    def next(self):
        if len(self) < self.warmup_period:
            logger.debug(f"Skipping bar {len(self)}: still in warmup period")
            return

        if not self.ready:
            self.ready = True
            logger.info(f"Strategy ready at bar {len(self)}")

        bar_time = self.datas[0].datetime.datetime(0)
        bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
        current_time = bar_time_ist.time()

        if current_time >= datetime.time(15, 15):
            if self.position:
                self.close()
                trade_logger.info("Force closed all positions at 15:15 IST")
            return

        if not (datetime.time(9, 15) <= current_time <= datetime.time(15, 5)):
            return

        if self.order:
            logger.debug(f"Order pending at bar {len(self)}")
            return

        if (
            np.isnan(self.adx[0])
            or np.isnan(self.ema[0])
            or np.isnan(self.macd.macd[0])
        ):
            logger.debug(f"Invalid indicator values at bar {len(self)}")
            return

        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "adx": self.adx[0],
                "ema": self.ema[0],
                "macd": self.macd.macd[0],
            }
        )

        if not self.position:
            if self.bullish_entry[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (High ADX + Above EMA + MACD Bullish) | Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f}"
                )
            elif self.bearish_entry[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (High ADX + Below EMA + MACD Bearish) | Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f}"
                )
        else:
            if self.position.size > 0 and self.bullish_exit[0]:
                self.order = self.sell()
                self.order_type = "exit_long"
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long) | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )
            elif self.position.size < 0 and self.bearish_exit[0]:
                self.order = self.buy()
                self.order_type = "exit_short"
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short) | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt).astimezone(
                pytz.timezone("Asia/Kolkata")
            )
            if self.order_type == "enter_long" and order.isbuy():
                position_info = {
                    "entry_time": exec_dt,
                    "entry_price": order.executed.price,
                    "size": order.executed.size,
                    "commission": order.executed.comm,
                    "ref": order.ref,
                    "direction": "long",
                }
                self.open_positions.append(position_info)
                trade_logger.info(
                    f"BUY EXECUTED (Enter Long) | Ref: {order.ref} | Price: {order.executed.price:.2f}"
                )
            elif self.order_type == "enter_short" and order.issell():
                position_info = {
                    "entry_time": exec_dt,
                    "entry_price": order.executed.price,
                    "size": order.executed.size,
                    "commission": order.executed.comm,
                    "ref": order.ref,
                    "direction": "short",
                }
                self.open_positions.append(position_info)
                trade_logger.info(
                    f"SELL EXECUTED (Enter Short) | Ref: {order.ref} | Price: {order.executed.price:.2f}"
                )
            elif self.order_type == "exit_long" and order.issell():
                if self.open_positions:
                    entry_info = self.open_positions.pop(0)
                    pnl = (order.executed.price - entry_info["entry_price"]) * abs(
                        entry_info["size"]
                    )
                    total_commission = entry_info["commission"] + abs(
                        order.executed.comm
                    )
                    trade_info = {
                        "ref": order.ref,
                        "entry_time": entry_info["entry_time"],
                        "exit_time": exec_dt,
                        "entry_price": entry_info["entry_price"],
                        "exit_price": order.executed.price,
                        "size": abs(entry_info["size"]),
                        "pnl": pnl,
                        "pnl_net": pnl - total_commission,
                        "commission": total_commission,
                        "status": "Won" if pnl > 0 else "Lost",
                        "direction": "Long",
                    }
                    self.completed_trades.append(trade_info)
                    self.trade_count += 1
                    trade_logger.info(
                        f"SELL EXECUTED (Exit Long) | Ref: {order.ref} | PnL: {pnl:.2f}"
                    )
            elif self.order_type == "exit_short" and order.isbuy():
                if self.open_positions:
                    entry_info = self.open_positions.pop(0)
                    pnl = (entry_info["entry_price"] - order.executed.price) * abs(
                        entry_info["size"]
                    )
                    total_commission = entry_info["commission"] + abs(
                        order.executed.comm
                    )
                    trade_info = {
                        "ref": order.ref,
                        "entry_time": entry_info["entry_time"],
                        "exit_time": exec_dt,
                        "entry_price": entry_info["entry_price"],
                        "exit_price": order.executed.price,
                        "size": abs(entry_info["size"]),
                        "pnl": pnl,
                        "pnl_net": pnl - total_commission,
                        "commission": total_commission,
                        "status": "Won" if pnl > 0 else "Lost",
                        "direction": "Short",
                    }
                    self.completed_trades.append(trade_info)
                    self.trade_count += 1
                    trade_logger.info(
                        f"BUY EXECUTED (Exit Short) | Ref: {order.ref} | PnL: {pnl:.2f}"
                    )
        if order.status in [
            order.Completed,
            order.Canceled,
            order.Margin,
            order.Rejected,
        ]:
            self.order = None
            self.order_type = None

    def notify_trade(self, trade):
        if trade.isclosed:
            trade_logger.info(
                f"TRADE CLOSED | Ref: {trade.ref} | Profit: {trade.pnl:.2f} | "
                f"Net Profit: {trade.pnlcomm:.2f} | Bars Held: {trade.barlen}"
            )

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        params = {
            "adx_period": trial.suggest_int("adx_period", 10, 20),
            "adx_threshold": trial.suggest_float("adx_threshold", 20.0, 30.0, step=1.0),
            "ema_period": trial.suggest_int("ema_period", 10, 30),
            "macd_fast": trial.suggest_int("macd_fast", 8, 16),
            "macd_slow": trial.suggest_int("macd_slow", 20, 30),
            "macd_signal": trial.suggest_int("macd_signal", 5, 12),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            adx_period = params.get("adx_period", 14)
            ema_period = params.get("ema_period", 20)
            macd_slow = params.get("macd_slow", 26)
            return max(adx_period, ema_period, macd_slow) + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 35
