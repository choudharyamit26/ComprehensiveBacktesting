import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class ChaikinMoneyFlow(bt.Indicator):
    lines = ("cmf",)
    params = (("period", 20),)
    plotinfo = dict(subplot=True)

    def __init__(self):
        hlc = self.data.high - self.data.low
        clv = (
            (self.data.close - self.data.low) - (self.data.high - self.data.close)
        ) / (hlc + 1e-10)
        mfv = clv * self.data.volume
        self.lines.cmf = bt.ind.SumN(mfv, period=self.p.period) / bt.ind.SumN(
            self.data.volume, period=self.p.period
        )


class OnBalanceVolume(bt.Indicator):
    lines = ("obv",)
    plotinfo = dict(subplot=True)

    def __init__(self):
        self.addminperiod(1)

    def next(self):
        if self.data.close[0] > self.data.close[-1]:
            self.lines.obv[0] = self.lines.obv[-1] + self.data.volume[0]
        elif self.data.close[0] < self.data.close[-1]:
            self.lines.obv[0] = self.lines.obv[-1] - self.data.volume[0]
        else:
            self.lines.obv[0] = self.lines.obv[-1]


class OBVCMFStrategy(bt.Strategy):
    """
    OBV + CMF Volume Confluence Intraday Strategy

    Entry:
    - Long: OBV rising AND CMF > threshold
    - Short: OBV falling AND CMF < -threshold

    Exit:
    - Long: OBV falling OR CMF < -threshold
    - Short: OBV rising OR CMF > threshold

    Force exit all positions at 15:15 IST.
    """

    params = (
        ("cmf_period", 20),
        ("cmf_threshold", 0.0),
        ("verbose", False),
    )

    optimization_params = {
        "cmf_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "cmf_threshold": {"type": "float", "low": 0.0, "high": 0.2, "step": 0.05},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.obv = OnBalanceVolume(self.data)
        self.cmf = ChaikinMoneyFlow(self.data, period=self.params.cmf_period)

        self.obv_rising = self.obv > self.obv(-1)
        self.obv_falling = self.obv < self.obv(-1)
        self.cmf_positive = self.cmf > self.params.cmf_threshold
        self.cmf_negative = self.cmf < -self.params.cmf_threshold

        self.bullish_entry = bt.And(self.obv_rising, self.cmf_positive)
        self.bearish_entry = bt.And(self.obv_falling, self.cmf_negative)
        self.bullish_exit = bt.Or(self.obv_falling, self.cmf_negative)
        self.bearish_exit = bt.Or(self.obv_rising, self.cmf_positive)

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = self.params.cmf_period + 5
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug("Initialized OBVCMFStrategy")

    def next(self):
        if len(self) < self.warmup_period:
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
            return

        if np.isnan(self.obv[0]) or np.isnan(self.cmf[0]):
            return

        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "obv": self.obv[0],
                "cmf": self.cmf[0],
            }
        )

        if not self.position:
            if self.bullish_entry[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )
            elif self.bearish_entry[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )
        else:
            if self.position.size > 0 and self.bullish_exit[0]:
                self.order = self.sell()
                self.order_type = "exit_long"
                trade_logger.info(
                    f"EXIT LONG | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )
            elif self.position.size < 0 and self.bearish_exit[0]:
                self.order = self.buy()
                self.order_type = "exit_short"
                trade_logger.info(
                    f"EXIT SHORT | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt).astimezone(
                pytz.timezone("Asia/Kolkata")
            )
            if self.order_type.startswith("enter"):
                direction = "long" if order.isbuy() else "short"
                self.open_positions.append(
                    {
                        "entry_time": exec_dt,
                        "entry_price": order.executed.price,
                        "size": order.executed.size,
                        "commission": order.executed.comm,
                        "ref": order.ref,
                        "direction": direction,
                    }
                )
                trade_logger.info(
                    f"{'BUY' if order.isbuy() else 'SELL'} EXECUTED | Ref: {order.ref} | Price: {order.executed.price:.2f}"
                )
            elif self.order_type.startswith("exit"):
                if self.open_positions:
                    entry_info = self.open_positions.pop(0)
                    pnl = (order.executed.price - entry_info["entry_price"]) * abs(
                        entry_info["size"]
                    )
                    if entry_info["direction"] == "short":
                        pnl = -pnl
                    total_commission = entry_info["commission"] + abs(
                        order.executed.comm
                    )
                    self.completed_trades.append(
                        {
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
                            "direction": entry_info["direction"].capitalize(),
                        }
                    )
                    self.trade_count += 1
                    trade_logger.info(
                        f"EXIT EXECUTED | Ref: {order.ref} | Net PnL: {pnl - total_commission:.2f}"
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
        return {
            "cmf_period": trial.suggest_int("cmf_period", 10, 30),
            "cmf_threshold": trial.suggest_float("cmf_threshold", 0.0, 0.2, step=0.05),
        }

    @classmethod
    def get_min_data_points(cls, params):
        try:
            return params.get("cmf_period", 20) + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 25
