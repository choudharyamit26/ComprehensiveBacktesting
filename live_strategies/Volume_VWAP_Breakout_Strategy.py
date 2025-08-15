import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


# Custom VWAP Indicator
class VWAP(bt.Indicator):
    alias = ("VolumeWeightedAveragePrice",)
    lines = ("vwap",)
    params = (("period", 20),)
    plotinfo = dict(subplot=False)
    plotlines = dict(vwap=dict(color="blue", linestyle="-", linewidth=2.0))

    def __init__(self):
        self.hlc = (self.data.high + self.data.low + self.data.close) / 3.0
        self.hlc_volume_sum = bt.ind.SumN(
            self.hlc * self.data.volume, period=self.p.period
        )
        self.volume_sum = bt.ind.SumN(self.data.volume, period=self.p.period)
        self.lines.vwap = bt.DivByZero(self.hlc_volume_sum, self.volume_sum, None)
        super(VWAP, self).__init__()


class VolumeVWAPBreakoutStrategy(bt.Strategy):
    """
    Volume + VWAP Breakout Strategy

    This strategy uses significant volume surges combined with VWAP breakouts to identify
    strong directional moves.

    Strategy Type: BREAKOUT
    ===============================
    This strategy enters trades when price breaks VWAP with volume exceeding 200% of average.
    It exits when price returns to VWAP or volume normalizes.

    Strategy Logic:
    ==============
    Long Position Rules:
    - Entry: Price breaks above VWAP AND volume > 200% of average volume
    - Exit: Price returns to VWAP OR volume < average volume

    Short Position Rules:
    - Entry: Price breaks below VWAP AND volume > 200% of average volume
    - Exit: Price returns to VWAP OR volume < average volume

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST
    - Uses warmup period for indicator stability
    - Prevents order overlap

    Indicators Used:
    ===============
    - VWAP: Volume Weighted Average Price
    - Volume: Compares current volume to average volume

    Parameters:
    ==========
    - vol_period (int): Volume average period (default: 20)
    - vol_multiplier (float): Volume threshold multiplier (default: 2.0)
    - verbose (bool): Enable detailed logging (default: False)

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(VolumeVWAPBreakoutStrategy, vol_period=20, vol_multiplier=2.0)
    cerebro.run()

    Best Market Conditions:
    ======================
    - High volume breakout scenarios
    - Avoid low volume or range-bound markets
    """

    params = (
        ("vol_period", 20),
        ("vol_multiplier", 2.0),
        ("verbose", False),
    )

    optimization_params = {
        "vol_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "vol_multiplier": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.5},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.vwap = VWAP(self.data, period=self.params.vol_period)  # Use custom VWAP
        self.avg_volume = btind.SimpleMovingAverage(
            self.data.volume, period=self.params.vol_period
        )

        # Optional: Add SuperTrend and OnBalanceVolume (commented out)
        # self.supertrend = SuperTrend(self.data, period=7, multiplier=3.0)
        # self.obv = OnBalanceVolume(self.data, length=12)

        self.high_volume = (
            self.data.volume > self.avg_volume * self.params.vol_multiplier
        )
        self.bullish_entry = bt.And(self.data.close > self.vwap.vwap, self.high_volume)
        self.bearish_entry = bt.And(self.data.close < self.vwap.vwap, self.high_volume)
        self.bullish_exit = bt.Or(
            self.data.close <= self.vwap.vwap, self.data.volume < self.avg_volume
        )
        self.bearish_exit = bt.Or(
            self.data.close >= self.vwap.vwap, self.data.volume < self.avg_volume
        )

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = self.params.vol_period + 5
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(
            f"Initialized VolumeVWAPBreakoutStrategy with params: {self.params}"
        )

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

        if np.isnan(self.vwap.vwap[0]) or np.isnan(self.avg_volume[0]):
            logger.debug(f"Invalid indicator values at bar {len(self)}")
            return

        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "vwap": self.vwap.vwap[0],
                "volume": self.data.volume[0],
            }
        )

        if not self.position:
            if self.bullish_entry[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (VWAP Breakout + High Volume) | Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f}"
                )
            elif self.bearish_entry[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (VWAP Breakout + High Volume) | Time: {bar_time_ist} | "
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
            "vol_period": trial.suggest_int("vol_period", 10, 30),
            "vol_multiplier": trial.suggest_float("vol_multiplier", 1.5, 3.0, step=0.5),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            vol_period = params.get("vol_period", 20)
            return vol_period + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 25
