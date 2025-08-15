import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class VWAP(bt.Indicator):
    alias = ("VolumeWeightedAveragePrice",)
    lines = ("vwap",)
    plotinfo = dict(subplot=False)
    plotlines = dict(vwap=dict(color="blue", linestyle="-", linewidth=2.0))

    def __init__(self):
        self.hlc = (self.data.high + self.data.low + self.data.close) / 3.0
        self.hlc_volume = self.hlc * self.data.volume
        # Cumulative sums for session-based VWAP
        self.cum_hlc_volume = self.hlc_volume
        self.cum_volume = self.data.volume
        self.lines.vwap = self.hlc  # Default to HLC for the first bar
        super(VWAP, self).__init__()

    def next(self):
        # Get current bar's datetime in IST
        bar_time = (
            self.datas[0].datetime.datetime(0).astimezone(pytz.timezone("Asia/Kolkata"))
        )
        current_time = bar_time.time()

        # Reset at the start of the session (9:15 AM IST or new day)
        if len(self) > 1:
            prev_time = (
                self.datas[0]
                .datetime.datetime(-1)
                .astimezone(pytz.timezone("Asia/Kolkata"))
                .time()
            )
            if (
                prev_time < datetime.time(9, 15)
                or bar_time.date() != self.datas[0].datetime.datetime(-1).date()
            ):
                # Reset cumulative sums at the start of a new session
                self.cum_hlc_volume[0] = self.hlc_volume[0]
                self.cum_volume[0] = self.data.volume[0]
            else:
                # Accumulate for the current session
                self.cum_hlc_volume[0] = self.cum_hlc_volume[-1] + self.hlc_volume[0]
                self.cum_volume[0] = self.cum_volume[-1] + self.data.volume[0]
        else:
            # First bar of the session
            self.cum_hlc_volume[0] = self.hlc_volume[0]
            self.cum_volume[0] = self.data.volume[0]

        # Calculate VWAP: use HLC if volume is zero, otherwise compute VWAP
        if self.cum_volume[0] != 0:
            self.lines.vwap[0] = self.cum_hlc_volume[0] / self.cum_volume[0]
        else:
            # Use current HLC or previous VWAP (if available) to avoid None
            self.lines.vwap[0] = self.hlc[0] if len(self) == 1 else self.lines.vwap[-1]


class VWAPEMAVolumeStrategy(bt.Strategy):
    """
    VWAP + EMA + Volume Strategy

    This strategy combines VWAP, EMA, and volume analysis to identify intraday trading opportunities.
    It enters trades when price aligns with VWAP and EMA, confirmed by volume spikes.

    Strategy Type: TREND + VOLUME
    =============================
    This strategy uses VWAP as a dynamic support/resistance, EMA for trend direction, and volume
    to confirm momentum. It exits when price fails to hold VWAP or EMA levels.

    Strategy Logic:
    ==============
    Long Position Rules:
    - Entry: Price above VWAP AND price above EMA AND volume > average volume
    - Exit: Price falls below VWAP OR price falls below EMA

    Short Position Rules:
    - Entry: Price below VWAP AND price below EMA AND volume > average volume
    - Exit: Price rises above VWAP OR price rises above EMA

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST
    - Uses warmup period for indicator stability
    - Prevents order overlap

    Indicators Used:
    ===============
    - VWAP: Volume Weighted Average Price (session-based)
    - EMA: Exponential Moving Average for trend direction
    - Volume: Simple Moving Average of volume to detect spikes

    Parameters:
    ==========
    - ema_period (int): EMA period (default: 20)
    - vol_period (int): Volume SMA period (default: 20)
    - vol_mult (float): Volume multiplier for spike detection (default: 1.5)
    - verbose (bool): Enable detailed logging (default: False)

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(VWAPEMAVolumeStrategy, ema_period=20, vol_period=20)
    cerebro.run()

    Best market conditions:
    ======================
    - Trending markets with strong volume
    - Avoid low-volume, choppy markets
    """

    params = (
        ("ema_period", 20),
        ("vol_period", 20),
        ("vol_mult", 1.5),
        ("verbose", False),
    )

    optimization_params = {
        "ema_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "vol_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "vol_mult": {"type": "float", "low": 1.0, "high": 2.0, "step": 0.1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.vwap = VWAP(self.data)  # Use the custom VWAP class directly
        self.ema = btind.EMA(self.data.close, period=self.params.ema_period)
        self.avg_volume = btind.SMA(self.data.volume, period=self.params.vol_period)

        self.bullish_entry = bt.And(
            self.data.close > self.vwap,
            self.data.close > self.ema,
            self.data.volume > self.avg_volume * self.params.vol_mult,
        )
        self.bearish_entry = bt.And(
            self.data.close < self.vwap,
            self.data.close < self.ema,
            self.data.volume > self.avg_volume * self.params.vol_mult,
        )
        self.bullish_exit = bt.Or(
            self.data.close < self.vwap, self.data.close < self.ema
        )
        self.bearish_exit = bt.Or(
            self.data.close > self.vwap, self.data.close > self.ema
        )

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = max(self.params.ema_period, self.params.vol_period) + 5
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized VWAPEMAVolumeStrategy with params: {self.params}")

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
            np.isnan(self.vwap[0])
            or np.isnan(self.ema[0])
            or np.isnan(self.avg_volume[0])
        ):
            logger.debug(f"Invalid indicator values at bar {len(self)}")
            return

        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "vwap": self.vwap[0],
                "ema": self.ema[0],
                "volume": self.data.volume[0],
            }
        )

        if not self.position:
            if self.bullish_entry[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Above VWAP + EMA + High Volume) | Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f}"
                )
            elif self.bearish_entry[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Below VWAP + EMA + High Volume) | Time: {bar_time_ist} | "
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
            "ema_period": trial.suggest_int("ema_period", 10, 30),
            "vol_period": trial.suggest_int("vol_period", 10, 30),
            "vol_mult": trial.suggest_float("vol_mult", 1.0, 2.0, step=0.1),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            ema_period = params.get("ema_period", 20)
            vol_period = params.get("vol_period", 20)
            return max(ema_period, vol_period) + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 25
