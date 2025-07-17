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


class VWAPBBADXStrategy(bt.Strategy):
    """
    VWAP + Bollinger Bands + ADX Strategy

    This strategy combines VWAP, Bollinger Bands, and ADX to capture strong intraday trends.
    VWAP acts as a reference level, Bollinger Bands identify breakouts, and ADX confirms trend strength.

    Strategy Type: TREND + MOMENTUM
    =============================
    This strategy uses VWAP deviation, Bollinger Band breakouts, and ADX to confirm strong trends.
    It exits when price reverts to VWAP or trend strength weakens.

    Strategy Logic:
    ==============
    Long Position Rules:
    - Entry: Price above VWAP AND price above upper BB AND ADX > threshold
    - Exit: Price falls below VWAP OR ADX falls below threshold

    Short Position Rules:
    - Entry: Price below VWAP AND price below lower BB AND ADX > threshold
    - Exit: Price rises above VWAP OR ADX falls below threshold

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST
    - Uses warmup period for indicator stability
    - Prevents order overlap

    Indicators Used:
    ===============
    - VWAP: Volume Weighted Average Price (session-based)
    - Bollinger Bands: Middle Band (SMA), Upper/Lower Bands (±SD)
    - ADX: Average Directional Index for trend strength

    Parameters:
    ==========
    - bb_period (int): Bollinger Bands period (default: 20)
    - bb_dev (float): BB standard deviation multiplier (default: 2.0)
    - adx_period (int): ADX period (default: 14)
    - adx_threshold (float): ADX threshold for trend strength (default: 25.0)
    - verbose (bool): Enable detailed logging (default: False)

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(VWAPBBADXStrategy, bb_period=20, adx_period=14)
    cerebro.run()

    Best market conditions:
    ======================
    - Strong trending markets with high volatility
    - Avoid range-bound markets
    """

    params = (
        ("bb_period", 20),
        ("bb_dev", 2.0),
        ("adx_period", 14),
        ("adx_threshold", 25.0),
        ("verbose", False),
    )

    optimization_params = {
        "bb_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "bb_dev": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.1},
        "adx_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "adx_threshold": {"type": "float", "low": 20.0, "high": 30.0, "step": 1.0},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.vwap = VWAP(self.data)
        self.bb = btind.BollingerBands(
            self.data.close, period=self.params.bb_period, devfactor=self.params.bb_dev
        )
        self.adx = btind.ADX(self.data, period=self.params.adx_period)

        self.bullish_entry = bt.And(
            self.data.close > self.vwap,
            self.data.close > self.bb.top,
            self.adx > self.params.adx_threshold,
        )
        self.bearish_entry = bt.And(
            self.data.close < self.vwap,
            self.data.close < self.bb.bot,
            self.adx > self.params.adx_threshold,
        )
        self.bullish_exit = bt.Or(
            self.data.close < self.vwap, self.adx < self.params.adx_threshold
        )
        self.bearish_exit = bt.Or(
            self.data.close > self.vwap, self.adx < self.params.adx_threshold
        )

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = max(self.params.bb_period, self.params.adx_period) + 5
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized VWAPBBADXStrategy with params: {self.params}")

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

        if np.isnan(self.vwap[0]) or np.isnan(self.bb.mid[0]) or np.isnan(self.adx[0]):
            logger.debug(f"Invalid indicator values at bar {len(self)}")
            return

        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "vwap": self.vwap[0],
                "bb_top": self.bb.top[0],
                "adx": self.adx[0],
            }
        )

        if not self.position:
            if self.bullish_entry[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Above VWAP + BB + High ADX) | Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f}"
                )
            elif self.bearish_entry[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Below VWAP + BB + High ADX) | Time: {bar_time_ist} | "
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
            "bb_period": trial.suggest_int("bb_period", 10, 30),
            "bb_dev": trial.suggest_float("bb_dev", 1.5, 2.5, step=0.1),
            "adx_period": trial.suggest_int("adx_period", 10, 20),
            "adx_threshold": trial.suggest_float("adx_threshold", 20.0, 30.0, step=1.0),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            bb_period = params.get("bb_period", 20)
            adx_period = params.get("adx_period", 14)
            return max(bb_period, adx_period) + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 25
