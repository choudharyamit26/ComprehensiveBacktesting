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


class VERV(bt.Strategy):
    """
    VWAP + EMA + RSI + Volume Deviation (VERV) Strategy
    Strategy Type: MEAN REVERSION + TREND + MOMENTUM + VOLUME
    ==========================================
    This strategy combines VWAP, EMA, RSI, and Volume Deviation for intraday trading on a 5-minute timeframe.

    Strategy Logic:
    ==============
    Long Entry: Price below VWAP + RSI rising + Price above EMA + Volume above average
    Short Entry: Price above VWAP + RSI falling + Price below EMA + Volume above average
    Exit: Price returns to VWAP or two or more signals conflict

    Parameters:
    ==========
    - vwap_period (int): VWAP period (default: 20)
    - ema_period (int): EMA period (default: 20)
    - rsi_period (int): RSI period (default: 14)
    - vol_sma_period (int): Volume SMA period (default: 14)
    - vol_threshold (float): Volume threshold multiplier (default: 1.5)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("vwap_period", 20),
        ("ema_period", 20),
        ("rsi_period", 14),
        ("vol_sma_period", 14),
        ("vol_threshold", 1.5),
        ("verbose", False),
    )

    optimization_params = {
        "vwap_period": {"type": "int", "low": 15, "high": 30, "step": 1},
        "ema_period": {"type": "int", "low": 15, "high": 30, "step": 1},
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "vol_sma_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "vol_threshold": {"type": "float", "low": 1.2, "high": 2.0, "step": 0.1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.vwap = VWAP(self.data, period=self.params.vwap_period)
        self.ema = btind.EMA(self.data.close, period=self.params.ema_period)
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.vol_sma = btind.SMA(self.data.volume, period=self.params.vol_sma_period)
        self.vol_ratio = self.data.volume / self.vol_sma

        # Debug: Log available lines and their types
        logger.debug(f"VWAP lines: {self.vwap.lines.getlinealiases()}")
        logger.debug(f"EMA lines: {self.ema.lines.getlinealiases()}")
        logger.debug(f"RSI lines: {self.rsi.lines.getlinealiases()}")
        logger.debug(f"Volume SMA lines: {self.vol_sma.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.vwap_period,
                self.params.ema_period,
                self.params.rsi_period,
                self.params.vol_sma_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized VERV with params: {self.params}")
        logger.info(
            f"VERV initialized with vwap_period={self.p.vwap_period}, "
            f"ema_period={self.p.ema_period}, rsi_period={self.p.rsi_period}, "
            f"vol_sma_period={self.p.vol_sma_period}, vol_threshold={self.p.vol_threshold}"
        )

    def next(self):
        if len(self) < self.warmup_period:
            logger.debug(
                f"Skipping bar {len(self)}: still in warmup period (need {self.warmup_period} bars)"
            )
            return

        if not self.ready:
            self.ready = True
            logger.info(f"Strategy ready at bar {len(self)}")

        bar_time = self.datas[0].datetime.datetime(0)
        bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
        current_time = bar_time_ist.time()

        # Force close positions at 15:15 IST
        if current_time >= datetime.time(15, 15):
            if self.position:
                self.close()
                trade_logger.info("Force closed all positions at 15:15 IST")
            return

        # Only trade during market hours (9:15 AM to 3:05 PM IST)
        if not (datetime.time(9, 15) <= current_time <= datetime.time(15, 5)):
            return

        if self.order:
            logger.debug(f"Order pending at bar {len(self)}")
            return

        # Check for invalid indicator values
        if (
            np.isnan(self.vwap[0])
            or np.isnan(self.ema[0])
            or np.isnan(self.rsi[0])
            or np.isnan(self.vol_ratio[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"VWAP={self.vwap[0]}, EMA={self.ema[0]}, "
                f"RSI={self.rsi[0]}, Volume Ratio={self.vol_ratio[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "vwap": self.vwap[0],
                "ema": self.ema[0],
                "rsi": self.rsi[0],
                "vol_ratio": self.vol_ratio[0],
            }
        )

        # Trading Logic
        rsi_rising = self.rsi[0] > self.rsi[-1] and 30 < self.rsi[0] < 70
        rsi_falling = self.rsi[0] < self.rsi[-1] and 30 < self.rsi[0] < 70
        price_below_vwap = self.data.close[0] < self.vwap[0]
        price_above_vwap = self.data.close[0] > self.vwap[0]
        price_above_ema = self.data.close[0] > self.ema[0]
        price_below_ema = self.data.close[0] < self.ema[0]
        high_volume = self.vol_ratio[0] > self.params.vol_threshold
        price_near_vwap = abs(self.data.close[0] - self.vwap[0]) / self.vwap[0] < 0.01
        rsi_reversal = (
            self.rsi[0] < self.rsi[-1]
            if self.position.size > 0
            else self.rsi[0] > self.rsi[-1]
        )
        ema_reversal = (
            self.data.close[0] < self.ema[0]
            if self.position.size > 0
            else self.data.close[0] > self.ema[0]
        )
        vol_decrease = self.vol_ratio[0] < 1.0
        signal_conflict = sum([rsi_reversal, ema_reversal, vol_decrease]) >= 2

        if not self.position:
            # Long Entry: Price below VWAP + RSI rising + Price above EMA + High volume
            if price_below_vwap and rsi_rising and price_above_ema and high_volume:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - VERV) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"VWAP: {self.vwap[0]:.2f} (Below) | "
                    f"RSI: {self.rsi[0]:.2f} (Rising) | "
                    f"EMA: {self.ema[0]:.2f} (Above) | "
                    f"Volume Ratio: {self.vol_ratio[0]:.2f} (High)"
                )
            # Short Entry: Price above VWAP + RSI falling + Price below EMA + High volume
            elif price_above_vwap and rsi_falling and price_below_ema and high_volume:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - VERV) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"VWAP: {self.vwap[0]:.2f} (Above) | "
                    f"RSI: {self.rsi[0]:.2f} (Falling) | "
                    f"EMA: {self.ema[0]:.2f} (Below) | "
                    f"Volume Ratio: {self.vol_ratio[0]:.2f} (High)"
                )
        elif self.position.size > 0:  # Long position
            # Exit: Price near VWAP or signals conflict
            if price_near_vwap or signal_conflict:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "Price near VWAP" if price_near_vwap else "Signals conflict"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - VERV) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"VWAP: {self.vwap[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"EMA: {self.ema[0]:.2f} | "
                    f"Volume Ratio: {self.vol_ratio[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            # Exit: Price near VWAP or signals conflict
            if price_near_vwap or signal_conflict:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "Price near VWAP" if price_near_vwap else "Signals conflict"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - VERV) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"VWAP: {self.vwap[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"EMA: {self.ema[0]:.2f} | "
                    f"Volume Ratio: {self.vol_ratio[0]:.2f}"
                )

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt)
            if exec_dt.tzinfo is None:
                exec_dt = exec_dt.replace(tzinfo=pytz.UTC)
            exec_dt = exec_dt.astimezone(pytz.timezone("Asia/Kolkata"))

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
                    f"BUY EXECUTED (Enter Long) | Ref: {order.ref} | "
                    f"Price: {order.executed.price:.2f} | "
                    f"Size: {order.executed.size} | "
                    f"Cost: {order.executed.value:.2f} | "
                    f"Comm: {order.executed.comm:.2f}"
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
                    f"SELL EXECUTED (Enter Short) | Ref: {order.ref} | "
                    f"Price: {order.executed.price:.2f} | "
                    f"Size: {order.executed.size} | "
                    f"Cost: {order.executed.value:.2f} | "
                    f"Comm: {order.executed.comm:.2f}"
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
                    pnl_net = pnl - total_commission
                    trade_info = {
                        "ref": order.ref,
                        "entry_time": entry_info["entry_time"],
                        "exit_time": exec_dt,
                        "entry_price": entry_info["entry_price"],
                        "exit_price": order.executed.price,
                        "size": abs(entry_info["size"]),
                        "pnl": pnl,
                        "pnl_net": pnl_net,
                        "commission": total_commission,
                        "status": "Won" if pnl > 0 else "Lost",
                        "direction": "Long",
                        "bars_held": (
                            exec_dt - entry_info["entry_time"]
                        ).total_seconds()
                        / 300,  # 5-min bars
                    }
                    self.completed_trades.append(trade_info)
                    self.trade_count += 1
                    trade_logger.info(
                        f"SELL EXECUTED (Exit Long) | Ref: {order.ref} | "
                        f"Price: {order.executed.price:.2f} | "
                        f"Size: {order.executed.size} | "
                        f"Cost: {order.executed.value:.2f} | "
                        f"Comm: {order.executed.comm:.2f} | "
                        f"PnL: {pnl:.2f}"
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
                    pnl_net = pnl - total_commission
                    trade_info = {
                        "ref": order.ref,
                        "entry_time": entry_info["entry_time"],
                        "exit_time": exec_dt,
                        "entry_price": entry_info["entry_price"],
                        "exit_price": order.executed.price,
                        "size": abs(entry_info["size"]),
                        "pnl": pnl,
                        "pnl_net": pnl_net,
                        "commission": total_commission,
                        "status": "Won" if pnl > 0 else "Lost",
                        "direction": "Short",
                        "bars_held": (
                            exec_dt - entry_info["entry_time"]
                        ).total_seconds()
                        / 300,  # 5-min bars
                    }
                    self.completed_trades.append(trade_info)
                    self.trade_count += 1
                    trade_logger.info(
                        f"BUY EXECUTED (Exit Short) | Ref: {order.ref} | "
                        f"Price: {order.executed.price:.2f} | "
                        f"Size: {order.executed.size} | "
                        f"Cost: {order.executed.value:.2f} | "
                        f"Comm: {order.executed.comm:.2f} | "
                        f"PnL: {pnl:.2f}"
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
                f"TRADE CLOSED | Ref: {trade.ref} | "
                f"Profit: {trade.pnl:.2f} | "
                f"Net Profit: {trade.pnlcomm:.2f} | "
                f"Bars Held: {trade.barlen} | "
                f"Trade Count: {self.trade_count}"
            )

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        params = {
            "vwap_period": trial.suggest_int("vwap_period", 15, 30),
            "ema_period": trial.suggest_int("ema_period", 15, 30),
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "vol_sma_period": trial.suggest_int("vol_sma_period", 10, 20),
            "vol_threshold": trial.suggest_float("vol_threshold", 1.2, 2.0, step=0.1),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            vwap_period = params.get("vwap_period", 20)
            ema_period = params.get("ema_period", 20)
            rsi_period = params.get("rsi_period", 14)
            vol_sma_period = params.get("vol_sma_period", 14)
            return max(vwap_period, ema_period, rsi_period, vol_sma_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
