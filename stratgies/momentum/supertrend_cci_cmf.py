import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class Supertrend(bt.Indicator):
    lines = ("supertrend",)
    params = (
        ("period", 10),  # ATR period
        ("multiplier", 3.0),  # Multiplier for bands
    )

    def __init__(self):
        self.atr = btind.ATR(self.data, period=self.params.period)
        self.lines.supertrend = self.data.close * 0  # Initialize with zeros
        self.mid = (self.data.high + self.data.low) / 2
        self.upperband = self.mid + self.params.multiplier * self.atr
        self.lowerband = self.mid - self.params.multiplier * self.atr
        self.last_supertrend = 0
        self.last_close = 0

    def next(self):
        if len(self) < 2:
            self.lines.supertrend[0] = self.last_supertrend
            return

        # Calculate current bands
        curr_upper = self.mid[0] + self.params.multiplier * self.atr[0]
        curr_lower = self.mid[0] - self.params.multiplier * self.atr[0]

        # Adjust bands based on previous Supertrend
        if self.last_supertrend > self.last_close:
            self.upperband[0] = min(curr_upper, self.upperband[-1])
            self.lowerband[0] = curr_lower
        else:
            self.upperband[0] = curr_upper
            self.lowerband[0] = max(curr_lower, self.lowerband[-1])

        # Determine Supertrend value
        if self.last_supertrend > self.last_close:
            if self.data.close[0] > self.upperband[0]:
                self.lines.supertrend[0] = self.lowerband[0]
            else:
                self.lines.supertrend[0] = self.upperband[0]
        else:
            if self.data.close[0] < self.lowerband[0]:
                self.lines.supertrend[0] = self.upperband[0]
            else:
                self.lines.supertrend[0] = self.lowerband[0]

        self.last_supertrend = self.lines.supertrend[0]
        self.last_close = self.data.close[0]

        # Handle NaN values
        if np.isnan(self.lines.supertrend[0]):
            self.lines.supertrend[0] = self.last_supertrend


class ChaikinMoneyFlow(bt.Indicator):
    lines = ("cmf",)
    params = (("period", 20),)  # CMF period

    def __init__(self):
        self.mfm = (
            (self.data.close - self.data.low) - (self.data.high - self.data.close)
        ) / (self.data.high - self.data.low)
        self.mfv = self.mfm * self.data.volume
        self.sum_mfv = btind.SumN(self.mfv, period=self.params.period)
        self.sum_volume = btind.SumN(self.data.volume, period=self.params.period)
        self.lines.cmf = self.sum_mfv / self.sum_volume

    def next(self):
        # Handle edge case where high == low or volume is zero
        if (
            self.data.high[0] == self.data.low[0]
            or self.data.volume[0] == 0
            or np.isnan(self.mfm[0])
        ):
            self.mfm[0] = 0
        if np.isnan(self.lines.cmf[0]) or self.sum_volume[0] == 0:
            self.lines.cmf[0] = 0


class SupertrendCCICMF(bt.Strategy):
    """
    Supertrend + CCI + CMF Strategy
    Strategy Type: TREND + MOMENTUM
    =============================
    This strategy uses Supertrend for trend direction, CCI for momentum, and CMF for money flow confirmation.

    Strategy Logic:
    ==============
    Long Entry: Supertrend bullish + CCI > 100 + CMF > 0
    Short Entry: Supertrend bearish + CCI < -100 + CMF < 0
    Exit: Majority of indicators reverse (at least two of: Supertrend flips, CCI crosses 0, CMF crosses 0)

    Parameters:
    ==========
    - supertrend_period (int): Supertrend period (default: 10)
    - supertrend_multiplier (float): Supertrend multiplier (default: 3.0)
    - cci_period (int): CCI period (default: 20)
    - cmf_period (int): CMF period (default: 20)
    - cci_threshold (int): CCI threshold (default: 100)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("supertrend_period", 10),
        ("supertrend_multiplier", 3.0),
        ("cci_period", 20),
        ("cmf_period", 20),
        ("cci_threshold", 100),
        ("verbose", False),
    )

    optimization_params = {
        "supertrend_period": {"type": "int", "low": 7, "high": 15, "step": 1},
        "supertrend_multiplier": {
            "type": "float",
            "low": 2.0,
            "high": 4.0,
            "step": 0.5,
        },
        "cci_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "cmf_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "cci_threshold": {"type": "int", "low": 80, "high": 120, "step": 10},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.supertrend = Supertrend(
            self.data,
            period=self.params.supertrend_period,
            multiplier=self.params.supertrend_multiplier,
        )
        self.cci = btind.CCI(self.data, period=self.params.cci_period)
        self.cmf = ChaikinMoneyFlow(self.data, period=self.params.cmf_period)

        # Debug: Log available lines and their types
        logger.debug(f"Supertrend lines: {self.supertrend.lines.getlinealiases()}")
        logger.debug(f"CCI lines: {self.cci.lines.getlinealiases()}")
        logger.debug(f"CMF lines: {self.cmf.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.supertrend_period,
                self.params.cci_period,
                self.params.cmf_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized SupertrendCCICMF with params: {self.params}")
        logger.info(
            f"SupertrendCCICMF initialized with supertrend_period={self.p.supertrend_period}, "
            f"supertrend_multiplier={self.p.supertrend_multiplier}, cci_period={self.p.cci_period}, "
            f"cmf_period={self.p.cmf_period}, cci_threshold={self.p.cci_threshold}"
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
            np.isnan(self.supertrend[0])
            or np.isnan(self.cci[0])
            or np.isnan(self.cmf[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"Supertrend={self.supertrend[0]}, CCI={self.cci[0]}, CMF={self.cmf[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "supertrend": self.supertrend[0],
                "cci": self.cci[0],
                "cmf": self.cmf[0],
            }
        )

        # Trading Logic
        supertrend_bullish = self.data.close[0] > self.supertrend[0]
        supertrend_bearish = self.data.close[0] < self.supertrend[0]

        if not self.position:
            # Long Entry: Supertrend bullish + CCI > threshold + CMF > 0
            if (
                supertrend_bullish
                and self.cci[0] > self.params.cci_threshold
                and self.cmf[0] > 0
            ):
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Supertrend + CCI + CMF) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Supertrend: {self.supertrend[0]:.2f} (Bullish) | "
                    f"CCI: {self.cci[0]:.2f} > {self.params.cci_threshold} | "
                    f"CMF: {self.cmf[0]:.4f} > 0"
                )
            # Short Entry: Supertrend bearish + CCI < -threshold + CMF < 0
            elif (
                supertrend_bearish
                and self.cci[0] < -self.params.cci_threshold
                and self.cmf[0] < 0
            ):
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Supertrend + CCI + CMF) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Supertrend: {self.supertrend[0]:.2f} (Bearish) | "
                    f"CCI: {self.cci[0]:.2f} < {-self.params.cci_threshold} | "
                    f"CMF: {self.cmf[0]:.4f} < 0"
                )
        elif self.position.size > 0:  # Long position
            reverse_count = sum(
                [not supertrend_bullish, self.cci[0] < 0, self.cmf[0] < 0]
            )
            if reverse_count >= 2:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "At least two indicators reversed (Supertrend, CCI, or CMF)"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - Supertrend + CCI + CMF) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Supertrend: {self.supertrend[0]:.2f} | "
                    f"CCI: {self.cci[0]:.2f} | "
                    f"CMF: {self.cmf[0]:.4f} | "
                    f"Reverse Count: {reverse_count}"
                )
        elif self.position.size < 0:  # Short position
            reverse_count = sum(
                [not supertrend_bearish, self.cci[0] > 0, self.cmf[0] > 0]
            )
            if reverse_count >= 2:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "At least two indicators reversed (Supertrend, CCI, or CMF)"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - Supertrend + CCI + CMF) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Supertrend: {self.supertrend[0]:.2f} | "
                    f"CCI: {self.cci[0]:.2f} | "
                    f"CMF: {self.cmf[0]:.4f} | "
                    f"Reverse Count: {reverse_count}"
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
                        / 60,
                    }
                    self.completed_trades.append(trade_info)
                    self.trade_count += 1
                    trade_logger.info(
                        f"SELL EXECUTED (Exit Long) | Ref: {order.ref} | "
                        f"Price: {order.executed.price:.2f} | "
                        f"Size: {order.executed.size} | "
                        f"Cost: {order.executed.value:.2f} | "
                        f"Comm: {order.executed.comm:.2f} | "
                        f"PnL: {pnl:.2f} | Net PnL: {pnl_net:.2f}"
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
                        / 60,
                    }
                    self.completed_trades.append(trade_info)
                    self.trade_count += 1
                    trade_logger.info(
                        f"BUY EXECUTED (Exit Short) | Ref: {order.ref} | "
                        f"Price: {order.executed.price:.2f} | "
                        f"Size: {order.executed.size} | "
                        f"Cost: {order.executed.value:.2f} | "
                        f"Comm: {order.executed.comm:.2f} | "
                        f"PnL: {pnl:.2f} | Net PnL: {pnl_net:.2f}"
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
            "supertrend_period": trial.suggest_int("supertrend_period", 7, 15),
            "supertrend_multiplier": trial.suggest_float(
                "supertrend_multiplier", 2.0, 4.0, step=0.5
            ),
            "cci_period": trial.suggest_int("cci_period", 10, 30),
            "cmf_period": trial.suggest_int("cmf_period", 10, 30),
            "cci_threshold": trial.suggest_int("cci_threshold", 80, 120, step=10),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            supertrend_period = params.get("supertrend_period", 10)
            cci_period = params.get("cci_period", 20)
            cmf_period = params.get("cmf_period", 20)
            return max(supertrend_period, cci_period, cmf_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
