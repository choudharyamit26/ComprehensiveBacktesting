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
        ) / (
            hlc + 1e-10
        )  # Prevent division by zero in CLV
        mfv = clv * self.data.volume
        mfv_sum = bt.ind.SumN(mfv, period=self.p.period)
        volume_sum = bt.ind.SumN(self.data.volume, period=self.p.period)
        # Prevent division by zero by checking volume_sum
        self.lines.cmf = mfv_sum / (
            volume_sum + 1e-10
        )  # Add small constant to denominator


class OnBalanceVolume(bt.Indicator):
    lines = ("obv",)
    params = (("size", None),)

    def __init__(self):
        self.lines.obv = self.data.close * 0  # Initialize OBV to 0
        self.lastobv = 0  # Track last OBV value

    def next(self):
        if len(self.data.close) < 2:
            self.lines.obv[0] = self.lastobv
            return

        if self.data.close[0] > self.data.close[-1]:
            self.lines.obv[0] = self.lastobv + self.data.volume[0]
        elif self.data.close[0] < self.data.close[-1]:
            self.lines.obv[0] = self.lastobv - self.data.volume[0]
        else:
            self.lines.obv[0] = self.lastobv

        self.lastobv = self.lines.obv[0]


class VolumeRate(bt.Indicator):
    lines = ("volume_rate",)
    params = (("period", 14),)

    def __init__(self):
        self.addminperiod(self.params.period)
        # Prevent division by zero in volume rate
        self.lines.volume_rate = self.data.volume / (
            self.data.volume(-self.params.period) + 1e-10
        )


class OBV_CMF_VolumeRate(bt.Strategy):
    """
    OBV + CMF + Volume Rate Strategy
    Strategy Type: VOLUME + MONEY FLOW
    ===============================
    This strategy combines OBV, Chaikin Money Flow, and Volume Rate for trade confirmation.

    Strategy Logic:
    ==============
    Long Entry: OBV rising + CMF positive + Volume Rate increasing
    Short Entry: OBV falling + CMF negative + Volume Rate decreasing
    Exit: At least two of the volume indicators weaken (OBV, CMF, or Volume Rate)

    Parameters:
    ==========
    - obv_period (int): OBV smoothing period (default: 20)
    - cmf_period (int): CMF period (default: 20)
    - vroc_period (int): Volume Rate of Change period (default: 14)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("obv_period", 20),
        ("cmf_period", 20),
        ("vroc_period", 14),
        ("verbose", False),
    )

    optimization_params = {
        "obv_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "cmf_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "vroc_period": {"type": "int", "low": 10, "high": 20, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.obv = OnBalanceVolume(self.data)
        self.obv_sma = btind.SMA(self.obv, period=self.params.obv_period)
        self.cmf = ChaikinMoneyFlow(self.data, period=self.params.cmf_period)
        self.vroc = VolumeRate(self.data, period=self.params.vroc_period)

        # Debug: Log available lines and their types
        logger.debug(f"OBV lines: {self.obv.lines.getlinealiases()}")
        logger.debug(f"OBV SMA lines: {self.obv_sma.lines.getlinealiases()}")
        logger.debug(f"CMF lines: {self.cmf.lines.getlinealiases()}")
        logger.debug(f"VROC lines: {self.vroc.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(self.params.obv_period, self.params.cmf_period, self.params.vroc_period)
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized OBV_CMF_VolumeRate with params: {self.params}")
        logger.info(
            f"OBV_CMF_VolumeRate initialized with obv_period={self.p.obv_period}, "
            f"cmf_period={self.p.cmf_period}, vroc_period={self.p.vroc_period}"
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
            np.isnan(self.obv_sma[0])
            or np.isnan(self.cmf[0])
            or np.isnan(self.vroc[0])
            or np.isinf(self.cmf[0])
            or np.isinf(self.vroc[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"OBV SMA={self.obv_sma[0]}, CMF={self.cmf[0]}, VROC={self.vroc[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "obv": self.obv[0],
                "obv_sma": self.obv_sma[0],
                "cmf": self.cmf[0],
                "vroc": self.vroc[0],
            }
        )

        # Trading Logic
        obv_rising = self.obv_sma[0] > self.obv_sma[-1]
        obv_falling = self.obv_sma[0] < self.obv_sma[-1]
        cmf_positive = self.cmf[0] > 0
        cmf_negative = self.cmf[0] < 0
        vroc_increasing = self.vroc[0] > 1
        vroc_decreasing = self.vroc[0] < 1

        if not self.position:
            # Long Entry: OBV rising + CMF positive + Volume Rate increasing
            if obv_rising and cmf_positive and vroc_increasing:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - OBV + CMF + VROC) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"OBV SMA: {self.obv_sma[0]:.2f} (Rising) | "
                    f"CMF: {self.cmf[0]:.2f} (Positive) | "
                    f"VROC: {self.vroc[0]:.2f} (Increasing)"
                )
            # Short Entry: OBV falling + CMF negative + Volume Rate decreasing
            elif obv_falling and cmf_negative and vroc_decreasing:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - OBV + CMF + VROC) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"OBV SMA: {self.obv_sma[0]:.2f} (Falling) | "
                    f"CMF: {self.cmf[0]:.2f} (Negative) | "
                    f"VROC: {self.vroc[0]:.2f} (Decreasing)"
                )
        elif self.position.size > 0:  # Long position
            reverse_count = sum([not obv_rising, not cmf_positive, not vroc_increasing])
            if reverse_count >= 2:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = "At least two volume indicators weakened"
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - OBV + CMF + VROC) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"OBV SMA: {self.obv_sma[0]:.2f} | "
                    f"CMF: {self.cmf[0]:.2f} | "
                    f"VROC: {self.vroc[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            reverse_count = sum(
                [not obv_falling, not cmf_negative, not vroc_decreasing]
            )
            if reverse_count >= 2:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = "At least two volume indicators weakened"
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - OBV + CMF + VROC) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"OBV SMA: {self.obv_sma[0]:.2f} | "
                    f"CMF: {self.cmf[0]:.2f} | "
                    f"VROC: {self.vroc[0]:.2f}"
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
                        "bars_held": (exec_dt - entry_info["entry_time"]).days,
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
                        "bars_held": (exec_dt - entry_info["entry_time"]).days,
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
            "obv_period": trial.suggest_int("obv_period", 10, 30),
            "cmf_period": trial.suggest_int("cmf_period", 10, 30),
            "vroc_period": trial.suggest_int("vroc_period", 10, 20),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            obv_period = params.get("obv_period", 20)
            cmf_period = params.get("cmf_period", 20)
            vroc_period = params.get("vroc_period", 14)
            return max(obv_period, cmf_period, vroc_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
