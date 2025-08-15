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
    lines = ("supertrend", "trend")
    params = (
        ("period", 10),
        ("multiplier", 3.0),
    )

    def __init__(self):
        self.addminperiod(self.params.period)
        self.atr = btind.ATR(self.data, period=self.params.period)
        self.basic_ub = (
            self.data.high + self.data.low
        ) / 2 + self.params.multiplier * self.atr
        self.basic_lb = (
            self.data.high + self.data.low
        ) / 2 - self.params.multiplier * self.atr
        self.final_ub = self.basic_ub
        self.final_lb = self.basic_lb
        self.lines.supertrend = (
            self.data.close * 0
        )  # Initialize supertrend as zero-filled series
        self.lines.trend = self.data.close * 0  # Initialize trend as zero-filled series

    def next(self):
        if len(self.data) < self.params.period:
            return

        if len(self) == 1:
            self.final_ub[0] = self.basic_ub[0]
            self.final_lb[0] = self.basic_lb[0]
            self.lines.supertrend[0] = self.data.close[0]
            self.lines.trend[0] = 0  # Initial trend set to neutral
        else:
            self.final_ub[0] = (
                self.basic_ub[0]
                if self.basic_ub[0] < self.final_ub[-1]
                or self.data.close[-1] > self.final_ub[-1]
                else self.final_ub[-1]
            )
            self.final_lb[0] = (
                self.basic_lb[0]
                if self.basic_lb[0] > self.final_lb[-1]
                or self.data.close[-1] < self.final_lb[-1]
                else self.final_lb[-1]
            )

            if self.lines.trend[-1] == 1:
                self.lines.supertrend[0] = (
                    self.final_lb[0]
                    if self.data.close[0] > self.final_lb[0]
                    else self.final_ub[0]
                )
                self.lines.trend[0] = 1 if self.data.close[0] > self.final_lb[0] else -1
            else:
                self.lines.supertrend[0] = (
                    self.final_ub[0]
                    if self.data.close[0] < self.final_ub[0]
                    else self.final_lb[0]
                )
                self.lines.trend[0] = -1 if self.data.close[0] < self.final_ub[0] else 1


class ATR_Supertrend_MACD(bt.Strategy):
    """
    ATR + Supertrend + MACD Strategy
    Strategy Type: VOLATILITY + TREND + MOMENTUM
    =======================================
    This strategy combines ATR, Supertrend, and MACD for volatility and trend-based trading.

    Strategy Logic:
    ==============
    Long Entry: ATR increasing + Supertrend bullish + MACD bullish crossover
    Short Entry: ATR increasing + Supertrend bearish + MACD bearish crossover
    Exit: Volatility decreases or Supertrend reverses or MACD momentum fails

    Parameters:
    ==========
    - atr_period (int): ATR period (default: 14)
    - supertrend_period (int): Supertrend period (default: 10)
    - supertrend_multiplier (float): Supertrend multiplier (default: 3.0)
    - macd_fast (int): MACD fast EMA period (default: 12)
    - macd_slow (int): MACD slow EMA period (default: 26)
    - macd_signal (int): MACD signal line period (default: 9)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("atr_period", 14),
        ("supertrend_period", 10),
        ("supertrend_multiplier", 3.0),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("verbose", False),
    )

    optimization_params = {
        "atr_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "supertrend_period": {"type": "int", "low": 7, "high": 15, "step": 1},
        "supertrend_multiplier": {
            "type": "float",
            "low": 2.0,
            "high": 4.0,
            "step": 0.5,
        },
        "macd_fast": {"type": "int", "low": 8, "high": 16, "step": 1},
        "macd_slow": {"type": "int", "low": 20, "high": 30, "step": 1},
        "macd_signal": {"type": "int", "low": 5, "high": 12, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.atr = btind.ATR(self.data, period=self.params.atr_period)
        self.supertrend = Supertrend(
            self.data,
            period=self.params.supertrend_period,
            multiplier=self.params.supertrend_multiplier,
        )
        self.macd = btind.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal,
        )

        # Debug: Log available lines and their types
        logger.debug(f"ATR lines: {self.atr.lines.getlinealiases()}")
        logger.debug(f"Supertrend lines: {self.supertrend.lines.getlinealiases()}")
        logger.debug(f"MACD lines: {self.macd.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.atr_period,
                self.params.supertrend_period,
                self.params.macd_slow,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized ATR_Supertrend_MACD with params: {self.params}")
        logger.info(
            f"ATR_Supertrend_MACD initialized with atr_period={self.p.atr_period}, "
            f"supertrend_period={self.p.supertrend_period}, supertrend_multiplier={self.p.supertrend_multiplier}, "
            f"macd_fast={self.p.macd_fast}, macd_slow={self.p.macd_slow}, macd_signal={self.p.macd_signal}"
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
            np.isnan(self.atr[0])
            or np.isnan(self.supertrend.supertrend[0])
            or np.isnan(self.macd.macd[0])
            or np.isnan(self.macd.signal[0])
            or np.isinf(self.atr[0])
            or np.isinf(self.supertrend.supertrend[0])
            or np.isinf(self.macd.macd[0])
            or np.isinf(self.macd.signal[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"ATR={self.atr[0]}, Supertrend={self.supertrend.supertrend[0]}, "
                f"MACD={self.macd.macd[0]}, Signal={self.macd.signal[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "atr": self.atr[0],
                "supertrend": self.supertrend.supertrend[0],
                "trend": self.supertrend.trend[0],
                "macd": self.macd.macd[0],
                "macd_signal": self.macd.signal[0],
            }
        )

        # Trading Logic
        atr_increasing = self.atr[0] > self.atr[-1]
        supertrend_bullish = self.supertrend.trend[0] == 1
        supertrend_bearish = self.supertrend.trend[0] == -1
        macd_bullish = (
            self.macd.macd[0] > self.macd.signal[0]
            and self.macd.macd[-1] <= self.macd.signal[-1]
        )
        macd_bearish = (
            self.macd.macd[0] < self.macd.signal[0]
            and self.macd.macd[-1] >= self.macd.signal[-1]
        )
        macd_momentum_failure = (
            (self.macd.macd[0] < self.macd.signal[0])
            if self.position.size > 0
            else (
                (self.macd.macd[0] > self.macd.signal[0])
                if self.position.size < 0
                else False
            )
        )

        if not self.position:
            # Long Entry: ATR increasing + Supertrend bullish + MACD bullish crossover
            if atr_increasing and supertrend_bullish and macd_bullish:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - ATR + Supertrend + MACD) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"ATR: {self.atr[0]:.2f} (Increasing) | "
                    f"Supertrend: {self.supertrend.supertrend[0]:.2f} (Bullish) | "
                    f"MACD: {self.macd.macd[0]:.2f} > Signal: {self.macd.signal[0]:.2f}"
                )
            # Short Entry: ATR increasing + Supertrend bearish + MACD bearish crossover
            elif atr_increasing and supertrend_bearish and macd_bearish:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - ATR + Supertrend + MACD) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"ATR: {self.atr[0]:.2f} (Increasing) | "
                    f"Supertrend: {self.supertrend.supertrend[0]:.2f} (Bearish) | "
                    f"MACD: {self.macd.macd[0]:.2f} < Signal: {self.macd.signal[0]:.2f}"
                )
        elif self.position.size > 0:  # Long position
            # Exit: Volatility decreases or Supertrend reverses or MACD momentum fails
            if not atr_increasing or not supertrend_bullish or macd_momentum_failure:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "Volatility decreasing"
                    if not atr_increasing
                    else (
                        "Supertrend reversal"
                        if not supertrend_bullish
                        else "MACD momentum failure"
                    )
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - ATR + Supertrend + MACD) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"ATR: {self.atr[0]:.2f} | "
                    f"Supertrend: {self.supertrend.supertrend[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.2f}, Signal: {self.macd.signal[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            # Exit: Volatility decreases or Supertrend reverses or MACD momentum fails
            if not atr_increasing or not supertrend_bearish or macd_momentum_failure:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "Volatility decreasing"
                    if not atr_increasing
                    else (
                        "Supertrend reversal"
                        if not supertrend_bearish
                        else "MACD momentum failure"
                    )
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - ATR + Supertrend + MACD) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"ATR: {self.atr[0]:.2f} | "
                    f"Supertrend: {self.supertrend.supertrend[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.2f}, Signal: {self.macd.signal[0]:.2f}"
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
                        / 60,
                    }
                    self.completed_trades.append(trade_info)
                    self.trade_count += 1
                    trade_logger.info(
                        f"BUY EXECUTED (Exit Short) | Ref: {order.ref} | "
                        f"Price: {self.data.close[0]:.2f} | "
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
            "atr_period": trial.suggest_int("atr_period", 10, 20),
            "supertrend_period": trial.suggest_int("supertrend_period", 7, 15),
            "supertrend_multiplier": trial.suggest_float(
                "supertrend_multiplier", 2.0, 4.0, step=0.5
            ),
            "macd_fast": trial.suggest_int("macd_fast", 8, 16),
            "macd_slow": trial.suggest_int("macd_slow", 20, 30),
            "macd_signal": trial.suggest_int("macd_signal", 5, 12),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            atr_period = params.get("atr_period", 14)
            supertrend_period = params.get("supertrend_period", 10)
            macd_slow = params.get("macd_slow", 26)
            return max(atr_period, supertrend_period, macd_slow) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
