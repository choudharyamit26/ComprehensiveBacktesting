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
        ("period", 10),
        ("multiplier", 3.0),
    )

    def __init__(self):
        self.atr = btind.ATR(self.data, period=self.params.period)
        self.basic_ub = (
            self.data.high + self.data.low
        ) / 2 + self.params.multiplier * self.atr
        self.basic_lb = (
            self.data.high + self.data.low
        ) / 2 - self.params.multiplier * self.atr
        self.lines.supertrend = self.data.close  # Initialize with close
        self.trend = 0
        self.addminperiod(self.params.period)

    def next(self):
        if len(self) == 1:
            self.lines.supertrend[0] = self.data.close[0]
            return

        ub = self.basic_ub[0]
        lb = self.basic_lb[0]
        prev_st = self.lines.supertrend[-1]
        prev_trend = self.trend

        if prev_trend == 1:
            if self.data.close[0] <= lb:
                self.trend = -1
                self.lines.supertrend[0] = ub
            else:
                self.trend = 1
                self.lines.supertrend[0] = min(lb, prev_st)
        else:
            if self.data.close[0] >= ub:
                self.trend = 1
                self.lines.supertrend[0] = lb
            else:
                self.trend = -1
                self.lines.supertrend[0] = max(ub, prev_st)


class SPMA(bt.Strategy):
    """
    Supertrend + Parabolic SAR + MACD + ADX (SPMA) Strategy
    Strategy Type: TREND + MOMENTUM
    ==========================================
    This strategy combines Supertrend, Parabolic SAR, MACD, and ADX for intraday trading on a 5-minute timeframe.

    Strategy Logic:
    ==============
    Long Entry: Supertrend bullish + PSAR below price + MACD bullish crossover + ADX rising
    Short Entry: Supertrend bearish + PSAR above price + MACD bearish crossover + ADX rising
    Exit: First major indicator reverses

    Parameters:
    ==========
    - supertrend_period (int): Supertrend period (default: 10)
    - supertrend_mult (float): Supertrend multiplier (default: 3.0)
    - psar_af (float): Parabolic SAR acceleration factor (default: 0.02)
    - psar_max_af (float): Parabolic SAR max acceleration factor (default: 0.2)
    - macd_fast (int): MACD fast EMA period (default: 12)
    - macd_slow (int): MACD slow EMA period (default: 26)
    - macd_signal (int): MACD signal period (default: 9)
    - adx_period (int): ADX period (default: 14)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("supertrend_period", 10),
        ("supertrend_mult", 3.0),
        ("psar_af", 0.02),
        ("psar_max_af", 0.2),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("adx_period", 14),
        ("verbose", False),
    )

    optimization_params = {
        "supertrend_period": {"type": "int", "low": 7, "high": 14, "step": 1},
        "supertrend_mult": {"type": "float", "low": 2.0, "high": 4.0, "step": 0.5},
        "psar_af": {"type": "float", "low": 0.01, "high": 0.05, "step": 0.01},
        "psar_max_af": {"type": "float", "low": 0.1, "high": 0.3, "step": 0.05},
        "macd_fast": {"type": "int", "low": 8, "high": 16, "step": 1},
        "macd_slow": {"type": "int", "low": 20, "high": 30, "step": 1},
        "macd_signal": {"type": "int", "low": 7, "high": 12, "step": 1},
        "adx_period": {"type": "int", "low": 10, "high": 20, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.supertrend = Supertrend(
            self.data,
            period=self.params.supertrend_period,
            multiplier=self.params.supertrend_mult,
        )
        self.psar = btind.ParabolicSAR(
            self.data, af=self.params.psar_af, afmax=self.params.psar_max_af
        )
        self.macd = btind.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal,
        )
        self.adx = btind.ADX(self.data, period=self.params.adx_period)

        # Debug: Log available lines and their types
        logger.debug(f"Supertrend lines: {self.supertrend.lines.getlinealiases()}")
        logger.debug(f"Parabolic SAR lines: {self.psar.lines.getlinealiases()}")
        logger.debug(f"MACD lines: {self.macd.lines.getlinealiases()}")
        logger.debug(f"ADX lines: {self.adx.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.supertrend_period,
                self.params.macd_slow,
                self.params.adx_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized SPMA with params: {self.params}")
        logger.info(
            f"SPMA initialized with supertrend_period={self.p.supertrend_period}, "
            f"supertrend_mult={self.p.supertrend_mult}, psar_af={self.p.psar_af}, "
            f"macd_fast={self.p.macd_fast}, adx_period={self.p.adx_period}"
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
            or np.isnan(self.psar[0])
            or np.isnan(self.macd.macd[0])
            or np.isnan(self.macd.signal[0])
            or np.isnan(self.adx[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"Supertrend={self.supertrend[0]}, PSAR={self.psar[0]}, "
                f"MACD={self.macd.macd[0]}, Signal={self.macd.signal[0]}, ADX={self.adx[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "supertrend": self.supertrend[0],
                "psar": self.psar[0],
                "macd": self.macd.macd[0],
                "signal": self.macd.signal[0],
                "adx": self.adx[0],
            }
        )

        # Trading Logic
        supertrend_bullish = self.data.close[0] > self.supertrend[0]
        supertrend_bearish = self.data.close[0] < self.supertrend[0]
        psar_bullish = self.psar[0] < self.data.close[0]
        psar_bearish = self.psar[0] > self.data.close[0]
        macd_bullish = (
            self.macd.macd[0] > self.macd.signal[0]
            and self.macd.macd[-1] <= self.macd.signal[-1]
        )
        macd_bearish = (
            self.macd.macd[0] < self.macd.signal[0]
            and self.macd.macd[-1] >= self.macd.signal[-1]
        )
        adx_rising = self.adx[0] > self.adx[-1] and self.adx[0] > 25
        supertrend_reversal = (
            self.data.close[0] < self.supertrend[0]
            if self.position.size > 0
            else self.data.close[0] > self.supertrend[0]
        )
        psar_reversal = (
            self.psar[0] > self.data.close[0]
            if self.position.size > 0
            else self.psar[0] < self.data.close[0]
        )
        macd_reversal = (
            self.macd.macd[0] < self.macd.signal[0]
            if self.position.size > 0
            else self.macd.macd[0] > self.macd.signal[0]
        )
        adx_reversal = self.adx[0] < self.adx[-1]

        if not self.position:
            # Long Entry: Supertrend bullish + PSAR below price + MACD bullish + ADX rising
            if supertrend_bullish and psar_bullish and macd_bullish and adx_rising:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - SPMA) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Supertrend: {self.supertrend[0]:.2f} (Bullish) | "
                    f"PSAR: {self.psar[0]:.2f} (Below) | "
                    f"MACD: {self.macd.macd[0]:.2f} (Bullish) | "
                    f"ADX: {self.adx[0]:.2f} (Rising)"
                )
            # Short Entry: Supertrend bearish + PSAR above price + MACD bearish + ADX rising
            elif supertrend_bearish and psar_bearish and macd_bearish and adx_rising:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - SPMA) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Supertrend: {self.supertrend[0]:.2f} (Bearish) | "
                    f"PSAR: {self.psar[0]:.2f} (Above) | "
                    f"MACD: {self.macd.macd[0]:.2f} (Bearish) | "
                    f"ADX: {self.adx[0]:.2f} (Rising)"
                )
        elif self.position.size > 0:  # Long position
            # Exit: Any major indicator reverses
            if supertrend_reversal or psar_reversal or macd_reversal or adx_reversal:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "Supertrend reversal"
                    if supertrend_reversal
                    else (
                        "PSAR reversal"
                        if psar_reversal
                        else "MACD reversal" if macd_reversal else "ADX reversal"
                    )
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - SPMA) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Supertrend: {self.supertrend[0]:.2f} | "
                    f"PSAR: {self.psar[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.2f} | "
                    f"ADX: {self.adx[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            # Exit: Any major indicator reverses
            if supertrend_reversal or psar_reversal or macd_reversal or adx_reversal:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "Supertrend reversal"
                    if supertrend_reversal
                    else (
                        "PSAR reversal"
                        if psar_reversal
                        else "MACD reversal" if macd_reversal else "ADX reversal"
                    )
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - SPMA) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Supertrend: {self.supertrend[0]:.2f} | "
                    f"PSAR: {self.psar[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.2f} | "
                    f"ADX: {self.adx[0]:.2f}"
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
            "supertrend_period": trial.suggest_int("supertrend_period", 7, 14),
            "supertrend_mult": trial.suggest_float(
                "supertrend_mult", 2.0, 4.0, step=0.5
            ),
            "psar_af": trial.suggest_float("psar_af", 0.01, 0.05, step=0.01),
            "psar_max_af": trial.suggest_float("psar_max_af", 0.1, 0.3, step=0.05),
            "macd_fast": trial.suggest_int("macd_fast", 8, 16),
            "macd_slow": trial.suggest_int("macd_slow", 20, 30),
            "macd_signal": trial.suggest_int("macd_signal", 7, 12),
            "adx_period": trial.suggest_int("adx_period", 10, 20),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            supertrend_period = params.get("supertrend_period", 10)
            macd_slow = params.get("macd_slow", 26)
            adx_period = params.get("adx_period", 14)
            return max(supertrend_period, macd_slow, adx_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
