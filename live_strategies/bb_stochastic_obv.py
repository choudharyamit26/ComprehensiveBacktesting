import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


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


class BBStochasticOBV(bt.Strategy):
    """
    Bollinger Bands + Stochastic + OBV Strategy
    Strategy Type: MOMENTUM + VOLUME
    =============================
    This strategy combines Bollinger Bands, Stochastic Oscillator, and OBV for trade confirmation.

    Strategy Logic:
    ==============
    Long Entry: Price at/above upper BB + Stochastic %K > %D + OBV rising
    Short Entry: Price at/below lower BB + Stochastic %K < %D + OBV falling
    Exit: Majority of indicators reverse (at least two of: BB touch ends, Stochastic reverses, OBV changes direction)

    Parameters:
    ==========
    - bb_period (int): Bollinger Bands period (default: 20)
    - bb_stddev (float): Bollinger Bands standard deviation (default: 2.0)
    - stoch_k (int): Stochastic %K period (default: 14)
    - stoch_d (int): Stochastic %D period (default: 3)
    - obv_period (int): OBV smoothing period (default: 20)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("bb_period", 20),
        ("bb_stddev", 2.0),
        ("stoch_k", 14),
        ("stoch_d", 3),
        ("obv_period", 20),
        ("verbose", False),
    )

    optimization_params = {
        "bb_period": {"type": "int", "low": 15, "high": 25, "step": 1},
        "bb_stddev": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.1},
        "stoch_k": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stoch_d": {"type": "int", "low": 2, "high": 5, "step": 1},
        "obv_period": {"type": "int", "low": 10, "high": 30, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.bb = btind.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_stddev,
        )
        self.stoch = btind.Stochastic(
            self.data, period=self.params.stoch_k, period_dfast=self.params.stoch_d
        )
        self.obv = OnBalanceVolume(self.data)
        self.obv_sma = btind.SMA(self.obv, period=self.params.obv_period)
        self.bb_upper_touch = self.data.close >= self.bb.lines.top
        self.bb_lower_touch = self.data.close <= self.bb.lines.bot

        # Debug: Log available lines and their types
        logger.debug(f"BollingerBands lines: {self.bb.lines.getlinealiases()}")
        logger.debug(f"Stochastic lines: {self.stoch.lines.getlinealiases()}")
        logger.debug(f"OBV lines: {self.obv.lines.getlinealiases()}")
        logger.debug(f"OBV SMA lines: {self.obv_sma.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            max(self.params.bb_period, self.params.stoch_k, self.params.obv_period) + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized BBStochasticOBV with params: {self.params}")
        logger.info(
            f"BBStochasticOBV initialized with bb_period={self.p.bb_period}, "
            f"bb_stddev={self.p.bb_stddev}, stoch_k={self.p.stoch_k}, "
            f"stoch_d={self.p.stoch_d}, obv_period={self.p.obv_period}"
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
            np.isnan(self.stoch.percK[0])
            or np.isnan(self.stoch.percD[0])
            or np.isnan(self.bb.lines.top[0])
            or np.isnan(self.bb.lines.bot[0])
            or np.isnan(self.obv_sma[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"Stochastic %K={self.stoch.percK[0]}, %D={self.stoch.percD[0]}, "
                f"BB Top={self.bb.lines.top[0]}, BB Bottom={self.bb.lines.bot[0]}, "
                f"OBV SMA={self.obv_sma[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "bb_top": self.bb.lines.top[0],
                "bb_mid": self.bb.lines.mid[0],
                "bb_bot": self.bb.lines.bot[0],
                "stoch_k": self.stoch.percK[0],
                "stoch_d": self.stoch.percD[0],
                "obv": self.obv[0],
                "obv_sma": self.obv_sma[0],
                "bb_upper_touch": self.bb_upper_touch[0],
                "bb_lower_touch": self.bb_lower_touch[0],
            }
        )

        # Trading Logic
        obv_rising = self.obv_sma[0] > self.obv_sma[-1]
        obv_falling = self.obv_sma[0] < self.obv_sma[-1]
        stoch_bullish = self.stoch.percK[0] > self.stoch.percD[0]
        stoch_bearish = self.stoch.percK[0] < self.stoch.percD[0]

        if not self.position:
            # Long Entry: Price at/above upper BB + Stochastic %K > %D + OBV rising
            if self.bb_upper_touch[0] and stoch_bullish and obv_rising:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - BB + Stochastic + OBV) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"BB Top: {self.bb.lines.top[0]:.2f} (Touch) | "
                    f"Stochastic %K: {self.stoch.percK[0]:.2f} > %D: {self.stoch.percD[0]:.2f} (Bullish) | "
                    f"OBV SMA: {self.obv_sma[0]:.2f} (Rising)"
                )
            # Short Entry: Price at/below lower BB + Stochastic %K < %D + OBV falling
            elif self.bb_lower_touch[0] and stoch_bearish and obv_falling:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - BB + Stochastic + OBV) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"BB Bottom: {self.bb.lines.bot[0]:.2f} (Touch) | "
                    f"Stochastic %K: {self.stoch.percK[0]:.2f} < %D: {self.stoch.percD[0]:.2f} (Bearish) | "
                    f"OBV SMA: {self.obv_sma[0]:.2f} (Falling)"
                )
        elif self.position.size > 0:  # Long position
            reverse_count = sum(
                [not self.bb_upper_touch[0], not stoch_bullish, not obv_rising]
            )
            if reverse_count >= 2:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "At least two indicators reversed (BB, Stochastic, or OBV)"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - BB + Stochastic + OBV) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"BB Top: {self.bb.lines.top[0]:.2f} | "
                    f"Stochastic %K: {self.stoch.percK[0]:.2f}, %D: {self.stoch.percD[0]:.2f} | "
                    f"OBV SMA: {self.obv_sma[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            reverse_count = sum(
                [not self.bb_lower_touch[0], not stoch_bearish, not obv_falling]
            )
            if reverse_count >= 2:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "At least two indicators reversed (BB, Stochastic, or OBV)"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - BB + Stochastic + OBV) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"BB Bottom: {self.bb.lines.bot[0]:.2f} | "
                    f"Stochastic %K: {self.stoch.percK[0]:.2f}, %D: {self.stoch.percD[0]:.2f} | "
                    f"OBV SMA: {self.obv_sma[0]:.2f}"
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
            "bb_period": trial.suggest_int("bb_period", 15, 25),
            "bb_stddev": trial.suggest_float("bb_stddev", 1.5, 2.5, step=0.1),
            "stoch_k": trial.suggest_int("stoch_k", 10, 20),
            "stoch_d": trial.suggest_int("stoch_d", 2, 5),
            "obv_period": trial.suggest_int("obv_period", 10, 30),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            bb_period = params.get("bb_period", 20)
            stoch_k = params.get("stoch_k", 14)
            obv_period = params.get("obv_period", 20)
            return max(bb_period, stoch_k, obv_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
