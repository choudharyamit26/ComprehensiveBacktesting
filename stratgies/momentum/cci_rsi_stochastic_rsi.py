import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class StochasticRSI(bt.Indicator):
    lines = ("percK", "percD")
    params = (
        ("period", 14),  # RSI period
        ("period_k", 14),  # Stochastic %K period
        ("period_d", 3),  # Stochastic %D period
    )

    def __init__(self):
        self.rsi = btind.RSI(self.data, period=self.params.period)
        self.rsi_high = btind.Highest(self.rsi, period=self.params.period_k)
        self.rsi_low = btind.Lowest(self.rsi, period=self.params.period_k)
        self.lines.percK = (
            (self.rsi - self.rsi_low) / (self.rsi_high - self.rsi_low)
        ) * 100
        self.lines.percD = btind.SMA(self.lines.percK, period=self.params.period_d)

    def next(self):
        # Handle edge case where rsi_high == rsi_low to avoid division by zero
        if self.rsi_high[0] == self.rsi_low[0]:
            self.lines.percK[0] = 50.0  # Neutral value when no range
        if np.isnan(self.lines.percK[0]) or np.isnan(self.rsi[0]):
            self.lines.percK[0] = 50.0
        if np.isnan(self.lines.percD[0]):
            self.lines.percD[0] = 50.0


class CCIRSIStochasticRSI(bt.Strategy):
    """
    CCI + RSI + Stochastic RSI Strategy
    Strategy Type: MULTI-OSCILLATOR
    =============================
    This strategy uses CCI, RSI, and Stochastic RSI for trade confirmation.

    Strategy Logic:
    ==============
    Long Entry: CCI > 100 + RSI > 50 + Stochastic RSI %K > %D
    Short Entry: CCI < -100 + RSI < 50 + Stochastic RSI %K < %D
    Exit: Two out of three oscillators reverse

    Parameters:
    ==========
    - cci_period (int): CCI period (default: 20)
    - rsi_period (int): RSI period (default: 14)
    - stochrsi_period (int): Stochastic RSI period (default: 14)
    - stochrsi_k (int): Stochastic RSI %K period (default: 14)
    - stochrsi_d (int): Stochastic RSI %D period (default: 3)
    - cci_threshold (int): CCI threshold (default: 100)
    - rsi_threshold (int): RSI threshold (default: 50)
    - verbose (bool): Enable detailed logging (default: False)

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses warmup period to ensure indicator stability
    - Prevents order overlap with pending order checks

    Best Market Conditions:
    ======================
    - Oscillating or range-bound markets with clear momentum shifts
    - Avoid strong trending markets where oscillators may lag
    - Works well in intraday timeframes with sufficient volatility
    """

    params = (
        ("cci_period", 20),
        ("rsi_period", 14),
        ("stochrsi_period", 14),
        ("stochrsi_k", 14),
        ("stochrsi_d", 3),
        ("cci_threshold", 100),
        ("rsi_threshold", 50),
        ("verbose", False),
    )

    optimization_params = {
        "cci_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stochrsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stochrsi_k": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stochrsi_d": {"type": "int", "low": 2, "high": 5, "step": 1},
        "cci_threshold": {"type": "int", "low": 80, "high": 120, "step": 10},
        "rsi_threshold": {"type": "int", "low": 45, "high": 55, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.cci = btind.CCI(self.data, period=self.params.cci_period)
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.stochrsi = StochasticRSI(
            self.data,
            period=self.params.stochrsi_period,
            period_k=self.params.stochrsi_k,
            period_d=self.params.stochrsi_d,
        )

        # Debug: Log available lines and their types
        logger.debug(f"CCI lines: {self.cci.lines.getlinealiases()}")
        logger.debug(f"RSI lines: {self.rsi.lines.getlinealiases()}")
        logger.debug(f"StochasticRSI lines: {self.stochrsi.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.cci_period,
                self.params.rsi_period,
                self.params.stochrsi_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized CCIRSIStochasticRSI with params: {self.params}")
        logger.info(
            f"CCIRSIStochasticRSI initialized with cci_period={self.p.cci_period}, "
            f"rsi_period={self.p.rsi_period}, stochrsi_period={self.p.stochrsi_period}, "
            f"stochrsi_k={self.p.stochrsi_k}, stochrsi_d={self.p.stochrsi_d}, "
            f"cci_threshold={self.p.cci_threshold}, rsi_threshold={self.p.rsi_threshold}"
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
            np.isnan(self.cci[0])
            or np.isnan(self.rsi[0])
            or np.isnan(self.stochrsi.percK[0])
            or np.isnan(self.stochrsi.percD[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"CCI={self.cci[0]}, RSI={self.rsi[0]}, "
                f"StochRSI %K={self.stochrsi.percK[0]}, %D={self.stochrsi.percD[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "cci": self.cci[0],
                "rsi": self.rsi[0],
                "stochrsi_k": self.stochrsi.percK[0],
                "stochrsi_d": self.stochrsi.percD[0],
            }
        )

        # Trading Logic
        stochrsi_bullish = self.stochrsi.percK[0] > self.stochrsi.percD[0]
        stochrsi_bearish = self.stochrsi.percK[0] < self.stochrsi.percD[0]

        if not self.position:
            # Long Entry: CCI > threshold + RSI > threshold + Stochastic RSI %K > %D
            if (
                self.cci[0] > self.params.cci_threshold
                and self.rsi[0] > self.params.rsi_threshold
                and stochrsi_bullish
            ):
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - CCI + RSI + StochRSI) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"CCI: {self.cci[0]:.2f} > {self.params.cci_threshold} | "
                    f"RSI: {self.rsi[0]:.2f} > {self.params.rsi_threshold} | "
                    f"StochRSI %K: {self.stochrsi.percK[0]:.2f} > %D: {self.stochrsi.percD[0]:.2f} (Bullish)"
                )
            # Short Entry: CCI < -threshold + RSI < (100 - threshold) + Stochastic RSI %K < %D
            elif (
                self.cci[0] < -self.params.cci_threshold
                and self.rsi[0] < (100 - self.params.rsi_threshold)
                and stochrsi_bearish
            ):
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - CCI + RSI + StochRSI) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"CCI: {self.cci[0]:.2f} < {-self.params.cci_threshold} | "
                    f"RSI: {self.rsi[0]:.2f} < {100 - self.params.rsi_threshold} | "
                    f"StochRSI %K: {self.stochrsi.percK[0]:.2f} < %D: {self.stochrsi.percD[0]:.2f} (Bearish)"
                )
        elif self.position.size > 0:  # Long position
            reverse_count = sum(
                [
                    self.cci[0] < 0,
                    self.rsi[0] < self.params.rsi_threshold,
                    not stochrsi_bullish,
                ]
            )
            if reverse_count >= 2:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "At least two oscillators reversed (CCI, RSI, or StochRSI)"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - CCI + RSI + StochRSI) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"CCI: {self.cci[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"StochRSI %K: {self.stochrsi.percK[0]:.2f}, %D: {self.stochrsi.percD[0]:.2f} | "
                    f"Reverse Count: {reverse_count}"
                )
        elif self.position.size < 0:  # Short position
            reverse_count = sum(
                [
                    self.cci[0] > 0,
                    self.rsi[0] > (100 - self.params.rsi_threshold),
                    not stochrsi_bearish,
                ]
            )
            if reverse_count >= 2:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "At least two oscillators reversed (CCI, RSI, or StochRSI)"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - CCI + RSI + StochRSI) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"CCI: {self.cci[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"StochRSI %K: {self.stochrsi.percK[0]:.2f}, %D: {self.stochrsi.percD[0]:.2f} | "
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
            "cci_period": trial.suggest_int("cci_period", 10, 30),
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "stochrsi_period": trial.suggest_int("stochrsi_period", 10, 20),
            "stochrsi_k": trial.suggest_int("stochrsi_k", 10, 20),
            "stochrsi_d": trial.suggest_int("stochrsi_d", 2, 5),
            "cci_threshold": trial.suggest_int("cci_threshold", 80, 120, step=10),
            "rsi_threshold": trial.suggest_int("rsi_threshold", 45, 55),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            cci_period = params.get("cci_period", 20)
            rsi_period = params.get("rsi_period", 14)
            stochrsi_period = params.get("stochrsi_period", 14)
            return max(cci_period, rsi_period, stochrsi_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
