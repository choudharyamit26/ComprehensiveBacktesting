import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class RSIMACDEMATrend(bt.Strategy):
    """
    RSI + MACD + EMA Trend Strategy
    Strategy Type: TREND + MOMENTUM
    =============================
    This strategy uses RSI, MACD, and EMA to confirm trades in the direction of the trend.

    Strategy Logic:
    ==============
    Long Entry: RSI > 50 + MACD bullish + Price above EMA
    Short Entry: RSI < 50 + MACD bearish + Price below EMA
    Exit: Majority of indicators reverse (at least two of: RSI crosses 50, MACD reverses, price crosses EMA)

    Parameters:
    ==========
    - rsi_period (int): RSI calculation period (default: 14)
    - macd_fast (int): MACD fast EMA period (default: 12)
    - macd_slow (int): MACD slow EMA period (default: 26)
    - macd_signal (int): MACD signal line period (default: 9)
    - ema_period (int): EMA period (default: 20)
    - rsi_threshold (int): RSI threshold (default: 50)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("rsi_period", 14),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("ema_period", 20),
        ("rsi_threshold", 50),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "macd_fast": {"type": "int", "low": 8, "high": 16, "step": 1},
        "macd_slow": {"type": "int", "low": 20, "high": 30, "step": 1},
        "macd_signal": {"type": "int", "low": 6, "high": 12, "step": 1},
        "ema_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "rsi_threshold": {"type": "int", "low": 45, "high": 55, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.macd = btind.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal,
        )
        self.ema = btind.EMA(self.data.close, period=self.params.ema_period)

        # Debug: Log available lines and their types
        logger.debug(f"RSI lines: {self.rsi.lines.getlinealiases()}")
        logger.debug(f"MACD lines: {self.macd.lines.getlinealiases()}")
        logger.debug(f"EMA lines: {self.ema.lines.getlinealiases()}")

        # Initialize order attributes
        self.order = None  # Initialize self.order to prevent AttributeError
        self.order_type = None
        self.is_order = None
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            max(self.params.rsi_period, self.params.macd_slow, self.params.ema_period)
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized RSIMACDEMATrend with params: {self.params}")
        logger.info(
            f"RSIMACDEMATrend initialized with rsi_period={self.p.rsi_period}, "
            f"macd_fast={self.p.macd_fast}, macd_slow={self.p.macd_slow}, "
            f"macd_signal={self.p.macd_signal}, ema_period={self.p.ema_period}, "
            f"rsi_threshold={self.p.rsi_threshold}"
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
            np.isnan(self.rsi[0])
            or np.isnan(self.macd.macd[0])
            or np.isnan(self.macd.signal[0])
            or np.isnan(self.ema[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"RSI={self.rsi[0]}, MACD={self.macd.macd[0]}, "
                f"MACD Signal={self.macd.signal[0]}, EMA={self.ema[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "rsi": self.rsi[0],
                "macd": self.macd.macd[0],
                "macd_signal": self.macd.signal[0],
                "ema": self.ema[0],
            }
        )

        # Trading Logic
        macd_bullish = self.macd.macd[0] > self.macd.signal[0]
        macd_bearish = self.macd.macd[0] < self.macd.signal[0]
        price_above_ema = self.data.close[0] > self.ema[0]
        price_below_ema = self.data.close[0] < self.ema[0]

        if not self.position:
            # Long Entry: RSI > threshold + MACD bullish + Price above EMA
            if (
                self.rsi[0] > self.params.rsi_threshold
                and macd_bullish
                and price_above_ema
            ):
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - RSI + MACD + EMA) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} > {self.params.rsi_threshold} | "
                    f"MACD: {self.macd.macd[0]:.2f} > {self.macd.signal[0]:.2f} (Bullish) | "
                    f"Price: {self.data.close[0]:.2f} > EMA: {self.ema[0]:.2f}"
                )
            # Short Entry: RSI < (100 - threshold) + MACD bearish + Price below EMA
            elif (
                self.rsi[0] < (100 - self.params.rsi_threshold)
                and macd_bearish
                and price_below_ema
            ):
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - RSI + MACD + EMA) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} < {100 - self.params.rsi_threshold} | "
                    f"MACD: {self.macd.macd[0]:.2f} < {self.macd.signal[0]:.2f} (Bearish) | "
                    f"Price: {self.data.close[0]:.2f} < EMA: {self.ema[0]:.2f}"
                )
        elif self.position.size > 0:  # Long position
            reverse_count = sum(
                [
                    self.rsi[0] < self.params.rsi_threshold,
                    self.macd.macd[0] < self.macd.signal[0],
                    self.data.close[0] < self.ema[0],
                ]
            )
            if reverse_count >= 2:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "At least two indicators reversed (RSI, MACD, or Price vs EMA)"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - RSI + MACD + EMA) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.2f}, Signal: {self.macd.signal[0]:.2f} | "
                    f"EMA: {self.ema[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            reverse_count = sum(
                [
                    self.rsi[0] > (100 - self.params.rsi_threshold),
                    self.macd.macd[0] > self.macd.signal[0],
                    self.data.close[0] > self.ema[0],
                ]
            )
            if reverse_count >= 2:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "At least two indicators reversed (RSI, MACD, or Price vs EMA)"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - RSI + MACD + EMA) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.2f}, Signal: {self.macd.signal[0]:.2f} | "
                    f"EMA: {self.ema[0]:.2f}"
                )

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt)
            if exec_dt.tzinfo is None:
                exec_dt = exec_dt.replace(tzinfo=pytz.timezone("Asia/Kolkata"))

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
                        "entry_User time is 11:10 PM IST on Wednesday, July 16, 2025.time": entry_info[
                            "entry_time"
                        ],
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
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "macd_fast": trial.suggest_int("macd_fast", 8, 16),
            "macd_slow": trial.suggest_int("macd_slow", 20, 30),
            "macd_signal": trial.suggest_int("macd_signal", 6, 12),
            "ema_period": trial.suggest_int("ema_period", 10, 30),
            "rsi_threshold": trial.suggest_int("rsi_threshold", 45, 55),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            macd_slow = params.get("macd_slow", 26)
            ema_period = params.get("ema_period", 20)
            return max(rsi_period, macd_slow, ema_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
