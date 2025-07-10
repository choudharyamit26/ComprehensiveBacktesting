import backtrader as bt
import backtrader.indicators as btind
import logging
import datetime
import pytz
import numpy as np
import os

# Setup dedicated trade logger
trade_logger = logging.getLogger("trade_logger")
trade_logger.setLevel(logging.INFO)

# Create logs directory if not exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Create file handler for trade logging
fh = logging.FileHandler("logs/trade_executions_ema_rsi.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
fh.setFormatter(formatter)
trade_logger.addHandler(fh)

logger = logging.getLogger(__name__)


class EMARSI(bt.Strategy):
    """EMA and RSI combined trading strategy with shorting logic"""

    params = (
        ("fast_ema_period", 12),
        ("slow_ema_period", 36),
        ("rsi_period", 14),
        ("rsi_upper", 70),
        ("rsi_lower", 30),
        ("verbose", False),
    )

    optimization_params = {
        "fast_ema_period": {"type": "int", "low": 5, "high": 15, "step": 1},
        "slow_ema_period": {"type": "int", "low": 16, "high": 30, "step": 1},
        "rsi_period": {"type": "int", "low": 8, "high": 20, "step": 1},
        "rsi_upper": {"type": "int", "low": 60, "high": 80, "step": 5},
        "rsi_lower": {"type": "int", "low": 20, "high": 40, "step": 5},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.fast_ema = btind.EMA(self.data.close, period=self.params.fast_ema_period)
        self.slow_ema = btind.EMA(self.data.close, period=self.params.slow_ema_period)
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.order = None
        self.order_type = None  # Track order type for shorting logic
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.fast_ema_period,
                self.params.slow_ema_period,
                self.params.rsi_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized EMARSI with params: {self.params}")
        logger.info(
            f"EMARSI initialized with fast_ema_period={self.p.fast_ema_period}, "
            f"slow_ema_period={self.p.slow_ema_period}, rsi_period={self.p.rsi_period}, "
            f"rsi_upper={self.p.rsi_upper}, rsi_lower={self.p.rsi_lower}"
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
            np.isnan(self.fast_ema[0])
            or np.isnan(self.slow_ema[0])
            or np.isnan(self.rsi[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"FastEMA={self.fast_ema[0]}, SlowEMA={self.slow_ema[0]}, "
                f"RSI={self.rsi[0]}"
            )
            return

        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "fast_ema": self.fast_ema[0],
                "slow_ema": self.slow_ema[0],
                "rsi": self.rsi[0],
            }
        )

        # Position management with shorting
        if not self.position:
            if (
                self.fast_ema[0] > self.slow_ema[0]
                and self.rsi[0] < self.params.rsi_lower
            ):
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"FastEMA: {self.fast_ema[0]:.2f} > SlowEMA: {self.slow_ema[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} < {self.params.rsi_lower}"
                )
            elif (
                self.fast_ema[0] < self.slow_ema[0]
                and self.rsi[0] > self.params.rsi_upper
            ):
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"FastEMA: {self.fast_ema[0]:.2f} < SlowEMA: {self.slow_ema[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} > {self.params.rsi_upper}"
                )
        elif self.position.size > 0:  # Long position
            if (
                self.fast_ema[0] < self.slow_ema[0]
                or self.rsi[0] > self.params.rsi_upper
            ):
                self.order = self.sell()
                self.order_type = "exit_long"
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"FastEMA: {self.fast_ema[0]:.2f} < SlowEMA: {self.slow_ema[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} > {self.params.rsi_upper}"
                )
        elif self.position.size < 0:  # Short position
            if (
                self.fast_ema[0] > self.slow_ema[0]
                or self.rsi[0] < self.params.rsi_lower
            ):
                self.order = self.buy()
                self.order_type = "exit_short"
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"FastEMA: {self.fast_ema[0]:.2f} > SlowEMA: {self.slow_ema[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} < {self.params.rsi_lower}"
                )

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt)
            if exec_dt.tzinfo is None:
                exec_dt = exec_dt.replace(tzinfo=pytz.UTC)

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
                    "size": order.executed.size,  # Negative for short
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
            "fast_ema_period": trial.suggest_int("fast_ema_period", 5, 12),
            "slow_ema_period": trial.suggest_int("slow_ema_period", 15, 25),
            "rsi_period": trial.suggest_int("rsi_period", 8, 15),
            "rsi_upper": trial.suggest_int("rsi_upper", 65, 75),
            "rsi_lower": trial.suggest_int("rsi_lower", 25, 35),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            fast_ema_period = params.get("fast_ema_period", 5)
            slow_ema_period = params.get("slow_ema_period", 10)
            rsi_period = params.get("rsi_period", 10)
            if slow_ema_period <= fast_ema_period:
                slow_ema_period = fast_ema_period + 5
            max_period = max(fast_ema_period, slow_ema_period, rsi_period)
            return max_period + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
