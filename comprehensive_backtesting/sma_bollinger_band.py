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
fh = logging.FileHandler("logs/trade_executions_sma_bollinger.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
fh.setFormatter(formatter)
trade_logger.addHandler(fh)

logger = logging.getLogger(__name__)


class SMABollinger(bt.Strategy):
    """SMA Crossover and Bollinger Bands trading strategy with shorting logic"""

    params = (
        ("fast_sma_period", 10),
        ("slow_sma_period", 20),
        ("bb_period", 20),
        ("bb_dev", 2.0),
        ("verbose", False),
    )

    optimization_params = {
        "fast_sma_period": {"type": "int", "low": 5, "high": 15, "step": 1},
        "slow_sma_period": {"type": "int", "low": 16, "high": 30, "step": 1},
        "bb_period": {"type": "int", "low": 15, "high": 25, "step": 1},
        "bb_dev": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.fast_sma = btind.SMA(self.data.close, period=self.params.fast_sma_period)
        self.slow_sma = btind.SMA(self.data.close, period=self.params.slow_sma_period)
        self.bbands = btind.BollingerBands(
            self.data.close, period=self.params.bb_period, devfactor=self.params.bb_dev
        )
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.fast_sma_period,
                self.params.slow_sma_period,
                self.params.bb_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized SMABollinger with params: {self.params}")
        logger.info(
            f"SMABollinger initialized with fast_sma_period={self.p.fast_sma_period}, "
            f"slow_sma_period={self.p.slow_sma_period}, bb_period={self.p.bb_period}, "
            f"bb_dev={self.p.bb_dev}"
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

        if self.order:
            logger.debug(f"Order pending at bar {len(self)}")
            return

        if (
            np.isnan(self.fast_sma[0])
            or np.isnan(self.slow_sma[0])
            or np.isnan(self.bbands.mid[0])
            or np.isnan(self.bbands.top[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"FastSMA={self.fast_sma[0]}, SlowSMA={self.slow_sma[0]}, "
                f"BB_Mid={self.bbands.mid[0]}, BB_Top={self.bbands.top[0]}"
            )
            return

        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "fast_sma": self.fast_sma[0],
                "slow_sma": self.slow_sma[0],
                "bb_mid": self.bbands.mid[0],
                "bb_top": self.bbands.top[0],
                "bb_bot": self.bbands.bot[0],
            }
        )

        # Position management with shorting
        if not self.position:
            if self.fast_sma[0] > self.slow_sma[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"FastSMA: {self.fast_sma[0]:.2f} > SlowSMA: {self.slow_sma[0]:.2f}"
                )
            elif self.fast_sma[0] < self.slow_sma[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"FastSMA: {self.fast_sma[0]:.2f} < SlowSMA: {self.slow_sma[0]:.2f}"
                )
        elif self.position.size > 0:  # Long position
            if self.fast_sma[0] < self.slow_sma[0]:
                self.order = self.sell()
                self.order_type = "exit_long"
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"FastSMA: {self.fast_sma[0]:.2f} < SlowSMA: {self.slow_sma[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            if self.fast_sma[0] > self.slow_sma[0]:
                self.order = self.buy()
                self.order_type = "exit_short"
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"FastSMA: {self.fast_sma[0]:.2f} > SlowSMA: {self.slow_sma[0]:.2f}"
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
    def get_min_data_points(cls, params):
        try:
            fast_sma_period = params.get("fast_sma_period", 10)
            slow_sma_period = params.get("slow_sma_period", 20)
            bb_period = params.get("bb_period", 20)
            if slow_sma_period <= fast_sma_period:
                slow_sma_period = fast_sma_period + 5
            max_period = max(fast_sma_period, slow_sma_period, bb_period)
            return max_period + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
