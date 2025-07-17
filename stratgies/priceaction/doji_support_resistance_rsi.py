import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class DojiSupportResistanceRSI(bt.Strategy):
    """
    Doji + Support/Resistance + RSI Strategy
    Strategy Type: INTRADAY REVERSAL
    =================================
    Detects Doji candlestick at support/resistance with RSI confirmation.

    Strategy Logic:
    ==============
    Long Entry: Doji at support (price near 20-SMA) + RSI < 30
    Short Entry: Doji at resistance (price near 20-SMA) + RSI > 70
    Exit: Direction confirmed (price moves away from SMA) or level breaks

    Parameters:
    ==========
    - rsi_period (int): RSI period (default: 14)
    - sma_period (int): SMA period for support/resistance (default: 20)
    - doji_threshold (float): Max body size for Doji (default: 0.1)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("rsi_period", 14),
        ("sma_period", 20),
        ("doji_threshold", 0.1),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "sma_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "doji_threshold": {"type": "float", "low": 0.05, "high": 0.2, "step": 0.05},
    }

    def __init__(self):
        self.rsi = btind.RSI_Safe(self.datas[0].close, period=self.params.rsi_period)
        self.sma = btind.SMA(self.datas[0].close, period=self.params.sma_period)
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = max(self.params.rsi_period, self.params.sma_period) + 2
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized DojiSupportResistanceRSI with params: {self.params}")

    def is_doji(self):
        """Detect Doji candlestick (small body relative to range)."""
        body = abs(self.datas[0].close[0] - self.datas[0].open[0])
        candle_range = self.datas[0].high[0] - self.datas[0].low[0]
        return (
            body <= self.params.doji_threshold * candle_range
            if candle_range > 0
            else False
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
            np.isnan(self.rsi[0])
            or np.isinf(self.rsi[0])
            or np.isnan(self.sma[0])
            or np.isinf(self.sma[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: RSI={self.rsi[0]}, SMA={self.sma[0]}"
            )
            return

        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.datas[0].close[0],
                "rsi": self.rsi[0],
                "sma": self.sma[0],
            }
        )

        is_doji = self.is_doji()
        at_support = abs(self.datas[0].close[0] - self.sma[0]) / self.sma[0] < 0.01
        at_resistance = at_support  # Same condition for simplicity

        if not self.position:
            if is_doji and at_support and self.rsi[0] < 30:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Doji at Support) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | RSI: {self.rsi[0]:.2f} | SMA: {self.sma[0]:.2f}"
                )
            elif is_doji and at_resistance and self.rsi[0] > 70:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Doji at Resistance) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | RSI: {self.rsi[0]:.2f} | SMA: {self.sma[0]:.2f}"
                )
        elif self.position.size > 0:
            if (
                self.datas[0].close[0] > self.sma[0] * 1.02
                or self.datas[0].close[0] < self.sma[0] * 0.98
            ):
                self.order = self.sell()
                self.order_type = "exit_long"
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | Reason: {'Direction confirmed' if self.datas[0].close[0] > self.sma[0] * 1.02 else 'Level break'}"
                )
        elif self.position.size < 0:
            if (
                self.datas[0].close[0] < self.sma[0] * 0.98
                or self.datas[0].close[0] > self.sma[0] * 1.02
            ):
                self.order = self.buy()
                self.order_type = "exit_short"
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | Reason: {'Direction confirmed' if self.datas[0].close[0] < self.sma[0] * 0.98 else 'Level break'}"
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
                    f"BUY EXECUTED (Enter Long) | Ref: {order.ref} | Price: {order.executed.price:.2f} | "
                    f"Size: {order.executed.size} | Cost: {order.executed.value:.2f} | Comm: {order.executed.comm:.2f}"
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
                    f"SELL EXECUTED (Enter Short) | Ref: {order.ref} | Price: {order.executed.price:.2f} | "
                    f"Size: {order.executed.size} | Cost: {order.executed.value:.2f} | Comm: {order.executed.comm:.2f}"
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
                        f"SELL EXECUTED (Exit Long) | Ref: {order.ref} | Price: {order.executed.price:.2f} | "
                        f"Size: {order.executed.size} | Cost: {order.executed.value:.2f} | Comm: {order.executed.comm:.2f} | PnL: {pnl:.2f}"
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
                        f"BUY EXECUTED (Exit Short) | Ref: {order.ref} | Price: {order.executed.price:.2f} | "
                        f"Size: {order.executed.size} | Cost: {order.executed.value:.2f} | Comm: {order.executed.comm:.2f} | PnL: {pnl:.2f}"
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
                f"TRADE CLOSED | Ref: {trade.ref} | Profit: {trade.pnl:.2f} | "
                f"Net Profit: {trade.pnlcomm:.2f} | Bars Held: {trade.barlen} | Trade Count: {self.trade_count}"
            )

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        params = {
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "sma_period": trial.suggest_int("sma_period", 10, 30),
            "doji_threshold": trial.suggest_float(
                "doji_threshold", 0.05, 0.2, step=0.05
            ),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            sma_period = params.get("sma_period", 20)
            return max(rsi_period, sma_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
