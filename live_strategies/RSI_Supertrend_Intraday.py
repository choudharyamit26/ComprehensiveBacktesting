import pandas as pd
import pandas_ta as ta
import numpy as np
import pytz
import datetime
import logging
from uuid import uuid4

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class RSISupertrendIntraday:
    """
    RSI + Supertrend Intraday Strategy
    Fixed version with proper position management
    """

    params = {
        "rsi_period": 14,
        "supertrend_period": 10,
        "supertrend_mult": 3.0,
        "target_percent": 2.0,
        "rsi_overbought": 80,
        "rsi_oversold": 20,
        "verbose": False,
    }

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "supertrend_period": {"type": "int", "low": 7, "high": 14, "step": 1},
        "supertrend_mult": {"type": "float", "low": 2.0, "high": 4.0, "step": 0.5},
        "target_percent": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.5},
    }

    def __init__(self, data, tickers=None, **kwargs):
        self.data = data.copy()
        self.data["datetime"] = pd.to_datetime(self.data["datetime"])
        self.params.update(kwargs)
        self.order = None
        self.order_type = None
        self.last_signal = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(self.params["rsi_period"], self.params["supertrend_period"]) + 5
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []
        self.entry_price = None

        # Calculate indicators
        self.data["rsi"] = ta.rsi(self.data["close"], length=self.params["rsi_period"])
        self.data["supertrend"] = ta.supertrend(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            length=self.params["supertrend_period"],
            multiplier=self.params["supertrend_mult"],
        )[f"SUPERT_{self.params['supertrend_period']}_{self.params['supertrend_mult']}"]

        # Calculate signals
        self.data["bullish_entry"] = (self.data["rsi"] > 50) & (
            self.data["close"] > self.data["supertrend"]
        )
        self.data["bearish_entry"] = (self.data["rsi"] < 50) & (
            self.data["close"] < self.data["supertrend"]
        )

        logger.debug(f"Initialized RSISupertrendIntraday with params: {self.params}")

    def run(self):
        self.last_signal = None
        for idx in range(len(self.data)):
            if idx < self.warmup_period:
                continue

            if not self.ready:
                self.ready = True
                logger.info(f"Strategy ready at row {idx}")

            bar_time = self.data.iloc[idx]["datetime"]
            bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
            current_time = bar_time_ist.time()

            # Force close positions at 15:15 IST
            if current_time >= datetime.time(15, 15):
                if self.open_positions:
                    self._close_position(idx, "Force close at 15:15 IST")
                    self.last_signal = None
                continue

            # Only trade during market hours (9:15 AM to 3:05 PM IST)
            if not (datetime.time(9, 15) <= current_time <= datetime.time(15, 5)):
                continue

            if self.order:
                logger.debug(f"Order pending at row {idx}")
                continue

            # Check for invalid indicator values
            if pd.isna(self.data.iloc[idx]["rsi"]) or pd.isna(
                self.data.iloc[idx]["supertrend"]
            ):
                continue

            # Store indicator data for analysis
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": self.data.iloc[idx]["close"],
                    "rsi": self.data.iloc[idx]["rsi"],
                    "supertrend": self.data.iloc[idx]["supertrend"],
                    "bullish_entry": self.data.iloc[idx]["bullish_entry"],
                    "bearish_entry": self.data.iloc[idx]["bearish_entry"],
                }
            )

            # Trading logic
            if not self.open_positions:
                # Long Entry
                if self.data.iloc[idx]["bullish_entry"]:
                    self.order = {
                        "ref": str(uuid4()),
                        "action": "buy",
                        "order_type": "enter_long",
                        "status": "Completed",
                        "executed_price": self.data.iloc[idx]["close"],
                        "size": 100,
                        "commission": abs(self.data.iloc[idx]["close"] * 100 * 0.001),
                        "executed_time": self.data.iloc[idx]["datetime"],
                    }
                    self.entry_price = self.data.iloc[idx]["close"]
                    self.last_signal = "buy"
                    self._notify_order(idx)
                    trade_logger.info(
                        f"BUY SIGNAL | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f}"
                    )
                # Short Entry
                elif self.data.iloc[idx]["bearish_entry"]:
                    self.order = {
                        "ref": str(uuid4()),
                        "action": "sell",
                        "order_type": "enter_short",
                        "status": "Completed",
                        "executed_price": self.data.iloc[idx]["close"],
                        "size": -100,
                        "commission": abs(self.data.iloc[idx]["close"] * 100 * 0.001),
                        "executed_time": self.data.iloc[idx]["datetime"],
                    }
                    self.entry_price = self.data.iloc[idx]["close"]
                    self.last_signal = "sell"
                    self._notify_order(idx)
                    trade_logger.info(
                        f"SELL SIGNAL | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f}"
                    )
            else:
                # Check if we have a valid open position and entry price
                if self.open_positions and self.entry_price is not None:
                    if self.open_positions[-1]["direction"] == "long":
                        target_price = self.entry_price * (
                            1 + self.params["target_percent"] / 100
                        )
                        if (
                            self.data.iloc[idx]["close"] >= target_price
                            or self.data.iloc[idx]["rsi"]
                            > self.params["rsi_overbought"]
                        ):
                            reason = (
                                "Target hit"
                                if self.data.iloc[idx]["close"] >= target_price
                                else "RSI overbought"
                            )
                            self._close_position(idx, reason, "sell", "exit_long")
                            self.last_signal = None
                    elif self.open_positions[-1]["direction"] == "short":
                        target_price = self.entry_price * (
                            1 - self.params["target_percent"] / 100
                        )
                        if (
                            self.data.iloc[idx]["close"] <= target_price
                            or self.data.iloc[idx]["rsi"] < self.params["rsi_oversold"]
                        ):
                            reason = (
                                "Target hit"
                                if self.data.iloc[idx]["close"] <= target_price
                                else "RSI oversold"
                            )
                            self._close_position(idx, reason, "buy", "exit_short")
                            self.last_signal = None
        return self.last_signal

    def _notify_order(self, idx):
        order = self.order
        exec_dt = order["executed_time"]
        if exec_dt.tzinfo is None:
            exec_dt = exec_dt.replace(tzinfo=pytz.UTC)

        if order["order_type"] == "enter_long" and order["action"] == "buy":
            position_info = {
                "entry_time": exec_dt,
                "entry_price": order["executed_price"],
                "size": order["size"],
                "commission": order["commission"],
                "ref": order["ref"],
                "direction": "long",
            }
            self.open_positions.append(position_info)
            trade_logger.info(
                f"BUY EXECUTED (Enter Long) | Ref: {order['ref']} | Price: {order['executed_price']:.2f}"
            )
        elif order["order_type"] == "enter_short" and order["action"] == "sell":
            position_info = {
                "entry_time": exec_dt,
                "entry_price": order["executed_price"],
                "size": order["size"],
                "commission": order["commission"],
                "ref": order["ref"],
                "direction": "short",
            }
            self.open_positions.append(position_info)
            trade_logger.info(
                f"SELL EXECUTED (Enter Short) | Ref: {order['ref']} | Price: {order['executed_price']:.2f}"
            )

        self.order = None
        self.order_type = None

    def _close_position(self, idx, reason, action=None, order_type=None):
        if not self.open_positions:
            return

        entry_info = self.open_positions[-1]

        # Use entry_price from the position info if self.entry_price is None
        if self.entry_price is None:
            self.entry_price = entry_info["entry_price"]
            logger.warning(
                f"entry_price was None, using position entry_price: {self.entry_price}"
            )

        self.order = {
            "ref": str(uuid4()),
            "action": action,
            "order_type": order_type,
            "status": "Completed",
            "executed_price": self.data.iloc[idx]["close"],
            "size": -entry_info["size"],
            "commission": abs(
                self.data.iloc[idx]["close"] * abs(entry_info["size"]) * 0.001
            ),
            "executed_time": self.data.iloc[idx]["datetime"],
        }
        order = self.order

        if order["order_type"] == "exit_long" and order["action"] == "sell":
            entry_info = (
                self.open_positions.pop()
            )  # Changed from pop(0) to pop() for LIFO
            pnl = (order["executed_price"] - entry_info["entry_price"]) * abs(
                entry_info["size"]
            )
            total_commission = entry_info["commission"] + abs(order["commission"])
            trade_info = {
                "ref": order["ref"],
                "entry_time": entry_info["entry_time"],
                "exit_time": order["executed_time"],
                "entry_price": entry_info["entry_price"],
                "exit_price": order["executed_price"],
                "size": abs(entry_info["size"]),
                "pnl": pnl,
                "pnl_net": pnl - total_commission,
                "commission": total_commission,
                "status": "Won" if pnl > 0 else "Lost",
                "direction": "Long",
                "bars_held": (
                    order["executed_time"] - entry_info["entry_time"]
                ).total_seconds()
                / 60,
            }
            self.completed_trades.append(trade_info)
            self.trade_count += 1
            trade_logger.info(
                f"SELL EXECUTED (Exit Long) | Ref: {order['ref']} | PnL: {pnl:.2f} | Reason: {reason}"
            )
        elif order["order_type"] == "exit_short" and order["action"] == "buy":
            entry_info = (
                self.open_positions.pop()
            )  # Changed from pop(0) to pop() for LIFO
            pnl = (entry_info["entry_price"] - order["executed_price"]) * abs(
                entry_info["size"]
            )
            total_commission = entry_info["commission"] + abs(order["commission"])
            trade_info = {
                "ref": order["ref"],
                "entry_time": entry_info["entry_time"],
                "exit_time": order["executed_time"],
                "entry_price": entry_info["entry_price"],
                "exit_price": order["executed_price"],
                "size": abs(entry_info["size"]),
                "pnl": pnl,
                "pnl_net": pnl - total_commission,
                "commission": total_commission,
                "status": "Won" if pnl > 0 else "Lost",
                "direction": "Short",
                "bars_held": (
                    order["executed_time"] - entry_info["entry_time"]
                ).total_seconds()
                / 60,
            }
            self.completed_trades.append(trade_info)
            self.trade_count += 1
            trade_logger.info(
                f"BUY EXECUTED (Exit Short) | Ref: {order['ref']} | PnL: {pnl:.2f} | Reason: {reason}"
            )

        self.order = None
        self.order_type = None
        # Only reset entry_price if no more open positions
        if not self.open_positions:
            self.entry_price = None

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        params = {
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "supertrend_period": trial.suggest_int("supertrend_period", 7, 14),
            "supertrend_mult": trial.suggest_float(
                "supertrend_mult", 2.0, 4.0, step=0.5
            ),
            "target_percent": trial.suggest_float("target_percent", 1.5, 3.0, step=0.5),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            supertrend_period = params.get("supertrend_period", 10)
            return max(rsi_period, supertrend_period) + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 25
