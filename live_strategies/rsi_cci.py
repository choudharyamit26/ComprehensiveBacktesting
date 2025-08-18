import pandas as pd
import pandas_ta as ta
import numpy as np
import pytz
import datetime
import logging
from uuid import uuid4

from live_strategies.common import COMMON_PARAMS

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class RSICCI:
    """
    RSI + CCI Double Momentum Strategy
    (Documentation remains unchanged)
    """

    params = {
        "rsi_period": 14,
        "cci_period": 14,
        "rsi_bullish": COMMON_PARAMS["rsi_overbought"],  # Use common value for bullish
        "rsi_bearish": COMMON_PARAMS["rsi_oversold"],  # Use common value for bearish
        "cci_bullish": 100,
        "cci_bearish": -100,
        "verbose": False,
    }

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "cci_period": {"type": "int", "low": 10, "high": 25, "step": 1},
        "rsi_bullish": {"type": "int", "low": 50, "high": 70, "step": 1},
        "rsi_bearish": {"type": "int", "low": 30, "high": 50, "step": 1},
        "cci_bullish": {"type": "int", "low": 80, "high": 150, "step": 10},
        "cci_bearish": {"type": "int", "low": -150, "high": -80, "step": 10},
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
            max(
                self.params["rsi_period"],
                self.params["cci_period"],
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        # Calculate indicators
        self.data["rsi"] = ta.rsi(self.data["close"], length=self.params["rsi_period"])
        self.data["cci"] = ta.cci(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            length=self.params["cci_period"],
        )

        # Calculate signals
        self.data["rsi_bullish_signal"] = self.data["rsi"] > self.params["rsi_bullish"]
        self.data["rsi_bearish_signal"] = self.data["rsi"] < self.params["rsi_bearish"]
        self.data["cci_bullish_signal"] = self.data["cci"] > self.params["cci_bullish"]
        self.data["cci_bearish_signal"] = self.data["cci"] < self.params["cci_bearish"]
        self.data["double_bullish"] = (
            self.data["rsi_bullish_signal"] & self.data["cci_bullish_signal"]
        )
        self.data["double_bearish"] = (
            self.data["rsi_bearish_signal"] & self.data["cci_bearish_signal"]
        )
        self.data["long_exit_signal"] = (
            self.data["rsi_bearish_signal"] | self.data["cci_bearish_signal"]
        )
        self.data["short_exit_signal"] = (
            self.data["rsi_bullish_signal"] | self.data["cci_bullish_signal"]
        )
        self.entry_signals = []
        logger.debug(f"Initialized RSICCI with params: {self.params}")

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
                self.data.iloc[idx]["cci"]
            ):
                continue

            # Store indicator data for analysis
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": self.data.iloc[idx]["close"],
                    "rsi": self.data.iloc[idx]["rsi"],
                    "cci": self.data.iloc[idx]["cci"],
                    "rsi_bullish": self.data.iloc[idx]["rsi_bullish_signal"],
                    "rsi_bearish": self.data.iloc[idx]["rsi_bearish_signal"],
                    "cci_bullish": self.data.iloc[idx]["cci_bullish_signal"],
                    "cci_bearish": self.data.iloc[idx]["cci_bearish_signal"],
                    "double_bullish": self.data.iloc[idx]["double_bullish"],
                    "double_bearish": self.data.iloc[idx]["double_bearish"],
                }
            )

            # Trading logic
            if not self.open_positions:
                # Long Entry
                if self.data.iloc[idx]["double_bullish"]:
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
                    self.last_signal = "buy"
                    self._notify_order(idx)
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "BUY"}
                    )
                    trade_logger.info(
                        f"BUY SIGNAL (Double Bullish) | Time: {bar_time_ist} | "
                        f"Price: {self.data.iloc[idx]['close']:.2f} | "
                        f"RSI: {self.data.iloc[idx]['rsi']:.2f} | "
                        f"CCI: {self.data.iloc[idx]['cci']:.2f}"
                    )
                # Short Entry
                elif self.data.iloc[idx]["double_bearish"]:
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
                    self.last_signal = "sell"
                    self._notify_order(idx)
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "SELL"}
                    )
                    trade_logger.info(
                        f"SELL SIGNAL (Double Bearish) | Time: {bar_time_ist} | "
                        f"Price: {self.data.iloc[idx]['close']:.2f} | "
                        f"RSI: {self.data.iloc[idx]['rsi']:.2f} | "
                        f"CCI: {self.data.iloc[idx]['cci']:.2f}"
                    )
            else:
                if self.open_positions[-1]["direction"] == "long":
                    # Long Exit
                    if self.data.iloc[idx]["long_exit_signal"]:
                        reversal_indicator = (
                            "RSI"
                            if self.data.iloc[idx]["rsi_bearish_signal"]
                            else "CCI"
                        )
                        self._close_position(
                            idx, f"{reversal_indicator} reversal", "sell", "exit_long"
                        )
                        self.last_signal = None
                elif self.open_positions[-1]["direction"] == "short":
                    # Short Exit
                    if self.data.iloc[idx]["short_exit_signal"]:
                        reversal_indicator = (
                            "RSI"
                            if self.data.iloc[idx]["rsi_bullish_signal"]
                            else "CCI"
                        )
                        self._close_position(
                            idx, f"{reversal_indicator} reversal", "buy", "exit_short"
                        )
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
            entry_info = self.open_positions.pop(0)
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
            entry_info = self.open_positions.pop(0)
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

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        params = {
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "cci_period": trial.suggest_int("cci_period", 10, 25),
            "rsi_bullish": trial.suggest_int("rsi_bullish", 50, 70),
            "rsi_bearish": trial.suggest_int("rsi_bearish", 30, 50),
            "cci_bullish": trial.suggest_int("cci_bullish", 80, 150),
            "cci_bearish": trial.suggest_int("cci_bearish", -150, -80),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            cci_period = params.get("cci_period", 14)
            return max(rsi_period, cci_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
