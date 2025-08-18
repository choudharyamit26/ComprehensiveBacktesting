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


class PivotCCI:
    """
    Pivot Points + CCI Strategy
    (Documentation remains unchanged)
    """

    params = {
        "cci_period": 14,
        "pivot_proximity": 0.5,
        "cci_oversold": COMMON_PARAMS["cci_oversold"],  # Use common value
        "cci_overbought": COMMON_PARAMS["cci_overbought"],  # Use common value,
        "cci_exit": 0,
        "verbose": False,
    }

    optimization_params = {
        "cci_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "pivot_proximity": {"type": "float", "low": 0.3, "high": 1.0, "step": 0.1},
        "cci_oversold": {"type": "int", "low": -150, "high": -50, "step": 10},
        "cci_overbought": {"type": "int", "low": 50, "high": 150, "step": 10},
        "cci_exit": {"type": "int", "low": -20, "high": 20, "step": 5},
    }

    def __init__(self, data, tickers=None, **kwargs):
        self.data = data.copy()
        self.data["datetime"] = pd.to_datetime(self.data["datetime"])
        self.params.update(kwargs)
        self.order = None
        self.order_type = None
        self.last_signal = None  # Initialize last_signal
        self.ready = False
        self.trade_count = 0
        self.warmup_period = self.params["cci_period"] + 5
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []
        self.entry_signals = []
        # Initialize indicators using pandas_ta
        self.data["cci"] = ta.cci(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            length=self.params["cci_period"],
        )

        # Calculate pivot points
        self.data["prev_high"] = self.data["high"].shift(1)
        self.data["prev_low"] = self.data["low"].shift(1)
        self.data["prev_close"] = self.data["close"].shift(1)
        self.data["pivot"] = (
            self.data["prev_high"] + self.data["prev_low"] + self.data["prev_close"]
        ) / 3
        self.data["r1"] = (2 * self.data["pivot"]) - self.data["prev_low"]
        self.data["s1"] = (2 * self.data["pivot"]) - self.data["prev_high"]
        self.data["r2"] = self.data["pivot"] + (
            self.data["prev_high"] - self.data["prev_low"]
        )
        self.data["s2"] = self.data["pivot"] - (
            self.data["prev_high"] - self.data["prev_low"]
        )

        # Define conditions
        self.data["near_s1"] = self.data.apply(
            lambda x: (
                abs(x["close"] - x["s1"]) / x["s1"]
                < self.params["pivot_proximity"] / 100
                if pd.notna(x["s1"])
                else False
            ),
            axis=1,
        )
        self.data["near_s2"] = self.data.apply(
            lambda x: (
                abs(x["close"] - x["s2"]) / x["s2"]
                < self.params["pivot_proximity"] / 100
                if pd.notna(x["s2"])
                else False
            ),
            axis=1,
        )
        self.data["near_r1"] = self.data.apply(
            lambda x: (
                abs(x["close"] - x["r1"]) / x["r1"]
                < self.params["pivot_proximity"] / 100
                if pd.notna(x["r1"])
                else False
            ),
            axis=1,
        )
        self.data["near_r2"] = self.data.apply(
            lambda x: (
                abs(x["close"] - x["r2"]) / x["r2"]
                < self.params["pivot_proximity"] / 100
                if pd.notna(x["r2"])
                else False
            ),
            axis=1,
        )
        self.data["bullish_entry"] = (
            self.data["cci"] < self.params["cci_oversold"]
        ) & (self.data["near_s1"] | self.data["near_s2"])
        self.data["bearish_entry"] = (
            self.data["cci"] > self.params["cci_overbought"]
        ) & (self.data["near_r1"] | self.data["near_r2"])
        self.data["bullish_exit"] = (self.data["cci"] > self.params["cci_exit"]) | (
            self.data["close"] >= self.data["pivot"]
        )
        self.data["bearish_exit"] = (self.data["cci"] < self.params["cci_exit"]) | (
            self.data["close"] <= self.data["pivot"]
        )

        logger.debug(f"Initialized PivotCCI with params: {self.params}")
        logger.info(
            f"PivotCCI initialized with cci_period={self.params['cci_period']}, "
            f"pivot_proximity={self.params['pivot_proximity']}, cci_oversold={self.params['cci_oversold']}, "
            f"cci_overbought={self.params['cci_overbought']}, cci_exit={self.params['cci_exit']}"
        )

    def run(self):
        self.last_signal = None  # Reset last_signal at the start of run
        for idx in range(len(self.data)):
            if idx < self.warmup_period:
                logger.debug(
                    f"Skipping row {idx}: still in warmup period (need {self.warmup_period} rows)"
                )
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
            if pd.isna(self.data.iloc[idx]["cci"]) or pd.isna(
                self.data.iloc[idx]["pivot"]
            ):
                logger.debug(f"Invalid indicator values at row {idx}")
                continue

            # Store indicator data for analysis
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": self.data.iloc[idx]["close"],
                    "cci": self.data.iloc[idx]["cci"],
                    "pivot": self.data.iloc[idx]["pivot"],
                    "s1": self.data.iloc[idx]["s1"],
                    "s2": self.data.iloc[idx]["s2"],
                    "r1": self.data.iloc[idx]["r1"],
                    "r2": self.data.iloc[idx]["r2"],
                }
            )

            # Check for trading signals
            if not self.open_positions:
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
                    self.last_signal = "buy"
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "BUY"}
                    )
                    self._notify_order(idx)
                    trade_logger.info(
                        f"BUY SIGNAL | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f} | "
                        f"CCI: {self.data.iloc[idx]['cci']:.2f} | Near S1: {self.data.iloc[idx]['near_s1']} | Near S2: {self.data.iloc[idx]['near_s2']}"
                    )
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
                    self.last_signal = "sell"
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "SELL"}
                    )
                    self._notify_order(idx)
                    trade_logger.info(
                        f"SELL SIGNAL | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f} | "
                        f"CCI: {self.data.iloc[idx]['cci']:.2f} | Near R1: {self.data.iloc[idx]['near_r1']} | Near R2: {self.data.iloc[idx]['near_r2']}"
                    )
                else:
                    self.last_signal = None
            else:
                if (
                    self.open_positions[-1]["direction"] == "long"
                    and self.data.iloc[idx]["bullish_exit"]
                ):
                    self._close_position(
                        idx, "Bullish exit condition", "sell", "exit_long"
                    )
                    self.last_signal = None
                    trade_logger.info(
                        f"EXIT LONG | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f} | "
                        f"CCI: {self.data.iloc[idx]['cci']:.2f} | Reached Pivot: {self.data.iloc[idx]['close'] >= self.data.iloc[idx]['pivot']}"
                    )
                elif (
                    self.open_positions[-1]["direction"] == "short"
                    and self.data.iloc[idx]["bearish_exit"]
                ):
                    self._close_position(
                        idx, "Bearish exit condition", "buy", "exit_short"
                    )
                    self.last_signal = None
                    trade_logger.info(
                        f"EXIT SHORT | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f} | "
                        f"CCI: {self.data.iloc[idx]['cci']:.2f} | Reached Pivot: {self.data.iloc[idx]['close'] <= self.data.iloc[idx]['pivot']}"
                    )
                else:
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
            "cci_period": trial.suggest_int("cci_period", 10, 20),
            "pivot_proximity": trial.suggest_float(
                "pivot_proximity", 0.3, 1.0, step=0.1
            ),
            "cci_oversold": trial.suggest_int("cci_oversold", -150, -50, step=10),
            "cci_overbought": trial.suggest_int("cci_overbought", 50, 150, step=10),
            "cci_exit": trial.suggest_int("cci_exit", -20, 20, step=5),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            cci_period = params.get("cci_period", 14)
            return cci_period + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 20
