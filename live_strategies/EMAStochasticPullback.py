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


class EMAStochasticPullback:
    """
    EMA + Stochastic Pullback Trading Strategy
    (Documentation remains unchanged)
    """

    params = {
        "ema_period": 20,
        "stoch_k_period": 14,
        "stoch_d_period": 3,
        "stoch_slowing": 3,
        "stoch_oversold": COMMON_PARAMS["stoch_oversold"],
        "stoch_overbought": COMMON_PARAMS["stoch_overbought"],
        "verbose": False,
    }

    optimization_params = {
        "ema_period": {"type": "int", "low": 10, "high": 50, "step": 5},
        "stoch_k_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stoch_d_period": {"type": "int", "low": 3, "high": 5, "step": 1},
        "stoch_slowing": {"type": "int", "low": 1, "high": 5, "step": 1},
        "stoch_oversold": {"type": "int", "low": 10, "high": 30, "step": 5},
        "stoch_overbought": {"type": "int", "low": 70, "high": 90, "step": 5},
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
        self.warmup_period = max(
            self.params["ema_period"],
            self.params["stoch_k_period"] + self.params["stoch_d_period"] + 2,
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []
        self.entry_signals = []
        # Calculate EMA
        self.data["ema"] = ta.ema(self.data["close"], length=self.params["ema_period"])

        # Calculate Stochastic
        stoch = ta.stoch(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            k=self.params["stoch_k_period"],
            d=self.params["stoch_d_period"],
            smooth_k=self.params["stoch_slowing"],
        )
        self.data["stoch_k"] = stoch[
            f"STOCHk_{self.params['stoch_k_period']}_{self.params['stoch_d_period']}_{self.params['stoch_slowing']}"
        ]
        self.data["stoch_d"] = stoch[
            f"STOCHd_{self.params['stoch_k_period']}_{self.params['stoch_d_period']}_{self.params['stoch_slowing']}"
        ]

        # Calculate EMA touch conditions
        self.data["ema_touch_long"] = self.data["close"] <= self.data["ema"]
        self.data["ema_touch_short"] = self.data["close"] >= self.data["ema"]

        # Calculate EMA distance
        self.data["ema_distance"] = (self.data["close"] - self.data["ema"]) / self.data[
            "ema"
        ]

        logger.debug(f"Initialized EMAStochasticPullback with params: {self.params}")

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
            if (
                pd.isna(self.data.iloc[idx]["ema"])
                or pd.isna(self.data.iloc[idx]["stoch_k"])
                or pd.isna(self.data.iloc[idx]["stoch_d"])
            ):
                continue

            # Store indicator data for analysis
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": self.data.iloc[idx]["close"],
                    "ema": self.data.iloc[idx]["ema"],
                    "stoch_k": self.data.iloc[idx]["stoch_k"],
                    "stoch_d": self.data.iloc[idx]["stoch_d"],
                    "ema_distance": self.data.iloc[idx]["ema_distance"],
                    "ema_touch_long": self.data.iloc[idx]["ema_touch_long"],
                    "ema_touch_short": self.data.iloc[idx]["ema_touch_short"],
                }
            )

            # Trading logic
            if not self.open_positions:
                # Long Entry
                if (
                    self.data.iloc[idx]["ema_touch_long"]
                    and self.data.iloc[idx]["stoch_k"] < self.params["stoch_oversold"]
                ):
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
                    # trade_logger.info(
                    #     f"BUY SIGNAL (Enter Long) | Time: {bar_time_ist} | "
                    #     f"Price: {self.data.iloc[idx]['close']:.2f} | "
                    #     f"Stoch_K: {self.data.iloc[idx]['stoch_k']:.2f} < {self.params['stoch_oversold']}"
                    # )
                # Short Entry
                elif (
                    self.data.iloc[idx]["ema_touch_short"]
                    and self.data.iloc[idx]["stoch_k"] > self.params["stoch_overbought"]
                ):
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
                    # trade_logger.info(
                    #     f"SELL SIGNAL (Enter Short) | Time: {bar_time_ist} | "
                    #     f"Price: {self.data.iloc[idx]['close']:.2f} | "
                    #     f"Stoch_K: {self.data.iloc[idx]['stoch_k']:.2f} > {self.params['stoch_overbought']}"
                    # )
            else:
                if self.open_positions[-1]["direction"] == "long":
                    # Long Exit
                    if (
                        self.data.iloc[idx]["stoch_k"] > self.params["stoch_overbought"]
                        or self.data.iloc[idx]["close"] < self.data.iloc[idx]["ema"]
                    ):
                        reason = (
                            "Stoch overbought"
                            if self.data.iloc[idx]["stoch_k"]
                            > self.params["stoch_overbought"]
                            else "Price below EMA"
                        )
                        self._close_position(idx, reason, "sell", "exit_long")
                        self.last_signal = None
                elif self.open_positions[-1]["direction"] == "short":
                    # Short Exit
                    if (
                        self.data.iloc[idx]["stoch_k"] < self.params["stoch_oversold"]
                        or self.data.iloc[idx]["close"] > self.data.iloc[idx]["ema"]
                    ):
                        reason = (
                            "Stoch oversold"
                            if self.data.iloc[idx]["stoch_k"]
                            < self.params["stoch_oversold"]
                            else "Price above EMA"
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
            # trade_logger.info(
            #     f"BUY EXECUTED (Enter Long) | Ref: {order['ref']} | Price: {order['executed_price']:.2f}"
            # )
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
            # trade_logger.info(
            #     f"SELL EXECUTED (Enter Short) | Ref: {order['ref']} | Price: {order['executed_price']:.2f}"
            # )

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
            # trade_logger.info(
            #     f"SELL EXECUTED (Exit Long) | Ref: {order['ref']} | PnL: {pnl:.2f} | Reason: {reason}"
            # )
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
            # trade_logger.info(
            #     f"BUY EXECUTED (Exit Short) | Ref: {order['ref']} | PnL: {pnl:.2f} | Reason: {reason}"
            # )

        self.order = None
        self.order_type = None

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        params = {
            "ema_period": trial.suggest_int("ema_period", 10, 50, step=5),
            "stoch_k_period": trial.suggest_int("stoch_k_period", 10, 20, step=1),
            "stoch_d_period": trial.suggest_int("stoch_d_period", 3, 5, step=1),
            "stoch_slowing": trial.suggest_int("stoch_slowing", 1, 5, step=1),
            "stoch_oversold": trial.suggest_int("stoch_oversold", 10, 30, step=5),
            "stoch_overbought": trial.suggest_int("stoch_overbought", 70, 90, step=5),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            ema_period = params.get("ema_period", 20)
            stoch_k_period = params.get("stoch_k_period", 14)
            stoch_d_period = params.get("stoch_d_period", 3)
            return max(ema_period, stoch_k_period + stoch_d_period + 2)
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
