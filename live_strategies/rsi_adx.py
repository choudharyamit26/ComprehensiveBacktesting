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


class RSIADX:
    """
    RSI + ADX Strength Strategy
    (Documentation remains unchanged)
    """

    params = {
        "rsi_period": 14,
        "adx_period": 14,
        "rsi_bullish": 55,
        "rsi_bearish": 45,
        "adx_strength": 25,
        "verbose": False,
    }

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "adx_period": {"type": "int", "low": 10, "high": 25, "step": 1},
        "rsi_bullish": {"type": "int", "low": 50, "high": 70, "step": 1},
        "rsi_bearish": {"type": "int", "low": 30, "high": 50, "step": 1},
        "adx_strength": {"type": "int", "low": 20, "high": 40, "step": 2},
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
                self.params["adx_period"],
            )
            + 5
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        # Calculate indicators
        self.data["rsi"] = ta.rsi(self.data["close"], length=self.params["rsi_period"])
        adx = ta.adx(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            length=self.params["adx_period"],
        )
        self.data["adx"] = adx[f"ADX_{self.params['adx_period']}"]

        # Calculate signals
        self.data["rsi_bullish_signal"] = self.data["rsi"] > self.params["rsi_bullish"]
        self.data["rsi_bearish_signal"] = self.data["rsi"] < self.params["rsi_bearish"]
        self.data["strong_trend_signal"] = (
            self.data["adx"] > self.params["adx_strength"]
        )
        self.data["strong_bullish"] = (
            self.data["rsi_bullish_signal"] & self.data["strong_trend_signal"]
        )
        self.data["strong_bearish"] = (
            self.data["rsi_bearish_signal"] & self.data["strong_trend_signal"]
        )
        self.data["weak_trend"] = self.data["adx"] < self.params["adx_strength"]
        self.data["volume_sma"] = ta.sma(self.data["volume"], length=20)
        self.data["volume_surge"] = (
            self.data["volume"]
            > self.data["volume_sma"] * COMMON_PARAMS["volume_surge_threshold"]
        )
        self.entry_signals = []
        logger.debug(f"Initialized RSIADX with params: {self.params}")

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
                self.data.iloc[idx]["adx"]
            ):
                continue

            # Store indicator data for analysis
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": self.data.iloc[idx]["close"],
                    "rsi": self.data.iloc[idx]["rsi"],
                    "adx": self.data.iloc[idx]["adx"],
                    "rsi_bullish": self.data.iloc[idx]["rsi_bullish_signal"],
                    "rsi_bearish": self.data.iloc[idx]["rsi_bearish_signal"],
                    "strong_trend": self.data.iloc[idx]["strong_trend_signal"],
                    "strong_bullish": self.data.iloc[idx]["strong_bullish"],
                    "strong_bearish": self.data.iloc[idx]["strong_bearish"],
                }
            )

            # Trading logic
            if not self.open_positions:
                # Long Entry
                if (
                    self.data.iloc[idx]["strong_bullish"]
                    and self.data.iloc[idx]["volume_surge"]
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
                    self._notify_order(idx)
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "BUY"}
                    )
                    trade_logger.info(
                        f"BUY SIGNAL (Strong Bullish) | Time: {bar_time_ist} | "
                        f"Price: {self.data.iloc[idx]['close']:.2f} | "
                        f"RSI: {self.data.iloc[idx]['rsi']:.2f} | "
                        f"ADX: {self.data.iloc[idx]['adx']:.2f}"
                    )
                # Short Entry
                elif (
                    self.data.iloc[idx]["strong_bearish"]
                    and self.data.iloc[idx]["volume_surge"]
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
                    self._notify_order(idx)
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "SELL"}
                    )
                    trade_logger.info(
                        f"SELL SIGNAL (Strong Bearish) | Time: {bar_time_ist} | "
                        f"Price: {self.data.iloc[idx]['close']:.2f} | "
                        f"RSI: {self.data.iloc[idx]['rsi']:.2f} | "
                        f"ADX: {self.data.iloc[idx]['adx']:.2f}"
                    )
            else:
                if self.data.iloc[idx]["weak_trend"]:
                    if self.open_positions[-1]["direction"] == "long":
                        self._close_position(idx, "Weak trend", "sell", "exit_long")
                        self.last_signal = None
                    elif self.open_positions[-1]["direction"] == "short":
                        self._close_position(idx, "Weak trend", "buy", "exit_short")
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
            "adx_period": trial.suggest_int("adx_period", 10, 25),
            "rsi_bullish": trial.suggest_int("rsi_bullish", 50, 70),
            "rsi_bearish": trial.suggest_int("rsi_bearish", 30, 50),
            "adx_strength": trial.suggest_int("adx_strength", 20, 40),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            adx_period = params.get("adx_period", 14)
            return max(rsi_period, adx_period) + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 35
