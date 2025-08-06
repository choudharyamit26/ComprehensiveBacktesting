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


class HeadShouldersConfirmation:
    """
    Head & Shoulders + Confirmation Strategy
    (Documentation remains unchanged)
    """

    params = {
        "rsi_period": 14,
        "volume_sma_period": 20,
        "lookback": 20,
        "target_multiplier": 1.0,
        "verbose": False,
    }

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "volume_sma_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "lookback": {"type": "int", "low": 15, "high": 30, "step": 1},
        "target_multiplier": {"type": "float", "low": 0.8, "high": 1.2, "step": 0.1},
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
                self.params["volume_sma_period"],
                self.params["lookback"],
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []
        self.target_price = None

        # Calculate indicators
        self.data["rsi"] = ta.rsi(self.data["close"], length=self.params["rsi_period"])
        self.data["volume_sma"] = ta.sma(
            self.data["volume"], length=self.params["volume_sma_period"]
        )
        self.data["volume_surge"] = self.data["volume"] > self.data["volume_sma"] * 1.5

        logger.debug(
            f"Initialized HeadShouldersConfirmation with params: {self.params}"
        )

    def detect_head_shoulders(self, idx):
        lookback = self.params["lookback"]
        if idx < lookback:
            return None, None

        highs = self.data["high"].iloc[idx - lookback : idx]
        lows = self.data["low"].iloc[idx - lookback : idx]
        closes = self.data["close"].iloc[idx - lookback : idx]

        head = highs.max()
        head_idx = highs.idxmax()

        if head_idx < 2 or head_idx > len(highs) - 3:
            return None, None

        left_shoulder = highs.iloc[:head_idx][-3:].max()
        right_shoulder = highs.iloc[head_idx + 1 :][:3].max()

        if left_shoulder >= head or right_shoulder >= head:
            return None, None

        neckline_lows = lows.iloc[head_idx - 2 : head_idx + 3]
        neckline = neckline_lows.mean()

        current_close = self.data["close"].iloc[idx]
        direction = (
            "bullish"
            if current_close > neckline
            else "bearish" if current_close < neckline else None
        )

        if direction == "bullish":
            target = neckline + (head - neckline) * self.params["target_multiplier"]
        elif direction == "bearish":
            target = neckline - (head - neckline) * self.params["target_multiplier"]
        else:
            target = None

        return direction, target

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
                self.data.iloc[idx]["volume_sma"]
            ):
                continue

            # Store indicator data for analysis
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": self.data.iloc[idx]["close"],
                    "rsi": self.data.iloc[idx]["rsi"],
                    "volume": self.data.iloc[idx]["volume"],
                    "volume_sma": self.data.iloc[idx]["volume_sma"],
                }
            )

            # Detect Head & Shoulders pattern
            direction, target = self.detect_head_shoulders(idx)
            volume_surge = self.data.iloc[idx]["volume_surge"]

            # Trading logic
            if not self.open_positions:
                if (
                    direction == "bullish"
                    and self.data.iloc[idx]["rsi"] > 30
                    and volume_surge
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
                    self.target_price = target
                    self.last_signal = "buy"
                    self._notify_order(idx)
                    trade_logger.info(
                        f"BUY SIGNAL (Head & Shoulders) | Time: {bar_time_ist} | "
                        f"Price: {self.data.iloc[idx]['close']:.2f} | "
                        f"Target: {target:.2f}"
                    )
                elif (
                    direction == "bearish"
                    and self.data.iloc[idx]["rsi"] < 70
                    and volume_surge
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
                    self.target_price = target
                    self.last_signal = "sell"
                    self._notify_order(idx)
                    trade_logger.info(
                        f"SELL SIGNAL (Head & Shoulders) | Time: {bar_time_ist} | "
                        f"Price: {self.data.iloc[idx]['close']:.2f} | "
                        f"Target: {target:.2f}"
                    )
            else:
                if self.open_positions[-1]["direction"] == "long":
                    # Long Exit
                    if (
                        self.data.iloc[idx]["close"] >= self.target_price
                        or self.data.iloc[idx]["rsi"] < 30
                    ):
                        reason = (
                            "Target reached"
                            if self.data.iloc[idx]["close"] >= self.target_price
                            else "RSI < 30"
                        )
                        self._close_position(idx, reason, "sell", "exit_long")
                        self.last_signal = None
                elif self.open_positions[-1]["direction"] == "short":
                    # Short Exit
                    if (
                        self.data.iloc[idx]["close"] <= self.target_price
                        or self.data.iloc[idx]["rsi"] > 70
                    ):
                        reason = (
                            "Target reached"
                            if self.data.iloc[idx]["close"] <= self.target_price
                            else "RSI > 70"
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
            "volume_sma_period": trial.suggest_int("volume_sma_period", 10, 30),
            "lookback": trial.suggest_int("lookback", 15, 30),
            "target_multiplier": trial.suggest_float(
                "target_multiplier", 0.8, 1.2, step=0.1
            ),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            volume_sma_period = params.get("volume_sma_period", 20)
            lookback = params.get("lookback", 20)
            return max(rsi_period, volume_sma_period, lookback) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
