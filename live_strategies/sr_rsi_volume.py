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


class SupportResistance:
    def __init__(self, period=20, min_touches=2, tolerance=0.002):
        self.period = period
        self.min_touches = min_touches
        self.tolerance = tolerance
        self.support_levels = []
        self.resistance_levels = []

    def update(self, highs, lows, closes):
        # Look for pivot points
        highs = highs[-self.period :]
        lows = lows[-self.period :]

        # Find local maxima and minima
        current_high = highs.max()
        current_low = lows.min()

        # Update support and resistance levels
        if not any(
            abs(current_high - level) / level < self.tolerance
            for level in self.resistance_levels
        ):
            self.resistance_levels.append(current_high)
        if not any(
            abs(current_low - level) / level < self.tolerance
            for level in self.support_levels
        ):
            self.support_levels.append(current_low)

        # Keep only recent levels
        self.resistance_levels = self.resistance_levels[-50:]
        self.support_levels = self.support_levels[-50:]

        # Get nearest levels
        current_price = closes.iloc[-1]
        valid_supports = [
            level for level in self.support_levels if level < current_price
        ]
        valid_resistances = [
            level for level in self.resistance_levels if level > current_price
        ]

        nearest_support = max(valid_supports) if valid_supports else None
        nearest_resistance = min(valid_resistances) if valid_resistances else None

        at_support = (
            nearest_support
            and abs(current_price - nearest_support) / nearest_support < self.tolerance
        )
        at_resistance = (
            nearest_resistance
            and abs(current_price - nearest_resistance) / nearest_resistance
            < self.tolerance
        )

        return nearest_support, nearest_resistance, at_support, at_resistance


class VolumeConfirmation:
    def __init__(self, period=20, surge_multiplier=1.5):
        self.period = period
        self.surge_multiplier = surge_multiplier

    def update(self, volumes):
        if len(volumes) < self.period:
            return 0, 0, False

        volume_avg = volumes[-self.period :].mean()
        volume_ratio = volumes.iloc[-1] / volume_avg if volume_avg > 0 else 1.0
        volume_surge = volume_ratio >= self.surge_multiplier
        return volume_avg, volume_ratio, volume_surge


class SRRSIVolume:
    """
    Support/Resistance + RSI + Volume Strategy
    (Documentation remains unchanged)
    """

    params = {
        "sr_period": 20,
        "sr_tolerance": 0.002,
        "rsi_oversold": COMMON_PARAMS["rsi_oversold"],  # Use common value
        "rsi_overbought": COMMON_PARAMS["rsi_overbought"],  # Use common value
        "rsi_overbought": 70,
        "volume_period": 20,
        "volume_surge": COMMON_PARAMS["volume_surge_threshold"],  # Use common value
        "verbose": False,
    }

    optimization_params = {
        "sr_period": {"type": "int", "low": 15, "high": 30, "step": 1},
        "sr_tolerance": {"type": "float", "low": 0.001, "high": 0.005, "step": 0.0005},
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "rsi_oversold": {"type": "int", "low": 20, "high": 35, "step": 1},
        "rsi_overbought": {"type": "int", "low": 65, "high": 80, "step": 1},
        "volume_period": {"type": "int", "low": 15, "high": 30, "step": 1},
        "volume_surge": {"type": "float", "low": 1.2, "high": 2.0, "step": 0.1},
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
                self.params["sr_period"] * 2,
                self.params["rsi_period"],
                self.params["volume_period"],
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        # Initialize indicators
        self.sr = SupportResistance(
            period=self.params["sr_period"], tolerance=self.params["sr_tolerance"]
        )
        self.volume_conf = VolumeConfirmation(
            period=self.params["volume_period"],
            surge_multiplier=self.params["volume_surge"],
        )

        # Calculate RSI
        self.data["rsi"] = ta.rsi(self.data["close"], length=self.params["rsi_period"])
        self.entry_signals = []
        logger.debug(f"Initialized SRRSIVolume with params: {self.params}")

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

            # Update indicators
            support, resistance, at_support, at_resistance = self.sr.update(
                self.data["high"].iloc[: idx + 1],
                self.data["low"].iloc[: idx + 1],
                self.data["close"].iloc[: idx + 1],
            )
            _, volume_ratio, volume_surge = self.volume_conf.update(
                self.data["volume"].iloc[: idx + 1]
            )

            # Check for invalid indicator values
            if (
                pd.isna(self.data.iloc[idx]["rsi"])
                or support is None
                or resistance is None
            ):
                continue

            # Store indicator data for analysis
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": self.data.iloc[idx]["close"],
                    "support": support,
                    "resistance": resistance,
                    "at_support": at_support,
                    "at_resistance": at_resistance,
                    "rsi": self.data.iloc[idx]["rsi"],
                    "volume": self.data.iloc[idx]["volume"],
                    "volume_ratio": volume_ratio,
                    "volume_surge": volume_surge,
                }
            )

            # Calculate RSI conditions
            rsi_oversold_recovery = (
                self.data.iloc[idx]["rsi"] > self.params["rsi_oversold"]
                and self.data.iloc[idx - 1]["rsi"] <= self.params["rsi_oversold"]
            )
            rsi_overbought_decline = (
                self.data.iloc[idx]["rsi"] < self.params["rsi_overbought"]
                and self.data.iloc[idx - 1]["rsi"] >= self.params["rsi_overbought"]
            )

            # Trading logic
            if not self.open_positions:
                # Long Entry
                if at_support and rsi_oversold_recovery and volume_surge:
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
                    # trade_logger.info(
                    #     f"BUY SIGNAL (S/R + RSI + Volume) | Time: {bar_time_ist} | "
                    #     f"Price: {self.data.iloc[idx]['close']:.2f} | "
                    #     f"Support: {support:.2f}"
                    # )
                # Short Entry
                elif at_resistance and rsi_overbought_decline and volume_surge:
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
                    #     f"SELL SIGNAL (S/R + RSI + Volume) | Time: {bar_time_ist} | "
                    #     f"Price: {self.data.iloc[idx]['close']:.2f} | "
                    #     f"Resistance: {resistance:.2f}"
                    # )
            else:
                if self.open_positions[-1]["direction"] == "long":
                    # Long Exit
                    if (
                        self.data.iloc[idx]["close"]
                        < support * (1 - self.params["sr_tolerance"])
                        or self.data.iloc[idx]["rsi"] >= self.params["rsi_overbought"]
                    ):
                        reason = (
                            "Support break"
                            if self.data.iloc[idx]["close"]
                            < support * (1 - self.params["sr_tolerance"])
                            else "RSI overbought"
                        )
                        self._close_position(idx, reason, "sell", "exit_long")
                        self.last_signal = None
                elif self.open_positions[-1]["direction"] == "short":
                    # Short Exit
                    if (
                        self.data.iloc[idx]["close"]
                        > resistance * (1 + self.params["sr_tolerance"])
                        or self.data.iloc[idx]["rsi"] <= self.params["rsi_oversold"]
                    ):
                        reason = (
                            "Resistance break"
                            if self.data.iloc[idx]["close"]
                            > resistance * (1 + self.params["sr_tolerance"])
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
            "sr_period": trial.suggest_int("sr_period", 15, 30),
            "sr_tolerance": trial.suggest_float(
                "sr_tolerance", 0.001, 0.005, step=0.0005
            ),
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 35),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 65, 80),
            "volume_period": trial.suggest_int("volume_period", 15, 30),
            "volume_surge": trial.suggest_float("volume_surge", 1.2, 2.0, step=0.1),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            sr_period = params.get("sr_period", 20)
            rsi_period = params.get("rsi_period", 14)
            volume_period = params.get("volume_period", 20)
            return max(sr_period * 2, rsi_period, volume_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 50
