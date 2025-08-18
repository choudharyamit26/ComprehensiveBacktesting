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


class MACDVolume:
    """
    MACD + Volume Confirmation Strategy (pandas_ta implementation)

    This version conforms to the same structure as sr_rsi.py:
    - params and optimization_params
    - __init__(data, **kwargs) computing indicators with pandas_ta
    - run() with market hour checks and position management
    - _notify_order() and _close_position() to manage positions and trade logs
    - get_param_space() and get_min_data_points()
    """

    params = {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "volume_period": 20,
        "volume_spike": COMMON_PARAMS["volume_surge_threshold"],  # Use common value
        "volume_dry": 1.0,
        "verbose": False,
    }

    optimization_params = {
        "fast_period": {"type": "int", "low": 8, "high": 16, "step": 1},
        "slow_period": {"type": "int", "low": 20, "high": 35, "step": 1},
        "signal_period": {"type": "int", "low": 6, "high": 12, "step": 1},
        "volume_period": {"type": "int", "low": 10, "high": 50, "step": 1},
        "volume_spike": {"type": "float", "low": 1.2, "high": 2.0, "step": 0.05},
        "volume_dry": {"type": "float", "low": 0.5, "high": 1.0, "step": 0.05},
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
                self.params["slow_period"] + self.params["signal_period"],
                self.params["volume_period"],
            )
            + 5
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []
        self.entry_signals = []
        # Indicators: MACD line, signal line, histogram
        macd = ta.macd(
            self.data["close"],
            fast=self.params["fast_period"],
            slow=self.params["slow_period"],
            signal=self.params["signal_period"],
        )
        self.data["macd_line"] = macd[
            f"MACD_{self.params['fast_period']}_{self.params['slow_period']}_{self.params['signal_period']}"
        ]
        self.data["macd_signal"] = macd[
            f"MACDs_{self.params['fast_period']}_{self.params['slow_period']}_{self.params['signal_period']}"
        ]
        self.data["macd_hist"] = macd[
            f"MACDh_{self.params['fast_period']}_{self.params['slow_period']}_{self.params['signal_period']}"
        ]

        # Volume indicators
        self.data["volume_sma"] = ta.sma(
            self.data["volume"], length=self.params["volume_period"]
        )
        self.data["vol_ratio"] = self.data["volume"] / self.data["volume_sma"]

        logger.debug(f"Initialized MACDVolume with params: {self.params}")

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

            # Validate indicator values
            if (
                pd.isna(self.data.iloc[idx]["macd_line"])
                or pd.isna(self.data.iloc[idx]["macd_signal"])
                or pd.isna(self.data.iloc[idx]["volume_sma"])
                or pd.isna(self.data.iloc[idx - 1]["macd_line"])
                or pd.isna(self.data.iloc[idx - 1]["macd_signal"])
                or self.data.iloc[idx]["volume_sma"] == 0
            ):
                continue

            current = self.data.iloc[idx]
            prev = self.data.iloc[idx - 1]

            # Signals
            cross_up = (current["macd_line"] > current["macd_signal"]) and (
                prev["macd_line"] <= prev["macd_signal"]
            )
            cross_down = (current["macd_line"] < current["macd_signal"]) and (
                prev["macd_line"] >= prev["macd_signal"]
            )
            vol_spike = current["vol_ratio"] > self.params["volume_spike"]
            volume_dry = current["vol_ratio"] < self.params["volume_dry"]

            # Store indicator data
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": current["close"],
                    "macd": current["macd_line"],
                    "signal": current["macd_signal"],
                    "hist": current["macd_hist"],
                    "vol_ratio": current["vol_ratio"],
                    "cross_up": cross_up,
                    "cross_down": cross_down,
                }
            )

            if not self.open_positions:
                if cross_up and vol_spike:
                    self.order = {
                        "ref": str(uuid4()),
                        "action": "buy",
                        "order_type": "enter_long",
                        "status": "Completed",
                        "executed_price": current["close"],
                        "size": 100,
                        "commission": abs(current["close"] * 100 * 0.001),
                        "executed_time": self.data.iloc[idx]["datetime"],
                    }
                    self.last_signal = "buy"
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "BUY"}
                    )
                    self._notify_order(idx)
                    trade_logger.info(
                        f"BUY SIGNAL (MACD Bullish + Volume Spike) | Time: {bar_time_ist} | "
                        f"Price: {current['close']:.2f} | MACD: {current['macd_line']:.4f} > Signal: {current['macd_signal']:.4f} | "
                        f"VolRatio: {current['vol_ratio']:.2f} > {self.params['volume_spike']:.2f}"
                    )
                elif cross_down and vol_spike:
                    self.order = {
                        "ref": str(uuid4()),
                        "action": "sell",
                        "order_type": "enter_short",
                        "status": "Completed",
                        "executed_price": current["close"],
                        "size": -100,
                        "commission": abs(current["close"] * 100 * 0.001),
                        "executed_time": self.data.iloc[idx]["datetime"],
                    }
                    self.last_signal = "sell"
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "SELL"}
                    )
                    self._notify_order(idx)
                    trade_logger.info(
                        f"SELL SIGNAL (MACD Bearish + Volume Spike) | Time: {bar_time_ist} | "
                        f"Price: {current['close']:.2f} | MACD: {current['macd_line']:.4f} < Signal: {current['macd_signal']:.4f} | "
                        f"VolRatio: {current['vol_ratio']:.2f} > {self.params['volume_spike']:.2f}"
                    )
                else:
                    self.last_signal = None
            else:
                # Exit logic
                if self.open_positions[-1]["direction"] == "long":
                    if cross_down or volume_dry:
                        reason = "MACD Bearish Cross" if cross_down else "Volume Dry"
                        self._close_position(idx, reason, "sell", "exit_long")
                        self.last_signal = None
                        trade_logger.info(
                            f"EXIT LONG | Time: {bar_time_ist} | Price: {current['close']:.2f} | Reason: {reason}"
                        )
                elif self.open_positions[-1]["direction"] == "short":
                    if cross_up or volume_dry:
                        reason = "MACD Bullish Cross" if cross_up else "Volume Dry"
                        self._close_position(idx, reason, "buy", "exit_short")
                        self.last_signal = None
                        trade_logger.info(
                            f"EXIT SHORT | Time: {bar_time_ist} | Price: {current['close']:.2f} | Reason: {reason}"
                        )

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
            "fast_period": trial.suggest_int("fast_period", 8, 16),
            "slow_period": trial.suggest_int("slow_period", 20, 35),
            "signal_period": trial.suggest_int("signal_period", 6, 12),
            "volume_period": trial.suggest_int("volume_period", 10, 50),
            "volume_spike": trial.suggest_float("volume_spike", 1.2, 2.0),
            "volume_dry": trial.suggest_float("volume_dry", 0.5, 1.0),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            slow_period = params.get("slow_period", 26)
            signal_period = params.get("signal_period", 9)
            volume_period = params.get("volume_period", 20)
            max_period = max(slow_period + signal_period, volume_period)
            return max_period + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 65
