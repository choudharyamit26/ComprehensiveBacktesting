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


class ATRVolumeExpansion:
    """
    ATR + Volume Expansion Trading Strategy
    (Rewritten to use pandas_ta)
    """

    params = {
        "atr_period": 14,
        "volume_period": 20,
        "expansion_factor": 1.1,
        "contraction_factor": 0.9,
        "volume_factor": COMMON_PARAMS["volume_surge_threshold"],  # Use common value
        "normal_factor": 1.2,
        "trailing_atr_mult": 2.0,
        "min_atr_threshold": 0.5,
        "verbose": False,
    }

    optimization_params = {
        "atr_period": {"type": "int", "low": 10, "high": 20, "step": 2},
        "volume_period": {"type": "int", "low": 15, "high": 30, "step": 5},
        "expansion_factor": {"type": "float", "low": 1.05, "high": 1.3, "step": 0.05},
        "contraction_factor": {"type": "float", "low": 0.8, "high": 0.95, "step": 0.05},
        "volume_factor": {"type": "float", "low": 1.2, "high": 2.0, "step": 0.1},
        "normal_factor": {"type": "float", "low": 1.0, "high": 1.5, "step": 0.1},
        "trailing_atr_mult": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.5},
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
            max(self.params["atr_period"], self.params["volume_period"], 10) + 5
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []
        self.entry_signals = []
        # Calculate indicators
        self.data["atr"] = ta.atr(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            length=self.params["atr_period"],
        )
        self.data["volume_sma"] = ta.sma(
            self.data["volume"], length=self.params["volume_period"]
        )
        self.data["high_sma"] = ta.sma(self.data["high"], length=10)
        self.data["low_sma"] = ta.sma(self.data["low"], length=10)
        self.data["trailing_stop_long"] = 0.0
        self.data["trailing_stop_short"] = 0.0

        logger.debug(f"Initialized ATRVolumeExpansion with params: {self.params}")

    def run(self):
        self.last_signal = None
        for idx in range(len(self.data)):
            if idx < self.warmup_period:
                continue

            if not self.ready:
                self.ready = True
                logger.info(f"Strategy ready at row {idx}")

            bar_time = pd.Timestamp(self.data.iloc[idx]["datetime"])
            if bar_time.tz is None:
                bar_time = bar_time.tz_localize(pytz.UTC)
            bar_time_ist = bar_time.tz_convert(pytz.timezone("Asia/Kolkata"))
            current_time = bar_time_ist.time()

            if current_time >= datetime.time(15, 15):
                if self.open_positions:
                    self._close_position(
                        idx,
                        "Force close at 15:15 IST",
                        (
                            "sell"
                            if self.open_positions[-1]["direction"] == "long"
                            else "buy"
                        ),
                        (
                            "exit_long"
                            if self.open_positions[-1]["direction"] == "long"
                            else "exit_short"
                        ),
                    )
                    self.last_signal = None
                continue

            if not (datetime.time(9, 15) <= current_time <= datetime.time(15, 5)):
                continue

            if self.order:
                logger.debug(f"Order pending at row {idx}")
                continue

            current_row = self.data.iloc[idx]
            prev_row = self.data.iloc[idx - 1]

            if (
                pd.isna(current_row["atr"])
                or pd.isna(current_row["volume_sma"])
                or current_row["atr"] < self.params["min_atr_threshold"]
            ):
                continue

            atr_expanding = current_row["atr"] > (
                prev_row["atr"] * self.params["expansion_factor"]
            )
            atr_contracting = current_row["atr"] < (
                prev_row["atr"] * self.params["contraction_factor"]
            )
            volume_surge = current_row["volume"] > (
                current_row["volume_sma"] * self.params["volume_factor"]
            )
            volume_normalizing = current_row["volume"] < (
                current_row["volume_sma"] * self.params["normal_factor"]
            )
            resistance_level = current_row["high_sma"]
            support_level = current_row["low_sma"]

            if self.open_positions:
                if self.open_positions[-1]["direction"] == "long":
                    new_stop = current_row["close"] - (
                        current_row["atr"] * self.params["trailing_atr_mult"]
                    )
                    self.data.loc[self.data.index[idx], "trailing_stop_long"] = max(
                        self.data.iloc[idx - 1]["trailing_stop_long"], new_stop
                    )
                else:
                    new_stop = current_row["close"] + (
                        current_row["atr"] * self.params["trailing_atr_mult"]
                    )
                    self.data.loc[self.data.index[idx], "trailing_stop_short"] = (
                        min(self.data.iloc[idx - 1]["trailing_stop_short"], new_stop)
                        if self.data.iloc[idx - 1]["trailing_stop_short"] != 0
                        else new_stop
                    )

            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": current_row["close"],
                    "atr": current_row["atr"],
                    "volume": current_row["volume"],
                    "volume_sma": current_row["volume_sma"],
                    "atr_expanding": atr_expanding,
                    "volume_surge": volume_surge,
                    "resistance": resistance_level,
                    "support": support_level,
                }
            )

            if not self.open_positions:
                if (
                    atr_expanding
                    and volume_surge
                    and current_row["close"] > resistance_level
                ):
                    self._place_order(idx, "buy", "enter_long")
                    self.last_signal = "buy"
                    self.data.loc[self.data.index[idx], "trailing_stop_long"] = (
                        current_row["close"]
                        - (current_row["atr"] * self.params["trailing_atr_mult"])
                    )
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "BUY"}
                    )
                    # trade_logger.info(
                    #     f"BUY SIGNAL (Enter Long - ATR+Volume Expansion) | Time: {bar_time_ist} | Price: {current_row['close']:.2f}"
                    # )
                elif (
                    atr_expanding
                    and volume_surge
                    and current_row["close"] < support_level
                ):
                    self._place_order(idx, "sell", "enter_short")
                    self.last_signal = "sell"
                    self.data.loc[self.data.index[idx], "trailing_stop_short"] = (
                        current_row["close"]
                        + (current_row["atr"] * self.params["trailing_atr_mult"])
                    )
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "SELL"}
                    )
                    # trade_logger.info(
                    #     f"SELL SIGNAL (Enter Short - ATR+Volume Expansion) | Time: {bar_time_ist} | Price: {current_row['close']:.2f}"
                    # )
            else:
                if self.open_positions[-1]["direction"] == "long":
                    if (
                        atr_contracting
                        or volume_normalizing
                        or current_row["close"]
                        <= self.data.iloc[idx]["trailing_stop_long"]
                    ):
                        reason = (
                            "ATR contracting"
                            if atr_contracting
                            else (
                                "Volume normalizing"
                                if volume_normalizing
                                else "Trailing stop hit"
                            )
                        )
                        self._close_position(idx, reason, "sell", "exit_long")
                        self.last_signal = None
                elif self.open_positions[-1]["direction"] == "short":
                    if (
                        atr_contracting
                        or volume_normalizing
                        or current_row["close"]
                        >= self.data.iloc[idx]["trailing_stop_short"]
                    ):
                        reason = (
                            "ATR contracting"
                            if atr_contracting
                            else (
                                "Volume normalizing"
                                if volume_normalizing
                                else "Trailing stop hit"
                            )
                        )
                        self._close_position(idx, reason, "buy", "exit_short")
                        self.last_signal = None
        return self.last_signal

    def _place_order(self, idx, action, order_type, size=100, commission_rate=0.001):
        price = self.data.iloc[idx]["close"]
        commission = abs(price * size * commission_rate)
        self.order = {
            "ref": str(uuid4()),
            "action": action,
            "order_type": order_type,
            "status": "Completed",
            "executed_price": price,
            "size": size if action == "buy" else -size,
            "commission": commission,
            "executed_time": self.data.iloc[idx]["datetime"],
        }
        self._notify_order(idx)

    def _notify_order(self, idx):
        order = self.order
        exec_dt = order["executed_time"]
        if exec_dt.tzinfo is None:
            exec_dt = exec_dt.replace(tzinfo=pytz.UTC)

        if order["order_type"] == "enter_long":
            self.open_positions.append(
                {
                    "entry_time": exec_dt,
                    "entry_price": order["executed_price"],
                    "size": order["size"],
                    "commission": order["commission"],
                    "ref": order["ref"],
                    "direction": "long",
                }
            )
            # trade_logger.info(
            # # f"BUY EXECUTED (Enter Long) | Ref: {order['ref']} | Price: {order['executed_price']:.2f}"
            # )
        elif order["order_type"] == "enter_short":
            self.open_positions.append(
                {
                    "entry_time": exec_dt,
                    "entry_price": order["executed_price"],
                    "size": order["size"],
                    "commission": order["commission"],
                    "ref": order["ref"],
                    "direction": "short",
                }
            )
            # trade_logger.info(
            # # f"SELL EXECUTED (Enter Short) | Ref: {order['ref']} | Price: {order['executed_price']:.2f}"
            # )

        self.order = None

    def _close_position(self, idx, reason, action, order_type):
        if not self.open_positions:
            return

        entry_info = self.open_positions.pop(0)
        price = self.data.iloc[idx]["close"]
        commission = abs(price * abs(entry_info["size"]) * 0.001)

        pnl = (
            (price - entry_info["entry_price"]) * entry_info["size"]
            if entry_info["direction"] == "long"
            else (entry_info["entry_price"] - price) * abs(entry_info["size"])
        )
        total_commission = entry_info["commission"] + commission

        trade_info = {
            "ref": str(uuid4()),
            "entry_time": entry_info["entry_time"],
            "exit_time": self.data.iloc[idx]["datetime"],
            "entry_price": entry_info["entry_price"],
            "exit_price": price,
            "size": abs(entry_info["size"]),
            "pnl": pnl,
            "pnl_net": pnl - total_commission,
            "commission": total_commission,
            "status": "Won" if pnl > 0 else "Lost",
            "direction": entry_info["direction"].capitalize(),
            "bars_held": (
                self.data.iloc[idx]["datetime"] - entry_info["entry_time"]
            ).total_seconds()
            / 60,
        }
        self.completed_trades.append(trade_info)
        self.trade_count += 1
        # trade_logger.info(
        #     f"{action.upper()} EXECUTED (Exit {entry_info['direction'].capitalize()}) | PnL: {pnl:.2f} | Reason: {reason}"
        # )

        self.order = None
        self.order_type = None

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        return {
            "atr_period": trial.suggest_int("atr_period", 10, 20, step=2),
            "volume_period": trial.suggest_int("volume_period", 15, 30, step=5),
            "expansion_factor": trial.suggest_float(
                "expansion_factor", 1.05, 1.3, step=0.05
            ),
            "contraction_factor": trial.suggest_float(
                "contraction_factor", 0.8, 0.95, step=0.05
            ),
            "volume_factor": trial.suggest_float("volume_factor", 1.2, 2.0, step=0.1),
            "normal_factor": trial.suggest_float("normal_factor", 1.0, 1.5, step=0.1),
            "trailing_atr_mult": trial.suggest_float(
                "trailing_atr_mult", 1.5, 3.0, step=0.5
            ),
        }

    @classmethod
    def get_min_data_points(cls, params):
        return (
            max(params.get("atr_period", 14), params.get("volume_period", 20), 10) + 5
        )
