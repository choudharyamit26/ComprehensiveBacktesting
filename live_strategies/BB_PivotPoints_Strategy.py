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


class BBPivotPointsStrategy:
    """
    Bollinger Bands + Pivot Points Strategy
    (Rewritten for consistency)
    """

    params = {
        "bb_period": 20,
        "bb_dev": COMMON_PARAMS["bb_stddev"],  # Standardized BB deviation
        "pivot_proximity": 0.5,
        "verbose": False,
    }

    optimization_params = {
        "bb_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "bb_dev": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.1},
        "pivot_proximity": {"type": "float", "low": 0.3, "high": 1.0, "step": 0.1},
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
        self.warmup_period = self.params["bb_period"] + 5
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        # Initialize indicators
        bb = ta.bbands(
            self.data["close"],
            length=self.params["bb_period"],
            std=self.params["bb_dev"],
        )
        self.data["bb_top"] = bb[
            f"BBU_{self.params['bb_period']}_{self.params['bb_dev']}"
        ]
        self.data["bb_mid"] = bb[
            f"BBM_{self.params['bb_period']}_{self.params['bb_dev']}"
        ]
        self.data["bb_bot"] = bb[
            f"BBL_{self.params['bb_period']}_{self.params['bb_dev']}"
        ]

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
        self.data["volume_sma"] = ta.sma(self.data["volume"], length=20)
        self.data["volume_surge"] = (
            self.data["volume"]
            > self.data["volume_sma"] * COMMON_PARAMS["volume_surge_threshold"]
        )
        self.entry_signals = []
        logger.debug(f"Initialized BBPivotPointsStrategy with params: {self.params}")

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
            if pd.isna(current_row["bb_mid"]) or pd.isna(current_row["pivot"]):
                continue

            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": current_row["close"],
                    "bb_top": current_row["bb_top"],
                    "bb_mid": current_row["bb_mid"],
                    "bb_bot": current_row["bb_bot"],
                    "pivot": current_row["pivot"],
                    "s1": current_row["s1"],
                    "s2": current_row["s2"],
                    "r1": current_row["r1"],
                    "r2": current_row["r2"],
                }
            )

            price_near_lower_bb = current_row["close"] <= current_row["bb_bot"] * (
                1 + self.params["pivot_proximity"] / 100
            )
            price_near_upper_bb = current_row["close"] >= current_row["bb_top"] * (
                1 - self.params["pivot_proximity"] / 100
            )
            near_s1 = (
                abs(current_row["close"] - current_row["s1"]) / current_row["s1"]
                < self.params["pivot_proximity"] / 100
                if pd.notna(current_row["s1"])
                else False
            )
            near_s2 = (
                abs(current_row["close"] - current_row["s2"]) / current_row["s2"]
                < self.params["pivot_proximity"] / 100
                if pd.notna(current_row["s2"])
                else False
            )
            near_r1 = (
                abs(current_row["close"] - current_row["r1"]) / current_row["r1"]
                < self.params["pivot_proximity"] / 100
                if pd.notna(current_row["r1"])
                else False
            )
            near_r2 = (
                abs(current_row["close"] - current_row["r2"]) / current_row["r2"]
                < self.params["pivot_proximity"] / 100
                if pd.notna(current_row["r2"])
                else False
            )
            bullish_entry = price_near_lower_bb and (near_s1 or near_s2)
            bearish_entry = price_near_upper_bb and (near_r1 or near_r2)
            bullish_exit = (current_row["close"] >= current_row["pivot"]) or (
                current_row["close"] >= current_row["bb_top"]
            )
            bearish_exit = (current_row["close"] <= current_row["pivot"]) or (
                current_row["close"] <= current_row["bb_bot"]
            )

            if not self.open_positions:
                if bullish_entry and self.data.iloc[idx]["volume_surge"]:
                    self._place_order(idx, "buy", "enter_long")
                    self.last_signal = "buy"
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "BUY"}
                    )
                    # trade_logger.info(
                    #     f"BUY SIGNAL | Time: {bar_time_ist} | Price: {current_row['close']:.2f}"
                    # )
                elif bearish_entry and self.data.iloc[idx]["volume_surge"]:
                    self._place_order(idx, "sell", "enter_short")
                    self.last_signal = "sell"
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "SELL"}
                    )
                    # trade_logger.info(
                    #     f"SELL SIGNAL | Time: {bar_time_ist} | Price: {current_row['close']:.2f}"
                    # )
            else:
                if self.open_positions[-1]["direction"] == "long" and bullish_exit:
                    self._close_position(
                        idx, "Bullish exit condition", "sell", "exit_long"
                    )
                    self.last_signal = None
                elif self.open_positions[-1]["direction"] == "short" and bearish_exit:
                    self._close_position(
                        idx, "Bearish exit condition", "buy", "exit_short"
                    )
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
            #     f"BUY EXECUTED (Enter Long) | Ref: {order['ref']} | Price: {order['executed_price']:.2f}"
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
            #     f"SELL EXECUTED (Enter Short) | Ref: {order['ref']} | Price: {order['executed_price']:.2f}"
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
            "bb_period": trial.suggest_int("bb_period", 10, 30, step=1),
            "bb_dev": trial.suggest_float("bb_dev", 1.5, 2.5, step=0.1),
            "pivot_proximity": trial.suggest_float(
                "pivot_proximity", 0.3, 1.0, step=0.1
            ),
        }

    @classmethod
    def get_min_data_points(cls, params):
        return params.get("bb_period", 20) + 5
