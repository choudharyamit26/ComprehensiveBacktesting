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


class BBSupertrendStrategy:
    """
    Bollinger Bands + Supertrend Strategy
    (Rewritten for consistency)
    """

    params = {
        "bb_period": 20,
        "bb_dev": 2.0,
        "supertrend_period": 10,
        "supertrend_mult": 3.0,
        "verbose": False,
    }

    optimization_params = {
        "bb_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "bb_dev": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.1},
        "supertrend_period": {"type": "int", "low": 7, "high": 14, "step": 1},
        "supertrend_mult": {"type": "float", "low": 2.0, "high": 4.0, "step": 0.5},
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
            max(self.params["bb_period"], self.params["supertrend_period"]) + 5
        )
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

        supertrend = ta.supertrend(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            length=self.params["supertrend_period"],
            multiplier=self.params["supertrend_mult"],
        )
        self.data["supertrend"] = supertrend[
            f"SUPERT_{self.params['supertrend_period']}_{self.params['supertrend_mult']}"
        ]

        logger.debug(f"Initialized BBSupertrendStrategy with params: {self.params}")

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
            if pd.isna(current_row["bb_mid"]) or pd.isna(current_row["supertrend"]):
                continue

            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": current_row["close"],
                    "bb_top": current_row["bb_top"],
                    "bb_mid": current_row["bb_mid"],
                    "bb_bot": current_row["bb_bot"],
                    "supertrend": current_row["supertrend"],
                }
            )

            bullish_entry = (current_row["close"] > current_row["bb_top"]) and (
                current_row["close"] > current_row["supertrend"]
            )
            bearish_entry = (current_row["close"] < current_row["bb_bot"]) and (
                current_row["close"] < current_row["supertrend"]
            )
            bullish_exit = (current_row["close"] < current_row["supertrend"]) or (
                current_row["close"] <= current_row["bb_mid"]
            )
            bearish_exit = (current_row["close"] > current_row["supertrend"]) or (
                current_row["close"] >= current_row["bb_mid"]
            )

            if not self.open_positions:
                if bullish_entry:
                    self._place_order(idx, "buy", "enter_long")
                    self.last_signal = "buy"
                    trade_logger.info(
                        f"BUY SIGNAL (BB Breakout + Supertrend Bullish) | Time: {bar_time_ist} | Price: {current_row['close']:.2f}"
                    )
                elif bearish_entry:
                    self._place_order(idx, "sell", "enter_short")
                    self.last_signal = "sell"
                    trade_logger.info(
                        f"SELL SIGNAL (BB Breakout + Supertrend Bearish) | Time: {bar_time_ist} | Price: {current_row['close']:.2f}"
                    )
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
            trade_logger.info(
                f"BUY EXECUTED (Enter Long) | Ref: {order['ref']} | Price: {order['executed_price']:.2f}"
            )
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
            trade_logger.info(
                f"SELL EXECUTED (Enter Short) | Ref: {order['ref']} | Price: {order['executed_price']:.2f}"
            )

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
        trade_logger.info(
            f"{action.upper()} EXECUTED (Exit {entry_info['direction'].capitalize()}) | PnL: {pnl:.2f} | Reason: {reason}"
        )

        self.order = None
        self.order_type = None

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        return {
            "bb_period": trial.suggest_int("bb_period", 10, 30, step=1),
            "bb_dev": trial.suggest_float("bb_dev", 1.5, 2.5, step=0.1),
            "supertrend_period": trial.suggest_int("supertrend_period", 7, 14, step=1),
            "supertrend_mult": trial.suggest_float(
                "supertrend_mult", 2.0, 4.0, step=0.5
            ),
        }

    @classmethod
    def get_min_data_points(cls, params):
        return max(params.get("bb_period", 20), params.get("supertrend_period", 10)) + 5
