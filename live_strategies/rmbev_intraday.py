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


class RMBEV:
    """
    RSI + MACD + Bollinger Bands + EMA + Volume (RMBEV) Strategy
    (Rewritten for consistency)
    """

    params = {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_stddev": COMMON_PARAMS["bb_stddev"],  # Use common value
        "ema_period": 20,
        "vol_sma_period": 14,
        "vol_threshold": COMMON_PARAMS["volume_surge_threshold"],  # Use common value
        "verbose": False,
    }

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "macd_fast": {"type": "int", "low": 8, "high": 16, "step": 1},
        "macd_slow": {"type": "int", "low": 20, "high": 30, "step": 1},
        "macd_signal": {"type": "int", "low": 7, "high": 12, "step": 1},
        "bb_period": {"type": "int", "low": 15, "high": 25, "step": 1},
        "bb_stddev": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.1},
        "ema_period": {"type": "int", "low": 15, "high": 30, "step": 1},
        "vol_sma_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "vol_threshold": {"type": "float", "low": 1.2, "high": 2.0, "step": 0.1},
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
                self.params["macd_slow"],
                self.params["bb_period"],
                self.params["ema_period"],
                self.params["vol_sma_period"],
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []
        self.entry_signals = []
        # Calculate indicators
        self.data["rsi"] = ta.rsi(self.data["close"], length=self.params["rsi_period"])
        macd = ta.macd(
            self.data["close"],
            fast=self.params["macd_fast"],
            slow=self.params["macd_slow"],
            signal=self.params["macd_signal"],
        )
        self.data["macd"] = macd[
            f"MACD_{self.params['macd_fast']}_{self.params['macd_slow']}_{self.params['macd_signal']}"
        ]
        self.data["macd_signal"] = macd[
            f"MACDs_{self.params['macd_fast']}_{self.params['macd_slow']}_{self.params['macd_signal']}"
        ]
        bbands = ta.bbands(
            self.data["close"],
            length=self.params["bb_period"],
            std=self.params["bb_stddev"],
        )
        self.data["bb_upper"] = bbands[
            f"BBU_{self.params['bb_period']}_{self.params['bb_stddev']}"
        ]
        self.data["bb_middle"] = bbands[
            f"BBM_{self.params['bb_period']}_{self.params['bb_stddev']}"
        ]
        self.data["bb_lower"] = bbands[
            f"BBL_{self.params['bb_period']}_{self.params['bb_stddev']}"
        ]
        self.data["ema"] = ta.ema(self.data["close"], length=self.params["ema_period"])
        self.data["vol_sma"] = ta.sma(
            self.data["volume"], length=self.params["vol_sma_period"]
        )
        self.data["vol_ratio"] = self.data["volume"] / self.data["vol_sma"]
        self.data["bb_upper_touch"] = self.data["close"] >= self.data["bb_upper"]
        self.data["bb_lower_touch"] = self.data["close"] <= self.data["bb_lower"]

        logger.debug(f"Initialized RMBEV with params: {self.params}")

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
            if (
                pd.isna(current_row["rsi"])
                or pd.isna(current_row["macd"])
                or pd.isna(current_row["bb_upper"])
                or pd.isna(current_row["ema"])
                or pd.isna(current_row["vol_ratio"])
            ):
                continue

            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": current_row["close"],
                    "rsi": current_row["rsi"],
                    "macd": current_row["macd"],
                    "signal": current_row["macd_signal"],
                    "bb_upper": current_row["bb_upper"],
                    "bb_middle": current_row["bb_middle"],
                    "bb_lower": current_row["bb_lower"],
                    "ema": current_row["ema"],
                    "vol_ratio": current_row["vol_ratio"],
                }
            )

            prev_row = self.data.iloc[idx - 1]
            rsi_rising = (
                current_row["rsi"] > prev_row["rsi"] and 30 < current_row["rsi"] < 70
            )
            rsi_falling = (
                current_row["rsi"] < prev_row["rsi"] and 30 < current_row["rsi"] < 70
            )
            macd_bullish = (
                current_row["macd"] > current_row["macd_signal"]
                and prev_row["macd"] <= prev_row["macd_signal"]
            )
            macd_bearish = (
                current_row["macd"] < current_row["macd_signal"]
                and prev_row["macd"] >= prev_row["macd_signal"]
            )
            price_above_ema = current_row["close"] > current_row["ema"]
            price_below_ema = current_row["close"] < current_row["ema"]
            high_volume = current_row["vol_ratio"] > self.params["vol_threshold"]

            if not self.open_positions:
                if (
                    rsi_rising
                    and macd_bullish
                    and price_above_ema
                    and current_row["bb_upper_touch"]
                    and high_volume
                ):
                    self._place_order(idx, "buy", "enter_long")
                    self.last_signal = "buy"
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "BUY"}
                    )
                    # trade_logger.info(
                    #     f"BUY SIGNAL (Enter Long - RMBEV) | Time: {bar_time_ist} | Price: {current_row['close']:.2f}"
                    # )
                elif (
                    rsi_falling
                    and macd_bearish
                    and price_below_ema
                    and current_row["bb_lower_touch"]
                    and high_volume
                ):
                    self._place_order(idx, "sell", "enter_short")
                    self.last_signal = "sell"
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "SELL"}
                    )
                    # trade_logger.info(
                    #     f"SELL SIGNAL (Enter Short - RMBEV) | Time: {bar_time_ist} | Price: {current_row['close']:.2f}"
                    # )
            else:
                if self.open_positions[-1]["direction"] == "long":
                    rsi_reversal = current_row["rsi"] < prev_row["rsi"]
                    macd_reversal = current_row["macd"] < current_row["macd_signal"]
                    ema_reversal = current_row["close"] < current_row["ema"]
                    bb_reversal = current_row["close"] <= current_row["bb_middle"]
                    vol_decrease = current_row["vol_ratio"] < 1.0
                    reversal_count = sum(
                        [
                            rsi_reversal,
                            macd_reversal,
                            ema_reversal,
                            bb_reversal,
                            vol_decrease,
                        ]
                    )
                    if reversal_count >= 3:
                        self._close_position(
                            idx,
                            f"{reversal_count} indicators reversed",
                            "sell",
                            "exit_long",
                        )
                        self.last_signal = None
                elif self.open_positions[-1]["direction"] == "short":
                    rsi_reversal = current_row["rsi"] > prev_row["rsi"]
                    macd_reversal = current_row["macd"] > current_row["macd_signal"]
                    ema_reversal = current_row["close"] > current_row["ema"]
                    bb_reversal = current_row["close"] >= current_row["bb_middle"]
                    vol_decrease = current_row["vol_ratio"] < 1.0
                    reversal_count = sum(
                        [
                            rsi_reversal,
                            macd_reversal,
                            ema_reversal,
                            bb_reversal,
                            vol_decrease,
                        ]
                    )
                    if reversal_count >= 3:
                        self._close_position(
                            idx,
                            f"{reversal_count} indicators reversed",
                            "buy",
                            "exit_short",
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
            #     f" BUY EXECUTED (Enter Long) | Ref: {order['ref']} | Price: {order['executed_price']:.2f}"
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
            #     f"# SELL EXECUTED (Enter Short) | Ref: {order['ref']} | Price: {order['executed_price']:.2f}"
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
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "macd_fast": trial.suggest_int("macd_fast", 8, 16),
            "macd_slow": trial.suggest_int("macd_slow", 20, 30),
            "macd_signal": trial.suggest_int("macd_signal", 7, 12),
            "bb_period": trial.suggest_int("bb_period", 15, 25),
            "bb_stddev": trial.suggest_float("bb_stddev", 1.5, 2.5, step=0.1),
            "ema_period": trial.suggest_int("ema_period", 15, 30),
            "vol_sma_period": trial.suggest_int("vol_sma_period", 10, 20),
            "vol_threshold": trial.suggest_float("vol_threshold", 1.2, 2.0, step=0.1),
        }

    @classmethod
    def get_min_data_points(cls, params):
        return (
            max(
                params.get("rsi_period", 14),
                params.get("macd_slow", 26),
                params.get("bb_period", 20),
                params.get("ema_period", 20),
                params.get("vol_sma_period", 14),
            )
            + 2
        )
