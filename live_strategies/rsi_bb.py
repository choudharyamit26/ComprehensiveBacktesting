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


class RSIBB:
    """
    RSI and Bollinger Bands Combined Mean Reversion Trading Strategy
    (Documentation remains unchanged)
    """

    params = {
        "rsi_period": 14,
        "bb_period": 20,
        "bb_stddev": COMMON_PARAMS["bb_stddev"],  # Standardized BB deviation
        "rsi_oversold": COMMON_PARAMS["rsi_oversold"],
        "rsi_overbought": COMMON_PARAMS["rsi_overbought"],
        "rsi_exit": 50,
        "verbose": False,
    }

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "bb_period": {"type": "int", "low": 15, "high": 25, "step": 1},
        "bb_stddev": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.1},
        "rsi_oversold": {"type": "int", "low": 20, "high": 35, "step": 1},
        "rsi_overbought": {"type": "int", "low": 65, "high": 80, "step": 1},
        "rsi_exit": {"type": "int", "low": 45, "high": 55, "step": 1},
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
        self.warmup_period = (
            max(self.params["rsi_period"], self.params["bb_period"]) + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        # Initialize indicators using pandas_ta
        self.data["rsi"] = ta.rsi(self.data["close"], length=self.params["rsi_period"])
        bb = ta.bbands(
            self.data["close"],
            length=self.params["bb_period"],
            std=self.params["bb_stddev"],
        )
        self.data["bb_top"] = bb[
            f"BBU_{self.params['bb_period']}_{self.params['bb_stddev']}"
        ]
        self.data["bb_mid"] = bb[
            f"BBM_{self.params['bb_period']}_{self.params['bb_stddev']}"
        ]
        self.data["bb_bot"] = bb[
            f"BBL_{self.params['bb_period']}_{self.params['bb_stddev']}"
        ]

        # Define conditions
        self.data["bb_lower_touch"] = self.data["close"] <= self.data["bb_bot"]
        self.data["bb_upper_touch"] = self.data["close"] >= self.data["bb_top"]
        self.data["bullish_entry"] = (
            self.data["rsi"] < self.params["rsi_oversold"]
        ) & self.data["bb_lower_touch"]
        self.data["bearish_entry"] = (
            self.data["rsi"] > self.params["rsi_overbought"]
        ) & self.data["bb_upper_touch"]
        self.data["bullish_exit"] = (self.data["rsi"] > self.params["rsi_exit"]) | (
            self.data["close"] >= self.data["bb_mid"]
        )
        self.data["bearish_exit"] = (self.data["rsi"] < self.params["rsi_exit"]) | (
            self.data["close"] <= self.data["bb_mid"]
        )
        self.data["volume_sma"] = ta.sma(self.data["volume"], length=20)
        self.data["volume_surge"] = (
            self.data["volume"]
            > self.data["volume_sma"] * COMMON_PARAMS["volume_surge_threshold"]
        )
        self.entry_signals = []
        logger.debug(f"Initialized RSIBB with params: {self.params}")
        logger.info(
            f"RSIBB initialized with rsi_period={self.params['rsi_period']}, "
            f"bb_period={self.params['bb_period']}, bb_stddev={self.params['bb_stddev']}, "
            f"rsi_oversold={self.params['rsi_oversold']}, rsi_overbought={self.params['rsi_overbought']}, "
            f"rsi_exit={self.params['rsi_exit']}"
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
            if pd.isna(self.data.iloc[idx]["rsi"]) or pd.isna(
                self.data.iloc[idx]["bb_mid"]
            ):
                logger.debug(f"Invalid indicator values at row {idx}")
                continue

            # Store indicator data for analysis
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": self.data.iloc[idx]["close"],
                    "rsi": self.data.iloc[idx]["rsi"],
                    "bb_top": self.data.iloc[idx]["bb_top"],
                    "bb_mid": self.data.iloc[idx]["bb_mid"],
                    "bb_bot": self.data.iloc[idx]["bb_bot"],
                }
            )

            # Check for trading signals
            if not self.open_positions:
                if (
                    self.data.iloc[idx]["bullish_entry"]
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
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "BUY"}
                    )
                    self._notify_order(idx)
                    trade_logger.info(
                        f"BUY SIGNAL | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f} | "
                        f"RSI: {self.data.iloc[idx]['rsi']:.2f} | BB Lower Touch: {self.data.iloc[idx]['bb_lower_touch']}"
                    )
                elif (
                    self.data.iloc[idx]["bearish_entry"]
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
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "SELL"}
                    )
                    self._notify_order(idx)
                    trade_logger.info(
                        f"SELL SIGNAL | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f} | "
                        f"RSI: {self.data.iloc[idx]['rsi']:.2f} | BB Upper Touch: {self.data.iloc[idx]['bb_upper_touch']}"
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
                        f"RSI: {self.data.iloc[idx]['rsi']:.2f} | Reached BB Mid: {self.data.iloc[idx]['close'] >= self.data.iloc[idx]['bb_mid']}"
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
                        f"RSI: {self.data.iloc[idx]['rsi']:.2f} | Reached BB Mid: {self.data.iloc[idx]['close'] <= self.data.iloc[idx]['bb_mid']}"
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
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "bb_period": trial.suggest_int("bb_period", 15, 25),
            "bb_stddev": trial.suggest_float("bb_stddev", 1.5, 2.5, step=0.1),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 35),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 65, 80),
            "rsi_exit": trial.suggest_int("rsi_exit", 45, 55),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            bb_period = params.get("bb_period", 20)
            return max(rsi_period, bb_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
