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


class SRRSI:
    """
    Support/Resistance Levels + RSI Confluence Trading Strategy
    (Documentation remains unchanged)
    """

    params = {
        "rsi_period": 14,
        "sr_lookback": 20,
        "sr_tolerance": 0.5,
        "rsi_oversold": COMMON_PARAMS["rsi_oversold"],
        "rsi_overbought": COMMON_PARAMS["rsi_overbought"],
        "rsi_exit": 50,
        "verbose": False,
    }

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "sr_lookback": {"type": "int", "low": 15, "high": 30, "step": 1},
        "sr_tolerance": {"type": "float", "low": 0.3, "high": 1.0, "step": 0.1},
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
        self.warmup_period = self.params["rsi_period"] + self.params["sr_lookback"] + 2
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        # Initialize indicators using pandas_ta
        self.data["rsi"] = ta.rsi(self.data["close"], length=self.params["rsi_period"])
        self.data["swing_high"] = (
            self.data["high"].rolling(window=self.params["sr_lookback"]).max()
        )
        self.data["swing_low"] = (
            self.data["low"].rolling(window=self.params["sr_lookback"]).min()
        )

        # Define conditions
        self.data["near_support"] = self.data.apply(
            lambda x: (
                abs(x["close"] - x["swing_low"]) / x["swing_low"]
                < self.params["sr_tolerance"] / 100
                if pd.notna(x["swing_low"])
                else False
            ),
            axis=1,
        )
        self.data["near_resistance"] = self.data.apply(
            lambda x: (
                abs(x["close"] - x["swing_high"]) / x["swing_high"]
                < self.params["sr_tolerance"] / 100
                if pd.notna(x["swing_high"])
                else False
            ),
            axis=1,
        )
        self.data["bullish_entry"] = (
            self.data["rsi"] < self.params["rsi_oversold"]
        ) & self.data["near_support"]
        self.data["bearish_entry"] = (
            self.data["rsi"] > self.params["rsi_overbought"]
        ) & self.data["near_resistance"]
        self.data["bullish_exit"] = (self.data["rsi"] > self.params["rsi_exit"]) | (
            self.data["close"] < self.data["swing_low"]
        )
        self.data["bearish_exit"] = (self.data["rsi"] < self.params["rsi_exit"]) | (
            self.data["close"] > self.data["swing_high"]
        )
        self.data["volume_sma"] = ta.sma(self.data["volume"], length=20)
        self.data["volume_surge"] = (
            self.data["volume"]
            > self.data["volume_sma"] * COMMON_PARAMS["volume_surge_threshold"]
        )
        self.entry_signals = []
        logger.debug(f"Initialized SRRSI with params: {self.params}")
        logger.info(
            f"SRRSI initialized with rsi_period={self.params['rsi_period']}, "
            f"sr_lookback={self.params['sr_lookback']}, sr_tolerance={self.params['sr_tolerance']}, "
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
            if (
                pd.isna(self.data.iloc[idx]["rsi"])
                or pd.isna(self.data.iloc[idx]["swing_high"])
                or pd.isna(self.data.iloc[idx]["swing_low"])
            ):
                logger.debug(f"Invalid indicator values at row {idx}")
                continue

            # Store indicator data for analysis
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": self.data.iloc[idx]["close"],
                    "rsi": self.data.iloc[idx]["rsi"],
                    "swing_high": self.data.iloc[idx]["swing_high"],
                    "swing_low": self.data.iloc[idx]["swing_low"],
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
                    self._notify_order(idx)
                    bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
                    self.entry_signals.append(
                        {"datetime": bar_time_ist, "signal": "BUY"}
                    )
                    # trade_logger.info(
                    #     f"BUY SIGNAL | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f} | "
                    #     f"RSI: {self.data.iloc[idx]['rsi']:.2f} | Near Support: {self.data.iloc[idx]['near_support']}"
                    # )
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
                    # trade_logger.info(
                    #     f"SELL SIGNAL | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f} | "
                    #     f"RSI: {self.data.iloc[idx]['rsi']:.2f} | Near Resistance: {self.data.iloc[idx]['near_resistance']}"
                    # )
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
                    # trade_logger.info(
                    #     f"EXIT LONG | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f} | "
                    #     f"RSI: {self.data.iloc[idx]['rsi']:.2f} | Broke Support: {self.data.iloc[idx]['close'] < self.data.iloc[idx]['swing_low']}"
                    # )
                elif (
                    self.open_positions[-1]["direction"] == "short"
                    and self.data.iloc[idx]["bearish_exit"]
                ):
                    self._close_position(
                        idx, "Bearish exit condition", "buy", "exit_short"
                    )
                    self.last_signal = None
                    # trade_logger.info(
                    #     f"EXIT SHORT | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f} | "
                    #     f"RSI: {self.data.iloc[idx]['rsi']:.2f} | Broke Resistance: {self.data.iloc[idx]['close'] > self.data.iloc[idx]['swing_high']}"
                    # )
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
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "sr_lookback": trial.suggest_int("sr_lookback", 15, 30),
            "sr_tolerance": trial.suggest_float("sr_tolerance", 0.3, 1.0, step=0.1),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 35),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 65, 80),
            "rsi_exit": trial.suggest_int("rsi_exit", 45, 55),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            sr_lookback = params.get("sr_lookback", 20)
            return max(rsi_period, sr_lookback) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
