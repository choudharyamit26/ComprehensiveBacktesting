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


class VWAPBounceRejection:
    """
    VWAP Bounce/Rejection Strategy
    (Documentation remains unchanged)
    """

    params = {
        "atr_period": 14,
        "volume_lookback": 5,
        "stop_loss_atr_mult": 1.5,
        "vwap_proximity_mult": 0.5,
        "profit_target_mult": 2.0,
        "verbose": False,
    }

    optimization_params = {
        "atr_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "volume_lookback": {"type": "int", "low": 3, "high": 7, "step": 1},
        "stop_loss_atr_mult": {"type": "float", "low": 1.0, "high": 2.5, "step": 0.25},
        "vwap_proximity_mult": {"type": "float", "low": 0.3, "high": 0.8, "step": 0.1},
        "profit_target_mult": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.25},
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
            max(self.params["atr_period"], self.params["volume_lookback"]) + 5
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []
        self.session_open_price = None
        self.current_date = None

        # Initialize indicators using pandas_ta
        self.data["vwap"] = ta.vwap(
            self.data["high"], self.data["low"], self.data["close"], self.data["volume"]
        )
        self.data["atr"] = ta.atr(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            length=self.params["atr_period"],
        )
        self.data["volume_avg"] = (
            self.data["volume"].rolling(window=self.params["volume_lookback"]).mean()
        )

        # Candlestick pattern detection
        self.data["bullish_pattern"] = self._detect_bullish_pattern()
        self.data["bearish_pattern"] = self._detect_bearish_pattern()

        # Define conditions
        self.data["volume_rising"] = self.data["volume"] > self.data["volume_avg"]
        self.data["near_vwap"] = self.data.apply(
            lambda x: (
                abs(x["close"] - x["vwap"])
                <= x["atr"] * self.params["vwap_proximity_mult"]
                if pd.notna(x["vwap"]) and pd.notna(x["atr"])
                else False
            ),
            axis=1,
        )

        logger.debug(f"Initialized VWAPBounceRejection with params: {self.params}")
        logger.info(
            f"VWAPBounceRejection initialized with atr_period={self.params['atr_period']}, "
            f"volume_lookback={self.params['volume_lookback']}, stop_loss_atr_mult={self.params['stop_loss_atr_mult']}, "
            f"vwap_proximity_mult={self.params['vwap_proximity_mult']}, profit_target_mult={self.params['profit_target_mult']}"
        )

    def _detect_bullish_pattern(self):
        body = abs(self.data["close"] - self.data["open"])
        upper_shadow = self.data["high"] - self.data[["close", "open"]].max(axis=1)
        lower_shadow = self.data[["close", "open"]].min(axis=1) - self.data["low"]
        return (
            (self.data["close"] > self.data["open"])
            & (body > upper_shadow)
            & (body > lower_shadow)
        )

    def _detect_bearish_pattern(self):
        body = abs(self.data["close"] - self.data["open"])
        upper_shadow = self.data["high"] - self.data[["close", "open"]].max(axis=1)
        lower_shadow = self.data[["close", "open"]].min(axis=1) - self.data["low"]
        return (
            (self.data["close"] < self.data["open"])
            & (body > upper_shadow)
            & (body > lower_shadow)
        )

    def _set_session_open(self, idx):
        bar_time = self.data.iloc[idx]["datetime"].astimezone(
            pytz.timezone("Asia/Kolkata")
        )
        current_date = bar_time.date()
        current_time = bar_time.time()

        if self.current_date != current_date:
            if current_time >= datetime.time(9, 15):
                self.current_date = current_date
                self.session_open_price = self.data.iloc[idx]["open"]
                logger.debug(
                    f"New session started on {current_date}, open price: {self.session_open_price}"
                )
        return self.session_open_price

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

            # Set session open price
            session_open = self._set_session_open(idx)
            if session_open is None:
                continue

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
            if pd.isna(self.data.iloc[idx]["vwap"]) or pd.isna(
                self.data.iloc[idx]["atr"]
            ):
                logger.debug(f"Invalid indicator values at row {idx}")
                continue

            # Store indicator data for analysis
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": self.data.iloc[idx]["close"],
                    "vwap": self.data.iloc[idx]["vwap"],
                    "atr": self.data.iloc[idx]["atr"],
                    "volume": self.data.iloc[idx]["volume"],
                    "volume_avg": self.data.iloc[idx]["volume_avg"],
                }
            )

            # Calculate stop-loss and profit target levels
            atr = self.data.iloc[idx]["atr"]
            stop_loss_level = atr * self.params["stop_loss_atr_mult"]
            profit_target_level = atr * self.params["profit_target_mult"]

            # Check for trading signals
            if not self.open_positions:
                bullish_entry = (
                    self.data.iloc[idx]["near_vwap"]
                    and self.data.iloc[idx]["bullish_pattern"]
                    and self.data.iloc[idx]["volume_rising"]
                    and self.data.iloc[idx]["close"] < session_open
                )
                bearish_entry = (
                    self.data.iloc[idx]["near_vwap"]
                    and self.data.iloc[idx]["bearish_pattern"]
                    and self.data.iloc[idx]["volume_rising"]
                    and self.data.iloc[idx]["close"] > session_open
                )

                if bullish_entry:
                    self.order = {
                        "ref": str(uuid4()),
                        "action": "buy",
                        "order_type": "enter_long",
                        "status": "Completed",
                        "executed_price": self.data.iloc[idx]["close"],
                        "size": 100,
                        "commission": abs(self.data.iloc[idx]["close"] * 100 * 0.001),
                        "executed_time": self.data.iloc[idx]["datetime"],
                        "stop_loss": self.data.iloc[idx]["close"] - stop_loss_level,
                        "profit_target": self.data.iloc[idx]["close"]
                        + profit_target_level,
                    }
                    self.last_signal = "buy"
                    self._notify_order(idx)
                    trade_logger.info(
                        f"BUY SIGNAL | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f} | "
                        f"Near VWAP: {self.data.iloc[idx]['near_vwap']} | Bullish Pattern: {self.data.iloc[idx]['bullish_pattern']} | "
                        f"Volume Rising: {self.data.iloc[idx]['volume_rising']} | Below Session Open: {self.data.iloc[idx]['close'] < session_open}"
                    )
                elif bearish_entry:
                    self.order = {
                        "ref": str(uuid4()),
                        "action": "sell",
                        "order_type": "enter_short",
                        "status": "Completed",
                        "executed_price": self.data.iloc[idx]["close"],
                        "size": -100,
                        "commission": abs(self.data.iloc[idx]["close"] * 100 * 0.001),
                        "executed_time": self.data.iloc[idx]["datetime"],
                        "stop_loss": self.data.iloc[idx]["close"] + stop_loss_level,
                        "profit_target": self.data.iloc[idx]["close"]
                        - profit_target_level,
                    }
                    self.last_signal = "sell"
                    self._notify_order(idx)
                    trade_logger.info(
                        f"SELL SIGNAL | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f} | "
                        f"Near VWAP: {self.data.iloc[idx]['near_vwap']} | Bearish Pattern: {self.data.iloc[idx]['bearish_pattern']} | "
                        f"Volume Rising: {self.data.iloc[idx]['volume_rising']} | Above Session Open: {self.data.iloc[idx]['close'] > session_open}"
                    )
                else:
                    self.last_signal = None
            else:
                position = self.open_positions[-1]
                current_price = self.data.iloc[idx]["close"]
                if position["direction"] == "long":
                    exit_condition = (
                        current_price <= position["stop_loss"]
                        or current_price >= position["profit_target"]
                        or abs(current_price - self.data.iloc[idx]["vwap"])
                        > stop_loss_level
                    )
                    if exit_condition:
                        self._close_position(
                            idx,
                            "Long exit: Stop-loss, profit target, or VWAP deviation",
                            "sell",
                            "exit_long",
                        )
                        self.last_signal = None
                        trade_logger.info(
                            f"EXIT LONG | Time: {bar_time_ist} | Price: {current_price:.2f} | "
                            f"Stop-Loss Hit: {current_price <= position['stop_loss']} | "
                            f"Profit Target Hit: {current_price >= position['profit_target']} | "
                            f"VWAP Deviation: {abs(current_price - self.data.iloc[idx]['vwap']) > stop_loss_level}"
                        )
                elif position["direction"] == "short":
                    exit_condition = (
                        current_price >= position["stop_loss"]
                        or current_price <= position["profit_target"]
                        or abs(current_price - self.data.iloc[idx]["vwap"])
                        > stop_loss_level
                    )
                    if exit_condition:
                        self._close_position(
                            idx,
                            "Short exit: Stop-loss, profit target, or VWAP deviation",
                            "buy",
                            "exit_short",
                        )
                        self.last_signal = None
                        trade_logger.info(
                            f"EXIT SHORT | Time: {bar_time_ist} | Price: {current_price:.2f} | "
                            f"Stop-Loss Hit: {current_price >= position['stop_loss']} | "
                            f"Profit Target Hit: {current_price <= position['profit_target']} | "
                            f"VWAP Deviation: {abs(current_price - self.data.iloc[idx]['vwap']) > stop_loss_level}"
                        )
                else:
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
                "stop_loss": order["stop_loss"],
                "profit_target": order["profit_target"],
            }
            self.open_positions.append(position_info)
            trade_logger.info(
                f"BUY EXECUTED (Enter Long) | Ref: {order['ref']} | Price: {order['executed_price']:.2f} | "
                f"Stop-Loss: {order['stop_loss']:.2f} | Profit Target: {order['profit_target']:.2f}"
            )
        elif order["order_type"] == "enter_short" and order["action"] == "sell":
            position_info = {
                "entry_time": exec_dt,
                "entry_price": order["executed_price"],
                "size": order["size"],
                "commission": order["commission"],
                "ref": order["ref"],
                "direction": "short",
                "stop_loss": order["stop_loss"],
                "profit_target": order["profit_target"],
            }
            self.open_positions.append(position_info)
            trade_logger.info(
                f"SELL EXECUTED (Enter Short) | Ref: {order['ref']} | Price: {order['executed_price']:.2f} | "
                f"Stop-Loss: {order['stop_loss']:.2f} | Profit Target: {order['profit_target']:.2f}"
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
            "atr_period": trial.suggest_int("atr_period", 10, 20),
            "volume_lookback": trial.suggest_int("volume_lookback", 3, 7),
            "stop_loss_atr_mult": trial.suggest_float(
                "stop_loss_atr_mult", 1.0, 2.5, step=0.25
            ),
            "vwap_proximity_mult": trial.suggest_float(
                "vwap_proximity_mult", 0.3, 0.8, step=0.1
            ),
            "profit_target_mult": trial.suggest_float(
                "profit_target_mult", 1.5, 3.0, step=0.25
            ),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            atr_period = params.get("atr_period", 14)
            volume_lookback = params.get("volume_lookback", 5)
            return max(atr_period, volume_lookback) + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 20
