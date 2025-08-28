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


class Supertrend:
    def __init__(self, period=10, multiplier=3.0):
        self.period = period
        self.multiplier = multiplier
        self.atr = None
        self.supertrend = []
        self.last_supertrend = 0

    def update(self, high, low, close):
        if len(close) < self.period:
            return self.last_supertrend

        # Calculate ATR
        if self.atr is None:
            self.atr = ta.atr(high, low, close, length=self.period)

        if len(close) < self.period or pd.isna(self.atr.iloc[-1]):
            return self.last_supertrend

        current_atr = self.atr.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]

        # Basic bands
        mid = (current_high + current_low) / 2
        upper_band = mid + self.multiplier * current_atr
        lower_band = mid - self.multiplier * current_atr

        # Adjust bands based on previous Supertrend
        if self.last_supertrend > close.iloc[-2] if len(close) > 1 else close.iloc[-1]:
            upper_band = min(
                upper_band, self.supertrend[-1] if self.supertrend else upper_band
            )
        else:
            lower_band = max(
                lower_band, self.supertrend[-1] if self.supertrend else lower_band
            )

        # Determine Supertrend value
        if self.last_supertrend > close.iloc[-2] if len(close) > 1 else close.iloc[-1]:
            supertrend_val = lower_band if current_close < lower_band else upper_band
        else:
            supertrend_val = upper_band if current_close > upper_band else lower_band

        self.supertrend.append(supertrend_val)
        self.last_supertrend = supertrend_val
        return supertrend_val


class ChaikinMoneyFlow:
    def __init__(self, period=20):
        self.period = period

    def update(self, high, low, close, volume):
        if len(close) < self.period:
            return 0

        # Calculate MFM and CMF
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1e-5)
        mfv = mfm * volume
        cmf = mfv.rolling(self.period).sum() / volume.rolling(
            self.period
        ).sum().replace(0, 1e-5)
        return cmf.iloc[-1]


class SupertrendCCICMF:
    """
    Supertrend + CCI + CMF Strategy
    (Documentation remains unchanged)
    """

    params = {
        "supertrend_period": 10,
        "supertrend_multiplier": 3.0,
        "cci_period": 20,
        "cmf_period": 20,
        "cci_threshold": 100,
        "verbose": False,
    }

    optimization_params = {
        "supertrend_period": {"type": "int", "low": 7, "high": 15, "step": 1},
        "supertrend_multiplier": {
            "type": "float",
            "low": 2.0,
            "high": 4.0,
            "step": 0.5,
        },
        "cci_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "cmf_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "cci_threshold": {"type": "int", "low": 80, "high": 120, "step": 10},
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
                self.params["supertrend_period"],
                self.params["cci_period"],
                self.params["cmf_period"],
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        # Initialize indicators
        self.supertrend_ind = Supertrend(
            period=self.params["supertrend_period"],
            multiplier=self.params["supertrend_multiplier"],
        )
        self.cmf_ind = ChaikinMoneyFlow(period=self.params["cmf_period"])

        # Calculate CCI
        self.data["cci"] = ta.cci(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            length=self.params["cci_period"],
        )
        self.entry_signals = []
        logger.debug(f"Initialized SupertrendCCICMF with params: {self.params}")

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
            supertrend_val = self.supertrend_ind.update(
                self.data["high"].iloc[: idx + 1],
                self.data["low"].iloc[: idx + 1],
                self.data["close"].iloc[: idx + 1],
            )
            cmf_val = self.cmf_ind.update(
                self.data["high"].iloc[: idx + 1],
                self.data["low"].iloc[: idx + 1],
                self.data["close"].iloc[: idx + 1],
                self.data["volume"].iloc[: idx + 1],
            )

            # Check for invalid indicator values
            if (
                pd.isna(supertrend_val)
                or pd.isna(self.data.iloc[idx]["cci"])
                or pd.isna(cmf_val)
            ):
                continue

            # Store indicator data for analysis
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": self.data.iloc[idx]["close"],
                    "supertrend": supertrend_val,
                    "cci": self.data.iloc[idx]["cci"],
                    "cmf": cmf_val,
                }
            )

            # Determine trend direction
            supertrend_bullish = self.data.iloc[idx]["close"] > supertrend_val
            supertrend_bearish = self.data.iloc[idx]["close"] < supertrend_val

            # Trading logic
            if not self.open_positions:
                # Long Entry
                if (
                    supertrend_bullish
                    and self.data.iloc[idx]["cci"] > self.params["cci_threshold"]
                    and cmf_val > 0
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
                    # trade_logger.info(
                    #     f"BUY SIGNAL (Supertrend + CCI + CMF) | Time: {bar_time_ist} | "
                    #     f"Price: {self.data.iloc[idx]['close']:.2f} | "
                    #     f"Supertrend: {supertrend_val:.2f}"
                    # )
                # Short Entry
                elif (
                    supertrend_bearish
                    and self.data.iloc[idx]["cci"] < -self.params["cci_threshold"]
                    and cmf_val < 0
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
                    #     f"SELL SIGNAL (Supertrend + CCI + CMF) | Time: {bar_time_ist} | "
                    #     f"Price: {self.data.iloc[idx]['close']:.2f} | "
                    #     f"Supertrend: {supertrend_val:.2f}"
                    # )
            else:
                if self.open_positions[-1]["direction"] == "long":
                    # Long Exit
                    reverse_count = sum(
                        [
                            not supertrend_bullish,
                            self.data.iloc[idx]["cci"] < 0,
                            cmf_val < 0,
                        ]
                    )
                    if reverse_count >= 2:
                        self._close_position(
                            idx,
                            f"{reverse_count} indicators reversed",
                            "sell",
                            "exit_long",
                        )
                        self.last_signal = None
                elif self.open_positions[-1]["direction"] == "short":
                    # Short Exit
                    reverse_count = sum(
                        [
                            not supertrend_bearish,
                            self.data.iloc[idx]["cci"] > 0,
                            cmf_val > 0,
                        ]
                    )
                    if reverse_count >= 2:
                        self._close_position(
                            idx,
                            f"{reverse_count} indicators reversed",
                            "buy",
                            "exit_short",
                        )
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
            "supertrend_period": trial.suggest_int("supertrend_period", 7, 15),
            "supertrend_multiplier": trial.suggest_float(
                "supertrend_multiplier", 2.0, 4.0, step=0.5
            ),
            "cci_period": trial.suggest_int("cci_period", 10, 30),
            "cmf_period": trial.suggest_int("cmf_period", 10, 30),
            "cci_threshold": trial.suggest_int("cci_threshold", 80, 120, step=10),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            supertrend_period = params.get("supertrend_period", 10)
            cci_period = params.get("cci_period", 20)
            cmf_period = params.get("cmf_period", 20)
            return max(supertrend_period, cci_period, cmf_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
