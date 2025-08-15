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


class VERV:
    """
    VWAP + EMA + RSI + Volume Deviation (pandas_ta implementation, sr_rsi structure)
    """

    params = {
        "vwap_period": 20,
        "ema_period": 20,
        "rsi_period": 14,
        "vol_sma_period": 14,
        "vol_threshold": 1.5,
        "verbose": False,
    }

    optimization_params = {
        "vwap_period": {"type": "int", "low": 15, "high": 30, "step": 1},
        "ema_period": {"type": "int", "low": 15, "high": 30, "step": 1},
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
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
                self.params["vwap_period"],
                self.params["ema_period"],
                self.params["rsi_period"],
                self.params["vol_sma_period"],
            )
            + 5
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        # EMA and RSI
        self.data["ema"] = ta.ema(self.data["close"], length=self.params["ema_period"])
        self.data["rsi"] = ta.rsi(self.data["close"], length=self.params["rsi_period"])

        # Volume ratio
        self.data["vol_sma"] = ta.sma(
            self.data["volume"], length=self.params["vol_sma_period"]
        )
        self.data["vol_ratio"] = self.data["volume"] / self.data["vol_sma"]

        # VWAP (session-like reset by date)
        typical = (self.data["high"] + self.data["low"] + self.data["close"]) / 3.0
        tpv = typical * self.data["volume"]
        dates = self.data["datetime"].dt.date
        vwap = []
        cum_tpv = 0.0
        cum_vol = 0.0
        last_date = None
        for i in range(len(self.data)):
            d = dates.iloc[i]
            if last_date is None or d != last_date:
                cum_tpv = 0.0
                cum_vol = 0.0
                last_date = d
            cum_tpv += tpv.iloc[i]
            cum_vol += self.data["volume"].iloc[i]
            vwap.append(cum_tpv / cum_vol if cum_vol > 0 else np.nan)
        self.data["vwap"] = pd.Series(vwap, index=self.data.index)

        logger.debug(f"Initialized VERV with params: {self.params}")

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

            row = self.data.iloc[idx]
            prev = self.data.iloc[idx - 1]
            required = [
                row["vwap"],
                row["ema"],
                row["rsi"],
                row["vol_ratio"],
                prev["rsi"],
            ]
            if any(pd.isna(x) for x in required):
                continue

            # States
            price = row["close"]
            price_below_vwap = price < row["vwap"]
            price_above_vwap = price > row["vwap"]
            rsi_rising = row["rsi"] > prev["rsi"] and 30 < row["rsi"] < 70
            rsi_falling = row["rsi"] < prev["rsi"] and 30 < row["rsi"] < 70
            price_above_ema = price > row["ema"]
            price_below_ema = price < row["ema"]
            high_volume = row["vol_ratio"] > self.params["vol_threshold"]
            price_near_vwap = (
                abs(price - row["vwap"]) / row["vwap"] < 0.01 if row["vwap"] else False
            )
            rsi_reversal_long = row["rsi"] < prev["rsi"]
            rsi_reversal_short = row["rsi"] > prev["rsi"]
            ema_reversal_long = price < row["ema"]
            ema_reversal_short = price > row["ema"]
            vol_decrease = row["vol_ratio"] < 1.0
            signal_conflict_long = (
                sum([rsi_reversal_long, ema_reversal_long, vol_decrease]) >= 2
            )
            signal_conflict_short = (
                sum([rsi_reversal_short, ema_reversal_short, vol_decrease]) >= 2
            )

            # Store snapshot
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": price,
                    "vwap": row["vwap"],
                    "ema": row["ema"],
                    "rsi": row["rsi"],
                    "vol_ratio": row["vol_ratio"],
                }
            )

            if not self.open_positions:
                # Long Entry
                if price_below_vwap and rsi_rising and price_above_ema and high_volume:
                    self.order = {
                        "ref": str(uuid4()),
                        "action": "buy",
                        "order_type": "enter_long",
                        "status": "Completed",
                        "executed_price": price,
                        "size": 100,
                        "commission": abs(price * 100 * 0.001),
                        "executed_time": self.data.iloc[idx]["datetime"],
                    }
                    self.last_signal = "buy"
                    self._notify_order(idx)
                    trade_logger.info(
                        f"BUY SIGNAL (VERV) | Time: {bar_time_ist} | Price: {price:.2f}"
                    )
                # Short Entry
                elif (
                    price_above_vwap and rsi_falling and price_below_ema and high_volume
                ):
                    self.order = {
                        "ref": str(uuid4()),
                        "action": "sell",
                        "order_type": "enter_short",
                        "status": "Completed",
                        "executed_price": price,
                        "size": -100,
                        "commission": abs(price * 100 * 0.001),
                        "executed_time": self.data.iloc[idx]["datetime"],
                    }
                    self.last_signal = "sell"
                    self._notify_order(idx)
                    trade_logger.info(
                        f"SELL SIGNAL (VERV) | Time: {bar_time_ist} | Price: {price:.2f}"
                    )
            else:
                if self.open_positions[-1]["direction"] == "long":
                    if price_near_vwap or signal_conflict_long:
                        self._close_position(
                            idx,
                            "Price near VWAP or signals conflict",
                            "sell",
                            "exit_long",
                        )
                        self.last_signal = None
                elif self.open_positions[-1]["direction"] == "short":
                    if price_near_vwap or signal_conflict_short:
                        self._close_position(
                            idx,
                            "Price near VWAP or signals conflict",
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
            "vwap_period": trial.suggest_int("vwap_period", 15, 30),
            "ema_period": trial.suggest_int("ema_period", 15, 30),
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "vol_sma_period": trial.suggest_int("vol_sma_period", 10, 20),
            "vol_threshold": trial.suggest_float("vol_threshold", 1.2, 2.0, step=0.1),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            vwap_period = params.get("vwap_period", 20)
            ema_period = params.get("ema_period", 20)
            rsi_period = params.get("rsi_period", 14)
            vol_sma_period = params.get("vol_sma_period", 14)
            return max(vwap_period, ema_period, rsi_period, vol_sma_period) + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 35
