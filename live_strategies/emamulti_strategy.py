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


class EMAMultiStrategy:
    """
    EMA Multi (fast, med1, med2, slow) + VWAP + StochRSI + CMF Strategy
    (pandas_ta implementation aligned with sr_rsi.py structure)
    """

    params = {
        "ema_fast": 5,
        "ema_med1": 9,
        "ema_med2": 13,
        "ema_slow": 21,
        "stochrsi_period": 14,
        "cmf_period": 20,
        "stochrsi_oversold": 20.0,
        "stochrsi_overbought": 80.0,
        "verbose": False,
    }

    optimization_params = {
        "ema_fast": {"type": "int", "low": 3, "high": 8, "step": 1},
        "ema_med1": {"type": "int", "low": 7, "high": 12, "step": 1},
        "ema_med2": {"type": "int", "low": 10, "high": 16, "step": 1},
        "ema_slow": {"type": "int", "low": 18, "high": 25, "step": 1},
        "stochrsi_period": {"type": "int", "low": 10, "high": 20, "step": 2},
        "cmf_period": {"type": "int", "low": 15, "high": 25, "step": 2},
        "stochrsi_oversold": {"type": "float", "low": 15, "high": 25, "step": 5},
        "stochrsi_overbought": {"type": "float", "low": 75, "high": 85, "step": 5},
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
                self.params["ema_slow"],
                self.params["stochrsi_period"],
                self.params["cmf_period"],
            )
            + 5
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        # Indicators
        self.data["ema_fast"] = ta.ema(
            self.data["close"], length=self.params["ema_fast"]
        )
        self.data["ema_med1"] = ta.ema(
            self.data["close"], length=self.params["ema_med1"]
        )
        self.data["ema_med2"] = ta.ema(
            self.data["close"], length=self.params["ema_med2"]
        )
        self.data["ema_slow"] = ta.ema(
            self.data["close"], length=self.params["ema_slow"]
        )

        # VWAP (use rolling typical price * volume / volume) by session approximation
        typical = (self.data["high"] + self.data["low"] + self.data["close"]) / 3.0
        tpv = typical * self.data["volume"]
        # Reset daily using date change
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

        # StochRSI via pandas_ta
        stochrsi = ta.stochrsi(
            self.data["close"],
            length=self.params["stochrsi_period"],
            rsi_length=self.params["stochrsi_period"],
            k=3,
            d=3,
        )
        # pandas_ta naming e.g., STOCHRSIk_14_14_3_3 and STOCHRSId_14_14_3_3
        k_name = [c for c in stochrsi.columns if c.startswith("STOCHRSIk_")][0]
        d_name = [c for c in stochrsi.columns if c.startswith("STOCHRSId_")][0]
        self.data["stochrsi_k"] = stochrsi[k_name]
        self.data["stochrsi_d"] = stochrsi[d_name]

        # Chaikin Money Flow
        # CMF = sum( MFM * Volume, n ) / sum(Volume, n)
        high = self.data["high"]
        low = self.data["low"]
        close = self.data["close"]
        vol = self.data["volume"]
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1e-9)
        mfv = mfm * vol
        self.data["cmf"] = mfv.rolling(self.params["cmf_period"]).sum() / vol.rolling(
            self.params["cmf_period"]
        ).sum().replace(0, 1e-9)

        logger.debug(f"Initialized EMAMultiStrategy with params: {self.params}")

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

            # Validate indicator values
            row = self.data.iloc[idx]
            required = [
                row["ema_fast"],
                row["ema_med1"],
                row["ema_med2"],
                row["ema_slow"],
                row["vwap"],
                row["stochrsi_k"],
                row["stochrsi_d"],
                row["cmf"],
            ]
            if any(pd.isna(x) for x in required):
                continue

            # EMA alignment
            bullish_ema = (
                row["ema_fast"] > row["ema_med1"] > row["ema_med2"] > row["ema_slow"]
            )
            bearish_ema = (
                row["ema_fast"] < row["ema_med1"] < row["ema_med2"] < row["ema_slow"]
            )
            price = row["close"]
            price_above_vwap = price > row["vwap"]
            price_below_vwap = price < row["vwap"]

            # StochRSI cross and slope
            prev = self.data.iloc[idx - 1]
            stochrsi_bull_cross = (
                row["stochrsi_k"] > row["stochrsi_d"]
                and prev["stochrsi_k"] <= prev["stochrsi_d"]
                and row["stochrsi_k"] > self.params["stochrsi_oversold"]
            )
            stochrsi_bear_cross = (
                row["stochrsi_k"] < row["stochrsi_d"]
                and prev["stochrsi_k"] >= prev["stochrsi_d"]
                and row["stochrsi_k"] < self.params["stochrsi_overbought"]
            )
            stochrsi_up = (
                row["stochrsi_k"] > prev["stochrsi_k"]
                and row["stochrsi_k"] < self.params["stochrsi_overbought"]
            )
            stochrsi_down = (
                row["stochrsi_k"] < prev["stochrsi_k"]
                and row["stochrsi_k"] > self.params["stochrsi_oversold"]
            )

            cmf_bullish = row["cmf"] > 0
            cmf_bearish = row["cmf"] < 0

            # Log snapshot
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": price,
                    "ema_fast": row["ema_fast"],
                    "ema_med1": row["ema_med1"],
                    "ema_med2": row["ema_med2"],
                    "ema_slow": row["ema_slow"],
                    "vwap": row["vwap"],
                    "stochrsi_k": row["stochrsi_k"],
                    "stochrsi_d": row["stochrsi_d"],
                    "cmf": row["cmf"],
                }
            )

            if not self.open_positions:
                # Long: Pullback scenario
                if bullish_ema and price_above_vwap and stochrsi_up and cmf_bullish:
                    self._place_order(idx, "buy", "enter_long")
                    self.last_signal = "buy"
                    # trade_logger.info(
                    #     f"BUY SIGNAL (EMAMulti Pullback) | Time: {bar_time_ist} | Price: {price:.2f}"
                    # )
                # Long: Breakout scenario
                elif (
                    bullish_ema
                    and price_above_vwap
                    and stochrsi_bull_cross
                    and cmf_bullish
                ):
                    self._place_order(idx, "buy", "enter_long")
                    self.last_signal = "buy"
                    # trade_logger.info(
                    #     f"BUY SIGNAL (EMAMulti Breakout) | Time: {bar_time_ist} | Price: {price:.2f}"
                    # )
                # Short: Bounce scenario
                elif bearish_ema and price_below_vwap and stochrsi_down and cmf_bearish:
                    self._place_order(idx, "sell", "enter_short")
                    self.last_signal = "sell"
                    # trade_logger.info(
                    #     f"SELL SIGNAL (EMAMulti Bounce) | Time: {bar_time_ist} | Price: {price:.2f}"
                    # )
                # Short: Breakdown scenario
                elif (
                    bearish_ema
                    and price_below_vwap
                    and stochrsi_bear_cross
                    and cmf_bearish
                ):
                    self._place_order(idx, "sell", "enter_short")
                    self.last_signal = "sell"
                    # trade_logger.info(
                    #     f"SELL SIGNAL (EMAMulti Breakdown) | Time: {bar_time_ist} | Price: {price:.2f}"
                    # )
            else:
                # Exits: EMA reversal or VWAP side flip or StochRSI extremes
                if self.open_positions[-1]["direction"] == "long":
                    ema_reversal = row["ema_fast"] < row["ema_med1"]
                    stoch_overbought = (
                        row["stochrsi_k"] >= self.params["stochrsi_overbought"]
                    )
                    if ema_reversal or (not price_above_vwap) or stoch_overbought:
                        reason = (
                            "EMA reversal"
                            if ema_reversal
                            else (
                                "Below VWAP"
                                if not price_above_vwap
                                else "StochRSI overbought"
                            )
                        )
                        self._close_position(idx, reason, "sell", "exit_long")
                        self.last_signal = None
                elif self.open_positions[-1]["direction"] == "short":
                    ema_reversal = row["ema_fast"] > row["ema_med1"]
                    stoch_oversold = (
                        row["stochrsi_k"] <= self.params["stochrsi_oversold"]
                    )
                    if ema_reversal or (not price_below_vwap) or stoch_oversold:
                        reason = (
                            "EMA reversal"
                            if ema_reversal
                            else (
                                "Above VWAP"
                                if not price_below_vwap
                                else "StochRSI oversold"
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
        self.order_type = None

    def _close_position(self, idx, reason, action=None, order_type=None):
        if not self.open_positions:
            return

        entry_info = self.open_positions.pop(0)
        # Default action/order_type if not provided (e.g., force close at cutoff time)
        if action is None:
            action = "sell" if entry_info["direction"] == "long" else "buy"
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
            "ema_fast": trial.suggest_int("ema_fast", 3, 8),
            "ema_med1": trial.suggest_int("ema_med1", 7, 12),
            "ema_med2": trial.suggest_int("ema_med2", 10, 16),
            "ema_slow": trial.suggest_int("ema_slow", 18, 25),
            "stochrsi_period": trial.suggest_int("stochrsi_period", 10, 20, step=2),
            "cmf_period": trial.suggest_int("cmf_period", 15, 25, step=2),
            "stochrsi_oversold": trial.suggest_float(
                "stochrsi_oversold", 15, 25, step=5
            ),
            "stochrsi_overbought": trial.suggest_float(
                "stochrsi_overbought", 75, 85, step=5
            ),
        }

    @classmethod
    def get_min_data_points(cls, params):
        try:
            ema_slow = params.get("ema_slow", 21)
            stochrsi_period = params.get("stochrsi_period", 14)
            cmf_period = params.get("cmf_period", 20)
            return max(ema_slow, stochrsi_period, cmf_period) + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 35
