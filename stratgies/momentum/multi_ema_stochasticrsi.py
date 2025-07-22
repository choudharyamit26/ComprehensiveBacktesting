import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class VWAP(bt.Indicator):
    """Volume Weighted Average Price indicator"""

    lines = ("vwap",)

    def __init__(self):
        self.typical_price = (self.data.high + self.data.low + self.data.close) / 3
        self.cumulative_volume = 0
        self.cumulative_pv = 0

    def next(self):
        volume = self.data.volume[0]
        typical = self.typical_price[0]

        self.cumulative_volume += volume
        self.cumulative_pv += typical * volume

        if self.cumulative_volume > 0:
            self.lines.vwap[0] = self.cumulative_pv / self.cumulative_volume
        else:
            self.lines.vwap[0] = self.data.close[0]


class StochRSI(bt.Indicator):
    """Stochastic RSI indicator"""

    lines = ("stochrsi", "stochrsi_signal")
    params = (
        ("rsi_period", 14),
        ("stoch_period", 14),
        ("smooth_k", 3),
        ("smooth_d", 3),
    )

    def __init__(self):
        self.rsi = btind.RSI(self.data, period=self.params.rsi_period)

        # Calculate Stochastic of RSI
        highest_rsi = btind.Highest(self.rsi, period=self.params.stoch_period)
        lowest_rsi = btind.Lowest(self.rsi, period=self.params.stoch_period)

        k_raw = 100 * (self.rsi - lowest_rsi) / (highest_rsi - lowest_rsi)

        # Smooth %K and %D
        self.lines.stochrsi = btind.SMA(k_raw, period=self.params.smooth_k)
        self.lines.stochrsi_signal = btind.SMA(
            self.lines.stochrsi, period=self.params.smooth_d
        )


class ChaikinMoneyFlow(bt.Indicator):
    """Chaikin Money Flow indicator"""

    lines = ("cmf",)
    params = (("period", 20),)

    def __init__(self):
        self.lines.cmf = self.data.close * 0  # Initialize with zeros

    def next(self):
        if len(self) < self.params.period:
            self.lines.cmf[0] = 0
            return

        mfv_sum = 0
        volume_sum = 0

        for i in range(self.params.period):
            idx = -i
            high_val = self.data.high[idx]
            low_val = self.data.low[idx]
            close_val = self.data.close[idx]
            volume_val = self.data.volume[idx]

            if (
                np.isnan(high_val)
                or np.isnan(low_val)
                or np.isnan(close_val)
                or np.isnan(volume_val)
                or volume_val <= 0
            ):
                continue

            price_range = high_val - low_val
            if price_range <= 0:
                mfm = 0
            else:
                mfm = ((close_val - low_val) - (high_val - close_val)) / price_range

            mfv = mfm * volume_val
            mfv_sum += mfv
            volume_sum += volume_val

        if volume_sum <= 0:
            self.lines.cmf[0] = 0
        else:
            self.lines.cmf[0] = mfv_sum / volume_sum

        if np.isnan(self.lines.cmf[0]):
            self.lines.cmf[0] = 0


class EMAMultiStrategy(bt.Strategy):
    """
    EMA Multi-Timeframe + VWAP + StochRSI + CMF Strategy
    Strategy Type: TREND + MOMENTUM + VOLUME
    =============================================
    This strategy uses multiple EMAs (5, 9, 13, 21) for trend identification,
    VWAP for price context, StochRSI for momentum, and CMF for volume confirmation.

    Strategy Logic:
    ==============
    Long Entry (Two scenarios):
    Pullback: EMA alignment (5>9>13>21) + price near 5-EMA support + StochRSI turning up + CMF > 0
    Breakout: EMA alignment + price above VWAP + StochRSI bullish crossover (>30) + CMF > 0

    Short Entry (Two scenarios):
    Bounce: EMA alignment (5<9<13<21) + price near 5-EMA resistance + StochRSI turning down + CMF < 0
    Breakdown: EMA alignment + price below VWAP + StochRSI bearish crossover (<70) + CMF < 0

    Exit Conditions:
    - EMA crossover reversal (5 EMA crosses against trend)
    - StochRSI extreme reversal
    - Price crosses VWAP against position

    Parameters:
    ==========
    - ema_fast (int): Fastest EMA period (default: 5)
    - ema_med1 (int): Medium EMA 1 period (default: 9)
    - ema_med2 (int): Medium EMA 2 period (default: 13)
    - ema_slow (int): Slowest EMA period (default: 21)
    - stochrsi_period (int): StochRSI period (default: 14)
    - cmf_period (int): CMF period (default: 20)
    - stochrsi_oversold (float): StochRSI oversold level (default: 20)
    - stochrsi_overbought (float): StochRSI overbought level (default: 80)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("ema_fast", 5),
        ("ema_med1", 9),
        ("ema_med2", 13),
        ("ema_slow", 21),
        ("stochrsi_period", 14),
        ("cmf_period", 20),
        ("stochrsi_oversold", 20),
        ("stochrsi_overbought", 80),
        ("verbose", False),
    )

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

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize EMA indicators
        self.ema_fast = btind.EMA(self.data, period=self.params.ema_fast)
        self.ema_med1 = btind.EMA(self.data, period=self.params.ema_med1)
        self.ema_med2 = btind.EMA(self.data, period=self.params.ema_med2)
        self.ema_slow = btind.EMA(self.data, period=self.params.ema_slow)

        # Initialize complementary indicators
        self.vwap = VWAP(self.data)
        self.stochrsi = StochRSI(self.data, rsi_period=self.params.stochrsi_period)
        self.cmf = ChaikinMoneyFlow(self.data, period=self.params.cmf_period)

        # Additional momentum indicators
        self.atr = btind.ATR(self.data, period=14)
        self.adx = btind.ADX(self.data, period=14)

        # Strategy state variables
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.ema_slow,
                self.params.stochrsi_period,
                self.params.cmf_period,
            )
            + 5
        )

        # Data storage
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        # Previous values for trend detection
        self.prev_stochrsi = None
        self.prev_ema_fast = None

        logger.info(
            f"EMAMultiStrategy initialized with EMAs: {self.p.ema_fast}, "
            f"{self.p.ema_med1}, {self.p.ema_med2}, {self.p.ema_slow}"
        )

    def next(self):
        if len(self) < self.warmup_period:
            logger.debug(
                f"Skipping bar {len(self)}: still in warmup period (need {self.warmup_period} bars)"
            )
            return

        if not self.ready:
            self.ready = True
            logger.info(f"Strategy ready at bar {len(self)}")

        bar_time = self.datas[0].datetime.datetime(0)
        bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
        current_time = bar_time_ist.time()

        # Force close positions at 15:15 IST
        if current_time >= datetime.time(15, 15):
            if self.position:
                self.close()
                trade_logger.info("Force closed all positions at 15:15 IST")
            return

        # Only trade during market hours (9:15 AM to 3:05 PM IST)
        if not (datetime.time(9, 15) <= current_time <= datetime.time(15, 5)):
            return

        if self.order:
            logger.debug(f"Order pending at bar {len(self)}")
            return

        # Check for invalid indicator values
        if (
            np.isnan(self.ema_fast[0])
            or np.isnan(self.ema_med1[0])
            or np.isnan(self.ema_med2[0])
            or np.isnan(self.ema_slow[0])
            or np.isnan(self.vwap[0])
            or np.isnan(self.stochrsi.stochrsi[0])
            or np.isnan(self.stochrsi.stochrsi_signal[0])
            or np.isnan(self.cmf[0])
        ):
            logger.debug(f"Invalid indicator values at bar {len(self)}")
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "ema_fast": self.ema_fast[0],
                "ema_med1": self.ema_med1[0],
                "ema_med2": self.ema_med2[0],
                "ema_slow": self.ema_slow[0],
                "vwap": self.vwap[0],
                "stochrsi": self.stochrsi.stochrsi[0],
                "stochrsi_signal": self.stochrsi.stochrsi_signal[0],
                "cmf": self.cmf[0],
                "atr": self.atr[0],
                "adx": self.adx[0],
            }
        )

        # EMA alignment checks
        price = self.data.close[0]
        bullish_ema = (
            self.ema_fast[0] > self.ema_med1[0] > self.ema_med2[0] > self.ema_slow[0]
        )
        bearish_ema = (
            self.ema_fast[0] < self.ema_med1[0] < self.ema_med2[0] < self.ema_slow[0]
        )

        # StochRSI signals
        stochrsi_bullish_crossover = (
            self.stochrsi.stochrsi[0] > self.stochrsi.stochrsi_signal[0]
            and self.stochrsi.stochrsi[-1] <= self.stochrsi.stochrsi_signal[-1]
            and self.stochrsi.stochrsi[0] > self.params.stochrsi_oversold
        )
        stochrsi_bearish_crossover = (
            self.stochrsi.stochrsi[0] < self.stochrsi.stochrsi_signal[0]
            and self.stochrsi.stochrsi[-1] >= self.stochrsi.stochrsi_signal[-1]
            and self.stochrsi.stochrsi[0] < self.params.stochrsi_overbought
        )
        stochrsi_turning_up = (
            self.stochrsi.stochrsi[0] > self.stochrsi.stochrsi[-1]
            and self.stochrsi.stochrsi[0] < self.params.stochrsi_overbought
        )
        stochrsi_turning_down = (
            self.stochrsi.stochrsi[0] < self.stochrsi.stochrsi[-1]
            and self.stochrsi.stochrsi[0] > self.params.stochrsi_oversold
        )

        # Price near 5-EMA for pullback/bounce
        atr = self.atr[0]
        price_near_ema_fast = (
            abs(price - self.ema_fast[0]) <= 0.5 * atr
        )  # Within 0.5 ATR of 5-EMA

        # CMF conditions
        cmf_bullish = self.cmf[0] > 0
        cmf_bearish = self.cmf[0] < 0

        # VWAP conditions
        price_above_vwap = price > self.vwap[0]
        price_below_vwap = price < self.vwap[0]

        # Trend strength (ADX > 20 indicates trending market)
        strong_trend = self.adx[0] > 20

        if not self.position:
            # Long Entry: Pullback Scenario
            if (
                bullish_ema
                and price_near_ema_fast
                and price > self.vwap[0]  # Ensure price is above VWAP
                and stochrsi_turning_up
                and cmf_bullish
                and strong_trend
            ):
                self.order = self.buy()
                self.order_type = "enter_long_pullback"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Pullback) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {price:.2f} | "
                    f"EMA Fast: {self.ema_fast[0]:.2f} | VWAP: {self.vwap[0]:.2f} | "
                    f"StochRSI: {self.stochrsi.stochrsi[0]:.2f} | CMF: {self.cmf[0]:.4f}"
                )

            # Long Entry: Breakout Scenario
            elif (
                bullish_ema
                and price_above_vwap
                and stochrsi_bullish_crossover
                and cmf_bullish
                and strong_trend
            ):
                self.order = self.buy()
                self.order_type = "enter_long_breakout"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Breakout) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {price:.2f} | "
                    f"EMA Fast: {self.ema_fast[0]:.2f} | VWAP: {self.vwap[0]:.2f} | "
                    f"StochRSI: {self.stochrsi.stochrsi[0]:.2f} | CMF: {self.cmf[0]:.4f}"
                )

            # Short Entry: Bounce Scenario
            elif (
                bearish_ema
                and price_near_ema_fast
                and price < self.vwap[0]  # Ensure price is below VWAP
                and stochrsi_turning_down
                and cmf_bearish
                and strong_trend
            ):
                self.order = self.sell()
                self.order_type = "enter_short_bounce"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Bounce) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {price:.2f} | "
                    f"EMA Fast: {self.ema_fast[0]:.2f} | VWAP: {self.vwap[0]:.2f} | "
                    f"StochRSI: {self.stochrsi.stochrsi[0]:.2f} | CMF: {self.cmf[0]:.4f}"
                )

            # Short Entry: Breakdown Scenario
            elif (
                bearish_ema
                and price_below_vwap
                and stochrsi_bearish_crossover
                and cmf_bearish
                and strong_trend
            ):
                self.order = self.sell()
                self.order_type = "enter_short_breakdown"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Breakdown) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {price:.2f} | "
                    f"EMA Fast: {self.ema_fast[0]:.2f} | VWAP: {self.vwap[0]:.2f} | "
                    f"StochRSI: {self.stochrsi.stochrsi[0]:.2f} | CMF: {self.cmf[0]:.4f}"
                )

        elif self.position.size > 0:  # Long position
            # Exit: EMA reversal, StochRSI overbought, or price below VWAP
            ema_reversal = self.ema_fast[0] < self.ema_med1[0]
            stochrsi_overbought = (
                self.stochrsi.stochrsi[0] >= self.params.stochrsi_overbought
            )
            if ema_reversal or stochrsi_overbought or price_below_vwap:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "EMA Reversal"
                    if ema_reversal
                    else (
                        "StochRSI Overbought"
                        if stochrsi_overbought
                        else "Price Below VWAP"
                    )
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - {exit_reason}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {price:.2f} | "
                    f"StochRSI: {self.stochrsi.stochrsi[0]:.2f} | CMF: {self.cmf[0]:.4f}"
                )

        elif self.position.size < 0:  # Short position
            # Exit: EMA reversal, StochRSI oversold, or price above VWAP
            ema_reversal = self.ema_fast[0] > self.ema_med1[0]
            stochrsi_oversold = (
                self.stochrsi.stochrsi[0] <= self.params.stochrsi_oversold
            )
            if ema_reversal or stochrsi_oversold or price_above_vwap:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "EMA Reversal"
                    if ema_reversal
                    else (
                        "StochRSI Oversold" if stochrsi_oversold else "Price Above VWAP"
                    )
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - {exit_reason}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {price:.2f} | "
                    f"StochRSI: {self.stochrsi.stochrsi[0]:.2f} | CMF: {self.cmf[0]:.4f}"
                )

        # Update previous values
        self.prev_stochrsi = self.stochrsi.stochrsi[0]
        self.prev_ema_fast = self.ema_fast[0]

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt)
            if exec_dt.tzinfo is None:
                exec_dt = exec_dt.replace(tzinfo=pytz.UTC)
            exec_dt = exec_dt.astimezone(pytz.timezone("Asia/Kolkata"))

            if (
                self.order_type in ["enter_long_pullback", "enter_long_breakout"]
                and order.isbuy()
            ):
                position_info = {
                    "entry_time": exec_dt,
                    "entry_price": order.executed.price,
                    "size": order.executed.size,
                    "commission": order.executed.comm,
                    "ref": order.ref,
                    "direction": "long",
                }
                self.open_positions.append(position_info)
                trade_logger.info(
                    f"BUY EXECUTED (Enter Long) | Ref: {order.ref} | "
                    f"Price: {order.executed.price:.2f} | Size: {order.executed.size}"
                )

            elif (
                self.order_type in ["enter_short_bounce", "enter_short_breakdown"]
                and order.issell()
            ):
                position_info = {
                    "entry_time": exec_dt,
                    "entry_price": order.executed.price,
                    "size": order.executed.size,
                    "commission": order.executed.comm,
                    "ref": order.ref,
                    "direction": "short",
                }
                self.open_positions.append(position_info)
                trade_logger.info(
                    f"SELL EXECUTED (Enter Short) | Ref: {order.ref} | "
                    f"Price: {order.executed.price:.2f} | Size: {order.executed.size}"
                )

            elif self.order_type in ["exit_long", "exit_short"]:
                if self.open_positions:
                    entry_info = self.open_positions.pop(0)
                    size = abs(entry_info["size"]) if entry_info["size"] != 0 else 1

                    if self.order_type == "exit_long":
                        pnl = (order.executed.price - entry_info["entry_price"]) * size
                    else:  # exit_short
                        pnl = (entry_info["entry_price"] - order.executed.price) * size

                    total_commission = entry_info["commission"] + abs(
                        order.executed.comm
                    )
                    pnl_net = pnl - total_commission

                    trade_info = {
                        "ref": order.ref,
                        "entry_time": entry_info["entry_time"],
                        "exit_time": exec_dt,
                        "entry_price": entry_info["entry_price"],
                        "exit_price": order.executed.price,
                        "size": size,
                        "pnl": pnl,
                        "pnl_net": pnl_net,
                        "commission": total_commission,
                        "status": "Won" if pnl > 0 else "Lost",
                        "direction": entry_info["direction"].title(),
                        "bars_held": max(
                            1, (exec_dt - entry_info["entry_time"]).total_seconds() / 60
                        ),
                    }
                    self.completed_trades.append(trade_info)
                    self.trade_count += 1

                    trade_logger.info(
                        f"{'SELL' if self.order_type == 'exit_long' else 'BUY'} EXECUTED "
                        f"(Exit {entry_info['direction'].title()}) | Ref: {order.ref} | "
                        f"Price: {order.executed.price:.2f} | PnL: {pnl:.2f} | Net PnL: {pnl_net:.2f}"
                    )

        if order.status in [
            order.Completed,
            order.Canceled,
            order.Margin,
            order.Rejected,
        ]:
            self.order = None
            self.order_type = None

    def notify_trade(self, trade):
        if trade.isclosed:
            trade_logger.info(
                f"TRADE CLOSED | Ref: {trade.ref} | "
                f"Profit: {trade.pnl:.2f} | Net Profit: {trade.pnlcomm:.2f} | "
                f"Bars Held: {trade.barlen} | Trade Count: {self.trade_count}"
            )

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        params = {
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
        return params

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
