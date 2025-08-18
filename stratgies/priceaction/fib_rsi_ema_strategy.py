import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class StochasticRSI(bt.Indicator):
    lines = ("percK", "percD")
    params = (
        ("period", 14),  # RSI period
        ("period_k", 14),  # Stochastic %K period
        ("period_d", 3),  # Stochastic %D period
    )

    def __init__(self):
        self.rsi = btind.RSI(self.data, period=self.params.period)
        self.rsi_high = btind.Highest(self.rsi, period=self.params.period_k)
        self.rsi_low = btind.Lowest(self.rsi, period=self.params.period_k)
        self.lines.percK = (
            (self.rsi - self.rsi_low) / (self.rsi_high - self.rsi_low)
        ) * 100
        self.lines.percD = btind.SMA(self.lines.percK, period=self.params.period_d)

    def next(self):
        # Handle edge case where rsi_high == rsi_low to avoid division by zero
        if self.rsi_high[0] == self.rsi_low[0]:
            self.lines.percK[0] = 50.0  # Neutral value when no range
        if np.isnan(self.lines.percK[0]) or np.isnan(self.rsi[0]):
            self.lines.percK[0] = 50.0
        if np.isnan(self.lines.percD[0]):
            self.lines.percD[0] = 50.0


class FibonacciLevels(bt.Indicator):
    """
    Enhanced Fibonacci Retracement Indicator
    Calculates Fib levels from yesterday's high/low with validation checks
    """

    lines = ("fib_382", "fib_50", "fib_618", "high", "low")
    params = (("lookback", 1), ("tolerance", 0.002))  # Days to look back for high/low

    def __init__(self):
        self.yesterday_high = 0.0
        self.yesterday_low = 0.0

    def next(self):
        if len(self) < 2:
            return

        # Get yesterday's high and low
        lookback_period = min(len(self), 288)  # Approx 1 day of 5-min bars

        if lookback_period > 1:
            self.yesterday_high = max(
                [self.data.high[-i] for i in range(1, lookback_period)]
            )
            self.yesterday_low = min(
                [self.data.low[-i] for i in range(1, lookback_period)]
            )

            # Calculate Fibonacci levels
            range_val = self.yesterday_high - self.yesterday_low

            self.lines.high[0] = self.yesterday_high
            self.lines.low[0] = self.yesterday_low
            self.lines.fib_382[0] = self.yesterday_high - (0.382 * range_val)
            self.lines.fib_50[0] = self.yesterday_high - (0.5 * range_val)
            self.lines.fib_618[0] = self.yesterday_high - (0.618 * range_val)

    def is_valid_level(self, price, direction):
        """Check if Fibonacci level has price action confirmation"""
        tol = self.params.tolerance
        prev_low = self.data.low[-1]
        prev_high = self.data.high[-1]
        prev_close = self.data.close[-1]
        prev_open = self.data.open[-1]

        for level in [
            self.lines.fib_382[0],
            self.lines.fib_50[0],
            self.lines.fib_618[0],
        ]:
            if abs(price - level) / level > tol:
                continue

            # Bullish validation (support)
            if direction == "long":
                # Rejection wick below level or bullish engulfing
                if (
                    prev_low <= level + tol
                    and min(prev_close, prev_open) > level
                    and prev_close > prev_open
                ):
                    return True

            # Bearish validation (resistance)
            elif direction == "short":
                # Rejection wick above level or bearish engulfing
                if (
                    prev_high >= level - tol
                    and max(prev_close, prev_open) < level
                    and prev_close < prev_open
                ):
                    return True

        return False


class VWAP(bt.Indicator):
    """Volume Weighted Average Price Indicator"""

    lines = ("vwap",)

    def __init__(self):
        self.addminperiod(1)

    def next(self):
        if len(self) == 1:
            self.lines.vwap[0] = self.data.close[0]
            self.cum_volume = self.data.volume[0]
            self.cum_pv = self.data.close[0] * self.data.volume[0]
        else:
            # Reset VWAP daily
            if len(self) % 288 == 1:
                self.cum_volume = self.data.volume[0]
                self.cum_pv = self.data.close[0] * self.data.volume[0]
            else:
                self.cum_volume += self.data.volume[0]
                self.cum_pv += self.data.close[0] * self.data.volume[0]

            self.lines.vwap[0] = (
                self.cum_pv / self.cum_volume
                if self.cum_volume > 0
                else self.data.close[0]
            )


class FibRSIEMAIntraday(bt.Strategy):
    """
    Enhanced Fibonacci + RSI + EMA Intraday Strategy
    ===============================================
    Major Improvements:
    1. Added Stochastic RSI and MACD for entry confirmation
    2. ATR-based dynamic stop loss and targets
    3. ADX filter for trend strength
    4. Volume filter for entry validity
    5. Price action validation at Fibonacci levels
    6. Time-based parameter adjustments
    """

    params = (
        ("rsi_period", 14),
        ("ema_period", 20),
        ("rsi_long_entry", 52),
        ("rsi_short_entry", 48),
        ("rsi_overbought", 75),
        ("rsi_oversold", 25),
        ("atr_multiplier", 2.0),  # ATR multiplier for stop loss
        ("atr_target_multiplier", 3.0),  # ATR multiplier for targets
        ("adx_threshold", 25),  # Minimum ADX for trending markets
        ("volume_multiplier", 1.2),  # Volume vs MA multiplier
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 2},
        "ema_period": {"type": "int", "low": 15, "high": 25, "step": 5},
        "rsi_long_entry": {"type": "int", "low": 35, "high": 45, "step": 5},
        "rsi_short_entry": {"type": "int", "low": 55, "high": 65, "step": 5},
        "atr_multiplier": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.25},
        "atr_target_multiplier": {
            "type": "float",
            "low": 2.0,
            "high": 4.0,
            "step": 0.5,
        },
        "adx_threshold": {"type": "int", "low": 20, "high": 30, "step": 2},
        "volume_multiplier": {"type": "float", "low": 1.1, "high": 1.5, "step": 0.1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Core Indicators
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.ema = btind.EMA(self.data.close, period=self.params.ema_period)
        self.vwap = VWAP(self.data)
        self.fib = FibonacciLevels(self.data)

        # Confirmation Indicators
        self.stoch_rsi = StochasticRSI(self.data.close, period=14)
        self.macd = btind.MACD(self.data.close)
        self.atr = btind.ATR(self.data, period=14)
        self.adx = btind.ADX(self.data, period=14)
        self.vol_ma = btind.SMA(self.data.volume, period=20)

        # Trade management
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(self.params.rsi_period, self.params.ema_period, 50) + 50  # For ADX/ATR
        )
        self.completed_trades = []
        self.open_positions = []
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

        logger.info(
            f"Initialized Enhanced FibRSIEMAIntraday with params: {self.params}"
        )

    def is_at_fib_level(self, price, direction):
        """Check if price is near validated Fibonacci level"""
        return self.fib.is_valid_level(price, direction)

    def next(self):
        if len(self) < self.warmup_period:
            return

        if not self.ready:
            self.ready = True
            logger.info(f"Strategy ready at bar {len(self)}")

        bar_time = self.datas[0].datetime.datetime(0)
        bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
        current_time = bar_time_ist.time()
        hour = bar_time_ist.hour

        # Force close at 3:15 PM IST
        if current_time >= datetime.time(15, 15):
            if self.position:
                self.close()
                trade_logger.info("Force closed all positions at 15:15 IST")
            return

        # Trading hours: 9:15 AM - 3:05 PM IST
        if not (datetime.time(9, 15) <= current_time <= datetime.time(15, 5)):
            return

        # Avoid early market noise (first 15 minutes)
        if hour == 9 and bar_time_ist.minute < 30:
            return

        # Afternoon parameter adjustments
        if hour >= 13:  # After 1 PM
            rsi_long_entry = 45
            rsi_short_entry = 55
            atr_target_multiplier = self.params.atr_target_multiplier * 0.8
        else:
            rsi_long_entry = self.params.rsi_long_entry
            rsi_short_entry = self.params.rsi_short_entry
            atr_target_multiplier = self.params.atr_target_multiplier

        # Market regime filters
        if self.adx[0] < self.params.adx_threshold:
            return  # Skip choppy markets

        if self.data.volume[0] < self.vol_ma[0] * self.params.volume_multiplier:
            return  # Require above-average volume

        # Check for NaN values
        nan_check = (
            np.isnan(self.rsi[0])
            or np.isnan(self.ema[0])
            or np.isnan(self.vwap[0])
            or np.isnan(self.fib.fib_50[0])
            or np.isnan(self.stoch_rsi[0])
            or np.isnan(self.macd.macd[0])
            or np.isnan(self.atr[0])
            or np.isnan(self.adx[0])
        )
        if nan_check:
            return

        current_price = self.data.close[0]

        if not self.position:
            # Long Entry Conditions
            long_conditions = (
                current_price > self.vwap[0]  # Above VWAP
                and self.is_at_fib_level(current_price, "long")  # Validated Fib support
                and self.rsi[0] > rsi_long_entry  # RSI bounce
                and current_price > self.ema[0]  # Above EMA
                and self.stoch_rsi.percK[0] > 30  # Stochastic RSI confirmation
                and self.macd.macd[0] > self.macd.signal[0]  # MACD bullish
            )

            # Short Entry Conditions
            short_conditions = (
                current_price < self.vwap[0]  # Below VWAP
                and self.is_at_fib_level(
                    current_price, "short"
                )  # Validated Fib resistance
                and self.rsi[0] < rsi_short_entry  # RSI rejection
                and current_price < self.ema[0]  # Below EMA
                and self.stoch_rsi.percK[0] < 70  # Stochastic RSI confirmation
                and self.macd.macd[0] < self.macd.signal[0]  # MACD bearish
            )

            if long_conditions:
                # Dynamic position sizing based on volatility
                atr_val = self.atr[0]
                self.stop_price = current_price - (atr_val * self.params.atr_multiplier)
                self.target_price = current_price + (atr_val * atr_target_multiplier)

                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL | Time: {bar_time_ist} | Price: {current_price:.2f} | "
                    f"Stop: {self.stop_price:.2f} | Target: {self.target_price:.2f} | "
                    f"ATR: {atr_val:.2f}"
                )

            elif short_conditions:
                atr_val = self.atr[0]
                self.stop_price = current_price + (atr_val * self.params.atr_multiplier)
                self.target_price = current_price - (atr_val * atr_target_multiplier)

                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL | Time: {bar_time_ist} | Price: {current_price:.2f} | "
                    f"Stop: {self.stop_price:.2f} | Target: {self.target_price:.2f} | "
                    f"ATR: {atr_val:.2f}"
                )
        else:
            # Exit Logic
            if self.position.size > 0:  # Long position
                exit_conditions = (
                    current_price >= self.target_price,
                    current_price <= self.stop_price,
                    self.rsi[0] > self.params.rsi_overbought,
                    hour >= 14 and self.rsi[0] < 50,  # Early exit if weak in afternoon
                )

                if any(exit_conditions):
                    self.order = self.sell()
                    self.order_type = "exit_long"
                    exit_reason = (
                        "Target"
                        if current_price >= self.target_price
                        else (
                            "Stop"
                            if current_price <= self.stop_price
                            else (
                                "RSI_OB"
                                if self.rsi[0] > self.params.rsi_overbought
                                else "Afternoon_Weakness"
                            )
                        )
                    )
                    trade_logger.info(
                        f"SELL SIGNAL (Exit Long - {exit_reason}) | Time: {bar_time_ist} | Price: {current_price:.2f}"
                    )

            elif self.position.size < 0:  # Short position
                exit_conditions = (
                    current_price <= self.target_price,
                    current_price >= self.stop_price,
                    self.rsi[0] < self.params.rsi_oversold,
                    hour >= 14 and self.rsi[0] > 50,  # Early exit if weak in afternoon
                )

                if any(exit_conditions):
                    self.order = self.buy()
                    self.order_type = "exit_short"
                    exit_reason = (
                        "Target"
                        if current_price <= self.target_price
                        else (
                            "Stop"
                            if current_price >= self.stop_price
                            else (
                                "RSI_OS"
                                if self.rsi[0] < self.params.rsi_oversold
                                else "Afternoon_Weakness"
                            )
                        )
                    )
                    trade_logger.info(
                        f"BUY SIGNAL (Exit Short - {exit_reason}) | Time: {bar_time_ist} | Price: {current_price:.2f}"
                    )

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt).astimezone(
                pytz.timezone("Asia/Kolkata")
            )

            if self.order_type == "enter_long" and order.isbuy():
                self.entry_price = order.executed.price
                self.open_positions.append(
                    {
                        "entry_time": exec_dt,
                        "entry_price": order.executed.price,
                        "size": order.executed.size,
                        "commission": order.executed.comm,
                        "ref": order.ref,
                        "direction": "long",
                        "stop_price": self.stop_price,
                        "target_price": self.target_price,
                        "atr": self.atr[0],
                    }
                )
                trade_logger.info(
                    f"BUY EXECUTED (Enter Long) | Ref: {order.ref} | Price: {order.executed.price:.2f}"
                )

            elif self.order_type == "enter_short" and order.issell():
                self.entry_price = order.executed.price
                self.open_positions.append(
                    {
                        "entry_time": exec_dt,
                        "entry_price": order.executed.price,
                        "size": order.executed.size,
                        "commission": order.executed.comm,
                        "ref": order.ref,
                        "direction": "short",
                        "stop_price": self.stop_price,
                        "target_price": self.target_price,
                        "atr": self.atr[0],
                    }
                )
                trade_logger.info(
                    f"SELL EXECUTED (Enter Short) | Ref: {order.ref} | Price: {order.executed.price:.2f}"
                )

            elif self.order_type == "exit_long" and order.issell():
                if self.open_positions:
                    entry_info = self.open_positions.pop(0)
                    pnl = (order.executed.price - entry_info["entry_price"]) * abs(
                        entry_info["size"]
                    )
                    total_commission = entry_info["commission"] + abs(
                        order.executed.comm
                    )

                    # Trade quality metrics
                    trade_duration = (
                        exec_dt - entry_info["entry_time"]
                    ).total_seconds() / 60
                    quality_score = min(1.0, trade_duration / 30) * min(
                        1.5, pnl / (entry_info["atr"] * 0.5)
                    )

                    self.completed_trades.append(
                        {
                            "ref": order.ref,
                            "entry_time": entry_info["entry_time"],
                            "exit_time": exec_dt,
                            "entry_price": entry_info["entry_price"],
                            "exit_price": order.executed.price,
                            "size": abs(entry_info["size"]),
                            "pnl": pnl,
                            "pnl_net": pnl - total_commission,
                            "commission": total_commission,
                            "status": "Won" if pnl > 0 else "Lost",
                            "direction": "Long",
                            "duration_min": trade_duration,
                            "quality_score": quality_score,
                        }
                    )
                    self.trade_count += 1
                    trade_logger.info(
                        f"SELL EXECUTED (Exit Long) | Ref: {order.ref} | PnL: {pnl:.2f} | "
                        f"Quality: {quality_score:.2f}"
                    )

            elif self.order_type == "exit_short" and order.isbuy():
                if self.open_positions:
                    entry_info = self.open_positions.pop(0)
                    pnl = (entry_info["entry_price"] - order.executed.price) * abs(
                        entry_info["size"]
                    )
                    total_commission = entry_info["commission"] + abs(
                        order.executed.comm
                    )

                    # Trade quality metrics
                    trade_duration = (
                        exec_dt - entry_info["entry_time"]
                    ).total_seconds() / 60
                    quality_score = min(1.0, trade_duration / 30) * min(
                        1.5, pnl / (entry_info["atr"] * 0.5)
                    )

                    self.completed_trades.append(
                        {
                            "ref": order.ref,
                            "entry_time": entry_info["entry_time"],
                            "exit_time": exec_dt,
                            "entry_price": entry_info["entry_price"],
                            "exit_price": order.executed.price,
                            "size": abs(entry_info["size"]),
                            "pnl": pnl,
                            "pnl_net": pnl - total_commission,
                            "commission": total_commission,
                            "status": "Won" if pnl > 0 else "Lost",
                            "direction": "Short",
                            "duration_min": trade_duration,
                            "quality_score": quality_score,
                        }
                    )
                    self.trade_count += 1
                    trade_logger.info(
                        f"BUY EXECUTED (Exit Short) | Ref: {order.ref} | PnL: {pnl:.2f} | "
                        f"Quality: {quality_score:.2f}"
                    )

        if order.status in [
            order.Completed,
            order.Canceled,
            order.Margin,
            order.Rejected,
        ]:
            if order.status in [order.Completed] and self.order_type in [
                "exit_long",
                "exit_short",
            ]:
                self.entry_price = None
                self.stop_price = None
                self.target_price = None
            self.order = None
            self.order_type = None

    def notify_trade(self, trade):
        if trade.isclosed:
            trade_logger.info(
                f"TRADE CLOSED | Ref: {trade.ref} | Profit: {trade.pnl:.2f} | Net Profit: {trade.pnlcomm:.2f}"
            )

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        return {
            "rsi_period": trial.suggest_int("rsi_period", 10, 20, step=2),
            "ema_period": trial.suggest_int("ema_period", 15, 25, step=5),
            "rsi_long_entry": trial.suggest_int("rsi_long_entry", 35, 45, step=5),
            "rsi_short_entry": trial.suggest_int("rsi_short_entry", 55, 65, step=5),
            "atr_multiplier": trial.suggest_float(
                "atr_multiplier", 1.5, 3.0, step=0.25
            ),
            "atr_target_multiplier": trial.suggest_float(
                "atr_target_multiplier", 2.0, 4.0, step=0.5
            ),
            "adx_threshold": trial.suggest_int("adx_threshold", 20, 30, step=2),
            "volume_multiplier": trial.suggest_float(
                "volume_multiplier", 1.1, 1.5, step=0.1
            ),
        }

    @classmethod
    def get_min_data_points(cls, params):
        try:
            return (
                max(
                    params.get("rsi_period", 14),
                    params.get("ema_period", 20),
                    50,  # For ADX/ATR
                )
                + 70
            )
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 100
