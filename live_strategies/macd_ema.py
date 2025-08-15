import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class MACDEMA(bt.Strategy):
    """
    MACD + EMA Trend Confirmation Strategy

    This strategy combines MACD (Moving Average Convergence Divergence) crossover signals with EMA
    (Exponential Moving Average) trend confirmation to identify high-probability trades that occur
    only when momentum signals align with the underlying trend direction.

    Strategy Type: MOMENTUM + TREND CONFIRMATION
    ===========================================
    This strategy uses MACD crossovers to identify momentum changes and EMA to confirm that
    price is moving in the direction of the underlying trend. It only enters trades when both
    momentum crossover and trend direction align, and exits when either signal reverses.

    Strategy Logic:
    ==============

    Long Position Rules:
    - Entry: MACD line crosses above signal line (bullish crossover) AND price > EMA (uptrend)
    - Exit: MACD line crosses below signal line (bearish crossover) OR price < EMA (downtrend)

    Short Position Rules:
    - Entry: MACD line crosses below signal line (bearish crossover) AND price < EMA (downtrend)
    - Exit: MACD line crosses above signal line (bullish crossover) OR price > EMA (uptrend)

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses warmup period to ensure indicator stability before trading
    - Prevents order overlap with pending order checks
    - Trend confirmation reduces false breakouts and whipsaws

    Indicators Used:
    ===============
    - MACD (Moving Average Convergence Divergence):
      * MACD Line: 12-period EMA - 26-period EMA
      * Signal Line: 9-period EMA of MACD line
      * Histogram: MACD line - Signal line
      * Crossover above signal = Bullish momentum
      * Crossover below signal = Bearish momentum

    - EMA (Exponential Moving Average):
      * Trend filter based on price position relative to EMA
      * Price > EMA = Uptrend confirmation
      * Price < EMA = Downtrend confirmation
      * Reacts faster to price changes than SMA

    Trend Confirmation Concept:
    ==========================
    - MACD bullish crossover + Price > EMA = Strong upward momentum with trend support
    - MACD bearish crossover + Price < EMA = Strong downward momentum with trend support
    - MACD crossover without EMA confirmation = Filtered out (potential false signal)
    - EMA acts as dynamic support/resistance level

    Features:
    =========
    - Comprehensive trade logging with IST timezone
    - Detailed PnL tracking for each completed trade
    - Position sizing and commission handling
    - Optimization-ready parameter space
    - Robust error handling and data validation
    - Support for both backtesting and live trading
    - MACD histogram analysis for momentum strength

    Parameters:
    ==========
    - fast_period (int): MACD fast EMA period (default: 12)
    - slow_period (int): MACD slow EMA period (default: 26)
    - signal_period (int): MACD signal line period (default: 9)
    - ema_period (int): EMA trend filter period (default: 50)
    - min_histogram (float): Minimum MACD histogram value for entry (default: 0.0)
    - verbose (bool): Enable detailed logging (default: False)

    Performance Metrics:
    ===================
    - Tracks win/loss ratio
    - Calculates net PnL including commissions
    - Records trade duration and timing
    - Provides detailed execution logs
    - Monitors MACD crossover frequency and success rate

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MACDEMA, ema_period=30, fast_period=8, slow_period=21)
    cerebro.run()

    Best Market Conditions:
    ======================
    - Trending markets with sustained directional moves
    - Markets with clear momentum shifts and trend alignment
    - Volatile markets where MACD can capture momentum changes
    - Avoid during choppy, sideways markets with frequent false crossovers

    Note:
    ====
    This strategy is selective as it requires both momentum crossover (MACD) and trend
    confirmation (EMA) to align. It prioritizes quality over quantity of trades,
    focusing on high-probability setups with trend support.
    """

    params = (
        ("fast_period", 12),
        ("slow_period", 26),
        ("signal_period", 9),
        ("ema_period", 50),
        ("min_histogram", 0.0),
        ("verbose", False),
    )

    optimization_params = {
        "fast_period": {"type": "int", "low": 8, "high": 16, "step": 1},
        "slow_period": {"type": "int", "low": 20, "high": 35, "step": 1},
        "signal_period": {"type": "int", "low": 6, "high": 12, "step": 1},
        "ema_period": {"type": "int", "low": 20, "high": 100, "step": 5},
        "min_histogram": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.05},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize MACD indicator
        self.macd = btind.MACD(
            self.data.close,
            period_me1=self.params.fast_period,
            period_me2=self.params.slow_period,
            period_signal=self.params.signal_period,
        )

        # Initialize EMA trend filter
        self.ema = btind.EMA(self.data.close, period=self.params.ema_period)

        # Calculate MACD histogram manually (since .histo doesn't exist)
        self.macd_histogram = self.macd.lines.macd - self.macd.lines.signal

        # MACD crossover signals - use single CrossOver
        self.macd_cross = btind.CrossOver(self.macd.lines.macd, self.macd.lines.signal)

        # Trend confirmation signals
        self.price_above_ema = self.data.close > self.ema
        self.price_below_ema = self.data.close < self.ema

        # Histogram strength filter
        self.macd_histogram = self.macd.lines.macd - self.macd.lines.signal
        self.strong_histogram = abs(self.macd_histogram) > self.params.min_histogram

        # Combined entry signals
        self.bullish_entry = bt.And(
            self.macd_cross > 0,  # Bullish crossover
            self.price_above_ema,
            self.strong_histogram,
        )
        self.bearish_entry = bt.And(
            self.macd_cross < 0,  # Bearish crossover
            self.price_below_ema,
            self.strong_histogram,
        )

        # Exit signals
        self.bullish_exit = bt.Or(self.macd_cross < 0, self.price_below_ema)
        self.bearish_exit = bt.Or(self.macd_cross > 0, self.price_above_ema)

        # Rest of initialization remains the same...
        self.order = None
        self.order_type = None
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.slow_period + self.params.signal_period,
                self.params.ema_period,
            )
            + 5  # Extra bars for MACD calculation stability
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized MACDEMA with params: {self.params}")
        logger.info(
            f"MACDEMA initialized with fast_period={self.p.fast_period}, "
            f"slow_period={self.p.slow_period}, signal_period={self.p.signal_period}, "
            f"ema_period={self.p.ema_period}, min_histogram={self.p.min_histogram}"
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
            np.isnan(self.macd.macd[0])
            or np.isnan(self.macd.signal[0])
            or np.isnan(self.ema[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"MACD={self.macd.macd[0]}, Signal={self.macd.signal[0]}, EMA={self.ema[0]}"
            )
            return

        # Calculate trend direction
        trend_direction = "NEUTRAL"
        if self.price_above_ema[0]:
            trend_direction = "UPTREND"
        elif self.price_below_ema[0]:
            trend_direction = "DOWNTREND"

        # Calculate momentum state
        momentum_state = "NEUTRAL"
        if self.macd.macd[0] > self.macd.signal[0]:
            momentum_state = "BULLISH"
        elif self.macd.macd[0] < self.macd.signal[0]:
            momentum_state = "BEARISH"

        # Calculate histogram strength
        histogram_strength = "WEAK"
        if abs(self.macd_histogram[0]) > self.params.min_histogram * 2:
            histogram_strength = "STRONG"
        elif abs(self.macd_histogram[0]) > self.params.min_histogram:
            histogram_strength = "MODERATE"

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "macd": self.macd.macd[0],
                "signal": self.macd.signal[0],
                "histogram": self.macd_histogram[0],
                "ema": self.ema[0],
                "price_above_ema": self.price_above_ema[0],
                "price_below_ema": self.price_below_ema[0],
                "macd_bullish_cross": self.macd_cross[0] > 0,  # Fixed reference
                "macd_bearish_cross": self.macd_cross[0] < 0,  # Fixed reference
                "bullish_entry": self.bullish_entry[0],
                "bearish_entry": self.bearish_entry[0],
                "trend_direction": trend_direction,
                "momentum_state": momentum_state,
                "histogram_strength": histogram_strength,
            }
        )

        # MACD + EMA Position Management
        if not self.position:
            # Long Entry: MACD bullish crossover + price above EMA
            if self.bullish_entry[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - MACD Bullish + EMA Uptrend) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} > Signal: {self.macd.signal[0]:.4f} (Bullish Cross) | "
                    f"EMA: {self.ema[0]:.2f} < Price (Uptrend) | "
                    f"Histogram: {self.macd_histogram[0]:.4f} | "
                    f"Trend: {trend_direction} | Momentum: {momentum_state} | "
                    f"Histogram Strength: {histogram_strength}"
                )
            # Short Entry: MACD bearish crossover + price below EMA
            elif self.bearish_entry[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - MACD Bearish + EMA Downtrend) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} < Signal: {self.macd.signal[0]:.4f} (Bearish Cross) | "
                    f"EMA: {self.ema[0]:.2f} > Price (Downtrend) | "
                    f"Histogram: {self.macd_histogram[0]:.4f} | "
                    f"Trend: {trend_direction} | Momentum: {momentum_state} | "
                    f"Histogram Strength: {histogram_strength}"
                )
        else:
            # Exit long position
            if self.position.size > 0 and self.bullish_exit[0]:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "MACD Bearish Cross"
                    if self.macd_cross[0] < 0
                    else "Price Below EMA"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - {exit_reason}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} | Signal: {self.macd.signal[0]:.4f} | "
                    f"EMA: {self.ema[0]:.2f} | "
                    f"Trend: {trend_direction} | Momentum: {momentum_state}"
                )
            # Exit short position
            elif self.position.size < 0 and self.bearish_exit[0]:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "MACD Bullish Cross"
                    if self.macd_cross[0] > 0
                    else "Price Above EMA"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - {exit_reason}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"posture_price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} | Signal: {self.macd.signal[0]:.4f} | "
                    f"EMA: {self.ema[0]:.2f} | "
                    f"Trend: {trend_direction} | Momentum: {momentum_state}"
                )

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt)
            if exec_dt.tzinfo is None:
                exec_dt = exec_dt.replace(tzinfo=pytz.UTC)

            if self.order_type == "enter_long" and order.isbuy():
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
                    f"Price: {order.executed.price:.2f} | "
                    f"Size: {order.executed.size} | "
                    f"Cost: {order.executed.value:.2f} | "
                    f"Comm: {order.executed.comm:.2f}"
                )
            elif self.order_type == "enter_short" and order.issell():
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
                    f"Price: {order.executed.price:.2f} | "
                    f"Size: {order.executed.size} | "
                    f"Cost: {order.executed.value:.2f} | "
                    f"Comm: {order.executed.comm:.2f}"
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
                    pnl_net = pnl - total_commission
                    trade_info = {
                        "ref": order.ref,
                        "entry_time": entry_info["entry_time"],
                        "exit_time": exec_dt,
                        "entry_price": entry_info["entry_price"],
                        "exit_price": order.executed.price,
                        "size": abs(entry_info["size"]),
                        "pnl": pnl,
                        "pnl_net": pnl_net,
                        "commission": total_commission,
                        "status": "Won" if pnl > 0 else "Lost",
                        "direction": "Long",
                        "bars_held": (exec_dt - entry_info["entry_time"]).days,
                    }
                    self.completed_trades.append(trade_info)
                    self.trade_count += 1
                    trade_logger.info(
                        f"SELL EXECUTED (Exit Long) | Ref: {order.ref} | "
                        f"Price: {order.executed.price:.2f} | "
                        f"Size: {order.executed.size} | "
                        f"Cost: {order.executed.value:.2f} | "
                        f"Comm: {order.executed.comm:.2f} | "
                        f"PnL: {pnl:.2f}"
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
                    pnl_net = pnl - total_commission
                    trade_info = {
                        "ref": order.ref,
                        "entry_time": entry_info["entry_time"],
                        "exit_time": exec_dt,
                        "entry_price": entry_info["entry_price"],
                        "exit_price": order.executed.price,
                        "size": abs(entry_info["size"]),
                        "pnl": pnl,
                        "pnl_net": pnl_net,
                        "commission": total_commission,
                        "status": "Won" if pnl > 0 else "Lost",
                        "direction": "Short",
                        "bars_held": (exec_dt - entry_info["entry_time"]).days,
                    }
                    self.completed_trades.append(trade_info)
                    self.trade_count += 1
                    trade_logger.info(
                        f"BUY EXECUTED (Exit Short) | Ref: {order.ref} | "
                        f"Price: {order.executed.price:.2f} | "
                        f"Size: {order.executed.size} | "
                        f"Cost: {order.executed.value:.2f} | "
                        f"Comm: {order.executed.comm:.2f} | "
                        f"PnL: {pnl:.2f}"
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
                f"Profit: {trade.pnl:.2f} | "
                f"Net Profit: {trade.pnlcomm:.2f} | "
                f"Bars Held: {trade.barlen} | "
                f"Trade Count: {self.trade_count}"
            )

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        params = {
            "fast_period": trial.suggest_int("fast_period", 8, 16),
            "slow_period": trial.suggest_int("slow_period", 20, 35),
            "signal_period": trial.suggest_int("signal_period", 6, 12),
            "ema_period": trial.suggest_int("ema_period", 20, 100),
            "min_histogram": trial.suggest_float("min_histogram", 0.0, 0.5),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            slow_period = params.get("slow_period", 26)
            signal_period = params.get("signal_period", 9)
            ema_period = params.get("ema_period", 50)
            max_period = max(slow_period + signal_period, ema_period)
            return max_period + 5  # Extra bars for MACD calculation
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 65
