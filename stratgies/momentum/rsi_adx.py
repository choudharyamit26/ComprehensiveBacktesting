import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class RSIADX(bt.Strategy):
    """
    RSI + ADX Strength Strategy

    This strategy combines RSI (Relative Strength Index) momentum signals with ADX
    (Average Directional Index) trend strength confirmation to identify high-probability
    trades that occur only when there's sufficient directional momentum in the market.

    Strategy Type: MOMENTUM + TREND STRENGTH CONFIRMATION
    ====================================================
    This strategy uses RSI to identify momentum direction and ADX to confirm that
    there's sufficient trend strength to support the momentum move. It only enters
    trades when both momentum and trend strength align, and exits when trend
    strength deteriorates regardless of momentum direction.

    Strategy Logic:
    ==============

    Long Position Rules:
    - Entry: RSI > bullish threshold AND ADX > strength threshold (strong uptrend)
    - Exit: ADX < strength threshold (trend strength weakening)

    Short Position Rules:
    - Entry: RSI < bearish threshold AND ADX > strength threshold (strong downtrend)
    - Exit: ADX < strength threshold (trend strength weakening)

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses warmup period to ensure indicator stability before trading
    - Prevents order overlap with pending order checks
    - Trend strength validation reduces false breakouts and whipsaws

    Indicators Used:
    ===============
    - RSI (Relative Strength Index):
      * Measures price momentum on 0-100 scale
      * > 55: Bullish momentum
      * < 45: Bearish momentum
      * 45-55: Neutral zone

    - ADX (Average Directional Index):
      * Measures trend strength on 0-100 scale
      * > 25: Strong trend (default threshold)
      * < 25: Weak trend or ranging market
      * > 50: Very strong trend
      * Independent of direction - only measures strength

    Trend Strength Concept:
    ======================
    - ADX > 25 + RSI bullish = Strong upward momentum with trend support
    - ADX > 25 + RSI bearish = Strong downward momentum with trend support
    - ADX < 25 = Weak trend, exit regardless of RSI direction
    - Higher ADX thresholds = More selective, stronger trend requirement

    Features:
    =========
    - Comprehensive trade logging with IST timezone
    - Detailed PnL tracking for each completed trade
    - Position sizing and commission handling
    - Optimization-ready parameter space
    - Robust error handling and data validation
    - Support for both backtesting and live trading
    - Trend strength analysis and monitoring

    Parameters:
    ==========
    - rsi_period (int): RSI calculation period (default: 14)
    - adx_period (int): ADX calculation period (default: 14)
    - rsi_bullish (int): RSI bullish threshold (default: 55)
    - rsi_bearish (int): RSI bearish threshold (default: 45)
    - adx_strength (int): ADX strength threshold (default: 25)
    - verbose (bool): Enable detailed logging (default: False)

    Performance Metrics:
    ===================
    - Tracks win/loss ratio
    - Calculates net PnL including commissions
    - Records trade duration and timing
    - Provides detailed execution logs
    - Monitors trend strength distribution

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(RSIADX, adx_strength=30, rsi_bullish=60, adx_period=20)
    cerebro.run()

    Best Market Conditions:
    ======================
    - Trending markets with sustained directional moves
    - Markets with clear trend strength and momentum alignment
    - Volatile markets where ADX can effectively measure trend strength
    - Avoid during low-volatility ranging markets

    Note:
    ====
    This strategy is conservative as it requires both momentum direction (RSI) and
    trend strength (ADX) to align. It prioritizes quality over quantity of trades,
    focusing on high-probability setups with strong trend support.
    """

    params = (
        ("rsi_period", 14),
        ("adx_period", 14),
        ("rsi_bullish", 55),
        ("rsi_bearish", 45),
        ("adx_strength", 25),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "adx_period": {"type": "int", "low": 10, "high": 25, "step": 1},
        "rsi_bullish": {"type": "int", "low": 50, "high": 70, "step": 1},
        "rsi_bearish": {"type": "int", "low": 30, "high": 50, "step": 1},
        "adx_strength": {"type": "int", "low": 20, "high": 40, "step": 2},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize RSI indicator
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)

        # Initialize ADX indicator - pass the entire data feed, not individual lines
        try:
            self.adx = btind.AverageDirectionalMovementIndex(
                self.data, period=self.params.adx_period  # Pass the entire data feed
            )
        except Exception as e:
            logger.error(f"Error initializing ADX indicator:-likelihood {str(e)}")
            # Fallback: try with explicit data lines if data feed has them
            try:
                if hasattr(self.data, "high") and hasattr(self.data, "low"):
                    self.adx = btind.AverageDirectionalMovementIndex(
                        self.data.high,
                        self.data.low,
                        self.data.close,
                        period=self.params.adx_period,
                    )
                else:
                    # If no high/low data, create a simple trend strength indicator
                    logger.warning(
                        "No OHLC data available. Creating simplified ADX using close prices."
                    )
                    self.adx = btind.DirectionalMovementIndex(
                        self.data, period=self.params.adx_period
                    )
            except Exception as e2:
                logger.error(f"Fallback ADX initialization failed: {str(e2)}")
                # Create a dummy indicator that always returns the threshold value
                self.adx = (
                    btind.SMA(self.data.close, period=1) * 0 + self.params.adx_strength
                )

        # Momentum and strength signals
        self.rsi_bullish_signal = self.rsi > self.params.rsi_bullish
        self.rsi_bearish_signal = self.rsi < self.params.rsi_bearish
        self.strong_trend_signal = self.adx > self.params.adx_strength

        # Combined entry signals using bt.And
        self.strong_bullish = bt.And(self.rsi_bullish_signal, self.strong_trend_signal)
        self.strong_bearish = bt.And(self.rsi_bearish_signal, self.strong_trend_signal)

        # Exit signal (trend strength weakness)
        self.weak_trend = self.adx < self.params.adx_strength

        self.order = None
        self.order_type = None
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.rsi_period,
                self.params.adx_period,
            )
            + 5  # ADX needs extra bars for DI+ and DI- calculation
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized RSIADX with params: {self.params}")
        logger.info(
            f"RSIADX initialized with rsi_period={self.p.rsi_period}, "
            f"adx_period={self.p.adx_period}, "
            f"rsi_bullish={self.p.rsi_bullish}, rsi_bearish={self.p.rsi_bearish}, "
            f"adx_strength={self.p.adx_strength}"
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
        if np.isnan(self.rsi[0]) or np.isnan(self.adx[0]):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"RSI={self.rsi[0]}, ADX={self.adx[0]}"
            )
            return

        # Calculate trend strength classification
        trend_strength = "WEAK"
        if self.adx[0] >= 50:
            trend_strength = "VERY_STRONG"
        elif self.adx[0] >= 35:
            trend_strength = "STRONG"
        elif self.adx[0] >= self.params.adx_strength:
            trend_strength = "MODERATE"

        # Calculate momentum direction
        momentum_direction = "NEUTRAL"
        if self.rsi[0] > self.params.rsi_bullish:
            momentum_direction = "BULLISH"
        elif self.rsi[0] < self.params.rsi_bearish:
            momentum_direction = "BEARISH"

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "rsi": self.rsi[0],
                "adx": self.adx[0],
                "rsi_bullish": self.rsi_bullish_signal[0],
                "rsi_bearish": self.rsi_bearish_signal[0],
                "strong_trend": self.strong_trend_signal[0],
                "strong_bullish": self.strong_bullish[0],
                "strong_bearish": self.strong_bearish[0],
                "trend_strength": trend_strength,
                "momentum_direction": momentum_direction,
            }
        )

        # RSI + ADX Strength Position Management
        if not self.position:
            # Long Entry: RSI bullish + ADX shows strong trend
            if self.strong_bullish[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Strong Bullish Trend) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} > {self.params.rsi_bullish} (Bullish) | "
                    f"ADX: {self.adx[0]:.2f} > {self.params.adx_strength} (Strong Trend) | "
                    f"Trend Strength: {trend_strength} | "
                    f"Momentum: {momentum_direction}"
                )
            # Short Entry: RSI bearish + ADX shows strong trend
            elif self.strong_bearish[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Strong Bearish Trend) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} < {self.params.rsi_bearish} (Bearish) | "
                    f"ADX: {self.adx[0]:.2f} > {self.params.adx_strength} (Strong Trend) | "
                    f"Trend Strength: {trend_strength} | "
                    f"Momentum: {momentum_direction}"
                )
        else:
            # Exit any position when trend strength weakens (ADX falls below threshold)
            if self.weak_trend[0]:
                if self.position.size > 0:  # Long position
                    self.order = self.sell()
                    self.order_type = "exit_long"
                    trade_logger.info(
                        f"SELL SIGNAL (Exit Long - Weak Trend) | Bar: {len(self)} | "
                        f"Time: {bar_time_ist} | "
                        f"Price: {self.data.close[0]:.2f} | "
                        f"ADX: {self.adx[0]:.2f} < {self.params.adx_strength} (Weak Trend) | "
                        f"RSI: {self.rsi[0]:.2f} | "
                        f"Trend Strength: {trend_strength}"
                    )
                elif self.position.size < 0:  # Short position
                    self.order = self.buy()
                    self.order_type = "exit_short"
                    trade_logger.info(
                        f"BUY SIGNAL (Exit Short - Weak Trend) | Bar: {len(self)} | "
                        f"Time: {bar_time_ist} | "
                        f"Price: {self.data.close[0]:.2f} | "
                        f"ADX: {self.adx[0]:.2f} < {self.params.adx_strength} (Weak Trend) | "
                        f"RSI: {self.rsi[0]:.2f} | "
                        f"Trend Strength: {trend_strength}"
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
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "adx_period": trial.suggest_int("adx_period", 10, 25),
            "rsi_bullish": trial.suggest_int("rsi_bullish", 50, 70),
            "rsi_bearish": trial.suggest_int("rsi_bearish", 30, 50),
            "adx_strength": trial.suggest_int("adx_strength", 20, 40),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            adx_period = params.get("adx_period", 14)
            max_period = max(rsi_period, adx_period)
            return max_period + 5  # Extra bars for ADX calculation
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 35
