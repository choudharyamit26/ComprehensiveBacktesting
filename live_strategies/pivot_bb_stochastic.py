import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class PivotPoint(bt.Indicator):
    """
    Custom Pivot Point indicator that calculates pivot levels based on historical high, low, and close prices.

    Lines:
        pivot: Main pivot point level
        r1: First resistance level
        s1: First support level

    Parameters:
        period (int): Number of periods to look back for high/low calculation (default: 20)
    """

    lines = ("pivot", "r1", "s1")
    params = (("period", 20),)

    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        if len(self.data) < self.params.period:
            return

        high = max(self.data.high.get(ago=0, size=self.params.period))
        low = min(self.data.low.get(ago=0, size=self.params.period))
        close = self.data.close[-1]

        self.lines.pivot[0] = (high + low + close) / 3
        self.lines.r1[0] = 2 * self.lines.pivot[0] - low
        self.lines.s1[0] = 2 * self.lines.pivot[0] - high


class Pivot_BB_Stochastic(bt.Strategy):
    """
    Pivot Point + Bollinger Bands + Stochastic Oscillator Trading Strategy

    STRATEGY OVERVIEW:
    ==================
    This is a multi-indicator convergence strategy that combines pivot point analysis,
    Bollinger Bands volatility measurement, and Stochastic momentum oscillator to
    identify high-probability trading opportunities.

    STRATEGY TYPE: PIVOT + VOLATILITY + MOMENTUM
    ============================================

    CORE CONCEPT:
    =============
    The strategy looks for convergence of three different market aspects:
    1. Price action around key pivot levels (support/resistance)
    2. Volatility breakouts using Bollinger Bands
    3. Momentum confirmation via Stochastic oscillator

    ENTRY SIGNALS:
    ==============

    Long Entry Conditions (ALL must be met):
    - Price is at or near the pivot point (within 1% tolerance)
    - Price touches or exceeds the upper Bollinger Band (volatility breakout upward)
    - Stochastic %K line is above %D line (bullish momentum)

    Short Entry Conditions (ALL must be met):
    - Price is at or near the pivot point (within 1% tolerance)
    - Price touches or goes below the lower Bollinger Band (volatility breakout downward)
    - Stochastic %K line is below %D line (bearish momentum)

    EXIT SIGNALS:
    =============

    Long Position Exits:
    - Price reaches the R1 resistance level (profit target)
    - Stochastic momentum reverses (%K drops below %D)

    Short Position Exits:
    - Price reaches the S1 support level (profit target)
    - Stochastic momentum reverses (%K rises above %D)

    RISK MANAGEMENT:
    ================
    - Automatic position closure at 15:15 IST (before market close)
    - Trading only during market hours: 9:15 AM to 3:05 PM IST
    - No new positions opened near market close
    - Single position limit (no position pyramiding)

    MARKET LOGIC:
    =============
    The strategy is based on the principle that:
    1. Pivot points act as natural support/resistance levels
    2. Bollinger Band touches indicate potential volatility breakouts
    3. Stochastic oscillator confirms the momentum direction
    4. When all three align, it suggests a high-probability trade setup

    TIMEFRAME SUITABILITY:
    ======================
    Best suited for:
    - Intraday trading on 5-minute to 15-minute charts
    - Liquid instruments with good volatility
    - Markets with clear trending behavior

    PERFORMANCE CHARACTERISTICS:
    ============================
    - Higher win rate during trending market conditions
    - May generate false signals in sideways/ranging markets
    - Profit targets based on pivot levels provide natural risk/reward ratios
    - Momentum confirmation helps reduce whipsaw trades

    OPTIMIZATION PARAMETERS:
    ========================
    - pivot_period: Affects sensitivity of pivot point calculations
    - bb_period: Controls Bollinger Bands responsiveness to price changes
    - bb_stddev: Adjusts the width of Bollinger Bands (volatility threshold)
    - stoch_k: Fast stochastic period (momentum sensitivity)
    - stoch_d: Slow stochastic period (signal smoothing)

    PARAMETER DEFAULTS:
    ===================
    - pivot_period: 20 (balanced between responsiveness and stability)
    - bb_period: 20 (standard Bollinger Bands period)
    - bb_stddev: 2.0 (captures ~95% of price action within bands)
    - stoch_k: 14 (classic stochastic period)
    - stoch_d: 3 (standard smoothing period)

    IMPLEMENTATION NOTES:
    =====================
    - Uses custom PivotPoint indicator for dynamic pivot calculations
    - Implements comprehensive trade logging for analysis
    - Stores indicator data for post-trade analysis
    - Includes parameter optimization support for backtesting
    - Handles timezone conversion for IST market hours

    USAGE EXAMPLE:
    ==============
    ```python
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Pivot_BB_Stochastic,
                       pivot_period=20,
                       bb_period=20,
                       bb_stddev=2.0,
                       stoch_k=14,
                       stoch_d=3,
                       verbose=True)
    ```

    AUTHOR: Trading Strategy Implementation
    VERSION: 1.0
    LAST_UPDATED: 2025
    """

    params = (
        ("pivot_period", 20),  # Pivot point calculation period
        ("bb_period", 20),  # Bollinger Bands period
        ("bb_stddev", 2.0),  # Bollinger Bands standard deviation multiplier
        ("stoch_k", 14),  # Stochastic %K period
        ("stoch_d", 3),  # Stochastic %D smoothing period
        ("verbose", False),  # Enable detailed logging
    )

    optimization_params = {
        "pivot_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "bb_period": {"type": "int", "low": 15, "high": 25, "step": 1},
        "bb_stddev": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.1},
        "stoch_k": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stoch_d": {"type": "int", "low": 2, "high": 5, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        """
        Initialize the Pivot_BB_Stochastic strategy with all required indicators.

        Args:
            tickers: List of ticker symbols (optional)
            analyzers: List of analyzers to attach (optional)
            **kwargs: Additional keyword arguments
        """
        # Initialize indicators
        self.pivot = PivotPoint(self.data, period=self.params.pivot_period)
        self.bb = btind.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_stddev,
        )
        self.stoch = btind.Stochastic(
            self.data, period=self.params.stoch_k, period_dfast=self.params.stoch_d
        )
        self.bb_upper_touch = self.data.close >= self.bb.lines.top
        self.bb_lower_touch = self.data.close <= self.bb.lines.bot

        # Debug: Log available lines and their types
        logger.debug(f"Pivot lines: {self.pivot.lines.getlinealiases()}")
        logger.debug(f"BollingerBands lines: {self.bb.lines.getlinealiases()}")
        logger.debug(f"Stochastic lines: {self.stoch.lines.getlinealiases()}")

        # Strategy state variables
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(self.params.pivot_period, self.params.bb_period, self.params.stoch_k)
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized Pivot_BB_Stochastic with params: {self.params}")
        logger.info(
            f"Pivot_BB_Stochastic initialized with pivot_period={self.p.pivot_period}, "
            f"bb_period={self.p.bb_period}, bb_stddev={self.p.bb_stddev}, "
            f"stoch_k={self.p.stoch_k}, stoch_d={self.p.stoch_d}"
        )

    def next(self):
        """
        Main strategy logic executed on each bar.

        This method contains:
        - Warmup period handling
        - Market hours validation
        - Signal generation logic
        - Entry and exit condition evaluation
        """
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
            np.isnan(self.pivot.pivot[0])
            or np.isnan(self.bb.lines.top[0])
            or np.isnan(self.bb.lines.bot[0])
            or np.isnan(self.stoch.percK[0])
            or np.isnan(self.stoch.percD[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"Pivot={self.pivot.pivot[0]}, BB Top={self.bb.lines.top[0]}, "
                f"BB Bottom={self.bb.lines.bot[0]}, Stochastic %K={self.stoch.percK[0]}, "
                f"Stochastic %D={self.stoch.percD[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "pivot": self.pivot.pivot[0],
                "r1": self.pivot.r1[0],
                "s1": self.pivot.s1[0],
                "bb_top": self.bb.lines.top[0],
                "bb_bot": self.bb.lines.bot[0],
                "stoch_k": self.stoch.percK[0],
                "stoch_d": self.stoch.percD[0],
            }
        )

        # Trading Logic
        price_near_pivot = (
            abs(self.data.close[0] - self.pivot.pivot[0]) / self.pivot.pivot[0] < 0.01
        )  # Within 1%
        stoch_bullish = self.stoch.percK[0] > self.stoch.percD[0]
        stoch_bearish = self.stoch.percK[0] < self.stoch.percD[0]

        if not self.position:
            # Long Entry: Price at/near pivot + Price at/above upper BB + Stochastic %K > %D
            if price_near_pivot and self.bb_upper_touch[0] and stoch_bullish:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Pivot + BB + Stochastic) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Pivot: {self.pivot.pivot[0]:.2f} (Near) | "
                    f"BB Top: {self.bb.lines.top[0]:.2f} (Touch) | "
                    f"Stochastic %K: {self.stoch.percK[0]:.2f} > %D: {self.stoch.percD[0]:.2f}"
                )
            # Short Entry: Price at/near pivot + Price at/below lower BB + Stochastic %K < %D
            elif price_near_pivot and self.bb_lower_touch[0] and stoch_bearish:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Pivot + BB + Stochastic) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Pivot: {self.pivot.pivot[0]:.2f} (Near) | "
                    f"BB Bottom: {self.bb.lines.bot[0]:.2f} (Touch) | "
                    f"Stochastic %K: {self.stoch.percK[0]:.2f} < %D: {self.stoch.percD[0]:.2f}"
                )
        elif self.position.size > 0:  # Long position
            # Exit: Price reaches R1 or Stochastic reversal
            if self.data.close[0] >= self.pivot.r1[0] or not stoch_bullish:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "Reached R1"
                    if self.data.close[0] >= self.pivot.r1[0]
                    else "Stochastic reversal"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - Pivot + BB + Stochastic) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Pivot: {self.pivot.pivot[0]:.2f} | "
                    f"BB Top: {self.bb.lines.top[0]:.2f} | "
                    f"Stochastic %K: {self.stoch.percK[0]:.2f}, %D: {self.stoch.percD[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            # Exit: Price reaches S1 or Stochastic reversal
            if self.data.close[0] <= self.pivot.s1[0] or not stoch_bearish:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "Reached S1"
                    if self.data.close[0] <= self.pivot.s1[0]
                    else "Stochastic reversal"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - Pivot + BB + Stochastic) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Pivot: {self.pivot.pivot[0]:.2f} | "
                    f"BB Bottom: {self.bb.lines.bot[0]:.2f} | "
                    f"Stochastic %K: {self.stoch.percK[0]:.2f}, %D: {self.stoch.percD[0]:.2f}"
                )

    def notify_order(self, order):
        """
        Handle order execution notifications.

        Args:
            order: Order object with execution details
        """
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt)
            if exec_dt.tzinfo is None:
                exec_dt = exec_dt.replace(tzinfo=pytz.UTC)
            exec_dt = exec_dt.astimezone(pytz.timezone("Asia/Kolkata"))

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
        """
        Handle trade completion notifications.

        Args:
            trade: Trade object with P&L details
        """
        if trade.isclosed:
            trade_logger.info(
                f"TRADE CLOSED | Ref: {trade.ref} | "
                f"Profit: {trade.pnl:.2f} | "
                f"Net Profit: {trade.pnlcomm:.2f} | "
                f"Bars Held: {trade.barlen} | "
                f"Trade Count: {self.trade_count}"
            )

    def get_completed_trades(self):
        """
        Return a copy of completed trades list for analysis.

        Returns:
            list: List of dictionaries containing trade information
        """
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        """
        Define parameter space for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            dict: Parameter dictionary for optimization
        """
        params = {
            "pivot_period": trial.suggest_int("pivot_period", 10, 30),
            "bb_period": trial.suggest_int("bb_period", 15, 25),
            "bb_stddev": trial.suggest_float("bb_stddev", 1.5, 2.5, step=0.1),
            "stoch_k": trial.suggest_int("stoch_k", 10, 20),
            "stoch_d": trial.suggest_int("stoch_d", 2, 5),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        """
        Calculate minimum data points required for strategy initialization.

        Args:
            params (dict): Strategy parameters

        Returns:
            int: Minimum number of data points needed
        """
        try:
            pivot_period = params.get("pivot_period", 20)
            bb_period = params.get("bb_period", 20)
            stoch_k = params.get("stoch_k", 14)
            return max(pivot_period, bb_period, stoch_k) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
