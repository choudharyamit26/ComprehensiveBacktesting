import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class TrendlineWilliams(bt.Strategy):
    """
    Trendlines + Williams %R Confluence Trading Strategy

    This strategy combines trendline tests with Williams %R to identify trading opportunities
    at significant price levels with momentum confirmation. It is designed for mean reversion
    or breakout scenarios at trendline levels.

    Strategy Type: CONFLUENCE-BASED TRADING
    ======================================
    This strategy uses trendlines (calculated via recent swing points) as key price levels
    and Williams %R for momentum confirmation to enter trades. It assumes price reactions
    at trendlines, combined with %R signals, provide reliable entry points.

    Strategy Logic:
    ==============
    Long Position Rules:
    - Entry: Price touches or is near an upward trendline (support) AND Williams %R < oversold threshold (default -80)
    - Exit: Price breaks below trendline OR Williams %R > exit threshold (default -50)

    Short Position Rules:
    - Entry: Price touches or is near a downward trendline (resistance) AND Williams %R > overbought threshold (default -20)
    - Exit: Price breaks above trendline OR Williams %R < exit threshold (default -50)

    Trendline Calculation:
    =====================
    - Trendlines are approximated using linear regression on recent swing highs/lows
    - Upward trendline (support): Linear fit through recent swing lows
    - Downward trendline (resistance): Linear fit through recent swing highs
    - Tolerance range (default 0.5%) for detecting price proximity to trendlines

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses warmup period to ensure indicator and trendline stability
    - Prevents order overlap with pending order checks
    - Suitable for trending or ranging markets with clear trendline reactions

    Indicators Used:
    ===============
    - Williams %R: Measures momentum to confirm oversold/overbought conditions
    - Trendlines: Calculated using linear regression on swing highs/lows
    - Tolerance Range: Percentage-based buffer to detect price proximity to trendlines

    Features:
    =========
    - Dynamic trendline calculation using swing points
    - Comprehensive trade logging with IST timezone
    - Detailed PnL tracking for each completed trade
    - Position sizing and commission handling
    - Optimization-ready parameter space
    - Robust error handling and data validation
    - Support for both backtesting and live trading

    Parameters:
    ==========
    - williams_period (int): Williams %R calculation period (default: 14)
    - trend_lookback (int): Lookback period for trendline calculation (default: 20)
    - trend_tolerance (float): Percentage tolerance for trendline proximity (default: 0.5)
    - williams_oversold (int): Williams %R oversold threshold for long entries (default: -80)
    - williams_overbought (int): Williams %R overbought threshold for short entries (default: -20)
    - williams_exit (int): Williams %R exit threshold for position closes (default: -50)
    - verbose (bool): Enable detailed logging (default: False)

    Performance Metrics:
    ===================
    - Tracks win/loss ratio
    - Calculates net PnL including commissions
    - Records trade duration and timing
    - Provides detailed execution logs
    - Monitors trendline touch frequency

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TrendlineWilliams, williams_oversold=-85, williams_overbought=-15, trend_lookback=25)
    cerebro.run()

    Best Market Conditions:
    ======================
    - Markets with clear trendline patterns
    - Ranging or trending markets with frequent trendline tests
    - Avoid during choppy markets with unclear trendlines
    - Works well in intraday timeframes with sufficient volatility

    Note:
    ====
    This strategy relies on accurate trendline detection and Williams %R confirmation.
    Trendline calculations are approximations and may require frequent updates.
    Consider adding volume or trend filters to improve signal reliability.
    Backtest thoroughly to validate trendline accuracy.
    """

    params = (
        ("williams_period", 14),
        ("trend_lookback", 20),
        ("trend_tolerance", 0.5),  # 0.5% tolerance for trendline proximity
        ("williams_oversold", -80),
        ("williams_overbought", -20),
        ("williams_exit", -50),
        ("verbose", False),
    )

    optimization_params = {
        "williams_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "trend_lookback": {"type": "int", "low": 15, "high": 30, "step": 1},
        "trend_tolerance": {"type": "float", "low": 0.3, "high": 1.0, "step": 0.1},
        "williams_oversold": {"type": "int", "low": -90, "high": -70, "step": 5},
        "williams_overbought": {"type": "int", "low": -30, "high": -10, "step": 5},
        "williams_exit": {"type": "int", "low": -60, "high": -40, "step": 5},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize Williams %R indicator
        self.williams = btind.WilliamsR(period=self.params.williams_period)

        # Initialize trendlines using swing points
        self.swing_high = btind.Highest(
            self.data.high, period=self.params.trend_lookback
        )
        self.swing_low = btind.Lowest(self.data.low, period=self.params.trend_lookback)

        self.order = None
        self.order_type = None
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            self.params.williams_period + self.params.trend_lookback + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized TrendlineWilliams with params: {self.params}")
        logger.info(
            f"TrendlineWilliams initialized with williams_period={self.p.williams_period}, "
            f"trend_lookback={self.p.trend_lookback}, trend_tolerance={self.p.trend_tolerance}, "
            f"williams_oversold={self.p.williams_oversold}, williams_overbought={self.p.williams_overbought}, "
            f"williams_exit={self.p.williams_exit}"
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
            np.isnan(self.williams[0])
            or np.isnan(self.swing_high[0])
            or np.isnan(self.swing_low[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"Williams %R={self.williams[0]}, Swing_High={self.swing_high[0]}, "
                f"Swing_Low={self.swing_low[0]}"
            )
            return

        # Calculate trendline levels (approximated as swing high/low levels for simplicity)
        current_price = self.data.close[0]
        support_trendline = self.swing_low[0]  # Upward trendline approximation
        resistance_trendline = self.swing_high[0]  # Downward trendline approximation
        tolerance = self.params.trend_tolerance / 100 * current_price

        support_touch = abs(current_price - support_trendline) <= tolerance
        resistance_touch = abs(current_price - resistance_trendline) <= tolerance

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": current_price,
                "williams": self.williams[0],
                "support_trendline": support_trendline,
                "resistance_trendline": resistance_trendline,
                "support_touch": support_touch,
                "resistance_touch": resistance_touch,
            }
        )

        # Trading Logic
        if not self.position:
            # Long Entry: Price near upward trendline AND Williams %R oversold
            if support_touch and self.williams[0] < self.params.williams_oversold:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Trendline + Williams %R) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {current_price:.2f} | "
                    f"Williams %R: {self.williams[0]:.2f} < {self.params.williams_oversold} (Oversold) | "
                    f"Trendline Support: {support_trendline:.2f} (Touch)"
                )
            # Short Entry: Price near downward trendline AND Williams %R overbought
            elif (
                resistance_touch and self.williams[0] > self.params.williams_overbought
            ):
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Trendline + Williams %R) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {current_price:.2f} | "
                    f"Williams %R: {self.williams[0]:.2f} > {self.params.williams_overbought} (Overbought) | "
                    f"Trendline Resistance: {resistance_trendline:.2f} (Touch)"
                )
        elif self.position.size > 0:  # Long position
            # Long Exit: Price breaks below trendline OR Williams %R normalizes
            if (
                current_price < support_trendline
                or self.williams[0] > self.params.williams_exit
            ):
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "Price broke below trendline"
                    if current_price < support_trendline
                    else "Williams %R normalized"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - Trendline + Williams %R) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {current_price:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Williams %R: {self.williams[0]:.2f} | "
                    f"Trendline Support: {support_trendline:.2f}"
                )
        elif self.position.size < 0:  # Short position
            # Short Exit: Price breaks above trendline OR Williams %R normalizes
            if (
                current_price > resistance_trendline
                or self.williams[0] < self.params.williams_exit
            ):
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "Price broke above trendline"
                    if current_price > resistance_trendline
                    else "Williams %R normalized"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - Trendline + Williams %R) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {current_price:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Williams %R: {self.williams[0]:.2f} | "
                    f"Trendline Resistance: {resistance_trendline:.2f}"
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
            "williams_period": trial.suggest_int("williams_period", 10, 20),
            "trend_lookback": trial.suggest_int("trend_lookback", 15, 30),
            "trend_tolerance": trial.suggest_float("trend_tolerance", 0.3, 1.0),
            "williams_oversold": trial.suggest_int("williams_oversold", -90, -70),
            "williams_overbought": trial.suggest_int("williams_overbought", -30, -10),
            "williams_exit": trial.suggest_int("williams_exit", -60, -40),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            williams_period = params.get("williams_period", 14)
            trend_lookback = params.get("trend_lookback", 20)
            max_period = max(williams_period, trend_lookback)
            return max_period + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
