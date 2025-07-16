import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class SRRSI(bt.Strategy):
    """
    Support/Resistance Levels + RSI Confluence Trading Strategy

    This strategy combines key support and resistance (S/R) levels with RSI (Relative Strength Index)
    to identify high-probability trading opportunities at significant price levels. It is designed for
    mean reversion or breakout scenarios, depending on price action at S/R levels.

    Strategy Type: CONFLUENCE-BASED TRADING
    ======================================
    This strategy uses S/R levels as primary price targets and RSI for momentum confirmation to
    enter trades. It assumes that price reactions at key S/R levels, combined with RSI signals,
    provide reliable entry points.

    Strategy Logic:
    ==============
    Long Position Rules:
    - Entry: Price touches or is near a support level (within tolerance) AND RSI < oversold threshold (default 30)
    - Exit: Price breaks below support OR RSI > exit threshold (default 50)

    Short Position Rules:
    - Entry: Price touches or is near a resistance level (within tolerance) AND RSI > overbought threshold (default 70)
    - Exit: Price breaks above resistance OR RSI < exit threshold (default 50)

    Support/Resistance Detection:
    ============================
    - Uses swing highs/lows to identify S/R levels dynamically
    - Applies a tolerance range (default 0.5%) to detect price proximity to S/R levels
    - Updates S/R levels based on recent price action

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses warmup period to ensure indicator and S/R stability before trading
    - Prevents order overlap with pending order checks
    - Suitable for ranging or mildly trending markets with clear S/R zones

    Indicators Used:
    ===============
    - RSI: Measures momentum to confirm oversold/overbought conditions
    - Support/Resistance Levels: Calculated using swing highs/lows over a lookback period
    - Tolerance Range: Percentage-based buffer to detect price proximity to S/R

    Features:
    =========
    - Dynamic S/R level detection using swing points
    - Comprehensive trade logging with IST timezone
    - Detailed PnL tracking for each completed trade
    - Position sizing and commission handling
    - Optimization-ready parameter space
    - Robust error handling and data validation
    - Support for both backtesting and live trading

    Parameters:
    ==========
    - rsi_period (int): RSI calculation period (default: 14)
    - sr_lookback (int): Lookback period for S/R level detection (default: 20)
    - sr_tolerance (float): Percentage tolerance for S/R proximity (default: 0.5)
    - rsi_oversold (int): RSI oversold threshold for long entries (default: 30)
    - rsi_overbought (int): RSI overbought threshold for short entries (default: 70)
    - rsi_exit (int): RSI exit threshold for position closes (default: 50)
    - verbose (bool): Enable detailed logging (default: False)

    Performance Metrics:
    ===================
    - Tracks win/loss ratio
    - Calculates net PnL including commissions
    - Records trade duration and timing
    - Provides detailed execution logs
    - Monitors S/R level touch frequency

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SRRSI, rsi_oversold=25, rsi_overbought=75, sr_lookback=20)
    cerebro.run()

    Best Market Conditions:
    ======================
    - Markets with well-defined support/resistance zones
    - Ranging or mildly trending markets
    - Avoid during strong trending markets without clear S/R levels
    - Works well in intraday timeframes with sufficient volatility

    Note:
    ====
    This strategy relies on accurate S/R level detection and RSI confirmation. Consider
    adding trend filters or volume confirmation to avoid false signals in low-liquidity
    markets. Backtest thoroughly to validate S/R level reliability.
    """

    params = (
        ("rsi_period", 14),
        ("sr_lookback", 20),
        ("sr_tolerance", 0.5),  # 0.5% tolerance for S/R proximity
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
        ("rsi_exit", 50),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "sr_lookback": {"type": "int", "low": 15, "high": 30, "step": 1},
        "sr_tolerance": {"type": "float", "low": 0.3, "high": 1.0, "step": 0.1},
        "rsi_oversold": {"type": "int", "low": 20, "high": 35, "step": 1},
        "rsi_overbought": {"type": "int", "low": 65, "high": 80, "step": 1},
        "rsi_exit": {"type": "int", "low": 45, "high": 55, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize RSI indicator
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)

        # Initialize S/R levels using swing highs/lows
        self.swing_high = btind.Highest(self.data.high, period=self.params.sr_lookback)
        self.swing_low = btind.Lowest(self.data.low, period=self.params.sr_lookback)

        self.order = None
        self.order_type = None
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = self.params.rsi_period + self.params.sr_lookback + 2
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized SRRSI with params: {self.params}")
        logger.info(
            f"SRRSI initialized with rsi_period={self.p.rsi_period}, "
            f"sr_lookback={self.p.sr_lookback}, sr_tolerance={self.p.sr_tolerance}, "
            f"rsi_oversold={self.p.rsi_oversold}, rsi_overbought={self.p.rsi_overbought}, "
            f"rsi_exit={self.p.rsi_exit}"
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
            np.isnan(self.rsi[0])
            or np.isnan(self.swing_high[0])
            or np.isnan(self.swing_low[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"RSI={self.rsi[0]}, Swing_High={self.swing_high[0]}, "
                f"Swing_Low={self.swing_low[0]}"
            )
            return

        # Calculate S/R proximity
        current_price = self.data.close[0]
        support_level = self.swing_low[0]
        resistance_level = self.swing_high[0]
        tolerance = self.params.sr_tolerance / 100 * current_price

        support_touch = abs(current_price - support_level) <= tolerance
        resistance_touch = abs(current_price - resistance_level) <= tolerance

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": current_price,
                "rsi": self.rsi[0],
                "support_level": support_level,
                "resistance_level": resistance_level,
                "support_touch": support_touch,
                "resistance_touch": resistance_touch,
            }
        )

        # Trading Logic
        if not self.position:
            # Long Entry: Price near support AND RSI oversold
            if support_touch and self.rsi[0] < self.params.rsi_oversold:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - S/R + RSI) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {current_price:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} < {self.params.rsi_oversold} (Oversold) | "
                    f"Support: {support_level:.2f} (Touch)"
                )
            # Short Entry: Price near resistance AND RSI overbought
            elif resistance_touch and self.rsi[0] > self.params.rsi_overbought:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - S/R + RSI) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {current_price:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} > {self.params.rsi_overbought} (Overbought) | "
                    f"Resistance: {resistance_level:.2f} (Touch)"
                )
        elif self.position.size > 0:  # Long position
            # Long Exit: Price breaks below support OR RSI normalizes
            if current_price < support_level or self.rsi[0] > self.params.rsi_exit:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "Price broke below support"
                    if current_price < support_level
                    else "RSI normalized"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - S/R + RSI) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {current_price:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"Support: {support_level:.2f}"
                )
        elif self.position.size < 0:  # Short position
            # Short Exit: Price breaks above resistance OR RSI normalizes
            if current_price > resistance_level or self.rsi[0] < self.params.rsi_exit:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "Price broke above resistance"
                    if current_price > resistance_level
                    else "RSI normalized"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - S/R + RSI) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {current_price:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"Resistance: {resistance_level:.2f}"
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
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "sr_lookback": trial.suggest_int("sr_lookback", 15, 30),
            "sr_tolerance": trial.suggest_float("sr_tolerance", 0.3, 1.0),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 35),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 65, 80),
            "rsi_exit": trial.suggest_int("rsi_exit", 45, 55),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            sr_lookback = params.get("sr_lookback", 20)
            max_period = max(rsi_period, sr_lookback)
            return max_period + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
