import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class RSIBB(bt.Strategy):
    """
    RSI and Bollinger Bands Combined Mean Reversion Trading Strategy

    This strategy combines RSI (Relative Strength Index) and Bollinger Bands to identify
    mean reversion opportunities by detecting oversold/overbought conditions at extreme
    price levels relative to the moving average.

    Strategy Type: MEAN REVERSION
    =============================
    This is a mean reversion strategy that assumes prices will return to their average
    after reaching extreme levels. It buys when prices are oversold and at the lower
    Bollinger Band, expecting a bounce back to the middle band (moving average).

    Strategy Logic:
    ==============

    Long Position Rules:
    - Entry: RSI < oversold threshold (default 30) AND price touches/crosses below lower Bollinger Band
    - Exit: RSI > exit threshold (default 50) OR price reaches middle Bollinger Band (SMA)

    Short Position Rules:
    - Entry: RSI > overbought threshold (default 70) AND price touches/crosses above upper Bollinger Band
    - Exit: RSI < exit threshold (default 50) OR price reaches middle Bollinger Band (SMA)

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses warmup period to ensure indicator stability before trading
    - Prevents order overlap with pending order checks
    - Mean reversion works best in ranging/sideways markets

    Indicators Used:
    ===============
    - RSI: Measures momentum strength to identify oversold/overbought conditions
    - Bollinger Bands: Statistical bands around moving average to identify price extremes
      * Upper Band: SMA + (2 * Standard Deviation)
      * Middle Band: Simple Moving Average (SMA)
      * Lower Band: SMA - (2 * Standard Deviation)

    Mean Reversion Concept:
    ======================
    - When price hits lower BB + RSI oversold = likely bounce upward
    - When price hits upper BB + RSI overbought = likely pullback downward
    - Exit at middle BB assumes price has reverted to mean (average)
    - RSI normalization confirms momentum shift back to neutral

    Features:
    =========
    - Comprehensive trade logging with IST timezone
    - Detailed PnL tracking for each completed trade
    - Position sizing and commission handling
    - Optimization-ready parameter space
    - Robust error handling and data validation
    - Support for both backtesting and live trading
    - Band touch detection for precise entry timing

    Parameters:
    ==========
    - rsi_period (int): RSI calculation period (default: 14)
    - bb_period (int): Bollinger Bands moving average period (default: 20)
    - bb_stddev (float): Bollinger Bands standard deviation multiplier (default: 2.0)
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
    - Monitors band touch frequency

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(RSIBB, rsi_oversold=25, rsi_overbought=75, bb_period=20)
    cerebro.run()

    Best Market Conditions:
    ======================
    - Ranging/sideways markets with clear support/resistance
    - High volatility periods with frequent mean reversion
    - Avoid during strong trending markets (momentum strategies better)
    - Works well in intraday timeframes with sufficient volatility

    Note:
    ====
    This is a mean reversion strategy that profits from price returning to average levels.
    It's opposite to momentum strategies and requires different market conditions to be profitable.
    Consider using trend filters to avoid trading against strong trends.
    """

    params = (
        ("rsi_period", 14),
        ("bb_period", 20),
        ("bb_stddev", 2.0),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
        ("rsi_exit", 50),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "bb_period": {"type": "int", "low": 15, "high": 25, "step": 1},
        "bb_stddev": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.1},
        "rsi_oversold": {"type": "int", "low": 20, "high": 35, "step": 1},
        "rsi_overbought": {"type": "int", "low": 65, "high": 80, "step": 1},
        "rsi_exit": {"type": "int", "low": 45, "high": 55, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.bb = btind.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_stddev,
        )

        # Bollinger Band components
        self.bb_top = self.bb.lines.top
        self.bb_mid = self.bb.lines.mid
        self.bb_bot = self.bb.lines.bot

        # Band touch detection
        self.bb_lower_touch = self.data.close <= self.bb_bot
        self.bb_upper_touch = self.data.close >= self.bb_top

        self.order = None
        self.order_type = None  # Track order type for shorting logic
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.rsi_period,
                self.params.bb_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized RSIBB with params: {self.params}")
        logger.info(
            f"RSIBB initialized with rsi_period={self.p.rsi_period}, "
            f"bb_period={self.p.bb_period}, bb_stddev={self.p.bb_stddev}, "
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
            or np.isnan(self.bb_top[0])
            or np.isnan(self.bb_mid[0])
            or np.isnan(self.bb_bot[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"RSI={self.rsi[0]}, BB_Top={self.bb_top[0]}, "
                f"BB_Mid={self.bb_mid[0]}, BB_Bot={self.bb_bot[0]}"
            )
            return

        # Calculate band position percentage
        bb_position = (self.data.close[0] - self.bb_bot[0]) / (
            self.bb_top[0] - self.bb_bot[0]
        )

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "rsi": self.rsi[0],
                "bb_top": self.bb_top[0],
                "bb_mid": self.bb_mid[0],
                "bb_bot": self.bb_bot[0],
                "bb_position": bb_position,
                "bb_lower_touch": self.bb_lower_touch[0],
                "bb_upper_touch": self.bb_upper_touch[0],
            }
        )

        # Mean Reversion Position Management
        if not self.position:
            # Long Entry: RSI oversold AND price at/below lower Bollinger Band
            if self.rsi[0] < self.params.rsi_oversold and self.bb_lower_touch[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Mean Reversion) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} < {self.params.rsi_oversold} (Oversold) | "
                    f"BB_Lower: {self.bb_bot[0]:.2f} (Touch) | "
                    f"BB_Position: {bb_position:.2%}"
                )
            # Short Entry: RSI overbought AND price at/above upper Bollinger Band
            elif self.rsi[0] > self.params.rsi_overbought and self.bb_upper_touch[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Mean Reversion) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} > {self.params.rsi_overbought} (Overbought) | "
                    f"BB_Upper: {self.bb_top[0]:.2f} (Touch) | "
                    f"BB_Position: {bb_position:.2%}"
                )
        elif self.position.size > 0:  # Long position
            # Long Exit: RSI normalizes OR price reaches middle BB (mean reversion complete)
            if (
                self.rsi[0] > self.params.rsi_exit
                or self.data.close[0] >= self.bb_mid[0]
            ):
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "RSI normalized"
                    if self.rsi[0] > self.params.rsi_exit
                    else "Price reached middle BB"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - Mean Reversion) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"BB_Mid: {self.bb_mid[0]:.2f} | "
                    f"BB_Position: {bb_position:.2%}"
                )
        elif self.position.size < 0:  # Short position
            # Short Exit: RSI normalizes OR price reaches middle BB (mean reversion complete)
            if (
                self.rsi[0] < self.params.rsi_exit
                or self.data.close[0] <= self.bb_mid[0]
            ):
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "RSI normalized"
                    if self.rsi[0] < self.params.rsi_exit
                    else "Price reached middle BB"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - Mean Reversion) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"BB_Mid: {self.bb_mid[0]:.2f} | "
                    f"BB_Position: {bb_position:.2%}"
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
                    "size": order.executed.size,  # Negative for short
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
            "bb_period": trial.suggest_int("bb_period", 15, 25),
            "bb_stddev": trial.suggest_float("bb_stddev", 1.5, 2.5),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 35),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 65, 80),
            "rsi_exit": trial.suggest_int("rsi_exit", 45, 55),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            bb_period = params.get("bb_period", 20)
            max_period = max(rsi_period, bb_period)
            return max_period + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
