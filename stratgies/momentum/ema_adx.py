import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class EMAADXTrend(bt.Strategy):
    """
    EMA + ADX Trend System Trading Strategy

    This strategy combines Exponential Moving Average (EMA) for trend direction
    and Average Directional Index (ADX) for trend strength to identify high-probability
    trend-following opportunities with strong momentum.

    Strategy Type: TREND FOLLOWING
    ==============================
    This is a trend-following strategy that enters positions when price is trending
    in the EMA direction and the ADX confirms strong trend strength, maximizing
    the probability of sustained price movement.

    Strategy Logic:
    ==============

    Long Position Rules:
    - Entry: Price above EMA AND ADX > min_adx_threshold AND ADX rising
    - Exit: Price below EMA OR ADX < exit_adx_threshold OR ADX falling significantly

    Short Position Rules:
    - Entry: Price below EMA AND ADX > min_adx_threshold AND ADX rising
    - Exit: Price above EMA OR ADX < exit_adx_threshold OR ADX falling significantly

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses ADX-based trend strength confirmation
    - Prevents order overlap with pending order checks
    - Works best in trending markets with sustained momentum

    Indicators Used:
    ===============
    - EMA: Exponential Moving Average for trend direction
    - ADX: Average Directional Index for trend strength measurement
    - DI+: Positive Directional Indicator (bullish momentum)
    - DI-: Negative Directional Indicator (bearish momentum)

    ADX Trend Strength Concept:
    ===========================
    - ADX > 25: Strong trend (default threshold)
    - ADX < 20: Weak trend or consolidation
    - Rising ADX: Strengthening trend
    - Falling ADX: Weakening trend
    - DI+ > DI-: Bullish momentum
    - DI- > DI+: Bearish momentum

    Features:
    =========
    - Comprehensive trade logging with IST timezone
    - ADX-based trend strength analysis
    - EMA trend direction confirmation
    - Dynamic exit based on trend weakening
    - Robust error handling and data validation
    - Support for both backtesting and live trading
    - Directional Indicator crossover analysis

    Parameters:
    ==========
    - ema_period (int): EMA calculation period (default: 21)
    - adx_period (int): ADX calculation period (default: 14)
    - min_adx_threshold (float): Minimum ADX for trend entry (default: 25.0)
    - exit_adx_threshold (float): ADX exit threshold (default: 20.0)
    - adx_rising_threshold (float): ADX rise requirement (default: 1.0)
    - adx_falling_threshold (float): ADX fall for exit (default: 3.0)
    - di_separation (float): Minimum DI+/DI- separation (default: 2.0)
    - price_ema_buffer (float): Price buffer above/below EMA (default: 0.1)
    - verbose (bool): Enable detailed logging (default: False)

    Performance Metrics:
    ===================
    - Tracks win/loss ratio
    - Calculates net PnL including commissions
    - Records trade duration and timing
    - Monitors trend strength and duration
    - Provides detailed execution logs

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EMADXTrend, ema_period=21, min_adx_threshold=30)
    cerebro.run()

    Best Market Conditions:
    ======================
    - Strong trending markets with sustained momentum
    - High ADX readings indicating trend strength
    - Clear directional moves with minimal whipsaws
    - Avoid during consolidation or choppy markets

    Note:
    ====
    This strategy requires strong trends to be profitable. It's designed for
    trend-following and performs best when markets have clear directional bias.
    Consider using additional filters during ranging markets.
    """

    params = (
        ("ema_period", 21),
        ("adx_period", 14),
        ("min_adx_threshold", 25.0),
        ("exit_adx_threshold", 20.0),
        ("adx_rising_threshold", 1.0),
        ("adx_falling_threshold", 3.0),
        ("di_separation", 2.0),
        ("price_ema_buffer", 0.1),
        ("verbose", False),
    )

    optimization_params = {
        "ema_period": {"type": "int", "low": 15, "high": 30, "step": 3},
        "adx_period": {"type": "int", "low": 10, "high": 20, "step": 2},
        "min_adx_threshold": {"type": "float", "low": 20.0, "high": 35.0, "step": 2.5},
        "exit_adx_threshold": {"type": "float", "low": 15.0, "high": 25.0, "step": 2.5},
        "adx_rising_threshold": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.5},
        "adx_falling_threshold": {
            "type": "float",
            "low": 2.0,
            "high": 5.0,
            "step": 1.0,
        },
        "di_separation": {"type": "float", "low": 1.0, "high": 5.0, "step": 1.0},
        "price_ema_buffer": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.ema = btind.EMA(self.data.close, period=self.params.ema_period)

        # Use DirectionalIndicator for +DI and -DI
        self.di = btind.DirectionalIndicator(period=self.params.adx_period)
        self.adx = btind.ADX(period=self.params.adx_period)
        self.di_plus = self.di.plusDI
        self.di_minus = self.di.minusDI

        # Trend analysis variables
        self.trend_direction = 0  # 1 for up, -1 for down, 0 for neutral
        self.adx_rising = False
        self.adx_falling = False
        self.strong_trend = False
        self.di_bullish = False
        self.di_bearish = False

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = max(self.params.ema_period, self.params.adx_period * 2) + 5

        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized EMADXTrend with params: {self.params}")
        logger.info(
            f"EMADXTrend initialized with ema_period={self.p.ema_period}, "
            f"adx_period={self.p.adx_period}, min_adx_threshold={self.p.min_adx_threshold}, "
            f"exit_adx_threshold={self.p.exit_adx_threshold}"
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
            np.isnan(self.ema[0])
            or np.isnan(self.adx[0])
            or np.isnan(self.di_plus[0])
            or np.isnan(self.di_minus[0])
            or len(self) < 3  # Need at least 3 bars for ADX analysis
        ):
            logger.debug(f"Invalid indicator values at bar {len(self)}")
            return

        # Analyze trend direction based on EMA
        price_above_ema = self.data.close[0] > (
            self.ema[0] + self.params.price_ema_buffer
        )
        price_below_ema = self.data.close[0] < (
            self.ema[0] - self.params.price_ema_buffer
        )

        if price_above_ema:
            self.trend_direction = 1  # Uptrend
        elif price_below_ema:
            self.trend_direction = -1  # Downtrend
        else:
            self.trend_direction = 0  # Neutral

        # Analyze ADX trend strength
        if len(self) >= 3:
            adx_change = self.adx[0] - self.adx[-1]
            self.adx_rising = adx_change >= self.params.adx_rising_threshold
            self.adx_falling = adx_change <= -self.params.adx_falling_threshold

        self.strong_trend = self.adx[0] > self.params.min_adx_threshold

        # Analyze Directional Indicators
        di_diff = abs(self.di_plus[0] - self.di_minus[0])
        self.di_bullish = (
            self.di_plus[0] > self.di_minus[0] and di_diff >= self.params.di_separation
        )
        self.di_bearish = (
            self.di_minus[0] > self.di_plus[0] and di_diff >= self.params.di_separation
        )

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "ema": self.ema[0],
                "adx": self.adx[0],
                "di_plus": self.di_plus[0],
                "di_minus": self.di_minus[0],
                "trend_direction": self.trend_direction,
                "strong_trend": self.strong_trend,
                "adx_rising": self.adx_rising,
                "di_bullish": self.di_bullish,
                "di_bearish": self.di_bearish,
            }
        )

        # Trading Logic
        if not self.position:
            # Long Entry: Price above EMA + Strong ADX + Rising ADX + Bullish DI
            if (
                self.trend_direction == 1
                and self.strong_trend
                and self.adx_rising
                and self.di_bullish
            ):
                self.order = self.buy()
                self.order_type = "enter_long"

                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - EMA+ADX Trend) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"EMA: {self.ema[0]:.2f} | ADX: {self.adx[0]:.2f} (Rising) | "
                    f"DI+: {self.di_plus[0]:.2f} > DI-: {self.di_minus[0]:.2f}"
                )

            # Short Entry: Price below EMA + Strong ADX + Rising ADX + Bearish DI
            elif (
                self.trend_direction == -1
                and self.strong_trend
                and self.adx_rising
                and self.di_bearish
            ):
                self.order = self.sell()
                self.order_type = "enter_short"

                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - EMA+ADX Trend) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"EMA: {self.ema[0]:.2f} | ADX: {self.adx[0]:.2f} (Rising) | "
                    f"DI-: {self.di_minus[0]:.2f} > DI+: {self.di_plus[0]:.2f}"
                )

        elif self.position.size > 0:  # Long position
            # Long Exit: Price below EMA OR Weak ADX OR Falling ADX
            if (
                self.trend_direction != 1
                or self.adx[0] < self.params.exit_adx_threshold
                or self.adx_falling
            ):
                self.order = self.sell()
                self.order_type = "exit_long"

                exit_reason = (
                    "Price below EMA"
                    if self.trend_direction != 1
                    else (
                        "Weak ADX"
                        if self.adx[0] < self.params.exit_adx_threshold
                        else "ADX falling"
                    )
                )

                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - EMA+ADX Trend) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | EMA: {self.ema[0]:.2f} | "
                    f"ADX: {self.adx[0]:.2f}"
                )

        elif self.position.size < 0:  # Short position
            # Short Exit: Price above EMA OR Weak ADX OR Falling ADX
            if (
                self.trend_direction != -1
                or self.adx[0] < self.params.exit_adx_threshold
                or self.adx_falling
            ):
                self.order = self.buy()
                self.order_type = "exit_short"

                exit_reason = (
                    "Price above EMA"
                    if self.trend_direction != -1
                    else (
                        "Weak ADX"
                        if self.adx[0] < self.params.exit_adx_threshold
                        else "ADX falling"
                    )
                )

                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - EMA+ADX Trend) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | EMA: {self.ema[0]:.2f} | "
                    f"ADX: {self.adx[0]:.2f}"
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
                    f"Price: {order.executed.price:.2f} | Size: {order.executed.size} | "
                    f"Cost: {order.executed.value:.2f} | Comm: {order.executed.comm:.2f}"
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
                    f"Price: {order.executed.price:.2f} | Size: {order.executed.size} | "
                    f"Cost: {order.executed.value:.2f} | Comm: {order.executed.comm:.2f}"
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
                        f"Price: {order.executed.price:.2f} | PnL: {pnl:.2f}"
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
                        f"Price: {order.executed.price:.2f} | PnL: {pnl:.2f}"
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
            "ema_period": trial.suggest_int("ema_period", 15, 30),
            "adx_period": trial.suggest_int("adx_period", 10, 20),
            "min_adx_threshold": trial.suggest_float("min_adx_threshold", 20.0, 35.0),
            "exit_adx_threshold": trial.suggest_float("exit_adx_threshold", 15.0, 25.0),
            "adx_rising_threshold": trial.suggest_float(
                "adx_rising_threshold", 0.5, 2.0
            ),
            "adx_falling_threshold": trial.suggest_float(
                "adx_falling_threshold", 2.0, 5.0
            ),
            "di_separation": trial.suggest_float("di_separation", 1.0, 5.0),
            "price_ema_buffer": trial.suggest_float("price_ema_buffer", 0.0, 0.5),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            ema_period = params.get("ema_period", 21)
            adx_period = params.get("adx_period", 14)
            max_period = max(ema_period, adx_period * 2)
            return max_period + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 40
