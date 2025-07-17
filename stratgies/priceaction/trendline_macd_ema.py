import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging
from scipy import stats

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class TrendlineDetector(bt.Indicator):
    """
    Dynamic Trendline Detection and Break Identification
    """

    lines = ("trendline", "trend_direction", "trendline_break", "breakout_direction")
    params = (
        ("period", 20),
        ("min_touches", 3),
        ("break_threshold", 0.001),  # 0.1% threshold for break confirmation
    )

    def __init__(self):
        self.addminperiod(self.params.period)
        self.trendline_slope = 0
        self.trendline_intercept = 0
        self.last_trendline_value = 0
        self.trend_points = []

    def next(self):
        if len(self.data) < self.params.period:
            return

        # Get recent price data
        recent_highs = [self.data.high[-i] for i in range(self.params.period)]
        recent_lows = [self.data.low[-i] for i in range(self.params.period)]
        recent_closes = [self.data.close[-i] for i in range(self.params.period)]

        # Detect trend direction using EMA comparison
        current_trend = self._detect_trend()
        self.lines.trend_direction[0] = current_trend

        # Calculate trendline based on trend direction
        if current_trend > 0:  # Uptrend - use lows for support trendline
            trendline_value = self._calculate_trendline(
                recent_lows, trend_type="support"
            )
        else:  # Downtrend - use highs for resistance trendline
            trendline_value = self._calculate_trendline(
                recent_highs, trend_type="resistance"
            )

        self.lines.trendline[0] = trendline_value

        # Detect trendline breaks
        if len(self.data) > 1:
            self.lines.trendline_break[0] = self._detect_break(trendline_value)
            self.lines.breakout_direction[0] = self._get_breakout_direction(
                trendline_value
            )
        else:
            self.lines.trendline_break[0] = False
            self.lines.breakout_direction[0] = 0

        self.last_trendline_value = trendline_value

    def _detect_trend(self):
        # Simple trend detection using price momentum
        if len(self.data) < 10:
            return 0

        recent_prices = [self.data.close[-i] for i in range(10)]
        x = np.arange(len(recent_prices))
        slope, _, _, _, _ = stats.linregress(x, recent_prices)

        return 1 if slope > 0 else -1

    def _calculate_trendline(self, price_data, trend_type="support"):
        # Find significant points for trendline calculation
        if trend_type == "support":
            # For support, find local minima
            significant_points = self._find_local_minima(price_data)
        else:
            # For resistance, find local maxima
            significant_points = self._find_local_maxima(price_data)

        if len(significant_points) < 2:
            return self.data.close[0]

        # Calculate trendline using linear regression
        x_vals = [point[0] for point in significant_points[-3:]]  # Use last 3 points
        y_vals = [point[1] for point in significant_points[-3:]]

        if len(x_vals) >= 2:
            slope, intercept, _, _, _ = stats.linregress(x_vals, y_vals)
            self.trendline_slope = slope
            self.trendline_intercept = intercept

            # Calculate current trendline value
            current_x = len(self.data) - 1
            trendline_value = slope * current_x + intercept
            return trendline_value

        return self.data.close[0]

    def _find_local_minima(self, price_data):
        minima = []
        for i in range(1, len(price_data) - 1):
            if price_data[i] < price_data[i - 1] and price_data[i] < price_data[i + 1]:
                minima.append((len(self.data) - len(price_data) + i, price_data[i]))
        return minima

    def _find_local_maxima(self, price_data):
        maxima = []
        for i in range(1, len(price_data) - 1):
            if price_data[i] > price_data[i - 1] and price_data[i] > price_data[i + 1]:
                maxima.append((len(self.data) - len(price_data) + i, price_data[i]))
        return maxima

    def _detect_break(self, current_trendline):
        if self.last_trendline_value == 0:
            return False

        current_price = self.data.close[0]

        # Check for break based on trend direction
        if self.lines.trend_direction[0] > 0:  # Uptrend support break
            return current_price < current_trendline * (1 - self.params.break_threshold)
        else:  # Downtrend resistance break
            return current_price > current_trendline * (1 + self.params.break_threshold)

    def _get_breakout_direction(self, current_trendline):
        if not self.lines.trendline_break[0]:
            return 0

        current_price = self.data.close[0]

        if current_price > current_trendline:
            return 1  # Bullish breakout
        elif current_price < current_trendline:
            return -1  # Bearish breakout
        else:
            return 0


class TrendlineMACD_EMA(bt.Strategy):
    """
    Trendline + MACD + EMA Strategy
    Strategy Type: BREAKOUT + MOMENTUM + TREND
    ==========================================
    This strategy combines trendline breaks, MACD momentum, and EMA trend confirmation.

    Strategy Logic:
    ==============
    Long Entry: Bullish trendline break + MACD bullish crossover + Price above EMA
    Short Entry: Bearish trendline break + MACD bearish crossover + Price below EMA
    Exit: False breakout (price returns to trendline) or MACD momentum failure

    Parameters:
    ==========
    - trendline_period (int): Trendline detection period (default: 20)
    - trendline_break_threshold (float): Break confirmation threshold (default: 0.001)
    - macd_fast (int): MACD fast EMA period (default: 12)
    - macd_slow (int): MACD slow EMA period (default: 26)
    - macd_signal (int): MACD signal line period (default: 9)
    - ema_period (int): Trend confirmation EMA period (default: 50)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("trendline_period", 20),
        ("trendline_break_threshold", 0.001),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("ema_period", 50),
        ("verbose", False),
    )

    optimization_params = {
        "trendline_period": {"type": "int", "low": 15, "high": 30, "step": 1},
        "trendline_break_threshold": {
            "type": "float",
            "low": 0.0005,
            "high": 0.002,
            "step": 0.0001,
        },
        "macd_fast": {"type": "int", "low": 8, "high": 16, "step": 1},
        "macd_slow": {"type": "int", "low": 20, "high": 30, "step": 1},
        "macd_signal": {"type": "int", "low": 5, "high": 12, "step": 1},
        "ema_period": {"type": "int", "low": 30, "high": 70, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.trendline = TrendlineDetector(
            self.data,
            period=self.params.trendline_period,
            break_threshold=self.params.trendline_break_threshold,
        )
        self.macd = btind.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal,
        )
        self.ema = btind.EMA(self.data.close, period=self.params.ema_period)

        # Debug: Log available lines and their types
        logger.debug(f"Trendline lines: {self.trendline.lines.getlinealiases()}")
        logger.debug(f"MACD lines: {self.macd.lines.getlinealiases()}")
        logger.debug(f"EMA lines: {self.ema.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.trendline_period,
                self.params.macd_slow,
                self.params.ema_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized TrendlineMACD_EMA with params: {self.params}")
        logger.info(
            f"TrendlineMACD_EMA initialized with trendline_period={self.p.trendline_period}, "
            f"trendline_break_threshold={self.p.trendline_break_threshold}, "
            f"macd_fast={self.p.macd_fast}, macd_slow={self.p.macd_slow}, "
            f"macd_signal={self.p.macd_signal}, ema_period={self.p.ema_period}"
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
            np.isnan(self.trendline.trendline[0])
            or np.isnan(self.macd.macd[0])
            or np.isnan(self.macd.signal[0])
            or np.isnan(self.ema[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"Trendline={self.trendline.trendline[0]}, "
                f"MACD={self.macd.macd[0]}, Signal={self.macd.signal[0]}, "
                f"EMA={self.ema[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "trendline": self.trendline.trendline[0],
                "trend_direction": self.trendline.trend_direction[0],
                "trendline_break": self.trendline.trendline_break[0],
                "breakout_direction": self.trendline.breakout_direction[0],
                "macd": self.macd.macd[0],
                "macd_signal": self.macd.signal[0],
                "ema": self.ema[0],
            }
        )

        # Trading Logic
        macd_bullish = (
            self.macd.macd[0] > self.macd.signal[0]
            and self.macd.macd[-1] <= self.macd.signal[-1]
        )
        macd_bearish = (
            self.macd.macd[0] < self.macd.signal[0]
            and self.macd.macd[-1] >= self.macd.signal[-1]
        )
        price_above_ema = self.data.close[0] > self.ema[0]
        price_below_ema = self.data.close[0] < self.ema[0]
        bullish_break = (
            self.trendline.trendline_break[0]
            and self.trendline.breakout_direction[0] == 1
        )
        bearish_break = (
            self.trendline.trendline_break[0]
            and self.trendline.breakout_direction[0] == -1
        )

        if not self.position:
            # Long Entry: Bullish trendline break + MACD bullish crossover + Price above EMA
            if bullish_break and macd_bullish and price_above_ema:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Trendline + MACD + EMA) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Trendline: {self.trendline.trendline[0]:.2f} (Bullish Break) | "
                    f"MACD: {self.macd.macd[0]:.2f} > Signal: {self.macd.signal[0]:.2f} (Bullish) | "
                    f"EMA: {self.ema[0]:.2f} (Price Above)"
                )
            # Short Entry: Bearish trendline break + MACD bearish crossover + Price below EMA
            elif bearish_break and macd_bearish and price_below_ema:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Trendline + MACD + EMA) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Trendline: {self.trendline.trendline[0]:.2f} (Bearish Break) | "
                    f"MACD: {self.macd.macd[0]:.2f} < Signal: {self.macd.signal[0]:.2f} (Bearish) | "
                    f"EMA: {self.ema[0]:.2f} (Price Below)"
                )
        elif self.position.size > 0:  # Long position
            # Exit: False breakout (price returns below trendline) or MACD momentum failure
            false_breakout = self.data.close[0] < self.trendline.trendline[0]
            macd_momentum_failure = self.macd.macd[0] < self.macd.signal[0]
            if false_breakout or macd_momentum_failure:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "False breakout" if false_breakout else "MACD momentum failure"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - Trendline + MACD + EMA) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Trendline: {self.trendline.trendline[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.2f}, Signal: {self.macd.signal[0]:.2f} | "
                    f"EMA: {self.ema[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            # Exit: False breakout (price returns above trendline) or MACD momentum failure
            false_breakout = self.data.close[0] > self.trendline.trendline[0]
            macd_momentum_failure = self.macd.macd[0] > self.macd.signal[0]
            if false_breakout or macd_momentum_failure:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "False breakout" if false_breakout else "MACD momentum failure"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - Trendline + MACD + EMA) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Trendline: {self.trendline.trendline[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.2f}, Signal: {self.macd.signal[0]:.2f} | "
                    f"EMA: {self.ema[0]:.2f}"
                )

    def notify_order(self, order):
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
                        "commission pinc": total_commission,
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
            "trendline_period": trial.suggest_int("trendline_period", 15, 30),
            "trendline_break_threshold": trial.suggest_float(
                "trendline_break_threshold", 0.0005, 0.002, step=0.0001
            ),
            "macd_fast": trial.suggest_int("macd_fast", 8, 16),
            "macd_slow": trial.suggest_int("macd_slow", 20, 30),
            "macd_signal": trial.suggest_int("macd_signal", 5, 12),
            "ema_period": trial.suggest_int("ema_period", 30, 70),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            trendline_period = params.get("trendline_period", 20)
            macd_slow = params.get("macd_slow", 26)
            ema_period = params.get("ema_period", 50)
            return max(trendline_period, macd_slow, ema_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 70
