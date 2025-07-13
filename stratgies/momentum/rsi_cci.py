import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging
from backtrader.lineseries import LineSeriesStub

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class RSICCI(bt.Strategy):
    """
    RSI + CCI Double Momentum Strategy

    This strategy combines RSI (Relative Strength Index) and CCI (Commodity Channel Index)
    to identify high-probability momentum opportunities when both oscillators confirm
    the same directional bias, creating a double momentum confirmation system.

    Strategy Type: DOUBLE MOMENTUM CONFIRMATION
    ==========================================
    This strategy uses two different momentum oscillators to confirm each other,
    reducing false signals and increasing the probability of successful trades.
    It enters when both indicators show the same momentum direction.

    Strategy Logic:
    ==============

    Long Position Rules:
    - Entry: RSI > bullish threshold AND CCI > bullish threshold (both bullish)
    - Exit: RSI < bearish threshold OR CCI < bearish threshold (either reverses)

    Short Position Rules:
    - Entry: RSI < bearish threshold AND CCI < bearish threshold (both bearish)
    - Exit: RSI > bullish threshold OR CCI > bullish threshold (either reverses)

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses warmup period to ensure indicator stability before trading
    - Prevents order overlap with pending order checks
    - Double confirmation reduces whipsaws and false signals

    Indicators Used:
    ===============
    - RSI (Relative Strength Index):
      * Measures price momentum on 0-100 scale
      * > 55: Bullish momentum
      * < 45: Bearish momentum
      * 45-55: Neutral zone

    - CCI (Commodity Channel Index):
      * Measures price deviation from statistical average
      * > +100: Bullish momentum (overbought but can continue)
      * < -100: Bearish momentum (oversold but can continue)
      * -100 to +100: Normal range

    Double Momentum Concept:
    =======================
    - RSI + CCI bullish = Strong upward momentum confirmation
    - RSI + CCI bearish = Strong downward momentum confirmation
    - Exit when either oscillator reverses = Early reversal detection
    - Different calculation methods provide diverse momentum perspective

    Features:
    =========
    - Comprehensive trade logging with IST timezone
    - Detailed PnL tracking for each completed trade
    - Position sizing and commission handling
    - Optimization-ready parameter space
    - Robust error handling and data validation
    - Support for both backtesting and live trading
    - Momentum alignment analysis
    - Automatic data format detection (Close-only vs OHLC)

    Parameters:
    ==========
    - rsi_period (int): RSI calculation period (default: 14)
    - cci_period (int): CCI calculation period (default: 14)
    - rsi_bullish (int): RSI bullish threshold (default: 55)
    - rsi_bearish (int): RSI bearish threshold (default: 45)
    - cci_bullish (int): CCI bullish threshold (default: 100)
    - cci_bearish (int): CCI bearish threshold (default: -100)
    - verbose (bool): Enable detailed logging (default: False)

    Performance Metrics:
    ===================
    - Tracks win/loss ratio
    - Calculates net PnL including commissions
    - Records trade duration and timing
    - Provides detailed execution logs
    - Monitors momentum alignment frequency

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(RSICCI, rsi_bullish=60, cci_bullish=120, cci_period=20)
    cerebro.run()

    Best Market Conditions:
    ======================
    - Strong momentum markets with clear directional moves
    - Trending markets with sustained momentum
    - Markets with good volatility for momentum generation
    - Avoid during choppy/ranging markets with conflicting signals

    Note:
    ====
    This double momentum strategy requires both oscillators to align, which
    reduces trade frequency but increases signal quality. The strategy works
    best when market conditions favor momentum-based moves.
    """

    params = (
        ("rsi_period", 14),
        ("cci_period", 14),
        ("rsi_bullish", 55),
        ("rsi_bearish", 45),
        ("cci_bullish", 100),
        ("cci_bearish", -100),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "cci_period": {"type": "int", "low": 10, "high": 25, "step": 1},
        "rsi_bullish": {"type": "int", "low": 50, "high": 65, "step": 1},
        "rsi_bearish": {"type": "int", "low": 35, "high": 50, "step": 1},
        "cci_bullish": {"type": "int", "low": 80, "high": 150, "step": 10},
        "cci_bearish": {"type": "int", "low": -150, "high": -80, "step": 10},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        logger.info(f"Initializing RSICCI Strategy with data: {self.data}")

        # Initialize RSI indicator
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)

        # Check if we have OHLC data
        self.has_ohlc = self._check_ohlc_data()
        logger.info(f"OHLC data available: {self.has_ohlc}")

        # Initialize CCI indicator - Fixed initialization
        if self.has_ohlc:
            try:
                # Use proper initialization for CCI with OHLC data
                # CCI in backtrader uses the data feed directly, not separate H,L,C parameters
                self.cci = btind.CommodityChannelIndex(
                    self.data,  # Pass the data feed directly
                    period=self.params.cci_period,
                )
                logger.info("Successfully initialized CCI with OHLC data")
            except Exception as e:
                logger.error(f"Failed to initialize CCI with OHLC data: {e}")
                # Fallback to close-only if OHLC fails
                logger.info("Falling back to close-based CCI calculation")
                # Create a custom CCI using close data only
                self.cci = self._create_close_based_cci()
        else:
            # If no OHLC data, create custom CCI using close data
            logger.info("Using close data for CCI calculation")
            self.cci = self._create_close_based_cci()

        # Momentum direction signals
        self.rsi_bullish_signal = self.rsi > self.params.rsi_bullish
        self.rsi_bearish_signal = self.rsi < self.params.rsi_bearish
        self.cci_bullish_signal = self.cci > self.params.cci_bullish
        self.cci_bearish_signal = self.cci < self.params.cci_bearish

        # Double momentum confirmations - FIXED: Use bt.And and bt.Or instead of 'and'/'or'
        self.double_bullish = bt.And(self.rsi_bullish_signal, self.cci_bullish_signal)
        self.double_bearish = bt.And(self.rsi_bearish_signal, self.cci_bearish_signal)

        # Exit conditions - use bt.Or for "either reverses"
        self.long_exit_signal = bt.Or(self.rsi_bearish_signal, self.cci_bearish_signal)
        self.short_exit_signal = bt.Or(self.rsi_bullish_signal, self.cci_bullish_signal)

        self.order = None
        self.order_type = None
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.rsi_period,
                self.params.cci_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized RSICCI with params: {self.params}")
        logger.info(
            f"RSICCI initialized with rsi_period={self.p.rsi_period}, "
            f"cci_period={self.p.cci_period}, "
            f"rsi_bullish={self.p.rsi_bullish}, rsi_bearish={self.p.rsi_bearish}, "
            f"cci_bullish={self.p.cci_bullish}, cci_bearish={self.p.cci_bearish}, "
            f"data_format={'OHLC' if self.has_ohlc else 'Close-only'}"
        )

    def _create_close_based_cci(self):
        """Create a custom CCI indicator using only close prices"""

        class CustomCCI(bt.Indicator):
            lines = ("cci",)
            params = (("period", 14),)

            def __init__(self):
                # Calculate typical price using close only (H=L=C for close-only data)
                typical_price = self.data.close

                # Calculate SMA of typical price
                sma_tp = bt.indicators.SimpleMovingAverage(
                    typical_price, period=self.p.period
                )

                # Calculate mean deviation
                mean_dev = bt.indicators.MeanDeviation(
                    typical_price, period=self.p.period
                )

                # CCI formula: (Typical Price - SMA) / (0.015 * Mean Deviation)
                self.lines.cci = (typical_price - sma_tp) / (0.015 * mean_dev)

        return CustomCCI(period=self.params.cci_period)

    def _check_ohlc_data(self):
        """Check if the data source has valid OHLC data"""
        try:
            # Check for presence of high, low, and close attributes
            if not (
                hasattr(self.data, "high")
                and hasattr(self.data, "low")
                and hasattr(self.data, "close")
            ):
                logger.info("Data missing high, low, or close attributes")
                return False

            # Try to access the first value to ensure data is available
            try:
                # Check if we can access current values
                if len(self.data) > 0:
                    high_val = self.data.high[0]
                    low_val = self.data.low[0]
                    close_val = self.data.close[0]

                    # Check for valid values and that H >= L >= 0
                    if (
                        np.isnan(high_val)
                        or np.isnan(low_val)
                        or np.isnan(close_val)
                        or np.isinf(high_val)
                        or np.isinf(low_val)
                        or np.isinf(close_val)
                        or high_val < low_val
                    ):
                        logger.info("OHLC data contains invalid values or H < L")
                        return False

                    logger.info(
                        f"Valid OHLC data detected - High: {high_val}, Low: {low_val}, Close: {close_val}"
                    )
                    return True
                else:
                    logger.info("No data available yet")
                    return False

            except (IndexError, AttributeError) as e:
                logger.warning(f"Error accessing OHLC data: {e}")
                return False

        except Exception as e:
            logger.warning(f"Error checking OHLC data availability: {e}")
            return False

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
            or np.isnan(self.cci[0])
            or np.isinf(self.rsi[0])
            or np.isinf(self.cci[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"RSI={self.rsi[0]}, CCI={self.cci[0]}"
            )
            return

        # Calculate momentum alignment for logging
        momentum_alignment = "NEUTRAL"
        if self.double_bullish[0]:
            momentum_alignment = "DOUBLE_BULLISH"
        elif self.double_bearish[0]:
            momentum_alignment = "DOUBLE_BEARISH"
        elif self.rsi_bullish_signal[0] or self.cci_bullish_signal[0]:
            momentum_alignment = "MIXED_BULLISH"
        elif self.rsi_bearish_signal[0] or self.cci_bearish_signal[0]:
            momentum_alignment = "MIXED_BEARISH"

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "rsi": self.rsi[0],
                "cci": self.cci[0],
                "rsi_bullish": self.rsi_bullish_signal[0],
                "rsi_bearish": self.rsi_bearish_signal[0],
                "cci_bullish": self.cci_bullish_signal[0],
                "cci_bearish": self.cci_bearish_signal[0],
                "double_bullish": self.double_bullish[0],
                "double_bearish": self.double_bearish[0],
                "momentum_alignment": momentum_alignment,
            }
        )

        # Double Momentum Position Management
        if not self.position:
            # Long Entry: Both RSI and CCI in bullish momentum zone
            if self.double_bullish[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Double Momentum) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} > {self.params.rsi_bullish} (Bullish) | "
                    f"CCI: {self.cci[0]:.2f} > {self.params.cci_bullish} (Bullish) | "
                    f"Momentum: {momentum_alignment}"
                )
            # Short Entry: Both RSI and CCI in bearish momentum zone
            elif self.double_bearish[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Double Momentum) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} < {self.params.rsi_bearish} (Bearish) | "
                    f"CCI: {self.cci[0]:.2f} < {self.params.cci_bearish} (Bearish) | "
                    f"Momentum: {momentum_alignment}"
                )
        elif self.position.size > 0:  # Long position
            # Long Exit: Either RSI or CCI reverses to bearish - use the pre-calculated signal
            if self.long_exit_signal[0]:
                self.order = self.sell()
                self.order_type = "exit_long"
                reversal_indicator = (
                    "RSI" if self.rsi[0] < self.params.rsi_bearish else "CCI"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - Momentum Reversal) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reversal: {reversal_indicator} turned bearish | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"CCI: {self.cci[0]:.2f} | "
                    f"Momentum: {momentum_alignment}"
                )
        elif self.position.size < 0:  # Short position
            # Short Exit: Either RSI or CCI reverses to bullish - use the pre-calculated signal
            if self.short_exit_signal[0]:
                self.order = self.buy()
                self.order_type = "exit_short"
                reversal_indicator = (
                    "RSI" if self.rsi[0] > self.params.rsi_bullish else "CCI"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - Momentum Reversal) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reversal: {reversal_indicator} turned bullish | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"CCI: {self.cci[0]:.2f} | "
                    f"Momentum: {momentum_alignment}"
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
            "cci_period": trial.suggest_int("cci_period", 10, 25),
            "rsi_bullish": trial.suggest_int("rsi_bullish", 50, 65),
            "rsi_bearish": trial.suggest_int("rsi_bearish", 35, 50),
            "cci_bullish": trial.suggest_int("cci_bullish", 80, 150),
            "cci_bearish": trial.suggest_int("cci_bearish", -150, -80),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            cci_period = params.get("cci_period", 14)
            max_period = max(rsi_period, cci_period)
            return max_period + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
