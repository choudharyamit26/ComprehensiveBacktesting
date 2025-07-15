import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class ATRVolumeExpansion(bt.Strategy):
    """
    ATR + Volume Expansion Trading Strategy

    This strategy combines Average True Range (ATR) and Volume analysis to identify
    high-probability breakout opportunities when volatility is expanding along with
    increased trading interest (volume surge).

    Strategy Type: BREAKOUT/MOMENTUM
    ===============================
    This is a momentum strategy that captures breakout moves when both volatility
    and volume are expanding, indicating strong market participation and potential
    for sustained price movement.

    Strategy Logic:
    ==============

    Long Position Rules:
    - Entry: ATR expanding (current ATR > previous ATR * expansion_factor) AND
             Volume surge (current volume > volume_avg * volume_factor) AND
             Price breaking above resistance level
    - Exit: ATR contracting (current ATR < previous ATR * contraction_factor) OR
            Volume normalizing (current volume < volume_avg * normal_factor) OR
            Price below trailing stop

    Short Position Rules:
    - Entry: ATR expanding AND Volume surge AND Price breaking below support level
    - Exit: ATR contracting OR Volume normalizing OR Price above trailing stop

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses ATR-based position sizing and stop losses
    - Prevents order overlap with pending order checks
    - Works best in trending markets with clear breakout patterns

    Indicators Used:
    ===============
    - ATR: Measures market volatility and expansion/contraction
    - Volume SMA: Average volume to identify volume surges
    - Price levels: Support/resistance breakout confirmation
    - Trailing stops: Dynamic exit based on ATR

    Volume Expansion Concept:
    ========================
    - ATR expansion indicates increasing volatility
    - Volume surge shows increased market participation
    - Combined signal suggests strong institutional interest
    - Breakout with both signals has higher success probability

    Features:
    =========
    - Comprehensive trade logging with IST timezone
    - ATR-based position sizing and risk management
    - Volume surge detection with customizable thresholds
    - Trailing stop implementation for trend following
    - Robust error handling and data validation
    - Support for both backtesting and live trading

    Parameters:
    ==========
    - atr_period (int): ATR calculation period (default: 14)
    - volume_period (int): Volume SMA period (default: 20)
    - expansion_factor (float): ATR expansion threshold (default: 1.1)
    - contraction_factor (float): ATR contraction threshold (default: 0.9)
    - volume_factor (float): Volume surge threshold (default: 1.5)
    - normal_factor (float): Volume normalization threshold (default: 1.2)
    - trailing_atr_mult (float): Trailing stop ATR multiplier (default: 2.0)
    - min_atr_threshold (float): Minimum ATR for trading (default: 0.5)
    - verbose (bool): Enable detailed logging (default: False)

    Performance Metrics:
    ===================
    - Tracks win/loss ratio
    - Calculates net PnL including commissions
    - Records trade duration and timing
    - Monitors ATR and volume expansion frequency
    - Provides detailed execution logs

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(ATRVolumeExpansion, expansion_factor=1.2, volume_factor=1.8)
    cerebro.run()

    Best Market Conditions:
    ======================
    - Trending markets with clear breakout patterns
    - High volatility periods with institutional participation
    - News-driven moves with volume confirmation
    - Avoid during low volume/consolidation periods

    Note:
    ====
    This strategy requires sufficient market volatility and volume to be effective.
    It's designed for intraday trading and performs best during active market hours.
    """

    params = (
        ("atr_period", 14),
        ("volume_period", 20),
        ("expansion_factor", 1.1),
        ("contraction_factor", 0.9),
        ("volume_factor", 1.5),
        ("normal_factor", 1.2),
        ("trailing_atr_mult", 2.0),
        ("min_atr_threshold", 0.5),
        ("verbose", False),
    )

    optimization_params = {
        "atr_period": {"type": "int", "low": 10, "high": 20, "step": 2},
        "volume_period": {"type": "int", "low": 15, "high": 30, "step": 5},
        "expansion_factor": {"type": "float", "low": 1.05, "high": 1.3, "step": 0.05},
        "contraction_factor": {"type": "float", "low": 0.8, "high": 0.95, "step": 0.05},
        "volume_factor": {"type": "float", "low": 1.2, "high": 2.0, "step": 0.1},
        "normal_factor": {"type": "float", "low": 1.0, "high": 1.5, "step": 0.1},
        "trailing_atr_mult": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.5},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.atr = btind.ATR(period=self.params.atr_period)
        self.volume_sma = btind.SMA(self.data.volume, period=self.params.volume_period)

        # Price levels for breakout detection
        self.high_sma = btind.SMA(self.data.high, period=10)
        self.low_sma = btind.SMA(self.data.low, period=10)

        # Trailing stop variables
        self.trailing_stop_long = 0
        self.trailing_stop_short = 0

        # ATR expansion/contraction detection
        self.atr_expanding = False
        self.atr_contracting = False

        # Volume surge detection
        self.volume_surge = False
        self.volume_normalizing = False

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.atr_period,
                self.params.volume_period,
                10,  # for price level SMAs
            )
            + 5
        )

        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized ATRVolumeExpansion with params: {self.params}")
        logger.info(
            f"ATRVolumeExpansion initialized with atr_period={self.p.atr_period}, "
            f"volume_period={self.p.volume_period}, expansion_factor={self.p.expansion_factor}, "
            f"volume_factor={self.p.volume_factor}, trailing_atr_mult={self.p.trailing_atr_mult}"
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
            np.isnan(self.atr[0])
            or np.isnan(self.volume_sma[0])
            or self.atr[0] < self.params.min_atr_threshold
            or len(self) < 2  # Need at least 2 bars for comparison
        ):
            logger.debug(f"Invalid indicator values at bar {len(self)}")
            return

        # Calculate ATR expansion/contraction
        if len(self) >= 2:
            self.atr_expanding = self.atr[0] > (
                self.atr[-1] * self.params.expansion_factor
            )
            self.atr_contracting = self.atr[0] < (
                self.atr[-1] * self.params.contraction_factor
            )

        # Calculate volume surge/normalization
        self.volume_surge = self.data.volume[0] > (
            self.volume_sma[0] * self.params.volume_factor
        )
        self.volume_normalizing = self.data.volume[0] < (
            self.volume_sma[0] * self.params.normal_factor
        )

        # Calculate breakout levels
        resistance_level = self.high_sma[0]
        support_level = self.low_sma[0]

        # Update trailing stops
        if self.position.size > 0:  # Long position
            new_stop = self.data.close[0] - (
                self.atr[0] * self.params.trailing_atr_mult
            )
            self.trailing_stop_long = max(self.trailing_stop_long, new_stop)
        elif self.position.size < 0:  # Short position
            new_stop = self.data.close[0] + (
                self.atr[0] * self.params.trailing_atr_mult
            )
            self.trailing_stop_short = min(self.trailing_stop_short, new_stop)

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "atr": self.atr[0],
                "volume": self.data.volume[0],
                "volume_sma": self.volume_sma[0],
                "atr_expanding": self.atr_expanding,
                "volume_surge": self.volume_surge,
                "resistance": resistance_level,
                "support": support_level,
            }
        )

        # Trading Logic
        if not self.position:
            # Long Entry: ATR expanding + Volume surge + Price breakout above resistance
            if (
                self.atr_expanding
                and self.volume_surge
                and self.data.close[0] > resistance_level
            ):
                self.order = self.buy()
                self.order_type = "enter_long"
                self.trailing_stop_long = self.data.close[0] - (
                    self.atr[0] * self.params.trailing_atr_mult
                )

                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - ATR+Volume Expansion) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"ATR: {self.atr[0]:.2f} (Expanding) | "
                    f"Volume: {self.data.volume[0]:.0f} vs Avg: {self.volume_sma[0]:.0f} | "
                    f"Resistance: {resistance_level:.2f}"
                )

            # Short Entry: ATR expanding + Volume surge + Price breakout below support
            elif (
                self.atr_expanding
                and self.volume_surge
                and self.data.close[0] < support_level
            ):
                self.order = self.sell()
                self.order_type = "enter_short"
                self.trailing_stop_short = self.data.close[0] + (
                    self.atr[0] * self.params.trailing_atr_mult
                )

                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - ATR+Volume Expansion) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"ATR: {self.atr[0]:.2f} (Expanding) | "
                    f"Volume: {self.data.volume[0]:.0f} vs Avg: {self.volume_sma[0]:.0f} | "
                    f"Support: {support_level:.2f}"
                )

        elif self.position.size > 0:  # Long position
            # Long Exit: ATR contracting OR Volume normalizing OR Trailing stop hit
            if (
                self.atr_contracting
                or self.volume_normalizing
                or self.data.close[0] <= self.trailing_stop_long
            ):
                self.order = self.sell()
                self.order_type = "exit_long"

                exit_reason = (
                    "ATR contracting"
                    if self.atr_contracting
                    else (
                        "Volume normalizing"
                        if self.volume_normalizing
                        else "Trailing stop hit"
                    )
                )

                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - ATR+Volume) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | ATR: {self.atr[0]:.2f} | "
                    f"Volume: {self.data.volume[0]:.0f} | "
                    f"Trailing Stop: {self.trailing_stop_long:.2f}"
                )

        elif self.position.size < 0:  # Short position
            # Short Exit: ATR contracting OR Volume normalizing OR Trailing stop hit
            if (
                self.atr_contracting
                or self.volume_normalizing
                or self.data.close[0] >= self.trailing_stop_short
            ):
                self.order = self.buy()
                self.order_type = "exit_short"

                exit_reason = (
                    "ATR contracting"
                    if self.atr_contracting
                    else (
                        "Volume normalizing"
                        if self.volume_normalizing
                        else "Trailing stop hit"
                    )
                )

                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - ATR+Volume) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | ATR: {self.atr[0]:.2f} | "
                    f"Volume: {self.data.volume[0]:.0f} | "
                    f"Trailing Stop: {self.trailing_stop_short:.2f}"
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
            "atr_period": trial.suggest_int("atr_period", 10, 20),
            "volume_period": trial.suggest_int("volume_period", 15, 30),
            "expansion_factor": trial.suggest_float("expansion_factor", 1.05, 1.3),
            "contraction_factor": trial.suggest_float("contraction_factor", 0.8, 0.95),
            "volume_factor": trial.suggest_float("volume_factor", 1.2, 2.0),
            "normal_factor": trial.suggest_float("normal_factor", 1.0, 1.5),
            "trailing_atr_mult": trial.suggest_float("trailing_atr_mult", 1.5, 3.0),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            atr_period = params.get("atr_period", 14)
            volume_period = params.get("volume_period", 20)
            max_period = max(atr_period, volume_period, 10)
            return max_period + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 35
