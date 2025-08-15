import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class SuperTrendPSAR(bt.Strategy):
    """
       Supertrend + Parabolic SAR Combined Trading Strategy

       This strategy combines Supertrend and Parabolic SAR indicators to identify
       high-probability trend-following opportunities when both indicators agree
       on market direction, providing strong confirmation signals.

       Strategy Type: TREND FOLLOWING
       ==============================
       This is a dual-confirmation trend-following strategy that enters positions
       only when both Supertrend and Parabolic SAR indicators signal the same
       direction, maximizing trend reliability and reducing false signals.

       Strategy Logic:
       ==============
       Long Position Rules:
       - Entry: Price > Supertrend AND PSAR < price (bullish)
       - Exit: Price < Supertrend OR PSAR > price (trend reversal)

       Short Position Rules:
       - Entry: Price < Supertrend AND PSAR > price (bearish)
       - Exit: Price > Supertrend OR PSAR < price (trend reversal)

       Risk Management:
       ===============
       - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
       - Force closes all positions at 3:15 PM IST to avoid overnight risk
       - Uses warmup period to ensure indicator stability
       - Prevents order overlap with pending order checks
       - Works best in trending markets with clear direction

       Indicators Used:
       ===============
       - Supertrend: ATR-based trend following indicator
         * Price > Supertrend: Bullish trend
         * Price < Supertrend: Bearish trend
       - Parabolic SAR: Time/price exit system
         * PSAR < price: Bullish trend
         * PSAR > price: Bearish trend

       Features:
       =========
       - Comprehensive trade logging with IST timezone
       - Detailed PnL tracking for each completed trade
       - Position sizing and commission handling
       - Optimization-ready parameter space
       - Robust error handling and data validation
       - Support for both backtesting and live trading

       Parameters:
       ==========
       - st_period (int): Supertrend ATR period (default: 10)
       - st_multiplier (float): Supertrend ATR multiplier (default: 3.0)
       - psar_af (float): PSAR acceleration factor (default: 0.02)
       - psar_afmax (float): PSAR maximum acceleration (default: 0.2)
       - verbose (bool): Enable detailed logging (default: False)

       Performance Metrics:
       ===================
    - Tracks win/loss ratio
       - Calculates net PnL including commissions
       - Records trade duration and timing
       - Provides detailed execution logs

       Usage:
       ======
       cerebro = bt.Cerebro()
       cerebro.addstrategy(SuperTrendPSAR, st_multiplier=2.5, psar_af=0.03)
       cerebro.run()

       Best Market Conditions:
       ======================
       - Strong trending markets with sustained momentum
       - Clear directional moves with minimal noise
       - High volatility periods with trend persistence
       - Avoid during sideways/consolidation periods
    """

    params = (
        ("st_period", 10),
        ("st_multiplier", 3.0),
        ("psar_af", 0.02),
        ("psar_afmax", 0.2),
        ("verbose", False),
    )

    optimization_params = {
        "st_period": {"type": "int", "low": 7, "high": 15, "step": 1},
        "st_multiplier": {"type": "float", "low": 2.0, "high": 4.0, "step": 0.5},
        "psar_af": {"type": "float", "low": 0.01, "high": 0.05, "step": 0.01},
        "psar_afmax": {"type": "float", "low": 0.15, "high": 0.25, "step": 0.05},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.supertrend = self._create_supertrend()
        self.psar = btind.ParabolicSAR(
            af=self.params.psar_af, afmax=self.params.psar_afmax
        )

        # Trend direction indicators
        self.uptrend = self.data.close > self.supertrend
        self.downtrend = self.data.close < self.supertrend
        self.psar_bullish = self.data.close > self.psar
        self.psar_bearish = self.data.close < self.psar

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = max(
            self.params.st_period + 5,  # Supertrend needs ATR warmup
            20,  # PSAR needs sufficient data
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized SuperTrendPSAR with params: {self.params}")
        logger.info(
            f"SuperTrendPSAR initialized with st_period={self.p.st_period}, "
            f"st_multiplier={self.p.st_multiplier}, psar_af={self.p.psar_af}, "
            f"psar_afmax={self.p.psar_afmax}"
        )

    def _create_supertrend(self):
        """Create Supertrend indicator"""

        class SuperTrend(bt.Indicator):
            lines = ("supertrend",)
            params = (
                ("period", 10),
                ("multiplier", 3.0),
            )

            def __init__(self):
                self.atr = btind.ATR(period=self.p.period)
                self.hl2 = (self.data.high + self.data.low) / 2.0
                self.basic_upperband = self.hl2 + (self.p.multiplier * self.atr)
                self.basic_lowerband = self.hl2 - (self.p.multiplier * self.atr)

            def next(self):
                if len(self) < 2:
                    self.lines.supertrend[0] = self.hl2[0]
                    return

                if (
                    self.basic_upperband[0] < self.basic_upperband[-1]
                    or self.data.close[-1] > self.basic_upperband[-1]
                ):
                    final_upperband = self.basic_upperband[0]
                else:
                    final_upperband = self.basic_upperband[-1]

                if (
                    self.basic_lowerband[0] > self.basic_lowerband[-1]
                    or self.data.close[-1] < self.basic_lowerband[-1]
                ):
                    final_lowerband = self.basic_lowerband[0]
                else:
                    final_lowerband = self.basic_lowerband[-1]

                if (
                    self.lines.supertrend[-1] == final_upperband
                    and self.data.close[0] < final_upperband
                ):
                    self.lines.supertrend[0] = final_upperband
                elif (
                    self.lines.supertrend[-1] == final_lowerband
                    and self.data.close[0] > final_lowerband
                ):
                    self.lines.supertrend[0] = final_lowerband
                elif self.data.close[0] <= final_lowerband:
                    self.lines.supertrend[0] = final_upperband
                else:
                    self.lines.supertrend[0] = final_lowerband

        return SuperTrend(
            period=self.params.st_period, multiplier=self.params.st_multiplier
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

        # Enhanced check for invalid or unavailable indicator data
        if (
            np.isnan(self.supertrend[0])
            or np.isnan(self.psar[0])
            or self.supertrend[0] is None
            or self.psar[0] is None
            or np.isnan(self.data.close[0])
        ):
            logger.warning(
                f"No indicator data available at bar {len(self)}: "
                f"Supertrend={self.supertrend[0]}, PSAR={self.psar[0]}, "
                f"Close={self.data.close[0]}"
            )
            return

        # Calculate trend strength
        price_st_diff = self.data.close[0] - self.supertrend[0]
        price_st_pct = (price_st_diff / self.supertrend[0]) * 100

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "supertrend": self.supertrend[0],
                "psar": self.psar[0],
                "price_st_diff": price_st_diff,
                "price_st_pct": price_st_pct,
                "uptrend": self.uptrend[0],
                "downtrend": self.downtrend[0],
                "psar_bullish": self.psar_bullish[0],
                "psar_bearish": self.psar_bearish[0],
            }
        )

        # Trend Following Position Management
        if not self.position:
            # Long Entry: Both indicators confirm bullish trend
            if self.uptrend[0] and self.psar_bullish[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Trend Following) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Supertrend: {self.supertrend[0]:.2f} | "
                    f"PSAR: {self.psar[0]:.2f} | "
                    f"Price vs Supertrend: {price_st_pct:.2f}% (Uptrend) | "
                    f"Trend: BULLISH"
                )
            # Short Entry: Both indicators confirm bearish trend
            elif self.downtrend[0] and self.psar_bearish[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Trend Following) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Supertrend: {self.supertrend[0]:.2f} | "
                    f"PSAR: {self.psar[0]:.2f} | "
                    f"Price vs Supertrend: {price_st_pct:.2f}% (Downtrend) | "
                    f"Trend: BEARISH"
                )
        elif self.position.size > 0:  # Long position
            # Long Exit: Either indicator signals trend reversal
            if self.downtrend[0] or self.psar_bearish[0]:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "PSAR turned bearish"
                    if self.psar_bearish[0]
                    else "Price crossed below Supertrend"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - Trend Reversal) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Supertrend: {self.supertrend[0]:.2f} | "
                    f"PSAR: {self.psar[0]:.2f} | "
                    f"Price vs Supertrend: {price_st_pct:.2f}%"
                )
        elif self.position.size < 0:  # Short position
            # Short Exit: Either indicator signals trend reversal
            if self.uptrend[0] or self.psar_bullish[0]:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "PSAR turned bullish"
                    if self.psar_bullish[0]
                    else "Price crossed above Supertrend"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - Trend Reversal) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Supertrend: {self.supertrend[0]:.2f} | "
                    f"PSAR: {self.psar[0]:.2f} | "
                    f"Price vs Supertrend: {price_st_pct:.2f}%"
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
            "st_period": trial.suggest_int("st_period", 7, 15),
            "st_multiplier": trial.suggest_float("st_multiplier", 2.0, 4.0, step=0.5),
            "psar_af": trial.suggest_float("psar_af", 0.01, 0.05, step=0.01),
            "psar_afmax": trial.suggest_float("psar_afmax", 0.15, 0.25, step=0.05),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            st_period = params.get("st_period", 10)
            return max(st_period + 5, 20)  # Supertrend ATR + PSAR warmup
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
