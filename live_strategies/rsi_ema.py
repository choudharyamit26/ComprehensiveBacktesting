import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class RSIEMA(bt.Strategy):
    """
    RSI + EMA Trend Following Strategy

    This strategy combines RSI (Relative Strength Index) with EMA (Exponential Moving Average)
    to identify trend-following opportunities. It uses RSI to confirm momentum direction
    and EMA to identify the underlying trend direction.

    Strategy Type: TREND FOLLOWING
    ==============================
    This is a trend-following strategy that assumes strong trends will continue.
    It buys when the trend is up and RSI confirms bullish momentum, and sells
    when the trend is down and RSI confirms bearish momentum.

    Strategy Logic:
    ==============

    Long Position Rules:
    - Entry: RSI > bullish threshold (default 55) AND price > EMA (uptrend)
    - Exit: RSI < bearish threshold (default 45) OR price < EMA (trend reversal)

    Short Position Rules:
    - Entry: RSI < bearish threshold (default 45) AND price < EMA (downtrend)
    - Exit: RSI > bullish threshold (default 55) OR price > EMA (trend reversal)

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses warmup period to ensure indicator stability before trading
    - Prevents order overlap with pending order checks
    - Trend following works best in trending markets

    Indicators Used:
    ===============
    - RSI: Measures momentum strength to confirm trend direction
      * > 55: Bullish momentum (buy signal when price > EMA)
      * < 45: Bearish momentum (sell signal when price < EMA)
      * 45-55: Neutral zone (no action)
    - EMA: Exponential Moving Average to identify trend direction
      * Price > EMA: Uptrend (allow long positions)
      * Price < EMA: Downtrend (allow short positions)

    Trend Following Concept:
    =======================
    - When price > EMA + RSI bullish = strong uptrend continuation
    - When price < EMA + RSI bearish = strong downtrend continuation
    - Exit when RSI momentum weakens or price crosses EMA (trend change)
    - EMA reacts faster than SMA for better trend identification

    Features:
    =========
    - Comprehensive trade logging with IST timezone
    - Detailed PnL tracking for each completed trade
    - Position sizing and commission handling
    - Optimization-ready parameter space
    - Robust error handling and data validation
    - Support for both backtesting and live trading
    - Trend strength analysis

    Parameters:
    ==========
    - rsi_period (int): RSI calculation period (default: 14)
    - ema_period (int): EMA calculation period (default: 21)
    - rsi_bullish (int): RSI bullish threshold for long entries (default: 55)
    - rsi_bearish (int): RSI bearish threshold for short entries (default: 45)
    - verbose (bool): Enable detailed logging (default: False)

    Performance Metrics:
    ===================
    - Tracks win/loss ratio
    - Calculates net PnL including commissions
    - Records trade duration and timing
    - Provides detailed execution logs
    - Monitors trend strength

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(RSIEMA, rsi_bullish=60, rsi_bearish=40, ema_period=21)
    cerebro.run()

    Best Market Conditions:
    ======================
    - Strong trending markets (both up and down)
    - Markets with clear momentum shifts
    - Avoid during ranging/sideways markets (mean reversion strategies better)
    - Works well in various timeframes with sufficient volatility

    Note:
    ====
    This is a trend-following strategy that profits from momentum continuation.
    It's opposite to mean reversion strategies and requires trending market conditions.
    Consider using volatility filters to avoid whipsaws in ranging markets.
    """

    params = (
        ("rsi_period", 14),
        ("ema_period", 21),
        ("rsi_bullish", 55),
        ("rsi_bearish", 45),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "ema_period": {"type": "int", "low": 15, "high": 30, "step": 1},
        "rsi_bullish": {"type": "int", "low": 50, "high": 65, "step": 1},
        "rsi_bearish": {"type": "int", "low": 35, "high": 50, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.ema = btind.EMA(self.data.close, period=self.params.ema_period)

        # Trend direction indicators
        self.uptrend = self.data.close > self.ema
        self.downtrend = self.data.close < self.ema

        # RSI momentum confirmations
        self.rsi_bullish_signal = self.rsi > self.params.rsi_bullish
        self.rsi_bearish_signal = self.rsi < self.params.rsi_bearish

        self.order = None
        self.order_type = None  # Track order type for shorting logic
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.rsi_period,
                self.params.ema_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized RSIEMA with params: {self.params}")
        logger.info(
            f"RSIEMA initialized with rsi_period={self.p.rsi_period}, "
            f"ema_period={self.p.ema_period}, "
            f"rsi_bullish={self.p.rsi_bullish}, rsi_bearish={self.p.rsi_bearish}"
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
        if np.isnan(self.rsi[0]) or np.isnan(self.ema[0]):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"RSI={self.rsi[0]}, EMA={self.ema[0]}"
            )
            return

        # Calculate trend strength
        price_ema_diff = self.data.close[0] - self.ema[0]
        price_ema_pct = (price_ema_diff / self.ema[0]) * 100

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "rsi": self.rsi[0],
                "ema": self.ema[0],
                "price_ema_diff": price_ema_diff,
                "price_ema_pct": price_ema_pct,
                "uptrend": self.uptrend[0],
                "downtrend": self.downtrend[0],
                "rsi_bullish": self.rsi_bullish_signal[0],
                "rsi_bearish": self.rsi_bearish_signal[0],
            }
        )

        # Trend Following Position Management
        if not self.position:
            # Long Entry: RSI confirms bullish momentum AND price above EMA (uptrend)
            if self.rsi[0] > self.params.rsi_bullish and self.uptrend[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Trend Following) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} > {self.params.rsi_bullish} (Bullish) | "
                    f"EMA: {self.ema[0]:.2f} | "
                    f"Price vs EMA: {price_ema_pct:.2f}% (Uptrend) | "
                    f"Trend: BULLISH"
                )
            # Short Entry: RSI confirms bearish momentum AND price below EMA (downtrend)
            elif self.rsi[0] < self.params.rsi_bearish and self.downtrend[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Trend Following) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} < {self.params.rsi_bearish} (Bearish) | "
                    f"EMA: {self.ema[0]:.2f} | "
                    f"Price vs EMA: {price_ema_pct:.2f}% (Downtrend) | "
                    f"Trend: BEARISH"
                )
        elif self.position.size > 0:  # Long position
            # Long Exit: RSI momentum weakens OR price crosses below EMA (trend reversal)
            if self.rsi[0] < self.params.rsi_bearish or self.downtrend[0]:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "RSI momentum weakened"
                    if self.rsi[0] < self.params.rsi_bearish
                    else "Price crossed below EMA"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - Trend Reversal) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"EMA: {self.ema[0]:.2f} | "
                    f"Price vs EMA: {price_ema_pct:.2f}%"
                )
        elif self.position.size < 0:  # Short position
            # Short Exit: RSI momentum strengthens OR price crosses above EMA (trend reversal)
            if self.rsi[0] > self.params.rsi_bullish or self.uptrend[0]:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "RSI momentum strengthened"
                    if self.rsi[0] > self.params.rsi_bullish
                    else "Price crossed above EMA"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - Trend Reversal) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"EMA: {self.ema[0]:.2f} | "
                    f"Price vs EMA: {price_ema_pct:.2f}%"
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
            "ema_period": trial.suggest_int("ema_period", 15, 30),
            "rsi_bullish": trial.suggest_int("rsi_bullish", 50, 65),
            "rsi_bearish": trial.suggest_int("rsi_bearish", 35, 50),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            ema_period = params.get("ema_period", 21)
            max_period = max(rsi_period, ema_period)
            return max_period + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
