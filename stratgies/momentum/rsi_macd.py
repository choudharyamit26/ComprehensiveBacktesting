import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class RSIMACD(bt.Strategy):
    """
    RSI and MACD Combined Intraday Trading Strategy

    This strategy combines RSI (Relative Strength Index) and MACD (Moving Average Convergence Divergence)
    indicators to identify momentum-based entry and exit signals for both long and short positions.

    Strategy Logic:
    ==============

    Long Position Rules:
    - Entry: RSI > threshold (default 50) AND MACD bullish crossover (MACD line crosses above signal line)
    - Exit: RSI < threshold OR MACD bearish crossover (MACD line crosses below signal line)

    Short Position Rules:
    - Entry: RSI < threshold (default 50) AND MACD bearish crossover (MACD line crosses below signal line)
    - Exit: RSI > threshold OR MACD bullish crossover (MACD line crosses above signal line)

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses warmup period to ensure indicator stability before trading
    - Prevents order overlap with pending order checks

    Indicators Used:
    ===============
    - RSI: Measures momentum strength (overbought/oversold conditions)
    - MACD: Trend-following momentum indicator using exponential moving averages
    - MACD Crossover: Detects bullish/bearish momentum shifts

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
    - rsi_period (int): RSI calculation period (default: 14)
    - macd_fast (int): MACD fast EMA period (default: 12)
    - macd_slow (int): MACD slow EMA period (default: 26)
    - macd_signal (int): MACD signal line EMA period (default: 9)
    - rsi_threshold (int): RSI threshold for entry/exit signals (default: 50)
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
    cerebro.addstrategy(RSIMACD, rsi_period=14, rsi_threshold=50)
    cerebro.run()

    Note:
    ====
    This is a momentum strategy that works best in trending markets. Consider using
    additional filters or risk management rules in choppy/sideways markets.
    """

    params = (
        ("rsi_period", 14),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("rsi_threshold", 50),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 8, "high": 20, "step": 1},
        "macd_fast": {"type": "int", "low": 8, "high": 15, "step": 1},
        "macd_slow": {"type": "int", "low": 20, "high": 35, "step": 1},
        "macd_signal": {"type": "int", "low": 7, "high": 12, "step": 1},
        "rsi_threshold": {"type": "int", "low": 45, "high": 55, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.macd = btind.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal,
        )
        self.macd_line = self.macd.macd
        self.macd_signal = self.macd.signal
        self.macd_crossover = btind.CrossOver(self.macd_line, self.macd_signal)

        self.order = None
        self.order_type = None  # Track order type for shorting logic
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.rsi_period,
                self.params.macd_slow,
                self.params.macd_signal,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized RSIMACD with params: {self.params}")
        logger.info(
            f"RSIMACD initialized with rsi_period={self.p.rsi_period}, "
            f"macd_fast={self.p.macd_fast}, macd_slow={self.p.macd_slow}, "
            f"macd_signal={self.p.macd_signal}, rsi_threshold={self.p.rsi_threshold}"
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
            or np.isnan(self.macd_line[0])
            or np.isnan(self.macd_signal[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"RSI={self.rsi[0]}, MACD={self.macd_line[0]}, "
                f"MACD_Signal={self.macd_signal[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "rsi": self.rsi[0],
                "macd": self.macd_line[0],
                "macd_signal": self.macd_signal[0],
                "macd_crossover": self.macd_crossover[0],
            }
        )

        # Position management with shorting
        if not self.position:
            # Long Entry: RSI > threshold AND MACD bullish crossover
            if self.rsi[0] > self.params.rsi_threshold and self.macd_crossover[0] > 0:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} > {self.params.rsi_threshold} | "
                    f"MACD: {self.macd_line[0]:.4f} > Signal: {self.macd_signal[0]:.4f} (Bullish Cross)"
                )
            # Short Entry: RSI < threshold AND MACD bearish crossover
            elif self.rsi[0] < self.params.rsi_threshold and self.macd_crossover[0] < 0:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} < {self.params.rsi_threshold} | "
                    f"MACD: {self.macd_line[0]:.4f} < Signal: {self.macd_signal[0]:.4f} (Bearish Cross)"
                )
        elif self.position.size > 0:  # Long position
            # Long Exit: RSI < threshold OR MACD bearish crossover
            if self.rsi[0] < self.params.rsi_threshold or self.macd_crossover[0] < 0:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "RSI < threshold"
                    if self.rsi[0] < self.params.rsi_threshold
                    else "MACD bearish cross"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"MACD: {self.macd_line[0]:.4f}, Signal: {self.macd_signal[0]:.4f}"
                )
        elif self.position.size < 0:  # Short position
            # Short Exit: RSI > threshold OR MACD bullish crossover
            if self.rsi[0] > self.params.rsi_threshold or self.macd_crossover[0] > 0:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "RSI > threshold"
                    if self.rsi[0] > self.params.rsi_threshold
                    else "MACD bullish cross"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"MACD: {self.macd_line[0]:.4f}, Signal: {self.macd_signal[0]:.4f}"
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
            "rsi_period": trial.suggest_int("rsi_period", 8, 20),
            "macd_fast": trial.suggest_int("macd_fast", 8, 15),
            "macd_slow": trial.suggest_int("macd_slow", 20, 35),
            "macd_signal": trial.suggest_int("macd_signal", 7, 12),
            "rsi_threshold": trial.suggest_int("rsi_threshold", 45, 55),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            macd_slow = params.get("macd_slow", 26)
            macd_signal = params.get("macd_signal", 9)
            max_period = max(rsi_period, macd_slow, macd_signal)
            return max_period + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
