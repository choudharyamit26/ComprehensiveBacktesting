import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class MACDBollinger(bt.Strategy):
    """
    MACD + Bollinger Bands Strategy

    Combines MACD crossover signals with Bollinger Bands squeeze/expansion to identify trades.
    Enters when MACD signals align with band expansion (breakout) or squeeze (volatility contraction).
    Exits on band contraction or MACD reversal.

    Strategy Type: MOMENTUM + VOLATILITY
    ===================================
    Uses MACD for momentum and Bollinger Bands for volatility signals.

    Strategy Logic:
    ==============
    Long Position Rules:
    - Entry: MACD bullish crossover AND (price > upper band OR BB bandwidth < threshold)
    - Exit: MACD bearish crossover OR BB bandwidth > threshold (contraction)

    Short Position Rules:
    - Entry: MACD bearish crossover AND (price < lower band OR BB bandwidth < threshold)
    - Exit: MACD bullish crossover OR BB bandwidth > threshold (contraction)

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST
    - Uses warmup period for indicator stability
    - Prevents order overlap

    Indicators Used:
    ===============
    - MACD:
      * MACD Line: 12-period EMA - 26-period EMA
      * Signal Line: 9-period EMA of MACD line
      * Histogram: MACD line - Signal line
    - Bollinger Bands:
      * Upper/Lower bands: 20-period SMA ± 2 standard deviations
      * Bandwidth: (Upper - Lower) / Middle band
      * Squeeze: Bandwidth < threshold (low volatility)
      * Expansion: Price breaks upper/lower band

    Parameters:
    ==========
    - fast_period (int): MACD fast EMA period (default: 12)
    - slow_period (int): MACD slow EMA period (default: 26)
    - signal_period (int): MACD signal line period (default: 9)
    - bb_period (int): Bollinger Bands period (default: 20)
    - bb_dev (float): Bollinger Bands standard deviation (default: 2.0)
    - bb_bandwidth (float): Bandwidth threshold for squeeze (default: 0.1)
    - verbose (bool): Enable detailed logging (default: False)

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MACDBollinger, bb_period=30, bb_dev=2.5)
    cerebro.run()
    """

    params = (
        ("fast_period", 12),
        ("slow_period", 26),
        ("signal_period", 9),
        ("bb_period", 20),
        ("bb_dev", 2.0),
        ("bb_bandwidth", 0.1),
        ("verbose", False),
    )

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize MACD indicator
        self.macd = btind.MACD(
            self.data.close,
            period_me1=self.params.fast_period,
            period_me2=self.params.slow_period,
            period_signal=self.params.signal_period,
        )

        # Calculate MACD histogram manually
        self.macd_histogram = self.macd.lines.macd - self.macd.lines.signal

        # MACD crossover signals
        self.macd_cross = btind.CrossOver(self.macd.lines.macd, self.macd.lines.signal)

        # Bollinger Bands
        self.bb = btind.BollingerBands(
            self.data.close, period=self.params.bb_period, devfactor=self.params.bb_dev
        )
        self.bb_bandwidth = (self.bb.top - self.bb.bot) / self.bb.mid
        self.bb_squeeze = self.bb_bandwidth < self.params.bb_bandwidth
        self.bb_expansion_up = self.data.close > self.bb.top
        self.bb_expansion_down = self.data.close < self.bb.bot
        self.bb_contraction = self.bb_bandwidth > self.params.bb_bandwidth

        # Combined entry signals
        self.bullish_entry = bt.And(
            self.macd_cross > 0, bt.Or(self.bb_expansion_up, self.bb_squeeze)
        )
        self.bearish_entry = bt.And(
            self.macd_cross < 0, bt.Or(self.bb_expansion_down, self.bb_squeeze)
        )

        # Exit signals
        self.bullish_exit = bt.Or(self.macd_cross < 0, self.bb_contraction)
        self.bearish_exit = bt.Or(self.macd_cross > 0, self.bb_contraction)

        # Initialize variables
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.slow_period + self.params.signal_period,
                self.params.bb_period,
            )
            + 5
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized MACDBollinger with params: {self.params}")

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
            np.isnan(self.macd.macd[0])
            or np.isnan(self.macd.signal[0])
            or np.isnan(self.bb.mid[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"MACD={self.macd.macd[0]}, Signal={self.macd.signal[0]}, BB Mid={self.bb.mid[0]}"
            )
            return

        # Calculate momentum and volatility state
        momentum_state = "NEUTRAL"
        if self.macd.macd[0] > self.macd.signal[0]:
            momentum_state = "BULLISH"
        elif self.macd.macd[0] < self.macd.signal[0]:
            momentum_state = "BEARISH"

        bb_state = "NORMAL"
        if self.bb_squeeze[0]:
            bb_state = "SQUEEZE"
        elif self.bb_expansion_up[0] or self.bb_expansion_down[0]:
            bb_state = "EXPANSION"
        elif self.bb_contraction[0]:
            bb_state = "CONTRACTION"

        # Store indicator data
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "macd": self.macd.macd[0],
                "signal": self.macd.signal[0],
                "histogram": self.macd_histogram[0],
                "bb_top": self.bb.top[0],
                "bb_mid": self.bb.mid[0],
                "bb_bot": self.bb.bot[0],
                "bb_bandwidth": self.bb_bandwidth[0],
                "bullish_entry": self.bullish_entry[0],
                "bearish_entry": self.bearish_entry[0],
                "momentum_state": momentum_state,
                "bb_state": bb_state,
            }
        )

        # Position Management
        if not self.position:
            if self.bullish_entry[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                condition = (
                    "BB Expansion Up" if self.bb_expansion_up[0] else "BB Squeeze"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - MACD Bullish + {condition}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} > Signal: {self.macd.signal[0]:.4f} | "
                    f"BB Bandwidth: {self.bb_bandwidth[0]:.4f}"
                )
            elif self.bearish_entry[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                condition = (
                    "BB Expansion Down" if self.bb_expansion_down[0] else "BB Squeeze"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - MACD Bearish + {condition}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} < Signal: {self.macd.signal[0]:.4f} | "
                    f"BB Bandwidth: {self.bb_bandwidth[0]:.4f}"
                )
        else:
            if self.position.size > 0 and self.bullish_exit[0]:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "MACD Bearish Cross" if self.macd_cross[0] < 0 else "BB Contraction"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - {exit_reason}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} | Signal: {self.macd.signal[0]:.4f} | "
                    f"BB Bandwidth: {self.bb_bandwidth[0]:.4f}"
                )
            elif self.position.size < 0 and self.bearish_exit[0]:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "MACD Bullish Cross" if self.macd_cross[0] > 0 else "BB Contraction"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - {exit_reason}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} | Signal: {self.macd.signal[0]:.4f} | "
                    f"BB Bandwidth: {self.bb_bandwidth[0]:.4f}"
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
                        f"Price: {order.executed.price:.2f} | Size: {order.executed.size} | "
                        f"Cost: {order.executed.value:.2f} | Comm: {order.executed.comm:.2f} | PnL: {pnl:.2f}"
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
                        f"Price: {order.executed.price:.2f} | Size: {order.executed.size} | "
                        f"Cost: {order.executed.value:.2f} | Comm: {order.executed.comm:.2f} | PnL: {pnl:.2f}"
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
            "fast_period": trial.suggest_int("fast_period", 8, 16),
            "slow_period": trial.suggest_int("slow_period", 20, 35),
            "signal_period": trial.suggest_int("signal_period", 6, 12),
            "bb_period": trial.suggest_int("bb_period", 10, 50),
            "bb_dev": trial.suggest_float("bb_dev", 1.5, 3.0),
            "bb_bandwidth": trial.suggest_float("bb_bandwidth", 0.05, 0.2),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            slow_period = params.get("slow_period", 26)
            signal_period = params.get("signal_period", 9)
            bb_period = params.get("bb_period", 20)
            max_period = max(slow_period + signal_period, bb_period)
            return max_period + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 65
