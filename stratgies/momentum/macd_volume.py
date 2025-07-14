import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class MACDVolume(bt.Strategy):
    """
    MACD + Volume Confirmation Strategy

    This strategy combines MACD crossover signals with volume spikes to identify high-probability
    trades. It enters trades when MACD signals align with a significant volume increase, indicating
    strong market participation, and exits when volume dries up or MACD reverses.

    Strategy Type: MOMENTUM + VOLUME CONFIRMATION
    ===========================================
    Uses MACD crossovers for momentum and volume spikes (>150% of average) for confirmation.
    Exits when volume falls below average or MACD signals reverse.

    Strategy Logic:
    ==============
    Long Position Rules:
    - Entry: MACD line crosses above signal line AND volume > 1.5 * average volume
    - Exit: MACD line crosses below signal line OR volume < average volume

    Short Position Rules:
    - Entry: MACD line crosses below signal line AND volume > 1.5 * average volume
    - Exit: MACD line crosses above signal line OR volume < average volume

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses warmup period for indicator stability
    - Prevents order overlap with pending order checks
    - Volume confirmation reduces false signals

    Indicators Used:
    ===============
    - MACD:
      * MACD Line: 12-period EMA - 26-period EMA
      * Signal Line: 9-period EMA of MACD line
      * Histogram: MACD line - Signal line
    - Volume:
      * Average volume over lookback period (default: 20)
      * Volume spike: >150% of average volume
      * Volume dry-up: <100% of average volume

    Parameters:
    ==========
    - fast_period (int): MACD fast EMA period (default: 12)
    - slow_period (int): MACD slow EMA period (default: 26)
    - signal_period (int): MACD signal line period (default: 9)
    - volume_period (int): Volume average lookback period (default: 20)
    - volume_spike (float): Volume multiplier for entry (default: 1.5)
    - volume_dry (float): Volume multiplier for exit (default: 1.0)
    - verbose (bool): Enable detailed logging (default: False)

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MACDVolume, volume_period=30, fast_period=8)
    cerebro.run()
    """

    params = (
        ("fast_period", 12),
        ("slow_period", 26),
        ("signal_period", 9),
        ("volume_period", 20),
        ("volume_spike", 1.5),
        ("volume_dry", 1.0),
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

        # Volume indicator
        self.avg_volume = btind.SMA(self.data.volume, period=self.params.volume_period)
        self.volume_spike = (
            self.data.volume > self.params.volume_spike * self.avg_volume
        )
        self.volume_dry = self.data.volume < self.params.volume_dry * self.avg_volume

        # Combined entry signals
        self.bullish_entry = bt.And(
            self.macd_cross > 0, self.volume_spike  # Bullish crossover
        )
        self.bearish_entry = bt.And(
            self.macd_cross < 0, self.volume_spike  # Bearish crossover
        )

        # Exit signals
        self.bullish_exit = bt.Or(self.macd_cross < 0, self.volume_dry)
        self.bearish_exit = bt.Or(self.macd_cross > 0, self.volume_dry)

        # Initialize variables
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.slow_period + self.params.signal_period,
                self.params.volume_period,
            )
            + 5
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized MACDVolume with params: {self.params}")

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
            or np.isnan(self.avg_volume[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"MACD={self.macd.macd[0]}, Signal={self.macd.signal[0]}, Avg Volume={self.avg_volume[0]}"
            )
            return

        # Calculate momentum and volume state
        momentum_state = "NEUTRAL"
        if self.macd.macd[0] > self.macd.signal[0]:
            momentum_state = "BULLISH"
        elif self.macd.macd[0] < self.macd.signal[0]:
            momentum_state = "BEARISH"

        volume_state = "NORMAL"
        if self.volume_spike[0]:
            volume_state = "SPIKE"
        elif self.volume_dry[0]:
            volume_state = "DRY"

        # Store indicator data
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "macd": self.macd.macd[0],
                "signal": self.macd.signal[0],
                "histogram": self.macd_histogram[0],
                "avg_volume": self.avg_volume[0],
                "volume": self.data.volume[0],
                "bullish_entry": self.bullish_entry[0],
                "bearish_entry": self.bearish_entry[0],
                "momentum_state": momentum_state,
                "volume_state": volume_state,
            }
        )

        # Position Management
        if not self.position:
            if self.bullish_entry[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - MACD Bullish + Volume Spike) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} > Signal: {self.macd.signal[0]:.4f} | "
                    f"Volume: {self.data.volume[0]:.2f} > {self.params.volume_spike} * Avg: {self.avg_volume[0]:.2f}"
                )
            elif self.bearish_entry[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - MACD Bearish + Volume Spike) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} < Signal: {self.macd.signal[0]:.4f} | "
                    f"Volume: {self.data.volume[0]:.2f} > {self.params.volume_spike} * Avg: {self.avg_volume[0]:.2f}"
                )
        else:
            if self.position.size > 0 and self.bullish_exit[0]:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "MACD Bearish Cross" if self.macd_cross[0] < 0 else "Volume Dry"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - {exit_reason}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} | Signal: {self.macd.signal[0]:.4f} | "
                    f"Volume: {self.data.volume[0]:.2f}"
                )
            elif self.position.size < 0 and self.bearish_exit[0]:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "MACD Bullish Cross" if self.macd_cross[0] > 0 else "Volume Dry"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - {exit_reason}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} | Signal: {self.macd.signal[0]:.4f} | "
                    f"Volume: {self.data.volume[0]:.2f}"
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
            "volume_period": trial.suggest_int("volume_period", 10, 50),
            "volume_spike": trial.suggest_float("volume_spike", 1.2, 2.0),
            "volume_dry": trial.suggest_float("volume_dry", 0.5, 1.0),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            slow_period = params.get("slow_period", 26)
            signal_period = params.get("signal_period", 9)
            volume_period = params.get("volume_period", 20)
            max_period = max(slow_period + signal_period, volume_period)
            return max_period + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 65
