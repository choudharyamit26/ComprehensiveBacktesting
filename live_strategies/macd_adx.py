import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class MACDADX(bt.Strategy):
    """
    MACD + ADX Trend Filter Strategy

    Combines MACD crossover signals with ADX trend strength to identify trades in strong trends.
    Enters when MACD signals align with a rising ADX above a threshold, exits on ADX decline or
    MACD reversal.

    Strategy Type: MOMENTUM + TREND STRENGTH
    =======================================
    Uses MACD for momentum and ADX to confirm trend strength.

    Strategy Logic:
    ==============
    Long Position Rules:
    - Entry: MACD bullish crossover AND ADX > threshold AND ADX rising
    - Exit: MACD bearish crossover OR ADX declining

    Short Position Rules:
    - Entry: MACD bearish crossover AND ADX > threshold AND ADX rising
    - Exit: MACD bullish crossover OR ADX declining

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
    - ADX:
      * Period: 14 (default)
      * Threshold: >20 indicates strong trend
      * Rising: ADX[0] > ADX[-1]
      * Declining: ADX[0] < ADX[-1]

    Parameters:
    ==========
    - fast_period (int): MACD fast EMA period (default: 12)
    - slow_period (int): MACD slow EMA period (default: 26)
    - signal_period (int): MACD signal line period (default: 9)
    - adx_period (int): ADX period (default: 14)
    - adx_threshold (float): ADX strength threshold (default: 20)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("fast_period", 12),
        ("slow_period", 26),
        ("signal_period", 9),
        ("adx_period", 14),
        ("adx_threshold", 20),
        ("verbose", False),
    )

    optimization_params = {
        "fast_period": {"type": "int", "low": 8, "high": 16, "step": 1},
        "slow_period": {"type": "int", "low": 20, "high": 35, "step": 1},
        "signal_period": {"type": "int", "low": 6, "high": 12, "step": 1},
        "adx_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "adx_threshold": {"type": "float", "low": 15, "high": 30},
    }

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

        # ADX indicator
        self.adx = btind.ADX(self.data, period=self.params.adx_period)
        self.adx_strong = self.adx > self.params.adx_threshold
        self.adx_rising = self.adx > self.adx(-1)
        self.adx_declining = self.adx < self.adx(-1)

        # Combined entry signals
        self.bullish_entry = bt.And(
            self.macd_cross > 0, self.adx_strong, self.adx_rising
        )
        self.bearish_entry = bt.And(
            self.macd_cross < 0, self.adx_strong, self.adx_rising
        )

        # Exit signals
        self.bullish_exit = bt.Or(self.macd_cross < 0, self.adx_declining)
        self.bearish_exit = bt.Or(self.macd_cross > 0, self.adx_declining)

        # Initialize variables
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.slow_period + self.params.signal_period,
                self.params.adx_period,
            )
            + 5
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized MACDADX with params: {self.params}")

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
            or np.isnan(self.adx[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"MACD={self.macd.macd[0]}, Signal={self.macd.signal[0]}, ADX={self.adx[0]}"
            )
            return

        # Calculate momentum and trend state
        momentum_state = "NEUTRAL"
        if self.macd.macd[0] > self.macd.signal[0]:
            momentum_state = "BULLISH"
        elif self.macd.macd[0] < self.macd.signal[0]:
            momentum_state = "BEARISH"

        trend_state = "WEAK"
        if self.adx_strong[0] and self.adx_rising[0]:
            trend_state = "STRONG_RISING"
        elif self.adx_declining[0]:
            trend_state = "DECLINING"

        # Store indicator data
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "macd": self.macd.macd[0],
                "signal": self.macd.signal[0],
                "histogram": self.macd_histogram[0],
                "adx": self.adx[0],
                "bullish_entry": self.bullish_entry[0],
                "bearish_entry": self.bearish_entry[0],
                "momentum_state": momentum_state,
                "trend_state": trend_state,
            }
        )

        # Position Management
        if not self.position:
            if self.bullish_entry[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - MACD Bullish + ADX Strong) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} > Signal: {self.macd.signal[0]:.4f} | "
                    f"ADX: {self.adx[0]:.2f} (Rising)"
                )
            elif self.bearish_entry[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - MACD Bearish + ADX Strong) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} < Signal: {self.macd.signal[0]:.4f} | "
                    f"ADX: {self.adx[0]:.2f} (Rising)"
                )
        else:
            if self.position.size > 0 and self.bullish_exit[0]:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "MACD Bearish Cross" if self.macd_cross[0] < 0 else "ADX Declining"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - {exit_reason}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} | Signal: {self.macd.signal[0]:.4f} | "
                    f"ADX: {self.adx[0]:.2f}"
                )
            elif self.position.size < 0 and self.bearish_exit[0]:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "MACD Bullish Cross" if self.macd_cross[0] > 0 else "ADX Declining"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - {exit_reason}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.4f} | Signal: {self.macd.signal[0]:.4f} | "
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
            "adx_period": trial.suggest_int("adx_period", 10, 20),
            "adx_threshold": trial.suggest_float("adx_threshold", 15, 30),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            slow_period = params.get("slow_period", 26)
            signal_period = params.get("signal_period", 9)
            adx_period = params.get("adx_period", 14)
            max_period = max(slow_period + signal_period, adx_period)
            return max_period + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 65
