import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class SuperTrend(bt.Indicator):
    """
    Custom Supertrend Indicator
    """

    lines = ("supertrend",)
    params = (("period", 10), ("multiplier", 3.0))

    def __init__(self):
        self.atr = btind.AverageTrueRange(self.data, period=self.p.period)
        self.hl2 = (self.data.high + self.data.low) / 2
        self.basic_ub = self.hl2 + self.p.multiplier * self.atr
        self.basic_lb = self.hl2 - self.p.multiplier * self.atr

        # Initialize state variables
        self.final_ub = 0.0
        self.final_lb = 0.0
        self.prev_close = float("nan")

    def next(self):
        # First bar initialization
        if len(self) == 1:
            self.final_ub = self.basic_ub[0]
            self.final_lb = self.basic_lb[0]
            self.prev_close = self.data.close[0]
            if self.data.close[0] > self.final_ub:
                self.lines.supertrend[0] = self.final_lb
            else:
                self.lines.supertrend[0] = self.final_ub
            return

        # Update final bands based on conditions
        if (self.basic_ub[0] < self.final_ub) or (self.prev_close > self.final_ub):
            self.final_ub = self.basic_ub[0]

        if (self.basic_lb[0] > self.final_lb) or (self.prev_close < self.final_lb):
            self.final_lb = self.basic_lb[0]

        # Determine supertrend value
        if self.data.close[0] <= self.final_ub:
            self.lines.supertrend[0] = self.final_ub
        else:
            self.lines.supertrend[0] = self.final_lb

        # Store previous close for next bar
        self.prev_close = self.data.close[0]


class BBSupertrendStrategy(bt.Strategy):
    """
    Bollinger Bands + Supertrend Strategy

    This strategy combines Bollinger Bands (BB) breakouts with Supertrend indicator confirmation
    to capture trending moves with reduced false signals.

    Strategy Type: MOMENTUM
    =============================
    This strategy uses BB breakouts to identify trend starts and Supertrend to confirm trend direction.
    It exits on Supertrend reversal or BB mean reversion.

    Strategy Logic:
    ==============
    Long Position Rules:
    - Entry: Price breaks above upper BB AND Supertrend is bullish
    - Exit: Supertrend turns bearish OR price returns to BB middle

    Short Position Rules:
    - Entry: Price breaks below lower BB AND Supertrend is bearish
    - Exit: Supertrend turns bullish OR price returns to BB middle

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST
    - Uses warmup period for indicator stability
    - Prevents order overlap

    Indicators Used:
    ===============
    - Bollinger Bands: Middle Band (20-period SMA), Upper/Lower Bands (±2 SD)
    - Supertrend: Trend indicator based on ATR and price levels

    Parameters:
    ==========
    - bb_period (int): Bollinger Bands period (default: 20)
    - bb_dev (float): BB standard deviation multiplier (default: 2.0)
    - supertrend_period (int): Supertrend period (default: 10)
    - supertrend_mult (float): Supertrend ATR multiplier (default: 3.0)
    - verbose (bool): Enable detailed logging (default: False)

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(BBSupertrendStrategy, bb_period=20, supertrend_period=10)
    cerebro.run()

    Best Market Conditions:
    ======================
    - Strong trending markets
    - Avoid range-bound or choppy markets
    """

    params = (
        ("bb_period", 20),
        ("bb_dev", 2.0),
        ("supertrend_period", 10),
        ("supertrend_mult", 3.0),
        ("verbose", False),
    )

    optimization_params = {
        "bb_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "bb_dev": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.1},
        "supertrend_period": {"type": "int", "low": 7, "high": 14, "step": 1},
        "supertrend_mult": {"type": "float", "low": 2.0, "high": 4.0, "step": 0.5},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.bb = btind.BollingerBands(
            self.data.close, period=self.params.bb_period, devfactor=self.params.bb_dev
        )
        self.supertrend = SuperTrend(
            self.data,
            period=self.params.supertrend_period,
            multiplier=self.params.supertrend_mult,
        )

        self.bullish_entry = bt.And(
            self.data.close > self.bb.top, self.data.close > self.supertrend
        )
        self.bearish_entry = bt.And(
            self.data.close < self.bb.bot, self.data.close < self.supertrend
        )
        self.bullish_exit = bt.Or(
            self.data.close < self.supertrend, self.data.close <= self.bb.mid
        )
        self.bearish_exit = bt.Or(
            self.data.close > self.supertrend, self.data.close >= self.bb.mid
        )

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(self.params.bb_period, self.params.supertrend_period) + 5
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized BBSupertrendStrategy with params: {self.params}")

    def next(self):
        if len(self) < self.warmup_period:
            logger.debug(f"Skipping bar {len(self)}: still in warmup period")
            return

        if not self.ready:
            self.ready = True
            logger.info(f"Strategy ready at bar {len(self)}")

        bar_time = self.datas[0].datetime.datetime(0)
        bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
        current_time = bar_time_ist.time()

        if current_time >= datetime.time(15, 15):
            if self.position:
                self.close()
                trade_logger.info("Force closed all positions at 15:15 IST")
            return

        if not (datetime.time(9, 15) <= current_time <= datetime.time(15, 5)):
            return

        if self.order:
            logger.debug(f"Order pending at bar {len(self)}")
            return

        if np.isnan(self.bb.mid[0]) or np.isnan(self.supertrend[0]):
            logger.debug(f"Invalid indicator values at bar {len(self)}")
            return

        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "bb_top": self.bb.top[0],
                "supertrend": self.supertrend[0],
            }
        )

        if not self.position:
            if self.bullish_entry[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (BB Breakout + Supertrend Bullish) | Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f}"
                )
            elif self.bearish_entry[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (BB Breakout + Supertrend Bearish) | Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f}"
                )
        else:
            if self.position.size > 0 and self.bullish_exit[0]:
                self.order = self.sell()
                self.order_type = "exit_long"
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long) | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )
            elif self.position.size < 0 and self.bearish_exit[0]:
                self.order = self.buy()
                self.order_type = "exit_short"
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short) | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt).astimezone(
                pytz.timezone("Asia/Kolkata")
            )
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
                    f"BUY EXECUTED (Enter Long) | Ref: {order.ref} | Price: {order.executed.price:.2f}"
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
                    f"SELL EXECUTED (Enter Short) | Ref: {order.ref} | Price: {order.executed.price:.2f}"
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
                    trade_info = {
                        "ref": order.ref,
                        "entry_time": entry_info["entry_time"],
                        "exit_time": exec_dt,
                        "entry_price": entry_info["entry_price"],
                        "exit_price": order.executed.price,
                        "size": abs(entry_info["size"]),
                        "pnl": pnl,
                        "pnl_net": pnl - total_commission,
                        "commission": total_commission,
                        "status": "Won" if pnl > 0 else "Lost",
                        "direction": "Long",
                    }
                    self.completed_trades.append(trade_info)
                    self.trade_count += 1
                    trade_logger.info(
                        f"SELL EXECUTED (Exit Long) | Ref: {order.ref} | PnL: {pnl:.2f}"
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
                    trade_info = {
                        "ref": order.ref,
                        "entry_time": entry_info["entry_time"],
                        "exit_time": exec_dt,
                        "entry_price": entry_info["entry_price"],
                        "exit_price": order.executed.price,
                        "size": abs(entry_info["size"]),
                        "pnl": pnl,
                        "pnl_net": pnl - total_commission,
                        "commission": total_commission,
                        "status": "Won" if pnl > 0 else "Lost",
                        "direction": "Short",
                    }
                    self.completed_trades.append(trade_info)
                    self.trade_count += 1
                    trade_logger.info(
                        f"BUY EXECUTED (Exit Short) | Ref: {order.ref} | PnL: {pnl:.2f}"
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
                f"TRADE CLOSED | Ref: {trade.ref} | Profit: {trade.pnl:.2f} | "
                f"Net Profit: {trade.pnlcomm:.2f} | Bars Held: {trade.barlen}"
            )

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        params = {
            "bb_period": trial.suggest_int("bb_period", 10, 30),
            "bb_dev": trial.suggest_float("bb_dev", 1.5, 2.5, step=0.1),
            "supertrend_period": trial.suggest_int("supertrend_period", 7, 14),
            "supertrend_mult": trial.suggest_float(
                "supertrend_mult", 2.0, 4.0, step=0.5
            ),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            bb_period = params.get("bb_period", 20)
            supertrend_period = params.get("supertrend_period", 10)
            return max(bb_period, supertrend_period) + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 25
