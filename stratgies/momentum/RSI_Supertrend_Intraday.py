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
        self.final_ub = 0.0
        self.final_lb = 0.0
        self.prev_close = float("nan")

    def next(self):
        if len(self) == 1:
            self.final_ub = self.basic_ub[0]
            self.final_lb = self.basic_lb[0]
            self.prev_close = self.data.close[0]
            if self.data.close[0] > self.final_ub:
                self.lines.supertrend[0] = self.final_lb
            else:
                self.lines.supertrend[0] = self.final_ub
            return

        if (self.basic_ub[0] < self.final_ub) or (self.prev_close > self.final_ub):
            self.final_ub = self.basic_ub[0]
        if (self.basic_lb[0] > self.final_lb) or (self.prev_close < self.final_lb):
            self.final_lb = self.basic_lb[0]

        if self.data.close[0] <= self.final_ub:
            self.lines.supertrend[0] = self.final_ub
        else:
            self.lines.supertrend[0] = self.final_lb

        self.prev_close = self.data.close[0]


class RSISupertrendIntraday(bt.Strategy):
    """
    RSI + Supertrend Intraday Strategy
    Strategy Type: MOMENTUM
    =============================
    This strategy uses RSI and Supertrend on a 5-minute timeframe for intraday trading.
    It enters positions based on RSI momentum and Supertrend direction, with exits on target hit or RSI extremes.

    Strategy Logic:
    ==============
    Long Entry: RSI > 50 AND Supertrend is bullish (price > Supertrend)
    Short Entry: RSI < 50 AND Supertrend is bearish (price < Supertrend)
    Long Exit: Target hit (2% profit) OR RSI > 80
    Short Exit: Target hit (2% profit) OR RSI < 20

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST
    - Uses warmup period for indicator stability
    - Prevents order overlap

    Parameters:
    ==========
    - rsi_period (int): RSI period (default: 14)
    - supertrend_period (int): Supertrend period (default: 10)
    - supertrend_mult (float): Supertrend ATR multiplier (default: 3.0)
    - target_percent (float): Target profit percentage (default: 2.0)
    - rsi_overbought (int): RSI overbought threshold (default: 80)
    - rsi_oversold (int): RSI oversold threshold (default: 20)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("rsi_period", 14),
        ("supertrend_period", 10),
        ("supertrend_mult", 3.0),
        ("target_percent", 2.0),
        ("rsi_overbought", 80),
        ("rsi_oversold", 20),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "supertrend_period": {"type": "int", "low": 7, "high": 14, "step": 1},
        "supertrend_mult": {"type": "float", "low": 2.0, "high": 4.0, "step": 0.5},
        "target_percent": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.5},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.supertrend = SuperTrend(
            self.data,
            period=self.params.supertrend_period,
            multiplier=self.params.supertrend_mult,
        )
        self.bullish_entry = bt.And(self.rsi > 50, self.data.close > self.supertrend)
        self.bearish_entry = bt.And(self.rsi < 50, self.data.close < self.supertrend)
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(self.params.rsi_period, self.params.supertrend_period) + 5
        )
        self.completed_trades = []
        self.open_positions = []
        self.entry_price = None
        logger.info(f"Initialized RSISupertrendIntraday with params: {self.params}")

    def next(self):
        if len(self) < self.warmup_period:
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
            return

        if np.isnan(self.rsi[0]) or np.isnan(self.supertrend[0]):
            return

        if not self.position:
            if self.bullish_entry[0]:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )
            elif self.bearish_entry[0]:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )
        else:
            if self.position.size > 0:
                target_price = self.entry_price * (1 + self.params.target_percent / 100)
                if (
                    self.data.close[0] >= target_price
                    or self.rsi[0] > self.params.rsi_overbought
                ):
                    self.order = self.sell()
                    self.order_type = "exit_long"
                    trade_logger.info(
                        f"SELL SIGNAL (Exit Long) | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                    )
            elif self.position.size < 0:
                target_price = self.entry_price * (1 - self.params.target_percent / 100)
                if (
                    self.data.close[0] <= target_price
                    or self.rsi[0] < self.params.rsi_oversold
                ):
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
                self.entry_price = order.executed.price  # Set entry_price here
                self.open_positions.append(
                    {
                        "entry_time": exec_dt,
                        "entry_price": order.executed.price,
                        "size": order.executed.size,
                        "commission": order.executed.comm,
                        "ref": order.ref,
                        "direction": "long",
                    }
                )
                trade_logger.info(
                    f"BUY EXECUTED (Enter Long) | Ref: {order.ref} | Price: {order.executed.price:.2f}"
                )
            elif self.order_type == "enter_short" and order.issell():
                self.entry_price = order.executed.price  # Set entry_price here
                self.open_positions.append(
                    {
                        "entry_time": exec_dt,
                        "entry_price": order.executed.price,
                        "size": order.executed.size,
                        "commission": order.executed.comm,
                        "ref": order.ref,
                        "direction": "short",
                    }
                )
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
                    self.completed_trades.append(
                        {
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
                    )
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
                    self.completed_trades.append(
                        {
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
                    )
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
            if order.status in [order.Completed] and self.order_type in [
                "exit_long",
                "exit_short",
            ]:
                self.entry_price = None  # Reset entry_price after exiting position
            self.order = None
            self.order_type = None

    def notify_trade(self, trade):
        if trade.isclosed:
            trade_logger.info(
                f"TRADE CLOSED | Ref: {trade.ref} | Profit: {trade.pnl:.2f} | Net Profit: {trade.pnlcomm:.2f}"
            )

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        return {
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "supertrend_period": trial.suggest_int("supertrend_period", 7, 14),
            "supertrend_mult": trial.suggest_float(
                "supertrend_mult", 2.0, 4.0, step=0.5
            ),
            "target_percent": trial.suggest_float("target_percent", 1.5, 3.0, step=0.5),
        }

    @classmethod
    def get_min_data_points(cls, params):
        try:
            return (
                max(params.get("rsi_period", 14), params.get("supertrend_period", 10))
                + 5
            )
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 25
