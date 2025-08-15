import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class RSIWilliamsStochastic(bt.Strategy):
    """
    RSI + Stochastic + Williams %R Strategy
    Strategy Type: MULTI-OSCILLATOR
    =============================
    This strategy uses three oscillators (RSI, Stochastic, Williams %R) for trade confirmation.

    Strategy Logic:
    ==============
    Long Entry: RSI > 50 + Stochastic %K > %D + Williams %R > -50
    Short Entry: RSI < 50 + Stochastic %K < %D + Williams %R < -50
    Exit: Majority of oscillators reverse or divergence appears

    Parameters:
    ==========
    - rsi_period (int): RSI period (default: 14)
    - stoch_k (int): Stochastic %K period (default: 14)
    - stoch_d (int): Stochastic %D period (default: 3)
    - willr_period (int): Williams %R period (default: 14)
    - rsi_threshold (int): RSI threshold (default: 50)
    - willr_threshold (int): Williams %R threshold (default: -50)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("rsi_period", 14),
        ("stoch_k", 14),
        ("stoch_d", 3),
        ("willr_period", 14),
        ("rsi_threshold", 50),
        ("willr_threshold", -50),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stoch_k": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stoch_d": {"type": "int", "low": 2, "high": 5, "step": 1},
        "willr_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "rsi_threshold": {"type": "int", "low": 45, "high": 55, "step": 1},
        "willr_threshold": {"type": "int", "low": -60, "high": -40, "step": 5},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.stoch = btind.Stochastic(
            self.data, period=self.params.stoch_k, period_dfast=self.params.stoch_d
        )
        self.willr = btind.WilliamsR(self.data, period=self.params.willr_period)
        self.order = None
        self.order_type = None
        self.ready = False
        self.warmup_period = (
            max(self.params.rsi_period, self.params.stoch_k, self.params.willr_period)
            + 2
        )
        self.trade_count = 0
        self.completed_trades = []
        self.open_positions = []
        logger.info(f"Initialized RSIWilliamsStochastic with params: {self.params}")

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

        if (
            np.isnan(self.rsi[0])
            or np.isnan(self.stoch.percK[0])
            or np.isnan(self.willr[0])
        ):
            return

        stoch_bullish = self.stoch.percK[0] > self.stoch.percD[0]
        stoch_bearish = self.stoch.percK[0] < self.stoch.percD[0]

        if not self.position:
            if (
                self.rsi[0] > self.params.rsi_threshold
                and stoch_bullish
                and self.willr[0] > self.params.willr_threshold
            ):
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )
            elif (
                self.rsi[0] < (100 - self.params.rsi_threshold)
                and stoch_bearish
                and self.willr[0] < -self.params.willr_threshold
            ):
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )
        elif self.position.size > 0:
            reverse_count = sum(
                [
                    self.rsi[0] < self.params.rsi_threshold,
                    not stoch_bullish,
                    self.willr[0] < self.params.willr_threshold,
                ]
            )
            if reverse_count >= 2:
                self.order = self.sell()
                self.order_type = "exit_long"
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long) | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )
        elif self.position.size < 0:
            reverse_count = sum(
                [
                    self.rsi[0] > (100 - self.params.rsi_threshold),
                    not stoch_bearish,
                    self.willr[0] > -self.params.willr_threshold,
                ]
            )
            if reverse_count >= 2:
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
            elif self.order_type == "enter_short" and order.issell():
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
                f"TRADE CLOSED | Ref: {trade.ref} | Profit: {trade.pnl:.2f} | Net Profit: {trade.pnlcomm:.2f}"
            )

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        return {
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "stoch_k": trial.suggest_int("stoch_k", 10, 20),
            "stoch_d": trial.suggest_int("stoch_d", 2, 5),
            "willr_period": trial.suggest_int("willr_period", 10, 20),
            "rsi_threshold": trial.suggest_int("rsi_threshold", 45, 55),
            "willr_threshold": trial.suggest_int("willr_threshold", -60, -40, step=5),
        }

    @classmethod
    def get_min_data_points(cls, params):
        try:
            return (
                max(
                    params.get("rsi_period", 14),
                    params.get("stoch_k", 14),
                    params.get("willr_period", 14),
                )
                + 2
            )
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
