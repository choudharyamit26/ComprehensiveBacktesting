import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class TripleMomentumConfirmation(bt.Strategy):
    """
    Triple Momentum Confirmation Strategy
    Strategy Type: MOMENTUM
    =============================
    This strategy combines RSI, MACD, and Volume to identify strong momentum trades
    with confirmation from multiple indicators.

    Strategy Logic:
    ==============
    Long Entry: RSI > 50 + MACD bullish crossover + Volume spike (above average)
    Short Entry: RSI < 50 + MACD bearish crossover + Volume spike (above average)
    Exit: Any of the three indicators turns bearish (RSI crosses 50, MACD reverses, or volume drops)

    Parameters:
    ==========
    - rsi_period (int): RSI calculation period (default: 14)
    - macd_fast (int): MACD fast EMA period (default: 12)
    - macd_slow (int): MACD slow EMA period (default: 26)
    - macd_signal (int): MACD signal line period (default: 9)
    - volume_period (int): Volume moving average period (default: 20)
    - volume_factor (float): Multiplier for volume spike detection (default: 1.5)
    - rsi_threshold (int): RSI threshold for momentum (default: 50)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("rsi_period", 14),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("volume_period", 20),
        ("volume_factor", 1.5),
        ("rsi_threshold", 50),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "macd_fast": {"type": "int", "low": 8, "high": 16, "step": 1},
        "macd_slow": {"type": "int", "low": 20, "high": 30, "step": 1},
        "macd_signal": {"type": "int", "low": 6, "high": 12, "step": 1},
        "volume_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "volume_factor": {"type": "float", "low": 1.2, "high": 2.0, "step": 0.1},
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
        self.volume_ma = btind.SMA(self.data.volume, period=self.params.volume_period)
        self.order = None
        self.order_type = None
        self.ready = False
        self.warmup_period = (
            max(
                self.params.rsi_period, self.params.macd_slow, self.params.volume_period
            )
            + 2
        )
        self.trade_count = 0
        self.completed_trades = []
        self.open_positions = []
        logger.info(
            f"Initialized TripleMomentumConfirmation with params: {self.params}"
        )

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
            or np.isnan(self.macd.macd[0])
            or np.isnan(self.volume_ma[0])
        ):
            return

        volume_spike = (
            self.data.volume[0] > self.volume_ma[0] * self.params.volume_factor
        )
        macd_bullish = (
            self.macd.macd[0] > self.macd.signal[0]
            and self.macd.macd[-1] <= self.macd.signal[-1]
        )
        macd_bearish = (
            self.macd.macd[0] < self.macd.signal[0]
            and self.macd.macd[-1] >= self.macd.signal[-1]
        )

        if not self.position:
            if (
                self.rsi[0] > self.params.rsi_threshold
                and macd_bullish
                and volume_spike
            ):
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )
            elif (
                self.rsi[0] < (100 - self.params.rsi_threshold)
                and macd_bearish
                and volume_spike
            ):
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )
        elif self.position.size > 0:
            if (
                self.rsi[0] < self.params.rsi_threshold
                or self.macd.macd[0] < self.macd.signal[0]
                or not volume_spike
            ):
                self.order = self.sell()
                self.order_type = "exit_long"
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long) | Time: {bar_time_ist} | Price: {self.data.close[0]:.2f}"
                )
        elif self.position.size < 0:
            if (
                self.rsi[0] > (100 - self.params.rsi_threshold)
                or self.macd.macd[0] > self.macd.signal[0]
                or not volume_spike
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
            "macd_fast": trial.suggest_int("macd_fast", 8, 16),
            "macd_slow": trial.suggest_int("macd_slow", 20, 30),
            "macd_signal": trial.suggest_int("macd_signal", 6, 12),
            "volume_period": trial.suggest_int("volume_period", 10, 30),
            "volume_factor": trial.suggest_float("volume_factor", 1.2, 2.0),
            "rsi_threshold": trial.suggest_int("rsi_threshold", 45, 55),
        }

    @classmethod
    def get_min_data_points(cls, params):
        try:
            return (
                max(
                    params.get("rsi_period", 14),
                    params.get("macd_slow", 26),
                    params.get("volume_period", 20),
                )
                + 2
            )
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
