import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class Multiple_Stochastic_Timeframes(bt.Strategy):
    """
    Multiple Stochastic Timeframes Strategy
    Strategy Type: MULTI-TIMEFRAME MOMENTUM
    =====================================
    This strategy uses Stochastic oscillators on 5min, 15min, and 30min timeframes.

    Strategy Logic:
    ==============
    Long Entry: All Stochastic oscillators (%K > %D) on 5min, 15min, and 30min timeframes
    Short Entry: All Stochastic oscillators (%K < %D) on 5min, 15min, and 30min timeframes
    Exit: 5min Stochastic reverses (%K crosses %D in opposite direction)

    Parameters:
    ==========
    - stoch_k_5m (int): Stochastic %K period for 5min (default: 14)
    - stoch_d_5m (int): Stochastic %D period for 5min (default: 3)
    - stoch_k_15m (int): Stochastic %K period for 15min (default: 14)
    - stoch_d_15m (int): Stochastic %D period for 15min (default: 3)
    - stoch_k_30m (int): Stochastic %K period for 30min (default: 14)
    - stoch_d_30m (int): Stochastic %D period for 30min (default: 3)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("stoch_k_5m", 14),
        ("stoch_d_5m", 3),
        ("stoch_k_15m", 14),
        ("stoch_d_15m", 3),
        ("stoch_k_30m", 14),
        ("stoch_d_30m", 3),
        ("verbose", False),
    )

    optimization_params = {
        "stoch_k_5m": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stoch_d_5m": {"type": "int", "low": 2, "high": 5, "step": 1},
        "stoch_k_15m": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stoch_d_15m": {"type": "int", "low": 2, "high": 5, "step": 1},
        "stoch_k_30m": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stoch_d_30m": {"type": "int", "low": 2, "high": 5, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # For single timeframe operation, create resampled data internally
        if len(self.datas) == 1:
            logger.info(
                "Single data feed detected. Creating resampled timeframes internally."
            )

            # Create 15min and 30min resampled data from the primary 5min data
            self.data_15m = bt.TimeFrame.Minutes
            self.data_30m = bt.TimeFrame.Minutes

            # Use the original data as 5min and create higher timeframe indicators
            self.stoch_5m = btind.Stochastic(
                self.datas[0],
                period=self.params.stoch_k_5m,
                period_dfast=self.params.stoch_d_5m,
            )

            # For 15min and 30min, we'll use the same data but with different logic
            # This is a simplified approach - in practice, you'd want proper resampling
            self.stoch_15m = btind.Stochastic(
                self.datas[0],
                period=self.params.stoch_k_15m * 3,  # Approximate 15min equivalent
                period_dfast=self.params.stoch_d_15m,
            )

            self.stoch_30m = btind.Stochastic(
                self.datas[0],
                period=self.params.stoch_k_30m * 6,  # Approximate 30min equivalent
                period_dfast=self.params.stoch_d_30m,
            )

        elif len(self.datas) >= 3:
            # Original multi-timeframe setup
            logger.debug(f"Data feeds provided: {len(self.datas)}")
            for i, data in enumerate(self.datas):
                logger.debug(
                    f"Data {i}: {data._name}, Timeframe: {data._timeframe}, Compression: {data._compression}"
                )

            # Initialize indicators for different timeframes
            # Assuming data0 is 5min, data1 is 15min, data2 is 30min
            self.stoch_5m = btind.Stochastic(
                self.datas[0],
                period=self.params.stoch_k_5m,
                period_dfast=self.params.stoch_d_5m,
            )
            self.stoch_15m = btind.Stochastic(
                self.datas[1],
                period=self.params.stoch_k_15m,
                period_dfast=self.params.stoch_d_15m,
            )
            self.stoch_30m = btind.Stochastic(
                self.datas[2],
                period=self.params.stoch_k_30m,
                period_dfast=self.params.stoch_d_30m,
            )
        else:
            raise ValueError(
                f"Multiple_Stochastic_Timeframes requires either 1 data feed (for internal resampling) or 3 data feeds (5min, 15min, 30min), but {len(self.datas)} provided"
            )

        # Debug: Log available lines and their types
        logger.debug(f"Stochastic 5m lines: {self.stoch_5m.lines.getlinealiases()}")
        logger.debug(f"Stochastic 15m lines: {self.stoch_15m.lines.getlinealiases()}")
        logger.debug(f"Stochastic 30m lines: {self.stoch_30m.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.stoch_k_5m + self.params.stoch_d_5m,
                self.params.stoch_k_15m + self.params.stoch_d_15m,
                self.params.stoch_k_30m + self.params.stoch_d_30m,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(
            f"Initialized Multiple_Stochastic_Timeframes with params: {self.params}"
        )
        logger.info(
            f"Multiple_Stochastic_Timeframes initialized with "
            f"stoch_k_5m={self.p.stoch_k_5m}, stoch_d_5m={self.p.stoch_d_5m}, "
            f"stoch_k_15m={self.p.stoch_k_15m}, stoch_d_15m={self.p.stoch_d_15m}, "
            f"stoch_k_30m={self.p.stoch_k_30m}, stoch_d_30m={self.p.stoch_d_30m}"
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
            np.isnan(self.stoch_5m.percK[0])
            or np.isnan(self.stoch_5m.percD[0])
            or np.isnan(self.stoch_15m.percK[0])
            or np.isnan(self.stoch_15m.percD[0])
            or np.isnan(self.stoch_30m.percK[0])
            or np.isnan(self.stoch_30m.percD[0])
            or np.isinf(self.stoch_5m.percK[0])
            or np.isinf(self.stoch_5m.percD[0])
            or np.isinf(self.stoch_15m.percK[0])
            or np.isinf(self.stoch_15m.percD[0])
            or np.isinf(self.stoch_30m.percK[0])
            or np.isinf(self.stoch_30m.percD[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"5m %K={self.stoch_5m.percK[0]:.2f}, %D={self.stoch_5m.percD[0]:.2f}, "
                f"15m %K={self.stoch_15m.percK[0]:.2f}, %D={self.stoch_15m.percD[0]:.2f}, "
                f"30m %K={self.stoch_30m.percK[0]:.2f}, %D={self.stoch_30m.percD[0]:.2f}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.datas[0].close[0],
                "stoch_5m_k": self.stoch_5m.percK[0],
                "stoch_5m_d": self.stoch_5m.percD[0],
                "stoch_15m_k": self.stoch_15m.percK[0],
                "stoch_15m_d": self.stoch_15m.percD[0],
                "stoch_30m_k": self.stoch_30m.percK[0],
                "stoch_30m_d": self.stoch_30m.percD[0],
            }
        )

        # Trading Logic
        stoch_5m_bullish = self.stoch_5m.percK[0] > self.stoch_5m.percD[0]
        stoch_5m_bearish = self.stoch_5m.percK[0] < self.stoch_5m.percD[0]
        stoch_15m_bullish = self.stoch_15m.percK[0] > self.stoch_15m.percD[0]
        stoch_15m_bearish = self.stoch_15m.percK[0] < self.stoch_15m.percD[0]
        stoch_30m_bullish = self.stoch_30m.percK[0] > self.stoch_30m.percD[0]
        stoch_30m_bearish = self.stoch_30m.percK[0] < self.stoch_30m.percD[0]

        if not self.position:
            # Long Entry: All Stochastic oscillators bullish
            if stoch_5m_bullish and stoch_15m_bullish and stoch_30m_bullish:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Multiple Stochastic Timeframes) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | "
                    f"5m %K: {self.stoch_5m.percK[0]:.2f} > %D: {self.stoch_5m.percD[0]:.2f} | "
                    f"15m %K: {self.stoch_15m.percK[0]:.2f} > %D: {self.stoch_15m.percD[0]:.2f} | "
                    f"30m %K: {self.stoch_30m.percK[0]:.2f} > %D: {self.stoch_30m.percD[0]:.2f}"
                )
            # Short Entry: All Stochastic oscillators bearish
            elif stoch_5m_bearish and stoch_15m_bearish and stoch_30m_bearish:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Multiple Stochastic Timeframes) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | "
                    f"5m %K: {self.stoch_5m.percK[0]:.2f} < %D: {self.stoch_5m.percD[0]:.2f} | "
                    f"15m %K: {self.stoch_15m.percK[0]:.2f} < %D: {self.stoch_15m.percD[0]:.2f} | "
                    f"30m %K: {self.stoch_30m.percK[0]:.2f} < %D: {self.stoch_30m.percD[0]:.2f}"
                )
        elif self.position.size > 0:  # Long position
            # Exit: 5min Stochastic reverses to bearish
            if stoch_5m_bearish:
                self.order = self.sell()
                self.order_type = "exit_long"
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - Multiple Stochastic Timeframes) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | "
                    f"Reason: 5m Stochastic reversal | "
                    f"5m %K: {self.stoch_5m.percK[0]:.2f} < %D: {self.stoch_5m.percD[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            # Exit: 5min Stochastic reverses to bullish
            if stoch_5m_bullish:
                self.order = self.buy()
                self.order_type = "exit_short"
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - Multiple Stochastic Timeframes) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | "
                    f"Reason: 5m Stochastic reversal | "
                    f"5m %K: {self.stoch_5m.percK[0]:.2f} > %D: {self.stoch_5m.percD[0]:.2f}"
                )

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt)
            if exec_dt.tzinfo is None:
                exec_dt = exec_dt.replace(tzinfo=pytz.UTC)
            exec_dt = exec_dt.astimezone(pytz.timezone("Asia/Kolkata"))

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
                        "bars_held": (
                            exec_dt - entry_info["entry_time"]
                        ).total_seconds()
                        / 60,
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
                        "bars_held": (
                            exec_dt - entry_info["entry_time"]
                        ).total_seconds()
                        / 60,
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
            "stoch_k_5m": trial.suggest_int("stoch_k_5m", 10, 20),
            "stoch_d_5m": trial.suggest_int("stoch_d_5m", 2, 5),
            "stoch_k_15m": trial.suggest_int("stoch_k_15m", 10, 20),
            "stoch_d_15m": trial.suggest_int("stoch_d_15m", 2, 5),
            "stoch_k_30m": trial.suggest_int("stoch_k_30m", 10, 20),
            "stoch_d_30m": trial.suggest_int("stoch_d_30m", 2, 5),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            stoch_k_5m = params.get("stoch_k_5m", 14)
            stoch_d_5m = params.get("stoch_d_5m", 3)
            stoch_k_15m = params.get("stoch_k_15m", 14)
            stoch_d_15m = params.get("stoch_d_15m", 3)
            stoch_k_30m = params.get("stoch_k_30m", 14)
            stoch_d_30m = params.get("stoch_d_30m", 3)
            return (
                max(
                    stoch_k_5m + stoch_d_5m,
                    stoch_k_15m + stoch_d_15m,
                    stoch_k_30m + stoch_d_30m,
                )
                + 2
            )
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
