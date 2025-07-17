import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class RSI_CCI_Williams_Stochastic(bt.Strategy):
    """
    RSI + CCI + Williams %R + Stochastic Strategy
    Strategy Type: MULTI-OSCILLATOR
    =================================
    This strategy combines RSI, CCI, Williams %R, and Stochastic for trade confirmation.

    Strategy Logic:
    ==============
    Long Entry: At least 3 oscillators bullish (RSI > 50, CCI > 0, Williams %R > -50, Stochastic %K > %D)
    Short Entry: At least 3 oscillators bearish (RSI < 50, CCI < 0, Williams %R < -50, Stochastic %K < %D)
    Exit: Majority of oscillators reverse (at least 3 show opposite signal)

    Parameters:
    ==========
    - rsi_period (int): RSI period (default: 14)
    - cci_period (int): CCI period (default: 20)
    - williams_period (int): Williams %R period (default: 14)
    - stoch_k (int): Stochastic %K period (default: 14)
    - stoch_d (int): Stochastic %D period (default: 3)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("rsi_period", 14),
        ("cci_period", 20),
        ("williams_period", 14),
        ("stoch_k", 14),
        ("stoch_d", 3),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "cci_period": {"type": "int", "low": 15, "high": 25, "step": 1},
        "williams_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stoch_k": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stoch_d": {"type": "int", "low": 2, "high": 5, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.cci = btind.CCI(self.data, period=self.params.cci_period)
        self.williams = btind.WilliamsR(self.data, period=self.params.williams_period)
        self.stoch = btind.Stochastic(
            self.data, period=self.params.stoch_k, period_dfast=self.params.stoch_d
        )

        # Debug: Log available lines and their types
        logger.debug(f"RSI lines: {self.rsi.lines.getlinealiases()}")
        logger.debug(f"CCI lines: {self.cci.lines.getlinealiases()}")
        logger.debug(f"Williams %R lines: {self.williams.lines.getlinealiases()}")
        logger.debug(f"Stochastic lines: {self.stoch.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.rsi_period,
                self.params.cci_period,
                self.params.williams_period,
                self.params.stoch_k,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(
            f"Initialized RSI_CCI_Williams_Stochastic with params: {self.params}"
        )
        logger.info(
            f"RSI_CCI_Williams_Stochastic initialized with rsi_period={self.p.rsi_period}, "
            f"cci_period={self.p.cci_period}, williams_period={self.p.williams_period}, "
            f"stoch_k={self.p.stoch_k}, stoch_d={self.p.stoch_d}"
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
            or np.isnan(self.cci[0])
            or np.isnan(self.williams[0])
            or np.isnan(self.stoch.percK[0])
            or np.isnan(self.stoch.percD[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"RSI={self.rsi[0]}, CCI={self.cci[0]}, "
                f"Williams %R={self.williams[0]}, Stochastic %K={self.stoch.percK[0]}, "
                f"Stochastic %D={self.stoch.percD[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "rsi": self.rsi[0],
                "cci": self.cci[0],
                "williams": self.williams[0],
                "stoch_k": self.stoch.percK[0],
                "stoch_d": self.stoch.percD[0],
            }
        )

        # Trading Logic
        rsi_bullish = self.rsi[0] > 50
        rsi_bearish = self.rsi[0] < 50
        cci_bullish = self.cci[0] > 0
        cci_bearish = self.cci[0] < 0
        williams_bullish = self.williams[0] > -50
        williams_bearish = self.williams[0] < -50
        stoch_bullish = self.stoch.percK[0] > self.stoch.percD[0]
        stoch_bearish = self.stoch.percK[0] < self.stoch.percD[0]

        bullish_count = sum([rsi_bullish, cci_bullish, williams_bullish, stoch_bullish])
        bearish_count = sum([rsi_bearish, cci_bearish, williams_bearish, stoch_bearish])

        if not self.position:
            # Long Entry: At least 3 oscillators bullish
            if bullish_count >= 3:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - RSI + CCI + Williams + Stochastic) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Bullish Count: {bullish_count}/4 | "
                    f"RSI: {self.rsi[0]:.2f} | CCI: {self.cci[0]:.2f} | "
                    f"Williams %R: {self.williams[0]:.2f} | "
                    f"Stochastic %K: {self.stoch.percK[0]:.2f}, %D: {self.stoch.percD[0]:.2f}"
                )
            # Short Entry: At least 3 oscillators bearish
            elif bearish_count >= 3:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - RSI + CCI + Williams + Stochastic) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Bearish Count: {bearish_count}/4 | "
                    f"RSI: {self.rsi[0]:.2f} | CCI: {self.cci[0]:.2f} | "
                    f"Williams %R: {self.williams[0]:.2f} | "
                    f"Stochastic %K: {self.stoch.percK[0]:.2f}, %D: {self.stoch.percD[0]:.2f}"
                )
        elif self.position.size > 0:  # Long position
            # Exit: At least 3 oscillators bearish
            if bearish_count >= 3:
                self.order = self.sell()
                self.order_type = "exit_long"
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - RSI + CCI + Williams + Stochastic) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: Majority reversal (Bearish Count: {bearish_count}/4) | "
                    f"RSI: {self.rsi[0]:.2f} | CCI: {self.cci[0]:.2f} | "
                    f"Williams %R: {self.williams[0]:.2f} | "
                    f"Stochastic %K: {self.stoch.percK[0]:.2f}, %D: {self.stoch.percD[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            # Exit: At least 3 oscillators bullish
            if bullish_count >= 3:
                self.order = self.buy()
                self.order_type = "exit_short"
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - RSI + CCI + Williams + Stochastic) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: Majority reversal (Bullish Count: {bullish_count}/4) | "
                    f"RSI: {self.rsi[0]:.2f} | CCI: {self.cci[0]:.2f} | "
                    f"Williams %R: {self.williams[0]:.2f} | "
                    f"Stochastic %K: {self.stoch.percK[0]:.2f}, %D: {self.stoch.percD[0]:.2f}"
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
            "cci_period": trial.suggest_int("cci_period", 15, 25),
            "williams_period": trial.suggest_int("williams_period", 10, 20),
            "stoch_k": trial.suggest_int("stoch_k", 10, 20),
            "stoch_d": trial.suggest_int("stoch_d", 2, 5),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            cci_period = params.get("cci_period", 20)
            williams_period = params.get("williams_period", 14)
            stoch_k = params.get("stoch_k", 14)
            return max(rsi_period, cci_period, williams_period, stoch_k) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
