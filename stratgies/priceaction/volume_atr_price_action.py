import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class VolumeSurge(bt.Indicator):
    lines = ("volume_surge",)
    params = (
        ("period", 14),
        ("threshold", 1.5),
    )

    def __init__(self):
        self.addminperiod(self.params.period)
        self.lines.volume_surge = self.data.volume / btind.SMA(
            self.data.volume, period=self.params.period
        )


class Volume_ATR_PriceAction(bt.Strategy):
    """
    Volume + ATR + Price Action Strategy
    Strategy Type: VOLUME + VOLATILITY + BREAKOUT
    ==========================================
    This strategy combines volume surges, ATR volatility expansion, and price breakouts.

    Strategy Logic:
    ==============
    Long Entry: Volume surge + ATR increase + Price breaks above recent high
    Short Entry: Volume surge + ATR increase + Price breaks below recent low
    Exit: Volume exhaustion or volatility contraction

    Parameters:
    ==========
    - volume_period (int): Volume surge period (default: 14)
    - volume_threshold (float): Volume surge threshold (default: 1.5)
    - atr_period (int): ATR period (default: 14)
    - breakout_period (int): Lookback period for price breakout (default: 20)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("volume_period", 14),
        ("volume_threshold", 1.5),
        ("atr_period", 14),
        ("breakout_period", 20),
        ("verbose", False),
    )

    optimization_params = {
        "volume_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "volume_threshold": {"type": "float", "low": 1.2, "high": 2.0, "step": 0.1},
        "atr_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "breakout_period": {"type": "int", "low": 15, "high": 30, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.volume_surge = VolumeSurge(
            self.data,
            period=self.params.volume_period,
            threshold=self.params.volume_threshold,
        )
        self.atr = btind.ATR(self.data, period=self.params.atr_period)
        self.highest = btind.Highest(self.data.high, period=self.params.breakout_period)
        self.lowest = btind.Lowest(self.data.low, period=self.params.breakout_period)

        # Debug: Log available lines and their types
        logger.debug(f"Volume Surge lines: {self.volume_surge.lines.getlinealiases()}")
        logger.debug(f"ATR lines: {self.atr.lines.getlinealiases()}")
        logger.debug(f"Highest lines: {self.highest.lines.getlinealiases()}")
        logger.debug(f"Lowest lines: {self.lowest.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.volume_period,
                self.params.atr_period,
                self.params.breakout_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized Volume_ATR_PriceAction with params: {self.params}")
        logger.info(
            f"Volume_ATR_PriceAction initialized with volume_period={self.p.volume_period}, "
            f"volume_threshold={self.p.volume_threshold}, atr_period={self.p.atr_period}, "
            f"breakout_period={self.p.breakout_period}"
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
            np.isnan(self.volume_surge[0])
            or np.isnan(self.atr[0])
            or np.isnan(self.highest[0])
            or np.isnan(self.lowest[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"Volume Surge={self.volume_surge[0]}, ATR={self.atr[0]}, "
                f"Highest={self.highest[0]}, Lowest={self.lowest[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "volume_surge": self.volume_surge[0],
                "atr": self.atr[0],
                "highest": self.highest[0],
                "lowest": self.lowest[0],
            }
        )

        # Trading Logic
        volume_surge = self.volume_surge[0] > self.params.volume_threshold
        atr_increasing = self.atr[0] > self.atr[-1]
        price_breakout_up = self.data.close[0] > self.highest[-1]
        price_breakout_down = self.data.close[0] < self.lowest[-1]
        volume_exhaustion = self.volume_surge[0] < 1.0
        volatility_contraction = self.atr[0] < self.atr[-1]

        if not self.position:
            # Long Entry: Volume surge + ATR increase + Price breaks above recent high
            if volume_surge and atr_increasing and price_breakout_up:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Volume + ATR + Price Action) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Volume Surge: {self.volume_surge[0]:.2f} | "
                    f"ATR: {self.atr[0]:.2f} (Increasing) | "
                    f"Breakout Above: {self.highest[-1]:.2f}"
                )
            # Short Entry: Volume surge + ATR increase + Price breaks below recent low
            elif volume_surge and atr_increasing and price_breakout_down:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Volume + ATR + Price Action) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Volume Surge: {self.volume_surge[0]:.2f} | "
                    f"ATR: {self.atr[0]:.2f} (Increasing) | "
                    f"Breakout Below: {self.lowest[-1]:.2f}"
                )
        elif self.position.size > 0:  # Long position
            if volume_exhaustion or volatility_contraction:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "Volume exhaustion"
                    if volume_exhaustion
                    else "Volatility contraction"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - Volume + ATR + Price Action) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Volume Surge: {self.volume_surge[0]:.2f} | "
                    f"ATR: {self.atr[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            if volume_exhaustion or volatility_contraction:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "Volume exhaustion"
                    if volume_exhaustion
                    else "Volatility contraction"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - Volume + ATR + Price Action) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Volume Surge: {self.volume_surge[0]:.2f} | "
                    f"ATR: {self.atr[0]:.2f}"
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
            "volume_period": trial.suggest_int("volume_period", 10, 20),
            "volume_threshold": trial.suggest_float(
                "volume_threshold", 1.2, 2.0, step=0.1
            ),
            "atr_period": trial.suggest_int("atr_period", 10, 20),
            "breakout_period": trial.suggest_int("breakout_period", 15, 30),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            volume_period = params.get("volume_period", 14)
            atr_period = params.get("atr_period", 14)
            breakout_period = params.get("breakout_period", 20)
            return max(volume_period, atr_period, breakout_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
