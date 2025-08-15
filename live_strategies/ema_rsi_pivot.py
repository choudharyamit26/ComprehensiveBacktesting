import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class PivotPoint(bt.Indicator):
    lines = ("pivot", "r1", "s1")
    params = (("period", 20),)

    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        if len(self.data) < self.params.period:
            return

        high = max(self.data.high.get(ago=0, size=self.params.period))
        low = min(self.data.low.get(ago=0, size=self.params.period))
        close = self.data.close[-1]

        self.lines.pivot[0] = (high + low + close) / 3
        self.lines.r1[0] = 2 * self.lines.pivot[0] - low
        self.lines.s1[0] = 2 * self.lines.pivot[0] - high


class EMA_RSI_Pivot(bt.Strategy):
    """
    EMA + RSI + Pivot Strategy
    Strategy Type: EMA + MOMENTUM + PIVOT
    =====================================
    This strategy combines EMA, RSI, and pivot points for intraday trading signals.

    Strategy Logic:
    ==============
    Long Entry: Price above 20 EMA + RSI above 52 + Price above pivot point
    Short Entry: Price below 20 EMA + RSI below 48 + Price below pivot point
    Exit: Price reaches next pivot level (R1/S1), reversal of entry conditions, or stop loss
    Stop Loss: RSI > 80 (long) or RSI < 25 (short)

    Parameters:
    ==========
    - ema_period (int): EMA period (default: 20)
    - rsi_period (int): RSI period (default: 14)
    - pivot_period (int): Pivot point calculation period (default: 20)
    - rsi_upper (float): RSI upper threshold for buy (default: 52)
    - rsi_lower (float): RSI lower threshold for sell (default: 48)
    - rsi_stop_long (float): RSI stop loss for long positions (default: 80)
    - rsi_stop_short (float): RSI stop loss for short positions (default: 25)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("ema_period", 20),
        ("rsi_period", 14),
        ("pivot_period", 20),
        ("rsi_upper", 52),
        ("rsi_lower", 48),
        ("rsi_stop_long", 80),
        ("rsi_stop_short", 25),
        ("verbose", False),
    )

    optimization_params = {
        "ema_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "pivot_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "rsi_upper": {"type": "float", "low": 50, "high": 60, "step": 1},
        "rsi_lower": {"type": "float", "low": 40, "high": 50, "step": 1},
        "rsi_stop_long": {"type": "float", "low": 75, "high": 85, "step": 1},
        "rsi_stop_short": {"type": "float", "low": 20, "high": 30, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.ema = btind.EMA(self.data.close, period=self.params.ema_period)
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.pivot = PivotPoint(self.data, period=self.params.pivot_period)

        # Debug: Log available lines
        logger.debug(f"EMA lines: {self.ema.lines.getlinealiases()}")
        logger.debug(f"RSI lines: {self.rsi.lines.getlinealiases()}")
        logger.debug(f"Pivot lines: {self.pivot.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.ema_period, self.params.rsi_period, self.params.pivot_period
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized EMA_RSI_Pivot with params: {self.params}")
        logger.info(
            f"EMA_RSI_Pivot initialized with ema_period={self.params.ema_period}, "
            f"rsi_period={self.params.rsi_period}, pivot_period={self.params.pivot_period}, "
            f"rsi_upper={self.params.rsi_upper}, rsi_lower={self.params.rsi_lower}, "
            f"rsi_stop_long={self.params.rsi_stop_long}, rsi_stop_short={self.params.rsi_stop_short}"
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
            np.isnan(self.ema[0])
            or np.isnan(self.rsi[0])
            or np.isnan(self.pivot.pivot[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"EMA={self.ema[0]}, RSI={self.rsi[0]}, Pivot={self.pivot.pivot[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "ema": self.ema[0],
                "rsi": self.rsi[0],
                "pivot": self.pivot.pivot[0],
                "r1": self.pivot.r1[0],
                "s1": self.pivot.s1[0],
            }
        )

        # Trading Logic
        price_above_ema = self.data.close[0] > self.ema[0]
        price_below_ema = self.data.close[0] < self.ema[0]
        price_above_pivot = self.data.close[0] > self.pivot.pivot[0]
        price_below_pivot = self.data.close[0] < self.pivot.pivot[0]
        rsi_bullish = self.rsi[0] > self.params.rsi_upper
        rsi_bearish = self.rsi[0] < self.params.rsi_lower
        rsi_overbought = self.rsi[0] > self.params.rsi_stop_long
        rsi_oversold = self.rsi[0] < self.params.rsi_stop_short
        exit_long_condition = (
            price_below_ema or price_below_pivot or self.rsi[0] < 50 or rsi_overbought
        )
        exit_short_condition = (
            price_above_ema or price_above_pivot or self.rsi[0] > 50 or rsi_oversold
        )

        if not self.position:
            # Long Entry: Price above EMA + RSI above 52 + Price above pivot
            if price_above_ema and rsi_bullish and price_above_pivot:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - EMA + RSI + Pivot) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"EMA: {self.ema[0]:.2f} (Above) | "
                    f"RSI: {self.rsi[0]:.2f} (>{self.params.rsi_upper}) | "
                    f"Pivot: {self.pivot.pivot[0]:.2f} (Above)"
                )
            # Short Entry: Price below EMA + RSI below 48 + Price below pivot
            elif price_below_ema and rsi_bearish and price_below_pivot:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - EMA + RSI + Pivot) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"EMA: {self.ema[0]:.2f} (Below) | "
                    f"RSI: {self.rsi[0]:.2f} (<{self.params.rsi_lower}) | "
                    f"Pivot: {self.pivot.pivot[0]:.2f} (Below)"
                )
        elif self.position.size > 0:  # Long position
            # Exit: Price reaches R1, reversal of entry conditions, or RSI stop loss
            if self.data.close[0] >= self.pivot.r1[0] or exit_long_condition:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "Reached R1"
                    if self.data.close[0] >= self.pivot.r1[0]
                    else (
                        "RSI Stop Loss"
                        if rsi_overbought
                        else "Reversal of entry conditions"
                    )
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - EMA + RSI + Pivot) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"EMA: {self.ema[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"Pivot: {self.pivot.pivot[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            # Exit: Price reaches S1, reversal of entry conditions, or RSI stop loss
            if self.data.close[0] <= self.pivot.s1[0] or exit_short_condition:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "Reached S1"
                    if self.data.close[0] <= self.pivot.s1[0]
                    else (
                        "RSI Stop Loss"
                        if rsi_oversold
                        else "Reversal of entry conditions"
                    )
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - EMA + RSI + Pivot) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"EMA: {self.ema[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"Pivot: {self.pivot.pivot[0]:.2f}"
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
            "ema_period": trial.suggest_int("ema_period", 10, 30),
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "pivot_period": trial.suggest_int("pivot_period", 10, 30),
            "rsi_upper": trial.suggest_float("rsi_upper", 50, 60),
            "rsi_lower": trial.suggest_float("rsi_lower", 40, 50),
            "rsi_stop_long": trial.suggest_float("rsi_stop_long", 75, 85),
            "rsi_stop_short": trial.suggest_float("rsi_stop_short", 20, 30),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            ema_period = params.get("ema_period", 20)
            rsi_period = params.get("rsi_period", 14)
            pivot_period = params.get("pivot_period", 20)
            return max(ema_period, rsi_period, pivot_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
