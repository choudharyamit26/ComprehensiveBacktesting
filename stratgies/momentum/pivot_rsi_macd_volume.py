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


class Pivot_RSI_MACD_Volume(bt.Strategy):
    """
    Pivot + RSI + MACD + Volume Strategy
    Strategy Type: PIVOT + MOMENTUM + VOLUME
    =======================================
    This strategy combines pivot points, RSI, MACD, and volume for trade confirmation.

    Strategy Logic:
    ==============
    Long Entry: Price above pivot + RSI bullish + MACD crossover + Volume increase
    Short Entry: Price below pivot + RSI bearish + MACD crossover + Volume increase
    Exit: Price reaches next pivot level or momentum failure

    Parameters:
    ==========
    - pivot_period (int): Pivot point calculation period (default: 20)
    - rsi_period (int): RSI period (default: 14)
    - macd_fast (int): MACD fast EMA period (default: 12)
    - macd_slow (int): MACD slow EMA period (default: 26)
    - macd_signal (int): MACD signal line period (default: 9)
    - volume_period (int): Volume SMA period (default: 14)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("pivot_period", 20),
        ("rsi_period", 14),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("volume_period", 14),
        ("verbose", False),
    )

    optimization_params = {
        "pivot_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "macd_fast": {"type": "int", "low": 8, "high": 16, "step": 1},
        "macd_slow": {"type": "int", "low": 20, "high": 30, "step": 1},
        "macd_signal": {"type": "int", "low": 5, "high": 12, "step": 1},
        "volume_period": {"type": "int", "low": 10, "high": 20, "step": 1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.pivot = PivotPoint(self.data, period=self.params.pivot_period)
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.macd = btind.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal,
        )
        self.volume_sma = btind.SMA(self.data.volume, period=self.params.volume_period)

        # Debug: Log available lines and their types
        logger.debug(f"Pivot lines: {self.pivot.lines.getlinealiases()}")
        logger.debug(f"RSI lines: {self.rsi.lines.getlinealiases()}")
        logger.debug(f"MACD lines: {self.macd.lines.getlinealiases()}")
        logger.debug(f"Volume SMA lines: {self.volume_sma.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.pivot_period,
                self.params.rsi_period,
                self.params.macd_slow,
                self.params.volume_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized Pivot_RSI_MACD_Volume with params: {self.params}")
        logger.info(
            f"Pivot_RSI_MACD_Volume initialized with pivot_period={self.p.pivot_period}, "
            f"rsi_period={self.p.rsi_period}, macd_fast={self.p.macd_fast}, "
            f"macd_slow={self.p.macd_slow}, macd_signal={self.p.macd_signal}, "
            f"volume_period={self.p.volume_period}"
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
            np.isnan(self.pivot.pivot[0])
            or np.isnan(self.rsi[0])
            or np.isnan(self.macd.macd[0])
            or np.isnan(self.macd.signal[0])
            or np.isnan(self.volume_sma[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"Pivot={self.pivot.pivot[0]}, RSI={self.rsi[0]}, "
                f"MACD={self.macd.macd[0]}, Signal={self.macd.signal[0]}, "
                f"Volume SMA={self.volume_sma[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "pivot": self.pivot.pivot[0],
                "r1": self.pivot.r1[0],
                "s1": self.pivot.s1[0],
                "rsi": self.rsi[0],
                "macd": self.macd.macd[0],
                "macd_signal": self.macd.signal[0],
                "volume_sma": self.volume_sma[0],
            }
        )

        # Trading Logic
        price_above_pivot = self.data.close[0] > self.pivot.pivot[0]
        price_below_pivot = self.data.close[0] < self.pivot.pivot[0]
        rsi_bullish = self.rsi[0] > 50
        rsi_bearish = self.rsi[0] < 50
        macd_bullish = (
            self.macd.macd[0] > self.macd.signal[0]
            and self.macd.macd[-1] <= self.macd.signal[-1]
        )
        macd_bearish = (
            self.macd.macd[0] < self.macd.signal[0]
            and self.macd.macd[-1] >= self.macd.signal[-1]
        )
        volume_increase = self.data.volume[0] > self.volume_sma[0]
        momentum_failure = (
            (self.macd.macd[0] < self.macd.signal[0])
            if self.position.size > 0
            else (
                (self.macd.macd[0] > self.macd.signal[0])
                if self.position.size < 0
                else False
            )
        )

        if not self.position:
            # Long Entry: Price above pivot + RSI bullish + MACD crossover + Volume increase
            if price_above_pivot and rsi_bullish and macd_bullish and volume_increase:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Pivot + RSI + MACD + Volume) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Pivot: {self.pivot.pivot[0]:.2f} (Above) | "
                    f"RSI: {self.rsi[0]:.2f} (Bullish) | "
                    f"MACD: {self.macd.macd[0]:.2f} > Signal: {self.macd.signal[0]:.2f} | "
                    f"Volume: {self.data.volume[0]:.2f} > SMA: {self.volume_sma[0]:.2f}"
                )
            # Short Entry: Price below pivot + RSI bearish + MACD crossover + Volume increase
            elif price_below_pivot and rsi_bearish and macd_bearish and volume_increase:
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Pivot + RSI + MACD + Volume) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Pivot: {self.pivot.pivot[0]:.2f} (Below) | "
                    f"RSI: {self.rsi[0]:.2f} (Bearish) | "
                    f"MACD: {self.macd.macd[0]:.2f} < Signal: {self.macd.signal[0]:.2f} | "
                    f"Volume: {self.data.volume[0]:.2f} > SMA: {self.volume_sma[0]:.2f}"
                )
        elif self.position.size > 0:  # Long position
            # Exit: Price reaches R1 or momentum failure
            if self.data.close[0] >= self.pivot.r1[0] or momentum_failure:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "Reached R1"
                    if self.data.close[0] >= self.pivot.r1[0]
                    else "Momentum failure"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - Pivot + RSI + MACD + Volume) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Pivot: {self.pivot.pivot[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.2f}, Signal: {self.macd.signal[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            # Exit: Price reaches S1 or momentum failure
            if self.data.close[0] <= self.pivot.s1[0] or momentum_failure:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "Reached S1"
                    if self.data.close[0] <= self.pivot.s1[0]
                    else "Momentum failure"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - Pivot + RSI + MACD + Volume) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Pivot: {self.pivot.pivot[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"MACD: {self.macd.macd[0]:.2f}, Signal: {self.macd.signal[0]:.2f}"
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
            "pivot_period": trial.suggest_int("pivot_period", 10, 30),
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "macd_fast": trial.suggest_int("macd_fast", 8, 16),
            "macd_slow": trial.suggest_int("macd_slow", 20, 30),
            "macd_signal": trial.suggest_int("macd_signal", 5, 12),
            "volume_period": trial.suggest_int("volume_period", 10, 20),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            pivot_period = params.get("pivot_period", 20)
            rsi_period = params.get("rsi_period", 14)
            macd_slow = params.get("macd_slow", 26)
            volume_period = params.get("volume_period", 14)
            return max(pivot_period, rsi_period, macd_slow, volume_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
