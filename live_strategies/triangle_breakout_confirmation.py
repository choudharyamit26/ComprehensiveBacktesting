import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class TriangleBreakoutConfirmation(bt.Strategy):
    """
    Triangle Breakout + Multi-Confirmation Strategy
    Strategy Type: INTRADAY BREAKOUT
    =================================
    Detects symmetrical triangle breakout with RSI, volume, and MACD confirmation.

    Strategy Logic:
    ==============
    Long Entry: Upper trendline break + RSI > 50 + Volume > 1.5 * SMA + MACD bullish
    Short Entry: Lower trendline break + RSI < 50 + Volume > 1.5 * SMA + MACD bearish
    Exit: Pattern target or false breakout (price returns to triangle)

    Parameters:
    ==========
    - rsi_period (int): RSI period (default: 14)
    - volume_sma_period (int): Volume SMA period (default: 20)
    - macd_fast (int): MACD fast EMA period (default: 12)
    - macd_slow (int): MACD slow EMA period (default: 26)
    - macd_signal (int): MACD signal period (default: 9)
    - lookback (int): Bars for triangle detection (default: 20)
    - target_multiplier (float): Multiplier for pattern target (default: 1.0)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("rsi_period", 14),
        ("volume_sma_period", 20),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("lookback", 20),
        ("target_multiplier", 1.0),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "volume_sma_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "macd_fast": {"type": "int", "low": 8, "high": 16, "step": 1},
        "macd_slow": {"type": "int", "low": 20, "high": 30, "step": 1},
        "macd_signal": {"type": "int", "low": 5, "high": 12, "step": 1},
        "lookback": {"type": "int", "low": 15, "high": 30, "step": 1},
        "target_multiplier": {"type": "float", "low": 0.8, "high": 1.2, "step": 0.1},
    }

    def __init__(self):
        self.rsi = btind.RSI_Safe(self.datas[0].close, period=self.params.rsi_period)
        self.volume_sma = btind.SMA(
            self.datas[0].volume, period=self.params.volume_sma_period
        )
        self.macd = btind.MACD(
            self.datas[0].close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal,
        )
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.rsi_period,
                self.params.volume_sma_period,
                self.params.macd_slow + self.params.macd_signal,
                self.params.lookback,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(
            f"Initialized TriangleBreakoutConfirmation with params: {self.params}"
        )

    def detect_triangle(self):
        """Detect symmetrical triangle within lookback period."""
        lookback = self.params.lookback
        if len(self.datas[0]) < lookback:
            return None, None, None

        highs = self.datas[0].high.get(size=lookback)
        lows = self.datas[0].low.get(size=lookback)
        closes = self.datas[0].close.get(size=lookback)

        # Detect converging highs and lows
        upper_trendline = np.polyfit(range(lookback), highs, 1)[0]  # Slope of highs
        lower_trendline = np.polyfit(range(lookback), lows, 1)[0]  # Slope of lows
        if (
            upper_trendline >= 0 or lower_trendline <= 0
        ):  # Converging if slopes are opposite
            return None, None, None

        upper_level = highs[-1]
        lower_level = lows[-1]
        current_close = closes[-1]
        direction = (
            "bullish"
            if current_close > upper_level
            else "bearish" if current_close < lower_level else None
        )
        triangle_height = max(highs) - min(lows)
        target = (
            current_close + triangle_height * self.params.target_multiplier
            if direction == "bullish"
            else current_close - triangle_height * self.params.target_multiplier
        )

        return direction, target, (upper_level, lower_level)

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

        if (
            np.isnan(self.rsi[0])
            or np.isinf(self.rsi[0])
            or np.isnan(self.volume_sma[0])
            or np.isinf(self.volume_sma[0])
            or np.isnan(self.macd.macd[0])
            or np.isinf(self.macd.macd[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: RSI={self.rsi[0]}, Volume SMA={self.volume_sma[0]}, MACD={self.macd.macd[0]}"
            )
            return

        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.datas[0].close[0],
                "rsi": self.rsi[0],
                "volume": self.datas[0].volume[0],
                "volume_sma": self.volume_sma[0],
                "macd": self.macd.macd[0],
                "macd_signal": self.macd.signal[0],
            }
        )

        direction, target, levels = self.detect_triangle()
        volume_surge = self.datas[0].volume[0] > self.volume_sma[0] * 1.5
        macd_bullish = self.macd.macd[0] > self.macd.signal[0]
        macd_bearish = self.macd.macd[0] < self.macd.signal[0]

        if not self.position:
            if (
                direction == "bullish"
                and self.rsi[0] > 50
                and volume_surge
                and macd_bullish
            ):
                self.order = self.buy()
                self.order_type = "enter_long"
                self.target_price = target
                self.upper_level, self.lower_level = levels
                trade_logger.info(
                    f"BUY SIGNAL (Triangle Breakout) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | RSI: {self.rsi[0]:.2f} | Volume Surge: {volume_surge} | MACD Bullish: {macd_bullish}"
                )
            elif (
                direction == "bearish"
                and self.rsi[0] < 50
                and volume_surge
                and macd_bearish
            ):
                self.order = self.sell()
                self.order_type = "enter_short"
                self.target_price = target
                self.upper_level, self.lower_level = levels
                trade_logger.info(
                    f"SELL SIGNAL (Triangle Breakout) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | RSI: {self.rsi[0]:.2f} | Volume Surge: {volume_surge} | MACD Bearish: {macd_bearish}"
                )
        elif self.position.size > 0:
            if (
                self.datas[0].close[0] >= self.target_price
                or self.datas[0].close[0] < self.lower_level
            ):
                self.order = self.sell()
                self.order_type = "exit_long"
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | Reason: {'Target reached' if self.datas[0].close[0] >= self.target_price else 'False breakout'}"
                )
        elif self.position.size < 0:
            if (
                self.datas[0].close[0] <= self.target_price
                or self.datas[0].close[0] > self.upper_level
            ):
                self.order = self.buy()
                self.order_type = "exit_short"
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | Reason: {'Target reached' if self.datas[0].close[0] <= self.target_price else 'False breakout'}"
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
                    f"BUY EXECUTED (Enter Long) | Ref: {order.ref} | Price: {order.executed.price:.2f} | "
                    f"Size: {order.executed.size} | Cost: {order.executed.value:.2f} | Comm: {order.executed.comm:.2f}"
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
                    f"SELL EXECUTED (Enter Short) | Ref: {order.ref} | Price: {order.executed.price:.2f} | "
                    f"Size: {order.executed.size} | Cost: {order.executed.value:.2f} | Comm: {order.executed.comm:.2f}"
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
                        f"SELL EXECUTED (Exit Long) | Ref: {order.ref} | Price: {order.executed.price:.2f} | "
                        f"Size: {order.executed.size} | Cost: {order.executed.value:.2f} | Comm: {order.executed.comm:.2f} | PnL: {pnl:.2f}"
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
                        f"BUY EXECUTED (Exit Short) | Ref: {order.ref} | Price: {order.executed.price:.2f} | "
                        f"Size: {order.executed.size} | Cost: {order.executed.value:.2f} | Comm: {order.executed.comm:.2f} | PnL: {pnl:.2f}"
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
                f"Net Profit: {trade.pnlcomm:.2f} | Bars Held: {trade.barlen} | Trade Count: {self.trade_count}"
            )

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        params = {
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "volume_sma_period": trial.suggest_int("volume_sma_period", 10, 30),
            "macd_fast": trial.suggest_int("macd_fast", 8, 16),
            "macd_slow": trial.suggest_int("macd_slow", 20, 30),
            "macd_signal": trial.suggest_int("macd_signal", 5, 12),
            "lookback": trial.suggest_int("lookback", 15, 30),
            "target_multiplier": trial.suggest_float(
                "target_multiplier", 0.8, 1.2, step=0.1
            ),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            volume_sma_period = params.get("volume_sma_period", 20)
            macd_slow = params.get("macd_slow", 26)
            macd_signal = params.get("macd_signal", 9)
            lookback = params.get("lookback", 20)
            return (
                max(rsi_period, volume_sma_period, macd_slow + macd_signal, lookback)
                + 2
            )
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
