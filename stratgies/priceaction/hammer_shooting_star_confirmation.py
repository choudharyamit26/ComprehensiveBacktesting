import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class HammerShootingStarConfirmation(bt.Strategy):
    """
    Hammer/Shooting Star + Multiple Confirmations Strategy
    Strategy Type: INTRADAY REVERSAL
    =================================
    Detects Hammer or Shooting Star candlestick with RSI and volume confirmation.

    Strategy Logic:
    ==============
    Long Entry: Hammer + RSI < 30 + Volume > 1.5 * SMA
    Short Entry: Shooting Star + RSI > 70 + Volume > 1.5 * SMA
    Exit: Reversal fails (price breaks opposite wick) or target reached (2 * wick size)

    Parameters:
    ==========
    - rsi_period (int): RSI period (default: 14)
    - volume_sma_period (int): Volume SMA period (default: 20)
    - hammer_threshold (float): Min lower wick size for Hammer (default: 2.0)
    - shooting_star_threshold (float): Min upper wick size for Shooting Star (default: 2.0)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("rsi_period", 14),
        ("volume_sma_period", 20),
        ("hammer_threshold", 2.0),
        ("shooting_star_threshold", 2.0),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "volume_sma_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "hammer_threshold": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.5},
        "shooting_star_threshold": {
            "type": "float",
            "low": 1.5,
            "high": 3.0,
            "step": 0.5,
        },
    }

    def __init__(self):
        self.rsi = btind.RSI_Safe(self.datas[0].close, period=self.params.rsi_period)
        self.volume_sma = btind.SMA(
            self.datas[0].volume, period=self.params.volume_sma_period
        )
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(self.params.rsi_period, self.params.volume_sma_period) + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(
            f"Initialized HammerShootingStarConfirmation with params: {self.params}"
        )

    def detect_candlestick(self):
        """Detect Hammer or Shooting Star candlestick."""
        body = abs(self.datas[0].close[0] - self.datas[0].open[0])
        candle_range = self.datas[0].high[0] - self.datas[0].low[0]
        lower_wick = (
            min(self.datas[0].open[0], self.datas[0].close[0]) - self.datas[0].low[0]
        )
        upper_wick = self.datas[0].high[0] - max(
            self.datas[0].open[0], self.datas[0].close[0]
        )

        if candle_range == 0:
            return None, None

        is_hammer = (
            lower_wick > self.params.hammer_threshold * body and upper_wick < body
        )
        is_shooting_star = (
            upper_wick > self.params.shooting_star_threshold * body
            and lower_wick < body
        )
        target = None
        if is_hammer:
            target = self.datas[0].close[0] + 2 * lower_wick
        elif is_shooting_star:
            target = self.datas[0].close[0] - 2 * upper_wick

        return (
            "hammer" if is_hammer else "shooting_star" if is_shooting_star else None
        ), target

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
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: RSI={self.rsi[0]}, Volume SMA={self.volume_sma[0]}"
            )
            return

        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.datas[0].close[0],
                "rsi": self.rsi[0],
                "volume": self.datas[0].volume[0],
                "volume_sma": self.volume_sma[0],
            }
        )

        candle_type, target = self.detect_candlestick()
        volume_surge = self.datas[0].volume[0] > self.volume_sma[0] * 1.5

        if not self.position:
            if candle_type == "hammer" and self.rsi[0] < 30 and volume_surge:
                self.order = self.buy()
                self.order_type = "enter_long"
                self.target_price = target
                trade_logger.info(
                    f"BUY SIGNAL (Hammer) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | RSI: {self.rsi[0]:.2f} | Volume Surge: {volume_surge}"
                )
            elif candle_type == "shooting_star" and self.rsi[0] > 70 and volume_surge:
                self.order = self.sell()
                self.order_type = "enter_short"
                self.target_price = target
                trade_logger.info(
                    f"SELL SIGNAL (Shooting Star) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | RSI: {self.rsi[0]:.2f} | Volume Surge: {volume_surge}"
                )
        elif self.position.size > 0:
            wick_low = (
                min(self.datas[0].open[-1], self.datas[0].close[-1])
                - self.datas[0].low[-1]
            )
            if (
                self.datas[0].close[0] >= self.target_price
                or self.datas[0].close[0] < self.datas[0].low[-1] - wick_low
            ):
                self.order = self.sell()
                self.order_type = "exit_long"
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | Reason: {'Target reached' if self.datas[0].close[0] >= self.target_price else 'Reversal failure'}"
                )
        elif self.position.size < 0:
            wick_high = self.datas[0].high[-1] - max(
                self.datas[0].open[-1], self.datas[0].close[-1]
            )
            if (
                self.datas[0].close[0] <= self.target_price
                or self.datas[0].close[0] > self.datas[0].high[-1] + wick_high
            ):
                self.order = self.buy()
                self.order_type = "exit_short"
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | Reason: {'Target reached' if self.datas[0].close[0] <= self.target_price else 'Reversal failure'}"
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
            "hammer_threshold": trial.suggest_float(
                "hammer_threshold", 1.5, 3.0, step=0.5
            ),
            "shooting_star_threshold": trial.suggest_float(
                "shooting_star_threshold", 1.5, 3.0, step=0.5
            ),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            volume_sma_period = params.get("volume_sma_period", 20)
            return max(rsi_period, volume_sma_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
