import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class FlagPennantMomentum(bt.Strategy):
    """
    Flag/Pennant + Momentum Burst Strategy
    Strategy Type: INTRADAY BREAKOUT
    =================================
    Detects flag/pennant pattern with momentum (ROC) and volume surge confirmation.

    Strategy Logic:
    ==============
    Long Entry: Upper boundary break + ROC > 0 + Volume > 1.5 * SMA
    Short Entry: Lower boundary break + ROC < 0 + Volume > 1.5 * SMA
    Exit: Momentum exhaustion (ROC reverses) or pattern failure

    Parameters:
    ==========
    - roc_period (int): Rate of Change period (default: 10)
    - volume_sma_period (int): Volume SMA period (default: 20)
    - lookback (int): Bars for flag/pennant detection (default: 15)
    - target_multiplier (float): Multiplier for pattern target (default: 1.0)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("roc_period", 10),
        ("volume_sma_period", 20),
        ("lookback", 15),
        ("target_multiplier", 1.0),
        ("verbose", False),
    )

    optimization_params = {
        "roc_period": {"type": "int", "low": 5, "high": 15, "step": 1},
        "volume_sma_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "lookback": {"type": "int", "low": 10, "high": 20, "step": 1},
        "target_multiplier": {"type": "float", "low": 0.8, "high": 1.2, "step": 0.1},
    }

    def __init__(self):
        self.roc = btind.ROC(self.datas[0].close, period=self.params.roc_period)
        self.volume_sma = btind.SMA(
            self.datas[0].volume, period=self.params.volume_sma_period
        )
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.roc_period,
                self.params.volume_sma_period,
                self.params.lookback,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized FlagPennantMomentum with params: {self.params}")

    def detect_flag_pennant(self):
        """Detect flag/pennant pattern within lookback period."""
        lookback = self.params.lookback
        if len(self.datas[0]) < lookback:
            return None, None, None

        highs = self.datas[0].high.get(size=lookback)
        lows = self.datas[0].low.get(size=lookback)
        closes = self.datas[0].close.get(size=lookback)

        # Detect consolidation after a strong move
        range_high = max(highs[-lookback // 2 :])
        range_low = min(lows[-lookback // 2 :])
        prior_move = abs(highs[0] - lows[0])
        if (range_high - range_low) > 0.5 * prior_move:
            return None, None, None

        upper_level = range_high
        lower_level = range_low
        current_close = closes[-1]
        direction = (
            "bullish"
            if current_close > upper_level
            else "bearish" if current_close < lower_level else None
        )
        target = (
            current_close + prior_move * self.params.target_multiplier
            if direction == "bullish"
            else current_close - prior_move * self.params.target_multiplier
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
            np.isnan(self.roc[0])
            or np.isinf(self.roc[0])
            or np.isnan(self.volume_sma[0])
            or np.isinf(self.volume_sma[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: ROC={self.roc[0]}, Volume SMA={self.volume_sma[0]}"
            )
            return

        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.datas[0].close[0],
                "roc": self.roc[0],
                "volume": self.datas[0].volume[0],
                "volume_sma": self.volume_sma[0],
            }
        )

        direction, target, levels = self.detect_flag_pennant()
        volume_surge = self.datas[0].volume[0] > self.volume_sma[0] * 1.5

        if not self.position:
            if direction == "bullish" and self.roc[0] > 0 and volume_surge:
                self.order = self.buy()
                self.order_type = "enter_long"
                self.target_price = target
                self.upper_level, self.lower_level = levels
                trade_logger.info(
                    f"BUY SIGNAL (Flag/Pennant) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | ROC: {self.roc[0]:.2f} | Volume Surge: {volume_surge}"
                )
            elif direction == "bearish" and self.roc[0] < 0 and volume_surge:
                self.order = self.sell()
                self.order_type = "enter_short"
                self.target_price = target
                self.upper_level, self.lower_level = levels
                trade_logger.info(
                    f"SELL SIGNAL (Flag/Pennant) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | ROC: {self.roc[0]:.2f} | Volume Surge: {volume_surge}"
                )
        elif self.position.size > 0:
            if self.roc[0] < 0 or self.datas[0].close[0] < self.lower_level:
                self.order = self.sell()
                self.order_type = "exit_long"
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | Reason: {'Momentum exhaustion' if self.roc[0] < 0 else 'Pattern failure'}"
                )
        elif self.position.size < 0:
            if self.roc[0] > 0 or self.datas[0].close[0] > self.upper_level:
                self.order = self.buy()
                self.order_type = "exit_short"
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short) | Bar: {len(self)} | Time: {bar_time_ist} | "
                    f"Price: {self.datas[0].close[0]:.2f} | Reason: {'Momentum exhaustion' if self.roc[0] > 0 else 'Pattern failure'}"
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
            "roc_period": trial.suggest_int("roc_period", 5, 15),
            "volume_sma_period": trial.suggest_int("volume_sma_period", 10, 30),
            "lookback": trial.suggest_int("lookback", 10, 20),
            "target_multiplier": trial.suggest_float(
                "target_multiplier", 0.8, 1.2, step=0.1
            ),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            roc_period = params.get("roc_period", 10)
            volume_sma_period = params.get("volume_sma_period", 20)
            lookback = params.get("lookback", 15)
            return max(roc_period, volume_sma_period, lookback) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
