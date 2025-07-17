import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class SupportResistance(bt.Indicator):
    """
    Support and Resistance Level Detection
    """

    lines = (
        "support",
        "resistance",
        "at_support",
        "at_resistance",
        "break_support",
        "break_resistance",
    )
    params = (
        ("period", 20),
        ("min_touches", 2),
        ("tolerance", 0.002),  # 0.2% tolerance for level detection
    )

    def __init__(self):
        self.addminperiod(self.params.period * 2)
        self.support_levels = []
        self.resistance_levels = []

    def next(self):
        if len(self.data) < self.params.period * 2:
            return

        # Look for pivot points in the recent period
        highs = [self.data.high[-i] for i in range(self.params.period)]
        lows = [self.data.low[-i] for i in range(self.params.period)]

        # Find local maxima and minima
        current_high = max(highs)
        current_low = min(lows)

        # Update support and resistance levels
        self._update_levels(current_high, current_low)

        # Determine current levels
        current_support = self._get_nearest_support()
        current_resistance = self._get_nearest_resistance()

        self.lines.support[0] = current_support if current_support else self.data.low[0]
        self.lines.resistance[0] = (
            current_resistance if current_resistance else self.data.high[0]
        )

        # Check if price is at levels
        self.lines.at_support[0] = self._is_at_level(
            self.data.close[0], current_support
        )
        self.lines.at_resistance[0] = self._is_at_level(
            self.data.close[0], current_resistance
        )

        # Check for level breaks
        if len(self.data) > 1:
            self.lines.break_support[0] = (
                (
                    self.data.close[-1] >= current_support
                    and self.data.close[0]
                    < current_support * (1 - self.params.tolerance)
                )
                if current_support
                else False
            )

            self.lines.break_resistance[0] = (
                (
                    self.data.close[-1] <= current_resistance
                    and self.data.close[0]
                    > current_resistance * (1 + self.params.tolerance)
                )
                if current_resistance
                else False
            )
        else:
            self.lines.break_support[0] = False
            self.lines.break_resistance[0] = False

    def _update_levels(self, high, low):
        # Add new levels if they don't exist
        if not any(
            abs(high - level) / level < self.params.tolerance
            for level in self.resistance_levels
        ):
            self.resistance_levels.append(high)
        if not any(
            abs(low - level) / level < self.params.tolerance
            for level in self.support_levels
        ):
            self.support_levels.append(low)

        # Keep only recent levels (last 50)
        self.resistance_levels = self.resistance_levels[-50:]
        self.support_levels = self.support_levels[-50:]

    def _get_nearest_support(self):
        if not self.support_levels:
            return None
        current_price = self.data.close[0]
        valid_supports = [
            level for level in self.support_levels if level < current_price
        ]
        return max(valid_supports) if valid_supports else None

    def _get_nearest_resistance(self):
        if not self.resistance_levels:
            return None
        current_price = self.data.close[0]
        valid_resistances = [
            level for level in self.resistance_levels if level > current_price
        ]
        return min(valid_resistances) if valid_resistances else None

    def _is_at_level(self, price, level):
        if level is None:
            return False
        return abs(price - level) / level < self.params.tolerance


class VolumeConfirmation(bt.Indicator):
    """
    Volume Confirmation Indicator
    """

    lines = ("volume_surge", "volume_avg", "volume_ratio")
    params = (
        ("period", 20),
        ("surge_multiplier", 1.5),
    )

    def __init__(self):
        self.volume_sma = btind.SMA(self.data.volume, period=self.params.period)

    def next(self):
        if len(self.data) < self.params.period:
            return

        self.lines.volume_avg[0] = self.volume_sma[0]
        self.lines.volume_ratio[0] = (
            self.data.volume[0] / self.volume_sma[0] if self.volume_sma[0] > 0 else 1.0
        )
        self.lines.volume_surge[0] = (
            self.lines.volume_ratio[0] >= self.params.surge_multiplier
        )


class SRRSIVolume(bt.Strategy):
    """
    Support/Resistance + RSI + Volume Strategy
    Strategy Type: BREAKOUT + OSCILLATOR + VOLUME
    =============================================
    This strategy combines Support/Resistance levels, RSI oscillator, and volume confirmation.

    Strategy Logic:
    ==============
    Long Entry: Price bounces off support + RSI oversold recovery + Volume surge
    Short Entry: Price rejected at resistance + RSI overbought decline + Volume surge
    Exit: Level break against position or RSI reaches opposite extreme

    Parameters:
    ==========
    - sr_period (int): Support/Resistance detection period (default: 20)
    - sr_tolerance (float): Level tolerance percentage (default: 0.002)
    - rsi_period (int): RSI period (default: 14)
    - rsi_oversold (int): RSI oversold level (default: 30)
    - rsi_overbought (int): RSI overbought level (default: 70)
    - volume_period (int): Volume average period (default: 20)
    - volume_surge (float): Volume surge multiplier (default: 1.5)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("sr_period", 20),
        ("sr_tolerance", 0.002),
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
        ("volume_period", 20),
        ("volume_surge", 1.5),
        ("verbose", False),
    )

    optimization_params = {
        "sr_period": {"type": "int", "low": 15, "high": 30, "step": 1},
        "sr_tolerance": {"type": "float", "low": 0.001, "high": 0.005, "step": 0.0005},
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "rsi_oversold": {"type": "int", "low": 20, "high": 35, "step": 1},
        "rsi_overbought": {"type": "int", "low": 65, "high": 80, "step": 1},
        "volume_period": {"type": "int", "low": 15, "high": 30, "step": 1},
        "volume_surge": {"type": "float", "low": 1.2, "high": 2.0, "step": 0.1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.sr = SupportResistance(
            self.data, period=self.params.sr_period, tolerance=self.params.sr_tolerance
        )
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.volume_conf = VolumeConfirmation(
            self.data,
            period=self.params.volume_period,
            surge_multiplier=self.params.volume_surge,
        )

        # State variables
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.sr_period * 2,
                self.params.rsi_period,
                self.params.volume_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized SRRSIVolume with params: {self.params}")
        logger.info(
            f"SRRSIVolume initialized with sr_period={self.p.sr_period}, "
            f"rsi_period={self.p.rsi_period}, volume_period={self.p.volume_period}"
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
            or np.isnan(self.sr.support[0])
            or np.isnan(self.sr.resistance[0])
            or np.isnan(self.volume_conf.volume_ratio[0])
        ):
            logger.debug(f"Invalid indicator values at bar {len(self)}")
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "support": self.sr.support[0],
                "resistance": self.sr.resistance[0],
                "at_support": self.sr.at_support[0],
                "at_resistance": self.sr.at_resistance[0],
                "break_support": self.sr.break_support[0],
                "break_resistance": self.sr.break_resistance[0],
                "rsi": self.rsi[0],
                "volume": self.data.volume[0],
                "volume_ratio": self.volume_conf.volume_ratio[0],
                "volume_surge": self.volume_conf.volume_surge[0],
            }
        )

        # Trading Logic
        rsi_oversold_recovery = (
            self.rsi[0] > self.params.rsi_oversold
            and self.rsi[-1] <= self.params.rsi_oversold
        )
        rsi_overbought_decline = (
            self.rsi[0] < self.params.rsi_overbought
            and self.rsi[-1] >= self.params.rsi_overbought
        )

        if not self.position:
            # Long Entry: Price bounces off support + RSI oversold recovery + Volume surge
            if (
                self.sr.at_support[0]
                and rsi_oversold_recovery
                and self.volume_conf.volume_surge[0]
            ):
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - S/R + RSI + Volume) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"Support: {self.sr.support[0]:.2f} | RSI: {self.rsi[0]:.2f} | "
                    f"Volume Ratio: {self.volume_conf.volume_ratio[0]:.2f}"
                )

            # Short Entry: Price rejected at resistance + RSI overbought decline + Volume surge
            elif (
                self.sr.at_resistance[0]
                and rsi_overbought_decline
                and self.volume_conf.volume_surge[0]
            ):
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - S/R + RSI + Volume) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"Resistance: {self.sr.resistance[0]:.2f} | RSI: {self.rsi[0]:.2f} | "
                    f"Volume Ratio: {self.volume_conf.volume_ratio[0]:.2f}"
                )

        elif self.position.size > 0:  # Long position
            # Exit: Support break or RSI overbought
            if self.sr.break_support[0] or self.rsi[0] >= self.params.rsi_overbought:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "Support break" if self.sr.break_support[0] else "RSI overbought"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - {exit_reason}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f}"
                )

        elif self.position.size < 0:  # Short position
            # Exit: Resistance break or RSI oversold
            if self.sr.break_resistance[0] or self.rsi[0] <= self.params.rsi_oversold:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "Resistance break"
                    if self.sr.break_resistance[0]
                    else "RSI oversold"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - {exit_reason}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f}"
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
                    f"Price: {order.executed.price:.2f} | Size: {order.executed.size} | "
                    f"Cost: {order.executed.value:.2f} | Comm: {order.executed.comm:.2f}"
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
                    f"Price: {order.executed.price:.2f} | Size: {order.executed.size} | "
                    f"Cost: {order.executed.value:.2f} | Comm: {order.executed.comm:.2f}"
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
                        f"Price: {order.executed.price:.2f} | PnL: {pnl:.2f}"
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
                        f"Price: {order.executed.price:.2f} | PnL: {pnl:.2f}"
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
                f"Net Profit: {trade.pnlcomm:.2f} | Bars Held: {trade.barlen} | "
                f"Trade Count: {self.trade_count}"
            )

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        params = {
            "sr_period": trial.suggest_int("sr_period", 15, 30),
            "sr_tolerance": trial.suggest_float(
                "sr_tolerance", 0.001, 0.005, step=0.0005
            ),
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 35),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 65, 80),
            "volume_period": trial.suggest_int("volume_period", 15, 30),
            "volume_surge": trial.suggest_float("volume_surge", 1.2, 2.0, step=0.1),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            sr_period = params.get("sr_period", 20)
            rsi_period = params.get("rsi_period", 14)
            volume_period = params.get("volume_period", 20)
            return max(sr_period * 2, rsi_period, volume_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 50
