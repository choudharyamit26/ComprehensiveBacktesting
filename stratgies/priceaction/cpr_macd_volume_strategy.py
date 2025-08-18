import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class CPR(bt.Indicator):
    """
    Central Pivot Range (CPR) Indicator
    Calculates Pivot, BC (Bottom Central), and TC (Top Central) levels
    """

    lines = ("pivot", "bc", "tc")
    params = (("lookback", 1),)  # Days to look back

    def __init__(self):
        self.prev_high = 0.0
        self.prev_low = 0.0
        self.prev_close = 0.0

    def next(self):
        if len(self) < 2:
            self.lines.pivot[0] = float("nan")
            self.lines.bc[0] = float("nan")
            self.lines.tc[0] = float("nan")
            return

        # Get previous day's OHLC (simplified - using previous session's range)
        lookback_period = min(len(self), 288)  # Approx 1 day of 5-min bars

        if lookback_period > 1:
            # Calculate previous session's high, low, close
            try:
                self.prev_high = max(
                    [
                        self.data.high[-i]
                        for i in range(1, min(lookback_period, len(self)))
                    ]
                )
                self.prev_low = min(
                    [
                        self.data.low[-i]
                        for i in range(1, min(lookback_period, len(self)))
                    ]
                )
                self.prev_close = self.data.close[-1]

                # CPR Calculations
                pivot = (self.prev_high + self.prev_low + self.prev_close) / 3
                bc = (self.prev_high + self.prev_low) / 2
                tc = pivot + (pivot - bc)

                self.lines.pivot[0] = pivot
                self.lines.bc[0] = bc
                self.lines.tc[0] = tc
            except (IndexError, ValueError):
                self.lines.pivot[0] = float("nan")
                self.lines.bc[0] = float("nan")
                self.lines.tc[0] = float("nan")
        else:
            self.lines.pivot[0] = float("nan")
            self.lines.bc[0] = float("nan")
            self.lines.tc[0] = float("nan")


class VolumeSpike(bt.Indicator):
    """
    Volume Spike Indicator
    Detects when current volume exceeds average volume by threshold
    """

    lines = ("volume_spike",)
    params = (("period", 10), ("threshold", 1.5))

    def __init__(self):
        self.volume_ma = btind.SMA(self.data.volume, period=self.p.period)

    def next(self):
        if len(self) < self.p.period:
            self.lines.volume_spike[0] = 0
            return

        try:
            current_volume = self.data.volume[0]
            avg_volume = self.volume_ma[0]

            if current_volume > (avg_volume * self.p.threshold):
                self.lines.volume_spike[0] = 1
            else:
                self.lines.volume_spike[0] = 0
        except (ZeroDivisionError, TypeError):
            self.lines.volume_spike[0] = 0


class CPRMACDVolumeIntraday(bt.Strategy):
    """
    CPR + MACD + Volume Intraday Strategy
    Strategy Type: BREAKOUT/MOMENTUM
    =================================
    This strategy uses Central Pivot Range (CPR) breakouts confirmed by MACD momentum
    and volume spikes on a 5-minute timeframe for intraday trading.

    Strategy Logic:
    ==============
    Long Entry: Price breaks and closes above TC (Top Central) + MACD histogram positive + Volume spike
    Short Entry: Price breaks and closes below BC (Bottom Central) + MACD histogram negative + Volume spike
    Long Exit: Target hit (2x risk) OR MACD histogram turns negative
    Short Exit: Target hit (2x risk) OR MACD histogram turns positive

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST
    - Stop loss at 0.4% from entry
    - Target at 2x Risk-Reward ratio
    - Volume confirmation required for entries

    Parameters:
    ==========
    - macd_fast (int): MACD fast EMA period (default: 12)
    - macd_slow (int): MACD slow EMA period (default: 26)
    - macd_signal (int): MACD signal line period (default: 9)
    - volume_period (int): Volume MA period (default: 10)
    - volume_threshold (float): Volume spike threshold (default: 1.5)
    - stop_percent (float): Stop loss percentage (default: 0.4)
    - risk_reward_ratio (float): Risk-reward ratio (default: 2.0)
    - min_cpr_width (float): Minimum CPR width filter (default: 0.1%)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("volume_period", 10),
        ("volume_threshold", 1.5),
        ("stop_percent", 0.4),
        ("risk_reward_ratio", 2.0),
        ("min_cpr_width", 0.1),
        ("verbose", False),
    )

    optimization_params = {
        "macd_fast": {"type": "int", "low": 10, "high": 15, "step": 1},
        "macd_slow": {"type": "int", "low": 24, "high": 30, "step": 2},
        "volume_threshold": {"type": "float", "low": 1.2, "high": 2.0, "step": 0.1},
        "stop_percent": {"type": "float", "low": 0.3, "high": 0.6, "step": 0.1},
        "risk_reward_ratio": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.5},
        "min_cpr_width": {"type": "float", "low": 0.05, "high": 0.2, "step": 0.05},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.cpr = CPR(self.data)
        self.macd = btind.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal,
        )
        self.volume_spike = VolumeSpike(
            self.data,
            period=self.params.volume_period,
            threshold=self.params.volume_threshold,
        )

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = max(self.params.macd_slow, self.params.volume_period) + 50
        self.completed_trades = []
        self.open_positions = []
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.prev_close = None
        self.breakout_confirmed = False

        logger.info(f"Initialized CPRMACDVolumeIntraday with params: {self.params}")

    def is_valid_cpr_width(self):
        """Check if CPR width meets minimum threshold for volatility"""
        try:
            if np.isnan(self.cpr.tc[0]) or np.isnan(self.cpr.bc[0]):
                return False

            cpr_width = abs(self.cpr.tc[0] - self.cpr.bc[0])
            price = self.data.close[0]
            width_percent = (cpr_width / price) * 100

            return width_percent >= self.params.min_cpr_width
        except (ZeroDivisionError, TypeError):
            return False

    def is_market_time(self, current_time):
        """Check if current time is within market hours"""
        return datetime.time(9, 15) <= current_time <= datetime.time(15, 5)

    def should_force_close(self, current_time):
        """Check if positions should be force closed"""
        return current_time >= datetime.time(15, 15)

    def next(self):
        if len(self) < self.warmup_period:
            return

        if not self.ready:
            self.ready = True
            if self.params.verbose:
                logger.info(f"Strategy ready at bar {len(self)}")

        try:
            bar_time = self.datas[0].datetime.datetime(0)
            bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
            current_time = bar_time_ist.time()
        except:
            # Fallback if timezone conversion fails
            current_time = datetime.time(10, 0)  # Default to market hours

        # Force close at 3:15 PM IST
        if self.should_force_close(current_time):
            if self.position:
                self.close()
                if self.params.verbose:
                    trade_logger.info("Force closed all positions at 15:15 IST")
            return

        # Trading hours: 9:15 AM - 3:05 PM IST
        if not self.is_market_time(current_time):
            return

        if self.order:
            return

        # Check for NaN values
        if (
            np.isnan(self.macd.macd[0])
            or np.isnan(self.macd.signal[0])
            or np.isnan(self.cpr.tc[0])
            or np.isnan(self.cpr.bc[0])
        ):
            return

        current_price = self.data.close[0]
        macd_histogram = self.macd.macd[0] - self.macd.signal[0]
        volume_spike_active = self.volume_spike[0] == 1

        # Store previous close for breakout confirmation
        if self.prev_close is None and len(self) > 1:
            self.prev_close = self.data.close[-1]

        if not self.position:
            # Check CPR width validity
            if not self.is_valid_cpr_width():
                return

            # Long Entry: Break above TC
            tc_breakout = (
                self.prev_close is not None
                and self.prev_close <= self.cpr.tc[0]
                and current_price > self.cpr.tc[0]
            )

            long_conditions = (
                tc_breakout
                and macd_histogram > 0  # MACD histogram positive
                and volume_spike_active  # Volume confirmation
            )

            # Short Entry: Break below BC
            bc_breakdown = (
                self.prev_close is not None
                and self.prev_close >= self.cpr.bc[0]
                and current_price < self.cpr.bc[0]
            )

            short_conditions = (
                bc_breakdown
                and macd_histogram < 0  # MACD histogram negative
                and volume_spike_active  # Volume confirmation
            )

            if long_conditions:
                self.order = self.buy()
                self.order_type = "enter_long"
                self.stop_price = current_price * (1 - self.params.stop_percent / 100)
                self.target_price = current_price * (
                    1 + (self.params.stop_percent * self.params.risk_reward_ratio) / 100
                )
                if self.params.verbose:
                    trade_logger.info(
                        f"BUY SIGNAL | Time: {bar_time_ist} | Price: {current_price:.2f} | "
                        f"TC: {self.cpr.tc[0]:.2f} | Stop: {self.stop_price:.2f} | Target: {self.target_price:.2f}"
                    )

            elif short_conditions:
                self.order = self.sell()
                self.order_type = "enter_short"
                self.stop_price = current_price * (1 + self.params.stop_percent / 100)
                self.target_price = current_price * (
                    1 - (self.params.stop_percent * self.params.risk_reward_ratio) / 100
                )
                if self.params.verbose:
                    trade_logger.info(
                        f"SELL SIGNAL | Time: {bar_time_ist} | Price: {current_price:.2f} | "
                        f"BC: {self.cpr.bc[0]:.2f} | Stop: {self.stop_price:.2f} | Target: {self.target_price:.2f}"
                    )
        else:
            # Exit Logic
            if self.position.size > 0:  # Long position
                if (
                    current_price >= self.target_price
                    or current_price <= self.stop_price
                    or macd_histogram <= 0
                ):  # MACD turns negative
                    self.order = self.sell()
                    self.order_type = "exit_long"
                    exit_reason = (
                        "Target"
                        if current_price >= self.target_price
                        else "Stop" if current_price <= self.stop_price else "MACD_NEG"
                    )
                    if self.params.verbose:
                        trade_logger.info(
                            f"SELL SIGNAL (Exit Long - {exit_reason}) | Time: {bar_time_ist} | Price: {current_price:.2f}"
                        )

            elif self.position.size < 0:  # Short position
                if (
                    current_price <= self.target_price
                    or current_price >= self.stop_price
                    or macd_histogram >= 0
                ):  # MACD turns positive
                    self.order = self.buy()
                    self.order_type = "exit_short"
                    exit_reason = (
                        "Target"
                        if current_price <= self.target_price
                        else "Stop" if current_price >= self.stop_price else "MACD_POS"
                    )
                    if self.params.verbose:
                        trade_logger.info(
                            f"BUY SIGNAL (Exit Short - {exit_reason}) | Time: {bar_time_ist} | Price: {current_price:.2f}"
                        )

        # Update previous close for next bar
        self.prev_close = current_price

    def notify_order(self, order):
        if order.status in [order.Completed]:
            try:
                exec_dt = bt.num2date(order.executed.dt).astimezone(
                    pytz.timezone("Asia/Kolkata")
                )
            except:
                exec_dt = datetime.datetime.now()

            if self.order_type == "enter_long" and order.isbuy():
                self.entry_price = order.executed.price
                self.open_positions.append(
                    {
                        "entry_time": exec_dt,
                        "entry_price": order.executed.price,
                        "size": order.executed.size,
                        "commission": order.executed.comm,
                        "ref": order.ref,
                        "direction": "long",
                        "stop_price": self.stop_price,
                        "target_price": self.target_price,
                    }
                )
                if self.params.verbose:
                    trade_logger.info(
                        f"BUY EXECUTED (Enter Long) | Ref: {order.ref} | Price: {order.executed.price:.2f}"
                    )

            elif self.order_type == "enter_short" and order.issell():
                self.entry_price = order.executed.price
                self.open_positions.append(
                    {
                        "entry_time": exec_dt,
                        "entry_price": order.executed.price,
                        "size": order.executed.size,
                        "commission": order.executed.comm,
                        "ref": order.ref,
                        "direction": "short",
                        "stop_price": self.stop_price,
                        "target_price": self.target_price,
                    }
                )
                if self.params.verbose:
                    trade_logger.info(
                        f"SELL EXECUTED (Enter Short) | Ref: {order.ref} | Price: {order.executed.price:.2f}"
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
                    if self.params.verbose:
                        trade_logger.info(
                            f"SELL EXECUTED (Exit Long) | Ref: {order.ref} | PnL: {pnl:.2f}"
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
                    if self.params.verbose:
                        trade_logger.info(
                            f"BUY EXECUTED (Exit Short) | Ref: {order.ref} | PnL: {pnl:.2f}"
                        )

        if order.status in [
            order.Completed,
            order.Canceled,
            order.Margin,
            order.Rejected,
        ]:
            if order.status in [order.Completed] and self.order_type in [
                "exit_long",
                "exit_short",
            ]:
                self.entry_price = None
                self.stop_price = None
                self.target_price = None
            self.order = None
            self.order_type = None

    def notify_trade(self, trade):
        if trade.isclosed and self.params.verbose:
            trade_logger.info(
                f"TRADE CLOSED | Ref: {trade.ref} | Profit: {trade.pnl:.2f} | Net Profit: {trade.pnlcomm:.2f}"
            )

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        return {
            "macd_fast": trial.suggest_int("macd_fast", 10, 15),
            "macd_slow": trial.suggest_int("macd_slow", 24, 30, step=2),
            "volume_threshold": trial.suggest_float(
                "volume_threshold", 1.2, 2.0, step=0.1
            ),
            "stop_percent": trial.suggest_float("stop_percent", 0.3, 0.6, step=0.1),
            "risk_reward_ratio": trial.suggest_float(
                "risk_reward_ratio", 1.5, 3.0, step=0.5
            ),
            "min_cpr_width": trial.suggest_float("min_cpr_width", 0.05, 0.2, step=0.05),
        }

    @classmethod
    def get_min_data_points(cls, params):
        try:
            return (
                max(params.get("macd_slow", 26), params.get("volume_period", 10)) + 50
            )
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 76
