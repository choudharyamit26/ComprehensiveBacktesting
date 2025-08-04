import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class DailyVWAP(bt.Indicator):
    """
    Daily VWAP Indicator that resets at the start of each trading day
    """

    lines = ("vwap",)
    params = (("timeframe", bt.TimeFrame.Days),)

    def __init__(self):
        self.volumes = []
        self.volumesums = []
        self.typical = []  # Typical Price (H+L+C)/3
        self.current_date = None

    def next(self):
        current_date = self.data.datetime.date(0)

        # Reset at start of new trading day
        if self.current_date != current_date:
            self.volumes = []
            self.volumesums = []
            self.typical = []
            self.current_date = current_date

        # Calculate typical price
        typical_price = (
            self.data.high[0] + self.data.low[0] + self.data.close[0]
        ) / 3.0
        volume = self.data.volume[0]

        # Update cumulative values
        if len(self.volumes) == 0:
            cum_vol = volume
            cum_val = typical_price * volume
        else:
            cum_vol = self.volumesums[-1] + volume
            cum_val = self.typical[-1] * self.volumesums[-1] + typical_price * volume

        self.volumes.append(volume)
        self.volumesums.append(cum_vol)
        self.typical.append(cum_val / cum_vol if cum_vol != 0 else typical_price)

        # Set current VWAP value
        self.lines.vwap[0] = self.typical[-1]


class OnBalanceVolume(bt.Indicator):
    """
    Custom On Balance Volume (OBV) Indicator
    """

    lines = ("obv",)

    def __init__(self):
        self.addminperiod(1)
        self.lines.obv = bt.If(
            self.data.close > self.data.close(-1), self.data.volume, -self.data.volume
        )
        self.lines.obv = bt.indicators.CumSum(self.lines.obv)


class VWAPBounceRejection(bt.Strategy):
    """
    Merged VWAP Bounce/Rejection Strategy
    Strategy Type: MEAN REVERSION + PRICE ACTION
    =======================================
    This strategy combines the best features from both implementations:
    - Uses true daily VWAP for accurate intraday reference
    - Incorporates session open price logic for trend context
    - Uses ATR-based dynamic stops and profit targets for better risk management
    - Clean trade extraction without issues
    Strategy Logic:
    ==============
    Long Entry: Price pulls back to VWAP after opening above it + bullish candlestick pattern + rising volume + OBV confirmation
    Short Entry: Price rallies to VWAP after opening below it + bearish candlestick pattern + volume spike + OBV confirmation
    Exit: ATR-based dynamic stop-loss or profit target or significant move away from VWAP
    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST
    - Uses warmup period for indicator stability
    - Prevents order overlap
    - ATR-based dynamic stop-loss and profit target
    Parameters:
    ==========
    - atr_period (int): ATR calculation period (default: 14)
    - volume_lookback (int): Lookback period for volume analysis (default: 5)
    - stop_loss_atr_mult (float): ATR multiplier for stop-loss (default: 1.5)
    - vwap_proximity_mult (float): ATR multiplier for VWAP proximity (default: 0.5)
    - profit_target_mult (float): ATR multiplier for profit target (default: 2.0)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("atr_period", 14),
        ("volume_lookback", 5),
        ("stop_loss_atr_mult", 1.5),
        ("vwap_proximity_mult", 0.5),
        ("profit_target_mult", 2.0),
        ("verbose", False),
    )
    optimization_params = {
        "atr_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "volume_lookback": {"type": "int", "low": 3, "high": 7, "step": 1},
        "stop_loss_atr_mult": {"type": "float", "low": 1.0, "high": 2.5, "step": 0.25},
        "vwap_proximity_mult": {"type": "float", "low": 0.3, "high": 0.8, "step": 0.1},
        "profit_target_mult": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.25},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.daily_vwap = DailyVWAP(self.data)
        self.atr = btind.ATR(self.data, period=self.params.atr_period)
        self.volume_avg = btind.SimpleMovingAverage(
            self.data.volume, period=self.params.volume_lookback
        )
        self.lowest = btind.Lowest(self.data.low, period=5)
        self.highest = btind.Highest(self.data.high, period=5)
        self.obv = OnBalanceVolume(self.data)  # Use custom OBV indicator
        # Initialize trading variables
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(self.params.atr_period, self.params.volume_lookback) + 5
        )
        self.completed_trades = []
        self.open_positions = []
        self.entry_price = None
        self.stop_loss_price = None
        self.profit_target_price = None  # New profit target price
        self.session_open_price = None
        self.current_date = None
        logger.info(f"Initialized MergedVWAPBounceRejection with params: {self.params}")

    def next(self):
        if len(self) < self.warmup_period:
            return
        if not self.ready:
            self.ready = True
            logger.info(f"Strategy ready at bar {len(self)}")
        bar_time = self.datas[0].datetime.datetime(0)
        bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
        current_time = bar_time_ist.time()
        current_date = self.data.datetime.date(0)
        # Reset session open price for new trading day
        if self.current_date != current_date:
            self.session_open_price = None
            self.current_date = current_date
        # Set session open price at market open (9:15 IST)
        if current_time >= datetime.time(9, 15) and self.session_open_price is None:
            self.session_open_price = self.data.open[0]
            if self.params.verbose:
                logger.info(f"Session open set: {self.session_open_price}")
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
            return
        # Check for invalid indicator values
        if (
            np.isnan(self.daily_vwap[0])
            or np.isnan(self.atr[0])
            or np.isnan(self.volume_avg[0])
            or self.session_open_price is None
            or self.daily_vwap[0] == 0
            or self.atr[0] == 0
        ):
            return
        # Enhanced candlestick pattern detection
        bullish_pattern = self._detect_bullish_pattern()
        bearish_pattern = self._detect_bearish_pattern()
        # Volume analysis (combining both approaches)
        volume_rising = self.data.volume[0] > self.volume_avg[0]
        volume_spike = self.data.volume[0] > 1.5 * self.volume_avg[0]
        # Price proximity to VWAP using ATR
        price_near_vwap = (
            abs(self.data.close[0] - self.daily_vwap[0])
            < self.atr[0] * self.params.vwap_proximity_mult
        )
        # Trading Logic
        if not self.position:
            # Long Entry: Session opened above VWAP, price pulled back to VWAP, bullish pattern, rising volume, OBV increasing
            if (
                self.session_open_price > self.daily_vwap[0]
                and price_near_vwap
                and self.data.low[0] <= self.daily_vwap[0]  # Price touched/crossed VWAP
                and bullish_pattern
                and volume_rising
                and self.obv[0] > self.obv[-1]  # OBV confirmation
            ):
                # Calculate dynamic stop-loss and profit target
                self.stop_loss_price = (
                    self.data.close[0] - self.atr[0] * self.params.stop_loss_atr_mult
                )
                self.profit_target_price = (
                    self.data.close[0] + self.atr[0] * self.params.profit_target_mult
                )

                # Risk validation
                risk = self.data.close[0] - self.stop_loss_price
                if risk > 0 and risk < self.atr[0] * 3:  # Reasonable risk
                    self.order = self.buy()
                    self.order_type = "enter_long"
                    trade_logger.info(
                        f"BUY SIGNAL | Time: {bar_time_ist} | "
                        f"Price: {self.data.close[0]:.2f} | "
                        f"VWAP: {self.daily_vwap[0]:.2f} | "
                        f"Stop: {self.stop_loss_price:.2f} | "
                        f"Profit Target: {self.profit_target_price:.2f}"
                    )
            # Short Entry: Session opened below VWAP, price rallied to VWAP, bearish pattern, volume spike, OBV decreasing
            elif (
                self.session_open_price < self.daily_vwap[0]
                and price_near_vwap
                and self.data.high[0]
                >= self.daily_vwap[0]  # Price touched/crossed VWAP
                and bearish_pattern
                and volume_spike
                and self.obv[0] < self.obv[-1]  # OBV confirmation
            ):
                # Calculate dynamic stop-loss and profit target
                self.stop_loss_price = (
                    self.data.close[0] + self.atr[0] * self.params.stop_loss_atr_mult
                )
                self.profit_target_price = (
                    self.data.close[0] - self.atr[0] * self.params.profit_target_mult
                )

                # Risk validation
                risk = self.stop_loss_price - self.data.close[0]
                if risk > 0 and risk < self.atr[0] * 3:  # Reasonable risk
                    self.order = self.sell()
                    self.order_type = "enter_short"
                    trade_logger.info(
                        f"SELL SIGNAL | Time: {bar_time_ist} | "
                        f"Price: {self.data.close[0]:.2f} | "
                        f"VWAP: {self.daily_vwap[0]:.2f} | "
                        f"Stop: {self.stop_loss_price:.2f} | "
                        f"Profit Target: {self.profit_target_price:.2f}"
                    )
        else:
            # Exit Logic
            if self.position.size > 0:  # Long position
                # Stop-loss or profit target or significant move away from VWAP
                if (
                    self.data.close[0] <= self.stop_loss_price
                    or self.data.close[0] >= self.profit_target_price
                    or self.data.close[0] > self.daily_vwap[0] + self.atr[0] * 2
                ):
                    self.order = self.sell()
                    self.order_type = "exit_long"
                    trade_logger.info(
                        f"SELL SIGNAL (Exit Long) | Time: {bar_time_ist} | "
                        f"Price: {self.data.close[0]:.2f}"
                    )

            elif self.position.size < 0:  # Short position
                # Stop-loss or profit target or significant move away from VWAP
                if (
                    self.data.close[0] >= self.stop_loss_price
                    or self.data.close[0] <= self.profit_target_price
                    or self.data.close[0] < self.daily_vwap[0] - self.atr[0] * 2
                ):
                    self.order = self.buy()
                    self.order_type = "exit_short"
                    trade_logger.info(
                        f"BUY SIGNAL (Exit Short) | Time: {bar_time_ist} | "
                        f"Price: {self.data.close[0]:.2f}"
                    )

    def _detect_bullish_pattern(self):
        """Enhanced bullish pattern detection"""
        if len(self) < 2:
            return False

        # Hammer pattern
        body = abs(self.data.close[0] - self.data.open[0])
        lower_shadow = min(self.data.close[0], self.data.open[0]) - self.data.low[0]
        upper_shadow = self.data.high[0] - max(self.data.close[0], self.data.open[0])

        hammer = (
            self.data.close[0] > self.data.open[0]
            and lower_shadow > 2 * body
            and upper_shadow < body
        )

        # Bullish Engulfing
        bullish_engulfing = (
            len(self) >= 2
            and self.data.close[-1] < self.data.open[-1]
            and self.data.close[0] > self.data.open[0]
            and self.data.close[0] > self.data.open[-1]
            and self.data.open[0] < self.data.close[-1]
        )

        # Doji at support (simplified)
        doji = body < (self.data.high[0] - self.data.low[0]) * 0.1

        return hammer or bullish_engulfing or doji

    def _detect_bearish_pattern(self):
        """Enhanced bearish pattern detection"""
        if len(self) < 2:
            return False

        # Shooting Star pattern
        body = abs(self.data.close[0] - self.data.open[0])
        upper_shadow = self.data.high[0] - max(self.data.close[0], self.data.open[0])
        lower_shadow = min(self.data.close[0], self.data.open[0]) - self.data.low[0]

        shooting_star = (
            self.data.close[0] < self.data.open[0]
            and upper_shadow > 2 * body
            and lower_shadow < body
        )

        # Bearish Engulfing
        bearish_engulfing = (
            len(self) >= 2
            and self.data.close[-1] > self.data.open[-1]
            and self.data.close[0] < self.data.open[0]
            and self.data.close[0] < self.data.open[-1]
            and self.data.open[0] > self.data.close[-1]
        )

        # Doji at resistance (simplified)
        doji = body < (self.data.high[0] - self.data.low[0]) * 0.1

        return shooting_star or bearish_engulfing or doji

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt).astimezone(
                pytz.timezone("Asia/Kolkata")
            )

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
                        "stop_loss": self.stop_loss_price,
                        "profit_target": self.profit_target_price,  # Store profit target
                    }
                )
                trade_logger.info(
                    f"BUY EXECUTED (Enter Long) | Ref: {order.ref} | "
                    f"Price: {order.executed.price:.2f} | "
                    f"Stop: {self.stop_loss_price:.2f} | "
                    f"Profit Target: {self.profit_target_price:.2f}"
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
                        "stop_loss": self.stop_loss_price,
                        "profit_target": self.profit_target_price,  # Store profit target
                    }
                )
                trade_logger.info(
                    f"SELL EXECUTED (Enter Short) | Ref: {order.ref} | "
                    f"Price: {order.executed.price:.2f} | "
                    f"Stop: {self.stop_loss_price:.2f} | "
                    f"Profit Target: {self.profit_target_price:.2f}"
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
                    trade_logger.info(
                        f"SELL EXECUTED (Exit Long) | Ref: {order.ref} | "
                        f"PnL: {pnl:.2f} | Net PnL: {pnl - total_commission:.2f}"
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
                    trade_logger.info(
                        f"BUY EXECUTED (Exit Short) | Ref: {order.ref} | "
                        f"PnL: {pnl:.2f} | Net PnL: {pnl - total_commission:.2f}"
                    )
        # Clean up order tracking
        if order.status in [
            order.Completed,
            order.Canceled,
            order.Margin,
            order.Rejected,
        ]:
            if order.status == order.Completed and self.order_type in [
                "exit_long",
                "exit_short",
            ]:
                self.entry_price = None
                self.stop_loss_price = None
                self.profit_target_price = None  # Reset profit target
            self.order = None
            self.order_type = None

    def notify_trade(self, trade):
        if trade.isclosed:
            trade_logger.info(
                f"TRADE CLOSED | Ref: {trade.ref} | "
                f"Profit: {trade.pnl:.2f} | Net Profit: {trade.pnlcomm:.2f}"
            )

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        return {
            "atr_period": trial.suggest_int("atr_period", 10, 20),
            "volume_lookback": trial.suggest_int("volume_lookback", 3, 7),
            "stop_loss_atr_mult": trial.suggest_float(
                "stop_loss_atr_mult", 1.0, 2.5, step=0.25
            ),
            "vwap_proximity_mult": trial.suggest_float(
                "vwap_proximity_mult", 0.3, 0.8, step=0.1
            ),
            "profit_target_mult": trial.suggest_float(
                "profit_target_mult", 1.5, 3.0, step=0.25
            ),
        }

    @classmethod
    def get_min_data_points(cls, params):
        try:
            return (
                max(params.get("atr_period", 14), params.get("volume_lookback", 5)) + 5
            )
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 25
