import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class BSAV(bt.Strategy):
    """
    Bollinger Bands + Stochastic + ADX + Volume (BSAV) Strategy
    Strategy Type: SQUEEZE + MOMENTUM + TREND + VOLUME
    ==========================================
    This strategy combines Bollinger Bands, Stochastic, ADX, and Volume for intraday trading on a 5-minute timeframe.

    Strategy Logic:
    ==============
    Long Entry: BB squeeze breakout (price above upper band) + Stochastic %K above %D + ADX rising + Volume above average
    Short Entry: BB squeeze breakout (price below lower band) + Stochastic %K below %D + ADX rising + Volume above average
    Exit: Trend weakens (ADX falling) or volume decreases significantly

    Parameters:
    ==========
    - bb_period (int): Bollinger Bands period (default: 20)
    - bb_stddev (float): Bollinger Bands standard deviation (default: 2.0)
    - stoch_period (int): Stochastic period (default: 14)
    - stoch_k (int): Stochastic %K smoothing (default: 3)
    - stoch_d (int): Stochastic %D smoothing (default: 3)
    - adx_period (int): ADX period (default: 14)
    - vol_sma_period (int): Volume SMA period (default: 14)
    - vol_threshold (float): Volume threshold multiplier (default: 1.5)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("bb_period", 20),
        ("bb_stddev", 2.0),
        ("stoch_period", 14),
        ("stoch_k", 3),
        ("stoch_d", 3),
        ("adx_period", 14),
        ("vol_sma_period", 14),
        ("vol_threshold", 1.5),
        ("verbose", False),
    )

    optimization_params = {
        "bb_period": {"type": "int", "low": 15, "high": 25, "step": 1},
        "bb_stddev": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.1},
        "stoch_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stoch_k": {"type": "int", "low": 2, "high": 5, "step": 1},
        "stoch_d": {"type": "int", "low": 2, "high": 5, "step": 1},
        "adx_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "vol_sma_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "vol_threshold": {"type": "float", "low": 1.2, "high": 2.0, "step": 0.1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.bb = btind.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_stddev,
        )
        self.stoch = btind.Stochastic(
            self.data,
            period=self.params.stoch_period,
            period_dfast=self.params.stoch_k,
            period_dslow=self.params.stoch_d,
        )
        self.adx = btind.ADX(self.data, period=self.params.adx_period)
        self.vol_sma = btind.SMA(self.data.volume, period=self.params.vol_sma_period)
        self.vol_ratio = self.data.volume / self.vol_sma
        self.bb_upper_touch = self.data.close >= self.bb.lines.top
        self.bb_lower_touch = self.data.close <= self.bb.lines.bot

        # Debug: Log available lines and their types
        logger.debug(f"Bollinger Bands lines: {self.bb.lines.getlinealiases()}")
        logger.debug(f"Stochastic lines: {self.stoch.lines.getlinealiases()}")
        logger.debug(f"ADX lines: {self.adx.lines.getlinealiases()}")
        logger.debug(f"Volume SMA lines: {self.vol_sma.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.bb_period,
                self.params.stoch_period,
                self.params.adx_period,
                self.params.vol_sma_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized BSAV with params: {self.params}")
        logger.info(
            f"BSAV initialized with bb_period={self.p.bb_period}, "
            f"bb_stddev={self.p.bb_stddev}, stoch_period={self.p.stoch_period}, "
            f"adx_period={self.p.adx_period}, vol_sma_period={self.p.vol_sma_period}, "
            f"vol_threshold={self.p.vol_threshold}"
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
            np.isnan(self.bb.lines.top[0])
            or np.isnan(self.bb.lines.bot[0])
            or np.isnan(self.stoch.percK[0])
            or np.isnan(self.stoch.percD[0])
            or np.isnan(self.adx[0])
            or np.isnan(self.vol_ratio[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"BB Top={self.bb.lines.top[0]}, BB Bot={self.bb.lines.bot[0]}, "
                f"Stoch %K={self.stoch.percK[0]}, Stoch %D={self.stoch.percD[0]}, "
                f"ADX={self.adx[0]}, Volume Ratio={self.vol_ratio[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "bb_top": self.bb.lines.top[0],
                "bb_mid": self.bb.lines.mid[0],
                "bb_bot": self.bb.lines.bot[0],
                "stoch_k": self.stoch.percK[0],
                "stoch_d": self.stoch.percD[0],
                "adx": self.adx[0],
                "vol_ratio": self.vol_ratio[0],
            }
        )

        # Trading Logic
        adx_rising = self.adx[0] > self.adx[-1] and self.adx[0] > 25
        stoch_bullish = (
            self.stoch.percK[0] > self.stoch.percD[0]
            and self.stoch.percK[-1] <= self.stoch.percD[-1]
        )
        stoch_bearish = (
            self.stoch.percK[0] < self.stoch.percD[0]
            and self.stoch.percK[-1] >= self.stoch.percD[-1]
        )
        high_volume = self.vol_ratio[0] > self.params.vol_threshold
        adx_falling = self.adx[0] < self.adx[-1]
        vol_decrease = self.vol_ratio[0] < 1.0

        if not self.position:
            # Long Entry: BB upper touch + Stochastic bullish + ADX rising + High volume
            if self.bb_upper_touch[0] and stoch_bullish and adx_rising and high_volume:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - BSAV) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"BB Top: {self.bb.lines.top[0]:.2f} (Touch) | "
                    f"Stoch %K: {self.stoch.percK[0]:.2f} (Bullish) | "
                    f"ADX: {self.adx[0]:.2f} (Rising) | "
                    f"Volume Ratio: {self.vol_ratio[0]:.2f} (High)"
                )
            # Short Entry: BB lower touch + Stochastic bearish + ADX rising + High volume
            elif (
                self.bb_lower_touch[0] and stoch_bearish and adx_rising and high_volume
            ):  # Fixed line
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - BSAV) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"BB Bot: {self.bb.lines.bot[0]:.2f} (Touch) | "
                    f"Stoch %K: {self.stoch.percK[0]:.2f} (Bearish) | "
                    f"ADX: {self.adx[0]:.2f} (Rising) | "
                    f"Volume Ratio: {self.vol_ratio[0]:.2f} (High)"
                )
        elif self.position.size > 0:  # Long position
            # Exit: ADX falling or volume decreases
            if adx_falling or vol_decrease:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = "ADX falling" if adx_falling else "Volume decreasing"
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - BSAV) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"ADX: {self.adx[0]:.2f} | "
                    f"Volume Ratio: {self.vol_ratio[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            # Exit: ADX falling or volume decreases
            if adx_falling or vol_decrease:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = "ADX falling" if adx_falling else "Volume decreasing"
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - BSAV) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"ADX: {self.adx[0]:.2f} | "
                    f"Volume Ratio: {self.vol_ratio[0]:.2f}"
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
                        "bars_held": (
                            exec_dt - entry_info["entry_time"]
                        ).total_seconds()
                        / 300,  # 5-min bars
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
                        "bars_held": (
                            exec_dt - entry_info["entry_time"]
                        ).total_seconds()
                        / 300,  # 5-min bars
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
            "bb_period": trial.suggest_int("bb_period", 15, 25),
            "bb_stddev": trial.suggest_float("bb_stddev", 1.5, 2.5, step=0.1),
            "stoch_period": trial.suggest_int("stoch_period", 10, 20),
            "stoch_k": trial.suggest_int("stoch_k", 2, 5),
            "stoch_d": trial.suggest_int("stoch_d", 2, 5),
            "adx_period": trial.suggest_int("adx_period", 10, 20),
            "vol_sma_period": trial.suggest_int("vol_sma_period", 10, 20),
            "vol_threshold": trial.suggest_float("vol_threshold", 1.2, 2.0, step=0.1),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            bb_period = params.get("bb_period", 20)
            stoch_period = params.get("stoch_period", 14)
            adx_period = params.get("adx_period", 14)
            vol_sma_period = params.get("vol_sma_period", 14)
            return max(bb_period, stoch_period, adx_period, vol_sma_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
