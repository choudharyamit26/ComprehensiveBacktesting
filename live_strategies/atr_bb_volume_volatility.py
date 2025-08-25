import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class VolumeVolatility(bt.Indicator):
    lines = ("vol_volatility",)
    params = (("period", 14),)

    def __init__(self):
        self.addminperiod(self.params.period)
        volume_sma = btind.SMA(self.data.volume, period=self.params.period)
        self.lines.vol_volatility = self.data.volume / volume_sma


class ATR_BB_VolumeVolatility(bt.Strategy):
    """
    ATR + Bollinger Bands + Volume Volatility Strategy
    Strategy Type: VOLATILITY + BAND + VOLUME
    ==========================================
    This strategy combines ATR, Bollinger Bands, and Volume Volatility for volatility-based trading.

    Strategy Logic:
    ==============
    Long Entry: ATR increasing + Price at/above upper BB + Volume volatility high
    Short Entry: ATR increasing + Price at/below lower BB + Volume volatility high
    Exit: Volatility contraction (ATR or Volume Volatility decreases) or price returns to BB middle

    Parameters:
    ==========
    - atr_period (int): ATR period (default: 14)
    - bb_period (int): Bollinger Bands period (default: 20)
    - bb_stddev (float): Bollinger Bands standard deviation (default: 2.0)
    - vol_vol_period (int): Volume Volatility period (default: 14)
    - vol_vol_threshold (float): Volume Volatility threshold (default: 1.5)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("atr_period", 14),
        ("bb_period", 20),
        ("bb_stddev", 2.0),
        ("vol_vol_period", 14),
        ("vol_vol_threshold", 1.5),
        ("verbose", False),
    )

    optimization_params = {
        "atr_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "bb_period": {"type": "int", "low": 15, "high": 25, "step": 1},
        "bb_stddev": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.1},
        "vol_vol_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "vol_vol_threshold": {"type": "float", "low": 1.2, "high": 2.0, "step": 0.1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.atr = btind.ATR(self.data, period=self.params.atr_period)
        self.bb = btind.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_stddev,
        )
        self.vol_vol = VolumeVolatility(self.data, period=self.params.vol_vol_period)
        self.bb_upper_touch = self.data.close >= self.bb.lines.top
        self.bb_lower_touch = self.data.close <= self.bb.lines.bot

        # Debug: Log available lines and their types
        logger.debug(f"ATR lines: {self.atr.lines.getlinealiases()}")
        logger.debug(f"BollingerBands lines: {self.bb.lines.getlinealiases()}")
        logger.debug(f"Volume Volatility lines: {self.vol_vol.lines.getlinealiases()}")

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.atr_period,
                self.params.bb_period,
                self.params.vol_vol_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized ATR_BB_VolumeVolatility with params: {self.params}")
        logger.info(
            f"ATR_BB_VolumeVolatility initialized with atr_period={self.p.atr_period}, "
            f"bb_period={self.p.bb_period}, bb_stddev={self.p.bb_stddev}, "
            f"vol_vol_period={self.p.vol_vol_period}, vol_vol_threshold={self.p.vol_vol_threshold}"
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
            np.isnan(self.atr[0])
            or np.isnan(self.bb.lines.top[0])
            or np.isnan(self.bb.lines.bot[0])
            or np.isnan(self.vol_vol[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"ATR={self.atr[0]}, BB Top={self.bb.lines.top[0]}, "
                f"BB Bottom={self.bb.lines.bot[0]}, Volume Volatility={self.vol_vol[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "atr": self.atr[0],
                "bb_top": self.bb.lines.top[0],
                "bb_mid": self.bb.lines.mid[0],
                "bb_bot": self.bb.lines.bot[0],
                "vol_vol": self.vol_vol[0],
            }
        )

        # Trading Logic
        atr_increasing = self.atr[0] > self.atr[-1]
        vol_vol_high = self.vol_vol[0] > self.params.vol_vol_threshold
        vol_vol_decreasing = self.vol_vol[0] < 1.0
        volatility_contraction = self.atr[0] < self.atr[-1]
        price_near_mid = (
            abs(self.data.close[0] - self.bb.lines.mid[0]) / self.bb.lines.mid[0] < 0.01
        )

        if not self.position:
            # Long Entry: ATR increasing + Price at/above upper BB + Volume volatility high
            if atr_increasing and self.bb_upper_touch[0] and vol_vol_high:
                self.order = self.buy()
                self.order_type = "enter_long"
                # trade_logger.info(
                #     f"BUY SIGNAL (Enter Long - ATR + BB + Volume Volatility) | Bar: {len(self)} | "
                #     f"Time: {bar_time_ist} | "
                #     f"Price: {self.data.close[0]:.2f} | "
                #     f"ATR: {self.atr[0]:.2f} (Increasing) | "
                #     f"BB Top: {self.bb.lines.top[0]:.2f} (Touch) | "
                #     f"Volume Volatility: {self.vol_vol[0]:.2f} (High)"
                # )
            # Short Entry: ATR increasing + Price at/below lower BB + Volume volatility high
            elif atr_increasing and self.bb_lower_touch[0] and vol_vol_high:
                self.order = self.sell()
                self.order_type = "enter_short"
                # trade_logger.info(
                #     f"SELL SIGNAL (Enter Short - ATR + BB + Volume Volatility) | Bar: {len(self)} | "
                #     f"Time: {bar_time_ist} | "
                #     f"Price: {self.data.close[0]:.2f} | "
                #     f"ATR: {self.atr[0]:.2f} (Increasing) | "
                #     f"BB Bottom: {self.bb.lines.bot[0]:.2f} (Touch) | "
                #     f"Volume Volatility: {self.vol_vol[0]:.2f} (High)"
                # )
        elif self.position.size > 0:  # Long position
            # Exit: Volatility contraction or price returns to BB middle
            if volatility_contraction or vol_vol_decreasing or price_near_mid:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "Volatility contraction"
                    if volatility_contraction
                    else (
                        "Volume volatility decreasing"
                        if vol_vol_decreasing
                        else "Price near BB middle"
                    )
                )
                # trade_logger.info(
                #     f"SELL SIGNAL (Exit Long - ATR + BB + Volume Volatility) | Bar: {len(self)} | "
                #     f"Time: {bar_time_ist} | "
                #     f"Price: {self.data.close[0]:.2f} | "
                #     f"Reason: {exit_reason} | "
                #     f"ATR: {self.atr[0]:.2f} | "
                #     f"BB Mid: {self.bb.lines.mid[0]:.2f} | "
                #     f"Volume Volatility: {self.vol_vol[0]:.2f}"
                # )
        elif self.position.size < 0:  # Short position
            # Exit: Volatility contraction or price returns to BB middle
            if volatility_contraction or vol_vol_decreasing or price_near_mid:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "Volatility contraction"
                    if volatility_contraction
                    else (
                        "Volume volatility decreasing"
                        if vol_vol_decreasing
                        else "Price near BB middle"
                    )
                )
                # trade_logger.info(
                #     f"BUY SIGNAL (Exit Short - ATR + BB + Volume Volatility) | Bar: {len(self)} | "
                #     f"Time: {bar_time_ist} | "
                #     f"Price: {self.data.close[0]:.2f} | "
                #     f"Reason: {exit_reason} | "
                #     f"ATR: {self.atr[0]:.2f} | "
                #     f"BB Mid: {self.bb.lines.mid[0]:.2f} | "
                #     f"Volume Volatility: {self.vol_vol[0]:.2f}"
                # )

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
                # trade_logger.info(
                #     f"BUY EXECUTED (Enter Long) | Ref: {order.ref} | "
                #     f"Price: {order.executed.price:.2f} | "
                #     f"Size: {order.executed.size} | "
                #     f"Cost: {order.executed.value:.2f} | "
                #     f"Comm: {order.executed.comm:.2f}"
                # )
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
                # trade_logger.info(
                #     f"SELL EXECUTED (Enter Short) | Ref: {order.ref} | "
                #     f"Price: {order.executed.price:.2f} | "
                #     f"Size: {order.executed.size} | "
                #     f"Cost: {order.executed.value:.2f} | "
                #     f"Comm: {order.executed.comm:.2f}"
                # )
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
                    # trade_logger.info(
                    #     f"SELL EXECUTED (Exit Long) | Ref: {order.ref} | "
                    #     f"Price: {order.executed.price:.2f} | "
                    #     f"Size: {order.executed.size} | "
                    #     f"Cost: {order.executed.value:.2f} | "
                    #     f"Comm: {order.executed.comm:.2f} | "
                    #     f"PnL: {pnl:.2f}"
                    # )
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
                    # trade_logger.info(
                    #     f"BUY EXECUTED (Exit Short) | Ref: {order.ref} | "
                    #     f"Price: {order.executed.price:.2f} | "
                    #     f"Size: {order.executed.size} | "
                    #     f"Cost: {order.executed.value:.2f} | "
                    #     f"Comm: {order.executed.comm:.2f} | "
                    #     f"PnL: {pnl:.2f}"
                    # )

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
            "atr_period": trial.suggest_int("atr_period", 10, 20),
            "bb_period": trial.suggest_int("bb_period", 15, 25),
            "bb_stddev": trial.suggest_float("bb_stddev", 1.5, 2.5, step=0.1),
            "vol_vol_period": trial.suggest_int("vol_vol_period", 10, 20),
            "vol_vol_threshold": trial.suggest_float(
                "vol_vol_threshold", 1.2, 2.0, step=0.1
            ),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            atr_period = params.get("atr_period", 14)
            bb_period = params.get("bb_period", 20)
            vol_vol_period = params.get("vol_vol_period", 14)
            return max(atr_period, bb_period, vol_vol_period) + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
