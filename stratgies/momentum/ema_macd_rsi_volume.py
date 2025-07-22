import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class VolumeOscillator(bt.Indicator):
    """
    Volume Oscillator Indicator
    Measures the difference between short-term and long-term volume moving averages
    """

    lines = ("vol_osc", "signal")
    params = (
        ("short_period", 5),
        ("long_period", 14),
        ("signal_period", 9),
        ("threshold", 0.0),  # Threshold for significant volume activity
    )

    def __init__(self):
        self.short_vol = btind.EMA(self.data.volume, period=self.params.short_period)
        self.long_vol = btind.EMA(self.data.volume, period=self.params.long_period)
        self.lines.vol_osc = self.short_vol - self.long_vol
        self.lines.signal = btind.EMA(
            self.lines.vol_osc, period=self.params.signal_period
        )
        self.addminperiod(max(self.params.long_period, self.params.signal_period))

    def next(self):
        pass  # Calculation handled in __init__


class EMAMACDRSIVolume(bt.Strategy):
    """
    EMA + MACD + RSI + Volume Oscillator Strategy
    Strategy Type: TREND-FOLLOWING + MOMENTUM + VOLUME
    =============================================
    This strategy combines multiple EMAs, MACD, RSI, and Volume Oscillator for intraday trading.

    Strategy Logic:
    ==============
    Long Entry: Price above all EMAs + MACD bullish crossover + RSI above oversold + Volume Oscillator above threshold
    Short Entry: Price below all EMAs + MACD bearish crossover + RSI below overbought + Volume Oscillator above threshold
    Exit: Opposite EMA alignment or MACD signal reversal or RSI reaching extreme levels

    Parameters:
    ==========
    - ema1_period (int): First EMA period (default: 5)
    - ema2_period (int): Second EMA period (default: 9)
    - ema3_period (int): Third EMA period (default: 13)
    - ema4_period (int): Fourth EMA period (default: 21)
    - macd_fast (int): MACD fast EMA period (default: 12)
    - macd_slow (int): MACD slow EMA period (default: 26)
    - macd_signal (int): MACD signal period (default: 9)
    - rsi_period (int): RSI period (default: 14)
    - rsi_oversold (int): RSI oversold level (default: 30)
    - rsi_overbought (int): RSI overbought level (default: 70)
    - vol_short_period (int): Volume Oscillator short period (default: 5)
    - vol_long_period (int): Volume Oscillator long period (default: 14)
    - vol_signal_period (int): Volume Oscillator signal period (default: 9)
    - vol_threshold (float): Volume Oscillator threshold (default: 0.0)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("ema1_period", 5),
        ("ema2_period", 9),
        ("ema3_period", 13),
        ("ema4_period", 21),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
        ("vol_short_period", 5),
        ("vol_long_period", 14),
        ("vol_signal_period", 9),
        ("vol_threshold", 0.0),
        ("verbose", False),
    )

    optimization_params = {
        "ema1_period": {"type": "int", "low": 3, "high": 8, "step": 1},
        "ema2_period": {"type": "int", "low": 7, "high": 12, "step": 1},
        "ema3_period": {"type": "int", "low": 10, "high": 16, "step": 1},
        "ema4_period": {"type": "int", "low": 18, "high": 25, "step": 1},
        "macd_fast": {"type": "int", "low": 10, "high": 15, "step": 1},
        "macd_slow": {"type": "int", "low": 20, "high": 30, "step": 1},
        "macd_signal": {"type": "int", "low": 7, "high": 12, "step": 1},
        "rsi_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "rsi_oversold": {"type": "int", "low": 20, "high": 35, "step": 1},
        "rsi_overbought": {"type": "int", "low": 65, "high": 80, "step": 1},
        "vol_short_period": {"type": "int", "low": 3, "high": 8, "step": 1},
        "vol_long_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "vol_signal_period": {"type": "int", "low": 7, "high": 12, "step": 1},
        "vol_threshold": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize EMAs
        self.ema1 = btind.EMA(self.data.close, period=self.params.ema1_period)
        self.ema2 = btind.EMA(self.data.close, period=self.params.ema2_period)
        self.ema3 = btind.EMA(self.data.close, period=self.params.ema3_period)
        self.ema4 = btind.EMA(self.data.close, period=self.params.ema4_period)

        # Initialize MACD
        self.macd = btind.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal,
        )

        # Initialize RSI
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)

        # Initialize Volume Oscillator
        self.vol_osc = VolumeOscillator(
            self.data,
            short_period=self.params.vol_short_period,
            long_period=self.params.vol_long_period,
            signal_period=self.params.vol_signal_period,
            threshold=self.params.vol_threshold,
        )

        # State variables
        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.ema4_period,
                self.params.macd_slow,
                self.params.rsi_period,
                self.params.vol_long_period,
                self.params.vol_signal_period,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized EMAMACDRSIVolume with params: {self.params}")
        logger.info(
            f"EMAMACDRSIVolume initialized with ema1={self.p.ema1_period}, "
            f"ema2={self.p.ema2_period}, ema3={self.p.ema3_period}, "
            f"ema4={self.p.ema4_period}, macd={self.p.macd_fast}/{self.p.macd_slow}/{self.p.macd_signal}, "
            f"rsi_period={self.p.rsi_period}, vol_period={self.p.vol_long_period}"
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
            np.isnan(self.ema1[0])
            or np.isnan(self.ema2[0])
            or np.isnan(self.ema3[0])
            or np.isnan(self.ema4[0])
            or np.isnan(self.macd.macd[0])
            or np.isnan(self.macd.signal[0])
            or np.isnan(self.rsi[0])
            or np.isnan(self.vol_osc.vol_osc[0])
        ):
            logger.debug(f"Invalid indicator values at bar {len(self)}")
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "ema1": self.ema1[0],
                "ema2": self.ema2[0],
                "em3": self.ema3[0],
                "ema4": self.ema4[0],
                "macd": self.macd.macd[0],
                "macd_signal": self.macd.signal[0],
                "rsi": self.rsi[0],
                "vol_osc": self.vol_osc.vol_osc[0],
                "vol_signal": self.vol_osc.signal[0],
            }
        )

        # EMA alignment checks
        price = self.data.close[0]
        bullish_ema = price > self.ema1[0] > self.ema2[0] > self.ema3[0] > self.ema4[0]
        bearish_ema = price < self.ema1[0] < self.ema2[0] < self.ema3[0] < self.ema4[0]

        # MACD signals
        macd_bullish = (
            self.macd.macd[0] > self.macd.signal[0]
            and self.macd.macd[-1] <= self.macd.signal[-1]
        )
        macd_bearish = (
            self.macd.macd[0] < self.macd.signal[0]
            and self.macd.macd[-1] >= self.macd.signal[-1]
        )

        # RSI conditions
        rsi_oversold_recovery = (
            self.rsi[0] > self.params.rsi_oversold
            and self.rsi[-1] <= self.params.rsi_oversold
        )
        rsi_overbought_decline = (
            self.rsi[0] < self.params.rsi_overbought
            and self.rsi[-1] >= self.params.rsi_overbought
        )

        # Volume Oscillator condition
        vol_confirm = self.vol_osc.vol_osc[0] > self.params.vol_threshold

        if not self.position:
            # Long Entry
            if bullish_ema and macd_bullish and rsi_oversold_recovery and vol_confirm:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - EMA + MACD + RSI + Volume) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"EMA1: {self.ema1[0]:.2f} | MACD: {self.macd.macd[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} | Vol Osc: {self.vol_osc.vol_osc[0]:.2f}"
                )

            # Short Entry
            elif (
                bearish_ema and macd_bearish and rsi_overbought_decline and vol_confirm
            ):
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - EMA + MACD + RSI + Volume) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"EMA1: {self.ema1[0]:.2f} | MACD: {self.macd.macd[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} | Vol Osc: {self.vol_osc.vol_osc[0]:.2f}"
                )

        elif self.position.size > 0:  # Long position
            # Exit: Bearish EMA alignment or MACD bearish crossover or RSI overbought
            if bearish_ema or macd_bearish or self.rsi[0] >= self.params.rsi_overbought:
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "Bearish EMA"
                    if bearish_ema
                    else "MACD Bearish" if macd_bearish else "RSI Overbought"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - {exit_reason}) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f}"
                )

        elif self.position.size < 0:  # Short position
            # Exit: Bullish EMA alignment or MACD bullish crossover or RSI oversold
            if bullish_ema or macd_bullish or self.rsi[0] <= self.params.rsi_oversold:
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "Bullish EMA"
                    if bullish_ema
                    else "MACD Bullish" if macd_bullish else "RSI Oversold"
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
            "ema1_period": trial.suggest_int("ema1_period", 3, 8),
            "ema2_period": trial.suggest_int("ema2_period", 7, 12),
            "ema3_period": trial.suggest_int("ema3_period", 10, 16),
            "ema4_period": trial.suggest_int("ema4_period", 18, 25),
            "macd_fast": trial.suggest_int("macd_fast", 10, 15),
            "macd_slow": trial.suggest_int("macd_slow", 20, 30),
            "macd_signal": trial.suggest_int("macd_signal", 7, 12),
            "rsiさい_period": trial.suggest_int("rsi_period", 10, 20),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 35),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 65, 80),
            "vol_short_period": trial.suggest_int("vol_short_period", 3, 8),
            "vol_long_period": trial.suggest_int("vol_long_period", 10, 20),
            "vol_signal_period": trial.suggest_int("vol_signal_period", 7, 12),
            "vol_threshold": trial.suggest_float("vol_threshold", 0.0, 0.5, step=0.1),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            ema4_period = params.get("ema4_period", 21)
            macd_slow = params.get("macd_slow", 26)
            rsi_period = params.get("rsi_period", 14)
            vol_long_period = params.get("vol_long_period", 14)
            vol_signal_period = params.get("vol_signal_period", 9)
            return (
                max(
                    ema4_period,
                    macd_slow,
                    rsi_period,
                    vol_long_period,
                    vol_signal_period,
                )
                + 2
            )
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 50
