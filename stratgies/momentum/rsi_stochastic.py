import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class RSIStochastic(bt.Strategy):
    """
    RSI and Stochastic Double Oscillator Intraday Trading Strategy

    This strategy combines RSI (Relative Strength Index) and Stochastic Oscillator
    indicators to identify momentum-based entry and exit signals for both long and short positions.

    Strategy Logic:
    ==============

    Long Position Rules:
    - Entry: RSI > threshold AND Stochastic %K > %D (bullish alignment)
    - Exit: RSI < threshold OR Stochastic %K < %D (divergence between oscillators)

    Short Position Rules:
    - Entry: RSI < threshold AND Stochastic %K < %D (bearish alignment)
    - Exit: RSI > threshold OR Stochastic %K > %D (divergence between oscillators)

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses warmup period to ensure indicator stability before trading
    - Prevents order overlap with pending order checks

    Indicators Used:
    ===============
    - RSI: Measures momentum strength (overbought/oversold conditions)
    - Stochastic: Compares closing price to price range over period
    - %K: Fast stochastic line (current close relative to recent range)
    - %D: Slow stochastic line (moving average of %K)

    Features:
    =========
    - Comprehensive trade logging with IST timezone
    - Detailed PnL tracking for each completed trade
    - Position sizing and commission handling
    - Optimization-ready parameter space
    - Robust error handling and data validation
    - Support for both backtesting and live trading

    Parameters:
    ==========
    - rsi_period (int): RSI calculation period (default: 14)
    - stoch_period (int): Stochastic %K period (default: 14)
    - stoch_period_dfast (int): Stochastic %D fast period (default: 3)
    - stoch_period_dslow (int): Stochastic %D slow period (default: 3)
    - rsi_threshold (int): RSI threshold for entry/exit signals (default: 50)
    - stoch_upper (int): Stochastic overbought level (default: 80)
    - stoch_lower (int): Stochastic oversold level (default: 20)
    - verbose (bool): Enable detailed logging (default: False)

    Performance Metrics:
    ===================
    - Tracks win/loss ratio
    - Calculates net PnL including commissions
    - Records trade duration and timing
    - Provides detailed execution logs

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(RSIStochastic, rsi_period=14, rsi_threshold=50)
    cerebro.run()

    Note:
    ====
    This is a momentum strategy that works best when both oscillators align.
    The strategy focuses on oscillator convergence and divergence patterns.
    """

    params = (
        ("rsi_period", 14),
        ("stoch_period", 14),
        ("stoch_period_dfast", 3),
        ("stoch_period_dslow", 3),
        ("rsi_threshold", 50),
        ("stoch_upper", 80),
        ("stoch_lower", 20),
        ("verbose", False),
    )

    optimization_params = {
        "rsi_period": {"type": "int", "low": 8, "high": 20, "step": 1},
        "stoch_period": {"type": "int", "low": 8, "high": 20, "step": 1},
        "stoch_period_dfast": {"type": "int", "low": 2, "high": 5, "step": 1},
        "stoch_period_dslow": {"type": "int", "low": 2, "high": 5, "step": 1},
        "rsi_threshold": {"type": "int", "low": 45, "high": 55, "step": 1},
        "stoch_upper": {"type": "int", "low": 70, "high": 90, "step": 5},
        "stoch_lower": {"type": "int", "low": 10, "high": 30, "step": 5},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize RSI
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)

        # Initialize Stochastic
        self.stochastic = btind.Stochastic(
            self.data,
            period=self.params.stoch_period,
            period_dfast=self.params.stoch_period_dfast,
            period_dslow=self.params.stoch_period_dslow,
        )

        # Stochastic components
        self.stoch_k = self.stochastic.percK  # Fast %K line
        self.stoch_d = self.stochastic.percD  # Slow %D line

        # Crossover signals for stochastic
        self.stoch_crossover = btind.CrossOver(self.stoch_k, self.stoch_d)

        self.order = None
        self.order_type = None  # Track order type for shorting logic
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            max(
                self.params.rsi_period,
                self.params.stoch_period
                + self.params.stoch_period_dfast
                + self.params.stoch_period_dslow,
            )
            + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized RSIStochastic with params: {self.params}")
        logger.info(
            f"RSIStochastic initialized with rsi_period={self.p.rsi_period}, "
            f"stoch_period={self.p.stoch_period}, "
            f"stoch_period_dfast={self.p.stoch_period_dfast}, "
            f"stoch_period_dslow={self.p.stoch_period_dslow}, "
            f"rsi_threshold={self.p.rsi_threshold}"
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
            or np.isnan(self.stoch_k[0])
            or np.isnan(self.stoch_d[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"RSI={self.rsi[0]}, Stoch_K={self.stoch_k[0]}, "
                f"Stoch_D={self.stoch_d[0]}"
            )
            return

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "rsi": self.rsi[0],
                "stoch_k": self.stoch_k[0],
                "stoch_d": self.stoch_d[0],
                "stoch_crossover": self.stoch_crossover[0],
            }
        )

        # Position management with shorting
        if not self.position:
            # Long Entry: RSI > threshold AND Stochastic %K > %D (bullish alignment)
            if (
                self.rsi[0] > self.params.rsi_threshold
                and self.stoch_k[0] > self.stoch_d[0]
            ):
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} > {self.params.rsi_threshold} | "
                    f"Stoch %K: {self.stoch_k[0]:.2f} > %D: {self.stoch_d[0]:.2f} (Bullish Alignment)"
                )
            # Short Entry: RSI < threshold AND Stochastic %K < %D (bearish alignment)
            elif (
                self.rsi[0] < self.params.rsi_threshold
                and self.stoch_k[0] < self.stoch_d[0]
            ):
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"RSI: {self.rsi[0]:.2f} < {self.params.rsi_threshold} | "
                    f"Stoch %K: {self.stoch_k[0]:.2f} < %D: {self.stoch_d[0]:.2f} (Bearish Alignment)"
                )
        elif self.position.size > 0:  # Long position
            # Long Exit: RSI < threshold OR Stochastic %K < %D (divergence)
            if (
                self.rsi[0] < self.params.rsi_threshold
                or self.stoch_k[0] < self.stoch_d[0]
            ):
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "RSI < threshold"
                    if self.rsi[0] < self.params.rsi_threshold
                    else "Stochastic divergence"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"Stoch %K: {self.stoch_k[0]:.2f}, %D: {self.stoch_d[0]:.2f}"
                )
        elif self.position.size < 0:  # Short position
            # Short Exit: RSI > threshold OR Stochastic %K > %D (divergence)
            if (
                self.rsi[0] > self.params.rsi_threshold
                or self.stoch_k[0] > self.stoch_d[0]
            ):
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "RSI > threshold"
                    if self.rsi[0] > self.params.rsi_threshold
                    else "Stochastic divergence"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"RSI: {self.rsi[0]:.2f} | "
                    f"Stoch %K: {self.stoch_k[0]:.2f}, %D: {self.stoch_d[0]:.2f}"
                )

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt)
            if exec_dt.tzinfo is None:
                exec_dt = exec_dt.replace(tzinfo=pytz.UTC)

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
                    "size": order.executed.size,  # Negative for short
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
            "rsi_period": trial.suggest_int("rsi_period", 8, 20),
            "stoch_period": trial.suggest_int("stoch_period", 8, 20),
            "stoch_period_dfast": trial.suggest_int("stoch_period_dfast", 2, 5),
            "stoch_period_dslow": trial.suggest_int("stoch_period_dslow", 2, 5),
            "rsi_threshold": trial.suggest_int("rsi_threshold", 45, 55),
            "stoch_upper": trial.suggest_int("stoch_upper", 70, 90),
            "stoch_lower": trial.suggest_int("stoch_lower", 10, 30),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            rsi_period = params.get("rsi_period", 14)
            stoch_period = params.get("stoch_period", 14)
            stoch_period_dfast = params.get("stoch_period_dfast", 3)
            stoch_period_dslow = params.get("stoch_period_dslow", 3)
            stoch_total_period = stoch_period + stoch_period_dfast + stoch_period_dslow
            max_period = max(rsi_period, stoch_total_period)
            return max_period + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 35
