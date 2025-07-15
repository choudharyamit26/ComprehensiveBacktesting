import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class EMAStochasticPullback(bt.Strategy):
    """
    EMA + Stochastic Pullback Trading Strategy

    This strategy combines Exponential Moving Average (EMA) and Stochastic Oscillator
    to identify mean reversion opportunities during price pullbacks to the EMA when
    the Stochastic indicates oversold conditions, expecting a reversion to the trend.

    Strategy Type: MEAN REVERSION
    =============================
    This is a mean reversion strategy that assumes prices will revert to the EMA
    after a pullback, confirmed by Stochastic oversold/overbought signals. It buys
    when prices pull back to the EMA in oversold conditions and sells when overbought
    or the price breaks the EMA.

    Strategy Logic:
    ==============

    Long Position Rules:
    - Entry: Price touches or crosses below EMA AND Stochastic %K < oversold threshold (default: 20)
    - Exit: Stochastic %K > overbought threshold (default: 80) OR price crosses below EMA

    Short Position Rules:
    - Entry: Price touches or crosses above EMA AND Stochastic %K > overbought threshold (default: 80)
    - Exit: Stochastic %K < oversold threshold (default: 20) OR price crosses above EMA

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses warmup period to ensure indicator stability before trading
    - Prevents order overlap with pending order checks
    - Mean reversion works best in ranging or mildly trending markets

    Indicators Used:
    ===============
    - EMA: Exponential Moving Average to identify the trend's anchor point
    - Stochastic Oscillator: Measures momentum to identify oversold/overbought conditions
      * %K: Fast stochastic line
      * %D: Slow stochastic line (signal line)

    Mean Reversion Concept:
    ======================
    - Price pulling back to EMA + Stochastic oversold = likely bounce upward
    - Price pulling back to EMA + Stochastic overbought = likely pullback downward
    - Exit on Stochastic reversal or EMA break assumes reversion completion
    - Stochastic confirms momentum shifts for entry and exit

    Features:
    =========
    - Comprehensive trade logging with IST timezone
    - Detailed PnL tracking for each completed trade
    - Position sizing and commission handling
    - Optimization-ready parameter space
    - Robust error handling and data validation
    - Support for both backtesting and live trading
    - Pullback detection for precise entry timing

    Parameters:
    ==========
    - ema_period (int): EMA calculation period (default: 20)
    - stoch_k_period (int): Stochastic %K period (default: 14)
    - stoch_d_period (int): Stochastic %D period (default: 3)
    - stoch_slowing (int): Stochastic slowing period (default: 3)
    - stoch_oversold (int): Stochastic oversold threshold for long entries (default: 20)
    - stoch_overbought (int): Stochastic overbought threshold for short entries (default: 80)
    - verbose (bool): Enable detailed logging (default: False)

    Performance Metrics:
    ===================
    - Tracks win/loss ratio
    - Calculates net PnL including commissions
    - Records trade duration and timing
    - Provides detailed execution logs
    - Monitors EMA pullback frequency

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EMAStochasticPullback, ema_period=50, stoch_oversold=25)
    cerebro.run()

    Best Market Conditions:
    ======================
    - Ranging or mildly trending markets with frequent pullbacks
    - Moderate Volatility periods with clear EMA support/resistance
    - Avoid during strong trending markets (trend-following strategies better)
    - Works well in intraday timeframes with sufficient price movement

    Note:
    ====
    This is a mean reversion strategy that profits from price reverting to the EMA.
    It's opposite to momentum strategies and requires different market conditions.
    Consider using trend filters to avoid trading against strong trends.
    """

    params = (
        ("ema_period", 20),
        ("stoch_k_period", 14),
        ("stoch_d_period", 3),
        ("stoch_slowing", 3),
        ("stoch_oversold", 20),
        ("stoch_overbought", 80),
        ("verbose", False),
    )

    optimization_params = {
        "ema_period": {"type": "int", "low": 10, "high": 50, "step": 5},
        "stoch_k_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "stoch_d_period": {"type": "int", "low": 3, "high": 5, "step": 1},
        "stoch_slowing": {"type": "int", "low": 1, "high": 5, "step": 1},
        "stoch_oversold": {"type": "int", "low": 10, "high": 30, "step": 5},
        "stoch_overbought": {"type": "int", "low": 70, "high": 90, "step": 5},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        # Initialize indicators
        self.ema = btind.EMA(self.data.close, period=self.params.ema_period)
        self.stoch = btind.Stochastic(
            period=self.params.stoch_k_period,
            period_dfast=self.params.stoch_d_period,
            period_dslow=self.params.stoch_slowing,
        )

        # Stochastic components
        self.stoch_k = self.stoch.percK
        self.stoch_d = self.stoch.percD

        # Pullback detection
        self.ema_touch_long = self.data.close <= self.ema
        self.ema_touch_short = self.data.close >= self.ema

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = max(
            self.params.ema_period,
            self.params.stoch_k_period + self.params.stoch_d_period + 2,
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized EMAStochasticPullback with params: {self.params}")
        logger.info(
            f"EMAStochasticPullback initialized with ema_period={self.p.ema_period}, "
            f"stoch_k_period={self.p.stoch_k_period}, stoch_d_period={self.p.stoch_d_period}, "
            f"stoch_slowing={self.p.stoch_slowing}, stoch_oversold={self.p.stoch_oversold}, "
            f"stoch_overbought={self.p.stoch_overbought}"
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
            np.isnan(self.ema[0])
            or np.isnan(self.stoch_k[0])
            or np.isnan(self.stoch_d[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"EMA={self.ema[0]}, Stoch_K={self.stoch_k[0]}, Stoch_D={self.stoch_d[0]}"
            )
            return

        # Calculate distance from EMA
        ema_distance = (self.data.close[0] - self.ema[0]) / self.ema[0]

        # Store indicator data for analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": self.data.close[0],
                "ema": self.ema[0],
                "stoch_k": self.stoch_k[0],
                "stoch_d": self.stoch_d[0],
                "ema_distance": ema_distance,
                "ema_touch_long": self.ema_touch_long[0],
                "ema_touch_short": self.ema_touch_short[0],
                "bar": len(self),
            }
        )

        # Mean Reversion Position Management
        if not self.position:
            # Long Entry: Price at/below EMA AND Stochastic oversold
            if self.ema_touch_long[0] and self.stoch_k[0] < self.params.stoch_oversold:
                self.order = self.buy()
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL (Enter Long - Pullback) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Stoch_K: {self.stoch_k[0]:.2f} < {self.params.stoch_oversold} (Oversold) | "
                    f"EMA: {self.ema[0]:.2f} (Touch) | "
                    f"EMA_Distance: {ema_distance:.2%}"
                )
            # Short Entry: Price at/above EMA AND Stochastic overbought
            elif (
                self.ema_touch_short[0]
                and self.stoch_k[0] > self.params.stoch_overbought
            ):
                self.order = self.sell()
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL (Enter Short - Pullback) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Stoch_K: {self.stoch_k[0]:.2f} > {self.params.stoch_overbought} (Overbought) | "
                    f"EMA: {self.ema[0]:.2f} (Touch) | "
                    f"EMA_Distance: {ema_distance:.2%}"
                )
        elif self.position.size > 0:  # Long position
            # Long Exit: Stochastic overbought OR price crosses below EMA
            if (
                self.stoch_k[0] > self.params.stoch_overbought
                or self.data.close[0] < self.ema[0]
            ):
                self.order = self.sell()
                self.order_type = "exit_long"
                exit_reason = (
                    "Stochastic Overbought"
                    if self.stoch_k[0] > self.params.stoch_overbought
                    else "Price below EMA"
                )
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long - Reversion) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Stoch_K: {self.stoch_k[0]:.2f} | "
                    f"EMA: {self.ema[0]:.2f} | "
                    f"EMA_Distance: {ema_distance:.2%}"
                )
        elif self.position.size < 0:  # Short position
            # Short Exit: Stochastic oversold OR price crosses above EMA
            if (
                self.stoch_k[0] < self.params.stoch_oversold
                or self.data.close[0] > self.ema[0]
            ):
                self.order = self.buy()
                self.order_type = "exit_short"
                exit_reason = (
                    "Stochastic Oversold"
                    if self.stoch_k[0] < self.params.stoch_oversold
                    else "Price above EMA"
                )
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short - Reversion) | Bar: {len(self)} | "
                    f"Time: {bar_time_ist} | "
                    f"Price: {self.data.close[0]:.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Stoch_K: {self.stoch_k[0]:.2f} | "
                    f"EMA: {self.ema[0]:.2f} | "
                    f"EMA_Distance: {ema_distance:.2%}"
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
                    "entry_bar": len(self),  # Store the bar index at entry
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
                    "entry_bar": len(self),  # Store the bar index at entry
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
                        "bars_held": len(self) - entry_info["entry_bar"],
                    }
                    self.completed_trades.append(trade_info)
                    self.trade_count += 1
                    trade_logger.info(
                        f"SELL EXECUTED (Exit Long) | Ref: {order.ref} | "
                        f"Price: {order.executed.price:.2f} | "
                        f"Size: {order.executed.size} | "
                        f"Cost: {order.executed.value:.2f} | "
                        f"Comm: {order.executed.comm:.2f} | "
                        f"PnL: {pnl:.2f} | Net PnL: {pnl_net:.2f}"
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
                        "bars_held": len(self) - entry_info["entry_bar"],
                    }
                    self.completed_trades.append(trade_info)
                    self.trade_count += 1
                    trade_logger.info(
                        f"BUY EXECUTED (Exit Short) | Ref: {order.ref} | "
                        f"Price: {order.executed.price:.2f} | "
                        f"Size: {order.executed.size} | "
                        f"Cost: {order.executed.value:.2f} | "
                        f"Comm: {order.executed.comm:.2f} | "
                        f"PnL: {pnl:.2f} | Net PnL: {pnl_net:.2f}"
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
            "ema_period": trial.suggest_int("ema_period", 10, 50, step=5),
            "stoch_k_period": trial.suggest_int("stoch_k_period", 10, 20),
            "stoch_d_period": trial.suggest_int("stoch_d_period", 3, 5),
            "stoch_slowing": trial.suggest_int("stoch_slowing", 1, 5),
            "stoch_oversold": trial.suggest_int("stoch_oversold", 10, 30, step=5),
            "stoch_overbought": trial.suggest_int("stoch_overbought", 70, 90, step=5),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        try:
            ema_period = params.get("ema_period", 20)
            stoch_k_period = params.get("stoch_k_period", 14)
            stoch_d_period = params.get("stoch_d_period", 3)
            return max(ema_period, stoch_k_period + stoch_d_period + 2)
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
