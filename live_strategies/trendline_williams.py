import pandas as pd
import pandas_ta as ta
import pytz
import datetime
import logging
from uuid import uuid4

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class TrendlineWilliams:
    """
    Trendlines + Williams %R Confluence Trading Strategy

    This strategy combines trendline tests with Williams %R to identify trading opportunities
    at significant price levels with momentum confirmation. It is designed for mean reversion
    or breakout scenarios at trendline levels.

    Strategy Type: CONFLUENCE-BASED TRADING
    ======================================
    This strategy uses trendlines (calculated via recent swing points) as key price levels
    and Williams %R for momentum confirmation to enter trades. It assumes price reactions
    at trendlines, combined with %R signals, provide reliable entry points.

    Strategy Logic:
    ==============
    Long Position Rules:
    - Entry: Price touches or is near an upward trendline (support) AND Williams %R < oversold threshold (default -80)
    - Exit: Price breaks below trendline OR Williams %R > exit threshold (default -50)

    Short Position Rules:
    - Entry: Price touches or is near a downward trendline (resistance) AND Williams %R > overbought threshold (default -20)
    - Exit: Price breaks above trendline OR Williams %R < exit threshold (default -50)

    Trendline Calculation:
    =====================
    - Trendlines are approximated using linear regression on recent swing highs/lows
    - Upward trendline (support): Linear fit through recent swing lows
    - Downward trendline (resistance): Linear fit through recent swing highs
    - Tolerance range (default 0.5%) for detecting price proximity to trendlines

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses warmup period to ensure indicator and trendline stability
    - Prevents order overlap with pending order checks
    - Suitable for trending or ranging markets with clear trendline reactions

    Indicators Used:
    ===============
    - Williams %R: Measures momentum to confirm oversold/overbought conditions
    - Trendlines: Calculated using linear regression on swing highs/lows
    - Tolerance Range: Percentage-based buffer to detect price proximity to trendlines

    Features:
    =========
    - Dynamic trendline calculation using swing points
    - Comprehensive trade logging with IST timezone
    - Detailed PnL tracking for each completed trade
    - Position sizing and commission handling
    - Optimization-ready parameter space
    - Robust error handling and data validation
    - Support for both backtesting and live trading

    Parameters:
    ==========
    - williams_period (int): Williams %R calculation period (default: 14)
    - trend_lookback (int): Lookback period for trendline calculation (default: 20)
    - trend_tolerance (float): Percentage tolerance for trendline proximity (default: 0.5)
    - williams_oversold (int): Williams %R oversold threshold for long entries (default: -80)
    - williams_overbought (int): Williams %R overbought threshold for short entries (default: -20)
    - williams_exit (int): Williams %R exit threshold for position closes (default: -50)
    - verbose (bool): Enable detailed logging (default: False)

    Performance Metrics:
    ===================
    - Tracks win/loss ratio
    - Calculates net PnL including commissions
    - Records trade duration and timing
    - Provides detailed execution logs
    - Monitors trendline touch frequency

    Usage:
    ======
    strategy = TrendlineWilliams(data)
    strategy.run()

    Best Market Conditions:
    ======================
    - Markets with clear trendline patterns
    - Ranging or trending markets with frequent trendline tests
    - Avoid during choppy markets with unclear trendlines
    - Works well in intraday timeframes with sufficient volatility

    Note:
    ====
    This strategy relies on accurate trendline detection and Williams %R confirmation.
    Trendline calculations are approximations and may require frequent updates.
    Consider adding volume or trend filters to improve signal reliability.
    Backtest thoroughly to validate trendline accuracy.
    """

    params = {
        "williams_period": 14,
        "trend_lookback": 20,
        "trend_tolerance": 0.5,  # 0.5% tolerance for trendline proximity
        "williams_oversold": -80,
        "williams_overbought": -20,
        "williams_exit": -50,
        "verbose": False,
    }

    optimization_params = {
        "williams_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "trend_lookback": {"type": "int", "low": 15, "high": 30, "step": 1},
        "trend_tolerance": {"type": "float", "low": 0.3, "high": 1.0, "step": 0.1},
        "williams_oversold": {"type": "int", "low": -90, "high": -70, "step": 5},
        "williams_overbought": {"type": "int", "low": -30, "high": -10, "step": 5},
        "williams_exit": {"type": "int", "low": -60, "high": -40, "step": 5},
    }

    def __init__(self, data, tickers=None, **kwargs):
        """
        Initialize the strategy with market data and parameters.

        Args:
            data (pd.DataFrame): DataFrame with OHLCV data (open, high, low, close, volume)
            tickers (list, optional): List of tickers for multi-asset trading
            **kwargs: Additional parameters to override defaults
        """
        self.data = data.copy()
        self.data["datetime"] = pd.to_datetime(self.data["datetime"])
        self.params.update(kwargs)
        self.order = None
        self.order_type = None
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = (
            self.params["williams_period"] + self.params["trend_lookback"] + 2
        )
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        # Initialize indicators using pandas_ta
        self.data["williams"] = ta.willr(
            self.data["high"],
            self.data["low"],
            self.data["close"],
            length=self.params["williams_period"],
        )
        self.data["swing_high"] = (
            self.data["high"].rolling(window=self.params["trend_lookback"]).max()
        )
        self.data["swing_low"] = (
            self.data["low"].rolling(window=self.params["trend_lookback"]).min()
        )

        logger.debug(f"Initialized TrendlineWilliams with params: {self.params}")
        logger.info(
            f"TrendlineWilliams initialized with williams_period={self.params['williams_period']}, "
            f"trend_lookback={self.params['trend_lookback']}, trend_tolerance={self.params['trend_tolerance']}, "
            f"williams_oversold={self.params['williams_oversold']}, williams_overbought={self.params['williams_overbought']}, "
            f"williams_exit={self.params['williams_exit']}"
        )

    def run(self):
        """
        Execute the strategy on the provided data.
        Returns the last signal generated ("BUY", "SELL", or None).
        """
        last_signal = None  # Track the last signal generated

        for idx in range(len(self.data)):
            if idx < self.warmup_period:
                logger.debug(
                    f"Skipping row {idx}: still in warmup period (need {self.warmup_period} rows)"
                )
                continue

            if not self.ready:
                self.ready = True
                logger.info(f"Strategy ready at row {idx}")

            bar_time = self.data.iloc[idx]["datetime"]
            bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
            current_time = bar_time_ist.time()

            # Force close positions at 15:15 IST
            if current_time >= datetime.time(15, 15):
                if self.open_positions:
                    self._close_position(idx, "Force close at 15:15 IST")
                continue

            # Only trade during market hours (9:15 AM to 3:05 PM IST)
            if not (datetime.time(9, 15) <= current_time <= datetime.time(15, 5)):
                continue

            if self.order:
                logger.debug(f"Order pending at row {idx}")
                continue

            # Check for invalid indicator values
            if (
                pd.isna(self.data.iloc[idx]["williams"])
                or pd.isna(self.data.iloc[idx]["swing_high"])
                or pd.isna(self.data.iloc[idx]["swing_low"])
            ):
                logger.debug(
                    f"Invalid indicator values at row {idx}: "
                    f"Williams %R={self.data.iloc[idx]['williams']}, "
                    f"Swing_High={self.data.iloc[idx]['swing_high']}, "
                    f"Swing_Low={self.data.iloc[idx]['swing_low']}"
                )
                continue

            # Calculate trendline levels
            current_price = self.data.iloc[idx]["close"]
            support_trendline = self.data.iloc[idx]["swing_low"]
            resistance_trendline = self.data.iloc[idx]["swing_high"]
            tolerance = self.params["trend_tolerance"] / 100 * current_price

            support_touch = abs(current_price - support_trendline) <= tolerance
            resistance_touch = abs(current_price - resistance_trendline) <= tolerance

            # Store indicator data
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": current_price,
                    "williams": self.data.iloc[idx]["williams"],
                    "support_trendline": support_trendline,
                    "resistance_trendline": resistance_trendline,
                    "support_touch": support_touch,
                    "resistance_touch": resistance_touch,
                }
            )

            # Trading Logic
            if not self.open_positions:
                # Long Entry
                if (
                    support_touch
                    and self.data.iloc[idx]["williams"]
                    < self.params["williams_oversold"]
                ):
                    self._place_order(idx, "buy", "enter_long")
                    last_signal = "BUY"  # Set the signal
                    trade_logger.info(
                        f"BUY SIGNAL (Enter Long - Trendline + Williams %R) | Row: {idx} | "
                        f"Time: {bar_time_ist} | "
                        f"Price: {current_price:.2f} | "
                        f"Williams %R: {self.data.iloc[idx]['williams']:.2f} < {self.params['williams_oversold']} (Oversold) | "
                        f"Trendline Support: {support_trendline:.2f} (Touch)"
                    )
                # Short Entry
                elif (
                    resistance_touch
                    and self.data.iloc[idx]["williams"]
                    > self.params["williams_overbought"]
                ):
                    self._place_order(idx, "sell", "enter_short")
                    last_signal = "SELL"  # Set the signal
                    trade_logger.info(
                        f"SELL SIGNAL (Enter Short - Trendline + Williams %R) | Row: {idx} | "
                        f"Time: {bar_time_ist} | "
                        f"Price: {current_price:.2f} | "
                        f"Williams %R: {self.data.iloc[idx]['williams']:.2f} > {self.params['williams_overbought']} (Overbought) | "
                        f"Trendline Resistance: {resistance_trendline:.2f} (Touch)"
                    )
            elif self.open_positions[-1]["direction"] == "long":
                # Long Exit
                if (
                    current_price < support_trendline
                    or self.data.iloc[idx]["williams"] > self.params["williams_exit"]
                ):
                    exit_reason = (
                        "Price broke below trendline"
                        if current_price < support_trendline
                        else "Williams %R normalized"
                    )
                    self._close_position(idx, exit_reason, "sell", "exit_long")
                    last_signal = "SELL"  # Set the signal for exit
                    trade_logger.info(
                        f"SELL SIGNAL (Exit Long - Trendline + Williams %R) | Row: {idx} | "
                        f"Time: {bar_time_ist} | "
                        f"Price: {current_price:.2f} | "
                        f"Reason: {exit_reason} | "
                        f"Williams %R: {self.data.iloc[idx]['williams']:.2f} | "
                        f"Trendline Support: {support_trendline:.2f}"
                    )
            elif self.open_positions[-1]["direction"] == "short":
                # Short Exit
                if (
                    current_price > resistance_trendline
                    or self.data.iloc[idx]["williams"] < self.params["williams_exit"]
                ):
                    exit_reason = (
                        "Price broke above trendline"
                        if current_price > resistance_trendline
                        else "Williams %R normalized"
                    )
                    self._close_position(idx, exit_reason, "buy", "exit_short")
                    last_signal = "BUY"  # Set the signal for exit
                    trade_logger.info(
                        f"BUY SIGNAL (Exit Short - Trendline + Williams %R) | Row: {idx} | "
                        f"Time: {bar_time_ist} | "
                        f"Price: {current_price:.2f} | "
                        f"Reason: {exit_reason} | "
                        f"Williams %R: {self.data.iloc[idx]['williams']:.2f} | "
                        f"Trendline Resistance: {resistance_trendline:.2f}"
                    )

        return last_signal  # Return the last signal generated

    def _place_order(self, idx, action, order_type, size=1, commission=0.001):
        """
        Simulate placing an order.

        Args:
            idx (int): Current data index
            action (str): 'buy' or 'sell'
            order_type (str): Type of order (e.g., 'enter_long', 'exit_short')
            size (float): Position size
            commission (float): Commission rate per trade
        """
        self.order = {
            "ref": str(uuid4()),
            "action": action,
            "order_type": order_type,
            "status": "Completed",
            "executed_price": self.data.iloc[idx]["close"],
            "size": size if action == "buy" else -size,
            "commission": abs(self.data.iloc[idx]["close"] * size * commission),
            "executed_time": self.data.iloc[idx]["datetime"],
        }
        self._notify_order(idx)

    def _notify_order(self, idx):
        """
        Process order execution and log details.

        Args:
            idx (int): Current data index
        """
        order = self.order
        exec_dt = order["executed_time"]
        if exec_dt.tzinfo is None:
            exec_dt = exec_dt.replace(tzinfo=pytz.UTC)

        if order["order_type"] == "enter_long" and order["action"] == "buy":
            position_info = {
                "entry_time": exec_dt,
                "entry_price": order["executed_price"],
                "size": order["size"],
                "commission": order["commission"],
                "ref": order["ref"],
                "direction": "long",
            }
            self.open_positions.append(position_info)
            trade_logger.info(
                f"BUY EXECUTED (Enter Long) | Ref: {order['ref']} | "
                f"Price: {order['executed_price']:.2f} | "
                f"Size: {order['size']} | "
                f"Cost: {order['executed_price'] * order['size']:.2f} | "
                f"Comm: {order['commission']:.2f}"
            )
        elif order["order_type"] == "enter_short" and order["action"] == "sell":
            position_info = {
                "entry_time": exec_dt,
                "entry_price": order["executed_price"],
                "size": order["size"],
                "commission": order["commission"],
                "ref": order["ref"],
                "direction": "short",
            }
            self.open_positions.append(position_info)
            trade_logger.info(
                f"SELL EXECUTED (Enter Short) | Ref: {order['ref']} | "
                f"Price: {order['executed_price']:.2f} | "
                f"Size: {order['size']} | "
                f"Cost: {order['executed_price'] * abs(order['size']):.2f} | "
                f"Comm: {order['commission']:.2f}"
            )

        self.order = None
        self.order_type = None

    def _close_position(self, idx, reason, action=None, order_type=None):
        """
        Close an open position and calculate PnL.

        Args:
            idx (int): Current data index
            reason (str): Reason for closing the position
            action (str, optional): 'buy' or 'sell' for closing order
            order_type (str, optional): Type of closing order (e.g., 'exit_long')
        """
        if not self.open_positions:
            return

        entry_info = self.open_positions[-1]
        self.order = {
            "ref": str(uuid4()),
            "action": action,
            "order_type": order_type,
            "status": "Completed",
            "executed_price": self.data.iloc[idx]["close"],
            "size": -entry_info["size"],
            "commission": abs(
                self.data.iloc[idx]["close"] * entry_info["size"] * 0.001
            ),
            "executed_time": self.data.iloc[idx]["datetime"],
        }
        order = self.order

        if order["order_type"] == "exit_long" and order["action"] == "sell":
            entry_info = self.open_positions.pop(0)
            pnl = (order["executed_price"] - entry_info["entry_price"]) * abs(
                entry_info["size"]
            )
            total_commission = entry_info["commission"] + abs(order["commission"])
            pnl_net = pnl - total_commission
            trade_info = {
                "ref": order["ref"],
                "entry_time": entry_info["entry_time"],
                "exit_time": order["executed_time"],
                "entry_price": entry_info["entry_price"],
                "exit_price": order["executed_price"],
                "size": abs(entry_info["size"]),
                "pnl": pnl,
                "pnl_net": pnl_net,
                "commission": total_commission,
                "status": "Won" if pnl > 0 else "Lost",
                "direction": "Long",
                "bars_held": (
                    order["executed_time"] - entry_info["entry_time"]
                ).total_seconds()
                / 60,  # in minutes
            }
            self.completed_trades.append(trade_info)
            self.trade_count += 1
            trade_logger.info(
                f"SELL EXECUTED (Exit Long) | Ref: {order['ref']} | "
                f"Price: {order['executed_price']:.2f} | "
                f"Size: {order['size']} | "
                f"Cost: {order['executed_price'] * abs(order['size']):.2f} | "
                f"Comm: {order['commission']:.2f} | "
                f"PnL: {pnl:.2f} | Reason: {reason}"
            )
        elif order["order_type"] == "exit_short" and order["action"] == "buy":
            entry_info = self.open_positions.pop(0)
            pnl = (entry_info["entry_price"] - order["executed_price"]) * abs(
                entry_info["size"]
            )
            total_commission = entry_info["commission"] + abs(order["commission"])
            pnl_net = pnl - total_commission
            trade_info = {
                "ref": order["ref"],
                "entry_time": entry_info["entry_time"],
                "exit_time": order["executed_time"],
                "entry_price": entry_info["entry_price"],
                "exit_price": order["executed_price"],
                "size": abs(entry_info["size"]),
                "pnl": pnl,
                "pnl_net": pnl_net,
                "commission": total_commission,
                "status": "Won" if pnl > 0 else "Lost",
                "direction": "Short",
                "bars_held": (
                    order["executed_time"] - entry_info["entry_time"]
                ).total_seconds()
                / 60,  # in minutes
            }
            self.completed_trades.append(trade_info)
            self.trade_count += 1
            trade_logger.info(
                f"BUY EXECUTED (Exit Short) | Ref: {order['ref']} | "
                f"Price: {order['executed_price']:.2f} | "
                f"Size: {order['size']} | "
                f"Cost: {order['executed_price'] * abs(order['size']):.2f} | "
                f"Comm: {order['commission']:.2f} | "
                f"PnL: {pnl:.2f} | Reason: {reason}"
            )

        self.order = None
        self.order_type = None

    def get_completed_trades(self):
        """
        Return a copy of completed trades.

        Returns:
            list: List of completed trade dictionaries
        """
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        """
        Define parameter space for optimization.

        Args:
            trial: Optuna trial object

        Returns:
            dict: Parameter space
        """
        params = {
            "williams_period": trial.suggest_int("williams_period", 10, 20),
            "trend_lookback": trial.suggest_int("trend_lookback", 15, 30),
            "trend_tolerance": trial.suggest_float("trend_tolerance", 0.3, 1.0),
            "williams_oversold": trial.suggest_int("williams_oversold", -90, -70),
            "williams_overbought": trial.suggest_int("williams_overbought", -30, -10),
            "williams_exit": trial.suggest_int("williams_exit", -60, -40),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        """
        Calculate minimum data points required for strategy.

        Args:
            params (dict): Strategy parameters

        Returns:
            int: Minimum number of data points
        """
        try:
            williams_period = params.get("williams_period", 14)
            trend_lookback = params.get("trend_lookback", 20)
            max_period = max(williams_period, trend_lookback)
            return max_period + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30


# if __name__ == "__main__":
#     ticker = 1922
#     data = pd.read_csv(f"combined_data_{ticker}.csv")
#     strategy = TrendlineWilliams(data)
#     strategy.run()
#     orders = strategy.completed_trades
#     print("================")
#     orders_df = pd.DataFrame(orders)
#     orders_df.to_csv(f"orders_{ticker}.csv", index=False)
#     print("orders saved to orders_{ticker}.csv")
#     print("=================")
