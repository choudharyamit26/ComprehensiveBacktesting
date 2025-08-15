import pandas as pd
import pandas_ta as ta
import pytz
import datetime
import logging
import numpy as np
from uuid import uuid4


# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class EMAADXTrend:
    """
    EMA + ADX Trend System Trading Strategy

    This strategy combines Exponential Moving Average (EMA) for trend direction
    and Average Directional Index (ADX) for trend strength to identify high-probability
    trend-following opportunities with strong momentum.

    Strategy Type: TREND FOLLOWING
    ==============================
    This is a trend-following strategy that enters positions when price is trending
    in the EMA direction and the ADX confirms strong trend strength, maximizing
    the probability of sustained price movement.

    Strategy Logic:
    ==============

    Long Position Rules:
    - Entry: Price above EMA AND ADX > min_adx_threshold AND ADX rising
    - Exit: Price below EMA OR ADX < exit_adx_threshold OR ADX falling significantly

    Short Position Rules:
    - Entry: Price below EMA AND ADX > min_adx_threshold AND ADX rising
    - Exit: Price above EMA OR ADX < exit_adx_threshold OR ADX falling significantly

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST to avoid overnight risk
    - Uses ADX-based trend strength confirmation
    - Prevents order overlap with pending order checks
    - Works best in trending markets with sustained momentum

    Indicators Used:
    ===============
    - EMA: Exponential Moving Average for trend direction
    - ADX: Average Directional Index for trend strength measurement
    - DI+: Positive Directional Indicator (bullish momentum)
    - DI-: Negative Directional Indicator (bearish momentum)

    ADX Trend Strength Concept:
    ===========================
    - ADX > 25: Strong trend (default threshold)
    - ADX < 20: Weak trend or consolidation
    - Rising ADX: Strengthening trend
    - Falling ADX: Weakening trend
    - DI+ > DI-: Bullish momentum
    - DI- > DI+: Bearish momentum

    Features:
    =========
    - Comprehensive trade logging with IST timezone
    - ADX-based trend strength analysis
    - EMA trend direction confirmation
    - Dynamic exit based on trend weakening
    - Robust error handling and data validation
    - Support for both backtesting and live trading
    - Directional Indicator crossover analysis

    Parameters:
    ==========
    - ema_period (int): EMA calculation period (default: 21)
    - adx_period (int): ADX calculation period (default: 14)
    - min_adx_threshold (float): Minimum ADX for trend entry (default: 25.0)
    - exit_adx_threshold (float): ADX exit threshold (default: 20.0)
    - adx_rising_threshold (float): ADX rise requirement (default: 1.0)
    - adx_falling_threshold (float): ADX fall for exit (default: 3.0)
    - di_separation (float): Minimum DI+/DI- separation (default: 2.0)
    - price_ema_buffer (float): Price buffer above/below EMA (default: 0.1)
    - verbose (bool): Enable detailed logging (default: False)

    Performance Metrics:
    ===================
    - Tracks win/loss ratio
    - Calculates net PnL including commissions
    - Records trade duration and timing
    - Monitors trend strength and duration
    - Provides detailed execution logs

    Usage:
    ======
    strategy = EMAADXTrend(data, ema_period=21, min_adx_threshold=30)
    last_signal = strategy.run()

    Best Market Conditions:
    ======================
    - Strong trending markets with sustained momentum
    - High ADX readings indicating trend strength
    - Clear directional moves with minimal whipsaws
    - Avoid during consolidation or choppy markets

    Note:
    ====
    This strategy requires strong trends to be profitable. It's designed for
    trend-following and performs best when markets have clear directional bias.
    Consider using additional filters during ranging markets.
    """

    params = {
        "ema_period": 21,
        "adx_period": 14,
        "min_adx_threshold": 25.0,
        "exit_adx_threshold": 20.0,
        "adx_rising_threshold": 1.0,
        "adx_falling_threshold": 3.0,
        "di_separation": 2.0,
        "price_ema_buffer": 0.1,
        "verbose": False,
    }

    optimization_params = {
        "ema_period": {"type": "int", "low": 15, "high": 30, "step": 3},
        "adx_period": {"type": "int", "low": 10, "high": 20, "step": 2},
        "min_adx_threshold": {"type": "float", "low": 20.0, "high": 35.0, "step": 2.5},
        "exit_adx_threshold": {"type": "float", "low": 15.0, "high": 25.0, "step": 2.5},
        "adx_rising_threshold": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.5},
        "adx_falling_threshold": {
            "type": "float",
            "low": 2.0,
            "high": 5.0,
            "step": 1.0,
        },
        "di_separation": {"type": "float", "low": 1.0, "high": 5.0, "step": 1.0},
        "price_ema_buffer": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
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

        # Trend analysis variables
        self.trend_direction = 0  # 1 for up, -1 for down, 0 for neutral
        self.adx_rising = False
        self.adx_falling = False
        self.strong_trend = False
        self.di_bullish = False
        self.di_bearish = False

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = (
            max(self.params["ema_period"], self.params["adx_period"] * 2) + 5
        )

        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        # Initialize indicators using pandas_ta
        self.data["ema"] = ta.ema(self.data["close"], length=self.params["ema_period"])

        # Calculate ADX and DI indicators - pandas_ta returns DataFrame with multiple columns
        try:
            adx_data = ta.adx(
                self.data["high"],
                self.data["low"],
                self.data["close"],
                length=self.params["adx_period"],
            )

            # Check if ADX data is returned and assign columns properly
            if adx_data is not None and not adx_data.empty:
                # The column names might vary, let's check what's available
                adx_columns = adx_data.columns.tolist()
                logger.debug(f"ADX columns available: {adx_columns}")

                # Find the ADX column (could be named differently)
                adx_col = None
                dmp_col = None
                dmn_col = None

                for col in adx_columns:
                    if "ADX" in col.upper():
                        adx_col = col
                    elif (
                        "DMP" in col.upper()
                        or "DI+" in col.upper()
                        or "DIPLUS" in col.upper()
                    ):
                        dmp_col = col
                    elif (
                        "DMN" in col.upper()
                        or "DI-" in col.upper()
                        or "DIMINUS" in col.upper()
                    ):
                        dmn_col = col

                # Fallback to standard naming if specific columns not found
                if adx_col is None:
                    # Try common column names
                    possible_adx_names = [
                        f'ADX_{self.params["adx_period"]}',
                        "ADX",
                        "adx",
                    ]
                    for name in possible_adx_names:
                        if name in adx_columns:
                            adx_col = name
                            break

                if dmp_col is None:
                    possible_dmp_names = [
                        f'DMP_{self.params["adx_period"]}',
                        "DMP",
                        "dmp",
                    ]
                    for name in possible_dmp_names:
                        if name in adx_columns:
                            dmp_col = name
                            break

                if dmn_col is None:
                    possible_dmn_names = [
                        f'DMN_{self.params["adx_period"]}',
                        "DMN",
                        "dmn",
                    ]
                    for name in possible_dmn_names:
                        if name in adx_columns:
                            dmn_col = name
                            break

                # Assign the data if columns are found
                if adx_col and adx_col in adx_data.columns:
                    self.data["adx"] = adx_data[adx_col]
                else:
                    logger.error(
                        f"ADX column not found in returned data. Available columns: {adx_columns}"
                    )
                    raise KeyError(
                        f"ADX column not found. Available columns: {adx_columns}"
                    )

                if dmp_col and dmp_col in adx_data.columns:
                    self.data["di_plus"] = adx_data[dmp_col]
                else:
                    logger.warning(
                        f"DI+ column not found. Available columns: {adx_columns}"
                    )
                    self.data["di_plus"] = np.nan

                if dmn_col and dmn_col in adx_data.columns:
                    self.data["di_minus"] = adx_data[dmn_col]
                else:
                    logger.warning(
                        f"DI- column not found. Available columns: {adx_columns}"
                    )
                    self.data["di_minus"] = np.nan

            else:
                logger.error("ADX calculation returned empty or None result")
                raise ValueError("ADX calculation failed")

        except Exception as e:
            logger.error(f"Error calculating ADX indicators: {str(e)}")
            # Fallback: create empty columns with NaN values
            self.data["adx"] = np.nan
            self.data["di_plus"] = np.nan
            self.data["di_minus"] = np.nan
            raise

        logger.debug(f"Initialized EMAADXTrend with params: {self.params}")
        logger.info(
            f"EMAADXTrend initialized with ema_period={self.params['ema_period']}, "
            f"adx_period={self.params['adx_period']}, min_adx_threshold={self.params['min_adx_threshold']}, "
            f"exit_adx_threshold={self.params['exit_adx_threshold']}"
        )

    def run(self):
        """
        Execute the strategy on the provided data.
        Returns the last signal generated ("BUY", "SELL", or None).
        """
        last_signal = None

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
            current_row = self.data.iloc[idx]
            if (
                pd.isna(current_row["ema"])
                or pd.isna(current_row["adx"])
                or pd.isna(current_row["di_plus"])
                or pd.isna(current_row["di_minus"])
                or idx < 3  # Need at least 3 bars for ADX analysis
            ):
                logger.debug(f"Invalid indicator values at row {idx}")
                continue

            # Analyze trend direction based on EMA
            current_price = current_row["close"]
            price_above_ema = current_price > (
                current_row["ema"] + self.params["price_ema_buffer"]
            )
            price_below_ema = current_price < (
                current_row["ema"] - self.params["price_ema_buffer"]
            )

            if price_above_ema:
                self.trend_direction = 1  # Uptrend
            elif price_below_ema:
                self.trend_direction = -1  # Downtrend
            else:
                self.trend_direction = 0  # Neutral

            # Analyze ADX trend strength
            if idx >= 3:
                adx_change = current_row["adx"] - self.data.iloc[idx - 1]["adx"]
                self.adx_rising = adx_change >= self.params["adx_rising_threshold"]
                self.adx_falling = adx_change <= -self.params["adx_falling_threshold"]

            self.strong_trend = current_row["adx"] > self.params["min_adx_threshold"]

            # Analyze Directional Indicators
            di_diff = abs(current_row["di_plus"] - current_row["di_minus"])
            self.di_bullish = (
                current_row["di_plus"] > current_row["di_minus"]
                and di_diff >= self.params["di_separation"]
            )
            self.di_bearish = (
                current_row["di_minus"] > current_row["di_plus"]
                and di_diff >= self.params["di_separation"]
            )

            # Store indicator data for analysis
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": current_price,
                    "ema": current_row["ema"],
                    "adx": current_row["adx"],
                    "di_plus": current_row["di_plus"],
                    "di_minus": current_row["di_minus"],
                    "trend_direction": self.trend_direction,
                    "strong_trend": self.strong_trend,
                    "adx_rising": self.adx_rising,
                    "di_bullish": self.di_bullish,
                    "di_bearish": self.di_bearish,
                }
            )

            # Trading Logic
            if not self.open_positions:
                # Long Entry: Price above EMA + Strong ADX + Rising ADX + Bullish DI
                if (
                    self.trend_direction == 1
                    and self.strong_trend
                    and self.adx_rising
                    and self.di_bullish
                ):
                    self._place_order(idx, "buy", "enter_long")
                    last_signal = "BUY"

                    trade_logger.info(
                        f"BUY SIGNAL (Enter Long - EMA+ADX Trend) | Row: {idx} | "
                        f"Time: {bar_time_ist} | Price: {current_price:.2f} | "
                        f"EMA: {current_row['ema']:.2f} | ADX: {current_row['adx']:.2f} (Rising) | "
                        f"DI+: {current_row['di_plus']:.2f} > DI-: {current_row['di_minus']:.2f}"
                    )

                # Short Entry: Price below EMA + Strong ADX + Rising ADX + Bearish DI
                elif (
                    self.trend_direction == -1
                    and self.strong_trend
                    and self.adx_rising
                    and self.di_bearish
                ):
                    self._place_order(idx, "sell", "enter_short")
                    last_signal = "SELL"

                    trade_logger.info(
                        f"SELL SIGNAL (Enter Short - EMA+ADX Trend) | Row: {idx} | "
                        f"Time: {bar_time_ist} | Price: {current_price:.2f} | "
                        f"EMA: {current_row['ema']:.2f} | ADX: {current_row['adx']:.2f} (Rising) | "
                        f"DI-: {current_row['di_minus']:.2f} > DI+: {current_row['di_plus']:.2f}"
                    )

            elif self.open_positions[-1]["direction"] == "long":
                # Long Exit: Price below EMA OR Weak ADX OR Falling ADX
                if (
                    self.trend_direction != 1
                    or current_row["adx"] < self.params["exit_adx_threshold"]
                    or self.adx_falling
                ):
                    exit_reason = (
                        "Price below EMA"
                        if self.trend_direction != 1
                        else (
                            "Weak ADX"
                            if current_row["adx"] < self.params["exit_adx_threshold"]
                            else "ADX falling"
                        )
                    )

                    self._close_position(idx, exit_reason, "sell", "exit_long")
                    last_signal = "SELL"

                    trade_logger.info(
                        f"SELL SIGNAL (Exit Long - EMA+ADX Trend) | Row: {idx} | "
                        f"Time: {bar_time_ist} | Price: {current_price:.2f} | "
                        f"Reason: {exit_reason} | EMA: {current_row['ema']:.2f} | "
                        f"ADX: {current_row['adx']:.2f}"
                    )

            elif self.open_positions[-1]["direction"] == "short":
                # Short Exit: Price above EMA OR Weak ADX OR Falling ADX
                if (
                    self.trend_direction != -1
                    or current_row["adx"] < self.params["exit_adx_threshold"]
                    or self.adx_falling
                ):
                    exit_reason = (
                        "Price above EMA"
                        if self.trend_direction != -1
                        else (
                            "Weak ADX"
                            if current_row["adx"] < self.params["exit_adx_threshold"]
                            else "ADX falling"
                        )
                    )

                    self._close_position(idx, exit_reason, "buy", "exit_short")
                    last_signal = "BUY"

                    trade_logger.info(
                        f"BUY SIGNAL (Exit Short - EMA+ADX Trend) | Row: {idx} | "
                        f"Time: {bar_time_ist} | Price: {current_price:.2f} | "
                        f"Reason: {exit_reason} | EMA: {current_row['ema']:.2f} | "
                        f"ADX: {current_row['adx']:.2f}"
                    )

        return last_signal

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
                f"Price: {order['executed_price']:.2f} | Size: {order['size']} | "
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
                f"Price: {order['executed_price']:.2f} | Size: {order['size']} | "
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

        # If no action specified, determine based on position direction
        if action is None:
            action = "sell" if entry_info["direction"] == "long" else "buy"
            order_type = (
                "exit_long" if entry_info["direction"] == "long" else "exit_short"
            )

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
        exec_dt = order["executed_time"]
        if exec_dt.tzinfo is None:
            exec_dt = exec_dt.replace(tzinfo=pytz.UTC)

        if order["order_type"] == "exit_long" and order["action"] == "sell":
            entry_info = self.open_positions.pop()
            pnl = (order["executed_price"] - entry_info["entry_price"]) * abs(
                entry_info["size"]
            )
            total_commission = entry_info["commission"] + abs(order["commission"])
            pnl_net = pnl - total_commission
            trade_info = {
                "ref": order["ref"],
                "entry_time": entry_info["entry_time"],
                "exit_time": exec_dt,
                "entry_price": entry_info["entry_price"],
                "exit_price": order["executed_price"],
                "size": abs(entry_info["size"]),
                "pnl": pnl,
                "pnl_net": pnl_net,
                "commission": total_commission,
                "status": "Won" if pnl > 0 else "Lost",
                "direction": "Long",
                "bars_held": (exec_dt - entry_info["entry_time"]).total_seconds()
                / 300,  # 5-min bars
            }
            self.completed_trades.append(trade_info)
            self.trade_count += 1
            trade_logger.info(
                f"SELL EXECUTED (Exit Long) | Ref: {order['ref']} | "
                f"Price: {order['executed_price']:.2f} | PnL: {pnl:.2f} | Reason: {reason}"
            )
        elif order["order_type"] == "exit_short" and order["action"] == "buy":
            entry_info = self.open_positions.pop()
            pnl = (entry_info["entry_price"] - order["executed_price"]) * abs(
                entry_info["size"]
            )
            total_commission = entry_info["commission"] + abs(order["commission"])
            pnl_net = pnl - total_commission
            trade_info = {
                "ref": order["ref"],
                "entry_time": entry_info["entry_time"],
                "exit_time": exec_dt,
                "entry_price": entry_info["entry_price"],
                "exit_price": order["executed_price"],
                "size": abs(entry_info["size"]),
                "pnl": pnl,
                "pnl_net": pnl_net,
                "commission": total_commission,
                "status": "Won" if pnl > 0 else "Lost",
                "direction": "Short",
                "bars_held": (exec_dt - entry_info["entry_time"]).total_seconds()
                / 300,  # 5-min bars
            }
            self.completed_trades.append(trade_info)
            self.trade_count += 1
            trade_logger.info(
                f"BUY EXECUTED (Exit Short) | Ref: {order['ref']} | "
                f"Price: {order['executed_price']:.2f} | PnL: {pnl:.2f} | Reason: {reason}"
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
            "ema_period": trial.suggest_int("ema_period", 15, 30),
            "adx_period": trial.suggest_int("adx_period", 10, 20),
            "min_adx_threshold": trial.suggest_float("min_adx_threshold", 20.0, 35.0),
            "exit_adx_threshold": trial.suggest_float("exit_adx_threshold", 15.0, 25.0),
            "adx_rising_threshold": trial.suggest_float(
                "adx_rising_threshold", 0.5, 2.0
            ),
            "adx_falling_threshold": trial.suggest_float(
                "adx_falling_threshold", 2.0, 5.0
            ),
            "di_separation": trial.suggest_float("di_separation", 1.0, 5.0),
            "price_ema_buffer": trial.suggest_float("price_ema_buffer", 0.0, 0.5),
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
            ema_period = params.get("ema_period", 21)
            adx_period = params.get("adx_period", 14)
            max_period = max(ema_period, adx_period * 2)
            return max_period + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 40
