import pandas as pd
import pandas_ta as ta
import numpy as np
import pytz
import datetime
import logging
from uuid import uuid4

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class BBVWAPStrategy:
    """
    Bollinger Bands + VWAP Strategy

    This strategy combines Bollinger Bands (BB) with Volume Weighted Average Price (VWAP) to identify
    high-probability intraday trades based on price extremes relative to BB and deviations from VWAP.

    Strategy Type: MEAN REVERSION
    =================================================
    This strategy uses BB to detect price extremes and VWAP to confirm directional bias.
    It enters trades when the price reaches a BB extreme and deviates significantly from VWAP,
    and exits when the price returns to VWAP or reaches the opposite BB.

    Strategy Logic:
    ==============
    Long Position Rules:
    - Entry: Price touches or crosses below lower BB AND price is below VWAP
    - Exit: Price returns to VWAP OR price reaches upper BB

    Short Position Rules:
    - Entry: Price touches or crosses above upper BB AND price is above VWAP
    - Exit: Price returns to VWAP OR price reaches lower BB

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST
    - Uses warmup period for indicator stability
    - Prevents order overlap
    - VWAP deviation filter reduces false signals

    Indicators Used:
    ===============
    - Bollinger Bands: Middle Band (20-period SMA), Upper/Lower Bands (±2 SD)
    - VWAP: Volume Weighted Average Price as dynamic support/resistance

    Parameters:
    ==========
    - bb_period (int): Bollinger Bands period (default: 20)
    - bb_dev (float): BB standard deviation multiplier (default: 2.0)
    - vwap_dev (float): Minimum VWAP deviation percentage (default: 0.5)
    - verbose (bool): Enable detailed logging (default: False)

    Usage:
    ======
    strategy = BBVWAPStrategy(data)
    strategy.run()

    Best Market Conditions:
    ======================
    - Range-bound markets with clear mean-reverting behavior
    - High intraday volatility
    - Avoid strong trending markets
    """

    params = {
        "bb_period": 20,
        "bb_dev": 2.0,
        "vwap_dev": 0.5,
        "verbose": False,
    }

    optimization_params = {
        "bb_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "bb_dev": {"type": "float", "low": 1.5, "high": 2.5, "step": 0.1},
        "vwap_dev": {"type": "float", "low": 0.3, "high": 1.0, "step": 0.1},
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
        self.trade_count = 0
        self.warmup_period = self.params["bb_period"] + 5
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        # Initialize indicators using pandas_ta
        bb = ta.bbands(
            self.data["close"],
            length=self.params["bb_period"],
            std=self.params["bb_dev"],
        )
        self.data["bb_top"] = bb[
            f"BBU_{self.params['bb_period']}_{self.params['bb_dev']}"
        ]
        self.data["bb_mid"] = bb[
            f"BBM_{self.params['bb_period']}_{self.params['bb_dev']}"
        ]
        self.data["bb_bot"] = bb[
            f"BBL_{self.params['bb_period']}_{self.params['bb_dev']}"
        ]

        # Calculate VWAP
        hlc = (self.data["high"] + self.data["low"] + self.data["close"]) / 3.0
        hlc_volume = hlc * self.data["volume"]
        self.data["vwap"] = (
            hlc_volume.rolling(window=self.params["bb_period"]).sum()
            / self.data["volume"].rolling(window=self.params["bb_period"]).sum()
        )

        # Define conditions
        self.data["price_below_lower_bb"] = self.data["close"] <= self.data["bb_bot"]
        self.data["price_above_upper_bb"] = self.data["close"] >= self.data["bb_top"]
        self.data["price_below_vwap"] = self.data["close"] < self.data["vwap"] * (
            1 - self.params["vwap_dev"] / 100
        )
        self.data["price_above_vwap"] = self.data["close"] > self.data["vwap"] * (
            1 + self.params["vwap_dev"] / 100
        )
        self.data["bullish_entry"] = (
            self.data["price_below_lower_bb"] & self.data["price_below_vwap"]
        )
        self.data["bearish_entry"] = (
            self.data["price_above_upper_bb"] & self.data["price_above_vwap"]
        )
        self.data["bullish_exit"] = (self.data["close"] >= self.data["vwap"]) | (
            self.data["close"] >= self.data["bb_top"]
        )
        self.data["bearish_exit"] = (self.data["close"] <= self.data["vwap"]) | (
            self.data["close"] <= self.data["bb_bot"]
        )

        logger.debug(f"Initialized BBVWAPStrategy with params: {self.params}")
        logger.info(
            f"BBVWAPStrategy initialized with bb_period={self.params['bb_period']}, "
            f"bb_dev={self.params['bb_dev']}, vwap_dev={self.params['vwap_dev']}"
        )

    def run(self):
        """
        Execute the strategy on the provided data.
        """
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
            if pd.isna(self.data.iloc[idx]["bb_mid"]) or pd.isna(
                self.data.iloc[idx]["vwap"]
            ):
                logger.debug(f"Invalid indicator values at row {idx}")
                continue

            # Store indicator data for analysis
            self.indicator_data.append(
                {
                    "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "close": self.data.iloc[idx]["close"],
                    "bb_top": self.data.iloc[idx]["bb_top"],
                    "bb_bot": self.data.iloc[idx]["bb_bot"],
                    "vwap": self.data.iloc[idx]["vwap"],
                }
            )

            # Trading Logic
            if not self.open_positions:
                if self.data.iloc[idx]["bullish_entry"]:
                    self._place_order(idx, "buy", "enter_long")
                    trade_logger.info(
                        f"BUY SIGNAL (Lower BB + Below VWAP) | Time: {bar_time_ist} | "
                        f"Price: {self.data.iloc[idx]['close']:.2f}"
                    )
                elif self.data.iloc[idx]["bearish_entry"]:
                    self._place_order(idx, "sell", "enter_short")
                    trade_logger.info(
                        f"SELL SIGNAL (Upper BB + Above VWAP) | Time: {bar_time_ist} | "
                        f"Price: {self.data.iloc[idx]['close']:.2f}"
                    )
            elif (
                self.open_positions[-1]["direction"] == "long"
                and self.data.iloc[idx]["bullish_exit"]
            ):
                self._close_position(idx, "Exit Long", "sell", "exit_long")
                trade_logger.info(
                    f"SELL SIGNAL (Exit Long) | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f}"
                )
            elif (
                self.open_positions[-1]["direction"] == "short"
                and self.data.iloc[idx]["bearish_exit"]
            ):
                self._close_position(idx, "Exit Short", "buy", "exit_short")
                trade_logger.info(
                    f"BUY SIGNAL (Exit Short) | Time: {bar_time_ist} | Price: {self.data.iloc[idx]['close']:.2f}"
                )

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
                f"BUY EXECUTED (Enter Long) | Ref: {order['ref']} | Price: {order['executed_price']:.2f}"
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
                f"SELL EXECUTED (Enter Short) | Ref: {order['ref']} | Price: {order['executed_price']:.2f}"
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
            trade_info = {
                "ref": order["ref"],
                "entry_time": entry_info["entry_time"],
                "exit_time": order["executed_time"],
                "entry_price": entry_info["entry_price"],
                "exit_price": order["executed_price"],
                "size": abs(entry_info["size"]),
                "pnl": pnl,
                "pnl_net": pnl - total_commission,
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
                f"SELL EXECUTED (Exit Long) | Ref: {order['ref']} | PnL: {pnl:.2f} | Reason: {reason}"
            )
        elif order["order_type"] == "exit_short" and order["action"] == "buy":
            entry_info = self.open_positions.pop(0)
            pnl = (entry_info["entry_price"] - order["executed_price"]) * abs(
                entry_info["size"]
            )
            total_commission = entry_info["commission"] + abs(order["commission"])
            trade_info = {
                "ref": order["ref"],
                "entry_time": entry_info["entry_time"],
                "exit_time": order["executed_time"],
                "entry_price": entry_info["entry_price"],
                "exit_price": order["executed_price"],
                "size": abs(entry_info["size"]),
                "pnl": pnl,
                "pnl_net": pnl - total_commission,
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
                f"BUY EXECUTED (Exit Short) | Ref: {order['ref']} | PnL: {pnl:.2f} | Reason: {reason}"
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
            "bb_period": trial.suggest_int("bb_period", 10, 30),
            "bb_dev": trial.suggest_float("bb_dev", 1.5, 2.5, step=0.1),
            "vwap_dev": trial.suggest_float("vwap_dev", 0.3, 1.0, step=0.1),
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
            bb_period = params.get("bb_period", 20)
            return bb_period + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 25


# if __name__ == "__main__":
#     ticker = 1922
#     data = pd.read_csv(f"combined_data_{ticker}.csv")
#     strategy = BBVWAPStrategy(data)
#     strategy.run()
#     orders = strategy.completed_trades
#     print("================")
#     if len(orders) > 0:
#         orders_df = pd.DataFrame(orders)
#         orders_df.to_csv(f"bb_orders_{ticker}.csv", index=False)
#         print("orders saved to bb_orders_{ticker}.csv")
#         print("=================")
#     else:
#         print("No orders to save.")
#         print("=================")
