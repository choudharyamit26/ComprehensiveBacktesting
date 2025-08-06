import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


class PivotCCI(bt.Strategy):
    """
    Pivot Points + CCI Confluence Trading Strategy

    This strategy combines Pivot Point levels with the Commodity Channel Index (CCI)
    to identify high-probability trading opportunities. It enters positions when
    price approaches key pivot levels (support/resistance) and CCI confirms
    oversold/overbought conditions.

    Trading Logic:
    - Long Entry: Price near support (S1/S2) AND CCI < oversold threshold
    - Short Entry: Price near resistance (R1/R2) AND CCI > overbought threshold
    - Long Exit: Price reaches pivot/R1 OR CCI normalizes above exit threshold
    - Short Exit: Price reaches pivot/S1 OR CCI normalizes below exit threshold

    Key Features:
    - Intraday trading with IST timezone handling
    - Force close positions at 15:15 IST
    - Trading hours: 9:15 AM to 3:05 PM IST
    - Comprehensive trade tracking and logging
    - Parameter optimization support via Optuna

    Parameters:
    - cci_period (int): Period for CCI calculation (default: 14)
    - pivot_tolerance (float): Percentage tolerance for pivot proximity (default: 0.5%)
    - cci_oversold (int): CCI threshold for oversold condition (default: -100)
    - cci_overbought (int): CCI threshold for overbought condition (default: 100)
    - cci_exit (int): CCI threshold for position exit (default: 0)
    - verbose (bool): Enable detailed logging (default: False)
    """

    params = (
        ("cci_period", 14),
        ("pivot_tolerance", 0.5),  # 0.5% tolerance for pivot proximity
        ("cci_oversold", -100),
        ("cci_overbought", 100),
        ("cci_exit", 0),
        ("verbose", False),
    )

    optimization_params = {
        "cci_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "pivot_tolerance": {"type": "float", "low": 0.3, "high": 1.0, "step": 0.1},
        "cci_oversold": {"type": "int", "low": -150, "high": -80, "step": 5},
        "cci_overbought": {"type": "int", "low": 80, "high": 150, "step": 5},
        "cci_exit": {"type": "int", "low": -20, "high": 20, "step": 5},
    }

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        """
        Initialize the PivotCCI strategy.

        Sets up technical indicators, initializes tracking variables, and
        configures logging for trade analysis.

        Args:
            tickers (list, optional): List of ticker symbols (unused in current implementation)
            analyzers (list, optional): List of analyzers to attach (unused in current implementation)
            **kwargs: Additional keyword arguments
        """
        # Initialize CCI indicator
        self.cci = btind.CCI(self.data, period=self.params.cci_period)

        # Initialize pivot points
        self.pivot = btind.PivotPoint(self.data)

        # Debug: Log available lines and their types
        logger.debug(f"PivotPoint lines: {self.pivot.lines.getlinealiases()}")
        for line_name in self.pivot.lines.getlinealiases():
            logger.debug(
                f"Line {line_name} type: {type(getattr(self.pivot.lines, line_name))}"
            )

        # Order management
        self.order = None
        self.order_type = None

        # Strategy state tracking
        self.ready = False
        self.stable_count = 0
        self.trade_count = 0
        self.warmup_period = self.params.cci_period + 2

        # Data collection for analysis
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        logger.debug(f"Initialized PivotCCI with params: {self.params}")
        logger.info(
            f"PivotCCI initialized with cci_period={self.p.cci_period}, "
            f"pivot_tolerance={self.p.pivot_tolerance}, cci_oversold={self.p.cci_oversold}, "
            f"cci_overbought={self.p.cci_overbought}, cci_exit={self.p.cci_exit}"
        )

    def next(self):
        """
        Main strategy logic executed on each bar.

        This method implements the core trading logic:
        1. Validates warmup period and market hours
        2. Calculates pivot level proximity
        3. Evaluates entry and exit conditions
        4. Places orders based on CCI and pivot confluence
        5. Handles position management and risk control
        """
        # Skip bars during warmup period
        if len(self) < self.warmup_period:
            logger.debug(
                f"Skipping bar {len(self)}: still in warmup period (need {self.warmup_period} bars)"
            )
            return

        # Mark strategy as ready after warmup
        if not self.ready:
            self.ready = True
            logger.info(f"Strategy ready at bar {len(self)}")

        # Convert bar time to IST for market hours validation
        bar_time = self.datas[0].datetime.datetime(0)
        bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
        current_time = bar_time_ist.time()

        # Force close positions at 15:15 IST (end of trading)
        if current_time >= datetime.time(15, 15):
            if self.position:
                self.close()
                trade_logger.info("Force closed all positions at 15:15 IST")
            return

        # Only trade during market hours (9:15 AM to 3:05 PM IST)
        if not (datetime.time(9, 15) <= current_time <= datetime.time(15, 5)):
            return

        # Skip if there's a pending order
        if self.order:
            logger.debug(f"Order pending at bar {len(self)}")
            return

        # Validate indicator values (skip if NaN)
        if (
            np.isnan(self.cci[0])
            or np.isnan(self.pivot.lines.p[0])
            or np.isnan(self.pivot.lines.s1[0])
            or np.isnan(self.pivot.lines.r1[0])
        ):
            logger.debug(
                f"Invalid indicator values at bar {len(self)}: "
                f"CCI={self.cci[0]}, Pivot={self.pivot.lines.p[0]}, "
                f"S1={self.pivot.lines.s1[0]}, R1={self.pivot.lines.r1[0]}"
            )
            return

        # Get current market data
        current_price = self.data.close[0]
        pivot_levels = {
            "P": self.pivot.lines.p[0],
            "S1": self.pivot.lines.s1[0],
            "S2": self.pivot.lines.s2[0],
            "R1": self.pivot.lines.r1[0],
            "R2": self.pivot.lines.r2[0],
        }

        # Calculate tolerance for pivot level proximity
        tolerance = self.params.pivot_tolerance / 100 * current_price

        # Check if price is near support or resistance levels
        support_touch = (
            abs(current_price - pivot_levels["S1"]) <= tolerance
            or abs(current_price - pivot_levels["S2"]) <= tolerance
        )
        resistance_touch = (
            abs(current_price - pivot_levels["R1"]) <= tolerance
            or abs(current_price - pivot_levels["R2"]) <= tolerance
        )

        # Store data for post-strategy analysis
        self.indicator_data.append(
            {
                "date": bar_time_ist.strftime("%Y-%m-%d %H:%M:%S"),
                "close": current_price,
                "cci": self.cci[0],
                "pivot": pivot_levels["P"],
                "s1": pivot_levels["S1"],
                "s2": pivot_levels["S2"],
                "r1": pivot_levels["R1"],
                "r2": pivot_levels["R2"],
                "support_touch": support_touch,
                "resistance_touch": resistance_touch,
            }
        )

        # Execute trading logic based on position status
        if not self.position:
            self._check_entry_signals(
                current_price,
                pivot_levels,
                support_touch,
                resistance_touch,
                bar_time_ist,
            )
        elif self.position.size > 0:  # Long position
            self._check_long_exit(current_price, pivot_levels, bar_time_ist)
        elif self.position.size < 0:  # Short position
            self._check_short_exit(current_price, pivot_levels, bar_time_ist)

    def _check_entry_signals(
        self, current_price, pivot_levels, support_touch, resistance_touch, bar_time_ist
    ):
        """
        Check for entry signals when no position is held.

        Args:
            current_price (float): Current market price
            pivot_levels (dict): Dictionary of pivot levels (P, S1, S2, R1, R2)
            support_touch (bool): Whether price is near support levels
            resistance_touch (bool): Whether price is near resistance levels
            bar_time_ist (datetime): Current bar time in IST
        """
        # Long Entry: Price near support (S1/S2) AND CCI oversold
        if support_touch and self.cci[0] < self.params.cci_oversold:
            self.order = self.buy()
            self.order_type = "enter_long"
            trade_logger.info(
                f"BUY SIGNAL (Enter Long - Pivot + CCI) | Bar: {len(self)} | "
                f"Time: {bar_time_ist} | "
                f"Price: {current_price:.2f} | "
                f"CCI: {self.cci[0]:.2f} < {self.params.cci_oversold} (Oversold) | "
                f"Support Level: {pivot_levels['S1']:.2f} or {pivot_levels['S2']:.2f} (Touch)"
            )
        # Short Entry: Price near resistance (R1/R2) AND CCI overbought
        elif resistance_touch and self.cci[0] > self.params.cci_overbought:
            self.order = self.sell()
            self.order_type = "enter_short"
            trade_logger.info(
                f"SELL SIGNAL (Enter Short - Pivot + CCI) | Bar: {len(self)} | "
                f"Time: {bar_time_ist} | "
                f"Price: {current_price:.2f} | "
                f"CCI: {self.cci[0]:.2f} > {self.params.cci_overbought} (Overbought) | "
                f"Resistance Level: {pivot_levels['R1']:.2f} or {pivot_levels['R2']:.2f} (Touch)"
            )

    def _check_long_exit(self, current_price, pivot_levels, bar_time_ist):
        """
        Check for long position exit signals.

        Args:
            current_price (float): Current market price
            pivot_levels (dict): Dictionary of pivot levels (P, S1, S2, R1, R2)
            bar_time_ist (datetime): Current bar time in IST
        """
        # Long Exit: Price reaches next pivot level (P or R1) OR CCI normalizes
        if (
            current_price >= pivot_levels["P"]
            or current_price >= pivot_levels["R1"]
            or self.cci[0] > self.params.cci_exit
        ):
            self.order = self.sell()
            self.order_type = "exit_long"
            exit_reason = (
                "Price reached pivot or R1"
                if current_price >= pivot_levels["P"]
                or current_price >= pivot_levels["R1"]
                else "CCI normalized"
            )
            trade_logger.info(
                f"SELL SIGNAL (Exit Long - Pivot + CCI) | Bar: {len(self)} | "
                f"Time: {bar_time_ist} | "
                f"Price: {current_price:.2f} | "
                f"Reason: {exit_reason} | "
                f"CCI: {self.cci[0]:.2f} | "
                f"Pivot: {pivot_levels['P']:.2f}, R1: {pivot_levels['R1']:.2f}"
            )

    def _check_short_exit(self, current_price, pivot_levels, bar_time_ist):
        """
        Check for short position exit signals.

        Args:
            current_price (float): Current market price
            pivot_levels (dict): Dictionary of pivot levels (P, S1, S2, R1, R2)
            bar_time_ist (datetime): Current bar time in IST
        """
        # Short Exit: Price reaches next pivot level (P or S1) OR CCI normalizes
        if (
            current_price <= pivot_levels["P"]
            or current_price <= pivot_levels["S1"]
            or self.cci[0] < self.params.cci_exit
        ):
            self.order = self.buy()
            self.order_type = "exit_short"
            exit_reason = (
                "Price reached pivot or S1"
                if current_price <= pivot_levels["P"]
                or current_price <= pivot_levels["S1"]
                else "CCI normalized"
            )
            trade_logger.info(
                f"BUY SIGNAL (Exit Short - Pivot + CCI) | Bar: {len(self)} | "
                f"Time: {bar_time_ist} | "
                f"Price: {current_price:.2f} | "
                f"Reason: {exit_reason} | "
                f"CCI: {self.cci[0]:.2f} | "
                f"Pivot: {pivot_levels['P']:.2f}, S1: {pivot_levels['S1']:.2f}"
            )

    def notify_order(self, order):
        """
        Handle order status notifications from the broker.

        This method processes order executions, tracks positions, and logs
        trade information for analysis. It handles both entry and exit orders
        for long and short positions.

        Args:
            order: Backtrader order object containing execution details
        """
        if order.status in [order.Completed]:
            # Convert execution time to timezone-aware datetime
            exec_dt = bt.num2date(order.executed.dt)
            if exec_dt.tzinfo is None:
                exec_dt = exec_dt.replace(tzinfo=pytz.UTC)

            if self.order_type == "enter_long" and order.isbuy():
                self._handle_long_entry(order, exec_dt)
            elif self.order_type == "enter_short" and order.issell():
                self._handle_short_entry(order, exec_dt)
            elif self.order_type == "exit_long" and order.issell():
                self._handle_long_exit(order, exec_dt)
            elif self.order_type == "exit_short" and order.isbuy():
                self._handle_short_exit(order, exec_dt)

        # Clean up order references when order is finished
        if order.status in [
            order.Completed,
            order.Canceled,
            order.Margin,
            order.Rejected,
        ]:
            self.order = None
            self.order_type = None

    def _handle_long_entry(self, order, exec_dt):
        """
        Handle long position entry order execution.

        Args:
            order: Executed buy order
            exec_dt: Execution datetime
        """
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

    def _handle_short_entry(self, order, exec_dt):
        """
        Handle short position entry order execution.

        Args:
            order: Executed sell order
            exec_dt: Execution datetime
        """
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

    def _handle_long_exit(self, order, exec_dt):
        """
        Handle long position exit order execution and calculate P&L.

        Args:
            order: Executed sell order
            exec_dt: Execution datetime
        """
        if self.open_positions:
            entry_info = self.open_positions.pop(0)
            pnl = (order.executed.price - entry_info["entry_price"]) * abs(
                entry_info["size"]
            )
            total_commission = entry_info["commission"] + abs(order.executed.comm)
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

    def _handle_short_exit(self, order, exec_dt):
        """
        Handle short position exit order execution and calculate P&L.

        Args:
            order: Executed buy order
            exec_dt: Execution datetime
        """
        if self.open_positions:
            entry_info = self.open_positions.pop(0)
            pnl = (entry_info["entry_price"] - order.executed.price) * abs(
                entry_info["size"]
            )
            total_commission = entry_info["commission"] + abs(order.executed.comm)
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

    def notify_trade(self, trade):
        """
        Handle trade completion notifications.

        This method is called when a complete trade (entry + exit) is closed.
        It provides additional logging for trade analysis.

        Args:
            trade: Backtrader trade object containing trade details
        """
        if trade.isclosed:
            trade_logger.info(
                f"TRADE CLOSED | Ref: {trade.ref} | "
                f"Profit: {trade.pnl:.2f} | "
                f"Net Profit: {trade.pnlcomm:.2f} | "
                f"Bars Held: {trade.barlen} | "
                f"Trade Count: {self.trade_count}"
            )

    def get_completed_trades(self):
        """
        Get a copy of all completed trades for analysis.

        Returns:
            list: List of dictionaries containing trade information including
                  entry/exit times, prices, P&L, commissions, etc.
        """
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        """
        Define parameter space for Optuna optimization.

        This method defines the search space for hyperparameter optimization
        using Optuna. It suggests parameter values within predefined ranges
        for strategy optimization.

        Args:
            trial: Optuna trial object for parameter suggestion

        Returns:
            dict: Dictionary of suggested parameter values
        """
        params = {
            "cci_period": trial.suggest_int("cci_period", 10, 20),
            "pivot_tolerance": trial.suggest_float("pivot_tolerance", 0.3, 1.0),
            "cci_oversold": trial.suggest_int("cci_oversold", -150, -80),
            "cci_overbought": trial.suggest_int("cci_overbought", 80, 150),
            "cci_exit": trial.suggest_int("cci_exit", -20, 20),
        }
        return params

    @classmethod
    def get_min_data_points(cls, params):
        """
        Calculate minimum required data points for strategy initialization.

        This method determines the minimum number of historical bars needed
        before the strategy can start making trading decisions, based on
        the indicator periods and warmup requirements.

        Args:
            params (dict): Strategy parameters dictionary

        Returns:
            int: Minimum number of data points required
        """
        try:
            cci_period = params.get("cci_period", 14)
            return cci_period + 2
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 30
