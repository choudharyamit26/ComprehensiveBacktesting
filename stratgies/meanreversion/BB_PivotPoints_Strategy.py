import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pytz
import datetime
import logging

# Set up loggers
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")


# Custom Pivot Point Indicator
class PivotPoint(bt.Indicator):
    lines = ("pivot", "s1", "s2", "r1", "r2")

    def __init__(self):
        high = self.data.high(-1)
        low = self.data.low(-1)
        close = self.data.close(-1)
        pivot = (high + low + close) / 3

        self.lines.pivot = pivot
        self.lines.r1 = (2 * pivot) - low
        self.lines.s1 = (2 * pivot) - high
        self.lines.r2 = pivot + (high - low)
        self.lines.s2 = pivot - (high - low)


class BBPivotPointsStrategy(bt.Strategy):
    """
    Bollinger Bands + Pivot Points Strategy

    This strategy combines Bollinger Bands (BB) levels with Pivot Points to identify trades
    where BB levels align with key support/resistance levels.

    Strategy Type: MEAN REVERSION
    ==========================================
    This strategy enters trades when price is near a BB level and a pivot point, confirming
    strong support/resistance zones. It exits at the next pivot level or BB reversal.

    Strategy Logic:
    ==============
    Long Position Rules:
    - Entry: Price near lower BB AND close to pivot support (S1 or S2)
    - Exit: Price reaches next pivot level (pivot or R1) OR upper BB

    Short Position Rules:
    - Entry: Price near upper BB AND close to pivot resistance (R1 or R2)
    - Exit: Price reaches next pivot level (pivot or S1) OR lower BB

    Risk Management:
    ===============
    - Operates only during Indian market hours (9:15 AM - 3:05 PM IST)
    - Force closes all positions at 3:15 PM IST
    - Uses warmup period for indicator stability
    - Prevents order overlap

    Indicators Used:
    ===============
    - Bollinger Bands: Middle Band (20-period SMA), Upper/Lower Bands (±2 SD)
    - Pivot Points: Daily levels (Pivot, S1, S2, R1, R2)

    Parameters:
    ==========
    - bb_period (int): Bollinger Bands period (default: 20)
    - bb_dev (float): BB standard deviation multiplier (default: 2.0)
    - pivot_proximity (float): Proximity to pivot level for entry (default: 0.5%)
    - verbose (bool): Enable detailed logging (default: False)

    Usage:
    ======
    cerebro = bt.Cerebro()
    cerebro.addstrategy(BBPivotPointsStrategy, bb_period=20, pivot_proximity=0.5)
    cerebro.run()

    Best Market Conditions:
    ======================
    - Markets with clear support/resistance levels
    - Avoid low volatility or trending markets without clear pivots
    """

    params = (
        ("bb_period", 20),
        ("bb_dev", 2.0),
        ("pivot_proximity", 0.5),
        ("verbose", False),
    )

    def __init__(self):
        self.data_intraday = self.datas[0]  # only one data feed
        self.bb = btind.BollingerBands(
            self.data_intraday.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_dev,
        )

        self.order = None
        self.order_type = None
        self.ready = False
        self.trade_count = 0
        self.warmup_period = self.params.bb_period + 5
        self.indicator_data = []
        self.completed_trades = []
        self.open_positions = []

        # Variables to hold previous day's OHLC
        self.prev_day = None
        self.day_high = float("-inf")
        self.day_low = float("inf")
        self.day_close = None
        self.pivot_p = self.s1 = self.s2 = self.r1 = self.r2 = None

    def next(self):
        dt = self.data_intraday.datetime.datetime(0)
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        dt_ist = dt.astimezone(pytz.timezone("Asia/Kolkata"))
        current_time = dt_ist.time()
        current_date = dt_ist.date()

        # Warmup check
        if len(self) < self.warmup_period:
            return

        if self.prev_day != current_date:
            # New day detected → calculate yesterday's pivot levels
            if self.day_close is not None:
                high = self.day_high
                low = self.day_low
                close = self.day_close
                pivot = (high + low + close) / 3
                self.pivot_p = pivot
                self.r1 = (2 * pivot) - low
                self.s1 = (2 * pivot) - high
                self.r2 = pivot + (high - low)
                self.s2 = pivot - (high - low)

            # Reset daily values
            self.prev_day = current_date
            self.day_high = float("-inf")
            self.day_low = float("inf")

        # Update today's OHLC tracking
        self.day_high = max(self.day_high, self.data_intraday.high[0])
        self.day_low = min(self.day_low, self.data_intraday.low[0])
        self.day_close = self.data_intraday.close[0]

        # Force exit at 15:15
        if current_time >= datetime.time(15, 15):
            if self.getposition(data=self.data_intraday).size:
                self.close(data=self.data_intraday)
                trade_logger.info("Force closed all positions at 15:15 IST")
            return

        if not (datetime.time(9, 15) <= current_time <= datetime.time(15, 5)):
            return
        if self.order:
            return
        if self.pivot_p is None:
            return  # skip day if no pivot available yet

        # Access indicators
        bb_top = self.bb.top[0]
        bb_bot = self.bb.bot[0]
        current_price = self.data_intraday.close[0]

        proximity_threshold = self.params.pivot_proximity / 100
        price_near_lower_bb = current_price <= bb_bot * (1 + proximity_threshold)
        price_near_upper_bb = current_price >= bb_top * (1 - proximity_threshold)

        near_s1 = (
            abs(current_price - self.s1) / abs(self.s1) < proximity_threshold
            if self.s1
            else False
        )
        near_s2 = (
            abs(current_price - self.s2) / abs(self.s2) < proximity_threshold
            if self.s2
            else False
        )
        near_r1 = (
            abs(current_price - self.r1) / abs(self.r1) < proximity_threshold
            if self.r1
            else False
        )
        near_r2 = (
            abs(current_price - self.r2) / abs(self.r2) < proximity_threshold
            if self.r2
            else False
        )

        bullish_entry = price_near_lower_bb and (near_s1 or near_s2)
        bearish_entry = price_near_upper_bb and (near_r1 or near_r2)
        bullish_exit = current_price >= self.pivot_p or current_price >= bb_top
        bearish_exit = current_price <= self.pivot_p or current_price <= bb_bot

        position_size = self.getposition(data=self.data_intraday).size
        if position_size == 0:
            if bullish_entry:
                self.order = self.buy(data=self.data_intraday)
                self.order_type = "enter_long"
                trade_logger.info(
                    f"BUY SIGNAL | Time: {dt_ist} | Price: {current_price:.2f}"
                )
            elif bearish_entry:
                self.order = self.sell(data=self.data_intraday)
                self.order_type = "enter_short"
                trade_logger.info(
                    f"SELL SIGNAL | Time: {dt_ist} | Price: {current_price:.2f}"
                )
        else:
            if position_size > 0 and bullish_exit:
                self.order = self.sell(data=self.data_intraday, size=position_size)
                self.order_type = "exit_long"
                trade_logger.info(
                    f"EXIT LONG | Time: {dt_ist} | Price: {current_price:.2f}"
                )
            elif position_size < 0 and bearish_exit:
                self.order = self.buy(data=self.data_intraday, size=abs(position_size))
                self.order_type = "exit_short"
                trade_logger.info(
                    f"EXIT SHORT | Time: {dt_ist} | Price: {current_price:.2f}"
                )

    def notify_order(self, order):
        if order.status in [order.Completed]:
            exec_dt = bt.num2date(order.executed.dt).astimezone(
                pytz.timezone("Asia/Kolkata")
            )
            if self.order_type == "enter_long" and order.isbuy():
                self.open_positions.append(
                    {
                        "entry_time": exec_dt,
                        "entry_price": order.executed.price,
                        "size": order.executed.size,
                        "commission": order.executed.comm,
                        "ref": order.ref,
                        "direction": "long",
                    }
                )
                trade_logger.info(
                    f"BUY EXECUTED | Ref: {order.ref} | Price: {order.executed.price:.2f}"
                )
            elif self.order_type == "enter_short" and order.issell():
                self.open_positions.append(
                    {
                        "entry_time": exec_dt,
                        "entry_price": order.executed.price,
                        "size": order.executed.size,
                        "commission": order.executed.comm,
                        "ref": order.ref,
                        "direction": "short",
                    }
                )
                trade_logger.info(
                    f"SELL EXECUTED | Ref: {order.ref} | Price: {order.executed.price:.2f}"
                )
            elif self.order_type == "exit_long" and order.issell():
                entry_info = self.open_positions.pop(0)
                pnl = (order.executed.price - entry_info["entry_price"]) * abs(
                    entry_info["size"]
                )
                total_comm = entry_info["commission"] + abs(order.executed.comm)
                self.completed_trades.append(
                    {
                        "ref": order.ref,
                        "entry_time": entry_info["entry_time"],
                        "exit_time": exec_dt,
                        "entry_price": entry_info["entry_price"],
                        "exit_price": order.executed.price,
                        "size": abs(entry_info["size"]),
                        "pnl": pnl,
                        "pnl_net": pnl - total_comm,
                        "commission": total_comm,
                        "status": "Won" if pnl > 0 else "Lost",
                        "direction": "Long",
                    }
                )
                self.trade_count += 1
                trade_logger.info(
                    f"EXIT LONG EXECUTED | Ref: {order.ref} | PnL: {pnl:.2f}"
                )
            elif self.order_type == "exit_short" and order.isbuy():
                entry_info = self.open_positions.pop(0)
                pnl = (entry_info["entry_price"] - order.executed.price) * abs(
                    entry_info["size"]
                )
                total_comm = entry_info["commission"] + abs(order.executed.comm)
                self.completed_trades.append(
                    {
                        "ref": order.ref,
                        "entry_time": entry_info["entry_time"],
                        "exit_time": exec_dt,
                        "entry_price": entry_info["entry_price"],
                        "exit_price": order.executed.price,
                        "size": abs(entry_info["size"]),
                        "pnl": pnl,
                        "pnl_net": pnl - total_comm,
                        "commission": total_comm,
                        "status": "Won" if pnl > 0 else "Lost",
                        "direction": "Short",
                    }
                )
                self.trade_count += 1
                trade_logger.info(
                    f"EXIT SHORT EXECUTED | Ref: {order.ref} | PnL: {pnl:.2f}"
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
                f"TRADE CLOSED | Ref: {trade.ref} | Profit: {trade.pnl:.2f} | Net: {trade.pnlcomm:.2f} | Held: {trade.barlen}"
            )

    def get_completed_trades(self):
        return self.completed_trades.copy()

    @classmethod
    def get_param_space(cls, trial):
        return {
            "bb_period": trial.suggest_int("bb_period", 10, 30),
            "bb_dev": trial.suggest_float("bb_dev", 1.5, 2.5, step=0.1),
            "pivot_proximity": trial.suggest_float(
                "pivot_proximity", 0.3, 1.0, step=0.1
            ),
        }

    @classmethod
    def get_min_data_points(cls, params):
        try:
            return params.get("bb_period", 20) + 5
        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 25
