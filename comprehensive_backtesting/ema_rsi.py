import backtrader as bt
import backtrader.indicators as btind
import logging
import datetime
import pytz
import numpy as np

logger = logging.getLogger(__name__)


class EMARSI(bt.Strategy):
    """EMA and RSI combined trading strategy with dynamic warmup."""

    params = (
        ("fast_ema_period", 12),
        ("slow_ema_period", 26),
        ("rsi_period", 14),
        ("rsi_upper", 60),  # Relaxed from 70
        ("rsi_lower", 40),  # Relaxed from 30
    )

    def __init__(self, tickers=None, analyzers=None, **kwargs):
        self.fast_ema = btind.EMA(self.data.close, period=self.params.fast_ema_period)
        self.slow_ema = btind.EMA(self.data.close, period=self.params.slow_ema_period)
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.order = None
        self.ready = False
        self.stable_count = 0
        logger.debug(f"Initialized EMARSI with params: {self.params}")
        self.is_intraday = self.data._timeframe != bt.TimeFrame.Days

    def next(self):
        # Skip processing until indicators are stable
        if not self.ready:
            if (
                not np.isnan(self.fast_ema[0])
                and not np.isnan(self.slow_ema[0])
                and not np.isnan(self.rsi[0])
            ):
                self.stable_count += 1
            else:
                self.stable_count = 0

            if self.stable_count >= 5:
                self.ready = True
                logger.info(f"Strategy ready at bar {len(self)}")
            else:
                return

        # Get current bar time in IST
        bar_time = self.datas[0].datetime.datetime(0)
        bar_time_ist = bar_time.astimezone(pytz.timezone("Asia/Kolkata"))
        current_time = bar_time_ist.time()

        # logger.info(
        #     f"Bar {len(self)} | Time (IST): {bar_time_ist} | FastEMA: {self.fast_ema[0]:.2f} | "
        #     f"SlowEMA: {self.slow_ema[0]:.2f} | RSI: {self.rsi[0]:.2f} | Position: {self.position.size if self.position else 0}"
        # )

        # Force close all positions at 15:15 IST
        if self.is_intraday and current_time >= datetime.time(15, 15):
            if self.position:
                self.close()
                # logger.info("Force closed all positions at 15:15 IST")
            # else:
            #     # logger.info("No position to close at 15:15 IST")
            # return

        # Only allow new trades between 09:15 and 15:05 IST
        # if not (datetime.time(9, 15) <= current_time < datetime.time(15, 5)):
        #     logger.info(f"Bar {len(self)} skipped due to time window: {current_time}")
        #     return

        if self.order:
            logger.info(f"Order pending at bar {len(self)}; skipping new signal.")
            return

        if not self.position:
            if (
                self.fast_ema[0] > self.slow_ema[0]
                and self.rsi[0] < self.params.rsi_lower
            ):
                self.order = self.buy()
                # logger.info(
                #     f"Buy signal: FastEMA {self.fast_ema[0]:.2f} > "
                #     f"SlowEMA {self.slow_ema[0]:.2f}, "
                #     f"RSI {self.rsi[0]:.2f} < {self.params.rsi_lower}"
                # )
        else:
            if (
                self.fast_ema[0] < self.slow_ema[0]
                or self.rsi[0] > self.params.rsi_upper
            ):
                self.order = self.sell()
                # logger.info(
                #     f"Sell signal: FastEMA {self.fast_ema[0]:.2f} < "
                #     f"SlowEMA {self.slow_ema[0]:.2f} or "
                #     f"RSI {self.rsi[0]:.2f} > {self.params.rsi_upper}"
                # )

    def notify_order(self, order):
        if order.status in [
            order.Completed,
            order.Canceled,
            order.Margin,
            order.Rejected,
        ]:
            self.order = None

    @classmethod
    def get_param_space(cls, trial):
        """Define the parameter space for optimization."""
        # logger.debug("Defining parameter space for EMARSI")
        params = {
            "fast_ema_period": trial.suggest_int("fast_ema_period", 5, 20),
            "slow_ema_period": trial.suggest_int("slow_ema_period", 21, 50),
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "rsi_upper": trial.suggest_int("rsi_upper", 60, 75),  # Wider range
            "rsi_lower": trial.suggest_int("rsi_lower", 25, 40),  # Wider range
        }
        # logger.debug(f"Parameter space: {params}")
        return params

    @classmethod
    def get_min_data_points(cls, params, interval=None):
        """Calculate minimum data points required with dynamic buffer."""
        # Get parameters or use defaults
        fast_ema = params.get("fast_ema_period", 12)
        slow_ema = params.get("slow_ema_period", 26)
        rsi_period = params.get("rsi_period", 14)

        # Ensure slow EMA > fast EMA
        slow_ema = max(slow_ema, fast_ema + 5)

        # Determine the longest indicator period
        max_period = max(fast_ema, slow_ema, rsi_period)

        # Calculate buffer dynamically (20% of max period but at least 5)
        buffer = max(5, int(max_period * 0.2))

        # For daily data, cap at 20 to prevent excessive requirements
        if interval == "1d":
            return min(max_period + buffer, 20)
        return max_period + buffer
