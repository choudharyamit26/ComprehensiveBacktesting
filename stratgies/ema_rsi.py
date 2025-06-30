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
        ("rsi_upper", 65),  # Relaxed from 70
        ("rsi_lower", 35),  # Relaxed from 30
    )

    def __init__(self):
        self.fast_ema = btind.EMA(self.data.close, period=self.params.fast_ema_period)
        self.slow_ema = btind.EMA(self.data.close, period=self.params.slow_ema_period)
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.order = None
        self.ready = False
        self.stable_count = 0
        logger.debug(f"Initialized EMARSI with params: {self.params}")

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

        # Force close all positions at 15:25 IST
        if current_time >= datetime.time(15, 15):
            if self.position:
                self.close()
                logger.debug("Force closed all positions at 15:25 IST")
            return

        # Only allow new trades between 09:15 and 15:05 IST
        if not (datetime.time(9, 15) <= current_time < datetime.time(15, 5)):
            return

        if self.order:
            return

        if not self.position:
            if (
                self.fast_ema[0] > self.slow_ema[0]
                and self.rsi[0] < self.params.rsi_lower
            ):
                self.order = self.buy()
                logger.debug(
                    f"Buy signal: FastEMA {self.fast_ema[0]:.2f} > "
                    f"SlowEMA {self.slow_ema[0]:.2f}, "
                    f"RSI {self.rsi[0]:.2f} < {self.params.rsi_lower}"
                )
        else:
            if (
                self.fast_ema[0] < self.slow_ema[0]
                or self.rsi[0] > self.params.rsi_upper
            ):
                self.order = self.sell()
                logger.debug(
                    f"Sell signal: FastEMA {self.fast_ema[0]:.2f} < "
                    f"SlowEMA {self.slow_ema[0]:.2f} or "
                    f"RSI {self.rsi[0]:.2f} > {self.params.rsi_upper}"
                )

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
        logger.debug("Defining parameter space for EMARSI")
        params = {
            "fast_ema_period": trial.suggest_int("fast_ema_period", 5, 20),
            "slow_ema_period": trial.suggest_int("slow_ema_period", 21, 50),
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "rsi_upper": trial.suggest_int("rsi_upper", 60, 75),  # Wider range
            "rsi_lower": trial.suggest_int("rsi_lower", 25, 40),  # Wider range
        }
        logger.debug(f"Parameter space: {params}")
        return params

    @classmethod
    def get_min_data_points(cls, params):
        """Calculate minimum data points required for the strategy."""
        try:
            fast_ema_period = params.get("fast_ema_period", 12)
            slow_ema_period = params.get("slow_ema_period", 26)
            rsi_period = params.get("rsi_period", 14)

            # Ensure slow_ema_period > fast_ema_period
            if slow_ema_period <= fast_ema_period:
                logger.warning(
                    f"Adjusting slow_ema_period ({slow_ema_period}) to be "
                    f"greater than fast_ema_period ({fast_ema_period})"
                )
                slow_ema_period = fast_ema_period + 5

            # Calculate minimum data points with reduced buffer
            max_period = max(fast_ema_period, slow_ema_period, rsi_period)
            min_data_points = max_period + 20  # Reduced buffer
            logger.debug(
                f"Min data points: {min_data_points} (fast:{fast_ema_period} "
                f"slow:{slow_ema_period} rsi:{rsi_period})"
            )
            return min_data_points

        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 50  # Conservative fallback
