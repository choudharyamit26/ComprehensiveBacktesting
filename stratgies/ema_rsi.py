import backtrader as bt
import backtrader.indicators as btind
import logging

logger = logging.getLogger(__name__)


class EMARSI(bt.Strategy):
    """EMA and RSI combined trading strategy."""

    params = (
        ("fast_ema_period", 12),
        ("slow_ema_period", 26),
        ("rsi_period", 14),
        ("rsi_upper", 70),
        ("rsi_lower", 30),
    )

    def __init__(self):
        self.fast_ema = btind.EMA(self.data.close, period=self.params.fast_ema_period)
        self.slow_ema = btind.EMA(self.data.close, period=self.params.slow_ema_period)
        self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)
        self.order = None
        logger.debug(f"Initialized EMARSI with params: {self.params}")

    def next(self):
        if self.order:
            return

        if not self.position:
            if (
                self.fast_ema[0] > self.slow_ema[0]
                and self.rsi[0] < self.params.rsi_lower
            ):
                self.order = self.buy()
                logger.debug("Buy signal triggered")
        else:
            if (
                self.fast_ema[0] < self.slow_ema[0]
                or self.rsi[0] > self.params.rsi_upper
            ):
                self.order = self.sell()
                logger.debug("Sell signal triggered")

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
        """Define the parameter space for optimization.

        Args:
            trial: Optuna trial object.

        Returns:
            dict: Parameter space for optimization.
        """
        logger.debug("Defining parameter space for EMARSI")
        params = {
            "fast_ema_period": trial.suggest_int("fast_ema_period", 5, 20),
            "slow_ema_period": trial.suggest_int("slow_ema_period", 21, 50),
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "rsi_upper": trial.suggest_int("rsi_upper", 60, 80),
            "rsi_lower": trial.suggest_int("rsi_lower", 20, 40),
        }
        logger.debug(f"Parameter space: {params}")
        return params

    @classmethod
    def get_min_data_points(cls, params):
        """Calculate minimum data points required for the strategy.

        Args:
            params (dict): Strategy parameters.

        Returns:
            int: Minimum number of data points.
        """
        try:
            # Use params dictionary directly with defaults
            fast_ema_period = params.get("fast_ema_period", 12)
            slow_ema_period = params.get("slow_ema_period", 26)
            rsi_period = params.get("rsi_period", 14)

            # Ensure slow_ema_period is greater than fast_ema_period
            if slow_ema_period <= fast_ema_period:
                logger.warning(
                    f"Invalid params: slow_ema_period ({slow_ema_period}) <= fast_ema_period ({fast_ema_period})"
                )
                return 100

            # Calculate minimum data points (e.g., max period + buffer for indicator stabilization)
            max_period = max(fast_ema_period, slow_ema_period, rsi_period)
            min_data_points = max_period + 100  # Buffer for EMA/RSI stabilization
            logger.debug(
                f"Calculated min_data_points: {min_data_points} (fast_ema: {fast_ema_period}, slow_ema: {slow_ema_period}, rsi: {rsi_period})"
            )
            return min_data_points

        except Exception as e:
            logger.error(f"Error calculating min_data_points: {str(e)}")
            return 100  # Fallback to default
