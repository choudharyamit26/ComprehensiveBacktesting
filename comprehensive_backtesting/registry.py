import logging
import backtrader as bt


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STRATEGY_REGISTRY = {}


def register_strategy(name: str, strategy_class):
    """Register a new strategy in the framework with multiple aliases.

    Args:
        name (str): Unique name for the strategy.
        strategy_class: Backtrader strategy class to register.

    Raises:
        ValueError: If the name is already registered or strategy_class is invalid.

    Example:
        >>> from ema_rsi import EMARSI
        >>> register_strategy("EMARSI", EMARSI)
    """
    import re

    if not isinstance(name, str) or not name:
        raise ValueError("Strategy name must be a non-empty string")
    if not issubclass(strategy_class, bt.Strategy):
        raise ValueError("strategy_class must be a subclass of backtrader.Strategy")

    # Only register the provided name and the class name as aliases
    aliases = set()
    aliases.add(name)
    aliases.add(strategy_class.__name__)

    for alias in aliases:
        if alias in STRATEGY_REGISTRY:
            logger.warning(f"Strategy alias '{alias}' already registered, overwriting")
        STRATEGY_REGISTRY[alias] = strategy_class
        logger.info(f"Registered strategy alias: {alias}")


def get_strategy(strategy_name: str):
    """Retrieve a strategy class by name.

    Args:
        strategy_name (str): Name of the strategy to retrieve.

    Returns:
        The strategy class or None if not found.

    Raises:
        KeyError: If the strategy_name is not registered.

    Example:
        >>> strategy = get_strategy("EMARSI")
    """
    strategy = STRATEGY_REGISTRY.get(strategy_name)
    if strategy is None:
        logger.error(f"Strategy '{strategy_name}' not found in registry")
        raise KeyError(f"Strategy '{strategy_name}' not found")
    return strategy


# Register default strategies
try:
    from .ema_rsi import EMARSI
    from .sma_bollinger_band import SMABollinger
    from stratgies.momentum.rsi_macd import RSIMACD
    from stratgies.meanreversion.rsi_bb import RSIBB
    from stratgies.momentum.rsi_ema import RSIEMA
    from stratgies.momentum.rsi_cci import RSICCI
    from stratgies.momentum.rsi_adx import RSIADX
    from stratgies.momentum.rsi_stochastic import RSIStochastic
    from stratgies.momentum.macd_ema import MACDEMA
    from stratgies.momentum.macd_volume import MACDVolume
    from stratgies.momentum.macd_bollinger import MACDBollinger
    from stratgies.momentum.macd_adx import MACDADX
    from stratgies.momentum.macd_williams import MACDWilliams
    from stratgies.momentum.BB_Supertrend_Strategy import BBSupertrendStrategy
    from stratgies.momentum.OBV_CMF_Strategy import OBVCMFStrategy
    from stratgies.meanreversion.BB_PivotPoints_Strategy import BBPivotPointsStrategy
    from stratgies.meanreversion.BB_VWAP_Strategy import BBVWAPStrategy
    from stratgies.breakout.BB_ATR_Strategy import BBATRStrategy
    from stratgies.breakout.Volume_VWAP_Breakout_Strategy import (
        VolumeVWAPBreakoutStrategy,
    )
    from stratgies.momentum.ema_adx import EMAADXTrend
    from stratgies.breakout.ATR_Volume_expansion import ATRVolumeExpansion
    from stratgies.momentum.supertrend_psar import SuperTrendPSAR
    from stratgies.breakout.EMAStochasticPullback import EMAStochasticPullback

    register_strategy("EMARSI", EMARSI)
    register_strategy("SMABollinger", SMABollinger)
    register_strategy("RSIMACD", RSIMACD)
    register_strategy("RSIBB", RSIBB)
    register_strategy("RSIEMA", RSIEMA)
    register_strategy("RSICCI", RSICCI)
    register_strategy("RSIADX", RSIADX)
    register_strategy("RSIStochastic", RSIStochastic)
    register_strategy("MACDEMA", MACDEMA)
    register_strategy("MACDVolume", MACDVolume)
    register_strategy("MACDBollinger", MACDBollinger)
    register_strategy("MACDADX", MACDADX)
    register_strategy("MACDWilliams", MACDWilliams)
    register_strategy("BBPivotPointsStrategy", BBPivotPointsStrategy)
    register_strategy("BBVWAPStrategy", BBVWAPStrategy)
    register_strategy("BBATRStrategy", BBATRStrategy)
    register_strategy("VolumeVWAPBreakoutStrategy", VolumeVWAPBreakoutStrategy)
    register_strategy("BBSupertrendStrategy", BBSupertrendStrategy)
    register_strategy("OBVCMFStrategy", OBVCMFStrategy)
    register_strategy("EMAADXTrend", EMAADXTrend)
    register_strategy("ATRVolumeExpansion", ATRVolumeExpansion)
    register_strategy("SuperTrendPSAR", SuperTrendPSAR)
    register_strategy("EMAStochasticPullback", EMAStochasticPullback)
except ImportError as e:
    logger.error(f"Failed to register EMARSI: {str(e)}")
