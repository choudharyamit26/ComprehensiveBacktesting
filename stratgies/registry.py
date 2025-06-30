import logging
import backtrader as bt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STRATEGY_REGISTRY = {}


def register_strategy(name: str, strategy_class):
    """Register a new strategy in the framework.

    Args:
        name (str): Unique name for the strategy.
        strategy_class: Backtrader strategy class to register.

    Raises:
        ValueError: If the name is already registered or strategy_class is invalid.

    Example:
        >>> from ema_rsi import EMARSI
        >>> register_strategy("EMARSI", EMARSI)
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Strategy name must be a non-empty string")
    if not issubclass(strategy_class, bt.Strategy):
        raise ValueError("strategy_class must be a subclass of backtrader.Strategy")
    if name in STRATEGY_REGISTRY:
        logger.warning(f"Strategy '{name}' already registered, overwriting")
    STRATEGY_REGISTRY[name] = strategy_class
    logger.info(f"Registered strategy: {name}")


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

    register_strategy("EMARSI", EMARSI)
except ImportError as e:
    logger.error(f"Failed to register EMARSI: {str(e)}")
