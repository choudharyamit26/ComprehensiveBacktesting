"""
Strategy registry for managing and loading trading strategies.
"""

import logging

logger = logging.getLogger("quant_trader")

STRATEGY_REGISTRY = {}


def register_strategy(name: str, strategy_class):
    """Register a new strategy in the framework with multiple aliases."""
    import re

    if not isinstance(name, str) or not name:
        raise ValueError("Strategy name must be a non-empty string")

    aliases = set()
    aliases.add(name)
    aliases.add(strategy_class.__name__)

    for alias in aliases:
        if alias in STRATEGY_REGISTRY:
            logger.warning(f"Strategy alias '{alias}' already registered, overwriting")
        STRATEGY_REGISTRY[alias] = strategy_class
        logger.info(f"Registered strategy alias: {alias}")


def get_strategy(strategy_name: str):
    """Retrieve a strategy class by name."""
    strategy = STRATEGY_REGISTRY.get(strategy_name)
    if strategy is None:
        logger.error(f"Strategy '{strategy_name}' not found in registry")
        raise KeyError(f"Strategy '{strategy_name}' not found")
    return strategy


def register_available_strategies():
    """Register strategies with correct module paths and class names"""
    strategies_to_load = [
        (
            "live_strategies.vwap_bounce_rejection",
            "VWAPBounceRejection",
            "VWAPBounceRejection",
        ),
        ("live_strategies.pivot_cci", "PivotCCI", "PivotCCI"),
        (
            "live_strategies.BB_PivotPoints_Strategy",
            "BBPivotPointsStrategy",
            "BBPivotPointsStrategy",
        ),
        ("live_strategies.BB_VWAP_Strategy", "BBVWAPStrategy", "BBVWAPStrategy"),
        ("live_strategies.rsi_bb", "RSIBB", "RSIBB"),
        (
            "live_strategies.head_shoulders_confirmation",
            "HeadShouldersConfirmation",
            "HeadShouldersConfirmation",
        ),
        (
            "live_strategies.RSI_Supertrend_Intraday",
            "RSISupertrendIntraday",
            "RSISupertrendIntraday",
        ),
        ("live_strategies.rsi_cci", "RSICCI", "RSICCI"),
        ("live_strategies.sr_rsi", "SRRSI", "SRRSI"),
        ("live_strategies.sr_rsi_volume", "SRRSIVolume", "SRRSIVolume"),
        ("live_strategies.rsi_adx", "RSIADX", "RSIADX"),
        (
            "live_strategies.EMAStochasticPullback",
            "EMAStochasticPullback",
            "EMAStochasticPullback",
        ),
        (
            "live_strategies.BB_Supertrend_Strategy",
            "BBSupertrendStrategy",
            "BBSupertrendStrategy",
        ),
        (
            "live_strategies.ATR_Volume_expansion",
            "ATRVolumeExpansion",
            "ATRVolumeExpansion",
        ),
        ("live_strategies.supertrend_cci_cmf", "SupertrendCCICMF", "SupertrendCCICMF"),
        ("live_strategies.bsav_intraday", "BSAV", "BSAV"),
        ("live_strategies.ema_adx", "EMAADXTrend", "EMAADXTrend"),
        ("live_strategies.emamulti_strategy", "EMAMultiStrategy", "EMAMultiStrategy"),
        ("live_strategies.macd_volume", "MACDVolume", "MACDVolume"),
        ("live_strategies.rmbev_intraday", "RMBEV", "RMBEV"),
        ("live_strategies.verv", "VERV", "VERV"),
        (
            "live_strategies.trendline_williams",
            "TrendlineWilliams",
            "TrendlineWilliams",
        ),
    ]

    loaded = 0
    failed_strategies = []

    for module_path, class_name, alias in strategies_to_load:
        try:
            module = __import__(module_path, fromlist=[class_name])
            strategy_class = getattr(module, class_name)
            register_strategy(alias, strategy_class)
            loaded += 1
            logger.debug(f"✅ Loaded strategy: {alias} from {module_path}")
        except ImportError as e:
            failed_strategies.append(f"{module_path} (ImportError: {e})")
            logger.warning(
                f"❌ Strategy import failed: {module_path}.{class_name} -> {e}"
            )
        except AttributeError as e:
            failed_strategies.append(
                f"{module_path}.{class_name} (AttributeError: {e})"
            )
            logger.warning(
                f"❌ Strategy class not found: {module_path}.{class_name} -> {e}"
            )
        except Exception as e:
            failed_strategies.append(f"{module_path}.{class_name} (Error: {e})")
            logger.warning(
                f"❌ Strategy registration failed: {module_path}.{class_name} -> {e}"
            )

    logger.info(
        f"Strategy registry initialized with {loaded}/{len(strategies_to_load)} strategies"
    )
    logger.info(f"Loaded strategies: {list(STRATEGY_REGISTRY.keys())}")

    if failed_strategies:
        logger.warning(f"Failed to load {len(failed_strategies)} strategies:")
        for failed in failed_strategies:
            logger.warning(f"  - {failed}")


# Initialize strategy registry
register_available_strategies()
