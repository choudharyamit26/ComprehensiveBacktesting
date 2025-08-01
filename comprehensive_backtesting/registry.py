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


try:
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
    from stratgies.meanreversion.sr_rsi import SRRSI
    from stratgies.meanreversion.pivot_cci import PivotCCI
    from stratgies.meanreversion.trendline_williams import TrendlineWilliams
    from stratgies.momentum.bb_rsi_volume_breakout import BBRHIVolumeBreakout
    from stratgies.momentum.bb_stochastic_obv import BBStochasticOBV
    from stratgies.momentum.cci_rsi_stochastic_rsi import CCIRSIStochasticRSI
    from stratgies.momentum.ema_adx_volume_trend import EMAAADXVolumeTrend
    from stratgies.momentum.rsi_williams_stochastic import RSIWilliamsStochastic
    from stratgies.momentum.triple_momentum_confirmation import (
        TripleMomentumConfirmation,
    )
    from stratgies.momentum.rsi_macd_ema_trend import RSIMACDEMATrend
    from stratgies.momentum.supertrend_cci_cmf import SupertrendCCICMF
    from stratgies.momentum.ADX_EMA_MACD_Strategy import ADXEMAMACDStrategy
    from stratgies.momentum.EMA_Supertrend_SAR_Strategy import EMASupertrendSARStrategy
    from stratgies.momentum.VWAP_BB_ADX_Strategy import VWAPBBADXStrategy
    from stratgies.momentum.VWAP_EMA_Volume_Strategy import VWAPEMAVolumeStrategy
    from stratgies.priceaction.sr_rsi_volume import SRRSIVolume
    from stratgies.priceaction.trendline_macd_ema import TrendlineMACD_EMA

    from stratgies.priceaction.volume_atr_price_action import Volume_ATR_PriceAction
    from stratgies.momentum.obv_cmf_volume_rate import OBV_CMF_VolumeRate
    from stratgies.momentum.pivot_rsi_macd_volume import Pivot_RSI_MACD_Volume
    from stratgies.momentum.pivot_bb_stochastic import Pivot_BB_Stochastic
    from stratgies.momentum.atr_bb_volume_volatility import ATR_BB_VolumeVolatility
    from stratgies.momentum.atr_supertrend_macd import ATR_Supertrend_MACD
    from stratgies.momentum.rsi_cci_williams_stochastic import (
        RSI_CCI_Williams_Stochastic,
    )
    from stratgies.momentum.multiple_stochastic_timeframes import (
        Multiple_Stochastic_Timeframes,
    )

    from stratgies.priceaction.doji_support_resistance_rsi import (
        DojiSupportResistanceRSI,
    )
    from stratgies.priceaction.flag_pennant_momentum import FlagPennantMomentum
    from stratgies.priceaction.hammer_shooting_star_confirmation import (
        HammerShootingStarConfirmation,
    )
    from stratgies.priceaction.head_shoulders_confirmation import (
        HeadShouldersConfirmation,
    )
    from stratgies.priceaction.triangle_breakout_confirmation import (
        TriangleBreakoutConfirmation,
    )
    from stratgies.momentum.bsav_intraday import BSAV
    from stratgies.momentum.motv_intraday import MOTV
    from stratgies.momentum.rmbev_intraday import RMBEV
    from stratgies.momentum.rmev_intraday import RMEV
    from stratgies.momentum.spma_intraday import SPMA
    from stratgies.momentum.verv_intraday import VERV
    from stratgies.momentum.multi_ema_stochasticrsi import EMAMultiStrategy
    from stratgies.momentum.ema_macd_rsi_volume import EMAMACDRSIVolume
    from stratgies.momentum.ema_rsi_pivot import EMA_RSI_Pivot
    from stratgies.momentum.RSI_Supertrend_Intraday import RSISupertrendIntraday

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
    register_strategy("SRRSI", SRRSI)
    register_strategy("PivotCCI", PivotCCI)
    register_strategy("TrendlineWilliams", TrendlineWilliams)
    register_strategy("BBRHIVolumeBreakout", BBRHIVolumeBreakout)
    register_strategy("BBStochasticOBV", BBStochasticOBV)
    register_strategy("CCIRSIStochasticRSI", CCIRSIStochasticRSI)
    register_strategy("EMAAADXVolumeTrend", EMAAADXVolumeTrend)
    register_strategy("RSIWilliamsStochastic", RSIWilliamsStochastic)
    register_strategy("SupertrendCCICMF", SupertrendCCICMF)
    register_strategy("TripleMomentumConfirmation", TripleMomentumConfirmation)
    register_strategy("RSIMACDEMATrend", RSIMACDEMATrend)
    register_strategy("ADXEMAMACDStrategy", ADXEMAMACDStrategy)
    register_strategy("EMASupertrendSARStrategy", EMASupertrendSARStrategy)
    register_strategy("VWAPBBADXStrategy", VWAPBBADXStrategy)
    register_strategy("VWAPEMAVolumeStrategy", VWAPEMAVolumeStrategy)
    register_strategy("SRRSIVolume", SRRSIVolume)
    register_strategy("TrendlineMACD_EMA", TrendlineMACD_EMA)
    register_strategy("Volume_ATR_PriceAction", Volume_ATR_PriceAction)
    register_strategy("OBV_CMF_VolumeRate", OBV_CMF_VolumeRate)
    register_strategy("Pivot_RSI_MACD_Volume", Pivot_RSI_MACD_Volume)
    register_strategy("Pivot_BB_Stochastic", Pivot_BB_Stochastic)
    register_strategy("ATR_BB_VolumeVolatility", ATR_BB_VolumeVolatility)
    register_strategy("ATR_Supertrend_MACD", ATR_Supertrend_MACD)
    register_strategy("RSI_CCI_Williams_Stochastic", RSI_CCI_Williams_Stochastic)
    register_strategy("Multiple_Stochastic_Timeframes", Multiple_Stochastic_Timeframes)
    register_strategy("DojiSupportResistanceRSI", DojiSupportResistanceRSI)
    register_strategy("FlagPennantMomentum", FlagPennantMomentum)
    register_strategy("HammerShootingStarConfirmation", HammerShootingStarConfirmation)
    register_strategy("HeadShouldersConfirmation", HeadShouldersConfirmation)
    register_strategy("TriangleBreakoutConfirmation", TriangleBreakoutConfirmation)
    register_strategy("BSAV", BSAV)
    register_strategy("MOTV", MOTV)
    register_strategy("RMBEV", RMBEV)
    register_strategy("RMEV", RMEV)
    register_strategy("SPMA", SPMA)
    register_strategy("VERV", VERV)
    register_strategy("EMAMultiStrategy", EMAMultiStrategy)
    register_strategy("EMAMACDRSIVolume", EMAMACDRSIVolume)
    register_strategy("EMA_RSI_Pivot", EMA_RSI_Pivot)
    register_strategy("RSISupertrendIntraday", RSISupertrendIntraday)
except ImportError as e:
    logger.error(f"Failed to register EMARSI: {str(e)}")
