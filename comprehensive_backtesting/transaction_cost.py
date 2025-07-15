import backtrader as bt
import numpy as np
from typing import Dict, Tuple
import pandas as pd


class RealisticTransactionCosts(bt.CommissionInfo):
    """Enhanced commission model with realistic market impact and slippage."""

    def __init__(self, commission=0.001, slippage_perc=0.0005, min_slippage=0.01):
        super().__init__(commission=commission)
        self.slippage_perc = slippage_perc
        self.min_slippage = min_slippage

    def getsize(self, price, cash):
        """Override to account for transaction costs in sizing."""
        gross_size = super().getsize(price, cash)
        net_size = gross_size * (1 - self.commission - self.slippage_perc)
        return max(1, int(net_size))


class MarketImpactModel:
    """Kyle's lambda model for market impact estimation."""

    def __init__(self, kappa=0.1):
        self.kappa = kappa

    def calculate_impact(self, order_size: float, avg_volume: float) -> float:
        """Calculate price impact based on order size and volume."""
        if avg_volume == 0:
            return 0.0
        impact = self.kappa * abs(order_size) / avg_volume
        return min(impact, 0.02)  # Cap at 2%


class RegimeDetector:
    """Detect market regimes using volatility and trend indicators."""

    def __init__(self, lookback=252):
        self.lookback = lookback

    def detect_regime(self, data: pd.DataFrame) -> Dict[str, str]:
        """Detect current market regime."""
        returns = data["Close"].pct_change()

        # Volatility regime
        rolling_vol = returns.rolling(self.lookback).std()
        vol_threshold = returns.std()
        volatility_regime = "high" if rolling_vol.iloc[-1] > vol_threshold else "low"

        # Trend regime
        sma_fast = data["Close"].rolling(50).mean()
        sma_slow = data["Close"].rolling(200).mean()
        trend_regime = "bull" if sma_fast.iloc[-1] > sma_slow.iloc[-1] else "bear"

        # Liquidity regime
        volume_ratio = (
            data["Volume"].rolling(20).mean().iloc[-1]
            / data["Volume"].rolling(60).mean().iloc[-1]
        )
        liquidity_regime = "high" if volume_ratio > 1.2 else "low"

        return {
            "volatility": volatility_regime,
            "trend": trend_regime,
            "liquidity": liquidity_regime,
        }


class DynamicPositionSizer(bt.Sizer):
    """Kelly Criterion-based position sizing with volatility adjustment."""

    params = (
        ("max_position_size", 0.1),  # Max 10% per position
        ("volatility_period", 20),
        ("kelly_fraction", 0.25),  # Use 25% of Kelly
    )

    def __init__(self):
        self.regime_detector = RegimeDetector()

    def _getsizing(self, comminfo, cash, data, isbuy):
        # Get current regime
        regime = self.regime_detector.detect_regime(data)

        # Calculate Kelly fraction adjusted for regime
        kelly_fraction = self.p.kelly_fraction

        if regime["volatility"] == "high":
            kelly_fraction *= 0.5
        if regime["trend"] == "bear":
            kelly_fraction *= 0.7

        # Calculate position size
        portfolio_value = self.strategy.broker.getvalue()
        max_position = portfolio_value * self.p.max_position_size

        # Volatility targeting
        returns = pd.Series(data.close.get(size=100)).pct_change()
        volatility = returns.rolling(self.p.volatility_period).std().iloc[-1]
        target_vol = 0.15 / np.sqrt(252)  # 15% annual

        vol_factor = min(target_vol / (volatility + 1e-8), 2.0)

        # Final position size
        position_size = min(
            max_position * vol_factor * kelly_fraction, cash / data.close[0]
        )

        return max(1, int(position_size))


class TransactionCostCalculator:
    """Calculate realistic transaction costs including spread and slippage."""

    def __init__(self, spread_bp=1, slippage_bp=0.5, market_impact=0.1):
        self.spread_bp = spread_bp
        self.slippage_bp = slippage_bp
        self.market_impact = market_impact

    def calculate_cost(self, price: float, size: int, avg_volume: float) -> float:
        """Calculate total transaction cost."""
        spread_cost = price * self.spread_bp * 0.0001
        slippage_cost = price * self.slippage_bp * 0.0001

        # Market impact based on Kyle's lambda
        impact_pct = min(self.market_impact * abs(size) / avg_volume, 0.01)
        impact_cost = price * impact_pct

        return spread_cost + slippage_cost + impact_cost
