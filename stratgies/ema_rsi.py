from backtrader import Strategy, Cerebro
from backtrader.indicators import ExponentialMovingAverage, RelativeStrengthIndex


class EMARSI(Strategy):
    params = (
        ("slow_ema_period", 20),
        ("fast_ema_period", 10),
        ("rsi_period", 14),
        ("rsi_upper", 70),
        ("rsi_lower", 30),
    )

    def __init__(self):
        self.slow_ema = ExponentialMovingAverage(
            self.data.close, period=self.params.slow_ema_period
        )
        self.fast_ema = ExponentialMovingAverage(
            self.data.close, period=self.params.fast_ema_period
        )
        self.rsi = RelativeStrengthIndex(period=self.params.rsi_period)

    # def next(self):
    #     if not self.position:
    #         # BUY: Fast EMA crosses above Slow EMA AND RSI < 30 (oversold)
    #         if (self.fast_ema[0] > self.slow_ema[0] and
    #             self.rsi[0] < self.params.rsi_lower):
    #             self.buy()
    #     else:
    #         # SELL: Fast EMA crosses below Slow EMA OR RSI > 70 (overbought)
    #         if (self.fast_ema[0] < self.slow_ema[0] or
    #             self.rsi[0] > self.params.rsi_upper):
    #             self.sell()
    def next(self):
        if not self.position:
            # BUY: Fast EMA crosses above Slow EMA AND RSI < 40 (relaxed)
            if (
                self.fast_ema[0] > self.slow_ema[0] and self.rsi[0] < 40
            ):  # Changed from 18 to 40
                self.buy()
        else:
            # SELL: Fast EMA crosses below Slow EMA OR RSI > 70 (conventional)
            if (
                self.fast_ema[0] < self.slow_ema[0] or self.rsi[0] > 70
            ):  # Changed from 77 to 70
                self.sell()
