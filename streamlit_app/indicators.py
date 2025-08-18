"""
Technical indicator detection and calculation functions.
"""

import pandas as pd
import numpy as np
import logging
import backtrader as bt

logger = logging.getLogger(__name__)


def detect_strategy_indicators(strategy):
    """Dynamically detect indicators used by a strategy."""
    indicators = {}
    try:
        for attr_name in dir(strategy):
            if attr_name.startswith("_"):
                continue

            attr = getattr(strategy, attr_name)

            # Improved indicator detection
            if isinstance(attr, bt.Indicator):
                indicators[attr_name] = {
                    "indicator": attr,
                    "type": attr.__class__.__name__,
                    "name": attr_name,
                }

                # Extract parameters
                if hasattr(attr, "params"):
                    params = {}
                    for pname in attr.params._getkeys():
                        try:
                            params[pname] = getattr(attr.params, pname)
                        except AttributeError:
                            continue
                    indicators[attr_name]["params"] = params

        logger.info(f"Detected indicators: {list(indicators.keys())}")
        return indicators

    except Exception as e:
        logger.error(f"Error detecting indicators: {e}")
        return {}


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(high, low, close, period):
    """Helper function to calculate ATR."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_indicator_values(data, indicator_info):
    """Calculate indicator values based on detected indicator info."""
    calculated_indicators = {}

    try:
        for name, info in indicator_info.items():
            indicator_type = info["type"]
            params = info.get("params", {})

            if indicator_type == "EMA":
                period = params.get("period", 20)
                calculated_indicators[name] = {
                    "values": data["Close"].ewm(span=period).mean(),
                    "type": "line",
                    "subplot": "price",
                    "color": "#ff6b35" if "fast" in name.lower() else "#004e89",
                    "name": f"{name.upper()} ({period})",
                }

            elif indicator_type == "SMA":
                period = params.get("period", 20)
                calculated_indicators[name] = {
                    "values": data["Close"].rolling(window=period).mean(),
                    "type": "line",
                    "subplot": "price",
                    "color": "#2e8b57",
                    "name": f"{name.upper()} ({period})",
                }

            elif indicator_type == "RSI":
                period = params.get("period", 14)
                rsi_values = calculate_rsi(data["Close"], period)
                calculated_indicators[name] = {
                    "values": rsi_values,
                    "type": "line",
                    "subplot": "oscillator",
                    "color": "#9d4edd",
                    "name": f"{name.upper()} ({period})",
                    "y_range": [0, 100],
                    "levels": {
                        "overbought": params.get("upperband", 70),
                        "oversold": params.get("lowerband", 30),
                        "neutral": 50,
                    },
                }

            elif indicator_type == "MACD":
                fast_period = params.get("period_me1", 12)
                slow_period = params.get("period_me2", 26)
                signal_period = params.get("period_signal", 9)

                ema_fast = data["Close"].ewm(span=fast_period).mean()
                ema_slow = data["Close"].ewm(span=slow_period).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=signal_period).mean()
                histogram = macd_line - signal_line

                calculated_indicators[f"{name}_line"] = {
                    "values": macd_line,
                    "type": "line",
                    "subplot": "oscillator",
                    "color": "#1f77b4",
                    "name": f"MACD Line",
                }
                calculated_indicators[f"{name}_signal"] = {
                    "values": signal_line,
                    "type": "line",
                    "subplot": "oscillator",
                    "color": "#ff7f0e",
                    "name": f"Signal Line",
                }
                calculated_indicators[f"{name}_histogram"] = {
                    "values": histogram,
                    "type": "bar",
                    "subplot": "oscillator",
                    "color": "#2ca02c",
                    "name": f"MACD Histogram",
                }

            elif indicator_type == "BollingerBands":
                period = params.get("period", 20)
                std_dev = params.get("devfactor", 2)

                sma = data["Close"].rolling(window=period).mean()
                std = data["Close"].rolling(window=period).std()
                upper_band = sma + (std * std_dev)
                lower_band = sma - (std * std_dev)

                calculated_indicators[f"{name}_upper"] = {
                    "values": upper_band,
                    "type": "line",
                    "subplot": "price",
                    "color": "#ff0000",
                    "name": f"BB Upper ({period}, {std_dev})",
                    "line_style": "dash",
                }
                calculated_indicators[f"{name}_middle"] = {
                    "values": sma,
                    "type": "line",
                    "subplot": "price",
                    "color": "#0000ff",
                    "name": f"BB Middle ({period})",
                }
                calculated_indicators[f"{name}_lower"] = {
                    "values": lower_band,
                    "type": "line",
                    "subplot": "price",
                    "color": "#ff0000",
                    "name": f"BB Lower ({period}, {std_dev})",
                    "line_style": "dash",
                }

            elif indicator_type == "Stochastic":
                k_period = params.get("period_k", 14)
                d_period = params.get("period_d", 3)

                lowest_low = data["Low"].rolling(window=k_period).min()
                highest_high = data["High"].rolling(window=k_period).max()
                k_percent = 100 * (
                    (data["Close"] - lowest_low) / (highest_high - lowest_low)
                )
                d_percent = k_percent.rolling(window=d_period).mean()

                calculated_indicators[f"{name}_k"] = {
                    "values": k_percent,
                    "type": "line",
                    "subplot": "oscillator",
                    "color": "#ff6b35",
                    "name": f"Stoch %K ({k_period})",
                    "y_range": [0, 100],
                    "levels": {"overbought": 80, "oversold": 20},
                }
                calculated_indicators[f"{name}_d"] = {
                    "values": d_percent,
                    "type": "line",
                    "subplot": "oscillator",
                    "color": "#004e89",
                    "name": f"Stoch %D ({d_period})",
                    "y_range": [0, 100],
                }

            elif indicator_type == "Supertrend":
                period = params.get("period", 10)
                multiplier = params.get("multiplier", 3.0)

                atr = calculate_atr(data["High"], data["Low"], data["Close"], period)
                hl2 = (data["High"] + data["Low"]) / 2.0
                basic_upperband = hl2 + (multiplier * atr)
                basic_lowerband = hl2 - (multiplier * atr)
                supertrend = pd.Series(index=data.index, dtype=float)
                supertrend.iloc[0] = hl2.iloc[0]

                for i in range(1, len(data)):
                    if (
                        basic_upperband.iloc[i] < supertrend.iloc[i - 1]
                        or data["Close"].iloc[i - 1] > supertrend.iloc[i - 1]
                    ):
                        final_upperband = basic_upperband.iloc[i]
                    else:
                        final_upperband = supertrend.iloc[i - 1]

                    if (
                        basic_lowerband.iloc[i] > supertrend.iloc[i - 1]
                        or data["Close"].iloc[i - 1] < supertrend.iloc[i - 1]
                    ):
                        final_lowerband = basic_lowerband.iloc[i]
                    else:
                        final_lowerband = supertrend.iloc[i - 1]

                    if (
                        supertrend.iloc[i - 1] == final_upperband
                        and data["Close"].iloc[i] <= final_upperband
                    ):
                        supertrend.iloc[i] = final_upperband
                    elif (
                        supertrend.iloc[i - 1] == final_lowerband
                        and data["Close"].iloc[i] >= final_lowerband
                    ):
                        supertrend.iloc[i] = final_lowerband
                    elif data["Close"].iloc[i] <= final_lowerband:
                        supertrend.iloc[i] = final_upperband
                    else:
                        supertrend.iloc[i] = final_lowerband

                calculated_indicators[name] = {
                    "values": supertrend,
                    "type": "line",
                    "subplot": "price",
                    "color": "#00ff00",
                    "name": f"Supertrend ({period}, {multiplier})",
                }

            elif indicator_type == "ParabolicSAR":
                af = params.get("af", 0.02)
                afmax = params.get("afmax", 0.2)

                psar = pd.Series(index=data.index, dtype=float)
                psar.iloc[0] = data["Low"].iloc[0]
                trend = 1  # 1 for uptrend, -1 for downtrend
                ep = data["High"].iloc[0]  # Extreme point
                af_current = af

                for i in range(1, len(data)):
                    if trend == 1:
                        psar.iloc[i] = psar.iloc[i - 1] + af_current * (
                            ep - psar.iloc[i - 1]
                        )
                        if data["High"].iloc[i] > ep:
                            ep = data["High"].iloc[i]
                            af_current = min(af_current + af, afmax)
                        if data["Low"].iloc[i] < psar.iloc[i]:
                            trend = -1
                            psar.iloc[i] = ep
                            ep = data["Low"].iloc[i]
                            af_current = af
                    else:
                        psar.iloc[i] = psar.iloc[i - 1] + af_current * (
                            ep - psar.iloc[i - 1]
                        )
                        if data["Low"].iloc[i] < ep:
                            ep = data["Low"].iloc[i]
                            af_current = min(af_current + af, afmax)
                        if data["High"].iloc[i] > psar.iloc[i]:
                            trend = 1
                            psar.iloc[i] = ep
                            ep = data["High"].iloc[i]
                            af_current = af

                calculated_indicators[name] = {
                    "values": psar,
                    "type": "scatter",
                    "subplot": "price",
                    "color": "#ff00ff",
                    "name": f"PSAR ({af}, {afmax})",
                    "marker": {"symbol": "dot", "size": 5},
                }

            elif indicator_type == "CCI":
                period = params.get("period", 14)
                constant = params.get("constant", 0.015)

                typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
                sma_tp = typical_price.rolling(window=period).mean()
                mean_deviation = typical_price.rolling(window=period).apply(
                    lambda x: np.mean(np.abs(x - x.mean())), raw=False
                )
                cci_values = (typical_price - sma_tp) / (constant * mean_deviation)

                calculated_indicators[name] = {
                    "values": cci_values,
                    "type": "line",
                    "subplot": "oscillator",
                    "color": "#4682b4",
                    "name": f"CCI ({period})",
                    "y_range": [-200, 200],
                    "levels": {
                        "overbought": params.get("overbought", 100),
                        "oversold": params.get("oversold", -100),
                        "neutral": 0,
                    },
                }

            elif indicator_type == "WilliamsR":
                period = params.get("period", 14)
                highest_high = data["High"].rolling(window=period).max()
                lowest_low = data["Low"].rolling(window=period).min()
                williams_r = -100 * (
                    (highest_high - data["Close"]) / (highest_high - lowest_low)
                )
                calculated_indicators[name] = {
                    "values": williams_r,
                    "type": "line",
                    "subplot": "oscillator",
                    "color": "#ff6b35",
                    "name": f"Williams %R ({period})",
                    "y_range": [-100, 0],
                    "levels": {
                        "overbought": params.get("overbought", -20),
                        "oversold": params.get("oversold", -80),
                        "neutral": params.get("neutral", -50),
                    },
                }

            elif indicator_type == "Trendline":
                period = params.get("period", 20)
                swing_high = data["High"].rolling(window=period).max()
                swing_low = data["Low"].rolling(window=period).min()
                calculated_indicators[f"{name}_support"] = {
                    "values": swing_low,
                    "type": "line",
                    "subplot": "price",
                    "color": "#00ff00",
                    "name": f"Trendline Support ({period})",
                    "line_style": "dash",
                }
                calculated_indicators[f"{name}_resistance"] = {
                    "values": swing_high,
                    "type": "line",
                    "subplot": "price",
                    "color": "#ff0000",
                    "name": f"Trendline Resistance ({period})",
                    "line_style": "dash",
                }

            elif indicator_type == "ATR":
                period = params.get("period", 14)
                atr_values = calculate_atr(
                    data["High"], data["Low"], data["Close"], period
                )
                calculated_indicators[name] = {
                    "values": atr_values,
                    "type": "line",
                    "subplot": "oscillator",
                    "color": "#8b008b",
                    "name": f"ATR ({period})",
                    "y_range": [0, None],  # ATR is non-negative
                }

            elif indicator_type == "VolumeVolatility":
                period = params.get("period", 14)
                volume_sma = data["Volume"].rolling(window=period).mean()
                vol_volatility = data["Volume"] / volume_sma
                calculated_indicators[name] = {
                    "values": vol_volatility,
                    "type": "line",
                    "subplot": "volume",
                    "color": "#ffa500",
                    "name": f"Volume Volatility ({period})",
                    "y_range": [0, None],  # Volume volatility is non-negative
                    "levels": {"threshold": params.get("threshold", 1.5)},
                }

            elif indicator_type == "VolumeRate":
                period = params.get("period", 14)
                volume_rate = data["Volume"] / data["Volume"].shift(period)
                calculated_indicators[name] = {
                    "values": volume_rate,
                    "type": "line",
                    "subplot": "volume",
                    "color": "#008080",
                    "name": f"Volume Rate ({period})",
                    "y_range": [0, None],  # Volume rate is non-negative
                    "levels": {"neutral": 1.0},
                }

            elif indicator_type == "OBV":
                obv = pd.Series(0.0, index=data.index)
                for i in range(1, len(data)):
                    if data["Close"].iloc[i] > data["Close"].iloc[i - 1]:
                        obv.iloc[i] = obv.iloc[i - 1] + data["Volume"].iloc[i]
                    elif data["Close"].iloc[i] < data["Close"].iloc[i - 1]:
                        obv.iloc[i] = obv.iloc[i - 1] - data["Volume"].iloc[i]
                    else:
                        obv.iloc[i] = obv.iloc[i - 1]
                calculated_indicators[name] = {
                    "values": obv,
                    "type": "line",
                    "subplot": "volume",
                    "color": "#6a5acd",
                    "name": "OBV",
                }

            elif indicator_type == "CMF":
                period = params.get("period", 20)
                mfm = (
                    (data["Close"] - data["Low"]) - (data["High"] - data["Close"])
                ) / (data["High"] - data["Low"])
                mfv = mfm * data["Volume"]
                cmf = (
                    mfv.rolling(window=period).sum()
                    / data["Volume"].rolling(window=period).sum()
                )
                calculated_indicators[name] = {
                    "values": cmf,
                    "type": "line",
                    "subplot": "oscillator",
                    "color": "#20b2aa",
                    "name": f"CMF ({period})",
                    "y_range": [-1, 1],
                    "levels": {"positive": 0, "negative": 0},
                }

        return calculated_indicators

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}


def create_dynamic_indicators_table(data, strategy):
    """Create a dynamic table showing current technical indicator values based on strategy."""
    try:
        # Detect indicators dynamically
        detected_indicators = detect_strategy_indicators(strategy)
        calculated_indicators = calculate_indicator_values(data, detected_indicators)

        if not calculated_indicators:
            return pd.DataFrame()

        # Get latest values (last 5 periods)
        latest_data = []
        for i in range(min(5, len(data))):
            idx = -(i + 1)  # Start from last and go backwards
            date = data.index[idx].strftime("%Y-%m-%d %H:%M")

            row_data = {"Date/Time": date, "Close Price": data["Close"].iloc[idx]}

            # Add all detected indicators
            for indicator_name, indicator_data in calculated_indicators.items():
                if not indicator_data["values"].empty and idx < len(
                    indicator_data["values"]
                ):
                    value = indicator_data["values"].iloc[idx]
                    if pd.notna(value):
                        row_data[indicator_data["name"]] = value

            # Add signal analysis for common indicators
            signals = []

            # EMA signals
            ema_indicators = [
                k for k in calculated_indicators.keys() if "ema" in k.lower()
            ]
            if len(ema_indicators) >= 2:
                fast_ema = None
                slow_ema = None
                for ema_name in ema_indicators:
                    if "fast" in ema_name.lower():
                        fast_ema = calculated_indicators[ema_name]["values"].iloc[idx]
                    elif "slow" in ema_name.lower():
                        slow_ema = calculated_indicators[ema_name]["values"].iloc[idx]

                if fast_ema is not None and slow_ema is not None:
                    signals.append(
                        "EMA: Bullish" if fast_ema > slow_ema else "EMA: Bearish"
                    )

            # RSI signals
            rsi_indicators = [
                k for k in calculated_indicators.keys() if "rsi" in k.lower()
            ]
            for rsi_name in rsi_indicators:
                rsi_data = calculated_indicators[rsi_name]
                if "levels" in rsi_data:
                    rsi_value = rsi_data["values"].iloc[idx]
                    levels = rsi_data["levels"]
                    if rsi_value > levels.get("overbought", 70):
                        signals.append("RSI: Overbought")
                    elif rsi_value < levels.get("oversold", 30):
                        signals.append("RSI: Oversold")
                    else:
                        signals.append("RSI: Neutral")

            # MACD signals
            macd_line_indicators = [
                k
                for k in calculated_indicators.keys()
                if "macd" in k.lower() and "line" in k.lower()
            ]
            macd_signal_indicators = [
                k
                for k in calculated_indicators.keys()
                if "macd" in k.lower() and "signal" in k.lower()
            ]

            if macd_line_indicators and macd_signal_indicators:
                macd_line = calculated_indicators[macd_line_indicators[0]][
                    "values"
                ].iloc[idx]
                macd_signal = calculated_indicators[macd_signal_indicators[0]][
                    "values"
                ].iloc[idx]
                signals.append(
                    "MACD: Bullish" if macd_line > macd_signal else "MACD: Bearish"
                )

            row_data["Signals"] = " | ".join(signals) if signals else "No signals"
            latest_data.append(row_data)

        return pd.DataFrame(latest_data)

    except Exception as e:
        logger.error(f"Error creating dynamic indicators table: {e}")
        return pd.DataFrame()
