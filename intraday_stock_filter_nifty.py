import yfinance as yf
import pandas as pd
import time
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


def read_stocks_from_csv(csv_file="ind_nifty500list.csv"):
    try:
        if not os.path.exists(csv_file):
            print(f"CSV file '{csv_file}' not found.")
            return []

        df = pd.read_csv(csv_file)
        if "ticker" not in df.columns:
            print("No 'ticker' column found in CSV.")
            return []

        symbols = df["ticker"].dropna().astype(str).tolist()
        cleaned_symbols = []
        for symbol in symbols:
            cleaned_symbol = symbol.strip().replace(".NS", "").replace(".BO", "")
            if (
                cleaned_symbol
                and len(cleaned_symbol) <= 15
                and cleaned_symbol.replace(".", "").replace("-", "").isalnum()
                and cleaned_symbol != "ETERNAL"
            ):
                cleaned_symbols.append(cleaned_symbol)

        print(f"📊 Read {len(cleaned_symbols)} stock symbols from '{csv_file}'")

        valid_symbols = []
        total_symbols = len(cleaned_symbols)
        print(f"🔍 Validating {total_symbols} tickers...")

        for i, symbol in enumerate(cleaned_symbols, 1):
            print(
                f"\r⏳ Validating ticker {i}/{total_symbols}: {symbol}...",
                end="",
                flush=True,
            )

            nse_ticker = symbol + ".NS"
            try:
                test_data = yf.download(
                    nse_ticker,
                    period="5d",
                    interval="1d",
                    progress=False,
                    auto_adjust=False,
                    timeout=5,
                )
                if not test_data.empty:
                    if isinstance(test_data.columns, pd.MultiIndex):
                        if nse_ticker not in [
                            col[1]
                            for col in test_data.columns
                            if isinstance(col, tuple)
                        ]:
                            continue
                        test_data = test_data.xs(
                            nse_ticker, level="Ticker", axis=1, drop_level=True
                        )
                    if all(
                        col in test_data.columns
                        for col in ["High", "Low", "Close", "Volume"]
                    ):
                        valid_symbols.append(symbol)
            except Exception:
                continue

        print(f"\n✅ Validated {len(valid_symbols)}/{len(cleaned_symbols)} tickers")
        if valid_symbols:
            valid_df = pd.DataFrame(valid_symbols, columns=["ticker"])
            valid_df.to_csv("validated_nifty500_tickers.csv", index=False)
            print("💾 Validated tickers saved to 'validated_nifty500_tickers.csv'")

        return valid_symbols

    except Exception as e:
        print(f"❌ Error reading CSV file '{csv_file}': {e}")
        return []


def get_historical_data(ticker, period="3mo", retries=3, delay=2.0, timeout=10):
    nse_ticker = ticker + ".NS"
    yf_ticker = yf.Ticker(nse_ticker)
    for attempt in range(retries):
        try:
            data = yf_ticker.history(
                period=period, interval="1d", auto_adjust=False, timeout=timeout
            )
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    if nse_ticker in [
                        col[1] for col in data.columns if isinstance(col, tuple)
                    ]:
                        data = data.xs(
                            nse_ticker, level="Ticker", axis=1, drop_level=True
                        )
                    else:
                        return None
                required_columns = ["High", "Low", "Close", "Volume"]
                if not all(col in data.columns for col in required_columns):
                    return None
                return data
            else:
                return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
            continue
    return None


def calculate_atr(data, period=14):
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return None
    if not all(col in data.columns for col in ["High", "Low", "Close"]):
        return None
    if (
        data["High"].isna().all()
        or data["Low"].isna().all()
        or data["Close"].isna().all()
    ):
        return None

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr.iloc[-1] if not atr.empty and pd.notna(atr.iloc[-1]) else None


def calculate_daily_range(data):
    if not isinstance(data, pd.DataFrame) or data.empty:
        return None

    required_columns = ["High", "Low", "Close"]
    for col in required_columns:
        if col not in data.columns or data[col].dropna().empty:
            return None

    recent_data = data.tail(5).dropna(subset=["High", "Low", "Close"])
    if recent_data.empty:
        return None

    range_pcts = []
    for index, row in recent_data.iterrows():
        try:
            high_val = row["High"]
            low_val = row["Low"]
            close_val = row["Close"]

            if not all(
                isinstance(val, (int, float, np.number))
                for val in [high_val, low_val, close_val]
            ):
                continue

            if (
                pd.notna(high_val)
                and pd.notna(low_val)
                and pd.notna(close_val)
                and close_val > 0
            ):
                range_pct = (high_val - low_val) / close_val * 100
                range_pcts.append(range_pct)
        except Exception:
            continue

    return np.mean(range_pcts) if range_pcts else None


def calculate_relative_volume(data):
    if (
        data is None
        or not isinstance(data, pd.DataFrame)
        or data.empty
        or "Volume" not in data.columns
    ):
        return None
    if data["Volume"].isna().all():
        return None

    recent_volume = data["Volume"].iloc[-1]
    if len(data) < 20:
        return None

    avg_volume = data["Volume"].iloc[-20:-1].mean()
    if pd.notna(recent_volume) and pd.notna(avg_volume) and avg_volume > 0:
        return float(recent_volume) / float(avg_volume)
    return None


def calculate_momentum(data, period=5):
    if (
        data is None
        or not isinstance(data, pd.DataFrame)
        or data.empty
        or "Close" not in data.columns
    ):
        return None
    if data["Close"].isna().all():
        return None

    if len(data) < period + 1:
        return None

    current_close = data["Close"].iloc[-1]
    past_close = data["Close"].iloc[-(period + 1)]

    if pd.notna(current_close) and pd.notna(past_close) and past_close > 0:
        return float(current_close - past_close) / float(past_close) * 100
    return None


def calculate_moving_averages(data, short_period=5, long_period=20):
    if (
        data is None
        or not isinstance(data, pd.DataFrame)
        or data.empty
        or "Close" not in data.columns
    ):
        return None, None

    if len(data) < long_period:
        return None, None

    short_ma = data["Close"].rolling(window=short_period).mean().iloc[-1]
    long_ma = data["Close"].rolling(window=long_period).mean().iloc[-1]

    return short_ma, long_ma


def calculate_rsi(data, period=14):
    if (
        data is None
        or not isinstance(data, pd.DataFrame)
        or data.empty
        or "Close" not in data.columns
    ):
        return None

    if len(data) < period + 1:
        return None

    closes = data["Close"]
    delta = closes.diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1] if not rsi.empty and pd.notna(rsi.iloc[-1]) else None


def calculate_macd(data, fast=12, slow=26, signal=9):
    if (
        data is None
        or not isinstance(data, pd.DataFrame)
        or data.empty
        or "Close" not in data.columns
    ):
        return None, None, None

    if len(data) < slow + signal:
        return None, None, None

    closes = data["Close"]
    ema_fast = closes.ewm(span=fast).mean()
    ema_slow = closes.ewm(span=slow).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line

    return (
        (
            macd_line.iloc[-1]
            if not macd_line.empty and pd.notna(macd_line.iloc[-1])
            else None
        ),
        (
            signal_line.iloc[-1]
            if not signal_line.empty and pd.notna(signal_line.iloc[-1])
            else None
        ),
        (
            histogram.iloc[-1]
            if not histogram.empty and pd.notna(histogram.iloc[-1])
            else None
        ),
    )


def generate_buy_sell_signal(data):
    if data is None or data.empty:
        return "HOLD", 0

    signals = []
    strength_scores = []

    # Moving Average Crossover Signal
    short_ma, long_ma = calculate_moving_averages(data)
    if short_ma is not None and long_ma is not None:
        ma_diff_pct = ((short_ma - long_ma) / long_ma) * 100
        if short_ma > long_ma:
            signals.append("BUY")
            strength_scores.append(min(abs(ma_diff_pct) * 2, 10))
        else:
            signals.append("SELL")
            strength_scores.append(min(abs(ma_diff_pct) * 2, 10))

    # RSI Signal
    rsi = calculate_rsi(data)
    if rsi is not None:
        if rsi < 30:
            signals.append("BUY")
            strength_scores.append((30 - rsi) * 0.3)
        elif rsi > 70:
            signals.append("SELL")
            strength_scores.append((rsi - 70) * 0.3)
        else:
            signals.append("HOLD")
            strength_scores.append(0)

    # MACD Signal
    macd_line, signal_line, histogram = calculate_macd(data)
    if macd_line is not None and signal_line is not None:
        macd_strength = abs(histogram) if histogram is not None else 0
        if macd_line > signal_line:
            signals.append("BUY")
            strength_scores.append(min(macd_strength * 50, 10))
        else:
            signals.append("SELL")
            strength_scores.append(min(macd_strength * 50, 10))

    # Momentum Signal
    momentum = calculate_momentum(data, period=5)
    if momentum is not None:
        if momentum > 2:
            signals.append("BUY")
            strength_scores.append(min(momentum * 0.5, 10))
        elif momentum < -2:
            signals.append("SELL")
            strength_scores.append(min(abs(momentum) * 0.5, 10))
        else:
            signals.append("HOLD")
            strength_scores.append(0)

    # Volume Signal
    rel_volume = calculate_relative_volume(data)
    if rel_volume is not None and rel_volume > 1.5:
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")

        volume_boost = min((rel_volume - 1.5) * 2, 5)

        if buy_count > sell_count:
            signals.append("BUY")
            strength_scores.append(volume_boost)
        elif sell_count > buy_count:
            signals.append("SELL")
            strength_scores.append(volume_boost)

    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    hold_count = signals.count("HOLD")

    total_strength = sum(strength_scores) if strength_scores else 0
    avg_strength = total_strength / len(strength_scores) if strength_scores else 0

    if buy_count > sell_count and buy_count > hold_count:
        return "BUY", round(avg_strength, 2)
    elif sell_count > buy_count and sell_count > hold_count:
        return "SELL", round(avg_strength, 2)
    else:
        return "HOLD", round(avg_strength, 2)


def calculate_recommendation_score(metrics):
    """Calculate a recommendation score (0-100) based on metrics"""
    score = 0

    # Signal Strength (0-20 points)
    if metrics["Signal Strength"] is not None:
        score += min(metrics["Signal Strength"] * 2, 20)

    # Daily Range (0-20 points)
    if metrics["Daily Range %"] is not None:
        if metrics["Daily Range %"] >= 3:
            score += 20
        elif metrics["Daily Range %"] >= 2:
            score += 15
        elif metrics["Daily Range %"] >= 1.5:
            score += 10

    # ATR (0-15 points)
    if metrics["ATR %"] is not None:
        if metrics["ATR %"] >= 3:
            score += 15
        elif metrics["ATR %"] >= 2:
            score += 10
        elif metrics["ATR %"] >= 1.5:
            score += 5

    # Volume (0-15 points)
    if metrics["Relative Volume"] is not None:
        if metrics["Relative Volume"] >= 2:
            score += 15
        elif metrics["Relative Volume"] >= 1.5:
            score += 10
        elif metrics["Relative Volume"] >= 1:
            score += 5

    # Momentum (0-10 points)
    if metrics["Momentum %"] is not None:
        momentum_score = min(abs(metrics["Momentum %"]) * 0.5, 10)
        score += momentum_score

    # RSI (0-8 points)
    if metrics["RSI"] is not None:
        if metrics["RSI"] < 30 or metrics["RSI"] > 70:
            score += 8
        elif metrics["RSI"] < 40 or metrics["RSI"] > 60:
            score += 4

    # Price Range (0-5 points)
    if metrics["Current Price"] is not None:
        if 100 <= metrics["Current Price"] <= 2000:
            score += 5
        elif 50 <= metrics["Current Price"] <= 3000:
            score += 3

    return min(round(score, 2), 100)


def get_recommendation_label(score, signal):
    """Assign recommendation label based on score and signal"""
    if signal not in ["BUY", "SELL"]:
        return f"{signal}"

    if score >= 50:
        return f"STRONG {signal}"
    elif score >= 35:
        return f"GOOD {signal}"
    elif score >= 20:
        return f"MODERATE {signal}"
    else:
        return f"WEAK {signal}"


signal_priority = {"BUY": 1, "SELL": 2, "HOLD": 3}


def is_suitable_for_intraday(ticker, data):
    if (
        data is None
        or not isinstance(data, pd.DataFrame)
        or data.empty
        or len(data) < 20
    ):
        return False, {}

    daily_range = calculate_daily_range(data)
    atr = calculate_atr(data)
    current_price = (
        data["Close"].iloc[-1]
        if "Close" in data.columns and not data["Close"].isna().all()
        else None
    )
    avg_volume = (
        data["Volume"].tail(20).mean()
        if "Volume" in data.columns and not data["Volume"].isna().all()
        else None
    )
    rel_volume = calculate_relative_volume(data)
    momentum = calculate_momentum(data)
    signal, signal_strength = generate_buy_sell_signal(data)

    # Additional technical indicators
    rsi = calculate_rsi(data)
    short_ma, long_ma = calculate_moving_averages(data)

    atr_pct = None
    if (
        atr is not None
        and current_price is not None
        and pd.notna(current_price)
        and current_price > 0
    ):
        atr_pct = atr / current_price * 100

    metrics = {
        "Daily Range %": round(daily_range, 2) if daily_range is not None else None,
        "ATR %": round(atr_pct, 2) if atr_pct is not None else None,
        "Avg Volume": int(avg_volume) if avg_volume is not None else None,
        "Relative Volume": round(rel_volume, 2) if rel_volume is not None else None,
        "Momentum %": round(momentum, 2) if momentum is not None else None,
        "Current Price": round(current_price, 2) if current_price is not None else None,
        "Signal": signal,
        "Signal Strength": signal_strength,
        "RSI": round(rsi, 2) if rsi is not None else None,
        "MA5": round(short_ma, 2) if short_ma is not None else None,
        "MA20": round(long_ma, 2) if long_ma is not None else None,
    }

    rec_score = calculate_recommendation_score(metrics)
    rec_label = get_recommendation_label(rec_score, signal)

    metrics["Recommendation Score"] = rec_score
    metrics["Recommendation"] = rec_label

    try:
        daily_range = float(daily_range) if daily_range is not None else None
        atr_pct = float(atr_pct) if atr_pct is not None else None
        avg_volume = float(avg_volume) if avg_volume is not None else None
        rel_volume = float(rel_volume) if rel_volume is not None else None
        current_price = float(current_price) if current_price is not None else None
        momentum = float(momentum) if momentum is not None else None
    except Exception:
        return False, {}

    criteria_met = (
        daily_range is not None
        and daily_range >= 1.5
        and atr_pct is not None
        and atr_pct >= 1.5
        and avg_volume is not None
        and avg_volume >= 1000000
        and rel_volume is not None
        and rel_volume >= 0.6
        and current_price is not None
        and current_price >= 50
        and current_price <= 3000
        and momentum is not None
        and momentum >= 0.0
        and signal in ["BUY", "SELL"]
    )

    return criteria_met, metrics


def process_ticker(ticker, progress_callback=None):
    try:
        data = get_historical_data(ticker, period="3mo", delay=2.0, timeout=15)
        if data is None:
            if progress_callback:
                progress_callback(ticker, False, "No data")
            return None

        is_suitable, metrics = is_suitable_for_intraday(ticker, data)
        if is_suitable:
            stock_info = {"Stock": ticker}
            stock_info.update(metrics)
            if progress_callback:
                progress_callback(ticker, True, metrics["Signal"])
            return stock_info
        else:
            if progress_callback:
                progress_callback(ticker, False, "Criteria not met")
            return None
    except Exception as e:
        if progress_callback:
            progress_callback(ticker, False, f"Error: {str(e)[:20]}")
        return None


def select_stocks_for_intraday(csv_file="ind_nifty500list.csv"):
    print("🚀 Starting stock selection for intraday trading...")
    print(f"📅 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    validated_csv = "validated_nifty500_tickers.csv"
    if os.path.exists(validated_csv):
        csv_file = validated_csv
        print(f"📂 Using validated tickers from '{csv_file}'")
    else:
        print(f"📂 Validated CSV not found, using '{csv_file}'")

    tickers = read_stocks_from_csv(csv_file)
    if not tickers:
        print("❌ No stock tickers found. Exiting...")
        return

    print(f"\n🔍 Analyzing {len(tickers)} stocks with parallel processing...")
    selected_stocks = []

    processed_count = 0
    selected_count = 0
    buy_signals = 0
    sell_signals = 0

    def progress_callback(ticker, selected, status):
        nonlocal processed_count, selected_count, buy_signals, sell_signals
        processed_count += 1

        if selected:
            selected_count += 1
            if status == "BUY":
                buy_signals += 1
            elif status == "SELL":
                sell_signals += 1

        if processed_count % 10 == 0 or processed_count == len(tickers):
            progress_pct = (processed_count / len(tickers)) * 100
            print(
                f"\r📊 Progress: {processed_count}/{len(tickers)} ({progress_pct:.1f}%) | "
                f"Selected: {selected_count} | 🟢 BUY: {buy_signals} | 🔴 SELL: {sell_signals}",
                end="",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(process_ticker, ticker, progress_callback): ticker
            for ticker in tickers
        }

        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
                if result:
                    selected_stocks.append(result)
            except Exception as e:
                print(f"\n❌ Error processing {ticker}: {e}")

    print()

    selected_stocks.sort(
        key=lambda x: (-x["Recommendation Score"], signal_priority.get(x["Signal"], 3))
    )

    if selected_stocks:
        print("\n" + "=" * 140)
        print("🎯 SELECTED STOCKS FOR INTRADAY TRADING (BUY/SELL SIGNALS ONLY)")
        print("=" * 140)
        print(
            f"{'No':<3} {'Stock':<10} {'Signal':<6} {'Recommendation':<16} {'Score':<5} {'Price':<8} {'Range%':<7} {'ATR%':<6} {'Volume':<10} {'RelVol':<6} {'RSI':<5}"
        )
        print("-" * 140)

        for i, stock in enumerate(selected_stocks, 1):
            signal_color = "🟢" if stock["Signal"] == "BUY" else "🔴"
            print(
                f"{i:<3} {stock['Stock']:<10} {signal_color}{stock['Signal']:<5} "
                f"{stock['Recommendation']:<16} {stock['Recommendation Score']:<5} "
                f"₹{stock['Current Price']:<7.2f} {stock['Daily Range %']:<6.2f}% "
                f"{stock['ATR %']:<5.2f}% {stock['Avg Volume']:>9,} "
                f"{stock['Relative Volume']:<5.2f} {stock['RSI'] or 'N/A':<5}"
            )

        buy_count = sum(1 for stock in selected_stocks if stock["Signal"] == "BUY")
        sell_count = sum(1 for stock in selected_stocks if stock["Signal"] == "SELL")

        strong_buy = sum(
            1 for stock in selected_stocks if "STRONG BUY" in stock["Recommendation"]
        )
        strong_sell = sum(
            1 for stock in selected_stocks if "STRONG SELL" in stock["Recommendation"]
        )
        good_buy = sum(
            1 for stock in selected_stocks if "GOOD BUY" in stock["Recommendation"]
        )
        good_sell = sum(
            1 for stock in selected_stocks if "GOOD SELL" in stock["Recommendation"]
        )

        print("\n" + "=" * 140)
        print("📈 SIGNAL SUMMARY:")
        print(
            f"🟢 BUY Signals: {buy_count} (🔥 Strong: {strong_buy}, ⭐ Good: {good_buy})"
        )
        print(
            f"🔴 SELL Signals: {sell_count} (🔥 Strong: {strong_sell}, ⭐ Good: {good_sell})"
        )
        print(f"📊 Total Selected: {len(selected_stocks)} stocks")
        print(f"🎯 Success Rate: {(len(selected_stocks)/len(tickers)*100):.1f}%")

        if len(selected_stocks) > 0:
            print(f"\n🏆 TOP 5 RECOMMENDATIONS:")
            for i, stock in enumerate(selected_stocks[:5], 1):
                signal_emoji = "🟢" if stock["Signal"] == "BUY" else "🔴"
                print(
                    f"{i}. {stock['Stock']} - {stock['Recommendation']} (Score: {stock['Recommendation Score']}) {signal_emoji}"
                )

        print("=" * 140)

        df = pd.DataFrame(selected_stocks)
        column_order = [
            "Stock",
            "Signal",
            "Recommendation",
            "Recommendation Score",
            "Current Price",
            "Daily Range %",
            "ATR %",
            "Avg Volume",
            "Relative Volume",
            "Momentum %",
            "RSI",
            "MA5",
            "MA20",
            "Signal Strength",
        ]
        df = df[column_order]
        df.to_csv("selected_stocks_with_recommendations.csv", index=False)
        print("💾 Results saved to 'selected_stocks_with_recommendations.csv'")

    else:
        print("\n" + "=" * 80)
        print("🎯 SELECTED STOCKS FOR INTRADAY TRADING")
        print("=" * 80)
        print("❌ No stocks with BUY/SELL signals meet the criteria.")
        print("\n💡 Try adjusting the filtering criteria:")
        print("- Lower the minimum daily range requirement")
        print("- Reduce the minimum volume requirement")
        print("- Adjust the price range filters")
        print("- Check market conditions (trending vs sideways)")


def print_usage_instructions():
    print("\n" + "=" * 80)
    print("🚀 ENHANCED STOCK FILTER SCRIPT - USAGE INSTRUCTIONS")
    print("=" * 80)
    print("\n✨ NEW FEATURES:")
    print("✅ Only BUY/SELL signals (HOLD signals filtered out)")
    print("✅ Recommendation scoring system (0-100 scale)")
    print("✅ Recommendation labels (🔥 STRONG, ⭐ GOOD, ✅ MODERATE, 🔸 WEAK)")
    print("✅ Stocks sorted by recommendation score (best first)")
    print("✅ Top 5 recommendations highlight")
    print("✅ Enhanced signal strength calculation")
    print("✅ Real-time progress tracking with counts")
    print("✅ Enhanced visual feedback with emojis")
    print("✅ Success rate calculation")
    print("✅ Improved CSV output with recommendations")
    print("✅ Better error handling and timeouts")
    print("\n🔧 USAGE:")
    print("select_stocks_for_intraday(csv_file='ind_nifty500list.csv')")
    print("\n📊 RECOMMENDATION SCORING:")
    print("- Signal Strength: Technical indicator consensus (0-20 points)")
    print("- Daily Range: Higher volatility = better for intraday (0-20 points)")
    print("- ATR: Average True Range percentage (0-15 points)")
    print("- Volume: Relative volume vs average (0-15 points)")
    print("- Momentum: Price momentum strength (0-10 points)")
    print("- RSI: Extreme values get bonus (0-8 points)")
    print("- Price Range: Bonus for optimal price range (0-5 points)")
    print("- Total Score: 0-100 (Higher = Better recommendation)")
    print("\n🏆 RECOMMENDATION LABELS:")
    print("🔥 STRONG BUY/SELL: Score ≥ 50")
    print("⭐ GOOD BUY/SELL: Score ≥ 35")
    print("✅ MODERATE BUY/SELL: Score ≥ 20")
    print("🔸 WEAK BUY/SELL: Score < 20")
    print("\n📄 CSV FILE FORMAT:")
    print("- Must contain a 'ticker' column with stock symbols")
    print("- Symbols can be with or without .NS suffix")
    print("- Example CSV content:")
    print("  ticker")
    print("  RELIANCE")
    print("  TCS")
    print("  HDFCBANK")
    print("=" * 80)


if __name__ == "__main__":
    print_usage_instructions()
    select_stocks_for_intraday(csv_file="ind_nifty500list.csv")
