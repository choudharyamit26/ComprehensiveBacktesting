import yfinance as yf
import pandas as pd
import time
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


def read_stocks_from_csv(csv_file="ind_nifty50list.csv"):
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

        print(f"Read {len(cleaned_symbols)} stock symbols from '{csv_file}'")

        valid_symbols = []
        for symbol in cleaned_symbols:
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
                            print(
                                f"[WARNING] {symbol}: Data does not contain {nse_ticker}. Available: {[col[1] for col in test_data.columns]}"
                            )
                            continue
                        test_data = test_data.xs(
                            nse_ticker, level="Ticker", axis=1, drop_level=True
                        )
                    if all(
                        col in test_data.columns
                        for col in ["High", "Low", "Close", "Volume"]
                    ):
                        valid_symbols.append(symbol)
                    else:
                        print(
                            f"[WARNING] {symbol}: Missing required columns. Available: {list(test_data.columns)}"
                        )
                else:
                    print(f"[WARNING] {symbol}: Empty DataFrame returned.")
            except Exception as e:
                print(f"[WARNING] {symbol}: Failed to validate ticker: {e}")

        print(f"Validated {len(valid_symbols)}/{len(cleaned_symbols)} tickers")
        if valid_symbols:
            valid_df = pd.DataFrame(valid_symbols, columns=["ticker"])
            valid_df.to_csv("validated_nifty50_tickers.csv", index=False)
            print("Validated tickers saved to 'validated_nifty50_tickers.csv'")

        return valid_symbols

    except Exception as e:
        print(f"Error reading CSV file '{csv_file}': {e}")
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
                        print(
                            f"[ERROR] {ticker}: Data does not contain {nse_ticker}. Available: {[col[1] for col in data.columns]}"
                        )
                        return None
                required_columns = ["High", "Low", "Close", "Volume"]
                if not all(col in data.columns for col in required_columns):
                    print(
                        f"[ERROR] {ticker}: Missing columns {required_columns}. Available: {list(data.columns)}"
                    )
                    return None
                return data
            else:
                print(f"[ERROR] {ticker}: Empty DataFrame returned.")
                return None
        except Exception as e:
            print(f"[ERROR] Attempt {attempt+1}/{retries} failed for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            continue
    print(f"[ERROR] Failed to fetch data for {ticker} after {retries} attempts")
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


def is_suitable_for_intraday(ticker, data):
    if (
        data is None
        or not isinstance(data, pd.DataFrame)
        or data.empty
        or len(data) < 20
    ):
        print(f"[DEBUG] {ticker}: Data is empty or insufficient.")
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
    }

    print(f"[DEBUG] {ticker}: Metrics: {metrics}")

    try:
        daily_range = float(daily_range) if daily_range is not None else None
        atr_pct = float(atr_pct) if atr_pct is not None else None
        avg_volume = float(avg_volume) if avg_volume is not None else None
        rel_volume = float(rel_volume) if rel_volume is not None else None
        current_price = float(current_price) if current_price is not None else None
        momentum = float(momentum) if momentum is not None else None
    except Exception:
        print(f"[DEBUG] {ticker}: Type casting failed for metrics.")
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
    )

    return criteria_met, metrics


def process_ticker(ticker):
    try:
        data = get_historical_data(ticker, period="3mo", delay=2.0, timeout=15)
        if data is None:
            return None

        is_suitable, metrics = is_suitable_for_intraday(ticker, data)
        if is_suitable:
            stock_info = {"Stock": ticker}
            stock_info.update(metrics)
            return stock_info
    except Exception as e:
        print(f"[ERROR] Error processing {ticker}: {e}")
    return None


def select_stocks_for_intraday(csv_file="ind_nifty50list.csv"):
    print("Starting stock selection for intraday trading...")
    validated_csv = "validated_nifty50_tickers.csv"
    if os.path.exists(validated_csv):
        csv_file = validated_csv
        print(f"Using validated tickers from '{csv_file}'")
    else:
        print(f"Validated CSV not found, using '{csv_file}'")

    tickers = read_stocks_from_csv(csv_file)
    if not tickers:
        print("No stock tickers found. Exiting...")
        return

    print(f"Analyzing {len(tickers)} stocks with parallel processing...")
    selected_stocks = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(process_ticker, ticker): ticker for ticker in tickers
        }

        for i, future in enumerate(as_completed(futures)):
            ticker = futures[future]
            try:
                result = future.result()
                if result:
                    selected_stocks.append(result)
                    print(
                        f"✓ Selected: {result['Stock']} - Range: {result['Daily Range %']}%"
                    )
            except Exception as e:
                print(f"[ERROR] Error processing {ticker}: {e}")

            if (i + 1) % 10 == 0 or (i + 1) == len(tickers):
                print(f"Processed {i+1}/{len(tickers)} stocks...")

    if selected_stocks:
        print("\n" + "=" * 80)
        print("SELECTED STOCKS FOR INTRADAY TRADING (5-min timeframe)")
        print("=" * 80)
        for i, stock in enumerate(selected_stocks, 1):
            print(
                f"{i}. {stock['Stock']:<12} | Price: ₹{stock['Current Price']:<7.2f} | "
                f"Range: {stock['Daily Range %']:<5.2f}% | ATR: {stock['ATR %']:<5.2f}% | "
                f"Volume: {stock['Avg Volume']:,} | RelVol: {stock['Relative Volume']:<5.2f}"
            )
    else:
        print("\n" + "=" * 80)
        print("SELECTED STOCKS FOR INTRADAY TRADING (5-min timeframe)")
        print("=" * 80)
        print("No stocks meet the criteria for intraday trading.")
        print("\nTry adjusting the filtering criteria:")
        print("- Lower the minimum daily range requirement")
        print("- Reduce the minimum volume requirement")
        print("- Adjust the price range filters")


def print_usage_instructions():
    print("\n" + "=" * 80)
    print("STOCK FILTER SCRIPT - USAGE INSTRUCTIONS")
    print("=" * 80)
    print("\nMETHOD: Using CSV File")
    print("select_stocks_for_intraday(csv_file='ind_nifty50list.csv')")
    print("\nCSV FILE FORMAT:")
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
    select_stocks_for_intraday(csv_file="ind_nifty50list.csv")
