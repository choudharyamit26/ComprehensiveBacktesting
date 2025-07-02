import yfinance as yf
import pandas as pd
import asyncio
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


async def get_data(ticker, start_date, end_date, interval="5m"):
    """
    Fetch historical stock data from Yahoo Finance asynchronously.

    Parameters:
    ticker (str): Stock ticker symbol.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    interval (str): Data interval (e.g., '1d', '5m'). Default is '5m'.

    Returns:
    pd.DataFrame: DataFrame containing the stock data with proper column names.
    """
    try:
        # Handle yfinance 60-day limit for intraday data
        intraday_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]
        if interval in intraday_intervals:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            max_days = 60
            if (end_dt - start_dt).days > max_days:
                print(
                    f"Warning: yfinance only provides up to 60 days of intraday data. Adjusting start_date from {start_date} to {(end_dt - pd.Timedelta(days=max_days)).strftime('%Y-%m-%d')}."
                )
                start_date = (end_dt - pd.Timedelta(days=max_days)).strftime("%Y-%m-%d")

        # Download data from Yahoo Finance asynchronously
        # df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        ticker_obj = yf.Ticker(ticker)
        df = await asyncio.to_thread(ticker_obj.history, period="60d", interval="5m")
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError(
                f"No data found for ticker {ticker} between {start_date} and {end_date} (interval={interval})"
            )

        # Reset index to make Date a column
        df.reset_index(inplace=True)

        # Handle the case where Date might be in the index
        if "Datetime" not in df.columns and df.index.name == "Datetime":
            df.reset_index(inplace=True)

        # Handle MultiIndex columns (common with yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten MultiIndex columns - take the first level (which contains OHLCV)
            df.columns = [
                col[0] if isinstance(col, tuple) else col for col in df.columns
            ]

        # Ensure we have the required columns and rename them to standard format
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        column_mapping = {}

        # Create mapping for case-insensitive column matching
        for col in df.columns:
            col_lower = str(col).lower()
            if "open" in col_lower:
                column_mapping[col] = "Open"
            elif "high" in col_lower:
                column_mapping[col] = "High"
            elif "low" in col_lower:
                column_mapping[col] = "Low"
            elif "close" in col_lower:
                column_mapping[col] = "Close"
            elif "volume" in col_lower:
                column_mapping[col] = "Volume"
            elif "adj" in col_lower and "close" in col_lower:
                column_mapping[col] = "Adj Close"

        # Rename columns using the mapping
        df.rename(columns=column_mapping, inplace=True)

        # Verify we have all required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Ensure Date column exists and is properly formatted
        if "Datetime" not in df.columns:
            if df.index.name == "Datetime" or "datetime" in str(df.index.name).lower():
                df.reset_index(inplace=True)
                df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
            else:
                raise ValueError("No Date column found in the data")

        # Convert Date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["Datetime"]):
            df["Datetime"] = pd.to_datetime(df["Datetime"])

        # Set Date as index for Backtrader compatibility
        df.set_index("Datetime", inplace=True)

        # Convert index to IST (Asia/Kolkata)
        if df.index.tz is None or str(df.index.tz) != "Asia/Kolkata":
            df.index = df.index.tz_convert("Asia/Kolkata")

        # Sort by date to ensure chronological order
        df.sort_index(inplace=True)

        # Remove any rows with NaN values in OHLCV columns
        df.dropna(subset=required_columns, inplace=True)

        # Ensure numeric data types for OHLCV columns
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Final check for any remaining NaN values
        df.dropna(subset=required_columns, inplace=True)

        if df.empty:
            raise ValueError(
                f"No valid data remaining after cleaning for ticker {ticker}"
            )

        print(f"Successfully loaded {len(df)} rows of data for {ticker}")
        print(
            f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
        )
        print(f"Columns: {list(df.columns)}")

        return df

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error fetching data for {ticker}: {str(e)}")
        raise


def validate_data(df, strict=False):
    """
    Validate the data format for backtesting with tolerance options.

    Parameters:
    df (pd.DataFrame): DataFrame to validate
    strict (bool): If True, use strict validation. If False, allow minor issues.
    """
    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    # Check index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        logger.warning("Index is not datetime type")
        if strict:
            return False

    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}")
        if strict:
            return False

    # Check for empty DataFrame
    if df.empty:
        logger.warning("DataFrame is empty")
        return False

    # Check for NaN values
    nan_columns = (
        df[required_columns].columns[df[required_columns].isna().any()].tolist()
    )
    if nan_columns:
        logger.warning(f"NaN values in columns: {nan_columns}")
        if strict:
            return False

    # Check High >= Low with tolerance
    invalid_high_low = df[df["High"] < df["Low"]]
    if not invalid_high_low.empty:
        logger.warning(f"High < Low in {len(invalid_high_low)} rows")
        if strict:
            return False

    # Check OHLC consistency with tolerance
    tolerance = 0.01  # 1% tolerance for price anomalies
    invalid_open = df[
        (df["Open"] > df["High"] * (1 + tolerance))
        | (df["Open"] < df["Low"] * (1 - tolerance))
    ]
    invalid_close = df[
        (df["Close"] > df["High"] * (1 + tolerance))
        | (df["Close"] < df["Low"] * (1 - tolerance))
    ]

    if not invalid_open.empty:
        logger.warning(f"Open price anomalies in {len(invalid_open)} rows")
    if not invalid_close.empty:
        logger.warning(f"Close price anomalies in {len(invalid_close)} rows")

    if strict and (not invalid_open.empty or not invalid_close.empty):
        return False

    logger.info("Data validation passed")
    return True


async def get_multiple_tickers_data(tickers, start_date, end_date):
    """
    Fetch data for multiple tickers asynchronously.

    Parameters:
    tickers (list): List of ticker symbols
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
    dict: Dictionary with ticker as key and DataFrame as value
    """
    data_dict = {}

    # Create tasks for concurrent execution
    tasks = []
    for ticker in tickers:
        print(f"Creating task for {ticker}...")
        task = asyncio.create_task(get_data(ticker, start_date, end_date))
        tasks.append((ticker, task))

    # Wait for all tasks to complete
    for ticker, task in tasks:
        try:
            print(f"Fetching data for {ticker}...")
            data_dict[ticker] = await task
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {str(e)}")
            continue

    return data_dict


def get_multiple_tickers_data_sync(tickers, start_date, end_date):
    """
    Fetch data for multiple tickers synchronously (wrapper for async function).

    Parameters:
    tickers (list): List of ticker symbols
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
    dict: Dictionary with ticker as key and DataFrame as value
    """
    return asyncio.run(get_multiple_tickers_data(tickers, start_date, end_date))


async def preview_data(ticker, start_date, end_date, rows=5):
    """
    Preview data for a ticker.

    Parameters:
    ticker (str): Stock ticker symbol
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    rows (int): Number of rows to display
    """
    try:
        df = await get_data(ticker, start_date, end_date)

        print(f"\nData Preview for {ticker}:")
        print("-" * 50)
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print("\nFirst {} rows:".format(rows))
        print(df.head(rows))
        print(f"\nLast {rows} rows:")
        print(df.tail(rows))
        print("\nData types:")
        print(df.dtypes)
        print("\nIndex info:")
        print(f"Index name: {df.index.name}")
        print(f"Index type: {type(df.index)}")
        print("\nBasic statistics:")
        print(df.describe())

        return df

    except Exception as e:
        print(f"Error previewing data for {ticker}: {str(e)}")
        return None


def preview_data_sync(ticker, start_date, end_date, rows=5):
    """
    Preview data for a ticker synchronously (wrapper for async function).

    Parameters:
    ticker (str): Stock ticker symbol
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    rows (int): Number of rows to display
    """
    return asyncio.run(preview_data(ticker, start_date, end_date, rows))


def get_data_sync(ticker, start_date, end_date, interval="5m"):
    """
    Fetch historical stock data from Yahoo Finance synchronously (wrapper for async function).

    Parameters:
    ticker (str): Stock ticker symbol.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    interval (str): Data interval (e.g., '1d', '5m'). Default is '5m'.

    Returns:
    pd.DataFrame: DataFrame containing the stock data with proper column names.
    """
    return asyncio.run(get_data(ticker, start_date, end_date, interval))
