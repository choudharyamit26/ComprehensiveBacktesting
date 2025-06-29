import yfinance as yf
import pandas as pd


def get_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.

    Parameters:
    ticker (str): Stock ticker symbol.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: DataFrame containing the stock data with proper column names.
    """
    try:
        # Download data from Yahoo Finance
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d")

        # Check if DataFrame is empty
        if df.empty:
            raise ValueError(
                f"No data found for ticker {ticker} between {start_date} and {end_date}"
            )

        # Reset index to make Date a column
        df.reset_index(inplace=True)

        # Handle the case where Date might be in the index
        if "Date" not in df.columns and df.index.name == "Date":
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
        if "Date" not in df.columns:
            if df.index.name == "Date" or "date" in str(df.index.name).lower():
                df.reset_index(inplace=True)
                df.rename(columns={df.columns[0]: "Date"}, inplace=True)
            else:
                raise ValueError("No Date column found in the data")

        # Convert Date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"])

        # Set Date as index for Backtrader compatibility
        df.set_index("Date", inplace=True)

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
        print(f"Error fetching data for {ticker}: {str(e)}")
        raise


def validate_data(df):
    """
    Validate the data format for backtesting.

    Parameters:
    df (pd.DataFrame): DataFrame to validate (should have Date as index)

    Returns:
    bool: True if data is valid, False otherwise
    """
    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    # Check if index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        print("Index is not datetime type")
        return False

    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False

    # Check for empty DataFrame
    if df.empty:
        print("DataFrame is empty")
        return False

    # Check for NaN values in OHLCV columns
    ohlcv_columns = ["Open", "High", "Low", "Close", "Volume"]
    nan_columns = []
    for col in ohlcv_columns:
        if df[col].isna().any():
            nan_columns.append(col)

    if nan_columns:
        print(f"NaN values found in columns: {nan_columns}")
        return False

    # Check if High >= Low for all rows
    invalid_high_low = df[df["High"] < df["Low"]]
    if not invalid_high_low.empty:
        print(f"Invalid High/Low values found in {len(invalid_high_low)} rows")
        return False

    # Check if Open and Close are within High/Low range
    invalid_open = df[(df["Open"] > df["High"]) | (df["Open"] < df["Low"])]
    invalid_close = df[(df["Close"] > df["High"]) | (df["Close"] < df["Low"])]

    if not invalid_open.empty:
        print(f"Invalid Open values found in {len(invalid_open)} rows")
    if not invalid_close.empty:
        print(f"Invalid Close values found in {len(invalid_close)} rows")

    if not invalid_open.empty or not invalid_close.empty:
        return False

    print("Data validation passed")
    return True


def get_multiple_tickers_data(tickers, start_date, end_date):
    """
    Fetch data for multiple tickers.

    Parameters:
    tickers (list): List of ticker symbols
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
    dict: Dictionary with ticker as key and DataFrame as value
    """
    data_dict = {}

    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            data_dict[ticker] = get_data(ticker, start_date, end_date)
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {str(e)}")
            continue

    return data_dict


def preview_data(ticker, start_date, end_date, rows=5):
    """
    Preview data for a ticker.

    Parameters:
    ticker (str): Stock ticker symbol
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    rows (int): Number of rows to display
    """
    try:
        df = get_data(ticker, start_date, end_date)

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


if __name__ == "__main__":
    # Test the data fetching function
    test_ticker = "AAPL"
    test_start = "2024-01-01"
    test_end = "2024-12-31"

    print(f"Testing data fetch for {test_ticker}")
    try:
        df = preview_data(test_ticker, test_start, test_end)
        if df is not None:
            is_valid = validate_data(df)
            print(f"Data validation result: {is_valid}")
    except Exception as e:
        print(f"Test failed: {str(e)}")
