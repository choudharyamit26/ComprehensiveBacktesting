import pandas as pd
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt


def fetch_and_process_data(
    tickers, labels, start_date, end_date, output_file, plot_title
):
    """
    Fetch historical data for given tickers, calculate daily percentage changes,
    save to CSV, and plot a comparison chart.

    Args:
        tickers (list): List of Yahoo Finance ticker symbols.
        labels (list): List of display names for each ticker.
        start_date (datetime): Start date for data fetch.
        end_date (datetime): End date for data fetch.
        output_file (str): Path to save the output CSV.
        plot_title (str): Title for the plot.

    Returns:
        None
    """
    # Fetch data for all tickers
    data_dict = {}
    try:
        for ticker in tickers:
            print(f"Fetching data for {ticker} from {start_date} to {end_date}")
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            if data.empty:
                print(f"No data retrieved for {ticker}. Check ticker or date range.")
                exit()
            data_dict[ticker] = data
    except Exception as e:
        print(f"Error fetching data: {e}")
        exit()

    # Extract closing prices from MultiIndex
    close_prices = {}
    for ticker in tickers:
        try:
            # Handle MultiIndex columns (e.g., ('Close', ticker))
            close_prices[ticker] = data_dict[ticker][("Close", ticker)]
        except KeyError as e:
            print(f"Error accessing 'Close' for {ticker}: {e}")
            print(f"Available columns for {ticker}: {data_dict[ticker].columns}")
            exit()

    # Calculate daily percentage changes
    pct_changes = {}
    for ticker, label in zip(tickers, labels):
        col_name = f"{label}_Pct_Change"
        pct_changes[col_name] = close_prices[ticker].pct_change() * 100

    # Create DataFrame with explicit column names
    df_data = {"Date": close_prices[tickers[0]].index}
    for label in labels:
        col_name = f"{label}_Pct_Change"
        df_data[col_name] = pct_changes[col_name].values
    df = pd.DataFrame(df_data)

    # Drop rows with NaN values
    df = df.dropna()
    if df.empty:
        print("No valid data after calculating percentage changes.")
        exit()

    # Debug: Print DataFrame columns
    print(f"DataFrame columns for {output_file}: {df.columns.tolist()}")

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

    # Plot the daily percentage changes
    plt.figure(figsize=(12, 6))
    colors = ["blue", "orange", "green"][
        : len(tickers)
    ]  # Adjust colors based on ticker count
    for label, color in zip(labels, colors):
        col_name = f"{label}_Pct_Change"
        plt.plot(df["Date"], df[col_name], label=label, color=color)
    plt.title(plot_title)
    plt.xlabel("Date")
    plt.ylabel("Daily % Change")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to configure and run comparisons for NIFTY 50 and BANKNIFTY groups."""
    # Define date range (May 1, 2025 to August 1, 2025)
    end_date = datetime(2025, 8, 1)
    start_date = end_date - relativedelta(months=3)

    # Define ticker groups
    comparisons = [
        {
            "tickers": ["^NSEI", "SETFNIF50.NS", "NIFTYBEES.NS"],
            "labels": ["NIFTY 50", "SBI Nifty 50 ETF", "Nippon Nifty 50 ETF"],
            "output_file": "nifty50_sbi_nippon_etf_daily_data.csv",
            "plot_title": "Daily Percentage Change: NIFTY 50 vs SBI Nifty 50 ETF vs Nippon Nifty 50 ETF (May 1, 2025 - Aug 1, 2025)",
        },
        {
            "tickers": ["^NSEBANK", "BANKBEES.NS"],
            "labels": ["BANKNIFTY", "Nippon Nifty Bank ETF"],
            "output_file": "banknifty_bankbees_daily_data.csv",
            "plot_title": "Daily Percentage Change: BANKNIFTY vs Nippon Nifty Bank ETF (May 1, 2025 - Aug 1, 2025)",
        },
    ]

    # Process each comparison
    for comp in comparisons:
        fetch_and_process_data(
            tickers=comp["tickers"],
            labels=comp["labels"],
            start_date=start_date,
            end_date=end_date,
            output_file=comp["output_file"],
            plot_title=comp["plot_title"],
        )


if __name__ == "__main__":
    main()
