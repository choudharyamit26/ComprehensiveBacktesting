import os
import yfinance as yf
import pandas as pd
import asyncio
import logging
import requests
import pytz
from datetime import datetime, time, timedelta, date
from dhanhq import dhanhq
from retrying import retry
from dotenv import load_dotenv

load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

CONFIG = {
    "CLIENT_ID": os.getenv("DHAN_CLIENT_ID"),
    "ACCESS_TOKEN": os.getenv("DHAN_ACCESS_TOKEN"),
    "EXCHANGE_SEGMENT": "NSE_EQ",
    "TIMEFRAME": 5,
    "MARKET_OPEN": time(9, 15),
    "MARKET_CLOSE": time(15, 30),
    "EXIT_BUFFER_MINUTES": 15,
    "CSV_FILE": "trading_signals.csv",
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID"),
    "DAYS_TO_FETCH": 2,
    "SIMULATE_MODE": True,
    "INITIAL_CAPITAL": 150000,
}
print(f"Loaded configuration: {CONFIG}")


def init_dhan_client():
    client_id = CONFIG["CLIENT_ID"]
    access_token = CONFIG["ACCESS_TOKEN"]
    if not client_id or not access_token:
        logger.error("DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN not set")
        return None
    try:
        dhan = dhanhq(client_id=client_id, access_token=access_token)
        logger.info("DhanHQ client initialized successfully")
        return dhan
    except Exception as e:
        logger.error(f"Error initializing DhanHQ client: {e}")
        return None


dhan = init_dhan_client()


def get_security_id(ticker, csv_path="ind_nifty500list.csv"):
    if not isinstance(ticker, str):
        logger.error(f"Invalid ticker: {ticker}, must be a string")
        return None
    if not os.path.exists(csv_path):
        logger.error(f"CSV file {csv_path} not found")
        return None
    try:
        df = pd.read_csv(csv_path)
        match = df[df["ticker"].str.upper() == ticker.upper()]
        if not match.empty:
            security_id = int(match["security_id"].iloc[0])
            logger.info(f"Found security ID {security_id} for ticker {ticker}")
            return security_id
        else:
            logger.warning(f"No security ID found for ticker {ticker} in {csv_path}")
            return None
    except Exception as e:
        logger.error(f"Failed to read security ID from {csv_path}: {e}")
        return None


def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{CONFIG['TELEGRAM_BOT_TOKEN']}/sendMessage"
        payload = {"chat_id": CONFIG["TELEGRAM_CHAT_ID"], "text": message}
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            logger.info(f"Telegram alert sent: {message}")
        else:
            logger.error(f"Failed to send Telegram alert: {response.text}")
    except Exception as e:
        logger.error(f"Error sending Telegram alert: {e}")


@retry(
    stop_max_attempt_number=3,
    wait_exponential_multiplier=2000,
    wait_exponential_max=30000,
)
async def fetch_historical_data(
    security_id, from_date, to_date, interval, exchange_segment="NSE_EQ"
):
    interval = (
        interval.replace("m", "") if "m" in interval else interval.replace("d", "")
    )
    if "m" in interval:
        interval = interval.strip().replace("m", "")
    elif "d" in interval:
        interval = interval.strip().replace("d", "")
    elif "h" in interval:
        interval = interval.strip().replace("h", "")
    if not isinstance(security_id, int):
        logger.error(f"Invalid security_id: {security_id}, must be an integer")
        return None
    if not dhan:
        logger.error("Cannot fetch historical data: Dhan client not initialized")
        return None
    try:
        ist_tz = pytz.timezone("Asia/Kolkata")
        if isinstance(from_date, date) and not isinstance(from_date, datetime):
            from_date = datetime.combine(from_date, time(0, 0), tzinfo=ist_tz)
        if isinstance(to_date, date) and not isinstance(to_date, datetime):
            to_date = datetime.combine(to_date, time(23, 59, 59), tzinfo=ist_tz)

        today = datetime.now(ist_tz)
        current_date = today - timedelta(days=1)
        trading_sessions = {security_id: []}
        days_checked = 0
        max_days_to_check = 10
        NSE_HOLIDAYS_2025 = [
            "2025-01-26",
            "2025-03-14",
        ]
        while (
            len(trading_sessions[security_id]) < 2 and days_checked < max_days_to_check
        ):
            if (
                current_date.weekday() >= 5
                or current_date.strftime("%Y-%m-%d") in NSE_HOLIDAYS_2025
            ):
                logger.info(f"Skipping non-trading day {current_date.date()}")
                current_date -= timedelta(days=1)
                days_checked += 1
                continue
            from_date_str = from_date
            to_date_str = to_date
            if isinstance(from_date, datetime):
                from_date_str = from_date.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(to_date, datetime):
                to_date_str = to_date.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(
                f"Fetching historical data for {security_id} from {from_date_str} to {to_date_str}"
            )
            print(
                f"Fetching data for {security_id} from {from_date_str} to {to_date_str} for {interval} minute interval"
            )
            data = dhan.intraday_minute_data(
                security_id=security_id,
                exchange_segment=exchange_segment,
                instrument_type="EQUITY",
                interval=int(interval),
                from_date=from_date_str,
                to_date=to_date_str,
            )
            if (
                data
                and data.get("status") == "success"
                and "data" in data
                and data["data"]
            ):
                df_chunk = pd.DataFrame(data["data"])
                required_fields = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "timestamp",
                ]
                missing_fields = [
                    field for field in required_fields if field not in df_chunk.columns
                ]
                if missing_fields:
                    logger.warning(
                        f"Missing fields {missing_fields} for {security_id} from {from_date_str} to {to_date_str}"
                    )
                    current_date -= timedelta(days=1)
                    days_checked += 1
                    continue
                df_chunk["datetime"] = pd.to_datetime(
                    df_chunk["timestamp"], unit="s", utc=True, errors="coerce"
                )
                df_chunk["datetime"] = df_chunk["datetime"].dt.tz_convert(ist_tz)
                df_chunk = df_chunk.dropna(subset=["datetime"])
                valid_date_range = (df_chunk["datetime"] >= from_date) & (
                    df_chunk["datetime"] <= to_date
                )
                df_chunk = df_chunk[valid_date_range]
                if len(df_chunk) < 50:
                    logger.info(
                        f"Insufficient data ({len(df_chunk)} rows) for {security_id} on {from_date.date()}"
                    )
                    current_date -= timedelta(days=1)
                    days_checked += 1
                    continue
                df_chunk = df_chunk[
                    ["datetime", "open", "high", "low", "close", "volume"]
                ]
                trading_sessions[security_id].append(df_chunk)
                logger.info(
                    f"Fetched {len(df_chunk)} rows for {security_id} from {from_date_str} to {to_date_str}"
                )
            else:
                logger.warning(
                    f"No data or failed fetch for {security_id} from {from_date_str} to {to_date_str}: {data.get('remarks', 'No remarks')}"
                )
            current_date -= timedelta(days=1)
            days_checked += 1
        logger.info(
            f"Checked {days_checked} days, found {len(trading_sessions[security_id])} trading sessions for {security_id}"
        )
        sessions = trading_sessions[security_id]
        if len(sessions) >= 2:
            df = pd.concat(sessions, ignore_index=True)
            df = df.drop_duplicates(subset=["datetime"], keep="last")
            df = df.sort_values("datetime").reset_index(drop=True)
            logger.info(
                f"Combined {len(df)} rows of historical data for {security_id} across {len(sessions)} trading sessions"
            )
            return df
        else:
            logger.error(
                f"Failed to fetch data for 2 trading sessions for {security_id} after checking {days_checked} days"
            )
            return None
    except Exception as e:
        logger.error(f"Error fetching historical data for {security_id}: {e}")
        raise


@retry(
    stop_max_attempt_number=3,
    wait_exponential_multiplier=2000,
    wait_exponential_max=30000,
)
async def fetch_intraday_data(ticker, exchange_segment="NSE_EQ"):
    """
    Fetch intraday data for multiple security IDs from DhanHQ API.

    Parameters:
    security_ids (list): List of security IDs to fetch data for.
    exchange_segment (str): Exchange segment, default is 'NSE_EQ'.

    Returns:
    pd.DataFrame: DataFrame containing intraday data for all securities or None if fetch fails.
    """

    if not dhan:
        logger.error("Cannot fetch intraday data: Dhan client not initialized")
        return None
    try:
        ist_tz = pytz.timezone("Asia/Kolkata")
        today = datetime.now(ist_tz).date()
        from_date = datetime.combine(today, time(9, 15), tzinfo=ist_tz)
        to_date = datetime.combine(today, time(15, 30), tzinfo=ist_tz)
        security_ids = get_security_id(ticker)
        logger.info(f"Fetching intraday data for {len(security_ids)} securities")
        securities = {exchange_segment: [security_ids]}
        data = dhan.quote_data(securities)

        if not data or data.get("status") != "success" or not data.get("data"):
            logger.error(
                f"API call failed: {data.get('remarks', 'Unknown error') if data else 'No response'}"
            )
            return None

        exchange_data = data["data"].get(exchange_segment)
        if not exchange_data:
            logger.error(f"No data for exchange segment {exchange_segment}")
            return None

        records = []
        for security_id_str, instrument_data in exchange_data.items():
            try:
                security_id = int(security_id_str)
                if security_id not in security_ids:
                    logger.debug(f"Skipping security {security_id} - not in watchlist")
                    continue
                if not instrument_data or not isinstance(instrument_data, dict):
                    logger.warning(f"No valid data for security {security_id}")
                    continue

                # Map security_id to ticker
                df_csv = pd.read_csv("ind_nifty500list.csv")
                match = df_csv[df_csv["security_id"] == security_id]
                symbol = (
                    match["ticker"].iloc[0]
                    if not match.empty
                    else f"UNKNOWN_{security_id}"
                )

                last_trade_time = instrument_data.get("last_trade_time")
                if last_trade_time:
                    try:
                        last_trade_time_parsed = pd.to_datetime(
                            last_trade_time, format="%d/%m/%Y %H:%M:%S", errors="coerce"
                        )
                        if pd.isna(last_trade_time_parsed):
                            for fmt in ["%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S"]:
                                try:
                                    last_trade_time_parsed = pd.to_datetime(
                                        last_trade_time, format=fmt, errors="raise"
                                    )
                                    break
                                except:
                                    continue
                            else:
                                raise ValueError(
                                    f"Could not parse timestamp: {last_trade_time}"
                                )
                        if last_trade_time_parsed.tz is None:
                            last_trade_time_parsed = ist_tz.localize(
                                last_trade_time_parsed
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse timestamp for {security_id}: {e}"
                        )
                        last_trade_time_parsed = datetime.now(ist_tz)
                else:
                    last_trade_time_parsed = datetime.now(ist_tz)

                ohlc = instrument_data.get("ohlc", {})
                if not ohlc or not isinstance(ohlc, dict):
                    logger.warning(f"No OHLC data for security {security_id}")
                    continue

                last_price = instrument_data.get("last_price", 0)
                record = {
                    "datetime": last_trade_time_parsed,
                    "open": float(ohlc.get("open", 0)),
                    "high": float(ohlc.get("high", 0)),
                    "low": float(ohlc.get("low", 0)),
                    "close": float(last_price),
                    "volume": float(instrument_data.get("volume", 0)),
                    "security_id": security_id,
                    "symbol": symbol,
                }

                required_fields = ["open", "high", "low", "close", "volume", "datetime"]
                missing_fields = [
                    field for field in required_fields if not record[field]
                ]
                if missing_fields:
                    logger.warning(
                        f"Missing fields {missing_fields} for security {security_id}"
                    )
                    continue

                if (
                    record["open"] >= 0
                    and record["high"] >= 0
                    and record["low"] >= 0
                    and record["close"] >= 0
                ):
                    records.append(record)
                    logger.info(
                        f"Added record for {symbol}: O={record['open']}, H={record['high']}, L={record['low']}, C={record['close']}, V={record['volume']}"
                    )
                else:
                    logger.warning(f"Invalid OHLC data for {symbol}: {record}")
            except Exception as e:
                logger.warning(f"Error processing data for {security_id}: {e}")
                continue

        if not records:
            logger.error("No valid records to process")
            return None

        df = pd.DataFrame(records)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        df["datetime"] = df["datetime"].dt.tz_convert(ist_tz)
        valid_date_range = (df["datetime"].dt.date == today) & (
            (df["datetime"].dt.time >= time(9, 15))
            & (df["datetime"].dt.time <= time(15, 30))
        )
        df = df[valid_date_range]

        if df.empty:
            logger.error(f"No valid intraday data for {exchange_segment} on {today}")
            return None

        df = df[
            [
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "security_id",
                "symbol",
            ]
        ]
        df = df.sort_values(["security_id", "datetime"]).reset_index(drop=True)
        logger.info(
            f"Combined {len(df)} rows of intraday data across {len(set(df['security_id']))} securities"
        )

        return df

    except Exception as e:
        logger.error(f"Error fetching intraday data: {e}")
        raise


async def get_data(ticker, start_date, end_date, interval):
    try:
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()
        ist_tz = pytz.timezone("Asia/Kolkata")
        today = datetime.now(ist_tz).date()
        if end_dt > today:
            logger.warning(f"End date {end_dt} is in the future. Adjusting to {today}.")
            end_dt = today
        chunk_size = timedelta(days=90)
        chunks = []
        current_start = start_dt
        while current_start <= end_dt:
            current_end = min(current_start + chunk_size - timedelta(days=1), end_dt)
            chunks.append((current_start, current_end))
            current_start = current_end + timedelta(days=1)
        security_id = get_security_id(ticker)
        dfs = []
        for chunk_start, chunk_end in chunks:
            logger.info(
                f"Fetching chunk for {ticker} from {chunk_start} to {chunk_end}"
            )
            df_chunk = await fetch_historical_data(
                security_id, chunk_start, chunk_end, interval, exchange_segment="NSE_EQ"
            )
            if df_chunk is not None and not df_chunk.empty:
                dfs.append(df_chunk)
            else:
                logger.warning(
                    f"No data fetched for {ticker} from {chunk_start} to {chunk_end}"
                )
        if not dfs:
            raise ValueError(
                f"No data found for ticker {ticker} between {start_date} and {end_date} (interval={interval})"
            )
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["datetime"], keep="last")
        df = df.sort_values("datetime").reset_index(drop=True)
        logger.info(
            f"Merged {len(dfs)} chunks into DataFrame with {len(df)} rows for {ticker}"
        )
        if df.empty:
            raise ValueError(
                f"No data found for ticker {ticker} between {start_date} and {end_date} (interval={interval})"
            )
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        column_mapping = {}
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
        df.rename(columns=column_mapping, inplace=True)
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        if "datetime" not in df.columns:
            raise ValueError("No 'datetime' column found in the data")
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")
        elif str(df.index.tz) != "Asia/Kolkata":
            df.index = df.index.tz_convert("Asia/Kolkata")
        df.sort_index(inplace=True)
        df.dropna(subset=required_columns, inplace=True)
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
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
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        raise


def validate_data(df, strict=False):
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        logger.warning("Index is not datetime type")
        if strict:
            return False
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}")
        if strict:
            return False
    if df.empty:
        logger.warning("DataFrame is empty")
        return False
    nan_columns = (
        df[required_columns].columns[df[required_columns].isna().any()].tolist()
    )
    if nan_columns:
        logger.warning(f"NaN values in columns: {nan_columns}")
        if strict:
            return False
    invalid_high_low = df[df["High"] < df["Low"]]
    if not invalid_high_low.empty:
        logger.warning(f"High < Low in {len(invalid_high_low)} rows")
        if strict:
            return False
    tolerance = 0.01
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
    data_dict = {}
    tasks = []
    for ticker in tickers:
        print(f"Creating task for {ticker}...")
        task = asyncio.create_task(get_data(ticker, start_date, end_date))
        tasks.append((ticker, task))
    for ticker, task in tasks:
        try:
            print(f"Fetching data for {ticker}...")
            data_dict[ticker] = await task
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {str(e)}")
            continue
    return data_dict


def get_multiple_tickers_data_sync(tickers, start_date, end_date):
    return asyncio.run(get_multiple_tickers_data(tickers, start_date, end_date))


async def preview_data(ticker, start_date, end_date, rows=5):
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
    return asyncio.run(preview_data(ticker, start_date, end_date, rows))


def get_data_sync(ticker, start_date, end_date, interval):
    return asyncio.run(get_data(ticker, start_date, end_date, interval))
