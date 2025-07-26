import asyncio
import pandas as pd
import backtrader as bt
import numpy as np
import os
import logging
from datetime import date, datetime, timedelta, time
import pytz
from retrying import retry
import sys
from io import StringIO
import requests

from app import run_filter_backtest
from comprehensive_backtesting.data import get_security_id, init_dhan_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trade_logger")

# Telegram bot settings from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError(
        "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in environment variables"
    )

# Market settings
MARKET_OPEN = "09:15:00"  # NSE market open time (IST)
MARKET_CLOSE = "15:30:00"  # NSE market close time (IST)
TRADING_END = "15:05:00"  # Stop new trades after 3:05 PM IST
FORCE_CLOSE = "15:15:00"  # Force close positions at 3:15 PM IST
MIN_VOTES = 2  # Minimum number of strategies agreeing for a signal
WINDOW_SIZE = 6  # 6 bars of 5-minute data = 30 minutes
MORNING_WINDOW_SIZE = 3  # 3 bars = 15 minutes for morning session

# Placeholder for DhanHQ client (replace with actual initialization)
dhan = init_dhan_client()  # Initialize your DhanHQ client here


async def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
        }
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


async def aggregate_ticks_to_5min(ticker, security_id, data_store):
    """Aggregate tick data into 5-minute OHLCV bars and store in data_store."""
    try:
        tick_buffer = []
        current_bar_start = None
        ist_tz = pytz.timezone("Asia/Kolkata")

        while True:
            tick_data = await fetch_intraday_data(ticker)
            if tick_data is None or tick_data.empty:
                await asyncio.sleep(1)
                continue

            tick_time = tick_data["datetime"].iloc[0]
            price = tick_data["close"].iloc[0]
            volume = tick_data["volume"].iloc[0]

            bar_start = tick_time.replace(second=0, microsecond=0)
            bar_start -= timedelta(minutes=bar_start.minute % 5)

            if current_bar_start is None:
                current_bar_start = bar_start
                tick_buffer = [{"time": tick_time, "price": price, "volume": volume}]
            elif bar_start != current_bar_start:
                if tick_buffer:
                    df = pd.DataFrame(tick_buffer)
                    bar_data = {
                        "datetime": current_bar_start,
                        "open": df["price"].iloc[0],
                        "high": df["price"].max(),
                        "low": df["price"].min(),
                        "close": df["price"].iloc[-1],
                        "volume": df["volume"].sum(),
                        "security_id": security_id,
                        "symbol": ticker,
                    }
                    bar_df = pd.DataFrame([bar_data])
                    bar_df["datetime"] = pd.to_datetime(bar_df["datetime"])
                    # Append to data_store
                    if security_id in data_store:
                        data_store[security_id] = (
                            pd.concat([data_store[security_id], bar_df])
                            .sort_values("datetime")
                            .reset_index(drop=True)
                        )
                    else:
                        data_store[security_id] = bar_df
                    logger.info(
                        f"Aggregated 5-min bar for {ticker} at {current_bar_start}"
                    )
                    yield bar_df
                tick_buffer = [{"time": tick_time, "price": price, "volume": volume}]
                current_bar_start = bar_start
            else:
                tick_buffer.append(
                    {"time": tick_time, "price": price, "volume": volume}
                )

            await asyncio.sleep(1)  # Fetch every second
    except Exception as e:
        logger.error(f"Error aggregating ticks for {ticker}: {e}")
        yield pd.DataFrame()


import asyncio
import pytz
import pandas as pd
import backtrader as bt
from datetime import datetime, timedelta
import logging


async def process_stock(ticker, security_id, strategies, data_store):
    """Process real-time 5-minute bars for a stock, running multiple strategies with voting."""
    logger = logging.getLogger(__name__)
    ist_tz = pytz.timezone("Asia/Kolkata")

    # Fetch historical data for indicator stability
    end_date = datetime.now(ist_tz).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=2)
    historical_data = await fetch_historical_data(
        security_id, start_date, end_date, "5m"
    )

    if historical_data is not None and not historical_data.empty:
        if security_id not in data_store:
            data_store[security_id] = historical_data
        else:
            data_store[security_id] = (
                pd.concat([data_store[security_id], historical_data])
                .sort_values("datetime")
                .reset_index(drop=True)
            )

    # Initialize cerebros for each strategy
    cerebros = []
    for strat in strategies:
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strat["class"], **strat["params"].get(ticker, {}))
        min_bars = strat["class"].get_min_data_points(strat["params"].get(ticker, {}))
        cerebros.append(
            {"name": strat["name"], "cerebro": cerebro, "min_bars": min_bars}
        )

    async for new_bar in aggregate_ticks_to_5min(ticker, security_id, data_store):
        if new_bar.empty:
            logger.debug(f"Empty bar received for {ticker}. Skipping.")
            continue
        data_window = data_store.get(security_id, pd.DataFrame())
        if data_window.empty:
            logger.warning(f"No data available for {ticker}")
            continue

        current_time = datetime.now(ist_tz).time()
        window_size = (
            MORNING_WINDOW_SIZE
            if current_time < datetime.strptime("10:30:00", "%H:%M:%S").time()
            else WINDOW_SIZE
        )

        max_min_bars = max(c["min_bars"] for c in cerebros)
        if len(data_window) < max_min_bars:
            logger.debug(
                f"Insufficient data for {ticker}: {len(data_window)} bars, need {max_min_bars}"
            )
            continue

        # Limit data window to required size and current date
        data_window = data_window[-max(max_min_bars, window_size) :]
        data_window = data_window[
            data_window["datetime"].dt.date <= datetime.now(ist_tz).date()
        ]

        signals = []
        for cerebro_info in cerebros:
            cerebro = cerebro_info["cerebro"]
            data_feed = bt.feeds.PandasData(
                dataname=data_window,
                datetime="datetime",
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
            )
            cerebro.datas = cerebro.datas[:0]  # Clear previous data
            cerebro.adddata(data_feed)
            cerebro.run()

            strategy_instance = cerebro.strategies[0]
            if strategy_instance.order and strategy_instance.order_type in [
                "enter_long",
                "enter_short",
            ]:
                # Base signal data
                signal_data = {
                    "strategy": cerebro_info["name"],
                    "signal": (
                        "BUY"
                        if strategy_instance.order_type == "enter_long"
                        else "SELL"
                    ),
                    "price": strategy_instance.data.close[0],
                    "momentum": (
                        strategy_instance.indicator_data[-1].get(
                            "momentum_alignment", "N/A"
                        )
                        if strategy_instance.indicator_data
                        else "N/A"
                    ),
                }

                # Dynamically retrieve strategy indicators
                indicators = {}
                try:
                    # Check if strategy has a get_indicators method
                    if hasattr(strategy_instance, "get_indicators"):
                        indicators = strategy_instance.get_indicators()
                    else:
                        # Fallback: Inspect strategy lines or attributes
                        for line_name in strategy_instance.lines.getlinealiases():
                            if line_name not in [
                                "datetime",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ]:
                                try:
                                    value = getattr(strategy_instance, line_name)[0]
                                    indicators[line_name] = value
                                except (IndexError, AttributeError):
                                    continue
                except Exception as e:
                    logger.warning(
                        f"Failed to retrieve indicators for {cerebro_info['name']} on {ticker}: {str(e)}"
                    )

                # Add indicators to signal data
                for ind_name, ind_value in indicators.items():
                    try:
                        signal_data[ind_name] = (
                            float(ind_value)
                            if isinstance(ind_value, (int, float))
                            else ind_value
                        )
                    except (ValueError, TypeError):
                        signal_data[ind_name] = ind_value  # Keep as is if not numeric

                signals.append(signal_data)

        if signals:
            buy_count = sum(1 for s in signals if s["signal"] == "BUY")
            sell_count = sum(1 for s in signals if s["signal"] == "SELL")
            timestamp = datetime.now(ist_tz).strftime("%Y-%m-%d %H:%M:%S")

            if buy_count >= MIN_VOTES and buy_count > sell_count:
                agreeing_strategies = [
                    s["strategy"] for s in signals if s["signal"] == "BUY"
                ]
                message = (
                    f"*{ticker} Signal (Majority Vote)*\n"
                    f"Signal: BUY\n"
                    f"Price: ₹{signals[0]['price']:.2f}\n"
                    f"Time: {timestamp}\n"
                    f"Agreeing Strategies: {', '.join(agreeing_strategies)}\n"
                )
                # Add indicators for agreeing strategies
                for s in signals:
                    if s["signal"] == "BUY":
                        message += f"\nStrategy: {s['strategy']}\n"
                        for ind_name, ind_value in s.items():
                            if ind_name not in [
                                "strategy",
                                "signal",
                                "price",
                                "momentum",
                            ]:
                                message += f"{ind_name.replace('_', ' ').title()} (Last): {ind_value:.2f if isinstance(ind_value, (int, float)) else ind_value}\n"
                        if s["momentum"] != "N/A":
                            message += f"Momentum (Last): {s['momentum']}\n"
                await send_telegram_alert(message)
            elif sell_count >= MIN_VOTES and sell_count > buy_count:
                agreeing_strategies = [
                    s["strategy"] for s in signals if s["signal"] == "SELL"
                ]
                message = (
                    f"*{ticker} Signal (Majority Vote)*\n"
                    f"Signal: SELL\n"
                    f"Price: ₹{signals[0]['price']:.2f}\n"
                    f"Time: {timestamp}\n"
                    f"Agreeing Strategies: {', '.join(agreeing_strategies)}\n"
                )
                # Add indicators for agreeing strategies
                for s in signals:
                    if s["signal"] == "SELL":
                        message += f"\nStrategy: {s['strategy']}\n"
                        for ind_name, ind_value in s.items():
                            if ind_name not in [
                                "strategy",
                                "signal",
                                "price",
                                "momentum",
                            ]:
                                message += f"{ind_name.replace('_', ' ').title()} (Last): {ind_value:.2f if isinstance(ind_value, (int, float)) else ind_value}\n"
                        if s["momentum"] != "N/A":
                            message += f"Momentum (Last): {s['momentum']}\n"
                await send_telegram_alert(message)
            else:
                logger.debug(
                    f"No majority signal for {ticker}: {buy_count} BUY, {sell_count} SELL"
                )


async def main():
    """Main function to process stocks concurrently using pre-generated strategies."""
    try:
        # Set up logging
        logger = logging.getLogger(__name__)

        # Get current time in IST
        ist_tz = pytz.timezone("Asia/Kolkata")
        now = datetime(
            2025, 7, 26, 11, 25, 0, tzinfo=ist_tz
        )  # Set to 11:25 AM IST, July 26, 2025

        # Check if it's a weekday (Monday to Friday)
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            logger.info("Market is closed on weekends (Saturday/Sunday). Exiting.")
            return

        # Define market open and close times for today
        market_open = datetime.strptime(
            f"{now.date()} {MARKET_OPEN}", "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=ist_tz)
        market_close = datetime.strptime(
            f"{now.date()} {MARKET_CLOSE}", "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=ist_tz)

        # Check if market is open
        if now.time() < market_open.time():
            wait_seconds = (market_open - now).total_seconds()
            logger.info(
                f"Market not yet open. Waiting {wait_seconds:.0f} seconds until {MARKET_OPEN}."
            )
            await asyncio.sleep(wait_seconds)
        elif now.time() > market_close.time():
            logger.info("Market is closed. Exiting.")
            return

        # Load strategies from CSV
        try:
            top_strategies_df = pd.read_csv("selected_stocks_strategies.csv")
        except FileNotFoundError:
            logger.error("File selected_stocks_strategies.csv not found. Exiting.")
            return
        except pd.errors.EmptyDataError:
            logger.error("File selected_stocks_strategies.csv is empty. Exiting.")
            return

        # Check if DataFrame is empty
        if top_strategies_df.empty:
            logger.error(
                "No strategies found in selected_stocks_strategies.csv. Exiting."
            )
            return

        # Group strategies by ticker to match top_strategies_per_stock format
        top_strategies_per_stock = {}
        for ticker in top_strategies_df["Ticker"].unique():
            stock_df = top_strategies_df[top_strategies_df["Ticker"] == ticker]
            strategies = stock_df[
                ["Strategy", "Composite_Win_Rate", "Composite_Sharpe"]
            ].to_dict(orient="records")
            top_strategies_per_stock[ticker] = strategies

        # Load Nifty 500 list for security IDs
        try:
            df_csv = pd.read_csv("ind_nifty500list.csv")
        except FileNotFoundError:
            logger.error("File ind_nifty500list.csv not found. Exiting.")
            return
        except pd.errors.EmptyDataError:
            logger.error("File ind_nifty500list.csv is empty. Exiting.")
            return

        # Extract stocks and their strategies
        stock_strategies = []
        for ticker, strategies in top_strategies_per_stock.items():
            match = df_csv[df_csv["ticker"] == ticker]
            if match.empty:
                logger.warning(f"No security ID found for ticker {ticker}. Skipping.")
                continue
            security_id = match["security_id"].iloc[0]
            stock_strategies.append(
                {
                    "ticker": ticker,
                    "security_id": int(security_id),
                    "strategies": strategies,
                }
            )

        if not stock_strategies:
            logger.error("No valid stocks with security IDs found. Exiting.")
            return

        # Initialize data store
        data_store = {}  # Dictionary to store OHLCV data for each security_id

        # Process stocks concurrently
        tasks = [
            process_stock(s["ticker"], s["security_id"], s["strategies"], data_store)
            for s in stock_strategies
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any task exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Error processing stock {stock_strategies[i]['ticker']}: {str(result)}"
                )

        logger.info("All stocks processed successfully.")

    except Exception as e:
        logger.exception(f"Critical error in main function: {str(e)}")
        return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped real-time signal generation")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
