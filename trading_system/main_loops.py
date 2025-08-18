"""
Main trading loops for live and simulation modes.
"""

import asyncio
import logging
import pandas as pd
import traceback
from datetime import datetime, timedelta

from .config import SIMULATION_MODE, IST
from .logging_setup import setup_logging
from .telegram_client import telegram_queue, send_telegram_alert
from .position_manager import position_manager
from .signal_execution import process_stock_with_exit_monitoring
from .market_utils import market_hours_check
from .data_manager import cache_warmup_for_trading
from .background_tasks import (
    schedule_square_off,
    send_enhanced_heartbeat,
    cache_maintenance,
    cleanup_system,
)
from live_data import initialize_live_data_from_config

logger, trade_logger = setup_logging()


async def main_trading_loop_with_cache():
    """Enhanced main trading loop with intelligent caching."""
    background_tasks = []  # Initialize early to avoid UnboundLocalError

    try:
        await telegram_queue.start()
        await send_telegram_alert("🚀 Bot started with enhanced caching")

        # Initialize live data system
        await initialize_live_data_from_config()

        # Pre-market cache warmup - Fixed timezone handling
        now = datetime.now(IST)
        # Use replace() instead of localize() for timezone objects
        from .config import MARKET_OPEN_TIME

        market_open = datetime.combine(now.date(), MARKET_OPEN_TIME).replace(tzinfo=IST)

        if now < market_open:
            time_to_market = (market_open - now).total_seconds()
            if time_to_market > 1800:  # More than 30 minutes before market
                logger.info("🔥 Starting pre-market cache warmup")
                cached_count, total_count = await cache_warmup_for_trading()
                await send_telegram_alert(
                    f"📊 Cache Warmup Complete\n"
                    f"Cached: {cached_count}/{total_count} securities\n"
                    f"Ready for market open!"
                )

        try:
            strategies_df = pd.read_csv("csv/selected_stocks_strategies.csv")
            nifty500 = pd.read_csv("csv/ind_nifty500list.csv")
        except Exception as e:
            logger.critical(f"Data load failed: {str(e)}")
            await send_telegram_alert(f"❌ Data load failed: {str(e)}")
            return

        # Prepare stock universe (existing logic)
        stock_universe = []
        ticker_to_security = nifty500.set_index("ticker")["security_id"].to_dict()

        for ticker in strategies_df["Ticker"].unique():
            if ticker in ticker_to_security:
                stock_data = strategies_df[strategies_df["Ticker"] == ticker]
                stock_universe.append(
                    {
                        "ticker": ticker,
                        "security_id": ticker_to_security[ticker],
                        "strategies": stock_data.to_dict("records"),
                    }
                )

        logger.info(f"Prepared {len(stock_universe)} stocks for trading")

        # Enhanced background tasks with cache maintenance
        background_tasks = [
            asyncio.create_task(schedule_square_off()),
            asyncio.create_task(send_enhanced_heartbeat()),  # Enhanced version
            asyncio.create_task(cache_maintenance()),  # New task
            asyncio.create_task(position_manager.monitor_positions()),
            asyncio.create_task(position_manager.load_trade_times()),
        ]

        logger.info(f"Started {len(background_tasks)} background tasks")

        batch_size = 3
        loop_count = 0
        INDIVIDUAL_TASK_TIMEOUT = 45  # Seconds per stock
        BATCH_TIMEOUT = 120  # Seconds per batch (up from 50)

        while await market_hours_check():
            loop_count += 1
            start_time = datetime.now(IST)

            logger.debug(f"Starting trading loop iteration {loop_count}")

            for i in range(0, len(stock_universe), batch_size):
                batch = stock_universe[i : i + batch_size]
                batch_tasks = []

                # Create tasks with individual timeouts
                for s in batch:
                    task = asyncio.create_task(
                        asyncio.wait_for(
                            process_stock_with_exit_monitoring(
                                s["ticker"], s["security_id"], s["strategies"]
                            ),
                            timeout=INDIVIDUAL_TASK_TIMEOUT,
                        )
                    )
                    batch_tasks.append(task)

                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=BATCH_TIMEOUT,
                    )

                    # Enhanced timeout handling
                    for j, result in enumerate(results):
                        ticker = batch[j]["ticker"]
                        if isinstance(result, asyncio.TimeoutError):
                            logger.warning(f"Timeout processing {ticker}")
                        elif isinstance(result, Exception):
                            logger.error(f"Error processing {ticker}: {result}")

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Batch timeout - cancelling {len(batch_tasks)} tasks"
                    )
                    for task in batch_tasks:
                        if not task.done():
                            task.cancel()
                    # Log specific tickers causing delay
                    for s in batch:
                        logger.debug(f"Pending: {s['ticker']}")
                except Exception as e:
                    logger.error(f"Error in batch processing: {str(e)}")

                # Small delay between batches
                await asyncio.sleep(1)

            elapsed = (datetime.now(IST) - start_time).total_seconds()
            sleep_time = max(30 - elapsed, 5)

            if loop_count % 10 == 0:
                logger.info(
                    f"Completed loop {loop_count}, elapsed: {elapsed:.2f}s, sleeping: {sleep_time:.1f}s"
                )

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        logger.info("Market hours ended - exiting trading loop")

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt - shutting down gracefully")
        await send_telegram_alert("⏹️ Bot stopped by user")

    except Exception as e:
        logger.critical(f"Enhanced main loop failure: {str(e)}")
        logger.error(traceback.format_exc())
        await send_telegram_alert(f"*CRITICAL ERROR*\nTrading stopped: {str(e)}")

    finally:
        # Cancel background tasks
        if background_tasks:
            logger.info(f"Cancelling {len(background_tasks)} background tasks...")
            for task in background_tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to finish cancellation
            try:
                await asyncio.wait(background_tasks, timeout=10.0)
                logger.info("Background tasks cancelled")
            except asyncio.TimeoutError:
                logger.warning("Some background tasks didn't cancel in time")

        # Cleanup system resources
        await cleanup_system()

        # Final telegram notification
        try:
            await send_telegram_alert("🛑 Bot shutdown complete")
        except Exception as e:
            logger.error(f"Failed to send shutdown alert: {e}")

        logger.info("Cleanup complete")


async def main_simulation_loop():
    """Continuous simulation run using offline data files"""
    try:
        await telegram_queue.start()
        logger.info("Starting continuous simulation mode")

        # Initialize position manager
        await position_manager.load_trade_times()

        try:
            strategies_df = pd.read_csv("csv/selected_stocks_strategies.csv")
            nifty500 = pd.read_csv("csv/ind_nifty500list.csv")
        except Exception as e:
            logger.critical(f"Data load failed: {str(e)}")
            return

        stock_universe = []
        ticker_to_security = nifty500.set_index("ticker")["security_id"].to_dict()
        for ticker in strategies_df["Ticker"].unique():
            if ticker in ticker_to_security:
                stock_data = strategies_df[strategies_df["Ticker"] == ticker]
                stock_universe.append(
                    {
                        "ticker": ticker,
                        "security_id": int(ticker_to_security[ticker]),
                        "strategies": stock_data.to_dict("records"),
                    }
                )

        logger.info(f"Prepared {len(stock_universe)} stocks for simulation")

        # Start background tasks
        background_tasks = [
            asyncio.create_task(position_manager.monitor_positions()),
            asyncio.create_task(send_enhanced_heartbeat()),
        ]

        # Continuous processing loop
        batch_size = 5
        while True:
            start_time = datetime.now(IST)
            opened_positions = 0

            # Process stocks in batches
            for i in range(0, len(stock_universe), batch_size):
                batch = stock_universe[i : i + batch_size]
                tasks = [
                    asyncio.create_task(
                        process_stock_with_exit_monitoring(
                            s["ticker"], s["security_id"], s["strategies"]
                        )
                    )
                    for s in batch
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(0)  # Yield control

                # Count new positions
                for s in batch:
                    if await position_manager.has_position(s["security_id"]):
                        opened_positions += 1

            # Log progress
            logger.info(f"Processed batch. Positions opened: {opened_positions}")

            # Add position monitoring
            await position_manager.monitor_positions()

            # Throttle processing
            elapsed = (datetime.now(IST) - start_time).total_seconds()
            sleep_time = max(30 - elapsed, 5)
            await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        logger.info("Simulation cancelled by user")
    except Exception as e:
        logger.critical(f"Simulation failure: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Close all positions at end
        closed_positions = 0
        async with position_manager.position_lock:
            for order_id, position in list(position_manager.open_positions.items()):
                await position_manager.close_position(
                    order_id, exit_price=position["entry_price"]  # For logging
                )
                closed_positions += 1

        # Generate summary
        summary = (
            f"📊 SIMULATION COMPLETE\n"
            f"Stocks: {len(stock_universe)}\n"
            f"Positions Opened: {opened_positions}\n"
            f"Positions Closed: {closed_positions}"
        )
        logger.info(summary)
        await send_telegram_alert(summary)
