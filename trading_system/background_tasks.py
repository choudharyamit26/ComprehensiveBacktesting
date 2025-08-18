"""
Background tasks for the trading system.
"""

import asyncio
import logging
from datetime import datetime, timedelta, time
from .config import IST, SQUARE_OFF_TIME
from .market_utils import market_hours_check, cleanup_market_times_cache
from .telegram_client import send_telegram_alert
from .position_manager import position_manager
from .order_management import place_market_order, api_client
from .cache_manager import cache_manager
from .data_manager import thread_pool
from historical_cache import get_cache_info, clear_cache_for_today

logger = logging.getLogger("quant_trader")


async def schedule_square_off():
    while True:
        try:
            now = datetime.now(IST)
            target_time = IST.localize(datetime.combine(now.date(), SQUARE_OFF_TIME))
            if now > target_time:
                target_time += timedelta(days=1)
            sleep_seconds = (target_time - now).total_seconds()
            if sleep_seconds > 0:
                await asyncio.sleep(min(sleep_seconds, 3600))
                if await market_hours_check():
                    async with position_manager.position_lock:
                        for order_id in list(position_manager.open_positions.keys()):
                            pos = position_manager.open_positions[order_id]
                            direction = "SELL" if pos["direction"] == "BUY" else "BUY"
                            await place_market_order(
                                pos["security_id"], direction, pos["quantity"]
                            )
                            await position_manager.close_position(
                                order_id, pos["entry_price"]
                            )
                            await send_telegram_alert(
                                f"*{pos['ticker']} SQUARED OFF* 🛑\nPrice: Market"
                            )
            else:
                await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Square off scheduler error: {e}")
            await asyncio.sleep(300)


async def send_enhanced_heartbeat():
    """Enhanced heartbeat with cache information."""
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour

            active_tasks = len([t for t in asyncio.all_tasks() if not t.done()])

            # Get cache sizes from both caching systems
            short_term_cache_sizes = {
                "depth": len(cache_manager.depth_cache),
                "historical": len(cache_manager.historical_cache),
                "volatility": len(cache_manager.volatility_cache),
                "volume": len(cache_manager.volume_cache),
            }

            # Get historical cache stats
            hist_cache_stats = get_cache_info()

            open_positions = len(position_manager.open_positions)

            message = (
                "💖 *SYSTEM HEARTBEAT*\n"
                f"Status: Operational\n"
                f"Active Tasks: {active_tasks}\n"
                f"Open Positions: {open_positions}\n"
                f"Short-term Cache: {short_term_cache_sizes}\n"
                f"Historical Cache: {hist_cache_stats['disk_cache_entries']} entries\n"
                f"Cached Securities: {hist_cache_stats['total_unique_securities']}\n"
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            )
            await send_telegram_alert(message)

        except Exception as e:
            logger.error(f"Enhanced heartbeat error: {e}")


async def cleanup_resources():
    while True:
        try:
            await asyncio.sleep(1800)
            cache_manager.log_cache_stats("depth")
            cache_manager.log_cache_stats("historical")
            cache_manager.log_cache_stats("volatility")
            cache_manager.log_cache_stats("volume")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def cache_maintenance():
    """
    Periodic cache maintenance to keep the system optimized.
    """
    while True:
        try:
            await asyncio.sleep(7200)  # Every 2 hours

            logger.info("🧹 Running cache maintenance")

            # Log cache statistics
            cache_manager.log_cache_stats("depth")
            cache_manager.log_cache_stats("historical")
            cache_manager.log_cache_stats("volatility")
            cache_manager.log_cache_stats("volume")

            # Get historical cache stats
            hist_stats = get_cache_info()
            logger.info(f"Historical cache stats: {hist_stats}")

            # Optional: Clear cache for current day if we're past market hours
            now = datetime.now(IST)
            market_close = time(15, 30)

            if now.time() > market_close:
                # Market is closed, we can clear today's intraday cache
                # But keep historical cache as it's still valid
                logger.info("Market closed - clearing intraday caches")

                # Clear short-term caches
                cache_manager.depth_cache.clear()
                # Keep historical cache as it contains valuable data
                clear_cache_for_today()  # Add this call

            # Clean up market times cache
            cleanup_market_times_cache()

        except Exception as e:
            logger.error(f"Cache maintenance error: {e}")


def shutdown_thread_pool_safely(pool, timeout=30):
    """Safely shutdown thread pool with timeout."""
    try:
        # Python 3.9+ has timeout parameter
        import sys

        if sys.version_info >= (3, 9):
            pool.shutdown(wait=True, timeout=timeout)
        else:
            # For older Python versions
            pool.shutdown(wait=True)
        logger.info("Thread pool shutdown successfully")
    except TypeError:
        # Fallback for versions without timeout parameter
        pool.shutdown(wait=True)
        logger.info("Thread pool shutdown (no timeout support)")
    except Exception as e:
        logger.error(f"Error shutting down thread pool: {e}")


async def cleanup_system():
    """Cleanup system resources on shutdown."""
    logger.info("Starting cleanup...")

    # Close API client
    try:
        await api_client.close()
        logger.info("API client closed")
    except Exception as e:
        logger.error(f"Error closing API client: {e}")

    # Shutdown thread pool
    try:
        shutdown_thread_pool_safely(thread_pool, timeout=30)
    except Exception as e:
        logger.error(f"Error shutting down thread pool: {e}")

    logger.info("Cleanup complete")
