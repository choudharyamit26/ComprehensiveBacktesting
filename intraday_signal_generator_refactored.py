#!/usr/bin/env python3
"""
Refactored Intraday Signal Generator
Main entry point for the trading system using modular architecture.
"""

import sys
import asyncio
import argparse
import logging

# Import configuration first
from trading_system.config import SIMULATION_MODE
from trading_system.logging_setup import setup_logging
from trading_system.main_loops import main_trading_loop_with_cache, main_simulation_loop

# Setup logging
logger, trade_logger = setup_logging()

# Validate environment variables for live trading
if not SIMULATION_MODE:
    from trading_system.config import (
        TELEGRAM_BOT_TOKEN,
        TELEGRAM_CHAT_ID,
        DHAN_ACCESS_TOKEN,
    )

    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_ACCESS_TOKEN]):
        logger.critical("Missing required environment variables for live trading")
        raise EnvironmentError("Required environment variables not set")
elif SIMULATION_MODE:
    logger.warning(
        "Simulation mode: skipping checks for TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DHAN_ACCESS_TOKEN"
    )


def main():
    """Main entry point for the trading system."""
    try:
        # Re-parse arguments for main execution
        parser = argparse.ArgumentParser(description="Intraday Signal Generator")
        parser.add_argument(
            "--mode",
            choices=["realtime", "simulate"],
            default="realtime",
            help="Run mode: realtime (default) or simulate (use combined_data)",
        )
        parser.add_argument(
            "--simulate", action="store_true", help="Shortcut for --mode simulate"
        )
        args = parser.parse_args()

        # Update SIMULATION_MODE based on final args
        simulation_mode = args.simulate or args.mode == "simulate"

        # Set event loop policy for Windows
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        if simulation_mode:
            logger.info("Starting SIMULATION MODE")
            asyncio.run(main_simulation_loop())
        else:
            logger.info("Starting LIVE TRADING MODE")
            asyncio.run(main_trading_loop_with_cache())

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.critical(f"System failure: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
