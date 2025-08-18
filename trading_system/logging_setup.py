"""
Logging configuration for the trading system.
"""

import os
import logging
from datetime import datetime
from .config import IST


class DateTimeFormatter(logging.Formatter):
    """Custom formatter with IST datetime"""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=IST)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " IST"


def setup_logging():
    """Improved logging setup with proper initialization checks"""

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    formatter = DateTimeFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    root_logger = logging.getLogger()

    # Clear existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler with proper error handling
    try:
        file_handler = logging.FileHandler(
            os.path.join("logs", "trading_system.log"), mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
    except PermissionError:
        print("Permission denied for logs directory, using current directory")
        file_handler = logging.FileHandler(
            "trading_system.log", mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
    except Exception as e:
        print(f"File handler creation failed: {e}")
        file_handler = None

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)

    # Create specialized loggers
    logger = logging.getLogger("quant_trader")
    logger.setLevel(logging.INFO)

    trade_logger = logging.getLogger("trade_execution")
    trade_logger.setLevel(logging.INFO)

    # Clear existing trade logger handlers
    for handler in trade_logger.handlers[:]:
        trade_logger.removeHandler(handler)

    # Trade-specific file handler
    try:
        trade_file_handler = logging.FileHandler(
            os.path.join("logs", "trades.log"), mode="a", encoding="utf-8"
        )
        trade_file_handler.setFormatter(formatter)
        trade_logger.addHandler(trade_file_handler)
        trade_logger.propagate = False
    except Exception as e:
        print(f"Trade log handler failed: {e}")
        trade_logger.propagate = True

    return logger, trade_logger
