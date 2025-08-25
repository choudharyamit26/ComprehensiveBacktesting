"""
Logging utilities for the intraday system.

Provides:
- DateTimeFormatter that prints timestamps in IST.
- setup_logging which configures root logger, console and file handlers.
- get_loggers returning the named system and trade loggers.

Rationale:
Separating logging setup avoids duplication and ensures that any module that
needs logging can import consistent loggers. The setup method is idempotent and
clears old handlers to prevent duplicate log lines when reloading modules.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Tuple

from intraday.constants import IST


class DateTimeFormatter(logging.Formatter):
    """Custom formatter with IST datetime.

    This formatter ensures consistent timestamp formatting across both console
    and file outputs. The format includes milliseconds and an explicit "IST"
    suffix for clarity when comparing with exchange/local times.
    """

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=IST)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " IST"


def setup_logging() -> Tuple[logging.Logger, logging.Logger]:
    """Configure application loggers.

    Creates console and file handlers, using ./logs directory when possible.
    Returns two named loggers:
    - quant_trader: general system/runtime logs
    - trade_execution: order/trade specific logs

    This function is safe to call multiple times; existing handlers are removed
    to avoid duplicate logs in environments that reload modules.
    """
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

    # File handler for system logs
    file_handler = None
    try:
        file_handler = logging.FileHandler(
            os.path.join("logs", "trading_system.log"), mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
    except Exception:
        # Fallback to current directory if logs path is not writable
        file_handler = logging.FileHandler(
            "trading_system.log", mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)

    logger = logging.getLogger("quant_trader")
    logger.setLevel(logging.INFO)

    trade_logger = logging.getLogger("trade_execution")
    trade_logger.setLevel(logging.INFO)

    # Ensure trade logger writes to its own file, without propagating to root twice
    for handler in trade_logger.handlers[:]:
        trade_logger.removeHandler(handler)

    try:
        trade_file_handler = logging.FileHandler(
            os.path.join("logs", "trades.log"), mode="a", encoding="utf-8"
        )
        trade_file_handler.setFormatter(formatter)
        trade_logger.addHandler(trade_file_handler)
        trade_logger.propagate = False
    except Exception:
        trade_logger.propagate = True

    return logger, trade_logger
