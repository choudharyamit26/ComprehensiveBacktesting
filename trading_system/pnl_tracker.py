"""
P&L tracking system for monitoring trading performance.
"""

import asyncio
import logging
from datetime import datetime
from .config import IST
from .rate_limiter import rate_limiter
from .data_manager import dhan

logger = logging.getLogger("quant_trader")


class PnLTracker:
    def __init__(self):
        self.cache = {
            "realized": 0.0,
            "unrealized": 0.0,
            "total": 0.0,
            "last_updated": None,
        }
        self.cache_ttl = 300  # Cache TTL in seconds (5 minutes)

    async def update_daily_pnl(self) -> dict:
        """Update daily P&L using dhan.get_positions()."""
        now = datetime.now(IST)
        if (
            self.cache["last_updated"]
            and (now - self.cache["last_updated"]).total_seconds() < self.cache_ttl
        ):
            logger.debug(f"Returning cached P&L: Total ₹{self.cache['total']:.2f}")
            return {
                "realized": self.cache["realized"],
                "unrealized": self.cache["unrealized"],
                "total": self.cache["total"],
            }

        # Retry logic: attempt up to 3 times with exponential backoff
        for attempt in range(3):
            try:
                await rate_limiter.acquire()
                # Run synchronous dhan.get_positions() in a separate thread
                response = await asyncio.to_thread(dhan.get_positions)

                if response.get("status") == "success" and response.get("data"):
                    realized_pnl = 0.0
                    unrealized_pnl = 0.0

                    # Process each position
                    for position in response["data"]:
                        try:
                            realized_pnl += float(position.get("realizedProfit", 0.0))
                            unrealized_pnl += float(
                                position.get("unrealizedProfit", 0.0)
                            )
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Error parsing P&L values for {position.get('tradingSymbol', 'Unknown')}: {e}"
                            )
                            continue

                    total_pnl = realized_pnl + unrealized_pnl

                    # Update cache
                    self.cache["realized"] = realized_pnl
                    self.cache["unrealized"] = unrealized_pnl
                    self.cache["total"] = total_pnl
                    self.cache["last_updated"] = now

                    logger.info(
                        f"Updated P&L - Realized: ₹{realized_pnl:.2f}, Unrealized: ₹{unrealized_pnl:.2f}, Total: ₹{total_pnl:.2f}"
                    )

                    return {
                        "realized": realized_pnl,
                        "unrealized": unrealized_pnl,
                        "total": total_pnl,
                    }

                elif response.get("status") == "success" and not response.get("data"):
                    # No positions found - this is valid
                    logger.info("No open positions found")
                    self.cache["realized"] = 0.0
                    self.cache["unrealized"] = 0.0
                    self.cache["total"] = 0.0
                    self.cache["last_updated"] = now

                    return {"realized": 0.0, "unrealized": 0.0, "total": 0.0}

                else:
                    logger.warning(
                        f"API call failed: {response.get('remarks', 'No remarks')}"
                    )
                    # Don't return 0 on first attempt, try again
                    if attempt == 2:  # Last attempt
                        return {
                            "realized": self.cache.get("realized", 0.0),
                            "unrealized": self.cache.get("unrealized", 0.0),
                            "total": self.cache.get("total", 0.0),
                        }

            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < 2:  # Not the last attempt
                    await asyncio.sleep(2**attempt)  # Exponential backoff: 1s, 2s, 4s
                else:
                    logger.error("Failed to update P&L after 3 attempts")
                    # Return cached values or zeros
                    return {
                        "realized": self.cache.get("realized", 0.0),
                        "unrealized": self.cache.get("unrealized", 0.0),
                        "total": self.cache.get("total", 0.0),
                    }

        # This shouldn't be reached, but just in case
        return {
            "realized": self.cache.get("realized", 0.0),
            "unrealized": self.cache.get("unrealized", 0.0),
            "total": self.cache.get("total", 0.0),
        }

    def get_current_pnl(self) -> dict:
        """Get current P&L without making API call."""
        return {
            "realized": self.cache.get("realized", 0.0),
            "unrealized": self.cache.get("unrealized", 0.0),
            "total": self.cache.get("total", 0.0),
            "last_updated": self.cache.get("last_updated"),
        }

    def get_position_summary(self, response_data: list) -> str:
        """Generate a summary of current positions."""
        if not response_data:
            return "No open positions"

        summary = []
        for pos in response_data:
            symbol = pos.get("tradingSymbol", "Unknown")
            position_type = pos.get("positionType", "Unknown")
            net_qty = pos.get("netQty", 0)
            unrealized = float(pos.get("unrealizedProfit", 0.0))

            summary.append(
                f"{symbol}: {position_type} {abs(net_qty)} (₹{unrealized:+.2f})"
            )

        return " | ".join(summary)


# Global P&L tracker instance
pnl_tracker = PnLTracker()
