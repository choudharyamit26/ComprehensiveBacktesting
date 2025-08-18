"""
Position management system for tracking and managing trading positions.
"""

import asyncio
import pickle
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
from .config import IST, SIMULATION_MODE
from .telegram_client import send_telegram_alert

logger = logging.getLogger("quant_trader")
trade_logger = logging.getLogger("trade_execution")


class PositionManager:
    def __init__(self):
        self.open_positions = {}  # order_id -> position_data
        self.positions_by_security = {}  # security_id -> order_id
        self.strategy_instances = {}  # order_id -> strategy_instance
        self.max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", 10))
        self.position_lock = asyncio.Lock()
        self.last_trade_times = {}
        self.last_trade_lock = asyncio.Lock()
        self.cooldown_minutes = int(os.getenv("COOLDOWN_MINUTES", 30))

    async def get_last_trade_time(self, ticker: str) -> Optional[datetime]:
        """Get last trade time for a ticker with thread safety"""
        async with self.last_trade_lock:
            return self.last_trade_times.get(ticker)

    async def update_last_trade_time(self, ticker: str, trade_time: datetime):
        """Update last trade time for a ticker with thread safety"""
        async with self.last_trade_lock:
            self.last_trade_times[ticker] = trade_time
            logger.debug(f"Updated last trade time for {ticker}: {trade_time}")
            await self.save_trade_times()

    async def save_trade_times(self):
        """Persist trade times to disk for restart resilience"""
        try:
            with open("last_trades.pkl", "wb") as f:
                pickle.dump(self.last_trade_times, f)
        except Exception as e:
            logger.error(f"Failed to save trade times: {e}")

    async def load_trade_times(self):
        """Load trade times from disk on startup"""
        try:
            if os.path.exists("last_trades.pkl"):
                with open("last_trades.pkl", "rb") as f:
                    self.last_trade_times = pickle.load(f)
                    logger.info(
                        f"Loaded {len(self.last_trade_times)} trade times from disk"
                    )
        except Exception as e:
            logger.error(f"Failed to load trade times: {e}")

    async def has_position(self, security_id: int) -> bool:
        """Check if we have an open position for this security"""
        async with self.position_lock:
            return security_id in self.positions_by_security

    async def get_position_direction(self, security_id: int) -> Optional[str]:
        """Get the direction of an open position"""
        async with self.position_lock:
            order_id = self.positions_by_security.get(security_id)
            if order_id and order_id in self.open_positions:
                return self.open_positions[order_id]["direction"]
            return None

    async def add_position(
        self,
        order_id: str,
        security_id: int,
        ticker: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        take_profit: float,
        direction: str,
        strategy_name: str,
        strategy_instance=None,
    ):
        """Add a new position to the manager"""
        async with self.position_lock:
            # Check for existing position
            if await self.has_position(security_id):
                existing_direction = await self.get_position_direction(security_id)
                logger.warning(
                    f"Position already exists for {ticker} (security_id: {security_id}, direction: {existing_direction})"
                )
                return False

            # Enforce max positions limit
            if len(self.open_positions) >= self.max_open_positions:
                logger.warning(
                    f"Max open positions reached ({self.max_open_positions})"
                )
                return False

            # Create new position
            position_data = {
                "security_id": security_id,
                "ticker": ticker,
                "entry_price": entry_price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "direction": direction,  # BUY = LONG position, SELL = SHORT position
                "strategy_name": strategy_name,
                "entry_time": datetime.now(IST),
                "last_updated": datetime.now(IST),
            }

            self.open_positions[order_id] = position_data
            self.positions_by_security[security_id] = order_id

            # Store strategy instance for exit monitoring
            if strategy_instance:
                self.strategy_instances[order_id] = strategy_instance

            trade_logger.info(
                f"{'[SIM] ' if SIMULATION_MODE else ''}NEW POSITION | {ticker} | "
                f"{direction} | Qty: {quantity} | Entry: ₹{entry_price:.2f} | "
                f"SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f} | Strategy: {strategy_name}"
            )

            logger.info(
                f"Added position {order_id} for {ticker}: {direction} @ ₹{entry_price:.2f}"
            )
            return True

    async def update_position(self, order_id: str, **updates):
        """Update position parameters (e.g., trailing stop)"""
        async with self.position_lock:
            if order_id in self.open_positions:
                position = self.open_positions[order_id]
                old_values = {k: position.get(k) for k in updates.keys()}
                position.update(updates)
                position["last_updated"] = datetime.now(IST)

                # Log meaningful updates
                if "stop_loss" in updates:
                    logger.info(
                        f"Updated stop loss for {position['ticker']}: "
                        f"₹{old_values.get('stop_loss', 0):.2f} -> ₹{updates['stop_loss']:.2f}"
                    )

    async def close_position(
        self, order_id: str, exit_price: float = None, reason: str = "Manual"
    ):
        """Remove a position from the manager"""
        async with self.position_lock:
            if order_id not in self.open_positions:
                logger.warning(f"Attempted to close non-existent position: {order_id}")
                return False

            position = self.open_positions[order_id]
            security_id = position["security_id"]
            ticker = position["ticker"]
            direction = position["direction"]

            # Calculate P&L if exit price is provided
            pnl_msg = ""
            pnl = 0.0
            if exit_price:
                entry = position["entry_price"]
                qty = position["quantity"]

                # Correct P&L calculation based on position direction
                if direction == "BUY":  # LONG position
                    pnl = (exit_price - entry) * qty
                else:  # SHORT position
                    pnl = (entry - exit_price) * qty

                pnl_msg = f" | Exit: ₹{exit_price:.2f} | P&L: ₹{pnl:.2f}"

            # Enhanced trade logging
            hold_time = datetime.now(IST) - position["entry_time"]
            hold_minutes = int(hold_time.total_seconds() / 60)

            trade_logger.info(
                f"{'[SIM] ' if SIMULATION_MODE else ''}CLOSED POSITION | {ticker} | "
                f"{direction} | Qty: {position['quantity']} | "
                f"Entry: ₹{position['entry_price']:.2f}{pnl_msg} | "
                f"Hold: {hold_minutes}min | Reason: {reason}"
            )

            # Remove from indexes
            if security_id in self.positions_by_security:
                del self.positions_by_security[security_id]

            if order_id in self.strategy_instances:
                del self.strategy_instances[order_id]

            del self.open_positions[order_id]

            logger.info(f"Closed position {order_id} for {ticker} (P&L: ₹{pnl:.2f})")
            return True

    async def check_strategy_exit_signals(self, security_id: int, current_data):
        """Check for exit signals from strategy instances"""
        async with self.position_lock:
            order_id = self.positions_by_security.get(security_id)
            if not order_id or order_id not in self.open_positions:
                return None

            position = self.open_positions[order_id]
            strategy_instance = self.strategy_instances.get(order_id)

            if not strategy_instance:
                return None

            try:
                # Update strategy with new data
                strategy_instance.data = current_data

                # Check strategy-specific exit conditions
                if hasattr(strategy_instance, "should_exit"):
                    exit_signal = strategy_instance.should_exit()
                    if exit_signal:
                        return {
                            "action": "exit",
                            "reason": exit_signal.get("reason", "Strategy exit"),
                            "price": current_data.iloc[-1]["close"],
                        }

                # Check if strategy automatically closed position
                if hasattr(strategy_instance, "open_positions") and hasattr(
                    strategy_instance, "completed_trades"
                ):
                    if (
                        not strategy_instance.open_positions
                        and strategy_instance.completed_trades
                    ):
                        last_trade = strategy_instance.completed_trades[-1]
                        return {
                            "action": "exit",
                            "reason": "Strategy closed position",
                            "price": last_trade.get(
                                "exit_price", current_data.iloc[-1]["close"]
                            ),
                        }

            except Exception as e:
                logger.error(f"Exit signal check error for {position['ticker']}: {e}")

            return None

    async def execute_strategy_exit(self, order_id: str, exit_info: dict):
        """Execute exit based on strategy signal"""
        from .order_management import place_market_order

        async with self.position_lock:
            if order_id not in self.open_positions:
                logger.warning(f"Cannot exit non-existent position: {order_id}")
                return False

            position = self.open_positions[order_id]
            ticker = position["ticker"]

            # Correct exit direction logic
            # LONG position (BUY) requires SELL to exit
            # SHORT position (SELL) requires BUY to exit
            exit_direction = "SELL" if position["direction"] == "BUY" else "BUY"

            logger.info(
                f"Executing exit for {ticker}: "
                f"Position direction: {position['direction']}, Exit direction: {exit_direction}"
            )

            # Place market order for exit
            exit_order = await place_market_order(
                position["security_id"], exit_direction, position["quantity"]
            )

            if exit_order and exit_order.get("orderId"):
                # Calculate P&L
                current_price = exit_info["price"]
                entry_price = position["entry_price"]
                quantity = position["quantity"]

                # Correct P&L calculation based on position direction
                if position["direction"] == "BUY":  # LONG position
                    pnl = (current_price - entry_price) * quantity
                else:  # SHORT position
                    pnl = (entry_price - current_price) * quantity

                # Send exit notification
                await self.send_exit_notification(
                    position, exit_info, pnl, current_price
                )

                # Remove position
                await self.close_position(order_id, current_price, exit_info["reason"])
                return True
            else:
                logger.error(f"Exit order failed for {ticker}: {exit_order}")
                return False

    async def send_exit_notification(
        self, position: dict, exit_info: dict, pnl: float, exit_price: float
    ):
        """Send detailed exit notification"""
        pnl_emoji = "📈" if pnl > 0 else "📉"
        status_emoji = "✅" if pnl > 0 else "❌"
        hold_time = datetime.now(IST) - position["entry_time"]
        hold_minutes = int(hold_time.total_seconds() / 60)
        hold_seconds = int(hold_time.total_seconds() % 60)
        position_type = "Long" if position["direction"] == "BUY" else "Short"

        message = (
            f"*{position['ticker']} EXIT SIGNAL* {status_emoji}\n"
            f"Strategy: `{position['strategy_name']}`\n"
            f"Position: {position_type} (Qty: {position['quantity']})\n"
            f"Entry: ₹{position['entry_price']:.2f} → Exit: ₹{exit_price:.2f}\n"
            f"P&L: *₹{pnl:.2f}* {pnl_emoji}\n"
            f"Hold Time: {hold_minutes}m {hold_seconds}s\n"
            f"Reason: `{exit_info['reason']}`\n"
            f"Time: {datetime.now(IST).strftime('%H:%M:%S')}"
        )

        await send_telegram_alert(message)

    async def monitor_positions(self):
        """Enhanced position monitoring with proper exit logic"""
        from .data_manager import get_combined_data, fetch_realtime_quote
        from .calculations import calculate_regime

        logger.info("Starting position monitoring task")

        while True:
            try:
                if SIMULATION_MODE and not self.open_positions:
                    await asyncio.sleep(5)
                    continue

                async with self.position_lock:
                    if not self.open_positions:
                        await asyncio.sleep(30)
                        continue

                    # Process each position
                    positions_to_close = []
                    for order_id, position in list(self.open_positions.items()):
                        security_id = position["security_id"]
                        ticker = position["ticker"]
                        direction = position["direction"]

                        try:
                            # Get current market data
                            combined_data = await get_combined_data(security_id)
                            if combined_data is None or len(combined_data) < 10:
                                continue

                            # Get current quote
                            quotes = await fetch_realtime_quote([security_id])
                            quote = quotes.get(security_id)
                            if not quote:
                                continue
                            current_price = quote["price"]

                            # 1. Check strategy exit signals first
                            exit_signal = await self.check_strategy_exit_signals(
                                security_id, combined_data
                            )
                            if exit_signal:
                                logger.info(
                                    f"{ticker} strategy exit signal: {exit_signal['reason']}"
                                )
                                positions_to_close.append((order_id, exit_signal))
                                continue

                            # 2. Check stop-loss/take-profit
                            exit_triggered = None

                            if direction == "BUY":  # LONG position
                                if current_price <= position["stop_loss"]:
                                    exit_triggered = {
                                        "reason": "Stop-loss hit",
                                        "price": current_price,
                                    }
                                elif current_price >= position["take_profit"]:
                                    exit_triggered = {
                                        "reason": "Take-profit hit",
                                        "price": current_price,
                                    }
                            else:  # SHORT position (direction == "SELL")
                                if current_price >= position["stop_loss"]:
                                    exit_triggered = {
                                        "reason": "Stop-loss hit",
                                        "price": current_price,
                                    }
                                elif current_price <= position["take_profit"]:
                                    exit_triggered = {
                                        "reason": "Take-profit hit",
                                        "price": current_price,
                                    }

                            if exit_triggered:
                                logger.info(
                                    f"{ticker} SL/TP triggered: {exit_triggered['reason']}"
                                )
                                positions_to_close.append((order_id, exit_triggered))
                                continue

                            # 3. Update trailing stops every 5 minutes
                            now = datetime.now(IST)
                            if (now - position["last_updated"]).total_seconds() > 300:
                                regime, adx_value, atr_value = calculate_regime(
                                    combined_data
                                )
                                if atr_value > 0:
                                    if direction == "BUY":  # LONG position
                                        new_sl = max(
                                            position["stop_loss"],
                                            current_price - atr_value * 1.5,
                                        )
                                        if new_sl > position["stop_loss"]:
                                            await self.update_position(
                                                order_id, stop_loss=new_sl
                                            )
                                    else:  # SHORT position
                                        new_sl = min(
                                            position["stop_loss"],
                                            current_price + atr_value * 1.5,
                                        )
                                        if new_sl < position["stop_loss"]:
                                            await self.update_position(
                                                order_id, stop_loss=new_sl
                                            )

                        except Exception as e:
                            logger.error(f"Error monitoring position {ticker}: {e}")

                # Execute exits outside the position lock to avoid deadlocks
                for order_id, exit_info in positions_to_close:
                    try:
                        await self.execute_strategy_exit(order_id, exit_info)
                    except Exception as e:
                        logger.error(f"Error executing exit for {order_id}: {e}")

                # Shorter sleep in simulation
                await asyncio.sleep(5 if SIMULATION_MODE else 30)

            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(5 if SIMULATION_MODE else 60)


# Global position manager instance
position_manager = PositionManager()
