"""
Position and order persistence layer using SQLite, with async-friendly wrappers.

Responsibilities:
- Ensure necessary tables and indexes exist (idempotent initialization).
- Track active positions separate from super_orders responses.
- Provide high level methods for add/update/close positions.
- Accessors to query orders and pending states.

Design notes:
- Uses synchronous sqlite3 but called from async code; keep DB operations small
  and fast. For heavy queries, consider running in executor.
- Avoid importing modules that depend on this one to prevent cycles.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import os
from comprehensive_backtesting.data import init_dhan_client
from intraday.constants import DB_PATH, IST, SIMULATION_MODE
from intraday.logging_setup import setup_logging
from intraday.realtime_quotes import fetch_realtime_quote
from intraday.telegram_alerts import send_telegram_alert

logger, trade_logger = setup_logging()

dhan = init_dhan_client()


class PositionManager:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.position_lock = asyncio.Lock()
        self.max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", 10))

        # Track daily traded securities to prevent multiple trades per day
        self.daily_traded_securities = set()
        self.last_reset_date = datetime.now().date()

        # Simulation-specific attributes
        self.simulated_pnl = 0.0
        self.simulated_trades = []

        # Initialize database tables if needed
        self._init_db()

    def _init_db(self):
        """Initialize position tracking tables in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create positions table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS active_positions (
                    order_id TEXT PRIMARY KEY,
                    security_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    original_quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    current_stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    strategy_name TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    breakeven_moved INTEGER DEFAULT 0,
                    partial_profit_taken INTEGER DEFAULT 0,
                    last_updated TEXT NOT NULL,
                    status TEXT DEFAULT 'ACTIVE'
                )
            """
            )
            # Main super orders table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS super_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE NOT NULL,
                    dhan_client_id TEXT,
                    correlation_id TEXT,
                    order_status TEXT,
                    transaction_type TEXT,
                    exchange_segment TEXT,
                    product_type TEXT,
                    order_type TEXT,
                    validity TEXT,
                    trading_symbol TEXT,
                    security_id TEXT,
                    quantity INTEGER,
                    remaining_quantity INTEGER,
                    ltp REAL,
                    price REAL,
                    after_market_order BOOLEAN,
                    leg_name TEXT,
                    exchange_order_id TEXT,
                    create_time TEXT,
                    update_time TEXT,
                    exchange_time TEXT,
                    oms_error_description TEXT,
                    average_traded_price REAL,
                    filled_qty INTEGER,
                    target_price REAL,
                    stop_loss_price REAL,
                    trailing_jump REAL,
                    request_payload TEXT,
                    response_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Order legs table for target and stop loss details
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS order_legs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_order_id TEXT,
                    order_id TEXT,
                    leg_name TEXT,
                    transaction_type TEXT,
                    total_quantity INTEGER,
                    remaining_quantity INTEGER,
                    triggered_quantity INTEGER,
                    price REAL,
                    order_status TEXT,
                    trailing_jump REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_order_id) REFERENCES super_orders (order_id)
                )
            """
            )

            # Create indexes for better performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_order_id ON super_orders (order_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_security_id ON super_orders (security_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_order_status ON super_orders (order_status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_parent_order ON order_legs (parent_order_id)"
            )

            conn.commit()
            conn.close()
            logger.info("Database initialized")

        except Exception as e:
            logger.error(f"Failed to initialize position database: {e}")

    async def get_order_id_by_security_id(self, security_id: int) -> str:
        """Fetch order_id using security_id from super_orders table"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT order_id FROM super_orders WHERE security_id = ? ORDER BY created_at DESC LIMIT 1",
                (str(security_id),),
            )
            result = cursor.fetchone()
            conn.close()
            if result:
                return result[0]
            return None
        except Exception as e:
            trade_logger.error(
                f"Failed to fetch order_id for security_id {security_id}: {str(e)}"
            )
            return None

    async def get_order_from_db(self, order_id: str) -> Optional[Dict]:
        """Fetch order details from super_orders table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT order_id, security_id, quantity, price, target_price, 
                       stop_loss_price, transaction_type, order_status
                FROM super_orders 
                WHERE order_id = ?
            """,
                (order_id,),
            )

            result = cursor.fetchone()
            conn.close()

            if result:
                return {
                    "order_id": result[0],
                    "security_id": int(result[1]),
                    "quantity": result[2],
                    "entry_price": result[3],
                    "take_profit": result[4],
                    "stop_loss": result[5],
                    "direction": result[6],
                    "status": result[7],
                }
            return None

        except Exception as e:
            logger.error(f"Error fetching order from DB: {e}")
            return None

    async def add_position(
        self, order_id: str, ticker: str, strategy_name: str, **kwargs
    ) -> bool:
        """Add position to tracking using order details from database"""
        async with self.position_lock:
            try:
                # Get order details from database
                order_data = await self.get_order_from_db(order_id)
                if not order_data:
                    logger.error(f"Order {order_id} not found in database")
                    return False

                # Add to active positions table
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO active_positions (
                        order_id, security_id, ticker, direction, quantity, 
                        original_quantity, entry_price, current_stop_loss, 
                        take_profit, strategy_name, entry_time, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        order_id,
                        order_data["security_id"],
                        ticker,
                        order_data["direction"],
                        order_data["quantity"],
                        order_data["quantity"],
                        order_data["entry_price"],
                        order_data["stop_loss"],
                        order_data["take_profit"],
                        strategy_name,
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                    ),
                )

                conn.commit()
                conn.close()

                # Mark as traded today
                # await self.mark_as_traded_today(order_data["security_id"])

                trade_logger.info(
                    f"NEW POSITION | {ticker} | {order_data['direction']} | "
                    f"Qty: {order_data['quantity']} | Entry: ₹{order_data['entry_price']:.2f} | "
                    f"SL: ₹{order_data['stop_loss']:.2f} | TP: ₹{order_data['take_profit']:.2f}"
                )

                return True

            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(
                    f"Error adding position: {e}. Tracebak:{str(traceback.print_exc())}"
                )
                return False

    async def get_active_positions(self) -> List[Dict]:
        """Get all active positions from database"""
        try:
            results = dhan.get_positions()
            positions = []
            for pos in results:
                # If response is dict-like and has 'positionType', filter out CLOSED
                if isinstance(pos, dict):
                    if pos.get("positionType") != "CLOSED":
                        positions.append(pos)
                # If response is row/tuple, fallback to old logic (for backward compatibility)
                elif isinstance(pos, (list, tuple)):
                    # If positionType is present and not CLOSED, add
                    position_type = None
                    if len(pos) > 3 and isinstance(pos[3], str):
                        position_type = pos[3]
                    if position_type != "CLOSED":
                        positions.append(
                            {
                                "order_id": self.get_order_id_by_security_id(pos[1]),
                                "security_id": pos[1],
                                "ticker": pos[2],
                                "direction": pos[3],
                                "quantity": pos[4],
                                "original_quantity": pos[5],
                                "entry_price": pos[6],
                                "current_stop_loss": pos[7],
                                "take_profit": pos[8],
                                "strategy_name": pos[9],
                                "entry_time": datetime.fromisoformat(pos[10]),
                                "breakeven_moved": bool(pos[11]),
                                "partial_profit_taken": bool(pos[12]),
                                "last_updated": datetime.fromisoformat(pos[13]),
                            }
                        )
            return positions
        except Exception as e:
            logger.error(f"Error getting active positions: {e}")
            return []

    async def calculate_profit_percentage(
        self, position: Dict, current_price: float
    ) -> float:
        """Calculate profit percentage for a position"""
        logger.info(f"Calculating profit percentage for {position,current_price}")
        entry_price = position["entry_price"]
        if position["direction"] == "BUY":
            return ((current_price - entry_price) / entry_price) * 100
        else:  # SHORT position
            return ((entry_price - current_price) / entry_price) * 100

    async def update_position_to_breakeven(self, order_id: str, position: Dict) -> bool:
        """Update stop loss to breakeven (entry price)"""
        try:
            logger.info(f"Breakeven triggered for {position['ticker']} at {position}%")

            new_stop_loss = position["entry_price"]

            # Update in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE active_positions 
                SET current_stop_loss = ?, breakeven_moved = 1, last_updated = ?
                WHERE order_id = ?
            """,
                (new_stop_loss, datetime.now().isoformat(), order_id),
            )

            conn.commit()
            conn.close()

            # Update actual order via Dhan API (if not simulation)
            if not SIMULATION_MODE:
                try:
                    modify_response = dhan.modify_order(
                        order_id=order_id,
                        order_type="STOP_LOSS",
                        price=new_stop_loss,
                        quantity=position["quantity"],
                    )
                    logger.info(
                        f"Modified order {order_id} stop loss to breakeven: ₹{new_stop_loss:.2f}. Response:{modify_response}"
                    )
                except Exception as e:
                    logger.error(f"Failed to modify order {order_id}: {e}")
                    return False

            # Send Telegram notification
            await send_telegram_alert(
                f"*{position['ticker']} BREAKEVEN MOVED* 🛡️\n"
                f"Stop Loss moved to Entry Price: ₹{new_stop_loss:.2f}\n"
                f"Position is now risk-free!\n"
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            )

            logger.info(
                f"{position['ticker']} moved to breakeven @ ₹{new_stop_loss:.2f}"
            )
            return True

        except Exception as e:
            logger.error(f"Error updating position to breakeven: {e}")
            return False

    async def take_partial_profit(
        self, order_id: str, position: Dict, current_price: float
    ) -> bool:
        """Sell half the quantity and lock in partial profit"""
        try:
            # Calculate half quantity (minimum 1 share)
            half_quantity = max(1, position["quantity"] // 2)
            remaining_quantity = position["quantity"] - half_quantity

            if remaining_quantity <= 0:
                logger.warning(
                    f"Cannot take partial profit for {position['ticker']}: insufficient quantity"
                )
                return False

            # Determine exit direction
            exit_direction = "SELL" if position["direction"] == "BUY" else "BUY"
            from intraday.orders import place_market_order

            # Place market order for partial exit
            exit_order = await place_market_order(
                position["security_id"], exit_direction, half_quantity
            )

            if exit_order and exit_order.get("orderId"):
                # Calculate partial profit
                entry_price = position["entry_price"]
                if position["direction"] == "BUY":
                    partial_pnl = (current_price - entry_price) * half_quantity
                else:
                    partial_pnl = (entry_price - current_price) * half_quantity

                # Update position in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE active_positions 
                    SET quantity = ?, partial_profit_taken = 1, last_updated = ?
                    WHERE order_id = ?
                """,
                    (remaining_quantity, datetime.now().isoformat(), order_id),
                )

                conn.commit()
                conn.close()

                # Track for simulation
                if SIMULATION_MODE:
                    self.simulated_pnl += partial_pnl
                    self.simulated_trades.append(
                        {
                            "ticker": position["ticker"],
                            "direction": exit_direction,
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "quantity": half_quantity,
                            "pnl": partial_pnl,
                            "reason": "Partial Profit",
                            "timestamp": datetime.now(),
                        }
                    )

                # Send Telegram notification
                await send_telegram_alert(
                    f"*{position['ticker']} PARTIAL PROFIT TAKEN* 💰\n"
                    f"Sold {half_quantity} shares @ ₹{current_price:.2f}\n"
                    f"Profit: ₹{partial_pnl:.2f}\n"
                    f"Remaining: {remaining_quantity} shares\n"
                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                )

                trade_logger.info(
                    f"PARTIAL PROFIT | {position['ticker']} | {exit_direction} | "
                    f"Qty: {half_quantity} | Price: ₹{current_price:.2f} | P&L: ₹{partial_pnl:.2f}"
                )

                logger.info(
                    f"{position['ticker']} partial profit: {half_quantity} shares @ ₹{current_price:.2f}, "
                    f"P&L: ₹{partial_pnl:.2f}, Remaining: {remaining_quantity}"
                )

                return True

            else:
                logger.error(f"Partial profit order failed for {position['ticker']}")
                return False

        except Exception as e:
            logger.error(f"Error taking partial profit: {e}")
            return False

    async def close_position(
        self, order_id: str, exit_price: float = None, reason: str = "Manual"
    ) -> bool:
        """Close position and remove from tracking"""
        try:
            # Get position details
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT ticker, direction, quantity, entry_price, strategy_name
                FROM active_positions WHERE order_id = ?
            """,
                (order_id,),
            )

            result = cursor.fetchone()
            if not result:
                logger.warning(f"Position {order_id} not found")
                return False

            ticker, direction, quantity, entry_price, strategy_name = result

            # Mark as closed in database
            cursor.execute(
                """
                UPDATE active_positions 
                SET status = 'CLOSED', last_updated = ?
                WHERE order_id = ?
            """,
                (datetime.now().isoformat(), order_id),
            )

            conn.commit()
            conn.close()

            # Calculate P&L if exit price provided
            pnl = 0.0
            if exit_price:
                if direction == "BUY":
                    pnl = (exit_price - entry_price) * quantity
                else:
                    pnl = (entry_price - exit_price) * quantity

                if SIMULATION_MODE:
                    self.simulated_pnl += pnl
                    self.simulated_trades.append(
                        {
                            "ticker": ticker,
                            "direction": direction,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "quantity": quantity,
                            "pnl": pnl,
                            "reason": reason,
                            "timestamp": datetime.now(),
                        }
                    )

            trade_logger.info(
                f"CLOSED POSITION | {ticker} | {direction} | Qty: {quantity} | "
                f"Entry: ₹{entry_price:.2f} | Exit: ₹{exit_price:.2f} | "
                f"P&L: ₹{pnl:.2f} | Reason: {reason}"
            )

            logger.info(f"Closed position {order_id} for {ticker}, P&L: ₹{pnl:.2f}")
            return True

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    async def monitor_positions(self):
        """Main position monitoring loop with breakeven and partial profit logic"""
        logger.info("Starting position monitoring with profit management")

        while True:
            try:
                positions = await self.get_active_positions()
                if not positions:
                    await asyncio.sleep(30)
                    continue

                for position in positions:
                    print("FROM monitor position:", position)
                    logger.info(f"Monitoring position: {position}")
                    try:

                        current_price = fetch_realtime_quote(position["security_id"])[
                            position["security_id"]
                        ]["price"]

                        # Calculate profit percentage
                        profit_pct = await self.calculate_profit_percentage(
                            position, current_price
                        )

                        # 1. Move to breakeven at 0.5% profit
                        if (
                            profit_pct >= 0.5
                            and not position["breakeven_moved"]
                            and not position["partial_profit_taken"]
                        ):
                            await self.update_position_to_breakeven(
                                position["order_id"], position
                            )
                            await self.take_partial_profit(
                                position["order_id"], position, current_price
                            )
                            continue

                        # 3. Check stop-loss/take-profit triggers
                        exit_triggered = False
                        reason = ""

                        if position["direction"] == "BUY":
                            if current_price <= position["current_stop_loss"]:
                                exit_triggered = True
                                reason = "Stop-loss hit"
                            elif current_price >= position["take_profit"]:
                                exit_triggered = True
                                reason = "Take-profit hit"
                        else:  # SHORT
                            if current_price >= position["current_stop_loss"]:
                                exit_triggered = True
                                reason = "Stop-loss hit"
                            elif current_price <= position["take_profit"]:
                                exit_triggered = True
                                reason = "Take-profit hit"

                        if exit_triggered:
                            # Place exit order
                            from intraday.orders import place_market_order

                            exit_direction = (
                                "SELL" if position["direction"] == "BUY" else "BUY"
                            )

                            exit_order = await place_market_order(
                                position["security_id"],
                                exit_direction,
                                position["quantity"],
                            )

                            if exit_order:
                                await self.close_position(
                                    position["order_id"], current_price, reason
                                )

                    except Exception as e:
                        logger.error(
                            f"Error monitoring position {position['ticker']}: {e}"
                        )

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(60)

    async def get_simulation_report(self) -> Dict:
        """Generate simulation performance report"""
        if not self.simulated_trades:
            return {"message": "No trades recorded"}

        winning_trades = [t for t in self.simulated_trades if t["pnl"] > 0]
        losing_trades = [t for t in self.simulated_trades if t["pnl"] <= 0]

        return {
            "total_pnl": self.simulated_pnl,
            "total_trades": len(self.simulated_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (
                len(winning_trades) / len(self.simulated_trades)
                if self.simulated_trades
                else 0
            ),
            "average_pnl_per_trade": (
                self.simulated_pnl / len(self.simulated_trades)
                if self.simulated_trades
                else 0
            ),
            "largest_win": (
                max([t["pnl"] for t in winning_trades]) if winning_trades else 0
            ),
            "largest_loss": (
                min([t["pnl"] for t in losing_trades]) if losing_trades else 0
            ),
            "trades": self.simulated_trades,
        }


# Convenience functions for super_orders persistence used by order placement
async def save_super_order_to_db(
    request_payload: dict,
    response_data: dict,
    security_id: int,
    transaction_type: str,
    current_price: float,
    stop_loss: float,
    take_profit: float,
    quantity: int,
) -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        order_id = response_data.get("orderId", "")
        order_status = response_data.get("orderStatus", "UNKNOWN")
        cursor.execute(
            """
            INSERT INTO super_orders (
                order_id, dhan_client_id, correlation_id, order_status,
                transaction_type, exchange_segment, product_type, order_type,
                security_id, quantity, price, target_price, stop_loss_price,
                trailing_jump, request_payload, response_data, update_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                request_payload.get("dhanClientId", ""),
                request_payload.get("correlationId", ""),
                order_status,
                transaction_type,
                request_payload.get("exchangeSegment", ""),
                request_payload.get("productType", ""),
                request_payload.get("orderType", ""),
                str(security_id),
                quantity,
                current_price,
                take_profit,
                stop_loss,
                request_payload.get("trailingJump", 0),
                json.dumps(request_payload),
                json.dumps(response_data),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        trade_logger.info(f"Super order saved to DB | OrderID: {order_id}")
    except Exception as e:
        trade_logger.error(f"Failed to save super order to DB: {str(e)}")


async def update_super_order_from_list_response(order_data: dict) -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        order_id = order_data.get("orderId", "")
        cursor.execute(
            """
            UPDATE super_orders SET
                dhan_client_id = ?, correlation_id = ?, order_status = ?,
                transaction_type = ?, exchange_segment = ?, product_type = ?,
                order_type = ?, validity = ?, trading_symbol = ?, security_id = ?,
                quantity = ?, remaining_quantity = ?, ltp = ?, price = ?,
                after_market_order = ?, leg_name = ?, exchange_order_id = ?,
                create_time = ?, update_time = ?, exchange_time = ?,
                oms_error_description = ?, average_traded_price = ?, filled_qty = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE order_id = ?
            """,
            (
                order_data.get("dhanClientId", ""),
                order_data.get("correlationId", ""),
                order_data.get("orderStatus", ""),
                order_data.get("transactionType", ""),
                order_data.get("exchangeSegment", ""),
                order_data.get("productType", ""),
                order_data.get("orderType", ""),
                order_data.get("validity", ""),
                order_data.get("tradingSymbol", ""),
                order_data.get("securityId", ""),
                order_data.get("quantity", 0),
                order_data.get("remainingQuantity", 0),
                order_data.get("ltp", 0.0),
                order_data.get("price", 0.0),
                order_data.get("afterMarketOrder", False),
                order_data.get("legName", ""),
                order_data.get("exchangeOrderId", ""),
                order_data.get("createTime", ""),
                order_data.get("updateTime", ""),
                order_data.get("exchangeTime", ""),
                order_data.get("omsErrorDescription", ""),
                order_data.get("averageTradedPrice", 0.0),
                order_data.get("filledQty", 0),
                order_id,
            ),
        )
        cursor.execute("DELETE FROM order_legs WHERE parent_order_id = ?", (order_id,))
        for leg in order_data.get("legDetails", []):
            cursor.execute(
                """
                INSERT INTO order_legs (
                    parent_order_id, order_id, leg_name, transaction_type,
                    total_quantity, remaining_quantity, triggered_quantity,
                    price, order_status, trailing_jump
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    leg.get("orderId", ""),
                    leg.get("legName", ""),
                    leg.get("transactionType", ""),
                    leg.get("totalQuatity", 0),
                    leg.get("remainingQuantity", 0),
                    leg.get("triggeredQuantity", 0),
                    leg.get("price", 0.0),
                    leg.get("orderStatus", ""),
                    leg.get("trailingJump", 0.0),
                ),
            )
        conn.commit()
        conn.close()
        trade_logger.info(f"Super order updated in DB | OrderID: {order_id}")
    except Exception as e:
        trade_logger.error(f"Failed to update super order in DB: {str(e)}")
