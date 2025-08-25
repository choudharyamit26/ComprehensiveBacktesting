import asyncio
from datetime import datetime
from typing import Dict, Optional
import aiohttp
from intraday.logging_setup import setup_logging
import intraday.positions as positions_store

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from intraday.constants import (
    DHAN_CLIENT_ID,
    IST,
    MAX_QUANTITY,
    QUOTE_API_RATE_LIMIT,
    SIMULATION_MODE,
)
from intraday.realtime_quotes import OptimizedRateLimiter
from intraday.telegram_alerts import send_telegram_alert
from intraday.utils import APIClient

rate_limiter = OptimizedRateLimiter(rate_limit=QUOTE_API_RATE_LIMIT)

logger, trade_logger = setup_logging()
api_client = APIClient()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (asyncio.TimeoutError, aiohttp.ClientConnectionError)
    ),
)
async def place_super_order(
    security_id: int,
    transaction_type: str,
    current_price: float,
    stop_loss: float,
    take_profit: float,
    quantity: int = MAX_QUANTITY,
) -> Optional[Dict]:
    """Enhanced order placement with proper logging and SQLite storage"""

    if SIMULATION_MODE:
        # Simulate successful order placement
        order_id = f"SIM-{security_id}-{int(datetime.now(IST).timestamp())}"

        # Create simulated request and response
        simulated_request = {
            "dhanClientId": DHAN_CLIENT_ID,
            "correlationId": f"{security_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "transactionType": transaction_type,
            "exchangeSegment": "NSE_EQ",
            "productType": "INTRADAY",
            "orderType": "LIMIT",
            "securityId": str(security_id),
            "quantity": quantity,
            "price": round(current_price, 2),
            "targetPrice": round(take_profit, 2),
            "stopLossPrice": round(stop_loss, 2),
            "trailingJump": 0.1,
        }

        simulated_response = {"orderId": order_id, "orderStatus": "PENDING"}

        # Save to database
        await positions_store.save_super_order_to_db(
            simulated_request,
            simulated_response,
            security_id,
            transaction_type,
            current_price,
            stop_loss,
            take_profit,
            quantity,
        )

        trade_logger.info(
            f"[SIM] SUPER ORDER | {security_id} | {transaction_type} | "
            f"Qty: {quantity} | Price: ₹{current_price:.2f} | "
            f"SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f} | OrderID: {order_id}"
        )
        return simulated_response

    try:
        # await rate_limiter.acquire()
        url = "https://api.dhan.co/v2/super/orders"
        payload = {
            "dhanClientId": DHAN_CLIENT_ID,
            "correlationId": f"{security_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "transactionType": transaction_type,
            "exchangeSegment": "NSE_EQ",
            "productType": "INTRADAY",
            "orderType": "LIMIT",
            "securityId": str(security_id),
            "quantity": quantity,
            "price": round(current_price, 2),
            "targetPrice": round(take_profit, 2),
            "stopLossPrice": round(stop_loss, 2),
            "trailingJump": 1 if current_price > 1000 else 0.1,
        }

        session = await api_client.get_session()
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                order_id = data.get("orderId")
                await send_telegram_alert(
                    f"*{security_id} Order Placed\n"
                    f"Signal: {transaction_type} at ₹{current_price:.2f}\n"
                    f"Reposne: {response}"
                )
                if order_id:
                    # Save to database
                    await positions_store.save_super_order_to_db(
                        payload,
                        data,
                        security_id,
                        transaction_type,
                        current_price,
                        stop_loss,
                        take_profit,
                        quantity,
                    )

                    trade_logger.info(
                        f"SUPER ORDER PLACED | {security_id} | {transaction_type} | "
                        f"Qty: {quantity} | Price: ₹{current_price:.2f} | "
                        f"SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f} | OrderID: {order_id}"
                    )
                    return data
                else:
                    trade_logger.error(f"Super order failed - no order ID: {data}")
            else:
                text = await response.text()
                trade_logger.error(f"Super order HTTP error {response.status}: {text}")
        return None
    except Exception as e:
        trade_logger.error(f"Super order exception: {str(e)}")
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(
        (asyncio.TimeoutError, aiohttp.ClientConnectionError)
    ),
)
async def place_market_order(
    security_id: int, transaction_type: str, quantity: int
) -> Optional[Dict]:
    """Enhanced market order placement for exits"""
    if SIMULATION_MODE:
        order_id = f"SIM-EXIT-{security_id}-{int(datetime.now(IST).timestamp())}"
        trade_logger.info(
            f"[SIM] MARKET ORDER | {security_id} | {transaction_type} | Qty: {quantity}"
        )
        return {"orderId": order_id, "status": "success"}

    try:
        await rate_limiter.acquire()
        url = "https://api.dhan.co/v2/orders"
        payload = {
            "dhanClientId": DHAN_CLIENT_ID,
            "exchangeSegment": "NSE_EQ",
            "securityId": str(security_id),
            "transactionType": transaction_type,
            "orderType": "MARKET",
            "productType": "INTRADAY",
            "quantity": quantity,
        }

        session = await api_client.get_session()
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                order_id = data.get("orderId")
                if order_id:
                    trade_logger.info(
                        f"MARKET ORDER PLACED | {security_id} | {transaction_type} | "
                        f"Qty: {quantity} | OrderID: {order_id}"
                    )
                    return data
                else:
                    trade_logger.error(f"Market order failed - no order ID: {data}")
            else:
                text = await response.text()
                trade_logger.error(f"Market order HTTP error {response.status}: {text}")
        return None
    except Exception as e:
        trade_logger.error(f"Market order exception: {str(e)}")
        return None
