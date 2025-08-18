"""
Order management system for placing and managing trades.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import aiohttp

from .config import SIMULATION_MODE, DHAN_CLIENT_ID, IST
from .rate_limiter import OptimizedRateLimiter
from .config import QUOTE_API_RATE_LIMIT

logger = logging.getLogger("quant_trader")
trade_logger = logging.getLogger("trade_execution")

# Rate limiter for API calls
rate_limiter = OptimizedRateLimiter(rate_limit=QUOTE_API_RATE_LIMIT)


class APIClient:
    def __init__(self):
        self.connector = None
        self.timeout = aiohttp.ClientTimeout(total=10, connect=5)
        self.session = None

    async def get_session(self):
        from .config import DHAN_ACCESS_TOKEN

        if self.session is None or self.session.closed:
            if self.connector is None:
                self.connector = aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=20,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True,
                )
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout,
                headers={"access-token": DHAN_ACCESS_TOKEN},
            )
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
        if self.connector and not self.connector.closed:
            await self.connector.close()


# Global API client instance
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
    quantity: int,
) -> Optional[Dict]:
    """Enhanced order placement with proper logging"""
    if SIMULATION_MODE:
        # Simulate successful order placement
        order_id = f"SIM-{security_id}-{int(datetime.now(IST).timestamp())}"
        trade_logger.info(
            f"[SIM] SUPER ORDER | {security_id} | {transaction_type} | "
            f"Qty: {quantity} | Price: ₹{current_price:.2f} | "
            f"SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f}"
        )
        return {"orderId": order_id, "status": "success"}

    try:
        await rate_limiter.acquire()
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
            "trailingJump": 0.1,
        }

        session = await api_client.get_session()
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                order_id = data.get("orderId")
                if order_id:
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
