"""
Real-time and simulated quote retrieval.

This module encapsulates:
- Dhan WebSocket client integration for tick/quote data.
- REST fallback batch quote functions.
- Simulation-mode quote handling from combined_data files.

Key entrypoints:
- fetch_realtime_quote(security_ids: list[int]) -> dict[int, Optional[dict]]
- shutdown_websocket()
- initialize_websocket()

The design provides a single high-level function to obtain the most recent
price for a list of security IDs, preferring WebSocket updates for lowest
latency; if unavailable, it falls back to REST. In simulation mode, it
reads the latest close from prepared offline data.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

import aiohttp
import pandas as pd
import websocket

from intraday.constants import (
    COMBINED_DATA_DIR,
    DHAN_ACCESS_TOKEN,
    DHAN_CLIENT_ID,
    IST,
    QUOTE_API_RATE_LIMIT,
    SIMULATION_MODE,
)
from intraday.logging_setup import setup_logging

logger, trade_logger = setup_logging()

# Local caches for simulation
SIM_DATA_CACHE: Dict[int, pd.DataFrame] = {}

# WebSocket globals
_websocket_client = None
_latest_ticks: Dict[int, dict] = {}
_tick_lock = threading.Lock()


@dataclass
class TickDataWS:
    """Standardized tick data structure for WebSocket updates.

    Attributes:
        security_id: The Dhan security identifier (int).
        last_price: The latest traded price or best bid/ask derived value.
        volume: Volume field if available per tick (0 if unknown).
        timestamp: Datetime when tick was created at the exchange or received.
    """

    security_id: int
    last_price: float
    volume: int
    timestamp: datetime


class DhanWebSocketClient:
    """Minimal WebSocket client for Dhan feed v2 with callbacks.

    Life-cycle:
        connect() -> starts a background thread using websocket-client.
        disconnect() -> sends a disconnect request and closes the socket.

    Callbacks:
        on_tick_callback is invoked with TickDataWS for every parsed packet.
    """

    def __init__(
        self,
        access_token: str,
        client_id: str,
        on_tick_callback: Callable[[TickDataWS], None],
    ):
        self.access_token = access_token
        self.client_id = client_id
        self.on_tick_callback = on_tick_callback
        self.ws: Optional[websocket.WebSocketApp] = None
        self.subscribed_instruments = set()
        self.is_connected = False

    def connect(self) -> None:
        url = f"wss://api-feed.dhan.co?version=2&token={self.access_token}&clientId={self.client_id}&authType=2"
        self.ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def subscribe_instruments(self, instruments: List[Dict]) -> None:
        # Dhan allows max 100 instruments per request
        for i in range(0, len(instruments), 100):
            batch = instruments[i : i + 100]
            request = {
                "RequestCode": 15,
                "InstrumentCount": len(batch),
                "InstrumentList": batch,
            }
            if self.ws and self.ws.sock and self.ws.sock.connected:
                self.ws.send(json.dumps(request))
                for instr in batch:
                    self.subscribed_instruments.add(instr["SecurityId"])

    def _on_open(self, ws):
        logger.info("Dhan WebSocket connection established")
        self.is_connected = True

    def _on_message(self, ws, message: bytes):
        try:
            if len(message) < 8:
                return
            header = struct.unpack("<BI2xI", message[:8])
            feed_response_code = header[0]
            security_id = header[2]
            if feed_response_code == 2:
                self._handle_ticker_packet(message, security_id)
            elif feed_response_code == 4:
                self._handle_quote_packet(message, security_id)
            elif feed_response_code == 8:
                self._handle_full_packet(message, security_id)
            elif feed_response_code in (41, 51):
                self._handle_market_depth_packet(
                    message, security_id, feed_response_code
                )
        except Exception as e:
            logger.error(f"Error parsing Dhan message: {e}")

    def _handle_ticker_packet(self, message: bytes, security_id: int) -> None:
        if len(message) < 16:
            return
        ltp_raw, ltt = struct.unpack("<II", message[8:16])
        ltp = ltp_raw / 100.0
        tick = TickDataWS(
            security_id=int(security_id),
            last_price=ltp,
            volume=0,
            timestamp=datetime.fromtimestamp(ltt),
        )
        self.on_tick_callback(tick)

    def _handle_quote_packet(self, message: bytes, security_id: int) -> None:
        if len(message) < 50:
            return
        data = struct.unpack("<IIHIIIIIIIIII", message[8:50])
        ltp = data[0] / 100.0
        ltt = data[1]
        volume = data[7]
        tick = TickDataWS(
            security_id=int(security_id),
            last_price=ltp,
            volume=volume,
            timestamp=datetime.fromtimestamp(ltt),
        )
        self.on_tick_callback(tick)

    def _handle_full_packet(self, message: bytes, security_id: int) -> None:
        if len(message) < 162:
            return
        data = struct.unpack("<IIHIIIIIIIIII", message[8:50])
        ltp = data[0] / 100.0
        ltt = data[1]
        volume = data[7]
        tick = TickDataWS(
            security_id=int(security_id),
            last_price=ltp,
            volume=volume,
            timestamp=datetime.fromtimestamp(ltt),
        )
        self.on_tick_callback(tick)

    def _handle_market_depth_packet(
        self, message: bytes, security_id: int, response_code: int
    ) -> None:
        if len(message) < 332:
            return
        if response_code == 41:
            price = struct.unpack("<d", message[12:20])[0]
            tick = TickDataWS(
                security_id=int(security_id),
                last_price=price,
                volume=0,
                timestamp=datetime.now(),
            )
            self.on_tick_callback(tick)

    def _on_error(self, ws, error):
        logger.error(f"Dhan WebSocket error: {error}")
        self.is_connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("Dhan WebSocket connection closed")
        self.is_connected = False

    def disconnect(self) -> None:
        if self.ws:
            try:
                self.ws.send(json.dumps({"RequestCode": 12}))
            except Exception:
                pass
            try:
                self.ws.close()
            finally:
                self.is_connected = False


# --------------------------------------------------------------------------------------
# WebSocket lifecycle helpers
# --------------------------------------------------------------------------------------


def _on_tick_update(tick: TickDataWS) -> None:
    global _latest_ticks
    with _tick_lock:
        _latest_ticks[tick.security_id] = {
            "price": tick.last_price,
            "timestamp": tick.timestamp,
            "volume": tick.volume,
        }


async def get_websocket_client():
    global _websocket_client
    if _websocket_client is None:
        if not DHAN_ACCESS_TOKEN or not DHAN_CLIENT_ID:
            logger.error("DHAN_ACCESS_TOKEN or DHAN_CLIENT_ID not set")
            return None
        _websocket_client = DhanWebSocketClient(
            access_token=DHAN_ACCESS_TOKEN,
            client_id=DHAN_CLIENT_ID,
            on_tick_callback=_on_tick_update,
        )
        _websocket_client.connect()
        # wait for connection up to 10s
        waited = 0.0
        while not _websocket_client.is_connected and waited < 10.0:
            await asyncio.sleep(0.5)
            waited += 0.5
        if not _websocket_client.is_connected:
            logger.error("Failed to establish WebSocket connection")
            return None
    return _websocket_client


async def initialize_websocket():
    try:
        await get_websocket_client()
        logger.info("WebSocket connection initialized")
    except Exception as e:
        logger.error(f"Failed to initialize WebSocket: {e}")


async def shutdown_websocket():
    global _websocket_client
    if _websocket_client:
        _websocket_client.disconnect()
        _websocket_client = None
        logger.info("WebSocket connection shutdown")


# --------------------------------------------------------------------------------------
# REST fallback and unified quote API
# --------------------------------------------------------------------------------------
from comprehensive_backtesting.data import init_dhan_client

# Initialize Dhan client respecting simulation mode
if not SIMULATION_MODE:
    dhan = init_dhan_client()
else:

    class _MockDhan:
        def get_fund_limits(self):
            return {"data": {"availabelBalance": 1e9}}

        def get_positions(self):
            return {"status": "success", "data": []}

        def quote_data(self, payload):
            return {"status": "failure"}

    dhan = _MockDhan()


class OptimizedRateLimiter:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.last_refill = datetime.now(IST)
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = datetime.now(IST)
            elapsed = (now - self.last_refill).total_seconds()
            if elapsed >= 1.0:
                tokens_to_add = int(elapsed * (self.rate_limit / 60.0))
                self.tokens = min(self.rate_limit, self.tokens + tokens_to_add)
                self.last_refill = now
            while self.tokens < 1:
                await asyncio.sleep(0.1)
                now = datetime.now(IST)
                elapsed = (now - self.last_refill).total_seconds()
                if elapsed >= 1.0:
                    tokens_to_add = int(elapsed * (self.rate_limit / 60.0))
                    self.tokens = min(self.rate_limit, self.tokens + tokens_to_add)
                    self.last_refill = now
            self.tokens -= 1


rate_limiter = OptimizedRateLimiter(rate_limit=QUOTE_API_RATE_LIMIT)


async def _fetch_quote_batch(security_ids: List[int]) -> Dict[int, Optional[Dict]]:
    try:
        payload = {"NSE_EQ": [int(sid) for sid in security_ids]}
        response = await asyncio.to_thread(dhan.quote_data, payload)
        if response.get("status") == "success":
            result: Dict[int, Optional[Dict]] = {}
            for sec_id in security_ids:
                sec_id_str = str(sec_id)
                quote_data = (
                    response.get("data", {})
                    .get("data", {})
                    .get("NSE_EQ", {})
                    .get(sec_id_str)
                )
                if quote_data:
                    try:
                        trade_time = datetime.strptime(
                            quote_data["last_trade_time"], "%d/%m/%Y %H:%M:%S"
                        ).replace(tzinfo=IST)
                        result[sec_id] = {
                            "price": float(quote_data["last_price"]),
                            "timestamp": trade_time,
                        }
                    except KeyError:
                        result[sec_id] = None
                else:
                    result[sec_id] = None
            return result
        elif response.get("status") == "failure":
            logger.warning(f"Quote API failed for {security_ids}: {response}")
            return {sec_id: None for sec_id in security_ids}
        else:
            logger.error(f"Unexpected quote API response: {response}")
            return {sec_id: None for sec_id in security_ids}
    except Exception as e:
        logger.error(f"Batch quote error for {security_ids}: {str(e)}")
        return {sec_id: None for sec_id in security_ids}


async def _fetch_quote_fallback_batch(
    security_ids: List[int],
) -> Dict[int, Optional[Dict]]:
    logger.warning(f"Falling back to REST for securities: {security_ids}")
    batch_size = 5
    results: Dict[int, Optional[Dict]] = {}
    for i in range(0, len(security_ids), batch_size):
        batch = security_ids[i : i + batch_size]
        await asyncio.sleep(0.2)
        batch_results = await _fetch_quote_batch(batch)
        results.update(batch_results)
    return results


async def _fetch_quote_fallback(security_id: int) -> Dict[int, Optional[Dict]]:
    return await _fetch_quote_fallback_batch([security_id])


async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)


async def _get_simulated_combined_data(security_id: int) -> Optional[pd.DataFrame]:
    try:
        if security_id in SIM_DATA_CACHE:
            return SIM_DATA_CACHE[security_id]
        file_path = os.path.join(COMBINED_DATA_DIR, f"combined_data_{security_id}.csv")
        if os.path.exists(file_path):
            df = await run_in_thread(pd.read_csv, file_path)
            df.columns = [c.lower() for c in df.columns]
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            elif "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.drop("timestamp", axis=1)
            required = ["datetime", "open", "high", "low", "close", "volume"]
            if any(c not in df.columns for c in required):
                return None
            df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")
            SIM_DATA_CACHE[security_id] = df
            return df
        return None
    except Exception as e:
        logger.error(f"Failed to load simulated data: {e}")
        return None


async def fetch_realtime_quote(security_ids: List[int]) -> Dict[int, Optional[Dict]]:
    """Return latest quotes keyed by security id.

    Behavior:
    - Simulation: returns last close from combined_data CSV.
    - Live: attempts WebSocket latest, otherwise REST fallback.
    """
    if not security_ids:
        return {}

    if SIMULATION_MODE:
        results: Dict[int, Optional[Dict]] = {}
        for sid in security_ids:
            try:
                df = SIM_DATA_CACHE.get(sid)
                if df is None:
                    df = await _get_simulated_combined_data(sid)
                if df is not None and not df.empty:
                    last_row = df.iloc[-1]
                    ts = last_row.get("datetime", datetime.now(IST))
                    if isinstance(ts, pd.Timestamp):
                        ts = ts.to_pydatetime()
                    if ts.tzinfo is None:
                        ts = IST.localize(ts)
                    results[sid] = {"price": float(last_row["close"]), "timestamp": ts}
                else:
                    results[sid] = None
            except Exception as e:
                logger.error(f"Simulated quote error for {sid}: {e}")
                results[sid] = None
        return results

    try:
        ws_client = await get_websocket_client()
        if not ws_client or not ws_client.is_connected:
            return await _fetch_quote_fallback_batch(security_ids)
        instruments = [
            {"SecurityId": sid, "ExchangeSegment": 1} for sid in security_ids
        ]
        ws_client.subscribe_instruments(instruments)

        results: Dict[int, Optional[Dict]] = {}
        current_time = datetime.now(IST)
        with _tick_lock:
            for sid in security_ids:
                tick_data = _latest_ticks.get(sid)
                if tick_data and tick_data.get("price") is not None:
                    ts = tick_data.get("timestamp", current_time)
                    if ts.tzinfo is None:
                        ts = IST.localize(ts)
                    results[sid] = {"price": float(tick_data["price"]), "timestamp": ts}
                else:
                    rest_result = await _fetch_quote_fallback(sid)
                    results[sid] = rest_result.get(sid) if rest_result else None
        return results
    except Exception as e:
        logger.error(f"WebSocket quote error: {e}")
        return await _fetch_quote_fallback_batch(security_ids)
