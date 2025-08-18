"""
Telegram client for sending alerts and notifications.
"""

import asyncio
import aiohttp
import logging
from datetime import datetime
from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SIMULATION_MODE, IST
from .rate_limiter import OptimizedRateLimiter

logger = logging.getLogger("quant_trader")


class TelegramQueue:
    def __init__(self):
        self.message_queue = asyncio.Queue(maxsize=100)
        self.is_running = False
        self._worker_task = None
        self.rate_limiter = OptimizedRateLimiter(
            rate_limit=30
        )  # 30 messages per minute

    async def start(self):
        if not self.is_running:
            self.is_running = True
            self._worker_task = asyncio.create_task(self._process_messages())
            logger.info("Telegram queue started")

    async def stop(self):
        self.is_running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Telegram queue stopped")

    async def _process_messages(self):
        while self.is_running:
            try:
                queue_size = self.message_queue.qsize()
                if queue_size > 80:
                    logger.warning(f"Telegram queue size high: {queue_size}/100")
                await self.rate_limiter.acquire()
                message = await asyncio.wait_for(self.message_queue.get(), timeout=5.0)
                success = await self._send_message_with_retry(message)
                if success:
                    logger.debug(f"Telegram message sent successfully")
                else:
                    logger.warning(f"Failed to send telegram message after retries")
                self.message_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Telegram queue error: {e}")
                await asyncio.sleep(5)

    async def _send_message_with_retry(
        self, message: str, max_retries: int = 3
    ) -> bool:
        for attempt in range(max_retries):
            try:
                success = await self._send_message(message)
                if success:
                    logger.debug(f"Telegram message sent on attempt {attempt + 1}")
                    return True
                logger.warning(f"Telegram send attempt {attempt + 1} failed")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
            except asyncio.TimeoutError as e:
                logger.error(f"Telegram send attempt {attempt + 1} timed out: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
            except Exception as e:
                logger.error(
                    f"Telegram send attempt {attempt + 1} failed with error: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)

        # Fallback: Log to file
        try:
            with open("failed_telegram_messages.log", "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} - {message}\n"
                )
            logger.info("Logged failed Telegram message to file")
        except Exception as e:
            logger.error(f"Failed to log message to file: {e}")
        return False

    async def _send_message(self, message: str) -> bool:
        try:
            if SIMULATION_MODE and not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
                logger.info(f"[SIM] Telegram message: {message[:100]}...")
                return True

            if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
                logger.error("Missing Telegram credentials")
                return False

            if len(message) > 4000:
                message = message[:3900] + "...\n[TRUNCATED]"

            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown",
            }

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return True
                    else:
                        text = await response.text()
                        logger.error(f"Telegram API error {response.status}: {text}")
                        return False
        except Exception as e:
            import traceback

            stack_trace = traceback.format_exc()
            logger.error(f"Telegram send failed: {stack_trace}")
            return False

    async def send_alert(self, message: str):
        try:
            if SIMULATION_MODE:
                message = f"[SIM] {message}"
            await self.message_queue.put(message)
            logger.debug(f"Queued telegram message: {message[:50]}...")
        except asyncio.QueueFull:
            logger.warning("Telegram queue full, dropping message")
        except Exception as e:
            logger.error(f"Failed to queue telegram message: {e}")


# Initialize telegram queue
telegram_queue = TelegramQueue()


async def send_telegram_alert(message: str):
    """Send telegram alert via queue"""
    await telegram_queue.send_alert(message)
