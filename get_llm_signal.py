import re
from langchain_groq import ChatGroq
import pandas as pd
import os
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)

# Ensure logging is configured
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LLM_RATE_LIMIT = asyncio.Semaphore(2)  # Max 2 concurrent LLM calls


def remove_think_tags(text: str) -> str:
    """Remove all content between and including <THINK> and </THINK> tags, case-insensitive"""
    cleaned_text = re.sub(
        r"<THINK\b[^>]*>.*?</THINK>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    logger.info(f"Cleaned text after removing <THINK> tags: {cleaned_text}")
    return cleaned_text.strip()


async def get_llm_signal(ticker: str, combined_data: pd.DataFrame) -> str:
    """Get trading signal from LLM based on market data, waiting for complete LLM response"""
    try:
        # Prepare market data summary for LLM
        latest_data = combined_data.tail(50)  # Last 50 bars for context

        # Calculate key technical indicators for LLM context
        current_price = latest_data["close"].iloc[-1]
        price_change_pct = (
            (current_price - latest_data["close"].iloc[-20])
            / latest_data["close"].iloc[-20]
        ) * 100
        volume_avg = latest_data["volume"].mean()
        current_volume = latest_data["volume"].iloc[-1]
        volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1

        # Recent price action
        recent_high = latest_data["high"].tail(10).max()
        recent_low = latest_data["low"].tail(10).min()
        price_position = (
            (current_price - recent_low) / (recent_high - recent_low)
            if recent_high != recent_low
            else 0.5
        )

        # Moving averages for trend analysis
        ma_5 = latest_data["close"].tail(5).mean()
        ma_20 = latest_data["close"].tail(20).mean()

        # RSI calculation (simple approximation)
        price_changes = latest_data["close"].diff().tail(14)
        gains = price_changes.where(price_changes > 0, 0).mean()
        losses = (-price_changes.where(price_changes < 0, 0)).mean()
        rsi = 100 - (100 / (1 + (gains / losses))) if losses != 0 else 50

        # Create few-shot prompt for LLM
        system_prompt = """You are an expert technical analyst specializing in short-term trading signals. Analyze market data and provide EXACTLY one word: BUY or SELL, based on strong technical patterns. Do NOT include reasoning, <think> tags, or any other text.

        Examples of your analysis:

        Example 1:
        Ticker: AAPL
        Current Price: 150.25
        20-period Price Change: +3.2%
        Volume Ratio: 1.8x
        Price Position: 0.85
        5-day MA: 148.50, 20-day MA: 145.20
        RSI: 68
        Recent pattern: Breaking above resistance with high volume
        Signal: BUY

        Example 2:
        Ticker: TSLA
        Current Price: 245.80
        20-period Price Change: -5.1%
        Volume Ratio: 2.1x
        Price Position: 0.15
        5-day MA: 250.30, 20-day MA: 255.40
        RSI: 32
        Recent pattern: Breaking below support with heavy volume
        Signal: SELL

        Example 3:
        Ticker: MSFT
        Current Price: 415.60
        20-period Price Change: +1.8%
        Volume Ratio: 1.2x
        Price Position: 0.55
        5-day MA: 414.20, 20-day MA: 410.80
        RSI: 58
        Recent pattern: Consolidating near highs with average volume
        Signal: BUY

        Example 4:
        Ticker: NVDA
        Current Price: 125.40
        20-period Price Change: -4.8%
        Volume Ratio: 1.9x
        Price Position: 0.20
        5-day MA: 128.90, 20-day MA: 132.10
        RSI: 28
        Recent pattern: Failed bounce from oversold levels
        Signal: SELL

        Rules:
        - Respond with EXACTLY one word: BUY or SELL
        - BUY: Strong bullish momentum, breakouts, uptrends with volume
        - SELL: Strong bearish momentum, breakdowns, downtrends with volume
        - Consider: Price vs MAs, RSI levels, volume confirmation, trend direction
        - Do NOT include <think> tags, reasoning, or any other text"""

        user_prompt = f"""
        Ticker: {ticker}
        Current Price: {current_price:.2f}
        20-period Price Change: {price_change_pct:.2f}%
        Volume Ratio: {volume_ratio:.2f}x
        Price Position: {price_position:.2f}
        5-day MA: {ma_5:.2f}, 20-day MA: {ma_20:.2f}
        RSI: {rsi:.2f}
        
        Recent Price Action (last 10 periods):
        {latest_data[['open', 'high', 'low', 'close', 'volume']].tail(10).to_string()}
        
        Signal:"""

        # Attempt LLM call with up to 2 retries for non-actionable responses
        max_attempts = 2
        attempt = 1
        while attempt <= max_attempts:
            logger.info(
                f"{ticker} - Initiating LLM API call (attempt {attempt}/{max_attempts})"
            )
            try:
                # Set a timeout of 30 seconds for the LLM call
                llm_response = await asyncio.wait_for(
                    call_llm_api(system_prompt, user_prompt), timeout=30.0
                )
                logger.info(
                    f"{ticker} - Received complete LLM response: {llm_response}"
                )

                # Log raw response for debugging
                logger.debug(f"{ticker} - Raw LLM response: {llm_response}")

                # Clean the response first
                signal = remove_think_tags(llm_response).strip().upper()
                logger.info(f"{ticker} - Cleaned LLM signal: {signal}")

                # Validate cleaned signal
                if signal in ["BUY", "SELL"]:
                    logger.info(f"{ticker} - Valid LLM signal received: {signal}")
                    print(f"LLM response for {ticker}: {signal}")
                    return signal
                else:
                    logger.warning(
                        f"{ticker} - Non-actionable response on attempt {attempt}: {llm_response} (cleaned: {signal})"
                    )
                    attempt += 1
                    if attempt <= max_attempts:
                        logger.info(
                            f"{ticker} - Retrying LLM call due to non-actionable response"
                        )
                        await asyncio.sleep(1)  # Brief delay before retry
                    continue
            except asyncio.TimeoutError:
                logger.error(
                    f"{ticker} - LLM API call timed out after 30 seconds on attempt {attempt}"
                )
                attempt += 1
                if attempt <= max_attempts:
                    logger.info(f"{ticker} - Retrying LLM call due to timeout")
                    await asyncio.sleep(1)
                continue

        logger.warning(f"{ticker} - All attempts failed to produce actionable signal")
        return "NONE"

    except Exception as e:
        logger.error(f"{ticker} - LLM signal generation failed: {e}")
        return "NONE"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((Exception,)),
)
async def call_llm_api(system_prompt: str, user_prompt: str) -> str:
    """
    LLM API call with few-shot prompting using Groq with rate limiting
    """
    async with LLM_RATE_LIMIT:  # Limit concurrent calls
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.error("GROQ_API_KEY not found in environment variables")
                return "NONE"

            llm = ChatGroq(
                api_key=api_key,
                model="deepseek-r1-distill-llama-70b",
                temperature=0.1,
                max_tokens=2000,  # Increased to allow complete response
                max_retries=0,  # Disable internal retries, we handle them manually
            )

            # Create the messages for the LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            logger.debug("Calling Groq LLM API...")

            # Add small delay to help with rate limiting
            await asyncio.sleep(0.5)

            # Make the actual API call
            response = await llm.ainvoke(messages)

            # Extract the content from the response
            if hasattr(response, "content"):
                signal = response.content.strip().upper()
                logger.info(f"LLM raw response: {response.content}, SIGNAL: {signal}")
                return signal
            else:
                logger.error("Unexpected response format from LLM")
                return "NONE"

        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                logger.warning(f"Rate limit hit, will retry: {str(e)}")
                raise  # Let tenacity handle the retry
            else:
                logger.error(f"LLM API call failed: {str(e)}")
                return "NONE"
