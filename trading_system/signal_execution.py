"""
Signal execution system for processing trading signals.
"""

import ast
import os
import logging
import traceback
from datetime import datetime, timedelta
from typing import List, Dict

from .config import (
    SIMULATION_MODE,
    MIN_VOTES,
    MAX_DAILY_LOSS_PERCENT,
    ACCOUNT_SIZE,
    VOLATILITY_THRESHOLD,
    DEFAULT_TICK_SIZE,
    IST,
)
from .calculations import calculate_vwap, calculate_risk_params, round_to_tick_size
from .data_manager import (
    fetch_realtime_quote,
    get_combined_data,
    calculate_stock_volatility,
)
from .order_management import place_super_order
from .position_manager import position_manager
from .pnl_tracker import pnl_tracker
from .strategy_registry import get_strategy, STRATEGY_REGISTRY
from .telegram_client import send_telegram_alert
from .rate_limiter import adaptive_semaphore

logger = logging.getLogger("quant_trader")
trade_logger = logging.getLogger("trade_execution")


async def execute_strategy_signal(
    ticker: str,
    security_id: int,
    signal: str,
    regime: str,
    adx_value: float,
    atr_value: float,
    hist_data,
    strategy_name: str,
    strategy_instance=None,
    **params,
) -> bool:
    """Enhanced signal execution with proper position direction handling."""
    try:
        # Check if we already have a position for this security
        if await position_manager.has_position(security_id):
            existing_direction = await position_manager.get_position_direction(
                security_id
            )
            logger.info(
                f"{ticker} - Already have {existing_direction} position, skipping new {signal} signal"
            )
            return False

        # Add volatility filter
        volatility = await calculate_stock_volatility(security_id)
        if volatility > VOLATILITY_THRESHOLD:
            logger.warning(
                f"Skipping {ticker} due to high volatility: {volatility:.4f}"
            )
            return False

        # Check daily loss limit
        pnl_data = await pnl_tracker.update_daily_pnl()
        if isinstance(pnl_data, dict):
            current_pnl = pnl_data.get("total", 0)
        else:
            current_pnl = pnl_data or 0

        if current_pnl <= -MAX_DAILY_LOSS_PERCENT * ACCOUNT_SIZE:
            message = (
                f"🛑 TRADING HALTED: Daily loss limit reached\n"
                f"Current P&L: ₹{current_pnl:.2f}\n"
                f"Limit: ₹{-MAX_DAILY_LOSS_PERCENT * ACCOUNT_SIZE:.2f}"
            )
            await send_telegram_alert(message)
            logger.critical("Daily loss limit reached - trading halted")
            return False

        # Get current quote
        quotes = await fetch_realtime_quote([security_id])
        quote = quotes.get(security_id)
        if not quote:
            logger.warning(f"Price unavailable for {ticker}")
            return False

        current_price = quote["price"]
        vwap = await calculate_vwap(hist_data)

        # Improved entry price logic
        entry_price = (
            min(current_price, vwap * 0.998)
            if signal == "BUY"
            else max(current_price, vwap * 1.002)
        )

        # Calculate risk parameters
        risk_params = calculate_risk_params(regime, atr_value, entry_price, signal)
        now = datetime.now(IST)

        tick_size = DEFAULT_TICK_SIZE

        # Round prices to the nearest tick size
        rounded_entry_price = round_to_tick_size(entry_price, tick_size)
        rounded_stop_loss = round_to_tick_size(risk_params["stop_loss"], tick_size)
        rounded_take_profit = round_to_tick_size(risk_params["take_profit"], tick_size)

        # Check available funds
        from .data_manager import dhan

        funds = dhan.get_fund_limits().get("data", {}).get("availabelBalance", 0)
        required_margin = rounded_entry_price * risk_params["position_size"]

        if funds < required_margin:
            logger.warning(
                f"{ticker} - Insufficient funds: ₹{funds:.2f} < ₹{required_margin:.2f}"
            )
            await send_telegram_alert(
                f"*{ticker} Order Failed* ❌\n"
                f"Signal: {signal} at ₹{rounded_entry_price:.2f}\n"
                f"Insufficient funds: ₹{funds:.2f} < ₹{required_margin:.2f}"
            )
            return False

        # Prepare entry notification
        direction_emoji = "🟢" if signal == "BUY" else "🔴"
        position_size = risk_params["position_size"]
        position_type = "Long" if signal == "BUY" else "Short"

        message = (
            f"*{ticker} ENTRY SIGNAL* {direction_emoji}\n"
            f"Strategy: `{strategy_name}`\n"
            f"Direction: {position_type}\n"
            f"Entry: ₹{rounded_entry_price:.2f} | VWAP: ₹{vwap:.2f}\n"
            f"Current: ₹{current_price:.2f}\n"
            f"Regime: {regime} (ADX: {adx_value:.2f})\n"
            f"Volatility: {volatility:.4f}\n"
            f"Size: {position_size} | SL: ₹{rounded_stop_loss:.2f}\n"
            f"TP: ₹{rounded_take_profit:.2f}\n"
            f"Risk: ₹{abs(rounded_entry_price - rounded_stop_loss) * position_size:.2f}\n"
            f"Time: {now.strftime('%H:%M:%S')}"
        )

        logger.info(f"Executing {signal} signal for {ticker}")
        await send_telegram_alert(message)

        # Place order
        try:
            order_response = await place_super_order(
                security_id,
                signal,
                rounded_entry_price,
                rounded_stop_loss,
                rounded_take_profit,
                position_size,
            )
        except Exception as e:
            logger.error(f"Order placement failed for {ticker}: {str(e)}")
            await send_telegram_alert(
                f"*{ticker} Order Failed* ❌\n"
                f"Signal: {signal} at ₹{rounded_entry_price:.2f}\n"
                f"Error: {str(e)}"
            )
            return False

        if order_response and order_response.get("orderId"):
            # Add position to manager
            success = await position_manager.add_position(
                order_response["orderId"],
                security_id,
                ticker,
                rounded_entry_price,
                position_size,
                rounded_stop_loss,
                rounded_take_profit,
                signal,  # This is the position direction (BUY = LONG, SELL = SHORT)
                strategy_name,
                strategy_instance,
            )
            if success:
                trade_logger.info(
                    f"{'[SIM] ' if SIMULATION_MODE else ''}ORDER EXECUTED | {ticker} | "
                    f"{signal} | Qty: {position_size} | Price: ₹{rounded_entry_price:.2f}"
                )
            return success
        else:
            logger.error(f"Order failed for {ticker}: {order_response}")
            await send_telegram_alert(
                f"*{ticker} Order Failed* ❌\n"
                f"Signal: {signal} at ₹{rounded_entry_price:.2f}\n"
                f"Order response: {order_response}"
            )
            return False

    except Exception as e:
        logger.error(f"Signal execution failed for {ticker}: {str(e)}")
        logger.error(traceback.format_exc())
        await send_telegram_alert(f"*{ticker} Execution Failed* ❌\nError: {str(e)}")
        return False


async def process_stock_with_exit_monitoring(
    ticker: str, security_id: int, strategies: List[Dict]
) -> None:
    """Separate entry and exit signal processing"""
    async with adaptive_semaphore:
        try:
            logger.debug(f"Processing {ticker} (ID: {security_id})")
            current_time = datetime.now(IST)

            # Get current combined data
            combined_data = await get_combined_data(security_id)
            if combined_data is None:
                logger.warning(f"{ticker} - No data available")
                return

            # Check for existing position FIRST
            if await position_manager.has_position(security_id):
                # We have an existing position - only check for exit signals
                logger.debug(
                    f"{ticker} - Has existing position, checking exit signals only"
                )

                exit_signal = await position_manager.check_strategy_exit_signals(
                    security_id, combined_data
                )
                if exit_signal:
                    order_id = position_manager.positions_by_security[security_id]
                    logger.info(
                        f"{ticker} - Exit signal detected: {exit_signal['reason']}"
                    )
                    await position_manager.execute_strategy_exit(order_id, exit_signal)
                return  # IMPORTANT: Return here to avoid processing entry signals

            # No existing position - check cooldown for new entries
            if not SIMULATION_MODE:
                last_trade_time = await position_manager.get_last_trade_time(ticker)
                if last_trade_time and current_time < last_trade_time + timedelta(
                    minutes=position_manager.cooldown_minutes
                ):
                    logger.debug(f"{ticker} - Skipping due to cooldown")
                    return

            # Check minimum data requirements
            data_length = len(combined_data)
            min_bars = max(
                [
                    get_strategy(s["Strategy"]).get_min_data_points(
                        ast.literal_eval(s["Best_Parameters"])
                        if isinstance(s["Best_Parameters"], str)
                        else s["Best_Parameters"]
                    )
                    for s in strategies
                    if s["Strategy"] in STRATEGY_REGISTRY
                ],
                default=30,
            )

            if data_length < min_bars:
                logger.warning(
                    f"{ticker} - Insufficient data ({data_length} < {min_bars})"
                )
                return

            # Calculate market regime
            from .calculations import calculate_regime

            regime, adx_value, atr_value = calculate_regime(combined_data)
            logger.debug(
                f"{ticker} - Regime: {regime} (ADX: {adx_value:.2f}, ATR: {atr_value:.2f})"
            )

            # Process entry signals from strategies
            signals = []
            strategy_instances = []

            for strat in strategies:
                strategy_name = strat["Strategy"]
                try:
                    strategy_class = get_strategy(strategy_name)
                    params = strat.get("Best_Parameters", {})
                    if isinstance(params, str) and params.strip():
                        try:
                            params = ast.literal_eval(params)
                        except (ValueError, SyntaxError):
                            params = {}

                    strategy_instance = strategy_class(combined_data, **params)
                    signal = strategy_instance.run()

                    if signal in ["BUY", "SELL"]:
                        signals.append(signal)
                        strategy_instances.append(
                            {
                                "instance": strategy_instance,
                                "name": strategy_name,
                                "signal": signal,
                                "params": params,
                            }
                        )
                        logger.debug(
                            f"{ticker} - {strategy_name} generated {signal} signal"
                        )

                except Exception as e:
                    logger.error(f"{ticker} - {strategy_name} failed: {e}")

            if not signals:
                return

            # Execute strongest signal based on votes
            buy_votes = signals.count("BUY")
            sell_votes = signals.count("SELL")
            min_vote_diff = int(os.getenv("MIN_VOTE_DIFF", 1))
            logger.info(f"{ticker} - Buy votes: {buy_votes}, Sell votes: {sell_votes}")
            executed = False
            if buy_votes >= MIN_VOTES and (buy_votes - sell_votes) >= min_vote_diff:
                primary_strategy = next(
                    s for s in strategy_instances if s["signal"] == "BUY"
                )
                executed = await execute_strategy_signal(
                    ticker,
                    security_id,
                    "BUY",
                    regime,
                    adx_value,
                    atr_value,
                    combined_data,
                    primary_strategy["name"],
                    primary_strategy["instance"],
                    **primary_strategy["params"],
                )

            elif sell_votes >= MIN_VOTES and (sell_votes - buy_votes) >= min_vote_diff:
                primary_strategy = next(
                    s for s in strategy_instances if s["signal"] == "SELL"
                )
                executed = await execute_strategy_signal(
                    ticker,
                    security_id,
                    "SELL",
                    regime,
                    adx_value,
                    atr_value,
                    combined_data,
                    primary_strategy["name"],
                    primary_strategy["instance"],
                    **primary_strategy["params"],
                )

            if executed:
                await position_manager.update_last_trade_time(ticker, current_time)

        except Exception as e:
            logger.error(f"{ticker} - Processing failed: {str(e)}")
            logger.error(traceback.format_exc())
