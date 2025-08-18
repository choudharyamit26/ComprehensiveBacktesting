"""
Data processing and analysis functions for the Streamlit application.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import pytz
import backtrader as bt
from comprehensive_backtesting.reports import PerformanceAnalyzer
from .config import IST
from .utils import get_strategy, get_strategy_params, safe_float

logger = logging.getLogger(__name__)


def create_consolidated_metrics(all_metrics):
    """Create consolidated metrics DataFrame with composite scores"""
    consolidated = []

    for metric_set in all_metrics:
        record = {"Ticker": metric_set["Ticker"], "Strategy": metric_set["Strategy"]}

        # Backtest metrics
        bt = metric_set.get("backtest", {})
        record.update({f"BT_{k}": v for k, v in bt.items()})

        # Optimization metrics
        opt = metric_set.get("optimization", {})
        record.update({f"OPT_{k}": v for k, v in opt.items()})

        # Walkforward metrics
        wf = metric_set.get("walkforward", {})
        record.update({f"WF_{k}": v for k, v in wf.items()})

        # Composite scores - include negative values
        win_rates = []
        if "win_rate" in bt and bt["win_rate"] is not None:
            win_rates.append(float(bt["win_rate"]) * 0.2)  # 20% weight
        if "win_rate" in opt and opt["win_rate"] is not None:
            win_rates.append(float(opt["win_rate"]) * 0.3)  # 30% weight
        if "win_rate" in wf and wf["win_rate"] is not None:
            win_rates.append(float(wf["win_rate"]) * 0.5)  # 50% weight
        record["Composite_Win_Rate"] = sum(win_rates) if win_rates else 0

        # Sharpe composite - include negative values
        sharpe_contrib = []
        if "sharpe_ratio" in bt and bt["sharpe_ratio"] is not None:
            sharpe_contrib.append(float(bt["sharpe_ratio"]) * 0.3)
        if "sharpe_ratio" in wf and wf["sharpe_ratio"] is not None:
            sharpe_contrib.append(float(wf["sharpe_ratio"]) * 0.7)
        record["Composite_Sharpe"] = sum(sharpe_contrib) if sharpe_contrib else 0

        # Degradation metric - compare optimization vs walkforward
        opt_return = (
            float(opt.get("total_return_pct", 0))
            if opt.get("total_return_pct") is not None
            else 0
        )
        wf_return = (
            float(wf.get("total_return_pct", 0))
            if wf.get("total_return_pct") is not None
            else 0
        )
        record["Degradation_Pct"] = opt_return - wf_return

        consolidated.append(record)

    if not consolidated:
        return pd.DataFrame()

    df = pd.DataFrame(consolidated)

    # Sort by composite scores
    if "Composite_Win_Rate" in df.columns:
        df = df.sort_values("Composite_Win_Rate", ascending=False)

    # Column ordering
    column_order = [
        "Ticker",
        "Strategy",
        "Composite_Win_Rate",
        "Composite_Sharpe",
        "Degradation_Pct",
        "BT_win_rate",
        "BT_sharpe_ratio",
        "BT_total_return_pct",
        "BT_profit_factor",
        "OPT_win_rate",
        "OPT_sharpe_ratio",
        "OPT_total_return_pct",
        "OPT_profit_factor",
        "WF_win_rate",
        "WF_sharpe_ratio",
        "WF_total_return_pct",
        "WF_profit_factor",
    ]

    # Return only existing columns
    return df[[col for col in column_order if col in df.columns]]


def generate_strategy_report(strategy_result, strategy_name, ticker, timeframe):
    """Generate a report for a strategy if it meets criteria"""
    try:
        # Get parameters from the strategy
        params = get_strategy_params(strategy_result)

        # Initialize analyzer based on result type
        if isinstance(strategy_result, dict):
            report = strategy_result
            summary = report.get("summary", {})
            trade_analysis = report.get("trade_analysis", {})
        else:
            analyzer = PerformanceAnalyzer(strategy_result)
            report = analyzer.generate_full_report()
            summary = report.get("summary", {})
            trade_analysis = report.get("trade_analysis", {})

        total_trades = trade_analysis.get("total_trades", 0)
        win_rate = trade_analysis.get("win_rate_percent", 0)

        if total_trades > 10 and win_rate > 50:
            params_str = ", ".join([f"{k}={v}" for k, v in params.items()])

            # Get detailed trade analysis
            detailed_trade_analysis = analyze_best_trades(strategy_result)
            if "error" in detailed_trade_analysis:
                logger.warning(
                    f"Could not get detailed trade analysis: {detailed_trade_analysis['error']}"
                )
                detailed_trade_analysis = {}

            return {
                "Strategy": strategy_name,
                "Ticker": ticker,
                "Timeframe": timeframe,
                "Total Trades": total_trades,
                "Win Rate (%)": win_rate,
                "Total Return (%)": summary.get("total_return_pct", 0),
                "Sharpe Ratio": summary.get("sharpe_ratio", 0),
                "Max Drawdown (%)": summary.get("max_drawdown_pct", 0),
                "Profit Factor": trade_analysis.get("profit_factor", 0),
                "Avg Trade Duration": trade_analysis.get("avg_trade_duration", 0),
                "Best Trade Return (%)": trade_analysis.get("best_trade_return_pct", 0),
                "Worst Trade Return (%)": trade_analysis.get(
                    "worst_trade_return_pct", 0
                ),
                "Total P&L": detailed_trade_analysis.get("total_pnl", 0),
                "Winning Trades": detailed_trade_analysis.get("winning_trades", 0),
                "Losing Trades": detailed_trade_analysis.get("losing_trades", 0),
                "Avg Winning Trade": detailed_trade_analysis.get(
                    "avg_winning_trade", 0
                ),
                "Avg Losing Trade": detailed_trade_analysis.get("avg_losing_trade", 0),
                "Best Trade P&L": detailed_trade_analysis.get("best_trade_pnl", 0),
                "Worst Trade P&L": detailed_trade_analysis.get("worst_trade_pnl", 0),
                "Avg Holding (Bars)": detailed_trade_analysis.get(
                    "avg_holding_bars", 0
                ),
                "Avg Winning Hold (Bars)": detailed_trade_analysis.get(
                    "avg_holding_won", 0
                ),
                "Avg Losing Hold (Bars)": detailed_trade_analysis.get(
                    "avg_holding_lost", 0
                ),
                "Parameters": params_str,
                "Params": params,
                "Start Date": summary.get("start_date", ""),
                "End Date": summary.get("end_date", ""),
            }
        return None
    except Exception as e:
        logger.error(f"Error generating strategy report: {str(e)}")
        return None


def analyze_best_trades(results):
    """Analyze and extract best performing trades."""
    try:
        strategy = get_strategy(results)
        # Robust trade extraction from PerformanceAnalyzer if available
        from comprehensive_backtesting.reports import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer(results)
        trade_analysis = analyzer.get_trade_analysis()
        trades = trade_analysis.get("trades", [])
        if trades:
            trade_list = []
            for t in trades:
                try:
                    entry_time = t["entry_time"]
                    exit_time = t["exit_time"]
                    pnl = t.get("pnl", 0)
                    size = t.get("size", 1)
                    trade_info = {
                        "trade_id": t.get("trade_id", None),
                        "entry_time": (
                            str(entry_time) if entry_time is not None else "-"
                        ),
                        "exit_time": str(exit_time) if exit_time is not None else "-",
                        "entry_price": t.get("price_in", 0),
                        "exit_price": t.get("price_out", 0),
                        "pnl": pnl,
                        "size": size,
                        "direction": t.get("direction", ""),
                        "return_pct": (
                            (pnl / (t.get("price_in", 1) * abs(size))) * 100
                            if t.get("price_in", 0) > 0 and size != 0
                            else 0
                        ),
                    }
                    trade_list.append(trade_info)
                except Exception as e:
                    logger.warning(f"Invalid trade data format: {e}")
                    continue
            if trade_list:
                # Sort trades by PnL to find best trades
                trade_list.sort(key=lambda x: x["pnl"], reverse=True)
                best_trades = trade_list[:5]
                total_pnl = sum(trade["pnl"] for trade in trade_list)
                winning_trades = [trade for trade in trade_list if trade["pnl"] > 0]
                losing_trades = [trade for trade in trade_list if trade["pnl"] < 0]
                return {
                    "best_trades": best_trades,
                    "total_trades": len(trade_list),
                    "winning_trades": len(winning_trades),
                    "losing_trades": len(losing_trades),
                    "total_pnl": total_pnl,
                    "avg_winning_trade": (
                        sum(trade["pnl"] for trade in winning_trades)
                        / len(winning_trades)
                        if winning_trades
                        else 0
                    ),
                    "avg_losing_trade": (
                        sum(trade["pnl"] for trade in losing_trades)
                        / len(losing_trades)
                        if losing_trades
                        else 0
                    ),
                    "best_trade_pnl": best_trades[0]["pnl"] if best_trades else 0,
                    "worst_trade_pnl": trade_list[-1]["pnl"] if trade_list else 0,
                }

        # fallback to legacy logic if no trades found
        trades = None
        if hasattr(strategy, "analyzers") and hasattr(
            strategy.analyzers, "tradeanalyzer"
        ):
            trades = strategy.analyzers.tradeanalyzer.get_analysis()
        elif hasattr(strategy, "analyzers") and hasattr(strategy.analyzers, "trades"):
            trades = strategy.analyzers.trades.get_analysis()

        if not trades:
            return {"error": "No trade data available"}

        # Try to extract per-trade list (preferred)
        trade_list = []
        closed_trades = trades.get("closed", []) or trades.get("trades", [])
        if (
            isinstance(closed_trades, list)
            and closed_trades
            and isinstance(closed_trades[0], dict)
        ):
            for i, trade in enumerate(closed_trades):
                try:
                    entry_time = (
                        pd.to_datetime(trade["datein"], unit="s")
                        .tz_localize("UTC")
                        .tz_convert(IST)
                    )
                    exit_time = (
                        pd.to_datetime(trade["dateout"], unit="s")
                        .tz_localize("UTC")
                        .tz_convert(IST)
                    )
                    pnl = trade.get("pnl", 0)

                    trade_info = {
                        "trade_id": i + 1,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "entry_price": trade["pricein"],
                        "exit_price": trade["priceout"],
                        "pnl": pnl,
                        "duration_hours": (exit_time - entry_time).total_seconds()
                        / 3600,
                        "return_pct": (
                            (pnl / (trade["pricein"] * trade.get("size", 1))) * 100
                            if trade["pricein"] > 0
                            else 0
                        ),
                    }
                    trade_list.append(trade_info)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Invalid trade data format: {e}")
                    continue
            if trade_list:
                # Sort trades by PnL to find best trades
                trade_list.sort(key=lambda x: x["pnl"], reverse=True)
                best_trades = trade_list[:5]
                total_pnl = sum(trade["pnl"] for trade in trade_list)
                winning_trades = [trade for trade in trade_list if trade["pnl"] > 0]
                losing_trades = [trade for trade in trade_list if trade["pnl"] < 0]
                return {
                    "best_trades": best_trades,
                    "total_trades": len(trade_list),
                    "winning_trades": len(winning_trades),
                    "losing_trades": len(losing_trades),
                    "total_pnl": total_pnl,
                    "avg_winning_trade": (
                        sum(trade["pnl"] for trade in winning_trades)
                        / len(winning_trades)
                        if winning_trades
                        else 0
                    ),
                    "avg_losing_trade": (
                        sum(trade["pnl"] for trade in losing_trades)
                        / len(losing_trades)
                        if losing_trades
                        else 0
                    ),
                    "best_trade_pnl": best_trades[0]["pnl"] if best_trades else 0,
                    "worst_trade_pnl": trade_list[-1]["pnl"] if trade_list else 0,
                }

        # If no per-trade list, use summary stats from TradeAnalyzer
        if "total" in trades and "won" in trades and "lost" in trades:
            total_trades = trades.get("total", {}).get("total", 0)
            winning_trades = trades.get("won", {}).get("total", 0)
            losing_trades = trades.get("lost", {}).get("total", 0)
            total_pnl = trades.get("pnl", {}).get("net", {}).get("total", 0)
            avg_winning_trade = trades.get("won", {}).get("pnl", {}).get("average", 0)
            avg_losing_trade = trades.get("lost", {}).get("pnl", {}).get("average", 0)
            best_trade_pnl = trades.get("won", {}).get("pnl", {}).get("max", 0)
            worst_trade_pnl = trades.get("lost", {}).get("pnl", {}).get("max", 0)
            # Holding period (bars)
            avg_holding_bars = trades.get("len", {}).get("average", 0)
            avg_holding_won = trades.get("len", {}).get("won", {}).get("average", 0)
            avg_holding_lost = trades.get("len", {}).get("lost", {}).get("average", 0)
            # Compose a pseudo-trade for UI
            best_trades = [
                {
                    "trade_id": 1,
                    "entry_time": None,
                    "exit_time": None,
                    "entry_price": None,
                    "exit_price": None,
                    "pnl": best_trade_pnl,
                    "duration_hours": avg_holding_bars,  # bars, not hours
                    "return_pct": None,
                }
            ]
            return {
                "best_trades": best_trades,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "total_pnl": total_pnl,
                "avg_winning_trade": avg_winning_trade,
                "avg_losing_trade": avg_losing_trade,
                "best_trade_pnl": best_trade_pnl,
                "worst_trade_pnl": worst_trade_pnl,
                "avg_holding_bars": avg_holding_bars,
                "avg_holding_won": avg_holding_won,
                "avg_holding_lost": avg_holding_lost,
            }
        return {"error": "No valid trades found"}
    except Exception as e:
        logger.error(f"Error analyzing best trades: {e}")
        return {"error": str(e)}


def analyze_best_time_ranges(results, data=None):
    """Analyze time ranges when most winning trades occurred using extract_trades."""
    try:
        from .table_generators import extract_trades

        strategy = get_strategy(results)
        # Use extract_trades for robust trade extraction
        trades_df = extract_trades(strategy, data)
        if trades_df.empty:
            logger.warning(
                "[analyze_best_time_ranges] No trade data available (trades_df is empty)"
            )
            return {"error": "No trade data available"}

        # Filter for winning trades
        winning_trades = trades_df[trades_df["pnl"] > 0].copy()
        if winning_trades.empty:
            logger.warning("[analyze_best_time_ranges] No winning trades found")
            return {"error": "No winning trades found"}

        # Ensure entry_time is datetime and in IST
        if not pd.api.types.is_datetime64_any_dtype(winning_trades["entry_time"]):
            winning_trades["entry_time"] = pd.to_datetime(winning_trades["entry_time"])
        if winning_trades["entry_time"].dt.tz is None:
            winning_trades["entry_time"] = (
                winning_trades["entry_time"].dt.tz_localize("UTC").dt.tz_convert(IST)
            )
        else:
            winning_trades["entry_time"] = winning_trades["entry_time"].dt.tz_convert(
                IST
            )

        # Add hour, day_of_week, month columns
        winning_trades["hour"] = winning_trades["entry_time"].dt.hour
        winning_trades["day_of_week"] = winning_trades["entry_time"].dt.day_name()
        winning_trades["month"] = winning_trades["entry_time"].dt.month_name()

        # Analyze by hour of day
        hourly_stats = (
            winning_trades.groupby("hour")
            .agg(count=("pnl", "count"), total_pnl=("pnl", "sum"))
            .to_dict("index")
        )
        best_hours = sorted(
            hourly_stats.items(), key=lambda x: x[1]["count"], reverse=True
        )[:3]

        # Analyze by day of week
        daily_stats = (
            winning_trades.groupby("day_of_week")
            .agg(count=("pnl", "count"), total_pnl=("pnl", "sum"))
            .to_dict("index")
        )
        best_days = sorted(
            daily_stats.items(), key=lambda x: x[1]["count"], reverse=True
        )[:3]

        # Analyze by month
        monthly_stats = (
            winning_trades.groupby("month")
            .agg(count=("pnl", "count"), total_pnl=("pnl", "sum"))
            .to_dict("index")
        )
        best_months = sorted(
            monthly_stats.items(), key=lambda x: x[1]["count"], reverse=True
        )[:3]

        return {
            "total_winning_trades": len(winning_trades),
            "best_hours": [
                {
                    "hour": f"{hour}:00",
                    "trades": stats["count"],
                    "total_pnl": stats["total_pnl"],
                }
                for hour, stats in best_hours
            ],
            "best_days": [
                {"day": day, "trades": stats["count"], "total_pnl": stats["total_pnl"]}
                for day, stats in best_days
            ],
            "best_months": [
                {
                    "month": month,
                    "trades": stats["count"],
                    "total_pnl": stats["total_pnl"],
                }
                for month, stats in best_months
            ],
            "hourly_distribution": hourly_stats,
            "daily_distribution": daily_stats,
            "monthly_distribution": monthly_stats,
        }
    except Exception as e:
        logger.error(f"Error analyzing best time ranges: {e}")
        return {"error": str(e)}
