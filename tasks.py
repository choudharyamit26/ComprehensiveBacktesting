import ast
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
from comprehensive_backtesting.registry import get_strategy
from celery_app import app
from comprehensive_backtesting.backtesting import (
    run_basic_backtest,
)
from comprehensive_backtesting.data import get_data_sync
from comprehensive_backtesting.registry import STRATEGY_REGISTRY
from comprehensive_backtesting.reports import PerformanceAnalyzer
from comprehensive_backtesting.walk_forward_analysis import (
    create_walk_forward_analysis,
)
from intraday_stock_filter_nifty import IntradayStockFilter
from app import (
    create_consolidated_metrics,
    extract_report_metrics,
    generate_strategy_report,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_strategy_default_params(strategy_name):
    """Get default parameters for a strategy"""
    try:
        strategy_class = get_strategy(strategy_name)
        if hasattr(strategy_class, "params"):
            # Extract default parameters from strategy class
            defaults = {}
            for param_tuple in strategy_class.params:
                if len(param_tuple) >= 2:
                    param_name, default_value = param_tuple[0], param_tuple[1]
                    # Skip non-configurable params like 'verbose'
                    if param_name not in ["verbose", "tickers", "analyzers"]:
                        defaults[param_name] = default_value
            return defaults

        # Try optimization_params if params doesn't exist
        if hasattr(strategy_class, "optimization_params"):
            defaults = {}
            for param_name, param_range in strategy_class.optimization_params.items():
                if isinstance(param_range, (list, tuple)) and len(param_range) > 0:
                    # Use middle value or first value as default
                    if len(param_range) >= 2 and isinstance(
                        param_range[0], (int, float)
                    ):
                        defaults[param_name] = param_range[len(param_range) // 2]
                    else:
                        defaults[param_name] = param_range[0]
            return defaults

    except Exception as e:
        logger.warning(f"Could not get default params for {strategy_name}: {e}")
    return {}


def extract_best_params_with_fallback(strategy_name, combined_metrics, ticker):
    """Extract best params with improved fallback logic"""
    best_params = {}

    # First try to get from walk-forward results
    for metric in combined_metrics:
        if metric["Ticker"] == ticker and metric["Strategy"] == strategy_name:
            # Check walk-forward results first (most reliable)
            if "walkforward" in metric and metric["walkforward"]:
                wf_data = metric["walkforward"]
                if "best_params" in wf_data and wf_data["best_params"]:
                    best_params = wf_data["best_params"]
                    if isinstance(best_params, str):
                        try:
                            best_params = ast.literal_eval(best_params)
                        except:
                            best_params = {}

                    if best_params and isinstance(best_params, dict):
                        logger.info(
                            f"Using walk-forward params for {ticker}-{strategy_name}: {best_params}"
                        )
                        return best_params

            # Then check optimization results
            if "optimization" in metric and metric["optimization"]:
                opt_data = metric["optimization"]
                if "best_params" in opt_data and opt_data["best_params"]:
                    best_params = opt_data["best_params"]
                    if isinstance(best_params, str):
                        try:
                            best_params = ast.literal_eval(best_params)
                        except:
                            best_params = {}

                    if best_params and isinstance(best_params, dict):
                        logger.info(
                            f"Using optimization params for {ticker}-{strategy_name}: {best_params}"
                        )
                        return best_params

            # Check backtest results for any saved params
            if "backtest" in metric and metric["backtest"]:
                bt_data = metric["backtest"]
                if "best_params" in bt_data and bt_data["best_params"]:
                    best_params = bt_data["best_params"]
                    if isinstance(best_params, str):
                        try:
                            best_params = ast.literal_eval(best_params)
                        except:
                            best_params = {}

                    if best_params and isinstance(best_params, dict):
                        logger.info(
                            f"Using backtest params for {ticker}-{strategy_name}: {best_params}"
                        )
                        return best_params

    # Fallback to strategy defaults
    default_params = get_strategy_default_params(strategy_name)
    if default_params:
        logger.info(
            f"Using default params for {ticker}-{strategy_name}: {default_params}"
        )
        return default_params

    # Last resort - return empty dict but log the issue
    logger.warning(
        f"No parameters found for {ticker}-{strategy_name}, using empty dict"
    )
    return {}


def run_complete_backtests(selected_stocks: List[Dict], strategies: List[str]):
    """Run comprehensive backtests for selected stocks"""

    end_date = datetime.today().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    interval = "5m"
    n_trials = 20
    # tickers = [stock["Stock"] for stock in selected_stocks]
    tickers = [
        "TATAMOTORS",
        "JSWSTEEL",
        "TATASTEEL",
        "HINDALCO",
        "SBIN",
        "ADANIENT",
        "ICICIBANK",
        "AXISBANK",
        "BAJFINANCE",
        "M&M",
        "KOTAKBANK",
        "EICHERMOT",
        "HDFCBANK",
        "INDUSINDBK",
        "MARUTI",
        "LT",
        "ULTRACEMCO",
        "BAJAJFINSV",
        "TECHM",
        "ONGC",
    ]
    all_basic_metrics = []
    all_walkforward_metrics = []

    # Step 1: Run basic backtest for all strategies and rank top 10
    logger.info("Running basic backtests for all strategies...")

    for ticker in tickers:
        data = get_data_sync(
            ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
        if data is None or data.empty:
            logger.warning(f"No data for {ticker}. Skipping backtest.")
            continue

        ticker_basic_metrics = {}

        for strategy_name in strategies:
            try:
                # Basic Backtest only
                results, cerebro = run_basic_backtest(
                    data=data,
                    strategy_class=strategy_name,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                )
                analyzer = PerformanceAnalyzer(results[0])
                report = analyzer.generate_full_report()
                bt_report = generate_strategy_report(
                    results, strategy_name, ticker, interval
                )

                if bt_report:
                    key = (ticker, strategy_name)
                    extracted_metrics = extract_report_metrics(bt_report)

                    # Store any parameters that might be in the bt_report
                    if hasattr(results[0], "_params") or "Params" in bt_report:
                        params = getattr(
                            results[0], "_params", bt_report.get("Params", {})
                        )
                        if params:
                            extracted_metrics["best_params"] = params

                    ticker_basic_metrics[key] = {"backtest": extracted_metrics}

            except Exception as e:
                logger.warning(
                    f"Error in basic backtesting {ticker} with {strategy_name}: {e}"
                )
                continue

        # Add ticker's basic metrics to overall collection
        for key, metrics in ticker_basic_metrics.items():
            if key[0] == ticker:
                all_basic_metrics.append(
                    {"Ticker": key[0], "Strategy": key[1], **metrics}
                )

    # Step 2: Create consolidated metrics and rank strategies
    if not all_basic_metrics:
        logger.error("No basic backtest results available for ranking.")
        return

    logger.info("Creating consolidated metrics and ranking strategies...")
    consolidated_df = create_consolidated_metrics(all_basic_metrics)

    # Calculate composite score for ranking (you can adjust weights as needed)
    consolidated_df["Composite_Score"] = (
        consolidated_df.get("Composite_Win_Rate", 0) * 0.4
        + consolidated_df.get("Composite_Sharpe", 0) * 0.6
    )

    # Rank strategies globally across all stocks
    top_strategies_global = consolidated_df.nlargest(10, "Composite_Score")[
        "Strategy"
    ].unique()[:10]

    logger.info(
        f"Top {len(top_strategies_global)} strategies selected: {list(top_strategies_global)}"
    )

    # Step 3: Run walk-forward analysis only on top 10 strategies
    logger.info("Running walk-forward analysis on top 10 strategies...")

    for ticker in tickers:
        data = get_data_sync(
            ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
        if data is None or data.empty:
            logger.warning(f"No data for {ticker}. Skipping walk-forward analysis.")
            continue

        ticker_walkforward_metrics = {}

        for strategy_name in top_strategies_global:
            try:
                strategy = get_strategy(strategy_name)

                # Walk-Forward Analysis using context manager for proper cleanup
                try:
                    with create_walk_forward_analysis(
                        data=data,
                        strategy_name=strategy_name,
                        optimization_params=strategy.optimization_params,
                        optimization_metric="sharpe_ratio",
                        training_ratio=0.6,
                        gap_ratio=0.05,  # Small gap to prevent look-ahead bias
                        testing_ratio=0.15,
                        step_ratio=0.2,
                        n_trials=n_trials,
                        verbose=False,
                    ) as wf:
                        # Run the analysis
                        wf.run_analysis()

                        # Get results
                        window_summary = wf.get_window_summary()
                        overall_metrics = wf.get_overall_metrics()

                        # Process window summary and extract best params
                        best_params_from_windows = {}
                        if (
                            not window_summary.empty
                            and "best_params" in window_summary.columns
                        ):
                            try:
                                # Convert string representations to actual dicts
                                window_summary["best_params"] = window_summary[
                                    "best_params"
                                ].apply(
                                    lambda x: (
                                        ast.literal_eval(x) if isinstance(x, str) else x
                                    )
                                )

                                # Get the best params from the most recent or best performing window
                                latest_window = window_summary.iloc[-1]
                                best_params_from_windows = latest_window.get(
                                    "best_params", {}
                                )

                                logger.info(
                                    f"Extracted best params from walk-forward: {best_params_from_windows}"
                                )

                            except Exception as param_error:
                                logger.warning(
                                    f"Could not convert best_params: {param_error}"
                                )
                                # Try to get params from the last window as backup
                                if not window_summary.empty:
                                    try:
                                        last_params = window_summary.iloc[-1].get(
                                            "best_params", {}
                                        )
                                        if isinstance(last_params, dict):
                                            best_params_from_windows = last_params
                                    except:
                                        pass

                        # Clean up percentage columns
                        percent_cols = [
                            col
                            for col in window_summary.columns
                            if "return" in col.lower()
                            or "drawdown" in col.lower()
                            or "in-sample" in col.lower()
                            or "out-sample" in col.lower()
                        ]
                        for col in percent_cols:
                            if col in window_summary.columns:
                                window_summary[col] = (
                                    window_summary[col]
                                    .astype(str)
                                    .str.replace("%", "", regex=False)
                                )
                                window_summary[col] = pd.to_numeric(
                                    window_summary[col], errors="coerce"
                                )

                        # Create a synthetic report for walk-forward results
                        wf_report = {
                            "Strategy": strategy_name,
                            "Ticker": ticker,
                            "Total Return": overall_metrics.get(
                                "out_sample_avg_return", 0
                            ),
                            "Sharpe Ratio": overall_metrics.get(
                                "out_sample_avg_sharpe", 0
                            ),
                            "Win Rate": overall_metrics.get("win_rate_out_sample", 0)
                            / 100,  # Convert to decimal
                            "Max Drawdown": 0,  # Could calculate from window results if needed
                            "Total Trades": overall_metrics.get("total_windows", 0),
                            "Params": best_params_from_windows,  # Include best params
                        }

                        key = (ticker, strategy_name)
                        extracted_wf_metrics = extract_report_metrics(wf_report)

                        # Ensure best_params is included
                        if best_params_from_windows:
                            extracted_wf_metrics["best_params"] = (
                                best_params_from_windows
                            )

                        ticker_walkforward_metrics[key] = {
                            "walkforward": extracted_wf_metrics
                        }

                        logger.info(
                            f"Completed walk-forward analysis for {ticker} - {strategy_name}"
                        )

                except Exception as wf_error:
                    logger.error(
                        f"Walk-forward analysis failed for {ticker} - {strategy_name}: {wf_error}"
                    )
                    continue

            except Exception as e:
                logger.warning(
                    f"Error in walk-forward analysis setup {ticker} with {strategy_name}: {e}"
                )
                continue

        # Add ticker's walk-forward metrics to overall collection
        for key, metrics in ticker_walkforward_metrics.items():
            if key[0] == ticker:
                all_walkforward_metrics.append(
                    {"Ticker": key[0], "Strategy": key[1], **metrics}
                )

    # Step 4: Combine basic backtest results with walk-forward results for final ranking
    logger.info("Combining results and generating final rankings...")

    # Create a combined metrics structure
    combined_metrics = []

    # First, add all basic backtest results
    for basic_metric in all_basic_metrics:
        ticker = basic_metric["Ticker"]
        strategy = basic_metric["Strategy"]

        # Find corresponding walk-forward result if it exists
        wf_metric = None
        for wf_metric_item in all_walkforward_metrics:
            if (
                wf_metric_item["Ticker"] == ticker
                and wf_metric_item["Strategy"] == strategy
            ):
                wf_metric = wf_metric_item
                break

        combined_metric = {
            "Ticker": ticker,
            "Strategy": strategy,
            "backtest": basic_metric.get("backtest", {}),
        }

        if wf_metric:
            combined_metric["walkforward"] = wf_metric.get("walkforward", {})

        combined_metrics.append(combined_metric)

    if combined_metrics:
        # Create consolidated metrics for final ranking
        final_consolidated_df = create_consolidated_metrics(combined_metrics)

        # Generate top 8 strategies per stock from the final results
        top_strategies_per_stock = {}
        for ticker in final_consolidated_df["Ticker"].unique():
            stock_df = final_consolidated_df[final_consolidated_df["Ticker"] == ticker]

            # Sort by composite metrics, but prioritize walk-forward results if available
            sort_columns = []
            if "Walkforward_Sharpe" in stock_df.columns:
                sort_columns.extend(["Walkforward_Sharpe", "Walkforward_Win_Rate"])
            sort_columns.extend(["Composite_Sharpe", "Composite_Win_Rate"])

            # Remove duplicates and keep valid columns
            sort_columns = [col for col in sort_columns if col in stock_df.columns]
            if not sort_columns:
                sort_columns = ["Composite_Sharpe", "Composite_Win_Rate"]

            # Get top 8 strategies instead of 5
            top_strategies = stock_df.sort_values(
                by=sort_columns, ascending=False
            ).head(8)[["Strategy"] + sort_columns]

            top_strategies_list = []
            for _, row in top_strategies.iterrows():
                strategy_name = row["Strategy"]
                best_params = extract_best_params_with_fallback(
                    strategy_name, combined_metrics, ticker
                )

                strategy_data = {
                    "Strategy": strategy_name,
                    "Best_Parameters": best_params,
                }

                # Add available metrics
                for col in sort_columns:
                    if col in row:
                        strategy_data[col] = row[col]

                top_strategies_list.append(strategy_data)

            top_strategies_per_stock[ticker] = top_strategies_list

        # Create final DataFrame
        top_strategies_df = pd.DataFrame(
            [
                {"Ticker": ticker, **strategy}
                for ticker, strategies in top_strategies_per_stock.items()
                for strategy in strategies
            ]
        )

        # Convert Best_Parameters to string for CSV export, but ensure they're not empty
        if not top_strategies_df.empty:

            def format_params(params):
                if isinstance(params, dict) and params:
                    return str(params)
                elif isinstance(params, str) and params.strip() and params != "{}":
                    return params
                else:
                    return "No parameters available"

            top_strategies_df["Best_Parameters"] = top_strategies_df[
                "Best_Parameters"
            ].apply(format_params)

            # Save to CSV
            top_strategies_df.to_csv(
                "selected_stocks_strategies_with_walkforward.csv", index=False
            )

            # Log summary statistics
            param_stats = top_strategies_df["Best_Parameters"].value_counts()
            logger.info(f"Parameter statistics: {param_stats}")

            logger.info(
                "Final backtest results with walk-forward analysis saved to 'selected_stocks_strategies.csv'"
            )
            logger.info(f"Total strategies analyzed: {len(strategies)}")
            logger.info(
                f"Top strategies selected for walk-forward: {len(top_strategies_global)}"
            )
            logger.info(
                f"Final results include {len(top_strategies_df)} strategy-stock combinations"
            )
            logger.info(f"Top 8 strategies saved per stock")
        else:
            logger.warning("No final results to save - empty DataFrame")
    else:
        logger.error("No combined metrics available for final ranking")


@app.task(bind=True)
def run_intraday_stock_filter(self, csv_file="ind_nifty50list.csv"):
    try:
        logging.info(f"Task started: {csv_file}")
        filter = IntradayStockFilter()
        selected_stocks = filter.select_stocks(csv_file="ind_nifty50list.csv")
        strategies = list(STRATEGY_REGISTRY.keys())

        # Run the complete backtests
        run_complete_backtests(selected_stocks=selected_stocks, strategies=strategies)

        logging.info(f"Task completed. {len(selected_stocks)} stocks selected.")
        return f"Successfully processed {len(selected_stocks)} stocks with {len(strategies)} strategies"

    except Exception as e:
        logging.exception("Task failed")
        raise self.retry(exc=e, countdown=10, max_retries=1)
