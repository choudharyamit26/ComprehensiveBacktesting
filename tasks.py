import ast
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
from app import get_strategy
from celery_app import app
from comprehensive_backtesting.backtesting import (
    run_basic_backtest,
    run_parameter_optimization,
)
from comprehensive_backtesting.data import get_data_sync
from comprehensive_backtesting.registry import STRATEGY_REGISTRY
from comprehensive_backtesting.reports import PerformanceAnalyzer
from comprehensive_backtesting.walk_forward_analysis import WalkForwardAnalysis
from intraday_stock_filter_nifty import IntradayStockFilter
from app import (
    create_consolidated_metrics,
    extract_report_metrics,
    generate_strategy_report,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_complete_backtests(selected_stocks: List[Dict], strategies: List[str]):
    """Run comprehensive backtests for selected stocks"""

    end_date = datetime.today().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    interval = "5m"
    n_trials = 20
    tickers = [stock["Stock"] for stock in selected_stocks]

    all_metrics = []

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

        strategy_metrics = {}

        for strategy_name in strategies:
            try:
                # strategy = get_strategy(strategy_name)

                # Basic Backtest
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
                    strategy_metrics[key] = {
                        "backtest": extract_report_metrics(bt_report)
                    }

                # Parameter Optimization
                opt_results = run_parameter_optimization(
                    data=data,
                    strategy_class=strategy_name,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    n_trials=n_trials,
                    interval=interval,
                )

                analyzer = PerformanceAnalyzer(opt_results["results"][0])
                report = analyzer.generate_full_report()
                opt_report = generate_strategy_report(
                    opt_results["results"], strategy_name, ticker, interval
                )

                if opt_report:
                    params_dict = opt_report.get("Params", {})
                    key = (ticker, strategy_name)
                    if key in strategy_metrics:
                        strategy_metrics[key]["optimization"] = {
                            **extract_report_metrics(opt_report),
                            "best_params": params_dict,
                        }
                    else:
                        strategy_metrics[key] = {
                            "optimization": {
                                **extract_report_metrics(opt_report),
                                "best_params": params_dict,
                            }
                        }
                strategy = get_strategy(strategy_name)

                # Walk-Forward Analysis
                wf = WalkForwardAnalysis(
                    data=data,
                    strategy_class=strategy.__name__,
                    optimization_params=strategy.optimization_params,
                    optimization_metric="sharpe_ratio",
                    training_ratio=0.6,
                    testing_ratio=0.15,
                    step_ratio=0.2,
                    n_trials=n_trials,
                    verbose=False,
                )
                wf.run_analysis()

                window_summary = wf.get_window_summary()

                if not window_summary.empty and "best_params" in window_summary.columns:
                    try:
                        window_summary["best_params"] = window_summary[
                            "best_params"
                        ].apply(
                            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                        )
                    except Exception:
                        logger.warning("Could not convert best_params string to dict")

                percent_cols = [
                    col
                    for col in window_summary.columns
                    if "return" in col.lower()
                    or "drawdown" in col.lower()
                    or "in-sample" in col.lower()
                    or "out-sample" in col.lower()
                ]
                for col in percent_cols:
                    window_summary[col] = (
                        window_summary[col]
                        .astype(str)
                        .str.replace("%", "", regex=False)
                    )
                    window_summary[col] = pd.to_numeric(
                        window_summary[col], errors="coerce"
                    )

                for i, window in enumerate(wf.results):
                    if window.get("valid") and "out_sample_performance" in window:
                        report = generate_strategy_report(
                            window["out_sample_performance"],
                            f"Walk-Forward Window {i+1}",
                            ticker,
                            interval,
                        )
                        if report:
                            key = (ticker, strategy_name)
                            params_dict = report.get("Params", {})
                            if key in strategy_metrics:
                                strategy_metrics[key]["walkforward"] = {
                                    **extract_report_metrics(report),
                                    "best_params": params_dict,
                                }
                            else:
                                strategy_metrics[key] = {
                                    "walkforward": {
                                        **extract_report_metrics(report),
                                        "best_params": params_dict,
                                    }
                                }

            except Exception as e:
                logger.warning(
                    f"Error in backtesting {ticker} with {strategy_name}: {e}"
                )
                continue

        for key, metrics in strategy_metrics.items():
            if key[0] == ticker:
                all_metrics.append({"Ticker": key[0], "Strategy": key[1], **metrics})

    if all_metrics:
        consolidated_df = create_consolidated_metrics(all_metrics)

        format_dict = {}
        for col in consolidated_df.columns:
            if "win_rate" in col.lower():
                format_dict[col] = "{:.2%}"
            elif "composite_win_rate" in col.lower():
                format_dict[col] = "{:.2%}"
            elif "ratio" in col.lower() and "sharpe" in col.lower():
                format_dict[col] = "{:.3f}"
            elif (
                "pct" in col.lower()
                or "return" in col.lower()
                or "degradation" in col.lower()
            ):
                format_dict[col] = "{:.2f}%"
            elif "sharpe" in col.lower():
                format_dict[col] = "{:.3f}"
            elif "factor" in col.lower():
                format_dict[col] = "{:.2f}"

        top_strategies_per_stock = {}
        for ticker in consolidated_df["Ticker"].unique():
            stock_df = consolidated_df[consolidated_df["Ticker"] == ticker]
            top_strategies = stock_df.sort_values(
                by=["Composite_Win_Rate", "Composite_Sharpe"], ascending=False
            ).head(3)[["Strategy", "Composite_Win_Rate", "Composite_Sharpe"]]

            top_strategies_list = []
            for _, row in top_strategies.iterrows():
                strategy_name = row["Strategy"]
                key = (ticker, strategy_name)
                best_params = {}
                if key in strategy_metrics and "walkforward" in strategy_metrics[key]:
                    best_params = strategy_metrics[key]["walkforward"].get(
                        "best_params", {}
                    )
                elif (
                    key in strategy_metrics and "optimization" in strategy_metrics[key]
                ):
                    best_params = strategy_metrics[key]["optimization"].get(
                        "best_params", {}
                    )
                top_strategies_list.append(
                    {
                        "Strategy": strategy_name,
                        "Composite_Win_Rate": row["Composite_Win_Rate"],
                        "Composite_Sharpe": row["Composite_Sharpe"],
                        "Best_Parameters": best_params,
                    }
                )
            top_strategies_per_stock[ticker] = top_strategies_list

        top_strategies_df = pd.DataFrame(
            [
                {"Ticker": ticker, **strategy}
                for ticker, strategies in top_strategies_per_stock.items()
                for strategy in strategies
            ]
        )
        top_strategies_df["Best_Parameters"] = top_strategies_df[
            "Best_Parameters"
        ].apply(lambda x: str(x) if isinstance(x, dict) else x)
        top_strategies_df.to_csv("selected_stocks_strategies.csv", index=False)
        logger.info("Backtest results saved to 'selected_stocks_strategies.csv'")


@app.task(bind=True)
def run_intraday_stock_filter(self, csv_file="ind_nifty50list.csv"):
    try:
        logging.info(f"Task started: {csv_file}")
        filter = IntradayStockFilter()
        selected_stocks = filter.select_stocks(csv_file="ind_nifty50list.csv")
        strategies = list(STRATEGY_REGISTRY.keys())
        filter = run_complete_backtests(
            selected_stocks=selected_stocks, strategies=strategies
        )
        logging.info(f"Task completed. {len(selected_stocks)} stocks selected.")
        return filter
    except Exception as e:
        logging.exception("Task failed")
        raise self.retry(exc=e, countdown=10, max_retries=1)
