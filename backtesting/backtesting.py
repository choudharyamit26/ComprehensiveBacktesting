import backtrader as bt

from stratgies.ema_rsi import EMARSI
from .data import get_data
from .parameter_optimization import optimize_strategy
from .reports import PerformanceAnalyzer, compare_strategies
from .validation import ValidationAnalyzer, run_validation_analysis


def run_backtest(
    strategy_class,
    ticker="AAPL",
    start_date="2022-01-01",
    end_date="2025-06-01",
    initial_cash=100000.0,
    commission=0.001,
    **strategy_params,
):
    """
    Run a backtest with specified parameters.

    Parameters:
    strategy_class: The trading strategy class
    ticker (str): Stock ticker symbol
    start_date (str): Start date for backtest
    end_date (str): End date for backtest
    initial_cash (float): Initial cash amount
    commission (float): Commission rate
    **strategy_params: Strategy-specific parameters

    Returns:
    tuple: (results, cerebro) - Results and cerebro instance
    """

    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Get data
    data_df = get_data(ticker, start_date, end_date)

    # Create Backtrader data feed with explicit datetime column mapping
    # The DataFrame should have Date as index and OHLCV as columns
    data = bt.feeds.PandasData(
        dataname=data_df,
        datetime=None,  # Use index as datetime
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        openinterest=None,
    )

    # Add strategy with parameters
    cerebro.addstrategy(strategy_class, **strategy_params)

    # Set broker parameters
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    # Add comprehensive analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Calmar, _name="calmar")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annualreturn")
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="transactions")
    cerebro.addanalyzer(bt.analyzers.PositionsValue, _name="positionsvalue")
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")

    # Add data to cerebro
    cerebro.adddata(data)

    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}")

    try:
        # Run backtest
        results = cerebro.run()
        print(f"Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
        return results, cerebro

    except Exception as e:
        print(f"Error during backtesting: {str(e)}")
        print("This might be due to insufficient data or strategy issues.")
        raise


def run_basic_backtest(ticker, start_date, end_date):
    """Run a basic backtest with default parameters."""
    print(f"\n=== BASIC BACKTEST: {ticker} ===")

    try:
        results, cerebro = run_backtest(
            strategy_class=EMARSI,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        analyzer = PerformanceAnalyzer(results)
        analyzer.print_report()

        return results, cerebro

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Basic backtest failed: {str(e)}")
        raise


def run_parameter_optimization(ticker, start_date, end_date):
    """Run parameter optimization analysis."""
    print(f"\n=== PARAMETER OPTIMIZATION: {ticker} ===")

    n_trials = int(
        input("Enter number of optimization trials (default: 50): ").strip() or "50"
    )

    print(f"Running optimization with {n_trials} trials...")
    try:
        optimization_results = optimize_strategy(
            strategy_class=EMARSI,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            n_trials=n_trials,
        )

        print("\nOptimized Strategy Performance:")
        analyzer_optimized = PerformanceAnalyzer(
            optimization_results["cerebro"].runstrats[0]
        )
        analyzer_optimized.print_report()

        return optimization_results

    except ImportError:
        print(
            "Optimization module not available. Running basic parameter comparison instead."
        )
        return run_basic_comparison_analysis(ticker, start_date, end_date)
    except Exception as e:
        print(f"Parameter optimization failed: {str(e)}")
        raise


def run_insample_outsample_analysis(ticker, start_date, end_date):
    """Run in-sample/out-of-sample validation analysis."""
    print(f"\n=== IN-SAMPLE / OUT-OF-SAMPLE ANALYSIS: {ticker} ===")

    try:
        validation_analyzer = ValidationAnalyzer(strategy_class=EMARSI, ticker=ticker)

        results = validation_analyzer.in_sample_out_sample_analysis(
            start_date=start_date, end_date=end_date
        )

        # Print in-sample report
        print("\nIn-Sample Performance:", results)
        print("-" * 60)
        if "in_sample_performance" in results:
            summary = results["in_sample_performance"]["summary"]
            print(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
            print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.3f}")
            print(f"Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
        else:
            print("Error: In-sample performance summary not available")

        # Print out-of-sample report
        print("\nOut-of-Sample Performance:", results)
        print("-" * 60)
        if "summary" in results["out_sample_performance"]:
            summary = results["out_sample_performance"]["summary"]
            print(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
            print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.3f}")
            print(f"Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%")
        else:
            print("Error: Out-of-sample performance summary not available")

        # Print validation metrics using the correct key
        print("\nValidation Summary:")
        degradation = results.get("performance_degradation", {})
        print(f"Return degradation: {degradation.get('return_degradation', 0):.2f}%")
        print(f"Sharpe degradation: {degradation.get('sharpe_degradation', 0):.3f}")

        return results

    except ImportError:
        print("ValidationAnalyzer not available. Running basic comparison instead.")
        return run_basic_comparison_analysis(ticker, start_date, end_date)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"In-sample/out-of-sample analysis failed: {str(e)}")
        return run_basic_comparison_analysis(ticker, start_date, end_date)


def run_walkforward_analysis(ticker, start_date, end_date):
    """Run walk-forward analysis."""
    print(f"\n=== WALK-FORWARD ANALYSIS: {ticker} ===")

    try:
        validation_analyzer = ValidationAnalyzer(
            strategy_class=EMARSI,
            ticker=ticker
        )

        window_months = int(
            input("Enter optimization window in months (default: 12): ").strip() or "12"
        )
        step_months = int(
            input("Enter step size in months (default: 3): ").strip() or "3"
        )

        results = validation_analyzer.walk_forward_analysis(
            optimization_window_months=window_months, step_months=step_months
        )

        print(f"\nWalk-Forward Analysis Results:")
        print(f"Number of periods: {len(results['period_results'])}")
        print(
            f"Average out-of-sample Sharpe: {results['summary']['avg_oos_sharpe']:.3f}"
        )
        print(
            f"Average out-of-sample return: {results['summary']['avg_oos_return']:.2f}%"
        )
        print(f"Consistency score: {results['summary']['consistency_score']:.2f}")

        return results

    except ImportError:
        print(
            "ValidationAnalyzer not available. Running parameter optimization instead."
        )
        return run_parameter_optimization(ticker, start_date, end_date)
    except Exception as e:
        print(f"Walk-forward analysis failed: {str(e)}")
        return run_parameter_optimization(ticker, start_date, end_date)


def run_comprehensive_validation(ticker, start_date, end_date):
    """Run comprehensive validation analysis."""
    print(f"\n=== COMPREHENSIVE VALIDATION ANALYSIS: {ticker} ===")

    try:
        results = run_validation_analysis(
            strategy_class=EMARSI,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        print("\nComprehensive Validation Complete!")
        print("Check generated reports for detailed analysis.")

        return results

    except ImportError:
        print("Full validation analysis not available. Running available analyses...")

        # Run what we can
        basic_results, _ = run_basic_backtest(ticker, start_date, end_date)

        try:
            opt_results = run_parameter_optimization(ticker, start_date, end_date)
        except:
            opt_results = None
            print(
                "Parameter optimization also failed, continuing with basic results only."
            )

        return {"basic_results": basic_results, "optimization_results": opt_results}
    except Exception as e:
        print(f"Comprehensive validation failed: {str(e)}")
        return run_basic_backtest(ticker, start_date, end_date)


def run_basic_comparison_analysis(ticker, start_date, end_date):
    """Run basic strategy comparison when validation module is not available."""
    print(f"\n=== STRATEGY COMPARISON ANALYSIS: {ticker} ===")

    # Test different parameter sets
    parameter_sets = {
        "Conservative": {
            "slow_ema_period": 21,
            "fast_ema_period": 50,
            "rsi_period": 14,
            "rsi_upper": 70,
            "rsi_lower": 30,
        },
        "Aggressive": {
            "slow_ema_period": 8,
            "fast_ema_period": 21,
            "rsi_period": 10,
            "rsi_upper": 80,
            "rsi_lower": 20,
        },
        "Balanced": {
            "slow_ema_period": 12,
            "fast_ema_period": 26,
            "rsi_period": 14,
            "rsi_upper": 75,
            "rsi_lower": 25,
        },
    }

    results_comparison = {}

    for name, params in parameter_sets.items():
        print(f"\nRunning {name} strategy...")
        try:
            results, _ = run_backtest(
                strategy_class=EMARSI,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                **params,
            )
            results_comparison[name] = results
        except Exception as e:
            print(f"Failed to run {name} strategy: {str(e)}")
            continue

    # Compare strategies if we have results
    if results_comparison:
        try:
            comparison_df = compare_strategies(results_comparison)
            if not comparison_df.empty:
                print("\nStrategy Comparison:")
                print(comparison_df.round(3))
        except Exception as e:
            print(f"Strategy comparison failed: {str(e)}")

    return results_comparison


def run_full_demo(ticker, start_date, end_date):
    """Run the complete demonstration of all available analyses."""
    print(f"\n=== FULL DEMO: {ticker} ===")

    results = {}

    # 1. Basic backtest
    print("\n" + "=" * 60)
    try:
        basic_results, basic_cerebro = run_basic_backtest(ticker, start_date, end_date)
        results["basic"] = basic_results
        results["basic_cerebro"] = basic_cerebro
    except Exception as e:
        print(f"Basic backtest failed: {str(e)}")
        return results

    # 2. Parameter optimization
    print("\n" + "=" * 60)
    try:
        opt_results = run_parameter_optimization(ticker, start_date, end_date)
        results["optimization"] = opt_results
    except Exception as e:
        print(f"Parameter optimization failed: {str(e)}")

    # 3. Validation analysis
    print("\n" + "=" * 60)
    try:
        validation_results = run_insample_outsample_analysis(
            ticker, start_date, end_date
        )
        results["validation"] = validation_results
    except Exception as e:
        print(f"Validation analysis failed: {str(e)}")

    # 4. Strategy comparison
    print("\n" + "=" * 60)
    try:
        comparison_results = run_basic_comparison_analysis(ticker, start_date, end_date)
        results["comparison"] = comparison_results
    except Exception as e:
        print(f"Strategy comparison failed: {str(e)}")

    # 5. Generate comprehensive report
    print("\n" + "=" * 60)
    print("Generating comprehensive reports...")

    try:
        basic_analyzer = PerformanceAnalyzer(basic_results)
        basic_analyzer.save_report_to_file(f"{ticker}_basic_report.json")
        print(f"Basic report saved: {ticker}_basic_report.json")

        if (
            "optimization" in results
            and results["optimization"]
            and "cerebro" in results["optimization"]
        ):
            opt_analyzer = PerformanceAnalyzer(
                results["optimization"]["cerebro"].runstrats[0]
            )
            opt_analyzer.save_report_to_file(f"{ticker}_optimized_report.json")
            print(f"Optimized report saved: {ticker}_optimized_report.json")

    except Exception as e:
        print(f"Report generation failed: {str(e)}")

    # 6. Plot option
    try:
        plot_choice = input("\nPlot results? (y/n): ").lower().strip()
        if plot_choice == "y":
            print("Plotting basic strategy results...")
            basic_cerebro.plot(style="candlestick", barup="green", bardown="red")

            if (
                "optimization" in results
                and results["optimization"]
                and "cerebro" in results["optimization"]
            ):
                print("Plotting optimized strategy results...")
                results["optimization"]["cerebro"].plot(
                    style="candlestick", barup="green", bardown="red"
                )
    except Exception as e:
        print(f"Plotting failed: {str(e)}")

    return results


def main():
    """Enhanced main function with comprehensive analysis options."""

    print("=== ENHANCED BACKTESTING FRAMEWORK ===\n")

    # Get user choice for analysis type
    print("Select analysis type:")
    print("1. Basic backtest")
    print("2. Parameter optimization")
    print("3. In-sample / Out-of-sample analysis")
    print("4. Walk-forward analysis")
    print("5. Comprehensive validation analysis")
    print("6. Full demo (all analyses)")

    choice = input("\nEnter your choice (1-6): ").strip()

    # Common parameters
    ticker = input("Enter ticker symbol (default: AAPL): ").strip() or "AAPL"
    start_date = (
        input("Enter start date (default: 2022-01-01): ").strip() or "2022-01-01"
    )
    end_date = input("Enter end date (default: 2025-06-01): ").strip() or "2025-06-01"

    try:
        if choice == "1":
            run_basic_backtest(ticker, start_date, end_date)

        elif choice == "2":
            run_parameter_optimization(ticker, start_date, end_date)

        elif choice == "3":
            run_insample_outsample_analysis(ticker, start_date, end_date)

        elif choice == "4":
            run_walkforward_analysis(ticker, start_date, end_date)

        elif choice == "5":
            run_comprehensive_validation(ticker, start_date, end_date)

        elif choice == "6":
            run_full_demo(ticker, start_date, end_date)

        else:
            print("Invalid choice. Running basic backtest...")
            run_basic_backtest(ticker, start_date, end_date)

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("This might be due to:")
        print("- Missing strategy module (strategies.ema_rsi)")
        print("- Missing dependencies")
        print("- Invalid ticker symbol")
        print("- Insufficient data")
        print("\nTrying basic fallback...")

        try:
            # Simple fallback test
            from .data import get_data, validate_data

            test_data = get_data(ticker, start_date, end_date)
            if validate_data(test_data):
                print("Data is valid. The issue might be with the strategy module.")
            else:
                print("Data validation failed.")
        except Exception as data_error:
            print(f"Data fetch also failed: {str(data_error)}")

    print("\n=== ANALYSIS COMPLETE ===")


def quick_test():
    """Quick test function for rapid development."""
    print("=== QUICK TEST ===")

    ticker = input("Enter ticker (default: AAPL): ").strip() or "AAPL"

    try:
        results, cerebro = run_backtest(
            strategy_class=EMARSI,
            ticker=ticker,
            start_date="2024-01-01",
            end_date="2025-06-01",
        )

        analyzer = PerformanceAnalyzer(results)
        analyzer.print_report()

        plot_choice = input("\nPlot results? (y/n): ").lower().strip()
        if plot_choice == "y":
            cerebro.plot(style="candlestick", barup="green", bardown="red")

    except Exception as e:
        print(f"Quick test failed: {str(e)}")

        # Test just the data loading
        try:
            from .data import get_data, validate_data, preview_data

            print("Testing data loading...")
            df = preview_data(ticker, "2024-01-01", "2024-12-31", rows=3)
            if df is not None:
                is_valid = validate_data(df)
                print(f"Data validation: {'PASSED' if is_valid else 'FAILED'}")
        except Exception as data_error:
            print(f"Data test also failed: {str(data_error)}")


if __name__ == "__main__":
    # Main execution logic
    print("Welcome to the Enhanced Backtesting Framework!")

    # Get user choice for execution mode
    mode_choice = input(
        "\nChoose mode:\n1. Full interactive menu\n2. Quick test\n3. Basic example\nEnter choice (1-3): "
    ).strip()

    try:
        if mode_choice == "1":
            main()
        elif mode_choice == "2":
            quick_test()
        elif mode_choice == "3":
            # Run basic example
            print("Running basic example...")

            try:
                cerebro = bt.Cerebro()

                # Get data
                from .data import get_data

                data_df = get_data(
                    ticker="AAPL", start_date="2022-01-01", end_date="2025-06-01"
                )

                # Create data feed
                data = bt.feeds.PandasData(
                    dataname=data_df,
                    datetime=None,
                    open="Open",
                    high="High",
                    low="Low",
                    close="Close",
                    volume="Volume",
                    openinterest=None,
                )

                # Add strategy and configure
                cerebro.addstrategy(EMARSI)
                cerebro.broker.setcash(100000.0)
                cerebro.broker.setcommission(commission=0.001)

                # Add comprehensive analyzers
                cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
                cerebro.addanalyzer(bt.analyzers.Calmar, _name="calmar")
                cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
                cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
                cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
                cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
                cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")
                cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
                cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annualreturn")
                cerebro.addanalyzer(bt.analyzers.Transactions, _name="transactions")
                cerebro.addanalyzer(bt.analyzers.PositionsValue, _name="positionsvalue")

                cerebro.adddata(data)

                print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
                results = cerebro.run()
                print(f"Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}")

                # Generate report
                analyzer = PerformanceAnalyzer(results)
                analyzer.print_report()

                # Plot option
                plot_choice = input("\nPlot results? (y/n): ").lower().strip()
                if plot_choice == "y":
                    cerebro.plot(style="candlestick", barup="green", bardown="red")

            except ImportError as ie:
                print(f"Import error: {str(ie)}")
                print("Make sure all required modules are available.")
            except Exception as e:
                print(f"Basic example failed: {str(e)}")

                # Even more basic test
                try:
                    from .data import preview_data

                    print("Testing data loading only...")
                    df = preview_data("AAPL", "2024-01-01", "2024-12-31")
                except Exception as data_error:
                    print(f"Data loading test failed: {str(data_error)}")
        else:
            print("Invalid choice. Running quick test...")
            quick_test()

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\nUnexpected error: {str(e)}")
        print("Please check your dependencies and try again.")

    print("\nThank you for using the Enhanced Backtesting Framework!")
