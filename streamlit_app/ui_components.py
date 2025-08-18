"""
UI components and layout functions for the Streamlit application.
"""

import streamlit as st
from datetime import datetime, timedelta
from comprehensive_backtesting.registry import STRATEGY_REGISTRY
from .config import (
    PAGE_CONFIG,
    CUSTOM_CSS,
    AVAILABLE_ANALYZERS,
    DEFAULT_ANALYZERS,
    OPTIMIZATION_PARAMETERS,
    TIMEFRAMES,
    ANALYSIS_TYPES,
    TICKER_INPUT_METHODS,
    get_default_dates,
)
from .utils import get_available_tickers, validate_ticker_format, save_tickers_to_file
from comprehensive_backtesting.utils import DEFAULT_TICKERS


def setup_page_config():
    """Set up Streamlit page configuration."""
    st.set_page_config(**PAGE_CONFIG)


def render_page_header():
    """Render the main page header and description."""
    st.title("📈 Comprehensive Backtesting Framework")
    st.markdown(
        """
    **Advanced Trading Strategy Analysis Platform**
    
    This framework provides comprehensive backtesting capabilities with:
    - 📊 **Enhanced Candlestick Charts** with trade markers and volume analysis
    - 📋 **Detailed Trade Tables** with comprehensive trade information
    - ⏰ **Best Trading Times Analysis** with hourly, daily, and monthly breakdowns
    - 🎯 **Parameter Optimization** with visual parameter importance analysis
    - 🗺️ **Optimization Landscape** visualization with contour plots
    - 📈 **Strategy Comparison** tables showing improvement metrics
    """
    )


def add_custom_css():
    """Add custom CSS for better styling."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_ticker_management():
    """Render the ticker management section in the sidebar."""
    with st.sidebar.expander("Ticker Management", expanded=False):
        st.write("**Manage Ticker List**")

        # Show current tickers
        current_tickers = get_available_tickers()
        st.write(f"Current tickers ({len(current_tickers)}):")
        st.write(
            ", ".join(current_tickers[:10])
            + ("..." if len(current_tickers) > 10 else "")
        )

        # Add new ticker
        new_ticker = st.text_input(
            "Add New Ticker", help="Enter ticker symbol to add to the list"
        )
        if st.button("Add Ticker") and new_ticker:
            new_ticker = new_ticker.strip().upper()
            is_valid, message = validate_ticker_format(new_ticker)
            if not is_valid:
                st.error(message)
            elif new_ticker not in current_tickers:
                updated_tickers = current_tickers + [new_ticker]
                if save_tickers_to_file(updated_tickers):
                    st.success(f"Added {new_ticker} to ticker list")
                    st.experimental_rerun()
                else:
                    st.error("Failed to save ticker list")
            else:
                st.warning(f"{new_ticker} already exists in the list")

        # Remove ticker
        if current_tickers:
            ticker_to_remove = st.selectbox("Remove Ticker", [""] + current_tickers)
            if st.button("Remove Ticker") and ticker_to_remove:
                updated_tickers = [t for t in current_tickers if t != ticker_to_remove]
                if save_tickers_to_file(updated_tickers):
                    st.success(f"Removed {ticker_to_remove} from ticker list")
                    st.experimental_rerun()
                else:
                    st.error("Failed to save ticker list")

        # Reset to defaults
        if st.button("Reset to Default Tickers"):
            if save_tickers_to_file(DEFAULT_TICKERS):
                st.success("Reset to default ticker list")
                st.experimental_rerun()
            else:
                st.error("Failed to reset ticker list")

        # Export ticker list
        if st.button("Export Ticker List"):
            ticker_text = "\n".join(current_tickers)
            st.download_button(
                label="Download tickers.txt",
                data=ticker_text,
                file_name="tickers.txt",
                mime="text/plain",
            )

        # Import ticker list
        uploaded_file = st.file_uploader("Import Ticker List", type=["txt"])
        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode("utf-8")
                imported_tickers = [
                    line.strip().upper() for line in content.split("\n") if line.strip()
                ]

                # Validate all tickers
                valid_tickers = []
                invalid_tickers = []
                for ticker in imported_tickers:
                    is_valid, _ = validate_ticker_format(ticker)
                    if is_valid:
                        valid_tickers.append(ticker)
                    else:
                        invalid_tickers.append(ticker)

                if valid_tickers:
                    if save_tickers_to_file(valid_tickers):
                        st.success(f"Imported {len(valid_tickers)} valid tickers")
                        if invalid_tickers:
                            st.warning(
                                f"Skipped {len(invalid_tickers)} invalid tickers: {', '.join(invalid_tickers[:5])}"
                            )
                        st.experimental_rerun()
                    else:
                        st.error("Failed to save imported tickers")
                else:
                    st.error("No valid tickers found in the uploaded file")
            except Exception as e:
                st.error(f"Error importing ticker list: {str(e)}")


def render_sidebar():
    """Render the sidebar configuration and collect user inputs."""
    st.sidebar.header("Backtest Configuration")
    render_ticker_management()

    # Strategy selection
    strategy_names = list(STRATEGY_REGISTRY.keys())
    strategy_options = ["Select All"] + strategy_names
    selected_strategy = st.sidebar.multiselect(
        "Select Strategy", strategy_options, default=["Select All"]
    )
    if "Select All" in selected_strategy:
        selected_strategy = strategy_names

    # Date inputs with validation
    start_date_default, end_date_default = get_default_dates()
    start_date = st.sidebar.date_input(
        "Start Date",
        value=start_date_default,
        max_value=end_date_default - timedelta(days=1),
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=end_date_default,
        min_value=start_date + timedelta(days=1),
        max_value=datetime.today().date(),
    )

    # Ticker input with validation - dynamic ticker selection
    ticker_input_method = st.sidebar.radio("Ticker Input Method", TICKER_INPUT_METHODS)

    if ticker_input_method == "Select from List":
        available_tickers = get_available_tickers()
        ticker_options = ["Select All"] + available_tickers
        ticker = st.sidebar.multiselect(
            "Ticker Symbol", ticker_options, default=["Select All"]
        )
        if "Select All" in ticker:
            ticker = available_tickers
    else:
        ticker = (
            st.sidebar.text_input(
                "Custom Ticker Symbol",
                value="",
                help="Enter ticker symbol (e.g., AAPL, GOOGL, RELIANCE.NS)",
            )
            .strip()
            .upper()
        )

    # Analysis type
    analysis_type = st.sidebar.selectbox("Analysis Type", ANALYSIS_TYPES)

    if analysis_type in ["Optimization", "Walk-Forward", "Complete Backtest"]:
        n_trials = st.sidebar.slider(
            "Number of Trials",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Number of optimization trials to run (increases in steps of 10)",
        )
        optimization_parameters = st.sidebar.selectbox(
            "Optimization Parameters",
            OPTIMIZATION_PARAMETERS,
            help="Choose whether to optimize all parameters or only selected ones",
        )
    else:
        n_trials = 20
        optimization_parameters = "total_return"

    # Timeframe selection
    timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES)

    # Analyzer selection
    selected_analyzers = st.sidebar.multiselect(
        "Select Analyzers",
        list(AVAILABLE_ANALYZERS.keys()),
        default=DEFAULT_ANALYZERS,
    )

    return {
        "selected_strategy": selected_strategy,
        "start_date": start_date,
        "end_date": end_date,
        "ticker": ticker,
        "ticker_input_method": ticker_input_method,
        "analysis_type": analysis_type,
        "n_trials": n_trials,
        "timeframe": timeframe,
        "selected_analyzers": selected_analyzers,
        "available_analyzers": AVAILABLE_ANALYZERS,
        "optimization_parameters": optimization_parameters,
    }


def validate_inputs(params):
    """Validate user inputs before running analysis."""
    errors = []

    # Validate date range
    if params["start_date"] >= params["end_date"]:
        errors.append("End date must be after start date.")

    # Validate ticker
    if not params["ticker"]:
        errors.append("Ticker symbol cannot be empty.")
    elif params["ticker_input_method"] == "Enter Custom Ticker":
        is_valid, message = validate_ticker_format(params["ticker"])
        if not is_valid:
            errors.append(message)

    # Validate analyzer dependencies
    if (
        "SortinoRatio" in params["selected_analyzers"]
        and "TimeReturn" not in params["selected_analyzers"]
    ):
        errors.append("SortinoRatio requires TimeReturn analyzer.")

    return errors
