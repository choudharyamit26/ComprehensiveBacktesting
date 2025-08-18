"""
Main Streamlit Application Entry Point

This is the refactored main application file that uses the modular components
from the streamlit_app package.
"""

from streamlit_app.ui_components import (
    setup_page_config,
    render_page_header,
    add_custom_css,
    render_sidebar,
)
from streamlit_app.analysis_runners import run_analysis
import streamlit as st


def main():
    """Main application function."""
    # Set up the page
    setup_page_config()
    render_page_header()
    add_custom_css()

    # Render sidebar and get parameters
    params = render_sidebar()

    # Add run button
    if st.sidebar.button("Run Analysis"):
        run_analysis(params)


if __name__ == "__main__":
    main()
