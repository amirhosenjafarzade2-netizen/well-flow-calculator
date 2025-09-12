# app.py
# Main entry point for the Well Pressure and Depth Calculator Streamlit application

import streamlit as st
from data_loader import load_reference_data
from utils import setup_logging
from config import INTERPOLATION_RANGES, PRODUCTION_RATES
from ui import run_p2_finder, run_natural_flow_finder, run_glr_graph_drawer  # moved imports here

# Initialize logger
logger = setup_logging()


def main():
    """
    Main function to run the Streamlit application.
    Initializes reference data and handles mode selection.
    """
    st.title("Well Pressure and Depth Calculator")

    # Load reference data into session state (only once)
    if "REFERENCE_DATA" not in st.session_state:
        logger.info("Loading reference data...")
        reference_data = load_reference_data()
        if reference_data is None:
            st.error("Failed to initialize application: Unable to load reference data. "
                     "Please check the Excel file or internet connection.")
            logger.error("Application initialization failed due to missing reference data.")
            return
        st.session_state.REFERENCE_DATA = reference_data
        logger.info("Reference data loaded successfully.")

    # Mode selection
    st.header("Select Calculation Mode")
    mode_options = ["p2 Finder", "Natural Flow Finder", "GLR Graph Drawer"]
    mode = st.selectbox("Choose a mode:", mode_options, key="mode_select")

    # Create tabs for cleaner layout
    tabs = st.tabs(["Calculation", "Help"])

    with tabs[0]:
        if mode == "p2 Finder":
            run_p2_finder(st.session_state.REFERENCE_DATA, INTERPOLATION_RANGES, PRODUCTION_RATES)
        elif mode == "Natural Flow Finder":
            run_natural_flow_finder(st.session_state.REFERENCE_DATA, INTERPOLATION_RANGES, PRODUCTION_RATES)
        elif mode == "GLR Graph Drawer":
            run_glr_graph_drawer(st.session_state.REFERENCE_DATA, INTERPOLATION_RANGES, PRODUCTION_RATES)

    with tabs[1]:
        st.write("### Help")
        st.write("""
        **Well Pressure and Depth Calculator**
        - **p2 Finder**: Calculate wellhead (p1, y1) and bottomhole (p2, y2) pressures and depths.
        - **Natural Flow Finder**: Find the natural flow rate by intersecting TPR and IPR curves.
        - **GLR Graph Drawer**: Plot pressure vs. depth curves for all GLRs at a given conduit size and production rate.
        
        **Inputs**:
        - Conduit size: 2.875 or 3.5 inches
        - Production rate: 50, 100, 200, 400, or 600 stb/day
        - GLR: Gas-Liquid Ratio (varies by conduit size and production rate)
        - Depth (D): Well length in feet (y1 + D ≤ 31000 ft)
        - Pressures: p1, pwh, pr (0 to 4000 psi)
        
        **Contact**: For issues, check the GitHub repository or contact support.
        """)


if __name__ == "__main__":
    main()
