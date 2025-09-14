# random_point_generator.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_reference_data
from config import PRODUCTION_RATES, INTERPOLATION_RANGES
from utils import export_plot_to_png, setup_logging
import xlsxwriter

logger = setup_logging()

def run_random_point_generator():
    """UI for generating and visualizing random well performance data using reference Excel data."""
    st.subheader("Random Point Generator")
    
    # Load reference data
    if "REFERENCE_DATA" not in st.session_state:
        logger.info("Loading reference data for Random Point Generator...")
        reference_data = load_reference_data()
        if reference_data is None or reference_data.empty:
            st.error("Failed to load reference data. Please check the Excel file.")
            logger.error("Failed to load reference data for Random Point Generator.")
            return
        st.session_state.REFERENCE_DATA = reference_data
    else:
        reference_data = st.session_state.REFERENCE_DATA

    # Initialize session state
    if 'random_point_inputs' not in st.session_state:
        st.session_state.random_point_inputs = {
            'num_points': 100,
            'conduit_size': 2.875
        }

    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        num_points = st.number_input(
            "Number of Points to Generate:",
            min_value=1,
            max_value=1000,
            value=st.session_state.random_point_inputs['num_points'],
            step=10,
            help="Number of random data points to generate."
        )
        st.session_state.random_point_inputs['num_points'] = num_points

    with col2:
        conduit_size = st.selectbox(
            "Conduit Size (in):",
            [2.875, 3.5],
            index=[2.875, 3.5].index(st.session_state.random_point_inputs['conduit_size']),
            help="Select conduit size (2.875 or 3.5 inches)."
        )
        st.session_state.random_point_inputs['conduit_size'] = conduit_size

    generate = st.button("Generate Random Points")

    if generate:
        with st.spinner("Generating random points..."):
            try:
                # Get valid ranges from REFERENCE_DATA
                valid_production_rates = PRODUCTION_RATES  # [50, 100, 200, 300, 400, 500, 600]
                valid_glr_ranges = INTERPOLATION_RANGES.get((conduit_size, valid_production_rates[0]), [100, 1000])
                min_glr, max_glr = valid_glr_ranges

                # Generate random data
                data = {
                    'conduit_size': [conduit_size] * num_points,
                    'production_rate': np.random.choice(valid_production_rates, size=num_points),
                    'glr': np.random.uniform(min_glr, max_glr, size=num_points),
                    'pressure': np.random.uniform(0, 4000, size=num_points),  # Based on ui.py constraints
                    'depth': np.random.uniform(0, 31000, size=num_points)     # Based on ui.py constraints
                }
                df = pd.DataFrame(data)

                # Store results in session state
                st.session_state.random_point_results = df

                # Display data
                st.subheader("Generated Data")
                st.dataframe(df)

                # Plot data
                fig, ax = plt.subplots()
                scatter = ax.scatter(df['production_rate'], df['pressure'], c=df['glr'], cmap='viridis', alpha=0.6)
                ax.set_xlabel("Production Rate (stb/day)")
                ax.set_ylabel("Pressure (psi)")
                ax.set_title(f"Random Well Performance Data (Conduit Size: {conduit_size} in)")
                plt.colorbar(scatter, label="GLR (scf/stb)")
                st.pyplot(fig)

                # Export plot
                try:
                    st.download_button(
                        label="Download Plot as PNG",
                        data=export_plot_to_png(fig),
                        file_name="random_points_plot.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Failed to export plot: {str(e)}")
                    logger.error(f"Plot export failed: {str(e)}")

                # Export data to Excel
                try:
                    output = pd.ExcelWriter('random_points.xlsx', engine='xlsxwriter')
                    df.to_excel(output, sheet_name='RandomPoints', index=False)
                    output.close()
                    with open('random_points.xlsx', 'rb') as f:
                        st.download_button(
                            label="Download Data as Excel",
                            data=f,
                            file_name="random_points.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f"Failed to export data to Excel: {str(e)}")
                    logger.error(f"Excel export failed: {str(e)}")

            except Exception as e:
                st.error(f"Failed to generate random points: {str(e)}")
                logger.error(f"Random point generation failed: {str(e)}")

    st.write("**Generation Logs**")
    st.write("Any warnings or informational messages will appear here.")
