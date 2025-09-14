# random_point_generator.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_reference_data
from config import PRODUCTION_RATES, INTERPOLATION_RANGES, COLORS
from utils import export_plot_to_png, setup_logging
import xlsxwriter
import re

logger = setup_logging()

def parse_cell(cell_value):
    """
    Parse a cell value (e.g., '2.875 in 50 stb-day 100 glr') into conduit size, production rate, and GLR.
    Returns (conduit_size, production_rate, glr) or (None, None, None) if parsing fails.
    """
    try:
        cell_value = str(cell_value).strip()
        if not re.match(r'[\d.]+\s*in\s*\d+\s*stb-day\s*\d+\s*glr', cell_value.lower()):
            logger.error(f"Invalid cell format: {cell_value}")
            return None, None, None
        parts = cell_value.split()
        conduit_size = float(parts[0])
        production_rate = float(parts[2])
        glr_str = parts[4].replace('glr', '')
        glr = float(glr_str)
        return conduit_size, production_rate, glr
    except (IndexError, ValueError) as e:
        logger.error(f"Failed to parse cell value: {cell_value}, error: {str(e)}")
        return None, None, None

def calc_y1(conduit_size, production_rate, glr, p1, reference_data):
    """
    Calculate wellhead depth (y1) based on inputs and reference data.
    Returns y1 (float) or None if calculation fails.
    """
    try:
        # Find matching reference data entry
        for entry in reference_data:
            if (entry['conduit_size'] == conduit_size and
                entry['production_rate'] == production_rate and
                entry['glr'] == glr):
                # Placeholder: Assume y1 is derived from coefficients or a simple model
                coefficients = entry['coefficients']
                # Example: y1 = a * p1 + b (simplified, adjust based on actual logic)
                y1 = coefficients['a'] * p1 + coefficients['b']
                return y1
        logger.warning(f"No matching reference data for conduit_size={conduit_size}, production_rate={production_rate}, glr={glr}")
        return None
    except Exception as e:
        logger.error(f"Failed to calculate y1: {str(e)}")
        return None

def solve_p2(conduit_size, production_rate, glr, p1, D, reference_data):
    """
    Solve for bottomhole pressure (p2) given wellhead pressure (p1), depth (D), and reference data.
    Returns p2 (float) or None if calculation fails.
    """
    try:
        # Find matching reference data entry
        for entry in reference_data:
            if (entry['conduit_size'] == conduit_size and
                entry['production_rate'] == production_rate and
                entry['glr'] == glr):
                coefficients = entry['coefficients']
                # Placeholder: Assume p2 is calculated using a polynomial model
                # Example: p2 = a * p1 + b * D + c (simplified, adjust based on actual logic)
                p2 = (coefficients['a'] * p1 +
                      coefficients['b'] * D +
                      coefficients['c'])
                return p2
        logger.warning(f"No matching reference data for conduit_size={conduit_size}, production_rate={production_rate}, glr={glr}")
        return None
    except Exception as e:
        logger.error(f"Failed to solve p2: {str(e)}")
        return None

def run_random_point_generator():
    """UI for generating and visualizing random well performance data using reference Excel data."""
    st.subheader("Random Point Generator")
    
    # Load reference data
    if "REFERENCE_DATA" not in st.session_state:
        logger.info("Loading reference data for Random Point Generator...")
        reference_data = load_reference_data()
        if reference_data is None or not reference_data:
            st.error("Failed to load reference data. Please check the Excel file or GitHub URL.")
            logger.error("Failed to load reference data for Random Point Generator.")
            return
        st.session_state.REFERENCE_DATA = reference_data
    else:
        reference_data = st.session_state.REFERENCE_DATA

    # Extract valid values from REFERENCE_DATA
    valid_conduit_sizes = sorted(set([entry['conduit_size'] for entry in reference_data]))
    valid_production_rates = sorted(set([entry['production_rate'] for entry in reference_data]))
    valid_glrs = sorted(set([entry['glr'] for entry in reference_data]))

    # Initialize session state
    if 'random_point_inputs' not in st.session_state:
        st.session_state.random_point_inputs = {
            'num_points': 100,
            'conduit_size': valid_conduit_sizes[0] if valid_conduit_sizes else 2.875
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
            valid_conduit_sizes,
            index=valid_conduit_sizes.index(st.session_state.random_point_inputs['conduit_size']) if st.session_state.random_point_inputs['conduit_size'] in valid_conduit_sizes else 0,
            help="Select conduit size from reference data."
        )
        st.session_state.random_point_inputs['conduit_size'] = conduit_size

    generate = st.button("Generate Random Points")

    if generate:
        with st.spinner("Generating random points..."):
            try:
                # Filter REFERENCE_DATA for selected conduit size
                filtered_data = [entry for entry in reference_data if entry['conduit_size'] == conduit_size]
                if not filtered_data:
                    st.error(f"No data found for conduit size {conduit_size} in reference data.")
                    logger.error(f"No data found for conduit size {conduit_size}.")
                    return

                # Get valid production rates and GLR ranges for this conduit size
                valid_production_rates = sorted(set([entry['production_rate'] for entry in filtered_data]))
                valid_glr_ranges = INTERPOLATION_RANGES.get((conduit_size, valid_production_rates[0]), [100, 1000])
                min_glr, max_glr = min([range[0] for range in valid_glr_ranges]), max([range[1] for range in valid_glr_ranges])

                # Generate random data
                data = {
                    'conduit_size': [conduit_size] * num_points,
                    'production_rate': np.random.choice(valid_production_rates, size=num_points),
                    'glr': np.random.uniform(min_glr, max_glr, size=num_points),
                    'pressure': np.random.uniform(0, 4000, size=num_points),  # From ui.py constraints
                    'depth': np.random.uniform(0, 31000, size=num_points)     # From ui.py constraints
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
