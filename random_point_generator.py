# random_point_generator.py
import streamlit as st
import numpy as np
import pandas as pd
import xlsxwriter
import zipfile
import io
from scipy.optimize import fsolve, bisect
from data_loader import load_reference_data
from config import PRODUCTION_RATES, INTERPOLATION_RANGES, COLORS
from utils import export_plot_to_png, setup_logging

logger = setup_logging()

def parse_cell(cell):
    try:
        col = ord(cell[0].upper()) - ord('A')
        row = int(cell[1:]) - 1
        if row < 0 or col < 0:
            raise ValueError("Row and column indices must be positive.")
        return row, col
    except (ValueError, IndexError):
        raise ValueError(f"Invalid cell reference format: {cell}. Use format like A1.")

def calc_y1(p, coeffs):
    try:
        y = 0
        for i, coef in enumerate(coeffs):
            y += coef * p ** (len(coeffs) - 1 - i)
        if not np.isfinite(y):
            return None
        return y
    except Exception:
        return None

def solve_p2(y2_val, p1, coeffs):
    def polynomial(x, coeffs):
        try:
            y = 0
            for i, coef in enumerate(coeffs):
                y += coef * x ** (len(coeffs) - 1 - i)
            return y
        except Exception:
            return np.nan

    def func(p2):
        y2 = polynomial(p2, coeffs)
        return y2 - y2_val if np.isfinite(y2) else np.inf

    try:
        p2_guess = p1 + 100
        p2 = bisect(func, 0, 4000) if func(p2_guess) * func(0) < 0 else fsolve(func, p2_guess)[0]
        if not (0 <= p2 <= 4000) or not np.isfinite(p2):
            return None
        return p2
    except Exception:
        return None

def run_random_point_generator():
    """UI for generating and visualizing random well performance data using reference Excel data."""
    st.subheader("Random Point Generator")
    
    # Load reference data (using Excel from GitHub via data_loader)
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

    # Initialize session state
    if 'random_point_inputs' not in st.session_state:
        st.session_state.random_point_inputs = {
            'num_points': 100,
            'conduit_size': valid_conduit_sizes[0] if valid_conduit_sizes else 2.875,
            'production_rate': valid_production_rates[0] if valid_production_rates else 100.0,
            'min_D': 500.0,
            'generate_graphs': False,
            'num_graph_sheets': 10
        }

    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        num_points = st.number_input(
            "Number of Points to Generate (n):",
            min_value=1,
            max_value=1000,
            value=st.session_state.random_point_inputs['num_points'],
            step=10,
            help="Number of random data points to generate."
        )
        st.session_state.random_point_inputs['num_points'] = num_points
        
        min_D = st.number_input(
            "Minimum Value for D (ft):",
            min_value=0.0,
            value=st.session_state.random_point_inputs['min_D'],
            step=100.0,
            help="Minimum well length for random generation."
        )
        st.session_state.random_point_inputs['min_D'] = min_D

    with col2:
        conduit_size = st.selectbox(
            "Conduit Size (in):",
            valid_conduit_sizes,
            index=valid_conduit_sizes.index(st.session_state.random_point_inputs['conduit_size']) if st.session_state.random_point_inputs['conduit_size'] in valid_conduit_sizes else 0,
            help="Select conduit size from reference data."
        )
        st.session_state.random_point_inputs['conduit_size'] = conduit_size
        
        production_rate = st.selectbox(
            "Production Rate (stb/day):",
            valid_production_rates,
            index=valid_production_rates.index(st.session_state.random_point_inputs['production_rate']) if st.session_state.random_point_inputs['production_rate'] in valid_production_rates else 0,
            help="Select production rate from reference data."
        )
        st.session_state.random_point_inputs['production_rate'] = production_rate

    generate_graphs = st.checkbox("Generate graph sheets in Excel?")
    st.session_state.random_point_inputs['generate_graphs'] = generate_graphs
    num_graph_sheets = 0
    if generate_graphs:
        num_graph_sheets = st.number_input(
            "How many graph sheets? (up to 10)",
            min_value=1,
            max_value=10,
            value=st.session_state.random_point_inputs['num_graph_sheets'],
            step=1
        )
        st.session_state.random_point_inputs['num_graph_sheets'] = num_graph_sheets

    generate = st.button("Generate Random Points")

    if generate:
        with st.spinner("Generating random points..."):
            try:
                # Filter REFERENCE_DATA for selected conduit size and production rate
                filtered_data = [entry for entry in reference_data 
                                 if entry['conduit_size'] == conduit_size and entry['production_rate'] == production_rate]
                if not filtered_data:
                    st.error(f"No data found for conduit size {conduit_size} and production rate {production_rate} in reference data.")
                    logger.error(f"No data found for conduit size {conduit_size} and production rate {production_rate}.")
                    return

                # Get valid GLR ranges for the selected conduit size and production rate
                valid_glr_ranges = INTERPOLATION_RANGES.get((conduit_size, production_rate), [(100, 1000)])
                min_glr, max_glr = min([r[0] for r in valid_glr_ranges]), max([r[1] for r in valid_glr_ranges])

                # Generate random data
                data = {
                    'conduit_size': [conduit_size] * num_points,
                    'production_rate': [production_rate] * num_points,
                    'glr': np.random.uniform(min_glr, max_glr, size=num_points),
                    'pressure': np.random.uniform(0, 4000, size=num_points),  # From ui.py constraints
                    'depth': np.random.uniform(min_D, 31000, size=num_points)  # From user input and ui.py constraints
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
                ax.set_title(f"Random Well Performance Data (Conduit Size: {conduit_size} in, Production Rate: {production_rate} stb/day)")
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
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Points', index=False)
                    if generate_graphs:
                        workbook = writer.book
                        points_sheet = writer.sheets['Points']
                        for sheet_num in range(1, num_graph_sheets + 1):
                            chart_sheet = workbook.add_chartsheet(f'Graph {sheet_num}')
                            chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
                            # Example series (adapt as needed)
                            chart.add_series({
                                'name': 'Pressure vs Depth',
                                'categories': ['Points', 1, 4, num_points, 4],  # Example: depth column (index 4)
                                'values': ['Points', 1, 3, num_points, 3],  # Example: pressure column (index 3)
                                'line': {'color': 'blue'}
                            })
                            # Configure axes (adapt as needed)
                            chart.set_x_axis({'name': 'Gradient Pressure, psi'})
                            chart.set_y_axis({'name': 'Depth, ft', 'reverse': True})
                            chart_sheet.set_chart(chart)

                excel_buffer.seek(0)
                st.download_button(
                    label="Download Excel",
                    data=excel_buffer,
                    file_name="random_points.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"Failed to generate random points: {str(e)}")
                logger.error(f"Random point generation failed: {str(e)}")

    st.write("**Generation Logs**")
    st.write("Any warnings or informational messages will appear here.")
