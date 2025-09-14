# random_point_generator.py
import streamlit as st
import numpy as np
import pandas as pd
import xlsxwriter
import io
from scipy.optimize import fsolve, bisect
from data_loader import load_reference_data
from config import PRODUCTION_RATES, INTERPOLATION_RANGES, COLORS
from utils import export_plot_to_png, setup_logging
import re
import matplotlib.pyplot as plt  # Added import to fix NameError

logger = setup_logging()

def parse_cell(cell):
    """
    Parse a cell reference (e.g., 'A1') into row and column indices.
    Returns (row, col) or raises ValueError.
    """
    try:
        col = ord(cell[0].upper()) - ord('A')
        row = int(cell[1:]) - 1
        if row < 0 or col < 0:
            raise ValueError("Row and column indices must be positive.")
        return row, col
    except (ValueError, IndexError):
        raise ValueError(f"Invalid cell reference format: {cell}. Use format like A1.")

def calc_y1(p, coeffs):
    """
    Calculate y1 (wellhead depth) using polynomial coefficients.
    Returns y1 (float) or None if calculation fails.
    """
    try:
        y = 0
        for i, coef in enumerate(coeffs):
            y += coef * p ** (len(coeffs) - 1 - i)
        if not np.isfinite(y):
            logger.warning(f"Non-finite y1 calculated for p={p}, coeffs={coeffs}")
            return None
        return y
    except Exception as e:
        logger.error(f"Failed to calculate y1: {str(e)}")
        return None

def solve_p2(y2_val, p1, coeffs):
    """
    Solve for p2 (bottomhole pressure) given y2, p1, and coefficients.
    Returns p2 (float) or None if calculation fails.
    """
    def polynomial(x, coeffs):
        try:
            y = 0
            for i, coef in enumerate(coeffs):
                y += coef * x ** (len(coeffs) - 1 - i)
            return y if np.isfinite(y) else np.nan
        except Exception:
            return np.nan

    def func(p2):
        y2 = polynomial(p2, coeffs)
        return y2 - y2_val if np.isfinite(y2) else np.inf

    try:
        # Improved initial guess: midpoint of valid range or adjusted based on p1
        p2_guess = min(max(p1 + 100, 2000), 3000)  # Avoid extremes
        # Try bisect first for stability
        if func(0) * func(4000) < 0:
            p2 = bisect(func, 0, 4000, maxiter=100)
        else:
            p2 = fsolve(func, p2_guess, maxfev=100)[0]
        if not (0 <= p2 <= 4000) or not np.isfinite(p2):
            logger.warning(f"Invalid p2={p2} for y2_val={y2_val}, p1={p1}, coeffs={coeffs}")
            return None
        return p2
    except Exception as e:
        logger.warning(f"Failed to solve p2 for y2_val={y2_val}, p1={p1}, coeffs={coeffs}: {str(e)}")
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
            max_value=31000.0,
            value=st.session_state.random_point_inputs['min_D'],
            step=100.0,
            help="Minimum well length for random generation (0 to 31000 ft)."
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

    generate_graphs = st.checkbox(
        "Generate graph sheets in Excel?",
        value=st.session_state.random_point_inputs['generate_graphs'],
        help="Check to include graph sheets in the output Excel file."
    )
    st.session_state.random_point_inputs['generate_graphs'] = generate_graphs

    num_graph_sheets = 0
    if generate_graphs:
        num_graph_sheets = st.number_input(
            "How many graph sheets? (1–10)",
            min_value=1,
            max_value=10,
            value=st.session_state.random_point_inputs['num_graph_sheets'],
            step=1,
            help="Number of graph sheets to include in the Excel file."
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
                    logger.error(f"No data found for conduit size {conduit_size} and production_rate {production_rate}.")
                    return

                # Get valid GLR ranges and coefficients
                valid_glr_ranges = INTERPOLATION_RANGES.get((conduit_size, production_rate), [(100, 1000)])
                min_glr, max_glr = min([r[0] for r in valid_glr_ranges]), max([r[1] for r in valid_glr_ranges])
                coeffs = filtered_data[0]['coefficients']  # Use first matching entry's coefficients

                # Generate random data
                data = {
                    'conduit_size': [conduit_size] * num_points,
                    'production_rate': [production_rate] * num_points,
                    'glr': np.random.uniform(min_glr, max_glr, size=num_points),
                    'p1': np.random.uniform(0, 4000, size=num_points),  # Wellhead pressure
                    'D': np.random.uniform(min_D, 31000, size=num_points)  # Depth
                }
                df = pd.DataFrame(data)

                # Calculate y1, p2, y2
                df['y1'] = df['p1'].apply(lambda p: calc_y1(p, list(coeffs.values())))
                df['y2'] = df['y1'] + df['D']
                df['p2'] = [solve_p2(y2, p1, list(coeffs.values())) for y2, p1 in zip(df['y2'], df['p1'])]

                # Drop rows with invalid calculations
                df = df.dropna()

                # Check if any valid data remains
                if df.empty:
                    st.error("No valid points generated. Please check input parameters or reference data.")
                    logger.error("No valid points generated after calculations.")
                    return

                # Store results in session state
                st.session_state.random_point_results = df

                # Display data
                st.subheader("Generated Data")
                st.dataframe(df)

                # Plot data
                fig, ax = plt.subplots()
                scatter = ax.scatter(df['p1'], df['y1'], c=df['glr'], cmap='viridis', alpha=0.6)
                ax.set_xlabel("Wellhead Pressure (p1, psi)")
                ax.set_ylabel("Wellhead Depth (y1, ft)")
                ax.set_title(f"Random Well Performance Data (Conduit Size: {conduit_size} in, Production Rate: {production_rate} stb/day)")
                plt.colorbar(scatter, label="GLR (scf/stb)")
                ax.invert_yaxis()  # Depth increases downward
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
                        # Shuffle the data for random subsets
                        df_shuffled = df.sample(frac=1).reset_index(drop=True)
                        points_per_sheet = len(df_shuffled) // num_graph_sheets
                        for sheet_num in range(1, num_graph_sheets + 1):
                            start_row = 1 + (sheet_num - 1) * points_per_sheet
                            end_row = start_row + points_per_sheet - 1
                            if sheet_num == num_graph_sheets:
                                end_row = len(df_shuffled)
                            chart_sheet = workbook.add_chartsheet(f'Graph {sheet_num}')
                            chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
                            chart.add_series({
                                'name': f'p1 vs y1 (Graph {sheet_num})',
                                'categories': ['Points', start_row, 4, end_row, 4],  # y1 column (index 4)
                                'values': ['Points', start_row, 3, end_row, 3],  # p1 column (index 3)
                                'line': {'color': COLORS[(sheet_num - 1) % len(COLORS)]}
                            })
                            chart.set_x_axis({
                                'name': 'Gradient Pressure, psi',
                                'min': 0,
                                'max': 4000,
                                'major_unit': 1000,
                                'minor_unit': 200,
                                'name_font': {'color': 'black'},
                                'num_font': {'color': 'black'},
                                'line': {'color': 'black'},
                                'major_gridlines': {'visible': True, 'line': {'color': '#D3D3D3'}},
                                'minor_gridlines': {'visible': True, 'line': {'color': '#D3D3D3', 'width': 0.5}}
                            })
                            chart.set_y_axis({
                                'name': 'Depth, ft',
                                'min': 0,
                                'max': 31000,
                                'major_unit': 10000,
                                'minor_unit': 2000,
                                'reverse': True,
                                'name_font': {'color': 'black'},
                                'num_font': {'color': 'black'},
                                'line': {'color': 'black'},
                                'major_gridlines': {'visible': True, 'line': {'color': '#D3D3D3'}},
                                'minor_gridlines': {'visible': True, 'line': {'color': '#D3D3D3', 'width': 0.5}}
                            })
                            chart.set_legend({
                                'position': 'right',
                                'font': {'size': 8},
                                'border': {'color': 'black', 'width': 0.5},
                                'layout': {'x_position': 0.85, 'y_position': 0.02},
                            })
                            chart.set_size({'width': 1000, 'height': 600})
                            chart_sheet.set_chart(chart)

                excel_buffer.seek(0)
                st.download_button(
                    label="Download Excel",
                    data=excel_buffer,
                    file_name="random_points.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                logger.error(f"Random point generation failed: {str(e)}")

    st.write("**Generation Logs**")
    st.write("Any warnings or informational messages will appear here.")
