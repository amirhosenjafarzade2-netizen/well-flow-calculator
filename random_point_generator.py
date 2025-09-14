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
import re
import matplotlib.pyplot as plt

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
        p2_guess = min(max(p1 + 100, 2000), 3000)
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

def generate_df(coeffs, num_points, min_D):
    """
    Generate DataFrame of random points for a given set of coefficients.
    Returns df or None if generation fails.
    """
    df = pd.DataFrame(columns=['conduit_size', 'production_rate', 'glr', 'p1', 'D', 'y1', 'y2', 'p2'])
    unique_D = set()
    x_range = (0, 4000)
    y_range = (0, 31000)
    attempts = 0
    max_attempts = num_points * 10  # Prevent infinite loops
    while len(df) < num_points and attempts < max_attempts:
        p1 = np.random.uniform(x_range[0], x_range[1])
        y1 = calc_y1(p1, coeffs)
        if y1 is None or y1 < y_range[0] or y1 > y_range[1] or not np.isfinite(y1):
            attempts += 1
            continue
        max_D = y_range[1] - y1
        D = np.random.uniform(max(min_D, 0), max_D)
        rounded_D = round(D, 8)
        if rounded_D in unique_D:
            attempts += 1
            continue
        unique_D.add(rounded_D)
        y2 = y1 + D
        p2 = solve_p2(y2, p1, coeffs)
        if p2 is None or p2 < x_range[0] or p2 > x_range[1] or not np.isfinite(p2):
            attempts += 1
            continue
        df = pd.concat([df, pd.DataFrame({
            'conduit_size': [np.nan],  # Placeholder, filled later
            'production_rate': [np.nan],
            'glr': [np.nan],
            'p1': [p1],
            'D': [D],
            'y1': [y1],
            'y2': [y2],
            'p2': [p2]
        })], ignore_index=True)
        attempts += 1
    if len(df) < num_points:
        logger.warning(f"Generated only {len(df)} of {num_points} points due to constraints.")
    return df if not df.empty else None

def generate_excel(entry, num_points, min_D, generate_graphs, num_graph_sheets):
    """
    Generate an Excel file for a single entry (GLR line).
    Returns (excel_buffer, file_name)
    """
    conduit_size = entry['conduit_size']
    production_rate = entry['production_rate']
    glr = entry['glr']
    coeffs_dict = entry['coefficients']
    coeffs = [coeffs_dict[k] for k in sorted(coeffs_dict.keys())]  # Ensure order: a, b, c, d, e, f

    # Generate data
    df = generate_df(coeffs, num_points, min_D)
    if df is None or df.empty:
        logger.error(f"No valid points generated for conduit_size={conduit_size}, production_rate={production_rate}, glr={glr}")
        return None, None

    # Fill metadata
    df['conduit_size'] = conduit_size
    df['production_rate'] = production_rate
    df['glr'] = glr

    # Excel file
    excel_buffer = io.BytesIO()
    file_name = f"{conduit_size}_in_{production_rate}_stb-day_{glr}_glr_random_points.xlsx"
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Points', index=False)
        if generate_graphs:
            workbook = writer.book
            points_sheet = writer.sheets['Points']
            # Select random rows for graphs
            if len(df) > num_graph_sheets:
                df_graph = df.sample(n=num_graph_sheets, random_state=42).reset_index(drop=True)
            else:
                df_graph = df
            for sheet_num in range(1, min(num_graph_sheets + 1, len(df_graph) + 1)):
                idx = sheet_num - 1
                row = df_graph.iloc[idx]
                sheet_name = f'Data_{sheet_num}'
                worksheet = workbook.add_worksheet(sheet_name)
                worksheet.hide()

                # Well path (p1,y1 → p2,y2)
                worksheet.write_row(0, 0, ["p", "y"])
                worksheet.write_row(1, 0, [row['p1'], row['y1']])
                worksheet.write_row(2, 0, [row['p2'], row['y2']])

                # Well length (p1,y1 → p1,y2)
                worksheet.write_row(4, 0, ["p", "y"])
                worksheet.write_row(5, 0, [row['p1'], row['y1']])
                worksheet.write_row(6, 0, [row['p1'], row['y2']])

                chart_sheet = workbook.add_chartsheet(f'Graph {sheet_num}')
                chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})

                # Add well path
                chart.add_series({
                    'name': f'Well Path {sheet_num}',
                    'categories': [sheet_name, 1, 0, 2, 0],
                    'values': [sheet_name, 1, 1, 2, 1],
                    'line': {'color': 'red'},
                    'marker': {'type': 'circle', 'size': 6, 'fill': {'color': 'red'}}
                })

                # Add well length
                chart.add_series({
                    'name': f'Well Length {sheet_num}',
                    'categories': [sheet_name, 5, 0, 6, 0],
                    'values': [sheet_name, 5, 1, 6, 1],
                    'line': {'color': 'blue', 'dash_type': 'dash'},
                    'marker': {'type': 'none'}
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
                    'layout': {'x_position': 0.85, 'y_position': 0.02}
                })
                chart.set_size({'width': 1000, 'height': 600})
                chart_sheet.set_chart(chart)

    excel_buffer.seek(0)
    return excel_buffer, file_name

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
            'glr': None,
            'min_D': 500.0,
            'all_glr': False,
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
            help="Number of random data points to generate per Excel."
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

    all_glr = st.checkbox(
        "Generate for all GLR lines?",
        value=st.session_state.random_point_inputs['all_glr'],
        help="Check to generate separate Excel files for all GLR in the selected conduit and production rate."
    )
    st.session_state.random_point_inputs['all_glr'] = all_glr

    glr = None
    if not all_glr:
        filtered_data = [entry for entry in reference_data 
                         if entry['conduit_size'] == conduit_size and entry['production_rate'] == production_rate]
        valid_glrs = sorted(set([entry['glr'] for entry in filtered_data]))
        if not valid_glrs:
            st.error("No valid GLR values found for the selected conduit size and production rate.")
            return
        glr = st.selectbox(
            "GLR (scf/stb):",
            valid_glrs,
            index=valid_glrs.index(st.session_state.random_point_inputs['glr']) if st.session_state.random_point_inputs['glr'] in valid_glrs else 0,
            help="Select GLR from reference data."
        )
        st.session_state.random_point_inputs['glr'] = glr

    generate_graphs = st.checkbox(
        "Generate graph sheets in Excel?",
        value=st.session_state.random_point_inputs['generate_graphs'],
        help="Check to include graph sheets in the output Excel file(s)."
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
            help="Number of graph sheets to include in each Excel file."
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

                if all_glr:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for entry in filtered_data:
                            excel_buffer, file_name = generate_excel(entry, num_points, min_D, generate_graphs, num_graph_sheets)
                            if excel_buffer and file_name:
                                zipf.writestr(file_name, excel_buffer.getvalue())
                    zip_buffer.seek(0)
                    st.download_button(
                        label="Download ZIP of All GLR Excels",
                        data=zip_buffer,
                        file_name="all_glr_random_points.zip",
                        mime="application/zip"
                    )
                else:
                    entry = next((e for e in filtered_data if e['glr'] == glr), None)
                    if entry is None:
                        st.error(f"No data found for GLR {glr}.")
                        logger.error(f"No data found for GLR {glr}.")
                        return
                    df = generate_df([entry['coefficients'][k] for k in sorted(entry['coefficients'].keys())], num_points, min_D)
                    if df is None or df.empty:
                        st.error(f"No valid points generated for GLR {glr}.")
                        logger.error(f"No valid points generated for GLR {glr}.")
                        return
                    # Fill metadata for display
                    df['conduit_size'] = conduit_size
                    df['production_rate'] = production_rate
                    df['glr'] = glr
                    # Display data for single GLR
                    st.subheader("Generated Data")
                    st.dataframe(df)
                    # Plot data
                    fig, ax = plt.subplots()
                    for idx, row in df.iterrows():
                        ax.plot([row['p1'], row['p2']], [row['y1'], row['y2']], color='red', label='Well Path' if idx == 0 else None)
                        ax.plot([row['p1'], row['p1']], [row['y1'], row['y2']], color='blue', linestyle='--', label='Well Length' if idx == 0 else None)
                    ax.set_xlabel("Gradient Pressure, psi")
                    ax.set_ylabel("Depth, ft")
                    ax.set_title(f"Random Well Performance Data (Conduit Size: {conduit_size} in, Production Rate: {production_rate} stb/day, GLR: {glr} scf/stb)")
                    ax.invert_yaxis()
                    ax.legend()
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
                    # Generate single Excel
                    excel_buffer, file_name = generate_excel(entry, num_points, min_D, generate_graphs, num_graph_sheets)
                    if excel_buffer and file_name:
                        st.download_button(
                            label="Download Excel",
                            data=excel_buffer,
                            file_name=file_name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                st.success("Generation complete!")

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                logger.error(f"Random point generation failed: {str(e)}")

    st.write("**Generation Logs**")
    st.write("Any warnings or informational messages will appear here.")
