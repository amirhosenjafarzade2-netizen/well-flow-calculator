import streamlit as st
import numpy as np
import pandas as pd
import xlsxwriter
import zipfile
import io
from scipy.optimize import fsolve, bisect
from data_loader import load_reference_data
from config import PRODUCTION_RATES, INTERPOLATION_RANGES, COLORS
from utils import setup_logging
import re

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
            return None
        return y
    except Exception:
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
            return y
        except Exception:
            return np.nan

    def root_function(x, target_depth, coeffs):
        return polynomial(x, coeffs) - target_depth

    p2 = None
    p_range = np.linspace(p1, 4000, 100)
    for i in range(len(p_range) - 1):
        p_start, p_end = p_range[i], p_range[i + 1]
        try:
            y_start = polynomial(p_start, coeffs)
            y_end = polynomial(p_end, coeffs)
            if not (np.isfinite(y_start) and np.isfinite(y_end)):
                continue
            f_start = root_function(p_start, y2_val, coeffs)
            f_end = root_function(p_end, y2_val, coeffs)
            if np.isfinite(f_start) and np.isfinite(f_end) and f_start * f_end <= 0:
                try:
                    candidate = fsolve(root_function, (p_start + p_end) / 2, args=(y2_val, coeffs), maxfev=20000)[0]
                    if p_start - 1e-6 <= candidate <= p_end + 1e-6:
                        y_candidate = polynomial(candidate, coeffs)
                        if np.isfinite(y_candidate) and 0 <= y_candidate <= 31000:
                            p2 = candidate
                            break
                except Exception:
                    pass
                try:
                    candidate = bisect(root_function, p_start, p_end, args=(y2_val, coeffs), maxiter=100)
                    y_candidate = polynomial(candidate, coeffs)
                    if np.isfinite(y_candidate) and 0 <= y_candidate <= 31000:
                        p2 = candidate
                        break
                except Exception:
                    continue
        except Exception:
            continue
    return p2

def generate_df(coeffs, num_points, min_D):
    """
    Generate DataFrame of random points for a given set of coefficients.
    Returns df or None if generation fails.
    """
    rows = []
    used_D_set = set()
    x_range = (0, 4000)
    y_range = (0, 31000)
    max_attempts = 50000
    attempts = 0

    while len(rows) < num_points and attempts < max_attempts:
        attempts += 1
        p1 = np.random.uniform(x_range[0], x_range[1])
        y1 = calc_y1(p1, coeffs)
        if y1 is None or not np.isfinite(y1) or y1 < 0 or y1 > y_range[1]:
            continue
        max_D = max(min_D, min(7000, y_range[1] - y1))
        if max_D < min_D:
            continue

        D_candidates = np.linspace(min_D, max_D, 1000)
        np.random.shuffle(D_candidates)
        D = None
        for d in D_candidates:
            if round(d, 8) not in used_D_set:
                D = d
                break
        if D is None:
            continue

        y2 = y1 + D
        if y2 > y_range[1]:
            y2 = y_range[1]
            D = y2 - y1

        p2 = solve_p2(y2, p1, coeffs)
        if p2 is None or p2 < x_range[0] or p2 > x_range[1]:
            continue

        used_D_set.add(round(D, 8))
        rows.append([p1, D, y1, y2, p2])

    df = pd.DataFrame(rows, columns=["p1", "D", "y1", "y2", "p2"])
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

    # Calculate dynamic x_range
    x_values = np.linspace(0, 4000, 1000)
    y_values = [calc_y1(x, coeffs) for x in x_values]
    max_x = 4000
    for x, y in zip(x_values, y_values):
        if y is not None and y >= 31000:
            max_x = x
            break
    x_range = (0, max_x)
    y_range = (0, 31000)

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
        df.to_excel(writer, sheet_name='Data', index=False)
        if generate_graphs:
            workbook = writer.book
            chartdata_sheet = workbook.add_worksheet('ChartData')
            chartdata_sheet.hide()

            # Write GLR curve data
            p1_full = np.linspace(x_range[0], x_range[1], 100)
            y1_full = [calc_y1(p, coeffs) for p in p1_full]
            chartdata_sheet.write_row(0, 0, p1_full)
            chartdata_sheet.write_row(1, 0, [y if y is not None and y <= y_range[1] else y_range[1] for y in y1_full])

            # Select random rows for graphs
            if len(df) > num_graph_sheets:
                selected_indices = np.random.choice(len(df), size=num_graph_sheets, replace=False)
                df_graph = df.iloc[selected_indices].reset_index(drop=True)
            else:
                selected_indices = range(len(df))
                df_graph = df

            # Write selected points to Points sheet without extra columns
            df_graph[['p1', 'D', 'y1', 'y2', 'p2']].to_excel(writer, sheet_name='Points', index=False)

            for sheet_num in range(1, min(num_graph_sheets + 1, len(df_graph) + 1)):
                idx = sheet_num - 1
                p1_val, D_val, y1_val, y2_val, p2_val = df_graph.iloc[idx]['p1'], df_graph.iloc[idx]['D'], df_graph.iloc[idx]['y1'], df_graph.iloc[idx]['y2'], df_graph.iloc[idx]['p2']

                # Write data for chart lines vertically
                row_offset = 2 + idx * 10
                # Vertical line at p1: (p1, y1) to (p1, 0)
                chartdata_sheet.write(row_offset, 0, p1_val)
                chartdata_sheet.write(row_offset, 1, y1_val)
                chartdata_sheet.write(row_offset + 1, 0, p1_val)
                chartdata_sheet.write(row_offset + 1, 1, 0)
                # Horizontal line at y1: (p1, y1) to (0, y1)
                chartdata_sheet.write(row_offset + 2, 0, p1_val)
                chartdata_sheet.write(row_offset + 2, 1, y1_val)
                chartdata_sheet.write(row_offset + 3, 0, 0)
                chartdata_sheet.write(row_offset + 3, 1, y1_val)
                # Vertical line at p2: (p2, y2) to (p2, 0)
                chartdata_sheet.write(row_offset + 4, 0, p2_val)
                chartdata_sheet.write(row_offset + 4, 1, y2_val)
                chartdata_sheet.write(row_offset + 5, 0, p2_val)
                chartdata_sheet.write(row_offset + 5, 1, 0)
                # Horizontal line at y2: (p2, y2) to (0, y2)
                chartdata_sheet.write(row_offset + 6, 0, p2_val)
                chartdata_sheet.write(row_offset + 6, 1, y2_val)
                chartdata_sheet.write(row_offset + 7, 0, 0)
                chartdata_sheet.write(row_offset + 7, 1, y2_val)
                # Well Length line: (0, y1) to (0, y2)
                chartdata_sheet.write(row_offset + 8, 0, 0)
                chartdata_sheet.write(row_offset + 8, 1, y1_val)
                chartdata_sheet.write(row_offset + 9, 0, 0)
                chartdata_sheet.write(row_offset + 9, 1, y2_val if y2_val <= y_range[1] else y_range[1])

                # Create chart
                chart_sheet = workbook.add_chartsheet(f'Graph_{sheet_num}')
                chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})

                # GLR curve series
                chart.add_series({
                    'name': 'GLR curve',
                    'categories': ['ChartData', 0, 0, 0, 99],
                    'values': ['ChartData', 1, 0, 1, 99],
                    'line': {'color': 'blue', 'width': 2.5},
                    'marker': {'type': 'none'},
                })

                # Add (p1, y1) and (p2, y2) markers
                chart.add_series({
                    'name': f'(p1, y1) = ({p1_val:.2f} psi, {y1_val:.2f} ft)',
                    'categories': ['Points', idx + 1, 0, idx + 1, 0],
                    'values': ['Points', idx + 1, 2, idx + 1, 2],
                    'marker': {'type': 'circle', 'size': 7, 'fill': {'color': 'blue'}},
                    'line': {'none': True},
                })
                chart.add_series({
                    'name': f'(p2, y2) = ({p2_val:.2f} psi, {y2_val:.2f} ft)',
                    'categories': ['Points', idx + 1, 4, idx + 1, 4],
                    'values': ['Points', idx + 1, 3, idx + 1, 3],
                    'marker': {'type': 'circle', 'size': 7, 'fill': {'color': 'blue'}},
                    'line': {'none': True},
                })

                # Red projection lines
                chart.add_series({
                    'name': 'Connecting Line',
                    'categories': ['ChartData', row_offset, 0, row_offset + 1, 0],
                    'values': ['ChartData', row_offset, 1, row_offset + 1, 1],
                    'line': {'color': 'red', 'width': 1},
                    'marker': {'type': 'none'},
                })
                chart.add_series({
                    'name': '',
                    'categories': ['ChartData', row_offset + 2, 0, row_offset + 3, 0],
                    'values': ['ChartData', row_offset + 2, 1, row_offset + 3, 1],
                    'line': {'color': 'red', 'width': 1},
                    'marker': {'type': 'none'},
                    'legend': {'none': True},
                })
                chart.add_series({
                    'name': '',
                    'categories': ['ChartData', row_offset + 4, 0, row_offset + 5, 0],
                    'values': ['ChartData', row_offset + 4, 1, row_offset + 5, 1],
                    'line': {'color': 'red', 'width': 1},
                    'marker': {'type': 'none'},
                    'legend': {'none': True},
                })
                chart.add_series({
                    'name': '',
                    'categories': ['ChartData', row_offset + 6, 0, row_offset + 7, 0],
                    'values': ['ChartData', row_offset + 6, 1, row_offset + 7, 1],
                    'line': {'color': 'red', 'width': 1},
                    'marker': {'type': 'none'},
                    'legend': {'none': True},
                })

                # Well Length line
                chart.add_series({
                    'name': f'Well Length ({D_val:.2f} ft)',
                    'categories': ['ChartData', row_offset + 8, 0, row_offset + 9, 0],
                    'values': ['ChartData', row_offset + 8, 1, row_offset + 9, 1],
                    'line': {'color': 'green', 'width': 4},
                    'marker': {'type': 'none'},
                })

                # Axes setup with dynamic ranges
                chart.set_x_axis({
                    'name': 'Gradient Pressure, psi',
                    'min': x_range[0],
                    'max': x_range[1],
                    'major_unit': (x_range[1] - x_range[0]) / 4,
                    'minor_unit': (x_range[1] - x_range[0]) / 20,
                    'name_font': {'color': 'black'},
                    'num_font': {'color': 'black'},
                    'line': {'color': 'black'},
                    'major_gridlines': {'visible': True, 'line': {'color': '#D3D3D3'}},
                    'minor_gridlines': {'visible': True, 'line': {'color': '#D3D3D3', 'width': 0.5}},
                    'minor_tick_mark': 'none',
                })

                chart.set_y_axis({
                    'name': 'Depth, ft',
                    'min': y_range[0],
                    'max': y_range[1],
                    'major_unit': (y_range[1] - y_range[0]) / 4,
                    'minor_unit': (y_range[1] - y_range[0]) / 20,
                    'reverse': True,
                    'name_font': {'color': 'black'},
                    'num_font': {'color': 'black'},
                    'line': {'color': 'black'},
                    'major_gridlines': {'visible': True, 'line': {'color': '#D3D3D3'}},
                    'minor_gridlines': {'visible': True, 'line': {'color': '#D3D3D3', 'width': 0.5}},
                    'minor_tick_mark': 'none',
                })

                chart.set_x2_axis({
                    'name': 'Gradient Pressure, psi',
                    'min': x_range[0],
                    'max': x_range[1],
                    'major_unit': (x_range[1] - x_range[0]) / 4,
                    'minor_unit': (x_range[1] - x_range[0]) / 20,
                    'name_font': {'color': 'black'},
                    'num_font': {'color': 'black'},
                    'line': {'color': 'black'},
                    'major_gridlines': {'visible': True, 'line': {'color': '#D3D3D3'}},
                    'minor_gridlines': {'visible': True, 'line': {'color': '#D3D3D3', 'width': 0.5}},
                    'minor_tick_mark': 'none',
                })

                chart.set_legend({
                    'position': 'right',
                    'font': {'size': 8},
                    'border': {'color': 'black', 'width': 0.5},
                    'layout': {'x_position': 0.85, 'y_position': 0.02},
                })

                chart.set_size({'width': 1000, 'height': 600})
                chart.set_title({'none': True})
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
