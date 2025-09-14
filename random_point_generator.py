# random_point_generator.py
# Module for Mode 4: Random Point Generator in Streamlit

import streamlit as st
import numpy as np
import pandas as pd
import xlsxwriter
import zipfile
import io
from scipy.optimize import fsolve, bisect
from utils import setup_logging

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
    st.subheader("Mode 4: Random Point Generator")
    
    uploaded_file = st.file_uploader("Upload the Excel file containing polynomial coefficients:", type=["xlsx"])
    if not uploaded_file:
        st.warning("Please upload the coefficient Excel file.")
        return

    try:
        df_coeffs = pd.read_excel(uploaded_file, header=None)
    except Exception as e:
        st.error(f"Error reading Excel: {str(e)}")
        logger.error(f"Excel read error: {str(e)}")
        return

    cell1 = st.text_input("Enter the first cell reference (e.g., A1) for 'name' of first polynomial:")
    cell2 = st.text_input("Enter the second cell reference (e.g., G1) for last coefficient of first polynomial:")
    cell3 = st.text_input("Enter the third cell reference (e.g., A2) for 'name' of last polynomial:")

    if not (cell1 and cell2 and cell3):
        st.warning("Please enter all cell references.")
        return

    try:
        row1, col1 = parse_cell(cell1)
        row2, col2 = parse_cell(cell2)
        row3, col3 = parse_cell(cell3)
    except ValueError as e:
        st.error(f"Cell parsing error: {str(e)}")
        return

    if row1 != row2 or col3 != col1:
        st.error("Invalid cell configuration.")
        return

    polynomials = []
    file_names = []
    for row in range(row1, row3 + 1):
        name = df_coeffs.iloc[row, col1]
        coeffs = df_coeffs.iloc[row, col1 + 1:col2 + 1].tolist()
        polynomials.append(coeffs)
        file_names.append(str(name))

    n = st.number_input("Enter the number of randomly generated points (n):", min_value=1, value=1000)
    min_D = st.number_input("Enter the minimum value for D:", min_value=0.0, value=500.0)
    generate_graphs = st.checkbox("Generate graph sheets?")
    num_graph_sheets = 0
    if generate_graphs:
        num_graph_sheets = st.number_input("How many graph sheets? (up to 10)", min_value=1, max_value=10, value=10)

    if st.button("Generate and Download ZIP"):
        with st.spinner("Generating files..."):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for poly_idx, (coeffs, file_name) in enumerate(zip(polynomials, file_names)):
                    df = pd.DataFrame(columns=['p1', 'D', 'y1', 'y2', 'p2'])
                    while len(df) < n:
                        p1 = np.random.uniform(0, 4000)
                        y1 = calc_y1(p1, coeffs)
                        if y1 is None or not (0 <= y1 <= 31000):
                            continue
                        D = np.random.uniform(min_D, 31000 - y1)
                        y2 = y1 + D
                        p2 = solve_p2(y2, p1, coeffs)
                        if p2 is not None and 0 <= p2 <= 4000:
                            df = pd.concat([df, pd.DataFrame({'p1': [p1], 'D': [D], 'y1': [y1], 'y2': [y2], 'p2': [p2]})], ignore_index=True)

                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='Points', index=False)
                        if generate_graphs:
                            workbook = writer.book
                            points_sheet = writer.sheets['Points']
                            for sheet_num in range(1, num_graph_sheets + 1):
                                chart_sheet = workbook.add_chartsheet(f'Graph {sheet_num}')
                                chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
                                # Add series for example (adapt as needed from snippet)
                                chart.add_series({
                                    'name': 'Pressure vs Depth',
                                    'categories': ['Points', 1, 2, len(df), 2],  # Example: y1 column
                                    'values': ['Points', 1, 0, len(df), 0],  # Example: p1 column
                                    'line': {'color': 'blue'}
                                })
                                # Configure axes (adapt from snippet)
                                chart.set_x_axis({'name': 'Gradient Pressure, psi'})
                                chart.set_y_axis({'name': 'Depth, ft', 'reverse': True})
                                chart_sheet.set_chart(chart)

                    excel_buffer.seek(0)
                    zipf.writestr(f"{file_name}.xlsx", excel_buffer.getvalue())

            zip_buffer.seek(0)
            st.download_button(
                label="Download ZIP of Generated Excels",
                data=zip_buffer,
                file_name="random_points.zip",
                mime="application/zip"
            )
            st.success("Generation complete!")
