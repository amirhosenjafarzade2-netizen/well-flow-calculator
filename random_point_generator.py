# random_point_generator.py
import streamlit as st
import numpy as np
import pandas as pd
import xlsxwriter
import io
from scipy.optimize import fsolve, bisect
from data_loader import load_reference_data
from config import INTERPOLATION_RANGES, COLORS
from utils import export_plot_to_png, setup_logging
import matplotlib.pyplot as plt

logger = setup_logging()


def calc_y1(p, coeffs):
    """Calculate y1 using polynomial coefficients."""
    try:
        y = 0
        for i, coef in enumerate(coeffs):
            y += coef * p ** (len(coeffs) - 1 - i)
        return y if np.isfinite(y) else None
    except Exception as e:
        logger.error(f"Failed to calculate y1: {str(e)}")
        return None


def solve_p2(y2_val, p1, coeffs):
    """Solve for p2 given y2, p1, and coefficients."""
    def polynomial(x, coeffs):
        y = sum(coef * x ** (len(coeffs) - 1 - i) for i, coef in enumerate(coeffs))
        return y if np.isfinite(y) else np.nan

    def root_function(x, target_depth, coeffs):
        return polynomial(x, coeffs) - target_depth

    try:
        p_range = np.linspace(p1, 4000, 100)
        for i in range(len(p_range) - 1):
            p_start, p_end = p_range[i], p_range[i + 1]
            y_start, y_end = polynomial(p_start, coeffs), polynomial(p_end, coeffs)
            if not (np.isfinite(y_start) and np.isfinite(y_end)):
                continue
            f_start, f_end = root_function(p_start, y2_val, coeffs), root_function(p_end, y2_val, coeffs)
            if np.isfinite(f_start) and np.isfinite(f_end) and f_start * f_end <= 0:
                try:
                    candidate = fsolve(root_function, (p_start + p_end) / 2,
                                       args=(y2_val, coeffs), maxfev=20000)[0]
                    if p_start <= candidate <= p_end:
                        return candidate
                except Exception:
                    pass
                try:
                    return bisect(root_function, p_start, p_end,
                                  args=(y2_val, coeffs), maxiter=100)
                except Exception:
                    continue
        return None
    except Exception as e:
        logger.warning(f"Failed to solve p2: {str(e)}")
        return None


def run_random_point_generator():
    """UI for generating and visualizing random well performance data."""
    st.subheader("Random Point Generator")

    # Load reference data
    if "REFERENCE_DATA" not in st.session_state:
        reference_data = load_reference_data()
        if reference_data is None or not reference_data:
            st.error("Failed to load reference data.")
            return
        st.session_state.REFERENCE_DATA = reference_data
    else:
        reference_data = st.session_state.REFERENCE_DATA

    valid_conduit_sizes = sorted(set(entry['conduit_size'] for entry in reference_data))
    valid_production_rates = sorted(set(entry['production_rate'] for entry in reference_data))

    if 'random_point_inputs' not in st.session_state:
        st.session_state.random_point_inputs = {
            'num_points': 100,
            'conduit_size': valid_conduit_sizes[0] if valid_conduit_sizes else 2.875,
            'production_rate': valid_production_rates[0] if valid_production_rates else 100.0,
            'min_D': 500.0,
            'generate_graphs': False,
            'num_graph_sheets': 10
        }

    # --- Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        num_points = st.number_input("Number of Points (n):", 1, 1000,
                                     st.session_state.random_point_inputs['num_points'], step=10)
        min_D = st.number_input("Minimum D (ft):", 0.0, 31000.0,
                                st.session_state.random_point_inputs['min_D'], step=100.0)
    with col2:
        conduit_size = st.selectbox("Conduit Size (in):", valid_conduit_sizes)
        production_rate = st.selectbox("Production Rate (stb/day):", valid_production_rates)

    generate_graphs = st.checkbox("Generate graph sheets?", st.session_state.random_point_inputs['generate_graphs'])
    num_graph_sheets = st.number_input("Number of graph sheets (1–10):", 1, 10,
                                       st.session_state.random_point_inputs['num_graph_sheets'], step=1) if generate_graphs else 0

    if st.button("Generate Random Points"):
        with st.spinner("Generating..."):
            filtered_data = [e for e in reference_data if e['conduit_size'] == conduit_size and e['production_rate'] == production_rate]
            if not filtered_data:
                st.error("No matching data.")
                return

            coeffs_dict = filtered_data[0]['coefficients']
            coeffs = [coeffs_dict[k] for k in sorted(coeffs_dict.keys())]

            # Valid ranges
            valid_glr_ranges = INTERPOLATION_RANGES.get((conduit_size, production_rate), [(100, 1000)])
            min_glr, max_glr = min(r[0] for r in valid_glr_ranges), max(r[1] for r in valid_glr_ranges)

            rows, used_D_set, attempts = [], set(), 0
            while len(rows) < num_points and attempts < 50000:
                attempts += 1
                p1 = np.random.uniform(0, 4000)
                y1 = calc_y1(p1, coeffs)
                if y1 is None or not (0 <= y1 <= 31000):
                    continue
                max_D = max(min_D, min(7000, 31000 - y1))
                if max_D < min_D:
                    continue
                D_candidates = np.linspace(min_D, max_D, 1000)
                np.random.shuffle(D_candidates)
                D = next((d for d in D_candidates if round(d, 8) not in used_D_set), None)
                if D is None:
                    continue
                y2 = min(y1 + D, 31000)
                p2 = solve_p2(y2, p1, coeffs)
                if p2 is None or not (0 <= p2 <= 4000):
                    continue
                used_D_set.add(round(D, 8))
                glr = np.random.uniform(min_glr, max_glr)
                rows.append([conduit_size, production_rate, glr, p1, D, y1, y2, p2])

            df = pd.DataFrame(rows, columns=["conduit_size", "production_rate", "glr", "p1", "D", "y1", "y2", "p2"])
            if df.empty:
                st.error("No valid points generated.")
                return

            st.session_state.random_point_results = df
            st.dataframe(df)

            fig, ax = plt.subplots()
            scatter = ax.scatter(df['p1'], df['y1'], c=df['glr'], cmap='viridis', alpha=0.6)
            ax.set_xlabel("Wellhead Pressure (psi)")
            ax.set_ylabel("Depth (ft)")
            ax.set_title("Random Well Performance Data")
            plt.colorbar(scatter, label="GLR (scf/stb)")
            ax.invert_yaxis()
            st.pyplot(fig)

            try:
                st.download_button("Download Plot as PNG", data=export_plot_to_png(fig),
                                   file_name="random_points_plot.png", mime="image/png")
            except Exception as e:
                logger.error(f"Plot export failed: {e}")

            # --- Excel export ---
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Points', index=False)
                if generate_graphs:
                    workbook = writer.book
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
                            'categories': ['Points', start_row, 3, end_row, 3],  # p1
                            'values': ['Points', start_row, 5, end_row, 5],      # y1
                            'line': {'color': COLORS[(sheet_num - 1) % len(COLORS)]}
                        })
                        chart.set_x_axis({'name': 'Pressure (psi)', 'min': 0, 'max': 4000})
                        chart.set_y_axis({'name': 'Depth (ft)', 'min': 0, 'max': 31000, 'reverse': True})
                        chart.set_size({'width': 1000, 'height': 600})
                        chart_sheet.set_chart(chart)

            excel_buffer.seek(0)
            st.download_button("Download Excel", data=excel_buffer,
                               file_name="random_points.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
