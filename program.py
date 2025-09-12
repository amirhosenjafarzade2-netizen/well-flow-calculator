# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import fsolve, bisect, curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import re

# Global variable to store reference data using session state
if 'REFERENCE_DATA' not in st.session_state:
    st.session_state.REFERENCE_DATA = None

# Interpolation ranges
INTERPOLATION_RANGES = {
    (2.875, 50): [(0, 10000), (10000, 17500)],
    (2.875, 100): [(0, 10000), (10000, 12500)],
    (2.875, 200): [(0, 6000), (6000, 8000)],
    (2.875, 400): [(0, 4000), (4000, 6500)],
    (2.875, 600): [(0, 3000), (3000, 5000)],
    (3.5, 50): [(0, 15000), (15000, 25000)],
    (3.5, 100): [(0, 10000), (10000, 17500)],
    (3.5, 200): [(0, 8000), (8000, 12000)],
    (3.5, 400): [(0, 8000), (8000, 9000)],
    (3.5, 600): [(0, 4000), (4000, 6000)]
}

# Available production rates for interpolation
PRODUCTION_RATES = [50, 100, 200, 400, 600]

# Function to parse the name column
def parse_name(name):
    parts = name.split()
    try:
        conduit_size = float(parts[0])
        production_rate = float(parts[2])
        glr_str = parts[4].replace('glr', '')
        glr = float(glr_str)
        return conduit_size, production_rate, glr
    except (IndexError, ValueError):
        st.error(f"Failed to parse reference data name: {name}")
        return None, None, None

# Function to load reference data
def load_reference_data():
    st.write("Please upload the reference Excel file (format: name in first column, coefficients in columns 2-7).")
    uploaded_file = st.file_uploader("Upload Reference Excel", type=["xlsx"])
    if uploaded_file is None:
        st.error("No file uploaded. Please upload the file.")
        return None
    try:
        df_ref = pd.read_excel(uploaded_file, header=None, engine='openpyxl')
        # Validate file structure
        if df_ref.shape[1] < 6:
            st.error("Invalid Excel file: Must have at least 6 columns (name + 5 or 6 coefficients).")
            return None
        data_ref = []
        for index, row in df_ref.iterrows():
            name = row[0]
            if pd.isna(name) or isinstance(name, (int, float)):
                st.warning(f"Skipping row {index} due to invalid name: {name}")
                continue
            name = str(name).strip()
            if not re.match(r'[\d.]+\s*in\s*\d+\s*stb-day\s*\d+\s*glr', name.lower()):
                st.warning(f"Failed to parse reference data name: {name}")
                continue
            conduit_size, production_rate, glr = parse_name(name)
            if conduit_size is None:
                continue
            try:
                coefficients = {
                    'a': float(row[1]),
                    'b': float(row[2]),
                    'c': float(row[3]),
                    'd': float(row[4]),
                    'e': float(row[5]),
                    'f': float(row[6]) if len(row) > 6 and pd.notna(row[6]) else 0.0
                }
            except (ValueError, TypeError) as e:
                st.error(f"Error parsing coefficients in row {index}: {e}")
                continue
            data_ref.append({
                'conduit_size': conduit_size,
                'production_rate': production_rate,
                'glr': glr,
                'coefficients': coefficients
            })
        if not data_ref:
            st.error("No valid data parsed from the reference Excel file.")
            return None
        return data_ref
    except Exception as e:
        st.error(f"Error loading reference Excel file: {str(e)}")
        return None

# Polynomial calculation function with dynamic range and production rate interpolation
def calculate_results(conduit_size_input, production_rate_input, glr_input, p1, D, data_ref):
    # Validate conduit size
    if conduit_size_input not in [2.875, 3.5]:
        st.error("Invalid conduit size. Must be 2.875 or 3.5.")
        return None, None, None, None, None, None, None

    # Validate production rate
    if not (50 <= production_rate_input <= 600):
        st.error("Production rate must be between 50 and 600 stb/day (interpolation supported).")
        return None, None, None, None, None, None, None

    # Find the two closest production rates for interpolation
    lower_prate = max([pr for pr in PRODUCTION_RATES if pr <= production_rate_input], default=50)
    higher_prate = min([pr for pr in PRODUCTION_RATES if pr >= production_rate_input], default=600)

    # Check if production rate is exact or needs interpolation
    if abs(lower_prate - higher_prate) < 1e-6:
        prate1 = prate2 = lower_prate
        production_interpolation_status = "exact"
    else:
        prate1 = lower_prate
        prate2 = higher_prate
        production_interpolation_status = "interpolated"

    # Validate GLR for both production rates
    valid_glr1 = False
    valid_range1 = None
    if (conduit_size_input, prate1) in INTERPOLATION_RANGES:
        for min_glr, max_glr in INTERPOLATION_RANGES[(conduit_size_input, prate1)]:
            if min_glr <= glr_input <= max_glr:
                valid_glr1 = True
                valid_range1 = (min_glr, max_glr)
                break
    if not valid_glr1:
        st.error(f"GLR {glr_input} is outside the valid interpolation ranges for conduit size {conduit_size_input} and production rate {prate1}.")
        return None, None, None, None, None, None, None

    valid_glr2 = False
    valid_range2 = None
    if (conduit_size_input, prate2) in INTERPOLATION_RANGES:
        for min_glr, max_glr in INTERPOLATION_RANGES[(conduit_size_input, prate2)]:
            if min_glr <= glr_input <= max_glr:
                valid_glr2 = True
                valid_range2 = (min_glr, max_glr)
                break
    if not valid_glr2:
        st.error(f"GLR {glr_input} is outside the valid interpolation ranges for conduit size {conduit_size_input} and production rate {prate2}.")
        return None, None, None, None, None, None, None

    # Function to get coefficients for a specific production rate and GLR
    def get_coefficients(conduit_size, production_rate, glr_input, valid_range, data_ref):
        # Check for exact match
        for entry in data_ref:
            if (abs(entry['conduit_size'] - conduit_size) < 1e-6 and
                abs(entry['production_rate'] - production_rate) < 1e-6 and
                abs(entry['glr'] - glr_input) < 1e-6):
                return entry['coefficients'], glr_input, glr_input, "exact"

        # Find rows for interpolation
        relevant_rows = [
            entry for entry in data_ref
            if (abs(entry['conduit_size'] - conduit_size) < 1e-6 and
                abs(entry['production_rate'] - production_rate) < 1e-6 and
                valid_range[0] <= entry['glr'] <= valid_range[1])
        ]
        if len(relevant_rows) < 1:
            st.error(f"No data points found for conduit size {conduit_size}, production rate {production_rate} in GLR range {valid_range}.")
            return None, None, None, None
        relevant_rows.sort(key=lambda x: x['glr'])

        if len(relevant_rows) == 1:
            if abs(relevant_rows[0]['glr'] - glr_input) < 1e-6:
                return relevant_rows[0]['coefficients'], glr_input, glr_input, "exact"
            else:
                st.error(f"Only one data point (GLR {relevant_rows[0]['glr']}) available for interpolation in range {valid_range}.")
                return None, None, None, None

        lower_row = None
        higher_row = None
        for entry in relevant_rows:
            if entry['glr'] <= glr_input:
                if lower_row is None or entry['glr'] > lower_row['glr']:
                    lower_row = entry
            if entry['glr'] >= glr_input:
                if higher_row is None or entry['glr'] < higher_row['glr']:
                    higher_row = entry
        if lower_row is None:
            lower_row = relevant_rows[0]
        if higher_row is None:
            higher_row = relevant_rows[-1]
        glr1 = lower_row['glr']
        glr2 = higher_row['glr']
        if glr1 == glr2:
            return lower_row['coefficients'], glr1, glr2, "exact"

        fraction = (glr_input - glr1) / (glr2 - glr1)
        coeffs = {
            'a': lower_row['coefficients']['a'] + fraction * (higher_row['coefficients']['a'] - lower_row['coefficients']['a']),
            'b': lower_row['coefficients']['b'] + fraction * (higher_row['coefficients']['b'] - lower_row['coefficients']['b']),
            'c': lower_row['coefficients']['c'] + fraction * (higher_row['coefficients']['c'] - lower_row['coefficients']['c']),
            'd': lower_row['coefficients']['d'] + fraction * (higher_row['coefficients']['d'] - lower_row['coefficients']['d']),
            'e': lower_row['coefficients']['e'] + fraction * (higher_row['coefficients']['e'] - lower_row['coefficients']['e']),
            'f': lower_row['coefficients']['f'] + fraction * (higher_row['coefficients']['f'] - lower_row['coefficients']['f'])
        }
        return coeffs, glr1, glr2, "interpolated"

    # Get coefficients for both production rates
    coeffs1, glr1_lower, glr1_higher, glr_status1 = get_coefficients(conduit_size_input, prate1, glr_input, valid_range1, data_ref)
    if coeffs1 is None:
        return None, None, None, None, None, None, None

    if production_interpolation_status == "exact":
        coeffs = coeffs1
        glr1 = glr1_lower
        glr2 = glr1_higher
        interpolation_status = glr_status1
    else:
        coeffs2, glr2_lower, glr2_higher, glr_status2 = get_coefficients(conduit_size_input, prate2, glr_input, valid_range2, data_ref)
        if coeffs2 is None:
            return None, None, None, None, None, None, None

        # Interpolate between production rates
        fraction_prate = (production_rate_input - prate1) / (prate2 - prate1)
        coeffs = {
            'a': coeffs1['a'] + fraction_prate * (coeffs2['a'] - coeffs1['a']),
            'b': coeffs1['b'] + fraction_prate * (coeffs2['b'] - coeffs1['b']),
            'c': coeffs1['c'] + fraction_prate * (coeffs2['c'] - coeffs1['c']),
            'd': coeffs1['d'] + fraction_prate * (coeffs2['d'] - coeffs1['d']),
            'e': coeffs1['e'] + fraction_prate * (coeffs2['e'] - coeffs1['e']),
            'f': coeffs1['f'] + fraction_prate * (coeffs2['f'] - coeffs1['f'])
        }
        glr1 = glr1_lower if glr_status1 == "exact" else min(glr1_lower, glr1_higher)
        glr2 = glr2_higher if glr_status2 == "exact" else max(glr2_lower, glr2_higher)
        interpolation_status = "interpolated" if glr_status1 == "interpolated" or glr_status2 == "interpolated" or production_interpolation_status == "interpolated" else "exact"

    def polynomial(x, coeffs):
        try:
            return (coeffs['a'] * x**5 + coeffs['b'] * x**4 + coeffs['c'] * x**3 +
                    coeffs['d'] * x**2 + coeffs['e'] * x + coeffs['f'])
        except Exception:
            return np.nan

    y1 = polynomial(p1, coeffs)
    if not np.isfinite(y1) or y1 < 0 or y1 > 31000:
        st.error(f"Computed y1 ({y1:.2f} ft) is invalid or outside valid depth range (0 to 31000 ft).")
        return None, None, None, None, None, None, None

    y2 = y1 + D
    if y2 > 31000:
        y2 = 31000
        D_adjusted = y2 - y1

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
            f_start = root_function(p_start, y2, coeffs)
            f_end = root_function(p_end, y2, coeffs)
            if np.isfinite(f_start) and np.isfinite(f_end) and f_start * f_end <= 0:
                try:
                    candidate = bisect(root_function, p_start, p_end, args=(y2, coeffs), maxiter=100)
                    y_candidate = polynomial(candidate, coeffs)
                    if np.isfinite(y_candidate) and 0 <= y_candidate <= 31000:
                        p2 = candidate
                        break
                except Exception:
                    continue
        except Exception:
            continue

    if p2 is None or p2 < 0 or p2 > 4000:
        st.error(f"No valid p2 found for y2 = {y2:.2f} ft within pressure range 0 to 4000 psi.")
        return None, None, None, None, None, None, None

    return y1, y2, p2, coeffs, interpolation_status, glr1, glr2

# Plotting function for polynomial results with dynamic cap
def plot_results(p1, y1, y2, p2, D, coeffs, glr_input, interpolation_status, production_rate):
    fig, ax = plt.subplots(figsize=(10, 6))
    p1_full = np.linspace(0, 4000, 100)
    def polynomial(x, coeffs):
        return (coeffs['a'] * x**5 + coeffs['b'] * x**4 + coeffs['c'] * x**3 +
                coeffs['d'] * x**2 + coeffs['e'] * x + coeffs['f'])
    y1_full = []
    crossing_x = None
    max_iterations = 100  # Added to prevent infinite loop
    iteration = 0
    for p in p1_full:
        if iteration >= max_iterations:
            break
        y = polynomial(p, coeffs)
        if np.isfinite(y) and y <= 31000:
            y1_full.append(y)
        else:
            if crossing_x is None and len(y1_full) > 0:
                def root_fn(x):
                    return polynomial(x, coeffs) - 31000
                try:
                    mid_guess = p1_full[max(0, len(y1_full) - 1)]
                    candidate = fsolve(root_fn, mid_guess, maxfev=20000)[0]
                    if 0 <= candidate <= 4000:
                        crossing_x = candidate
                        y1_full.append(31000)
                        break
                except Exception:
                    crossing_x = p1_full[len(y1_full) - 1]
                    y1_full.append(31000)
                    break
            else:
                y1_full.append(31000)
        iteration += 1
    ax.plot(p1_full[:len(y1_full)], y1_full, color='blue', linewidth=2.5,
            label=f'GLR curve ({"Interpolated" if interpolation_status == "interpolated" else "Exact"}, Q0={production_rate} stb/day, GLR={glr_input})')
    ax.scatter([p1], [y1], color='blue', s=50, label=f'(p1, y1) = ({p1:.2f} psi, {y1:.2f} ft)')
    ax.scatter([p2], [y2], color='blue', s=50, label=f'(p2, y2) = ({p2:.2f} psi, {y2:.2f} ft)')
    ax.plot([p1, p1], [y1, 0], color='red', linewidth=1, label='Connecting Line')
    ax.plot([p1, 0], [y1, y1], color='red', linewidth=1)
    ax.plot([p2, p2], [y2, 0], color='red', linewidth=1)
    ax.plot([p2, 0], [y2, y2], color='red', linewidth=1)
    ax.plot([0, 0], [y1, y2], color='green', linewidth=4, label=f'Well Length ({D:.2f} ft)')
    ax.set_xlabel('Gradient Pressure, psi', fontsize=10)
    ax.set_ylabel('Depth, ft', fontsize=10)
    ax.set_xlim(0, 4000)
    ax.set_ylim(0, 31000)
    ax.invert_yaxis()
    ax.grid(True, which='major', color='#D3D3D3')
    ax.grid(True, which='minor', color='#D3D3D3', linestyle='-', alpha=0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(200))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1000))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(200))
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, frameon=True, edgecolor='black')
    return fig

# Modular function to calculate TPR points
def calculate_tpr_points(conduit_size, glr, D, pwh, data_ref):
    production_rates = [50, 100, 200, 400, 600]
    tpr_points = []
    for prate in production_rates:
        result = calculate_results(conduit_size, prate, glr, pwh, D, data_ref)
        if result[0] is None:
            raise ValueError(f"Failed to compute p2 for production rate {prate} stb/day.")
        y1, y2, p2, coeffs, interpolation_status, glr1, glr2 = result
        tpr_points.append((prate, p2))
    return tpr_points

# Modular function to calculate IPR parameters and points using Fetkovich method
def calculate_ipr_fetkovich(pr, c=None, n=None, q01=None, pwf1=None, q02=None, pwf2=None, q03=None, pwf3=None, q04=None, pwf4=None):
    points = None
    if c is None or n is None:
        points = []
        if q01 is not None and pwf1 is not None and q01 > 0 and pwf1 > 0:
            points.append((q01, pwf1))
        if q02 is not None and pwf2 is not None and q02 > 0 and pwf2 > 0:
            points.append((q02, pwf2))
        if q03 is not None and pwf3 is not None and q03 > 0 and pwf3 > 0:
            points.append((q03, pwf3))
        if q04 is not None and pwf4 is not None and q04 > 0 and pwf4 > 0:
            points.append((q04, pwf4))

        if len(points) < 2:
            raise ValueError("At least two valid points (Q0 > 0, Pwf > 0) required for Fetkovich parameters.")

        if len(points) == 2:
            # Two-point method with corrected formula
            q01, pwf1 = points[0]
            q02, pwf2 = points[1]
            if pwf1 == pwf2 or q01 == q02 or pwf1 <= 0 or pwf2 <= 0 or q01 <= 0 or q02 <= 0:
                raise ValueError("Invalid Fetkovich input parameters: Pwf1, Pwf2, Q01, Q02 must be positive and distinct.")
            delta_p1 = pr**2 - pwf1**2
            delta_p2 = pr**2 - pwf2**2
            if delta_p1 <= 0 or delta_p2 <= 0 or delta_p1 == delta_p2:
                raise ValueError("Invalid delta pressures: Pr^2 - Pwf^2 must be positive and distinct.")
            n = np.log10(q02 / q01) / np.log10(delta_p2 / delta_p1)
            c = q01 / (delta_p1 ** n)
        else:
            # Multi-point method using regression
            q_points_list, pwf_points_list = zip(*points)
            q_points_array = np.array(q_points_list)
            pwf_points_array = np.array(pwf_points_list)
            def fetkovich_model(pwf, c, n):
                return c * (pr**2 - pwf**2)**n
            popt, _ = curve_fit(fetkovich_model, pwf_points_array, q_points_array, p0=[1e-5, 0.5], maxfev=10000)
            c, n = popt

            # Draw log-log plot for multi-point fit
            st.write("Generating Fetkovich log-log plot for multi-point fit...")
            delta_p = pr**2 - pwf_points_array**2
            x = np.log10(delta_p)
            y = np.log10(q_points_array)
            fig_log, ax_log = plt.subplots(figsize=(8, 6))
            ax_log.scatter(x, y, color='blue', s=50, label='Data Points')
            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = np.log10(c) + n * x_fit
            ax_log.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit: n={n:.4f}, C={c:.4e}')
            ax_log.set_xlabel('log(Pr² - Pwf²)', fontsize=10)
            ax_log.set_ylabel('log(Q0)', fontsize=10)
            ax_log.set_title('Fetkovich Log-Log Plot')
            ax_log.legend(loc='upper left')
            ax_log.grid(True)
            st.pyplot(fig_log)

            # Draw flow after flow plot if 4 points are provided
            if len(points) == 4:
                st.write("Generating Flow After Flow Plot...")
                fig_flow, ax_flow = plt.subplots(figsize=(8, 6))
                ax_flow.scatter(pwf_points_array, q_points_array, color='blue', s=50, label='Test Points')
                pwf_range = np.linspace(0, pr, 100)
                q_flow = fetkovich_model(pwf_range, c, n)
                ax_flow.plot(pwf_range, q_flow, 'r-', linewidth=2, label=f'IPR Fit: n={n:.4f}, C={c:.4e}')
                ax_flow.set_xlabel('Flowing Bottomhole Pressure (Pwf, psi)', fontsize=10)
                ax_flow.set_ylabel('Production Rate (Q0, stb/day)', fontsize=10)
                ax_flow.set_title('Flow After Flow Test Results')
                ax_flow.legend(loc='upper left')
                ax_flow.grid(True)
                st.pyplot(fig_flow)

    # Validate c and n
    if c <= 0 or not np.isfinite(c) or n <= 0 or n > 2.0 or not np.isfinite(n):
        st.error(f"Invalid Fetkovich parameters: C={c}, n={n}. C must be positive, n must be in (0, 2].")
        raise ValueError(f"Invalid Fetkovich parameters: C={c}, n={n}")

    # Generate IPR points (for both direct and point-based inputs)
    pwf_values = np.linspace(0, pr, 15)
    ipr_points = []
    for pwf in pwf_values:
        q0 = c * (pr**2 - pwf**2)**n
        if np.isfinite(q0) and q0 >= 0:  # Relaxed q0 upper limit
            ipr_points.append((q0, pwf))
    if len(ipr_points) < 2:
        raise ValueError("Insufficient valid IPR points to plot (need at least 2).")
    return c, n, ipr_points, points

    
# Modular function to calculate IPR parameters and points using Vogel method
def calculate_ipr_vogel(pr, q_max):
    pwf_values = np.linspace(0, pr, 15)
    ipr_points = []
    for pwf in pwf_values:
        q0 = q_max * (1 - 0.2 * (pwf / pr) - 0.8 * (pwf / pr)**2)
        if np.isfinite(q0) and 0 <= q0 <= 1000:
            ipr_points.append((q0, pwf))
    if len(ipr_points) < 2:
        raise ValueError("Insufficient valid IPR points to plot (need at least 2).")
    return q_max, ipr_points

# Modular function to calculate IPR parameters and points using Composite method
def calculate_ipr_composite(pr, j_star, p_b):
    pwf_values = np.linspace(0, pr, 15)
    ipr_points = []
    for pwf in pwf_values:
        if pwf > p_b:
            q0 = j_star * (pr - pwf)
        else:
            q0 = j_star * (pr - p_b) + (j_star * p_b / 1.8) * (1 - 0.2 * (pwf / p_b) - 0.8 * (pwf / p_b)**2)
        if np.isfinite(q0) and 0 <= q0 <= 1000:
            ipr_points.append((q0, pwf))
    if len(ipr_points) < 2:
        raise ValueError("Insufficient valid IPR points to plot (need at least 2).")
    return j_star, p_b, ipr_points

# Modular function to find intersection with feasibility check and bisect solver
def find_intersection(tpr_points, ipr_points, pr):
    # Interpolate IPR as Pwf vs Q0
    ipr_q0, ipr_pwf = zip(*ipr_points)
    ipr_interp = interp1d(ipr_q0, ipr_pwf, kind='linear', fill_value='extrapolate')

    # Interpolate TPR as Pwf vs Q0
    tpr_q0, tpr_p2 = zip(*tpr_points)
    tpr_interp = interp1d(tpr_q0, tpr_p2, kind='linear', fill_value='extrapolate')

    # Define difference function in Q0 space
    def diff_func(q):
        try:
            return ipr_interp(q) - tpr_interp(q)
        except Exception as e:
            st.write(f"Intersection function error at q={q}: {str(e)}")
            return np.inf

    # Find root between min and max Q0 ranges
    try:
        q_min = max(min(ipr_q0), min(tpr_q0))
        q_max = min(max(ipr_q0), max(tpr_q0))
        if q_min >= q_max:
            st.write(f"Invalid Q0 range: q_min={q_min:.2f}, q_max={q_max:.2f}")
            return None, None
        f_min = diff_func(q_min)
        f_max = diff_func(q_max)
        if not (np.isfinite(f_min) and np.isfinite(f_max)):
            st.write(f"Non-finite function values: f_min={f_min}, f_max={f_max}")
            return None, None
        if f_min * f_max > 0:
            st.write(f"No sign change detected: f_min={f_min:.2f}, f_max={f_max:.2f}")
            return None, None
        intersection_q0 = bisect(diff_func, q_min, q_max, maxiter=200)
        intersection_p = ipr_interp(intersection_q0)
        # Validate intersection
        if (0 <= intersection_q0 <= 750 and 0 <= intersection_p <= max(pr, 4000) and
            abs(ipr_interp(intersection_q0) - tpr_interp(intersection_q0)) < 1.0):
            st.write(f"Intersection found: Q0={intersection_q0:.2f} stb/day, P={intersection_p:.2f} psi")
            return intersection_q0, intersection_p
        else:
            st.write(f"Intersection rejected: Q0={intersection_q0:.2f}, P={intersection_p:.2f}, "
                     f"diff={abs(ipr_interp(intersection_q0) - tpr_interp(intersection_q0)):.4f}")
            return None, None
    except Exception as e:
        st.write(f"Intersection search failed: {str(e)}")
        return None, None

# Modular function to plot TPR and IPR curves
def plot_curves(tpr_points, ipr_points, intersection_q0, intersection_p, conduit_size, glr, D, pwh, pr, ipr_params):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot TPR
    tpr_q0, tpr_p2 = zip(*tpr_points)
    ax.plot(tpr_q0, tpr_p2, 'b-o', linewidth=2, label=f'TPR (Conduit: {conduit_size} in, GLR: {glr}, D: {D} ft, Pwh: {pwh} psi)')

    # Plot IPR
    ipr_q0, ipr_pwf = zip(*ipr_points)
    ax.plot(ipr_q0, ipr_pwf, 'r-', linewidth=2, label=f'IPR (Pr: {pr} psi, Params: {ipr_params})')

    # Plot intersection point if found
    if intersection_p is not None and intersection_q0 is not None:
        ax.scatter([intersection_q0], [intersection_p], color='green', s=100, marker='*',
                   label=f'Natural Flow Point (Q0: {intersection_q0:.2f} stb/day, P: {intersection_p:.2f} psi)')

    # Graph settings
    ax.set_xlabel('Production Rate, Q0 (stb/day)', fontsize=10)
    ax.set_ylabel('Pressure, psi', fontsize=10)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, max(pr, 4000))
    ax.grid(True, which='major', color='#D3D3D3')
    ax.grid(True, which='minor', color='#D3D3D3', linestyle='-', alpha=0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1000))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(200))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, frameon=True, edgecolor='black')
    ax.set_title('TPR and IPR Curves with Natural Flow Point')
    return fig

# Function to calculate Fetkovich parameters and plot TPR and IPR lines
def plot_natural_flow(conduit_size, glr, D, pwh, pr, ipr_method, ipr_params, data_ref=None):
    st.write("Starting plot_natural_flow function")

    # Validate inputs
    if not (conduit_size in [2.875, 3.5]):
        st.error("Invalid conduit size. Must be 2.875 or 3.5.")
        return None, None, None
    if D <= 0:
        st.error("Well length D must be positive.")
        return None, None, None
    if pwh < 0 or pwh > 4000:
        st.error("Wellhead pressure Pwh must be between 0 and 4000 psi.")
        return None, None, None
    if pr <= 0:
        st.error("Reservoir pressure Pr must be positive.")
        return None, None, None

    # Calculate TPR points
    try:
        tpr_points = calculate_tpr_points(conduit_size, glr, D, pwh, data_ref)
        st.write(f"TPR points calculated: {tpr_points}")
    except ValueError as e:
        st.error(str(e))
        return None, None, None

    # Calculate IPR based on method
    try:
        if ipr_method == 'fetkovich':
            c, n, ipr_points, fetkovich_points = calculate_ipr_fetkovich(pr, **ipr_params)
            ipr_params_str = f'n={n:.4f}, C={c:.4e}'
            def ipr_func(p):
                return c * (pr**2 - p**2)**n
        elif ipr_method == 'vogel':
            q_max = ipr_params['q_max']
            ipr_points = calculate_ipr_vogel(pr, q_max)[1]
            ipr_params_str = f'q_max={q_max:.4f}'
            def ipr_func(p):
                return q_max * (1 - 0.2 * (p / pr) - 0.8 * (p / pr)**2)
        elif ipr_method == 'composite':
            j_star = ipr_params['j_star']
            p_b = ipr_params['p_b']
            ipr_points = calculate_ipr_composite(pr, j_star, p_b)[2]
            ipr_params_str = f'J*={j_star:.4f}, P_b={p_b:.4f}'
            def ipr_func(p):
                if p > p_b:
                    return j_star * (pr - p)
                else:
                    return j_star * (pr - p_b) + (j_star * p_b / 1.8) * (1 - 0.2 * (p / p_b) - 0.8 * (p / p_b)**2)
        else:
            raise ValueError("Invalid IPR method selected.")
    except ValueError as e:
        st.error(str(e))
        return None, None, None

    # Find intersection with feasibility check
    intersection_q0, intersection_p = find_intersection(tpr_points, ipr_points, pr)
    if intersection_q0 is None:
        st.warning("Well cannot flow naturally; artificial lift required. Showing TPR and IPR curves for visualization.")

    # Plot curves regardless of intersection
    try:
        fig = plot_curves(tpr_points, ipr_points, intersection_q0, intersection_p, conduit_size, glr, D, pwh, pr, ipr_params_str)
        return fig, intersection_q0, intersection_p
    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")
        return None, None, None

# Post-task menu to handle repeat, change task, or exit
def post_task_menu(mode):
    st.subheader("Next Action")
    next_action = st.selectbox("Choose:", ["Repeat Task", "Choose Another Task", "End Program"], key=f"next_action_{mode}")
    if next_action == "Repeat Task":
        # No change, app will rerun with same mode
        pass
    elif next_action == "Choose Another Task":
        st.session_state.mode = None  # Allow mode selection again
    else:  # End Program
        st.write("Program terminated.")

# p2 Finder task with validation for p1 and D
def run_p2_finder():
    st.header("p2 Finder")
    st.write("Enter parameters to calculate pressure and depth values using polynomial formulas.")
    conduit_size = st.selectbox("Conduit Size:", [2.875, 3.5])
    production_rate = st.number_input("Production Rate (stb/day, interpolated between 50 and 600):", value=100.0, min_value=50.0, max_value=600.0)
    glr = st.number_input("GLR:", value=200.0, min_value=0.0)
    p1 = st.number_input("Pressure p1 (psi):", value=1000.0, min_value=0.0, max_value=4000.0)
    D = st.number_input("Depth Offset D (ft):", value=1000.0, min_value=0.0)

    # Validate GLR
    valid_glr = False
    for prate in PRODUCTION_RATES:
        if (conduit_size, prate) in INTERPOLATION_RANGES:
            for min_glr, max_glr in INTERPOLATION_RANGES[(conduit_size, prate)]:
                if min_glr <= glr <= max_glr:
                    valid_glr = True
                    break
            if valid_glr:
                break
    if not valid_glr:
        st.error(f"GLR {glr} is outside valid ranges for conduit size {conduit_size} and available production rates.")
        return

    if st.button("Calculate"):
        if st.session_state.REFERENCE_DATA is None:
            st.error("Reference data not loaded. Please restart the program and upload the reference Excel file.")
            return

        # Validate p1 by checking if y1 <= 31000
        result = calculate_results(conduit_size, production_rate, glr, p1, D, st.session_state.REFERENCE_DATA)
        if result[0] is None:
            st.error("Invalid input parameters (e.g., GLR or production rate). Please check and try again.")
            return
        y1 = result[0]

        if y1 > 31000:
            st.error(f"Invalid pressure: p1 = {p1} psi results in y1 = {y1:.2f} ft, which exceeds 31000 ft.")
            return

        # Validate D by checking if y1 + D <= 31000
        if y1 + D > 31000:
            st.error(f"Invalid well length: D = {D} ft results in y1 + D = {y1 + D:.2f} ft, which exceeds 31000 ft.")
            return

        # Proceed with calculation
        y1, y2, p2, coeffs, interpolation_status, glr1, glr2 = result
        st.subheader("Results")
        st.write(f"Conduit Size: {conduit_size}")
        st.write(f"Production Rate: {production_rate} stb/day")
        st.write(f"GLR: {glr}")
        st.write(f"Pressure p1: {p1} psi")
        st.write(f"Depth Offset D: {D} ft")
        st.write("p2 Finder Results")
        if interpolation_status == "exact":
            st.write("Using exact polynomial coefficients from data.")
        else:
            st.write(f"Interpolated polynomial coefficients between GLR {glr1} and {glr2}.")
        st.write(f"Depth y1 at p1: {y1:.2f} ft")
        st.write(f"Target Depth y2: {y2:.2f} ft")
        st.write(f"Pressure p2: {p2:.2f} psi")
        st.write("Pressure vs Depth Plot")
        fig = plot_results(p1, y1, y2, p2, D, coeffs, glr, interpolation_status, production_rate)
        st.pyplot(fig)
        post_task_menu("p2 Finder")

# Point of Natural Flow Finder task
def run_natural_flow_finder():
    st.header("Point of Natural Flow Finder")
    st.write("Enter parameters to find the point of natural flow using TPR and IPR curves.")
    conduit_size = st.selectbox("Conduit Size:", [2.875, 3.5])
    glr = st.number_input("GLR:", value=200.0, min_value=0.0)
    D = st.number_input("Well Length D (ft):", value=1000.0, min_value=1e-6)
    pwh = st.number_input("Wellhead Pressure Pwh (psi):", value=1000.0, min_value=0.0, max_value=4000.0)
    pr = st.number_input("Reservoir Pressure Pr (psi):", value=3000.0, min_value=1e-6)
    ipr_method = st.selectbox("IPR Method:", ["Fetkovich", "Vogel", "Composite"])

    # Validate GLR
    valid_glr = False
    for prate in PRODUCTION_RATES:
        if (conduit_size, prate) in INTERPOLATION_RANGES:
            for min_glr, max_glr in INTERPOLATION_RANGES[(conduit_size, prate)]:
                if min_glr <= glr <= max_glr:
                    valid_glr = True
                    break
            if valid_glr:
                break
    if not valid_glr:
        st.error(f"GLR {glr} is outside valid ranges for conduit size {conduit_size} and available production rates.")
        return

    fetkovich_method = None
    c = None
    n = None
    q01 = None
    pwf1 = None
    q02 = None
    pwf2 = None
    q03 = None
    pwf3 = None
    q04 = None
    pwf4 = None
    q_max = None
    j_star = None
    p_b = None

    if ipr_method == "Fetkovich":
        fetkovich_method = st.selectbox("Fetkovich Method:", ["Enter C and n directly", "Calculate C and n from points"])
        if fetkovich_method == "Enter C and n directly":
            c = st.number_input("C:", value=1e-5, step=1e-6, format="%.6e", min_value=1e-10)
            n = st.number_input("n:", value=0.5, step=0.01, min_value=1e-6)
        else:
            q01 = st.number_input("Q01 (stb/day):", value=100.0, min_value=1e-6)
            pwf1 = st.number_input("Pwf1 (psi):", value=2000.0, min_value=1e-6)
            q02 = st.number_input("Q02 (stb/day):", value=200.0, min_value=1e-6)
            pwf2 = st.number_input("Pwf2 (psi):", value=1500.0, min_value=1e-6)
            q03 = st.number_input("Q03 (stb/day):", value=0.0, min_value=0.0)
            pwf3 = st.number_input("Pwf3 (psi):", value=0.0, min_value=0.0)
            q04 = st.number_input("Q04 (stb/day):", value=0.0, min_value=0.0)
            pwf4 = st.number_input("Pwf4 (psi):", value=0.0, min_value=0.0)
    elif ipr_method == "Vogel":
        q_max = st.number_input("q_max (stb/day):", value=500.0, min_value=1e-6)
    elif ipr_method == "Composite":
        j_star = st.number_input("J* (stb/day/psi):", value=0.5, min_value=1e-6)
        p_b = st.number_input("P_b (psi):", value=2000.0, min_value=1e-6)

    if st.button("Calculate and Plot"):
        if st.session_state.REFERENCE_DATA is None:
            st.error("Reference data not loaded. Please restart the program and upload the reference Excel file.")
            return

        # Validate depth using a sample production rate
        sample_prate = PRODUCTION_RATES[0]  # Use first production rate for validation
        result = calculate_results(conduit_size, sample_prate, glr, pwh, D, st.session_state.REFERENCE_DATA)
        if result[0] is None:
            st.error("Invalid input parameters (e.g., GLR or production rate). Please check and try again.")
            return
        y1 = result[0]
        if y1 > 31000:
            st.error(f"Invalid wellhead pressure: Pwh = {pwh} psi results in y1 = {y1:.2f} ft, which exceeds 31000 ft.")
            return
        if y1 + D > 31000:
            st.error(f"Invalid well length: D = {D} ft results in y1 + D = {y1 + D:.2f} ft, which exceeds 31000 ft.")
            return

        st.write("Processing inputs...")
        try:
            ipr_params = {}
            if ipr_method == "Fetkovich":
                if fetkovich_method == "Enter C and n directly":
                    ipr_params['c'] = float(c)
                    ipr_params['n'] = float(n)
                else:
                    ipr_params['q01'] = float(q01)
                    ipr_params['pwf1'] = float(pwf1)
                    ipr_params['q02'] = float(q02)
                    ipr_params['pwf2'] = float(pwf2)
                    ipr_params['q03'] = float(q03)
                    ipr_params['pwf3'] = float(pwf3)
                    ipr_params['q04'] = float(q04)
                    ipr_params['pwf4'] = float(pwf4)
            elif ipr_method == "Vogel":
                ipr_params['q_max'] = float(q_max)
            elif ipr_method == "Composite":
                ipr_params['j_star'] = float(j_star)
                ipr_params['p_b'] = float(p_b)

            st.write("Calling plot_natural_flow...")
            fig, intersection_q0, intersection_p = plot_natural_flow(
                conduit_size,
                glr,
                D,
                pwh,
                pr,
                ipr_method.lower(),
                ipr_params,
                st.session_state.REFERENCE_DATA
            )
            if fig is not None:
                st.subheader("Point of Natural Flow Results")
                st.write(f"IPR Method: {ipr_method}")
                if intersection_q0 is not None and intersection_p is not None:
                    st.write(f"Natural Flow Point: Q0 = {intersection_q0:.2f} stb/day, P = {intersection_p:.2f} psi")
                else:
                    st.write("Could not determine the natural flow point.")
                st.write("TPR and IPR Curves (Intersection indicates Point of Natural Flow)")
                st.pyplot(fig)
            else:
                st.error("Failed to generate plot. Please check inputs and try again.")
            post_task_menu("Point of Natural Flow Finder")
        except Exception as e:
            st.error(f"Error processing inputs: {str(e)}")

# Main UI function to handle initial data upload and mode selection
def main():
    st.title("Well Pressure and Depth Calculator")
    st.write("Well Pressure and Depth Calculator")
    if st.session_state.REFERENCE_DATA is None:
        st.session_state.REFERENCE_DATA = load_reference_data()
        if st.session_state.REFERENCE_DATA is None:
            st.error("Cannot proceed without reference data. Please try again.")
            return

    mode = st.sidebar.selectbox("Select Mode:", ["p2 Finder", "Point of Natural Flow Finder"])
    if mode == "p2 Finder":
        run_p2_finder()
    else:
        run_natural_flow_finder()

if __name__ == "__main__":
    main()
