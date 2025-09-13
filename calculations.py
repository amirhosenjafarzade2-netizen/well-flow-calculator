import streamlit as st
import numpy as np
from scipy.optimize import root_scalar, curve_fit
from scipy.interpolate import interp1d
from config import INTERPOLATION_RANGES, PRODUCTION_RATES
from utils import polynomial, setup_logging
from validators import validate_conduit_size, validate_production_rate, validate_glr

# Initialize logger
logger = setup_logging()

@st.cache_data
def calculate_results(conduit_size, production_rate, glr_input, p1, D, data_ref):
    """
    Calculate depths (y1, y2) and pressure (p2) based on polynomial interpolation.
    Returns (y1, y2, p2, coeffs, interpolation_status, glr1, glr2) or (None, ...) if invalid.
    """
    # Validate inputs
    if not validate_conduit_size(conduit_size):
        logger.error("Invalid conduit size in calculate_results.")
        return None, None, None, None, None, None, None
    if not validate_production_rate(production_rate):
        logger.error("Invalid production rate in calculate_results.")
        return None, None, None, None, None, None, None

    # Find closest production rates for interpolation
    lower_prate = max([pr for pr in PRODUCTION_RATES if pr <= production_rate], default=50)
    higher_prate = min([pr for pr in PRODUCTION_RATES if pr >= production_rate], default=600)
    production_interpolation_status = "exact" if abs(lower_prate - higher_prate) < 1e-6 else "interpolated"
    prate1, prate2 = lower_prate, higher_prate

    # Validate GLR for both production rates
    valid_glr1, valid_range1 = False, None
    ranges1 = INTERPOLATION_RANGES.get((conduit_size, prate1), [])
    for min_glr, max_glr in ranges1:
        if min_glr <= glr_input <= max_glr:
            valid_glr1, valid_range1 = True, (min_glr, max_glr)
            break
    if not valid_glr1:
        logger.error(f"GLR {glr_input} outside valid ranges for conduit {conduit_size}, production {prate1}.")
        return None, None, None, None, None, None, None

    valid_glr2, valid_range2 = False, None
    ranges2 = INTERPOLATION_RANGES.get((conduit_size, prate2), [])
    for min_glr, max_glr in ranges2:
        if min_glr <= glr_input <= max_glr:
            valid_glr2, valid_range2 = True, (min_glr, max_glr)
            break
    if not valid_glr2:
        logger.error(f"GLR {glr_input} outside valid ranges for conduit {conduit_size}, production {prate2}.")
        return None, None, None, None, None, None, None

    # Check for extrapolation risks
    if valid_glr1 and (glr_input < valid_range1[0] * 1.05 or glr_input > valid_range1[1] * 0.95):
        logger.warning(f"GLR {glr_input} is near edge of range {valid_range1} for production {prate1}.")
        st.warning(f"GLR {glr_input} is near the edge of valid range {valid_range1}. Results may be less accurate.")
    if valid_glr2 and (glr_input < valid_range2[0] * 1.05 or glr_input > valid_range2[1] * 0.95):
        logger.warning(f"GLR {glr_input} is near edge of range {valid_range2} for production {prate2}.")
        st.warning(f"GLR {glr_input} is near the edge of valid range {valid_range2}. Results may be less accurate.")

    def get_coefficients(conduit_size, production_rate, glr_input, valid_range, data_ref):
        """
        Get polynomial coefficients by interpolating between GLR values.
        Returns (coeffs, glr1, glr2, status) or (None, ...) if invalid.
        """
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
            logger.error(f"No data points for conduit {conduit_size}, production {production_rate}, GLR range {valid_range}.")
            st.error(f"No data points found for conduit size {conduit_size}, production rate {production_rate} in GLR range {valid_range}.")
            return None, None, None, None
        relevant_rows.sort(key=lambda x: x['glr'])

        if len(relevant_rows) == 1:
            if abs(relevant_rows[0]['glr'] - glr_input) < 1e-6:
                return relevant_rows[0]['coefficients'], glr_input, glr_input, "exact"
            logger.error(f"Only one GLR point ({relevant_rows[0]['glr']}) available for interpolation.")
            st.error(f"Only one GLR point ({relevant_rows[0]['glr']}) available in range {valid_range}.")
            return None, None, None, None

        lower_row = max([r for r in relevant_rows if r['glr'] <= glr_input], key=lambda x: x['glr'], default=relevant_rows[0])
        higher_row = min([r for r in relevant_rows if r['glr'] >= glr_input], key=lambda x: x['glr'], default=relevant_rows[-1])
        glr1, glr2 = lower_row['glr'], higher_row['glr']
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
    coeffs1, glr1_lower, glr1_higher, glr_status1 = get_coefficients(conduit_size, prate1, glr_input, valid_range1, data_ref)
    if coeffs1 is None:
        return None, None, None, None, None, None, None

    if production_interpolation_status == "exact":
        coeffs, glr1, glr2, interpolation_status = coeffs1, glr1_lower, glr1_higher, glr_status1
    else:
        coeffs2, glr2_lower, glr2_higher, glr_status2 = get_coefficients(conduit_size, prate2, glr_input, valid_range2, data_ref)
        if coeffs2 is None:
            return None, None, None, None, None, None, None
        fraction_prate = (production_rate - prate1) / (prate2 - prate1)
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
        interpolation_status = "interpolated" if glr_status1 == "interpolated" or glr_status2 == "interpolated" else "exact"

    # Calculate y1
    y1 = polynomial(p1, coeffs)
    if not np.isfinite(y1) or y1 < 0 or y1 > 31000:
        logger.error(f"Computed y1 ({y1:.2f} ft) is invalid or outside range 0 to 31000 ft.")
        st.error(f"Computed depth y1 ({y1:.2f} ft) is invalid or exceeds 31000 ft. Please adjust inputs.")
        return None, None, None, None, None, None, None

    y2 = y1 + D
    if y2 > 31000:
        y2 = 31000
        D_adjusted = y2 - y1
        logger.info(f"Adjusted y2 to {y2:.2f} ft, D_adjusted = {D_adjusted:.2f} ft to stay within 31000 ft.")

    # Find p2 using robust root-finding
    def root_function(x, target_depth, coeffs):
        return polynomial(x, coeffs) - target_depth

    p2 = None
    try:
        result = root_scalar(
            root_function, args=(y2, coeffs),
            bracket=[p1, 4000], method='brentq'
        )
        if result.converged:
            p2 = result.root
    except ValueError:
        try:
            result = root_scalar(
                root_function, args=(y2, coeffs),
                x0=p1, method='secant'
            )
            if result.converged:
                p2 = result.root
        except ValueError:
            logger.error(f"No valid p2 found for y2 = {y2:.2f} ft within 0 to 4000 psi.")
            st.error(f"No valid pressure p2 found for depth y2 = {y2:.2f} ft. Please adjust inputs.")
            return None, None, None, None, None, None, None

    if p2 is None or p2 < 0 or p2 > 4000:
        logger.error(f"Computed p2 ({p2:.2f} psi) is invalid or outside range 0 to 4000 psi.")
        st.error(f"Computed pressure p2 ({p2:.2f} psi) is invalid or exceeds 4000 psi. Please adjust inputs.")
        return None, None, None, None, None, None, None

    logger.info(f"Calculated: y1={y1:.2f} ft, y2={y2:.2f} ft, p2={p2:.2f} psi, interpolation_status={interpolation_status}")
    return y1, y2, p2, coeffs, interpolation_status, glr1, glr2

@st.cache_data
def calculate_tpr_points(conduit_size, glr, D, pwh, data_ref):
    """
    Calculate TPR points for given conduit size, GLR, depth, and wellhead pressure.
    Returns list of (production_rate, p2) tuples.
    """
    if not validate_conduit_size(conduit_size):
        raise ValueError("Invalid conduit size.")
    if not validate_glr(conduit_size, PRODUCTION_RATES[0], glr):
        raise ValueError(f"GLR {glr} is invalid for conduit size {conduit_size}.")
    tpr_points = []
    for prate in PRODUCTION_RATES:
        result = calculate_results(conduit_size, prate, glr, pwh, D, data_ref)
        if result[0] is None:
            logger.warning(f"Failed to compute p2 for production rate {prate} stb/day.")
            continue
        y1, y2, p2, coeffs, interpolation_status, glr1, glr2 = result
        tpr_points.append((prate, p2))
    if not tpr_points:
        logger.error("No valid TPR points computed.")
        raise ValueError("No valid TPR points computed.")
    logger.info(f"TPR points: {tpr_points}")
    return tpr_points

@st.cache_data
def calculate_ipr_fetkovich(pr, c=None, n=None, q01=None, pwf1=None, q02=None, pwf2=None, q03=None, pwf3=None, q04=None, pwf4=None):
    """
    Calculate IPR parameters and points using Fetkovich method.
    If c and n are provided, use them directly; otherwise, calculate from points.
    Returns (c, n, ipr_points, fetkovich_points).
    """
    points = []
    if q01 is not None and pwf1 is not None and q01 > 0 and pwf1 > 0:
        points.append((q01, pwf1))
    if q02 is not None and pwf2 is not None and q02 > 0 and pwf2 > 0:
        points.append((q02, pwf2))
    if q03 is not None and pwf3 is not None and q03 > 0 and pwf3 > 0:
        points.append((q03, pwf3))
    if q04 is not None and pwf4 is not None and q04 > 0 and pwf4 > 0:
        points.append((q04, pwf4))

    if c is not None and n is not None:
        # Validate provided c and n
        if c <= 0 or not np.isfinite(c) or n <= 0 or n > 2.0 or not np.isfinite(n):
            logger.error(f"Invalid Fetkovich parameters: C={c}, n={n}.")
            st.error(f"Invalid Fetkovich parameters: C={c}, n={n}. C must be positive, n must be in (0, 2].")
            raise ValueError(f"Invalid Fetkovich parameters: C={c}, n={n}")
    else:
        # Calculate c and n from points
        if len(points) < 2:
            logger.error("At least two valid points required for Fetkovich parameters.")
            st.error("At least two valid points (Q0 > 0, Pwf > 0) required for Fetkovich method. Please provide valid inputs.")
            raise ValueError("Insufficient valid points for Fetkovich calculation.")
        
        if len(points) == 2:
            q01, pwf1 = points[0]
            q02, pwf2 = points[1]
            if pwf1 == pwf2 or q01 == q02 or pwf1 <= 0 or pwf2 <= 0 or q01 <= 0 or q02 <= 0:
                logger.error("Invalid Fetkovich inputs: Pwf1, Pwf2, Q01, Q02 must be positive and distinct.")
                st.error("Invalid Fetkovich inputs: Pwf1, Pwf2, Q01, Q02 must be positive and distinct.")
                raise ValueError("Invalid Fetkovich input parameters.")
            delta_p1 = pr**2 - pwf1**2
            delta_p2 = pr**2 - pwf2**2
            if delta_p1 <= 0 or delta_p2 <= 0 or delta_p1 == delta_p2:
                logger.error("Invalid delta pressures for Fetkovich calculation.")
                st.error("Invalid pressures: Pr^2 - Pwf^2 must be positive and distinct.")
                raise ValueError("Invalid delta pressures.")
            n = np.log10(q02 / q01) / np.log10(delta_p2 / delta_p1)
            c = q01 / (delta_p1 ** n)
        else:
            q_points_list, pwf_points_list = zip(*points)
            q_points_array = np.array(q_points_list)
            pwf_points_array = np.array(pwf_points_list)
            def fetkovich_model(pwf, c, n):
                return c * (pr**2 - pwf**2)**n
            try:
                popt, _ = curve_fit(fetkovich_model, pwf_points_array, q_points_array, p0=[1e-5, 0.5], maxfev=10000)
                c, n = popt
            except Exception as e:
                logger.error(f"Curve fit failed for Fetkovich: {str(e)}")
                st.error(f"Failed to compute Fetkovich parameters: {str(e)}. Please check input points.")
                raise ValueError("Curve fit failed.")

        if c <= 0 or not np.isfinite(c) or n <= 0 or n > 2.0 or not np.isfinite(n):
            logger.error(f"Invalid Fetkovich parameters: C={c}, n={n}.")
            st.error(f"Invalid Fetkovich parameters: C={c}, n={n}. C must be positive, n must be in (0, 2].")
            raise ValueError(f"Invalid Fetkovich parameters: C={c}, n={n}")

    # Generate IPR points
    pwf_values = np.linspace(0, pr, 15)
    ipr_points = []
    for pwf in pwf_values:
        q0 = c * (pr**2 - pwf**2)**n
        if np.isfinite(q0) and q0 >= 0:
            ipr_points.append((q0, pwf))
    if len(ipr_points) < 2:
        logger.error("Insufficient valid IPR points for Fetkovich.")
        st.error("Insufficient valid IPR points generated. Please adjust inputs.")
        raise ValueError("Insufficient valid IPR points.")

    logger.info(f"Fetkovich parameters: C={c:.4e}, n={n:.4f}, points={len(ipr_points)}")
    return c, n, ipr_points, points

@st.cache_data
def calculate_ipr_vogel(pr, q_max):
    """
    Calculate IPR points using Vogel method.
    Returns (q_max, ipr_points).
    """
    pwf_values = np.linspace(0, pr, 15)
    ipr_points = []
    for pwf in pwf_values:
        q0 = q_max * (1 - 0.2 * (pwf / pr) - 0.8 * (pwf / pr)**2)
        if np.isfinite(q0) and 0 <= q0 <= 1000:
            ipr_points.append((q0, pwf))
    if len(ipr_points) < 2:
        logger.error("Insufficient valid IPR points for Vogel.")
        st.error("Insufficient valid IPR points generated for Vogel method. Please adjust q_max.")
        raise ValueError("Insufficient valid IPR points.")
    logger.info(f"Vogel parameters: q_max={q_max:.4f}, points={len(ipr_points)}")
    return q_max, ipr_points

@st.cache_data
def calculate_ipr_composite(pr, j_star, p_b):
    """
    Calculate IPR points using Composite method.
    Returns (j_star, p_b, ipr_points).
    """
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
        logger.error("Insufficient valid IPR points for Composite.")
        st.error("Insufficient valid IPR points generated for Composite method. Please adjust j_star or p_b.")
        raise ValueError("Insufficient valid IPR points.")
    logger.info(f"Composite parameters: j_star={j_star:.4f}, p_b={p_b:.4f}, points={len(ipr_points)}")
    return j_star, p_b, ipr_points

def find_intersection(tpr_points, ipr_points, pr):
    """
    Find intersection between TPR and IPR curves using interpolation.
    Returns (intersection_q0, intersection_p) or (None, None) if no valid intersection.
    """
    # Validate input points
    if not tpr_points or not ipr_points:
        logger.error("Empty TPR or IPR points provided.")
        st.error("No valid TPR or IPR points provided for intersection calculation.")
        return None, None

    # Filter valid points and ensure finite values
    tpr_points = [(q, p) for q, p in tpr_points if np.isfinite(q) and np.isfinite(p) and q >= 0 and p >= 0]
    ipr_points = [(q, p) for q, p in ipr_points if np.isfinite(q) and np.isfinite(p) and q >= 0 and p >= 0]
    
    if len(tpr_points) < 2 or len(ipr_points) < 2:
        logger.error(f"Insufficient points: TPR={len(tpr_points)}, IPR={len(ipr_points)}")
        st.error("Insufficient valid points for TPR or IPR curves. At least two points required per curve.")
        return None, None

    try:
        tpr_q0, tpr_p2 = zip(*tpr_points)
        ipr_q0, ipr_pwf = zip(*ipr_points)
        
        # Ensure arrays are sorted by q0
        tpr_indices = np.argsort(tpr_q0)
        tpr_q0 = np.array(tpr_q0)[tpr_indices]
        tpr_p2 = np.array(tpr_p2)[tpr_indices]
        
        ipr_indices = np.argsort(ipr_q0)
        ipr_q0 = np.array(ipr_q0)[ipr_indices]
        ipr_pwf = np.array(ipr_pwf)[ipr_indices]

        # Create interpolation functions
        tpr_interp = interp1d(tpr_q0, tpr_p2, kind='linear', fill_value='extrapolate')
        ipr_interp = interp1d(ipr_q0, ipr_pwf, kind='linear', fill_value='extrapolate')

        # Define difference function
        def diff_func(q):
            try:
                return ipr_interp(q) - tpr_interp(q)
            except Exception as e:
                logger.debug(f"Intersection function error at q={q}: {str(e)}")
                return np.inf

        # Determine valid intersection range
        q_min = max(min(tpr_q0), min(ipr_q0))
        q_max = min(max(tpr_q0), max(ipr_q0))
        
        if q_min >= q_max:
            logger.warning(f"Invalid Q0 range: q_min={q_min:.2f}, q_max={q_max:.2f}")
            st.warning(f"No valid intersection range: Q0 from {q_min:.2f} to {q_max:.2f}. Check TPR and IPR points.")
            return None, None

        # Check for sign change
        f_min = diff_func(q_min)
        f_max = diff_func(q_max)
        if not (np.isfinite(f_min) and np.isfinite(f_max)):
            logger.warning(f"Non-finite function values: f_min={f_min}, f_max={f_max}")
            st.warning("Intersection calculation failed due to invalid values in TPR or IPR curves.")
            return None, None
        
        if f_min * f_max > 0:
            logger.warning(f"No sign change detected: f_min={f_min:.2f}, f_max={f_max:.2f}")
            st.warning("No intersection found between TPR and IPR curves.")
            return None, None

        # Find intersection
        try:
            result = root_scalar(diff_func, bracket=[q_min, q_max], method='brentq')
            intersection_q0 = result.root if result.converged else None
            if intersection_q0 is not None:
                intersection_p = ipr_interp(intersection_q0)
                # Validate intersection
                if (0 <= intersection_q0 <= 750 and
                    0 <= intersection_p <= max(pr, 4000) and
                    abs(ipr_interp(intersection_q0) - tpr_interp(intersection_q0)) < 1.0):
                    logger.info(f"Intersection found: Q0={intersection_q0:.2f} stb/day, P={intersection_p:.2f} psi")
                    st.write(f"Intersection found: Q0={intersection_q0:.2f} stb/day, P={intersection_p:.2f} psi")
                    return intersection_q0, intersection_p
                else:
                    logger.warning(f"Intersection rejected: Q0={intersection_q0:.2f}, P={intersection_p:.2f}")
                    st.warning(f"Invalid intersection: Q0={intersection_q0:.2f}, P={intersection_p:.2f}.")
                    return None, None
        except Exception as e:
            logger.error(f"Intersection search failed: {str(e)}")
            st.warning(f"Failed to find intersection: {str(e)}.")
            return None, None
    except Exception as e:
        logger.error(f"Intersection calculation failed: {str(e)}")
        st.error(f"Calculation failed: {str(e)}")
        return None, None
