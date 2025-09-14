import streamlit as st
import numpy as np
from scipy.optimize import root_scalar, curve_fit, OptimizeWarning
from scipy.interpolate import interp1d
from config import INTERPOLATION_RANGES, PRODUCTION_RATES
from utils import polynomial, setup_logging
from validators import validate_conduit_size, validate_production_rate, validate_glr, validate_fetkovich_parameters
import warnings

# Initialize logger
logger = setup_logging()

@st.cache_data
def calculate_results(conduit_size, production_rate, glr_input, p1, D, data_ref):
    """
    Calculate depths (y1, y2) and pressure (p2) based on polynomial interpolation.
    Returns (y1, y2, p2, coeffs, interpolation_status, glr1, glr2) or (None, ...) if invalid.
    """
    logger.info(f"calculate_results inputs: conduit_size={conduit_size}, production_rate={production_rate}, "
                f"glr_input={glr_input}, p1={p1}, D={D}")
    
    if not validate_conduit_size(conduit_size):
        logger.error(f"Invalid conduit size: {conduit_size}")
        return None, None, None, None, None, None, None
    if not validate_production_rate(production_rate):
        logger.error(f"Invalid production rate: {production_rate}")
        return None, None, None, None, None, None, None
    if not validate_glr(conduit_size, production_rate, glr_input):
        logger.error(f"Invalid GLR {glr_input} for conduit {conduit_size}, production {production_rate}")
        return None, None, None, None, None, None, None

    lower_prate = max([pr for pr in PRODUCTION_RATES if pr <= production_rate], default=50)
    higher_prate = min([pr for pr in PRODUCTION_RATES if pr >= production_rate], default=600)
    production_interpolation_status = "exact" if abs(lower_prate - higher_prate) < 1e-6 else "interpolated"
    prate1, prate2 = lower_prate, higher_prate
    logger.info(f"Production rates for interpolation: prate1={prate1}, prate2={prate2}, status={production_interpolation_status}")

    valid_glr1, valid_range1 = False, None
    ranges1 = INTERPOLATION_RANGES.get((conduit_size, prate1), [])
    for min_glr, max_glr in ranges1:
        if min_glr <= glr_input <= max_glr:
            valid_glr1, valid_range1 = True, (min_glr, max_glr)
            break
    if not valid_glr1:
        logger.error(f"GLR {glr_input} outside valid ranges {ranges1} for conduit {conduit_size}, production {prate1}")
        return None, None, None, None, None, None, None

    valid_glr2, valid_range2 = False, None
    ranges2 = INTERPOLATION_RANGES.get((conduit_size, prate2), [])
    for min_glr, max_glr in ranges2:
        if min_glr <= glr_input <= max_glr:
            valid_glr2, valid_range2 = True, (min_glr, max_glr)
            break
    if not valid_glr2:
        logger.error(f"GLR {glr_input} outside valid ranges {ranges2} for conduit {conduit_size}, production {prate2}")
        return None, None, None, None, None, None, None

    if valid_glr1 and (glr_input < valid_range1[0] * 1.05 or glr_input > valid_range1[1] * 0.95):
        logger.warning(f"GLR {glr_input} is near edge of range {valid_range1} for production {prate1}")
    if valid_glr2 and (glr_input < valid_range2[0] * 1.05 or glr_input > valid_range2[1] * 0.95):
        logger.warning(f"GLR {glr_input} is near edge of range {valid_range2} for production {prate2}")

    def get_coefficients(conduit_size, production_rate, glr_input, valid_range, data_ref):
        """
        Get polynomial coefficients by interpolating between GLR values.
        Returns (coeffs, glr1, glr2, status) or (None, ...) if invalid.
        """
        logger.info(f"get_coefficients: conduit_size={conduit_size}, production_rate={production_rate}, "
                    f"glr_input={glr_input}, valid_range={valid_range}")
        
        # Check for exact match
        for entry in data_ref:
            if (abs(entry['conduit_size'] - conduit_size) < 1e-6 and
                abs(entry['production_rate'] - production_rate) < 1e-6 and
                abs(entry['glr'] - glr_input) < 1e-6):
                logger.info(f"Exact match found: coefficients={entry['coefficients']}")
                return entry['coefficients'], glr_input, glr_input, "exact"

        # Find relevant data points for interpolation
        relevant_rows = [
            entry for entry in data_ref
            if (abs(entry['conduit_size'] - conduit_size) < 1e-6 and
                abs(entry['production_rate'] - production_rate) < 1e-6 and
                valid_range[0] <= entry['glr'] <= valid_range[1])
        ]
        relevant_rows.sort(key=lambda x: x['glr'])
        if len(relevant_rows) < 2:
            logger.error(f"Insufficient data points for interpolation: {len(relevant_rows)} found")
            return None, None, None, None

        glr1 = max([entry['glr'] for entry in relevant_rows if entry['glr'] <= glr_input], default=valid_range[0])
        glr2 = min([entry['glr'] for entry in relevant_rows if entry['glr'] >= glr_input], default=valid_range[1])
        if glr1 == glr2:
            for entry in relevant_rows:
                if abs(entry['glr'] - glr1) < 1e-6:
                    return entry['coefficients'], glr1, glr2, "exact"

        coeffs1 = None
        coeffs2 = None
        for entry in relevant_rows:
            if abs(entry['glr'] - glr1) < 1e-6:
                coeffs1 = entry['coefficients']
            if abs(entry['glr'] - glr2) < 1e-6:
                coeffs2 = entry['coefficients']

        if coeffs1 is None or coeffs2 is None:
            logger.error(f"Could not find coefficients for glr1={glr1}, glr2={glr2}")
            return None, None, None, None

        # Linear interpolation of coefficients
        weight = (glr_input - glr1) / (glr2 - glr1) if glr2 != glr1 else 0
        coeffs = {}
        for key in coeffs1:
            coeffs[key] = coeffs1[key] + weight * (coeffs2[key] - coeffs1[key])
        logger.info(f"Interpolated coefficients: {coeffs}, glr1={glr1}, glr2={glr2}")
        return coeffs, glr1, glr2, "interpolated"

    coeffs1, glr1_1, glr1_2, status1 = get_coefficients(conduit_size, prate1, glr_input, valid_range1, data_ref)
    coeffs2, glr2_1, glr2_2, status2 = get_coefficients(conduit_size, prate2, glr_input, valid_range2, data_ref)
    
    if coeffs1 is None or coeffs2 is None:
        logger.error("Failed to get coefficients for one or both production rates")
        return None, None, None, None, None, None, None

    try:
        y1 = polynomial(p1, coeffs1)
        y2 = polynomial(p1, coeffs2)
        if not (np.isfinite(y1) and np.isfinite(y2)):
            logger.error(f"Invalid y1={y1} or y2={y2} from polynomial calculation")
            return None, None, None, None, None, None, None

        if production_interpolation_status == "exact":
            coeffs = coeffs1
            y_final = y1
            p2 = polynomial(y1 + D, coeffs)
        else:
            weight = (production_rate - prate1) / (prate2 - prate1)
            coeffs = {}
            for key in coeffs1:
                coeffs[key] = coeffs1[key] + weight * (coeffs2[key] - coeffs1[key])
            y_final = y1 + weight * (y2 - y1)
            p2 = polynomial(y_final + D, coeffs)

        if not np.isfinite(p2):
            logger.error(f"Invalid p2={p2} from polynomial calculation")
            return None, None, None, None, None, None, None

        logger.info(f"Calculated: y1={y1:.2f}, y2={y_final:.2f}, p2={p2:.2f}, interpolation_status={production_interpolation_status}")
        return y1, y_final, p2, coeffs, production_interpolation_status, glr1_1, glr2_1
    except Exception as e:
        logger.error(f"Calculation failed: {str(e)}")
        return None, None, None, None, None, None, None

def calculate_tpr_points(conduit_size, production_rate, glr, pwh, D, data_ref):
    """
    Calculate TPR points (Q0, p2) for given inputs.
    Returns list of (Q0, p2) tuples or None if invalid.
    """
    logger.info(f"Calculating TPR points: conduit_size={conduit_size}, production_rate={production_rate}, "
                f"glr={glr}, pwh={pwh}, D={D}")
    
    try:
        q0_values = np.linspace(0, 1000, 50)
        tpr_points = []
        for q0 in q0_values:
            y1, y2, p2, coeffs, status, glr1, glr2 = calculate_results(conduit_size, q0, glr, pwh, D, data_ref)
            if p2 is not None and np.isfinite(p2) and 0 <= p2 <= 4000:
                tpr_points.append((q0, p2))
        if not tpr_points:
            logger.error("No valid TPR points calculated")
            return None
        logger.info(f"Generated {len(tpr_points)} TPR points")
        return tpr_points
    except Exception as e:
        logger.error(f"TPR calculation failed: {str(e)}")
        return None

def calculate_ipr_fetkovich(pr, c, n):
    """
    Calculate IPR points using Fetkovich model: q0 = c * (pr^2 - pwf^2)^n
    Returns list of (q0, pwf) tuples or None if invalid.
    """
    logger.info(f"Calculating Fetkovich IPR: pr={pr}, c={c}, n={n}")
    
    try:
        if not validate_fetkovich_parameters(c, n):
            logger.error(f"Invalid Fetkovich parameters: c={c}, n={n}")
            return None
        
        pwf_values = np.linspace(0, pr, 50)
        ipr_points = []
        for pwf in pwf_values:
            delta_p2 = pr**2 - pwf**2
            if delta_p2 < 0:
                continue
            q0 = c * delta_p2 ** n
            if np.isfinite(q0) and q0 >= 0:
                ipr_points.append((q0, pwf))
        if not ipr_points:
            logger.error("No valid IPR points calculated for Fetkovich model")
            return None
        logger.info(f"Generated {len(ipr_points)} Fetkovich IPR points")
        return ipr_points
    except Exception as e:
        logger.error(f"Fetkovich IPR calculation failed: {str(e)}")
        return None

def fit_fetkovich_parameters(pr, points):
    """
    Fit Fetkovich parameters c and n from test points using curve_fit.
    
    Parameters:
    - pr: Reservoir pressure (psi)
    - points: List of (q0, pwf) tuples
    
    Returns:
    - (c, n) if fit succeeds, else (None, None)
    """
    def fetkovich_model(delta_p2, c, n):
        return c * delta_p2 ** n
    
    delta_p2 = []
    q_data = []
    for q, pwf in points:
        delta = pr**2 - pwf**2
        if delta > 0:
            delta_p2.append(delta)
            q_data.append(q)
    
    if len(delta_p2) < 2:
        logger.warning("Insufficient valid delta_p2 points for Fetkovich fit")
        return None, None
    
    delta_p2 = np.array(delta_p2)
    q_data = np.array(q_data)
    
    try:
        # Suppress warnings for optimization issues
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                fetkovich_model, 
                delta_p2, 
                q_data, 
                p0=[0.001, 0.5],  # Initial guesses
                bounds=(0, [np.inf, 2]),  # Bounds: c > 0, 0 < n <= 2
                maxfev=10000  # Increase iterations for convergence
            )
        c, n = popt
        if not validate_fetkovich_parameters(c, n):
            return None, None
        logger.info(f"Fitted Fetkovich parameters: c={c:.6f}, n={n:.2f}")
        return c, n
    except Exception as e:
        logger.error(f"Fetkovich fit failed: {str(e)}")
        return None, None

def calculate_ipr_vogel(pr, qmax):
    """
    Calculate IPR points using Vogel's model: q0 = qmax * (1 - 0.2*(pwf/pr) - 0.8*(pwf/pr)^2)
    Returns list of (q0, pwf) tuples or None if invalid.
    """
    logger.info(f"Calculating Vogel IPR: pr={pr}, qmax={qmax}")
    
    try:
        if qmax <= 0 or not np.isfinite(qmax):
            logger.error(f"Invalid qmax: {qmax}")
            return None
        
        pwf_values = np.linspace(0, pr, 50)
        ipr_points = []
        for pwf in pwf_values:
            pwf_pr = pwf / pr
            q0 = qmax * (1 - 0.2 * pwf_pr - 0.8 * pwf_pr**2)
            if np.isfinite(q0) and q0 >= 0:
                ipr_points.append((q0, pwf))
        if not ipr_points:
            logger.error("No valid IPR points calculated for Vogel model")
            return None
        logger.info(f"Generated {len(ipr_points)} Vogel IPR points")
        return ipr_points
    except Exception as e:
        logger.error(f"Vogel IPR calculation failed: {str(e)}")
        return None

def calculate_ipr_composite(pr, qmax, n):
    """
    Calculate IPR points using composite model (Vogel below pr/2, Fetkovich above pr/2).
    Returns list of (q0, pwf) tuples or None if invalid.
    """
    logger.info(f"Calculating Composite IPR: pr={pr}, qmax={qmax}, n={n}")
    
    try:
        if qmax <= 0 or not np.isfinite(qmax) or n <= 0 or n > 2 or not np.isfinite(n):
            logger.error(f"Invalid parameters: qmax={qmax}, n={n}")
            return None
        
        pwf_values = np.linspace(0, pr, 50)
        ipr_points = []
        for pwf in pwf_values:
            if pwf <= pr / 2:
                # Vogel model
                pwf_pr = pwf / pr
                q0 = qmax * (1 - 0.2 * pwf_pr - 0.8 * pwf_pr**2)
            else:
                # Fetkovich model
                delta_p2 = pr**2 - pwf**2
                q0 = qmax * (delta_p2 / (pr**2)) ** n
            if np.isfinite(q0) and q0 >= 0:
                ipr_points.append((q0, pwf))
        if not ipr_points:
            logger.error("No valid IPR points calculated for Composite model")
            return None
        logger.info(f"Generated {len(ipr_points)} Composite IPR points")
        return ipr_points
    except Exception as e:
        logger.error(f"Composite IPR calculation failed: {str(e)}")
        return None

def find_intersection(tpr_points, ipr_points, pr):
    """
    Find intersection between TPR and IPR curves using interpolation.
    Returns (intersection_q0, intersection_p) or (None, None) if no valid intersection.
    """
    logger.info(f"find_intersection inputs: tpr_points={tpr_points}, ipr_points={ipr_points}, pr={pr}")
    if not tpr_points or not ipr_points:
        logger.error("Empty TPR or IPR points provided")
        return None, None

    # Filter valid points
    tpr_points = [(q, p) for q, p in tpr_points if np.isfinite(q) and np.isfinite(p) and q >= 0 and p >= 0]
    ipr_points = [(q, p) for q, p in ipr_points if np.isfinite(q) and np.isfinite(p) and q >= 0 and p >= 0]
    
    if len(tpr_points) < 2 or len(ipr_points) < 2:
        logger.error(f"Insufficient points: TPR={len(tpr_points)}, IPR={len(ipr_points)}")
        return None, None

    try:
        tpr_q0, tpr_p2 = zip(*tpr_points)
        ipr_q0, ipr_pwf = zip(*ipr_points)
        
        tpr_q0 = np.array(tpr_q0, dtype=float)
        tpr_p2 = np.array(tpr_p2, dtype=float)
        ipr_q0 = np.array(ipr_q0, dtype=float)
        ipr_pwf = np.array(ipr_pwf, dtype=float)
        
        # Sort points by q0
        tpr_indices = np.argsort(tpr_q0)
        tpr_q0 = tpr_q0[tpr_indices]
        tpr_p2 = tpr_p2[tpr_indices]
        
        ipr_indices = np.argsort(ipr_q0)
        ipr_q0 = ipr_q0[ipr_indices]
        ipr_pwf = ipr_pwf[ipr_indices]

        logger.info(f"TPR points: {len(tpr_points)}, IPR points: {len(ipr_points)}")
        logger.debug(f"TPR q0: {tpr_q0.tolist()}, TPR p2: {tpr_p2.tolist()}")
        logger.debug(f"IPR q0: {ipr_q0.tolist()}, IPR pwf: {ipr_pwf.tolist()}")

        # Determine common q0 range
        q_min = max(min(tpr_q0), min(ipr_q0))
        q_max = min(max(tpr_q0), max(ipr_q0))
        
        if q_min >= q_max:
            logger.warning(f"No valid intersection range: q_min={q_min:.2f}, q_max={q_max:.2f}")
            return None, None

        # Create interpolation functions
        try:
            tpr_interp = interp1d(tpr_q0, tpr_p2, kind='linear', fill_value='extrapolate')
            ipr_interp = interp1d(ipr_q0, ipr_pwf, kind='linear', fill_value='extrapolate')
        except Exception as e:
            logger.error(f"Interpolation failed: {str(e)}")
            return None, None

        def diff_func(q):
            try:
                tpr_val = tpr_interp(q)
                ipr_val = ipr_interp(q)
                if not (np.isfinite(tpr_val) and np.isfinite(ipr_val)):
                    logger.debug(f"Non-finite interpolation at q={q}: tpr_val={tpr_val}, ipr_val={ipr_val}")
                    return np.inf
                return ipr_val - tpr_val
            except Exception as e:
                logger.debug(f"Intersection function error at q={q}: {str(e)}")
                return np.inf

        # Check for sign change to ensure intersection exists
        f_min = diff_func(q_min)
        f_max = diff_func(q_max)
        if not (np.isfinite(f_min) and np.isfinite(f_max)):
            logger.warning(f"Non-finite function values: f_min={f_min}, f_max={f_max}")
            return None, None

        if f_min * f_max >= 0:
            logger.warning("No sign change in interpolation function, no intersection found")
            return None, None

        # Find intersection
        try:
            result = root_scalar(diff_func, bracket=[q_min, q_max], method='brentq')
            if result.converged:
                intersection_q0 = result.root
                intersection_p = tpr_interp(intersection_q0)
                if (0 <= intersection_q0 <= 1000 and 0 <= intersection_p <= pr):
                    logger.info(f"Intersection found: q0={intersection_q0:.2f}, p={intersection_p:.2f}")
                    return intersection_q0, intersection_p
                else:
                    logger.warning(f"Intersection out of bounds: q0={intersection_q0:.2f}, p={intersection_p:.2f}")
                    return None, None
            else:
                logger.warning("Intersection calculation did not converge")
                return None, None
        except Exception as e:
            logger.warning(f"Intersection calculation failed: {str(e)}")
            return None, None
    except Exception as e:
        logger.error(f"Intersection calculation failed: {str(e)}")
        return None, None
