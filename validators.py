import streamlit as st
import numpy as np
from config import INTERPOLATION_RANGES, PRODUCTION_RATES
from utils import setup_logging

# Initialize logger
logger = setup_logging()

def validate_conduit_size(conduit_size):
    """
    Validate conduit size. Must be 2.875 or 3.5.
    Returns True if valid, else False and displays a warning.
    """
    valid_sizes = [2.875, 3.5]
    if conduit_size not in valid_sizes:
        logger.warning(f"Invalid conduit size: {conduit_size}. Must be one of {valid_sizes}.")
        st.warning(f"Invalid conduit size: {conduit_size}. Please select 2.875 or 3.5.")
        return False
    return True

def validate_production_rate(production_rate):
    """
    Validate production rate. Must be between 50 and 600 stb/day.
    Returns True if valid, else False and displays a warning.
    """
    if not (50 <= production_rate <= 600):
        logger.warning(f"Invalid production rate: {production_rate}. Must be between 50 and 600 stb/day.")
        st.warning(f"Invalid production rate: {production_rate}. Please enter a value between 50 and 600 stb/day.")
        return False
    return True

def validate_glr(conduit_size, production_rate, glr):
    """
    Validate GLR against interpolation ranges for given conduit size and production rate.
    Returns True if valid, else False and displays a warning with valid ranges.
    """
    ranges = INTERPOLATION_RANGES.get((conduit_size, production_rate), [])
    for min_glr, max_glr in ranges:
        if min_glr <= glr <= max_glr:
            return True
    valid_ranges = get_valid_glr_range(conduit_size, production_rate)
    logger.warning(f"GLR {glr} is outside valid ranges for conduit size {conduit_size} and production rate {production_rate}: {valid_ranges}")
    st.warning(f"GLR {glr} is invalid for conduit size {conduit_size} and production rate {production_rate}. Valid ranges: {valid_ranges}")
    return False

def validate_depth_and_pressure(y1, D, max_depth=31000):
    """
    Validate depth and pressure. Ensures y1 <= 31000 ft and y1 + D <= 31000 ft.
    Returns True if valid, else False and displays a warning.
    """
    if y1 > max_depth:
        logger.warning(f"Invalid depth: y1 = {y1:.2f} ft exceeds maximum depth of {max_depth} ft.")
        st.warning(f"Invalid pressure: Resulting depth y1 = {y1:.2f} ft exceeds maximum depth of {max_depth} ft.")
        return False
    if y1 + D > max_depth:
        logger.warning(f"Invalid well length: y1 + D = {y1 + D:.2f} ft exceeds maximum depth of {max_depth} ft.")
        st.warning(f"Invalid well length: Depth y1 + D = {y1 + D:.2f} ft exceeds maximum depth of {max_depth} ft.")
        return False
    return True

def validate_pressure(p, p_type, max_pressure=4000):
    """
    Validate pressure (e.g., p1, pwh, pr). Must be non-negative and <= max_pressure.
    Returns True if valid, else False and displays a warning.
    """
    if p < 0 or p > max_pressure:
        logger.warning(f"Invalid {p_type}: {p} psi. Must be between 0 and {max_pressure} psi.")
        st.warning(f"Invalid {p_type}: {p} psi. Please enter a value between 0 and {max_pressure} psi.")
        return False
    return True

def validate_fetkovich_parameters(c, n):
    """
    Validate Fetkovich parameters C and n.
    C must be positive, n must be between 0 and 2.
    Returns True if valid, else False and displays a warning.
    """
    if c is None or c <= 0 or not np.isfinite(c):
        logger.warning(f"Invalid Fetkovich parameter: C = {c}. Must be positive.")
        st.warning(f"Invalid Fetkovich parameter: C = {c}. Please enter a positive value.")
        return False
    if n is None or n <= 0 or n > 2 or not np.isfinite(n):
        logger.warning(f"Invalid Fetkovich parameter: n = {n}. Must be between 0 and 2.")
        st.warning(f"Invalid Fetkovich parameter: n = {n}. Please enter a value between 0 and 2.")
        return False
    return True

def validate_fetkovich_points(points, pr):
    """
    Validate Fetkovich test points (q0, pwf).
    At least two valid points required, q0 > 0, 0 < pwf <= pr, points must be distinct.
    points: List of (q0, pwf) tuples.
    Returns True if valid, else False and displays a warning.
    """
    valid_points = [(q, p) for q, p in points if q is not None and p is not None and q > 0 and 0 < p <= pr and np.isfinite(q) and np.isfinite(p)]
    if len(valid_points) < 2:
        logger.warning(f"Insufficient valid Fetkovich points: {len(valid_points)} provided, at least 2 required.")
        st.warning("At least two valid points (Q0 > 0, 0 < Pwf ≤ Pr) required for Fetkovich calculation.")
        return False
    
    # Check for distinct points
    q_values = [q for q, _ in valid_points]
    p_values = [p for _, p in valid_points]
    if len(set(q_values)) < len(q_values) or len(set(p_values)) < len(p_values):
        logger.warning("Fetkovich points are not distinct: duplicate Q0 or Pwf values.")
        st.warning("Fetkovich points must have distinct Q0 and Pwf values.")
        return False
    
    # Check for valid pressure differences
    for q, pwf in valid_points:
        delta_p = pr**2 - pwf**2
        if delta_p <= 0:
            logger.warning(f"Invalid Fetkovich point: Q0 = {q}, Pwf = {pwf}, Pr = {pr}. Pr^2 - Pwf^2 must be positive.")
            st.warning(f"Invalid point: Q0 = {q}, Pwf = {pwf}. Pwf must be less than Pr ({pr} psi).")
            return False
    
    logger.info(f"Validated {len(valid_points)} Fetkovich points.")
    return True

def get_valid_glr_range(conduit_size, production_rate):
    """
    Get valid GLR range for given conduit size and production rate.
    Returns a string describing valid ranges.
    """
    ranges = INTERPOLATION_RANGES.get((conduit_size, production_rate), [])
    if not ranges:
        return f"No valid GLR ranges for conduit size {conduit_size} and production rate {production_rate}."
    return f"{', '.join([f'[{min_glr}, {max_glr}]' for min_glr, max_glr in ranges])}"

def get_valid_options(conduit_size):
    """
    Get valid production rates and GLRs for a given conduit size.
    Returns (valid_production_rates, valid_glrs_dict) for use in dropdowns.
    """
    valid_production_rates = [pr for pr in PRODUCTION_RATES if (conduit_size, pr) in INTERPOLATION_RANGES]
    valid_glrs = {}
    for pr in valid_production_rates:
        ranges = INTERPOLATION_RANGES.get((conduit_size, pr), [])
        glrs = set()
        for min_glr, max_glr in ranges:
            glrs.update(np.arange(int(min_glr), int(max_glr) + 1, 100))
        valid_glrs[pr] = sorted(glrs)
    logger.info(f"Valid production rates for conduit {conduit_size}: {valid_production_rates}")
    logger.info(f"Valid GLRs for conduit {conduit_size}: {valid_glrs}")
    return valid_production_rates, valid_glrs
