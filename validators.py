# validators.py
# Input validation functions for the Well Pressure and Depth Calculator

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
