# utils.py
# Shared utility functions for the Well Pressure and Depth Calculator

import numpy as np
import pandas as pd
import io
import logging
import plotly.graph_objects as go

def polynomial(x, coeffs):
    """
    Calculate polynomial values for given input x and coefficients.
    Vectorized to handle arrays efficiently.
    """
    try:
        x = np.asarray(x)
        return (coeffs['a'] * x**5 +
                coeffs['b'] * x**4 +
                coeffs['c'] * x**3 +
                coeffs['d'] * x**2 +
                coeffs['e'] * x +
                coeffs['f'])
    except Exception:
        return np.nan

def setup_logging():
    """
    Set up logging configuration for debugging and application logs.
    Logs to both console and a file ('app.log').
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)

def export_results_to_excel(tpr_points, ipr_points, intersection_q0, intersection_p):
    """
    Export TPR, IPR, and intersection results to an Excel file.
    Returns bytes for Streamlit download.
    """
    df = pd.DataFrame({
        'TPR_Q0': [q for q, _ in tpr_points] if tpr_points else [],
        'TPR_P2': [p for _, p in tpr_points] if tpr_points else [],
        'IPR_Q0': [q for q, _ in ipr_points] if ipr_points else [],
        'IPR_Pwf': [p for _, p in ipr_points] if ipr_points else [],
        'Intersection_Q0': [intersection_q0] if intersection_q0 is not None else [],
        'Intersection_P': [intersection_p] if intersection_p is not None else []
    })
    output = io.BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    return output.getvalue()

def export_plot_to_png(fig):
    """
    Export a Plotly figure to PNG format.
    Returns bytes for Streamlit download.
    """
    buf = io.BytesIO()
    fig.write_image(buf, format='png')
    return buf.getvalue()

def export_plot_to_pdf(fig):
    """
    Export a Plotly figure to PDF format.
    Returns bytes for Streamlit download.
    """
    buf = io.BytesIO()
    fig.write_image(buf, format='pdf')
    return buf.getvalue()
