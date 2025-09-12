import numpy as np
import pandas as pd
import io
import logging
import matplotlib.pyplot as plt

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
    Export a matplotlib figure to PNG format.
    Returns bytes for Streamlit download. Returns empty bytes if fig is invalid/empty.
    """
    logger = setup_logging()
    if fig is None:
        logger.warning("Cannot export: Figure is None")
        return b''  # Empty bytes instead of error
    if len(fig.axes) == 0:
        logger.warning("Cannot export: No axes in figure")
        return b''
    
    ax = fig.axes[0]
    # Broader check for content (lines, scatters, text, etc.)
    if (len(ax.lines) + len(ax.patches) + len(ax.collections) + len(ax.texts) == 0):
        logger.warning("Cannot export: Empty axes (no visible content)")
        return b''
    
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        png_data = buf.getvalue()
        logger.info(f"Exported PNG: {len(png_data)} bytes")
        return png_data
    except Exception as e:
        logger.error(f"Savefig failed: {str(e)}")
        return b''  # Return empty instead of raising

def export_plot_to_pdf(fig):
    """
    Export a matplotlib figure to PDF format.
    Returns bytes for Streamlit download. Returns empty bytes if fig is invalid/empty.
    """
    logger = setup_logging()
    if fig is None:
        logger.warning("Cannot export: Figure is None")
        return b''
    if len(fig.axes) == 0:
        logger.warning("Cannot export: No axes in figure")
        return b''
    
    ax = fig.axes[0]
    if (len(ax.lines) + len(ax.patches) + len(ax.collections) + len(ax.texts) == 0):
        logger.warning("Cannot export: Empty axes (no visible content)")
        return b''
    
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='pdf', dpi=300, bbox_inches='tight')
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Savefig PDF failed: {str(e)}")
        return b''

def export_plot_to_jpg(fig):
    """
    Export a matplotlib figure to JPG format with high resolution.
    Returns bytes for Streamlit download. Returns empty bytes if fig is invalid/empty.
    """
    logger = setup_logging()
    if fig is None:
        logger.warning("Cannot export: Figure is None")
        return b''
    if len(fig.axes) == 0:
        logger.warning("Cannot export: No axes in figure")
        return b''
    
    ax = fig.axes[0]
    if (len(ax.lines) + len(ax.patches) + len(ax.collections) + len(ax.texts) == 0):
        logger.warning("Cannot export: Empty axes (no visible content)")
        return b''
    
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='jpg', dpi=300, bbox_inches='tight', quality=95)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Savefig JPG failed: {str(e)}")
        return b''
