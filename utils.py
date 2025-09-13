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
    Returns bytes for Streamlit download. Returns empty bytes if inputs are invalid.
    """
    logger = setup_logging()
    try:
        # Validate input types
        if not isinstance(tpr_points, (list, tuple)):
            logger.error(f"Invalid tpr_points type: {type(tpr_points)}")
            return b''
        if not isinstance(ipr_points, (list, tuple)):
            logger.error(f"Invalid ipr_points type: {type(ipr_points)}")
            return b''

        # Validate tuple structure
        for points, name in [(tpr_points, 'TPR'), (ipr_points, 'IPR')]:
            for i, point in enumerate(points):
                if not isinstance(point, (list, tuple)):
                    logger.error(f"Invalid {name} point at index {i}: {point} (not a tuple/list)")
                    return b''
                if len(point) != 2:
                    logger.error(f"Invalid {name} point at index {i}: {point} (expected 2 elements, got {len(point)})")
                    return b''
                # Ensure elements are numeric
                for val in point:
                    if not isinstance(val, (int, float, np.number)) or not np.isfinite(val):
                        logger.error(f"Invalid {name} point at index {i}: {point} (non-numeric or non-finite value)")
                        return b''

        # Create DataFrame
        df = pd.DataFrame({
            'TPR_Q0': [q for q, _ in tpr_points] if tpr_points else [],
            'TPR_P2': [p for _, p in tpr_points] if tpr_points else [],
            'IPR_Q0': [q for q, _ in ipr_points] if ipr_points else [],
            'IPR_Pwf': [p for _, p in ipr_points] if ipr_points else [],
            'Intersection_Q0': [intersection_q0] if intersection_q0 is not None and np.isfinite(intersection_q0) else [],
            'Intersection_P': [intersection_p] if intersection_p is not None and np.isfinite(intersection_p) else []
        })
        output = io.BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        logger.info(f"Exported Excel: {len(output.getvalue())} bytes")
        return output.getvalue()
    except Exception as e:
        logger.error(f"Failed to export to Excel: {str(e)}")
        return b''

def export_plot_to_png(fig):
    """
    Export a matplotlib figure to PNG format.
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
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        png_data = buf.getvalue()
        logger.info(f"Exported PNG: {len(png_data)} bytes")
        return png_data
    except Exception as e:
        logger.error(f"Savefig failed: {str(e)}")
        return b''

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
