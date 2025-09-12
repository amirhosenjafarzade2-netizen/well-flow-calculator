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
    logger = setup_logging()
    if fig is None or not hasattr(fig, 'data') or not fig.data:
        logger.error("Invalid or empty Plotly figure provided for PNG export.")
        raise ValueError("Cannot export plot: Invalid or empty figure.")
    
    try:
        import kaleido
        buf = io.BytesIO()
        fig.write_image(buf, format='png')
        return buf.getvalue()
    except ImportError:
        logger.error("Kaleido is not installed, cannot export plot to PNG.")
        raise ImportError("Kaleido is required for PNG export but is not installed.")
    except RuntimeError as e:
        logger.error(f"Failed to export plot to PNG: {str(e)}")
        raise RuntimeError("Failed to export plot to PNG: Chrome browser may be missing or misconfigured.")
    except Exception as e:
        logger.error(f"Unexpected error during PNG export: {str(e)}")
        raise

def export_plot_to_pdf(fig):
    """
    Export a Plotly figure to PDF format.
    Returns bytes for Streamlit download.
    """
    logger = setup_logging()
    if fig is None or not hasattr(fig, 'data') or not fig.data:
        logger.error("Invalid or empty Plotly figure provided for PDF export.")
        raise ValueError("Cannot export plot: Invalid or empty figure.")
    
    try:
        import kaleido
        buf = io.BytesIO()
        fig.write_image(buf, format='pdf')
        return buf.getvalue()
    except ImportError:
        logger.error("Kaleido is not installed, cannot export plot to PDF.")
        raise ImportError("Kaleido is required for PDF export but is not installed.")
    except RuntimeError as e:
        logger.error(f"Failed to export plot to PDF: {str(e)}")
        raise RuntimeError("Failed to export plot to PDF: Chrome browser may be missing or misconfigured.")
    except Exception as e:
        logger.error(f"Unexpected error during PDF export: {str(e)}")
        raise

def export_plot_to_jpg(fig):
    """
    Export a Plotly figure to JPG format with high resolution.
    Returns bytes for Streamlit download.
    """
    logger = setup_logging()
    if fig is None or not hasattr(fig, 'data') or not fig.data:
        logger.error("Invalid or empty Plotly figure provided for JPG export.")
        raise ValueError("Cannot export plot: Invalid or empty figure.")
    
    try:
        import kaleido
        buf = io.BytesIO()
        fig.write_image(buf, format='jpeg', scale=2)  # ~300 DPI
        return buf.getvalue()
    except ImportError:
        logger.error("Kaleido is not installed, cannot export plot to JPG.")
        raise ImportError("Kaleido is required for JPG export but is not installed.")
    except RuntimeError as e:
        logger.error(f"Failed to export plot to JPG: {str(e)}")
        raise RuntimeError("Failed to export plot to JPG: Chrome browser may be missing or misconfigured.")
    except Exception as e:
        logger.error(f"Unexpected error during JPG export: {str(e)}")
        raise
