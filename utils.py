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
    Returns bytes for Streamlit download. Handles mismatched lengths and empty inputs.
    """
    logger = setup_logging()
    logger.info(f"Exporting to Excel: tpr_points={tpr_points}, ipr_points={ipr_points}, "
                f"intersection_q0={intersection_q0}, intersection_p={intersection_p}")
    
    try:
        # Initialize lists for valid points
        valid_tpr_q0 = []
        valid_tpr_p2 = []
        valid_ipr_q0 = []
        valid_ipr_pwf = []
        
        # Validate and filter TPR points
        if isinstance(tpr_points, (list, tuple)) and tpr_points:
            for i, point in enumerate(tpr_points):
                if not isinstance(point, (list, tuple)):
                    logger.warning(f"Skipping invalid TPR point at index {i}: {point} (not a tuple/list)")
                    continue
                if len(point) != 2:
                    logger.warning(f"Skipping invalid TPR point at index {i}: {point} (expected 2 elements, got {len(point)})")
                    continue
                q, p = point
                if not (isinstance(q, (int, float, np.number)) and isinstance(p, (int, float, np.number)) and
                        np.isfinite(q) and np.isfinite(p)):
                    logger.warning(f"Skipping invalid TPR point at index {i}: {point} (non-numeric or non-finite)")
                    continue
                valid_tpr_q0.append(q)
                valid_tpr_p2.append(p)
        else:
            logger.warning(f"Invalid or empty tpr_points: {type(tpr_points)}")
        
        # Validate and filter IPR points
        if isinstance(ipr_points, (list, tuple)) and ipr_points:
            for i, point in enumerate(ipr_points):
                if not isinstance(point, (list, tuple)):
                    logger.warning(f"Skipping invalid IPR point at index {i}: {point} (not a tuple/list)")
                    continue
                if len(point) != 2:
                    logger.warning(f"Skipping invalid IPR point at index {i}: {point} (expected 2 elements, got {len(point)})")
                    continue
                q, p = point
                if not (isinstance(q, (int, float, np.number)) and isinstance(p, (int, float, np.number)) and
                        np.isfinite(q) and np.isfinite(p)):
                    logger.warning(f"Skipping invalid IPR point at index {i}: {point} (non-numeric or non-finite)")
                    continue
                valid_ipr_q0.append(q)
                valid_ipr_pwf.append(p)
        else:
            logger.warning(f"Invalid or empty ipr_points: {type(ipr_points)}")
        
        # Validate intersection points
        valid_intersection_q0 = [intersection_q0] if (intersection_q0 is not None and
                                                     np.isfinite(intersection_q0)) else []
        valid_intersection_p = [intersection_p] if (intersection_p is not None and
                                                   np.isfinite(intersection_p)) else []
        
        # Log lengths of all lists
        logger.info(f"List lengths: TPR_Q0={len(valid_tpr_q0)}, TPR_P2={len(valid_tpr_p2)}, "
                    f"IPR_Q0={len(valid_ipr_q0)}, IPR_Pwf={len(valid_ipr_pwf)}, "
                    f"Intersection_Q0={len(valid_intersection_q0)}, Intersection_P={len(valid_intersection_p)}")
        
        # Determine maximum length for DataFrame
        max_length = max(
            len(valid_tpr_q0),
            len(valid_tpr_p2),
            len(valid_ipr_q0),
            len(valid_ipr_pwf),
            len(valid_intersection_q0) or 1,  # Ensure at least 1 to avoid empty DataFrame
            len(valid_intersection_p) or 1
        )
        
        # Pad lists with None to match max_length
        valid_tpr_q0 += [None] * (max_length - len(valid_tpr_q0))
        valid_tpr_p2 += [None] * (max_length - len(valid_tpr_p2))
        valid_ipr_q0 += [None] * (max_length - len(valid_ipr_q0))
        valid_ipr_pwf += [None] * (max_length - len(valid_ipr_pwf))
        valid_intersection_q0 += [None] * (max_length - len(valid_intersection_q0))
        valid_intersection_p += [None] * (max_length - len(valid_intersection_p))
        
        # Create DataFrame
        df = pd.DataFrame({
            'TPR_Q0': valid_tpr_q0,
            'TPR_P2': valid_tpr_p2,
            'IPR_Q0': valid_ipr_q0,
            'IPR_Pwf': valid_ipr_pwf,
            'Intersection_Q0': valid_intersection_q0,
            'Intersection_P': valid_intersection_p
        })
        
        # Log DataFrame info
        logger.info(f"DataFrame created with {len(df)} rows and columns: {df.columns.tolist()}")
        
        # If no valid TPR or IPR points, add a note
        if not valid_tpr_q0 and not valid_ipr_q0:
            logger.warning("No valid TPR or IPR points, adding note to Excel")
            df = pd.DataFrame({
                "Note": ["No valid TPR or IPR points. Check input data or calculations."]
            })
        elif not valid_tpr_q0:
            logger.warning("No valid TPR points, adding note to Excel")
            df["Note"] = ["No valid TPR points"] + [None] * (len(df) - 1)
        elif not valid_ipr_q0:
            logger.warning("No valid IPR points, adding note to Excel")
            df["Note"] = ["No valid IPR points"] + [None] * (len(df) - 1)
        
        # Export to Excel
        output = io.BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        excel_data = output.getvalue()
        logger.info(f"Exported Excel: {len(excel_data)} bytes")
        return excel_data
    
    except Exception as e:
        logger.error(f"Failed to export to Excel: {str(e)}")
        # Create minimal Excel file with error message
        try:
            df = pd.DataFrame({"Error": [f"Export failed: {str(e)}"]})
            output = io.BytesIO()
            df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            excel_data = output.getvalue()
            logger.info(f"Exported minimal error Excel: {len(excel_data)} bytes")
            return excel_data
        except Exception as e2:
            logger.error(f"Failed to create minimal error Excel: {str(e2)}")
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
