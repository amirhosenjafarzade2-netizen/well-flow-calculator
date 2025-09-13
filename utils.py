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

def export_results_to_excel(mode, results, inputs):
    """
    Export results to Excel for specific mode (p2 Finder, Natural Flow Finder, GLR Curves).
    Includes input parameters and mode-specific results.
    Returns bytes for Streamlit download. Handles empty/invalid inputs with specific notes.
    
    Parameters:
    - mode: str, one of 'p2_finder', 'natural_flow', 'glr_curves'
    - results: dict, mode-specific results (e.g., {'y1': ..., 'y2': ..., 'p2': ...} for p2_finder)
    - inputs: dict, input parameters (e.g., {'conduit_size': ..., 'glr': ..., ...})
    """
    logger = setup_logging()
    logger.info(f"Exporting to Excel for mode: {mode}, inputs: {inputs}, results: {results}")
    
    try:
        df = None
        notes = []
        filename = f"{mode}_results.xlsx"

        if mode == "p2_finder":
            # Expected inputs: conduit_size, production_rate, glr, p1, D
            # Expected results: y1, y2, p2, interpolation_status, coeffs, glr1, glr2
            required_inputs = ['conduit_size', 'production_rate', 'glr', 'p1', 'D']
            required_results = ['y1', 'y2', 'p2', 'interpolation_status']
            
            # Validate inputs
            missing_inputs = [k for k in required_inputs if k not in inputs or inputs[k] is None]
            if missing_inputs:
                notes.append(f"Missing or invalid inputs: {', '.join(missing_inputs)}")
                logger.warning(f"Missing inputs for p2_finder: {missing_inputs}")

            # Validate results
            missing_results = [k for k in required_results if k not in results or results[k] is None]
            if missing_results:
                notes.append(f"Missing or invalid results: {', '.join(missing_results)}")
                logger.warning(f"Missing results for p2_finder: {missing_results}")

            # Create DataFrame
            data = {
                'Conduit_Size_in': [inputs.get('conduit_size', None)],
                'Production_Rate_stb_day': [inputs.get('production_rate', None)],
                'GLR_scf_stb': [inputs.get('glr', None)],
                'Wellhead_Pressure_p1_psi': [inputs.get('p1', None)],
                'Well_Length_D_ft': [inputs.get('D', None)],
                'Depth_y1_ft': [results.get('y1', None)],
                'Depth_y2_ft': [results.get('y2', None)],
                'Bottomhole_Pressure_p2_psi': [results.get('p2', None)],
                'Interpolation_Status': [results.get('interpolation_status', None)],
                'GLR_Range_Lower_scf_stb': [results.get('glr1', None)],
                'GLR_Range_Upper_scf_stb': [results.get('glr2', None)]
            }
            if notes:
                data['Notes'] = [', '.join(notes)]
            df = pd.DataFrame(data)
            logger.info(f"p2_finder DataFrame created with columns: {df.columns.tolist()}")

        elif mode == "natural_flow":
            # Expected inputs: conduit_size, production_rate, glr, pwh, D, pr, ipr_method, ipr_params
            # Expected results: tpr_points, ipr_points, intersection_q0, intersection_p, ipr_params
            required_inputs = ['conduit_size', 'production_rate', 'glr', 'pwh', 'D', 'pr', 'ipr_method']
            required_results = ['tpr_points', 'ipr_points', 'intersection_q0', 'intersection_p']
            
            # Validate inputs
            missing_inputs = [k for k in required_inputs if k not in inputs or inputs[k] is None]
            if missing_inputs:
                notes.append(f"Missing or invalid inputs: {', '.join(missing_inputs)}")
                logger.warning(f"Missing inputs for natural_flow: {missing_inputs}")

            # Validate results
            missing_results = [k for k in required_results if k not in results or results[k] is None]
            if missing_results:
                notes.append(f"Missing or invalid results: {', '.join(missing_results)}")
                logger.warning(f"Missing results for natural_flow: {missing_results}")

            # Process TPR and IPR points
            tpr_q0 = []
            tpr_p2 = []
            if results.get('tpr_points', []):
                try:
                    for i, point in enumerate(results['tpr_points']):
                        if not isinstance(point, (list, tuple)) or len(point) != 2:
                            logger.warning(f"Invalid TPR point at index {i}: {point}")
                            continue
                        q, p = point
                        if not (isinstance(q, (int, float, np.number)) and isinstance(p, (int, float, np.number)) and
                                np.isfinite(q) and np.isfinite(p) and q >= 0 and p >= 0):
                            logger.warning(f"Skipping invalid TPR point: {point}")
                            continue
                        tpr_q0.append(q)
                        tpr_p2.append(p)
                except Exception as e:
                    notes.append(f"Error processing TPR points: {str(e)}")
                    logger.error(f"TPR points processing failed: {str(e)}")
            else:
                notes.append("No valid TPR points provided")
                logger.warning("No valid TPR points provided")

            ipr_q0 = []
            ipr_pwf = []
            if results.get('ipr_points', []):
                try:
                    for i, point in enumerate(results['ipr_points']):
                        if not isinstance(point, (list, tuple)) or len(point) != 2:
                            logger.warning(f"Invalid IPR point at index {i}: {point}")
                            continue
                        q, p = point
                        if not (isinstance(q, (int, float, np.number)) and isinstance(p, (int, float, np.number)) and
                                np.isfinite(q) and np.isfinite(p) and q >= 0 and p >= 0):
                            logger.warning(f"Skipping invalid IPR point: {point}")
                            continue
                        ipr_q0.append(q)
                        ipr_pwf.append(p)
                except Exception as e:
                    notes.append(f"Error processing IPR points: {str(e)}")
                    logger.error(f"IPR points processing failed: {str(e)}")
            else:
                notes.append("No valid IPR points provided")
                logger.warning("No valid IPR points provided")

            # Validate intersection
            intersection_q0 = results.get('intersection_q0')
            intersection_p = results.get('intersection_p')
            if (intersection_q0 is None or intersection_p is None or
                not np.isfinite(intersection_q0) or not np.isfinite(intersection_p) or
                intersection_q0 < 0 or intersection_p < 0):
                notes.append(f"Invalid intersection: q0={intersection_q0}, p={intersection_p}")
                logger.warning(f"Invalid intersection: q0={intersection_q0}, p={intersection_p}")
                intersection_q0 = None
                intersection_p = None

            # Prepare IPR parameters
            ipr_params_str = results.get('ipr_params', '')
            ipr_method = inputs.get('ipr_method', '').capitalize()
            if ipr_method == 'Fetkovich' and 'fetkovich_input_method' in inputs:
                ipr_params_str += f", Input_Method={inputs['fetkovich_input_method']}"
                if inputs['fetkovich_input_method'] == 'Points':
                    ipr_params_str += f", Points=[{inputs.get('q01', 0)},{inputs.get('pwf1', 0)},{inputs.get('q02', 0)},{inputs.get('pwf2', 0)}]"

            # Create DataFrame
            max_length = max(
                len(tpr_q0), len(tpr_p2), len(ipr_q0), len(ipr_pwf),
                1 if intersection_q0 is not None else 0
            )
            tpr_q0 += [None] * (max_length - len(tpr_q0))
            tpr_p2 += [None] * (max_length - len(tpr_p2))
            ipr_q0 += [None] * (max_length - len(ipr_q0))
            ipr_pwf += [None] * (max_length - len(ipr_pwf))
            intersection_q0_list = [intersection_q0] + [None] * (max_length - 1)
            intersection_p_list = [intersection_p] + [None] * (max_length - 1)

            data = {
                'Conduit_Size_in': [inputs.get('conduit_size', None)] * max_length,
                'Production_Rate_stb_day': [inputs.get('production_rate', None)] * max_length,
                'GLR_scf_stb': [inputs.get('glr', None)] * max_length,
                'Wellhead_Pressure_pwh_psi': [inputs.get('pwh', None)] * max_length,
                'Well_Length_D_ft': [inputs.get('D', None)] * max_length,
                'Reservoir_Pressure_pr_psi': [inputs.get('pr', None)] * max_length,
                'IPR_Method': [ipr_method] * max_length,
                'IPR_Parameters': [ipr_params_str] * max_length,
                'TPR_Q0_stb_day': tpr_q0,
                'TPR_P2_psi': tpr_p2,
                'IPR_Q0_stb_day': ipr_q0,
                'IPR_Pwf_psi': ipr_pwf,
                'Intersection_Q0_stb_day': intersection_q0_list,
                'Intersection_P_psi': intersection_p_list
            }
            if notes:
                data['Notes'] = [', '.join(notes)] + [None] * (max_length - 1)
            df = pd.DataFrame(data)
            logger.info(f"natural_flow DataFrame created with {len(df)} rows and columns: {df.columns.tolist()}")

        elif mode == "glr_curves":
            # Expected inputs: conduit_size, production_rate
            # Expected results: glr_data (list of dicts with glr, pressures, depths)
            required_inputs = ['conduit_size', 'production_rate']
            required_results = ['glr_data']
            
            # Validate inputs
            missing_inputs = [k for k in required_inputs if k not in inputs or inputs[k] is None]
            if missing_inputs:
                notes.append(f"Missing or invalid inputs: {', '.join(missing_inputs)}")
                logger.warning(f"Missing inputs for glr_curves: {missing_inputs}")

            # Validate results
            missing_results = [k for k in required_results if k not in results or results[k] is None]
            if missing_results:
                notes.append(f"Missing or invalid results: {', '.join(missing_results)}")
                logger.warning(f"Missing results for glr_curves: {missing_results}")

            # Process GLR data
            glr_values = []
            pressures = []
            depths = []
            if results.get('glr_data', []):
                try:
                    for item in results['glr_data']:
                        if not isinstance(item, dict) or 'glr' not in item or 'pressures' not in item or 'depths' not in item:
                            logger.warning(f"Invalid GLR data item: {item}")
                            continue
                        glr = item['glr']
                        glr_pressures = item['pressures']
                        glr_depths = item['depths']
                        if not (isinstance(glr, (int, float)) and isinstance(glr_pressures, list) and isinstance(glr_depths, list)):
                            logger.warning(f"Invalid GLR data format: glr={glr}, pressures={glr_pressures}, depths={glr_depths}")
                            continue
                        if len(glr_pressures) != len(glr_depths):
                            logger.warning(f"Mismatched pressures and depths for GLR {glr}: len(pressures)={len(glr_pressures)}, len(depths)={len(glr_depths)}")
                            continue
                        for p, d in zip(glr_pressures, glr_depths):
                            if not (isinstance(p, (int, float, np.number)) and isinstance(d, (int, float, np.number)) and
                                    np.isfinite(p) and np.isfinite(d) and p >= 0 and d >= 0):
                                logger.warning(f"Skipping invalid GLR point for GLR {glr}: pressure={p}, depth={d}")
                                continue
                            glr_values.append(glr)
                            pressures.append(p)
                            depths.append(d)
                except Exception as e:
                    notes.append(f"Error processing GLR data: {str(e)}")
                    logger.error(f"GLR data processing failed: {str(e)}")
            else:
                notes.append("No valid GLR data provided")
                logger.warning("No valid GLR data provided")

            # Create DataFrame
            max_length = len(glr_values) or 1
            data = {
                'Conduit_Size_in': [inputs.get('conduit_size', None)] * max_length,
                'Production_Rate_stb_day': [inputs.get('production_rate', None)] * max_length,
                'GLR_scf_stb': glr_values + [None] * (max_length - len(glr_values)),
                'Pressure_psi': pressures + [None] * (max_length - len(pressures)),
                'Depth_ft': depths + [None] * (max_length - len(depths))
            }
            if notes:
                data['Notes'] = [', '.join(notes)] + [None] * (max_length - 1)
            df = pd.DataFrame(data)
            logger.info(f"glr_curves DataFrame created with {len(df)} rows and columns: {df.columns.tolist()}")

        else:
            logger.error(f"Unknown mode: {mode}")
            df = pd.DataFrame({'Error': [f"Unknown mode: {mode}"]})

        # Export to Excel
        if df is None or df.empty:
            df = pd.DataFrame({'Error': ['No valid data to export. Check inputs and results.']})
            if notes:
                df['Notes'] = [', '.join(notes)]
            logger.warning("Created minimal DataFrame due to no valid data")

        output = io.BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        excel_data = output.getvalue()
        logger.info(f"Exported Excel for {mode}: {len(excel_data)} bytes")
        return excel_data, filename

    except Exception as e:
        logger.error(f"Failed to export to Excel for {mode}: {str(e)}")
        try:
            df = pd.DataFrame({'Error': [f"Export failed: {str(e)}"]})
            if notes:
                df['Notes'] = [', '.join(notes)]
            output = io.BytesIO()
            df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            excel_data = output.getvalue()
            logger.info(f"Exported minimal error Excel for {mode}: {len(excel_data)} bytes")
            return excel_data, f"{mode}_error.xlsx"
        except Exception as e2:
            logger.error(f"Failed to create minimal error Excel for {mode}: {str(e2)}")
            return b'', f"{mode}_error.xlsx"

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
