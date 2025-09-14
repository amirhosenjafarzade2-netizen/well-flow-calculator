import matplotlib.pyplot as plt
import numpy as np
from config import COLORS
from utils import setup_logging, polynomial
from calculations import find_pressure

# Global variable for GLR color mapping to ensure consistent coloring across plots
GLR_COLOR_MAP = {}

# Initialize logger for debugging and error tracking
logger = setup_logging()

def validate_plotting_inputs(data, param_name, min_len=2, non_negative=True, finite=True):
    """
    Validate input data for plotting to ensure it meets requirements.
    
    Parameters:
    - data: List of tuples or array-like data to validate
    - param_name: Name of the parameter for logging
    - min_len: Minimum number of valid points required
    - non_negative: If True, ensures values are non-negative
    - finite: If True, ensures values are finite
    
    Returns:
    - List of validated tuples or None if validation fails
    """
    if not data:
        logger.error(f"No data provided for {param_name}")
        return None
    
    # Filter valid points
    validated_data = [
        (x, y) for x, y in data
        if (np.isfinite(x) and np.isfinite(y) if finite else True) and
           (x >= 0 and y >= 0 if non_negative else True)
    ]
    
    if len(validated_data) < min_len:
        logger.error(f"Insufficient valid points for {param_name}: {len(validated_data)} points")
        return None
    
    logger.debug(f"Validated {len(validated_data)} points for {param_name}")
    return validated_data

def configure_axes(ax, x_label, y_label, x_lim=None, y_lim=None, title=None, mode='color', is_log_log=False):
    """
    Configure matplotlib axes with consistent styling.
    
    Parameters:
    - ax: Matplotlib axes object
    - x_label, y_label: Axis labels
    - x_lim, y_lim: Tuple of (min, max) for axis limits
    - title: Plot title
    - mode: 'color' or 'bw' for styling
    - is_log_log: If True, use tick locators suitable for log-log plots
    """
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    if title:
        ax.set_title(title)
    
    grid_color = '#D3D3D3' if mode == 'color' else 'black'
    ax.grid(True, which='major', color=grid_color, alpha=0.5)
    ax.grid(True, which='minor', color=grid_color, linestyle='-', alpha=0.2 if mode == 'bw' else 0.5)
    
    if is_log_log:
        # Use AutoLocator for log-log plots to ensure visible ticks
        ax.xaxis.set_major_locator(plt.AutoLocator())
        ax.xaxis.set_minor_locator(plt.NullLocator())  # Disable minor ticks for clarity
        ax.yaxis.set_major_locator(plt.AutoLocator())
        ax.yaxis.set_minor_locator(plt.NullLocator())
    else:
        # Use linear locators for standard plots
        ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(200))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1000))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(200))
    
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.margins(x=0.05, y=0.05)

def plot_results(p1, y1, y2, p2, D, coeffs, glr_input, interpolation_status, production_rate, mode='color'):
    """
    Plot pressure vs. depth for p2 Finder / Natural Flow Finder.

    Important behavior preserved:
    - GLR curve computed from polynomial(coeffs), capped at 31,000 ft
    - (p1, y1) and (p2, y2) scatter points shown
    - Thick green vertical line at x=0 from y1 to y2 representing D (well length)
    """
    logger.info(f"Plotting pressure vs. depth: p1={p1:.2f}, y1={y1:.2f}, p2={p2:.2f}, y2={y2:.2f}, D={D:.2f}, Q0={production_rate}, GLR={glr_input}")

    try:
        # --- defensive input checks ---
        if not np.isfinite(p1) or not np.isfinite(p2) or not np.isfinite(y1) or not np.isfinite(y2):
            logger.error("One of p1, p2, y1 or y2 is not finite.")
            return None
        if D is None or D < 0:
            logger.warning("D is not positive; continuing but check D value.")

        # local polynomial to avoid NameError if not in global scope
        def polynomial(x, c):
            try:
                return (c['a'] * x**5 + c['b'] * x**4 + c['c'] * x**3 +
                        c['d'] * x**2 + c['e'] * x + c.get('f', 0.0))
            except Exception:
                return np.nan

        # Initialize figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        bg = '#F5F5F5' if mode == 'color' else 'white'
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        # --- build GLR curve and cap at 31,000 ft ---
        p_vals = np.linspace(0, 4000, 400)
        y_vals = np.array([polynomial(p, coeffs) for p in p_vals], dtype=float)

        # Mark invalid / out-of-range values
        valid_mask = np.isfinite(y_vals) & (y_vals <= 31000)

        # If there is a crossing above 31k, cap once at the first crossing and stop plot there
        end_index = None
        if not np.all(valid_mask):
            # find first index where invalid occurs after some valid points
            for i in range(len(y_vals)):
                if not valid_mask[i]:
                    if i > 0 and np.isfinite(y_vals[i-1]):
                        end_index = i
                        break
            if end_index is None:
                # either everything invalid or everything valid but >31000 — handle conservatively
                if np.any(np.isfinite(y_vals)):
                    # find last finite index
                    finite_idx = np.where(np.isfinite(y_vals))[0]
                    if finite_idx.size > 0:
                        end_index = finite_idx[-1]
                else:
                    logger.error("No finite polynomial points to plot for GLR curve.")
                    plt.close(fig)
                    return None
        else:
            end_index = len(p_vals) - 1

        # Build final arrays to plot (include the cap point at 31000 if needed)
        if end_index is not None and end_index < len(p_vals) - 1:
            plot_p = p_vals[:end_index+1].copy()
            plot_y = y_vals[:end_index+1].copy()
            # set the last value to 31000 to show cap
            plot_y[-1] = 31000.0
        else:
            plot_p = p_vals
            plot_y = y_vals
            # clip to 31000 for safety
            plot_y = np.where(np.isfinite(plot_y), np.minimum(plot_y, 31000.0), np.nan)

        # Require at least two valid plotted points
        if np.sum(np.isfinite(plot_y)) < 2:
            logger.error("Insufficient valid GLR curve points to plot.")
            plt.close(fig)
            return None

        # Plot GLR curve
        curve_color = 'blue' if mode == 'color' else 'black'
        ax.plot(plot_p, plot_y, color=curve_color, linewidth=2.5,
                label=f'GLR curve ({interpolation_status.capitalize()}, Q0={production_rate} stb/day, GLR={glr_input})')

        # --- Scatter key points ---
        ax.scatter([p1], [y1], color=curve_color, s=50,
                   label=f'(p1, y1) = ({p1:.2f} psi, {y1:.2f} ft)', zorder=6)
        ax.scatter([p2], [y2], color=curve_color, s=50,
                   label=f'(p2, y2) = ({p2:.2f} psi, {y2:.2f} ft)', zorder=6)

        # --- Reference red lines (thin) ---
        ax.plot([p1, p1], [y1, 0], color='red', linewidth=1, zorder=2)
        ax.plot([p1, 0], [y1, y1], color='red', linewidth=1, zorder=2)
        ax.plot([p2, p2], [y2, 0], color='red', linewidth=1, zorder=2)
        ax.plot([p2, 0], [y2, y2], color='red', linewidth=1, zorder=2)

        # --- Thick green vertical D line at x=0 (ensure visible) ---
        green_color = 'green' if mode == 'color' else 'black'
        # Ensure ordering y_low -> y_high for plotting
        y_low, y_high = (y1, y2) if y1 <= y2 else (y2, y1)
        ax.plot([0, 0], [y_low, y_high],
                color=green_color, linewidth=6, solid_capstyle='butt', zorder=10,
                label=f'Well Length (D = {D:.2f} ft)')

        # --- Configure axes limits with a small margin so x=0 is visible ---
        max_x = max(np.nanmax(plot_p[np.isfinite(plot_y)]), float(p1), float(p2), 4000.0)
        max_y = max(float(y_low), float(y_high), 1000.0)
        ax.set_xlim(0, max_x * 1.03)
        ax.set_ylim(0, min(31000.0, max_y * 1.05))

        # Axis labels / invert y-axis (depth)
        ax.set_xlabel('Gradient Pressure, psi', fontsize=10)
        ax.set_ylabel('Depth, ft', fontsize=10)
        ax.invert_yaxis()

        # Grid and ticks similar to your original style
        ax.grid(True, which='major', color='#D3D3D3')
        ax.grid(True, which='minor', color='#D3D3D3', linestyle='-', alpha=0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(200))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1000))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(200))

        # Legend & layout
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.28), fontsize=8, frameon=True, edgecolor='black', ncol=1)
        plt.tight_layout()

        logger.info("Pressure vs. depth plot generated successfully (robust version).")
        return fig

    except Exception as e:
        logger.error(f"Failed to plot pressure vs. depth: {e}")
        plt.close()
        return None



def plot_curves(tpr_points, ipr_points, intersection_q0, intersection_p, conduit_size, glr, D, pwh, pr, ipr_params, mode='color'):
    """
    Plot TPR and IPR curves with intersection point for Natural Flow Finder.
    
    Parameters:
    - tpr_points, ipr_points: Lists of (q0, pressure) tuples
    - intersection_q0, intersection_p: Intersection point
    - conduit_size, glr, D, pwh, pr: Parameters for plot annotation
    - ipr_params: String for IPR parameters
    - mode: 'color' or 'bw' for colorful or black-and-white plots
    
    Returns:
    - Matplotlib figure object or None if invalid
    """
    logger.info(f"Plotting TPR/IPR curves: GLR={glr}, conduit_size={conduit_size}, pr={pr:.2f}")
    
    try:
        # Validate input points
        tpr_points = validate_plotting_inputs(tpr_points, "TPR points")
        ipr_points = validate_plotting_inputs(ipr_points, "IPR points")
        if not tpr_points or not ipr_points:
            return None
        
        # Unpack and sort points
        tpr_q0, tpr_p2 = zip(*tpr_points)
        ipr_q0, ipr_pwf = zip(*ipr_points)
        
        tpr_q0 = np.array(tpr_q0, dtype=float)
        tpr_p2 = np.array(tpr_p2, dtype=float)
        ipr_q0 = np.array(ipr_q0, dtype=float)
        ipr_pwf = np.array(ipr_pwf, dtype=float)
        
        tpr_indices = np.argsort(tpr_q0)
        ipr_indices = np.argsort(ipr_q0)
        tpr_q0 = tpr_q0[tpr_indices]
        tpr_p2 = tpr_p2[tpr_indices]
        ipr_q0 = ipr_q0[ipr_indices]
        ipr_pwf = ipr_pwf[ipr_indices]
        
        logger.debug(f"TPR q0 shape: {tpr_q0.shape}, TPR p2 shape: {tpr_p2.shape}")
        logger.debug(f"IPR q0 shape: {ipr_q0.shape}, IPR pwf shape: {ipr_pwf.shape}")
        
        # Initialize figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        fig.patch.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
        ax.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
        
        # Plot TPR and IPR curves
        tpr_color = 'blue' if mode == 'color' else 'black'
        ax.plot(tpr_q0, tpr_p2, color=tpr_color, marker='o', linewidth=2, 
                label=f'TPR (Conduit: {conduit_size} in, GLR: {glr}, D: {D} ft, Pwh: {pwh} psi)')
        
        ipr_color = 'red' if mode == 'color' else 'gray'
        ax.plot(ipr_q0, ipr_pwf, color=ipr_color, linewidth=2, 
                label=f'IPR ({ipr_params})')
        
        # Plot intersection point
        if intersection_q0 is not None and intersection_p is not None and np.isfinite(intersection_q0) and np.isfinite(intersection_p):
            intersect_color = 'green' if mode == 'color' else 'black'
            ax.scatter([intersection_q0], [intersection_p], color=intersect_color, s=100, marker='*',
                       label=f'Natural Flow Point (Q0: {intersection_q0:.2f} stb/day, P: {intersection_p:.2f} psi)')
        
        # Configure axes
        configure_axes(
            ax,
            x_label='Production Rate, Q0 (stb/day)',
            y_label='Pressure, psi',
            x_lim=(0, max(max(tpr_q0), max(ipr_q0), 600) * 1.1),
            y_lim=(0, max(max(tpr_p2), max(ipr_pwf), pr, 4000) * 1.1),
            title='TPR and IPR Curves with Natural Flow Point',
            mode=mode,
            is_log_log=False
        )
        ax.xaxis.set_major_locator(plt.MultipleLocator(100))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
        
        # Add legend
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize=8, frameon=True, 
                  edgecolor='black', ncol=1)
        plt.tight_layout()
        
        # Check for empty plot
        if len(ax.lines) == 0:
            logger.error("Empty plot in plot_curves - closing and returning None")
            plt.close(fig)
            return None
        
        logger.info("TPR/IPR curves plot generated successfully")
        return fig
    
    except Exception as e:
        logger.error(f"Failed to plot TPR/IPR curves: {str(e)}")
        plt.close()
        return None

def plot_fetkovich_log_log(points, pr, c, n, mode='color'):
    """
    Plot log-log graph for Fetkovich IPR calculation.
    
    Parameters:
    - points: List of (q0, pwf) tuples
    - pr: Reservoir pressure
    - c, n: Fetkovich parameters
    - mode: 'color' or 'bw' for colorful or black-and-white plots
    
    Returns:
    - Matplotlib figure object or None if invalid
    """
    logger.info(f"Plotting Fetkovich log-log graph: pr={pr:.2f}, c={c:.4e}, n={n:.4f}")
    
    try:
        # Validate points
        points = validate_plotting_inputs(points, "Fetkovich points", non_negative=True, finite=True)
        if not points:
            return None
        
        # Unpack and process points
        q_points, pwf_points = zip(*points)
        delta_p = pr**2 - np.array(pwf_points)**2
        valid_indices = np.isfinite(delta_p) & (delta_p > 0)
        if not np.any(valid_indices):
            logger.error("No valid delta_p values for Fetkovich log-log plot")
            return None
        
        delta_p = delta_p[valid_indices]
        q_points = np.array(q_points)[valid_indices]
        x = np.log10(delta_p)
        y = np.log10(q_points)
        
        # Initialize figure
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        fig.patch.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
        ax.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
        
        # Plot data points and fit
        point_color = 'blue' if mode == 'color' else 'black'
        ax.scatter(x, y, color=point_color, s=50, label='Data Points')
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = np.log10(c) + n * x_fit
        fit_color = 'red' if mode == 'color' else 'gray'
        ax.plot(x_fit, y_fit, color=fit_color, linewidth=2, label=f'Fit: n={n:.4f}, C={c:.4e}')
        
        # Configure axes
        configure_axes(
            ax,
            x_label='log(Pr² - Pwf²)',
            y_label='log(Q0)',
            title='Fetkovich Log-Log Plot',
            mode=mode,
            is_log_log=True
        )
        
        # Add legend
        ax.legend(loc='upper left', fontsize=8, frameon=True, edgecolor='black')
        plt.tight_layout()
        
        # Check for empty plot
        if len(ax.lines) == 0:
            logger.error("Empty plot in plot_fetkovich_log_log - closing and returning None")
            plt.close(fig)
            return None
        
        logger.info("Fetkovich log-log plot generated successfully")
        return fig
    
    except Exception as e:
        logger.error(f"Failed to plot Fetkovich log-log graph: {str(e)}")
        plt.close()
        return None

def plot_fetkovich_flow_after_flow(points, pr, c, n, mode='color'):
    """
    Plot flow-after-flow graph for Fetkovich IPR calculation.
    
    Parameters:
    - points: List of (q0, pwf) tuples
    - pr: Reservoir pressure
    - c, n: Fetkovich parameters
    - mode: 'color' or 'bw' for colorful or black-and-white plots
    
    Returns:
    - Matplotlib figure object or None if invalid
    """
    logger.info(f"Plotting Fetkovich flow-after-flow graph: pr={pr:.2f}, c={c:.4e}, n={n:.4f}")
    
    try:
        # Validate points
        points = validate_plotting_inputs(points, "Fetkovich flow-after-flow points", non_negative=True, finite=True)
        if not points:
            return None
        
        # Unpack points
        q_points, pwf_points = zip(*points)
        
        def fetkovich_model(pwf, c, n):
            return c * (pr**2 - pwf**2)**n
        
        # Initialize figure
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        fig.patch.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
        ax.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
        
        # Plot data points and fit
        point_color = 'blue' if mode == 'color' else 'black'
        ax.scatter(pwf_points, q_points, color=point_color, s=50, label='Test Points')
        pwf_range = np.linspace(0, pr, 100)
        q_flow = fetkovich_model(pwf_range, c, n)
        fit_color = 'red' if mode == 'color' else 'gray'
        ax.plot(pwf_range, q_flow, color=fit_color, linewidth=2, label=f'IPR Fit: n={n:.4f}, C={c:.4e}')
        
        # Configure axes
        configure_axes(
            ax,
            x_label='Flowing Bottomhole Pressure (Pwf, psi)',
            y_label='Production Rate (Q0, stb/day)',
            title='Flow After Flow Test Results',
            mode=mode,
            is_log_log=False
        )
        
        # Add legend
        ax.legend(loc='upper left', fontsize=8, frameon=True, edgecolor='black')
        plt.tight_layout()
        
        # Check for empty plot
        if len(ax.lines) == 0:
            logger.error("Empty plot in plot_fetkovich_flow_after_flow - closing and returning None")
            plt.close(fig)
            return None
        
        logger.info("Fetkovich flow-after-flow plot generated successfully")
        return fig
    
    except Exception as e:
        logger.error(f"Failed to plot Fetkovich flow-after-flow graph: {str(e)}")
        plt.close()
        return None

def plot_glr_graphs(reference_data, conduit_size, production_rate, mode='color'):
    """
    Plot pressure vs. depth curves for all GLRs in a given conduit size and production rate.
    
    Parameters:
    - reference_data: List of reference data dictionaries
    - conduit_size: Conduit size (2.875 or 3.5)
    - production_rate: Production rate in stb/day
    - mode: 'color' or 'bw' for colorful or black-and-white plots
    
    Returns:
    - Matplotlib figure object or None if invalid
    """
    logger.info(f"Plotting GLR graphs for conduit={conduit_size}, production_rate={production_rate}, mode={mode}")
    
    try:
        # Filter relevant data
        relevant_rows = [
            entry for entry in reference_data
            if (abs(entry['conduit_size'] - conduit_size) < 1e-6 and
                abs(entry['production_rate'] - production_rate) < 1e-6)
        ]
        relevant_rows.sort(key=lambda x: x['glr'])
        if not relevant_rows:
            logger.warning(f"No data points found for conduit {conduit_size} in, production {production_rate} stb/day")
            return None

        # Assign colors for GLR curves
        if mode == 'color':
            all_glrs = sorted(set(entry['glr'] for entry in reference_data))
            for i, glr in enumerate(all_glrs):
                GLR_COLOR_MAP[glr] = COLORS[i % len(COLORS)]
        else:
            for glr in set(entry['glr'] for entry in relevant_rows):
                GLR_COLOR_MAP[glr] = 'black'

        # Initialize figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        fig.patch.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
        ax.set_facecolor('#F5F5F5' if mode == 'color' else 'white')

        p1_full = np.linspace(0, 4000, 100)
        label_positions = []
        traces_added = 0

        # Plot GLR curves
        for entry in relevant_rows:
            glr = entry['glr']
            coeffs = entry['coefficients']

            p_plot = []
            y_plot = []
            for p in p1_full:
                y = polynomial(p, coeffs)
                if np.isfinite(y) and 0 <= y <= 31000:
                    p_plot.append(p)
                    y_plot.append(y)
                else:
                    if y > 31000:
                        break

            if len(p_plot) < 2:
                logger.warning(f"GLR {glr} has insufficient valid points ({len(p_plot)}). Skipping")
                continue

            line_color = GLR_COLOR_MAP[glr]
            label_text = f'GLR {int(glr) if glr.is_integer() else glr}' if mode == 'color' else None
            ax.plot(p_plot, y_plot, color=line_color, linewidth=2.5, label=label_text)
            traces_added += 1

            # Add text labels for black-and-white mode
            if mode == 'bw' and p_plot and y_plot:
                label_value = int(glr/100) if (glr/100).is_integer() else glr/100
                end_x, end_y = p_plot[-1], y_plot[-1] - 300
                overlap = False
                for prev_x, prev_y in label_positions:
                    if abs(end_y - prev_y) < 300 and abs(end_x - prev_x) < 100:
                        overlap = True
                        break
                if overlap:
                    index = max(0, len(p_plot) - 11)
                    end_x, end_y = p_plot[index], y_plot[index] - 300
                ax.text(end_x, end_y, f'{label_value}', fontsize=8, ha='left', va='center')
                label_positions.append((end_x, end_y))

        if traces_added == 0:
            logger.error("No valid curves added to GLR plot - closing and returning None")
            plt.close(fig)
            return None

        # Configure axes
        configure_axes(
            ax,
            x_label='Gradient Pressure, psi',
            y_label='Depth, ft',
            x_lim=(0, 4000),
            y_lim=(0, 31000),
            title=f"GLR Curves (Conduit: {conduit_size} in, Production: {production_rate} stb/day)",
            mode=mode,
            is_log_log=False
        )
        ax.invert_yaxis()
        
        # Add legend
        if mode == 'color':
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, frameon=True, edgecolor='black')
        else:
            ax.legend(['Multiply GLR line\nnumbers by 100'], loc='center left', 
                     bbox_to_anchor=(1.05, 0.5), fontsize=8, frameon=True, edgecolor='black')
        
        plt.tight_layout()
        
        # Check for empty plot
        if len(ax.lines) == 0:
            logger.error("Empty plot after setup in plot_glr_graphs - closing and returning None")
            plt.close(fig)
            return None
        
        logger.info(f"GLR graphs generated successfully with {traces_added} curves")
        return fig
    
    except Exception as e:
        logger.error(f"Failed to plot GLR curves: {str(e)}")
        plt.close()
        return None
