import matplotlib.pyplot as plt
import numpy as np
from config import COLORS
from utils import setup_logging

# Global variable for GLR color map
GLR_COLOR_MAP = {}

# Initialize logger
logger = setup_logging()

def plot_results(p1, y1, y2, p2, D, coeffs, glr_input, interpolation_status, production_rate, mode='color'):
    """
    Plot the pressure vs. depth curve for p2 Finder or Natural Flow Finder.
    Parameters:
    - p1, y1, p2, y2: Pressure and depth points
    - D: Well length
    - coeffs: Polynomial coefficients
    - glr_input: Gas-Liquid Ratio
    - interpolation_status: 'exact' or 'interpolated'
    - production_rate: Production rate in stb/day
    - mode: 'color' or 'bw' for colorful or black-and-white plots
    Returns matplotlib figure.
    """
    logger.info(f"Plotting pressure vs. depth: p1={p1:.2f}, y1={y1:.2f}, p2={p2:.2f}, y2={y2:.2f}, Q0={production_rate}, GLR={glr_input}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    p1_full = np.linspace(0, 4000, 100)
    
    def polynomial(x, coeffs):
        return (coeffs['a'] * x**5 + coeffs['b'] * x**4 + coeffs['c'] * x**3 +
                coeffs['d'] * x**2 + coeffs['e'] * x + coeffs['f'])
    
    y1_full = []
    crossing_x = None
    max_iterations = 100
    iteration = 0
    for p in p1_full:
        if iteration >= max_iterations:
            break
        y = polynomial(p, coeffs)
        if np.isfinite(y) and y <= 31000:
            y1_full.append(y)
        else:
            if crossing_x is None and len(y1_full) > 0:
                def root_fn(x):
                    return polynomial(x, coeffs) - 31000
                try:
                    mid_guess = p1_full[max(0, len(y1_full) - 1)]
                    candidate = np.linspace(mid_guess, p, 10)[5]  # Simple approximation for fsolve
                    if 0 <= candidate <= 4000:
                        crossing_x = candidate
                        y1_full.append(31000)
                        break
                except Exception:
                    crossing_x = p1_full[len(y1_full) - 1]
                    y1_full.append(31000)
                    break
            else:
                y1_full.append(31000)
        iteration += 1
    
    curve_color = 'blue' if mode == 'color' else 'black'
    ax.plot(p1_full[:len(y1_full)], y1_full, color=curve_color, linewidth=2.5,
            label=f'GLR curve ({interpolation_status.capitalize()}, Q0={production_rate} stb/day, GLR={glr_input})')
    ax.scatter([p1], [y1], color=curve_color, s=50, label=f'(p1, y1) = ({p1:.2f} psi, {y1:.2f} ft)')
    ax.scatter([p2], [y2], color=curve_color, s=50, label=f'(p2, y2) = ({p2:.2f} psi, {y2:.2f} ft)')
    ax.plot([p1, p1], [y1, 0], color='red', linewidth=1, label='Connecting Line')
    ax.plot([p1, 0], [y1, y1], color='red', linewidth=1)
    ax.plot([p2, p2], [y2, 0], color='red', linewidth=1)
    ax.plot([p2, 0], [y2, y2], color='red', linewidth=1)
    ax.plot([0, 0], [y1, y2], color='green', linewidth=4, label=f'Well Length ({D:.2f} ft)')
    ax.set_xlabel('Gradient Pressure, psi', fontsize=10)
    ax.set_ylabel('Depth, ft', fontsize=10)
    ax.set_xlim(0, 4000)
    ax.set_ylim(0, 31000)
    ax.invert_yaxis()
    # Fix tilt: Add symmetric padding
    ax.margins(x=0.05, y=0.05)
    grid_color = '#D3D3D3' if mode == 'color' else 'black'
    ax.grid(True, which='major', color=grid_color)
    ax.grid(True, which='minor', color=grid_color, linestyle='-', alpha=0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(200))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1000))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(200))
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize=8, frameon=True, edgecolor='black', ncol=1)
    plt.tight_layout()
    fig.patch.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
    ax.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
    
    # Safety check for empty plot
    if len(ax.lines) == 0:
        logger.error("Empty plot in plot_results - closing and returning None")
        plt.close(fig)
        return None
    
    logger.info("Pressure vs. depth plot generated successfully.")
    return fig

def plot_curves(tpr_points, ipr_points, intersection_q0, intersection_p, conduit_size, glr, D, pwh, pr, ipr_params, mode='color'):
    """
    Plot TPR and IPR curves with intersection point for Natural Flow Finder.
    Parameters:
    - tpr_points, ipr_points: Lists of (q0, pressure) tuples
    - intersection_q0, intersection_p: Intersection point
    - conduit_size: Conduit size
    - glr: Gas-Liquid Ratio
    - D: Well length
    - pwh: Wellhead pressure
    - pr: Reservoir pressure
    - ipr_params: String for IPR parameters
    - mode: 'color' or 'bw' for colorful or black-and-white plots
    Returns matplotlib figure.
    """
    logger.info(f"Plotting TPR/IPR curves: GLR={glr}, conduit_size={conduit_size}, pr={pr:.2f}")
    
    tpr_q0, tpr_p2 = zip(*tpr_points)
    ipr_q0, ipr_pwf = zip(*ipr_points)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tpr_color = 'blue' if mode == 'color' else 'black'
    ax.plot(tpr_q0, tpr_p2, color=tpr_color, marker='o', linewidth=2, label=f'TPR (Conduit: {conduit_size} in, GLR: {glr}, D: {D} ft, Pwh: {pwh} psi)')
    
    ipr_color = 'red' if mode == 'color' else 'gray'
    ax.plot(ipr_q0, ipr_pwf, color=ipr_color, linewidth=2, label=f'IPR ({ipr_params})')
    
    if intersection_q0 is not None and intersection_p is not None:
        intersect_color = 'green' if mode == 'color' else 'black'
        ax.scatter([intersection_q0], [intersection_p], color=intersect_color, s=100, marker='*',
                   label=f'Natural Flow Point (Q0: {intersection_q0:.2f} stb/day, P: {intersection_p:.2f} psi)')
    
    ax.set_xlabel('Production Rate, Q0 (stb/day)', fontsize=10)
    ax.set_ylabel('Pressure, psi', fontsize=10)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, max(pr, 4000))
    # Fix tilt: Add symmetric padding
    ax.margins(x=0.05, y=0.05)
    grid_color = '#D3D3D3' if mode == 'color' else 'black'
    ax.grid(True, which='major', color=grid_color)
    ax.grid(True, which='minor', color=grid_color, linestyle='-', alpha=0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1000))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(200))
    ax.set_title('TPR and IPR Curves with Natural Flow Point')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize=8, frameon=True, 
              edgecolor='black', ncol=1)
    plt.tight_layout()
    fig.patch.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
    ax.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
    
    # Safety check for empty plot
    if len(ax.lines) == 0:
        logger.error("Empty plot in plot_curves - closing and returning None")
        plt.close(fig)
        return None
    
    logger.info("TPR/IPR curves plot generated successfully.")
    return fig

def plot_fetkovich_log_log(points, pr, c, n, mode='color'):
    """
    Plot log-log graph for Fetkovich IPR calculation.
    Parameters:
    - points: List of (q0, pwf) tuples
    - pr: Reservoir pressure
    - c, n: Fetkovich parameters
    - mode: 'color' or 'bw' for colorful or black-and-white plots
    Returns matplotlib figure.
    """
    logger.info(f"Plotting Fetkovich log-log graph: pr={pr:.2f}, c={c:.4e}, n={n:.4f}")
    
    if not points:
        logger.error("No valid points provided for Fetkovich log-log plot.")
        return None
    
    q_points, pwf_points = zip(*points)
    delta_p = pr**2 - np.array(pwf_points)**2
    x = np.log10(delta_p)
    y = np.log10(q_points)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    point_color = 'blue' if mode == 'color' else 'black'
    ax.scatter(x, y, color=point_color, s=50, label='Data Points')
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = np.log10(c) + n * x_fit
    fit_color = 'red' if mode == 'color' else 'gray'
    ax.plot(x_fit, y_fit, color=fit_color, linewidth=2, label=f'Fit: n={n:.4f}, C={c:.4e}')
    ax.set_xlabel('log(Pr² - Pwf²)', fontsize=10)
    ax.set_ylabel('log(Q0)', fontsize=10)
    ax.set_title('Fetkovich Log-Log Plot')
    ax.legend(loc='upper left')
    ax.grid(True)
    grid_color = '#D3D3D3' if mode == 'color' else 'black'
    ax.grid(True, color=grid_color, alpha=0.5)
    # Fix tilt: Add symmetric padding
    ax.margins(x=0.05, y=0.05)
    fig.patch.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
    ax.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
    
    # Safety check for empty plot
    if len(ax.lines) == 0:
        logger.error("Empty plot in plot_fetkovich_log_log - closing and returning None")
        plt.close(fig)
        return None
    
    logger.info("Fetkovich log-log plot generated successfully.")
    return fig

def plot_fetkovich_flow_after_flow(points, pr, c, n, mode='color'):
    """
    Plot flow-after-flow graph for Fetkovich IPR calculation.
    Parameters:
    - points: List of (q0, pwf) tuples
    - pr: Reservoir pressure
    - c, n: Fetkovich parameters
    - mode: 'color' or 'bw' for colorful or black-and-white plots
    Returns matplotlib figure.
    """
    logger.info(f"Plotting Fetkovich flow-after-flow graph: pr={pr:.2f}")
    
    if not points:
        logger.error("No valid points provided for Fetkovich flow-after-flow plot.")
        return None
    
    q_points, pwf_points = zip(*points)
    
    def fetkovich_model(pwf, c, n):
        return c * (pr**2 - pwf**2)**n
    
    fig, ax = plt.subplots(figsize=(8, 6))
    point_color = 'blue' if mode == 'color' else 'black'
    ax.scatter(pwf_points, q_points, color=point_color, s=50, label='Test Points')
    pwf_range = np.linspace(0, pr, 100)
    q_flow = fetkovich_model(pwf_range, c, n)
    fit_color = 'red' if mode == 'color' else 'gray'
    ax.plot(pwf_range, q_flow, color=fit_color, linewidth=2, label=f'IPR Fit: n={n:.4f}, C={c:.4e}')
    ax.set_xlabel('Flowing Bottomhole Pressure (Pwf, psi)', fontsize=10)
    ax.set_ylabel('Production Rate (Q0, stb/day)', fontsize=10)
    ax.set_title('Flow After Flow Test Results')
    ax.legend(loc='upper left')
    ax.grid(True)
    grid_color = '#D3D3D3' if mode == 'color' else 'black'
    ax.grid(True, color=grid_color, alpha=0.5)
    # Fix tilt: Add symmetric padding
    ax.margins(x=0.05, y=0.05)
    fig.patch.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
    ax.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
    
    # Safety check for empty plot
    if len(ax.lines) == 0:
        logger.error("Empty plot in plot_fetkovich_flow_after_flow - closing and returning None")
        plt.close(fig)
        return None
    
    logger.info("Fetkovich flow-after-flow plot generated successfully.")
    return fig

def plot_glr_graphs(reference_data, conduit_size, production_rate, mode='color'):
    """
    Plot pressure vs. depth curves for all GLRs in a given conduit size and production rate.
    Parameters:
    - reference_data: List of reference data dictionaries
    - conduit_size: Conduit size (2.875 or 3.5)
    - production_rate: Production rate in stb/day
    - mode: 'color' or 'bw' for colorful or black-and-white plots
    Returns matplotlib figure or None if invalid.
    """
    logger.info(f"Plotting GLR graphs for conduit={conduit_size}, production_rate={production_rate}, mode={mode}")
    
    # Filter relevant rows
    relevant_rows = [
        entry for entry in reference_data
        if (abs(entry['conduit_size'] - conduit_size) < 1e-6 and
            abs(entry['production_rate'] - production_rate) < 1e-6)
    ]
    relevant_rows.sort(key=lambda x: x['glr'])
    if not relevant_rows:
        logger.warning(f"No data points found for conduit {conduit_size} in, production {production_rate} stb/day.")
        return None

    # Assign colors to GLR values for consistency in colorful mode
    if mode == 'color':
        all_glrs = sorted(set(entry['glr'] for entry in reference_data))
        for i, glr in enumerate(all_glrs):
            GLR_COLOR_MAP[glr] = COLORS[i % len(COLORS)]
    else:
        for glr in set(entry['glr'] for entry in relevant_rows):
            GLR_COLOR_MAP[glr] = 'black'

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    # Set background based on mode
    fig.patch.set_facecolor('#F5F5F5' if mode == 'color' else 'white')
    ax.set_facecolor('#F5F5F5' if mode == 'color' else 'white')

    p1_full = np.linspace(0, 4000, 100)

    # Store label positions to check for overlaps in black-and-white mode
    label_positions = []
    traces_added = 0  # Counter for safety check

    for entry in relevant_rows:
        glr = entry['glr']
        coeffs = entry['coefficients']

        def polynomial(x, coeffs):
            try:
                return (coeffs['a'] * x**5 +
                        coeffs['b'] * x**4 +
                        coeffs['c'] * x**3 +
                        coeffs['d'] * x**2 +
                        coeffs['e'] * x +
                        coeffs['f'])
            except Exception:
                return np.nan

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
            logger.warning(f"GLR {glr} has insufficient valid points ({len(p_plot)}). Skipping.")
            continue

        # Plot the line
        line_color = GLR_COLOR_MAP[glr]
        label_text = f'GLR {int(glr) if glr.is_integer() else glr}' if mode == 'color' else None
        ax.plot(p_plot, y_plot, color=line_color, linewidth=2.5, 
                label=label_text)
        traces_added += 1

        # Add end-point label for black-and-white mode
        if mode == 'bw' and p_plot and y_plot:
            label_value = int(glr/100) if (glr/100).is_integer() else glr/100
            # Start with the last point, offset by 300 units upward
            end_x, end_y = p_plot[-1], y_plot[-1] - 300
            # Check for overlap with previous labels
            overlap = False
            for prev_x, prev_y in label_positions:
                # Check if the new label is too close (within 300 units in y-direction)
                if abs(end_y - prev_y) < 300 and abs(end_x - prev_x) < 100:
                    overlap = True
                    break
            if overlap:
                # Move up the curve by selecting an earlier point (e.g., 10 points back)
                index = max(0, len(p_plot) - 11)  # Ensure we don't go out of bounds
                end_x, end_y = p_plot[index], y_plot[index] - 300
            # Add the label
            ax.text(end_x, end_y, f'{label_value}', fontsize=8, ha='left', va='center')
            # Store the label position
            label_positions.append((end_x, end_y))

    # Safety check: No traces added
    if traces_added == 0:
        logger.error("No valid curves added to GLR plot - closing and returning None")
        plt.close(fig)
        return None

    ax.set_xlabel('Gradient Pressure, psi', fontsize=10)
    ax.set_ylabel('Depth, ft', fontsize=10)
    ax.set_xlim(0, 4000)
    ax.set_ylim(0, 31000)
    ax.invert_yaxis()
    # Fix tilt: Add symmetric padding
    ax.margins(x=0.05, y=0.05)
    # Grid lines: weaker black for black-and-white, light gray for colorful
    grid_color = '#D3D3D3' if mode == 'color' else 'black'
    ax.grid(True, which='major', color=grid_color, 
            alpha=0.5 if mode == 'color' else 0.3)
    ax.grid(True, which='minor', color=grid_color, 
            linestyle='-', alpha=0.5 if mode == 'color' else 0.2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(200))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1000))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(200))
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    # Legend: customized for black-and-white mode
    if mode == 'color':
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, frameon=True, edgecolor='black')
    else:
        ax.legend(['Multiply GLR line\nnumbers by 100'], loc='center left', 
                 bbox_to_anchor=(1.05, 0.5), fontsize=8, frameon=True, edgecolor='black')
    ax.set_title(f"GLR Curves (Conduit: {conduit_size} in, Production: {production_rate} stb/day)")
    
    # Final safety check for empty plot
    if len(ax.lines) == 0:
        logger.error("Empty plot after setup in plot_glr_graphs - closing and returning None")
        plt.close(fig)
        return None
    
    logger.info(f"GLR graphs generated successfully with {traces_added} curves.")
    return fig
