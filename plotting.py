import numpy as np
import plotly.graph_objects as go
from config import COLORS, GLR_COLOR_MAP
from utils import polynomial, setup_logging
from validators import get_valid_glr_range

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
    Returns Plotly figure.
    """
    logger.info(f"Plotting pressure vs. depth: p1={p1:.2f}, y1={y1:.2f}, p2={p2:.2f}, y2={y2:.2f}, Q0={production_rate}, GLR={glr_input}")
    
    # Generate data points for the curve
    p1_full = np.linspace(0, 4000, 100)
    y1_full = polynomial(p1_full, coeffs)
    y1_full = np.where((y1_full >= 0) & (y1_full <= 31000), y1_full, np.nan)

    # Create Plotly figure
    fig = go.Figure()
    line_color = COLORS[0] if mode == 'color' else 'black'
    
    # Add the main curve
    fig.add_trace(go.Scatter(
        x=p1_full,
        y=y1_full,
        mode='lines',
        name=f'GLR curve ({interpolation_status}, Q0={production_rate}, GLR={glr_input})',
        line=dict(color=line_color, width=2.5),
        hovertemplate='Pressure: %{x:.2f} psi<br>Depth: %{y:.2f} ft'
    ))
    
    # Add points for (p1, y1) and (p2, y2)
    fig.add_trace(go.Scatter(
        x=[p1],
        y=[y1],
        mode='markers',
        name=f'(p1, y1) = ({p1:.2f} psi, {y1:.2f} ft)',
        marker=dict(color=line_color if mode == 'color' else 'black', size=10, symbol='circle'),
        hovertemplate='p1: %{x:.2f} psi<br>y1: %{y:.2f} ft'
    ))
    fig.add_trace(go.Scatter(
        x=[p2],
        y=[y2],
        mode='markers',
        name=f'(p2, y2) = ({p2:.2f} psi, {y2:.2f} ft)',
        marker=dict(color=line_color if mode == 'color' else 'black', size=10, symbol='square'),
        hovertemplate='p2: %{x:.2f} psi<br>y2: %{y:.2f} ft'
    ))
    
    # Common grid settings
    gridcolor = '#D3D3D3' if mode == 'color' else 'black'
    minor_gridcolor = '#D3D3D3' if mode == 'color' else 'black'
    minor_gridopacity = 0.5 if mode == 'color' else 0.2
    
    # Update layout
    fig.update_layout(
        title=f'Pressure vs. Depth (Q0={production_rate} stb/day, GLR={glr_input}, {interpolation_status})',
        xaxis_title='Gradient Pressure, psi',
        yaxis_title='Depth, ft',
        yaxis_autorange='reversed',
        xaxis_range=[0, 4000],
        yaxis_range=[0, 31000],
        showlegend=True,
        template='plotly_white' if mode == 'color' else 'plotly',
        hovermode='closest',
        xaxis=dict(
            tick0=0,
            dtick=1000,
            showgrid=True,
            gridcolor=gridcolor,
            gridwidth=1,
            minor=dict(
                dtick=200,
                showgrid=True,
                gridcolor=minor_gridcolor,
                gridwidth=1
            ),
            title_standoff=10,
            side='top'
        ),
        yaxis=dict(
            tick0=0,
            dtick=1000,
            showgrid=True,
            gridcolor=gridcolor,
            gridwidth=1,
            minor=dict(
                dtick=200,
                showgrid=True,
                gridcolor=minor_gridcolor,
                gridwidth=1
            ),
            title_standoff=10
        ),
        legend=dict(
            x=1,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            bordercolor='black',
            borderwidth=1
        ),
        plot_bgcolor='#F5F5F5' if mode == 'color' else 'white',
        paper_bgcolor='#F5F5F5' if mode == 'color' else 'white'
    )
    
    logger.info("Pressure vs. depth plot generated successfully.")
    return fig

def plot_curves(tpr_points, ipr_points, pr, intersection_q0, intersection_p, glr, conduit_size, mode='color'):
    """
    Plot TPR and IPR curves with intersection point for Natural Flow Finder.
    Parameters:
    - tpr_points, ipr_points: Lists of (q0, pressure) tuples
    - pr: Reservoir pressure
    - intersection_q0, intersection_p: Intersection point
    - glr: Gas-Liquid Ratio
    - conduit_size: Conduit size (2.875 or 3.5)
    - mode: 'color' or 'bw' for colorful or black-and-white plots
    Returns Plotly figure.
    """
    logger.info(f"Plotting TPR/IPR curves: GLR={glr}, conduit_size={conduit_size}, pr={pr:.2f}")
    
    # Prepare data
    tpr_q0, tpr_p2 = zip(*tpr_points)
    ipr_q0, ipr_pwf = zip(*ipr_points)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add TPR curve
    tpr_color = COLORS[0] if mode == 'color' else 'black'
    fig.add_trace(go.Scatter(
        x=tpr_q0,
        y=tpr_p2,
        mode='lines+markers',
        name=f'TPR (GLR={glr}, conduit={conduit_size})',
        line=dict(color=tpr_color, width=2.5),
        marker=dict(size=8),
        hovertemplate='Q0: %{x:.2f} stb/day<br>P2: %{y:.2f} psi'
    ))
    
    # Add IPR curve
    ipr_color = COLORS[1] if mode == 'color' else 'gray'
    fig.add_trace(go.Scatter(
        x=ipr_q0,
        y=ipr_pwf,
        mode='lines+markers',
        name='IPR',
        line=dict(color=ipr_color, width=2.5),
        marker=dict(size=8),
        hovertemplate='Q0: %{x:.2f} stb/day<br>Pwf: %{y:.2f} psi'
    ))
    
    # Add intersection point if valid
    if intersection_q0 is not None and intersection_p is not None:
        fig.add_trace(go.Scatter(
            x=[intersection_q0],
            y=[intersection_p],
            mode='markers',
            name=f'Intersection (Q0={intersection_q0:.2f}, P={intersection_p:.2f})',
            marker=dict(color=COLORS[2] if mode == 'color' else 'black', size=12, symbol='star'),
            hovertemplate='Q0: %{x:.2f} stb/day<br>P: %{y:.2f} psi'
        ))
    
    # Common grid settings
    gridcolor = '#D3D3D3' if mode == 'color' else 'black'
    minor_gridcolor = '#D3D3D3' if mode == 'color' else 'black'
    minor_gridopacity = 0.5 if mode == 'color' else 0.2
    
    # Update layout
    fig.update_layout(
        title=f'TPR and IPR Curves (GLR={glr}, Conduit={conduit_size}, Pr={pr:.2f} psi)',
        xaxis_title='Production Rate, stb/day',
        yaxis_title='Pressure, psi',
        xaxis_range=[0, max(max(tpr_q0), max(ipr_q0)) * 1.1],
        yaxis_range=[0, max(pr, 4000) * 1.1],
        showlegend=True,
        template='plotly_white' if mode == 'color' else 'plotly',
        hovermode='closest',
        xaxis=dict(
            tick0=0,
            dtick=100,
            showgrid=True,
            gridcolor=gridcolor,
            gridwidth=1,
            minor=dict(
                dtick=20,
                showgrid=True,
                gridcolor=minor_gridcolor,
                gridwidth=1
            ),
            title_standoff=10
        ),
        yaxis=dict(
            tick0=0,
            dtick=1000,
            showgrid=True,
            gridcolor=gridcolor,
            gridwidth=1,
            minor=dict(
                dtick=200,
                showgrid=True,
                gridcolor=minor_gridcolor,
                gridwidth=1
            ),
            title_standoff=10
        ),
        legend=dict(
            x=0.5,
            y=-0.3,
            xanchor='center',
            yanchor='top',
            bordercolor='black',
            borderwidth=1
        ),
        plot_bgcolor='#F5F5F5' if mode == 'color' else 'white',
        paper_bgcolor='#F5F5F5' if mode == 'color' else 'white'
    )
    
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
    Returns Plotly figure.
    """
    logger.info(f"Plotting Fetkovich log-log graph: pr={pr:.2f}, c={c:.4e}, n={n:.4f}")
    
    if not points:
        logger.error("No valid points provided for Fetkovich log-log plot.")
        return None
    
    # Prepare data
    q_points, pwf_points = zip(*points)
    delta_p_square = [(pr**2 - pwf**2) for pwf in pwf_points]
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add data points
    fig.add_trace(go.Scatter(
        x=delta_p_square,
        y=q_points,
        mode='markers',
        name='Data Points',
        marker=dict(color=COLORS[0] if mode == 'color' else 'black', size=10),
        hovertemplate='ΔP²: %{x:.2f} psi²<br>Q0: %{y:.2f} stb/day'
    ))
    
    # Add fitted curve
    delta_p_range = np.logspace(np.log10(min(delta_p_square)), np.log10(max(delta_p_square)), 100)
    q_fitted = c * delta_p_range**n
    fig.add_trace(go.Scatter(
        x=delta_p_range,
        y=q_fitted,
        mode='lines',
        name=f'Fitted Curve (n={n:.2f})',
        line=dict(color=COLORS[1] if mode == 'color' else 'gray', width=2.5),
        hovertemplate='ΔP²: %{x:.2f} psi²<br>Q0: %{y:.2f} stb/day'
    ))
    
    # Common grid settings
    gridcolor = '#D3D3D3' if mode == 'color' else 'black'
    minor_gridcolor = '#D3D3D3' if mode == 'color' else 'black'
    minor_gridopacity = 0.5 if mode == 'color' else 0.2
    
    # Update layout
    fig.update_layout(
        title=f'Fetkovich Log-Log Plot (C={c:.2e}, n={n:.2f})',
        xaxis_title='ΔP² (Pr² - Pwf²), psi²',
        yaxis_title='Production Rate, stb/day',
        xaxis_type='log',
        yaxis_type='log',
        showlegend=True,
        template='plotly_white' if mode == 'color' else 'plotly',
        hovermode='closest',
        xaxis=dict(
            showgrid=True,
            gridcolor=gridcolor,
            gridwidth=1,
            minor=dict(
                showgrid=True,
                gridcolor=minor_gridcolor,
                gridwidth=1
            ),
            title_standoff=10
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=gridcolor,
            gridwidth=1,
            minor=dict(
                showgrid=True,
                gridcolor=minor_gridcolor,
                gridwidth=1
            ),
            title_standoff=10
        ),
        legend=dict(
            x=0,
            y=1,
            xanchor='left',
            yanchor='top',
            bordercolor='black',
            borderwidth=1
        ),
        plot_bgcolor='#F5F5F5' if mode == 'color' else 'white',
        paper_bgcolor='#F5F5F5' if mode == 'color' else 'white'
    )
    
    logger.info("Fetkovich log-log plot generated successfully.")
    return fig

def plot_fetkovich_flow_after_flow(points, pr, mode='color'):
    """
    Plot flow-after-flow graph for Fetkovich IPR calculation.
    Parameters:
    - points: List of (q0, pwf) tuples
    - pr: Reservoir pressure
    - mode: 'color' or 'bw' for colorful or black-and-white plots
    Returns Plotly figure.
    """
    logger.info(f"Plotting Fetkovich flow-after-flow graph: pr={pr:.2f}")
    
    if not points:
        logger.error("No valid points provided for Fetkovich flow-after-flow plot.")
        return None
    
    # Prepare data
    q_points, pwf_points = zip(*points)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add data points
    fig.add_trace(go.Scatter(
        x=q_points,
        y=pwf_points,
        mode='markers',
        name='Data Points',
        marker=dict(color=COLORS[0] if mode == 'color' else 'black', size=10),
        hovertemplate='Q0: %{x:.2f} stb/day<br>Pwf: %{y:.2f} psi'
    ))
    
    # Common grid settings
    gridcolor = '#D3D3D3' if mode == 'color' else 'black'
    minor_gridcolor = '#D3D3D3' if mode == 'color' else 'black'
    minor_gridopacity = 0.5 if mode == 'color' else 0.2
    
    # Update layout
    fig.update_layout(
        title=f'Fetkovich Flow-After-Flow Plot (Pr={pr:.2f} psi)',
        xaxis_title='Production Rate, stb/day',
        yaxis_title='Flowing Bottomhole Pressure, psi',
        yaxis_autorange='reversed',
        showlegend=True,
        template='plotly_white' if mode == 'color' else 'plotly',
        hovermode='closest',
        xaxis=dict(
            tick0=0,
            showgrid=True,
            gridcolor=gridcolor,
            gridwidth=1,
            minor=dict(
                showgrid=True,
                gridcolor=minor_gridcolor,
                gridwidth=1
            ),
            title_standoff=10
        ),
        yaxis=dict(
            tick0=0,
            dtick=1000,
            showgrid=True,
            gridcolor=gridcolor,
            gridwidth=1,
            minor=dict(
                dtick=200,
                showgrid=True,
                gridcolor=minor_gridcolor,
                gridwidth=1
            ),
            title_standoff=10
        ),
        legend=dict(
            x=0,
            y=1,
            xanchor='left',
            yanchor='top',
            bordercolor='black',
            borderwidth=1
        ),
        plot_bgcolor='#F5F5F5' if mode == 'color' else 'white',
        paper_bgcolor='#F5F5F5' if mode == 'color' else 'white'
    )
    
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
    Returns Plotly figure or None if invalid.
    """
    logger.info(f"Plotting GLR graphs for conduit={conduit_size}, production_rate={production_rate}, mode={mode}")
    
    # Validate inputs
    valid_ranges = get_valid_glr_range(conduit_size, production_rate)
    if not valid_ranges.startswith('['):
        logger.error(f"Invalid inputs for GLR plot: {valid_ranges}")
        return None
    
    # Validate reference_data structure
    if not isinstance(reference_data, list):
        logger.error("reference_data must be a list of dictionaries")
        return None
    
    for row in reference_data:
        if not isinstance(row, dict) or not all(key in row for key in ['conduit_size', 'production_rate', 'glr', 'coefficients']):
            logger.error("Invalid reference_data format: each row must be a dict with 'conduit_size', 'production_rate', 'glr', and 'coefficients'")
            return None
        if not isinstance(row['coefficients'], dict) or not all(key in row['coefficients'] for key in ['a', 'b', 'c', 'd', 'e', 'f']):
            logger.error("Invalid coefficients format in reference_data")
            return None
    
    # Filter reference data
    relevant_rows = [
        row for row in reference_data
        if (abs(row['conduit_size'] - conduit_size) < 1e-6 and
            abs(row['production_rate'] - production_rate) < 1e-6)
    ]
    relevant_rows.sort(key=lambda x: x['glr'])
    if not relevant_rows:
        logger.error(f"No reference data for conduit={conduit_size}, production_rate={production_rate}")
        return None
    
    # Assign colors for GLR values
    if mode == 'color':
        all_glrs = sorted(set(row['glr'] for row in reference_data))
        for i, glr in enumerate(all_glrs):
            if glr not in GLR_COLOR_MAP:
                GLR_COLOR_MAP[glr] = COLORS[i % len(COLORS)]
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Plot curves for each GLR
    p1_full = np.linspace(0, 4000, 100)
    traces_added = 0
    label_positions = []
    for row in relevant_rows:
        glr = row['glr']
        coeffs = row['coefficients']
        p_plot = []
        y_plot = []
        for p in p1_full:
            y = polynomial(p, coeffs)
            if np.isfinite(y) and 0 <= y <= 31000:
                p_plot.append(p)
                y_plot.append(y)
            else:
                if y > 31000:
                    break  # Stop plotting if depth exceeds 31000 ft
        
        if len(p_plot) < 2:
            logger.warning(f"GLR {glr} has insufficient valid points ({len(p_plot)}). Skipping.")
            continue
        
        # Assign color
        line_color = GLR_COLOR_MAP.get(glr, 'black') if mode == 'color' else 'black'
        
        # Add curve
        fig.add_trace(go.Scatter(
            x=p_plot,
            y=y_plot,
            mode='lines',
            name=f'GLR {int(glr) if glr.is_integer() else glr}' if mode == 'color' else None,
            line=dict(color=line_color, width=2.5),
            hovertemplate='Pressure: %{x:.2f} psi<br>Depth: %{y:.2f} ft'
        ))
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
            fig.add_annotation(
                x=end_x,
                y=end_y,
                text=f'{label_value}',
                showarrow=False,
                font=dict(size=8),
                xanchor='left',
                yanchor='middle'
            )
            label_positions.append((end_x, end_y))
    
    # Check if any traces were added
    if traces_added == 0:
        logger.error("No valid traces added to GLR plot")
        return None
    
    # Common grid settings
    gridcolor = '#D3D3D3' if mode == 'color' else 'black'
    minor_gridcolor = '#D3D3D3' if mode == 'color' else 'black'
    minor_gridopacity = 0.5 if mode == 'color' else 0.2
    
    # Update layout
    fig.update_layout(
        title=f'GLR Curves (Conduit: {conduit_size} in, Production: {production_rate} stb/day)',
        xaxis_title='Gradient Pressure, psi',
        yaxis_title='Depth, ft',
        yaxis_autorange='reversed',
        xaxis_range=[0, 4000],
        yaxis_range=[0, 31000],
        showlegend=True,
        template='plotly_white' if mode == 'color' else 'plotly',
        hovermode='closest',
        xaxis=dict(
            tick0=0,
            dtick=1000,
            showgrid=True,
            gridcolor=gridcolor,
            gridwidth=1,
            minor=dict(
                dtick=200,
                showgrid=True,
                gridcolor=minor_gridcolor,
                gridwidth=1
            ),
            title_standoff=10,
            side='top'
        ),
        yaxis=dict(
            tick0=0,
            dtick=1000,
            showgrid=True,
            gridcolor=gridcolor,
            gridwidth=1,
            minor=dict(
                dtick=200,
                showgrid=True,
                gridcolor=minor_gridcolor,
                gridwidth=1
            ),
            title_standoff=10
        ),
        legend=dict(
            x=1,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            bordercolor='black',
            borderwidth=1,
            title=dict(text='Multiply GLR line numbers by 100') if mode == 'bw' else None
        ),
        plot_bgcolor='#F5F5F5' if mode == 'color' else 'white',
        paper_bgcolor='#F5F5F5' if mode == 'color' else 'white'
    )
    
    logger.info(f"GLR graphs generated successfully with {traces_added} traces.")
    return fig
