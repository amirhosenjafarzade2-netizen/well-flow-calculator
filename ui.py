# ui.py
import streamlit as st
import numpy as np
from calculations import (calculate_results, calculate_tpr_points, calculate_ipr_fetkovich,
                         calculate_ipr_vogel, calculate_ipr_composite, find_intersection)
from plotting import (plot_results, plot_curves, plot_fetkovich_log_log,
                     plot_fetkovich_flow_after_flow, plot_glr_graphs)
from validators import (validate_conduit_size, validate_production_rate, validate_glr,
                       validate_depth_and_pressure, validate_pressure, get_valid_options,
                       get_valid_glr_range, validate_fetkovich_parameters, validate_fetkovich_points)
from utils import export_plot_to_png, setup_logging
from config import COLORS
from random_point_generator import run_random_point_generator
from ml_module import run_machine_learning

logger = setup_logging()

def apply_theme():
    """Apply dark or light theme based on session state."""
    if st.session_state.get('theme', 'light') == 'dark':
        st.markdown("""
            <style>
                .stApp {
                    background-color: #1e1e1e;
                    color: #ffffff;
                }
                .stTextInput > div > div > input, .stSelectbox > div > div > select {
                    background-color: #333333;
                    color: #ffffff;
                }
                .stButton > button {
                    background-color: #4CAF50;
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)
        return 'plotly_dark'
    return 'plotly_white'

def run_p2_finder(reference_data, interpolation_ranges, production_rates):
    """UI for p2 Finder: Calculate wellhead and bottomhole pressures and depths."""
    logger.info("Running p2 Finder UI")
    
    if 'p2_finder_inputs' not in st.session_state:
        st.session_state.p2_finder_inputs = {
            'conduit_size': 2.875,
            'production_rate': 100.0,
            'glr': 200.0,
            'p1': 1000.0,
            'D': 1000.0
        }
    
    st.subheader("p2 Finder Inputs")
    col1, col2 = st.columns(2)
    
    with col1:
        valid_conduits = [2.875, 3.5]
        conduit_size = st.selectbox(
            "Conduit Size (in):",
            valid_conduits,
            index=valid_conduits.index(st.session_state.p2_finder_inputs['conduit_size']),
            help="Select the conduit size (2.875 or 3.5 inches)."
        )
        st.session_state.p2_finder_inputs['conduit_size'] = conduit_size
        
        valid_prates, valid_glrs = get_valid_options(conduit_size)
        valid_prates = [float(pr) for pr in valid_prates]
        production_rate = st.selectbox(
            "Production Rate (stb/day):",
            valid_prates,
            index=valid_prates.index(st.session_state.p2_finder_inputs['production_rate']) if st.session_state.p2_finder_inputs['production_rate'] in valid_prates else 0,
            help="Select the production rate (50 to 600 stb/day)."
        )
        st.session_state.p2_finder_inputs['production_rate'] = production_rate
        
        valid_glrs_dict = {pr: [float(glr) for glr in glrs] for pr, glrs in valid_glrs.items()}
        glr_option = st.selectbox(
            "GLR (scf/stb):",
            ["Custom"] + valid_glrs_dict.get(production_rate, []),
            help="Select a valid GLR or enter a custom value."
        )
        if glr_option == "Custom":
            glr = st.number_input(
                "Custom GLR:",
                min_value=0.0,
                value=float(st.session_state.p2_finder_inputs['glr']),
                step=100.0,
                help="Enter a custom GLR value (must be within valid ranges)."
            )
        else:
            glr = float(glr_option)
        st.session_state.p2_finder_inputs['glr'] = glr
    
    with col2:
        p1 = st.number_input(
            "Wellhead Pressure, p1 (psi):",
            min_value=0.0,
            max_value=4000.0,
            value=float(st.session_state.p2_finder_inputs['p1']),
            step=10.0,
            help="Enter the wellhead pressure (0 to 4000 psi)."
        )
        st.session_state.p2_finder_inputs['p1'] = p1
        
        D = st.number_input(
            "Well Length, D (ft):",
            min_value=0.0,
            max_value=31000.0,
            value=float(st.session_state.p2_finder_inputs['D']),
            step=100.0,
            help="Enter the well length (y1 + D ≤ 31000 ft)."
        )
        st.session_state.p2_finder_inputs['D'] = D
    
    calculate = st.button("Calculate p2")
    
    if calculate:
        with st.spinner("Calculating..."):
            errors = []
            if not validate_conduit_size(conduit_size):
                errors.append("Invalid conduit size. Must be 2.875 or 3.5 inches.")
            if not validate_production_rate(production_rate):
                errors.append("Invalid production rate. Must be between 50 and 600 stb/day.")
            if not validate_glr(conduit_size, production_rate, glr):
                try:
                    valid_range = get_valid_glr_range(conduit_size, production_rate)
                    errors.append(f"Invalid GLR. Valid ranges: {valid_range}")
                    ranges = interpolation_ranges.get((conduit_size, production_rate), [])
                    if ranges:
                        min_glr, max_glr = ranges[0]
                        glr = min(max_glr, max(min_glr, glr))
                        st.info(f"GLR auto-corrected to {glr:.2f} scf/stb.")
                except Exception as e:
                    errors.append(f"Failed to validate GLR: {str(e)}")
                    logger.error(f"GLR validation error: {str(e)}")
            if not validate_pressure(p1, "wellhead pressure"):
                errors.append("Invalid wellhead pressure. Must be between 0 and 4000 psi.")
            if not validate_depth_and_pressure(0, D):
                errors.append("Invalid well length. Must be between 0 and 31000 ft.")
            
            if errors:
                for error in errors:
                    st.error(error)
                logger.error(f"p2 Finder errors: {errors}")
            else:
                try:
                    result = calculate_results(conduit_size, production_rate, glr, p1, D, reference_data)
                    if result[0] is None:
                        st.error("Calculation failed. Please check inputs and try again.")
                        logger.error("p2 Finder calculation returned None")
                    else:
                        y1, y2, p2, coeffs, interpolation_status, glr1, glr2 = result
                        st.subheader("p2 Finder Results")
                        st.write(f"**Depth y1**: {y1:.2f} ft")
                        st.write(f"**Depth y2**: {y2:.2f} ft")
                        st.write(f"**Bottomhole Pressure p2**: {p2:.2f} psi")
                        st.write(f"**Interpolation Status**: {interpolation_status}")
                        st.write(f"**GLR Range**: [{glr1:.2f}, {glr2:.2f}]")
                        st.session_state.p2_finder_results = {
                            'y1': y1, 'y2': y2, 'p2': p2, 'coeffs': coeffs,
                            'interpolation_status': interpolation_status,
                            'glr1': glr1, 'glr2': glr2
                        }
                        
                        try:
                            fig = plot_results(
                                p1, y1, y2, p2, D, coeffs, glr, interpolation_status,
                                production_rate, mode='color'
                            )
                            if fig is not None:
                                st.subheader("Pressure vs Depth Plot")
                                st.pyplot(fig)
                                
                                if len(fig.axes) > 0 and len(fig.axes[0].lines) > 0:
                                    try:
                                        st.download_button(
                                            label="Download Plot as PNG",
                                            data=export_plot_to_png(fig),
                                            file_name="pressure_vs_depth_plot.png",
                                            mime="image/png"
                                        )
                                    except Exception as e:
                                        st.error(f"Failed to export plot as PNG: {str(e)}")
                                        logger.error(f"PNG export failed: {str(e)}")
                                else:
                                    st.warning("Plot is empty - cannot export.")
                            else:
                                st.warning("Failed to generate plot.")
                        except Exception as e:
                            st.warning(f"Failed to plot pressure vs. depth graph: {str(e)}")
                            logger.error(f"Plotting failed: {str(e)}")
                except Exception as e:
                    st.error(f"Calculation failed: {str(e)}")
                    logger.error(f"p2 Finder calculation failed: {str(e)}")
    
    st.write("**Calculation Logs**")
    st.write("Any warnings or informational messages will appear here.")

def run_natural_flow_finder(reference_data, interpolation_ranges, production_rates):
    """UI for Natural Flow Finder: Calculate natural flow rate by intersecting TPR and IPR curves."""
    logger.info("Running Natural Flow Finder UI")
    
    if 'natural_flow_inputs' not in st.session_state:
        st.session_state.natural_flow_inputs = {
            'conduit_size': 2.875,
            'glr': 200.0,
            'D': 1000.0,
            'pwh': 1000.0,
            'pr': 3000.0,
            'ipr_method': 'Fetkovich',
            'fetkovich_input_method': 'Enter C and n directly',
            'c': 1e-5,
            'n': 0.5,
            'q01': 100.0,
            'pwf1': 2000.0,
            'q02': 200.0,
            'pwf2': 1500.0,
            'q03': 0.0,
            'pwf3': 0.0,
            'q04': 0.0,
            'pwf4': 0.0,
            'q_max': 500.0,
            'j_star': 0.5,
            'p_b': 2000.0
        }
    
    st.subheader("Natural Flow Finder Inputs")
    col1, col2 = st.columns(2)
    
    with col1:
        valid_conduits = [2.875, 3.5]
        conduit_size = st.selectbox(
            "Conduit Size (in):",
            valid_conduits,
            index=valid_conduits.index(st.session_state.natural_flow_inputs['conduit_size']),
            help="Select the conduit size (2.875 or 3.5 inches)."
        )
        st.session_state.natural_flow_inputs['conduit_size'] = conduit_size
        
        glr = st.number_input(
            "GLR (scf/stb):",
            min_value=0.0,
            value=st.session_state.natural_flow_inputs['glr'],
            step=100.0,
            help="Enter the Gas-Liquid Ratio."
        )
        st.session_state.natural_flow_inputs['glr'] = glr
        
        D = st.number_input(
            "Well Length, D (ft):",
            min_value=0.0,
            max_value=31000.0,
            value=st.session_state.natural_flow_inputs['D'],
            step=100.0,
            help="Enter the well length (must satisfy y1 + D ≤ 31000 ft)."
        )
        st.session_state.natural_flow_inputs['D'] = D
    
    with col2:
        pwh = st.number_input(
            "Wellhead Pressure, Pwh (psi):",
            min_value=0.0,
            max_value=4000.0,
            value=st.session_state.natural_flow_inputs['pwh'],
            step=10.0,
            help="Enter the wellhead pressure (0 to 4000 psi)."
        )
        st.session_state.natural_flow_inputs['pwh'] = pwh
        
        pr = st.number_input(
            "Reservoir Pressure, Pr (psi):",
            min_value=0.0,
            max_value=10000.0,
            value=st.session_state.natural_flow_inputs['pr'],
            step=10.0,
            help="Enter the reservoir pressure (0 to 10000 psi)."
        )
        st.session_state.natural_flow_inputs['pr'] = pr
    
    ipr_method = st.selectbox(
        "IPR Method:",
        ["Fetkovich", "Vogel", "Composite"],
        index=["Fetkovich", "Vogel", "Composite"].index(st.session_state.natural_flow_inputs['ipr_method']),
        help="Select the Inflow Performance Relationship method."
    )
    st.session_state.natural_flow_inputs['ipr_method'] = ipr_method
    
    if ipr_method == "Fetkovich":
        fetkovich_input_method = st.selectbox(
            "Fetkovich Input Method:",
            ["Enter C and n directly", "Calculate C and n from points"],
            index=0 if st.session_state.natural_flow_inputs['fetkovich_input_method'] == "Enter C and n directly" else 1
        )
        st.session_state.natural_flow_inputs['fetkovich_input_method'] = fetkovich_input_method
        
        if fetkovich_input_method == "Enter C and n directly":
            c = st.number_input(
                "Fetkovich C:",
                min_value=0.0,
                value=st.session_state.natural_flow_inputs['c'],
                step=1e-6,
                format="%.6e",
                help="Enter positive value for C."
            )
            st.session_state.natural_flow_inputs['c'] = c
            
            n = st.number_input(
                "Fetkovich n:",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.natural_flow_inputs['n'],
                step=0.01,
                help="Enter value between 0 and 2 for n."
            )
            st.session_state.natural_flow_inputs['n'] = n
        else:
            col3, col4 = st.columns(2)
            with col3:
                q01 = st.number_input(
                    "Q01 (stb/day):",
                    min_value=0.0,
                    value=st.session_state.natural_flow_inputs['q01'],
                    step=10.0
                )
                st.session_state.natural_flow_inputs['q01'] = q01
                
                pwf1 = st.number_input(
                    "Pwf1 (psi):",
                    min_value=0.0,
                    value=st.session_state.natural_flow_inputs['pwf1'],
                    step=10.0
                )
                st.session_state.natural_flow_inputs['pwf1'] = pwf1
                
                q02 = st.number_input(
                    "Q02 (stb/day):",
                    min_value=0.0,
                    value=st.session_state.natural_flow_inputs['q02'],
                    step=10.0
                )
                st.session_state.natural_flow_inputs['q02'] = q02
                
                pwf2 = st.number_input(
                    "Pwf2 (psi):",
                    min_value=0.0,
                    value=st.session_state.natural_flow_inputs['pwf2'],
                    step=10.0
                )
                st.session_state.natural_flow_inputs['pwf2'] = pwf2
            
            with col4:
                q03 = st.number_input(
                    "Q03 (stb/day):",
                    min_value=0.0,
                    value=st.session_state.natural_flow_inputs['q03'],
                    step=10.0
                )
                st.session_state.natural_flow_inputs['q03'] = q03
                
                pwf3 = st.number_input(
                    "Pwf3 (psi):",
                    min_value=0.0,
                    value=st.session_state.natural_flow_inputs['pwf3'],
                    step=10.0
                )
                st.session_state.natural_flow_inputs['pwf3'] = pwf3
                
                q04 = st.number_input(
                    "Q04 (stb/day):",
                    min_value=0.0,
                    value=st.session_state.natural_flow_inputs['q04'],
                    step=10.0
                )
                st.session_state.natural_flow_inputs['q04'] = q04
                
                pwf4 = st.number_input(
                    "Pwf4 (psi):",
                    min_value=0.0,
                    value=st.session_state.natural_flow_inputs['pwf4'],
                    step=10.0
                )
                st.session_state.natural_flow_inputs['pwf4'] = pwf4
    
    elif ipr_method == "Vogel":
        q_max = st.number_input(
            "q_max (stb/day):",
            min_value=0.0,
            value=st.session_state.natural_flow_inputs['q_max'],
            step=10.0,
            help="Enter the maximum production rate for Vogel method."
        )
        st.session_state.natural_flow_inputs['q_max'] = q_max
    
    elif ipr_method == "Composite":
        j_star = st.number_input(
            "J* (stb/day/psi):",
            min_value=0.0,
            value=st.session_state.natural_flow_inputs['j_star'],
            step=0.01,
            help="Enter the productivity index above bubble point."
        )
        st.session_state.natural_flow_inputs['j_star'] = j_star
        
        p_b = st.number_input(
            "P_b (psi):",
            min_value=0.0,
            value=st.session_state.natural_flow_inputs['p_b'],
            step=10.0,
            help="Enter the bubble point pressure."
        )
        st.session_state.natural_flow_inputs['p_b'] = p_b
    
    calculate = st.button("Calculate Natural Flow")
    
    if calculate:
        with st.spinner("Calculating..."):
            errors = []
            if not validate_conduit_size(conduit_size):
                errors.append("Invalid conduit size. Must be 2.875 or 3.5 inches.")
            if not validate_glr(conduit_size, 100, glr):  # Using average production rate for validation
                valid_range = get_valid_glr_range(conduit_size, 100)
                errors.append(f"Invalid GLR. Valid ranges: {valid_range}")
            if not validate_pressure(pwh, "wellhead pressure"):
                errors.append("Invalid wellhead pressure. Must be between 0 and 4000 psi.")
            if not validate_pressure(pr, "reservoir pressure", max_pressure=10000):
                errors.append("Invalid reservoir pressure. Must be between 0 and 10000 psi.")
            if not validate_depth_and_pressure(0, D):
                errors.append("Invalid well length. Must be such that y1 + D ≤ 31000 ft.")
            
            if ipr_method == "Fetkovich":
                if st.session_state.natural_flow_inputs['fetkovich_input_method'] == "Enter C and n directly":
                    if not validate_fetkovich_parameters(st.session_state.natural_flow_inputs['c'], st.session_state.natural_flow_inputs['n']):
                        errors.append("Invalid Fetkovich parameters C or n.")
                else:
                    fetkovich_points = [
                        (st.session_state.natural_flow_inputs['q01'], st.session_state.natural_flow_inputs['pwf1']),
                        (st.session_state.natural_flow_inputs['q02'], st.session_state.natural_flow_inputs['pwf2']),
                        (st.session_state.natural_flow_inputs['q03'], st.session_state.natural_flow_inputs['pwf3']),
                        (st.session_state.natural_flow_inputs['q04'], st.session_state.natural_flow_inputs['pwf4'])
                    ]
                    if not validate_fetkovich_points(fetkovich_points, pr):
                        errors.append("Invalid Fetkovich test points.")
            elif ipr_method == "Vogel":
                if st.session_state.natural_flow_inputs['q_max'] <= 0:
                    errors.append("Invalid q_max for Vogel method. Must be positive.")
            elif ipr_method == "Composite":
                if st.session_state.natural_flow_inputs['j_star'] <= 0 or st.session_state.natural_flow_inputs['p_b'] <= 0 or st.session_state.natural_flow_inputs['p_b'] > pr:
                    errors.append("Invalid parameters for Composite method.")
            
            if errors:
                for error in errors:
                    st.error(error)
                logger.error(f"Natural Flow Finder errors: {errors}")
            else:
                try:
                    # Calculate TPR points
                    tpr_points = calculate_tpr_points(conduit_size, glr, D, pwh, reference_data)
                    
                    # Calculate IPR points and params
                    ipr_params = {}
                    ipr_points = []
                    if ipr_method == "Fetkovich":
                        if st.session_state.natural_flow_inputs['fetkovich_input_method'] == "Enter C and n directly":
                            c = st.session_state.natural_flow_inputs['c']
                            n = st.session_state.natural_flow_inputs['n']
                            _, _, ipr_points, _ = calculate_ipr_fetkovich(pr, c=c, n=n)
                        else:
                            fetkovich_points = [
                                (st.session_state.natural_flow_inputs['q01'], st.session_state.natural_flow_inputs['pwf1']),
                                (st.session_state.natural_flow_inputs['q02'], st.session_state.natural_flow_inputs['pwf2']),
                                (st.session_state.natural_flow_inputs['q03'], st.session_state.natural_flow_inputs['pwf3']),
                                (st.session_state.natural_flow_inputs['q04'], st.session_state.natural_flow_inputs['pwf4'])
                            ]
                            c, n, ipr_points, _ = calculate_ipr_fetkovich(pr, points=fetkovich_points)
                        ipr_params = {'c': c, 'n': n}
                    elif ipr_method == "Vogel":
                        q_max = st.session_state.natural_flow_inputs['q_max']
                        _, ipr_points = calculate_ipr_vogel(pr, q_max)
                        ipr_params = {'q_max': q_max}
                    elif ipr_method == "Composite":
                        j_star = st.session_state.natural_flow_inputs['j_star']
                        p_b = st.session_state.natural_flow_inputs['p_b']
                        _, _, ipr_points = calculate_ipr_composite(pr, j_star, p_b)
                        ipr_params = {'j_star': j_star, 'p_b': p_b}
                    
                    # Find intersection
                    intersection_q0, intersection_p = find_intersection(tpr_points, ipr_points, pr)
                    
                    # Plot curves
                    fig = plot_curves(tpr_points, ipr_points, intersection_q0, intersection_p, conduit_size, glr, D, pwh, pr, ipr_params)
                    if fig is not None:
                        st.subheader("TPR and IPR Curves")
                        st.pyplot(fig)
                        
                        if len(fig.axes) > 0 and len(fig.axes[0].lines) > 0:
                            try:
                                st.download_button(
                                    label="Download Curves Plot as PNG",
                                    data=export_plot_to_png(fig),
                                    file_name="tpr_ipr_curves.png",
                                    mime="image/png"
                                )
                            except Exception as e:
                                st.error(f"Failed to export curves plot as PNG: {str(e)}")
                                logger.error(f"Curves PNG export failed: {str(e)}")
                        else:
                            st.warning("Curves plot is empty - cannot export.")
                        
                        # Additional plots for Fetkovich with points
                        if ipr_method == "Fetkovich" and st.session_state.natural_flow_inputs['fetkovich_input_method'] == "Calculate C and n from points":
                            try:
                                log_log_fig = plot_fetkovich_log_log(fetkovich_points, pr, c, n, mode='color')
                                if log_log_fig is not None:
                                    st.subheader("Fetkovich Log-Log Plot")
                                    st.pyplot(log_log_fig)
                                    
                                    if len(log_log_fig.axes) > 0 and len(log_log_fig.axes[0].lines) > 0:
                                        try:
                                            st.download_button(
                                                label="Download Log-Log Plot as PNG",
                                                data=export_plot_to_png(log_log_fig),
                                                file_name="fetkovich_log_log_plot.png",
                                                mime="image/png"
                                            )
                                        except Exception as e:
                                            st.error(f"Failed to export log-log plot as PNG: {str(e)}")
                                            logger.error(f"Log-log PNG export failed: {str(e)}")
                                    else:
                                        st.warning("Log-log plot is empty - cannot export.")
                                else:
                                    st.warning("Failed to generate Fetkovich log-log plot.")
                            except Exception as e:
                                st.warning(f"Failed to plot Fetkovich log-log graph: {str(e)}")
                                logger.error(f"Log-log plotting failed: {str(e)}")
                            
                            if len(fetkovich_points) >= 4:
                                try:
                                    flow_fig = plot_fetkovich_flow_after_flow(fetkovich_points, pr, c, n, mode='color')
                                    if flow_fig is not None:
                                        st.subheader("Fetkovich Flow-After-Flow Plot")
                                        st.pyplot(flow_fig)
                                        
                                        if len(flow_fig.axes) > 0 and len(flow_fig.axes[0].lines) > 0:
                                            try:
                                                st.download_button(
                                                    label="Download Flow-After-Flow Plot as PNG",
                                                    data=export_plot_to_png(flow_fig),
                                                    file_name="fetkovich_flow_after_flow_plot.png",
                                                    mime="image/png"
                                                )
                                            except Exception as e:
                                                st.error(f"Failed to export flow-after-flow plot as PNG: {str(e)}")
                                                logger.error(f"Flow-after-flow PNG export failed: {str(e)}")
                                        else:
                                            st.warning("Flow-after-flow plot is empty - cannot export.")
                                    else:
                                        st.warning("Failed to generate Fetkovich flow-after-flow plot.")
                                except Exception as e:
                                    st.warning(f"Failed to plot Fetkovich flow-after-flow graph: {str(e)}")
                                    logger.error(f"Flow-after-flow plotting failed: {str(e)}")
                    
                    # Display intersection results
                    st.subheader("Point of Natural Flow Results")
                    if intersection_q0 is not None and intersection_p is not None:
                        st.write(f"**Production Rate (Q0)**: {intersection_q0:.2f} stb/day")
                        st.write(f"**Pressure (P)**: {intersection_p:.2f} psi")
                        st.session_state.natural_flow_results = {
                            'intersection_q0': intersection_q0,
                            'intersection_p': intersection_p,
                            'tpr_points': tpr_points,
                            'ipr_points': ipr_points,
                            'ipr_params': ipr_params
                        }
                    else:
                        st.warning("No valid intersection point found. TPR and IPR curves are plotted above.")
                        logger.warning("No valid intersection point found")
                
                except Exception as e:
                    st.error(f"Calculation failed: {str(e)}")
                    logger.error(f"Natural Flow Finder calculation failed: {str(e)}")
    
    st.write("**Calculation Logs**")
    st.write("Any warnings or informational messages will appear here.")

def run_glr_graph_drawer(reference_data, interpolation_ranges, production_rates):
    """UI for GLR Curves: Plot pressure vs. depth curves for all GLRs at given conduit size and production rate."""
    logger.info("Running GLR Graph Drawer UI")
    
    if 'glr_graph_inputs' not in st.session_state:
        st.session_state.glr_graph_inputs = {
            'conduit_size': 2.875,
            'production_rate': 100.0
        }
    
    st.subheader("GLR Curves Inputs")
    col1, col2 = st.columns(2)
    
    with col1:
        valid_conduits = [2.875, 3.5]
        conduit_size = st.selectbox(
            "Conduit Size (in):",
            valid_conduits,
            index=valid_conduits.index(st.session_state.glr_graph_inputs['conduit_size']),
            key="glr_conduit",
            help="Select the conduit size (2.875 or 3.5 inches)."
        )
        st.session_state.glr_graph_inputs['conduit_size'] = conduit_size
    
    with col2:
        valid_prates, _ = get_valid_options(conduit_size)
        valid_prates = [float(pr) for pr in valid_prates]
        production_rate = st.selectbox(
            "Production Rate (stb/day):",
            valid_prates,
            index=valid_prates.index(st.session_state.glr_graph_inputs['production_rate']) if st.session_state.glr_graph_inputs['production_rate'] in valid_prates else 0,
            key="glr_prate",
            help="Select the production rate (50 to 600 stb/day)."
        )
        st.session_state.glr_graph_inputs['production_rate'] = production_rate
    
    graph_style = st.selectbox("Graph Style:", ["Colorful", "Black-and-White"])
    mode = 'color' if graph_style == "Colorful" else 'bw'
    
    plot_glr = st.button("Plot GLR Curves")
    
    if plot_glr:
        with st.spinner("Generating GLR Curves..."):
            errors = []
            if not validate_conduit_size(conduit_size):
                errors.append("Invalid conduit size. Must be 2.875 or 3.5 inches.")
            if not validate_production_rate(production_rate):
                errors.append("Invalid production rate. Must be between 50 and 600 stb/day.")
            
            if errors:
                for error in errors:
                    st.error(error)
                logger.error(f"GLR Graph Drawer errors: {errors}")
            else:
                try:
                    # Placeholder: Assume plot_glr_graphs returns (fig, glr_data)
                    fig = plot_glr_graphs(reference_data, conduit_size, production_rate, mode=mode)
                    glr_data = [...]  # Replace with actual glr_data from plot_glr_graphs
                    if fig is not None:
                        st.subheader("GLR Pressure vs Depth Curves")
                        st.pyplot(fig)
                        
                        if len(fig.axes) > 0 and len(fig.axes[0].lines) > 0:
                            try:
                                st.download_button(
                                    label="Download GLR Plot as PNG",
                                    data=export_plot_to_png(fig),
                                    file_name="glr_curves.png",
                                    mime="image/png"
                                )
                            except Exception as e:
                                st.error(f"Failed to export GLR plot as PNG: {str(e)}")
                                logger.error(f"GLR PNG export failed: {str(e)}")
                        else:
                            st.warning("GLR plot is empty - cannot export.")
                    else:
                        st.error("Failed to generate GLR plot.")
                        logger.error("GLR plot returned None")
                except Exception as e:
                    st.error(f"Failed to plot GLR curves: {str(e)}")
                    logger.error(f"GLR plotting failed: {str(e)}")
    
    st.write("**Plotting Logs**")
    st.write("Any warnings or informational messages will appear here.")

def main():
    """Main function to set up the Streamlit app with tabs for different functionalities."""
    st.title("Well Performance Calculator")
    apply_theme()
    
    # Placeholder for reference data and interpolation ranges
    reference_data = st.session_state.get('REFERENCE_DATA', [...])  # Use session state if available
    interpolation_ranges = INTERPOLATION_RANGES
    production_rates = PRODUCTION_RATES
    
    # Debug button for REFERENCE_DATA
    if st.button("Debug REFERENCE_DATA"):
        st.write(st.session_state.get('REFERENCE_DATA', "REFERENCE_DATA not found in session_state"))
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["p2 Finder", "Natural Flow Finder", "GLR Curves", "Random Point Generator", "Machine Learning"])
    
    with tab1:
        run_p2_finder(reference_data, interpolation_ranges, production_rates)
    
    with tab2:
        run_natural_flow_finder(reference_data, interpolation_ranges, production_rates)
    
    with tab3:
        run_glr_graph_drawer(reference_data, interpolation_ranges, production_rates)
    
    with tab4:
        run_random_point_generator()
    
    with tab5:
        run_machine_learning()

if __name__ == "__main__":
    main()
