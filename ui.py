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
from config import COLORS, INTERPOLATION_RANGES, PRODUCTION_RATES
from random_point_generator import run_random_point_generator
from ml_module import run_machine_learning
from theme_module import apply_theme

logger = setup_logging()

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
            "Wellhead Pressure (p1, psi):",
            min_value=0.0,
            max_value=4000.0,
            value=float(st.session_state.p2_finder_inputs['p1']),
            step=10.0,
            help="Enter the wellhead pressure (0 to 4000 psi)."
        )
        st.session_state.p2_finder_inputs['p1'] = p1
        
        D = st.number_input(
            "Depth Difference (D, ft):",
            min_value=0.0,
            max_value=31000.0,
            value=float(st.session_state.p2_finder_inputs['D']),
            step=100.0,
            help="Enter the depth difference (y1 + D ≤ 31000 ft)."
        )
        st.session_state.p2_finder_inputs['D'] = D
    
    if st.button("Calculate p2"):
        with st.spinner("Calculating p2..."):
            errors = []
            if not validate_conduit_size(conduit_size):
                errors.append("Invalid conduit size. Must be 2.875 or 3.5 inches.")
            if not validate_production_rate(production_rate):
                errors.append("Invalid production rate. Must be between 50 and 600 stb/day.")
            if not validate_glr(glr, conduit_size, production_rate, interpolation_ranges):
                errors.append("Invalid GLR for the selected conduit size and production rate.")
            if not validate_depth_and_pressure(D, p1, y1=0):
                errors.append("Invalid depth or pressure: Ensure p1 ≤ 4000 psi and y1 + D ≤ 31000 ft.")
            
            if errors:
                for error in errors:
                    st.error(error)
                logger.error(f"p2 Finder errors: {errors}")
            else:
                try:
                    results = calculate_results(reference_data, conduit_size, production_rate, glr, p1, D)
                    st.session_state.p2_finder_results = results
                    st.subheader("p2 Finder Results")
                    st.write(f"Bottomhole Pressure (p2): {results['p2']:.2f} psi")
                    st.write(f"Wellhead Depth (y1): 0 ft")
                    st.write(f"Bottomhole Depth (y2): {results['y2']:.2f} ft")
                    
                    fig = plot_results(results, conduit_size, production_rate, glr, colors=COLORS if st.session_state.get('theme', 'Light') != 'Monochrome' else None)
                    if fig:
                        st.pyplot(fig)
                        try:
                            st.download_button(
                                label="Download Plot as PNG",
                                data=export_plot_to_png(fig),
                                file_name="p2_finder_plot.png",
                                mime="image/png"
                            )
                        except Exception as e:
                            st.error(f"Failed to export plot as PNG: {str(e)}")
                            logger.error(f"p2 Finder PNG export failed: {str(e)}")
                    else:
                        st.error("Failed to generate plot.")
                        logger.error("p2 Finder plot returned None")
                except Exception as e:
                    st.error(f"Failed to calculate p2: {str(e)}")
                    logger.error(f"p2 Finder calculation failed: {str(e)}")

def run_natural_flow_finder(reference_data, interpolation_ranges, production_rates):
    """UI for Natural Flow Finder: Find natural flow rate by intersecting TPR and IPR curves."""
    logger.info("Running Natural Flow Finder UI")
    
    if 'natural_flow_inputs' not in st.session_state:
        st.session_state.natural_flow_inputs = {
            'conduit_size': 2.875,
            'production_rate': 100.0,
            'glr': 200.0,
            'pwh': 1000.0,
            'pr': 2000.0,
            'method': 'Fetkovich',
            'C': 0.01,
            'n': 1.0,
            'test_points': [(100, 1500), (200, 1200)]
        }
    
    st.subheader("Natural Flow Finder Inputs")
    col1, col2 = st.columns(2)
    
    with col1:
        conduit_size = st.selectbox(
            "Conduit Size (in):",
            [2.875, 3.5],
            index=[2.875, 3.5].index(st.session_state.natural_flow_inputs['conduit_size']),
            help="Select the conduit size (2.875 or 3.5 inches)."
        )
        st.session_state.natural_flow_inputs['conduit_size'] = conduit_size
        
        valid_prates, valid_glrs = get_valid_options(conduit_size)
        valid_prates = [float(pr) for pr in valid_prates]
        production_rate = st.selectbox(
            "Production Rate (stb/day):",
            valid_prates,
            index=valid_prates.index(st.session_state.natural_flow_inputs['production_rate']) if st.session_state.natural_flow_inputs['production_rate'] in valid_prates else 0,
            help="Select the production rate (50 to 600 stb/day)."
        )
        st.session_state.natural_flow_inputs['production_rate'] = production_rate
        
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
                value=float(st.session_state.natural_flow_inputs['glr']),
                step=100.0,
                help="Enter a custom GLR value (must be within valid ranges)."
            )
        else:
            glr = float(glr_option)
        st.session_state.natural_flow_inputs['glr'] = glr
    
    with col2:
        pwh = st.number_input(
            "Wellhead Pressure (pwh, psi):",
            min_value=0.0,
            max_value=4000.0,
            value=float(st.session_state.natural_flow_inputs['pwh']),
            step=10.0,
            help="Enter the wellhead pressure (0 to 4000 psi)."
        )
        st.session_state.natural_flow_inputs['pwh'] = pwh
        
        pr = st.number_input(
            "Reservoir Pressure (pr, psi):",
            min_value=0.0,
            max_value=4000.0,
            value=float(st.session_state.natural_flow_inputs['pr']),
            step=10.0,
            help="Enter the reservoir pressure (0 to 4000 psi)."
        )
        st.session_state.natural_flow_inputs['pr'] = pr
    
    method = st.selectbox(
        "IPR Method:",
        ["Fetkovich", "Vogel", "Composite"],
        index=["Fetkovich", "Vogel", "Composite"].index(st.session_state.natural_flow_inputs['method']),
        help="Select the IPR method for natural flow calculation."
    )
    st.session_state.natural_flow_inputs['method'] = method
    
    if method == "Fetkovich":
        input_type = st.radio("Fetkovich Input:", ["Direct Parameters", "Test Points"])
        if input_type == "Direct Parameters":
            C = st.number_input(
                "Fetkovich C:",
                min_value=0.0,
                value=float(st.session_state.natural_flow_inputs['C']),
                step=0.001,
                help="Enter Fetkovich C parameter (> 0)."
            )
            n = st.number_input(
                "Fetkovich n:",
                min_value=0.0,
                max_value=2.0,
                value=float(st.session_state.natural_flow_inputs['n']),
                step=0.1,
                help="Enter Fetkovich n parameter (0 to 2)."
            )
            st.session_state.natural_flow_inputs['C'] = C
            st.session_state.natural_flow_inputs['n'] = n
        else:
            num_points = st.number_input(
                "Number of Test Points (1-4):",
                min_value=1,
                max_value=4,
                value=len(st.session_state.natural_flow_inputs['test_points']),
                step=1
            )
            test_points = []
            for i in range(num_points):
                st.write(f"Test Point {i+1}")
                col3, col4 = st.columns(2)
                with col3:
                    Q0 = st.number_input(
                        f"Flow Rate (Q0, stb/day) {i+1}:",
                        min_value=0.0,
                        value=float(st.session_state.natural_flow_inputs['test_points'][i][0] if i < len(st.session_state.natural_flow_inputs['test_points']) else 100),
                        step=10.0
                    )
                with col4:
                    Pwf = st.number_input(
                        f"Bottomhole Pressure (Pwf, psi) {i+1}:",
                        min_value=0.0,
                        max_value=4000.0,
                        value=float(st.session_state.natural_flow_inputs['test_points'][i][1] if i < len(st.session_state.natural_flow_inputs['test_points']) else 1500),
                        step=10.0
                    )
                test_points.append((Q0, Pwf))
            st.session_state.natural_flow_inputs['test_points'] = test_points
    
    if st.button("Find Natural Flow"):
        with st.spinner("Calculating natural flow..."):
            errors = []
            if not validate_conduit_size(conduit_size):
                errors.append("Invalid conduit size. Must be 2.875 or 3.5 inches.")
            if not validate_production_rate(production_rate):
                errors.append("Invalid production rate. Must be between 50 and 600 stb/day.")
            if not validate_glr(glr, conduit_size, production_rate, interpolation_ranges):
                errors.append("Invalid GLR for the selected conduit size and production rate.")
            if not validate_pressure(pwh, max_value=4000.0):
                errors.append("Invalid wellhead pressure. Must be between 0 and 4000 psi.")
            if not validate_pressure(pr, max_value=4000.0):
                errors.append("Invalid reservoir pressure. Must be between 0 and 4000 psi.")
            if method == "Fetkovich" and input_type == "Direct Parameters":
                if not validate_fetkovich_parameters(C, n):
                    errors.append("Invalid Fetkovich parameters: C must be > 0, n must be between 0 and 2.")
            elif method == "Fetkovich":
                if not validate_fetkovich_points(test_points):
                    errors.append("Invalid test points: Q0 and Pwf must be > 0, up to 4 points.")
            
            if errors:
                for error in errors:
                    st.error(error)
                logger.error(f"Natural Flow Finder errors: {errors}")
            else:
                try:
                    if method == "Fetkovich" and input_type == "Test Points":
                        Q0 = [tp[0] for tp in test_points]
                        Pwf = [tp[1] for tp in test_points]
                        C, n = calculate_ipr_fetkovich(Q0, Pwf, pr)
                    elif method == "Fetkovich":
                        pass  # Use provided C, n
                    else:
                        C, n = None, None
                    
                    tpr_points = calculate_tpr_points(reference_data, conduit_size, production_rate, glr, pwh)
                    if method == "Fetkovich":
                        ipr_points = calculate_ipr_fetkovich(C, n, pr)
                    elif method == "Vogel":
                        ipr_points = calculate_ipr_vogel(pr)
                    else:
                        ipr_points = calculate_ipr_composite(pr)
                    
                    intersection = find_intersection(tpr_points, ipr_points)
                    st.session_state.natural_flow_results = {
                        'tpr_points': tpr_points,
                        'ipr_points': ipr_points,
                        'intersection': intersection
                    }
                    
                    st.subheader("Natural Flow Finder Results")
                    if intersection:
                        st.write(f"Natural Flow Rate: {intersection['Q']:.2f} stb/day")
                        st.write(f"Bottomhole Pressure: {intersection['Pwf']:.2f} psi")
                    else:
                        st.warning("No intersection found between TPR and IPR curves.")
                    
                    fig = plot_curves(tpr_points, ipr_points, intersection, method, colors=COLORS if st.session_state.get('theme', 'Light') != 'Monochrome' else None)
                    if fig:
                        st.pyplot(fig)
                        try:
                            st.download_button(
                                label="Download Plot as PNG",
                                data=export_plot_to_png(fig),
                                file_name="natural_flow_plot.png",
                                mime="image/png"
                            )
                        except Exception as e:
                            st.error(f"Failed to export plot as PNG: {str(e)}")
                            logger.error(f"Natural Flow PNG export failed: {str(e)}")
                    else:
                        st.error("Failed to generate plot.")
                        logger.error("Natural Flow plot returned None")
                    
                    if method == "Fetkovich" and input_type == "Test Points":
                        fig_log = plot_fetkovich_log_log(test_points, C, n, pr, colors=COLORS if st.session_state.get('theme', 'Light') != 'Monochrome' else None)
                        if fig_log:
                            st.subheader("Fetkovich Log-Log Plot")
                            st.pyplot(fig_log)
                            try:
                                st.download_button(
                                    label="Download Log-Log Plot as PNG",
                                    data=export_plot_to_png(fig_log),
                                    file_name="fetkovich_log_log.png",
                                    mime="image/png"
                                )
                            except Exception as e:
                                st.error(f"Failed to export log-log plot as PNG: {str(e)}")
                                logger.error(f"Fetkovich log-log PNG export failed: {str(e)}")
                        fig_flow = plot_fetkovich_flow_after_flow(test_points, pr, colors=COLORS if st.session_state.get('theme', 'Light') != 'Monochrome' else None)
                        if fig_flow:
                            st.subheader("Fetkovich Flow-After-Flow Plot")
                            st.pyplot(fig_flow)
                            try:
                                st.download_button(
                                    label="Download Flow-After-Flow Plot as PNG",
                                    data=export_plot_to_png(fig_flow),
                                    file_name="fetkovich_flow_after_flow.png",
                                    mime="image/png"
                                )
                            except Exception as e:
                                st.error(f"Failed to export flow-after-flow plot as PNG: {str(e)}")
                                logger.error(f"Fetkovich flow-after-flow PNG export failed: {str(e)}")
                except Exception as e:
                    st.error(f"Failed to calculate natural flow: {str(e)}")
                    logger.error(f"Natural Flow calculation failed: {str(e)}")

def run_glr_graph_drawer(reference_data, interpolation_ranges, production_rates):
    """UI for GLR Graph Drawer: Plot pressure vs. depth curves for all GLRs."""
    logger.info("Running GLR Graph Drawer UI")
    
    if 'glr_graph_inputs' not in st.session_state:
        st.session_state.glr_graph_inputs = {
            'conduit_size': 2.875,
            'production_rate': 100.0
        }
    
    st.subheader("GLR Graph Drawer Inputs")
    col1, col2 = st.columns(2)
    
    with col1:
        conduit_size = st.selectbox(
            "Conduit Size (in):",
            [2.875, 3.5],
            index=[2.875, 3.5].index(st.session_state.glr_graph_inputs['conduit_size']),
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
                    # Apply theme colors to plots
                    theme = st.session_state.get('theme', 'Light')
                    if mode == 'color' and theme != 'Monochrome':
                        fig = plot_glr_graphs(reference_data, conduit_size, production_rate, mode=mode, colors=COLORS)
                    else:
                        fig = plot_glr_graphs(reference_data, conduit_size, production_rate, mode=mode)
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

def run_random_point_generator():
    """UI for Random Point Generator: Generate and visualize random well performance data."""
    logger.info("Running Random Point Generator UI")
    run_random_point_generator()

def run_machine_learning():
    """UI for Machine Learning: Perform ML-based analysis on well performance data."""
    logger.info("Running Machine Learning UI")
    run_machine_learning()

def main():
    """Main function to set up the Streamlit app with tabs for different functionalities."""
    st.title("Well Performance Calculator")
    
    # Apply theme on page load
    apply_theme(st.session_state.get('theme', 'Light'), st.session_state.get('custom_colors', {}))
    
    # Placeholder for reference data and interpolation ranges
    reference_data = st.session_state.get('REFERENCE_DATA', [])
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
