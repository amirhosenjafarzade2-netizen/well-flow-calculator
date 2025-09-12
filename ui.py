import streamlit as st
import numpy as np
from calculations import (calculate_results, calculate_tpr_points, calculate_ipr_fetkovich,
                         calculate_ipr_vogel, calculate_ipr_composite, find_intersection)
from plotting import (plot_results, plot_curves, plot_fetkovich_log_log,
                     plot_fetkovich_flow_after_flow, plot_glr_graphs)
from validators import (validate_conduit_size, validate_production_rate, validate_glr,
                       validate_depth_and_pressure, validate_pressure, get_valid_options, get_valid_glr_range)
from utils import export_results_to_excel, export_plot_to_png, setup_logging
from config import COLORS

# Initialize logger
logger = setup_logging()

def apply_theme():
    """Apply dark or light theme based on session state."""
    theme = st.session_state.get('theme', 'light')
    
    if theme == 'dark':
        st.markdown("""
            <style>
                .stApp {
                    background-color: #1e1e1e !important;
                    color: #ffffff !important;
                    padding-top: 50px !important; /* Add padding to avoid overlap with toggle button */
                }
                .stTextInput > div > div > input,
                .stNumberInput > div > div > input,
                .stSelectbox > div > div > select {
                    background-color: #333333 !important;
                    color: #ffffff !important;
                    border: 1px solid #555555 !important;
                }
                .stButton > button {
                    background-color: #4CAF50 !important;
                    color: white !important;
                    border: 1px solid #4CAF50 !important;
                }
                .stButton > button:hover {
                    background-color: #45a049 !important;
                }
                .stMarkdown, .stMarkdown p, .stMarkdown div {
                    color: #ffffff !important;
                }
                .stSelectbox > div > div > div,
                .stNumberInput > div > div > div,
                .stTextInput > div > div > div {
                    color: #ffffff !important;
                }
                /* Ensure sidebar follows theme */
                .css-1d391kg, .css-1v3fvcr {
                    background-color: #1e1e1e !important;
                    color: #ffffff !important;
                }
                /* Ensure headers and labels are styled */
                h1, h2, h3, h4, h5, h6, label {
                    color: #ffffff !important;
                }
            </style>
        """, unsafe_allow_html=True)
        return 'plotly_dark'
    else:
        st.markdown("""
            <style>
                .stApp {
                    background-color: #ffffff !important;
                    color: #000000 !important;
                    padding-top: 50px !important; /* Add padding to avoid overlap with toggle button */
                }
                .stTextInput > div > div > input,
                .stNumberInput > div > div > input,
                .stSelectbox > div > div > select {
                    background-color: #ffffff !important;
                    color: #000000 !important;
                    border: 1px solid #cccccc !important;
                }
                .stButton > button {
                    background-color: #4CAF50 !important;
                    color: white !important;
                    border: 1px solid #4CAF50 !important;
                }
                .stButton > button:hover {
                    background-color: #45a049 !important;
                }
                .stMarkdown, .stMarkdown p, .stMarkdown div {
                    color: #000000 !important;
                }
                .stSelectbox > div > div > div,
                .stNumberInput > div > div > div,
                .stTextInput > div > div > div {
                    color: #000000 !important;
                }
                /* Ensure sidebar follows theme */
                .css-1d391kg, .css-1v3fvcr {
                    background-color: #ffffff !important;
                    color: #000000 !important;
                }
                h1, h2, h3, h4, h5, h6, label {
                    color: #000000 !important;
                }
            </style>
        """, unsafe_allow_html=True)
        return 'plotly_white'

def render_theme_toggle():
    """Render a theme toggle button positioned at the top-right."""
    current_theme = st.session_state.get('theme', 'light')
    st.markdown(
        """
        <style>
            .theme-toggle-container {
                position: absolute;
                top: 10px;
                right: 10px;
                z-index: 1000;
            }
            .theme-toggle-button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }
            .theme-toggle-button:hover {
                background-color: #45a049;
            }
        </style>
        <div class="theme-toggle-container">
            <button class="theme-toggle-button" onclick="streamlitRerun()">
                Switch to {theme} Mode
            </button>
        </div>
        <script>
            function streamlitRerun() {
                // Simulate a click on a hidden Streamlit button to trigger rerun
                const hiddenButton = document.createElement('button');
                hiddenButton.style.display = 'none';
                hiddenButton.setAttribute('data-st-click', 'theme_toggle_hidden');
                document.body.appendChild(hiddenButton);
                hiddenButton.click();
                document.body.removeChild(hiddenButton);
            }
        </script>
        """.format(theme='Light' if current_theme == 'dark' else 'Dark'),
        unsafe_allow_html=True
    )
    
    # Hidden Streamlit button to handle the rerun
    if st.button("Toggle Theme", key="theme_toggle_hidden"):
        st.session_state.theme = 'dark' if current_theme == 'light' else 'light'
        st.rerun()

def run_p2_finder(reference_data, interpolation_ranges, production_rates):
    """UI for p2 Finder: Calculate wellhead and bottomhole pressures and depths."""
    logger.info("Running p2 Finder UI")
    
    # Render theme toggle at the top-right
    render_theme_toggle()
    
    # Initialize session state for inputs
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
    
    calculate = st.button("Calculate")
    
    if calculate:
        with st.spinner("Calculating..."):
            # Validate inputs
            errors = []
            if not validate_conduit_size(conduit_size):
                errors.append("Invalid conduit size.")
            if not validate_production_rate(production_rate):
                errors.append("Invalid production rate.")
            if not validate_glr(conduit_size, production_rate, glr):
                try:
                    valid_range = get_valid_glr_range(conduit_size, production_rate)
                    logger.debug(f"Valid GLR range for conduit_size={conduit_size}, production_rate={production_rate}: {valid_range}")
                    errors.append(f"Invalid GLR. Valid ranges: {str(valid_range)}")
                    ranges = interpolation_ranges.get((conduit_size, production_rate), [])
                    if ranges:
                        min_glr, max_glr = ranges[0]
                        glr = min(max_glr, max(min_glr, glr))
                        st.info(f"GLR auto-corrected to {glr} scf/stb.")
                except Exception as e:
                    errors.append(f"Failed to validate GLR: {str(e)}")
                    logger.error(f"GLR validation error: {str(e)}")
            
            if not validate_pressure(p1, "wellhead pressure"):
                errors.append("Invalid wellhead pressure.")
            if not validate_depth_and_pressure(0, D):
                errors.append("Invalid well length.")
            
            if errors:
                for error in errors:
                    st.error(error)
                logger.error(f"p2 Finder errors: {errors}")
            else:
                result = calculate_results(conduit_size, production_rate, glr, p1, D, reference_data)
                if result[0] is None:
                    st.error("Calculation failed. Please check inputs and try again.")
                else:
                    y1, y2, p2, coeffs, interpolation_status, glr1, glr2 = result
                    st.subheader("Results")
                    st.write(f"**Depth y1**: {y1:.2f} ft")
                    st.write(f"**Depth y2**: {y2:.2f} ft")
                    st.write(f"**Pressure p2**: {p2:.2f} psi")
                    st.write(f"**Interpolation Status**: {interpolation_status}")
                    st.write(f"**GLR Range**: [{glr1}, {glr2}]")
                    st.session_state.p2_finder_results = {
                        'y1': y1, 'y2': y2, 'p2': p2, 'coeffs': coeffs,
                        'interpolation_status': interpolation_status,
                        'glr_input': glr, 'production_rate': production_rate
                    }
                    
                    fig = plot_results(
                        p1, y1, y2, p2, D, coeffs, glr, interpolation_status, production_rate,
                        mode='color'
                    )
                    st.subheader("Pressure vs Depth Plot")
                    st.pyplot(fig)
                    
                    # Check for valid fig before download
                    if fig is not None and len(fig.axes) > 0 and len(fig.axes[0].lines) > 0:
                        try:
                            st.download_button(
                                label="Download Plot as PNG",
                                data=export_plot_to_png(fig),
                                file_name="p2_finder_plot.png",
                                mime="image/png"
                            )
                        except Exception as e:
                            st.error(f"Failed to export plot as PNG: {str(e)}")
                            logger.error(f"PNG export failed: {str(e)}")
                    else:
                        st.warning("Plot is empty - cannot export.")
    
    st.write("**Calculation Logs**")
    st.write("Any warnings or informational messages will appear here.")

def run_natural_flow_finder(reference_data, interpolation_ranges, production_rates):
    """UI for Natural Flow Finder: Find natural flow rate by intersecting TPR and IPR."""
    logger.info("Running Natural Flow Finder UI")
    
    # Render theme toggle at the top-right
    render_theme_toggle()
    
    if 'natural_flow_inputs' not in st.session_state:
        st.session_state.natural_flow_inputs = {
            'conduit_size': 2.875,
            'production_rate': 100.0,
            'glr': 200.0,
            'pwh': 1000.0,
            'D': 1000.0,
            'pr': 2000.0,
            'ipr_method': 'Fetkovich',
            'c': 0.0001,
            'n': 0.5,
            'q_max': 500.0,
            'j_star': 0.5,
            'p_b': 1000.0,
            'q01': 100.0,
            'pwf1': 1500.0,
            'q02': 200.0,
            'pwf2': 1000.0
        }
    
    st.subheader("Natural Flow Finder Inputs")
    col1, col2 = st.columns(2)
    
    with col1:
        valid_conduits = [2.875, 3.5]
        conduit_size = st.selectbox(
            "Conduit Size (in):",
            valid_conduits,
            index=valid_conduits.index(st.session_state.natural_flow_inputs['conduit_size']),
            key="nf_conduit",
            help="Select the conduit size (2.875 or 3.5 inches)."
        )
        st.session_state.natural_flow_inputs['conduit_size'] = conduit_size
        
        valid_prates, valid_glrs = get_valid_options(conduit_size)
        valid_prates = [float(pr) for pr in valid_prates]
        production_rate = st.selectbox(
            "Production Rate (stb/day):",
            valid_prates,
            index=valid_prates.index(st.session_state.natural_flow_inputs['production_rate']) if st.session_state.natural_flow_inputs['production_rate'] in valid_prates else 0,
            key="nf_prate",
            help="Select the production rate (50 to 600 stb/day)."
        )
        st.session_state.natural_flow_inputs['production_rate'] = production_rate
        
        valid_glrs_dict = {pr: [float(glr) for glr in glrs] for pr, glrs in valid_glrs.items()}
        glr_option = st.selectbox(
            "GLR (scf/stb):",
            ["Custom"] + valid_glrs_dict.get(production_rate, []),
            key="nf_glr",
            help="Select a valid GLR or enter a custom value."
        )
        if glr_option == "Custom":
            glr = st.number_input(
                "Custom GLR:",
                min_value=0.0,
                value=float(st.session_state.natural_flow_inputs['glr']),
                step=100.0,
                key="nf_custom_glr",
                help="Enter a custom GLR value (must be within valid ranges)."
            )
        else:
            glr = float(glr_option)
        st.session_state.natural_flow_inputs['glr'] = glr
    
    with col2:
        pwh = st.number_input(
            "Wellhead Pressure, pwh (psi):",
            min_value=0.0,
            max_value=4000.0,
            value=float(st.session_state.natural_flow_inputs['pwh']),
            step=10.0,
            help="Enter the wellhead pressure (0 to 4000 psi)."
        )
        st.session_state.natural_flow_inputs['pwh'] = pwh
        
        D = st.number_input(
            "Well Length, D (ft):",
            min_value=0.0,
            max_value=31000.0,
            value=float(st.session_state.natural_flow_inputs['D']),
            step=100.0,
            help="Enter the well length (y1 + D ≤ 31000 ft)."
        )
        st.session_state.natural_flow_inputs['D'] = D
        
        pr = st.number_input(
            "Reservoir Pressure, Pr (psi):",
            min_value=0.0,
            max_value=4000.0,
            value=float(st.session_state.natural_flow_inputs['pr']),
            step=10.0,
            help="Enter the reservoir pressure (0 to 4000 psi)."
        )
        st.session_state.natural_flow_inputs['pr'] = pr
    
    st.subheader("IPR Method")
    ipr_method = st.selectbox(
        "Select IPR Method:",
        ["Fetkovich", "Vogel", "Composite"],
        index=["Fetkovich", "Vogel", "Composite"].index(st.session_state.natural_flow_inputs['ipr_method']),
        help="Choose the IPR calculation method."
    )
    st.session_state.natural_flow_inputs['ipr_method'] = ipr_method
    
    c = None
    n = None
    q_max = None
    j_star = None
    p_b = None
    q01 = None
    pwf1 = None
    q02 = None
    pwf2 = None
    
    if ipr_method == "Fetkovich":
        col3, col4 = st.columns(2)
        with col3:
            c = st.number_input(
                "C (optional):",
                min_value=0.0,
                value=float(st.session_state.natural_flow_inputs['c']),
                step=0.00001,
                format="%.6f",
                help="Fetkovich C parameter (leave blank to calculate)."
            )
            n = st.number_input(
                "n (optional):",
                min_value=0.0,
                max_value=2.0,
                value=float(st.session_state.natural_flow_inputs['n']),
                step=0.1,
                help="Fetkovich n parameter (leave blank to calculate)."
            )
        with col4:
            q01 = st.number_input(
                "Q01 (stb/day):",
                min_value=0.0,
                value=float(st.session_state.natural_flow_inputs['q01']),
                step=10.0
            )
            pwf1 = st.number_input(
                "Pwf1 (psi):",
                min_value=0.0,
                value=float(st.session_state.natural_flow_inputs['pwf1']),
                step=10.0
            )
            q02 = st.number_input(
                "Q02 (stb/day):",
                min_value=0.0,
                value=float(st.session_state.natural_flow_inputs['q02']),
                step=10.0
            )
            pwf2 = st.number_input(
                "Pwf2 (psi):",
                min_value=0.0,
                value=float(st.session_state.natural_flow_inputs['pwf2']),
                step=10.0
            )
        st.session_state.natural_flow_inputs.update({'c': c, 'n': n, 'q01': q01, 'pwf1': pwf1, 'q02': q02, 'pwf2': pwf2})
    elif ipr_method == "Vogel":
        q_max = st.number_input(
            "Q_max (stb/day):",
            min_value=0.0,
            value=float(st.session_state.natural_flow_inputs['q_max']),
            step=10.0,
            help="Maximum production rate for Vogel method."
        )
        st.session_state.natural_flow_inputs['q_max'] = q_max
    elif ipr_method == "Composite":
        col3, col4 = st.columns(2)
        with col3:
            j_star = st.number_input(
                "J* (stb/day/psi):",
                min_value=0.0,
                value=float(st.session_state.natural_flow_inputs['j_star']),
                step=0.1,
                help="Productivity index for Composite method."
            )
        with col4:
            p_b = st.number_input(
                "Bubble Point Pressure, P_b (psi):",
                min_value=0.0,
                value=float(st.session_state.natural_flow_inputs['p_b']),
                step=10.0,
                help="Bubble point pressure for Composite method."
            )
        st.session_state.natural_flow_inputs.update({'j_star': j_star, 'p_b': p_b})
    
    calculate = st.button("Calculate and Plot", key="nf_calculate")
    
    if calculate:
        with st.spinner("Calculating..."):
            errors = []
            if not validate_conduit_size(conduit_size):
                errors.append("Invalid conduit size.")
            if not validate_production_rate(production_rate):
                errors.append("Invalid production rate.")
            if not validate_glr(conduit_size, production_rate, glr):
                try:
                    valid_range = get_valid_glr_range(conduit_size, production_rate)
                    logger.debug(f"Valid GLR range for conduit_size={conduit_size}, production_rate={production_rate}: {valid_range}")
                    errors.append(f"Invalid GLR. Valid ranges: {str(valid_range)}")
                    ranges = interpolation_ranges.get((conduit_size, production_rate), [])
                    if ranges:
                        min_glr, max_glr = ranges[0]
                        glr = min(max_glr, max(min_glr, glr))
                        st.info(f"GLR auto-corrected to {glr} scf/stb.")
                except Exception as e:
                    errors.append(f"Failed to validate GLR: {str(e)}")
                    logger.error(f"GLR validation error: {str(e)}")
            if not validate_pressure(pwh, "wellhead pressure"):
                errors.append("Invalid wellhead pressure.")
            if not validate_pressure(pr, "reservoir pressure"):
                errors.append("Invalid reservoir pressure.")
            if not validate_depth_and_pressure(0, D):
                errors.append("Invalid well length.")
            
            if errors:
                for error in errors:
                    st.error(error)
                logger.error(f"Natural Flow Finder errors: {errors}")
            else:
                try:
                    tpr_points = calculate_tpr_points(conduit_size, glr, D, pwh, reference_data)
                    fetkovich_points = []
                    if ipr_method == "Fetkovich":
                        c_val, n_val, ipr_points, fetkovich_points = calculate_ipr_fetkovich(
                            pr, c=c if c is not None and c > 0 else None, n=n if n is not None and n > 0 else None, 
                            q01=q01, pwf1=pwf1, q02=q02, pwf2=pwf2
                        )
                        c = c_val
                        n = n_val
                    elif ipr_method == "Vogel":
                        _, ipr_points = calculate_ipr_vogel(pr, q_max)
                    else:  # Composite
                        _, _, ipr_points = calculate_ipr_composite(pr, j_star, p_b)
                    
                    intersection_q0, intersection_p = find_intersection(tpr_points, ipr_points, pr)
                    
                    st.subheader("Point of Natural Flow Results")
                    if intersection_q0 is not None and intersection_p is not None:
                        st.write(f"**Natural Flow Rate**: {intersection_q0:.2f} stb/day")
                        st.write(f"**Flowing Bottomhole Pressure**: {intersection_p:.2f} psi")
                    else:
                        st.warning("No valid natural flow point found.")
                    
                    st.session_state.natural_flow_results = {
                        'tpr_points': tpr_points,
                        'ipr_points': ipr_points,
                        'intersection_q0': intersection_q0,
                        'intersection_p': intersection_p,
                        'pr': pr,
                        'glr': glr,
                        'conduit_size': conduit_size,
                        'D': D,
                        'pwh': pwh,
                        'fetkovich_points': fetkovich_points,
                        'ipr_method': ipr_method,
                        'c': c,
                        'n': n
                    }
                    
                    if tpr_points and ipr_points:
                        st.download_button(
                            label="Download Results as Excel",
                            data=export_results_to_excel(tpr_points, ipr_points, intersection_q0, intersection_p),
                            file_name="natural_flow_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    ipr_params_str = f'Pr: {pr} psi, Params: {ipr_method}'
                    fig = plot_curves(
                        tpr_points, ipr_points, intersection_q0, intersection_p, conduit_size, glr, D, pwh, pr, ipr_params_str,
                        mode='color'
                    )
                    st.subheader("TPR and IPR Curves (Intersection indicates Point of Natural Flow)")
                    st.pyplot(fig)
                    
                    # Check for valid fig before download
                    if fig is not None and len(fig.axes) > 0 and len(fig.axes[0].lines) > 0:
                        try:
                            st.download_button(
                                label="Download TPR/IPR Plot as PNG",
                                data=export_plot_to_png(fig),
                                file_name="tpr_ipr_plot.png",
                                mime="image/png"
                            )
                        except Exception as e:
                            st.error(f"Failed to export plot as PNG: {str(e)}")
                            logger.error(f"PNG export failed: {str(e)}")
                    else:
                        st.warning("Plot is empty - cannot export.")
                    
                    if ipr_method == "Fetkovich" and fetkovich_points:
                        fig_log = plot_fetkovich_log_log(fetkovich_points, pr, c, n, mode='color')
                        if fig_log is not None and len(fig_log.axes) > 0:
                            st.subheader("Fetkovich Log-Log Plot")
                            st.pyplot(fig_log)
                            if len(fig_log.axes) > 0 and len(fig_log.axes[0].lines) > 0:
                                try:
                                    st.download_button(
                                        label="Download Log-Log Plot as PNG",
                                        data=export_plot_to_png(fig_log),
                                        file_name="fetkovich_log_log.png",
                                        mime="image/png"
                                    )
                                except Exception as e:
                                    st.error(f"Failed to export plot as PNG: {str(e)}")
                                    logger.error(f"PNG export failed: {str(e)}")
                        
                        fig_faf = plot_fetkovich_flow_after_flow(fetkovich_points, pr, c, n, mode='color')
                        if fig_faf is not None and len(fig_faf.axes) > 0:
                            st.subheader("Flow After Flow Plot")
                            st.pyplot(fig_faf)
                            if len(fig_faf.axes) > 0 and len(fig_faf.axes[0].lines) > 0:
                                try:
                                    st.download_button(
                                        label="Download Flow-After-Flow Plot as PNG",
                                        data=export_plot_to_png(fig_faf),
                                        file_name="fetkovich_flow_after_flow.png",
                                        mime="image/png"
                                    )
                                except Exception as e:
                                    st.error(f"Failed to export plot as PNG: {str(e)}")
                                    logger.error(f"PNG export failed: {str(e)}")
                
                except ValueError as e:
                    st.error(f"Calculation failed: {str(e)}")
                    logger.error(f"Natural Flow Finder calculation failed: {str(e)}")
    
    st.write("**Calculation Logs**")
    st.write("Any warnings or informational messages will appear here.")

def run_glr_graph_drawer(reference_data, interpolation_ranges, production_rates):
    """UI for GLR Graph Drawer: Plot pressure vs. depth for all GLRs."""
    logger.info("Running GLR Graph Drawer UI")
    
    # Render theme toggle at the top-right
    render_theme_toggle()
    
    st.subheader("GLR Graph Drawer Inputs")
    col1, col2 = st.columns(2)
    
    with col1:
        valid_conduits = [2.875, 3.5]
        conduit_size = st.selectbox(
            "Conduit Size (in):",
            valid_conduits,
            index=0,
            help="Select the conduit size (2.875 or 3.5 inches)."
        )
    
    with col2:
        valid_prates, _ = get_valid_options(conduit_size)
        valid_prates = [float(pr) for pr in valid_prates]
        production_rate = st.selectbox(
            "Production Rate (stb/day):",
            valid_prates,
            index=0,
            help="Select the production rate (50 to 600 stb/day)."
        )
    
    graph_style = st.selectbox(
        "Graph Style:",
        ["Colorful", "Black-and-White"],
        index=0,
        help="Choose colorful or black-and-white GLR graphs."
    )
    plot_mode = "color" if graph_style == "Colorful" else "bw"
    
    plot = st.button("Generate GLR Graphs")
    
    if plot:
        with st.spinner("Generating graphs..."):
            errors = []
            if not validate_conduit_size(conduit_size):
                errors.append("Invalid conduit size.")
            if not validate_production_rate(production_rate):
                errors.append("Invalid production rate.")
            if not reference_data:
                errors.append("Reference data is empty or invalid.")
            
            if errors:
                for error in errors:
                    st.error(error)
                logger.error(f"GLR Graph Drawer errors: {errors}")
            else:
                try:
                    fig = plot_glr_graphs(reference_data, conduit_size, production_rate, mode=plot_mode)
                    if fig is not None:
                        st.subheader("GLR Graphs")
                        st.write(f"Conduit Size: {conduit_size} in, Production Rate: {production_rate} stb/day")
                        st.pyplot(fig)
                        
                        # Check for valid fig before download
                        if len(fig.axes) > 0 and len(fig.axes[0].lines) > 0:
                            try:
                                st.download_button(
                                    label="Download GLR Plot as PNG",
                                    data=export_plot_to_png(fig),
                                    file_name=f"glr_plot_conduit{conduit_size}_q0{production_rate}.png",
                                    mime="image/png"
                                )
                            except Exception as e:
                                st.error(f"Failed to export plot as PNG: {str(e)}")
                                logger.error(f"PNG export failed: {str(e)}")
                        else:
                            st.warning("Plot is empty - cannot export.")
                    else:
                        st.error("No valid GLR curves generated. Please check reference data.")
                        logger.error("No valid GLR curves generated.")
                
                except Exception as e:
                    st.error(f"Failed to generate GLR graphs: {str(e)}")
                    logger.error(f"GLR Graph Drawer failed: {str(e)}")
    
    st.write("**Plotting Logs**")
    st.write("Any warnings or informational messages will appear here.")

def post_task_menu():
    """Display a button to return to the main menu."""
    # Render theme toggle at the top-right
    render_theme_toggle()
    
    if st.button("Back to Main Menu"):
        st.session_state.mode_select = None
        st.rerun()

# Apply theme at the start to ensure consistent styling
apply_theme()

# Example main function to tie everything together (if needed)
def main():
    st.title("Well Performance Analysis Tool")
    if 'mode_select' not in st.session_state:
        st.session_state.mode_select = None
    
    # Render theme toggle for the main menu
    render_theme_toggle()
    
    # Example menu logic (adjust based on your actual main function)
    mode = st.selectbox(
        "Select Mode:",
        ["Select a mode", "p2 Finder", "Natural Flow Finder", "GLR Graph Drawer"],
        index=0 if st.session_state.mode_select is None else ["Select a mode", "p2 Finder", "Natural Flow Finder", "GLR Graph Drawer"].index(st.session_state.mode_select)
    )
    
    # Placeholder for reference_data, interpolation_ranges, production_rates
    # Replace with actual data loading logic
    reference_data = {}  # Example placeholder
    interpolation_ranges = {}  # Example placeholder
    production_rates = []  # Example placeholder
    
    if mode != "Select a mode":
        st.session_state.mode_select = mode
        if mode == "p2 Finder":
            run_p2_finder(reference_data, interpolation_ranges, production_rates)
        elif mode == "Natural Flow Finder":
            run_natural_flow_finder(reference_data, interpolation_ranges, production_rates)
        elif mode == "GLR Graph Drawer":
            run_glr_graph_drawer(reference_data, interpolation_ranges, production_rates)
    else:
        st.write("Please select a mode to proceed.")

if __name__ == "__main__":
    main()
