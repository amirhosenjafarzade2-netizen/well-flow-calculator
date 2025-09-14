# theme_module.py
# Module for theme selection and application in Streamlit

import streamlit as st

def apply_theme(theme_name, custom_colors=None):
    """
    Apply a theme to the Streamlit app using CSS and Streamlit's theming.
    Args:
        theme_name (str): Name of the predefined theme ('Light', 'Dark', 'Blue', 'Green', 'Purple', 'High-Contrast', 'Monochrome', 'Custom').
        custom_colors (dict): Optional custom colors for background and text.
    """
    # Predefined themes
    themes = {
        'Light': {
            'primaryColor': '#FF4B4B',
            'backgroundColor': '#FFFFFF',
            'secondaryBackgroundColor': '#F0F2F6',
            'textColor': '#31333F',
            'font': 'sans serif'
        },
        'Dark': {
            'primaryColor': '#FF4B4B',
            'backgroundColor': '#1A1B1E',
            'secondaryBackgroundColor': '#2C2D30',
            'textColor': '#FFFFFF',
            'font': 'sans serif'
        },
        'Blue': {
            'primaryColor': '#1E90FF',
            'backgroundColor': '#E6F0FA',
            'secondaryBackgroundColor': '#B3D4FF',
            'textColor': '#000080',
            'font': 'sans serif'
        },
        'Green': {
            'primaryColor': '#32CD32',
            'backgroundColor': '#E8F5E9',
            'secondaryBackgroundColor': '#C8E6C9',
            'textColor': '#004D40',
            'font': 'sans serif'
        },
        'Purple': {
            'primaryColor': '#9C27B0',
            'backgroundColor': '#F3E5F5',
            'secondaryBackgroundColor': '#E1BEE7',
            'textColor': '#4A148C',
            'font': 'sans serif'
        },
        'High-Contrast': {
            'primaryColor': '#FFFF00',
            'backgroundColor': '#000000',
            'secondaryBackgroundColor': '#333333',
            'textColor': '#FFFFFF',
            'font': 'sans serif'
        },
        'Monochrome': {
            'primaryColor': '#666666',
            'backgroundColor': '#D3D3D3',
            'secondaryBackgroundColor': '#B0B0B0',
            'textColor': '#333333',
            'font': 'sans serif'
        }
    }

    # Use custom colors if provided, else use predefined theme
    if theme_name == 'Custom' and custom_colors:
        theme = {
            'primaryColor': custom_colors.get('primary', '#FF4B4B'),
            'backgroundColor': custom_colors.get('background', '#FFFFFF'),
            'secondaryBackgroundColor': custom_colors.get('secondary_background', '#F0F2F6'),
            'textColor': custom_colors.get('text', '#31333F'),
            'font': 'sans serif'
        }
    else:
        theme = themes.get(theme_name, themes['Light'])

    # Apply custom CSS for app-wide styling
    css = f"""
    <style>
        .stApp {{
            background-color: {theme['backgroundColor']};
            color: {theme['textColor']};
        }}
        .stButton>button {{
            background-color: {theme['primaryColor']};
            color: {theme['textColor']};
            border: 1px solid {theme['textColor']};
        }}
        .stTextInput>div>input, .stSelectbox>div>select, .stNumberInput>div>input {{
            background-color: {theme['secondaryBackgroundColor']};
            color: {theme['textColor']};
            border: 1px solid {theme['textColor']};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {theme['textColor']};
        }}
        .stMarkdown, .stRadio > label, .stCheckbox > label {{
            color: {theme['textColor']};
        }}
        .stPlotlyChart, .stPlotlyChart canvas {{
            background-color: {theme['secondaryBackgroundColor']} !important;
        }}
        /* Theme selector container styling */
        .theme-selector {{
            position: absolute;
            top: 10px;
            left: 10px;
            background-color: {theme['secondaryBackgroundColor']};
            padding: 10px;
            border-radius: 5px;
            border: 1px solid {theme['textColor']};
            z-index: 1000;
            width: 300px;
        }}
        .theme-selector select, .theme-selector button {{
            width: 100%;
            margin-bottom: 10px;
        }}
        .theme-selector .stColorPicker {{
            margin-bottom: 10px;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def theme_selector():
    """
    Display a theme selection interface in the top left and apply the selected theme.
    Persist theme in st.session_state.
    """
    # Wrap theme selector in a styled div
    st.markdown('<div class="theme-selector">', unsafe_allow_html=True)
    st.subheader("Theme Settings")
    
    # Initialize theme in session state
    if 'theme' not in st.session_state:
        st.session_state.theme = 'Light'
    if 'custom_colors' not in st.session_state:
        st.session_state.custom_colors = {}

    # Theme selection
    theme_options = ['Light', 'Dark', 'Blue', 'Green', 'Purple', 'High-Contrast', 'Monochrome', 'Custom']
    selected_theme = st.selectbox("Select Theme:", theme_options, index=theme_options.index(st.session_state.theme))
    
    custom_colors = st.session_state.custom_colors
    if selected_theme == 'Custom':
        st.write("Customize Colors:")
        col1, col2 = st.columns(2)
        with col1:
            custom_colors['background'] = st.color_picker("Background Color", value=custom_colors.get('background', '#FFFFFF'))
            custom_colors['text'] = st.color_picker("Text Color", value=custom_colors.get('text', '#31333F'))
        with col2:
            custom_colors['primary'] = st.color_picker("Primary Color (Buttons)", value=custom_colors.get('primary', '#FF4B4B'))
            custom_colors['secondary_background'] = st.color_picker("Secondary Background (Inputs)", value=custom_colors.get('secondary_background', '#F0F2F6'))

    # Apply theme when button is clicked
    if st.button("Apply Theme"):
        st.session_state.theme = selected_theme
        st.session_state.custom_colors = custom_colors if selected_theme == 'Custom' else {}
        apply_theme(selected_theme, custom_colors)
        st.success(f"Applied {selected_theme} theme!")

    # Apply theme on page load
    apply_theme(st.session_state.theme, st.session_state.custom_colors)
    
    st.markdown('</div>', unsafe_allow_html=True)
