import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import requests
import io
import re
import time
from config import INTERPOLATION_RANGES, PRODUCTION_RATES, GITHUB_URL
from utils import setup_logging
from random_point_generator import generate_df
from validators import validate_conduit_size, validate_production_rate, get_valid_options, get_valid_glr_range, validate_glr, validate_positive_integer

logger = setup_logging()

def load_reference_data():
    """
    Load reference Excel file (referenceexcel.xlsx) from GitHub and parse into a list of dictionaries.
    Returns None if loading or parsing fails.
    """
    logger.info("Loading reference Excel file from GitHub...")
    try:
        response = requests.get(GITHUB_URL)
        response.raise_for_status()
        file_like_object = io.BytesIO(response.content)
        df_ref = pd.read_excel(file_like_object, header=None, engine='openpyxl')
        if df_ref.shape[1] < 6:
            st.error("Invalid Excel file: Must have at least 6 columns (name + 5 or 6 coefficients).")
            logger.error("Excel file has insufficient columns.")
            return None
        data_ref = []
        for index, row in df_ref.iterrows():
            name = row[0]
            if pd.isna(name) or isinstance(name, (int, float)):
                logger.warning(f"Skipping row {index} due to invalid name: {name}")
                continue
            name = str(name).strip()
            if not re.match(r'[\d.]+\s*in\s*\d+\s*stb-day\s*\d+\s*glr', name.lower()):
                logger.warning(f"Failed to parse reference data name: {name}")
                continue
            parts = name.split()
            try:
                conduit_size = float(parts[0])
                production_rate = float(parts[2])
                glr = float(parts[4].replace('glr', ''))
                coefficients = {
                    'a': float(row[1]),
                    'b': float(row[2]),
                    'c': float(row[3]),
                    'd': float(row[4]),
                    'e': float(row[5]),
                    'f': float(row[6]) if len(row) > 6 and pd.notna(row[6]) else 0.0
                }
                data_ref.append({
                    'conduit_size': conduit_size,
                    'production_rate': production_rate,
                    'glr': glr,
                    'coefficients': coefficients
                })
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing row {index}: {e}")
                continue
        if not data_ref:
            st.error("No valid data parsed from referenceexcel.xlsx.")
            logger.error("No valid data parsed from the Excel file.")
            return None
        logger.info("Reference data loaded successfully from referenceexcel.xlsx.")
        return data_ref
    except Exception as e:
        st.error(f"Error loading referenceexcel.xlsx from GitHub: {str(e)}")
        logger.error(f"Error loading reference Excel: {str(e)}")
        return None

def load_ml_data(reference_data, conduit_size, production_rate, num_points, glr=None, min_D=1000, all_prates=False, both_conduits=False):
    """
    Generate ML data using random_point_generator's generate_df function.
    If glr is None, use all valid GLRs. If all_prates is True, use all valid production rates.
    If both_conduits is True, use both conduit sizes (2.875 and 3.5).
    """
    dfs_ml = []
    required_cols = ["p1", "D", "p2"]  # Simplified, excluding y1 and y2 as they are derivable
    
    conduit_sizes = [2.875, 3.5] if both_conduits else [conduit_size]
    valid_prates_dict = {cs: [float(pr) for pr in get_valid_options(cs)[0]] for cs in conduit_sizes}
    
    total_iterations = 0
    for cs in conduit_sizes:
        prates = valid_prates_dict[cs] if all_prates else [production_rate]
        for pr in prates:
            filtered_data = [
                entry for entry in reference_data 
                if entry['conduit_size'] == cs 
                and entry['production_rate'] == pr
                and (glr is None or entry['glr'] == glr)
            ]
            total_iterations += len(filtered_data) if glr is None else 1
    
    if total_iterations == 0:
        valid_range = get_valid_glr_range(conduit_size, production_rate)
        st.error(f"No data found for conduit size(s) {conduit_sizes}, production rate(s) {prates}" +
                 (f", GLR {glr}. Valid ranges: {valid_range}" if glr else f". Valid ranges: {valid_range}"))
        logger.error(f"No data found for conduit_size(s)={conduit_sizes}, production_rate(s)={prates}, glr={glr}")
        return None
    
    progress_bar = st.progress(0)
    current_iteration = 0
    
    for cs in conduit_sizes:
        prates = valid_prates_dict[cs] if all_prates else [production_rate]
        for pr in prates:
            filtered_data = [
                entry for entry in reference_data 
                if entry['conduit_size'] == cs 
                and entry['production_rate'] == pr
                and (glr is None or entry['glr'] == glr)
            ]
            if not filtered_data:
                valid_range = get_valid_glr_range(cs, pr)
                logger.warning(f"No data for conduit_size={cs}, production_rate={pr}, glr={glr}. Valid ranges: {valid_range}")
                current_iteration += 1 if glr is not None else len(get_valid_glr_range(cs, pr))
                continue
            for entry in filtered_data:
                coeffs = [entry['coefficients'][k] for k in sorted(entry['coefficients'].keys())]
                df_temp = generate_df(coeffs, num_points, min_D)
                if df_temp is None or df_temp.empty:
                    logger.warning(f"Failed to generate data for conduit={entry['conduit_size']}, prod={entry['production_rate']}, glr={entry['glr']}")
                    current_iteration += 1
                    continue
                # Drop y1 and y2 as they are not needed for this prediction task
                df_temp = df_temp.drop(columns=['y1', 'y2'])
                df_temp['conduit_size'] = entry['conduit_size']
                df_temp['production_rate'] = entry['production_rate']
                df_temp['GLR'] = entry['glr']
                dfs_ml.append(df_temp)
                logger.info(f"Generated {len(df_temp)} points for conduit={cs}, production_rate={pr}, glr={entry['glr']}")
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)
    
    if not dfs_ml:
        valid_range = get_valid_glr_range(conduit_size, production_rate)
        st.error(f"No valid data generated for conduit size(s) {conduit_sizes}, production rate(s) {prates}" +
                 (f", GLR {glr}. Valid ranges: {valid_range}" if glr else f". Valid ranges: {valid_range}"))
        logger.error(f"No valid data generated for conduit_size(s)={conduit_sizes}, production_rate(s)={prates}, glr={glr}")
        return None
    
    df_ml = pd.concat(dfs_ml, ignore_index=True)
    logger.info(f"Generated ML data with {len(df_ml)} points across {len(dfs_ml)} configurations")
    return df_ml

def train_model(df_ml, model_type):
    """
    Train the selected ML model with a progress bar where applicable.
    Returns model, scaler
    """
    if df_ml.empty:
        return None, None
    features = ['p1', 'D', 'conduit_size', 'production_rate', 'GLR']
    target = 'p2'
    X = df_ml[features]
    y = df_ml[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.write(f"Training {model_type} model...")
    progress = st.progress(0)
    
    if model_type == "Neural Network":
        model = Sequential([
            Input(shape=(X.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        epochs = 50
        for epoch in range(epochs):
            model.fit(X_scaled, y, epochs=1, batch_size=32, validation_split=0.2, verbose=0)
            progress.progress((epoch + 1) / epochs)
            time.sleep(0.05)
    
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        progress.progress(1.0)
    
    elif model_type == "Gradient Boosting":
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        progress.progress(1.0)
    
    elif model_type == "Stacking Ensemble":
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('lgb', lgb.LGBMRegressor(n_estimators=50, random_state=42))
        ]
        model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
        model.fit(X_scaled, y)
        progress.progress(1.0)
    
    return model, scaler

def predict_p2(model, scaler, p1, D, conduit_size, production_rate, glr):
    features = ['p1', 'D', 'conduit_size', 'production_rate', 'GLR']
    input_data = pd.DataFrame([{
        'p1': p1, 'D': D, 'conduit_size': conduit_size,
        'production_rate': production_rate, 'GLR': glr
    }])
    input_scaled = scaler.transform(input_data[features])
    prediction = model.predict(input_scaled)[0]
    if isinstance(prediction, np.ndarray):
        prediction = prediction[0]
    return prediction

def run_ml_predictor():
    """
    UI for Bottomhole Pressure Predictor: Train selected ML model and predict p2.
    """
    st.subheader("Mode 6: Bottomhole Pressure Predictor")
    
    if 'reference_data' not in st.session_state:
        with st.spinner("Loading referenceexcel.xlsx from GitHub..."):
            reference_data = load_reference_data()
            if reference_data is None:
                st.error("Failed to load referenceexcel.xlsx.")
                return
            st.session_state.reference_data = reference_data
    else:
        reference_data = st.session_state.reference_data
    
    st.subheader("Input Parameters for Training Data")
    col1, col2 = st.columns(2)
    
    with col1:
        conduit_size = st.selectbox(
            "Conduit Size (in):",
            [2.875, 3.5],
            help="Select the conduit size (2.875 or 3.5 inches)."
        )
        both_conduits = st.checkbox(
            "Use Both Conduit Sizes",
            value=False,
            help="Check to generate data for both conduit sizes (2.875 and 3.5 inches)."
        )
    
    with col2:
        valid_prates, valid_glrs = get_valid_options(conduit_size)
        valid_prates = [float(pr) for pr in valid_prates]
        production_rate = st.selectbox(
            "Production Rate (stb/day):",
            valid_prates,
            help="Select the production rate (50 to 600 stb/day)."
        )
        all_prates = st.checkbox(
            "Use All Production Rates",
            value=False,
            help="Check to generate data for all valid production rates for the selected conduit size(s)."
        )
    
    num_points = st.number_input(
        "Number of Random Points per GLR Curve:",
        min_value=1,
        value=1000,
        step=100,
        help="Number of random data points to generate per GLR curve."
    )
    
    all_glr = st.checkbox("Use All GLRs for Selected Production Rate", value=True)
    glr = None
    
    if not all_glr:
        valid_glrs = valid_glrs.get(production_rate, [])
        if valid_glrs:
            glr = st.selectbox(
                "GLR (scf/stb):",
                [float(g) for g in valid_glrs],
                help="Select a valid GLR for the chosen conduit size and production rate."
            )
    
    if st.button("Generate Data"):
        errors = []
        if not both_conduits and not validate_conduit_size(conduit_size):
            errors.append("Invalid conduit size.")
        if not all_prates and not validate_production_rate(production_rate):
            errors.append("Invalid production rate.")
        if not validate_positive_integer(num_points, "number of random points"):
            errors.append("Invalid number of random points.")
        if not all_glr and glr is not None and not validate_glr(conduit_size, production_rate, glr):
            valid_range = get_valid_glr_range(conduit_size, production_rate)
            errors.append(f"Invalid GLR {glr}. Valid ranges: {valid_range}")
        
        if errors:
            for error in errors:
                st.error(error)
            logger.error(f"Bottomhole Pressure Predictor errors: {errors}")
            return
        
        with st.spinner("Generating data..."):
            df_ml = load_ml_data(st.session_state.reference_data, conduit_size, production_rate, num_points, glr, all_prates=all_prates, both_conduits=both_conduits)
            if df_ml is None or df_ml.empty:
                valid_range = get_valid_glr_range(conduit_size, production_rate)
                st.error(f"Failed to generate ML data. Valid ranges: {valid_range}")
                return
            st.session_state.df_ml_pred = df_ml
            st.subheader("Generated Data Preview")
            st.dataframe(df_ml.head())
            st.success("Data generation complete!")
    
    if 'df_ml_pred' in st.session_state:
        model_type = st.selectbox(
            "Select ML Model:",
            ["Neural Network", "Random Forest", "Gradient Boosting", "Stacking Ensemble"],
            key="model_type_selectbox"
        )
        if st.button("Train Model"):
            try:
                model, scaler = train_model(st.session_state.df_ml_pred, model_type)
                if model is None:
                    st.error("Training failed.")
                    return
                st.session_state.model_pred = model
                st.session_state.scaler_pred = scaler
                st.success("Training complete!")
            except Exception as e:
                st.error(f"Error in training: {str(e)}")
                logger.error(f"Error in training: {str(e)}")
    
    if 'model_pred' in st.session_state:
        st.subheader("Predict Bottomhole Flowing Pressure (p2)")
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            pred_p1 = st.number_input("Wellhead Pressure (p1, psi):", min_value=0.0, max_value=4000.0, value=1000.0)
            pred_D = st.number_input("Well Length (D, ft):", min_value=0.0, max_value=31000.0, value=1000.0)
        
        with pred_col2:
            pred_conduit = st.selectbox("Conduit Size (in):", [2.875, 3.5])
            pred_prod = st.selectbox("Production Rate (stb/day):", PRODUCTION_RATES)
            pred_glr = st.number_input("GLR (scf/stb):", min_value=0.0, max_value=25000.0, value=200.0)
        
        if st.button("Predict p2"):
            errors = []
            if not validate_conduit_size(pred_conduit):
                errors.append("Invalid conduit size for prediction.")
            if not validate_production_rate(pred_prod):
                errors.append("Invalid production rate for prediction.")
            if not validate_glr(pred_conduit, pred_prod, pred_glr):
                valid_range = get_valid_glr_range(pred_conduit, pred_prod)
                errors.append(f"Invalid GLR {pred_glr} for prediction. Valid ranges: {valid_range}")
            # Add more validations if needed
            
            if errors:
                for error in errors:
                    st.error(error)
                return
            
            try:
                p2_pred = predict_p2(st.session_state.model_pred, st.session_state.scaler_pred, pred_p1, pred_D, pred_conduit, pred_prod, pred_glr)
                st.success(f"Predicted Bottomhole Flowing Pressure (p2): {p2_pred:.2f} psi")
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                logger.error(f"Error in prediction: {str(e)}")
    else:
        st.warning("Please generate data and train the model before making predictions.")
