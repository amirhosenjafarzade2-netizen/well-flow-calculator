import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Hardcoded configuration
PRODUCTION_RATES = [50, 100, 200, 400, 600]
CONDUIT_SIZES = [2.875, 3.5]
GITHUB_URL = "https://raw.githubusercontent.com/amirhosenjafarzade2-netizen/well-flow-calculator/main/referenceexcel.xlsx"
VALID_GLRS = {
    2.875: {50: [200, 400, 600], 100: [200, 400], 200: [200], 400: [200], 600: [200]},
    3.5: {50: [200, 400, 600, 800], 100: [200, 400, 600], 200: [200, 400], 400: [200], 600: [200]}
}

def load_reference_data():
    """
    Load reference Excel file from GitHub and parse into a list of dictionaries.
    Returns None if loading or parsing fails.
    """
    try:
        response = requests.get(GITHUB_URL)
        response.raise_for_status()
        file_like_object = io.BytesIO(response.content)
        df_ref = pd.read_excel(file_like_object, header=None, engine='openpyxl')
        if df_ref.shape[1] < 6:
            st.error("Invalid Excel file: Must have at least 6 columns.")
            return None
        data_ref = []
        for index, row in df_ref.iterrows():
            name = row[0]
            if pd.isna(name) or isinstance(name, (int, float)):
                continue
            name = str(name).strip().lower()
            parts = name.split()
            if len(parts) < 5 or 'in' not in name or 'stb-day' not in name or 'glr' not in name:
                continue
            try:
                conduit_size = float(parts[0])
                production_rate = float(parts[2])
                glr = float(parts[4].replace('glr', ''))
                if conduit_size not in CONDUIT_SIZES or production_rate not in PRODUCTION_RATES:
                    continue
                if glr not in VALID_GLRS.get(conduit_size, {}).get(production_rate, []):
                    continue
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
            except (ValueError, TypeError):
                continue
        if not data_ref:
            st.error("No valid data parsed from reference Excel file.")
            return None
        return data_ref
    except Exception as e:
        st.error(f"Error loading reference Excel file: {str(e)}")
        return None

def generate_training_data(reference_data, conduit_size, production_rate, num_points, glr=None, min_D=1000):
    """
    Generate synthetic training data for ML model.
    """
    try:
        filtered_data = [
            entry for entry in reference_data
            if entry['conduit_size'] == conduit_size
            and entry['production_rate'] == production_rate
            and (glr is None or entry['glr'] == glr)
        ]
        if not filtered_data:
            st.error("No data found for the selected parameters.")
            return None
        dfs = []
        for entry in filtered_data:
            coeffs = entry['coefficients']
            p1 = np.random.uniform(100, 4000, num_points)
            D = np.random.uniform(min_D, 31000, num_points)
            p2 = (
                coeffs['a'] * p1 +
                coeffs['b'] * D +
                coeffs['c'] * p1 * D +
                coeffs['d'] +
                coeffs['e'] * np.random.normal(0, 0.1, num_points) +
                coeffs['f']
            )
            df_temp = pd.DataFrame({
                'p1': p1,
                'D': D,
                'p2': p2,
                'conduit_size': entry['conduit_size'],
                'production_rate': entry['production_rate'],
                'GLR': entry['glr']
            })
            dfs.append(df_temp)
        if not dfs:
            st.error("No valid data generated.")
            return None
        return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"Error generating training data: {str(e)}")
        return None

def train_model(df_ml):
    """
    Train a Random Forest model on the generated data.
    """
    if df_ml is None or df_ml.empty:
        st.error("No data available for training.")
        return None, None
    features = ['p1', 'D', 'conduit_size', 'production_rate', 'GLR']
    target = 'p2'
    if not all(col in df_ml.columns for col in features + [target]):
        st.error(f"DataFrame missing required columns: {df_ml.columns}")
        return None, None
    X = df_ml[features]
    y = df_ml[target]
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        return model, scaler
    except Exception as e:
        st.error(f"Training error: {str(e)}")
        return None, None

def predict_p2(model, scaler, p1, D, conduit_size, production_rate, glr):
    """
    Predict p2 using the trained model.
    """
    features = ['p1', 'D', 'conduit_size', 'production_rate', 'GLR']
    input_data = pd.DataFrame([{
        'p1': p1, 'D': D, 'conduit_size': conduit_size,
        'production_rate': production_rate, 'GLR': glr
    }])
    try:
        input_scaled = scaler.transform(input_data[features])
        prediction = model.predict(input_scaled)[0]
        return prediction if isinstance(prediction, (int, float)) else prediction.item()
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def run_ml_predictor():
    """
    Streamlit UI for Bottomhole Pressure Predictor.
    """
    st.subheader("Bottomhole Pressure Predictor")
    
    # Load reference data
    if 'reference_data' not in st.session_state:
        with st.spinner("Loading reference data..."):
            reference_data = load_reference_data()
            if reference_data is None:
                return
            st.session_state.reference_data = reference_data
    
    # Training data inputs
    st.subheader("Input Parameters for Training Data")
    col1, col2 = st.columns(2)
    with col1:
        conduit_size = st.selectbox("Conduit Size (in):", CONDUIT_SIZES, key="ml_pred_conduit")
    with col2:
        production_rate = st.selectbox("Production Rate (stb/day):", PRODUCTION_RATES, key="ml_pred_prod_rate")
    
    num_points = st.number_input("Number of Random Points:", min_value=1, value=1000, step=100, key="ml_pred_num_points")
    all_glr = st.checkbox("Use All GLRs", value=True, key="ml_pred_all_glr")
    glr = None
    if not all_glr:
        valid_glrs = VALID_GLRS.get(conduit_size, {}).get(production_rate, [])
        if not valid_glrs:
            st.error(f"No valid GLRs for conduit size {conduit_size} and production rate {production_rate}.")
            return
        glr = st.selectbox("GLR (scf/stb):", valid_glrs, key="ml_pred_glr")
    
    if st.button("Generate Data", key="ml_pred_generate"):
        if conduit_size not in CONDUIT_SIZES:
            st.error("Invalid conduit size.")
            return
        if production_rate not in PRODUCTION_RATES:
            st.error("Invalid production rate.")
            return
        if num_points <= 0:
            st.error("Number of points must be positive.")
            return
        if not all_glr and glr not in VALID_GLRS.get(conduit_size, {}).get(production_rate, []):
            st.error(f"Invalid GLR {glr} for selected conduit size and production rate.")
            return
        
        with st.spinner("Generating data..."):
            df_ml = generate_training_data(st.session_state.reference_data, conduit_size, production_rate, num_points, glr)
            if df_ml is None:
                return
            st.session_state.df_ml_pred = df_ml
            st.subheader("Generated Data Preview")
            st.dataframe(df_ml.head())
            st.success("Data generation complete!")
    
    # Model training
    if 'df_ml_pred' in st.session_state:
        if st.button("Train Model", key="ml_pred_train"):
            with st.spinner("Training model..."):
                model, scaler = train_model(st.session_state.df_ml_pred)
                if model is None:
                    return
                st.session_state.model_pred = model
                st.session_state.scaler_pred = scaler
                st.success("Training complete!")
    
    # Prediction UI
    if 'model_pred' in st.session_state:
        st.subheader("Predict Bottomhole Flowing Pressure (p2)")
        pred_col1, pred_col2 = st.columns(2)
        with pred_col1:
            pred_p1 = st.number_input("Wellhead Pressure (p1, psi):", min_value=0.0, max_value=4000.0, value=1000.0, key="ml_pred_p1")
            pred_D = st.number_input("Well Length (D, ft):", min_value=0.0, max_value=31000.0, value=1000.0, key="ml_pred_D")
        with pred_col2:
            pred_conduit = st.selectbox("Conduit Size (in):", CONDUIT_SIZES, key="ml_pred_conduit_pred")
            pred_prod = st.selectbox("Production Rate (stb/day):", PRODUCTION_RATES, key="ml_pred_prod_pred")
            pred_glr = st.number_input("GLR (scf/stb):", min_value=0.0, max_value=25000.0, value=200.0, key="ml_pred_glr_pred")
        
        if st.button("Predict p2", key="ml_pred_predict"):
            if pred_conduit not in CONDUIT_SIZES:
                st.error("Invalid conduit size for prediction.")
                return
            if pred_prod not in PRODUCTION_RATES:
                st.error("Invalid production rate for prediction.")
                return
            if pred_glr not in VALID_GLRS.get(pred_conduit, {}).get(pred_prod, []):
                st.error(f"Invalid GLR {pred_glr} for selected conduit size and production rate.")
                return
            p2_pred = predict_p2(st.session_state.model_pred, st.session_state.scaler_pred,
                                 pred_p1, pred_D, pred_conduit, pred_prod, pred_glr)
            if p2_pred is not None:
                st.success(f"Predicted Bottomhole Flowing Pressure (p2): {p2_pred:.2f} psi")
    else:
        st.warning("Please generate data and train the model before making predictions.")
