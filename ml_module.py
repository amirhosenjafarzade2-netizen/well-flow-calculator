# ml_module.py
# Module for Mode 5: Machine Learning in Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools
import random
import requests
import io
import re
import time
from config import INTERPOLATION_RANGES, PRODUCTION_RATES, GITHUB_URL
from utils import setup_logging
from random_point_generator import generate_df, calc_y1, solve_p2
from validators import validate_conduit_size, validate_production_rate, get_valid_options

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

def load_ml_data(reference_data, conduit_size, production_rate, num_points, glr=None, min_D=1000):
    """
    Generate ML data using random_point_generator's generate_df function.
    If glr is None, use all valid GLRs for the conduit size and production rate.
    """
    dfs_ml = []
    required_cols = ["p1", "D", "y1", "y2", "p2"]
    
    filtered_data = [
        entry for entry in reference_data 
        if entry['conduit_size'] == conduit_size 
        and entry['production_rate'] == production_rate
        and (glr is None or entry['glr'] == glr)
    ]
    
    if not filtered_data:
        st.error(f"No data found for conduit size {conduit_size}, production rate {production_rate}" +
                 (f", GLR {glr}" if glr else ""))
        logger.error(f"No data found for conduit_size={conduit_size}, production_rate={production_rate}, glr={glr}")
        return None
    
    progress = st.progress(0)
    for i, entry in enumerate(filtered_data):
        coeffs = [entry['coefficients'][k] for k in sorted(entry['coefficients'].keys())]
        df_temp = generate_df(coeffs, num_points, min_D)
        if df_temp is None or df_temp.empty:
            logger.warning(f"Failed to generate data for conduit={entry['conduit_size']}, prod={entry['production_rate']}, glr={entry['glr']}")
            continue
        df_temp['conduit_size'] = entry['conduit_size']
        df_temp['production_rate'] = entry['production_rate']
        df_temp['GLR'] = entry['glr']
        dfs_ml.append(df_temp)
        progress.progress((i + 1) / len(filtered_data))
    
    return pd.concat(dfs_ml, ignore_index=True) if dfs_ml else None

def train_neural_network(df_ml):
    """
    Train the neural network with a progress bar for epochs.
    """
    if df_ml.empty:
        return None, None
    features = ['p1', 'D', 'y1', 'y2', 'conduit_size', 'production_rate', 'GLR']
    target = 'p2'
    X = df_ml[features]
    y = df_ml[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Progress bar for training
    st.write("Training neural network...")
    progress = st.progress(0)
    epochs = 50
    for epoch in range(epochs):
        model.fit(X_scaled, y, epochs=1, batch_size=32, validation_split=0.2, verbose=0)
        progress.progress((epoch + 1) / epochs)
        time.sleep(0.05)  # Smooth progress bar animation
    return model, scaler

def analyze_parameter_effects(model, scaler, df_ml):
    """
    Generate parameter effect plots matching the main Colab program.
    """
    for conduit_size in [2.875, 3.5]:
        for production_rate in PRODUCTION_RATES:
            glr_min, glr_max = get_valid_glr_range(conduit_size, production_rate)
            glr_values = np.linspace(glr_min, glr_max, 100)
            base_values = df_ml.mean().to_dict()
            base_values['conduit_size'] = conduit_size
            base_values['production_rate'] = production_rate
            X_test_glr = pd.DataFrame([base_values] * 100)
            X_test_glr['GLR'] = glr_values
            X_test_glr_scaled = scaler.transform(X_test_glr)
            glr_predictions = model.predict(X_test_glr_scaled, verbose=0).flatten()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(glr_values, glr_predictions, label=f'Conduit: {conduit_size} in, Prod: {production_rate} stb/day')
            ax.set_xlabel('GLR (SCF/STB)')
            ax.set_ylabel('Pressure Gradient (psi)')
            ax.set_title('Pressure Gradient vs. GLR')
            ax.grid(True)
            ax.legend()
            st.subheader(f"Effect of GLR (Conduit: {conduit_size} in, Production: {production_rate} stb/day)")
            st.pyplot(fig)

    # Overall Pressure vs. GLR
    glr_values = np.linspace(0, 25000, 100)
    base_values = df_ml.mean().to_dict()
    X_test_glr = pd.DataFrame([base_values] * 100)
    X_test_glr['GLR'] = glr_values
    X_test_glr_scaled = scaler.transform(X_test_glr)
    glr_predictions = model.predict(X_test_glr_scaled, verbose=0).flatten()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(glr_values, glr_predictions, label='Overall Pressure vs. GLR')
    ax.set_xlabel('GLR (SCF/STB)')
    ax.set_ylabel('Pressure Gradient (psi)')
    ax.set_title('Overall Pressure Gradient vs. GLR')
    ax.grid(True)
    ax.legend()
    st.subheader("Overall Pressure vs. GLR")
    st.pyplot(fig)

    # Effect of D, production_rate, conduit_size
    for param in ['D', 'production_rate', 'conduit_size']:
        if param == 'conduit_size':
            param_values = [2.875, 3.5]
        elif param == 'production_rate':
            param_values = PRODUCTION_RATES
        else:
            param_values = np.linspace(df_ml[param].min(), df_ml[param].max(), 100)
        base_values = df_ml.mean().to_dict()
        X_test_param = pd.DataFrame([base_values] * len(param_values))
        X_test_param[param] = param_values
        X_test_param_scaled = scaler.transform(X_test_param)
        param_predictions = model.predict(X_test_param_scaled, verbose=0).flatten()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(param_values, param_predictions, label=f'Pressure Gradient vs. {param}')
        ax.set_xlabel(param.capitalize())
        ax.set_ylabel('Pressure Gradient (psi)')
        ax.set_title(f'Pressure Gradient vs. {param.capitalize()}')
        ax.grid(True)
        ax.legend()
        st.subheader(f"Effect of {param}")
        st.pyplot(fig)

def get_valid_glr_range(conduit_size, production_rate):
    ranges = INTERPOLATION_RANGES.get((conduit_size, production_rate), [])
    return min(r[0] for r in ranges) if ranges else 0, max(r[1] for r in ranges) if ranges else 25000

def evaluate_individual(individual, model, scaler):
    conduit_size, production_rate, glr = individual
    input_data = pd.DataFrame([{
        'p1': 1000, 'D': 1000, 'y1': 5000, 'y2': 6000,
        'conduit_size': conduit_size, 'production_rate': production_rate, 'GLR': glr
    }])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled, verbose=0)[0][0]
    return (prediction,)

def optimize_neural_network_conditions(model, scaler, df_ml):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_conduit", random.choice, [2.875, 3.5])
    toolbox.register("attr_prod", random.choice, PRODUCTION_RATES)
    toolbox.register("attr_glr", random.uniform, 0, 25000)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_conduit, toolbox.attr_prod, toolbox.attr_glr), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual, model=model, scaler=scaler)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=50)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    best_fitness_history = []
    progress = st.progress(0)
    for gen in range(20):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        best_ind = tools.selBest(pop, 1)[0]
        best_fitness_history.append(best_ind.fitness.values[0])
        progress.progress((gen + 1) / 20)
    
    st.write(f"Optimal Conditions: Conduit {best_ind[0]} in, Production {best_ind[1]} stb/day, GLR {best_ind[2]:.2f} SCF/STB")
    st.write(f"Predicted Minimal Pressure Gradient: {best_fitness_history[-1]:.2f} psi")
    
    # Plot fitness evolution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 21), best_fitness_history, marker='o')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Pressure Gradient (psi)')
    ax.set_title('Optimization Progress: Best Fitness per Generation')
    ax.grid(True)
    st.subheader("Optimization Progress: Best Fitness per Generation")
    st.pyplot(fig)
    
    # Optimization graphs
    base_values = df_ml.mean().to_dict()
    base_values['conduit_size'] = best_ind[0]
    base_values['production_rate'] = best_ind[1]
    base_values['GLR'] = best_ind[2]
    
    # Pressure Gradient vs. Production Rate
    production_rates = PRODUCTION_RATES
    X_test_prod = pd.DataFrame([base_values] * len(production_rates))
    X_test_prod['production_rate'] = production_rates
    X_test_prod_scaled = scaler.transform(X_test_prod)
    prod_predictions = model.predict(X_test_prod_scaled, verbose=0).flatten()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(production_rates, prod_predictions, marker='o', label='Pressure Gradient vs. Production Rate')
    ax.set_xlabel('Production Rate (stb/day)')
    ax.set_ylabel('Pressure Gradient (psi)')
    ax.set_title('Pressure Gradient vs. Production Rate')
    ax.grid(True)
    ax.legend()
    st.subheader("Pressure Gradient vs. Production Rate (Optimization)")
    st.pyplot(fig)
    
    # Pressure Gradient vs. GLR
    glr_min, glr_max = get_valid_glr_range(best_ind[0], best_ind[1])
    glr_values = np.linspace(glr_min, glr_max, 100)
    X_test_glr = pd.DataFrame([base_values] * 100)
    X_test_glr['GLR'] = glr_values
    X_test_glr_scaled = scaler.transform(X_test_glr)
    glr_predictions = model.predict(X_test_glr_scaled, verbose=0).flatten()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(glr_values, glr_predictions, label='Pressure Gradient vs. GLR')
    ax.set_xlabel('GLR (SCF/STB)')
    ax.set_ylabel('Pressure Gradient (psi)')
    ax.set_title('Pressure Gradient vs. GLR')
    ax.grid(True)
    ax.legend()
    st.subheader("Pressure Gradient vs. GLR (Optimization)")
    st.pyplot(fig)
    
    # Pressure Gradient vs. Depth
    depth_values = np.linspace(df_ml['D'].min(), df_ml['D'].max(), 100)
    X_test_depth = pd.DataFrame([base_values] * 100)
    X_test_depth['D'] = depth_values
    X_test_depth_scaled = scaler.transform(X_test_depth)
    depth_predictions = model.predict(X_test_depth_scaled, verbose=0).flatten()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depth_values, depth_predictions, label='Pressure Gradient vs. Depth')
    ax.set_xlabel('Depth Offset (ft)')
    ax.set_ylabel('Pressure Gradient (psi)')
    ax.set_title('Pressure Gradient vs. Depth')
    ax.grid(True)
    ax.legend()
    st.subheader("Pressure Gradient vs. Depth (Optimization)")
    st.pyplot(fig)

def run_machine_learning():
    st.subheader("Mode 5: Machine Learning Analysis")
    
    # Load reference data from GitHub
    with st.spinner("Loading referenceexcel.xlsx from GitHub..."):
        reference_data = load_reference_data()
        if reference_data is None:
            st.error("Failed to load referenceexcel.xlsx.")
            return
    
    # Input form
    st.subheader("Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        conduit_size = st.selectbox(
            "Conduit Size (in):",
            [2.875, 3.5],
            help="Select the conduit size (2.875 or 3.5 inches)."
        )
        if not validate_conduit_size(conduit_size):
            return
    
    with col2:
        valid_prates, valid_glrs = get_valid_options(conduit_size)
        valid_prates = [float(pr) for pr in valid_prates]
        production_rate = st.selectbox(
            "Production Rate (stb/day):",
            valid_prates,
            help="Select the production rate (50 to 600 stb/day)."
        )
        if not validate_production_rate(production_rate):
            return
    
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
    
    if st.button("Generate Data and Train"):
        with st.spinner("Generating data..."):
            df_ml = load_ml_data(reference_data, conduit_size, production_rate, num_points, glr)
            if df_ml is None or df_ml.empty:
                st.error("Failed to generate ML data.")
                return
            st.subheader("Generated Data Preview")
            st.dataframe(df_ml.head())
        
        model, scaler = train_neural_network(df_ml)
        if model is None:
            st.error("Training failed.")
            return
        
        st.success("Training complete!")
        
        option = st.selectbox("Choose: 1. Parameter Analysis or 2. Optimize Conditions", ["1", "2"])
        if option == "1":
            analyze_parameter_effects(model, scaler, df_ml)
        elif option == "2":
            optimize_neural_network_conditions(model, scaler, df_ml)
