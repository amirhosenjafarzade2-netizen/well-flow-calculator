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
from config import INTERPOLATION_RANGES, PRODUCTION_RATES, GITHUB_URL
from utils import setup_logging
from random_point_generator import parse_cell, calc_y1, solve_p2  # Reuse generation functions
import requests
import io

logger = setup_logging()

def load_ml_data(df_coeffs, row1, col1, row2, col2, row3, col3, choice, polynomials, file_names):
    dfs_ml = []
    required_cols = ["p1", "D", "y1", "y2", "p2"]
    for poly_idx, (coeffs, file_name) in enumerate(zip(polynomials, file_names)):
        match = pd.Series(file_name.lower()).str.extract(r'([\d.]+)\s*in\s*(\d+)\s*stb-day\s*(\d+)\s*glr', expand=False).iloc[0]
        if match.isna().any():
            continue
        conduit_size = float(match[0])
        production_rate = float(match[1])
        glr = float(match[2])
        # Generate points internally (like mode 4, but in memory)
        df_temp = pd.DataFrame(columns=required_cols)
        while len(df_temp) < 10000:  # Generate excess to sample from
            p1 = np.random.uniform(0, 4000)
            y1 = calc_y1(p1, coeffs)
            if y1 is None or not (0 <= y1 <= 31000):
                continue
            D = np.random.uniform(0, 31000 - y1)  # min_D=0 for ML
            y2 = y1 + D
            p2 = solve_p2(y2, p1, coeffs)
            if p2 is not None and 0 <= p2 <= 4000:
                df_temp = pd.concat([df_temp, pd.DataFrame({'p1': [p1], 'D': [D], 'y1': [y1], 'y2': [y2], 'p2': [p2]})], ignore_index=True)
        df_temp = df_temp[required_cols].dropna()
        if choice != 'all':
            n_rows = int(choice)
            if 0 < n_rows < len(df_temp):
                df_temp = df_temp.sample(n=n_rows)
        df_temp['conduit_size'] = conduit_size
        df_temp['production_rate'] = production_rate
        df_temp['GLR'] = glr
        dfs_ml.append(df_temp)
    return pd.concat(dfs_ml, ignore_index=True) if dfs_ml else None

def train_neural_network(df_ml):
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
    model.fit(X_scaled, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    return model, scaler

def analyze_parameter_effects(model, scaler, df_ml):
    # Adapt plots from snippet, display with st.pyplot
    # Example for one plot (repeat for others)
    base_values = df_ml.mean().to_dict()
    production_rates = PRODUCTION_RATES
    X_test_prod = pd.DataFrame([base_values] * len(production_rates))
    X_test_prod['production_rate'] = production_rates
    X_test_prod_scaled = scaler.transform(X_test_prod)
    prod_predictions = model.predict(X_test_prod_scaled, verbose=0).flatten()
    
    fig, ax = plt.subplots()
    ax.plot(production_rates, prod_predictions, marker='o', label='Pressure Gradient vs. Production Rate')
    ax.set_xlabel('Production Rate (stb/day)')
    ax.set_ylabel('Pressure Gradient (psi)')
    ax.set_title('Pressure Gradient vs. Production Rate')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    # Add similar for GLR, Depth, etc.

def get_valid_glr_range(conduit_size, production_rate):
    ranges = INTERPOLATION_RANGES.get((conduit_size, production_rate), [])
    return min(r[0] for r in ranges), max(r[1] for r in ranges)

def evaluate_individual(individual, model, scaler):
    conduit_size, production_rate, glr = individual
    input_data = pd.DataFrame([{
        'p1': 1000, 'D': 1000, 'y1': 5000, 'y2': 6000,  # Defaults
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
    
    for gen in range(40):
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
    st.write(f"Optimal: Conduit {best_ind[0]}, Prod {best_ind[1]}, GLR {best_ind[2]:.2f}")
    # Add plots similar to analysis

def run_machine_learning():
    st.subheader("Mode 5: Machine Learning Analysis")
    
    # Load the coefficient Excel from GitHub (no uploader)
    with st.spinner("Loading coefficient Excel from GitHub..."):
        try:
            response = requests.get(GITHUB_URL)
            response.raise_for_status()
            file_like_object = io.BytesIO(response.content)
            df_coeffs = pd.read_excel(file_like_object, header=None, engine='openpyxl')
            st.success("Coefficient data loaded successfully from GitHub.")
        except Exception as e:
            st.error(f"Error loading coefficient Excel from GitHub: {str(e)}")
            logger.error(f"Error loading reference Excel: {str(e)}")
            return

    # Cell inputs same as before
    cell1 = st.text_input("First cell (e.g., A1):")
    cell2 = st.text_input("Second cell (e.g., G1):")
    cell3 = st.text_input("Third cell (e.g., A2):")
    if not (cell1 and cell2 and cell3):
        return

    try:
        row1, col1 = parse_cell(cell1)
        row2, col2 = parse_cell(cell2)
        row3, col3 = parse_cell(cell3)
        polynomials = []
        file_names = []
        for row in range(row1, row3 + 1):
            name = df_coeffs.iloc[row, col1]
            coeffs = df_coeffs.iloc[row, col1 + 1:col2 + 1].tolist()
            polynomials.append(coeffs)
            file_names.append(str(name))
    except Exception as e:
        st.error(f"Error processing inputs: {str(e)}")
        return

    choice = st.number_input("Number of random rows per polynomial (or 0 for all):", min_value=0, value=1000)
    if choice == 0:
        choice = 'all'

    if st.button("Load Data and Train"):
        with st.spinner("Generating data and training..."):
            df_ml = load_ml_data(df_coeffs, row1, col1, row2, col2, row3, col3, choice, polynomials, file_names)
            if df_ml is None or df_ml.empty:
                st.error("Failed to generate ML data.")
                return
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
