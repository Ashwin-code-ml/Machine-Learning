import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ==========================
# Load saved model + encoders + scaler
# ==========================
model = joblib.load('model.pkl')
encoders = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Ford Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Ford Car Price Prediction")
st.write("Provide the car details below and get an estimated selling price.")

# Car models (from dataset)
models = [
    "Fiesta", "Focus", "Kuga", "EcoSport", "C-MAX", "Ka+", "Mondeo", "B-MAX", 
    "S-MAX", "Grand C-MAX", "Galaxy", "Edge", "KA", "Puma", "Tourneo Custom",
    "Grand Tourneo Connect", "Mustang", "Tourneo Connect", "Fusion", 
    "Streetka", "Ranger", "Escort", "Transit Tourneo"
]

# Input Fields
col1, col2 = st.columns(2)
with col1:
    car_model = st.selectbox('Car Model', models)
    transmission = st.selectbox('Transmission Type', ['Manual', 'Automatic', 'Semi-Auto'])
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Hybrid', 'Electric'])
    car_age = st.number_input('Car Age (in years)', min_value=0, max_value=50, value=5)
with col2:
    mileage = st.number_input('Mileage (in miles)', min_value=0, step=1000)
    tax = st.number_input('Tax', min_value=0, step=10)
    mpg = st.number_input('Miles Per Gallon (MPG)', min_value=0.0, format="%.2f")
    engine_size = st.number_input('Engine Size (in litres)', min_value=0.0, format="%.1f")

# ==========================
# Prediction
# ==========================
if st.button('🔮 Predict Price'):
    # Create input DataFrame
    input_df = pd.DataFrame({
        'model': [car_model],
        'transmission': [transmission],
        'mileage': [mileage],
        'fuelType': [fuel_type],
        'tax': [tax],
        'mpg': [mpg],
        'engineSize': [engine_size],
        'car_age': [car_age]
    })

    try:
        # Apply encoders to categorical columns
        for col in ['model', 'transmission', 'fuelType']:
            if col in encoders:
                le = encoders[col]
                input_df[col] = le.transform(input_df[col])

        # Apply scaling to numeric columns
        numeric_cols = ['mileage', 'tax', 'mpg', 'engineSize', 'car_age']
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Predict
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Price: {prediction:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# ==========================
# Footer
# ==========================
st.caption("Built with ❤️ using Streamlit + Scikit-Learn")
