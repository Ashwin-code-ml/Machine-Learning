import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("xgb_model.pkl")

# Load preprocessors only if you used them
scaler = joblib.load("scaler.pkl") if st.session_state.get("use_scaler", True) else None

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")
st.write("Predict house price using XGBoost Regression model")

# Input UI
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0)
sqft_living = st.number_input("Sqft Living Area", value=1500)
floors = st.number_input("Floors", value=1.0)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.slider("View Rating", 0, 4, 0)
condition = st.slider("Condition", 1, 5, 3)
grade = st.slider("Grade", 1, 13, 7)
sqft_above = st.number_input("Sqft Above", value=1200)
sqft_basement = st.number_input("Sqft Basement", value=300)
yr_built = st.number_input("Year Built", value=2000)
yr_renovated = st.number_input("Year Renovated (0 if not)", value=0)
city = st.selectbox("City", ["default_city"])  # Replace with actual cities

year = st.number_input("Year Sold", value=2015)
month = st.number_input("Month Sold", min_value=1, max_value=12, value=6)
day = st.number_input("Day Sold", min_value=1, max_value=31, value=15)

# Use same column order as training
df = pd.DataFrame([[
    bedrooms, bathrooms, sqft_living, floors, waterfront,
    view, condition, grade, sqft_above, sqft_basement,
    yr_built, yr_renovated, year, month, day
]], columns=[
    "bedrooms", "bathrooms", "sqft_living", "floors", "waterfront",
    "view", "condition", "grade", "sqft_above", "sqft_basement",
    "yr_built", "yr_renovated", "year", "month", "day"
])

# Apply scaler if used
if scaler:
    df[:] = scaler.transform(df)

# Predict
if st.button("Predict Price"):
    pred_log = model.predict(df)[0]
    price = np.exp(pred_log)
    st.success(f"Estimated House Price: ₹ {price:,.2f}")
