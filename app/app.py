# app/app.py

import streamlit as st
import pickle
import numpy as np

# Load the model
with open('models/rain_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🌧️ Rainfall Prediction App")

# Input form
location = st.number_input("Location (Encoded number)", min_value=0, max_value=49, step=1)
min_temp = st.number_input("Min Temperature (°C)")
max_temp = st.number_input("Max Temperature (°C)")
rainfall = st.number_input("Rainfall (mm)")
humidity_9am = st.number_input("Humidity at 9AM (%)")
humidity_3pm = st.number_input("Humidity at 3PM (%)")
rain_today = st.selectbox("Did it rain today?", ["No", "Yes"])
rain_today_encoded = 1 if rain_today == "Yes" else 0

# Predict button
if st.button("Predict"):
    features = np.array([[location, min_temp, max_temp, rainfall, humidity_9am, humidity_3pm, rain_today_encoded]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("🌧️ It will rain tomorrow.")
    else:
        st.info("☀️ No rain expected tomorrow.")
