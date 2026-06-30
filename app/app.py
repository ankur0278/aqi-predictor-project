import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")

# Load model
model = joblib.load('../models/aqi_random_forest_model.pkl')

# City mapping (from Step 3, Cell 14 — paste your actual city_mapping dict here)
city_mapping = {
    'Ahmedabad': 0, 'Aizawl': 1, 'Amaravati': 2, 'Amritsar': 3, 'Bengaluru': 4,
    'Bhopal': 5, 'Brajrajnagar': 6, 'Chandigarh': 7, 'Chennai': 8, 'Coimbatore': 9,
    'Delhi': 10, 'Ernakulam': 11, 'Gurugram': 12, 'Guwahati': 13, 'Hyderabad': 14,
    'Jaipur': 15, 'Jorapokhar': 16, 'Kochi': 17, 'Kolkata': 18, 'Lucknow': 19,
    'Mumbai': 20, 'Patna': 21, 'Shillong': 22, 'Talcher': 23, 'Thiruvananthapuram': 24,
    'Visakhapatnam': 25
    # ⚠️ Replace with YOUR actual mapping from notebook Cell 14 output
}

season_map = {'Winter':0, 'Spring':1, 'Summer':2, 'Monsoon':3}

# AQI category function
def aqi_category(aqi):
    if aqi <= 50: return "Good", "🟢"
    elif aqi <= 100: return "Satisfactory", "🟡"
    elif aqi <= 200: return "Moderate", "🟠"
    elif aqi <= 300: return "Poor", "🔴"
    elif aqi <= 400: return "Very Poor", "🟣"
    else: return "Severe", "⚫"

# ---- HEADER ----
st.title("🌫️ AQI Predictor")
st.markdown("Predict Air Quality Index based on pollutant levels — powered by Random Forest (R² = 0.91)")
st.divider()

# ---- SIDEBAR INPUTS ----
st.sidebar.header("📥 Input Parameters")

city = st.sidebar.selectbox("City", list(city_mapping.keys()))
season = st.sidebar.selectbox("Season", list(season_map.keys()))
year = st.sidebar.number_input("Year", min_value=2015, max_value=2030, value=2024)
month = st.sidebar.slider("Month", 1, 12, 6)
day = st.sidebar.slider("Day", 1, 31, 15)

st.sidebar.subheader("🧪 Pollutant Levels")
pm25 = st.sidebar.slider("PM2.5", 0.0, 500.0, 50.0)
pm10 = st.sidebar.slider("PM10", 0.0, 600.0, 80.0)
no = st.sidebar.slider("NO", 0.0, 100.0, 10.0)
no2 = st.sidebar.slider("NO2", 0.0, 150.0, 25.0)
nox = st.sidebar.slider("NOx", 0.0, 150.0, 30.0)
nh3 = st.sidebar.slider("NH3", 0.0, 100.0, 15.0)
co = st.sidebar.slider("CO", 0.0, 20.0, 1.0)
so2 = st.sidebar.slider("SO2", 0.0, 100.0, 10.0)
o3 = st.sidebar.slider("O3", 0.0, 200.0, 30.0)
benzene = st.sidebar.slider("Benzene", 0.0, 50.0, 2.0)
toluene = st.sidebar.slider("Toluene", 0.0, 100.0, 5.0)

# ---- PREDICTION ----
if st.sidebar.button("🔍 Predict AQI", use_container_width=True):
    input_data = pd.DataFrame([{
        'PM2.5': pm25, 'PM10': pm10, 'NO': no, 'NO2': no2, 'NOx': nox,
        'NH3': nh3, 'CO': co, 'SO2': so2, 'O3': o3,
        'Benzene': benzene, 'Toluene': toluene,
        'Year': year, 'Month': month, 'Day': day,
        'City_Encoded': city_mapping[city],
        'Season_Encoded': season_map[season]
    }])

    prediction = model.predict(input_data)[0]
    category, emoji = aqi_category(prediction)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted AQI", f"{prediction:.0f}")
    col2.metric("Category", f"{emoji} {category}")
    col3.metric("City", city)

    st.divider()

    # Health advisory
    advisories = {
        "Good": "Air quality is excellent. Enjoy outdoor activities!",
        "Satisfactory": "Air quality is acceptable. Sensitive groups should be cautious.",
        "Moderate": "May cause breathing discomfort to sensitive people.",
        "Poor": "Breathing discomfort to most people on prolonged exposure.",
        "Very Poor": "Respiratory illness on prolonged exposure. Avoid outdoor activity.",
        "Severe": "Serious health impact. Stay indoors, use air purifiers."
    }
    st.info(f"**Health Advisory:** {advisories[category]}")

else:
    st.info("👈 Adjust pollutant levels in the sidebar and click 'Predict AQI'")

# ---- FOOTER ----
st.divider()
st.caption("AQI Predictor | Master's Project | Random Forest Model (R² = 0.91)")