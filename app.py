import streamlit as st
import numpy as np
import joblib

# Load trained model and encoders
model = joblib.load("crop_yield_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

area_encoder = label_encoders["Area"]
crop_encoder = label_encoders["Crop"]

# Page config
st.set_page_config(
    page_title="AI Crop Yield Forecasting",
    layout="centered"
)

# Title
st.title("🌾 AI-Based Crop Yield Forecasting System")
st.write("Predict crop yield using weather and agricultural inputs")

st.markdown("---")

# -------- USER INPUT SECTION --------
st.subheader("📥 Enter Farm Details")

area = st.selectbox(
    "Select Region / Area",
    area_encoder.classes_
)

crop = st.selectbox(
    "Select Crop",
    crop_encoder.classes_
)

year = st.number_input(
    "Year",
    min_value=2000,
    max_value=2030,
    value=2024
)

rainfall = st.number_input(
    "Average Rainfall (mm/year)",
    min_value=0.0,
    max_value=5000.0,
    value=800.0
)

temperature = st.number_input(
    "Average Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    value=25.0
)

pesticide = st.number_input(
    "Pesticide Usage (kg/ha)",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

# -------- PREDICTION --------
if st.button("🔍 Predict Crop Yield"):
    # Encode categorical inputs
    area_encoded = area_encoder.transform([area])[0]
    crop_encoded = crop_encoder.transform([crop])[0]

    # Arrange input in same order as training
    input_data = np.array([[
        area_encoded,
        crop_encoded,
        year,
        rainfall,
        temperature,
        pesticide
    ]])

    prediction = model.predict(input_data)[0]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.success(f"🌱 Predicted Crop Yield: **{prediction:.2f} hg/ha**")

    # -------- RECOMMENDATION SYSTEM --------
    st.subheader("💡 Recommendations")

    if prediction < 20000:
        st.warning(
            "• Improve irrigation practices\n"
            "• Enhance soil nutrient management\n"
            "• Monitor weather conditions closely"
        )
    elif prediction < 40000:
        st.info(
            "• Yield is moderate\n"
            "• Optimize fertilizer and pesticide usage\n"
            "• Regular crop monitoring is recommended"
        )
    else:
        st.success(
            "• Excellent yield conditions\n"
            "• Maintain current farming practices\n"
            "• Ensure sustainable resource usage"
        )
