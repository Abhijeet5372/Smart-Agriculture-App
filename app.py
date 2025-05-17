import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sklearn # Import sklearn to display version for debugging

# --- Streamlit App Configuration ---
# set_page_config() must be the first Streamlit command
st.set_page_config(page_title="Smart Agriculture Recommendations", layout="wide")

# --- Custom CSS for Background Image ---
# Ensure 'crop2.jpg' is in the same directory as app.py
# Use st.markdown with a style tag to inject CSS

# Background image via custom CSS
def set_background_image():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.yourstory.com/cs/2/f02aced0d86311e98e0865c1f0fe59a2/agritech-1590421634963.png?mode=crop&crop=faces&ar=2%3A1&format=auto&w=1920&q=75");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to set background
set_background_image()

st.title("🌱 Smart Agriculture System")
st.markdown("Get recommendations for crop cultivation and fertilizer usage based on environmental conditions.")

# Optional: Display scikit-learn version for debugging
st.write(f"Scikit-learn version used by app: {sklearn.__version__}")


# --- Load Saved Models and Components ---

# Define file paths (adjust if your files are in a different directory)
CROP_MODEL_PATH = 'best_crop_model.sav'
CROP_SCALER_PATH = 'crop_scaler.sav'
CROP_LABEL_ENCODER_PATH = 'crop_label_encoder.sav'

FERTILIZER_MODEL_PATH = 'best_fertilizer_model.sav'
FERTILIZER_SCALER_PATH = 'fertilizer_scaler.sav'
FERTILIZER_LABEL_ENCODER_PATH = 'fertilizer_label_encoder.sav'
FERTILIZER_COLUMNS_PATH = 'fertilizer_columns.sav'

@st.cache_resource # Cache the loading of resources
def load_crop_components():
    """Loads the saved crop prediction model, scaler, and label encoder."""
    try:
        model = pickle.load(open(CROP_MODEL_PATH, 'rb'))
        scaler = pickle.load(open(CROP_SCALER_PATH, 'rb'))
        label_encoder = pickle.load(open(CROP_LABEL_ENCODER_PATH, 'rb'))
        return model, scaler, label_encoder
    except FileNotFoundError:
        st.error("Error loading crop model components. Make sure the files are in the correct directory.")
        return None, None, None

@st.cache_resource # Cache the loading of resources
def load_fertilizer_components():
    """Loads the saved fertilizer recommendation model, scaler, label encoder, and columns."""
    try:
        model = pickle.load(open(FERTILIZER_MODEL_PATH, 'rb'))
        scaler = pickle.load(open(FERTILIZER_SCALER_PATH, 'rb'))
        label_encoder = pickle.load(open(FERTILIZER_LABEL_ENCODER_PATH, 'rb'))
        columns = pickle.load(open(FERTILIZER_COLUMNS_PATH, 'rb'))
        return model, scaler, label_encoder, columns
    except FileNotFoundError:
        st.error("Error loading fertilizer model components. Make sure the files are in the correct directory.")
        return None, None, None, None

# Load components on app startup
crop_model, crop_scaler, crop_label_encoder = load_crop_components()
fertilizer_model, fertilizer_scaler, fertilizer_label_encoder, fertilizer_columns = load_fertilizer_components()

# --- Prediction Functions ---

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """Predicts the best crop based on environmental parameters."""
    if crop_model is None or crop_scaler is None or crop_label_encoder is None:
        return "Model not loaded."

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    transformed_features = crop_scaler.transform(features)
    prediction_encoded = crop_model.predict(transformed_features)
    predicted_crop = crop_label_encoder.inverse_transform(prediction_encoded)
    return predicted_crop[0]

def predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
    """Recommends the best fertilizer based on environmental and crop parameters."""
    if fertilizer_model is None or fertilizer_scaler is None or fertilizer_label_encoder is None or fertilizer_columns is None:
        return "Model not loaded."

    # Create a DataFrame from the input
    new_data_input = {
        'temperature': temperature,
        'humidity': humidity,
        'moisture': moisture,
        'soil_type': soil_type,
        'crop_type': crop_type,
        'nitrogen': nitrogen,
        'potassium': potassium,
        'phosphorous': phosphorous
    }
    new_data_df = pd.DataFrame([new_data_input])

    # Apply one-hot encoding, aligning with training columns
    new_data_df = pd.get_dummies(new_data_df, columns=['soil_type', 'crop_type'], drop_first=True)

    # Ensure the new data DataFrame has the same columns as the training data
    # Add missing columns with a value of 0
    for col in fertilizer_columns:
        if col not in new_data_df.columns:
            new_data_df[col] = 0

    # Reorder columns to match the training data order
    new_data_df = new_data_df[fertilizer_columns]

    # Scale the numerical features
    numerical_features_for_scaling = ['temperature', 'humidity', 'moisture', 'nitrogen', 'potassium', 'phosphorous']
    new_data_df[numerical_features_for_scaling] = fertilizer_scaler.transform(new_data_df[numerical_features_for_scaling])

    # Make a prediction
    predicted_label_encoded = fertilizer_model.predict(new_data_df)

    # Decode the predicted label
    predicted_fertilizer = fertilizer_label_encoder.inverse_transform(predicted_label_encoded)

    return predicted_fertilizer[0]

# --- Streamlit App Layout ---

# Use tabs for better organization
tab1, tab2 = st.tabs(["🌾 Crop Recommendation", "🧪 Fertilizer Recommendation"])

with tab1:
    st.header("Crop Recommendation")
    st.write("Enter the environmental parameters to get a crop recommendation.")

    # Input fields for crop prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        n_value = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0)
    with col2:
        p_value = st.number_input("Phosphorous (P)", min_value=0.0, value=42.0)
    with col3:
        k_value = st.number_input("Potassium (K)", min_value=0.0, value=43.0)

    col4, col5, col6 = st.columns(3)
    with col4:
        temp_value = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0)
    with col5:
        humidity_value = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    with col6:
        ph_value = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5)

    rainfall_value = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0)

    if st.button("Get Crop Recommendation"):
        if crop_model and crop_scaler and crop_label_encoder:
            crop_recommendation = predict_crop(n_value, p_value, k_value, temp_value, humidity_value, ph_value, rainfall_value)
            st.success(f"Recommended Crop: **{crop_recommendation}**")
        else:
            st.warning("Crop prediction model not loaded. Please ensure model files are in the correct directory and scikit-learn versions match.")

with tab2:
    st.header("Fertilizer Recommendation")
    st.write("Enter the environmental, soil, and crop parameters to get a fertilizer recommendation.")

    # Input fields for fertilizer recommendation
    col1, col2, col3 = st.columns(3)
    with col1:
        fert_temp_value = st.number_input("Temperature (°C)", key='fert_temp', min_value=-50.0, max_value=60.0, value=26.0)
    with col2:
        fert_humidity_value = st.number_input("Humidity (%)", key='fert_humidity', min_value=0.0, max_value=100.0, value=52.0)
    with col3:
        fert_moisture_value = st.number_input("Moisture (%)", key='fert_moisture', min_value=0.0, max_value=100.0, value=38.0)

    col4, col5 = st.columns(2)
    with col4:
        # These lists should ideally come from the unique values in your training data
        soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'] # Add all unique soil types from your data
        fert_soil_type = st.selectbox("Soil Type", soil_types, key='fert_soil_type')
    with col5:
        # These lists should ideally come from the unique values in your training data
        crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts'] # Add all unique crop types from your data
        fert_crop_type = st.selectbox("Crop Type", crop_types, key='fert_crop_type')

    col6, col7, col8 = st.columns(3)
    with col6:
        fert_nitrogen_value = st.number_input("Nitrogen (N)", key='fert_nitrogen', min_value=0.0, value=37.0)
    with col7:
        fert_potassium_value = st.number_input("Potassium (K)", key='fert_potassium', min_value=0.0, value=0.0)
    with col8:
        fert_phosphorous_value = st.number_input("Phosphorous (P)", key='fert_phosphorous', min_value=0.0, value=0.0)


    if st.button("Get Fertilizer Recommendation"):
        if fertilizer_model and fertilizer_scaler and fertilizer_label_encoder and fertilizer_columns:
            fertilizer_recommendation = predict_fertilizer(
                fert_temp_value, fert_humidity_value, fert_moisture_value,
                fert_soil_type, fert_crop_type, fert_nitrogen_value,
                fert_potassium_value, fert_phosphorous_value
            )
            st.success(f"Recommended Fertilizer: **{fertilizer_recommendation}**")
        else:
            st.warning("Fertilizer recommendation model not loaded. Please ensure model files are in the correct directory and scikit-learn versions match.")

st.markdown("---")
st.markdown("Developed as a Smart Agriculture System using Machine Learning.")
st.markdown("© 2025 Abhijeet Sawant - EDUNET Foundation")
st.markdown("For educational purposes only. All rights reserved.")


