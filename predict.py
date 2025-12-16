import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def safe_transform(encoder: LabelEncoder, value: str):
    if value not in encoder.classes_:
        print(f"Warning: '{value}' not seen during training. Using fallback '{encoder.classes_[0]}'")
        return encoder.transform([encoder.classes_[0]])[0]
    else:
        return encoder.transform([value])[0]


def predict_fertilizer(
    soil_temp,
    soil_moisture,
    air_temp,
    air_humidity,
    paddy_growth_stage,
    purpose
):
    print("Loading trained model.")

    model_dir = "models"
    paths = {
        "model": os.path.join(model_dir, "fertilizer_model.pkl"),
        "scaler": os.path.join(model_dir, "scaler.pkl"),
        "target_encoder": os.path.join(model_dir, "target_encoder.pkl"),
        "feature_encoders": os.path.join(model_dir, "feature_encoders.pkl"),
        "feature_names": os.path.join(model_dir, "feature_names.pkl")
    }

    for f in paths.values():
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing required file: {f}")

    model = joblib.load(paths["model"])
    scaler = joblib.load(paths["scaler"])
    target_encoder = joblib.load(paths["target_encoder"])
    label_encoders = joblib.load(paths["feature_encoders"])
    feature_names = joblib.load(paths["feature_names"])

    print("Model loaded successfully!\n")

    # Prepare input
    input_data = pd.DataFrame([{
        "Soil_Temperature (°C)": soil_temp,
        "Soil_Moisture (%)": soil_moisture,
        "Air_Temperature (°C)": air_temp,
        "Air_Humidity (%)": air_humidity,
        "Paddy_Growth_Stage": paddy_growth_stage,
        "Purpose": purpose,
        "Quantity_kg_per_acre": 100
    }])

    # Encode categorical columns
    for col, encoder in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = input_data[col].apply(lambda v: safe_transform(encoder, v))

    # Align features
    X = input_data[feature_names]
    X_scaled = scaler.transform(X)

    # Prediction
    pred_encoded = model.predict(X_scaled)
    fertilizer = target_encoder.inverse_transform(pred_encoded)[0]

    print("Predicted Fertilizer Recommendation:")
    print(f"Fertilizer Type: {fertilizer}")
    print(f"Input Conditions: {soil_temp}°C soil, {soil_moisture}% moisture, {air_temp}°C air, {air_humidity}% humidity")
    print(f"Paddy Stage: {paddy_growth_stage}")
    print(f"Purpose: {purpose}")

    return fertilizer


# prediction
if __name__ == "__main__":
    fertilizer = predict_fertilizer(
        soil_temp=25.3,
        soil_moisture=10.8,
        air_temp=20.2,
        air_humidity=10.5,

        paddy_growth_stage = "Tillering",
        purpose = "Improves soil structure and moisture retention"

    )
    print(f"\nFinal Recommended Fertilizer: {fertilizer}")
