# predict.py
import joblib
import pandas as pd

def predict_total_cost(input_data):
    """
    Predict total paddy farming cost for given input data (dict).
    """
    model = joblib.load("trained_paddy_cost_model.pkl")
    encoders = joblib.load("label_encoders.pkl")

    df_input = pd.DataFrame([input_data])

    # Encode categorical columns using saved encoders
    for col, le in encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col])

    # Predict cost
    predicted_cost = model.predict(df_input)[0]
    return round(predicted_cost, 2)

if __name__ == "__main__":
    # Example GPS-based sample
    sample = {
        "Latitude": 7.35,
        "Longitude": 81.65,
        "Year": 2025,
        "Season": "Maha",
        "Soil_Type": "Clay",
        "Rainfall_mm": 9200,
        "Temperature_C": 10.5,
        "Humidity_%": 25,
        "Area_acres": 23.5,
        "Seed_Type": "BG352",
        "Seed_Cost (LKR)": 8000,
        "Fertilizer_Cost (LKR)": 15000,
        "Pesticide_Cost (LKR)": 4000,
        "Labor_Cost (LKR)": 25000,
        "Water_Cost (LKR)": 5000,
        "Machinery_Cost (LKR)": 10000,
        "Other_Costs (LKR)": 20000
    }

    cost = predict_total_cost(sample)
    print(f"Predicted Total Cost: LKR {cost}")
