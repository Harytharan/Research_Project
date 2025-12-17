import serial
import time
import traceback
import sys
import joblib
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

#  LABEL TRANSFORM
def safe_transform(encoder: LabelEncoder, value: str):
    if value not in encoder.classes_:
        print(f" Warning: '{value}' not seen during training. Using fallback '{encoder.classes_[0]}'")
        return encoder.transform([encoder.classes_[0]])[0]
    return encoder.transform([value])[0]

#  FERTILIZER PREDICTION
def predict_fertilizer(
    soil_temp,
    soil_moisture,
    air_temp,
    air_humidity,
    paddy_growth_stage,
    purpose
):
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

    input_data = pd.DataFrame([{
        "Soil_Temperature (°C)": soil_temp,
        "Soil_Moisture (%)": soil_moisture,
        "Air_Temperature (°C)": air_temp,
        "Air_Humidity (%)": air_humidity,
        "Paddy_Growth_Stage": paddy_growth_stage,
        "Purpose": purpose,
        "Quantity_kg_per_acre": 0
    }])

    for col, encoder in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = input_data[col].apply(lambda v: safe_transform(encoder, v))

    X = input_data[feature_names]
    X_scaled = scaler.transform(X)

    pred_encoded = model.predict(X_scaled)
    fertilizer = target_encoder.inverse_transform(pred_encoded)[0]

    print(f"Fertilizer Recommendation: {fertilizer}")
    print(f"Input Conditions: {soil_temp:.2f}°C soil, {soil_moisture:.2f}% moisture, {air_temp:.2f}°C air, {air_humidity:.2f}% humidity")
    return fertilizer

#  SERIAL MONITOR
class SerialMonitor:
    def __init__(self):
        self.port = "COM3"
        self.baudrate = 115200
        self.ser = None
        self.sensor_values = {}

    def connect(self):
        print("SERIAL MONITOR SYSTEM - INITIALIZATION")
        print(f"Using PORT: {self.port}, BAUD: {self.baudrate}")
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to {self.port} successfully\n")
        except Exception:
            print("ERROR: Cannot open serial port!")
            traceback.print_exc()
            sys.exit(1)

    def parse_sensor_line(self, text):
        try:
            if "DHT11 Temperature" in text:
                self.sensor_values["air_temp"] = float(text.split(":")[1].replace("°C","").strip())
            elif "DHT11 Humidity" in text:
                self.sensor_values["air_humidity"] = float(text.split(":")[1].replace("%","").strip())
            elif "DS18B20 Temperature" in text:
                self.sensor_values["soil_temp"] = float(text.split(":")[1].replace("°C","").strip())
            elif "Soil Moisture Value" in text:
                raw = float(text.split(":")[1].strip())
                self.sensor_values["soil_moisture"] = (raw / 4095) * 100
        except Exception:
            print(f"Failed to parse sensor line: {text}")

    def read_once(self):
        print("Reading sensor data ...\n")
        while True:
            raw = self.ser.readline()
            if raw:
                try:
                    text = raw.decode('utf-8').strip()
                except UnicodeDecodeError:
                    text = raw.decode(errors="ignore").strip()

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Raw Serial: {text}")

                self.parse_sensor_line(text)

                # Check  required sensor values
                required_keys = ["soil_temp", "soil_moisture", "air_temp", "air_humidity"]
                if all(k in self.sensor_values for k in required_keys):
                    predict_fertilizer(
                        soil_temp=self.sensor_values["soil_temp"],
                        soil_moisture=self.sensor_values["soil_moisture"],
                        air_temp=self.sensor_values["air_temp"],
                        air_humidity=self.sensor_values["air_humidity"],
                        paddy_growth_stage="Panicle Initiation",
                        purpose="Phosphorus for root development"
                    )
                    break

if __name__ == "__main__":
    monitor = SerialMonitor()
    monitor.connect()
    monitor.read_once()
