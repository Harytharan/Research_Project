import serial
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from load import load_dataset
from pre_process import preprocess
from predict import (
    add_rolling_and_seasonal,
    add_price_momentum,
    predict_price_future_enhanced,
    predict_demand_future_enhanced
)

PORT = "COM7"
BAUD = 115200
N_DAYS = 14


def extract_value(line):
    try:
        return float(line.split(":")[1].strip())
    except:
        return None


def plot_predictions(hist_dates, hist_price, hist_demand,
                     pred_dates, price_preds, demand_preds):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Price
    ax1.plot(hist_dates, hist_price, label="Historical Price", linewidth=2)
    ax1.plot(pred_dates, price_preds, 'ro-', label="Predicted Price")
    ax1.axvline(hist_dates.iloc[-1], linestyle="--", color="gray")
    ax1.set_ylabel("Price (LKR/kg)")
    ax1.set_title("Paddy Price Prediction")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Demand
    ax2.plot(hist_dates, hist_demand, label="Historical Demand", linewidth=2)
    ax2.plot(pred_dates, demand_preds, 'mo-', label="Predicted Demand")
    ax2.axvline(hist_dates.iloc[-1], linestyle="--", color="gray")
    ax2.set_ylabel("Demand (Tons)")
    ax2.set_xlabel("Date")
    ax2.set_title("Demand Prediction")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_combined(pred_dates, price_preds, demand_preds):
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.set_ylabel("Price (LKR/kg)", color="red")
    ax1.plot(pred_dates, price_preds, 'ro-', label="Price")
    ax1.tick_params(axis='y', labelcolor="red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Demand (Tons)", color="blue")
    ax2.plot(pred_dates, demand_preds, 'bs-', label="Demand")
    ax2.tick_params(axis='y', labelcolor="blue")

    ax1.set_title("Price vs Demand Trend")
    ax1.grid(alpha=0.3)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels)

    plt.show()


def main():
    # Load models
    lstm = load_model("models/lstm_price_model_final.h5")
    xgb = joblib.load("models/xgb_demand_model_best_optimized.joblib")
    feature_cols_lstm = joblib.load("models/lstm_feature_columns.joblib")
    feature_cols_xgb = joblib.load("models/feature_columns_optimized.joblib")
    training_info = joblib.load("models/training_info.joblib")
    window_size = training_info["window_size"]

    # Load dataset
    df = load_dataset()
    df_raw, df_mm, df_std, _ = preprocess(df, save_artifacts=False)

    df_mm = add_rolling_and_seasonal(df_mm)
    df_mm = add_price_momentum(df_mm)
    df_mm = df_mm.dropna().reset_index(drop=True)

    df_std = add_rolling_and_seasonal(df_std)
    df_std = add_price_momentum(df_std)
    df_std = df_std.dropna().reset_index(drop=True)

    # Historical data (last 60 days)
    hist_days = 60
    hist_dates = pd.to_datetime(df_raw["Date"].iloc[-hist_days:])
    hist_price = df_raw["Paddy_Price_LKR_per_kg"].iloc[-hist_days:]
    hist_demand = df_raw["Demand_Tons"].iloc[-hist_days:]

    # Serial
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("Waiting for sensor data...")

    n = p = k = None

    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue

        print("RAW:", line)

        if "Nitrogen" in line:
            n = extract_value(line)
        elif "Phosphorus" in line:
            p = extract_value(line)
        elif "Potassium" in line:
            k = extract_value(line)

        if n is not None and p is not None and k is not None:
            print(f"\nSensor → N:{n} P:{p} K:{k}")

            # Inject sensor values
            df_mm.loc[df_mm.index[-1], ["Nitrogen_N", "Phosphorus_P", "Potassium_K"]] = [n, p, k]

            start_date = pd.Timestamp.now() + pd.Timedelta(days=1)

            # Predict
            price_preds, pred_dates = predict_price_future_enhanced(
                df_mm, lstm, feature_cols_lstm,
                start_date, n_steps=N_DAYS, window_size=window_size
            )

            demand_preds = predict_demand_future_enhanced(
                df_std, xgb, feature_cols_xgb,
                price_preds, pred_dates
            )

            # Print
            print("\nPREDICTIONS")
            for d, pr, dm in zip(pred_dates, price_preds, demand_preds):
                print(f"{d.date()} → Price {pr:.2f} | Demand {dm:.2f}")

            # Graphs
            plot_predictions(
                hist_dates, hist_price, hist_demand,
                pred_dates, price_preds, demand_preds
            )

            plot_combined(pred_dates, price_preds, demand_preds)

            break

    ser.close()
    print("System finished.")


if __name__ == "__main__":
    main()
