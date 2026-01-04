# ===========================
# predict_from_sensor.py  (uses ΔPrice LSTM model) ✅ non-flat multi-day prices
# ===========================
import serial
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from load import load_dataset
from pre_process import preprocess

PORT = "COM7"
BAUD = 115200
N_DAYS = 14


def extract_value(line):
    try:
        return float(line.split(":")[1].strip())
    except Exception:
        return None


# -------------------------
# Feature engineering (same as train)
# -------------------------
def add_rolling_and_seasonal(df):
    df_copy = df.copy()

    for window in [3, 7, 14, 21]:
        df_copy[f"Demand_roll{window}"] = df_copy["Demand_Tons"].rolling(window).mean()
        df_copy[f"Price_roll{window}"] = df_copy["Paddy_Price_LKR_per_kg"].rolling(window).mean()
        df_copy[f"Temperature_roll{window}"] = df_copy["Temperature_C"].rolling(window).mean()
        df_copy[f"Rainfall_roll{window}"] = df_copy["Rainfall_mm"].rolling(window).mean()

    df_copy["Demand_roll7_std"] = df_copy["Demand_Tons"].rolling(7).std()
    df_copy["Price_roll7_std"] = df_copy["Paddy_Price_LKR_per_kg"].rolling(7).std()

    df_copy["Month"] = pd.to_datetime(df_copy["Date"]).dt.month
    df_copy["month_sin"] = np.sin(2 * np.pi * df_copy["Month"] / 12)
    df_copy["month_cos"] = np.cos(2 * np.pi * df_copy["Month"] / 12)

    df_copy["Quarter"] = pd.to_datetime(df_copy["Date"]).dt.quarter
    df_copy["quarter_sin"] = np.sin(2 * np.pi * df_copy["Quarter"] / 4)
    df_copy["quarter_cos"] = np.cos(2 * np.pi * df_copy["Quarter"] / 4)

    df_copy["day_of_week"] = pd.to_datetime(df_copy["Date"]).dt.dayofweek
    df_copy["dow_sin"] = np.sin(2 * np.pi * df_copy["day_of_week"] / 7)
    df_copy["dow_cos"] = np.cos(2 * np.pi * df_copy["day_of_week"] / 7)

    df_copy["day_of_year"] = pd.to_datetime(df_copy["Date"]).dt.dayofyear
    df_copy["doy_sin"] = np.sin(2 * np.pi * df_copy["day_of_year"] / 365)
    df_copy["doy_cos"] = np.cos(2 * np.pi * df_copy["day_of_year"] / 365)

    df_copy["year_progress"] = df_copy["day_of_year"] / 365.0
    df_copy["is_weekend"] = (df_copy["day_of_week"] >= 5).astype(int)
    return df_copy


def add_price_momentum(df):
    df_copy = df.copy()

    df_copy["price_diff_1"] = df_copy["Paddy_Price_LKR_per_kg"].diff()
    df_copy["price_diff_3"] = df_copy["Paddy_Price_LKR_per_kg"].diff(3)
    df_copy["price_diff_7"] = df_copy["Paddy_Price_LKR_per_kg"].diff(7)

    df_copy["price_momentum_3"] = df_copy["Paddy_Price_LKR_per_kg"].pct_change(3)
    df_copy["price_momentum_7"] = df_copy["Paddy_Price_LKR_per_kg"].pct_change(7)

    df_copy["price_volatility_7"] = df_copy["Paddy_Price_LKR_per_kg"].rolling(7).std()
    return df_copy


def ensure_cols(df, cols):
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df


# -------------------------
# Forecast ΔPrice then reconstruct price
# -------------------------
def predict_price_future_delta(df_base, model, feature_cols, X_scaler, y_scaler, window_size, start_date, n_days):
    df_ext = df_base.copy().reset_index(drop=True)
    preds = []
    dates = []

    for i in range(n_days):
        next_date = start_date + pd.Timedelta(days=i)

        # append placeholder row for date
        next_row = df_ext.iloc[-1].copy()
        next_row["Date"] = next_date
        df_tmp = pd.concat([df_ext, pd.DataFrame([next_row])], ignore_index=True)

        df_tmp = add_rolling_and_seasonal(df_tmp)
        df_tmp = add_price_momentum(df_tmp)
        df_tmp = df_tmp.dropna().reset_index(drop=True)

        # make sure we have columns
        df_tmp = ensure_cols(df_tmp, feature_cols)

        feat = df_tmp[feature_cols].values
        feat_scaled = X_scaler.transform(feat)

        window = feat_scaled[-window_size:, :]
        X_in = window.reshape(1, window_size, len(feature_cols))

        # predict delta in scaled space -> back to real delta
        delta_scaled = float(model.predict(X_in, verbose=0)[0][0])
        delta_real = float(y_scaler.inverse_transform([[delta_scaled]])[0][0])

        last_price = float(df_tmp["Paddy_Price_LKR_per_kg"].iloc[-1])
        next_price = last_price + delta_real

        preds.append(next_price)
        dates.append(next_date)

        df_tmp.loc[df_tmp.index[-1], "Paddy_Price_LKR_per_kg"] = next_price
        df_ext = df_tmp

    return np.array(preds, dtype=float), dates


def predict_demand_future_xgb(df_std_base, xgb, feature_cols_xgb, price_preds, pred_dates):
    df_future = []
    for pr, d in zip(price_preds, pred_dates):
        row = df_std_base.iloc[-1].copy()
        row["Date"] = d
        row["Paddy_Price_LKR_per_kg"] = float(pr)
        row["Price_LSTM_pred"] = float(pr)
        df_future.append(row)

    df_future = pd.DataFrame(df_future)

    # simple engineered features same as your older approach
    df_all = pd.concat([df_std_base, df_future], ignore_index=True)
    df_all = add_rolling_and_seasonal(df_all)
    df_all = add_price_momentum(df_all)

    df_future_eng = df_all.iloc[-len(df_future):].copy()
    df_future_eng = df_future_eng.fillna(method="ffill").fillna(method="bfill").fillna(0)

    df_future_eng = ensure_cols(df_future_eng, feature_cols_xgb)
    X_future = df_future_eng[feature_cols_xgb]

    return np.array(xgb.predict(X_future), dtype=float)


# -------------------------
# Plot
# -------------------------
def plot_predictions(hist_dates, hist_price, hist_demand, pred_dates, price_preds, demand_preds):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(hist_dates, hist_price, label="Historical Price", linewidth=2)
    ax1.plot(pred_dates, price_preds, "ro-", label="Predicted Price")
    ax1.axvline(hist_dates.iloc[-1], linestyle="--", color="gray")
    ax1.set_ylabel("Price (LKR/kg)")
    ax1.set_title("Paddy Price Prediction")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(hist_dates, hist_demand, label="Historical Demand", linewidth=2)
    ax2.plot(pred_dates, demand_preds, "mo-", label="Predicted Demand")
    ax2.axvline(hist_dates.iloc[-1], linestyle="--", color="gray")
    ax2.set_ylabel("Demand (Tons)")
    ax2.set_xlabel("Date")
    ax2.set_title("Demand Prediction")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    # Load models + artifacts
    lstm = load_model("models/lstm_price_delta_model_final.h5")
    xgb = joblib.load("models/xgb_demand_model_best_optimized.joblib")

    feature_cols_lstm = joblib.load("models/lstm_feature_columns.joblib")
    feature_cols_xgb = joblib.load("models/feature_columns_optimized.joblib")

    training_info = joblib.load("models/training_info.joblib")
    window_size = int(training_info.get("window_size", 21))

    X_scaler = joblib.load("models/lstm_X_scaler.joblib")
    y_scaler = joblib.load("models/lstm_y_scaler.joblib")

    # Load data
    df = load_dataset()
    df_raw, df_mm, df_std, _ = preprocess(df, save_artifacts=False)

    # Build df_mm for LSTM (same feature eng as train)
    df_mm = add_rolling_and_seasonal(df_mm)
    df_mm = add_price_momentum(df_mm)
    df_mm = df_mm.dropna().reset_index(drop=True)

    # For XGB demand base
    df_std = add_rolling_and_seasonal(df_std)
    df_std = add_price_momentum(df_std)
    df_std = df_std.dropna().reset_index(drop=True)

    # History window for plot: ✅ ~2.5 months (about 75 days)
    hist_days = min(75, len(df_raw))
    hist_dates = pd.to_datetime(df_raw["Date"].iloc[-hist_days:])
    hist_price = df_raw["Paddy_Price_LKR_per_kg"].iloc[-hist_days:]
    hist_demand = df_raw["Demand_Tons"].iloc[-hist_days:]

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("Waiting for sensor data...")

    n = p = k = None

    try:
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

                # Inject sensor into last row (if columns exist)
                if "Nitrogen_N" in df_mm.columns:
                    df_mm.loc[df_mm.index[-1], "Nitrogen_N"] = float(n)
                if "Phosphorus_P" in df_mm.columns:
                    df_mm.loc[df_mm.index[-1], "Phosphorus_P"] = float(p)
                if "Potassium_K" in df_mm.columns:
                    df_mm.loc[df_mm.index[-1], "Potassium_K"] = float(k)

                start_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)

                price_preds, pred_dates = predict_price_future_delta(
                    df_base=df_mm,
                    model=lstm,
                    feature_cols=feature_cols_lstm,
                    X_scaler=X_scaler,
                    y_scaler=y_scaler,
                    window_size=window_size,
                    start_date=start_date,
                    n_days=N_DAYS
                )

                demand_preds = predict_demand_future_xgb(
                    df_std_base=df_std,
                    xgb=xgb,
                    feature_cols_xgb=feature_cols_xgb,
                    price_preds=price_preds,
                    pred_dates=pred_dates
                )

                print("\nPREDICTIONS")
                for d, pr, dm in zip(pred_dates, price_preds, demand_preds):
                    print(f"{pd.to_datetime(d).date()} → Price {pr:.2f} | Demand {dm:.2f}")

                plot_predictions(hist_dates, hist_price, hist_demand, pred_dates, price_preds, demand_preds)
                break

    finally:
        ser.close()
        print("System finished.")


if __name__ == "__main__":
    main()
