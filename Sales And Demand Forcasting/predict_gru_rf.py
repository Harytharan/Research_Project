import os
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Plot backend (still saves PNG even if GUI fails)
try:
    import matplotlib
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from load import load_dataset


# =======================
# FEATURE ENGINEERING (same as training)
# =======================

def add_time_features(df):
    df = df.copy()
    date = pd.to_datetime(df["Date"])

    df["Month"] = date.dt.month
    df["Quarter"] = date.dt.quarter
    df["day_of_week"] = date.dt.dayofweek
    df["day_of_year"] = date.dt.dayofyear

    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * df["Quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["Quarter"] / 4)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    df["year_progress"] = df["day_of_year"] / 365.0
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_price_features(df, price_col="Paddy_Price_LKR_per_kg"):
    df = df.copy()

    for w in [3, 7, 14, 21]:
        df[f"price_roll{w}"] = df[price_col].rolling(w).mean()

    df["price_roll7_std"] = df[price_col].rolling(7).std()

    df["price_diff_1"] = df[price_col].diff(1)
    df["price_diff_3"] = df[price_col].diff(3)
    df["price_diff_7"] = df[price_col].diff(7)

    df["price_momentum_3"] = df[price_col].pct_change(3)
    df["price_momentum_7"] = df[price_col].pct_change(7)

    df["price_volatility_7"] = df[price_col].rolling(7).std()
    return df


def add_demand_features(df, demand_col="Demand_Tons"):
    df = df.copy()
    for w in [3, 7, 14, 21]:
        df[f"demand_roll{w}"] = df[demand_col].rolling(w).mean()
    df["demand_roll7_std"] = df[demand_col].rolling(7).std()
    return df


def add_demand_lags(df, demand_col="Demand_Tons", n_lags=21):
    df = df.copy()
    for i in range(1, n_lags + 1):
        df[f"{demand_col}_lag{i}"] = df[demand_col].shift(i)
    return df


def add_optional_weather_rolls(df):
    df = df.copy()
    if "Temperature_C" in df.columns:
        for w in [3, 7, 14, 21]:
            df[f"temp_roll{w}"] = df["Temperature_C"].rolling(w).mean()
    if "Rainfall_mm" in df.columns:
        for w in [3, 7, 14, 21]:
            df[f"rain_roll{w}"] = df["Rainfall_mm"].rolling(w).mean()
    return df


# =======================
# ENCODERS (LOAD + APPLY)
# =======================

def load_encoders(models_dir="models", prefix="gru_"):
    """
    Loads all encoders matching: models/{prefix}*_encoder.joblib
    Returns dict {colname: encoder}
    """
    encoders = {}
    if not os.path.isdir(models_dir):
        return encoders

    for fname in os.listdir(models_dir):
        if fname.startswith(prefix) and fname.endswith("_encoder.joblib"):
            col = fname[len(prefix):-len("_encoder.joblib")]
            encoders[col] = joblib.load(os.path.join(models_dir, fname))
    return encoders


def safe_label_transform(series, encoder):

    values = series.astype(str).values
    known = set(getattr(encoder, "classes_", []))

    out = np.empty(len(values), dtype=int)
    for i, v in enumerate(values):
        if v in known:
            out[i] = int(encoder.transform([v])[0])
        else:
            out[i] = -1  # unseen category
    return out


def apply_encoders(df, encoders):
    df = df.copy()
    for col, enc in encoders.items():
        if col in df.columns:
            df[col] = safe_label_transform(df[col], enc)
    return df


# =======================
# RECOMMENDATION
# =======================

def generate_farmer_recommendation(price_preds, demand_preds, horizon_days=7):
    horizon_days = min(horizon_days, len(price_preds))
    p = np.array(price_preds[:horizon_days], dtype=float)
    d = np.array(demand_preds[:horizon_days], dtype=float)

    price_change_pct = ((p[-1] - p[0]) / (p[0] + 1e-9)) * 100
    demand_change_pct = ((d[-1] - d[0]) / (d[0] + 1e-9)) * 100

    if price_change_pct <= -2:
        return "✅ Recommendation: SELL within 1 week (price expected to drop)."
    elif price_change_pct >= 2 and demand_change_pct >= 0:
        return "✅ Recommendation: DELAY selling (price expected to rise)."
    elif demand_change_pct <= -5:
        return "✅ Recommendation: SELL soon (demand expected to drop)."
    else:
        return "✅ Recommendation: MONITOR market (no strong signal)."


# =======================
# TRUE FUTURE PRICE FORECAST (DELTA ACCUMULATION)
# =======================

def forecast_future_price_gru_delta(df_hist, gru_model, gru_features, x_scaler, y_scaler,
                                   start_date, n_days, window_size, gru_encoders):
    df = df_hist.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    last_price = float(df["Paddy_Price_LKR_per_kg"].iloc[-1])

    # Build initial engineered features
    df_feat = df.copy()
    df_feat = add_time_features(df_feat)
    df_feat = add_price_features(df_feat, "Paddy_Price_LKR_per_kg")
    df_feat = add_optional_weather_rolls(df_feat)

    # ✅ encode categoricals BEFORE scaling
    df_feat = apply_encoders(df_feat, gru_encoders)

    df_feat = df_feat.fillna(method="ffill").fillna(method="bfill").fillna(0)

    missing = set(gru_features) - set(df_feat.columns)
    if missing:
        raise ValueError(f"Missing GRU features in prediction data: {missing}")

    price_preds, dates = [], []

    for i in range(n_days):
        d = start_date + pd.Timedelta(days=i)
        dates.append(d)

        # last window
        X_window = df_feat[gru_features].values[-window_size:]

        # Make sure numeric
        X_window = X_window.astype(float)

        X_window_scaled = x_scaler.transform(X_window)
        X = X_window_scaled.reshape(1, window_size, len(gru_features))

        pred_delta_scaled = float(gru_model.predict(X, verbose=0)[0][0])
        pred_delta = float(y_scaler.inverse_transform([[pred_delta_scaled]])[0][0])

        next_price = last_price + pred_delta
        price_preds.append(next_price)
        last_price = next_price

        # Append new future row
        new_row = df_feat.iloc[-1].copy()
        new_row["Date"] = d
        new_row["Paddy_Price_LKR_per_kg"] = next_price

        df_feat = pd.concat([df_feat, pd.DataFrame([new_row])], ignore_index=True)

        # Rebuild features + encode again (safe)
        df_feat = add_time_features(df_feat)
        df_feat = add_price_features(df_feat, "Paddy_Price_LKR_per_kg")
        df_feat = add_optional_weather_rolls(df_feat)
        df_feat = apply_encoders(df_feat, gru_encoders)

        df_feat = df_feat.fillna(method="ffill").fillna(method="bfill").fillna(0)

    return price_preds, dates


# =======================
# TRUE FUTURE DEMAND FORECAST (RECURSIVE)
# =======================

def forecast_future_demand_rf(df_hist, rf_model, rf_features, future_dates, future_prices,
                              n_lags=21, rf_encoders=None):
    if rf_encoders is None:
        rf_encoders = {}

    df = df_hist.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    demand_preds = []

    for d, p in zip(future_dates, future_prices):
        new_row = df.iloc[-1].copy()
        new_row["Date"] = pd.to_datetime(d)
        new_row["Paddy_Price_LKR_per_kg"] = float(p)

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df_feat = df.copy()
        df_feat = add_time_features(df_feat)
        df_feat = add_price_features(df_feat, "Paddy_Price_LKR_per_kg")
        df_feat = add_optional_weather_rolls(df_feat)
        df_feat = add_demand_features(df_feat, "Demand_Tons")
        df_feat = add_demand_lags(df_feat, "Demand_Tons", n_lags=n_lags)

        # encode categoricals
        df_feat = apply_encoders(df_feat, rf_encoders)

        df_feat = df_feat.fillna(method="ffill").fillna(method="bfill").fillna(0)

        missing = set(rf_features) - set(df_feat.columns)
        if missing:
            raise ValueError(f"Missing RF features in prediction data: {missing}")

        X_row = df_feat[rf_features].iloc[[-1]].astype(float)
        pred_demand = float(rf_model.predict(X_row)[0])
        demand_preds.append(pred_demand)

        # feedback
        df.at[df.index[-1], "Demand_Tons"] = pred_demand

    return demand_preds

# =======================
# MAIN
# =======================

def main():
    print("\nGRU (Delta) + RF Prediction System (TRUE FUTURE)")
    print("=" * 60)

    start_date_input = input("Enter start date (YYYY-MM-DD) or press Enter for tomorrow: ").strip()
    if start_date_input == "":
        start_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    else:
        start_date = pd.to_datetime(start_date_input, errors="coerce")
        if pd.isna(start_date):
            print("❌ Invalid date. Use YYYY-MM-DD (e.g., 2026-01-20)")
            return

    n_days_input = input("Enter number of days (default 7): ").strip()
    n_days = int(n_days_input) if n_days_input else 7

    print("\nLoading models/artifacts...")
    gru = load_model("models/gru_price_model_final.h5")
    rf = joblib.load("models/rf_demand_model.joblib")
    gru_features = joblib.load("models/gru_feature_columns.joblib")
    rf_features = joblib.load("models/rf_feature_columns.joblib")
    x_scaler = joblib.load("models/gru_x_scaler.joblib")
    y_scaler = joblib.load("models/gru_y_scaler.joblib")
    info_gru = joblib.load("models/training_info_gru.joblib")
    window_size = int(info_gru.get("window_size", 21))

    gru_encoders = load_encoders("models", prefix="gru_")
    rf_encoders = load_encoders("models", prefix="rf_")

    print("Loaded GRU encoders:", list(gru_encoders.keys()))
    print("Loaded RF encoders :", list(rf_encoders.keys()))

    print("\nLoading dataset...")
    df = load_dataset()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    required = {"Date", "Paddy_Price_LKR_per_kg", "Demand_Tons"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    print("\nForecasting future price (GRU delta recursive)...")
    price_preds, dates = forecast_future_price_gru_delta(
        df, gru, gru_features, x_scaler, y_scaler, start_date, n_days, window_size, gru_encoders
    )

    print("\nForecasting future demand (RF recursive)...")
    demand_preds = forecast_future_demand_rf(
        df, rf, rf_features, dates, price_preds, n_lags=21, rf_encoders=rf_encoders
    )

    advice = generate_farmer_recommendation(price_preds, demand_preds, horizon_days=min(7, n_days))

    print("\nPREDICTION RESULTS")
    print("-" * 60)
    for d, p, dem in zip(dates, price_preds, demand_preds):
        print(f"{pd.to_datetime(d).strftime('%Y-%m-%d')} | Price: {p:.2f} LKR | Demand: {dem:.2f} Tons")

    print("\nFARMER RECOMMENDATION")
    print("-" * 60)
    print(advice)

    # =======================
    # PLOTS (2) + SAVE
    # =======================
    results_df = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Predicted_Price": price_preds,
        "Predicted_Demand": demand_preds
    })

    os.makedirs("outputs", exist_ok=True)

    # Graph 1: Trends
    plt.figure()
    plt.plot(results_df["Date"], results_df["Predicted_Price"], marker="o", label="Predicted Price (LKR)")
    plt.plot(results_df["Date"], results_df["Predicted_Demand"], marker="o", label="Predicted Demand (Tons)")
    plt.title("Future Forecast: Price & Demand Trends (GRU-Delta + RF)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/gru_rf_trends.png", dpi=300)
    plt.gcf().text(0.02, 0.02, advice, fontsize=10)

    try:
        plt.show(block=True)
    except Exception:
        pass

    # Graph 2: Relationship
    plt.figure()
    plt.scatter(results_df["Predicted_Price"], results_df["Predicted_Demand"])
    plt.title("Predicted Demand vs Predicted Price (GRU-Delta + RF)")
    plt.xlabel("Predicted Price (LKR)")
    plt.ylabel("Predicted Demand (Tons)")
    plt.tight_layout()
    plt.savefig("outputs/gru_rf_price_vs_demand.png", dpi=300)

    try:
        plt.show(block=True)
    except Exception:
        pass

    print("\n✅ Graphs saved in outputs/")
    print("✅ Done ✔")


if __name__ == "__main__":
    main()
