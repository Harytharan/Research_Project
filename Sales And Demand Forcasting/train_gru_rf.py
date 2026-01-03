import os
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from load import load_dataset


# =======================
# FEATURE ENGINEERING
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


def encode_categoricals(df, save_dir="models", prefix=""):
    """
    Label-encode object columns (except Date). Saves encoders for prediction time.
    """
    df = df.copy()
    os.makedirs(save_dir, exist_ok=True)

    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != "Date"]
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        joblib.dump(le, os.path.join(save_dir, f"{prefix}{col}_encoder.joblib"))
        encoders[col] = le

    return df, cat_cols


# =======================
# GRU SEQUENCES (NEXT-STEP)
# =======================

def create_sequences_next_step(df, feature_cols, target_col, window_size):
    X, y = [], []
    feat = df[feature_cols].values
    tgt = df[target_col].values

    for end in range(window_size, len(df) - 1):
        start = end - window_size
        X.append(feat[start:end])
        y.append(tgt[end + 1])  # next day
    return np.array(X), np.array(y)


def build_gru_model(input_shape):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.25),
        GRU(32, return_sequences=False),
        Dropout(0.20),
        Dense(32, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae", "mape"])
    return model


# =======================
# MAIN
# =======================

def main():
    os.makedirs("models", exist_ok=True)

    print("Loading dataset...")
    df = load_dataset()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    required = {"Date", "Paddy_Price_LKR_per_kg", "Demand_Tons"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # =====================================================
    # [1/2] GRU TRAIN (PRICE DELTA)
    # =====================================================
    print("\n[1/2] Training GRU for PRICE DELTA...")

    df_price = df.copy()
    df_price = add_time_features(df_price)
    df_price = add_price_features(df_price, "Paddy_Price_LKR_per_kg")
    df_price = add_optional_weather_rolls(df_price)

    # Encode categoricals BEFORE scaling
    df_price, gru_cat_cols = encode_categoricals(df_price, save_dir="models", prefix="gru_")
    if gru_cat_cols:
        print("Encoded GRU categorical columns:", gru_cat_cols)

    # Delta target
    df_price["Price_Delta"] = df_price["Paddy_Price_LKR_per_kg"].diff()
    df_price = df_price.dropna().reset_index(drop=True)

    window_size = 21
    target_col = "Price_Delta"
    gru_feature_cols = [c for c in df_price.columns if c not in {"Date", target_col}]

    # Scale X and y
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_all = df_price[gru_feature_cols].values.astype(float)
    y_all = df_price[[target_col]].values.astype(float)

    X_all_scaled = X_scaler.fit_transform(X_all)
    y_all_scaled = y_scaler.fit_transform(y_all).ravel()

    df_price_scaled = df_price.copy()
    df_price_scaled[gru_feature_cols] = X_all_scaled
    df_price_scaled[target_col] = y_all_scaled

    X_seq, y_seq = create_sequences_next_step(df_price_scaled, gru_feature_cols, target_col, window_size)

    split = int(0.8 * len(X_seq))
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    tf.keras.backend.clear_session()
    gru = build_gru_model((X_train.shape[1], X_train.shape[2]))
    gru.summary()

    ckpt_path = os.path.join("models", "best_gru_price_delta.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7, verbose=1),
    ]

    print("Training GRU...")
    gru.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        shuffle=False,
        callbacks=callbacks,
        verbose=1
    )

    # Sanity check
    val_preds = gru.predict(X_val, verbose=0).flatten()
    print("\nGRU sanity check:")
    print("  Pred delta std:", float(np.std(val_preds)))
    print("  True delta std:", float(np.std(y_val)))

    # Save artifacts
    gru.save("models/gru_price_model_final.h5")
    joblib.dump(gru_feature_cols, "models/gru_feature_columns.joblib")
    joblib.dump(X_scaler, "models/gru_x_scaler.joblib")
    joblib.dump(y_scaler, "models/gru_y_scaler.joblib")
    joblib.dump({"window_size": window_size, "target": "Price_Delta"}, "models/training_info_gru.joblib")
    print("✅ Saved GRU model + features + scalers + encoders.")

    # =====================================================
    # [2/2] RF TRAIN (DEMAND)
    # =====================================================
    print("\n[2/2] Training RandomForest for DEMAND...")

    df_dem = df.copy()
    df_dem = add_time_features(df_dem)
    df_dem = add_price_features(df_dem, "Paddy_Price_LKR_per_kg")
    df_dem = add_optional_weather_rolls(df_dem)
    df_dem = add_demand_features(df_dem, "Demand_Tons")
    df_dem = add_demand_lags(df_dem, "Demand_Tons", n_lags=21)

    # Encode categoricals for RF (can be same cols, but keep separate files to be safe)
    df_dem, rf_cat_cols = encode_categoricals(df_dem, save_dir="models", prefix="rf_")
    if rf_cat_cols:
        print("Encoded RF categorical columns:", rf_cat_cols)

    df_dem = df_dem.dropna().reset_index(drop=True)

    target_demand = "Demand_Tons"
    rf_feature_cols = [c for c in df_dem.columns if c not in {"Date", target_demand}]

    X_tab = df_dem[rf_feature_cols].astype(float)
    y_tab = df_dem[target_demand].astype(float).values

    split = int(0.8 * len(X_tab))
    X_train, X_test = X_tab.iloc[:split], X_tab.iloc[split:]
    y_train, y_test = y_tab[:split], y_tab[split:]

    rf = RandomForestRegressor(
        n_estimators=600,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    print("Training RF...")
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\nRF Evaluation:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  R²  : {r2:.4f}")

    joblib.dump(rf, "models/rf_demand_model.joblib")
    joblib.dump(rf_feature_cols, "models/rf_feature_columns.joblib")

    training_info = {
        "window_size": window_size,
        "gru_features_count": len(gru_feature_cols),
        "rf_features_count": len(rf_feature_cols),
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rf_rmse": float(rmse),
        "rf_mae": float(mae),
        "rf_r2": float(r2),
        "gru_cat_cols": gru_cat_cols,
        "rf_cat_cols": rf_cat_cols,
    }
    joblib.dump(training_info, "models/training_info.joblib")

    print("\n✅ All models and artifacts saved in models/ folder.")


if __name__ == "__main__":
    main()
