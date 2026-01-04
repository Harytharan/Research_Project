# ===========================
# train.py  (ΔPrice LSTM + XGB Demand)  ✅ fixes flat price forecasts
# ===========================
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

import xgboost as xgb

from load import load_dataset
from pre_process import preprocess
from featureeng import create_lstm_sequences


# -------------------------
# SETTINGS
# -------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

WINDOW_SIZE = 21
EPOCHS = 200
BATCH_SIZE = 64


# -------------------------
# Feature Engineering (same as yours)
# -------------------------
def add_lag_features(df, target_col, n_lags=21):
    df_copy = df.copy()
    for lag in range(1, n_lags + 1):
        df_copy[f"{target_col}_lag{lag}"] = df_copy[target_col].shift(lag)
    return df_copy


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


# -------------------------
# LSTM (predict ΔPrice)
# -------------------------
def build_delta_lstm(input_shape):
    model = Sequential([
        LSTM(96, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.20),

        LSTM(48, return_sequences=False,
             kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.15),

        Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.10),
        Dense(1, activation="linear")
    ])

    opt = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=opt, loss=Huber(), metrics=["mae"])
    return model


# -------------------------
# Helpers
# -------------------------
def safe_mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def main():
    print("Loading dataset...")
    df = load_dataset()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # preprocess from your pipeline
    df_raw, df_mm, df_std, artifacts = preprocess(df, save_artifacts=True, artifact_dir=MODEL_DIR)

    # -------------------------
    # LSTM PRICE (ΔPrice)
    # -------------------------
    print("\nApplying feature engineering for LSTM...")
    df_mm = add_rolling_and_seasonal(df_mm)
    df_mm = add_price_momentum(df_mm)

    # ✅ key: ΔPrice target
    df_mm["Price_Delta"] = df_mm["Paddy_Price_LKR_per_kg"].diff()

    df_mm = df_mm.dropna().reset_index(drop=True)

    exclude = {"Date", "Paddy_Price_LKR_per_kg", "Demand_Tons", "Price_Delta"}
    feature_cols_lstm = [c for c in df_mm.columns if c not in exclude]

    print(f"LSTM Feature columns ({len(feature_cols_lstm)}): {feature_cols_lstm}")

    # Scale X and y_delta
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = df_mm.copy()
    X_scaled[feature_cols_lstm] = X_scaler.fit_transform(X_scaled[feature_cols_lstm].values)

    y_scaled = df_mm.copy()
    y_scaled["Price_Delta"] = y_scaler.fit_transform(y_scaled[["Price_Delta"]].values)

    # Save scalers + columns
    joblib.dump(X_scaler, os.path.join(MODEL_DIR, "lstm_X_scaler.joblib"))
    joblib.dump(y_scaler, os.path.join(MODEL_DIR, "lstm_y_scaler.joblib"))
    joblib.dump(feature_cols_lstm, os.path.join(MODEL_DIR, "lstm_feature_columns.joblib"))

    # sequences
    X_seq, y_seq = create_lstm_sequences(X_scaled, feature_cols_lstm, "Price_Delta", WINDOW_SIZE)
    print(f"\nLSTM sequences: X={X_seq.shape}, y={y_seq.shape}")

    split = int(0.8 * len(X_seq))
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    tf.keras.backend.clear_session()
    lstm = build_delta_lstm((X_train.shape[1], X_train.shape[2]))
    lstm.summary()

    ckpt = os.path.join(MODEL_DIR, "best_lstm_price_delta.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt, save_best_only=True, monitor="val_loss", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1),
    ]

    print("\nTraining LSTM ΔPrice model...")
    lstm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate ΔPrice in real units
    y_train_pred = lstm.predict(X_train, verbose=0)
    y_val_pred = lstm.predict(X_val, verbose=0)

    y_train_real = y_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_val_real = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_train_pred_real = y_scaler.inverse_transform(y_train_pred).flatten()
    y_val_pred_real = y_scaler.inverse_transform(y_val_pred).flatten()

    print("\nLSTM ΔPrice Evaluation (delta units):")
    print(f"  Train R²: {r2_score(y_train_real, y_train_pred_real):.4f} | RMSE: {mean_squared_error(y_train_real, y_train_pred_real, squared=False):.4f}")
    print(f"  Val   R²: {r2_score(y_val_real, y_val_pred_real):.4f} | RMSE: {mean_squared_error(y_val_real, y_val_pred_real, squared=False):.4f}")

    # Save final model
    lstm.save(os.path.join(MODEL_DIR, "lstm_price_delta_model_final.h5"))
    print("\nSaved: models/lstm_price_delta_model_final.h5")

    # -------------------------
    # XGBOOST DEMAND
    # -------------------------
    print("\nPreparing demand training data...")

    # Build Price_LSTM_pred for df_std using ΔPrice predictions aligned to df_mm rows
    # We will predict delta for every available window and reconstruct price for those rows.
    X_full_seq, _ = create_lstm_sequences(X_scaled, feature_cols_lstm, "Price_Delta", WINDOW_SIZE)
    pred_delta_scaled = lstm.predict(X_full_seq, verbose=0).flatten()
    pred_delta = y_scaler.inverse_transform(pred_delta_scaled.reshape(-1, 1)).flatten()

    # Align: each sequence predicts delta at position i+WINDOW_SIZE
    # reconstruct predicted price = actual price at t-1 + predicted delta at t
    # Use df_mm actual price for previous day alignment
    prices = df_mm["Paddy_Price_LKR_per_kg"].values
    pred_price = np.zeros_like(pred_delta)

    # target index in df_mm that this delta belongs to:
    # t = WINDOW_SIZE..WINDOW_SIZE+len(pred_delta)-1
    for j in range(len(pred_delta)):
        t = WINDOW_SIZE + j
        prev_price = prices[t - 1]
        pred_price[j] = prev_price + pred_delta[j]

    # Map these predicted prices into df_std length safely (best-effort alignment)
    # df_std was produced by preprocess; we align by min length after trimming.
    pred_prices_full = np.zeros(len(df_std), dtype=float)

    start = WINDOW_SIZE
    end = min(start + len(pred_price), len(pred_prices_full))
    pred_prices_full[start:end] = pred_price[:(end - start)]
    if end > 0:
        pred_prices_full[:start] = pred_prices_full[start] if start < len(pred_prices_full) else np.mean(pred_price)

    df_std = df_std.copy()
    df_std["Price_LSTM_pred"] = pred_prices_full

    # Demand features
    df_std = add_lag_features(df_std, "Demand_Tons", n_lags=21)
    df_std = add_rolling_and_seasonal(df_std)
    df_std = add_price_momentum(df_std)
    df_std = df_std.dropna().reset_index(drop=True)

    # Encode categoricals
    cat_cols = [c for c in df_std.columns if df_std[c].dtype == "object" and c != "Date"]
    for col in cat_cols:
        le = LabelEncoder()
        df_std[col] = le.fit_transform(df_std[col].astype(str))
        joblib.dump(le, os.path.join(MODEL_DIR, f"{col}_encoder.joblib"))

    target_col = "Demand_Tons"
    X_tab = df_std.drop(columns=[target_col, "Date"])
    y_tab = df_std[target_col].values

    feature_cols_xgb = X_tab.columns.tolist()
    joblib.dump(feature_cols_xgb, os.path.join(MODEL_DIR, "feature_columns_optimized.joblib"))

    split_idx = int(0.8 * len(X_tab))
    X_train_tab, X_test_tab = X_tab.iloc[:split_idx], X_tab.iloc[split_idx:]
    y_train_tab, y_test_tab = y_tab[:split_idx], y_tab[split_idx:]

    model_xgb = xgb.XGBRegressor(
        n_estimators=2500,
        learning_rate=0.02,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED,
        objective="reg:squarederror",
        tree_method="hist"
    )

    print("\nTraining XGBoost demand model...")
    model_xgb.fit(
        X_train_tab, y_train_tab,
        eval_set=[(X_test_tab, y_test_tab)],
        eval_metric="rmse",
        early_stopping_rounds=100,
        verbose=100
    )

    y_pred = model_xgb.predict(X_test_tab)
    print("\nXGBoost Demand Evaluation:")
    print(f"  R²  : {r2_score(y_test_tab, y_pred):.4f}")
    print(f"  MAE : {mean_absolute_error(y_test_tab, y_pred):.4f}")
    print(f"  RMSE: {mean_squared_error(y_test_tab, y_pred, squared=False):.4f}")
    print(f"  MAPE: {safe_mape(y_test_tab, y_pred):.2f}%")

    joblib.dump(model_xgb, os.path.join(MODEL_DIR, "xgb_demand_model_best_optimized.joblib"))

    training_info = {
        "window_size": WINDOW_SIZE,
        "lstm_features": feature_cols_lstm,
        "xgb_features": feature_cols_xgb,
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_shape": df.shape,
        "price_model": "delta"
    }
    joblib.dump(training_info, os.path.join(MODEL_DIR, "training_info.joblib"))

    print("\nAll saved to models/:")
    print("  - lstm_price_delta_model_final.h5")
    print("  - best_lstm_price_delta.keras")
    print("  - lstm_X_scaler.joblib / lstm_y_scaler.joblib")
    print("  - lstm_feature_columns.joblib")
    print("  - xgb_demand_model_best_optimized.joblib")
    print("  - feature_columns_optimized.joblib")
    print("  - training_info.joblib")


if __name__ == "__main__":
    main()
