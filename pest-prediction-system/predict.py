import argparse
import numpy as np
import joblib
import json

MODEL_PATH = "paddy_pest_rf.joblib"
META_PATH = "paddy_meta.joblib"

def load_artifacts(model_path=MODEL_PATH, meta_path=META_PATH):
    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    return model, meta

def predict_from_features(values, model, meta):
    arr = np.array(values, dtype=float).reshape(1, -1)

    scaler = meta["scaler"]
    X_scaled = scaler.transform(arr)

    pred_idx = model.predict(X_scaled)[0]
    label = meta["label_encoder"].inverse_transform([pred_idx])[0]

    try:
        prob = float(np.max(model.predict_proba(X_scaled)[0]))
    except:
        prob = None

    # Dynamic action from dataset
    action_map = meta.get("action_map", {})
    action = action_map.get(label, "No recommended action found in dataset.")

    return {
        "predicted_pest": label,
        "probability": prob,
        "recommended_action": action
    }

def parse_sample_string(s):
    parts = s.split(",")
    if len(parts) != 5:
        raise ValueError("Use: temp,hum,pressure,light,dayofyear")
    return [float(p.strip()) for p in parts]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str)
    parser.add_argument("--batch_csv", type=str)
    args = parser.parse_args()

    model, meta = load_artifacts()

    if args.sample:
        features = parse_sample_string(args.sample)
        out = predict_from_features(features, model, meta)
        print(json.dumps(out, indent=2))

    elif args.batch_csv:
        import pandas as pd
        df = pd.read_csv(args.batch_csv, header=None)
        results = []
        for i, row in df.iterrows():
            feats = row.values[:5]
            res = predict_from_features(feats, model, meta)
            res["row_index"] = int(i)
            results.append(res)
        print(json.dumps(results, indent=2))

    else:
        # user input

        T = float(input("Enter Temperature: "))
        H = float(input("Enter Humidity: "))
        P = float(input("Enter Pressure: "))
        L = float(input("Enter LDR: "))

        demo = [T, H, P, L]
        print(json.dumps(predict_from_features(demo, model, meta), indent=2))

if __name__ == "__main__":
    main()
