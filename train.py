import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import load_and_preprocess_data
import os

def main():
    print("Starting training pipeline for Fertilizer Recommendation System...\n")

    #  Load and preprocess data
    (
        X_train, X_test, y_train, y_test,
        scaler, target_encoder, label_encoders, feature_names
    ) = load_and_preprocess_data()

    # Model setup
    print("\nInitializing Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
    )

    # raining
    print("Training model...")
    model.fit(X_train, y_train)
    print("Model training complete!\n")

    #  Evaluation
    print("Evaluating model on test data...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model & encoders
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fertilizer_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(target_encoder, "models/target_encoder.pkl")
    joblib.dump(label_encoders, "models/feature_encoders.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")

    print("\nAll models and encoders saved successfully to /models/")
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
