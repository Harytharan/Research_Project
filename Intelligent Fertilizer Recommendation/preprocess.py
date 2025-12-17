import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path="fertilizer_recommendation_dataset.csv"):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    print(f"Columns found: {list(df.columns)}")

    # Rename column
    if 'Recommended_Fertilizer' in df.columns:
        df.rename(columns={'Recommended_Fertilizer': 'Fertilizer'}, inplace=True)
        print("Renamed 'Recommended_Fertilizer' → 'Fertilizer'")

    # Drop unnecessary
    drop_cols = ['Sustainability_Note']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Separate features and target
    y = df['Fertilizer']
    X = df.drop(columns=['Fertilizer'])

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    print(f"Found {len(categorical_cols)} categorical features: {categorical_cols}")
    print(f"Found {len(numeric_cols)} numeric features: {numeric_cols}")

    # Encode columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Encode target column
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    # Scale numeric
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"Data preprocessing complete! Training size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, target_encoder, label_encoders, X.columns.tolist()

if __name__ == "__main__":
    load_and_preprocess_data()
