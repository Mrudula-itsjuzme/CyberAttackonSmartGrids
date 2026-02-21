import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
import os
import json

# Sample automated pipeline for Hyper Automation and XAI integration
def load_and_preprocess_data(file_path):
    """Load and preprocess dataset."""
    print("[INFO] Loading dataset...")
    df = pd.read_csv(file_path)

    # Example preprocessing: handling missing values, normalizing
    print("[INFO] Handling missing values...")
    df.fillna(df.mean(), inplace=True)

    print("[INFO] Normalizing features...")
    scaler = StandardScaler()
    feature_cols = df.columns[:-1]  # Assuming last column is the target
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df

# Automated model training and anomaly detection
def train_model(df):
    """Train a RandomForest model and evaluate it."""
    print("[INFO] Splitting dataset...")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model, X_test

# XAI - Explainable AI Integration
def explain_model(model, X_test):
    """Use SHAP for explainability."""
    print("[INFO] Generating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    print("[INFO] Visualizing feature importance...")
    shap.summary_plot(shap_values, X_test)
    return shap_values

# Automation for Real-Time Detection and Logging
def real_time_detection(model, input_data, log_file="alerts_log.json"):
    """Simulate real-time detection and log alerts."""
    print("[INFO] Real-time detection initiated...")

    # Simulated real-time data processing
    for idx, sample in input_data.iterrows():
        prediction = model.predict([sample.values])[0]
        if prediction == 1:  # Assuming '1' indicates an anomaly
            alert = {
                "id": idx,
                "prediction": int(prediction),
                "details": sample.to_dict()
            }
            print(f"[ALERT] Anomaly detected: {alert}")

            # Log alert to a file
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    logs = json.load(f)
            else:
                logs = []

            logs.append(alert)
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=4)

# Main Execution
if __name__ == "__main__":
    # Load and preprocess data
    dataset_path = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"  # Replace with your dataset path
    df = load_and_preprocess_data(dataset_path)

    # Train model
    model, X_test = train_model(df)

    # Explain model decisions
    shap_values = explain_model(model, X_test)

    # Simulate real-time detection
    print("[INFO] Simulating real-time detection...")
    real_time_sample = df.iloc[:10, :-1]  # Simulate the first 10 rows as real-time data
    real_time_detection(model, real_time_sample)
