import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Define dataset path
processed_data_path = r"H:\sem1_project\Cyberattack_on_smartGrid\Processed_Data\Processed_Dataset.csv"
model_path = r"H:\sem1_project\Cyberattack_on_smartGrid\Models\best_model_RF.pkl"

# Load preprocessed dataset
if os.path.exists(processed_data_path):
    df = pd.read_csv(processed_data_path)
    print("Loaded preprocessed dataset! Shape:", df.shape)
else:
    print("[!] Error: Processed dataset not found!")
    exit()

# Identify target column
possible_labels = ['Label', 'label', 'target', 'class', 'attack_type']
target_col = next((col for col in df.columns if col in possible_labels), None)

if not target_col:
    print("[!] No valid target column found! Fix the dataset.")
    exit()

print(f"[+] Using '{target_col}' as the target column.")

# Ensure no NaN in target column
df.dropna(subset=[target_col], inplace=True)

# Split dataset
print("[+] Splitting dataset into training and testing sets...")
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
print("[+] Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"[+] Model saved at {model_path}")
