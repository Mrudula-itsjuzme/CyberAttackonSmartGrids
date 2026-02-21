import os
import pandas as pd
import numpy as np
import joblib
import random
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
print("Booting the Chaos Engine... 💣")
start_time = time.time()

# ------------ PATHS ------------
dataset_path = r"H:\sem1_project\Cyberattack_on_smartGrid\Processed_Data\Processed_Dataset.csv"
attacked_dataset_path = r"H:\sem1_project\Cyberattack_on_smartGrid\IDS\attacked_dataset.csv"
model_dir = r"H:\sem1_project\Cyberattack_on_smartGrid\Models"

# ------------ LOAD DATASET ------------
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

df = pd.read_csv(dataset_path)
print(f"Dataset loaded. Shape: {df.shape}")

# ------------ DETECT TARGET COLUMN ------------
target_col = next((col for col in df.columns if col.lower() in ['label', 'target', 'class', 'attack_type']), None)
if not target_col:
    raise ValueError("No target column found in dataset.")

print(f"Using '{target_col}' as the target.")

# ------------ CLEANING ------------
df[target_col] = df[target_col].astype(str)
df.fillna(df.mean(numeric_only=True), inplace=True)
df[target_col].fillna(df[target_col].mode()[0], inplace=True)

# Encode target labels
label_encoder = LabelEncoder()
df[target_col] = label_encoder.fit_transform(df[target_col])
print(f"Classes found: {list(label_encoder.classes_)}")
joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
print("Target labels encoded.")
# ------------ ATTACK SIMULATION ------------
def manipulate_timestamps(df):
   if 'timestamp' in df.columns:
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp']) + pd.to_timedelta(np.random.randint(-10, 10, df.shape[0]), unit='s')
    except:
        pass  # Or log it

def inject_noise(df, noise_level=0.05):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference([target_col])
    df[numeric_cols] += noise_level * np.random.normal(size=df[numeric_cols].shape)
    return df

def adversarial_corruption(df, corruption_rate=0.1):
    indices = random.sample(range(len(df)), int(len(df) * corruption_rate))
    for col in df.columns.difference([target_col]):
        try:
            df.loc[indices, col] = df[col].dropna().sample(len(indices), replace=True).values
        except Exception:
            continue
    return df

print("Injecting chaos into the dataset...")
df = manipulate_timestamps(df)
df = inject_noise(df)
df = adversarial_corruption(df)

# Save corrupted dataset
os.makedirs(os.path.dirname(attacked_dataset_path), exist_ok=True)
df.to_csv(attacked_dataset_path, index=False)
print(f"Attacked dataset saved to: {attacked_dataset_path}")

# ------------ MODEL TRAINING ------------
X = df.drop(columns=[target_col])
y = df[target_col]

# Drop non-numeric features if any slipped in
X = X.select_dtypes(include=[np.number])

# Sanity check
if X.shape[1] == 0:
    raise ValueError("No usable numeric features found for training.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
}

best_model = None
best_score = 0
best_name = None

print("Training models...\n")

for name, model in models.items():
    print(f"--- Training: {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    if acc > best_score:
        best_score = acc
        best_model = model
        best_name = name

# ------------ SAVE MODEL ------------
if best_model:
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"best_model_{best_name}.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n🎯 Best model ({best_name}) saved to: {model_path}")
else:
    print("No model performed well enough to save. 😭")

print(f"Done in {time.time() - start_time:.2f} seconds.")
