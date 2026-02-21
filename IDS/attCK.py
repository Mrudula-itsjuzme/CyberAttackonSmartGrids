import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Paths
dataset_path = r"H:\sem1_project\Cyberattack_on_smartGrid\IDS\attacked_dataset.csv"
model_dir = r"H:\sem1_project\Cyberattack_on_smartGrid\Models"

print("Loading attacked dataset...")
df = pd.read_csv(dataset_path)
print(f"Dataset shape: {df.shape}")

# Target Column
target_col = next((col for col in df.columns if col.lower() in ['label', 'target', 'class', 'attack_type']), None)
if not target_col:
    raise ValueError("No target column found!")

print(f"Using target column: {target_col}")

# Fill & Encode
df[target_col] = df[target_col].astype(str)
df.fillna(df.mean(numeric_only=True), inplace=True)
df[target_col].fillna(df[target_col].mode()[0], inplace=True)
label_encoder = LabelEncoder()
df[target_col] = label_encoder.fit_transform(df[target_col])
print(f"Classes: {list(label_encoder.classes_)}")

# Drop non-numeric columns (like IPs, Flow ID etc.)
X = df.drop(columns=[target_col])
X = X.select_dtypes(include=[np.number])
y = df[target_col]

print(f"Training features shape: {X.shape}, Target shape: {y.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
}

best_model = None
best_score = 0
best_name = None

# Train
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    if acc > best_score:
        best_model = model
        best_score = acc
        best_name = name

# Save
if best_model:
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"best_model_{best_name}.joblib")
    joblib.dump(best_model, model_path)
    print(f"✅ Saved best model ({best_name}) at: {model_path}")
else:
    print("😭 No model saved.")

