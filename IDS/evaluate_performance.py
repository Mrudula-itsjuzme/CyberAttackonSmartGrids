import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

print("💥 Full Retrain with Label Encoding Activated! Let’s slay this IDS! 💅")

# === Paths ===
dataset_path = r"H:\sem1_project\Cyberattack_on_smartGrid\IDS\attacked_dataset.csv"
model_path = r"H:\sem1_project\Cyberattack_on_smartGrid\best_model_xgboost.joblib"
scaler_path = r"H:\sem1_project\Cyberattack_on_smartGrid\scaler_xgboost.joblib"
features_path = r"H:\sem1_project\Cyberattack_on_smartGrid\feature_names_xgboost.joblib"
label_encoder_path = r"H:\sem1_project\Cyberattack_on_smartGrid\label_encoder_xgboost.joblib"
eval_result_path = r"H:\sem1_project\Cyberattack_on_smartGrid\IDS\evaluation_results.txt"

# === Load dataset ===
df = pd.read_csv(dataset_path)
print(f"✅ Loaded dataset. Shape: {df.shape}")

# === Target column detection ===
possible_targets = ['Label', 'label', 'class', 'target', 'attack_type']
target_col = next((col for col in df.columns if col in possible_targets), None)
if not target_col:
    raise ValueError("💀 No valid target column found.")

# === Handle missing values ===
df[target_col] = df[target_col].fillna(df[target_col].mode()[0])

# === Encode the label column ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[target_col])
joblib.dump(label_encoder, label_encoder_path)

# === Encode non-numeric columns (IP, Flow ID, Protocol, etc.) ===
X = df.drop(columns=[target_col])
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

print(f"🎨 Categorical columns encoded. Final feature shape: {X.shape}")

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# === Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, scaler_path)

# === Train model ===
model = XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train_scaled, y_train)
joblib.dump(model, model_path)
joblib.dump(X.columns.tolist(), features_path)

# === Evaluate ===
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)
conf_matrix = confusion_matrix(y_test, y_pred)

with open(eval_result_path, "w") as f:
    f.write("🎯 Classification Report:\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(np.array2string(conf_matrix))

print(f"🎯 Model Accuracy: {accuracy:.4f}")
print("📊 Classification Report saved!")

# === Confusion Matrix Plot ===
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='magma',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()  # Show plot

