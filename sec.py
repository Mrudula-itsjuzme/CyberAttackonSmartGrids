import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset (update with actual file path)
df = pd.read_csv(r"H:\sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv")

# Data Preprocessing
scaler = StandardScaler()
X = df.drop(columns=["Label"])  # Assuming 'Label' is the target column
X_scaled = scaler.fit_transform(X)
y = df["Label"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Isolation Forest Anomaly Detection
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_forest.fit(X_train)
anomaly_scores = iso_forest.decision_function(X_test)
y_pred_anomaly = iso_forest.predict(X_test)

# Convert anomaly labels
y_pred_anomaly = np.where(y_pred_anomaly == -1, 1, 0)  # Convert -1 to 1 (anomaly), 1 to 0 (normal)

# Random Forest Classification
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
print("Anomaly Detection (Isolation Forest):")
print(classification_report(y_test, y_pred_anomaly))
print("\nClassification Model (Random Forest):")
print(classification_report(y_test, y_pred))

# PCA for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
df_pca["Label"] = y

# Interactive Plotly Visualization
fig = px.scatter(df_pca, x="PCA1", y="PCA2", color=df_pca["Label"].astype(str),
                 title="PCA Visualization of Attack Data")
fig.show()

# Heatmap of Feature Correlations
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()
