import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc, roc_curve
import pickle
import os
import time

# Create a Marimo app

# Function to generate simulated firewall logs
def generate_firewall_logs():
    np.random.seed(42)
    ip_addresses = [f"192.168.1.{i}" for i in range(1, 20)] + [f"10.0.0.{i}" for i in range(1, 10)]
    actions = ["BLOCK", "ALLOW", "BLOCK", "BLOCK", "ALERT"]
    protocols = ["TCP", "UDP", "ICMP", "HTTP", "HTTPS"]
    ports = [22, 80, 443, 8080, 3389, 21]
    attack_types = ["Port Scan", "DoS Attempt", "Brute Force", "SQL Injection", "Normal Traffic"]
    logs_data = []
    for _ in range(100):
        logs_data.append({
            "timestamp": pd.Timestamp.now() - pd.Timedelta(minutes=np.random.randint(1, 60)),
            "source_ip": np.random.choice(ip_addresses),
            "destination_ip": "172.16.1.10",  
            "port": np.random.choice(ports),
            "protocol": np.random.choice(protocols),
            "action": np.random.choice(actions),
            "attack_type": np.random.choice(attack_types),
            "severity": np.random.choice(["Low", "Medium", "High", "Critical"], p=[0.5, 0.3, 0.15, 0.05])
        })
    logs_df = pd.DataFrame(logs_data)
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])
    logs_df = logs_df.sort_values("timestamp", ascending=False)
    return logs_df

def load_data():
    dataset_path = r"H:\sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
    df = pd.read_csv(dataset_path, low_memory=False)
    print("📥 Loading dataset from file...")
    return df
    print("📥 Loading dataset...")
    df = pd.DataFrame({
        'voltage': np.random.normal(220, 5, 1000),
        'current': np.random.normal(10, 1, 1000),
        'power': np.random.normal(2200, 100, 1000),
        'frequency': np.random.normal(50, 0.5, 1000),
        'phase': np.random.normal(0, 0.1, 1000),
    })
    print("✅ Dataset loaded successfully!")
    print(df.head())
    return df

class AnomalyDetection:
    def __init__(self, data):
        self.data = data

    def detect_isolation_forest(self, contamination=0.1):
        model = IsolationForest(contamination=contamination, random_state=42)
        scores = model.fit_predict(self.data)
        return scores

    def detect_one_class_svm(self, nu=0.1, gamma=0.1):
        model = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
        scores = model.fit_predict(self.data)
        return scores

    def detect_dbscan(self, eps=0.5, min_samples=5):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(self.data)
        return labels == -1

    def visualize_anomalies(self, scores, title="Anomaly Visualization"):
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.data)
        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=(scores == -1), cmap="coolwarm", s=5)
        plt.title(title)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(label="Anomaly (-1) / Normal (1)")
        print(plt)

def compare_anomaly_detection_methods(df_scaled):
    detector = AnomalyDetection(df_scaled)
    isolation_scores = detector.detect_isolation_forest(contamination=0.1)
    svm_scores = detector.detect_one_class_svm(nu=0.1, gamma=0.1)
    dbscan_scores = detector.detect_dbscan(eps=0.5, min_samples=5)
    detector.visualize_anomalies(isolation_scores, "Isolation Forest")
    detector.visualize_anomalies(svm_scores, "One-Class SVM")
    detector.visualize_anomalies(dbscan_scores, "DBSCAN")
    return {
        "Isolation Forest": isolation_scores,
        "One-Class SVM": svm_scores,
        "DBSCAN": dbscan_scores
    }

def generate_firewall_logs_and_display():
    logs_df = generate_firewall_logs()
    print(logs_df.head())
    return logs_df

if __name__ == "__main__":
    print("Running security analysis...")
    df = load_data()
    from ids1 import IEC104IntrusionDetector
    detector = IEC104IntrusionDetector()
    df = detector.preprocess_features(df)
    scaled_results = compare_anomaly_detection_methods(df)
    firewall_logs = generate_firewall_logs_and_display()
    print("Security analysis complete!")
