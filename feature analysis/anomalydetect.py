# Script 4: Anomaly Detection and Comparison

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class AnomalyDetection:
    def __init__(self, data):
        self.data = data

    def detect_isolation_forest(self, contamination=0.1):
        print("Detecting anomalies using Isolation Forest...")
        model = IsolationForest(contamination=contamination, random_state=42)
        scores = model.fit_predict(self.data)
        print(f"Anomalies detected by Isolation Forest: {np.sum(scores == -1)}")
        return scores

    def detect_one_class_svm(self, nu=0.1, gamma=0.1):
        print("Detecting anomalies using One-Class SVM...")
        model = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
        scores = model.fit_predict(self.data)
        print(f"Anomalies detected by One-Class SVM: {np.sum(scores == -1)}")
        return scores

    def detect_dbscan(self, eps=0.5, min_samples=5):
        print("Detecting anomalies using DBSCAN...")
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(self.data)
        anomalies = (labels == -1)
        print(f"Anomalies detected by DBSCAN: {np.sum(anomalies)}")
        return anomalies

    def visualize_anomalies(self, scores, title="Anomaly Visualization"):
        print(f"Visualizing anomalies for {title}...")
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.data)
        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=(scores == -1), cmap="coolwarm", s=5)
        plt.title(title)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(label="Anomaly (-1) / Normal (1)")
        plt.show()

    def compare_anomaly_detection_methods(self):
        print("Comparing anomaly detection methods...")

        print("Running Isolation Forest...")
        isolation_forest_scores = self.detect_isolation_forest(contamination=0.1)

        print("Running One-Class SVM...")
        one_class_svm_scores = self.detect_one_class_svm(nu=0.1, gamma=0.1)

        print("Running DBSCAN...")
        dbscan_scores = self.detect_dbscan(eps=0.5, min_samples=5)

        # Visualize results
        self.visualize_anomalies(isolation_forest_scores, "Isolation Forest")
        self.visualize_anomalies(one_class_svm_scores, "One-Class SVM")
        self.visualize_anomalies(dbscan_scores, "DBSCAN")

        return {
            "Isolation Forest": isolation_forest_scores,
            "One-Class SVM": one_class_svm_scores,
            "DBSCAN": dbscan_scores
        }

if __name__ == "__main__":
    print("Running Script 4: Anomaly Detection and Comparison")

    # Simulated data
    data = np.random.rand(1000, 10)  # Example data
    df = pd.DataFrame(data, columns=[f"Feature_{i}" for i in range(10)])

    # Perform anomaly detection
    detector = AnomalyDetection(data)
    print("Running individual anomaly detection methods...")
    isolation_scores = detector.detect_isolation_forest()
    svm_scores = detector.detect_one_class_svm()
    dbscan_scores = detector.detect_dbscan()

    print("Running comparison of methods...")
    detection_results = detector.compare_anomaly_detection_methods()

    print("Script 4 completed: Anomaly Detection and Comparison")
