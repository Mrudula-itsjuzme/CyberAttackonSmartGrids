# Script 5: Visualization and Reporting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class Visualization:
    def __init__(self, data, anomalies):
        self.data = data
        self.anomalies = anomalies

    def generate_pca_plot(self, output_path="pca_plot.png"):
        print("Generating PCA plot...")
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.data)
        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=self.anomalies, cmap="coolwarm", s=5)
        plt.title("PCA Visualization of Anomalies")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(label="Anomaly (-1) / Normal (1)")
        plt.savefig(output_path)
        plt.close()
        print(f"PCA plot saved to {output_path}")

    def generate_tsne_plot(self, output_path="tsne_plot.png"):
        print("Generating t-SNE plot...")
        tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
        reduced_data = []
        for chunk in tqdm(np.array_split(self.data, 10), desc="Running t-SNE in chunks"):
            reduced_data.append(tsne.fit_transform(chunk))
        reduced_data = np.vstack(reduced_data)
        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=self.anomalies, cmap="coolwarm", s=5)
        plt.title("t-SNE Visualization of Anomalies")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.colorbar(label="Anomaly (-1) / Normal (1)")
        plt.savefig(output_path)
        plt.close()
        print(f"t-SNE plot saved to {output_path}")

    def generate_report(self, pca_path="pca_plot.png", tsne_path="tsne_plot.png"):
        print("Generating summary report...")
        with open("anomaly_report.txt", "w") as report:
            report.write("Anomaly Detection Report\n")
            report.write("========================\n\n")
            report.write(f"PCA Plot: {pca_path}\n")
            report.write(f"t-SNE Plot: {tsne_path}\n")
            report.write("\nAnomaly Counts:\n")
            unique, counts = np.unique(self.anomalies, return_counts=True)
            for label, count in zip(unique, counts):
                report.write(f"  {label}: {count}\n")
        print("Report generated: anomaly_report.txt")

if __name__ == "__main__":
    print("Running Script 5: Visualization and Reporting")

    # Load your dataset
    file_path = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
    print(f"Loading dataset from: {file_path}")
    data = pd.read_csv(file_path)

    # Preprocessing: Handle missing values and problematic data
    print("Preprocessing dataset...")
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    # Replace infinities with NaN
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Fill NaN values with column means
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    # Standardizing the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])

    # Simulate anomalies (replace with actual anomaly labels if available)
    anomalies = np.random.choice([-1, 1], size=scaled_data.shape[0], p=[0.3, 0.7])

    # Perform visualization
    visualizer = Visualization(scaled_data, anomalies)
    visualizer.generate_pca_plot(output_path="pca_plot.png")
    visualizer.generate_tsne_plot(output_path="tsne_plot.png")
    visualizer.generate_report(pca_path="pca_plot.png", tsne_path="tsne_plot.png")

    print("Script 5 completed: Visualization and Reporting")
