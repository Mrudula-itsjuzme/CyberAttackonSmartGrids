import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import os
import re
import multiprocessing

def print_progress(message):
    print(f"[INFO] {message}")

class SmartGridAnalyzerIEEE:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        self.output_dir = "smart_grid_analysis_ieee"
        os.makedirs(self.output_dir, exist_ok=True)
        print_progress("Data loaded and output directory created.")

    def save_plot(self, plt, name):
        """Save plot with IEEE-style formatting."""
        safe_name = re.sub(r'[\/*?:"<>|]', "_", name)
        plot_path = os.path.join(self.output_dir, f"{safe_name}.png")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.xticks(fontsize=14, fontweight="bold")
        plt.yticks(fontsize=14, fontweight="bold")
        plt.xlabel(plt.gca().get_xlabel(), fontsize=16, fontweight="bold")
        plt.ylabel(plt.gca().get_ylabel(), fontsize=16, fontweight="bold")
        plt.title(plt.gca().get_title(), fontsize=18, fontweight="bold")
        for spine in plt.gca().spines.values():
            spine.set_linewidth(2)
            spine.set_color("black")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        print_progress(f"Saved IEEE plot: {safe_name}.png")

    def analyze_distributions(self):
        """Analyze and save distribution plots with enhanced readability."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        print_progress("Starting distribution analysis...")
        
        def process_column(col):
            print_progress(f"Processing distribution for {col}...")
            plt.figure(figsize=(12, 6))
            sns.histplot(data=self.data, x=col, bins=30, kde=True, edgecolor="black")
            plt.title(f'Distribution of {col}', fontsize=18, fontweight="bold")
            plt.xlabel(col, fontsize=16, fontweight="bold")
            plt.ylabel("Frequency", fontsize=16, fontweight="bold")
            self.save_plot(plt, f'dist_{col}')

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        pool.map(process_column, numeric_cols)
        pool.close()
        pool.join()
        print_progress("Feature distribution analysis completed.")

    def analyze_correlations(self):
        """Analyze and save correlation heatmap with IEEE-style formatting."""
        print_progress("Starting correlation analysis...")
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        plt.figure(figsize=(14, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=1, linecolor='black')
        plt.title('Feature Correlation Matrix', fontsize=18, fontweight="bold")
        self.save_plot(plt, 'correlation_matrix')
        print_progress("Correlation analysis completed.")

    def detect_anomalies(self):
        """Perform and save anomaly detection results with IEEE formatting."""
        print_progress("Starting anomaly detection...")
        numeric_data = self.data.select_dtypes(include=[np.number])
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        scores = iso_forest.fit_predict(numeric_data)
        plt.figure(figsize=(12, 6))
        plt.hist(scores, bins=50, edgecolor="black", alpha=0.75)
        plt.title('Anomaly Score Distribution', fontsize=18, fontweight="bold")
        plt.xlabel('Anomaly Score', fontsize=16, fontweight="bold")
        plt.ylabel('Frequency', fontsize=16, fontweight="bold")
        self.save_plot(plt, 'anomaly_scores')
        print_progress("Anomaly detection completed.")

    def analyze_all(self):
        """Run all analyses and save results."""
        print_progress("Starting full IEEE-compliant analysis...")
        self.analyze_distributions()
        self.analyze_correlations()
        self.detect_anomalies()
        print_progress("IEEE-Formatted Analysis complete! All results saved in:", self.output_dir)

if __name__ == "__main__":
    file_path = r"H:\sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"  # Update with your file path
    analyzer = SmartGridAnalyzerIEEE(file_path)
    analyzer.analyze_all()
