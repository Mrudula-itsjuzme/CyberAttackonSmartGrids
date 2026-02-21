import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def feature_analysis(file_path):
    # Load the processed dataset
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"📂 Loading file for feature analysis: {file_path}")
    df = pd.read_csv(file_path)

    # Identify columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns

    print("\n📊 Statistical Summary:")
    print("Numerical Features:")
    print(df[numerical_columns].describe())
    print("\nCategorical Features:")
    print(df[categorical_columns].describe())

    print("\n📈 Generating correlation heatmap...")
    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numerical_columns].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    print("✅ Correlation heatmap saved as 'correlation_heatmap.png'")

    print("\n📊 Plotting feature distributions...")
    # Feature Distributions
    for col in numerical_columns[:5]:  # Limit to first 5 numerical features for simplicity
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=30, color='blue')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"distribution_{col}.png")
        print(f"✅ Distribution plot saved for {col} as 'distribution_{col}.png'")

    print("\n📉 Performing PCA for visualization...")
    # PCA Analysis
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[numerical_columns])
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', data=df, alpha=0.6)
    plt.title("PCA Scatter Plot")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig("pca_scatter_plot.png")
    print("✅ PCA scatter plot saved as 'pca_scatter_plot.png'")

    print("\n✅ Feature analysis complete!")

if __name__ == "__main__":
    # Path to the processed dataset
    file_path = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\FRESH\processed_dataset.csv"

    # Perform feature analysis
    feature_analysis(file_path)
