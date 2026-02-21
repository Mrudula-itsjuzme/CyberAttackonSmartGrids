import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from tqdm import tqdm

# PCA Visualization
def plot_pca(df, numerical_columns, output_file="pca_scatter_plot.png"):
    print("\n📉 Performing PCA for visualization...")
    pbar = tqdm(total=1, desc="PCA", leave=False)
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
    plt.savefig(output_file)
    print(f"✅ PCA scatter plot saved as '{output_file}'")
    pbar.update(1)
    pbar.close()

# t-SNE with Multiprocessing
def tsne_batch_processing(data, perplexity=30, n_iter=1000, init='pca', learning_rate='auto'):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, init=init, learning_rate=learning_rate, random_state=42)
    return tsne.fit_transform(data)

def parallel_tsne(df, numerical_columns, sample_size=5000, n_jobs=-1, batch_size=500, output_file="tsne_scatter_plot.png"):
    print("\n📉 Performing t-SNE with multiprocessing...")

    # Sample the data
    if len(df) > sample_size:
        print(f"🔍 Sampling {sample_size} points from the dataset for t-SNE...")
        df_sampled = df.sample(n=sample_size, random_state=42)
    else:
        df_sampled = df

    # Ensure numerical columns are clean
    df_sampled[numerical_columns] = df_sampled[numerical_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    data = df_sampled[numerical_columns].values

    # Divide data into batches
    num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    data_batches = [data[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

    # Process batches in parallel
    print(f"🚀 Processing {num_batches} batches with {n_jobs if n_jobs > 0 else 'all available'} cores...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(tsne_batch_processing)(batch, perplexity=30) for batch in tqdm(data_batches, desc="t-SNE Batches")
    )

    # Combine results
    tsne_result = np.vstack(results)
    df_sampled['tSNE1'] = tsne_result[:, 0]
    df_sampled['tSNE2'] = tsne_result[:, 1]

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='tSNE1', y='tSNE2', data=df_sampled, alpha=0.6)
    plt.title("t-SNE Scatter Plot (Parallelized)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"✅ t-SNE scatter plot saved as '{output_file}'")

# Class Distribution (Imbalance Check)
def class_distribution_analysis(df):
    print("\n📊 Analyzing class distribution...")
    target_columns = [col for col in df.columns if col.lower() in ['attack_type', 'label', 'target']]

    if not target_columns:
        print("❌ No target column found! Expected 'attack_type', 'label', or 'target'.")
        return

    target_column = target_columns[0]
    print(f"Detected target column: {target_column}")

    pbar = tqdm(total=1, desc="Class Distribution", leave=False)
    class_counts = df[target_column].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    output_file = "class_distribution.png"
    plt.savefig(output_file)
    print(f"✅ Class distribution plot saved as '{output_file}'")
    pbar.update(1)
    pbar.close()
    return class_counts

if __name__ == "__main__":
    # Path to your processed dataset
    file_path = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\FRESH\processed_dataset.csv"

    # Load dataset
    df = pd.read_csv(file_path, low_memory=False)
    print(f"✅ Dataset loaded with shape: {df.shape}")

    # Identify columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns

    # Perform various feature analyses
    plot_pca(df, numerical_columns)
    parallel_tsne(df, numerical_columns)
    class_distribution_analysis(df)
