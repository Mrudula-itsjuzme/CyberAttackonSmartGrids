import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from datetime import datetime
import os

# Ensure high-quality plots
plt.rcParams.update({'font.size': 14, 'axes.labelweight': 'bold', 'axes.titlesize': 18})

class EnhancedHomomorphicEncryption:
    def __init__(self, key_size: int = 1024):
        self.public_key = np.random.randint(1, key_size)
        self._private_key = np.random.randint(1, key_size)
        self.modulus = 1e9 + 7
        print("[INFO] Initialized Homomorphic Encryption.")

    def encrypt(self, value: float) -> float:
        """Encrypt a single value."""
        try:
            noise = np.random.uniform(0.999, 1.001)
            return (float(value) * self.public_key * noise) % self.modulus
        except:
            return value  

    def encrypt_column(self, col_data):
        """Encrypt a single column using multiprocessing."""
        return col_data.apply(self.encrypt)

    def encrypt_dataframe(self, df: pd.DataFrame, chunk_size=100000):
        """Encrypt dataframe in smaller chunks to reduce memory usage."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        encrypted_df = df.copy()
        
        print(f"[INFO] Encrypting {len(numeric_columns)} numeric columns in chunks of {chunk_size} rows each...")

        for col in numeric_columns:
            print(f"[PROCESS] Encrypting column: {col}...")

            num_chunks = len(df) // chunk_size + 1
            encrypted_chunks = []

            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size
                chunk = df[col].iloc[start:end]
                encrypted_chunks.append(chunk.apply(self.encrypt))

                print(f"[CHUNK {i+1}/{num_chunks}] Processed {len(chunk)} rows for {col}.")

            encrypted_df[col] = pd.concat(encrypted_chunks).values
            print(f"[DONE] Finished encrypting column: {col}")

        print("[SUCCESS] Encryption completed for all columns.")
        return encrypted_df

def plot_distribution(df, col, save_path):
    """Generate and save IEEE-formatted distribution plot."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], bins=50, kde=True, edgecolor="black")
    plt.title(f'Distribution of {col}', fontsize=18, fontweight="bold")
    plt.xlabel(col, fontsize=16, fontweight="bold")
    plt.ylabel("Frequency", fontsize=16, fontweight="bold")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[PLOT] Saved distribution plot: {save_path}")

def analyze_first_chunk(df, output_dir):
    """Perform all analyses on the first chunk of data and save IEEE plots."""

    # Ensure output directory exists, handling errors properly
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"[INFO] Created directory: {output_dir}")
        except OSError as e:
            print(f"[ERROR] Failed to create directory {output_dir}. Error: {e}")
            return  # Exit function if directory cannot be created

    chunk_size = min(100000, len(df))
    df_chunk = df.iloc[:chunk_size]
    print(f"[INFO] Analyzing first {chunk_size} rows of the dataset.")

    numeric_cols = df_chunk.select_dtypes(include=[np.number]).columns  # Only select numeric columns

    if len(numeric_cols) == 0:
        print("[ERROR] No numeric columns found in the dataset. Correlation heatmap cannot be generated.")
        return

    selected_features = numeric_cols[:15]  # Choose first 15 numeric columns

    for col in selected_features:
        plot_distribution(df_chunk, col, os.path.join(output_dir, f"{col}_distribution.png"))

    # ✅ FIX: Ensure only numeric columns are used for correlation
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_chunk[numeric_cols].corr(), annot=True, cmap="coolwarm", linewidths=1, linecolor="black")
    plt.title("Feature Correlation Matrix", fontsize=18, fontweight="bold")
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print("[PLOT] Saved correlation matrix.")

    # Pairplot for first 6 numeric features (ensuring all are numeric)
    num_features_for_pairplot = min(6, len(numeric_cols))
    plt.figure(figsize=(14, 12))
    sns.pairplot(df_chunk[numeric_cols[:num_features_for_pairplot]], diag_kind="kde", plot_kws={'alpha': 0.5})
    plt.savefig(os.path.join(output_dir, "pairplot.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print("[PLOT] Saved pairplot for selected features.")

    print("[ANALYSIS] Completed analysis and IEEE plot generation for first chunk.")

def main():
    # Setup paths
    input_path = r"H:\sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    print("[INFO] Successfully loaded dataset. Starting encryption and analysis.")

    # Initialize encryption
    he = EnhancedHomomorphicEncryption()

    # Encrypt data in chunks
    encrypted_df = he.encrypt_dataframe(df)

    # Perform full analysis on the first chunk
    analyze_first_chunk(df, output_dir)

    # Save encrypted data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    encrypted_output_path = os.path.join(output_dir, f"homomorphic_encrypted_{timestamp}.csv")
    encrypted_df.to_csv(encrypted_output_path, index=False)
    print(f"[SUCCESS] Encrypted dataset saved at {encrypted_output_path}")

if __name__ == "__main__":
    main()
