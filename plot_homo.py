import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def clean_and_normalize_data(df, features):
    """Clean and normalize data for selected features"""
    df_clean = df[features].apply(pd.to_numeric, errors='coerce')
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(df_clean.median())
    
    # Normalize data (Min-Max scaling)
    for col in df_clean.columns:
        min_val = df_clean[col].min()
        max_val = df_clean[col].max()
        if min_val != max_val:
            df_clean[col] = (df_clean[col] - min_val) / (max_val - min_val)
        else:
            df_clean[col] = 0  # Handle constant columns with no variance

    return df_clean

def create_no_overlap_plot(original_df_path, encrypted_df_path, features, samples=15, scaling_factor=5):
    """Generate a comparison plot with no overlap for selected features"""
    try:
        # Load data with specified number of samples
        orig_data = pd.read_csv(original_df_path, nrows=samples)
        enc_data = pd.read_csv(encrypted_df_path, nrows=samples)

        # Clean and normalize the selected features from both datasets
        orig_normalized = clean_and_normalize_data(orig_data, features)
        enc_normalized = clean_and_normalize_data(enc_data, features)

        # Calculate the means for each feature
        orig_means = orig_normalized.mean() * scaling_factor
        enc_means = enc_normalized.mean() * scaling_factor + 1.5  # Add offset to separate lines visually

        # Create the plot
        plt.figure(figsize=(15, 7))

        x = np.arange(len(features))

        # Plot original data (blue line)
        plt.plot(x, orig_means, 'o--', color='blue', label='Original', markersize=8, linewidth=2)

        # Plot transformed data (orange line)
        plt.plot(x, enc_means, 's-', color='orange', label='Transformed', markersize=8, linewidth=2)

        # Add labels, grid, and title
        plt.xticks(ticks=x, labels=features, rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Normalized Values')
        plt.title('Comparison: Original vs. Transformed Data (No Overlap)')
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.legend()

        # Save the plot
        plt.savefig('no_overlap_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Plot saved as 'no_overlap_comparison.png'")

    except Exception as e:
        print(f"Error during plotting: {e}")

def main():
    # Features to extract dynamically from files
    original_df_path = "D:/sem1_project/Cyberattack_on_smartGrid/intermediate_combined_data.csv"
    encrypted_df_path = "encrypted_data/homomorphic_encrypted_20250117_105524.csv"

    # Dynamically extract column names
    orig_data = pd.read_csv(original_df_path, nrows=1)
    enc_data = pd.read_csv(encrypted_df_path, nrows=1)

    # Use only 15 shared features
    features = list(set(orig_data.columns).intersection(set(enc_data.columns)))[:15]

    # Generate plot
    create_no_overlap_plot(original_df_path, encrypted_df_path, features)

if __name__ == "__main__":
    main()
