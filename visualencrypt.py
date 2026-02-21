import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import numpy as np

def get_numeric_columns(df):
    """Get only the columns that contain purely numeric data"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def plot_correlation_matrix(data, title_prefix, timestamp, sample_size=10000):
    """Plot correlation matrix with sampling for large datasets"""
    if len(data) > sample_size:
        data_sample = data.sample(n=sample_size, random_state=42)
    else:
        data_sample = data
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(data_sample.corr(), annot=False, cmap='coolwarm')  # Removed annotations for speed
    plt.title(f'{title_prefix} Data Correlation Matrix')
    plt.xticks(rotation=90)  # Changed to 90 degrees for better readability
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'plots/{title_prefix.lower()}_correlation_{timestamp}.png', dpi=100)
    plt.close()

def plot_distributions(original_data, encrypted_data, column, timestamp, bins=50, sample_size=10000):
    """Plot distributions with sampling for large datasets"""
    if len(original_data) > sample_size:
        original_sample = original_data.sample(n=sample_size, random_state=42)
        encrypted_sample = encrypted_data.sample(n=sample_size, random_state=42)
    else:
        original_sample = original_data
        encrypted_sample = encrypted_data
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(data=original_sample[column], bins=bins, ax=ax1)
    ax1.set_title(f'Original {column} Distribution')
    
    sns.histplot(data=encrypted_sample[column], bins=bins, ax=ax2)
    ax2.set_title(f'Encrypted {column} Distribution')
    
    plt.tight_layout()
    plt.savefig(f'plots/distribution_{column.replace("/", "_")}_{timestamp}.png', dpi=100)
    plt.close()

def plot_scatter(original_data, encrypted_data, column, timestamp, sample_size=10000):
    """Plot scatter with sampling for large datasets"""
    if len(original_data) > sample_size:
        step = len(original_data) // sample_size
        original_sample = original_data.iloc[::step]
        encrypted_sample = encrypted_data.iloc[::step]
    else:
        original_sample = original_data
        encrypted_sample = encrypted_data
    
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(original_sample)), original_sample[column], 
               alpha=0.3, label='Original Data', color='blue', s=0.5)
    plt.scatter(range(len(encrypted_sample)), encrypted_sample[column], 
               alpha=0.3, label='Encrypted Data', color='red', s=0.5)
    plt.title(f'Original vs Encrypted: {column}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend(loc='upper right', markerscale=4)
    plt.tight_layout()
    plt.savefig(f'plots/scatter_{column.replace("/", "_")}_{timestamp}.png', dpi=100)
    plt.close()

def plot_encrypted_comparisons(original_df, encrypted_df):
    """Create visualization comparisons between original and encrypted data"""
    os.makedirs('plots', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    numeric_columns = get_numeric_columns(original_df)
    print(f"Processing numeric columns: {len(numeric_columns)}")
    
    # Create correlation matrices
    for data, title_prefix in [(original_df[numeric_columns], 'Original'), 
                              (encrypted_df[numeric_columns], 'Encrypted')]:
        plot_correlation_matrix(data, title_prefix, timestamp)
    
    # Process only the first 20 columns for detailed plots to avoid overwhelming
    for column in numeric_columns[:20]:
        try:
            plot_distributions(original_df, encrypted_df, column, timestamp)
            plot_scatter(original_df, encrypted_df, column, timestamp)
        except Exception as e:
            print(f"Error processing column {column}: {e}")
            continue

def print_data_summary(original_df, encrypted_df):
    """Print summary statistics of the data"""
    numeric_columns = get_numeric_columns(original_df)
    
    print(f"\nTotal columns: {len(original_df.columns)}")
    print(f"Numeric columns: {len(numeric_columns)}")
    
    # Calculate statistics for all columns at once
    original_stats = original_df[numeric_columns].agg(['mean', 'std', 'min', 'max'])
    encrypted_stats = encrypted_df[numeric_columns].agg(['mean', 'std', 'min', 'max'])
    
    # Print stats for first 10 columns only
    for col in numeric_columns[:10]:
        print(f"\nColumn: {col}")
        print("Original Data:")
        print(f"  Mean: {original_stats.loc['mean', col]:.2f}")
        print(f"  Std: {original_stats.loc['std', col]:.2f}")
        print(f"  Min: {original_stats.loc['min', col]:.2f}")
        print(f"  Max: {original_stats.loc['max', col]:.2f}")
        print("Encrypted Data:")
        print(f"  Mean: {encrypted_stats.loc['mean', col]:.2f}")
        print(f"  Std: {encrypted_stats.loc['std', col]:.2f}")
        print(f"  Min: {encrypted_stats.loc['min', col]:.2f}")
        print(f"  Max: {encrypted_stats.loc['max', col]:.2f}")

def main():
    original_df = pd.read_csv("D:/sem1_project/Cyberattack_on_smartGrid/intermediate_combined_data.csv")
    encrypted_df = pd.read_csv("encrypted_data/homomorphic_encrypted_20250117_105524.csv")
    
    print_data_summary(original_df, encrypted_df)
    plot_encrypted_comparisons(original_df, encrypted_df)
    print("\nVisualizations have been created in the 'plots' directory")

if __name__ == "__main__":
    main()