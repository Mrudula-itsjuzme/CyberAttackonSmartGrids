import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')

# ------------------- Optimization: Read Data in Chunks -------------------
def load_data(filepath, chunk_size=50000):
    """Load large CSV in chunks and reduce memory usage."""
    print("\n📂 Loading dataset in chunks to reduce memory usage...")
    chunk_list = []
    total_rows = 0

    for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
        total_rows += len(chunk)
        
        # Convert float64 → float32 to reduce memory
        for col in chunk.select_dtypes(include=['float64']).columns:
            chunk[col] = chunk[col].astype('float32')
        
        chunk_list.append(chunk)

    df = pd.concat(chunk_list, ignore_index=True)
    print(f"✅ Dataset loaded: {total_rows} rows and {df.shape[1]} columns")
    print(f"🔻 Reduced Memory Usage: {df.memory_usage(deep=True).sum() / (1024**3):.2f} GB")
    
    return df

# ------------------- Feature Selection Optimization -------------------
def analyze_top_10_features(df, sample_size=500000):
    """Efficiently analyze top 10 features by variance with reduced memory."""
    print("\n📊 ANALYZING TOP 10 FEATURES BY VARIANCE")

    # Limit to numerical columns and downsample data
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

    # Use nlargest(10) instead of sorting all variances
    top_10_features = df_sample[numerical_cols].var().nlargest(10)
    
    print("\n🏆 Top 10 features by variance:")
    for i, (feature, variance) in enumerate(top_10_features.items(), 1):
        print(f"{i}. {feature}: {variance:.2e}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    bar_plot = sns.barplot(x=top_10_features.index, y=top_10_features.values, palette="coolwarm")
    plt.xticks(rotation=30)
    plt.title('Top 10 Features by Variance')
    plt.ylabel('Variance')
    plt.xlabel('Features')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on bars
    for p in bar_plot.patches:
        plt.text(p.get_x() + p.get_width()/2, p.get_height() + 0.0005, f"{p.get_height():.2e}",
                 ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return top_10_features.index[:3]  # Return only top 3 for detailed analysis

# ------------------- Correlation Matrix Optimization -------------------
def plot_correlation_matrix(df, top_features, sample_size=500000):
    """Optimized correlation matrix visualization."""
    print("\n🔗 Computing Correlation Matrix...")

    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    correlation_matrix = df_sample[top_features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix - Top Features')
    plt.tight_layout()
    plt.show()

# ------------------- Main Execution -------------------
def main(filepath):
    """Main execution function with memory optimizations."""
    start_time = time.time()

    # Load dataset efficiently
    df = load_data(filepath)

    # Step 1: Identify Top Features by Variance
    top_3_features = analyze_top_10_features(df)

    # Step 2: Compute Correlation Matrix for Top Features
    plot_correlation_matrix(df, top_3_features)

    end_time = time.time() - start_time
    print(f"\n⏱ Total Execution Time: {end_time:.2f} seconds")

# ------------------- Run Script -------------------
if __name__ == "__main__":
    filepath = "H:\sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
    main(filepath)

    print("\n✅ Complete!") 

    # Save the plot as a PDF file
    plt.savefig('correlation_matrix.pdf', format='pdf', bbox_inches='tight')    # Save as PDF file
    plt.show()  # Display the plot

    print("\n✅ Complete!")  # Confirmation message
# ------------------- End of Script -------------------