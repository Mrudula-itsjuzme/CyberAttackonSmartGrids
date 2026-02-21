import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
import time

# ------------------- Create Directory for Saving Plots -------------------
def create_output_dir(output_dir="IEEE_Plots"):
    """Ensure output directory exists."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# ------------------- Load Data in Chunks -------------------
def load_data_in_chunks(filepath, chunk_size=50000):
    """Load large CSV in chunks, process incrementally, and handle non-numeric values."""
    print("\n📂 Loading dataset in chunks...")

    chunk_list = []
    total_rows = 0
    numerical_cols = None

    # Read the dataset in chunks
    for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
        total_rows += len(chunk)

        # Drop completely empty columns
        chunk.dropna(axis=1, how='all', inplace=True)

        # Select only numeric columns, avoiding errors on string data
        chunk_numeric = chunk.select_dtypes(include=[np.number])

        # Convert float64 → float32 to reduce memory
        chunk_numeric = chunk_numeric.astype(np.float32)

        # Store numerical column names (only from first chunk)
        if numerical_cols is None:
            numerical_cols = chunk_numeric.columns.tolist()

        # Append processed chunk
        chunk_list.append(chunk_numeric)

    # Concatenate all chunks
    df = pd.concat(chunk_list, ignore_index=True)
    
    print(f"✅ Loaded {total_rows} rows with reduced memory.")
    print(f"🔻 Memory Usage: {df.memory_usage(deep=True).sum() / (1024**3):.2f} GB")
    
    return df

# ------------------- Feature Analysis -------------------
def analyze_top_features(df, top_n=3):
    """Find and analyze top features by variance."""
    print("\n📊 Finding Top Features by Variance...")

    # Ensure only numeric columns are considered
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    # Compute variance on numeric columns
    top_features = df[numerical_cols].var().nlargest(top_n).index
    print(f"🎯 Selected Top Features: {list(top_features)}")

    return top_features

# ------------------- IEEE Graph Generation (Separate Plots) -------------------
def generate_ieee_graphs(df, feature, output_dir):
    """Generate and save separate IEEE-compliant plots for a specific feature."""
    print(f"\n📊 Generating IEEE-Format Graphs for: {feature}...")

    # Standardize feature for better visualization
    scaler = StandardScaler()
    df['standardized_feature'] = scaler.fit_transform(df[[feature]])

    # === 1. Histogram ===
    plt.figure(figsize=(8, 6))
    sns.histplot(df, x=feature, bins=50, color='steelblue', edgecolor='black')
    plt.title(f'Histogram of {feature}', fontsize=14, fontweight='bold')
    plt.xlabel(f'{feature} Value', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{feature}_Histogram.png", dpi=600, bbox_inches='tight')
    print(f"💾 Saved: {feature}_Histogram.png")
    plt.close()

    # === 2. Q-Q Plot ===
    plt.figure(figsize=(8, 6))
    stats.probplot(df[feature], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {feature}', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{feature}_QQPlot.png", dpi=600, bbox_inches='tight')
    print(f"💾 Saved: {feature}_QQPlot.png")
    plt.close()

    # === 3. Box Plot ===
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[feature], color="steelblue")
    plt.title(f'Box Plot of {feature}', fontsize=14, fontweight='bold')
    plt.xlabel(f'{feature}', fontsize=12, fontweight='bold')  # Force x-axis label
    plt.xticks([0], [feature])  # Ensure x-axis is visible
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{feature}_BoxPlot.png", dpi=600, bbox_inches='tight')
    print(f"💾 Saved: {feature}_BoxPlot.png")
    plt.close()

    # === 5. Violin Plot ===
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=df[feature], color="steelblue")
    plt.title(f'Violin Plot of {feature}', fontsize=14, fontweight='bold')
    plt.xlabel(f'{feature}', fontsize=12, fontweight='bold')  # Force x-axis label
    plt.xticks([0], [feature])  # Ensure x-axis is visible
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{feature}_ViolinPlot.png", dpi=600, bbox_inches='tight')
    print(f"💾 Saved: {feature}_ViolinPlot.png")
    plt.close()

    # === 6. Scatter Plot ===
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df.index, y=df[feature], color="steelblue")
    plt.title(f'Scatter Plot of {feature}', fontsize=14, fontweight='bold')
    plt.xlabel('Index', fontsize=12, fontweight='bold')
    plt.ylabel(f'{feature} Value', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.7)    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{feature}_ScatterPlot.png", dpi=600, bbox_inches='tight')    
    print(f"💾 Saved: {feature}_ScatterPlot.png")
    plt.close()

    # === 4. Time Series Plot ===
    df_sorted = df.sort_index()
    plt.figure(figsize=(8, 6))
    plt.plot(df_sorted[feature], label='Feature Trend', alpha=0.7, color='blue')
    plt.title(f'Time Series Analysis of {feature}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{feature}_TimeSeries.png", dpi=600, bbox_inches='tight')
    print(f"💾 Saved: {feature}_TimeSeries.png")
    plt.close()

    # === 5. Feature Distribution ===
    plt.figure(figsize=(8, 6))  
    sns.histplot(df[feature], bins=50, kde=True, color='green')
    plt.title(f'Distribution of {feature}', fontsize=14, fontweight='bold')
    plt.xlabel(f'{feature} Value', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{feature}_Distribution.png", dpi=600, bbox_inches='tight')
    print(f"💾 Saved: {feature}_Distribution.png")

    # === 6. Standardized Feature Distribution ===
    plt.figure(figsize=(8, 6))
    sns.histplot(df['standardized_feature'], bins=50, kde=True, color='purple')
    plt.title(f'Standardized Distribution of {feature}', fontsize=14, fontweight='bold')
    plt.xlabel('Standardized Value', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{feature}_StandardizedDistribution.png", dpi=600, bbox_inches='tight')
    print(f"💾 Saved: {feature}_StandardizedDistribution.png")
    plt.close()

# ------------------- Main Execution -------------------
def main(filepath):
    """Load dataset in chunks, analyze top features, and generate IEEE graphs separately."""
    start_time = time.time()

    # Step 1: Create output directory
    output_dir = create_output_dir()

    # Step 2: Load data in chunks (without loading entire dataset in RAM)
    df = load_data_in_chunks(filepath)

    # Step 3: Identify Top Features
    top_features = analyze_top_features(df)

    # Step 4: Generate IEEE-style graphs separately for each feature
    for feature in top_features:
        generate_ieee_graphs(df, feature, output_dir)

    end_time = time.time() - start_time
    print(f"\n⏱ Total Execution Time: {end_time:.2f} seconds")

# ------------------- Run Script -------------------
if __name__ == "__main__":
    filepath = "H:\sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
    main(filepath)
