import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import gc  # For garbage collection

def load_and_process_csv(file_path, required_columns=None):
    """
    Load CSV in chunks with memory optimization.
    Only loads specified columns if provided.
    """
    try:
        # Get total rows first
        total_rows = sum(1 for _ in open(file_path)) - 1
        chunk_size = min(100000, total_rows)  # Adjust chunk size based on file size
        
        # Initialize empty list for chunks
        processed_chunks = []
        
        # Read CSV in chunks
        for chunk in pd.read_csv(
            file_path,
            usecols=required_columns,
            chunksize=chunk_size,
            low_memory=False
        ):
            # Only keep numeric columns and add timestamp
            numeric_cols = chunk.select_dtypes(include=[np.number]).columns
            chunk = chunk[numeric_cols]
            
            # Basic cleaning per chunk
            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk = chunk.fillna(chunk.mean())
            
            processed_chunks.append(chunk)
            
            # Force garbage collection
            gc.collect()
        
        # Combine all chunks
        df = pd.concat(processed_chunks, axis=0, ignore_index=True)
        del processed_chunks
        gc.collect()
        
        return df
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def add_timestamps(df):
    """Add synthetic timestamps to the dataframe."""
    try:
        timestamps = pd.date_range(
            start=datetime.now().replace(hour=0, minute=0, second=0),
            periods=len(df),
            freq='S'
        )
        df.insert(0, 'timestamp', timestamps)
        return df
    except Exception as e:
        print(f"Error adding timestamps: {e}")
        return df

def generate_plot(plot_func, df, filename, output_dir):
    """Wrapper function to safely generate and save plots."""
    try:
        plt.figure(figsize=(12, 8))
        plot_func(df)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        gc.collect()
    except Exception as e:
        print(f"Error generating {filename}: {e}")

def plot_correlation_heatmap(df):
    """Generate correlation heatmap for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')

def plot_single_feature_trend(df, feature):
    """Plot trend for a single feature."""
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df[feature])
        plt.title(f'Trend of {feature} Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature}_trend.png'))
        plt.close()
        gc.collect()
    except Exception as e:
        print(f"Error plotting trend for {feature}: {e}")

def plot_single_distribution(df, feature):
    """Plot distribution for a single feature."""
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature}_distribution.png'))
        plt.close()
        gc.collect()
    except Exception as e:
        print(f"Error plotting distribution for {feature}: {e}")

def main():
    # File paths
    file_path = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
    output_dir = 'analysis_outputs'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting analysis...")
    
    try:
        # Load and process data
        print("Loading dataset...")
        df = load_and_process_csv(file_path)
        
        if df is None or df.empty:
            print("Error: Failed to load dataset")
            return
            
        # Add timestamps
        print("Adding timestamps...")
        df = add_timestamps(df)
        
        # Generate correlation heatmap
        print("Generating correlation heatmap...")
        generate_plot(
            plot_correlation_heatmap,
            df,
            'correlation_heatmap.png',
            output_dir
        )
        
        # Plot trends for top features
        print("Plotting feature trends...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        top_features = df[numeric_cols].var().nlargest(5).index
        
        for feature in top_features:
            plot_single_feature_trend(df, feature)
        
        # Plot distributions
        print("Plotting feature distributions...")
        for feature in numeric_cols[:10]:  # First 10 numeric features
            plot_single_distribution(df, feature)
            
        print("Analysis complete. Outputs saved in 'analysis_outputs' directory.")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
    
    finally:
        # Clean up
        plt.close('all')
        gc.collect()

if __name__ == "__main__":
    output_dir = 'analysis_outputs'
    main()