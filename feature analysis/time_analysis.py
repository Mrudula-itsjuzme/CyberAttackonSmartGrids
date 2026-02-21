import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def analyze_time_patterns(file_path, output_dir='time_analysis_outputs'):
    """Analyze time-based patterns in network traffic"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Ensure timestamp column is properly formatted
    if 'timestamp' in df.columns:
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Activity Volume Over Time
    plt.figure(figsize=(15, 6))
    df.groupby('timestamp').size().plot()
    plt.title('Network Traffic Volume Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Number of Records')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'traffic_volume.png'))
    plt.close()
    
    # 2. Inter-arrival Time Analysis
    if 'Flow Duration' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Flow Duration'], bins=50)
        plt.title('Distribution of Flow Durations')
        plt.xlabel('Duration')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'flow_duration_dist.png'))
        plt.close()
    
    # 3. Peak Activity Detection
    activity_counts = df.groupby('timestamp').size()
    threshold = activity_counts.mean() + 2*activity_counts.std()
    peak_times = activity_counts[activity_counts > threshold]
    
    with open(os.path.join(output_dir, 'peak_activity_report.txt'), 'w') as f:
        f.write("Peak Activity Analysis\n")
        f.write("=====================\n")
        f.write(f"Average activity: {activity_counts.mean():.2f} records/timestamp\n")
        f.write(f"Activity threshold: {threshold:.2f} records/timestamp\n")
        f.write(f"Number of peak activity periods: {len(peak_times)}\n")
        f.write("\nTop 10 Peak Activity Timestamps:\n")
        for time, count in peak_times.nlargest(10).items():
            f.write(f"{time}: {count} records\n")

if __name__ == "__main__":
    file_path = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
    analyze_time_patterns(file_path)
    print("Time-based analysis complete. Check time_analysis_outputs directory.")