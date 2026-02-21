import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_traffic_patterns(file_path, output_dir='traffic_analysis_outputs'):
    """Analyze traffic patterns and packet characteristics"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # 1. Packet Size Analysis
    packet_columns = [col for col in df.columns if 'length' in col.lower()]
    for col in packet_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col].dropna(), bins=50)
        plt.title(f'Distribution of {col}')
        plt.xlabel('Size')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
        plt.close()
    
    # 2. Flow Statistics
    flow_stats = pd.DataFrame()
    if 'Flow Duration' in df.columns:
        flow_stats['Duration_Stats'] = df['Flow Duration'].describe()
    
    # Calculate packets per flow if available
    packet_count_cols = [col for col in df.columns if 'packets' in col.lower()]
    if packet_count_cols:
        flow_stats['Packets_Per_Flow'] = df[packet_count_cols].sum(axis=1).describe()
    
    # Save flow statistics
    flow_stats.to_csv(os.path.join(output_dir, 'flow_statistics.csv'))
    
    # 3. Protocol Analysis if available
    protocol_cols = [col for col in df.columns if 'protocol' in col.lower()]
    if protocol_cols:
        for col in protocol_cols:
            protocol_counts = df[col].value_counts()
            plt.figure(figsize=(10, 6))
            protocol_counts.plot(kind='bar')
            plt.title(f'{col} Distribution')
            plt.xlabel('Protocol')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
            plt.close()

if __name__ == "__main__":
    file_path = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
    analyze_traffic_patterns(file_path)
    print("Traffic pattern analysis complete. Check traffic_analysis_outputs directory.")