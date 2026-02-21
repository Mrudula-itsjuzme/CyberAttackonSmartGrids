# utils.py
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

def sanitize_filename(name):
    """
    Convert a string to a valid filename by removing or replacing invalid characters
    """
    # Replace problematic characters with underscores
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', name)
    # Replace spaces and slashes with underscores
    sanitized = sanitized.replace(' ', '_').replace('/', '_')
    return sanitized

def setup_logging():
    """Configure logging with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = "encryption_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/encryption_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_output_dirs():
    """Create necessary output directories"""
    dirs = ['plots', 'encrypted_data', 'encryption_logs']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    return dirs

def plot_comparison(original_data, encrypted_data, column_name, plot_type='scatter'):
    """Plot comparison between original and encrypted data"""
    plt.figure(figsize=(12, 6))
    
    if plot_type == 'scatter':
        plt.scatter(range(len(original_data)), original_data, 
                   alpha=0.5, label='Original Data', color='blue')
        plt.scatter(range(len(encrypted_data)), encrypted_data, 
                   alpha=0.5, label='Encrypted Data', color='red')
    else:
        plt.plot(original_data, label='Original Data', alpha=0.5)
        plt.plot(encrypted_data, label='Encrypted Data', alpha=0.5)
    
    plt.title(f'Comparison of Original vs Encrypted {column_name}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Sanitize the column name for the filename
    safe_column_name = sanitize_filename(column_name)
    
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/comparison_{safe_column_name}_{timestamp}.png')
    plt.close()

def load_and_prepare_data(path):
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv(path)
        logger = logging.getLogger(__name__)
        logger.info(f"Successfully loaded dataset with shape: {df.shape}")
        logger.info(f"Columns: {', '.join(df.columns)}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise