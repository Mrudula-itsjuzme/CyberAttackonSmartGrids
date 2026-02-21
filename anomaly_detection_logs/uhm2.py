from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibrationDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import shap
from sklearn.model_selection import learning_curve
import optuna
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from concurrent.futures import ThreadPoolExecutor

class EnhancedCybersecurityAnalyzer:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.output_dir = "output"  # Default output directory
        self.dirs = {
            'reports': os.path.join(self.output_dir, 'reports'),
            'model_explanations': os.path.join(self.output_dir, 'model_explanations'),
            'feature_analysis': os.path.join(self.output_dir, 'feature_analysis'),
            'saved_models': os.path.join(self.output_dir, 'saved_models'),
            'cross_validation': os.path.join(self.output_dir, 'cross_validation')
        }
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def print_and_log(self, message: str):
        """Print and log a message."""
        print(message)
        # Here, you can also add functionality to log messages to a file

    def load_and_analyze_dataset(self):
        """Load the dataset from the given path."""
        self.print_and_log(f"Loading dataset from {self.dataset_path}...")
        data = pd.read_csv(self.dataset_path, low_memory=False)
        self.print_and_log(f"Dataset loaded with shape: {data.shape}")
        return data

    def generate_data_profile(self, data: pd.DataFrame) -> None:
        """Generate a comprehensive data profile report with progress logging and parallel processing."""
        self.print_and_log("Generating data profile...")

        profile = {
            'basic_info': {
                'rows': len(data),
                'columns': len(data.columns),
                'duplicate_rows': data.duplicated().sum(),
                'missing_values': data.isnull().sum().to_dict()
            },
            'column_stats': {}
        }

        # Define a function for column profiling
        def process_column(column):
            col_data = data[column]
            stats = {
                'dtype': str(col_data.dtype),
                'unique_values': len(col_data.unique()),
                'missing_percentage': (col_data.isnull().sum() / len(col_data)) * 100
            }
            column_profile_path = os.path.join(self.dirs['feature_analysis'], f'{column}_profile.json')
            
            if pd.api.types.is_numeric_dtype(col_data):
                stats.update({
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis()
                })

                # Generate histogram
                plt.figure(figsize=(10, 6))
                sns.histplot(col_data, kde=True)
                plt.title(f'Distribution of {column}')
                plot_path = os.path.join(self.dirs['feature_analysis'], f'{column}_distribution.png')
                plt.savefig(plot_path)
                plt.close()
                stats['distribution_plot'] = plot_path

            else:
                value_counts = col_data.value_counts().head(10).to_dict()
                stats.update({'top_values': value_counts})

                # Generate bar plot for categorical variables
                plt.figure(figsize=(10, 6))
                sns.countplot(data=data, y=column, order=col_data.value_counts().index[:10])
                plt.title(f'Top 10 Categories in {column}')
                plot_path = os.path.join(self.dirs['feature_analysis'], f'{column}_categories.png')
                plt.savefig(plot_path)
                plt.close()
                stats['categories_plot'] = plot_path

            # Save stats for the column
            with open(column_profile_path, 'w') as f:
                json.dump(stats, f, indent=4)
            
            return column, stats

        # Parallelize column processing
        num_workers = min(8, os.cpu_count() or 4)  # Adjust number of workers as needed
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_column, column): column for column in data.columns}
            for i, future in enumerate(futures):
                column = futures[future]
                try:
                    column_name, column_stats = future.result()
                    profile['column_stats'][column_name] = column_stats
                    self.print_and_log(f"Processed column {column_name} ({i + 1}/{len(data.columns)})")
                except Exception as e:
                    self.print_and_log(f"Error processing column {column}: {str(e)}")

        # Generate correlation matrix for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plot_path = os.path.join(self.dirs['feature_analysis'], 'correlation_matrix.png')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            profile['correlation_matrix_plot'] = plot_path

        # Save overall profile report
        profile_path = os.path.join(self.dirs['reports'], 'data_profile.json')
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=4, default=str)
        self.print_and_log(f"Data profile saved to {profile_path}")

    # Placeholder for other methods (e.g., advanced_preprocessing, handle_imbalanced_data, etc.)
    # Define other methods as needed...

def main():
    dataset_path = r"D:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
    analyzer = EnhancedCybersecurityAnalyzer(dataset_path)
    
    # Load and analyze data
    data = analyzer.load_and_analyze_dataset()
    
    # Generate custom data profile
    analyzer.generate_data_profile(data)

if __name__ == "__main__":
    main()
