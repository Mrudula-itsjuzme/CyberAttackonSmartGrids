import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from scipy.fftpack import fft
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import os
import logging
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

class ComprehensiveAnalysisPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f'analysis_output_{self.timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        log_file = self.output_dir / 'analysis.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        try:
            self.logger.info("Loading dataset.")
            df = pd.read_csv(self.data_path)
            self.logger.info(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def perform_statistical_analysis(self, df):
        try:
            self.logger.info("Performing statistical analysis.")
            numerical_cols = df.select_dtypes(include=[np.number]).columns

            # Basic statistics
            basic_stats = df[numerical_cols].describe()
            basic_stats.to_csv(self.output_dir / 'basic_stats.csv')

            # Advanced statistics
            advanced_stats = pd.DataFrame(index=numerical_cols)
            for col in numerical_cols:
                data = df[col].dropna()
                advanced_stats.loc[col, 'Mean'] = data.mean()
                advanced_stats.loc[col, 'Median'] = data.median()
                advanced_stats.loc[col, 'Std'] = data.std()
                advanced_stats.loc[col, 'Variance'] = data.var()
                advanced_stats.loc[col, 'Skewness'] = data.skew()
                advanced_stats.loc[col, 'Kurtosis'] = data.kurtosis()

            advanced_stats.to_csv(self.output_dir / 'advanced_stats.csv')

            # Normality tests
            normality_results = []
            for col in numerical_cols:
                data = df[col].dropna()
                if len(data) >= 3:
                    shapiro_stat, shapiro_p = stats.shapiro(data)
                    normality_results.append({
                        'Column': col,
                        'Shapiro_Stat': shapiro_stat,
                        'Shapiro_P': shapiro_p
                    })

            pd.DataFrame(normality_results).to_csv(
                self.output_dir / 'normality_tests.csv', index=False
            )
        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {e}")
            raise

    def create_visualizations(self, df):
        try:
            self.logger.info("Creating visualizations.")
            numerical_cols = df.select_dtypes(include=[np.number]).columns

            # Correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
            plt.title("Correlation Heatmap")
            plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300)
            plt.close()

            # Distribution plots
            for col in numerical_cols:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f"Distribution of {col}")
                plt.savefig(self.output_dir / f'distribution_{col}.png', dpi=300)
                plt.close()

            # PCA Analysis
            if len(numerical_cols) > 1:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[numerical_cols].dropna())
                pca = PCA()
                pca_result = pca.fit_transform(scaled_data)

                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                         np.cumsum(pca.explained_variance_ratio_), marker='o')
                plt.title("PCA Cumulative Explained Variance")
                plt.xlabel("Number of Components")
                plt.ylabel("Cumulative Variance")
                plt.grid()
                plt.savefig(self.output_dir / 'pca_variance.png', dpi=300)
                plt.close()

        except Exception as e:
            self.logger.error(f"Error in visualizations: {e}")
            raise

    def perform_time_and_frequency_analysis(self, df):
        try:
            time_col = 'time'  # Replace with the actual time column name if present
            numerical_cols = df.select_dtypes(include=[np.number]).columns

            if time_col in df.columns:
                self.logger.info("Performing time-domain analysis.")
                for col in numerical_cols:
                    plt.figure(figsize=(12, 6))
                    plt.plot(df[time_col], df[col])
                    plt.title(f"Time Domain Analysis of {col}")
                    plt.xlabel("Time")
                    plt.ylabel(col)
                    plt.savefig(self.output_dir / f'time_domain_{col}.png', dpi=300)
                    plt.close()

            self.logger.info("Performing frequency-domain analysis.")
            for col in numerical_cols:
                data = df[col].dropna()
                freqs, power = welch(data)

                plt.figure(figsize=(10, 6))
                plt.semilogy(freqs, power)
                plt.title(f"Frequency Domain Analysis of {col}")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Power Spectral Density")
                plt.savefig(self.output_dir / f'frequency_domain_{col}.png', dpi=300)
                plt.close()

        except Exception as e:
            self.logger.error(f"Error in time and frequency analysis: {e}")
            raise

    def analyze_outliers(self, df):
        try:
            self.logger.info("Analyzing outliers.")
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            outlier_results = []

            for col in numerical_cols:
                data = df[col].dropna()
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = data[(data < lower_bound) | (data > upper_bound)]
                outlier_results.append({
                    'Column': col,
                    'Outlier_Count': len(outliers),
                    'Outlier_Percentage': (len(outliers) / len(data)) * 100
                })

                plt.figure(figsize=(10, 6))
                sns.boxplot(data)
                plt.title(f"Outlier Analysis for {col}")
                plt.savefig(self.output_dir / f'outliers_{col}.png', dpi=300)
                plt.close()

            pd.DataFrame(outlier_results).to_csv(
                self.output_dir / 'outlier_analysis.csv', index=False
            )

        except Exception as e:
            self.logger.error(f"Error in outlier analysis: {e}")
            raise

    def run_pipeline(self):
        try:
            self.logger.info("Starting analysis pipeline.")
            df = self.load_data()
            self.perform_statistical_analysis(df)
            self.create_visualizations(df)
            self.perform_time_and_frequency_analysis(df)
            self.analyze_outliers(df)
            self.logger.info("Analysis pipeline completed successfully.")
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise

if __name__ == "__main__":
    data_path = r"D:\sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv" # Replace with your dataset path
    pipeline = ComprehensiveAnalysisPipeline(data_path)
    pipeline.run_pipeline()

