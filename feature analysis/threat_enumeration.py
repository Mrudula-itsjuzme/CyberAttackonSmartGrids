import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import networkx as nx
import gc
import os
import logging
from tqdm import tqdm  # Import tqdm for progress bar

class IEC104SecurityAnalyzer:
    def __init__(self, file_path, output_dir='analysis_outputs'):
        self.file_path = file_path
        self.output_dir = output_dir
        self.df = None
        self.anomaly_scores = None
        self.preprocessed_data = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Initialized analyzer. Output directory: {output_dir}")
        
    def load_data(self, required_columns=None):
        """Enhanced data loader with memory optimization"""
        self.logger.info(f"Loading data from: {self.file_path}")
        try:
            self.logger.info("Reading CSV file...")
            self.df = pd.read_csv(self.file_path, usecols=required_columns)
            self.logger.info(f"Successfully loaded {len(self.df)} rows and {len(self.df.columns)} columns")
            
            self.logger.info("Starting data preprocessing...")
            self._preprocess_data()
            return self
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _preprocess_data(self):
        """Preprocess the data for analysis"""
        try:
            self.logger.info("Identifying numeric columns...")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.logger.info(f"Found {len(numeric_cols)} numeric columns")
            
            self.logger.info("Handling infinities and NaN values...")
            with tqdm(total=len(numeric_cols), desc="Handling NaN and infinities") as pbar:
                for col in numeric_cols:
                    self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                    pbar.update(1)
            
            self.logger.info("Scaling features...")
            scaler = StandardScaler()
            self.preprocessed_data = scaler.fit_transform(self.df[numeric_cols])
            self.logger.info("Preprocessing complete")
            return self
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def detect_anomalies(self, contamination=0.1):
        """Anomaly detection using Isolation Forest"""
        try:
            self.logger.info("Starting anomaly detection...")
            
            self.logger.info("Running Isolation Forest...")
            with tqdm(total=1, desc="Isolation Forest") as pbar:
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                iso_scores = iso_forest.fit_predict(self.preprocessed_data)
                pbar.update(1)
            self.logger.info(f"Found {sum(iso_scores == -1)} anomalies with Isolation Forest")
            
            self.anomaly_scores = {
                'isolation_forest': iso_scores
            }
            self.logger.info("Anomaly detection complete")
            return self
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            raise
    
    def generate_threat_visualization(self):
        """Generate advanced threat visualizations"""
        try:
            self.logger.info("Generating parallel coordinates plot...")
            gc.collect()  # Free memory before visualizations
            with tqdm(total=1, desc="Parallel Coordinates") as pbar:
                # Sample data for visualization
                sample_data = self.preprocessed_data[np.random.choice(self.preprocessed_data.shape[0], 10000, replace=False)]
                fig = go.Figure(data=
                    go.Parcoords(
                        line=dict(color=self.anomaly_scores['isolation_forest'][:10000],
                                 colorscale='Viridis'),
                        dimensions=[
                            dict(range=[sample_data[:, i].min(), 
                                      sample_data[:, i].max()],
                                 label=f'Feature {i}',
                                 values=sample_data[:, i])
                            for i in range(min(10, sample_data.shape[1]))  # Limit to 10 features
                        ]
                    )
                )
                fig.write_html(f"{self.output_dir}/parallel_coordinates.html")
                pbar.update(1)
            self.logger.info("Parallel coordinates plot saved")

            self.logger.info("Generating t-SNE visualization...")
            with tqdm(total=1, desc="t-SNE Visualization") as pbar:
                tsne = TSNE(n_components=2, random_state=42)
                tsne_results = tsne.fit_transform(sample_data)
                pbar.update(1)

            plt.figure(figsize=(10, 8))
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                       c=self.anomaly_scores['isolation_forest'][:10000], cmap='viridis')
            plt.title('t-SNE visualization of anomalies')
            plt.savefig(f"{self.output_dir}/tsne_anomalies.png")
            plt.close()
            self.logger.info("t-SNE visualization saved")

        except Exception as e:
            self.logger.error(f"Error in threat visualization: {str(e)}")
            raise

    def create_temporal_graph(self):
        """Create temporal graph analysis"""
        try:
            self.logger.info("Creating temporal graph...")
            G = nx.Graph()
            
            edge_count = 0
            for i in range(len(self.df) - 1):
                if self.anomaly_scores['isolation_forest'][i] == -1:
                    G.add_edge(f"T{i}", f"T{i+1}", weight=1)
                    edge_count += 1
            
            self.logger.info(f"Created graph with {len(G.nodes)} nodes and {edge_count} edges")
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, node_color='lightblue', 
                    with_labels=True, node_size=500, alpha=0.5)
            plt.savefig(f"{self.output_dir}/temporal_graph.png")
            plt.close()
            self.logger.info("Temporal graph saved")
            
        except Exception as e:
            self.logger.error(f"Error in temporal graph creation: {str(e)}")
            raise

def main():
    try:
        file_path = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
        print(f"Starting analysis on file: {file_path}")
        
        analyzer = IEC104SecurityAnalyzer(file_path)
        print("Analyzer initialized")
        
        print("Loading and preprocessing data...")
        analyzer.load_data()
        
        print("Running anomaly detection...")
        analyzer.detect_anomalies()
        
        print("Generating visualizations...")
        analyzer.generate_threat_visualization()
        
        print("Creating temporal graph...")
        analyzer.create_temporal_graph()
        
        print("Analysis complete! Check the output directory for results.")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        logging.error(f"Fatal error in main execution: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
