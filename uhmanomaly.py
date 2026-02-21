import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
import logging
import os
from datetime import datetime
import psutil
import gc
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedAnomalyDetector:
    def __init__(self, max_memory_mb=1000, contamination=0.1):
        self.max_memory_mb = max_memory_mb
        self.contamination = contamination
        self.scaler = RobustScaler()
        self.models = {
            'IsolationForest': IsolationForest(
                contamination=self.contamination,
                n_estimators=200,
                max_samples='auto',
                random_state=42,
                n_jobs=-1
            ),
            'LocalOutlierFactor': LocalOutlierFactor(
                contamination=self.contamination,
                n_neighbors=20,
                novelty=True,
                n_jobs=-1
            ),
            'RobustCovariance': EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            ),
            'OneClassSVM': OneClassSVM(
                kernel='rbf',
                nu=self.contamination
            ),
            'GaussianMixture': GaussianMixture(
                n_components=2,
                random_state=42
            )
        }
        self.feature_columns = None
        self.results = {}
        self.chunk_size = 10000  # Increased for better performance

    def preprocess_chunk(self, chunk, target_column):
        """Optimized preprocessing to avoid DataFrame fragmentation"""
        # Convert target to binary
        y = pd.Series(np.where(chunk[target_column].str.contains('Attack', case=False, na=False), 1, 0))
        
        # Drop target and get features
        X = chunk.drop(columns=[target_column])
        
        # Handle categorical columns efficiently
        cat_columns = X.select_dtypes(include=['object']).columns
        if not cat_columns.empty:
            # Create dummies all at once
            X = pd.get_dummies(X, columns=cat_columns)
        
        # First chunk: set feature columns
        if self.feature_columns is None:
            self.feature_columns = X.columns
            return X.values.astype(np.float32), y.values
        
        # Subsequent chunks: align columns
        current_cols = set(X.columns)
        expected_cols = set(self.feature_columns)
        
        # Handle missing columns efficiently
        if current_cols != expected_cols:
            # Create empty DataFrame with all expected columns
            aligned_df = pd.DataFrame(0, index=X.index, columns=self.feature_columns)
            
            # Fill in existing columns
            common_cols = current_cols.intersection(expected_cols)
            aligned_df[list(common_cols)] = X[list(common_cols)]
            
            X = aligned_df
        
        return X.values.astype(np.float32), y.values

    def fit(self, file_path, target_column):
        """Process data and train models with minimal memory usage"""
        # Initialize variables for chunked processing
        chunks = []
        total_size = 0
        chunk_count = 0
        
        print("Reading and preprocessing data in chunks...")
        
        # Read and preprocess in chunks
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
            X_chunk, y_chunk = self.preprocess_chunk(chunk, target_column)
            chunks.append((X_chunk, y_chunk))
            
            total_size += X_chunk.nbytes + y_chunk.nbytes
            chunk_count += 1
            
            if chunk_count % 5 == 0:
                print(f"Processed {chunk_count} chunks...")
        
        # Combine chunks efficiently
        print("\nCombining chunks and scaling data...")
        X = np.vstack([x for x, _ in chunks])
        y = np.concatenate([y for _, y in chunks])
        
        # Clear chunks to free memory
        chunks.clear()
        gc.collect()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print("\nTraining and evaluating models...")
        best_score = -1
        best_model = None
        
        # Train and evaluate each model
        for name, model in self.models.items():
            try:
                print(f"\nProcessing {name}...")
                model.fit(X_scaled)
                
                if name == 'GaussianMixture':
                    scores = -model.score_samples(X_scaled)
                    predictions = (scores > np.percentile(scores, (1 - self.contamination) * 100)).astype(int)
                else:
                    predictions = (model.predict(X_scaled) == -1).astype(int)
                
                accuracy = np.mean(predictions == y)
                f1 = f1_score(y, predictions)
                
                self.results[name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'conf_matrix': confusion_matrix(y, predictions),
                    'report': classification_report(y, predictions)
                }
                
                print(f"{name} Results:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"F1 Score: {f1:.4f}")
                print("\nClassification Report:")
                print(self.results[name]['report'])
                
                if f1 > best_score:
                    best_score = f1
                    best_model = name
                
            except Exception as e:
                print(f"Error with {name}: {str(e)}")
        
        print(f"\nBest performing model: {best_model}")
        print(f"Best F1 score: {best_score:.4f}")
        
        return self

def main():
    # Set file path and target column
    file_path = r"G:\Sem1\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
    target_column = "Label"  # Update this to match your target column name
    
    # Initialize and run detector
    detector = EnhancedAnomalyDetector(max_memory_mb=2000, contamination=0.1)
    detector.fit(file_path, target_column)

if __name__ == "__main__":
    main()