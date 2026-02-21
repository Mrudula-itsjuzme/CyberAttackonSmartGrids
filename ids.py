import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import logging
from sklearn.preprocessing import label_binarize
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import xgboost as xgb
import joblib
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb

class IEC104IntrusionDetector:
    def __init__(self):
        self.dataset_path = "intermediate_combined_data.csv"
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None  # Store feature columns for consistency
        
        # Setup logging
        logging.basicConfig(
            filename=f'ids_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Create output directories
        self.output_dir = "ids_output"
        self.models_dir = os.path.join(self.output_dir, "models")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def analyze_dataset(self):
        """Analyze the dataset structure and content."""
        print("\n📊 Analyzing dataset structure...")
        
        # Read the first few rows to understand the structure
        df_sample = pd.read_csv(self.dataset_path, nrows=1000, low_memory=False)
        
        # Store feature columns for consistency
        self.feature_columns = df_sample.columns.tolist()
        
        # Analyze column types
        print("\nColumn types:")
        for col in self.feature_columns:
            print(f"{col}: {df_sample[col].dtype}")
        
        # Count missing values
        missing_counts = df_sample.isnull().sum()
        print("\nColumns with missing values:")
        print(missing_counts[missing_counts > 0])
        
        # Analyze unique values
        print("\nUnique values in each column:")
        for col in self.feature_columns:
            unique_count = df_sample[col].nunique()
            print(f"{col}: {unique_count} unique values")
            
            # Show sample values for columns with mixed types
            if df_sample[col].dtype == 'object':
                sample_values = df_sample[col].dropna().head(3).tolist()
                print(f"  Sample values: {sample_values}")
        
        return self.feature_columns

    def preprocess_features(self, df):
        """Preprocess features with consistent columns."""
        print("\n🔄 Preprocessing features...")
        
        # Ensure all expected columns are present
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in chunk: {missing_cols}")
        
        # Reorder columns to ensure consistency
        df = df[self.feature_columns]
        
        processed_columns = {}  # Use a dictionary to store columns before creating the DataFrame
        feature_stats = {}
        
        for col in self.feature_columns:
            try:
                series = df[col]
                
                # Log initial statistics
                feature_stats[col] = {
                    'original_type': series.dtype,
                    'missing_values': series.isnull().sum(),
                    'unique_values': series.nunique()
                }
                
                if series.dtype == 'object':
                    # For string columns, try label encoding
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    
                    # Handle missing values before encoding
                    series = series.fillna('missing')
                    
                    try:
                        encoded_values = self.label_encoders[col].fit_transform(series)
                        processed_columns[col] = encoded_values
                        feature_stats[col]['encoding'] = 'label_encoding'
                    except Exception as e:
                        print(f"Warning: Could not encode column {col}: {str(e)}")
                        # Use a default value instead of skipping
                        processed_columns[col] = 0
                        feature_stats[col]['encoding'] = 'failed'
                        
                else:
                    # For numeric columns, handle missing values and convert
                    numeric_values = pd.to_numeric(series, errors='coerce')
                    median_value = numeric_values.median()
                    numeric_values = numeric_values.replace([np.inf, -np.inf], np.nan)  # Handle infinities
                    numeric_values = numeric_values.clip(-1e6, 1e6)  # Clip large values
                    numeric_values = numeric_values.fillna(median_value)
                    processed_columns[col] = numeric_values
                    feature_stats[col]['encoding'] = 'numeric'
                
                # Log successful processing
                feature_stats[col]['processed'] = True
                
            except Exception as e:
                print(f"Error processing column {col}: {str(e)}")
                # Use a default value instead of skipping
                processed_columns[col] = 0
                feature_stats[col]['processed'] = False
                feature_stats[col]['error'] = str(e)
        
        # Combine all processed columns into a single DataFrame
        processed_data = pd.DataFrame(processed_columns)
        
        # Verify the processed data has all expected columns
        if set(processed_data.columns) != set(self.feature_columns):
            raise ValueError("Processed data columns don't match expected columns")
        
        return processed_data

    def load_data(self):
        """Load and preprocess the dataset with consistent features."""
        print("\n📝 Loading dataset...")
        
        # Analyze dataset structure and set feature columns
        self.analyze_dataset()
        
        # First pass: identify target column
        df_sample = pd.read_csv(self.dataset_path, nrows=1000, low_memory=False)
        
        # Look for binary columns
        potential_targets = []
        for col in self.feature_columns:
            unique_vals = df_sample[col].dropna().unique()
            if len(unique_vals) == 2:
                potential_targets.append(col)
                print(f"Potential target column found: {col}")
                print(f"Unique values: {unique_vals}")
        
        if not potential_targets:
            raise ValueError("No suitable binary target column found")
        
        target_column = potential_targets[0]
        print(f"\nSelected target column: {target_column}")
        
        # Remove target column from feature columns
        self.feature_columns.remove(target_column)
        
        # Process data in chunks
        chunk_size = 10000
        chunks = pd.read_csv(self.dataset_path, chunksize=chunk_size, low_memory=False)
        
        processed_chunks = []
        labels = []
        
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            try:
                print(f"\nProcessing chunk {i+1}")
                
                # Extract target variable
                y_chunk = chunk[target_column].copy()
                X_chunk = chunk[self.feature_columns]
                
                # Preprocess features
                processed_chunk = self.preprocess_features(X_chunk)
                
                if processed_chunk is not None and not processed_chunk.empty:
                    processed_chunks.append(processed_chunk.values)
                    labels.append(y_chunk.values)
                    
                    print(f"Chunk {i+1} processed successfully")
                    print(f"Chunk shape: {processed_chunk.shape}")
                
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
                continue
        
        if not processed_chunks:
            raise ValueError("No valid data remained after preprocessing")
        
        # Combine all chunks
        X = np.vstack(processed_chunks)
        y = np.concatenate(labels)
        
        print("\nFinal dataset statistics:")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # Ensure target is numeric
        if not np.issubdtype(y.dtype, np.number):
            y = LabelEncoder().fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, self.feature_columns

    # Rest of the class implementation remains the same...

    def train_models(self):
        """Train multiple models for high accuracy."""
        # Load data
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        # Handle missing values using mean imputation
        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Initialize models
        models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=12,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=12,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'decision_tree': DecisionTreeClassifier(
                max_depth=30,
                random_state=42
            )
        }

        # Create output directories
        self.output_dir = "ids1_output"
        self.models_dir = os.path.join(self.output_dir, "models")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        results = {}
        model_accuracies = []

        for name, model in tqdm(models.items(), desc="Training models"):
            print(f"\nTraining {name.upper()}...")
            logging.info(f"Training {name}...")

            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }
            model_accuracies.append((name.upper(), accuracy))

            # Save the model
            model_file_path = os.path.join(self.models_dir, f'{name}_model.pkl')
            joblib.dump(model, model_file_path)
            print(f"Model saved to: {model_file_path}")

            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name.upper()}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(self.plots_dir, f'{name}_confusion_matrix.png'))
            plt.close()

            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                plt.figure(figsize=(12, 6))
                sns.barplot(x='importance', y='feature', data=importance.head(15))
                plt.title(f'Top 15 Important Features - {name.upper()}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, f'{name}_feature_importance.png'))
                plt.close()

            logging.info(f"{name.upper()} Accuracy: {accuracy:.4f}")
            logging.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

        # Compare model accuracies
        model_accuracies.sort(key=lambda x: x[1], reverse=True)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=[x[0] for x in model_accuracies], y=[x[1] for x in model_accuracies])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xlabel('Model')
        plt.savefig(os.path.join(self.plots_dir, 'model_accuracy_comparison.png'))
        plt.close()

        self.models = models
        return results

if __name__ == "__main__":
    # Initialize and train the IDS
    ids = IEC104IntrusionDetector()
    results = ids.train_models()

    # Print final results
    print("\nModel Accuracies:")
    for model_name, result in results.items():
        print(f"{model_name.upper()}: {result['accuracy']:.4f}")
