import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.tree import DecisionTreeClassifier
import joblib
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

class IEC104IntrusionDetector:
    def __init__(self):
        self.dataset_path = r"H:\sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
        logging.basicConfig(
            filename=f'enhanced_ids_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"enhanced_ids_output_{self.version}"
        self.models_dir = os.path.join(self.output_dir, "models")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def analyze_dataset(self):
        """Analyze the dataset structure and content."""
        print("\n📊 Analyzing dataset structure...")
        
        df_sample = pd.read_csv(self.dataset_path, nrows=1000, low_memory=False)
        self.feature_columns = df_sample.columns.tolist()
        
        print("\nColumn types:")
        for col in self.feature_columns:
            print(f"{col}: {df_sample[col].dtype}")
        
        missing_counts = df_sample.isnull().sum()
        print("\nColumns with missing values:")
        print(missing_counts[missing_counts > 0])
        
        print("\nUnique values in each column:")
        for col in self.feature_columns:
            unique_count = df_sample[col].nunique()
            print(f"{col}: {unique_count} unique values")
            
            if df_sample[col].dtype == 'object':
                sample_values = df_sample[col].dropna().head(3).tolist()
                print(f"  Sample values: {sample_values}")
        
        return self.feature_columns

    def preprocess_features(self, df):
        """Preprocess features with consistent columns."""
        print("\n🔄 Preprocessing features...")
        
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in chunk: {missing_cols}")
        
        df = df[self.feature_columns]
        processed_columns = {}
        feature_stats = {}
        
        for col in self.feature_columns:
            try:
                series = df[col]
                feature_stats[col] = {
                    'original_type': series.dtype,
                    'missing_values': series.isnull().sum(),
                    'unique_values': series.nunique()
                }
                
                if series.dtype == 'object':
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    
                    series = series.fillna('missing')
                    
                    try:
                        encoded_values = self.label_encoders[col].fit_transform(series)
                        processed_columns[col] = encoded_values
                        feature_stats[col]['encoding'] = 'label_encoding'
                    except Exception as e:
                        print(f"Warning: Could not encode column {col}: {str(e)}")
                        processed_columns[col] = 0
                        feature_stats[col]['encoding'] = 'failed'
                        
                else:
                    numeric_values = pd.to_numeric(series, errors='coerce')
                    median_value = numeric_values.median()
                    numeric_values = numeric_values.replace([np.inf, -np.inf], np.nan)
                    numeric_values = numeric_values.clip(-1e6, 1e6)
                    numeric_values = numeric_values.fillna(median_value)
                    processed_columns[col] = numeric_values
                    feature_stats[col]['encoding'] = 'numeric'
                
                feature_stats[col]['processed'] = True
                
            except Exception as e:
                print(f"Error processing column {col}: {str(e)}")
                processed_columns[col] = 0
                feature_stats[col]['processed'] = False
                feature_stats[col]['error'] = str(e)
        
        processed_data = pd.DataFrame(processed_columns)
        
        if set(processed_data.columns) != set(self.feature_columns):
            raise ValueError("Processed data columns don't match expected columns")
        
        return processed_data

    def load_data(self):
        """Load and preprocess the dataset with consistent features."""
        print("\n📝 Loading dataset...")
        
        # Analyze dataset structure
        self.analyze_dataset()
        
        # Identify target column
        df_sample = pd.read_csv(self.dataset_path, nrows=1000, low_memory=False)
        
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
        
        # Remove target from features
        self.feature_columns.remove(target_column)
        
        # Process data in chunks
        chunk_size = 10000
        chunks = pd.read_csv(self.dataset_path, chunksize=chunk_size, low_memory=False)
        
        processed_chunks = []
        labels = []
        
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            try:
                print(f"\nProcessing chunk {i+1}")
                
                y_chunk = chunk[target_column].copy()
                X_chunk = chunk[self.feature_columns]
                
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
        
        X = np.vstack(processed_chunks)
        y = np.concatenate(labels)
        
        print("\nFinal dataset statistics:")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        if not np.issubdtype(y.dtype, np.number):
            y = LabelEncoder().fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, self.feature_columns

    def create_advanced_models(self):
        """Create an enhanced set of models with optimized hyperparameters."""
        base_models = {
            'rf': RandomForestClassifier(
                n_estimators=300,
                max_depth=40,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                bootstrap=True
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=12,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method='hist'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=12,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbose=-1
            ),
            'catboost': cb.CatBoostClassifier(
                iterations=300,
                depth=10,
                learning_rate=0.05,
                random_seed=42,
                verbose=False
            ),
            'gbc': GradientBoostingClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.85,
                random_state=42
            ),
            'ada': AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                early_stopping=True,
                random_state=42
            )
        }

        # Create ensemble models
        estimators = [
            ('rf', base_models['rf']),
            ('xgb', base_models['xgb']),
            ('lgb', base_models['lightgbm'])
        ]

        # Voting Classifier
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )

        # Stacking Classifier
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                random_state=42
            )
        )

        base_models.update({
            'voting': voting_clf,
            'stacking': stacking_clf
        })

        return base_models

    def train_models(self):
        """Train multiple models with enhanced evaluation and visualization."""
        X_train, X_test, y_train, y_test, feature_names = self.load_data()

        imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        models = self.create_advanced_models()
        results = {}
        model_accuracies = []

        for name, model in tqdm(models.items(), desc="Training enhanced models"):
            print(f"\nTraining {name.upper()}...")
            logging.info(f"Training {name}...")

            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                logging.info(f"{name} CV Scores: {cv_scores}")
                logging.info(f"{name} CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'cv_scores': cv_scores,
                    'probabilities': y_pred_proba
                }
                model_accuracies.append((name.upper(), accuracy))

                model_file_path = os.path.join(self.models_dir, f'{name}_model.pkl')
                joblib.dump(model, model_file_path)

                self.generate_model_visualizations(
                    name, model, X_test, y_test, y_pred, y_pred_proba, feature_names
                )

                logging.info(f"{name.upper()} Accuracy: {accuracy:.4f}")
                logging.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

            except Exception as e:
                logging.error(f"Error training {name}: {str(e)}")
                continue

        self.generate_comparison_visualizations(model_accuracies, results)
        
        self.models = models
        return results

    def generate_model_visualizations(self, name, model, X_test, y_test, y_pred, y_pred_proba, feature_names):
        """Generate comprehensive visualizations for each model."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.plots_dir, f'{name}_confusion_matrix.png'))
        plt.close()

        if y_pred_proba is not None:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name.upper()}')
            plt.legend()
            plt.savefig(os.path.join(self.plots_dir, f'{name}_roc_curve.png'))
            plt.close()

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

    def generate_comparison_visualizations(self, model_accuracies, results):
        """Generate visualizations comparing all models."""
        # Model Accuracy Comparison
        plt.figure(figsize=(12, 6))
        model_accuracies.sort(key=lambda x: x[1], reverse=True)
        sns.barplot(x=[x[0] for x in model_accuracies], y=[x[1] for x in model_accuracies])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'model_accuracy_comparison.png'))
        plt.close()

        # Cross-validation Score Comparison
        plt.figure(figsize=(12, 6))
        cv_data = []
        for name, result in results.items():
            if 'cv_scores' in result:
                cv_data.extend([(name.upper(), score) for score in result['cv_scores']])
        
        cv_df = pd.DataFrame(cv_data, columns=['Model', 'CV Score'])
        sns.boxplot(x='Model', y='CV Score', data=cv_df)
        plt.title('Cross-validation Score Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'cv_score_comparison.png'))
        plt.close()

if __name__ == "__main__":
    try:
        print("Starting IEC104 Intrusion Detection System...")
        ids = IEC104IntrusionDetector()
        results = ids.train_models()

        print("\nFinal Model Accuracies:")
        for model_name, result in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Test Accuracy: {result['accuracy']:.4f}")
            print(f"  CV Mean Accuracy: {result['cv_scores'].mean():.4f}")
            print(f"  CV Std: {result['cv_scores'].std():.4f}")

        print("\nTraining complete! Check the output directory for detailed results and visualizations.")
        
    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Please check the log file for more details.")
        raise