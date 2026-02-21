import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
import joblib
import gc
import warnings
import os
from datetime import datetime
import logging
import sys

# Set up logging configuration
def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create file handler
    log_file = f"{log_dir}/training_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Set up logger
    logger = logging.getLogger('MLPipeline')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()
warnings.filterwarnings('ignore')

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
    logger.info(f"Current memory usage: {memory_usage:.2f} MB")

def load_and_prepare_data(filepath):
    logger.info("Starting data loading and preparation...")
    logger.info(f"Loading data from: {filepath}")
    
    # Read data in chunks
    chunks = []
    chunk_size = 5000
    total_rows = 0
    
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        # Remove duplicate columns early in the process
        chunk = chunk.loc[:, ~chunk.columns.duplicated()]
        chunks.append(chunk)
        total_rows += len(chunk)
        logger.info(f"Loaded {total_rows} rows...")
    
    df = pd.concat(chunks)
    logger.info(f"Initial shape: {df.shape}")
    
    del chunks
    gc.collect()
    log_memory_usage()
    
    # Detect target column
    target_column = next(col for col in df.columns if any(x in col.lower() for x in ['attack', 'label', 'class']))
    logger.info(f"Detected target column: {target_column}")
    
    # Check for and remove any remaining duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        logger.warning(f"Found duplicate columns: {duplicate_cols}")
        df = df.loc[:, ~df.columns.duplicated()]
    
    # Memory optimization
    logger.info("Optimizing memory usage...")
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            if df[col].min() >= -128 and df[col].max() <= 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() >= -32768 and df[col].max() <= 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
    
    # Handle class imbalance
    logger.info("Handling class imbalance...")
    class_counts = df[target_column].value_counts()
    logger.info(f"Class distribution before balancing:\n{class_counts}")
    
    min_class_size = class_counts.min()
    balanced_dfs = []
    
    for class_label in class_counts.index:
        class_data = df[df[target_column] == class_label]
        if len(class_data) > min_class_size * 3:
            class_data = class_data.sample(n=min_class_size * 3, random_state=42)
        balanced_dfs.append(class_data)
    
    df = pd.concat(balanced_dfs)
    logger.info(f"Balanced shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    del df
    gc.collect()
    
    # Preprocessing
    logger.info("Preprocessing features...")
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Handle missing values
    numeric_columns = X.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
    for col in numeric_columns:
        if X[col].isnull().sum() > 0:
            logger.info(f"Filling missing values in column: {col}")
            X[col] = X[col].fillna(X[col].median())
    
    # Encode categorical variables
    logger.info("Encoding categorical variables...")
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column].astype(str))
    
    # Encode target
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)
    
    # Scale features
    logger.info("Scaling features...")
    scaler = RobustScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Final check for duplicate columns
    if any(X.columns.duplicated()):
        logger.warning("Removing duplicate columns before splitting...")
        X = X.loc[:, ~X.columns.duplicated()]
    
    # Split data
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    del X, y
    gc.collect()
    log_memory_usage()
    
    return X_train, X_test, y_train, y_test, label_encoder_y, label_encoders, scaler

def select_features(X_train, y_train, X_test):
    logger.info("Starting feature selection...")
    
    # First, ensure no duplicate columns exist
    if any(X_train.columns.duplicated()):
        logger.warning("Found duplicate columns in training data. Removing duplicates...")
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        X_test = X_test.loc[:, ~X_test.columns.duplicated()]
    
    # Log all column names to help with debugging
    logger.info(f"Total features before selection: {len(X_train.columns)}")
    logger.debug(f"Feature names: {X_train.columns.tolist()}")
    
    try:
        # Try LightGBM feature selection first
        selector = LGBMClassifier(
            n_estimators=200,
            importance_type='gain',
            random_state=42
        )
        selector.fit(X_train, y_train)
        
        importance_threshold = np.percentile(selector.feature_importances_, 25)
        selected_features = X_train.columns[selector.feature_importances_ > importance_threshold].tolist()
        
    except Exception as e:
        logger.warning(f"LightGBM feature selection failed: {str(e)}")
        logger.info("Falling back to Random Forest feature selection...")
        
        # Fallback to Random Forest for feature selection
        rf_selector = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=42
        )
        rf_selector.fit(X_train, y_train)
        
        importance_threshold = np.percentile(rf_selector.feature_importances_, 25)
        selected_features = X_train.columns[rf_selector.feature_importances_ > importance_threshold].tolist()
        
        # Log feature importances from Random Forest
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 features selected by Random Forest:")
        logger.info(feature_importance.head(10))
    
    # Verify selected features exist in both datasets
    selected_features = [f for f in selected_features if f in X_train.columns and f in X_test.columns]
    
    # Ensure we have at least some minimum number of features
    if len(selected_features) < 10:
        logger.warning("Too few features selected. Including top 10 features by correlation with target...")
        correlations = []
        for column in X_train.columns:
            correlation = np.corrcoef(X_train[column], y_train)[0, 1]
            correlations.append((column, abs(correlation)))
        
        top_corr_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:10]
        additional_features = [f[0] for f in top_corr_features if f[0] not in selected_features]
        selected_features.extend(additional_features)
    
    # Final validation of selected features
    selected_features = list(set(selected_features))  # Remove any duplicates
    logger.info(f"Selected {len(selected_features)} features")
    
    # Create final feature sets
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Verify no duplicates in final selection
    assert not any(X_train_selected.columns.duplicated()), "Duplicate columns found in final training set"
    assert not any(X_test_selected.columns.duplicated()), "Duplicate columns found in final test set"
    
    # Log memory usage after feature selection
    log_memory_usage()
    
    return X_train_selected, X_test_selected, selected_features

def create_advanced_models():
    logger.info("Creating model configurations...")
    models = {
        'CatBoost': CatBoostClassifier(
            iterations=2500,
            depth=10,
            learning_rate=0.02,
            l2_leaf_reg=3,
            verbose=False,
            thread_count=-1,
            random_state=42
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=2500,
            max_depth=12,
            learning_rate=0.02,
            num_leaves=64,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=20,
            n_jobs=-1,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=2500,
            max_depth=10,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            tree_method='hist',
            n_jobs=-1,
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=1000,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=8,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=42
        ),
        'NeuralNet': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
    }
    return models

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, label_encoder_y, model_name):
    logger.info(f"\nTraining {model_name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=label_encoder_y.classes_))
    
    return accuracy, model

def save_model(model, model_name, accuracy, selected_features, scaler, label_encoders, label_encoder_y, timestamp):
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    model_filename = f"{save_dir}/{model_name}_{accuracy:.4f}_{timestamp}"
    
    model_data = {
        'model': model,
        'selected_features': selected_features,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'label_encoder_y': label_encoder_y,
        'accuracy': accuracy,
        'timestamp': timestamp
    }
    
    joblib.dump(model_data, f"{model_filename}.joblib")
    logger.info(f"Model saved: {model_filename}.joblib")

def create_stacking_ensemble(base_models):
    logger.info("Creating stacking ensemble...")
    estimators = [(name, model) for name, model in base_models.items()]
    
    level1 = StackingClassifier(
        estimators=estimators[:4],
        final_estimator=LGBMClassifier(n_estimators=1000),
        cv=5,
        n_jobs=-1
    )
    
    level2 = StackingClassifier(
        estimators=[
            ('stack1', level1),
            ('catboost', base_models['CatBoost']),
            ('lightgbm', base_models['LightGBM'])
        ],
        final_estimator=XGBClassifier(n_estimators=1000),
        cv=5,
        n_jobs=-1
    )
    
    return level2

def create_voting_ensemble(trained_models):
    logger.info("Creating voting ensemble...")
    return VotingClassifier(
        estimators=[
            (name, model) for name, model in trained_models.items()
            if name not in ['SVM', 'NeuralNet']
        ],
        voting='soft',
        n_jobs=-1
    )

def main():
    logger.info("Starting ML pipeline...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filepath = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\FRESH\processed_dataset.csv"
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, label_encoder_y, label_encoders, scaler = \
        load_and_prepare_data(filepath)
    
    # Select features
    X_train_selected, X_test_selected, selected_features = select_features(X_train, y_train, X_test)
    
    del X_train, X_test
    gc.collect()
    
    # Get all models
    models = create_advanced_models()
    
    # Train individual models
    trained_models = {}
    best_accuracy = 0
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        accuracy, trained_model = train_and_evaluate_model(
            model, X_train_selected, X_test_selected, y_train, y_test, 
            label_encoder_y, name
        )
        trained_models[name] = trained_model
        
        # Save each model
        save_model(
            trained_model, name, accuracy, selected_features,
            scaler, label_encoders, label_encoder_y, timestamp
        )
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = trained_model
            best_model_name = name
        
        # Free memory for large models
        if name in ['SVM', 'NeuralNet']:
            del trained_models[name]
            gc.collect()
    
    # Train and evaluate stacking ensemble
    logger.info("\nTraining Stacking Ensemble...")
    stacking_ensemble = create_stacking_ensemble(models)
    stacking_accuracy, stacking_model = train_and_evaluate_model(
        stacking_ensemble, X_train_selected, X_test_selected, y_train, y_test,
        label_encoder_y, "Stacking Ensemble"
    )
    
    save_model(
        stacking_model, "Stacking_Ensemble", stacking_accuracy,
        selected_features, scaler, label_encoders, label_encoder_y, timestamp
    )
    
    # Train and evaluate voting ensemble
    # Continuing the main() function...
    
    # Train and evaluate voting ensemble
    logger.info("\nTraining Voting Ensemble...")
    voting_ensemble = create_voting_ensemble(trained_models)
    voting_accuracy, voting_model = train_and_evaluate_model(
        voting_ensemble, X_train_selected, X_test_selected, y_train, y_test,
        label_encoder_y, "Voting Ensemble"
    )
    
    save_model(
        voting_model, "Voting_Ensemble", voting_accuracy,
        selected_features, scaler, label_encoders, label_encoder_y, timestamp
    )
    
    # Update best model if ensembles performed better
    if stacking_accuracy > best_accuracy:
        best_accuracy = stacking_accuracy
        best_model = stacking_model
        best_model_name = "Stacking Ensemble"
    
    if voting_accuracy > best_accuracy:
        best_accuracy = voting_accuracy
        best_model = voting_model
        best_model_name = "Voting Ensemble"
    
    # Save summary report
    logger.info("\n=== Final Results ===")
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")
    
    # Save summary to file
    summary_dir = 'summaries'
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = f"{summary_dir}/training_summary_{timestamp}.txt"
    
    with open(summary_file, 'w') as f:
        f.write("=== Training Summary ===\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {filepath}\n")
        f.write(f"Number of features selected: {len(selected_features)}\n\n")
        
        f.write("Model Accuracies:\n")
        for name, model in trained_models.items():
            y_pred = model.predict(X_test_selected)
            acc = accuracy_score(y_test, y_pred)
            f.write(f"{name}: {acc:.4f}\n")
        
        f.write(f"\nStacking Ensemble: {stacking_accuracy:.4f}\n")
        f.write(f"Voting Ensemble: {voting_accuracy:.4f}\n\n")
        
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Best Accuracy: {best_accuracy:.4f}\n")
        
        f.write("\nFeature Importance Summary:\n")
        if hasattr(best_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': selected_features,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            f.write(importance.to_string())
    
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info("\n✅ Pipeline completed successfully!")
    
    # Final cleanup
    gc.collect()
    log_memory_usage()
    
    return best_model_name, best_accuracy

if __name__ == "__main__":
    try:
        # Add psutil import at the top of the file
        import psutil
        
        # Create directories
        for directory in ['logs', 'saved_models', 'summaries']:
            os.makedirs(directory, exist_ok=True)
        
        # Run pipeline
        best_model_name, best_accuracy = main()
        
        # Final success message
        logger.info(f"\n🎉 Training completed successfully!")
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best accuracy: {best_accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"❌ An error occurred: {str(e)}", exc_info=True)
        raise