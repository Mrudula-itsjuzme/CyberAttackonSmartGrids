import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
import joblib
import gc
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    print("🗋 Loading data efficiently...")
    # Read data in chunks to manage memory
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=10000):
        chunks.append(chunk)
    df = pd.concat(chunks)
    print(f"Initial shape: {df.shape}")
    
    # Free memory
    del chunks
    gc.collect()
    
    # Detect target column
    target_column = next(col for col in df.columns if any(x in col.lower() for x in ['attack', 'label', 'class']))
    
    # Memory-efficient preprocessing
    print("🔄 Preprocessing data...")
    
    # Convert datatypes to save memory
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    del df
    gc.collect()
    
    # Basic preprocessing
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # Encode categorical variables
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column].astype(str))
    
    # Encode target
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Free memory
    del X, y
    gc.collect()
    
    return X_train, X_test, y_train, y_test, label_encoder_y, label_encoders, scaler

def select_features(X_train, y_train, X_test):
    print("🔄 Selecting features...")
    selector = LGBMClassifier(n_estimators=100, random_state=42)
    selector.fit(X_train, y_train)
    
    # Select features based on importance
    importance_threshold = np.percentile(selector.feature_importances_, 50)  # Top 50% features
    selected_features = X_train.columns[selector.feature_importances_ > importance_threshold].tolist()
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Free memory
    del selector
    gc.collect()
    
    print(f"Selected {len(selected_features)} features")
    return X_train_selected, X_test_selected, selected_features

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, label_encoder_y, model_name):
    print(f"\n🚀 Training {model_name}...")
    model.fit(X_train, y_train)
    
    print(f"📈 Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder_y.classes_))
    
    return accuracy, model

def save_model(model, model_name, accuracy, selected_features, scaler, label_encoders, label_encoder_y):
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    print(f"\n💾 Model saved: {model_filename}.joblib")

def main():
    filepath = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\FRESH\processed_dataset.csv"
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, label_encoder_y, label_encoders, scaler = \
        load_and_prepare_data(filepath)
    
    # Select features
    X_train_selected, X_test_selected, selected_features = select_features(X_train, y_train, X_test)
    
    # Free memory
    del X_train, X_test
    gc.collect()
    
    # Define models with optimized parameters
    models = {
        'CatBoost': CatBoostClassifier(
            iterations=1000,
            depth=8,
            learning_rate=0.01,
            l2_leaf_reg=3,
            verbose=False,
            thread_count=-1,
            random_state=42
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.01,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=20,
            n_jobs=-1,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            tree_method='hist',
            n_jobs=-1,
            random_state=42
        )
    }
    
    # Train and evaluate models
    best_accuracy = 0
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        accuracy, trained_model = train_and_evaluate_model(
            model, X_train_selected, X_test_selected, y_train, y_test, 
            label_encoder_y, name
        )
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = trained_model
            best_model_name = name
        
        # Free memory after each model
        if name != best_model_name:
            del model
            gc.collect()
    
    print(f"\n✨ Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Save best model
    save_model(
        best_model, best_model_name, best_accuracy,
        selected_features, scaler, label_encoders, label_encoder_y
    )
    
    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()