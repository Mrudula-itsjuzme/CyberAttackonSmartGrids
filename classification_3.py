import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import gc
import warnings
import os
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

def setup_logging():
    """Setup logging directory and return log filename."""
    log_dir = 'ml_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(log_dir, f'ml_run_{timestamp}.log')

def log_message(message, log_file):
    """Log message to file and print to console."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}\n"
    print(log_msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_msg)

def optimize_dtypes(df):
    """Optimize memory usage by converting datatypes."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def handle_problematic_values(df, numeric_columns):
    """Handle problematic values in numeric columns."""
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    df[numeric_columns] = df[numeric_columns].clip(lower=-1e6, upper=1e6)
    return df

def load_and_prepare_data(filepath, log_file, target_column='Label', chunk_size=50000):
    """Efficiently process dataset in chunks and avoid memory overflow."""
    try:
        log_message("📝 Loading dataset in chunks...", log_file)
        
        feature_columns, categorical_columns, numeric_columns = None, None, None
        label_encoder_y, label_encoders = None, {}
        
        def process_chunk(chunk):
            nonlocal feature_columns, categorical_columns, numeric_columns, label_encoder_y
            
            if feature_columns is None:
                feature_columns = [col for col in chunk.columns if col not in [target_column, 'timestamp', 'src_ip', 'dst_ip', 'mac_address']]
                categorical_columns = chunk[feature_columns].select_dtypes(include=['object']).columns
                numeric_columns = [col for col in feature_columns if col not in categorical_columns]
                label_encoder_y = LabelEncoder()
                label_encoder_y.fit(chunk[target_column].dropna())
                log_message(f"Feature columns: {feature_columns}", log_file)
                log_message(f"Categorical columns: {list(categorical_columns)}", log_file)
                log_message(f"Numeric columns: {list(numeric_columns)}", log_file)
            
            for col in categorical_columns:
                chunk[col] = chunk[col].astype(str)  # Ensure conversion to string
                if col not in label_encoders:
                    label_encoders[col] = LabelEncoder().fit(chunk[col].dropna())
                chunk[col] = chunk[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else np.nan)
            
            # Handle unseen labels safely
            chunk[target_column] = chunk[target_column].map(lambda x: label_encoder_y.transform([x])[0] if x in label_encoder_y.classes_ else np.nan)
            
            chunk = handle_problematic_values(chunk, numeric_columns)
            return chunk[feature_columns], chunk[target_column].dropna()
        
        X_list, y_list = [], []
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, dtype=str, low_memory=False):
            X_chunk, y_chunk = process_chunk(chunk)
            X_list.append(X_chunk)
            y_list.append(y_chunk)
            del chunk  # Free memory
            gc.collect()
        
        X = pd.concat(X_list, axis=0)
        y = pd.concat(y_list, axis=0)
        
        scaler = StandardScaler()
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
        
        log_message("Splitting dataset...", log_file)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        return X_train, X_test, y_train, y_test, label_encoder_y, class_weight_dict
    except Exception as e:
        log_message(f"❌ Error in data preparation: {str(e)}", log_file)
        raise

def train_decision_tree(X_train, y_train, class_weight_dict, log_file):
    """Train Decision Tree with class weights and debug info."""
    log_message("🌲 Training Decision Tree...", log_file)
    dt_model = DecisionTreeClassifier(
        max_depth=20,
        class_weight=class_weight_dict,
        random_state=42,
    )
    dt_model.fit(X_train, y_train)
    log_message("✅ Decision Tree training complete!", log_file)
    return dt_model

def evaluate_model(model, X_test, y_test, label_encoder_y, log_file):
    """Evaluate model and log results with more debug details."""
    log_message("📈 Evaluating Decision Tree...", log_file)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    log_message(f"Accuracy: {accuracy:.4f}", log_file)
    log_message("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)), log_file)
    log_message("Classification Report:\n" + classification_report(y_test, y_pred, target_names=label_encoder_y.classes_), log_file)
    
    misclassified_indices = np.where(y_test != y_pred)[0]
    log_message(f"Total Misclassified Samples: {len(misclassified_indices)}", log_file)
    return accuracy

def main():
    log_file = setup_logging()
    filepath = r"G:\\Sem1\\Cyberattack_on_smartGrid\\intermediate_combined_data.csv"
    try:
        X_train, X_test, y_train, y_test, label_encoder_y, class_weight_dict = load_and_prepare_data(filepath, log_file)
        dt_model = train_decision_tree(X_train, y_train, class_weight_dict, log_file)
        evaluate_model(dt_model, X_test, y_test, label_encoder_y, log_file)
    except Exception as e:
        log_message(f"❌ Error: {str(e)}", log_file)

if __name__ == "__main__":
    main()