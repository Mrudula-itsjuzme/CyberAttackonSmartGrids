import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def load_and_preprocess(filepath, non_numeric_cols=['src_ip', 'dst_ip']):
    """Load and preprocess the dataset."""
    print("📝 Loading dataset...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"📊 Initial shape: {df.shape}")
    
    # Handle non-numeric columns
    ip_data = df[non_numeric_cols].copy()
    numeric_cols = df.drop(columns=non_numeric_cols).columns
    df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Clean data
    print("🧹 Cleaning data...")
    df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid_cols = df_numeric.columns[df_numeric.notna().any()].tolist()
    df_numeric = df_numeric[valid_cols]
    
    # Normalize
    df_normalized = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())
    df_final = pd.concat([ip_data, df_normalized], axis=1)
    
    return df_final

def train_isolation_forest(X, contamination=0.1):
    """Train Isolation Forest for anomaly detection."""
    print("🌲 Training Isolation Forest...")
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
    return model

def train_autoencoder(X, encoding_dim=16):
    """Train Autoencoder for anomaly detection."""
    print("🤖 Training Autoencoder...")
    input_dim = X.shape[1]
    inputs = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(inputs)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(inputs, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=10, batch_size=32, verbose=0)
    return autoencoder

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance."""
    print(f"📊 Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'{model_name}_confusion.png')
    plt.close()
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def save_models(models, filename_prefix="security_model"):
    """Save trained models."""
    print("💾 Saving models...")
    for name, model in models.items():
        joblib.dump(model, f'{filename_prefix}_{name}.joblib')

def main():
    filepath = r"E:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
    data = load_and_preprocess(filepath)
    
    # Train models
    models = {
        'isolation_forest': train_isolation_forest(data),
        'autoencoder': train_autoencoder(data)
    }
    
    # Save models
    save_models(models)
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()