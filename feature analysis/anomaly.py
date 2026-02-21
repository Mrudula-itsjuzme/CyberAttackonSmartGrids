import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_and_prepare_anomaly_data(filepath):
    """
    Load and prepare data for anomaly detection.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV dataset
    
    Returns:
    --------
    X : numpy array
        Preprocessed features for anomaly detection
    df : pandas DataFrame
        Original dataframe for reference
    """
    print("📝 Loading dataset for anomaly detection...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"📝 Dataset loaded with shape: {df.shape}")
    
    # Separate features for anomaly detection
    # Exclude categorical and target columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    X = df.drop(columns=list(categorical_columns) + ['attack_type'])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"🔍 Prepared {X_scaled.shape[1]} features for anomaly detection")
    
    return X_scaled, df, scaler

def train_anomaly_detectors(X):
    """
    Train multiple anomaly detection models.
    
    Parameters:
    -----------
    X : numpy array
        Scaled feature matrix
    
    Returns:
    --------
    dict : Trained anomaly detection models
    """
    print("🕵️ Training Anomaly Detection Models...")
    
    # Isolation Forest
    iso_forest = IsolationForest(
        contamination=0.1,  # Expect 10% of data to be anomalous
        random_state=42,
        n_estimators=200,
        max_samples='auto',
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1
    )
    iso_forest.fit(X)
    
    # One-Class SVM
    one_class_svm = OneClassSVM(
        kernel='rbf',
        nu=0.1,  # Expected proportion of outliers
        gamma='scale'
    )
    one_class_svm.fit(X)
    
    return {
        'isolation_forest': iso_forest,
        'one_class_svm': one_class_svm
    }

def evaluate_anomaly_detection(models, X, df):
    """
    Evaluate and visualize anomaly detection results.
    
    Parameters:
    -----------
    models : dict
        Trained anomaly detection models
    X : numpy array
        Scaled feature matrix
    df : pandas DataFrame
        Original dataframe
    """
    print("🔬 Evaluating Anomaly Detection...")
    
    # Predictions from both models
    results = {}
    for name, model in models.items():
        # -1 indicates anomaly, 1 indicates normal
        predictions = model.predict(X)
        results[name] = predictions
        
        # Anomaly percentage
        anomaly_percentage = np.mean(predictions == -1) * 100
        print(f"{name.replace('_', ' ').title()}:")
        print(f"  📊 Anomaly Percentage: {anomaly_percentage:.2f}%")
    
    # Create a combined anomaly detection result
    combined_anomalies = (
        (results['isolation_forest'] == -1) & 
        (results['one_class_svm'] == -1)
    )
    combined_anomaly_percentage = np.mean(combined_anomalies) * 100
    print(f"\n🚨 Combined Anomaly Percentage: {combined_anomaly_percentage:.2f}%")
    
    # Visualization of anomalies
    plt.figure(figsize=(15, 6))
    
    # Isolation Forest Anomalies
    plt.subplot(1, 2, 1)
    sns.countplot(
        x='attack_type', 
        hue=results['isolation_forest'] == -1, 
        data=df,
        palette=['green', 'red']
    )
    plt.title('Isolation Forest: Anomalies by Attack Type')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Anomaly', labels=['Normal', 'Anomalous'])
    
    # One-Class SVM Anomalies
    plt.subplot(1, 2, 2)
    sns.countplot(
        x='attack_type', 
        hue=results['one_class_svm'] == -1, 
        data=df,
        palette=['green', 'red']
    )
    plt.title('One-Class SVM: Anomalies by Attack Type')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Anomaly', labels=['Normal', 'Anomalous'])
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_results.png')
    plt.close()
    
    # Save anomaly detection models and scaler
    joblib.dump(models['isolation_forest'], 'isolation_forest_anomaly_detector.joblib')
    joblib.dump(models['one_class_svm'], 'one_class_svm_anomaly_detector.joblib')
    
    return results

def main():
    # File path to your processed dataset
    filepath = r'D:\IEC 60870-5-104 SEG Intrusion Detection Dataset\processed_dataset.csv'
    
    try:
        # Load and prepare data
        X_scaled, df, scaler = load_and_prepare_anomaly_data(filepath)
        
        # Train anomaly detection models
        anomaly_models = train_anomaly_detectors(X_scaled)
        
        # Evaluate and visualize
        anomaly_results = evaluate_anomaly_detection(anomaly_models, X_scaled, df)
        
        print("\n✅ Anomaly Detection Complete!")
    
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()