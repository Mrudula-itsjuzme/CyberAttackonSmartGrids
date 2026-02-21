import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler
import joblib
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def print_section(message, section_num, total_sections):
    print(f"\n{'='*80}")
    print(f"[{section_num}/{total_sections}] {message}")
    print('='*80)

def train_and_visualize_isolation_forest(dataset_path, model_save_path, scaler_save_path, plot_save_path):
    total_sections = 9
    start_time = time.time()
    
    # Section 1: Loading Dataset
    print_section("Loading dataset...", 1, total_sections)
    try:
        data = pd.read_csv(dataset_path, low_memory=False)
        print("Successfully loaded dataset")
        print(f"Dataset shape: {data.shape}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise
    
    # Section 2: Preprocessing Labels
    print_section("Preprocessing labels...", 2, total_sections)
    attack_keywords = ["attack", "malicious", "threat", "dos", "mitm", "breach", "intrusion", "exploit"]
    normal_keywords = ["normal", "benign", "legitimate"]
    
    def categorize_label(label):
        label_str = str(label).lower()
        return 1 if any(keyword in label_str for keyword in attack_keywords) else (
            0 if any(keyword in label_str for keyword in normal_keywords) else 1)
    
    data['Binary_Label'] = data['Label'].apply(categorize_label)
    label_dist = data['Binary_Label'].value_counts()
    print("Label distribution:")
    print(f"Normal (0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(data)*100:.2f}%)")
    print(f"Attack (1): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(data)*100:.2f}%)")
    
    # Section 3: Processing Features
    print_section("Processing features...", 3, total_sections)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    features = data[numeric_cols].drop(columns=['Binary_Label'], errors='ignore').astype('float32')
    print(f"Initial feature count: {features.shape[1]}")
    
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    for col in features.columns:
        features[col].fillna(features[col].median(), inplace=True)
    
    variance_threshold = features.var().mean() * 0.01
    features = features.loc[:, features.var() > variance_threshold]
    print(f"Remaining features: {features.shape[1]}")
    
    # Section 5: Scaling Features
    print_section("Scaling features...", 5, total_sections)
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    print("Scaling completed")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, data['Binary_Label'], test_size=0.2, random_state=42, stratify=data['Binary_Label']
    )
    
    # Section 6: Grid Search Setup
    print_section("Setting up grid search...", 6, total_sections)
    param_grid = {
        'n_estimators': [200, 300],
        'contamination': [0.1, 0.15],
        'max_samples': ['auto'],
        'max_features': [0.7, 1.0]
    }
    
    model = IsolationForest(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    # Section 7: Model Training
    print_section("Training model...", 7, total_sections)
    grid_search.fit(X_train)
    best_model = grid_search.best_estimator_
    
    # Save and Evaluate
    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    
    predictions = best_model.predict(X_test)
    anomalies = np.where(predictions == -1, 1, 0)
    accuracy = accuracy_score(y_test, anomalies)
    
    cm = confusion_matrix(y_test, anomalies)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.4f})")
    plt.savefig(plot_save_path)
    plt.close()
    
    return accuracy, grid_search.best_params_

if __name__ == "__main__":
    dataset_path = "F:\\mrudula college\\Sem1_project\\Cyberattack_on_smartGrid\\intermediate_combined_data.csv"
    model_path = "F:\\mrudula college\\Sem1_project\\Cyberattack_on_smartGrid\\saved_models\\isolation_forest_model.pkl"
    scaler_path = "F:\\mrudula college\\Sem1_project\\Cyberattack_on_smartGrid\\saved_models\\scaler.pkl"
    save_plot_path = "F:\\mrudula college\\Sem1_project\\Cyberattack_on_smartGrid\\saved_models\\confusion_matrix_plot.png"
    
    accuracy, best_params = train_and_visualize_isolation_forest(
        dataset_path, model_path, scaler_path, save_plot_path
    )
    print(f"\nFinal Results:")
    print(f"Best accuracy: {accuracy:.4f}")
    print(f"Best parameters: {best_params}")