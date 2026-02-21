import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_prepare_data(filepath):
    """Load and prepare the dataset for classification."""
    print("📝 Loading dataset...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"📝 Dataset loaded with shape: {df.shape}")
    
    # Separate features and target
    X = df.drop('attack_type', axis=1)
    y = df['attack_type']
    
    # Encode categorical variables
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column].astype(str))
    
    # Encode target variable
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)
    
    # Split the data
    print("🔄 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"🔄 Data split into {X_train.shape[0]} training and {X_test.shape[0]} testing samples.")
    
    return X_train, X_test, y_train, y_test, label_encoder_y, label_encoders

def train_optimized_random_forest(X_train, y_train):
    """Train Random Forest with optimized parameters."""
    print("🌳 Training Random Forest with optimized parameters...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=1,  # Avoid parallel processing
        random_state=42,
        verbose = 2
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def train_optimized_xgboost(X_train, y_train):
    """Train XGBoost with optimized parameters."""
    print("🚀 Training XGBoost with optimized parameters...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        tree_method='hist',  # Memory efficient algorithm
        n_jobs=1,  # Avoid parallel processing
        random_state=42366032
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

def evaluate_model(model, X_test, y_test, label_encoder_y, model_name):
    """Evaluate the model and display results."""
    print(f"📈 Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy: {accuracy:.4f}")
    
    # Get classification report
    class_names = label_encoder_y.classes_
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Create confusion matrix
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return accuracy

def save_best_model(best_model, label_encoders, label_encoder_y, model_name):
    """Save the best model and encoders."""
    print(f"💾 Saving {model_name} and encoders...")
    joblib.dump(best_model, f'best_model_{model_name.lower().replace(" ", "_")}.joblib')
    joblib.dump(label_encoders, 'label_encoders.joblib')
    joblib.dump(label_encoder_y, 'label_encoder_y.joblib')

def main():
    # File path to your processed dataset
    filepath = r'D:\IEC 60870-5-104 SEG Intrusion Detection Dataset\processed_dataset.csv'
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, label_encoder_y, label_encoders = load_and_prepare_data(filepath)
    
    # Train models
    models = {
        'Random Forest': train_optimized_random_forest(X_train, y_train),
        'XGBoost': train_optimized_xgboost(X_train, y_train)
    }
    
    # Evaluate models and select the best one
    best_accuracy = 0
    best_model_name = None
    
    for name, model in models.items():
        accuracy = evaluate_model(model, X_test, y_test, label_encoder_y, name)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
    
    # Save the best model
    save_best_model(models[best_model_name], label_encoders, label_encoder_y, best_model_name)
    
    print(f"\n✨ Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    print("\n✅ Classification pipeline complete!")

if __name__ == "__main__":
    main()
