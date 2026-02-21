import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Load and preprocess the dataset
def load_and_prepare_data(filepath):
    print("🗋 Loading dataset...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"🗋 Dataset loaded with shape: {df.shape}")
    
    # Detect target column
    possible_targets = [col for col in df.columns if 'attack' in col.lower() or 'label' in col.lower() or 'class' in col.lower()]
    if not possible_targets:
        raise ValueError("Could not automatically detect a target column. Please specify the target column manually.")
    target_column = possible_targets[0]
    print(f"🎯 Detected target column: {target_column}")
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Handle duplicate columns
    print("🔄 Checking for duplicate feature names...")
    if X.columns.duplicated().any():
        print(f"Duplicate columns found: {X.columns[X.columns.duplicated()].tolist()}")
        X = X.loc[:, ~X.columns.duplicated()]
    
    # Handle infinity and large values
    print("🔄 Checking and replacing infinity or very large values...")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Encode categorical variables
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column].astype(str))
    
    # Encode target variable
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)
    
    # Handle missing values by imputing with mean
    print("🔄 Imputing missing values...")
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Normalize features
    print("🔄 Normalizing features...")
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Split the data
    print("🔄 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"🔄 Data split into {X_train.shape[0]} training and {X_test.shape[0]} testing samples.")
    
    return X_train, X_test, y_train, y_test, label_encoder_y, label_encoders


# Dynamically select top features
def select_features(X_train, y_train, X_test):
    print("🔄 Selecting top features...")
    k = min(50, X_train.shape[1])  # Dynamically choose k to avoid exceeding available features
    print(f"🔍 Selecting top {k} features out of {X_train.shape[1]} available features...")
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected


# Compute class weights
def compute_weights(y_train):
    print("🔄 Computing class weights...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    return {i: weight for i, weight in enumerate(class_weights)}


# Model training functions
def train_random_forest(X_train, y_train, class_weights):
    print("🌳 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=30, n_jobs=-1, random_state=42, class_weight=class_weights
    )
    rf_model.fit(X_train, y_train)
    return rf_model


def train_xgboost(X_train, y_train, class_weights):
    print("🚀 Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=class_weights[1] / class_weights[0], tree_method='hist', n_jobs=-1, random_state=42
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model


def train_lightgbm(X_train, y_train, class_weights):
    print("🚀 Training LightGBM...")
    lgb_model = LGBMClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1, class_weight=class_weights, n_jobs=-1, random_state=42
    )
    lgb_model.fit(X_train, y_train)
    return lgb_model


def train_decision_tree(X_train, y_train, class_weights):
    print("🌳 Training Decision Tree...")
    dt_model = DecisionTreeClassifier(max_depth=25, random_state=42, class_weight=class_weights)
    dt_model.fit(X_train, y_train)
    return dt_model


def train_stacking_classifier(X_train, y_train, class_weights):
    print("🔄 Training Stacking Classifier...")
    
    # Use a subset of training data for the meta-model
    subset_size = int(0.2 * X_train.shape[0])  # 20% of the training data
    subset_indices = np.random.choice(X_train.shape[0], subset_size, replace=False)
    X_subset = X_train[subset_indices]
    y_subset = y_train[subset_indices]
    
    stacked_model = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1, class_weight=class_weights)),
            ('xgb', XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1)),
            ('lgb', LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1, class_weight=class_weights))
        ],
        final_estimator=RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        n_jobs=-1
    )
    stacked_model.fit(X_subset, y_subset)
    return stacked_model


# Evaluation function
def evaluate_model(model, X_test, y_test, label_encoder_y, model_name):
    print(f"📈 Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder_y.classes_))
    return accuracy


# Main pipeline
def main():
    filepath = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\FRESH\processed_dataset.csv"
    X_train, X_test, y_train, y_test, label_encoder_y, _ = load_and_prepare_data(filepath)
    class_weights = compute_weights(y_train)
    X_train_selected, X_test_selected = select_features(X_train, y_train, X_test)
    
    models = {
        'Random Forest': train_random_forest(X_train_selected, y_train, class_weights),
        'XGBoost': train_xgboost(X_train_selected, y_train, class_weights),
        'LightGBM': train_lightgbm(X_train_selected, y_train, class_weights),
        'Decision Tree': train_decision_tree(X_train_selected, y_train, class_weights),
        'Stacking Classifier': train_stacking_classifier(X_train_selected, y_train, class_weights)
    }

    best_accuracy = 0
    best_model = None
    for name, model in models.items():
        accuracy = evaluate_model(model, X_test_selected, y_test, label_encoder_y, name)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print(f"\n✨ Best model accuracy: {best_accuracy:.4f}")
    print("\n✅ Complete!")

if __name__ == "__main__":
    main()
