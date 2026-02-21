import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import VotingClassifier

def load_and_preprocess(filepath, target_col="Label"):
    """Load and preprocess the dataset with enhanced cleaning."""
    print("📝 Loading dataset...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"📊 Initial shape: {df.shape}")
    
    print(f"Target column distribution before encoding:\n{df[target_col].value_counts()}")

    # Binary encoding of the target column
    df[target_col] = df[target_col].apply(lambda x: "NORMAL" if x == "NORMAL" else "ANOMALY")
    y = (df[target_col] == "ANOMALY").astype(int)
    
    # Drop the target column from features
    df_features = df.drop(columns=[target_col])
    
    # Select numeric columns only
    df_numeric = df_features.select_dtypes(include=[np.number])
    
    # Enhanced cleaning
    df_numeric = handle_outliers(df_numeric)
    df_numeric = handle_missing_values(df_numeric)
    
    # Use RobustScaler instead of StandardScaler for better handling of outliers
    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df_numeric)
    
    return pd.DataFrame(df_scaled, columns=df_numeric.columns), y

def handle_outliers(df):
    """Handle outliers using IQR method."""
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower_bound, upper_bound)
    return df

def handle_missing_values(df):
    """Enhanced missing value handling."""
    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # For each column, fill NaN with median if skewed, mean otherwise
    for column in df.columns:
        skewness = df[column].skew()
        if abs(skewness) > 1:
            df[column].fillna(df[column].median(), inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)
    
    return df

def select_features(X, y):
    """Select most important features using Random Forest."""
    print("🔍 Selecting important features...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = SelectFromModel(rf, prefit=False)
    selector.fit(X, y)
    
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Selected {len(selected_features)} features")
    
    return X[selected_features]

def optimize_isolation_forest(X, y):
    """Find optimal parameters for Isolation Forest using GridSearchCV."""
    print("🔄 Optimizing Isolation Forest parameters...")
    
    # Convert Isolation Forest predictions to match the target format
    def custom_scorer(estimator, X, y):
        y_pred = np.where(estimator.predict(X) == 1, 0, 1)
        return accuracy_score(y, y_pred)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_samples': ['auto', 100, 200],
        'contamination': [0.1, 0.2, 0.3],
        'max_features': [0.5, 0.7, 1.0],
        'bootstrap': [True, False]
    }
    
    base_model = IsolationForest(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=custom_scorer,
        cv=3,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    print(f"Best parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def create_ensemble(X, y):
    """Create an ensemble of anomaly detection models."""
    print("🤝 Creating ensemble model...")
    
    # Create base models with different parameters
    models = [
        IsolationForest(contamination=0.2, n_estimators=100, random_state=42),
        IsolationForest(contamination=0.3, n_estimators=200, random_state=43),
        IsolationForest(contamination=0.25, n_estimators=300, random_state=44)
    ]
    
    # Train each model
    predictions = []
    for i, model in enumerate(models):
        model.fit(X)
        pred = np.where(model.predict(X) == 1, 0, 1)
        predictions.append(pred)
    
    # Combine predictions using majority voting
    ensemble_pred = np.array([1 if sum(row) >= 2 else 0 for row in zip(*predictions)])
    
    return ensemble_pred

def evaluate_model(y_true, y_pred):
    """Enhanced model evaluation with additional metrics."""
    acc = accuracy_score(y_true, y_pred)
    print(f"🎯 Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Enhanced visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    report = classification_report(y_true, y_pred, target_names=["NORMAL", "ANOMALY"])
    print("\nClassification Report:")
    print(report)
    
    return acc, cm, report

def main():
    filepath = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\combined_dataset.csv"
    
    # Load and preprocess data
    X, y = load_and_preprocess(filepath)
    
    # Select important features
    X_selected = select_features(X, y)
    
    # Reduce dimensionality with optimized variance retention
    pca = PCA(n_components=0.98)  # Increased variance retention
    X_reduced = pca.fit_transform(X_selected)
    
    # Create and evaluate ensemble
    y_pred = create_ensemble(X_reduced, y)
    
    # Evaluate results
    accuracy, confusion_mat, report = evaluate_model(y, y_pred)
    
    # Save results
    with open("enhanced_evaluation_report.txt", "w") as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_mat))
        f.write("\n\nClassification Report:\n")
        f.write(report)
    
    print("✅ Enhanced analysis complete!")

if __name__ == "__main__":
    main()