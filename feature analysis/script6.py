# Script 6: Integration and Validation
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class IntegrationValidation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.features = None
        self.labels = None
        self.model = None

    def load_and_preprocess_data(self):
        print("Loading and preprocessing data...")
        # Load data
        data = pd.read_csv(self.file_path)

        # Sample a subset of data
        data = data.sample(frac=0.1, random_state=42)

        # Identify numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # Replace infinities with NaN and fill NaN with column mean
        data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

        # Assume last column is the label (modify as necessary)
        self.features = data.iloc[:, :-1]
        self.labels = data.iloc[:, -1]

        # Reduce dimensionality with PCA
        print("Reducing dimensionality with PCA...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=20)  # Keep top 20 components
        self.features = pca.fit_transform(self.features)

        # Standardize features
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

        print("Data preprocessing complete.")

    def train_model(self):
        print("Training model...")
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)

        # Train a Random Forest model with fewer estimators
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

    def train_model(self):
        print("Training model...")
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)

        # Train a Random Forest model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

    def feature_importance_analysis(self):
        print("Analyzing feature importance...")
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(self.features.shape[1]), importances[indices], align="center")
        plt.xticks(range(self.features.shape[1]), indices)
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.show()

    def cross_validate_model(self, cv=5):
        print("Performing cross-validation...")
        scores = cross_val_score(self.model, self.features, self.labels, cv=cv)
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {np.mean(scores):.2f}")

if __name__ == "__main__":
    print("Running Script 6: Integration and Validation")

    # File path to your dataset
    file_path = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"

    # Initialize and run integration and validation
    integrator = IntegrationValidation(file_path)
    integrator.load_and_preprocess_data()
    integrator.train_model()
    integrator.feature_importance_analysis()
    integrator.cross_validate_model()

    print("Script 6 completed: Integration and Validation")
