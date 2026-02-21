# Script 7: Defense Tactics and Recommendations

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class DefenseTactics:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.features = None
        self.labels = None
        self.model = None

    def load_data(self):
        print("Loading data...")
        # Load data without preprocessing
        self.data = pd.read_csv(self.file_path, low_memory=False)

        # Assume last column is the label (modify as necessary)
        self.features = self.data.iloc[:, :-1]
        self.labels = self.data.iloc[:, -1]

        print("Data loading complete.")

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
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

    def recommend_defense_tactics(self):
        print("Generating defense tactics and recommendations...")

        recommendations = [
            "1. Apply differential privacy to protect sensitive features during data collection.",
            "2. Use homomorphic encryption for secure data sharing and analysis.",
            "3. Implement anomaly detection models to monitor traffic in real-time.",
            "4. Train the system to identify common attack patterns using reinforcement learning.",
            "5. Perform regular feature analysis to detect evolving threats.",
            "6. Use explainable AI models to validate predictions and build trust.",
            "7. Conduct regular system penetration tests to identify vulnerabilities.",
            "8. Enforce role-based access control to minimize unauthorized data access."
        ]

        print("Recommended Defense Tactics:")
        for recommendation in recommendations:
            print(recommendation)

        # Save recommendations to a file
        with open("defense_tactics.txt", "w") as f:
            f.write("Recommended Defense Tactics:\n")
            f.write("===========================\n")
            f.write("\n".join(recommendations))

if __name__ == "__main__":
    print("Running Script 7: Defense Tactics and Recommendations")

    # File path to your dataset
    file_path = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"

    # Initialize and run defense tactics
    defense = DefenseTactics(file_path)
    defense.load_data()
    defense.train_model()
    defense.recommend_defense_tactics()

    print("Script 7 completed: Defense Tactics and Recommendations")
