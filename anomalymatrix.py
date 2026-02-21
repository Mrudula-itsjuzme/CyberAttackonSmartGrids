import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
dataset_path = "F:\\mrudula college\\Sem1_project\\Cyberattack_on_smartGrid\\intermediate_combined_data.csv"
logging.info(f"Loading dataset from {dataset_path}")
data = pd.read_csv(dataset_path)
logging.info(f"Dataset loaded successfully with shape {data.shape}")

# Automatically detect the target column
logging.info("Detecting target column")
target_column = None
for col in data.columns:
    if data[col].nunique() == 2 and sorted(data[col].unique()) == [0, 1]:
        target_column = col
        break

if target_column is None:
    logging.error("No suitable target column found. Please verify the dataset.")
    raise KeyError("Target column with binary labels (0 and 1) not found in the dataset.")

logging.info(f"Detected target column: {target_column}")

# Preprocess the dataset
logging.info(f"Dropping target column '{target_column}' for feature extraction")
features = data.drop(columns=[target_column])
target = data[target_column]

# Handle missing values based on data type
logging.info("Handling missing values by data type")
for column in features.columns:
    if features[column].dtype == 'object':  # For string/categorical columns
        features[column] = features[column].fillna(features[column].mode()[0])
    else:  # For numeric columns
        features[column] = features[column].fillna(features[column].mean())

# Convert categorical variables to numeric
logging.info("Converting categorical variables to numeric")
categorical_columns = features.select_dtypes(include=['object']).columns
for column in categorical_columns:
    features[column] = pd.Categorical(features[column]).codes

# Train-Test Split
from sklearn.model_selection import train_test_split
logging.info("Splitting data into train and test sets")
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Fit the Isolation Forest model
logging.info("Initializing Isolation Forest model")
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
logging.info("Fitting the Isolation Forest model to the training data")
model.fit(X_train)
logging.info("Model training completed")

# Predict anomalies
logging.info("Making predictions on training and test data")
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Convert predictions to match target format (0 for normal, 1 for anomaly)
def format_predictions(predictions):
    return np.where(predictions == -1, 1, 0)

logging.info("Formatting predictions")
y_train_pred = format_predictions(train_predictions)
y_test_pred = format_predictions(test_predictions)

# Evaluate the model
logging.info("Evaluating model performance")
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:\n", cm)
logging.info(f"Confusion Matrix:\n{cm}")

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
logging.info("Plotting the confusion matrix")
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Save the model
model_path = "isolation_forest_model.pkl"
logging.info(f"Saving the model to {model_path}")
joblib.dump(model, model_path)
logging.info(f"Model saved to {model_path}")