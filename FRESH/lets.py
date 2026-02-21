import pandas as pd
import numpy as np
import logging
from scapy.all import PcapReader
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import Parallel, delayed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='process_log.log', filemode='w')

# Step 1: Extract Features from PCAP Files
def extract_features_from_pcap(pcap_file, chunk_size=5000, save_progress_every=10000):
    logging.info(f"Starting feature extraction from PCAP: {pcap_file}")
    print(f"Starting feature extraction from PCAP: {pcap_file}")
    data = []
    packet_count = 0

    with PcapReader(pcap_file) as pcap_reader:
        for pkt in pcap_reader:
            packet_count += 1
            if hasattr(pkt, "time") and hasattr(pkt, "len"):
                data.append({
                    "timestamp": pkt.time,
                    "length": len(pkt),
                    "src_ip": pkt.src if hasattr(pkt, 'src') else None,
                    "dst_ip": pkt.dst if hasattr(pkt, 'dst') else None,
                    "protocol": pkt.proto if hasattr(pkt, 'proto') else None,
                })

            # Save intermediate progress
            if packet_count % save_progress_every == 0:
                partial_df = pd.DataFrame(data)
                partial_df.to_csv(f"intermediate_pcap_features_{packet_count}.csv", index=False)
                logging.info(f"Saved {packet_count} packets to intermediate file.")

    logging.info(f"Extracted {len(data)} packets from PCAP.")
    print(f"Extracted {len(data)} packets from PCAP.")
    return pd.DataFrame(data)

# Step 2: Combine PCAP Data with Existing CSV Data
def merge_pcap_and_csv(pcap_features, csv_data):
    logging.info("Merging PCAP features with CSV data.")
    print("Merging PCAP features with CSV data...")

    # Ensure alignment and handle missing values
    pcap_features.fillna(0, inplace=True)
    csv_data.fillna(0, inplace=True)

    combined_data = pd.concat([pcap_features, csv_data], axis=1)

    # Save intermediate combined data
    combined_data.to_csv("intermediate_combined_data.csv", index=False)
    logging.info("Saved intermediate combined data.")
    print("Saved intermediate combined data.")

    return combined_data

# Step 3: Train and Evaluate a Single Model
def train_model(name, model, X_train, X_test, y_train, y_test):
    logging.info(f"Training {name}...")
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logging.info(f"Accuracy for {name}: {acc}")
    logging.info(f"Classification Report for {name}:\n{report}")

    print(f"Accuracy for {name}: {acc}")
    print(f"Classification Report for {name}:\n{report}")

    return name, acc

# Step 4: Classification with Multiple Models (Parallel)
def perform_classification_with_models(data):
    logging.info("Starting classification...")
    print("Starting classification...")

    # Assume the target column is "label" (last column)
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target

    # Train-test split
    logging.info("Splitting data into training and testing sets.")
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Models
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=42)
    }

    # Train and evaluate models in parallel
    results = Parallel(n_jobs=-1)(
        delayed(train_model)(name, model, X_train, X_test, y_train, y_test)
        for name, model in models.items()
    )

    # Display results
    logging.info("Model training complete. Summary:")
    print("\nSummary of Model Results:")
    for name, acc in results:
        logging.info(f"{name}: Accuracy = {acc}")
        print(f"{name}: Accuracy = {acc}")

# Step 5: Main Execution
if __name__ == "__main__":
    # Paths to PCAP file and CSV data
    pcap_file = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\FRESH\combined_output.pcap"  # Replace with your PCAP file path
    csv_file = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\FRESH\combined_output.csv"    # Replace with your CSV file path

    # Extract features from PCAP
    logging.info("Starting PCAP feature extraction...")
    print("Starting PCAP feature extraction...")
    pcap_features = extract_features_from_pcap(pcap_file)
    logging.info(f"Extracted {pcap_features.shape[0]} packets.")
    print(f"Extracted {pcap_features.shape[0]} packets.")

    # Load CSV data
    logging.info("Loading CSV data...")
    print("Loading CSV data...")
    csv_data = pd.read_csv(csv_file)

    # Merge PCAP and CSV data
    combined_data = merge_pcap_and_csv(pcap_features, csv_data)

    # Perform classification with multiple models
    logging.info("Performing classification using multiple models...")
    print("Performing classification using multiple models...")
    perform_classification_with_models(combined_data)
