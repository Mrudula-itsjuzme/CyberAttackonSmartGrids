import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import IsolationForest

# File paths
input_path = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\FRESH\processed_dataset.csv"
output_path = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\FRESH\final_secure_dataset.csv"

# Privacy and Anomaly Settings
chunk_size = 50000  # Process 50,000 rows at a time
epsilon = 0.5  # Differential privacy budget
contamination = 0.05  # Proportion of anomalies for Isolation Forest

# Function to add Laplace noise (Differential Privacy)
def add_laplace_noise(column, epsilon):
    sensitivity = column.max() - column.min()
    noise = np.random.laplace(0, sensitivity / epsilon, len(column))
    return column + noise

# Function to detect anomalies (Anomaly Detection with Isolation Forest)
def detect_anomalies(chunk):
    numerical_cols = chunk.select_dtypes(include=['float64', 'int64']).columns
    if numerical_cols.empty:
        print("⚠️ No numerical columns found for anomaly detection.")
        return chunk

    print("🔍 Detecting anomalies...")
    model = IsolationForest(contamination=contamination, random_state=42)
    chunk['Anomaly'] = model.fit_predict(chunk[numerical_cols])
    chunk['Anomaly'] = chunk['Anomaly'].apply(lambda x: 'Normal' if x == 1 else 'Anomaly')
    return chunk

# Function to simulate attacks
def simulate_attack(chunk):
    print("⚡ Simulating replay attack...")
    attack_chunk = chunk.sample(frac=0.01, random_state=42)  # Simulate attacks on 1% of the data
    attack_chunk['Attack_Type'] = 'Replay'
    chunk['Attack_Type'] = np.where(chunk.index.isin(attack_chunk.index), 'Replay', 'Normal')
    return chunk

# Function to process chunks (Differential Privacy, Anomaly Detection, Attack Simulation)
def process_chunk(chunk):
    print(f"🚀 Processing chunk of size {len(chunk)}")

    # Clean mixed-type columns
    for col in chunk.columns:
        if chunk[col].dtype == object:
            print(f"⚠️ Converting column {col} to numeric.")
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)

    # Differential Privacy
    numerical_cols = chunk.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        print(f"🔒 Adding noise to {col}")
        chunk[col] = add_laplace_noise(chunk[col], epsilon)

    # Anomaly Detection
    chunk = detect_anomalies(chunk)

    # Attack Simulation
    chunk = simulate_attack(chunk)

    return chunk

# Function to visualize key features
def visualize_features(data, features):
    for feature in features:
        plt.figure(figsize=(10, 6))
        plt.hist(data[feature], bins=50, alpha=0.7, label='Histogram')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        plt.show()

# Process the large dataset in chunks
def process_large_dataset(input_path, output_path, chunk_size):
    chunks = pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)
    processed_chunks = []

    print("📂 Starting dataset processing...")
    for idx, chunk in tqdm(enumerate(chunks), desc="Processing chunks"):
        print(f"🔄 Processing chunk {idx + 1}")
        processed_chunk = process_chunk(chunk)
        processed_chunks.append(processed_chunk)

    # Combine processed chunks
    print("✅ Combining processed chunks...")
    processed_data = pd.concat(processed_chunks, axis=0)

    # Save final dataset
    processed_data.to_csv(output_path, index=False)
    print(f"🎉 Final dataset saved to: {output_path}")

    # Visualization of sample features
    sample_features = processed_data.select_dtypes(include=['float64', 'int64']).columns[:3]
    visualize_features(processed_data, sample_features)

# Run the script
process_large_dataset(input_path, output_path, chunk_size)
