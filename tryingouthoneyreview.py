import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

# ---- CONFIG ----
DATA_PATH = "G:/Sem1/Cyberattack_on_smartGrid/intermediate_combined_data.csv"  # path to dataset
CHUNKSIZE = 50000
SAMPLE_SIZE = 500  # per category per chunk
EPSILON = 0.05
FEATURES = [
    'Flow Duration',
    'Fwd IAT Mean',
    'Bwd IAT Std',
    'Flow IAT Max',
    'flow packet APDU length mean',
    'flow down/up ratio',
]
# Output
combined_tsne = []
combined_labels = []
feature_deltas = []

# ---- Adversarial FGSM-Style Perturbation ----
def fgsm_perturb(x, epsilon=EPSILON):
    grad = np.sign(np.random.randn(*x.shape))
    return np.clip(x + epsilon * grad, -3, 3)

# ---- Process Each Chunk ----
for i, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=CHUNKSIZE)):
    print(f"Processing chunk {i+1}...")

    if not all(col in chunk.columns for col in FEATURES):
        print(f"Chunk {i+1} skipped: required features missing.")
        continue

    # Drop NaNs and sample
    df_chunk = chunk[FEATURES].dropna()
    if len(df_chunk) < SAMPLE_SIZE * 2:
        print(f"Chunk {i+1} skipped: not enough data.")
        continue

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_chunk)

    X_benign = X_scaled[:SAMPLE_SIZE]
    X_real = X_scaled[-SAMPLE_SIZE:]
    X_adv = np.array([fgsm_perturb(xi) for xi in X_benign])

    # Stack for TSNE
    X_total = np.vstack([X_benign, X_adv, X_real])
    labels = (['Benign'] * SAMPLE_SIZE +
              ['Adversarial'] * SAMPLE_SIZE +
              ['RealAttack'] * SAMPLE_SIZE)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_total)
    combined_tsne.append(X_tsne)
    combined_labels.extend(labels)

    # Feature deltas
    df_benign = pd.DataFrame(X_benign, columns=FEATURES)
    df_adv = pd.DataFrame(X_adv, columns=FEATURES)
    df_real = pd.DataFrame(X_real, columns=FEATURES)

    delta_adv = (df_adv - df_benign).abs().mean()
    delta_real = (df_real - df_benign).abs().mean()

    delta_df = pd.DataFrame({
        'Feature': FEATURES,
        'Δ_Adversarial': delta_adv.values,
        'Δ_RealAttack': delta_real.values,
        'Chunk': i + 1
    })
    feature_deltas.append(delta_df)

# ---- Final Visualizations ----

# Combine t-SNEs
if combined_tsne:
    tsne_all = np.vstack(combined_tsne)
    df_vis = pd.DataFrame(tsne_all, columns=["x", "y"])
    df_vis['Label'] = combined_labels

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_vis, x='x', y='y', hue='Label', palette='Set2', alpha=0.6)
    plt.title("t-SNE Across Chunks: Benign vs Adversarial vs Real Attack")
    plt.savefig("chunkwise_tsne_comparison.png", dpi=300)
    plt.show()

# Combine deltas
if feature_deltas:
    full_delta = pd.concat(feature_deltas, ignore_index=True)
    full_delta.to_csv("chunkwise_feature_shift.csv", index=False)
    print("Feature delta report saved to chunkwise_feature_shift.csv")
