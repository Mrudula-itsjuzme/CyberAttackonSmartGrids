import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from itertools import groupby

# **Define Paths**
DATA_PATH = r"H:\sem1_project\Cyberattack_on_smartGrid\Processed_Data\Processed_Dataset.csv"
OUTPUT_PATH = r"H:\sem1_project\Cyberattack_on_smartGrid\Graphs\Cosine_Similarity_RLE.png"

# **Create output directory**
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# **Load datasets**
print("📂 Loading dataset...")
df_processed = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded successfully!")

# **Fix Column Name Mismatch**
df_processed.columns = df_processed.columns.astype(str)

# **Convert to float32 to reduce memory usage**
df_processed = df_processed.astype(np.float32)

# **🔹 Apply RLE Encryption**
def rle_encode(column):
    """Applies RLE encoding to a single column."""
    encoded = []
    for key, group in groupby(column):
        encoded.append((key, len(list(group))))
    return np.array(encoded).flatten()  # Flatten to maintain array shape

print("🔐 Applying RLE Encryption...")
df_encrypted = df_processed.apply(rle_encode, axis=0)

# **Handle Dimension Mismatch**
max_len = max(len(col) for col in df_encrypted)
df_encrypted = df_encrypted.apply(lambda col: np.pad(col, (0, max_len - len(col)), 'constant', constant_values=np.nan))

# **🔍 Normalize Before Computing Similarity**
df_encrypted = (df_encrypted - df_encrypted.min()) / (df_encrypted.max() - df_encrypted.min())

# **🔹 Compute Cosine Similarity for each feature**
print("🔍 Calculating Cosine Similarity...")
cosine_sim_values = df_processed.corrwith(df_encrypted, method=cosine)

# **Handle NaNs by replacing with the column-wise mean**
cosine_sim_mean = np.nanmean(cosine_sim_values)
cosine_sim_values = cosine_sim_values.fillna(cosine_sim_mean)

# **🔧 Rescale for better visibility**
scaler = StandardScaler()
cosine_sim_scaled = scaler.fit_transform(cosine_sim_values.values.reshape(-1, 1)).flatten()

# **🔹 Compute Absolute Differences to Highlight Changes**
feature_diff = np.abs(df_processed.mean() - df_encrypted.mean())

# **Fix NaN issue: Compute percentiles safely**
if not np.isnan(feature_diff).all():
    low_threshold = np.nanpercentile(feature_diff, 10)  # 10th percentile
    mid_threshold = np.nanpercentile(feature_diff, 50)  # Median
    high_threshold = np.nanpercentile(feature_diff, 90)  # 90th percentile
else:
    low_threshold, mid_threshold, high_threshold = 0.01, 0.05, 0.1  # Fallback values

# **Clamp Cosine Similarity Values**
cosine_sim_scaled = np.clip(cosine_sim_scaled, -2, 3)

# **Prepare feature indices**
feature_indices = np.arange(len(cosine_sim_values))

# **🔹 Generate Line Graph with Proper Thresholds**
plt.figure(figsize=(12, 6))
plt.plot(feature_indices, cosine_sim_scaled, marker='o', linestyle='-', color='blue', label="Cosine Similarity")

# **Thresholds based on detected variations**
plt.axhline(y=low_threshold, color='green', linestyle='--', label=f'Threshold {low_threshold:.2f}')
plt.axhline(y=mid_threshold, color='orange', linestyle='--', label=f'Threshold {mid_threshold:.2f}')
plt.axhline(y=high_threshold, color='red', linestyle='--', label=f'Threshold {high_threshold:.2f}')

# **Highlight Points with Variations**
plt.scatter(feature_indices, cosine_sim_scaled, color='blue', edgecolors='black', s=30, zorder=3)

# **Labels and Title**
plt.xlabel("Feature Index", fontsize=12, fontweight='bold')
plt.ylabel("Cosine Similarity (Standardized)", fontsize=12, fontweight='bold')
plt.title("Cosine Similarity Analysis (RLE Encryption)", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# **Save and Show**
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"✅ Cosine Similarity Plot Saved at: {OUTPUT_PATH}")
plt.show()
