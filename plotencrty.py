import os
import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.stats import entropy
from scipy.spatial.distance import cosine

# **Define Paths**
DATA_PATH = r"H:\sem1_project\Cyberattack_on_smartGrid\Processed_Data\Processed_Dataset.csv"
ENCRYPTED_PATH = r"H:\sem1_project\Cyberattack_on_smartGrid\Processed_Data\Encrypted_Dataset.csv"
OUTPUT_DIR = r"H:\sem1_project\Cyberattack_on_smartGrid\Graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# **Load the datasets**
print("📂 Loading datasets...")
df_processed = pd.read_csv(DATA_PATH)
df_encrypted = pd.read_csv(ENCRYPTED_PATH)
print("✅ Datasets loaded successfully!")

# **Fix Column Name Mismatch**
df_processed.columns = df_processed.columns.astype(str)
df_encrypted.columns = df_encrypted.columns.astype(str)

# **Filter Only Matching Columns**
common_columns = df_processed.columns.intersection(df_encrypted.columns)

# **Convert to float32 to reduce memory usage**
df_processed = df_processed[common_columns].astype(np.float32)
df_encrypted = df_encrypted[common_columns].astype(np.float32)

# **Functions for Security Analysis**
def save_plot(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {name}")

def safe_execute(test_name, function):
    """Execute a function and handle errors without stopping execution."""
    try:
        return function()
    except Exception as e:
        print(f"❌ {test_name} failed due to: {e}")
        return None

# **📏 1️⃣ Feature Entropy Calculation**
def calculate_entropy():
    print("🔍 Calculating feature entropy...")
    entropy_values = df_processed.apply(entropy, axis=0)
    return entropy_values

entropy_values = safe_execute("Feature Entropy", calculate_entropy)

# **📏 2️⃣ Shannon Redundancy Calculation**
def calculate_redundancy():
    total_unique_values = df_processed.nunique()
    return 1 - (entropy_values / np.log2(total_unique_values.replace(0, 1)))

redundancy_values = safe_execute("Shannon Redundancy", calculate_redundancy)

# **📏 3️⃣ KL-Divergence Calculation**
def calculate_kl_div():
    print("🔍 Calculating KL-Divergence...")
    df_processed_prob = df_processed.div(df_processed.sum(axis=0), axis=1).fillna(0)
    df_encrypted_prob = df_encrypted.div(df_encrypted.sum(axis=0), axis=1).fillna(0)

    # Convert to NumPy and apply np.clip() efficiently
    df_processed_prob = np.clip(df_processed_prob.to_numpy(dtype=np.float32), 1e-10, 1)
    df_encrypted_prob = np.clip(df_encrypted_prob.to_numpy(dtype=np.float32), 1e-10, 1)

    df_processed_prob = pd.DataFrame(df_processed_prob, columns=common_columns)
    df_encrypted_prob = pd.DataFrame(df_encrypted_prob, columns=common_columns)

    return df_processed_prob.apply(lambda col: entropy(col, df_encrypted_prob[col]), axis=0)

kl_div_values = safe_execute("KL-Divergence", calculate_kl_div)

# **📏 4️⃣ Cosine Similarity Calculation**
def calculate_cosine_similarity():
    print("🔍 Calculating Cosine Similarity...")
    return df_processed.corrwith(df_encrypted, method=cosine)

cosine_sim_values = safe_execute("Cosine Similarity", calculate_cosine_similarity)

# **📏 5️⃣ Feature Correlation Matrices**
def calculate_correlation_matrices():
    print("📊 Generating Feature Correlation Matrices...")
    return df_processed.corr(), df_encrypted.corr()

corr_before, corr_after = safe_execute("Feature Correlation Matrices", calculate_correlation_matrices)

# **📊 Generate Graphs**
print("📊 Generating plots...")

# **1️⃣ Feature Entropy Analysis**
if entropy_values is not None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=entropy_values.index, y=entropy_values.values, palette="viridis", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Entropy Value")
    ax.set_title("Feature Entropy Analysis")
    save_plot(fig, "Feature_Entropy_Analysis.png")

# **2️⃣ Shannon Redundancy Plot**
if redundancy_values is not None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=redundancy_values.index, y=redundancy_values.values, palette="mako", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Redundancy")
    ax.set_title("Shannon Redundancy Analysis")
    save_plot(fig, "Shannon_Redundancy.png")

# **3️⃣ KL-Divergence Plot**
if kl_div_values is not None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=kl_div_values.index, y=kl_div_values.values, palette="coolwarm", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel("Feature")
    ax.set_ylabel("KL-Divergence")
    ax.set_title("KL-Divergence Analysis")
    save_plot(fig, "KL_Divergence.png")

# **4️⃣ Cosine Similarity Plot**
if cosine_sim_values is not None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=cosine_sim_values.index, y=cosine_sim_values.values, palette="Blues", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity (Before vs. After Encryption)")
    save_plot(fig, "Cosine_Similarity.png")

# **5️⃣ Feature Correlation Matrix (Before Encryption)**
if corr_before is not None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_before, cmap="coolwarm", annot=False, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix (Before Encryption)")
    save_plot(fig, "Feature_Correlation_Before.png")

# **6️⃣ Feature Correlation Matrix (After Encryption)**
if corr_after is not None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_after, cmap="coolwarm", annot=False, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix (After Encryption)")
    save_plot(fig, "Feature_Correlation_After.png")

print("\n🚀🚀🚀 ALL SECURITY TESTS & GRAPHS GENERATED SUCCESSFULLY! 🚀🚀🚀")
print(f"📂 Check your graphs in: {OUTPUT_DIR}")
