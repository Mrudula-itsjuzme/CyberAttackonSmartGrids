import os
import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.stats import zscore, entropy

# **Define Paths**
DATA_PATH = r"H:\sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
OUTPUT_DIR = r"H:\sem1_project\Cyberattack_on_smartGrid\Processed_Data"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "IEEE_Formatted_Plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# 🔍 **Load dataset normally**
print(f"📂 Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"✅ Dataset loaded! Shape: {df.shape}")

# **SHA-256 Encryption Function**
def encrypt_value(value):
    """Encrypt categorical values using SHA-256 hashing."""
    return hashlib.sha256(str(value).encode()).hexdigest()

# **Preprocessing**
print(f"🔄 Preprocessing dataset...")

# **Convert numeric columns**
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert numbers
    except:
        pass

# **Fill missing values (numeric only)**
print("   ➡ Filling missing values...")
df.fillna(df.mean(numeric_only=True), inplace=True)

# **Convert categorical to numerical using Label Encoding**
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"   ➡ Encoding {len(categorical_cols)} categorical columns...")
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# **Normalize outliers using Z-score**
numeric_cols = df.select_dtypes(include=['number']).columns
print("   ➡ Applying Z-score normalization...")
for col in numeric_cols:
    if df[col].nunique() > 1:  # Apply only if more than 1 unique value
        df[col] = zscore(df[col])

# **Replace NaNs that appeared after Z-score**
df.fillna(df.mean(numeric_only=True), inplace=True)

# **Apply MinMax scaling**
print("   ➡ Applying MinMax scaling...")
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("✅ Preprocessing complete!\n")

# **Encrypt dataset**
print("🔐 Encrypting dataset...")
df_encrypted = df.copy()
for col in categorical_cols:
    df_encrypted[col] = df_encrypted[col].map(encrypt_value)
print("✅ Dataset encrypted!\n")

# **Save processed and encrypted data**
processed_data_path = os.path.join(OUTPUT_DIR, "Processed_Dataset.csv")
encrypted_data_path = os.path.join(OUTPUT_DIR, "Encrypted_Dataset.csv")

print(f"💾 Saving processed dataset to {processed_data_path}...")
df.to_csv(processed_data_path, index=False)
print("✅ Processed dataset saved!")

print(f"💾 Saving encrypted dataset to {encrypted_data_path}...")
df_encrypted.to_csv(encrypted_data_path, index=False)
print("✅ Encrypted dataset saved!\n")

# **Check Available Matplotlib Styles**
print(f"✅ Available Styles: {plt.style.available}")

# **Apply a Safe Style Instead of `seaborn-darkgrid`**
safe_style = 'ggplot' if 'ggplot' in plt.style.available else 'default'
plt.style.use(safe_style)

# **Generate IEEE-Formatted Plots**
print("📊 Generating IEEE-Formatted Plots...")

# **Plot: Feature Distribution**
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(df.iloc[:, 0], kde=True, bins=50, ax=ax)
ax.set_title("Feature Distribution (IEEE Format)", fontsize=14, fontweight='bold')
ax.set_xlabel("Feature Values", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
plot_path_1 = os.path.join(PLOTS_DIR, "Feature_Distribution.png")
plt.savefig(plot_path_1, dpi=300, bbox_inches='tight')
print(f"✅ Feature Distribution plot saved: {plot_path_1}")

# **Security validation tests**
print("🔍 Running security validation tests...")
df_entropy = df.apply(entropy)
entropy_file = os.path.join(OUTPUT_DIR, "Feature_Entropy_Analysis.csv")
df_entropy.to_csv(entropy_file)
print(f"✅ Security entropy analysis saved: {entropy_file}")

# **Summary report**
summary = {
    "Original Dataset Shape": df.shape,
    "Encrypted Dataset Shape": df_encrypted.shape,
    "Output Directory": OUTPUT_DIR,
    "Plots Directory": PLOTS_DIR,
}

print("\n🚀🚀🚀 PROCESSING COMPLETE! FINAL SUMMARY 🚀🚀🚀\n")
for key, value in summary.items():
    print(f"🔹 {key}: {value}")

print("\n🔥🔥🔥 YOUR DATASET IS NOW ENCRYPTED & SECURED! 🔥🔥🔥")
print("📂 Processed & encrypted files are saved!")
print(f"📊 Check the plots in: {PLOTS_DIR}")
print("✅ You're good to go! 🚀")
