import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from datetime import datetime

# === CONFIGURATION ===
model_path = r"H:\sem1_project\Cyberattack_on_smartGrid\best_model_xgboost.joblib"
feature_path = r"H:\sem1_project\Cyberattack_on_smartGrid\trained_feature_columns.txt"
encoder_path = r"H:\sem1_project\Cyberattack_on_smartGrid\label_encoder.pkl"
csv_path = r"H:\sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
honeypot_log = "honeypot_log.txt"
blocked_ips_path = "blocked_ips.txt"
chunk_size = 10000

# === STREAMLIT HEADER ===
st.set_page_config("CyberDefense Hub: Final Form", layout="wide")
st.title("🛡️ CyberDefense Hub: Final Form")
st.caption("IDS + Honeypot + Feature Alignment + IP Blocking + Visualization")

# === LOAD STUFF ===
try:
    model = joblib.load(model_path)
    st.success("Model loaded.")
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

try:
    with open(feature_path, "r") as f:
        trained_features = [line.strip() for line in f]
    st.info("Trained features loaded.")
except:
    st.error("Feature list not found.")
    st.stop()

try:
    label_encoder = joblib.load(encoder_path)
    st.info("LabelEncoder loaded.")
except:
    st.warning("LabelEncoder not found. Labels won't be decoded.")
    label_encoder = None

# === CSV LOAD ===
@st.cache_data
def load_csv(path, chunk_size=chunk_size):
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks)

try:
    df_raw = load_csv(csv_path)
    st.success(f"Dataset loaded. Shape: {df_raw.shape}")
except Exception as e:
    st.error(f"CSV load failed: {e}")
    st.stop()

# === HONEYPOT MODULE ===
st.subheader("Honeypot Simulation")
if st.button("Simulate Attacker Visit"):
    fake_ip = f"192.168.1.{len(open(honeypot_log).readlines()) % 255}" if os.path.exists(honeypot_log) else "192.168.1.1"
    with open(honeypot_log, "a") as f:
        f.write(f"[{pd.Timestamp.now()}] Visit from {fake_ip}\n")
    st.success(f"Access logged from {fake_ip}")

if os.path.exists(honeypot_log):
    logs = open(honeypot_log).readlines()[-5:]
    st.text_area("Recent Honeypot Logs", "".join(logs), height=120)

# === FEATURE ALIGNMENT + ENCODING ===
df = df_raw.copy()
X = df.copy()

# Drop columns that should not be used
drop_cols = ['Dst IP', 'Src IP', 'Flow ID', 'Timestamp', 'Label']
X.drop(columns=[col for col in drop_cols if col in X.columns], inplace=True)

# Encode string/object columns
for col in X.select_dtypes(include='object').columns:
    try:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        st.info(f"Encoded: {col}")
    except:
        X[col] = 0
        st.warning(f"Failed to encode {col}")

# Add missing columns
for col in trained_features:
    if col not in X.columns:
        X[col] = 0
        st.warning(f"Added missing feature: {col}")

# Drop extras
X = X[trained_features]

# === PREDICTIONS IN CHUNKS ===
st.subheader("Prediction (Chunked)")
predictions = []
model_features = model.get_booster().feature_names
num_chunks = len(X) // chunk_size + 1

for i in range(num_chunks):
    chunk = X.iloc[i * chunk_size: (i + 1) * chunk_size].copy()

    if chunk.empty:
        continue

    # Ensure model input alignment
    for col in model_features:
        if col not in chunk.columns:
            chunk[col] = 0
    chunk = chunk[model_features]

    try:
        preds = model.predict(chunk)
        predictions.extend(preds)
        st.success(f"Chunk {i+1}/{num_chunks} OK")
    except Exception as e:
        st.error(f"Chunk {i+1} failed: {e}")
        predictions.extend([-1]*len(chunk))

df["Prediction"] = predictions[:len(df)]

# === DECODE LABELS ===
if label_encoder:
    df["Predicted_Label"] = label_encoder.inverse_transform(df["Prediction"].astype(int))
    st.success("Labels decoded.")

# === VISUALIZATION ===
st.subheader("Prediction Distribution")
class_counts = df["Prediction"].value_counts().sort_index()
fig, ax = plt.subplots()
class_counts.plot(kind='bar', ax=ax)
ax.set_title("Prediction Class Distribution")
st.pyplot(fig)

if st.checkbox("Pie Chart"):
    fig2, ax2 = plt.subplots()
    class_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
    ax2.set_ylabel("")
    st.pyplot(fig2)

# === ACCURACY METRIC ===
if "Label" in df.columns:
    try:
        actual = df["Label"]
        if label_encoder:
            actual = label_encoder.transform(actual)
        acc = accuracy_score(actual, df["Prediction"])
        st.metric("Model Accuracy", f"{acc*100:.2f}%")
    except:
        st.warning("Accuracy calculation failed.")

# === OUTPUT SAVE ===
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"predicted_output_{ts}.csv"
df.to_csv(output_file, index=False)
st.success(f"Saved to: {output_file}")

# === FIREWALL MODULE ===
st.subheader("IP Firewall Management")
if not os.path.exists(blocked_ips_path):
    with open(blocked_ips_path, "w") as f:
        f.write("")

blocked_ips = open(blocked_ips_path).read().splitlines()
ip_input = st.text_input("IP to block")
if st.button("Block IP"):
    if ip_input and ip_input not in blocked_ips:
        with open(blocked_ips_path, "a") as f:
            f.write(ip_input + "\n")
        st.success(f"Blocked: {ip_input}")

if blocked_ips:
    st.warning("Currently Blocked:")
    st.code("\n".join(blocked_ips))
    ip_unblock = st.selectbox("Unblock IP", blocked_ips)
    if st.button("Unblock IP"):
        blocked_ips.remove(ip_unblock)
        with open(blocked_ips_path, "w") as f:
            f.write("\n".join(blocked_ips))
        st.success(f"Unblocked: {ip_unblock}")
