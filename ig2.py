import os
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import numpy as np

st.set_page_config(layout="wide", page_title="SCADA IDS Dashboard")

# Paths
DATA_PATH = r"G:\Sem 1\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
MODEL_PATH = r"G:\Sem 1\Cyberattack_on_smartGrid\ids_output\models\decision_tree_model.pkl"

# Load model safely
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

# Load data (limited sample to avoid memory crash)
@st.cache_data(ttl=300, max_entries=1)
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False, nrows=100000)  # Reduced rows for performance
        df = df.select_dtypes(include=[np.number])  # Only numeric columns
        return df
    except Exception as e:
        st.error(f"Dataset loading failed: {e}")
        return pd.DataFrame()

model = load_model()
df = load_data()

if df.empty or model is None:
    st.stop()

# Match features with model
required_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else df.columns[:model.n_features_in_]
df = df[[col for col in df.columns if col in required_features]]

# UI TITLE
st.title("🔐 SCADA Intrusion Detection System with Honeypot & Firewall")
st.markdown("Live monitoring of network activity, attack detection, and firewall actions.")

# Fake alert logs for now
alert_log = []
blocked_ips = set()

# Controls
st.sidebar.header("📡 System Controls")
st.sidebar.write(f"Total Packets: **{len(df)}**")
st.sidebar.write(f"Blocked IPs: **{len(blocked_ips)}**")

# Sample-based prediction
st.subheader("🧪 Predict on Sample Packet")
row_id = st.slider("Select Packet ID", 0, len(df)-1, 0)
sample = df.iloc[row_id:row_id+1]

try:
    if sample.shape[1] == model.n_features_in_:
        pred = model.predict(sample)
        conf = round(model.predict_proba(sample)[0][1] * 100, 2)
        st.success(f"Prediction: **{'Attack' if pred[0] else 'Normal'}** with Confidence: `{conf:.2f}%`")
        if pred[0] == 1:
            fake_ip = f"10.127.2.{(row_id * 17) % 255}"
            blocked_ips.add(fake_ip)
            alert_log.append({
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "src": fake_ip,
                "dst": f"172.64.{(row_id * 11) % 255}.{(row_id * 23) % 255}",
                "conf": conf
            })
    else:
        st.warning(f"Feature mismatch: model expects {model.n_features_in_} features, sample has {sample.shape[1]}")
except Exception as e:
    st.warning(f"Prediction failed: {e}")

# Simulate IDS alerts from full dataset
st.subheader("📋 Recent Alerts Table")
alert_samples = df.sample(n=200, random_state=42)
for i, row in alert_samples.iterrows():
    try:
        row_trimmed = row.dropna().values[:model.n_features_in_].reshape(1, -1)
        pred = model.predict(row_trimmed)
        if pred[0] == 1:
            conf = round(model.predict_proba(row_trimmed)[0][1] * 100, 2)
            alert_log.append({
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "src": f"10.127.{i % 255}.{(i * 3) % 255}",
                "dst": f"172.64.{(i * 5) % 255}.{(i * 7) % 255}",
                "conf": conf
            })
            blocked_ips.add(alert_log[-1]['src'])
    except:
        continue

if alert_log:
    alerts_df = pd.DataFrame(alert_log)
    for idx, row in alerts_df.iterrows():
        alert_level = "HIGH" if row["conf"] > 90 else "MEDIUM"
        st.markdown(f"""
        <div style='background-color:#ffd6d6;padding:10px;border-radius:8px;margin-bottom:10px'>
            <b>{row['timestamp']}</b> - {alert_level} alert<br>
            Source: {row['src']} → {row['dst']}<br>
            Confidence: {row['conf']:.1f}%
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No alerts to display.")

# Alert Visualization
st.subheader("🚨 Blocked Attacker IPs (Mock)")
if blocked_ips:
    alert_df = pd.DataFrame(alert_log)
    fig = px.bar(alert_df, x="src", y="conf", title="Alerted IPs by Confidence", labels={"src": "Attacker IP", "conf": "Confidence (%)"})
    st.plotly_chart(fig)
else:
    st.info("No attacker IPs yet.")

# Honeypot status
st.subheader("🛡️ Honeypot Status")
st.success("Honeypot is actively listening on Port 8080 (simulated)")

# Firewall Logs
st.subheader("🧰 Firewall Logs")
if blocked_ips:
    st.table(pd.DataFrame({"Blocked IP": list(blocked_ips)}))
else:
    st.info("No connections blocked by firewall.")

# Footer
st.markdown("---")
st.caption("SCADA IDS Dashboard for Secure Smart Grid Monitoring © 2025")
