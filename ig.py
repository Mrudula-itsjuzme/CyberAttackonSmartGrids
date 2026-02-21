import os
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

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

# Load data safely
@st.cache_data

def load_data():
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
        return df
    except Exception as e:
        st.error(f"Dataset loading failed: {e}")
        return pd.DataFrame()

model = load_model()
df = load_data()

if df.empty or model is None:
    st.stop()

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
sample = df.dropna(axis=1).iloc[row_id:row_id+1]

try:
    pred = model.predict(sample)
    conf = model.predict_proba(sample)[0][1] * 100
    st.success(f"Prediction: **{'Attack' if pred[0] else 'Normal'}** with Confidence: `{conf:.2f}%`")
    if pred[0] == 1:
        fake_ip = f"192.168.0.{row_id % 255}"
        blocked_ips.add(fake_ip)
        alert_log.append({"ip": fake_ip, "confidence": conf})
except Exception as e:
    st.warning(f"Prediction failed: {e}")

# Alert Visualization
st.subheader("🚨 Blocked Attacker IPs (Mock)")
if blocked_ips:
    alert_df = pd.DataFrame(alert_log)
    fig = px.bar(alert_df, x="ip", y="confidence", title="Alerted IPs by Confidence", labels={"ip": "Attacker IP", "confidence": "Confidence (%)"})
    st.plotly_chart(fig)
else:
    st.info("No attacker IPs yet.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by your paranoid AI overlord.")
