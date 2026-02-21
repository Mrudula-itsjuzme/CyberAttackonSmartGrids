import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc, roc_curve
import pickle
import os
import time

# Set page title and configuration
st.set_page_config(
    page_title="Smart Grid Security Dashboard",
    page_icon="🔐",
    layout="wide"
)

# Function to load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("G:\\Sem1\\Cyberattack_on_smartGrid\\intermediate_combined_data.csv")
    except FileNotFoundError:
        st.warning("⚠️ Dataset file not found! Using sample data instead.")
        df = pd.DataFrame({
            'voltage': np.random.normal(220, 5, 1000),
            'current': np.random.normal(10, 1, 1000),
            'power': np.random.normal(2200, 100, 1000),
            'frequency': np.random.normal(50, 0.5, 1000),
            'phase': np.random.normal(0, 0.1, 1000),
        })
        anomaly_indices = np.random.choice(len(df), size=50, replace=False)
        df.loc[anomaly_indices, 'voltage'] *= 1.5
        df.loc[anomaly_indices, 'current'] *= 1.3
    return df

# Function to preprocess data
def preprocess_data(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df_numeric = df[numeric_cols].dropna()
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=numeric_cols)
    return df_scaled

# Function to train anomaly detection model
def get_anomaly_model(df):
    model_path = "isolation_forest_model.pkl"
    df_scaled = preprocess_data(df)
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        st.success("✅ Loaded existing anomaly detection model")
    else:
        with st.spinner("Training anomaly detection model..."):
            model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
            model.fit(df_scaled)
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
        st.success("✅ Trained and saved new anomaly detection model")
    return model, df_scaled

# Function to train classifier
def train_classifier(df_scaled):
    if 'Anomaly' not in df_scaled.columns:
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        df_scaled['Anomaly'] = model.fit_predict(df_scaled)
    X = df_scaled.drop(columns=['Anomaly'])
    y = df_scaled['Anomaly']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return model, X_test, y_test, y_pred, y_proba

# Main dashboard
st.title("🔐 Smart Grid Security Dashboard")
option = st.sidebar.selectbox("Choose Section", ["Intrusion Detection", "Firewall & Honeypot Logs", "Real-time Monitoring", "Model Performance"])

df = load_data()

if option == "Intrusion Detection":
    st.header("🛡 Intrusion Detection System")
    model, df_scaled = get_anomaly_model(df)
    anomalies = model.predict(df_scaled)
    df_scaled["Anomaly"] = anomalies
    anomaly_count = sum(anomalies == -1)
    st.metric(label="Total Anomalies", value=f"{anomaly_count}")
    feature = st.selectbox("Select feature to visualize", df_scaled.columns[:-1])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df_scaled.index, y=df_scaled[feature], hue=df_scaled["Anomaly"], palette={1: "blue", -1: "red"}, ax=ax)
    plt.title(f"Anomaly Detection in {feature}")
    st.pyplot(fig)

elif option == "Firewall & Honeypot Logs":
    st.header("🔥 Firewall & Honeypot Logs")
    st.write("View real-time attack logs and firewall actions.")
    attack_logs = pd.read_csv("firewall_logs.csv")
    st.dataframe(attack_logs)

elif option == "Real-time Monitoring":
    st.header("📊 Real-time Smart Grid Monitoring")
    st.warning("This is a simulation of real-time monitoring.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Voltage", f"{np.random.normal(220, 3):.1f} V")
    with col2:
        st.metric("Current", f"{np.random.normal(10, 0.5):.1f} A")
    with col3:
        st.metric("Frequency", f"{np.random.normal(50, 0.1):.2f} Hz")

elif option == "Model Performance":
    st.header("🧠 Anomaly Detection Model Performance")
    model, df_scaled = get_anomaly_model(df)
    classifier, X_test, y_test, y_pred, y_proba = train_classifier(df_scaled)
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    st.pyplot(fig)
    if y_proba is not None:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.legend()
        st.pyplot(fig)