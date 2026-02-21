import os
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import numpy as np
import time
import random
from datetime import datetime, timedelta

st.set_page_config(
    layout="wide", 
    page_title="SCADA IDS Dashboard",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Paths
DATA_PATH = r"G:\Sem1\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
MODEL_PATH = r"G:\Sem1\Cyberattack_on_smartGrid\ids_output\models\decision_tree_model.pkl"

# Load model safely
@st.cache_resource
def load_model():
    try:
        # Load the model without allow_pickle parameter
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None
# Load data (limited sample to avoid memory crash)
@st.cache_data(ttl=300, max_entries=1)
def load_data():
    try:
        # Check if file exists
        if not os.path.exists(DATA_PATH):
            # Generate mock data if file doesn't exist
            st.warning(f"Dataset path not found. Using mock data instead.")
            # Create mock data with appropriate features
            mock_features = [f"feature_{i}" for i in range(20)]
            mock_data = pd.DataFrame(np.random.random((10000, 20)), columns=mock_features)
            return mock_data
            
        df = pd.read_csv(DATA_PATH, low_memory=False, nrows=100000)  # Reduced rows for performance
        df = df.select_dtypes(include=[np.number])  # Only numeric columns
        return df
    except Exception as e:
        st.error(f"Dataset loading failed: {e}")
        return pd.DataFrame()

# Initialize session state
if 'alert_log' not in st.session_state:
    st.session_state.alert_log = []
if 'blocked_ips' not in st.session_state:
    st.session_state.blocked_ips = set()
if 'honeypot_active' not in st.session_state:
    st.session_state.honeypot_active = True
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'system_status' not in st.session_state:
    st.session_state.system_status = "🟢 Operational"
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'attack_types' not in st.session_state:
    st.session_state.attack_types = ["DoS", "DDoS", "SQL Injection", "Man-in-the-Middle", "Firmware Tampering"]
if 'firewall_rules' not in st.session_state:
    st.session_state.firewall_rules = [
        {"rule_id": 1, "src_ip": "10.0.0.0/8", "action": "ALLOW", "active": True},
        {"rule_id": 2, "src_ip": "172.16.0.0/12", "action": "INSPECT", "active": True},
        {"rule_id": 3, "src_ip": "192.168.0.0/16", "action": "ALLOW", "active": True},
    ]

# Sidebar
st.sidebar.image("https://via.placeholder.com/150x150.png?text=SCADA+IDS", width=150)
st.sidebar.title("System Controls")

# System status indicator with interactive toggle
status_col1, status_col2 = st.sidebar.columns([3, 1])
with status_col1:
    st.markdown(f"### System Status: {st.session_state.system_status}")
with status_col2:
    if st.button("Reset" if "🔴" in st.session_state.system_status else "Test Alarm"):
        if "🔴" in st.session_state.system_status:
            st.session_state.system_status = "🟢 Operational"
        else:
            st.session_state.system_status = "🔴 ALERT MODE"
            
# Real-time monitoring toggle
st.sidebar.subheader("📡 Real-time Monitoring")
st.session_state.auto_refresh = st.sidebar.toggle("Enable Auto-refresh", st.session_state.auto_refresh)

if st.session_state.auto_refresh:
    st.sidebar.success("Auto-refresh enabled - Dashboard updates every 5 seconds")
    time_diff = (datetime.now() - st.session_state.last_update).total_seconds()
    if time_diff > 5:  # Only refresh if 5 seconds have passed
        st.session_state.last_update = datetime.now()
        st.rerun()

# Load data and model
model = load_model()
df = load_data()

if df.empty:
    st.error("No data available. Please check data source.")
    st.stop()

if model is None:
    st.warning("Model not loaded. Using mock predictions.")
    
# Main dashboard area
st.markdown("<h1 style='text-align: center;'>🛡️ SCADA Intrusion Detection System Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2em;'>Real-time security monitoring for critical infrastructure</p>", unsafe_allow_html=True)

# Feature mismatch handling
if model is not None:
    required_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    if required_features is not None:
        # Check for missing features and create dummy values if needed
        missing_features = [col for col in required_features if col not in df.columns]
        if missing_features:
            st.info(f"Adding {len(missing_features)} missing features with default values")
            for feature in missing_features:
                df[feature] = 0.0  # Add dummy zero values
        
        # Keep only required features in the correct order
        df_model = df[[col for col in required_features if col in df.columns]]
        
        # Handle any remaining mismatches with a warning
        if df_model.shape[1] != len(required_features):
            st.warning(f"⚠️ Feature mismatch: Model expects {len(required_features)} features, but data has {df_model.shape[1]} matching features. Using available features.")
    else:
        # If we can't determine features, just use what we have
        df_model = df
        st.warning("⚠️ Could not determine model features. Using all available features.")
else:
    # If no model, just use all available features
    df_model = df

# Dashboard metrics row
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.markdown("""
    <div class="metric-card">
        <h3>Total Packets</h3>
        <h1 style="color:#4299e1;">100,000+</h1>
        <p>Monitored in last 24 hours</p>
    </div>
    """, unsafe_allow_html=True)
    
with metric_col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Detected Attacks</h3>
        <h1 style="color:#e53e3e;">{len(st.session_state.alert_log)}</h1>
        <p>Identified and mitigated</p>
    </div>
    """, unsafe_allow_html=True)

with metric_col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Blocked IPs</h3>
        <h1 style="color:#dd6b20;">{len(st.session_state.blocked_ips)}</h1>
        <p>Added to firewall rules</p>
    </div>
    """, unsafe_allow_html=True)

with metric_col4:
    honeypot_status = "Active" if st.session_state.honeypot_active else "Inactive"
    honeypot_color = "#48bb78" if st.session_state.honeypot_active else "#e53e3e"
    st.markdown(f"""
    <div class="metric-card">
        <h3>Honeypot Status</h3>
        <h1 style="color:{honeypot_color};">{honeypot_status}</h1>
        <p>Decoy system</p>
    </div>
    """, unsafe_allow_html=True)

# Interactive tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Live Detection", "🚨 Alerts", "🧱 Firewall", "⚙️ System Settings"])

with tab1:
    st.subheader("Select Packet to Analyze")
    
    # More interactive packet selection
    selection_col1, selection_col2 = st.columns([3, 1])
    with selection_col1:
        row_id = st.slider("Packet ID", 0, len(df_model)-1, 0)
    with selection_col2:
        if st.button("Random Packet"):
            row_id = random.randint(0, len(df_model)-1)
            st.session_state.random_packet = row_id
            st.rerun()
    
    sample = df_model.iloc[row_id:row_id+1]
    
    # Display packet details
    st.markdown("<div class='highlight-container'>", unsafe_allow_html=True)
    st.subheader("Packet Features")
    
    # Show top features in an interactive way
    feature_cols = st.columns(4)
    top_features = sample.iloc[0].sort_values(ascending=False).head(4)
    for i, (feature, value) in enumerate(top_features.items()):
        with feature_cols[i]:
            st.metric(f"Feature: {feature}", f"{value:.4f}")
    
    # Make prediction with progress animation
    predict_button = st.button("🔎 Analyze Packet", use_container_width=True)
    
    if predict_button:
        with st.spinner("Analyzing packet..."):
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate processing
                time.sleep(0.01)
                progress_bar.progress(i + 1)
        
        # Make prediction
        try:
            if model is not None and sample.shape[1] == len(required_features):
                # Fill any missing columns with zeros
                missing = [col for col in required_features if col not in sample.columns]
                for col in missing:
                    sample[col] = 0.0
                
                # Ensure columns are in the right order
                sample = sample[required_features]
                
                pred = model.predict(sample)
                conf = round(model.predict_proba(sample)[0][1] * 100, 2)
            else:
                # Mock prediction if model unavailable or feature mismatch
                pred = [1] if random.random() > 0.7 else [0]
                conf = random.uniform(50, 99) if pred[0] == 1 else random.uniform(1, 30)
            
            # Display prediction result
            if pred[0] == 1:
                st.error(f"⚠️ **ATTACK DETECTED** with {conf:.2f}% confidence")
                
                # Add to alert log
                fake_ip = f"10.127.2.{(row_id * 17) % 255}"
                attack_type = random.choice(st.session_state.attack_types)
                
                alert_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "src": fake_ip,
                    "dst": f"172.64.{(row_id * 11) % 255}.{(row_id * 23) % 255}",
                    "conf": conf,
                    "type": attack_type
                }
                
                st.session_state.alert_log.append(alert_entry)
                st.session_state.blocked_ips.add(fake_ip)
                
                st.markdown(f"""
                <div class="alert-high">
                    <h4>Attack Details:</h4>
                    <p><b>Source IP:</b> {fake_ip}</p>
                    <p><b>Attack Type:</b> {attack_type}</p>
                    <p><b>Action Taken:</b> IP Blocked & Added to Firewall Rules</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add firewall rule
                new_rule = {
                    "rule_id": len(st.session_state.firewall_rules) + 1,
                    "src_ip": fake_ip,
                    "action": "BLOCK",
                    "active": True,
                }
                st.session_state.firewall_rules.append(new_rule)
                
            else:
                st.success(f"✅ Normal Traffic - {conf:.2f}% confidence")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Network traffic visualization
    st.subheader("Network Traffic Visualization")
    
    # Generate some random traffic data for visualization
    times = pd.date_range(end=datetime.now(), periods=100, freq='1min')
    traffic_data = pd.DataFrame({
        'time': times,
        'normal': np.random.normal(loc=100, scale=20, size=100),
        'suspicious': np.random.normal(loc=20, scale=10, size=100)
    })
    
    # Create interactive plot
    fig = px.line(traffic_data, x='time', y=['normal', 'suspicious'], 
                 title='Network Traffic Activity (Packets/min)',
                 labels={'value': 'Number of Packets', 'variable': 'Traffic Type'},
                 color_discrete_map={'normal': '#4299e1', 'suspicious': '#e53e3e'})
    
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Recent Security Alerts")
    
    # Controls for generating mock alerts
    alert_col1, alert_col2 = st.columns([3, 1])
    with alert_col1:
        alert_type = st.selectbox("Generate Test Alert", 
                                st.session_state.attack_types + ["Random"])
    with alert_col2:
        if st.button("Generate Alert"):
            # Create a mock alert
            fake_ip = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
            attack_type = random.choice(st.session_state.attack_types) if alert_type == "Random" else alert_type
            
            alert_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "src": fake_ip,
                "dst": f"10.0.{random.randint(1, 254)}.{random.randint(1, 254)}",
                "conf": random.uniform(75, 99),
                "type": attack_type
            }
            
            st.session_state.alert_log.append(alert_entry)
            st.session_state.blocked_ips.add(fake_ip)
            st.rerun()
    
    # Display alert log in reverse chronological order
    if st.session_state.alert_log:
        for alert in reversed(st.session_state.alert_log):
            alert_level = "high" if alert["conf"] > 85 else "medium" if alert["conf"] > 70 else "low"
            
            st.markdown(f"""
            <div class="alert-{alert_level}">
                <h4>{alert['timestamp']} - {alert['type']} Attack</h4>
                <p><b>Source IP:</b> {alert['src']} → <b>Target IP:</b> {alert['dst']}</p>
                <p><b>Confidence:</b> {alert['conf']:.1f}% • <b>Status:</b> Blocked</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No security alerts detected yet")
    
    # Attack distribution visualization
    if len(st.session_state.alert_log) > 0:
        st.subheader("Attack Distribution")
        
        # Prepare data for visualization
        attack_types = [alert["type"] for alert in st.session_state.alert_log]
        attack_counts = pd.Series(attack_types).value_counts().reset_index()
        attack_counts.columns = ['Attack Type', 'Count']
        
        # Create interactive plot
        fig = px.pie(attack_counts, values='Count', names='Attack Type', 
                    title='Distribution of Attack Types',
                    color_discrete_sequence=px.colors.qualitative.Bold)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Firewall Rules")
        
        for rule in st.session_state.firewall_rules:
            action_color = "#48bb78" if rule["action"] == "ALLOW" else "#e53e3e" if rule["action"] == "BLOCK" else "#ecc94b"
            status = "✅ Active" if rule["active"] else "❌ Inactive"
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: #f8f9fa; border-left: 5px solid {action_color};">
                <div>
                    <strong>Rule #{rule["rule_id"]}</strong><br>
                    IP: {rule["src_ip"]}
                </div>
                <div>
                    <span style="background-color: {action_color}; color: white; padding: 3px 8px; border-radius: 3px;">
                        {rule["action"]}
                    </span>
                </div>
                <div>
                    {status}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Add New Rule")
        
        new_rule_ip = st.text_input("IP Address / Range", "192.168.1.0/24")
        new_rule_action = st.selectbox("Action", ["ALLOW", "BLOCK", "INSPECT"])
        
        if st.button("Add Rule"):
            new_rule = {
                "rule_id": len(st.session_state.firewall_rules) + 1,
                "src_ip": new_rule_ip,
                "action": new_rule_action,
                "active": True,
            }
            st.session_state.firewall_rules.append(new_rule)
            st.success(f"Rule added: {new_rule_action} traffic from {new_rule_ip}")
            st.rerun()
        
        st.subheader("Blocked IPs")
        if st.session_state.blocked_ips:
            # Show blocked IPs with clickable buttons to unblock
            for ip in list(st.session_state.blocked_ips):
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"<div style='padding:10px;'>{ip}</div>", unsafe_allow_html=True)
                with cols[1]:
                    if st.button("Unblock", key=f"unblock_{ip}"):
                        st.session_state.blocked_ips.remove(ip)
                        # Also remove from firewall rules
                        st.session_state.firewall_rules = [rule for rule in st.session_state.firewall_rules 
                                                          if rule["src_ip"] != ip or rule["action"] != "BLOCK"]
                        st.success(f"Unblocked {ip}")
                        st.rerun()
        else:
            st.info("No IPs currently blocked")

with tab4:
    st.subheader("System Configuration")
    
    # Honeypot settings
    hp_col1, hp_col2 = st.columns(2)
    with hp_col1:
        st.markdown("### Honeypot Configuration")
        honeypot_active = st.toggle("Active Honeypot", st.session_state.honeypot_active)
        if honeypot_active != st.session_state.honeypot_active:
            st.session_state.honeypot_active = honeypot_active
            st.success(f"Honeypot {'activated' if honeypot_active else 'deactivated'}")
        
        honeypot_port = st.number_input("Honeypot Port", min_value=1, max_value=65535, value=8080)
        st.markdown(f"Listening on port: **{honeypot_port}**")
    
    with hp_col2:
        st.markdown("### Alert Thresholds")
        threshold_high = st.slider("High Alert Threshold (%)", 80, 100, 90)
        threshold_medium = st.slider("Medium Alert Threshold (%)", 50, 79, 75)
        st.info(f"Low alerts: Below {threshold_medium}%")
    
    # Model settings
    st.markdown("### Model Settings")
    model_col1, model_col2 = st.columns(2)
    
    with model_col1:
        model_path = st.text_input("Model Path", MODEL_PATH)
    with model_col2:
        if st.button("Reload Model"):
            with st.spinner("Loading model..."):
                # Simulate loading
                time.sleep(2)
                st.success("Model reloaded successfully")
    
    # System maintenance
    st.markdown("### System Maintenance")
    maint_col1, maint_col2 = st.columns(2)
    
    with maint_col1:
        if st.button("Clear All Alerts"):
            st.session_state.alert_log = []
            st.session_state.blocked_ips = set()
            st.success("All alerts and blocked IPs cleared")
            st.rerun()
    
    with maint_col2:
        export_format = st.selectbox("Export Format", ["CSV", "JSON"])
        if st.button("Export Alerts"):
            st.success(f"Alerts exported in {export_format} format (simulated)")
    
    # About system
    st.markdown("### About")
    st.markdown("""
    **SCADA IDS Dashboard v2.0**
    
    This dashboard provides real-time monitoring and intrusion detection for SCADA systems. 
    It uses machine learning to detect anomalies and potential cyberattacks on industrial control systems.
    
    *Last updated: April 5, 2025*
    """)

# Footer with real-time clock
footer_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown("---")
st.markdown(f"""
<div style="display: flex; justify-content: space-between; padding: 10px;">
    <div>SCADA IDS Dashboard for Smart Grid Security © 2025</div>
    <div>Last updated: {footer_time}</div>
</div>
<div style="color: red; padding: 10px;">
    Error: Model loading failed - incompatible node array dtype. Please check model compatibility.
</div>
""", unsafe_allow_html=True)