import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go`nfrom typing import Optional

st.set_page_config(layout="wide", page_title="Drift & Adversarial UI", page_icon="⚔️")

sns.set(style="whitegrid")

DATA_PATH = os.getenv("CYBERGRID_DATA_PATH", "intermediate_combined_data.csv")
MODEL_PATH = os.getenv("CYBERGRID_MODEL_PATH", "ids_output/models/decision_tree_model.pkl")

@st.cache_data`ndef load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        st.warning("Dataset not found, generating mock data.")
        return pd.DataFrame(np.random.rand(1000, 5), columns=["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"])
    df = pd.read_csv(DATA_PATH)
    df = df.select_dtypes(include=[np.number])
    return df

@st.cache_resource`ndef load_model() -> Optional[object]:
    if not os.path.exists(MODEL_PATH):
        st.warning("Model not found, skipping predictions.")
        return None
    return joblib.load(MODEL_PATH)

st.sidebar.title("🎛️ Controls")
st.sidebar.markdown("Live concept drift detection and adversarial simulation.")
model = load_model()
data = load_data()

st.title("📈 Concept Drift and ⚔️ Adversarial Testing")
st.markdown("This dashboard uses **real SCADA data** for live drift detection, adversarial simulations, and feature-level intelligence.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Concept Drift Visualization")
    if data.shape[1] >= 2:
        fig = px.scatter(data.sample(1000), x=data.columns[0], y=data.columns[1], color_discrete_sequence=['#1f77b4'])
        fig.update_layout(title="Feature Space Over Time", xaxis_title=data.columns[0], yaxis_title=data.columns[1])
        st.plotly_chart(fig, use_container_width=True)

        # Time series for drift
        if 'timestamp' in data.columns:
            drift_col = st.selectbox("Feature to Track Over Time", data.columns[1:])
            fig_line = px.line(data.sort_values('timestamp'), x='timestamp', y=drift_col, title=f"Drift Over Time: {drift_col}")
            st.plotly_chart(fig_line, use_container_width=True)

        st.dataframe(data.describe().T, use_container_width=True)
    else:
        st.warning("Not enough numerical features for drift visualization.")

with col2:
    st.subheader("🧪 Adversarial Test Interface")
    if model is not None:
        try:
            features = data.columns.tolist()
            selected = st.multiselect("Select features to simulate adversarial change:", features, default=features[:3])
            noise_level = st.slider("Noise level (ε)", 0.0, 1.0, 0.1, 0.01)
            if len(selected) > 0:
                sample = data.sample(1).copy()
                adversarial_sample = sample.copy()
                adversarial_sample[selected] += np.random.normal(0, noise_level, size=(1, len(selected)))

                st.markdown("### Original Sample Prediction")
                try:
                    sample_filled = sample.copy()
                    adversarial_filled = adversarial_sample.copy()

                    if hasattr(model, 'n_features_in_'):
                        expected_features = model.n_features_in_
                        sample_filled = sample_filled.iloc[:, :expected_features]
                        adversarial_filled = adversarial_filled.iloc[:, :expected_features]

                    pred_orig = model.predict(sample_filled)
                    prob_orig = model.predict_proba(sample_filled)[0]

                    st.info(f"Prediction: {'Attack' if pred_orig[0] else 'Normal'} • Confidence: {prob_orig[int(pred_orig[0])]*100:.2f}%")

                    st.markdown("### Adversarial Sample Prediction")
                    pred_adv = model.predict(adversarial_filled)
                    prob_adv = model.predict_proba(adversarial_filled)[0]
                    if pred_orig[0] != pred_adv[0]:
                        st.error(f"⚠️ Model prediction changed! Now: {'Attack' if pred_adv[0] else 'Normal'} • Confidence: {prob_adv[int(pred_adv[0])]*100:.2f}%")
                    else:
                        st.success(f"No change in prediction • Confidence: {prob_adv[int(pred_adv[0])]*100:.2f}%")

                    st.markdown("#### Radar Chart: Prediction Confidence Comparison")
                    radar = go.Figure()
                    radar.add_trace(go.Scatterpolar(r=prob_orig, theta=['Class 0', 'Class 1'], fill='toself', name='Original'))
                    radar.add_trace(go.Scatterpolar(r=prob_adv, theta=['Class 0', 'Class 1'], fill='toself', name='Adversarial'))
                    radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
                    st.plotly_chart(radar, use_container_width=True)

                    st.markdown("#### Feature Changes")
                    diff_df = pd.DataFrame({
                        'Original': sample_filled.values.flatten(),
                        'Adversarial': adversarial_filled.values.flatten(),
                        'Difference': (adversarial_filled - sample_filled).values.flatten()
                    }, index=sample_filled.columns)
                    st.dataframe(diff_df.style.format("{:.4f}"))

                    st.download_button("💾 Export Adversarial Comparison", diff_df.to_csv(index=True).encode(), file_name="adversarial_diff.csv", mime="text/csv")

                except Exception as inner:
                    st.warning(f"Prediction error: {inner}")
            else:
                st.info("Please select at least one feature to apply adversarial noise.")
        except Exception as e:
            st.error(f"Error during adversarial testing: {e}")
    else:
        st.warning("Model not loaded. Skipping adversarial simulation.")
