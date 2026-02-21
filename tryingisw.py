import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
import base64
import io

# --- SETUP ---
st.set_page_config(layout='wide')
st.title("⚔️ Adversarial Testing Visualizer")

# --- LOAD DUMMY OR REAL DATA ---
@st.cache_data
def load_sample_data():
    df = pd.DataFrame(np.random.rand(1, 10), columns=[f"F{i}" for i in range(10)])
    return df

try:
    df = pd.read_csv("intermediate_combined_data.csv")
    model = pd.read_pickle("G:\Sem1\Cyberattack_on_smartGrid\ids_output\models\decision_tree_model.pkl")
    all_features = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else df.select_dtypes(include=np.number).columns.tolist()
except Exception as e:
    st.warning("⚠️ Could not load real data/model. Using dummy values instead.")
    df = load_sample_data()
    model = None
    all_features = df.columns.tolist()

# --- SELECT A SAMPLE ---
sample = df.sample(1).copy()
eps = st.slider("⚙️ Adversarial Noise ε", 0.0, 1.0, 0.1, 0.01)
adv_sample = sample.copy()

# Apply adversarial noise
for col in all_features:
    if pd.api.types.is_numeric_dtype(sample[col]):
        adv_sample[col] = adv_sample[col] + np.random.normal(0, eps)

# --- DIFF CALCULATION (NUMERIC ONLY) ---
numeric_cols = sample.select_dtypes(include=np.number).columns
diff = (adv_sample[numeric_cols] - sample[numeric_cols]).abs().T.rename(columns={sample.index[0]: "Change"})

# --- BAR CHART OF CHANGES ---
st.subheader("📊 Feature Change Magnitude")
st.bar_chart(diff)

# --- MODEL PREDICTION CONFIDENCE ---
if model:
    try:
        labels = model.classes_
        original_probs = model.predict_proba(sample[model.feature_names_in_])[0]
        adv_probs = model.predict_proba(adv_sample[model.feature_names_in_])[0]
    except Exception as e:
        labels = ["Normal", "Attack"]
        original_probs = np.random.rand(2)
        adv_probs = np.random.rand(2)
else:
    labels = ["Normal", "Attack"]
    original_probs = np.random.rand(2)
    adv_probs = np.random.rand(2)

# --- RADAR PLOT ---
radar_df = pd.DataFrame({
    "Label": labels,
    "Original": original_probs,
    "Adversarial": adv_probs
})
fig_radar = px.line_polar(
    radar_df.melt(id_vars="Label"),
    r='value',
    theta='Label',
    color='variable',
    line_close=True,
    title="🔁 Confidence Radar Plot"
)
st.plotly_chart(fig_radar, use_container_width=True)

# --- PCA VISUALIZATION ---
try:
    pca = PCA(n_components=2)
    combined = pd.concat([sample[numeric_cols], adv_sample[numeric_cols]])
    transformed = pca.fit_transform(combined)
    pca_df = pd.DataFrame(transformed, columns=["PC1", "PC2"])
    pca_df["Point"] = ["Original", "Adversarial"]
    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Point", title="🧬 PCA Projection")
    st.plotly_chart(fig_pca, use_container_width=True)
except Exception:
    st.error("❌ PCA failed — maybe not enough numeric features or invalid format.")

# --- EXPORT COMPARISON TABLE ---
csv = pd.concat([
    sample[numeric_cols].T.rename(columns={sample.index[0]: "Original"}),
    adv_sample[numeric_cols].T.rename(columns={adv_sample.index[0]: "Adversarial"}),
    diff
], axis=1)

csv_buffer = io.StringIO()
csv.to_csv(csv_buffer)
b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="adversarial_comparison.csv">🗂️ Download Adversarial Comparison CSV</a>'
st.markdown(href, unsafe_allow_html=True)
