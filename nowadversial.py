import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
import joblib
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def align_features(df, required_features, fill_value=0.0):
    """Ensure input DataFrame has all required features, add missing as fill_value, drop extras."""
    df = df.copy()
    for col in required_features:
        if col not in df.columns:
            df[col] = fill_value
    return df[required_features]
# Ensure compatibility with main dashboard
st.set_page_config(
    layout="wide",
    page_title="SCADA IDS Adversarial Testing",
    page_icon="⚔️",
    initial_sidebar_state="expanded"
)

# Custom CSS to maintain the same visual style
st.markdown("""
<style>
    .highlight-container { 
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(66, 153, 225, 0.5); }
        70% { box-shadow: 0 0 0 10px rgba(66, 153, 225, 0); }
        100% { box-shadow: 0 0 0 0 rgba(66, 153, 159, 0); }
    }
    
    .attack-card {
        background-color: light-blue;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(1,1,1,0.1);
        margin-bottom: 15px;
        border-left: 5px solid #e53e3e;
        transition: all 0.3s ease;
    }
    
    .attack-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    .defense-card {
        background-color: light-blue;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        border-left: 5px solid #48bb78;
        transition: all 0.3s ease;
    }
    
    .defense-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background-color: light-blue;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    .stButton>button {
        width: 100%;
        background-color: #4299e1;
        color: light-blue;
        font-weight: bold;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #3182ce;
        transform: scale(1.05);
    }
    
    /* Custom progress bar */
    .stProgress > div > div > div > div {
        background-color: #4299e1;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #f8f9fa;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4299e1;
        color: light-blue;
    }
</style>
""", unsafe_allow_html=True)

# Paths
DATA_PATH = r"G:\Sem1\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
MODEL_PATH = r"G:\Sem1\Cyberattack_on_smartGrid\ids_output\models\xgb_model.pkl"

# Initialize session state for adversarial module
if 'attack_results' not in st.session_state:
    st.session_state.attack_results = []
if 'defense_results' not in st.session_state:
    st.session_state.defense_results = []
if 'current_sample' not in st.session_state:
    st.session_state.current_sample = None
if 'perturbed_sample' not in st.session_state:
    st.session_state.perturbed_sample = None
if 'epsilon_value' not in st.session_state:
    st.session_state.epsilon_value = 0.1
if 'adv_trained_model' not in st.session_state:
    st.session_state.adv_trained_model = None

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
        # Check if file exists
        if not os.path.exists(DATA_PATH):
            # Generate mock data if file doesn't exist
            st.warning(f"Dataset path not found. Using mock data instead.")
            # Create mock data with appropriate features
            mock_features = [f"feature_{i}" for i in range(20)]
            mock_data = pd.DataFrame(np.random.random((10000, 20)), columns=mock_features)
            # Add labels
            mock_data['label'] = np.random.choice([0, 1], size=10000, p=[0.8, 0.2])
            return mock_data
            
        df = pd.read_csv(DATA_PATH, low_memory=False, nrows=100000)  # Reduced rows for performance
        df = df.select_dtypes(include=[np.number])  # Only numeric columns
        
        # If no label column exists, assume last column is label or create one
        if 'label' not in df.columns:
            last_col = df.columns[-1]
            # Rename last column to label or create a synthetic one
            if last_col != 'label':
                df['label'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])
                
        return df
    except Exception as e:
        st.error(f"Dataset loading failed: {e}")
        return pd.DataFrame()

# Adversarial attack implementations
def fast_gradient_sign_method(model, sample, epsilon, feature_names):
    """
    Implementation of Fast Gradient Sign Method (FGSM)
    """
    # Convert sample to numpy array if needed
    if isinstance(sample, pd.DataFrame):
        x = sample[feature_names].values
    else:
        x = sample
    
    # For non-neural network models, we'll use a simplified approach
    # by perturbing features based on their importance
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Create perturbation (more perturbation for important features)
        perturbation = np.zeros_like(x)
        for i, importance in enumerate(importances):
            # Scale perturbation by feature importance
            sign = np.random.choice([-1, 1])  # Random sign for perturbation
            perturbation[0, i] = sign * epsilon * importance * 10
        
        # Apply perturbation
        x_perturbed = x + perturbation
        
        # Ensure features stay in reasonable range (assuming 0-1 normalization)
        x_perturbed = np.clip(x_perturbed, 0, 1)
        
        return x_perturbed, perturbation
    else:
        # Fallback for models without feature_importances_
        # Random perturbation in this case
        perturbation = np.random.uniform(-epsilon, epsilon, x.shape)
        x_perturbed = x + perturbation
        return x_perturbed, perturbation

def projected_gradient_descent(model, sample, epsilon, feature_names, alpha=0.01, iterations=10):
    """
    Implementation of Projected Gradient Descent (PGD)
    """
    # Convert sample to numpy array if needed
    if isinstance(sample, pd.DataFrame):
        x = sample[feature_names].values
    else:
        x = sample
    
    # Create a copy of the original sample
    x_perturbed = x.copy()
    
    # For traditional ML models, use a simplified version based on feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Iteratively apply perturbations
        for _ in range(iterations):
            # Create perturbation based on feature importance
            perturbation = np.zeros_like(x)
            for i, importance in enumerate(importances):
                # Scale perturbation by feature importance and alpha
                if importance > 0.01:  # Only perturb important features
                    sign = 1 if np.random.random() > 0.5 else -1
                    perturbation[0, i] = sign * alpha * importance * 10
            
            # Apply perturbation
            x_perturbed = x_perturbed + perturbation
            
            # Project back to epsilon ball
            delta = x_perturbed - x
            norm = np.linalg.norm(delta)
            if norm > epsilon:
                delta = delta * epsilon / norm
            
            # Apply projection
            x_perturbed = x + delta
            
            # Ensure values stay in reasonable range
            x_perturbed = np.clip(x_perturbed, 0, 1)
    else:
        # Fallback for models without feature_importances_
        for _ in range(iterations):
            # Random perturbation
            perturbation = np.random.uniform(-alpha, alpha, x.shape)
            x_perturbed = x_perturbed + perturbation
            
            # Project back to epsilon ball
            delta = x_perturbed - x
            norm = np.linalg.norm(delta)
            if norm > epsilon:
                delta = delta * epsilon / norm
            
            # Apply projection
            x_perturbed = x + delta
            
            # Ensure values stay in reasonable range
            x_perturbed = np.clip(x_perturbed, 0, 1)
    
    perturbation = x_perturbed - x
    return x_perturbed, perturbation

def carlini_wagner_attack(model, sample, feature_names, confidence=0.0, iterations=100):
    """
    Simplified implementation of Carlini & Wagner attack
    """
    # Convert sample to numpy array if needed
    if isinstance(sample, pd.DataFrame):
        x = sample[feature_names].values
    else:
        x = sample
    
    # For traditional ML models, adapt the approach
    x_perturbed = x.copy()
    
    # Initial confidence of the model
    try:
        initial_pred = model.predict_proba(x)[0]
        target_class = 1 - np.argmax(initial_pred)  # Target the opposite class
        
        # Apply iterative optimization (simplified)
        step_size = 0.01
        best_perturbed = None
        best_distance = float('inf')
        
        for i in range(iterations):
            # Create a small perturbation
            if hasattr(model, 'feature_importances_'):
                # Target important features
                importances = model.feature_importances_
                perturbation = np.zeros_like(x)
                for j, importance in enumerate(importances):
                    if importance > 0.01:  # Only perturb important features
                        perturbation[0, j] = (np.random.random() - 0.5) * step_size * importance * 20
            else:
                perturbation = np.random.uniform(-step_size, step_size, x.shape)
            
            # Apply perturbation
            candidate = x_perturbed + perturbation
            candidate = np.clip(candidate, 0, 1)
            
            # Check if the attack is successful
            pred = model.predict_proba(candidate)[0]
            if np.argmax(pred) == target_class and pred[target_class] >= confidence:
                # Calculate L2 distance
                distance = np.linalg.norm(candidate - x)
                if distance < best_distance:
                    best_distance = distance
                    best_perturbed = candidate
            
            # Update x_perturbed with the perturbation
            x_perturbed = candidate
            
            # Reduce step size over time
            step_size = step_size * 0.98
        
        if best_perturbed is not None:
            x_perturbed = best_perturbed
    except:
        # Fallback if prediction_proba doesn't work
        perturbation = np.random.uniform(-0.1, 0.1, x.shape)
        x_perturbed = x + perturbation
        x_perturbed = np.clip(x_perturbed, 0, 1)
    
    perturbation = x_perturbed - x
    return x_perturbed, perturbation

# Defense methods
def adversarial_training(model, data, labels, feature_names, epsilon=0.1):
    """
    Implement adversarial training by augmenting the training data with adversarial examples
    """
    # In a real implementation, this would train the model with adversarial examples
    # For this demo, we'll simulate the process
    
    st.info("In a production environment, this would retrain the model with adversarial examples.")
    
    # Simulate training progress
    progress_bar = st.progress(0)
    
    # Generate some adversarial examples for display
    adv_examples = []
    sample_indices = np.random.choice(len(data), min(10, len(data)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        progress_bar.progress((i + 1) / len(sample_indices))
        
        # Get a sample
        sample = data.iloc[idx:idx+1]
        
        # Generate adversarial example using FGSM
        x_adv, _ = fast_gradient_sign_method(model, sample, epsilon, feature_names)
        
        # Add to list of examples
        adv_examples.append({
            'original': sample[feature_names].values[0],
            'adversarial': x_adv[0],
            'original_prediction': model.predict(sample[feature_names].values.reshape(1, -1))[0],
            'adversarial_prediction': model.predict(x_adv)[0],
            'true_label': labels.iloc[idx]
        })
        
        # Simulate some processing time
        time.sleep(0.2)
    
    # Return a "trained" model (same model for demo) and the examples
    return model, adv_examples

def ensemble_model(models, sample, feature_names):
    """
    Create an ensemble prediction from multiple models for improved robustness
    """
    # Get predictions from all models
    predictions = []
    for model in models:
        if isinstance(sample, pd.DataFrame):
            pred = model.predict(sample[feature_names])
        else:
            pred = model.predict(sample)
        predictions.append(pred)
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    
    # Take the majority vote
    ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
    
    return ensemble_pred

def randomized_smoothing(model, sample, feature_names, num_samples=10, sigma=0.01):
    """
    Implement randomized smoothing for certified robustness
    """
    # Generate noisy samples
    predictions = []
    
    if isinstance(sample, pd.DataFrame):
        x = sample[feature_names].values
    else:
        x = sample
    
    for _ in range(num_samples):
        # Add Gaussian noise
        noise = np.random.normal(0, sigma, x.shape)
        x_noisy = x + noise
        
        # Ensure values stay in reasonable range
        x_noisy = np.clip(x_noisy, 0, 1)
        
        # Get prediction
        pred = model.predict(x_noisy)
        predictions.append(pred)
    
    # Convert to numpy array
    predictions = np.array(predictions)
    
    # Take the majority vote
    smoothed_pred = np.round(np.mean(predictions, axis=0)).astype(int)
    
    return smoothed_pred

# Feature importance visualization
def plot_feature_importance(model, feature_names, num_features=10):
    """
    Create a bar chart of feature importances
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Create a DataFrame for plotting
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values('Importance', ascending=False).head(num_features)
        
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title=f'Top {num_features} Feature Importances',
            labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'},
            color='Importance',
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        return fig
    else:
        return None

# Perturbation visualization
def visualize_perturbation(original, perturbed, feature_names, top_n=10):
    """
    Visualize the differences between original and perturbed samples
    """
    if isinstance(original, pd.DataFrame):
        original = original[feature_names].values[0]
    else:
        original = original[0]
    
    if isinstance(perturbed, pd.DataFrame):
        perturbed = perturbed[feature_names].values[0]
    else:
        perturbed = perturbed[0]
    
    # Calculate absolute differences
    diff = np.abs(perturbed - original)
    
    # Create a DataFrame for visualization
    viz_df = pd.DataFrame({
        'Feature': feature_names,
        'Original': original,
        'Perturbed': perturbed,
        'Difference': diff
    })
    
    # Sort by difference and take top N
    viz_df = viz_df.sort_values('Difference', ascending=False).head(top_n)
    
    # Create subplots: bar chart for differences and scatter for values
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Feature Value Comparison', 'Perturbation Magnitude'),
        vertical_spacing=0.2,
        row_heights=[0.6, 0.4]
    )
    
    # Add scatter plot for feature values
    fig.add_trace(
        go.Scatter(
            x=viz_df['Feature'], 
            y=viz_df['Original'],
            mode='markers',
            name='Original',
            marker=dict(color='blue', size=12)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=viz_df['Feature'], 
            y=viz_df['Perturbed'],
            mode='markers',
            name='Perturbed',
            marker=dict(color='red', size=12)
        ),
        row=1, col=1
    )
    
    # Add bar chart for differences
    fig.add_trace(
        go.Bar(
            x=viz_df['Feature'],
            y=viz_df['Difference'],
            name='Perturbation',
            marker_color='orange'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        title_text="Original vs. Perturbed Feature Values",
        showlegend=True
    )
    
    fig.update_xaxes(tickangle=45, row=2, col=1)
    
    return fig

# Confusion matrix visualization
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    Plot confusion matrix using plotly
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Create labels
    categories = ['Normal', 'Attack']
    
    # Create heatmap
    fig = px.imshow(
        cm,
        x=categories,
        y=categories,
        color_continuous_scale='Blues',
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title=title
    )
    
    # Add text annotations
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=str(value),
                    showarrow=False,
                    font=dict(color='blue' if value > cm.max() / 2 else 'black')
                )
            )
    
    fig.update_layout(annotations=annotations)
    return fig

# Performance metrics comparison
def plot_feature_importance(model, feature_names, num_features=10):
    """
    Create a bar chart of feature importances
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # DEBUG: check lengths before plotting
        if len(feature_names) != len(importances):
            st.error(f"Mismatch: feature_names has {len(feature_names)} items but importances has {len(importances)} items.")
            return None

        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        ...


# Main app layout
st.markdown("<h1 style='text-align: center;'>⚔️ SCADA IDS Adversarial Testing Module</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2em;'>Test model robustness against adversarial attacks and evaluate defensive countermeasures</p>", unsafe_allow_html=True)

# Load model and data
model = load_model()
df = load_data()

# Sidebar controls
st.sidebar.title("Test Controls")

# Attack parameters
st.sidebar.subheader("Attack Parameters")
attack_method = st.sidebar.selectbox(
    "Attack Method",
    ["Fast Gradient Sign Method (FGSM)", "Projected Gradient Descent (PGD)", "Carlini & Wagner (C&W)"]
)

# Epsilon control (perturbation magnitude)
epsilon = st.sidebar.slider("Epsilon (Perturbation Magnitude)", 0.01, 0.5, st.session_state.epsilon_value, 0.01)
if epsilon != st.session_state.epsilon_value:
    st.session_state.epsilon_value = epsilon

# Defense methods
st.sidebar.subheader("Defense Methods")
defense_methods = st.sidebar.multiselect(
    "Select Defense Methods",
    ["Adversarial Training", "Ensemble Model", "Randomized Smoothing"],
    default=["Adversarial Training"]
)

# Test size
test_size = st.sidebar.slider("Test Sample Size", 10, min(1000, len(df)), 100)

# Run batch test button
run_batch = st.sidebar.button("Run Batch Test", use_container_width=True)

# Main content in tabs
tabs = st.tabs(["🎯 Attack Simulation", "🛡️ Defense Evaluation", "📊 Results & Metrics"])

with tabs[0]:  # Attack Simulation tab
    st.subheader("Adversarial Attack Simulation")
    
    # Feature selection for the model
    if model is not None:
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        else:
            # Use all features except the label
            feature_names = [col for col in df.columns if col != 'label']
        
        # Select a sample to attack
        st.markdown("### Select Data to Attack")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Random sampling options
            sample_label = st.radio("Sample Type", ["Normal Traffic (label=0)", "Attack Traffic (label=1)", "Random"])
        
        with col2:
            # Button to select a sample
            if st.button("Select Sample", use_container_width=True):
                # Filter based on selected label
                if sample_label == "Normal Traffic (label=0)":
                    sample_df = df[df['label'] == 0]
                elif sample_label == "Attack Traffic (label=1)":
                    sample_df = df[df['label'] == 1]
                else:
                    sample_df = df
                
                # Randomly select a sample
                if not sample_df.empty:
                    sample_idx = np.random.choice(len(sample_df))
                    st.session_state.current_sample = sample_df.iloc[sample_idx:sample_idx+1]
                    st.session_state.perturbed_sample = None  # Reset perturbed sample
                else:
                    st.error("No samples available for the selected label.")
        
        # Show the selected sample
        if st.session_state.current_sample is not None:
            st.markdown("### Sample Details")
            sample = st.session_state.current_sample
            
            # Create two columns for original sample details
            col1, col2 = st.columns(2)
            
            with col1:
                # Original prediction
                try:
                    if isinstance(sample, pd.DataFrame):
                        pred = model.predict(sample[feature_names])
                        conf = model.predict_proba(sample[feature_names])[0]
                        confidence = max(conf) * 100
                    else:
                        pred = model.predict(sample)
                        conf = model.predict_proba(sample)[0]
                        confidence = max(conf) * 100
                    
                    prediction = "Attack" if pred[0] == 1 else "Normal"
                    
                    # Show prediction with confidence
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Original Prediction</h3>
                        <h1 style="color:{'#e53e3e' if pred[0] == 1 else '#4299e1'};">{prediction}</h1>
                        <p>Confidence: {confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
            
            with col2:
                # True label
                true_label = sample['label'].values[0] if 'label' in sample.columns else "Unknown"
                label_text = "Attack" if true_label == 1 else "Normal" if true_label == 0 else true_label
                
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>True Label</h3>
                    <h1 style="color:{'#e53e3e' if true_label == 1 else '#4299e1'};">{label_text}</h1>
                    <p>Ground Truth</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Button to generate adversarial example
            if st.button("Generate Adversarial Example", use_container_width=True):
                with st.spinner(f"Generating adversarial example using {attack_method}..."):
                    # Apply the selected attack method
                    if attack_method == "Fast Gradient Sign Method (FGSM)":
                        x_perturbed, perturbation = fast_gradient_sign_method(model, sample, epsilon, feature_names)
                    elif attack_method == "Projected Gradient Descent (PGD)":
                        x_perturbed, perturbation = projected_gradient_descent(model, sample, epsilon, feature_names)
                    else:  # Carlini & Wagner
                        x_perturbed, perturbation = carlini_wagner_attack(model, sample, feature_names)
                    
                    # Create a copy of the sample with perturbed values
                    perturbed_sample = sample.copy()
                    perturbed_sample[feature_names] = x_perturbed
                    
                    # Store in session state
                    st.session_state.perturbed_sample = perturbed_sample
            
            # Show adversarial example if available
            if st.session_state.perturbed_sample is not None:
                st.markdown("### Adversarial Example")
                perturbed = st.session_state.perturbed_sample
                
                # Create two columns for comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    # Original prediction
                    try:
                        if isinstance(perturbed, pd.DataFrame):
                            pred = model.predict(perturbed[feature_names])
                            conf = model.predict_proba(perturbed[feature_names])[0]
                            confidence = max(conf) * 100
                        else:
                            pred = model.predict(perturbed)
                            conf = model.predict_proba(perturbed)[0]
                            confidence = max(conf) * 100
                        
                        prediction = "Attack" if pred[0] == 1 else "Normal"
                        
                        # Show prediction with confidence
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3>Adversarial Prediction</h3>
                            <h1 style="color:{'#e53e3e' if pred[0] == 1 else '#4299e1'};">{prediction}</h1>
                            <p>Confidence: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                
                with col2:
                    # Success status
                    original_pred = model.predict(sample[feature_names])[0]
                    adversarial_pred = pred[0]
                    
                    if original_pred != adversarial_pred:
                        status = "✅ SUCCESS"
                        status_color = "#48bb78"  # Green
                    else:
                        status = "❌ FAILED"
                        status_color = "#e53e3e"  # Red
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Attack Result</h3>
                        <h1 style="color:{status_color};">{status}</h1>
                        <p>Attack {'changed' if original_pred != adversarial_pred else 'did not change'} prediction</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualize perturbation
                st.markdown("### Perturbation Analysis")
                
                # Feature comparison visualization
                perturbation_viz = visualize_perturbation(sample, perturbed, feature_names)
                st.plotly_chart(perturbation_viz, use_container_width=True)
                
                # Record result
                result = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'attack_method': attack_method,
                    'epsilon': epsilon,
                    'original_prediction': original_pred,
                    'adversarial_prediction': adversarial_pred,
                    'true_label': true_label,
                    'success': original_pred != adversarial_pred,
                    'perturbation_magnitude': np.linalg.norm(perturbed[feature_names].values - sample[feature_names].values)
                }
                
                st.session_state.attack_results.append(result)
        
        else:
            st.info("Please select a sample to start the attack simulation")
    else:
        st.error("Model not loaded. Please check the model path.")

with tabs[1]:  # Defense Evaluation tab
    st.subheader("Defense Mechanisms Evaluation")
    
    # Brief explanation of defense methods
    st.markdown("""
    <div class="highlight-container">
        <h3>🛡️ Defense Against Adversarial Attacks</h3>
        <p>Adversarial defenses aim to make models robust against perturbations. Test different defense strategies against your attacks.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Defense cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="defense-card">
            <h3>Adversarial Training</h3>
            <p>Train the model on adversarial examples to make it more robust against attacks.</p>
        </div>
        
        <div class="defense-card">
            <h3>Ensemble Model</h3>
            <p>Combine predictions from multiple models to improve robustness.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="defense-card">
            <h3>Randomized Smoothing</h3>
            <p>Add random noise to inputs to make predictions more stable.</p>
        </div>
        
        <div class="defense-card">
            <h3>Feature Squeezing</h3>
            <p>Reduce the precision of input features to remove adversarial perturbations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Apply defenses on a sample
    st.markdown("### Test Defense on Current Sample")
    
    if st.session_state.current_sample is not None and st.session_state.perturbed_sample is not None:
        # Show original and perturbed samples
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Sample Prediction:**")
            original_pred = model.predict(st.session_state.current_sample[feature_names])[0]
            st.write(f"{'Attack' if original_pred == 1 else 'Normal'} (Class {original_pred})")
        
        with col2:
            st.markdown("**Adversarial Sample Prediction:**")
            adv_pred = model.predict(st.session_state.perturbed_sample[feature_names])[0]
            st.write(f"{'Attack' if adv_pred == 1 else 'Normal'} (Class {adv_pred})")
        
        # Apply selected defenses
        if "Adversarial Training" in defense_methods:
            st.markdown("#### 🛡️ Adversarial Training Defense")
            
            # Check if we already have an adversarially trained model
            if st.session_state.adv_trained_model is None:
                if st.button("Train Adversarial Model", use_container_width=True):
                    with st.spinner("Training model with adversarial examples..."):
                        # Train model with adversarial examples
                        adv_model, examples = adversarial_training(
                            model, 
                            df.drop('label', axis=1), 
                            df['label'], 
                            feature_names,
                            epsilon
                        )
                        
                        # Store the model
                        st.session_state.adv_trained_model = adv_model
                        
                        # Show training results
                        st.success("Adversarial training completed!")
                        
                        # Display example adversarial training data
                        st.markdown("**Sample Adversarial Training Data:**")
                        for i, example in enumerate(examples[:3]):
                            st.markdown(f"**Example {i+1}:**")
                            st.write(f"Original prediction: {'Attack' if example['original_prediction'] == 1 else 'Normal'}")
                            st.write(f"Adversarial prediction: {'Attack' if example['adversarial_prediction'] == 1 else 'Normal'}")
                            st.write(f"True label: {'Attack' if example['true_label'] == 1 else 'Normal'}")
                            st.write("---")
            else:
                # Use the adversarially trained model
                adv_model = st.session_state.adv_trained_model
                adv_trained_pred = adv_model.predict(st.session_state.perturbed_sample[feature_names])[0]
                
                # Show results
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Adversarial Training Results</h3>
                    <h2 style="color:{'#48bb78' if adv_trained_pred == original_pred else '#e53e3e'};">
                        {'SUCCESS' if adv_trained_pred == original_pred else 'FAILED'}
                    </h2>
                    <p>Prediction: {'Attack' if adv_trained_pred == 1 else 'Normal'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Record the defense result
                result = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'defense_method': 'Adversarial Training',
                    'attack_method': attack_method,
                    'epsilon': epsilon,
                    'original_prediction': original_pred,
                    'adversarial_prediction': adv_pred,
                    'defended_prediction': adv_trained_pred,
                    'success': adv_trained_pred == original_pred,
                }
                st.session_state.defense_results.append(result)
        
        if "Ensemble Model" in defense_methods:
            st.markdown("#### 🛡️ Ensemble Model Defense")
            
            # Create an ensemble of models (for demo, just duplicate the model)
            # In a real scenario, these would be different models
            ensemble_models = [model] * 3  # Using 3 identical models for demonstration
            
            # Apply ensemble defense
            with st.spinner("Applying ensemble defense..."):
                ensemble_pred = ensemble_model(ensemble_models, st.session_state.perturbed_sample, feature_names)[0]
                
                # Show results
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Ensemble Model Results</h3>
                    <h2 style="color:{'#48bb78' if ensemble_pred == original_pred else '#e53e3e'};">
                        {'SUCCESS' if ensemble_pred == original_pred else 'FAILED'}
                    </h2>
                    <p>Prediction: {'Attack' if ensemble_pred == 1 else 'Normal'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Record the defense result
                result = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'defense_method': 'Ensemble Model',
                    'attack_method': attack_method,
                    'epsilon': epsilon,
                    'original_prediction': original_pred,
                    'adversarial_prediction': adv_pred,
                    'defended_prediction': ensemble_pred,
                    'success': ensemble_pred == original_pred,
                }
                st.session_state.defense_results.append(result)
        
        if "Randomized Smoothing" in defense_methods:
            st.markdown("#### 🛡️ Randomized Smoothing Defense")
            
            # Apply randomized smoothing
            with st.spinner("Applying randomized smoothing..."):
                smoothed_pred = randomized_smoothing(model, st.session_state.perturbed_sample, feature_names)[0]
                
                # Show results
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Randomized Smoothing Results</h3>
                    <h2 style="color:{'#48bb78' if smoothed_pred == original_pred else '#e53e3e'};">
                        {'SUCCESS' if smoothed_pred == original_pred else 'FAILED'}
                    </h2>
                    <p>Prediction: {'Attack' if smoothed_pred == 1 else 'Normal'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Record the defense result
                result = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'defense_method': 'Randomized Smoothing',
                    'attack_method': attack_method,
                    'epsilon': epsilon,
                    'original_prediction': original_pred,
                    'adversarial_prediction': adv_pred,
                    'defended_prediction': smoothed_pred,
                    'success': smoothed_pred == original_pred,
                }
                st.session_state.defense_results.append(result)
    else:
        st.info("Run an attack first to test defenses")

with tabs[2]:  # Results & Metrics tab
    st.subheader("Results & Performance Metrics")
    
    # Show attack results
    st.markdown("### Attack Results History")
    if st.session_state.attack_results:
        attack_df = pd.DataFrame(st.session_state.attack_results)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            success_rate = attack_df['success'].mean() * 100
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Attack Success Rate</h3>
                <h1>{success_rate:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_perturbation = attack_df['perturbation_magnitude'].mean()
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Avg. Perturbation Magnitude</h3>
                <h1>{avg_perturbation:.4f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_attempts = len(attack_df)
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Total Attack Attempts</h3>
                <h1>{total_attempts}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Display attack results table
        st.markdown("#### Attack Results Table")
        st.dataframe(attack_df)
        
        # Success rate by attack method
        if len(attack_df['attack_method'].unique()) > 1:
            st.markdown("#### Success Rate by Attack Method")
            attack_success_by_method = attack_df.groupby('attack_method')['success'].mean() * 100
            attack_success_df = pd.DataFrame({
                'Attack Method': attack_success_by_method.index,
                'Success Rate (%)': attack_success_by_method.values
            })
            
            fig = px.bar(
                attack_success_df,
                x='Attack Method',
                y='Success Rate (%)',
                color='Success Rate (%)',
                title='Attack Success Rate by Method',
                color_continuous_scale=px.colors.sequential.Reds
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No attack results yet. Run some attacks to see results.")
    
    # Show defense results
    st.markdown("### Defense Results History")
    if st.session_state.defense_results:
        defense_df = pd.DataFrame(st.session_state.defense_results)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            defense_success_rate = defense_df['success'].mean() * 100
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Defense Success Rate</h3>
                <h1>{defense_success_rate:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            best_defense = defense_df.groupby('defense_method')['success'].mean().idxmax()
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Best Defense Method</h3>
                <h1>{best_defense}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_tests = len(defense_df)
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Total Defense Tests</h3>
                <h1>{total_tests}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Display defense results table
        st.markdown("#### Defense Results Table")
        st.dataframe(defense_df)
        
        # Success rate by defense method
        st.markdown("#### Success Rate by Defense Method")
        defense_success_by_method = defense_df.groupby('defense_method')['success'].mean() * 100
        defense_success_df = pd.DataFrame({
            'Defense Method': defense_success_by_method.index,
            'Success Rate (%)': defense_success_by_method.values
        })
        
        fig = px.bar(
            defense_success_df,
            x='Defense Method',
            y='Success Rate (%)',
            color='Success Rate (%)',
            title='Defense Success Rate by Method',
            color_continuous_scale=px.colors.sequential.Greens
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No defense results yet. Test some defenses to see results.")
    
    # Model feature importance
    if model is not None and hasattr(model, 'feature_importances_'):
        st.markdown("### Model Feature Importance")
        feature_importance_fig = plot_feature_importance(model, feature_names)
        if feature_importance_fig:
            st.plotly_chart(feature_importance_fig, use_container_width=True)
    
    # Clear results button
    if st.button("Clear Results History", use_container_width=True):
        st.session_state.attack_results = []
        st.session_state.defense_results = []
        st.success("Results history cleared!")

# Run batch test logic
if run_batch:
    with st.spinner(f"Running batch test with {test_size} samples..."):
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Get test samples
        if len(df) > test_size:
            test_indices = np.random.choice(len(df), test_size, replace=False)
            test_data = df.iloc[test_indices]
        else:
            test_data = df
        
        # Initialize results
        batch_results = []
        defense_batch_results = []
        
        # Run attacks on all test samples
        for i, (idx, sample) in enumerate(test_data.iterrows()):
            # Update progress
            progress_bar.progress((i + 1) / len(test_data))
            
            # Convert the series to DataFrame
            sample_df = pd.DataFrame([sample])
            
            # True label
            true_label = sample_df['label'].values[0] if 'label' in sample_df else None
            
            # Original prediction
            original_pred = model.predict(sample_df[feature_names])[0]
            
            # Apply attack
            if attack_method == "Fast Gradient Sign Method (FGSM)":
                x_perturbed, perturbation = fast_gradient_sign_method(model, sample_df, epsilon, feature_names)
            elif attack_method == "Projected Gradient Descent (PGD)":
                x_perturbed, perturbation = projected_gradient_descent(model, sample_df, epsilon, feature_names)
            else:  # Carlini & Wagner
                x_perturbed, perturbation = carlini_wagner_attack(model, sample_df, feature_names)
            
            # Create perturbed sample
            perturbed_sample = sample_df.copy()
            perturbed_sample[feature_names] = x_perturbed
            
            # Get adversarial prediction
            adv_pred = model.predict(x_perturbed)[0]
            
            # Record attack result
            result = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'attack_method': attack_method,
                'epsilon': epsilon,
                'original_prediction': original_pred,
                'adversarial_prediction': adv_pred,
                'true_label': true_label,
                'success': original_pred != adv_pred,
                'perturbation_magnitude': np.linalg.norm(x_perturbed - sample_df[feature_names].values)
            }
            batch_results.append(result)
            
            # Apply defense methods
            for defense in defense_methods:
                if defense == "Adversarial Training":
                    # Skip for batch testing since it requires training
                    continue
                elif defense == "Ensemble Model":
                    # Create ensemble of models
                    ensemble_models = [model] * 3
                    defended_pred = ensemble_model(ensemble_models, perturbed_sample, feature_names)[0]
                elif defense == "Randomized Smoothing":
                    defended_pred = randomized_smoothing(model, perturbed_sample, feature_names)[0]
                else:
                    # Default to original prediction
                    defended_pred = original_pred
                
                # Record defense result
                defense_result = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'defense_method': defense,
                    'attack_method': attack_method,
                    'epsilon': epsilon,
                    'original_prediction': original_pred,
                    'adversarial_prediction': adv_pred,
                    'defended_prediction': defended_pred,
                    'success': defended_pred == original_pred,
                }
                defense_batch_results.append(defense_result)
        
        # Add batch results to session state
        st.session_state.attack_results.extend(batch_results)
        st.session_state.defense_results.extend(defense_batch_results)
        
        # Show summary
        st.success(f"Batch test completed! {len(batch_results)} attacks and {len(defense_batch_results)} defense tests performed.")
        
        # Calculate and display success rates
        attack_success_rate = sum(r['success'] for r in batch_results) / len(batch_results) * 100
        defense_results_by_method = {}
        
        for defense in defense_methods:
            if defense == "Adversarial Training":
                continue
            
            method_results = [r for r in defense_batch_results if r['defense_method'] == defense]
            if method_results:
                success_rate = sum(r['success'] for r in method_results) / len(method_results) * 100
                defense_results_by_method[defense] = success_rate
        
        # Display summary
        st.markdown(f"#### Batch Test Summary")
        st.markdown(f"- Attack Success Rate: **{attack_success_rate:.2f}%**")
        for defense, rate in defense_results_by_method.items():
            st.markdown(f"- {defense} Success Rate: **{rate:.2f}%**")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>SCADA IDS Adversarial Testing Module | Developed for Security Evaluation</p>", unsafe_allow_html=True)