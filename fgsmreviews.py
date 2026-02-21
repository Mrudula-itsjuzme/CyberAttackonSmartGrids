import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import shap
import multiprocessing as mp
from functools import partial
import gc
import os

# --- CONFIG ---
DATA_PATH = r"G:\Sem1\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
OUTPUT_CSV = "adv_dt_comparison_output.csv"
CHUNK_SIZE = 50000  # Process 50k rows at a time
epsilon = 0.2

def load_data_chunked(file_path, chunk_size=CHUNK_SIZE):
    """Load data in chunks to manage memory"""
    print(f"📊 Loading data in chunks of {chunk_size:,} rows...")
    
    # Get total rows for progress tracking
    total_rows = sum(1 for _ in open(file_path)) - 1
    print(f"📈 Total rows: {total_rows:,}")
    
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk.dropna(inplace=True)
        chunks.append(chunk)
        print(f"✅ Loaded chunk: {len(chunk):,} rows")
    
    # Combine chunks
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    
    print(f"🎯 Final dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df

def process_correlation_chunk(chunk_data, threshold=0.95):
    """Process correlation analysis on a chunk"""
    feature_cols, chunk = chunk_data
    
    if len(chunk) < 100:  # Skip small chunks
        return []
    
    # Calculate correlation for this chunk
    corr_matrix = chunk[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    
    return drop_cols

def parallel_correlation_analysis(df, feature_cols, n_cores=None):
    """Parallel correlation analysis with chunking"""
    if n_cores is None:
        n_cores = min(mp.cpu_count(), 4)
    
    print(f"🔧 Running correlation analysis with {n_cores} cores...")
    
    # Split data into chunks for parallel processing
    chunk_size = len(df) // n_cores
    chunks = [(feature_cols, df[i:i+chunk_size]) for i in range(0, len(df), chunk_size)]
    
    with mp.Pool(n_cores) as pool:
        results = pool.map(process_correlation_chunk, chunks)
    
    # Combine results and get most frequently dropped columns
    all_drop_cols = [col for sublist in results for col in sublist]
    drop_counts = pd.Series(all_drop_cols).value_counts()
    
    # Keep columns that appear in majority of chunks
    threshold_count = len(chunks) // 2
    final_drop_cols = drop_counts[drop_counts >= threshold_count].index.tolist()
    
    return final_drop_cols

def chunked_mutual_info(X, y, chunk_size=10000):
    """Calculate mutual information in chunks"""
    print("🧠 Calculating mutual information in chunks...")
    
    if len(X) <= chunk_size:
        return mutual_info_classif(X, y, discrete_features=False)
    
    # Process in chunks and average results
    mi_scores = np.zeros(X.shape[1])
    n_chunks = 0
    
    for i in range(0, len(X), chunk_size):
        end_idx = min(i + chunk_size, len(X))
        X_chunk = X.iloc[i:end_idx]
        y_chunk = y.iloc[i:end_idx]
        
        chunk_mi = mutual_info_classif(X_chunk, y_chunk, discrete_features=False)
        mi_scores += chunk_mi
        n_chunks += 1
        
        print(f"  ✅ Processed chunk {n_chunks}: rows {i:,}-{end_idx:,}")
    
    return mi_scores / n_chunks

def memory_efficient_shap(model, X, max_samples=5000):
    """Memory-efficient SHAP calculation"""
    print(f"🔍 Calculating SHAP values (sampling {max_samples:,} rows)...")
    
    # Sample data if too large
    if len(X) > max_samples:
        sample_idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X.iloc[sample_idx]
    else:
        X_sample = X
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    return shap_values

def process_adversarial_chunk(chunk_data):
    """Process adversarial examples in chunks"""
    X_chunk, epsilon = chunk_data
    return X_chunk + epsilon * np.sign(np.random.normal(0, 1, X_chunk.shape))

def parallel_adversarial_generation(X_test, epsilon, n_cores=None):
    """Generate adversarial examples in parallel"""
    if n_cores is None:
        n_cores = min(mp.cpu_count(), 4)
    
    print(f"💥 Generating adversarial examples with {n_cores} cores...")
    
    # Split data into chunks
    chunk_size = len(X_test) // n_cores
    chunks = [(X_test[i:i+chunk_size], epsilon) for i in range(0, len(X_test), chunk_size)]
    
    with mp.Pool(n_cores) as pool:
        results = pool.map(process_adversarial_chunk, chunks)
    
    return np.vstack(results)

def main():
    print("🚀 Starting Memory-Efficient ML Pipeline...")
    
    # Step 1: Load data in chunks
    df = load_data_chunked(DATA_PATH)
    
    target = 'Label'
    feature_cols = [col for col in df.columns if col != target]
    
    # Step 2: Parallel correlation analysis
    n_cores = min(mp.cpu_count(), 4)
    drop_cols = parallel_correlation_analysis(df, feature_cols, n_cores)
    
    df_reduced = df.drop(columns=drop_cols)
    print(f"🧼 Dropped {len(drop_cols)} highly correlated features.")
    
    # Memory cleanup
    del df
    gc.collect()
    
    # Step 3: Chunked mutual information
    X = df_reduced.drop(columns=[target])
    y = df_reduced[target]
    
    mi_scores = chunked_mutual_info(X, y)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    top_mi_features = mi_series.head(20).index.tolist()
    
    print(f"🎯 Top 20 MI features selected")
    
    # Step 4: RFE (memory efficient)
    print("🔧 Running RFE...")
    X_mi = X[top_mi_features]
    
    model_for_rfe = DecisionTreeClassifier(max_depth=5, random_state=42)
    rfe = RFE(estimator=model_for_rfe, n_features_to_select=10)
    rfe.fit(X_mi, y)
    selected_rfe_features = list(X_mi.columns[rfe.support_])
    
    # Memory cleanup
    del X_mi
    gc.collect()
    
    # Step 5: Memory-efficient SHAP
    X_rfe = X[selected_rfe_features]
    final_model = DecisionTreeClassifier(max_depth=8, random_state=42)
    final_model.fit(X_rfe, y)
    
    shap_values = memory_efficient_shap(final_model, X_rfe)
    mean_shap_vals = np.abs(shap_values[1]).mean(axis=0)
    
    shap_df = pd.DataFrame({
        'Feature': selected_rfe_features,
        'SHAP_Value': mean_shap_vals
    }).sort_values(by='SHAP_Value', ascending=False)
    
    final_selected = shap_df.head(10)['Feature'].tolist()
    print(f"✅ Final Selected Features: {final_selected}")
    
    # Step 6: Final processing
    X_final = X[final_selected]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    # Memory cleanup
    del X, X_final
    gc.collect()
    
    # Step 7: Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    dt_model = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Step 8: Original predictions
    y_pred_orig = dt_model.predict(X_test)
    acc_orig = accuracy_score(y_test, y_pred_orig)
    f1_orig = f1_score(y_test, y_pred_orig)
    
    # Step 9: Parallel adversarial generation
    X_adv = parallel_adversarial_generation(X_test, epsilon, n_cores)
    y_pred_adv = dt_model.predict(X_adv)
    acc_adv = accuracy_score(y_test, y_pred_adv)
    f1_adv = f1_score(y_test, y_pred_adv)
    
    # Step 10: Save results in chunks
    print("💾 Saving results...")
    
    X_test_unscaled = scaler.inverse_transform(X_test)
    X_adv_unscaled = scaler.inverse_transform(X_adv)
    
    # Create comparison dataframe in chunks to avoid memory issues
    chunk_size = 10000
    comparison_chunks = []
    
    for i in range(0, len(X_test_unscaled), chunk_size):
        end_idx = min(i + chunk_size, len(X_test_unscaled))
        
        chunk_df = pd.DataFrame(
            X_test_unscaled[i:end_idx], 
            columns=[f + '_orig' for f in final_selected]
        )
        
        for idx, feat in enumerate(final_selected):
            chunk_df[feat + '_adv'] = X_adv_unscaled[i:end_idx, idx]
        
        chunk_df['Label'] = y_test.iloc[i:end_idx].reset_index(drop=True)
        chunk_df['Prediction_Original'] = y_pred_orig[i:end_idx]
        chunk_df['Prediction_Adversarial'] = y_pred_adv[i:end_idx]
        
        comparison_chunks.append(chunk_df)
    
    # Combine and save
    comparison_df = pd.concat(comparison_chunks, ignore_index=True)
    comparison_df.to_csv(OUTPUT_CSV, index=False)
    
    print("🔥 DONE — Saved to adv_dt_comparison_output.csv")
    print(f"🎯 Original Acc: {acc_orig*100:.2f}%, F1: {f1_orig*100:.2f}%")
    print(f"💥 Adversarial Acc: {acc_adv*100:.2f}%, F1: {f1_adv*100:.2f}%")
    print(f"📊 Memory-efficient processing completed!")

if __name__ == "__main__":
    main()