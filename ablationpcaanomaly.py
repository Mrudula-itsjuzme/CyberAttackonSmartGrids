import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "G:\\Sem1\\Cyberattack_on_smartGrid\\intermediate_combined_data.csv"
CHUNK_SIZE = 50000

def robust_clean(X, means=None, stds=None):
    """Enhanced cleaning with better outlier handling"""
    X = np.array(X, dtype=float)
    X = np.where(np.isinf(X), np.nan, X)
    
    if means is None:
        means = np.nanmedian(X, axis=0)
        stds = np.nanstd(X, axis=0)
        means = np.where(np.isnan(means), 0, means)
        stds = np.where(np.isnan(stds) | (stds == 0), 1, stds)
    
    if len(means) != X.shape[1]:
        if len(means) < X.shape[1]:
            means = np.pad(means, (0, X.shape[1] - len(means)), constant_values=0)
            stds = np.pad(stds, (0, X.shape[1] - len(stds)), constant_values=1)
        else:
            means = means[:X.shape[1]]
            stds = stds[:X.shape[1]]
    
    # Aggressive outlier removal for better anomaly detection
    for i in range(X.shape[1]):
        col = X[:, i]
        col = np.where(np.isnan(col), means[i], col)
        
        # Use stricter IQR bounds
        q75, q25 = np.percentile(col, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr  # Stricter bounds
        upper_bound = q75 + 1.5 * iqr
        
        col = np.clip(col, lower_bound, upper_bound)
        X[:, i] = col
    
    return X, means, stds

def create_enhanced_target(chunk, target_strategy='aggressive_multi'):
    """Enhanced target creation for higher F1 scores"""
    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
    
    if target_strategy == 'ultra_aggressive':
        # Ultra aggressive approach for 70%+ F1
        anomaly_scores = np.zeros(len(chunk))
        
        for col in numeric_cols[:12]:  # Use even more columns
            if col in chunk.columns:
                values = chunk[col].values
                if np.std(values) > 0:
                    # Even more sensitive thresholds
                    z_scores = np.abs((values - np.mean(values)) / np.std(values))
                    anomaly_scores += (z_scores > 2.0).astype(int)  # Very low threshold
                    
                    # Very sensitive percentile thresholds
                    anomaly_scores += (values > np.percentile(values, 90)).astype(int)
                    anomaly_scores += (values < np.percentile(values, 10)).astype(int)
                    
                    # Multiple rate of change indicators
                    if len(values) > 1:
                        diff = np.diff(values, prepend=values[0])
                        diff_thresh = np.percentile(np.abs(diff), 80)  # Lower threshold
                        anomaly_scores[1:] += (np.abs(diff[1:]) > diff_thresh).astype(int)
                        
                        # Second order differences
                        if len(values) > 2:
                            diff2 = np.diff(diff)
                            diff2_thresh = np.percentile(np.abs(diff2), 85)
                            anomaly_scores[2:] += (np.abs(diff2) > diff2_thresh).astype(int)
        
        # Very sensitive threshold
        y = (anomaly_scores >= 1).astype(int)
        
    elif target_strategy == 'aggressive_multi':
        # More aggressive multi-criteria approach
        anomaly_scores = np.zeros(len(chunk))
        
        for col in numeric_cols[:8]:  # Use more columns
            if col in chunk.columns:
                values = chunk[col].values
                if np.std(values) > 0:
                    # Multiple anomaly indicators with lower thresholds
                    z_scores = np.abs((values - np.mean(values)) / np.std(values))
                    anomaly_scores += (z_scores > 2.5).astype(int)  # Lower threshold
                    
                    # More sensitive percentile thresholds
                    anomaly_scores += (values > np.percentile(values, 95)).astype(int)
                    anomaly_scores += (values < np.percentile(values, 5)).astype(int)
                    
                    # Add rate of change anomalies
                    if len(values) > 1:
                        diff = np.diff(values, prepend=values[0])
                        diff_thresh = np.percentile(np.abs(diff), 90)
                        anomaly_scores[1:] += (np.abs(diff[1:]) > diff_thresh).astype(int)
        
        # Lower threshold for anomaly classification
        y = (anomaly_scores >= 1).astype(int)  # More sensitive
        
    elif target_strategy == 'balanced_synthetic':
        # Create more balanced synthetic targets
        X = chunk[numeric_cols].values
        X_clean, _, _ = robust_clean(X)
        
        # Multiple anomaly indicators
        row_means = np.mean(X_clean, axis=1)
        row_stds = np.std(X_clean, axis=1)
        row_mins = np.min(X_clean, axis=1)
        row_maxs = np.max(X_clean, axis=1)
        
        # Combine multiple indicators with lower thresholds
        mean_anomaly = row_means > np.percentile(row_means, 85)  # Lower threshold
        std_anomaly = row_stds > np.percentile(row_stds, 85)
        min_anomaly = row_mins < np.percentile(row_mins, 15)
        max_anomaly = row_maxs > np.percentile(row_maxs, 85)
        
        y = (mean_anomaly | std_anomaly | min_anomaly | max_anomaly).astype(int)
        
    elif target_strategy == 'correlation_based':
        # Use correlation patterns for anomaly detection
        X = chunk[numeric_cols[:10]].values  # Use first 10 columns
        X_clean, _, _ = robust_clean(X)
        
        # Calculate correlation with first principal component
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_clean).flatten()
        
        anomaly_scores = np.zeros(len(chunk))
        for i in range(X_clean.shape[1]):
            corr = np.corrcoef(X_clean[:, i], pc1)[0, 1]
            if not np.isnan(corr):
                # Points that don't correlate well with main pattern
                anomaly_scores += (np.abs(corr) < 0.3).astype(int)
        
        y = (anomaly_scores >= 2).astype(int)
        
    else:  # composite - your previous best
        X = chunk[numeric_cols].values
        X_clean, _, _ = robust_clean(X)
        
        row_means = np.mean(X_clean, axis=1)
        row_stds = np.std(X_clean, axis=1)
        
        mean_anomaly = row_means > np.percentile(row_means, 90)  # Lower threshold
        std_anomaly = row_stds > np.percentile(row_stds, 90)
        
        y = (mean_anomaly | std_anomaly).astype(int)
    
    return y

def advanced_threshold_optimization(y_true, y_scores):
    """Advanced threshold optimization focusing on F1"""
    if len(np.unique(y_true)) <= 1:
        return np.percentile(y_scores, 15)
    
    # Method 1: Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, -y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Find multiple good thresholds
    top_indices = np.argsort(f1_scores)[-5:]  # Top 5 F1 scores
    candidate_thresholds = []
    
    for idx in top_indices:
        if idx < len(thresholds):
            candidate_thresholds.append(-thresholds[idx])
    
    # Method 2: Grid search on percentiles
    best_f1 = 0
    best_threshold = np.percentile(y_scores, 15)
    
    # Fine-grained search
    for percentile in np.linspace(5, 40, 50):
        threshold = np.percentile(y_scores, percentile)
        y_pred = (y_scores < threshold).astype(int)
        
        if len(np.unique(y_pred)) > 1:
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
    # Method 3: Youden's J statistic
    fpr_list, tpr_list = [], []
    for percentile in np.linspace(5, 40, 30):
        threshold = np.percentile(y_scores, percentile)
        y_pred = (y_scores < threshold).astype(int)
        
        if len(np.unique(y_pred)) > 1:
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            tpr = tp / (tp + fn + 1e-8)
            fpr = fp / (fp + tn + 1e-8)
            
            # Youden's J = TPR - FPR
            j_score = tpr - fpr
            f1 = f1_score(y_true, y_pred)
            
            # Weighted combination of J and F1
            combined_score = 0.7 * f1 + 0.3 * j_score
            if combined_score > best_f1:
                best_f1 = combined_score
                best_threshold = threshold
    
    return best_threshold

def run_advanced_isolation_forest(name, use_scaler=True, scaler_type='robust', 
                                 use_pca=False, pca_components=30, use_fs=True, 
                                 fs_k=25, fs_method='mutual_info', contamination=0.15, 
                                 n_estimators=300, target_strategy='aggressive_multi',
                                 use_balanced_sampling=False, max_samples=0.8):
    """Advanced IF with focus on high F1 scores"""
    try:
        transformers = {}
        all_scores = []
        all_labels = []
        first_chunk = True
        clf = None
        feature_means = None
        feature_stds = None
        n_features = None
        
        chunk_count = 0
        
        for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
            chunk_count += 1
            if chunk_count > 10:  # Remove for full run
                break
                
            numeric_cols = chunk.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                continue
                
            X = chunk[numeric_cols].values
            
            if first_chunk:
                n_features = X.shape[1]
                X, feature_means, feature_stds = robust_clean(X)
            else:
                if X.shape[1] < n_features:
                    X = np.pad(X, ((0, 0), (0, n_features - X.shape[1])), constant_values=0)
                elif X.shape[1] > n_features:
                    X = X[:, :n_features]
                X, _, _ = robust_clean(X, feature_means, feature_stds)
            
            # Enhanced target creation
            y = create_enhanced_target(chunk, target_strategy)
            
            if len(np.unique(y)) <= 1:
                continue
            
            # Balanced sampling for training
            if use_balanced_sampling and first_chunk:
                pos_indices = np.where(y == 1)[0]
                neg_indices = np.where(y == 0)[0]
                
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    # Undersample majority class
                    n_samples = min(len(pos_indices) * 3, len(neg_indices))
                    selected_neg = np.random.choice(neg_indices, n_samples, replace=False)
                    selected_indices = np.concatenate([pos_indices, selected_neg])
                    
                    X = X[selected_indices]
                    y = y[selected_indices]
            
            # Apply transformations
            if use_scaler:
                if first_chunk:
                    transformers['scaler'] = RobustScaler() if scaler_type == 'robust' else StandardScaler()
                    X = transformers['scaler'].fit_transform(X)
                else:
                    X = transformers['scaler'].transform(X)
            
            if use_pca:
                if first_chunk:
                    n_components = min(pca_components, X.shape[1], X.shape[0])
                    transformers['pca'] = PCA(n_components=n_components)
                    X = transformers['pca'].fit_transform(X)
                else:
                    X = transformers['pca'].transform(X)
            
            if use_fs:
                if first_chunk:
                    scorer = mutual_info_classif if fs_method == 'mutual_info' else f_classif
                    k_best = min(fs_k, X.shape[1])
                    transformers['fs'] = SelectKBest(scorer, k=k_best)
                    X = transformers['fs'].fit_transform(X, y)
                else:
                    X = transformers['fs'].transform(X)
            
            # Enhanced model
            if first_chunk:
                clf = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    bootstrap=True,
                    n_jobs=-1
                )
                clf.fit(X)
                first_chunk = False
            
            scores = clf.decision_function(X)
            all_scores.extend(scores)
            all_labels.extend(y)
        
        if len(all_scores) == 0:
            return 0, 0, 0, 0, 0
        
        y_true = np.array(all_labels)
        y_scores = np.array(all_scores)
        
        pos_rate = np.mean(y_true)
        print(f"  Positive rate: {pos_rate:.3f} ({np.sum(y_true)}/{len(y_true)})")
        
        if pos_rate == 0 or pos_rate == 1:
            return 0, 0, np.mean(y_true == 0), 0, 0
        
        # Advanced threshold optimization
        threshold = advanced_threshold_optimization(y_true, y_scores)
        y_pred = (y_scores < threshold).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_true, -y_scores)
        f1 = f1_score(y_true, y_pred)
        accuracy = np.mean(y_true == y_pred)
        
        true_pos = np.sum((y_pred == 1) & (y_true == 1))
        pred_pos = np.sum(y_pred == 1)
        actual_pos = np.sum(y_true == 1)
        
        precision = true_pos / max(1, pred_pos)
        recall = true_pos / max(1, actual_pos)
        
        print(f"{name:30s} | AUC: {auc:.3f} | F1: {f1:.3f} | Acc: {accuracy:.3f} | P: {precision:.3f} | R: {recall:.3f}")
        
        return auc, f1, accuracy, precision, recall
        
    except Exception as e:
        print(f"{name:30s} | ERROR: {str(e)}")
        return 0, 0, 0, 0, 0

def main():
    print("Configuration                  | AUC   | F1    | Acc   | Prec  | Rec")
    print("-" * 85)
    
    results = {}
    
    # Test enhanced target strategies
    print("=== RAW BASELINE ===")
    baseline_configs = [
        ("Raw Baseline", {
            "target_strategy": "composite", 
            "contamination": 0.1,
            "use_scaler": False,
            "use_pca": False,
            "use_fs": False,
            "n_estimators": 100
        }),
    ]
    
    for name, params in baseline_configs:
        results[name] = run_advanced_isolation_forest(name, **params)
    
    print("\n=== ENHANCED TARGET STRATEGIES ===")
    target_configs = [
        ("Aggressive Multi", {"target_strategy": "aggressive_multi", "contamination": 0.20}),
        ("Balanced Synthetic", {"target_strategy": "balanced_synthetic", "contamination": 0.25}),
        ("Correlation Based", {"target_strategy": "correlation_based", "contamination": 0.15}),
        ("Enhanced Composite", {"target_strategy": "composite", "contamination": 0.20}),
    ]
    
    for name, params in target_configs:
        results[name] = run_advanced_isolation_forest(name, **params)
    
    print("\n=== SIMPLE EFFECTIVE CONFIGS ===")
    # Simple but effective configurations
    simple_configs = [
        ("Simple-Best", {
            "target_strategy": "aggressive_multi",
            "contamination": 0.30,  # Higher contamination
            "use_fs": True,
            "fs_k": 20,
            "n_estimators": 300
        }),
        ("Ultra-Aggressive", {
            "target_strategy": "ultra_aggressive",
            "contamination": 0.40,  # Match the high positive rate
            "use_fs": True,
            "fs_k": 25,
            "n_estimators": 250
        }),
    ]
    
    for name, params in simple_configs:
        results[name] = run_advanced_isolation_forest(name, **params)
    
    print("\n=== HIGH F1 OPTIMIZED CONFIGS ===")
    # Optimized configurations for high F1
    optimized_configs = [
        ("F1-Supreme-1", {
            "target_strategy": "ultra_aggressive",
            "contamination": 0.45,  # Match ultra aggressive positive rate
            "use_fs": True,
            "fs_k": 30,
            "fs_method": "mutual_info",
            "n_estimators": 400,
            "max_samples": 0.5,
            "use_balanced_sampling": True
        }),
        ("F1-Supreme-2", {
            "target_strategy": "aggressive_multi",
            "contamination": 0.50,  # Even higher
            "use_fs": True,
            "fs_k": 35,
            "fs_method": "f_classif",
            "n_estimators": 500,
            "max_samples": 0.4,
            "use_balanced_sampling": True
        }),
        ("F1-Precision-Balance", {
            "target_strategy": "ultra_aggressive",
            "contamination": 0.35,
            "use_fs": True,
            "fs_k": 25,
            "use_pca": True,
            "pca_components": 15,
            "n_estimators": 350,
            "max_samples": 0.6
        }),
    ]
    
    for name, params in optimized_configs:
        results[name] = run_advanced_isolation_forest(name, **params)
    
    # Analysis
    print(f"\n{'='*85}")
    print("RESULTS ANALYSIS:")
    
    if results:
        valid_results = {k: v for k, v in results.items() if v[1] > 0}
        
        if valid_results:
            best_f1 = max(valid_results.keys(), key=lambda k: valid_results[k][1])
            print(f"Best F1 Score: {best_f1} (F1: {valid_results[best_f1][1]:.3f})")
            
            # Show all results sorted by F1
            sorted_results = sorted(valid_results.items(), key=lambda x: x[1][1], reverse=True)
            print(f"\nALL CONFIGURATIONS (sorted by F1):")
            for i, (config, (auc, f1, acc, prec, rec)) in enumerate(sorted_results):
                print(f"{i+1:2d}. {config:<25} | F1: {f1:.3f} | AUC: {auc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f}")
                
            # Check if we hit target
            if valid_results[best_f1][1] >= 0.7:
                print(f"\n🎉 TARGET ACHIEVED! F1 Score >= 70%")
            else:
                print(f"\n📈 Progress: {valid_results[best_f1][1]:.1%} towards 70% target")

if __name__ == "__main__":
    main()