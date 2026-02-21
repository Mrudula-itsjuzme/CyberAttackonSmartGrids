import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('shap_analysis.log'),
        logging.StreamHandler()
    ]
)

def load_model_and_data(model_path, dataset_path, label_encoders_path, label_encoder_y_path, sample_size=5000):
    logging.info(f"Loading model from: {model_path}")
    logging.info(f"Loading dataset from: {dataset_path}")
    
    try:
        model = joblib.load(model_path)
        logging.info(f"Model successfully loaded: {type(model)._name_}")
        
        df = pd.read_csv(dataset_path, low_memory=False)
        logging.info(f"Dataset loaded. Total rows: {len(df)}")
        
        df_sampled = df.sample(n=min(sample_size, len(df)), random_state=42)
        logging.info(f"Sampled dataset size: {len(df_sampled)}")
        
        X = df_sampled.drop('attack_type', axis=1)
        y = df_sampled['attack_type']
        
        label_encoders = joblib.load(label_encoders_path)
        label_encoder_y = joblib.load(label_encoder_y_path)
        
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        logging.info(f"Categorical columns to encode: {list(categorical_columns)}")
        
        for column in categorical_columns:
            X[column] = label_encoders[column].transform(X[column].astype(str))
        
        feature_names = list(X.columns)
        logging.info(f"Total features: {len(feature_names)}")
        
        return model, X, y, feature_names, label_encoder_y
    
    except Exception as e:
        logging.error(f"Error in data loading: {e}")
        raise

def local_feature_importance(model, X, feature_names, label_encoder_y, num_samples=10):
    logging.info("Starting local feature importance analysis")
    
    try:
        explainer = shap.TreeExplainer(model)
        logging.info("SHAP explainer created")
        
        shap_values = explainer.shap_values(X)
        logging.info(f"SHAP values shape: {shap_values.shape}")
        
        # Handle multi-class scenario
        if shap_values.ndim == 3:
            logging.info("Multi-class scenario detected")
            shap_values = shap_values[:, :, 1]  # Select one class for analysis
        
        sample_indices = np.random.choice(X.shape[0], num_samples, replace=False)
        logging.info(f"Analyzing {num_samples} random samples")

        for index in sample_indices:
            logging.info(f"Processing sample {index}")
            plt.figure(figsize=(12, 6))
            sample = X.iloc[index]
            sample_shap_values = shap_values[index]
            
            predicted_class = model.predict(sample.to_frame().T)[0]
            true_class = label_encoder_y.inverse_transform([predicted_class])[0]
            logging.info(f"Sample predicted class: {true_class}")

            # Manual waterfall plot instead of shap.plots.waterfall
            plt.title(f'Local Feature Impact - Sample {index}\nPredicted: {true_class}')
            
            # Sort features by absolute SHAP values
            sorted_idx = np.abs(sample_shap_values).argsort()[::-1]
            top_features = [feature_names[i] for i in sorted_idx[:10]]
            top_shap_values = sample_shap_values[sorted_idx[:10]]
            
            # Plot horizontal bar chart
            plt.barh(top_features, top_shap_values)
            plt.xlabel('SHAP Value')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig(f'local_shap_sample_{index}.png')
            logging.info(f"Saved local SHAP plot for sample {index}")
            plt.close()
        
        logging.info("Local feature importance analysis completed")
    
    except Exception as e:
        logging.error(f"Error in local feature importance: {e}")
        
def dependency_plots(model, X, feature_names, num_features=5):
    logging.info("Starting dependency plot generation")
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Handle multi-class scenario
        if shap_values.ndim == 3:
            logging.info("Multi-class scenario detected")
            shap_values = shap_values[:, :, 1]  # Select one class
        
        feature_importances = np.abs(shap_values).mean(axis=0)
        top_feature_indices = feature_importances.argsort()[-num_features:][::-1]
        logging.info(f"Top {num_features} features for dependency analysis")

        for idx in top_feature_indices:
            feature = feature_names[idx]
            logging.info(f"Creating dependency plot for feature: {feature}")
            plt.figure(figsize=(12, 6))
            shap.dependence_plot(feature, shap_values, X)
            plt.title(f'SHAP Dependency Plot: {feature}')
            plt.tight_layout()
            plt.savefig(f'shap_dependency_{feature}.png')
            logging.info(f"Saved dependency plot for {feature}")
            plt.close()
        
        logging.info("Dependency plot generation completed")
    
    except Exception as e:
        logging.error(f"Error in dependency plots: {e}")

def main():
    logging.info("Starting SHAP Analysis")
    
    model_path = 'best_model_decision_tree.joblib'
    dataset_path = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
    label_encoders_path = 'label_encoders.joblib'
    label_encoder_y_path = 'label_encoder_y.joblib'

    try:
        model, X, y, feature_names, label_encoder_y = load_model_and_data(
            model_path, dataset_path, label_encoders_path, label_encoder_y_path
        )

        local_feature_importance(model, X, feature_names, label_encoder_y)
        dependency_plots(model, X, feature_names)
        
        logging.info("SHAP Analysis completed successfully")
    
    except Exception as e:
        logging.error(f"SHAP Analysis failed: {e}")

if __name__ == "_main_":
    main()