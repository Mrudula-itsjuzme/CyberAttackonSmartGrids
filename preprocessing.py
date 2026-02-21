import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def optimize_dtypes(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    return df

def preprocess_single_csv(file_path):
    errors = []

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None, [f"File not found: {file_path}"]

    try:
        print(f"📂 Loading file: {file_path}")
        df = pd.read_csv(file_path)

        print("🔍 Optimizing data types...")
        df = optimize_dtypes(df)

        # Identify columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns

        print("🧹 Handling missing values...")
        # Handle missing values
        df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        print("⚠️ Handling infinite values...")
        # Handle infinities
        df[numerical_columns] = df[numerical_columns].replace([np.inf, -np.inf], np.nan)
        df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

        print("⚖️ Removing low-variance features...")
        # Remove low-variance features
        selector = VarianceThreshold(threshold=0.01)
        numerical_df = df[numerical_columns]
        selector.fit(numerical_df)
        selected_features = numerical_df.columns[selector.get_support()].tolist()

        # Keep only selected numerical features and categorical columns
        df = df[selected_features + list(categorical_columns)]

        print("⚖️ Removing highly correlated features...")
        # Remove highly correlated features
        correlation_matrix = df[selected_features].corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
        df = df.drop(columns=to_drop)

        print("📊 Scaling numerical features...")
        # Scale remaining numerical features
        remaining_numerical = [col for col in selected_features if col not in to_drop]
        scaler = StandardScaler()
        df[remaining_numerical] = scaler.fit_transform(df[remaining_numerical])

        return df, errors

    except Exception as e:
        errors.append(f"Error processing file {file_path}: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return None, errors

def save_results_single(combined_df, errors, output_path):
    # Save processed dataset
    if combined_df is not None:
        output_file = os.path.join(output_path, 'processed_dataset.csv')
        combined_df.to_csv(output_file, index=False)
        print(f"\n✅ Processed dataset saved to: {output_file}")

        print("\n📋 Final Dataset Summary:")
        print(f"• Total samples: {len(combined_df)}")
        print(f"• Total features: {len(combined_df.columns)}")
        print(f"• Numerical features: {len(combined_df.select_dtypes(include=[np.number]).columns)}")
        print(f"• Categorical features: {len(combined_df.select_dtypes(exclude=[np.number]).columns)}")

    # Save errors if any
    if errors:
        error_file = os.path.join(output_path, 'error_log.txt')
        with open(error_file, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(errors))
        print(f"\n📜 Error log saved to: {error_file}")

if __name__ == "__main__":
    # Path to the input CSV file
    file_path = r"F:\mrudula college\Sem1_project\Cyberattack_on_smartGrid\FRESH\intermediate_combined_data.csv"
    output_path = os.path.dirname(file_path)  # Save output in the same directory as input

    # Process the CSV file
    combined_df, errors = preprocess_single_csv(file_path)

    # Save results
    save_results_single(combined_df, errors, output_path)

    print("\n✅ Preprocessing complete!")
