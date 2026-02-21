import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from joblib import dump, Parallel, delayed
from scipy.sparse import vstack
import gc
import os


def clean_memory():
    print("🧹 Cleaning up memory...")
    gc.collect()


def load_dataset(file_path, chunk_size=50000):
    print("📦 Loading dataset in chunks...")
    chunks = [chunk for chunk in pd.read_csv(file_path, chunksize=chunk_size)]
    df = pd.concat(chunks, axis=0)
    del chunks
    gc.collect()
    print(f"✅ Dataset loaded with shape: {df.shape}")
    return df


def handle_missing_and_outliers(df):
    print("🔧 Handling missing values and outliers...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Label" in numeric_cols:
        numeric_cols.remove("Label")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return clean_outliers(df)


def clean_outliers(df, chunk_size=250000):
    cleaned_chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size].copy()
        numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
        if "Label" in numeric_cols:
            numeric_cols.remove("Label")
        Q1 = chunk[numeric_cols].quantile(0.25)
        Q3 = chunk[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        condition = ~((chunk[numeric_cols] < (Q1 - 1.5 * IQR)) | (chunk[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        chunk = chunk[condition]
        cleaned_chunks.append(chunk)
    result = pd.concat(cleaned_chunks, axis=0)
    print(f"✅ After IQR outlier removal: {result.shape}")
    return result


def encode_features(df):
    print("🔤 Encoding features...")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    ordinal_cols = [col for col in categorical_cols if 2 <= df[col].nunique() <= 10 and col != 'Label']
    onehot_cols = [col for col in categorical_cols if col not in ordinal_cols and col != 'Label']
    df[ordinal_cols + onehot_cols] = df[ordinal_cols + onehot_cols].astype(str)

    y = df["Label"]
    X = df.drop("Label", axis=1)

    sample_X = X.sample(n=10000, random_state=42)
    encoder = ColumnTransformer(transformers=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True), onehot_cols)
    ], remainder='passthrough')
    encoder.fit(sample_X)

    def encode_chunk_sparse(chunk):
        return encoder.transform(chunk)

    chunk_count = 8
    chunk_size = int(np.ceil(len(X) / chunk_count))
    X_chunks = [X.iloc[i:i + chunk_size] for i in range(0, len(X), chunk_size)]
    encoded_chunks = Parallel(n_jobs=-1)(delayed(encode_chunk_sparse)(chunk) for chunk in X_chunks)

    X_encoded = vstack(encoded_chunks).astype(np.float32)
    X_encoded.data[np.isinf(X_encoded.data)] = 0
    X_encoded.data[np.isnan(X_encoded.data)] = 0

    return X_encoded, y


def train_and_evaluate_decision_tree(X_encoded, y):
    print("🧪 Splitting dataset and training Decision Tree...")
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=50,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        class_weight=class_weights,
        random_state=42
    )
    model.fit(X_train, y_train)

    chunked_preds = []
    chunk_size_pred = 10000
    for i in range(0, X_test.shape[0], chunk_size_pred):
        preds = model.predict(X_test[i:i+chunk_size_pred])
        chunked_preds.extend(preds)

    y_pred = np.array(chunked_preds)
    print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
    print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("📋 Classification Report:\n", classification_report(y_test, y_pred))

    return model


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(model, path)
    print(f"💾 Model saved to: {path}")


def main():
    clean_memory()
    file_path = r"intermediate_combined_data.csv"
    df = load_dataset(file_path)
    df = handle_missing_and_outliers(df)
    X_encoded, y = encode_features(df)
    model = train_and_evaluate_decision_tree(X_encoded, y)
    save_model(model, r"G:\\Sem 1\\Cyberattack_on_smartGrid\\decision_tree_chunked.joblib")


if __name__ == "__main__":
    main()
