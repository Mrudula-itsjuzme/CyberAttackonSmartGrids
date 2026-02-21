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
import matplotlib.pyplot as plt
import seaborn as sns
import gc  # garbage collection

print("\U0001F9F9 Cleaning up memory before starting...")
gc.collect()

# 📂 Load in CHUNKS
file_path = r"intermediate_combined_data.csv"
chunk_size = 50000
print(f"📦 Loading CSV in chunks of {chunk_size} rows...")

chunks = []
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    chunks.append(chunk)
    print(f"✅ Loaded chunk: {chunk.shape}")

df = pd.concat(chunks, axis=0)
del chunks  # Free memory
gc.collect()

print(f"\n🧠 Final combined dataset shape: {df.shape}")

# 🔍 Analyze dataset
print("\n🔍 Descriptive statistics of the dataset:")
print(df.describe(include='all'))

print("\n🗇 Number of unique values in each column:")
print(df.nunique())

# Handle Missing Values
print("\n🔍 Checking for missing values...")
missing_cols = df.columns[df.isnull().any()]
print(f"⚠️ Columns with missing values: {list(missing_cols)}")

numeric_cols_with_na = [col for col in missing_cols if df[col].dtype in ['float64', 'int64']]
non_numeric_cols_with_na = [col for col in missing_cols if col not in numeric_cols_with_na]

print(f"🗇 Numeric columns to impute: {numeric_cols_with_na}")

# Replace inf with nan before imputation (outlier handling)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df[numeric_cols_with_na] = df[numeric_cols_with_na].fillna(df[numeric_cols_with_na].mean())
print("✅ Numeric missing values imputed with mean!")
print(f"📄 Non-numeric columns with missing values (not imputed): {non_numeric_cols_with_na}")

# Handle non-numeric encoding
print("\n🌐 Encoding non-numeric columns...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"ℹ️ Categorical columns: {categorical_cols}")

ordinal_cols = [col for col in categorical_cols if 2 <= df[col].nunique() <= 10 and col != 'Label']
onehot_cols = [col for col in categorical_cols if col not in ordinal_cols and col != 'Label']

print(f"🔹 Ordinal encoded columns: {ordinal_cols}")
print(f"🔹 One-hot encoded columns: {onehot_cols}")

df[ordinal_cols + onehot_cols] = df[ordinal_cols + onehot_cols].astype(str)

y = df["Label"]
X = df.drop("Label", axis=1)

# 🔀 Fit encoder on a sample instead of the full dataset
print("\n🔀 Fitting encoder on a sample of the dataset...")
sample_X = X.sample(n=10000, random_state=42)
full_encoder = ColumnTransformer(transformers=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True), onehot_cols)
], remainder='passthrough')

full_encoder.fit(sample_X)

# Function to transform chunk using already-fitted encoder
def encode_chunk_sparse(chunk):
    return full_encoder.transform(chunk)

print("\n🤁 Encoding in parallel chunks (sparse matrices)...")
n_chunks = 8
chunk_size = int(np.ceil(len(X) / n_chunks))
X_chunks = [X.iloc[i:i + chunk_size] for i in range(0, len(X), chunk_size)]
encoded_chunks = Parallel(n_jobs=-1)(delayed(encode_chunk_sparse)(chunk) for chunk in X_chunks)

# ✅ Combine sparse matrices
X_encoded = vstack(encoded_chunks).astype(np.float32)

# ⛔️ Clean inf/nan again in final encoded matrix
X_encoded.data[np.isinf(X_encoded.data)] = 0
X_encoded.data[np.isnan(X_encoded.data)] = 0

print("✅ Encoding complete!")

del X_chunks, encoded_chunks  # Free up memory
gc.collect()

print("\n✂️ Splitting train-test with stratification...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)
print(f"✅ Training: {X_train.shape} | Testing: {X_test.shape}")

# ⚖️ Class weights
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))
print("⚖️ Class weights calculated!")

# 🌳 Build the model
print("🌪️ Training Decision Tree...")
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight=class_weights,
    random_state=42
)

model.fit(X_train, y_train)
print("🌟 Model trained!")

# 🕰️ Predict in CHUNKS
print("🔄 Predicting test data in chunks...")
chunked_preds = []
chunk_size_pred = 10000

for i in range(0, X_test.shape[0], chunk_size_pred):
    X_chunk = X_test[i:i+chunk_size_pred]
    preds = model.predict(X_chunk)
    chunked_preds.extend(preds)
    print(f"✅ Predicted chunk {i // chunk_size_pred + 1}")

y_pred = np.array(chunked_preds)

# 📈 Evaluate
print("\n📊 Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 📂 Save model
save_path = r"G:\\Sem 1\\Cyberattack_on_smartGrid\\decision_tree_chunked.joblib"
dump(model, save_path)
print(f"\n📂 Model saved to: {save_path}")
