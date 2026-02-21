import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# Paths
DATA_PATH = r"G:\Sem1\Cyberattack_on_smartGrid\Processed_Data\Processed_Dataset.csv"
OUTPUT_DIR = r"G:\Sem1\Cyberattack_on_smartGrid\Attacked_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Identify label column
LABEL_COL = "Label"  # Modify if needed
FEATURE_COLS = [col for col in df.columns if col != LABEL_COL]

# Convert categorical labels to numerical (if needed)
df[LABEL_COL], label_mapping = pd.factorize(df[LABEL_COL])

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(df[FEATURE_COLS], df[LABEL_COL], test_size=0.2, random_state=42)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
}

# Function to train & evaluate models
def train_evaluate(models, X_train, y_train, X_test, y_test, dataset_name):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"✅ {name} Accuracy on {dataset_name}: {acc:.4f}")
    return results

# Train on original dataset
print("\n🚀 Training on Original Dataset...")
baseline_results = train_evaluate(models, X_train, y_train, X_test, y_test, "Original Dataset")

# Save baseline results
baseline_results["Dataset"] = "Original"
performance_data = [baseline_results]

# Function to apply label flipping
def flip_labels(df, attack_ratio):
    df_attacked = df.copy()
    indices_to_flip = df_attacked.sample(frac=attack_ratio).index

    for idx in indices_to_flip:
        current_label = df_attacked.at[idx, LABEL_COL]
        possible_labels = [l for l in df_attacked[LABEL_COL].unique() if l != current_label]
        df_attacked.at[idx, LABEL_COL] = random.choice(possible_labels)

    return df_attacked

# Define attack levels
attack_levels = {
    "Mild": 0.10,
    "Moderate": 0.25,
    "Extreme": 0.50
}

# Apply attacks & evaluate performance
for attack_name, ratio in attack_levels.items():
    print(f"\n🚨 Applying {attack_name} Label Flipping Attack ({ratio*100}% flipped labels)...")
    df_attacked = flip_labels(df, ratio)

    # Split attacked data
    X_train_attacked, X_test_attacked, y_train_attacked, y_test_attacked = train_test_split(
        df_attacked[FEATURE_COLS], df_attacked[LABEL_COL], test_size=0.2, random_state=42
    )

    # Train and evaluate models
    attack_results = train_evaluate(models, X_train_attacked, y_train_attacked, X_test_attacked, y_test_attacked, f"{attack_name} Dataset")
    attack_results["Dataset"] = attack_name
    performance_data.append(attack_results)

# Convert to DataFrame
df_performance = pd.DataFrame(performance_data)

# =======================
# 📊 Plot Performance Drop
# =======================

fig, ax = plt.subplots(figsize=(8, 5))
for model_name in ["Decision Tree", "XGBoost"]:
    ax.plot(df_performance["Dataset"], df_performance[model_name], marker="o", label=model_name)

ax.set_xlabel("Attack Level")
ax.set_ylabel("Accuracy")
ax.set_title("Impact of Label Flipping Attacks on Model Performance")
ax.legend()
ax.grid()

# Save & show the plot
impact_plot_path = os.path.join(OUTPUT_DIR, "Model_Accuracy_Drop.png")
plt.savefig(impact_plot_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\n📉 Accuracy drop graph saved: {impact_plot_path}")
