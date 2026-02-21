import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load only the necessary data (FAST - No plotting required)
filepath = "H:\sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"
df = pd.read_csv(filepath, usecols=['Flow Duration', 'Fwd IAT Tot'])  # Adjust column names as needed

# Print actual unique values from x-axis
print("\n📌 X-Axis Values for Box Plot & Violin Plot:\n")

for feature in df.columns:
    values = df[feature].dropna().unique()  # Remove NaNs and get unique values
    print(f"🔹 {feature} X-Axis Values ({len(values)} unique values shown):")
    print(values[:10])  # Print first 10 values only for readability
    print("...\n" if len(values) > 10 else "\n")

# Print actual unique values from y-axis    
print("\n📌 Y-Axis Values for Box Plot & Violin Plot:\n")

for feature in df.columns:
    values = df[feature].dropna().unique()  # Remove NaNs and get unique values
    print(f"🔹 {feature} Y-Axis Values ({len(values)} unique values shown):")
    print(values[:10])  # Print first 10 values only for readability
    print("...\n" if len(values) > 10 else "\n")
# Compare this snippet from poi.py:

# Load the existing image
image_path = r"H:\sem1_project\Cyberattack_on_smartGrid\IEEE_Feature_Analysis_Flow Duration.png"  # Replace with your actual file path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (for correct color display)

# Define x-axis values to overlay (adjust based on your dataset)
x_values = [
    1.18476352e+08, 1.19701989e+08, 9.14000000e+02, 1.15000000e+02,
    8.26572080e+07, 1.95153350e+07, 7.94847510e+07, 9.85323230e+07,
    1.27160000e+04, 1.41000000e+02
]

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))

# Show the original plot as background
ax.imshow(image)
ax.set_xticks([])  # Hide default x-ticks
ax.set_yticks([])  # Hide default y-ticks

# Get image dimensions
height, width, _ = image.shape

# Define x positions for labels (spread evenly across the x-axis)
x_positions = np.linspace(100, width - 100, len(x_values))

# Overlay X-axis values manually
for i, (pos, value) in enumerate(zip(x_positions, x_values)):
    ax.text(pos, height - 30, f'{value:.2e}', fontsize=12, fontweight='bold', 
            color='black', ha='center', va='bottom')

# Save the modified plot with x-axis values
output_path = "modified_plot_with_xaxis.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.show()

print(f"✅ Modified image saved as: {output_path}")
