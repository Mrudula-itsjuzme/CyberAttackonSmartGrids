import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Read the data
print("Reading data from CSV file...")
data_path = os.getenv("CYBERGRID_DATA_PATH", "intermediate_combined_data.csv")
print(f"Using dataset: {data_path}")
df = pd.read_csv(
    data_path,
    low_memory=False,
)

# Create figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
plt.suptitle('Time Series Analysis of Cyber Attacks', y=0.95)

# 1. Distribution histogram
sns.histplot(data=df, x='timestamp', bins=30, ax=ax1, 
            color='lightblue', edgecolor='black')
ax1.set_title('Distribution of timestamp')
ax1.set_xlabel('timestamp')
ax1.set_ylabel('Frequency')
ax1.ticklabel_format(axis='x', style='sci', scilimits=(9,9))
ax1.ticklabel_format(axis='y', style='plain')
# Add legend
ax1.legend(['Frequency'], loc='upper right')

# 2. Q-Q plot
stats.probplot(df['timestamp'], plot=ax2)
ax2.set_title('Q-Q Plot of timestamp')
ax2.ticklabel_format(axis='y', style='sci', scilimits=(9,9))
# Add grid
ax2.grid(True, linestyle=':', alpha=0.6)

# 3. Box plot - this needs special attention
plt.sca(ax3)
box_plot = plt.boxplot(df['timestamp'], vert=True, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', color='black'),
                      whiskerprops=dict(color='black'),
                      capprops=dict(color='black'),
                      medianprops=dict(color='black'))
ax3.set_title('Box Plot of timestamp')
ax3.set_ylabel('timestamp')
ax3.ticklabel_format(axis='y', style='sci', scilimits=(9,9))
# Remove x-axis labels as they're not needed
ax3.set_xticks([])

# 4. Violin plot
sns.violinplot(y=df['timestamp'], ax=ax4, color='lightblue')
ax4.set_title('Violin Plot of timestamp')
ax4.set_ylabel('timestamp')
ax4.ticklabel_format(axis='y', style='sci', scilimits=(9,9))

# Adjust layout
plt.tight_layout()

plt.show()

print("Basic statistics of the timestamp column:")
print(df['timestamp'].describe())