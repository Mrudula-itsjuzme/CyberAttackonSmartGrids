import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib import rcParams

# Set the font parameters for IEEE paper standards with LARGER and BOLDER text
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12  # Increased base font size
rcParams['font.weight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.linewidth'] = 2.0  # Thicker axes lines
rcParams['axes.labelsize'] = 16  # Larger axis labels
rcParams['axes.titlesize'] = 18  # Larger titles
rcParams['xtick.labelsize'] = 14  # Larger tick labels
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 14
rcParams['figure.titlesize'] = 20
rcParams['figure.titleweight'] = 'bold'

# Read the data with sampling - use only a subset of rows
print("Reading data from CSV file with sampling...")
# Read only the timestamp column to save memory
df_sample = pd.read_csv("H:\sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv", 
                       usecols=['timestamp'], 
                       nrows=100000)  # Adjust based on memory constraints

print(f"Using {len(df_sample)} samples out of the full dataset")

# Create figure with proper spacing
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))  # Larger figure size
fig.suptitle('Time Series Analysis of Cyber Attacks', fontsize=22, fontweight='bold', y=0.98)

# 1. Distribution histogram with sampled data
sns.histplot(data=df_sample, x='timestamp', bins=30, ax=ax1, 
            color='skyblue', edgecolor='black', linewidth=2.0)
ax1.set_title('Distribution of Timestamp', fontweight='bold', fontsize=18)
ax1.set_xlabel('Timestamp (s)', fontweight='bold', fontsize=16)
ax1.set_ylabel('Frequency', fontweight='bold', fontsize=16)
ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
ax1.ticklabel_format(axis='y', style='plain')
ax1.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
for spine in ax1.spines.values():
    spine.set_linewidth(2.0)
# Make legend text larger and bolder
legend = ax1.legend(['Frequency'], loc='upper right', frameon=True, framealpha=0.9)
plt.setp(legend.get_texts(), fontsize=14, fontweight='bold')

# 2. Q-Q plot with sampled data
res = stats.probplot(df_sample['timestamp'], plot=ax2)
ax2.set_title('Q-Q Plot of Timestamp', fontweight='bold', fontsize=18)
ax2.set_xlabel('Theoretical Quantiles', fontweight='bold', fontsize=16)
ax2.set_ylabel('Sample Quantiles', fontweight='bold', fontsize=16)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax2.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
for spine in ax2.spines.values():
    spine.set_linewidth(2.0)
# Make the line and markers bolder
for line in ax2.get_lines():
    line.set_markeredgewidth(2.0)
    line.set_markersize(8)
    line.set_linewidth(2.0)

# 3. Box plot with sampled data
timestamp_data = df_sample['timestamp'].values
q1, q3 = np.percentile(timestamp_data, [25, 75])
iqr = q3 - q1
lower_whisker = max(np.min(timestamp_data), q1 - 1.5 * iqr)
upper_whisker = min(np.max(timestamp_data), q3 + 1.5 * iqr)
y_min = lower_whisker - 0.1 * (upper_whisker - lower_whisker)
y_max = upper_whisker + 0.1 * (upper_whisker - lower_whisker)

box_plot = ax3.boxplot(timestamp_data, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='skyblue', color='black', linewidth=2.0),
                     whiskerprops=dict(color='black', linewidth=2.0),
                     capprops=dict(color='black', linewidth=2.0),
                     medianprops=dict(color='black', linewidth=3.0),  # Extra bold median line
                     flierprops=dict(markeredgewidth=2.0, markersize=8))
ax3.set_title('Box Plot of Timestamp', fontweight='bold', fontsize=18)
ax3.set_ylabel('Timestamp (s)', fontweight='bold', fontsize=16)
ax3.set_xlabel('Timestamp', fontweight='bold', fontsize=16)  # Added X-axis label
ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax3.set_ylim(y_min, y_max)
for spine in ax3.spines.values():
    spine.set_linewidth(2.0)
ax3.set_xticks([1])
ax3.set_xticklabels(['Timestamp'], fontweight='bold', fontsize=14)
ax3.grid(True, axis='y', linestyle='--', alpha=0.7, linewidth=1.5)

# 4. Violin plot with sampled data
violin = sns.violinplot(y=df_sample['timestamp'], ax=ax4, color='skyblue', linewidth=2.0)
# Make violin plot lines thicker
for l in violin.collections:
    l.set_linewidth(2.0)
ax4.set_title('Violin Plot of Timestamp', fontweight='bold', fontsize=18)
ax4.set_ylabel('Timestamp (s)', fontweight='bold', fontsize=16)
ax4.set_xlabel('Distribution', fontweight='bold', fontsize=16)  # Made X-axis label more visible
ax4.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
for spine in ax4.spines.values():
    spine.set_linewidth(2.0)
ax4.grid(True, axis='y', linestyle='--', alpha=0.7, linewidth=1.5)

# Ensure x-axis is visible for all plots
for ax in [ax1, ax2, ax3, ax4]:
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.xaxis.set_tick_params(which='both', width=2.0, length=6, labelsize=14, labelcolor='black')
    
    # Make tick labels bolder for all plots
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

# Adjust layout with proper spacing
plt.tight_layout()
fig.subplots_adjust(top=0.92, wspace=0.25, hspace=0.3)  # Added more space between plots

# Save high resolution figure for IEEE paper
plt.savefig('time_series_analysis.png', dpi=600, bbox_inches='tight')  # Higher DPI for better print quality
plt.savefig('time_series_analysis.pdf', format='pdf', bbox_inches='tight')

plt.show()

# Print statistics with better formatting
print("\n============ Basic Statistics of the Timestamp Column ============")
stats_df = df_sample['timestamp'].describe()
print(stats_df.to_string())
print("\n================================================================")