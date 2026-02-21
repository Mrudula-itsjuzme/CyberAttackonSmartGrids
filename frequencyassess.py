import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# File path to the dataset
file_path = r"D:\sem1_project\Cyberattack_on_smartGrid\intermediate_combined_data.csv"

# Load the dataset
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    print(f"Columns in the dataset: {df.columns.tolist()}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Strip column names to remove extra spaces
df.columns = df.columns.str.strip()

# Check for appropriate columns
if 'src_ip' not in df.columns or 'dst_ip' not in df.columns:
    print("Error: Dataset must contain 'src_ip' and 'dst_ip' columns for source and target.")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

# Create a graph from the edge list using 'src_ip' and 'dst_ip'
try:
    G = nx.from_pandas_edgelist(df, source='src_ip', target='dst_ip')
    print(f"Graph constructed successfully.")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
except Exception as e:
    print(f"Error constructing graph: {e}")
    exit()

# Frequency analysis
# Calculate degree distribution
degree_freq = nx.degree_histogram(G)
degrees = range(len(degree_freq))

# Print degree distribution summary
print("Degree distribution summary:")
for degree, count in enumerate(degree_freq):
    if count > 0:
        print(f"Degree {degree}: {count} nodes")

# Plot degree distribution
plt.figure(figsize=(10, 6))
plt.bar(degrees, degree_freq, color='skyblue', alpha=0.7)
plt.title("Degree Frequency Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.yscale('log')  # Log scale for better visualization of large ranges
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and show the plot
output_plot = r"D:\sem1_project\Cyberattack_on_smartGrid\degree_distribution_plot.png"
plt.savefig(output_plot)
plt.show()
print(f"Degree distribution plot saved to {output_plot}")
