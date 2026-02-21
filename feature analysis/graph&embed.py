# Script 3: Graph Analysis and Embedding Techniques

import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd

class GraphAnalysis:
    def __init__(self, df, anomalies):
        self.df = df
        self.anomalies = anomalies

    def create_temporal_graph(self, max_nodes=1000):
        print("Creating temporal graph for anomaly analysis...")
        G = nx.Graph()
        for i in tqdm(range(len(self.df) - 1), desc="Adding Edges"):
            if self.anomalies[i] == -1:
                G.add_edge(f"T{i}", f"T{i+1}", weight=1)
        if len(G.nodes) > max_nodes:
            sampled_nodes = np.random.choice(list(G.nodes), size=max_nodes, replace=False)
            G = G.subgraph(sampled_nodes)
        # Add more random edges to simulate denser connections
        for _ in range(200):  # Add 200 random connections
            if len(G.nodes) > 1:
                node1, node2 = np.random.choice(list(G.nodes), size=2, replace=False)
                G.add_edge(node1, node2, weight=1)
        return G

    def visualize_graph(self, G, output_path="temporal_graph.png"):
        print("Visualizing the temporal graph...")
        try:
            import matplotlib
            matplotlib.use("Agg")  # Use a non-interactive backend

            # Extract the largest connected component for visualization
            components = list(nx.connected_components(G))
            largest_component = max(components, key=len)
            subgraph = G.subgraph(largest_component)

            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(subgraph, k=0.05)  # Adjust 'k' for better spacing
            nx.draw_networkx_nodes(subgraph, pos, node_size=100, node_color="blue", alpha=0.8)
            nx.draw_networkx_edges(subgraph, pos, edge_color="gray", alpha=0.5)
            plt.title("Temporal Graph Visualization (Largest Component)", fontsize=16)
            plt.savefig(output_path, format="png")  # Save as PNG
            plt.close()
            print(f"Graph saved to {output_path}")
        except Exception as e:
            print(f"Error during visualization: {str(e)}")

    def calculate_graph_metrics(self, G):
        print("Calculating graph metrics...")
        metrics = {
            "Number of Nodes": len(G.nodes),
            "Number of Edges": len(G.edges),
            "Average Degree": np.mean([d for n, d in G.degree]),
            "Density": nx.density(G),
        }
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        return metrics

if __name__ == "__main__":
    print("Running Script 3: Graph Analysis")

    # Simulated data
    data = np.random.rand(1000, 10)
    anomalies = np.random.choice([-1, 1], size=1000, p=[0.5, 0.5])  # Increased anomaly probability
    df = pd.DataFrame(data, columns=[f"Feature_{i}" for i in range(10)])

    # Perform graph analysis
    graph_analyzer = GraphAnalysis(df, anomalies)
    G = graph_analyzer.create_temporal_graph()
    graph_analyzer.visualize_graph(G, output_path="temporal_graph.png")
    metrics = graph_analyzer.calculate_graph_metrics(G)

    print("Script 3 completed: Graph Analysis")
