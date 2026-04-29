import torch
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np
import matplotlib.pyplot as plt

def compute_wl_hash(G):
    """
    Computes the Weisfeiler-Lehman graph hash for a given NetworkX graph.
    Since our decoder only generates structural adjacency matrices (without node labels),
    we initialize node labels using their degrees to enable the WL algorithm.
    """
    for node in G.nodes():
        if "label" not in G.nodes[node]:
            # Use node degree as the initial label for structural comparison
            G.nodes[node]["label"] = str(G.degree(node))
    return nx.weisfeiler_lehman_graph_hash(G, node_attr="label")

def evaluate_novelty_and_uniqueness(model, train_dataset, num_samples=1000, device='cpu'):
    """
    Samples graphs from the VAE and compares them against the training dataset
    to calculate Novelty, Uniqueness, and Novel & Unique percentages.
    """
    model.eval()
    
    print("Extracting training graphs...")
    train_hashes = set()
    for data in train_dataset:
        # Convert PyG graph to NetworkX
        G = to_networkx(data, to_undirected=True)
        # Remove isolated nodes to focus on the core structure
        G.remove_nodes_from(list(nx.isolates(G)))
        train_hashes.add(compute_wl_hash(G))
        
    print(f"Computed WL hashes for {len(train_hashes)} unique training graphs.")
    
    print(f"Sampling {num_samples} graphs from the model...")
    with torch.no_grad():
        # Sample latent vectors from the prior
        z = model.prior().sample(torch.Size([num_samples])).to(device)
        
        # Decode into adjacency probabilities and threshold to binary
        # adj_probs = model.decoder(z).mean
        # adj_binary = (adj_probs > 0.2).cpu().numpy()

        # Decode into continuous adjacency values
        adj_continuous = model.decoder(z).mean.cpu().numpy()
        
    generated_hashes = []
    
    for i in range(num_samples):
        # Threshold the continuous values strictly for WL hashing extraction!
        # (e.g., treat anything greater than 0.5 as an edge)
        adj_binary_for_eval = (adj_continuous[i] > 0.5).astype(np.int8)
        
        # Convert the THRESHOLDED matrix to NetworkX graph
        G = nx.from_numpy_array(adj_binary_for_eval)
        G.remove_nodes_from(list(nx.isolates(G)))
        
        # Sanity check: 
        # if i < 5:  # Just peek at the first 5 samples
        #     print(f"Sample {i}: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")
        
        # If the graph is completely empty, assign a specific hash
        if G.number_of_nodes() == 0:
            generated_hashes.append("empty_graph")
        else:
            generated_hashes.append(compute_wl_hash(G))
            
    # Calculate Metrics based on Weisfeiler-Lehman hashes
    novel_graphs = [h for h in generated_hashes if h not in train_hashes]
    unique_graphs = set(generated_hashes)
    novel_and_unique_graphs = unique_graphs - train_hashes
    
    novelty = len(novel_graphs) / num_samples
    uniqueness = len(unique_graphs) / num_samples
    novel_and_unique = len(novel_and_unique_graphs) / num_samples
    
    print("\n" + "-" * 40)
    print("EVALUATION METRICS (Weisfeiler-Lehman)")
    print("-" * 40)
    print(f"Novel:            {novelty * 100:>6.2f}%")
    print(f"Unique:           {uniqueness * 100:>6.2f}%")
    print(f"Novel and unique: {novel_and_unique * 100:>6.2f}%")
    print("-" * 40 + "\n")
    
    return novelty, uniqueness, novel_and_unique

def evaluate_baseline_novelty_and_uniqueness(baseline, train_dataset, num_samples=1000):
    """
    Samples graphs from the baseline model and compares them against the training dataset
    to calculate Novelty, Uniqueness, and Novel & Unique percentages.
    """
    print("Extracting training graphs for baseline evaluation...")
    train_hashes = set()
    for data in train_dataset:
        G = to_networkx(data, to_undirected=True)
        G.remove_nodes_from(list(nx.isolates(G)))
        train_hashes.add(compute_wl_hash(G))
        
    print(f"Sampling {num_samples} graphs from the Erdös-Rényi baseline...")
    baseline_samples = baseline.sample(num_samples=num_samples)
    
    generated_hashes = []
    for G in baseline_samples:
        # Copy to avoid mutating the original sampled graph
        G_copy = G.copy()
        G_copy.remove_nodes_from(list(nx.isolates(G_copy)))
        
        if G_copy.number_of_nodes() == 0:
            generated_hashes.append("empty_graph")
        else:
            generated_hashes.append(compute_wl_hash(G_copy))
            
    # Calculate Metrics based on Weisfeiler-Lehman hashes
    novel_graphs = [h for h in generated_hashes if h not in train_hashes]
    unique_graphs = set(generated_hashes)
    novel_and_unique_graphs = unique_graphs - train_hashes
    
    novelty = len(novel_graphs) / num_samples
    uniqueness = len(unique_graphs) / num_samples
    novel_and_unique = len(novel_and_unique_graphs) / num_samples
    
    print("\n" + "-" * 40)
    print("BASELINE METRICS (Weisfeiler-Lehman)")
    print("-" * 40)
    print(f"Novel:            {novelty * 100:>6.2f}%")
    print(f"Unique:           {uniqueness * 100:>6.2f}%")
    print(f"Novel and unique: {novel_and_unique * 100:>6.2f}%")
    print("-" * 40 + "\n")
    
    return novelty, uniqueness, novel_and_unique


_is_interactive_plotting_enabled = False

def plot_training_loss(train_losses, title='VAE Training Loss', ylabel='Negative ELBO'):
    """
    Interactively plots the training curve. Updates the figure without blocking.
    """
    global _is_interactive_plotting_enabled
    if not _is_interactive_plotting_enabled:
        plt.ion()  # Enable interactive mode
        _is_interactive_plotting_enabled = True

    plt.figure(title).clf()
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    
    # Force draw the current plot (equivalent to the drawnow() function)
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()


def _get_graph_stats(graphs, is_networkx=False):
    """Helper to extract structural stats from a list of graphs."""
    degrees = []
    clusterings = []
    centralities = []
    
    for g in graphs:
        # Convert to NetworkX if it's a PyG graph from the dataset
        if not is_networkx:
            G = to_networkx(g, to_undirected=True)
        else:
            G = g
            
        # Optional: remove isolates focusing on the core structure (similar to WL eval)
        G.remove_nodes_from(list(nx.isolates(G)))
        
        if G.number_of_nodes() == 0:
            continue
            
        # 1. Node degree
        degrees.extend([d for _, d in G.degree()])
        
        # 2. Clustering coefficient
        clusterings.extend(list(nx.clustering(G).values()))
        
        # 3. Eigenvector centrality
        try:
            # use numpy version to avoid ConvergenceWarning/errors in disconnected graphs
            centrality = nx.eigenvector_centrality_numpy(G)
            centralities.extend(list(centrality.values()))
        except Exception:
            pass
            
    return np.array(degrees), np.array(clusterings), np.array(centralities)

def plot_graph_statistics(real_graphs, baseline_graphs, vae_graphs):
    """Plots a 3x3 grid comparing metrics across empirical, baseline, and VAE distributions."""
    print("Calculating statistics for Empirical graphs...")
    real_deg, real_clus, real_cent = _get_graph_stats(real_graphs, is_networkx=False)
    
    print("Calculating statistics for Baseline graphs...")
    base_deg, base_clus, base_cent = _get_graph_stats(baseline_graphs, is_networkx=True)
    
    print("Calculating statistics for VAE graphs...")
    vae_deg, vae_clus, vae_cent = _get_graph_stats(vae_graphs, is_networkx=True)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Rows: 1=Degree, 2=Clustering, 3=Centrality
    metrics_data = [
        ("Node Degree", real_deg, base_deg, vae_deg),
        ("Clustering Coefficient", real_clus, base_clus, vae_clus),
        ("Eigenvector Centrality", real_cent, base_cent, vae_cent)
    ]
    
    # Cols: Empirical, Baseline, VAE
    col_titles = ["Empirical (Training)", "Baseline (Erdös-Rényi)", "Graph VAE"]
    
    for i, (metric_name, real_d, base_d, vae_d) in enumerate(metrics_data):
        # Calculate shared bins across all models' data for 1-to-1 visual comparison
        all_data = np.concatenate([real_d, base_d, vae_d])
        bins = np.histogram_bin_edges(all_data, bins=25)
        
        data_sets = [real_d, base_d, vae_d]
        for j in range(3):
            ax = axes[i, j]
            ax.hist(data_sets[j], bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
            
            if i == 0:
                ax.set_title(col_titles[j], fontsize=14, pad=10)
            if j == 0:
                ax.set_ylabel(metric_name, fontsize=12, labelpad=10)
                
            ax.grid(alpha=0.3)
            
    plt.tight_layout()
    plt.savefig('graph_statistics.png', dpi=300)
    print("Saved graph statistics plot to 'graph_statistics.png'")
    plt.show()
if __name__ == "__main__":
    print("Import `evaluate_novelty_and_uniqueness` from this module to use it in your main script.")