import torch
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np

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

if __name__ == "__main__":
    print("Import `evaluate_novelty_and_uniqueness` from this module to use it in your main script.")