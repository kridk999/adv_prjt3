import torch
import numpy as np
import networkx as nx
from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split

class ErdosRenyiBaseline:
    def __init__(self, train_dataset):
        """
        Initializes the baseline by computing the empirical distribution of nodes
        and the link probability 'r' for each node count 'N' in the training data.
        """
        self.node_counts = []
        self.edges_for_N = {}
        
        # Gather data to compute empirical distribution and densities
        for data in train_dataset:
            N = data.num_nodes
            self.node_counts.append(N)
            
            # PyTorch Geometric undirected graphs typically store 2 * edges in edge_index
            num_edges = data.num_edges / 2
            
            if N not in self.edges_for_N:
                self.edges_for_N[N] = []
            self.edges_for_N[N].append(num_edges)
            
        self.density_for_N = {}
        # 2. Compute the link probability r as the graph density for each N
        for N, edges_list in self.edges_for_N.items():
            if N > 1:
                max_possible_edges = N * (N - 1) / 2
                avg_edges = np.mean(edges_list)
                self.density_for_N[N] = min(1.0, avg_edges / max_possible_edges) # Clamp to 1.0 just in case
            else:
                self.density_for_N[N] = 0.0

    def sample(self, num_samples=1):
        """
        Generates random graphs using the Erdös-Rényi model based on empirical distributions.
        """
        sampled_graphs = []
        for _ in range(num_samples):
            # 1. Sample the number of nodes N from the empirical distribution
            N = int(np.random.choice(self.node_counts))
            
            # 2. Get the pre-computed link probability r for graphs with N nodes
            r = self.density_for_N[N]
            
            # 3. Sample a random graph with N nodes and edge probability r
            # NetworkX provides a convenient function for Erdös-Rényi graph sampling
            G = nx.erdos_renyi_graph(n=N, p=r)
            sampled_graphs.append(G)
            
        return sampled_graphs

if __name__ == "__main__":
    # Load the MUTAG dataset
    dataset = TUDataset(root='./data/', name='MUTAG')

    # Split dataset identical to graph_VAE.py setup
    rng = torch.Generator().manual_seed(0)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)
    
    # Initialize baseline
    baseline = ErdosRenyiBaseline(train_dataset)
    
    # Sample 5 random graphs
    samples = baseline.sample(num_samples=5)
    
    for i, G in enumerate(samples):
        print(f"Sample {i+1}: N={G.number_of_nodes()}, Edges={G.number_of_edges()}, Link Probability (r)={baseline.density_for_N[G.number_of_nodes()]:.4f}")