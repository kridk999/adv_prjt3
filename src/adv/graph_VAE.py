import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
import numpy as np
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
from metrics import evaluate_novelty_and_uniqueness, plot_training_loss, plot_graph_statistics
from graph_baseline import ErdosRenyiBaseline
import networkx as nx

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)




class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Gaussian decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.log_std = nn.Parameter(torch.zeros(784) - 2, requires_grad=True)


    def forward(self, z):
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        mean = self.decoder_net(z)  # (batch, 28, 28)
        std = torch.exp(self.log_std)
        return td.Independent(td.Normal(loc=mean, scale=std), 1)




class GraphVAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(GraphVAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, data, beta=1.0):
        q = self.encoder(data)
        z = q.rsample()
        
        target_adj = to_dense_adj(data.edge_index, data.batch, max_num_nodes=self.decoder.max_nodes)
        

        reconstruction_loss = self.decoder(z).log_prob(target_adj)
        kl_divergence = q.log_prob(z) - self.prior().log_prob(z)
        
        return torch.mean(reconstruction_loss - beta * kl_divergence, dim=0)

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x, beta=1.0):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x, beta)



    
class GCNEncoderNet(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logstd = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Message passing
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Graph-level pooling
        x = global_mean_pool(x, batch)
        
        return self.fc_mu(x), self.fc_logstd(x)
    
class DenseAdjacencyDecoder(nn.Module):
    def __init__(self, decoder_net, max_nodes):
        """
        Define a dense adjacency decoder distribution based on a given network.

        Parameters:
        decoder_net: [torch.nn.Module]
            The decoder network that takes a tensor of dim `(batch_size, latent_dim)`
            and outputs a tensor of dimension `(batch_size, max_nodes * max_nodes)`.
        max_nodes: [int]
            The maximum number of nodes in the graph to reshape the output.
        """
        super().__init__()
        self.max_nodes = max_nodes
        self.decoder_net = decoder_net

    def forward(self, z):
        batch_size = z.size(0)
        
        adj_logits = self.decoder_net(z).view(batch_size, self.max_nodes, self.max_nodes)
        
        adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2.0
        
        return td.Independent(td.Bernoulli(logits=adj_logits), 2)
    

class ContinuousAdjacencyDecoder(nn.Module):
    def __init__(self, decoder_net, max_nodes):
        super().__init__()
        self.max_nodes = max_nodes
        self.decoder_net = decoder_net
        # Learnable global standard deviation parameter
        self.log_std = nn.Parameter(torch.zeros(1) - 2, requires_grad=True)

    def forward(self, z):
        batch_size = z.size(0)
        
        # Decode to continuous mean values
        adj_mean = self.decoder_net(z).view(batch_size, self.max_nodes, self.max_nodes)
        
        # Symmetrize to ensure undirected graph properties
        adj_mean = (adj_mean + adj_mean.transpose(1, 2)) / 2.0
        
        # Optional: Apply torch.sigmoid(adj_mean) here if you need 
        # the continuous weights to be strictly bounded between 0.0 and 1.0.
        
        std = torch.exp(self.log_std).expand_as(adj_mean)
        
        # Return independent Normal distributions for the continuous edges
        return td.Independent(td.Normal(loc=adj_mean, scale=std), 2)

class GaussianGraphEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, data):
        mean, log_std = self.encoder_net(data)
        
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(log_std)), 1)


def train(model, optimizer, data_loader, epochs, device, beta=1.0, scheduler=None):
    """
    Train a VAE model with interactive plotting and optional LR scheduling.
    """
    from metrics import plot_training_loss  # Import the new plotting function
    
    model.train()

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    epoch_losses = []  # List to track the average loss per epoch

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        
        running_epoch_loss = 0.0
        num_batches = 0
        
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x, beta)
            loss.backward()
            optimizer.step()
            
            running_epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()
            
        # At the end of each epoch, compute the average loss and update the plot
        avg_epoch_loss = running_epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        plot_training_loss(epoch_losses)
        
        # Step the scheduler if it was provided
        if scheduler is not None:
            scheduler.step()

def evalELBO(model, test_loader, device):
    total_elbo = 0.0
    num_batches = 0
    with torch.no_grad():
        for x, _ in test_loader:
            elbo = model.elbo(x.to(device))
            total_elbo += elbo.item()
            num_batches += 1
    print(f"Average ELBO: {total_elbo / num_batches:.4f}")


def train_and_eval_multiple_runs(model, optimizer, train_loader, test_loader, epochs, device, num_runs=10, prior_type=None):
    """
    Train the model multiple times and evaluate the ELBO for each trained model.

    Parameters:
    model: [VAE]
        The VAE model to train and evaluate.
    optimizer: [torch.optim.Optimizer]
        The optimizer to use for training.
    train_loader: [torch.utils.data.DataLoader]
        The data loader for the training set.
    test_loader: [torch.utils.data.DataLoader]
        The data loader for the test set.
    epochs: [int]
        Number of epochs to train for each run.
    device: [torch.device]
        The device to use for training and evaluation.
    num_runs: [int]
        Number of training and evaluation runs.

    Returns:
    mean_elbo: [float]
        The mean ELBO over the runs.
    std_elbo: [float]
        The standard deviation of the ELBO over the runs.
    """
    elbo_values = []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, train_loader, epochs, device)

        total_elbo = 0.0
        num_batches = 0
        with torch.no_grad():
            for x, _ in test_loader:
                elbo = model.elbo(x.to(device))
                total_elbo += elbo.item()
                num_batches += 1
        avg_elbo = total_elbo / num_batches
        elbo_values.append(avg_elbo)
        print(f"Run {run + 1} ELBO: {avg_elbo:.4f}")

    mean_elbo = np.mean(elbo_values)
    std_elbo = np.std(elbo_values)

    # Write the mean and standard deviation to the file
    with open(f"elbo_values_{prior_type}.txt", "w") as f:
        for elbo in elbo_values:
            f.write(f"{elbo:.4f}\n")    
        f.write(f"\nMean: {mean_elbo:.4f}, Std: {std_elbo:.4f}\n")
    return mean_elbo, std_elbo

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, nargs='?', default='sample', choices=['train', 'sample', 'eval', 'train-multiple', 'plot-stats'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='2.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='M', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--num-components', type=int, default=3, metavar='K', help='number of MoG prior components (default: %(default)s)')
    parser.add_argument('--beta', type=float, default=1.0, help='beta value for beta-VAE (default: %(default)s)')
    parser.add_argument('--max-nodes', type=int, default=28, help='maximum number of nodes in the graph (default: %(default)s)')
    parser.add_argument('--use-scheduler', action='store_true', help='enable ExponentialLR scheduler (default: False)')
    parser.add_argument('--plot-stats', action='store_true', help='plot structural statistics 3x3 histogram grid')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device
    latent_dim = args.latent_dim
    max_nodes = args.max_nodes



    decoder_net = nn.Sequential(
        nn.Linear(latent_dim, 400),
        nn.ReLU(),
        nn.Linear(400, max_nodes * max_nodes)
    ).to(device)

    encoder_net = GCNEncoderNet(7, 64, latent_dim).to(device)
    encoder = GaussianGraphEncoder(encoder_net)
    decoder = ContinuousAdjacencyDecoder(decoder_net, max_nodes)
    prior = GaussianPrior(latent_dim)


    # Load the MUTAG dataset
    dataset = TUDataset(root='./data/', name='MUTAG').to(device)
    node_feature_dim = dataset.num_node_features

    
    # Split dataset
    rng = torch.Generator().manual_seed(0)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


    model = GraphVAE(prior, decoder, encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Initialize the scheduler if the flag is passed
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        print("Using ExponentialLR scheduler with gamma=0.995")
    else:
        scheduler = None

    if args.mode == 'train':
        train(model, optimizer, train_loader, args.epochs, device, args.beta, scheduler=scheduler)
        torch.save(model.state_dict(), args.model)
    
    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=device))
        
        print("\n--- Evaluating VAE Model ---")
        evaluate_novelty_and_uniqueness(model, train_dataset, num_samples=1000, device=device)
        
        print("\n--- Evaluating Erdös-Rényi Baseline ---")
        from metrics import evaluate_baseline_novelty_and_uniqueness
        
        baseline = ErdosRenyiBaseline(train_dataset)
        evaluate_baseline_novelty_and_uniqueness(baseline, train_dataset, num_samples=1000)

    elif args.mode == 'plot-stats':
        print("\n--- Generating Graphs for Statistics ---")
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        
        num_samples = 1000
        
        # 1. Baseline generation
        baseline = ErdosRenyiBaseline(train_dataset)
        baseline_graphs = baseline.sample(num_samples=num_samples)
        
        # 2. VAE generation
        with torch.no_grad():
            z = prior().sample(torch.Size([num_samples])).to(device)
            adj_continuous = decoder(z).mean.cpu().numpy()
            
        vae_graphs = []
        for i in range(num_samples):
            # Threshold to binary representation as in your WL metric
            adj_binary = (adj_continuous[i] > 0.5).astype(np.int8)
            G = nx.from_numpy_array(adj_binary)
            vae_graphs.append(G)
            
        # 3. Real graphs come from train_dataset
        # Plot the grid
        plot_graph_statistics(train_dataset, baseline_graphs, vae_graphs)


