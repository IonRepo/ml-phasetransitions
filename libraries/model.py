import numpy               as np
import torch               as torch
import torch.nn.functional as F

from scipy.optimize          import curve_fit
from torch_geometric.loader  import DataLoader
from torch_geometric.data    import Batch
from torch.nn                import Linear
from torch_geometric.nn      import GraphConv, global_mean_pool

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNN(
    torch.nn.Module
):
    """Graph convolution neural network.
    """
    
    def __init__(
            self,
            features_channels,
            pdropout
    ):
        """Initializes the Graph Convolutional Neural Network.

        Args:
            features_channels (int):   Number of input features.
            pdropout          (float): Dropout probability for regularization.

        Returns:
            None
        """
        
        super(GCNN, self).__init__()
        
        # Set random seed for reproducibility
        torch.manual_seed(12345)
        
        # Define graph convolution layers
        self.conv1 = GraphConv(features_channels, 512)
        self.conv2 = GraphConv(512, 512)
        
        # Define linear layers
        self.linconv1 = Linear(512, 64)
        self.linconv2 = Linear(64, 16)
        self.lin      = Linear(16, 1)
        
        self.pdropout = pdropout

    def forward(
            self,
            x,
            edge_index,
            edge_attr,
            batch,
            return_graph_embedding=False
    ):
        ## CONVOLUTION
        
        # Apply graph convolution with ReLU activation function
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()

        ## POOLING
        
        # Apply global mean pooling to reduce dimensionality
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        if return_graph_embedding:
            return x

        # Apply dropout regularization
        x = F.dropout(x, p=self.pdropout, training=self.training)
        
        # Apply linear convolution with ReLU activation function
        x = self.linconv1(x)
        x = x.relu()
        x = self.linconv2(x)
        x = x.relu()
        
        ## REGRESSION
        
        # Apply final linear layer to make prediction
        x = self.lin(x)
        return x


def make_predictions(
        reference_dataset,
        pred_dataset,
        model,
        standardized_parameters
):
    """Make predictions.
    
    Args:
        reference_dataset       (list):            The reference dataset, as a list of graphs in PyTorch Geometric's Data format.
        pred_dataset            (list):            List of graphs in PyTorch Geometric's Data format for predictions.
        model                   (torch.nn.Module): PyTorch model for predictions.
        standardized_parameters (dict):            Parameters needed to re-scale predicted properties from the dataset.
    
    Returns:
        numpy.ndarray: Predicted values.
    """

    # Read dataset parameters for re-scaling
    target_mean = standardized_parameters['target_mean']
    scale       = standardized_parameters['scale']
    target_std  = standardized_parameters['target_std']

    # Computing the predictions
    dataset = DataLoader(pred_dataset, batch_size=64, shuffle=False)

    predictions   = []
    uncertainties = []
    with torch.no_grad():  # No gradients for prediction
        for data in dataset:
            # Moving data to device
            data = data.to(device)

            # Perform a single forward pass
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch).flatten()
            
            # Estimate out of distribution
            uncer = estimate_out_of_distribution(reference_dataset, data.to_data_list(), model)
            
            # Append predictions to lists
            predictions.append(pred.cpu().detach())
            uncertainties.append(uncer)

    # Concatenate predictions and ground truths into single arrays
    predictions   = torch.cat(predictions) * target_std / scale + target_mean
    uncertainties = np.concatenate(uncertainties)
    return predictions.cpu().numpy(), uncertainties


def Helmholtz_free_energy_function(
        x,
        alpha,
        beta,
        gamma
):
    """Smoothes the Helmholtz free energy with a physically informed function.
    The parameters beta and gamma are defined positively and re-scaled for improved fitting.

    Args:
        x     (ndarray): The input array containing temperature data.
        alpha (float):   A parameter defining the baseline energy level.
        beta  (float):   A positive parameter controlling the quadratic term.
        gamma (float):   A positive parameter controlling the quartic term.

    Returns:
        smoothed_energy (ndarray): An array of smoothed Helmholtz free energy values.
    """

    # Define beta, gamma positively
    beta  = - np.abs(beta)
    gamma = - np.abs(gamma)

    # Re-scale variables
    beta  *= 1e-5
    gamma *= 1e-10

    return alpha + beta * x ** 2 + gamma * x ** 4


def compute_coefficients(
        temperatures,
        predictions
):
    """Smooth the predictions and obtain parameters from Helmholtz_free_energy_function.
    Beta and gamma are set to negative, which actually does not change anything at Helmholtz_free_energy_function fitting.
    
    Args:
        temperatures (list):          List of temperatures.
        predictions  (numpy.ndarray): Predicted values.

    Returns:
        numpy.ndarray: Coefficients of the smoothing function.
    """

    n_graphs = int(len(predictions) / len(temperatures))

    coefficients = []
    for i in range(n_graphs):
        idx0 = i     * len(temperatures)
        idxf = (i+1) * len(temperatures)

        # Concatenate array
        F_exp = predictions[idx0:idxf]

        # Fitting to a 4-degree polynomial
        try:
            _beta_, _s_beta_ = curve_fit(Helmholtz_free_energy_function,
                                         temperatures, F_exp,
                                         p0=(1, 0, 0),
                                         maxfev=3000)

            # Make beta and gamma negative, which actually does not change anything in Helmholtz_free_energy_function
            _beta_[1:] = - np.abs(_beta_[1:])
        except RuntimeError:  # Convergence not achieved
            _beta_ = [np.nan]*3

        # Append coefficients and metrics
        coefficients.append(_beta_)
    
    # Convert into array and transpose
    coefficients = np.array(coefficients).T
    return coefficients


def compute_Fv(
        temperatures,
        coefficients
):
    """Compute predictions from coefficients.
    
    Args:
        temperatures (list):          List of temperatures.
        coefficients (numpy.ndarray): Coefficients for the smoothing function.
    
    Returns:
        numpy.ndarray: Predicted vibrational energies.
    """

    Fv_pred = []
    for i in range(np.shape(coefficients)[1]):
        _beta_ = coefficients[:, i]
        vibrational_energy = Helmholtz_free_energy_function(temperatures, *_beta_)
        Fv_pred.append(vibrational_energy)

    return np.array(Fv_pred)


def estimate_out_of_distribution(
        r_dataset,
        t_dataset,
        model,
        k=3
):
    """We use the pooling from a graph neural network, which is a vector representation of the
    material, to assess the similarity between the target graph with respect to the dataset.

    Args:
        r_dataset (list):            The reference dataset, as a list of graphs in PyTorch Geometric's Data format.
        t_dataset (list):            Target dataset to assess the similarity on.
        model     (torch.nn.Module): PyTorch model for predictions.
        k         (int):             Number of neighbors for mean distance (default: 3).

    Returns:
        list ints:   Indexes of the closest example referred to the reference dataset.
        list floats: Distances to the distribution.
    """

    # Generate embeddings for target dataset
    t_batch = Batch.from_data_list(t_dataset).to(device)
    t_embeddings = model(t_batch.x, t_batch.edge_index, t_batch.edge_attr, t_batch.batch,
                         return_graph_embedding=True)

    closest_distances = torch.full((t_embeddings.size(0),), float('inf'), device=device)

    # Create a DataLoader for the reference dataset
    r_loader = DataLoader(r_dataset, batch_size=32, shuffle=False)
    all_r_embeddings = []

    # Process the reference dataset in batches using the DataLoader
    for r_batch in r_loader:
        r_batch = r_batch.to(device)
        r_embeddings = model(
            r_batch.x, r_batch.edge_index, r_batch.edge_attr, r_batch.batch,
            return_graph_embedding=True
        )
        all_r_embeddings.append(r_embeddings)

        # Compute pairwise distances
        pairwise_distances = torch.cdist(t_embeddings, r_embeddings)

        # Update global closest distances
        closest_distances = torch.minimum(closest_distances, torch.min(pairwise_distances, dim=1).values)

    r_embeddings = torch.cat(all_r_embeddings, dim=0)  # [N_r, d]

    # Pairwise distances
    pairwise = torch.cdist(r_embeddings, r_embeddings)  # [N_r, N_r]
    
    # k-NN distances (exclude self by skipping first column)
    knn_dists, _ = torch.topk(pairwise, k + 1, largest=False, dim=1)
    knn_means = knn_dists[:, 1:].mean(dim=1)  # [N_r]
    knn_means_min = torch.min(knn_means)

    # Move results to CPU and return as a list
    return np.sqrt(closest_distances.cpu().numpy() / knn_means_min.cpu().numpy())
