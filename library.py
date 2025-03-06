import numpy               as np
import torch               as torch
import torch.nn.functional as F
import os

from scipy.optimize          import curve_fit
from torch_geometric.loader  import DataLoader
from torch_geometric.data    import Data, Batch
from torch.nn                import Linear
from torch_geometric.nn      import GraphConv, global_mean_pool
from pymatgen.core.structure import Structure

import sys
sys.path.append('../../UPC')
from GenerativeModels.libraries.graph import graph_POSCAR_encoding
from GenerativeModels.libraries.model import add_features_to_graph

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNN(torch.nn.Module):
    """Graph convolution neural network.
    """
    
    def __init__(self, features_channels, pdropout):
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

    def forward(self, x, edge_index, edge_attr, batch, return_graph_embedding=False):
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


def generate_graph(current_path, encoding_type='sphere-images', distance_thd=6):
    """Generate a graph from a given POSCAR file.
    
    Args:
        current_path (str):             Path to the POSCAR file.
        temperature  (float, optional): Temperature value. Defaults to None.
        encoding_type (str):            Encoding architecture. Defaults to 'sphere-images'.
        distance_thd (float, optional): Distance threshold for encoding. Defaults to 6.
    
    Returns:
        Data: A graph in PyTorch Geometric's Data format.
    """

    # Read POSCAR information
    structure = Structure.from_file(f'{current_path}/POSCAR')

    try:
        nodes, edges, attributes = graph_POSCAR_encoding(structure,
                                                         encoding_type=encoding_type,
                                                         distance_threshold=distance_thd
                                                         )
    
        # Generate the graph
        temp = Data(x=nodes,
                    edge_index=edges.t().contiguous(),
                    edge_attr=attributes
                   )
        return temp
    except TypeError:
        return None


def create_predictions_dataset(path_to_folder, path_to_material=False, path_to_polymorph=False):
    """Create dataset for predictions.
    
    Args:
        path_to_folder (str): Path to the folder containing POSCAR files.
    
    Returns:
        list: List of graphs in PyTorch Geometric's Data format for predictions and their labels.
    """

    dataset = []
    labels  = []

    elements = os.listdir(path_to_folder)
    if path_to_material:
        elements = ['./']

    for element in elements:
        path_to_element = os.path.join(path_to_folder, element)
        if os.path.isdir(path_to_element):
            polymorphs = os.listdir(path_to_element)
            if path_to_polymorph:
                polymorphs = ['./']

            for polymorf in polymorphs:
                path_to_polymorf = os.path.join(path_to_element, polymorf)
                if os.path.isdir(path_to_polymorf):
                    # Generate graph
                    try:
                        data = generate_graph(path_to_polymorf)
                    except ValueError:
                        print(f'Error: some element is not available for {polymorf}')
                        continue

                    if data is None:
                        continue

                    # Append graph and label
                    dataset.append(data)
                    labels.append(f'{element} {polymorf}')
    return dataset, labels


def standarize_dataset(dataset, standardized_parameters, transformation='inverse-quadratic'):
    """Standardize the dataset. Non-linear normalization is also implemented.
    
    Args:
        dataset                 (list):  List of graphs in PyTorch Geometric's Data format.
        standardized_parameters (dict):  Parameters needed to re-scale predicted properties from the dataset.
    
    Returns:
        list: Standardized dataset.
    """

    # Read dataset parameters for re-scaling
    edge_mean = standardized_parameters['edge_mean']
    feat_mean = standardized_parameters['feat_mean']
    scale     = standardized_parameters['scale']
    edge_std  = standardized_parameters['edge_std']
    feat_std  = standardized_parameters['feat_std']
    
    # Check if non-linear standardization
    if transformation == 'inverse-quadratic':
        for data in dataset:
            data.edge_attr = 1 / data.edge_attr.pow(2)

    for data in dataset:
        data.edge_attr = (data.edge_attr - edge_mean) * scale / edge_std

    for feat_index in range(dataset[0].num_node_features):
        for data in dataset:
            data.x[:, feat_index] = (data.x[:, feat_index] - feat_mean[feat_index]) * scale / feat_std[feat_index]
    return dataset


def include_temperatures(dataset, temperatures, standardized_parameters):
    """Include temperatures (standardized as well).
    
    Args:
        dataset                 (list):  List of graphs in PyTorch Geometric's Data format.
        temperatures            (list):  List of temperatures to include.
        standardized_parameters (dict):  Parameters needed to re-scale predicted properties from the dataset.

    Returns:
        list: Dataset with included temperatures.
    """

    # Read dataset parameters for re-scaling
    feat_mean = standardized_parameters['feat_mean'][-1].cpu().numpy()
    scale     = standardized_parameters['scale'].cpu().numpy()
    feat_std  = standardized_parameters['feat_std'][-1].cpu().numpy()

    # Normalize the temperature
    normalized_temperatures = (temperatures - feat_mean) * scale / feat_std

    pred_dataset = []
    for data in dataset:
        for normalized_temperature in normalized_temperatures:
            # Clone graph
            temp_data = data.clone()

            # Include the temperature as a new feature in node attributes
            temp_data = add_features_to_graph(temp_data,
                                              torch.tensor([normalized_temperature],
                                                           dtype=torch.float)
                                              )

            # Append the new graph
            pred_dataset.append(temp_data)
    return pred_dataset


def make_predictions(reference_dataset, pred_dataset, model, standardized_parameters):
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
            uncertainties.append(np.sqrt(uncer))

    # Concatenate predictions and ground truths into single arrays
    predictions   = torch.cat(predictions) * target_std / scale + target_mean
    uncertainties = np.concatenate(uncertainties)
    return predictions.cpu().numpy(), uncertainties


def Helmholtz_free_energy_function(x, alpha, beta, gamma):
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


def compute_coefficients(temperatures, predictions, uncertainties, s):
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
        s_exp = uncertainties[idx0:idxf]

        if np.all(np.max(s_exp) < s * np.max(np.abs(F_exp))):
            # Fitting to a 4-degree polynomial
            try:
                _beta_, _s_beta_ = curve_fit(Helmholtz_free_energy_function,
                                             temperatures, F_exp,
                                             p0=(1, 0, 0),
                                             maxfev=3000)
    
                # Make beta and gamma negative, which actually does not change anything in Helmholtz_free_energy_function
                _beta_[1:] = - np.abs(_beta_[1:])
            except RuntimeError:  # Convergence not achieved
                _beta_ = [np.NaN]*3
        else:
            _beta_ = [np.NaN]*3

        # Append coefficients and metrics
        coefficients.append(_beta_)
    
    # Convert into array and transpose
    coefficients = np.array(coefficients).T
    return coefficients


def compute_Fv(temperatures, coefficients):
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


def estimate_out_of_distribution(r_dataset, t_dataset, model):
    """We use the pooling from a graph neural network, which is a vector representation of the
    material, to assess the similarity between the target graph with respect to the dataset.

    Args:
        r_dataset (list):            The reference dataset, as a list of graphs in PyTorch Geometric's Data format.
        t_dataset (list):            Target dataset to assess the similarity on.
        model     (torch.nn.Module): PyTorch model for predictions.

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
    r_loader = DataLoader(r_dataset, batch_size=64, shuffle=False)

    # Process the reference dataset in batches using the DataLoader
    for r_batch in r_loader:
        r_batch = r_batch.to(device)
        r_embeddings = model(
            r_batch.x, r_batch.edge_index, r_batch.edge_attr, r_batch.batch,
            return_graph_embedding=True
        )

        # Compute pairwise distances
        pairwise_distances = torch.cdist(t_embeddings, r_embeddings)

        # Update global closest distances
        closest_distances = torch.minimum(closest_distances, torch.min(pairwise_distances, dim=1).values)

    # Move results to CPU and return as a list
    return closest_distances.cpu().numpy()
