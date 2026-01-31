import numpy               as np
import torch               as torch
import torch.nn.functional as F
import torch.nn            as nn
import os

from scipy.interpolate      import RBFInterpolator
from scipy.optimize         import curve_fit
from torch_geometric.loader import DataLoader
from torch.nn               import Linear
from torch_geometric.nn     import GraphConv, global_mean_pool

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
            batch,
            return_graph_embedding=False
    ):
        ## CONVOLUTION
        
        # Apply graph convolution with ReLU activation function
        x = self.conv1(batch.x, batch.edge_index, batch.edge_attr)
        x = x.relu()
        x = self.conv2(x, batch.edge_index, batch.edge_attr)
        x = x.relu()

        ## POOLING
        
        # Apply global mean pooling to reduce dimensionality
        x = global_mean_pool(x, batch.batch)  # [batch_size, hidden_channels]

        # Apply dropout regularization
        x = F.dropout(x, p=self.pdropout, training=self.training)
        
        # Apply linear convolution with ReLU activation function
        x = self.linconv1(x)
        x = x.relu()
        x = self.linconv2(x)
        if return_graph_embedding:
            return x
        x = x.relu()
        
        ## REGRESSION
        
        # Apply final linear layer to make prediction
        x = self.lin(x)
        return x


def make_predictions(
        pred_dataset,
        model,
        standardized_parameters,
        r_uncertainty_data,
        interpolator
):
    """Make predictions.
    
    Args:
        pred_dataset            (list):            List of graphs in PyTorch Geometric's Data format for predictions.
        model                   (torch.nn.Module): PyTorch model for predictions.
        standardized_parameters (dict):            Parameters needed to re-scale predicted properties from the dataset.
        r_uncertainty_data      (dict):            Uncertainty data for the reference dataset.
        interpolator            (RBFInterpolator): Fitted RBFInterpolator object for uncertainty estimation.
    
    Returns:
        numpy.ndarray: Predicted values.
    """
    # Read dataset parameters for re-scaling
    target_mean  = standardized_parameters['target_mean']
    target_scale = standardized_parameters['scale']
    target_std   = standardized_parameters['target_std']

    # Read uncertainty parameters for re-scaling
    uncert_mean  = r_uncertainty_data['uncert_mean']
    uncert_std   = r_uncertainty_data['uncert_std']
    uncert_scale = r_uncertainty_data['uncert_scale']

    # Computing the predictions
    dataset = DataLoader(pred_dataset, batch_size=16, shuffle=False)

    predictions   = []
    uncertainties = []
    with torch.no_grad():  # No gradients for prediction
        for data in dataset:
            # Move data to device
            data = data.to(device)

            # Perform a single forward pass
            pred = model(data).flatten()
            
            # Estimate uncertainty
            uncer = estimate_uncertainty(data.to_data_list(), model, interpolator)
            
            # Append predictions to lists
            predictions.append(pred.cpu().detach())
            uncertainties.append(uncer)

    # Concatenate predictions and ground truths into single arrays
    predictions   = torch.cat(predictions) * target_std / target_scale + target_mean
    uncertainties = np.concatenate(uncertainties) * uncert_std / uncert_scale + uncert_mean  # De-standardize predictions
    return predictions.cpu().numpy(), uncertainties


def predict_single_material(
        data,
        temperatures,
        model,
        standardized_parameters,
        uncertainty_data,
        interpolator
):
    """Make predictions for a single material across temperatures.
    
    Adds predictions, uncertainties, and polynomial coefficients directly to the Data object.
    
    Args:
        data                    (Data):            Single graph in PyTorch Geometric's Data format.
        temperatures            (np.ndarray):      Array of temperatures for predictions.
        model                   (torch.nn.Module): PyTorch model for predictions.
        standardized_parameters (dict):            Parameters for standardization/de-standardization.
        uncertainty_data        (dict):            Uncertainty data for estimation.
        interpolator            (RBFInterpolator): Fitted RBFInterpolator for uncertainty.
    
    Returns:
        Data: The input data object with added attributes (predictions, uncertainties, coefficients).
    """
    from libraries.dataset import standardize_dataset_from_keys, include_temperatures
    
    # Standardize the single data point
    std_data = standardize_dataset_from_keys([data], standardized_parameters)[0]
    
    # Include temperatures to create batch
    std_data_w_temp = include_temperatures([std_data], temperatures, standardized_parameters)
    
    # Make predictions
    predictions, uncertainties = make_predictions(
        std_data_w_temp, model, standardized_parameters, uncertainty_data, interpolator
    )

    # Process uncertainties
    max_uncert    = np.max(np.abs(uncertainties))
    uncertainties = np.ones(len(temperatures)) * max_uncert
    
    # Compute coefficients
    coefficients = compute_coefficients(temperatures, predictions)
    
    # Add to original data object
    data.predictions   = predictions
    data.uncertainties = uncertainties
    data.coefficients  = coefficients
    
    return data


def compute_transition_temperature_uncertainty(
        d_E0,
        d_alpha,
        d_beta,
        d_gamma,
        uncert_Fv,
        c_temperature
):
    """Compute uncertainty in transition temperature given free energy uncertainty.
    
    Solves the quadratic equation for transition temperature with uncertainty bounds
    and returns the minimum absolute deviation from the critical temperature.
    
    Args:
        d_E0          (float): Difference in ground state energies (meV/atom).
        d_alpha       (float): Difference in alpha coefficients.
        d_beta        (float): Difference in beta coefficients (scaled by 1e-5).
        d_gamma       (float): Difference in gamma coefficients (scaled by 1e-10).
        uncert_Fv     (float): Uncertainty in free energy (meV/atom).
        c_temperature (float): Critical transition temperature (K).
    
    Returns:
        float: Uncertainty in transition temperature (K), or np.nan if no valid roots.
    """
    # Two possible transition temperatures for negative uncertainty
    root_1_n = (-d_beta + np.sqrt(d_beta**2 - 4 * (d_E0 + d_alpha - uncert_Fv) * d_gamma)) / (2 * d_gamma)
    root_2_n = (-d_beta - np.sqrt(d_beta**2 - 4 * (d_E0 + d_alpha - uncert_Fv) * d_gamma)) / (2 * d_gamma)
    
    # Two possible transition temperatures for positive uncertainty
    root_1_p = (-d_beta + np.sqrt(d_beta**2 - 4 * (d_E0 + d_alpha + uncert_Fv) * d_gamma)) / (2 * d_gamma)
    root_2_p = (-d_beta - np.sqrt(d_beta**2 - 4 * (d_E0 + d_alpha + uncert_Fv) * d_gamma)) / (2 * d_gamma)
    
    # Filter valid (real and positive) temperatures
    uncert_temperature = np.array(
        [x for x in [np.sqrt(root_1_n), np.sqrt(root_2_n), np.sqrt(root_1_p), np.sqrt(root_2_p)] if x >= 0]
    )
    
    # Return minimum deviation from critical temperature
    return np.min(np.abs(uncert_temperature - c_temperature)) if len(uncert_temperature) > 0 else np.nan


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


def estimate_uncertainty(
        t_dataset,
        model,
        interpolator
):
    """Estimate uncertainty on predictions and whether the target dataset is in the interpolation regime.

    Args:
        r_dataset          (list):            Reference dataset, as a list of graphs in PyTorch Geometric's Data format.
        t_dataset          (list):            Target dataset, as a list of graphs in PyTorch Geometric's Data format.
        model              (torch.nn.Module): The trained model.
        r_uncertainty_data (dict):            Uncertainty data for the reference dataset.

    Returns:
        numpy.ndarray: Uncertainties of the target dataset.
        numpy.ndarray: Boolean array indicating if the target embeddings
    """
    # Create a DataLoader for the target dataset
    t_embeddings = extract_embeddings(t_dataset, model)

    # Determine the uncertainty on the predictions
    t_uncertainties = interpolator(t_embeddings)
    return t_uncertainties


def fit_interpolator(
    r_uncertainty_data,
    r_dataset,
    model
):
    """
    Fit RBFInterpolator to reference uncertainties.

    Args:
        r_uncertainty_data (dict):            Uncertainty data for the reference dataset.
        r_dataset          (list):            Reference dataset, as a list of graphs in PyTorch Geometric's Data format.
        model              (torch.nn.Module): The trained model.

    Returns:
        RBFInterpolator: Fitted RBFInterpolator object.
    """
    # Create a DataLoader for the reference dataset
    r_embeddings = extract_embeddings(r_dataset, model)

    # Extract labels from r_dataset
    r_labels = [data.label for data in r_dataset]

    # Extract uncertainties for each reference example
    r_uncertainties = np.asarray([r_uncertainty_data[label] for label in r_labels])

    # Interpolate uncertainties for the target dataset
    interpolator = RBFInterpolator(r_embeddings, r_uncertainties, smoothing=0)
    return interpolator


def extract_embeddings(
        dataset,
        model
):
    """Extract embeddings from a dataset using a trained model.

    Args:
        dataset (list):            Dataset, as a list of graphs in PyTorch Geometric's Data format.
        model   (torch.nn.Module): The trained model.

    Returns:
        numpy.ndarray: Embeddings extracted from the dataset.
    """
    # Create a DataLoader for the dataset
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Process the reference dataset in batches using the DataLoader
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embedding = model(batch, return_graph_embedding=True).cpu().numpy()
            embeddings.append(embedding)

    # Concatenate all batch embeddings into a single array
    return np.concatenate(embeddings, axis=0)
