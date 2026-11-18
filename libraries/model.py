import numpy               as np
import torch               as torch
import torch.nn.functional as F
import torch.nn            as nn
import os

from scipy.interpolate      import RBFInterpolator
from scipy.optimize         import curve_fit
from torch_geometric.loader import DataLoader
from torch.nn               import Linear
from torch_geometric.nn     import TransformerConv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation

import libraries.electronic as cle

# Checking if pytorch can run in GPU, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCNN(
    torch.nn.Module
):
    """Graph neural network.

    Message passing is a stack of pre-norm residual blocks. Each block has
        1) multi-head graph attention that uses the edge features (TransformerConv, with a learned gate between the
           aggregated message and the node's own features, beta=True), and
        2) a gated (GLU) feed-forward layer.
    The edge attribute (standardized inverse squared distance) goes through a small MLP, so its sign no longer
    matters. Pooling concatenates the global mean and a learned soft-attention pooling. The temperature is still the
    last node feature. The linear head (and the 16-d graph embedding used for the uncertainty) is unchanged.
    """
    
    def __init__(
            self,
            features_channels,
            pdropout,
            hidden_channels=128,
            heads=4,
            n_layers=3,
            edge_channels=32
    ):
        """Initializes the Graph Neural Network.

        Args:
            features_channels (int):   Number of input features.
            pdropout          (float): Dropout probability for regularization.
            hidden_channels   (int):   Width of the message-passing layers.
            heads             (int):   Attention heads.
            n_layers          (int):   Number of message-passing blocks.
            edge_channels     (int):   Width of the edge embedding.

        Returns:
            None
        """
        
        super(GCNN, self).__init__()
        
        # Set random seed for reproducibility
        torch.manual_seed(12345)
        
        # Node and edge embeddings
        self.embed      = Linear(features_channels, hidden_channels)
        self.edge_embed = nn.Sequential(Linear(1, edge_channels), nn.SiLU(), Linear(edge_channels, edge_channels))
        
        # Message-passing blocks
        self.attn_norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(n_layers)])
        self.convs      = nn.ModuleList([TransformerConv(hidden_channels, hidden_channels // heads, heads=heads,
                                                         edge_dim=edge_channels, beta=True)
                                         for _ in range(n_layers)])
        self.ffn_norms  = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(n_layers)])
        self.ffn_in     = nn.ModuleList([Linear(hidden_channels, 2 * hidden_channels) for _ in range(n_layers)])
        self.ffn_out    = nn.ModuleList([Linear(hidden_channels, hidden_channels) for _ in range(n_layers)])
        
        # Attention pooling
        self.pool = AttentionalAggregation(gate_nn=Linear(hidden_channels, 1))
        
        # Define linear layers
        self.linconv1 = Linear(2 * hidden_channels, 64)
        self.linconv2 = Linear(64, 16)
        self.lin      = Linear(16, 1)
        
        self.pdropout = pdropout

    def forward(
            self,
            batch,
            return_graph_embedding=False
    ):
        ## MESSAGE PASSING
        
        x = self.embed(batch.x)
        e = self.edge_embed(batch.edge_attr)
        for attn_norm, conv, ffn_norm, ffn_in, ffn_out in zip(self.attn_norms, self.convs, self.ffn_norms,
                                                              self.ffn_in, self.ffn_out):
            x = x + conv(attn_norm(x), batch.edge_index, e)
            x = x + ffn_out(F.glu(ffn_in(ffn_norm(x)), dim=-1))

        ## POOLING
        
        # Global mean pooling and soft-attention pooling
        x = torch.cat([global_mean_pool(x, batch.batch), self.pool(x, batch.batch)], dim=-1)  # [batch_size, 2 * hidden_channels]

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
    std_data = standardize_dataset_from_keys([data.clone()], standardized_parameters)[0]  # clone: the input graph is left untouched
    
    # Include temperatures to create batch
    std_data_w_temp = include_temperatures([std_data], temperatures, standardized_parameters)
    
    # Make predictions
    predictions, uncertainties = make_predictions(
        std_data_w_temp, model, standardized_parameters, uncertainty_data, interpolator
    )

    # Process uncertainties - take max as conservative estimate
    max_uncert    = np.max(np.abs(uncertainties))
    uncertainties = np.ones(len(temperatures)) * max_uncert
    
    # NEW: electronic free energy from the DOS (zero when the material has no DOS file), meV/atom.
    # The smooth fit below is done on F_v + F_el, so the transitions include the electronic contribution.
    electronic = np.zeros(len(temperatures))
    if getattr(data, 'dos_file', None):
        electronic = cle.electronic_free_energy_from_file(data.dos_file, temperatures, natoms=data.x.size(0),
                                                          bandgap=getattr(data, 'bandgap', None))
    
    # Compute coefficients - compute_coefficients expects predictions for single material
    coefficients = compute_coefficients(temperatures, predictions + electronic)
    
    # Add to original data object
    data.predictions   = predictions
    data.electronic    = electronic
    data.uncertainties = uncertainties
    data.coefficients  = coefficients
    
    return data


def transition_temperatures(
        d_E0_alpha,
        d_beta,
        d_gamma
):
    """Temperatures (K) at which two polymorphs have the same free energy: d_E0_alpha + d_beta T^2 + d_gamma T^4 = 0.

    The roots of the quadratic in T^2 are computed with the numerically stable form (the textbook formula loses the
    small root to cancellation when d_gamma is small, and divides by zero when it vanishes), and only real, positive
    ones are returned, as T.

    Args:
        d_E0_alpha (float): E0 + alpha difference between the two polymorphs (meV/atom).
        d_beta     (float): Difference in beta coefficients (scaled by 1e-5, already re-scaled to meV/K^2 here).
        d_gamma    (float): Difference in gamma coefficients (already re-scaled to meV/K^4).

    Returns:
        numpy.ndarray: Sorted transition temperatures.
    """
    c, b, a = d_E0_alpha, d_beta, d_gamma
    if not np.all(np.isfinite([a, b, c])):
        return np.array([])

    if a == 0:  # Linear in T^2
        roots = [-c / b] if b != 0 else []
    else:
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return np.array([])
        q = -0.5 * (b + (1.0 if b >= 0 else -1.0) * np.sqrt(discriminant))
        roots = [q / a] + ([c / q] if q != 0 else [])
    return np.sort(np.sqrt([x for x in roots if x > 0]))


def compute_transition_temperature_uncertainty(
        d_E0,
        d_alpha,
        d_beta,
        d_gamma,
        uncert_Fv,
        c_temperature
):
    """Compute uncertainty in transition temperature given free energy uncertainty.

    The free-energy difference is shifted by -uncert_Fv and +uncert_Fv; in each case the root closest to the critical
    temperature (the same branch) is taken, and the larger of the two deviations is returned. Before it returned the
    smallest deviation over the four roots of the two shifted equations, i.e. the less conservative side only.

    Args:
        d_E0          (float): Difference in ground state energies (meV/atom).
        d_alpha       (float): Difference in alpha coefficients.
        d_beta        (float): Difference in beta coefficients (scaled by 1e-5).
        d_gamma       (float): Difference in gamma coefficients (scaled by 1e-10).
        uncert_Fv     (float): Uncertainty in free energy (meV/atom).
        c_temperature (float): Critical transition temperature (K).

    Returns:
        float: Uncertainty in transition temperature (K), or np.nan if the transition disappears (no real root)
            within the free-energy uncertainty.
    """
    deviations = []
    for shift in (-uncert_Fv, uncert_Fv):
        roots = transition_temperatures(d_E0 + d_alpha + shift, d_beta, d_gamma)
        if not len(roots):
            return np.nan
        deviations.append(np.min(np.abs(roots - c_temperature)))
    return max(deviations)


def Helmholtz_free_energy_function(
        x,
        alpha,
        beta,
        gamma
):
    """Smoothes the Helmholtz free energy with a physically informed function: alpha + beta T^2 + gamma T^4.
    The parameters beta and gamma are re-scaled for improved fitting.
    They are no longer forced to be negative: that was fine for F_v alone, but the electronic free energy of a metal
    has a positive T^4 term (Sommerfeld expansion), so the constraint would give a wrong fit of F_v + F_el.

    Args:
        x     (ndarray): The input array containing temperature data.
        alpha (float):   A parameter defining the baseline energy level.
        beta  (float):   A parameter controlling the quadratic term (scaled by 1e-5).
        gamma (float):   A parameter controlling the quartic term (scaled by 1e-10).

    Returns:
        smoothed_energy (ndarray): An array of smoothed Helmholtz free energy values.
    """
    # Re-scale variables
    beta  = beta  * 1e-5
    gamma = gamma * 1e-10
    return alpha + beta * x ** 2 + gamma * x ** 4


def compute_coefficients(
        temperatures,
        predictions
):
    """Smooth the predictions and obtain parameters from Helmholtz_free_energy_function.
    
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

        except (RuntimeError, ValueError):  # Convergence not achieved, or NaN/inf in the data (e.g. a failed F_el)
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
    model,
    neighbors=None
):
    """
    Fit RBFInterpolator to reference uncertainties.

    Args:
        r_uncertainty_data (dict):            Uncertainty data for the reference dataset.
        r_dataset          (list):            Reference dataset, as a list of graphs in PyTorch Geometric's Data format.
        model              (torch.nn.Module): The trained model.
        neighbors          (int):             If given, each query uses only its nearest `neighbors` reference points
                                              (the default, all points, needs O(N^2) memory).

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
    interpolator = RBFInterpolator(r_embeddings, r_uncertainties, smoothing=0, neighbors=neighbors)
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


def load_model(
        n_node_features,
        pdropout=0,
        device='cpu',
        model_name=None,
        mode='eval'
):
    # Load Graph Neural Network model
    model = GCNN(features_channels=n_node_features, pdropout=pdropout)

    if model_name is not None and os.path.exists(model_name):
        # Load Graph Neural Network model
        model.load_state_dict(torch.load(model_name, map_location="cpu", weights_only=True))  # weights only: no code execution
    
    # Moving model to device
    model = model.to(device)

    if mode == 'eval':
        model.eval()
    elif mode == 'train':
        model.train()

    # NOTE: nn.DataParallel was removed. It cannot split a PyG batch (it hands the whole batch to every GPU and
    # concatenates the outputs, so out and data.y have different sizes on multi-GPU machines), and it prefixed
    # the saved state-dict keys with 'module.'.
    return model


def train(
        model,
        criterion,
        train_loader,
        target_factor,
        target_mean,
        optimizer
):
    """Train the model using the provided optimizer and criterion on the training dataset.

    Args:
        model        (torch.nn.Module):             The model to train.
        optimizer    (torch.optim.Optimizer):       The optimizer to use for updating model parameters.
        criterion    (torch.nn.Module):             The loss function to use.
        train_loader (torch.utils.data.DataLoader): The training dataset loader.

    Returns:
        float: The average training loss.
    """
    model.train()
    train_loss = 0
    predictions   = []
    ground_truths = []
    for data in train_loader:  # Iterate in batches over the training dataset
        # Moving data to device
        data = data.to(device)
        
        # Perform a single forward pass
        out = model(data).flatten()
        
        # Compute the loss
        loss = criterion(out, data.y)
        
        # Accumulate the training loss
        train_loss += loss.item()

        # Append predictions and ground truths to lists
        predictions.append(out.detach().cpu().numpy())
        ground_truths.append(data.y.detach().cpu().numpy())
        
        # Derive gradients
        loss.backward()
        
        # Update parameters based on gradients
        optimizer.step()
        
        # Clear gradients
        optimizer.zero_grad()
    
    # Compute the average training loss
    avg_train_loss = train_loss / len(train_loader)
    
    # Concatenate predictions and ground truths into single arrays
    predictions   = np.concatenate(predictions)   * target_factor + target_mean
    ground_truths = np.concatenate(ground_truths) * target_factor + target_mean
    return avg_train_loss, predictions, ground_truths


def test(
        model,
        criterion,
        test_loader,
        target_factor,
        target_mean
):
    """Evaluate the performance of a given model on a test dataset.

    Args:
        model       (torch.nn.Module):             The model to evaluate.
        criterion   (torch.nn.Module):             The loss function to use.
        test_loader (torch.utils.data.DataLoader): The test dataset loader.

    Returns:
        float: The average loss on the test dataset.
    """
    model.eval()
    test_loss = 0
    predictions   = []
    ground_truths = []
    with torch.no_grad():
        for data in test_loader:  # Iterate in batches over the train/test dataset
            # Moving data to device
            data = data.to(device)
            
            # Perform a single forward pass
            out = model(data).flatten()
            
            # Compute the loss
            loss = criterion(out, data.y)
            
            # Accumulate the training loss
            test_loss += loss.item()

            # Append predictions and ground truths to lists
            predictions.append(out.detach().cpu().numpy())
            ground_truths.append(data.y.detach().cpu().numpy())
    
    # Compute the average test loss
    avg_test_loss = test_loss / len(test_loader)
    
    # Concatenate predictions and ground truths into single arrays
    predictions   = np.concatenate(predictions)   * target_factor + target_mean
    ground_truths = np.concatenate(ground_truths) * target_factor + target_mean
    return avg_test_loss, predictions, ground_truths


class EarlyStopping():
    def __init__(
            self,
            patience=5,
            delta=0,
            wandb_run=None,
            model_name='model.pt'
    ):
        """Initializes the EarlyStopping object. Saves a model if accuracy is improved.
        Declares early_stop = True if training does not improve in patience steps within a delta threshold.

        Args:
            patience   (int):          Number of steps with no improvement.
            delta      (float):        Threshold for a score to be considered an improvement.
            wandb_run  (wandb object): Optional WandB run object for logging.
            model_name (str):          Name of the saved model checkpoint file.
        """
        self.patience = patience  # Number of steps with no improvement
        self.delta = delta  # Threshold for a score to be an improvement
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.wandb_run = wandb_run
        self.model_name = model_name

    def __call__(
            self,
            val_loss,
            model
    ):
        """Call method to check and update early stopping.

        Args:
            val_loss (float):           Current validation loss.
            model    (torch.nn.Module): The PyTorch model being trained.
        """
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.wandb_run:
                self.wandb_run.log({'Early Stopping Counter': self.counter})

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(
            self,
            val_loss,
            model
    ):
        """Save the model checkpoint if the validation loss has decreased.
        It uses model.module, allowing models loaded to nn.DataParallel.

        Args:
            val_loss (float):           Current validation loss.
            model    (torch.nn.Module): The PyTorch model being trained.
        """
        if val_loss < self.val_loss_min:
            torch.save(getattr(model, 'module', model).state_dict(), self.model_name)  # also works for a DataParallel-wrapped model
            self.val_loss_min = val_loss
