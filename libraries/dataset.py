import matplotlib.pyplot as plt
import seaborn           as sns
import numpy as np
import torch as torch
import json
import os

from torch_geometric.data    import Data
from libraries.graph import generate_graph, add_features_to_graph, graph_POSCAR_encoding

def create_predictions_dataset(
        path_to_folder,
        path_to_material=False,
        path_to_polymorph=False
):
    """Create dataset for predictions.
    
    Always loads properties (EPA, LTC, centrosymmetry, space_group, mass_per_atom) into graph objects.
    
    Args:
        path_to_folder    (str):  Path to the folder containing POSCAR files.
        path_to_material  (bool): If True, only process current folder as material.
        path_to_polymorph (bool): If True, only process current folder as polymorph.
    
    Returns:
        list: List of graphs in PyTorch Geometric's Data format with loaded properties.
    """
    elements = os.listdir(path_to_folder)
    if path_to_material:
        elements = ['./']

    dataset = []
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
                        data = generate_graph(path_to_polymorf, label=f'{element} {polymorf}')
                    except ValueError:
                        print(f'Error: some element is not available for {polymorf}')
                        continue

                    if data is None:
                        continue

                    # Append graph and label
                    dataset.append(data)
    return dataset


def check_finite_attributes(data):
    # Check all node attributes
    if not torch.all(torch.isfinite(data.x)):
        return False
    # Check all edge attributes
    if not torch.all(torch.isfinite(data.edge_attr)):
        return False
    # Check all target values
    if not torch.all(torch.isfinite(data.y)):
        return False
    return True


def standardize_dataset(
        dataset,
        transformation=None
):
    """Standardizes a given dataset (both nodes features and edge attributes).
    Typically, a normal distribution is applied, although it be easily modified to apply other distributions.
    Check those graphs with finite attributes and retains labels accordingly.

    Currently: normal distribution.

    Args:
        dataset        (list): List containing graph structures.
        transformation (str):  Type of transformation strategy for edge attributes (None, 'inverse-quadratic').

    Returns:
        Tuple: A tuple containing the normalized dataset and parameters needed to re-scale predicted properties.
            - dataset_std        (list): Normalized dataset.
            - labels_std         (list): Labels from valid graphs.
            - dataset_parameters (dict): Parameters needed to re-scale predicted properties from the dataset.
    """

    # Clone the dataset and labels
    dataset_std = []
    for graph in dataset:
        if check_finite_attributes(graph):
            dataset_std.append(graph.clone())

    # Number of graphs
    n_graphs = len(dataset_std)

    # Number of features per node
    n_features = dataset_std[0].num_node_features
    
    # Number of features per graph
    n_y = dataset_std[0].y.shape[0]

    # Check if non-linear standardization
    if transformation == 'inverse-quadratic':
        for data in dataset_std:
            data.edge_attr = 1 / data.edge_attr.pow(2)

    epsilon = 1e-8

    # Compute means
    target_mean = torch.zeros(n_y)
    for target_index in range(n_y):
        target_mean[target_index] = sum([data.y[target_index] for data in dataset_std]) / n_graphs

    edge_mean = sum([data.edge_attr.mean() for data in dataset_std]) / n_graphs

    # Compute standard deviations
    target_std = torch.zeros(n_y)
    for target_index in range(n_y):
        var = sum([(data.y[target_index] - target_mean[target_index]).pow(2).sum() for data in dataset_std]) / (n_graphs * (n_graphs - 1))
        target_std[target_index] = torch.sqrt(var) if var > epsilon else torch.tensor(epsilon)

    edge_var = sum([(data.edge_attr - edge_mean).pow(2).sum() for data in dataset_std]) / (n_graphs * (n_graphs - 1))
    edge_std = torch.sqrt(edge_var) if edge_var > epsilon else torch.tensor(epsilon)

    # In case we want to increase the values of the normalization
    scale = torch.tensor(1e0)

    target_factor = target_std / scale
    target_factor[target_factor == 0] = epsilon
    edge_factor   = edge_std / scale if edge_std != 0 else epsilon

    # Update normalized values into the database
    for data in dataset_std:
        data.y         = (data.y         - target_mean) / target_factor
        data.edge_attr = (data.edge_attr - edge_mean)   / edge_factor

    # Same for the node features
    feat_mean = torch.zeros(n_features)
    feat_std  = torch.zeros(n_features)
    for feat_index in range(n_features):
        # Compute mean
        temp_feat_mean = sum([data.x[:, feat_index].mean() for data in dataset_std]) / n_graphs
        # Compute standard deviations
        temp_feat_var = sum([(data.x[:, feat_index] - temp_feat_mean).pow(2).sum() for data in dataset_std]) / (n_graphs * (n_graphs - 1))
        temp_feat_std = torch.sqrt(temp_feat_var) if temp_feat_var > epsilon else torch.tensor(epsilon)
        # Update normalized values into the database
        for data in dataset_std:
            data.x[:, feat_index] = (data.x[:, feat_index] - temp_feat_mean) * scale / temp_feat_std
        
        # Append corresponding values for saving
        feat_mean[feat_index] = temp_feat_mean
        feat_std[feat_index]  = temp_feat_std

    # Create and save as a dictionary
    dataset_parameters = {
        'transformation': transformation,
        'target_mean':    np.array(target_mean.cpu().numpy()),
        'feat_mean':      np.array(feat_mean.cpu().numpy()),
        'edge_mean':      edge_mean.cpu().numpy(),
        'target_std':     np.array(target_std.cpu().numpy()),
        'feat_std':       np.array(feat_std.cpu().numpy()),
        'edge_std':       edge_std.cpu().numpy(),
        'scale':          scale.cpu().numpy()
    }
    return dataset_std, dataset_parameters


def standardize_dataset_from_keys(
        dataset,
        standardized_parameters
):
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
    target_mean = standardized_parameters['target_mean']
    target_std  = standardized_parameters['target_std'] 
    
    # Check if non-linear standardization
    if standardized_parameters['transformation'] == 'inverse-quadratic':
        for data in dataset:
            data.edge_attr = 1 / data.edge_attr.pow(2)

    for data in dataset:
        data.edge_attr = (data.edge_attr - edge_mean) * scale / edge_std
        if data.y is not None:
            data.y = (data.y - target_mean) * scale / target_std

    for feat_index in range(dataset[0].num_node_features):
        for data in dataset:
            data.x[:, feat_index] = (data.x[:, feat_index] - feat_mean[feat_index]) * scale / feat_std[feat_index]
    return dataset


def include_temperatures(
        dataset,
        temperatures,
        standardized_parameters
):
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


def load_atomic_masses(
        filename
):
    """Load atomic masses from file.

    Args:
        filename (str): Path to file containing atomic masses.

    Returns:
        atomic_masses (dict): Dictionary containing atomic masses.
    """
    atomic_masses = {}
    with open(filename, 'r') as atomic_masses_file:
        for line in atomic_masses_file:
            (key, mass, _, _, _) = line.split()
            atomic_masses[key] = mass
    return atomic_masses


def load_json(
        file_name
):
    """Loads a JSON file and converts torch tensors to torch tensors.

    Args:
        file_name (str): Path to the JSON file.
        to        (str): Convert torch tensors to torch tensors.

    Returns:
        dict: Dictionary containing the data from the JSON file.
    """
    # Load the data from the JSON file
    with open(file_name, 'r') as json_file:
        file = json.load(json_file)

    for key, value in file.items():
        try:
            file[key] = np.array(value, dtype=np.float32)
        except:
            pass
    return file


def get_min_max(*data):
    """Determine the minimum and maximum values in a stack of data.

    Args:
        data: list of torch.tensor

    Returns:
        _min_: float
        _max_: float
    """
    stack = np.concatenate(data)
    _min_ = np.nanmin(stack)
    _max_ = np.nanmax(stack)
    return _min_, _max_


def load_datasets(
        files_names
):
    """Loads the training, validation, and testing datasets from disk.

    Args:
        files_names (dict): Dictionary containing the file names for the training, validation, and testing datasets.

    Returns:
        Tuple: A tuple containing the training, validation, and testing datasets
    """
    train_dataset = torch.load(files_names['train_dataset_std'], weights_only=False)
    val_dataset   = torch.load(files_names['val_dataset_std'],   weights_only=False)
    test_dataset  = torch.load(files_names['test_dataset_std'],  weights_only=False)

    standardized_parameters = load_json(files_names['std_parameters'])
    return train_dataset, val_dataset, test_dataset, standardized_parameters


def get_datasets(
        subset_labels,
        dataset_labels,
        dataset
):
    """Get datasets filtered, non-ordered by labels.

    Args:
        subset_labels  (list): List of labels to filter by.
        dataset_labels (list): List of material labels.
        dataset        (list): List of data elements.

    Returns:
        list: Filtered dataset containing elements corresponding to the specified labels (not ordered).
    """

    subset_labels  = np.array(subset_labels)
    dataset_labels = np.array(dataset_labels)
    
    dataset_idxs = []
    for dataset_idx, dataset_label in enumerate(dataset_labels):
        for subset_idx, subset_label in enumerate(subset_labels):
            if dataset_label.split()[0] == subset_label:
                dataset_idxs.append(dataset_idx)
        if not len(subset_labels):
            break
    return [dataset[idx] for idx in dataset_idxs]


def split_dataset(
        train_ratio,
        test_ratio,
        dataset,
        target_folder
):
    """Splits the dataset into training, validation, and testing datasets regarding their labels.

    Args:
        train_ratio   (float): Ratio of the dataset to be used for training.
        test_ratio    (float): Ratio of the dataset to be used for testing.
        dataset       (list):  List of graphs in PyTorch Geometric's Data format.
        target_folder (str):   Path to folder to save the splittings.

    Returns:
        Tuple: A tuple containing the training, validation, and testing datasets.
    """
    # Splitting into train-test sets considering that Fvs from the same materials must be in the same dataset
    material_labels = [data.label.split()[0] for data in dataset]
    
    # Define unique labels
    unique_labels = np.unique(material_labels)
    
    # Shuffle the list of unique labels
    np.random.shuffle(unique_labels)

    # Define the sizes of the train and test sets
    # Corresponds to the size wrt the number of unique materials in the dataset
    train_size = int(train_ratio * len(unique_labels))
    test_size  = int(test_ratio  * len(unique_labels))
    
    train_labels = unique_labels[:train_size]
    val_labels   = unique_labels[train_size:-test_size]
    test_labels  = unique_labels[-test_size:]
    
    # Save this splitting for transfer-learning approaches
    np.savetxt(f'{target_folder}/train_labels.txt',      train_labels, fmt='%s')
    np.savetxt(f'{target_folder}/validation_labels.txt', val_labels,   fmt='%s')
    np.savetxt(f'{target_folder}/test_labels.txt',       test_labels,  fmt='%s')

    # Use the computed indexes to generate train and test sets
    # We iteratively check where labels equals a unique train/test labels and append the index to a list
    train_dataset = get_datasets(train_labels, material_labels, dataset)
    val_dataset   = get_datasets(val_labels,   material_labels, dataset)
    test_dataset  = get_datasets(test_labels,  material_labels, dataset)
    return train_dataset, val_dataset, test_dataset


def save_datasets(
        train_dataset,
        val_dataset,
        test_dataset,
        files_names
):
    """Saves the training, validation, and testing datasets to disk.

    Args:
        train_dataset (list): List of graphs in PyTorch Geometric's Data format.

    Returns:
        None
    """
    torch.save(train_dataset, files_names['train_dataset_std'])
    torch.save(val_dataset,   files_names['val_dataset_std'])
    torch.save(test_dataset,  files_names['test_dataset_std'])


def save_json(
        file,
        file_name
):
    """Saves a dictionary to a JSON file.

    Args:
        file      (dict): Dictionary containing the data to be saved.
        file_name (str):  Path to the JSON file.

    Returns:
        None
    """
    # Convert torch tensors to numpy arrays
    for key, value in file.items():
        try:
            file[key] = value.tolist()
        except:
            pass

    # Dump the dictionary with numpy arrays to a JSON file
    with open(file_name, 'w') as json_file:
        json.dump(file, json_file)


def parity_plot(
        train=np.array([np.nan, np.nan]),
        validation=np.array([np.nan, np.nan]),
        test=np.array([np.nan, np.nan]),
        figsize=(3, 3),
        save_to=None
):
    """Plots the computed vs. predicted values for the training, validation, and testing datasets.

    Args:
        train       (list): List containing the computed and predicted values for the training dataset.
        validation  (list): List containing the computed and predicted values for the validation dataset.
        test        (list): List containing the computed and predicted values for the testing dataset.
        figsize    (tuple): Size of the figure.

    Returns:
        None
    """
    x_train, y_train = train
    x_val,   y_val   = validation
    x_test,  y_test  = test

    plt.figure(figsize=figsize)

    if np.any(~np.isnan(train)):
        plt.plot(x_train, y_train, '.', label='Train')
    if np.any(~np.isnan(validation)):
        plt.plot(x_val, y_val, '.', label='Validation')
    if np.any(~np.isnan(test)):
        plt.plot(x_test, y_test, '.', label='Test')

    _min_, _max_ = get_min_max(train.flatten(), validation.flatten(), test.flatten())
    plt.xlabel('Computed')
    plt.ylabel('Predicted ')
    plt.plot([_min_, _max_], [_min_, _max_], '-r')
    plt.legend(loc='best')
    if save_to is not None:
        plt.savefig(save_to, dpi=50, bbox_inches='tight')
    plt.show()


def losses_plot(
        train_losses,
        val_losses,
        to_log=True,
        figsize=(3, 3),
        save_to=None
):
    """Plots the training and validation losses.

    Args:
        train_losses (list): List containing the training losses.
        val_losses   (list): List containing the validation losses.
        to_log       (bool): If True, the losses are plotted in log scale.
        figsize      (tuple): Size of the figure.

    Returns:
        None
    """
    if to_log:
        plt.plot(np.log10(train_losses), label='Train loss')
        plt.plot(np.log10(val_losses) , label='Val  loss')
    else:
        plt.plot(train_losses, label='Train loss')
        plt.plot(val_losses,   label='Val  loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    if save_to is not None:
        plt.savefig(save_to, dpi=50, bbox_inches='tight')
    plt.show()