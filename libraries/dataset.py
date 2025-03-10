import torch as torch
import os

from libraries.graph import generate_graph, add_features_to_graph


def create_predictions_dataset(
        path_to_folder,
        path_to_material=False,
        path_to_polymorph=False
):
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
                        data = generate_graph(path_to_polymorf, label=f'{element} {polymorf}')
                    except ValueError:
                        print(f'Error: some element is not available for {polymorf}')
                        continue

                    if data is None:
                        continue

                    # Append graph and label
                    dataset.append(data)
    return dataset


def standarize_dataset(
        dataset,
        standardized_parameters,
        transformation='inverse-quadratic'
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
