import numpy as np
import torch as torch
import sys
import os

import libraries.dataset as cld

from torch_geometric.data    import Data
from pymatgen.core.structure import Structure


def generate_graph(
        current_path,
        label='',
        distance_thd=6
):
    """Generate a graph from a given POSCAR file.
    
    Args:
        current_path (str):             Path to the POSCAR file.
        temperature  (float, optional): Temperature value. Defaults to None.
        distance_thd (float, optional): Distance threshold for encoding. Defaults to 6.
    
    Returns:
        Data: A graph in PyTorch Geometric's Data format.
    """

    # Read POSCAR information
    structure = Structure.from_file(f'{current_path}/POSCAR')

    try:
        nodes, edges, attributes = graph_POSCAR_encoding(structure,
                                                         distance_threshold=distance_thd
                                                         )
    
        # Generate the graph
        temp = Data(x=nodes,
                    edge_index=edges.t().contiguous(),
                    edge_attr=attributes,
                    label=label
                   )
        
        # Load additional properties if requested:
        # 1) Ground state energy (EPA) - from eV/atom to meV/atom
        # 2) Lattice thermal conductivity (LTC) - in W/mK
        # 3) Centrosymmetry (1 is centrosymmetric, 0 is not)
        # 4) Space group

        # Define file paths
        epa_file    = f'{current_path}/EPA'
        ltc_file    = f'{current_path}/LTC'
        centro_file = f'{current_path}/is_centrosymmetric'
        sg_file     = f'{current_path}/space_group'

        # Load properties
        temp.gs_energy = float(np.loadtxt(epa_file)) * 1e3 if os.path.isfile(epa_file) else None
        temp.conductivity = float(np.loadtxt(ltc_file)) if os.path.isfile(ltc_file) else np.nan
        temp.centrosymmetry = int(np.loadtxt(centro_file, dtype=int)) if os.path.isfile(centro_file) else None
        temp.space_group = str(np.loadtxt(sg_file, dtype=str)) if os.path.isfile(sg_file) else None
        
        # Calculate and store total mass from POSCAR
        poscar_file = f'{current_path}/POSCAR'
        try:
            with open(poscar_file, 'r') as pf:
                poscar_lines = pf.readlines()
                
                composition = poscar_lines[5].split()
                concentration = np.array(poscar_lines[6].split(), dtype=float)
                
                # Load atomic masses for calculation
                atomic_masses = cld.load_atomic_masses('../input/atomic_masses.dat')

                # Calculate total mass per atom
                temp.mass_per_atom_m = sum(
                    float(atomic_masses[el]) * conc for el, conc in zip(composition, concentration)
                    ) / sum(concentration)
        except (ValueError, IndexError, KeyError):
            temp.mass_per_atom_m = None
        
        return temp
    except TypeError:
        return None


def get_sphere_images_tessellation(
        atomic_data,
        structure,
        distance_threshold=6
):
    """Gets the distances by pairs of particles, considering images with periodic boundary conditions (PBC).

    Args:
        atomic_data        (dict):                      A dictionary with all node features.
        structure          (pymatgen Structure object): Structure from which the graph is to be generated
        distance_threshold (float, optional):           The distance threshold for edge creation (default is 6).

    Returns:
        nodes      (list): A tensor containing node attributes.
        edges      (list): A tensor containing edge indices.
        attributes (list): A tensor containing edge attributes (distances).
    """

    # structure.get_all_neighbors returns a list of neighbor lists per site
    neighbors = structure.get_all_neighbors(distance_threshold)

    # Adding nodes and edges.
    nodes = []
    edges = []
    attributes = []
    for i, site in enumerate(structure.sites):
        # Adding the nodes (mass, charge, electronegativity and ionization energies)
        nodes.append([atomic_data[site.species_string]['atomic_mass'],
                      atomic_data[site.species_string]['charge'],
                      atomic_data[site.species_string]['electronegativity'],
                      atomic_data[site.species_string]['ionization_energy']])
        
        for neighbor in neighbors[i]:
            j = neighbor.index
            distance = neighbor.nn_distance
    
            if neighbor.nn_distance > 0:
                # Append edge i->j and j->i to make it undirected
                edges.append([i, j])
                attributes.append([distance])
                if i != j:
                    edges.append([j, i])
                    attributes.append([distance])
    return nodes, edges, attributes


def graph_POSCAR_encoding(
        structure,
        distance_threshold=6
):
    """Generates a graph parameters from a POSCAR.
    There are the following implementations:
        1. Voronoi tessellation.
        2. All particles inside a sphere of radius distance_threshold.
        3. Filled space given a cubic box of dimension [0-Lx, 0-Ly, 0-Lz] considering all necessary images.
           It links every particle with the rest for the given set of nodes and edges.

    Args:
        structure          (pymatgen Structure object): Structure from which the graph is to be generated.
        distance_threshold (float):  Distance threshold for sphere-images tessellation.
    Returns:
        nodes      (torch tensor): Generated nodes with corresponding features.
        edges      (torch tensor): Generated connections between nodes.
        attributes (torch tensor): Corresponding weights of the generated connections.
    """

    # Loading dictionary of atomic masses
    atomic_data = {}
    with open('input/atomic_masses.dat', 'r') as atomic_data_file:
        for line in atomic_data_file:
            key, atomic_mass, charge, electronegativity, ionization_energy = line.split()
            atomic_data[key] = {
                'atomic_mass':       float(atomic_mass) if atomic_mass != 'None' else None,
                'charge':            int(charge) if charge != 'None' else None,
                'electronegativity': float(electronegativity) if electronegativity != 'None' else None,
                'ionization_energy': float(ionization_energy) if ionization_energy != 'None' else None
            }

    # Get edges and attributes for the corresponding tessellation
    nodes, edges, attributes = get_sphere_images_tessellation(atomic_data,
                                                              structure,
                                                              distance_threshold=distance_threshold)

    # Convert to torch tensors and return
    nodes      = torch.tensor(nodes,      dtype=torch.float)
    edges      = torch.tensor(edges,      dtype=torch.long)
    attributes = torch.tensor(attributes, dtype=torch.float)
    return nodes, edges, attributes


def add_features_to_graph(
        graph_0,
        node_features
):
    """Include some more information to the node features. The generated graph does not modify the input graph.

    Args:
        graph_0       (torch_geometric.data.Data): The input graph containing edge indexes and attributes.
        node_features (torch.array of size 1):     Information to be added to the graph (target,
                                                   step of the diffusing/denoising process, etc.).

    Returns:
        graph (torch_geometric.data.Data): Updated graph, with node_features as a new node feature for every atom.
    """

    graph = graph_0.clone()

    # Check that the size of node_features is the expected by the function
    if len(torch.Tensor.size(node_features)) != 1:
        sys.exit('Error: node_features does not have the expected size')

    # Concatenate tensors along the second dimension (dim=1) and update the graph with the new node features
    graph.x = torch.cat((graph.x, node_features.unsqueeze(0).repeat(graph.x.size(0), 1)), dim=1)
    return graph
