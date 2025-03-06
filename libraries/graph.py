import numpy as np
import torch as torch
import sys

from torch_geometric.data    import Data
from pymatgen.core.structure import Structure


def generate_graph(
        current_path,
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
                    edge_attr=attributes
                   )
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

    # Extract direct positions, composition and concentration as lists
    positions     = np.array([site.frac_coords for site in structure.sites])
    composition   = [element.symbol for element in structure.composition.elements]
    concentration = np.array([sum(site.species_string == element for site in structure.sites) for element in composition])

    # Counting number of particles
    total_particles = np.sum(concentration)

    # Generating graph structure, getting particle types
    particle_types = []
    for i in range(len(composition)):
        particle_types += [i] * concentration[i]

    # Adding nodes and edges.
    nodes = []
    edges = []
    attributes = []
    for index_0 in range(total_particles):
        # Get particle type (index of type wrt composition in POSCAR)
        particle_type = particle_types[index_0]

        # Name of the current species
        species_name = composition[particle_type]

        # Adding the nodes (mass, charge, electronegativity and ionization energies)
        nodes.append([atomic_data[species_name]['atomic_mass'],
                      atomic_data[species_name]['charge'],
                      atomic_data[species_name]['electronegativity'],
                      atomic_data[species_name]['ionization_energy']])

        # Get the initial position
        position_0 = positions[index_0]
        position_cartesian_0 = np.dot(position_0, structure.lattice.matrix)

        # Explore images of all particles in the system
        # Starting on index_0, thus exploring possible images with itself (except for i,j,k=0, exact same particle)
        for index_i in np.arange(index_0, total_particles):
            # Get the initial position
            position_i = positions[index_i]

            reference_distance_i = np.nan  # So it outputs False when first compared with another distance
            i = 0
            alpha_i = 1
            while True:
                minimum_distance_i   = np.nan
                reference_distance_j = np.nan
                j = 0
                alpha_j = 1
                while True:
                    minimum_distance_j   = np.nan
                    reference_distance_k = np.nan
                    k = 0
                    alpha_k = 1
                    while True:
                        # Move to the corresponding image and convert to cartesian distances
                        position_cartesian_i = np.dot(position_i + [i, j, k], structure.lattice.matrix)

                        # New distance as Euclidean distance between both reference and new image particle
                        new_distance = np.linalg.norm([position_cartesian_0 - position_cartesian_i])

                        # Condition that remove exact same particle
                        same_index_condition     = (index_0 == index_i)
                        all_index_null_condition = np.all([i, j, k] == [0]*3)
                        same_particle_condition  = (same_index_condition and all_index_null_condition)

                        # Applying threshold to images
                        if (new_distance <= distance_threshold) and not same_particle_condition:
                            # Append this point as a edge connection to particle 0
                            edges.append([index_0, index_i])
                            attributes.append([new_distance])

                        # Change direction or update i,j if the box is far
                        elif new_distance > reference_distance_k:
                            # Explore other direction or cancel
                            if alpha_k == 1:
                                k = 0
                                alpha_k = -1
                            else:
                                break

                        reference_distance_k = new_distance
                        k += alpha_k

                        if not minimum_distance_j <= reference_distance_k:
                            minimum_distance_j = reference_distance_k

                    # If k worked fine, j is fine as well thus continue; else, explore other direction or cancel
                    if minimum_distance_j > reference_distance_j:
                        if alpha_j == 1:
                            j = 0
                            alpha_j = -1
                        else:
                            break

                    # Update j
                    j += alpha_j
                    reference_distance_j = minimum_distance_j

                    if not minimum_distance_i <= reference_distance_j:
                        minimum_distance_i = reference_distance_j

                # If j did not work fine, explore other direction or cancel
                if minimum_distance_i > reference_distance_i:
                    if alpha_i == 1:
                        i = 0
                        alpha_i = -1
                    else:
                        break

                # Update i
                i += alpha_i
                reference_distance_i = minimum_distance_i
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