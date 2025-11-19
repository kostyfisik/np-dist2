"""Scientific analysis functions for np-dist2."""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from np_dist2.io import read_timestep
from np_dist2.grid_generator import generate_sphere_points
from pathlib import Path


# Functions from lat-plot.py
def get_lat(ref, atoms_pos):
    """Calculate lattice parameter for a reference atom."""
    ref_arr = np.ones(atoms_pos.shape) * ref
    lat_arr = np.sort(np.linalg.norm(np.abs(atoms_pos - ref_arr), axis=1))
    lat_arr = lat_arr[:20]
    return np.min(lat_arr)


def get_lat_vs_r(atoms_pos):
    """Calculate lattice parameters vs radial distance."""
    lats = []
    rs = []
    for i in range(atoms_pos.shape[0]):
        ref = atoms_pos[i]
        atoms_pos_cp = np.copy(atoms_pos)
        lat = get_lat(ref, np.delete(atoms_pos_cp, i, axis=0))
        lats.append(lat)
        rs.append(np.linalg.norm(ref))
    return lats, rs


# Functions from ion-plot.py
def vol(r, h2):
    """Calculate volume of a spherical shell with thickness 2*h2."""
    r1 = r - h2
    r2 = r + h2
    return 4.0 / 3.0 * np.pi * (r2**3 - r1**3)


def calculate_ion_density(atoms_r, r, h2):
    """Calculate average ion density inside a spherical shell of thickness 2*h2."""
    # Handle edge case where h2 is zero
    if h2 <= 0:
        return 0.0

    r1 = r - h2
    r2 = r + h2
    in_span = atoms_r[(atoms_r >= r1) & (atoms_r <= r2)]
    volume = vol(r, h2)

    # Handle edge case where volume is zero
    if volume <= 0:
        return 0.0

    return len(in_span) / volume


# Functions from dist-plot.py
def get_lat_extended(ref, atoms_pos):
    """Calculate extended lattice parameters for a reference atom."""
    ref_arr = np.ones(atoms_pos.shape) * ref
    lat_arr = np.linalg.norm(np.abs(atoms_pos - ref_arr), axis=1)
    lat_arr = np.sort(np.linalg.norm(np.abs(atoms_pos - ref_arr), axis=1))
    lat_arr = lat_arr[:20]
    lat = np.min(lat_arr)
    lat2 = lat_arr[lat_arr < lat * 1.2]

    # Handle edge case where lat2 is empty
    if len(lat2) == 0:
        lat2_mean = 0.0
    else:
        lat2_mean = np.mean(lat2)

    r = np.linalg.norm(ref)
    return lat, r, len(lat2), lat2_mean


def get_lat_vs_r_extended(atoms_pos):
    """Calculate extended lattice parameters vs radial distance."""
    lats = []
    rs = []
    ns = []
    lat2s = []
    for i in range(atoms_pos.shape[0]):
        ref = atoms_pos[i]
        atoms_pos_cp = np.copy(atoms_pos)
        lat, r, n, lat2 = get_lat_extended(ref, np.delete(atoms_pos_cp, i, axis=0))
        lats.append(lat)
        rs.append(r)
        ns.append(n)
        lat2s.append(lat2)
    return lats, rs, ns, lat2s


def average_timesteps(directory: str, num_steps: int) -> np.ndarray:
    """Reads and averages atomic positions from a series of LAMMPS dump files."""

    # Get a sorted list of dump files
    dump_files = sorted(Path(directory).glob("dump_superficie.*"))

    if not dump_files:
        raise FileNotFoundError(f"No dump files found in directory: {directory}")

    # Initialize based on the first timestep
    initial_pos, atoms_num = read_timestep(str(dump_files[0]))
    sum_pos = initial_pos

    # Loop through the remaining files up to num_steps
    count = 1
    for i in range(1, min(num_steps, len(dump_files))):
        pos, _ = read_timestep(str(dump_files[i]))
        sum_pos += pos
        count += 1

    return sum_pos / count


# Functions from rdf-plot.py
def radial_distribution(atoms_pos, r_min, r_max, cutoff, resolution):
    """Calculate radial distribution function."""
    # Handle edge case of empty array
    if atoms_pos.size == 0:
        x = np.linspace(r_min, r_max, resolution)
        g = np.zeros(resolution - 1)  # Return resolution-1 values to match histogram behavior
        # Return bin centers instead of bin edges
        x_centers = (x[:-1] + x[1:]) / 2
        return x_centers, g

    # Calculate pairwise distances
    distances = cdist(atoms_pos, atoms_pos)
    # Remove self-distances (diagonal elements)
    distances = distances[distances > 0]

    # Create histogram
    x = np.linspace(r_min, r_max, resolution)
    hist, bin_edges = np.histogram(distances, bins=x)

    # Normalize by volume of shells
    g = np.zeros(resolution - 1)
    for i in range(resolution - 1):
        r1, r2 = bin_edges[i], bin_edges[i + 1]
        volume = 4.0 / 3.0 * np.pi * (r2**3 - r1**3)
        if volume > 0:
            # Normalize by number of atoms and volume
            g[i] = hist[i] / volume / len(atoms_pos)

    # Normalize by average density
    rho = len(atoms_pos) / ((4.0 / 3.0) * np.pi * (np.max(atoms_pos) ** 3))
    if rho > 0:
        g = g / rho

    # Return bin centers instead of bin edges
    x_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return x_centers, g


# Functions for RDF plotting
def plot_radial_distribution(x, g, output_file):
    """Plot radial distribution function."""
    plt.figure()
    plt.bar(x, g, width=0.9*(x[1] - x[0]))
    plt.xlabel("Distance")
    plt.ylabel("Radial Distribution")
    plt.savefig(output_file, dpi=600)
    plt.close()


# New functions for particle analysis
def generate_random_directions(num_directions: int) -> np.ndarray:
    """
    Generate uniformly distributed random unit vectors on a sphere.

    Args:
        num_directions: Number of random directions to generate

    Returns:
        Array of shape (num_directions, 3) containing unit vectors
    """
    # Leverage existing generate_sphere_points function
    # It returns shape (3, N), so we transpose to get (N, 3)
    directions = generate_sphere_points(num_directions).T
    return directions


def find_atoms_in_cylinder(atoms: np.ndarray, axis: np.ndarray, radius: float) -> np.ndarray:
    """
    Find all atoms within a cylinder of specified radius along a given axis.

    The cylinder is oriented along the axis vector passing through the origin.

    Args:
        atoms: Array of shape (n_atoms, 3) containing atomic positions
        axis: Unit vector defining cylinder axis direction, shape (3,)
        radius: Cylinder radius

    Returns:
        Array of atom positions that lie within the cylinder
    """
    if atoms.size == 0:
        return np.array([]).reshape(0, 3)

    # Normalize the axis vector
    axis_normalized = axis / np.linalg.norm(axis)

    # Calculate perpendicular distance from each atom to the axis
    # Distance = |atom - (atom · axis) * axis|
    projection_lengths = np.dot(atoms, axis_normalized)
    projections = np.outer(projection_lengths, axis_normalized)
    perpendicular_vectors = atoms - projections
    perpendicular_distances = np.linalg.norm(perpendicular_vectors, axis=1)

    # Select atoms within the cylinder radius
    mask = perpendicular_distances <= radius
    return atoms[mask]


def calculate_effective_radius(
    atoms: np.ndarray, num_directions: int, cylinder_radius: float
) -> tuple[float, np.ndarray]:
    """
    Calculate the effective radius of a nanoparticle by sampling random directions.

    For each random direction, finds atoms within a cylinder along that direction
    and determines the maximum radial distance. The effective radius is the average
    of these maximum radii across all directions.

    Args:
        atoms: Array of shape (n_atoms, 3) containing atomic positions
        num_directions: Number of random directions to sample
        cylinder_radius: Radius of the cylinder used for sampling

    Returns:
        Tuple of (effective_radius, all_radii) where:
        - effective_radius: Average of maximum radii across all directions
        - all_radii: Array of maximum radii for each direction
    """
    if atoms.size == 0:
        return 0.0, np.array([])

    # Generate random directions
    directions = generate_random_directions(num_directions)

    # Calculate maximum radius for each direction
    all_radii = np.zeros(num_directions)

    for i, direction in enumerate(directions):
        # Find atoms in cylinder along this direction
        cylinder_atoms = find_atoms_in_cylinder(atoms, direction, cylinder_radius)

        if cylinder_atoms.size > 0:
            # Calculate radial distances from origin
            radial_distances = np.linalg.norm(cylinder_atoms, axis=1)
            all_radii[i] = np.max(radial_distances)
        else:
            all_radii[i] = 0.0

    # Calculate effective radius as the average
    effective_radius = np.mean(all_radii)

    return effective_radius, all_radii


def find_near_neighbors(
    atom_index: int, all_atoms: np.ndarray, dist_factor: float = 1.2
) -> tuple[np.ndarray, float]:
    """
    Find near neighbors for a specific atom.

    Identifies the nearest neighbor and then finds all atoms whose distance is
    no more than dist_factor times the nearest neighbor distance.

    Args:
        atom_index: Index of the target atom
        all_atoms: Array of shape (n_atoms, 3) containing all atomic positions
        dist_factor: Multiplier for nearest neighbor distance to define "near" (default 1.2)

    Returns:
        Tuple of (neighbor_indices, nearest_distance) where:
        - neighbor_indices: Indices of near neighbor atoms
        - nearest_distance: Distance to the nearest neighbor
    """
    if atom_index < 0 or atom_index >= len(all_atoms):
        raise ValueError(f"atom_index {atom_index} out of range for atoms array of length {len(all_atoms)}")

    target_atom = all_atoms[atom_index]

    # Calculate distances to all other atoms
    distances = np.linalg.norm(all_atoms - target_atom, axis=1)

    # Find nearest neighbor (minimum non-zero distance)
    non_zero_distances = distances[distances > 0]

    if len(non_zero_distances) == 0:
        # No other atoms
        return np.array([], dtype=int), 0.0

    d_min = np.min(non_zero_distances)

    # Find all atoms within dist_factor * d_min (excluding self)
    threshold = d_min * dist_factor
    mask = (distances > 0) & (distances <= threshold)
    neighbor_indices = np.where(mask)[0]

    return neighbor_indices, d_min


def calculate_lattice_parameter_distribution(
    atoms: np.ndarray, num_directions: int, cylinder_radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate local lattice parameters for atoms in cylindrical sections.

    Samples multiple random directions and for each direction, identifies atoms
    within a cylinder. For each such atom, calculates the local lattice parameter
    as the average distance to its near neighbors. Radial distances are normalized
    so that the outermost atom in each direction is at the effective radius.

    The particle is not perfectly spherical relative to its center of mass, so
    this function:
    1. Calculates the effective radius using all directions
    2. For each direction, normalizes radial distances so the outermost atom
       is at the effective radius
    3. Removes atoms with negative normalized radii
    4. Combines all directions into a single dataset

    Args:
        atoms: Array of shape (n_atoms, 3) containing atomic positions
        num_directions: Number of random directions to sample
        cylinder_radius: Radius of the cylinder used for sampling

    Returns:
        Tuple of (radial_distances, lattice_parameters) where both are 1D arrays
        containing the normalized radial position and local lattice parameter for each sampled atom
    """
    if atoms.size == 0:
        return np.array([]), np.array([])

    # First, calculate the effective radius of the particle
    effective_radius, _ = calculate_effective_radius(atoms, num_directions, cylinder_radius)

    if effective_radius == 0.0:
        return np.array([]), np.array([])

    # Generate random directions
    directions = generate_random_directions(num_directions)

    # Initialize lists to collect results
    all_r_normalized = []
    all_a_local = []

    for direction in directions:
        # Find atoms in cylinder along this direction
        cylinder_atoms = find_atoms_in_cylinder(atoms, direction, cylinder_radius)

        if cylinder_atoms.size == 0:
            continue

        # Calculate radial distances for atoms in this direction
        radial_distances_raw = np.linalg.norm(cylinder_atoms, axis=1)

        if len(radial_distances_raw) == 0:
            continue

        # Find maximum radius in this direction
        r_max_direction = np.max(radial_distances_raw)

        if r_max_direction == 0.0:
            continue

        # Normalize radii: shift and scale so the outermost atom is at effective_radius
        # Formula: r_normalized = (r_raw / r_max_direction) * effective_radius
        # This ensures the outermost atom in each direction is at the effective radius

        # Store temporary results for this direction
        direction_r_normalized = []
        direction_a_local = []

        # For each atom in the cylinder
        for idx, cylinder_atom in enumerate(cylinder_atoms):
            # Find the atom's index in the original array
            atom_index = np.where(np.all(atoms == cylinder_atom, axis=1))[0][0]

            # Find near neighbors
            neighbor_indices, d_min = find_near_neighbors(atom_index, atoms, dist_factor=1.2)

            if len(neighbor_indices) == 0:
                continue

            # Calculate average distance to near neighbors (local lattice parameter)
            neighbor_positions = atoms[neighbor_indices]
            distances = np.linalg.norm(neighbor_positions - cylinder_atom, axis=1)
            a_local = np.mean(distances)

            # Get raw radial distance
            r_raw = radial_distances_raw[idx]

            # Normalize radius relative to effective radius
            r_normalized = (r_raw / r_max_direction) * effective_radius

            # Store results for this atom
            direction_r_normalized.append(r_normalized)
            direction_a_local.append(a_local)

        # Add all atoms from this direction to the combined dataset
        # (No negative radii with this normalization approach since we're scaling, not shifting)
        all_r_normalized.extend(direction_r_normalized)
        all_a_local.extend(direction_a_local)

    return np.array(all_r_normalized), np.array(all_a_local)


def convert_lattice_to_density(
    lattice_params: np.ndarray, structure: str = 'fcc'
) -> np.ndarray:
    """
    Convert local lattice parameters to local ionic densities.

    For FCC structure, the lattice parameter given is d = a*sqrt(2)/2 where a is
    the conventional lattice constant. The density is calculated as n = 4/a^3.

    Args:
        lattice_params: Array of local lattice parameters (nearest neighbor distances)
        structure: Crystal structure type, currently only 'fcc' is supported

    Returns:
        Array of local ionic densities
    """
    if structure != 'fcc':
        raise ValueError(f"Unsupported structure type: {structure}. Only 'fcc' is supported.")

    # For FCC: d = a * sqrt(2) / 2, so a = d * sqrt(2)
    # Volume per atom = a^3 / 4
    # Density n = 4 / a^3 = 4 / (d * sqrt(2))^3

    a = lattice_params * np.sqrt(2)
    density = 4.0 / (a ** 3)

    return density
