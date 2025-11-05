"""Scientific analysis functions for np-dist2."""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from np_dist2.io import read_timestep
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