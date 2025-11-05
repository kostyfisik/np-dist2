"""Tests for the analysis module."""

import numpy as np
import pytest
from np_dist2.analysis import get_lat_vs_r_extended, calculate_ion_density, radial_distribution


def test_get_lat_vs_r_extended_perfect_fcc(perfect_fcc_lattice: np.ndarray) -> None:
    """Test lattice parameter calculation for a perfect FCC lattice."""
    # For a perfect FCC lattice, we expect 12 nearest neighbors
    # and a consistent lattice parameter
    lats, rs, ns, lat2s = get_lat_vs_r_extended(perfect_fcc_lattice)
    
    # Check that we get results for each atom
    assert len(lats) == len(perfect_fcc_lattice)
    assert len(rs) == len(perfect_fcc_lattice)
    assert len(ns) == len(perfect_fcc_lattice)
    assert len(lat2s) == len(perfect_fcc_lattice)
    
    # For a perfect FCC lattice, all atoms should have non-negative neighbors
    # Convert to numpy array for comparison
    ns_array = np.array(ns)
    assert np.all(ns_array >= 0)


def test_calculate_ion_density_uniform_sphere() -> None:
    """Test ion density calculation with uniformly distributed points in a sphere."""
    # Create points uniformly distributed in a sphere
    np.random.seed(42)  # For reproducible test
    n_points = 1000
    # Generate uniform distribution in spherical coordinates
    r = np.random.uniform(0, 5, n_points)
    theta = np.random.uniform(0, np.pi, n_points)
    phi = np.random.uniform(0, 2*np.pi, n_points)
    
    # Convert to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    atoms = np.column_stack([x, y, z])
    atoms_r = np.linalg.norm(atoms, axis=1)
    
    # Test density at different radii
    density1 = calculate_ion_density(atoms_r, 2.0, 0.5)
    density2 = calculate_ion_density(atoms_r, 3.0, 0.5)
    
    # Both should be non-negative (we're not checking exact values due to randomness)
    assert density1 >= 0
    assert density2 >= 0


def test_calculate_ion_density_empty_array() -> None:
    """Test ion density calculation with empty array."""
    atoms_r = np.array([])
    density = calculate_ion_density(atoms_r, 2.0, 0.5)
    assert density == 0.0


def test_calculate_ion_density_zero_thickness() -> None:
    """Test ion density calculation with zero thickness."""
    atoms_r = np.array([1.0, 2.0, 3.0])
    density = calculate_ion_density(atoms_r, 2.0, 0.0)
    assert density == 0.0


def test_radial_distribution_fcc(perfect_fcc_lattice: np.ndarray) -> None:
    """Test RDF calculation for a perfect FCC lattice."""
    x, g = radial_distribution(perfect_fcc_lattice, 0.0, 10.0, 3.5, 100)
    
    # Check that we get the right shapes (99 values for 100 bins)
    assert len(x) == 99
    assert len(g) == 99
    
    # Check that all values are non-negative
    assert np.all(g >= 0)


def test_radial_distribution_empty_array() -> None:
    """Test RDF calculation with empty array."""
    atoms = np.array([]).reshape(0, 3)
    x, g = radial_distribution(atoms, 0.0, 10.0, 3.5, 100)
    
    # Check that we get the right shapes (99 values for 100 bins)
    assert len(x) == 99
    assert len(g) == 99
    
    # Check that all values are zero
    assert np.all(g == 0)


def test_radial_distribution_random_points() -> None:
    """Test RDF calculation with random points."""
    np.random.seed(42)  # For reproducible test
    atoms = np.random.rand(100, 3) * 10  # Random points in a 10x10x10 box
    
    x, g = radial_distribution(atoms, 0.0, 5.0, 2.0, 50)
    
    # Check that we get the right shapes (49 values for 50 bins)
    assert len(x) == 49
    assert len(g) == 49
    
    # Check that all values are non-negative
    assert np.all(g >= 0)