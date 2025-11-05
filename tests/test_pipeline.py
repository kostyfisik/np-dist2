"""Integration tests for the np-dist2 pipeline."""

import numpy as np
import pytest
from pathlib import Path
from np_dist2.io import read_timestep
from np_dist2.analysis import get_lat_vs_r_extended


def test_read_and_analyze_fcc(valid_lammps_dump: Path) -> None:
    """Test the complete pipeline from file reading to analysis."""
    # Read the data
    atoms, atoms_num = read_timestep(str(valid_lammps_dump))
    
    # Analyze the data
    lats, rs, ns, lat2s = get_lat_vs_r_extended(atoms)
    
    # Check that we get results for each atom
    assert len(lats) == len(atoms)
    assert len(rs) == len(atoms)
    assert len(ns) == len(atoms)
    assert len(lat2s) == len(atoms)
    
    # Check that all values are reasonable
    # Convert to numpy arrays for comparison
    rs_array = np.array(rs)
    ns_array = np.array(ns)
    lat2s_array = np.array(lat2s)
    
    assert np.all(rs_array >= 0)
    assert np.all(ns_array >= 0)
    assert np.all(lat2s_array > 0)


def test_read_and_analyze_shuffled(shuffled_lammps_dump: Path) -> None:
    """Test the complete pipeline with shuffled atom IDs."""
    # Read the data (should sort by atom ID)
    atoms, atoms_num = read_timestep(str(shuffled_lammps_dump))
    
    # Analyze the data
    lats, rs, ns, lat2s = get_lat_vs_r_extended(atoms)
    
    # Check that we get results for each atom
    assert len(lats) == len(atoms)
    assert len(rs) == len(atoms)
    assert len(ns) == len(atoms)
    assert len(lat2s) == len(atoms)
    
    # Check that all values are reasonable
    # Convert to numpy arrays for comparison
    rs_array = np.array(rs)
    ns_array = np.array(ns)
    lat2s_array = np.array(lat2s)
    
    assert np.all(rs_array >= 0)
    assert np.all(ns_array >= 0)
    assert np.all(lat2s_array > 0)