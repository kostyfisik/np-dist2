"""Tests for the io module."""

import numpy as np
import pytest
from pathlib import Path
from np_dist2.io import read_timestep


def test_read_timestep_success(valid_lammps_dump: Path) -> None:
    """Test reading a valid LAMMPS dump file."""
    atoms, atoms_num = read_timestep(str(valid_lammps_dump))
    
    # Check that we have the right number of atoms
    assert atoms.shape == (4, 3)
    assert atoms_num == 4
    
    # Check that the coordinates are correct
    expected = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0]
    ])
    np.testing.assert_array_equal(atoms, expected)


def test_read_timestep_shuffled_ids(shuffled_lammps_dump: Path) -> None:
    """Test reading a LAMMPS dump file with shuffled atom IDs."""
    atoms, atoms_num = read_timestep(str(shuffled_lammps_dump))
    
    # Check that we have the right number of atoms
    assert atoms.shape == (4, 3)
    assert atoms_num == 4
    
    # Check that the coordinates are sorted by atom ID (should be the same as valid dump)
    expected = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0]
    ])
    np.testing.assert_array_equal(atoms, expected)


def test_read_timestep_file_not_found() -> None:
    """Test that FileNotFoundError is raised for non-existent file."""
    with pytest.raises(FileNotFoundError):
        read_timestep("non_existent_file.dump")


def test_read_timestep_malformed(malformed_lammps_dump: Path) -> None:
    """Test that ValueError is raised for malformed dump file."""
    with pytest.raises(ValueError):
        read_timestep(str(malformed_lammps_dump))


def test_read_timestep_missing_header(missing_header_dump: Path) -> None:
    """Test that an error is raised for a dump file missing the ATOMS header."""
    with pytest.raises(Exception):  # Or a more specific custom exception
        read_timestep(str(missing_header_dump))


def test_read_timestep_wrong_atom_count(wrong_atom_count_dump: Path) -> None:
    """Test that an error is raised for a file with a mismatched atom count."""
    with pytest.raises(Exception):  # Or a more specific custom exception
        read_timestep(str(wrong_atom_count_dump))