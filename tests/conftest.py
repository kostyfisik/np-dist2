"""Test configuration and fixtures for np-dist2."""

import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def valid_lammps_dump(tmp_path: Path) -> Path:
    """Create a valid LAMMPS dump file fixture."""
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 1.0 1.0 1.0
2 1 2.0 2.0 2.0
3 1 3.0 3.0 3.0
4 1 4.0 4.0 4.0
"""
    dump_file = tmp_path / "valid_dump.dump"
    dump_file.write_text(dump_content)
    return dump_file


@pytest.fixture
def shuffled_lammps_dump(tmp_path: Path) -> Path:
    """Create a LAMMPS dump file with shuffled atom IDs."""
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
3 1 3.0 3.0 3.0
1 1 1.0 1.0 1.0
4 1 4.0 4.0 4.0
2 1 2.0 2.0 2.0
"""
    dump_file = tmp_path / "shuffled_dump.dump"
    dump_file.write_text(dump_content)
    return dump_file


@pytest.fixture
def malformed_lammps_dump(tmp_path: Path) -> Path:
    """Create a malformed LAMMPS dump file."""
    dump_content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 1.0 1.0
2 1 2.0 2.0 2.0
3 1 3.0 3.0 3.0
4 1 4.0 4.0 4.0
"""
    dump_file = tmp_path / "malformed_dump.dump"
    dump_file.write_text(dump_content)
    return dump_file


@pytest.fixture
def missing_header_dump(tmp_path: Path) -> Path:
    """Creates a dump file missing the ATOMS header."""
    content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
1 1 1.0 1.0 1.0
2 1 2.0 2.0 2.0
"""
    file = tmp_path / "missing_header.dump"
    file.write_text(content)
    return file


@pytest.fixture
def wrong_atom_count_dump(tmp_path: Path) -> Path:
    """Creates a dump file with a wrong atom count in the header."""
    content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
5 
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 1.0 1.0 1.0
2 1 2.0 2.0 2.0
"""
    file = tmp_path / "wrong_count.dump"
    file.write_text(content)
    return file


@pytest.fixture
def valid_npy_file(tmp_path: Path) -> Path:
    """Create a valid .npy file with atomic coordinates."""
    # Create a simple FCC-like structure
    atoms = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0]
    ])
    npy_file = tmp_path / "valid_atoms.npy"
    np.save(npy_file, atoms)
    return npy_file


@pytest.fixture
def empty_npy_file(tmp_path: Path) -> Path:
    """Create an empty .npy file."""
    atoms = np.array([]).reshape(0, 3)
    npy_file = tmp_path / "empty_atoms.npy"
    np.save(npy_file, atoms)
    return npy_file


@pytest.fixture
def perfect_fcc_lattice() -> np.ndarray:
    """Create a perfect FCC lattice fixture."""
    # Create a 3x3x3 FCC lattice with lattice constant 4.0896
    lattice_constant = 4.0896
    atoms = []
    
    # FCC lattice positions
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corner atoms
                atoms.append([i * lattice_constant, j * lattice_constant, k * lattice_constant])
                # Face-centered atoms
                atoms.append([i * lattice_constant + lattice_constant/2, j * lattice_constant + lattice_constant/2, k * lattice_constant])
                atoms.append([i * lattice_constant + lattice_constant/2, j * lattice_constant, k * lattice_constant + lattice_constant/2])
                atoms.append([i * lattice_constant, j * lattice_constant + lattice_constant/2, k * lattice_constant + lattice_constant/2])
    
    return np.array(atoms)