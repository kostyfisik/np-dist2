"""Input/output functions for np-dist2."""

from pathlib import Path
import numpy as np


def read_timestep(filename):
    """Read a timestep from a LAMMPS dump file."""
    with open(filename) as f:
        [f.readline() for _ in range(3)]
        atoms_num = int(f.readline().strip())
        atoms_pos = np.zeros((atoms_num, 5))
        line = f.readline()
        # Add a counter to prevent infinite looping
        attempts = 0
        max_attempts = 1000  # Prevent infinite loop
        while "ITEM: ATOMS" not in line and attempts < max_attempts:
            line = f.readline()
            if not line:  # End of file
                raise ValueError("Could not find ITEM: ATOMS header in file")
            attempts += 1
            
        if attempts >= max_attempts:
            raise ValueError("Could not find ITEM: ATOMS header in file")
            
        for i in range(atoms_num):
            line = f.readline().split(" ")
            arr = np.array([float(x) for x in line[:5]])
            atoms_pos[i, :] = arr[:]
    idxs = np.argsort(atoms_pos[:, 0])
    all_data = atoms_pos[idxs]
    return all_data[:, 2:5], atoms_num


def read_data(filename):
    """Read atomic positions from a LAMMPS data file."""
    with open(filename) as f:
        f.readline()
        line = f.readline()
        atoms_num = int(f.readline().split(" ")[0])
        atoms_pos = np.zeros((atoms_num, 3))
        # Add a counter to prevent infinite looping
        attempts = 0
        max_attempts = 1000  # Prevent infinite loop
        while "atomic" not in line and attempts < max_attempts:
            line = f.readline()
            if not line:  # End of file
                raise ValueError("Could not find 'atomic' keyword in file")
            attempts += 1
            
        if attempts >= max_attempts:
            raise ValueError("Could not find 'atomic' keyword in file")
            
        f.readline()
        for i in range(atoms_num):
            line = f.readline().split(" ")[2:5]
            arr = np.array([float(x) for x in line])
            atoms_pos[i, :] = arr[:]
    return atoms_pos


def read_raw_data(filename):
    """Read raw data from a file."""
    with open(filename) as f:
        f.readline()
        f.readline()
        f.readline()
        atoms_num = int(f.readline().split(" ")[0])
        atoms_pos = np.loadtxt(filename, skiprows=9)

    assert atoms_num == atoms_pos.shape[0], 'Incorrect number of atoms'

    return atoms_pos


def list_files_with_extension(directory, extension):
    """List files with a specific extension in a directory."""
    return list(Path(directory).rglob(f"*.{extension}"))