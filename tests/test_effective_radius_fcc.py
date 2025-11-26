"""Tests for effective radius calculation on FCC cubes."""

import numpy as np
import pytest
from np_dist2.grid_generator import generate_fcc_lattice, shift_grid_to_origin
from np_dist2.analysis import calculate_effective_radius


def get_equivalent_sphere_radius_from_cube_vol(side_length: float) -> float:
    """Calculate the radius of a sphere that has the same volume as a cube."""
    volume = side_length ** 3
    return (3 * volume / (4 * np.pi)) ** (1/3)


@pytest.mark.parametrize("size", [4, 8, 12])
def test_fcc_cube_effective_radius_comparison(size):
    """
    Compare accuracy of v3 (Solid Angle) and v4 (Unique Atoms) algorithms.
    """
    # 1. Generate the FCC lattice
    atoms = generate_fcc_lattice(size)
    atoms_centered = shift_grid_to_origin(atoms)
    
    # 2. Theoretical Expectation
    # We use the bounding box extent of atom centers as the reference dimension
    coords_min = np.min(atoms_centered, axis=0)
    coords_max = np.max(atoms_centered, axis=0)
    L_effective = np.mean(coords_max - coords_min)
    r_theoretical = get_equivalent_sphere_radius_from_cube_vol(L_effective)
    
    print(f"\nAnalysis for FCC Cube (size={size}):")
    print(f"  Number of atoms: {len(atoms)}")
    print(f"  Side Length (L): {L_effective:.4f}")
    print(f"  Theoretical Radius: {r_theoretical:.4f}")

    # 3. Run v3 (Solid Angle)
    # This weights by directional hit frequency.
    # On small cubes with finite probe radius, corners are "seen" 
    # from a wider angle than they strictly occupy, causing overestimation.
    r_v3, _ = calculate_effective_radius(
        atoms_centered, num_directions=5000, cylinder_radius=1.0, method='v3'
    )
    diff_v3 = abs(r_v3 - r_theoretical)
    err_v3 = (diff_v3 / r_theoretical) * 100
    
    print(f"  v3 (Solid Angle) Radius: {r_v3:.4f} (Err: {err_v3:.2f}%)")

    # 4. Run v4 (Unique Atoms)
    # This weights every unique surface atom equally.
    # For a cube, this reduces the weight of corners (fewer atoms) 
    # compared to faces (many atoms), countering the overestimation 
    # caused by corner protrusion.
    r_v4, _ = calculate_effective_radius(
        atoms_centered, num_directions=5000, cylinder_radius=1.0, method='v4'
    )
    diff_v4 = abs(r_v4 - r_theoretical)
    err_v4 = (diff_v4 / r_theoretical) * 100
    
    print(f"  v4 (Unique Atoms) Radius: {r_v4:.4f} (Err: {err_v4:.2f}%)")
    
    # 5. Assertions
    # Verify that the new algorithm (v4) provides a better estimate 
    # for these shapes than the previous one (v3).
    assert err_v4 < err_v3, \
        f"v4 error ({err_v4:.2f}%) should be lower than v3 error ({err_v3:.2f}%)"
    
    # Verify v4 is reasonably accurate (< 10%)
    assert err_v4 < 10.0, f"v4 error {err_v4:.2f}% is too high"


def test_effective_radius_scaling():
    """Verify that effective radius scales linearly with cube size (using default v4)."""
    size_small = 5
    size_large = 10
    
    atoms_small = shift_grid_to_origin(generate_fcc_lattice(size_small))
    r_small, _ = calculate_effective_radius(atoms_small, 5000, 1.0, method='v4')
    
    atoms_large = shift_grid_to_origin(generate_fcc_lattice(size_large))
    r_large, _ = calculate_effective_radius(atoms_large, 5000, 1.0, method='v4')
    
    # Ratio should match geometric scaling
    L_small = np.mean(np.ptp(atoms_small, axis=0))
    L_large = np.mean(np.ptp(atoms_large, axis=0))
    ratio_expected = L_large / L_small
    
    ratio_calc = r_large / r_small
    
    assert np.isclose(ratio_calc, ratio_expected, rtol=0.05)