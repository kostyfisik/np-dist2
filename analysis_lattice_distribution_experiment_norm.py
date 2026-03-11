#!/usr/bin/env python3
"""Compare analytical atom counts from fit parameters with actual atom counts.

Reads fit_results.json produced by analysis_lattice_distribution_fit.py,
integrates the piecewise density profile analytically assuming spherical
symmetry:

    N = 4π ∫₀^R density(r) r² dr

    density(r) = core                          for r ≤ a  (a = R_max - width)
               = core + factor·(r - a)²        for r > a

Analytical result:
    N = 4π · [ core·a³/3
              + core·(W³/3 + a·W² + a²·W)
              + factor·(W⁵/5 + a·W⁴/2 + a²·W³/3) ]
where W = width.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np


def analytical_atom_count(factor: float, core: float, width: float, r_max: float) -> float:
    """
    Analytically integrate the piecewise density profile over a sphere.

    N = 4π ∫₀^R density(r) r² dr

    Args:
        factor: Parabolic slope coefficient (atoms/Å⁵)
        core:   Core density (atoms/Å³)
        width:  Parabolic surface width (Å)
        r_max:  Total particle radius (Å)

    Returns:
        Estimated total number of atoms
    """
    a = r_max - width
    W = width
    # Core sphere [0, a]
    n_core_inner = core * a**3 / 3
    # Shell [a, R]: core contribution + parabolic contribution
    n_core_outer = core * (W**3 / 3 + a * W**2 + a**2 * W)
    n_para       = factor * (W**5 / 5 + a * W**4 / 2 + a**2 * W**3 / 3)
    return 4 * np.pi * (n_core_inner + n_core_outer + n_para)


def main():
    parser = argparse.ArgumentParser(
        description="Compare analytical atom counts from fit to actual atom counts"
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default="full_data/res-2025-01-13/fit_results.json",
        help="Path to fit_results.json (default: full_data/res-2025-01-13/fit_results.json)"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing original .npy files (default: same dir as json_file)"
    )
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: '{json_path}' not found", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as f:
        fit = json.load(f)

    data_dir = Path(args.data_dir) if args.data_dir else json_path.parent

    has_combined = fit.get("combined_factor") is not None

    print("=" * 90)
    print("Analytical Atom Count vs Actual")
    print("=" * 90)
    print(f"Fit file : {json_path}")
    print(f"Data dir : {data_dir}")
    if has_combined:
        print(f"Combined : factor={fit['combined_factor']:.6f}, "
              f"width={fit['combined_width']:.4f}")
    print()

    header = (f"{'Dataset':<30} {'Actual':>8}  "
              f"{'N_indiv':>10} {'err_i%':>8}  "
              f"{'N_comb':>10} {'err_c%':>8}")
    print(header)
    print("-" * len(header))

    for ds in fit["datasets"]:
        npy_path = data_dir / ds["filename"]
        if npy_path.exists():
            atoms = np.load(npy_path)
            actual = len(atoms)
        else:
            # Fall back to stored value
            actual = ds["num_atoms"]
            print(f"  Warning: {npy_path.name} not found, using stored num_atoms={actual}",
                  file=sys.stderr)

        n_indiv = analytical_atom_count(
            ds["indiv_factor"], ds["indiv_core"], ds["indiv_width"], ds["r_max"]
        )
        err_i = (n_indiv - actual) / actual * 100

        row = (f"{ds['filename']:<30} {actual:>8d}  "
               f"{n_indiv:>10.1f} {err_i:>+8.2f}%")

        if has_combined and ds.get("comb_core") is not None:
            n_comb = analytical_atom_count(
                fit["combined_factor"], ds["comb_core"], fit["combined_width"], ds["r_max"]
            )
            err_c = (n_comb - actual) / actual * 100
            row += f"  {n_comb:>10.1f} {err_c:>+8.2f}%"

        print(row)

    print("-" * len(header))
    print()
    print("N_indiv / N_comb: atom count from analytical integration of individual / combined fit")
    print("err%: (N_analytical - N_actual) / N_actual × 100")


if __name__ == "__main__":
    main()
