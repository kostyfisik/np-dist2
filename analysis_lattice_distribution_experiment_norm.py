#!/usr/bin/env python3
"""Compare analytical atom counts from fit parameters with actual atom counts,
and compute density-normalised core values matching experimental densities.

Reads fit_results.json produced by analysis_lattice_distribution_fit.py,
integrates the piecewise density profile analytically assuming spherical
symmetry:

    N = 4π ∫₀^R density(r) r² dr

    density(r) = core                          for r ≤ a  (a = R_max - width)
               = core + factor·(r - a)²        for r > a

Analytical result:
    N = 4π · [ core·A + factor·B ]
    A = a³/3 + W³/3 + aW² + a²W
    B = W⁵/5 + aW⁴/2 + a²W³/3
where W = width.

To match experimental average density ρ_exp (atoms/Å³):
    ρ_exp = N / V_sphere = N / (4π/3 · R³)
    → N_target = ρ_exp · (4π/3) · R³
    → core_new = (ρ_exp · R³/3 − factor·B) / A
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np


# Conversion: 10^28 m^-3 → atoms/Å³
# 1 m = 1e10 Å → 1 m^3 = 1e30 Å³ → 1 m^-3 = 1e-30 Å^-3
# N × 10^28 m^-3 = N × 10^28 × 10^-30 Å^-3 = N × 1e-2 Å^-3
EXP_DENSITY_SCALE = 1e-2  # (10^28 m^-3) → atoms/Å³


def _geometry_factors(width: float, r_max: float):
    """Return (A, B) geometry integrals for given width and r_max.

    A = R³/3  (algebraically equivalent form, valid only when a = r_max - width > 0)
    B = ∫₀^W u²(u+a)² du  where W = width, u = r − a
    """
    a = r_max - width
    if a <= 0:
        raise ValueError(
            f"width ({width:.4f}) must be strictly less than r_max ({r_max:.4f}); "
            "the core region vanishes when a = r_max - width ≤ 0"
        )
    W = width
    A = a**3 / 3 + W**3 / 3 + a * W**2 + a**2 * W  # equals r_max³/3
    B = W**5 / 5 + a * W**4 / 2 + a**2 * W**3 / 3
    return A, B


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
    A, B = _geometry_factors(width, r_max)
    return 4 * np.pi * (core * A + factor * B)


def core_from_experimental_density(rho_exp_angs: float, factor: float,
                                   width: float, r_max: float) -> float:
    """
    Solve analytically for core such that the average sphere density equals rho_exp.

    Average density = N / V = N / (4π/3 · R³) = ρ_exp
    → 4π·(core·A + factor·B) = ρ_exp · (4π/3) · R³
    → core = (ρ_exp · R³/3 − factor·B) / A

    Args:
        rho_exp_angs: Experimental density in atoms/Å³
        factor:       Parabolic slope coefficient
        width:        Parabolic surface width (Å)
        r_max:        Total particle radius (Å)

    Returns:
        core value (atoms/Å³) that yields the target average density

    Raises:
        ValueError: if the result would be non-positive (unphysical)
    """
    A, B = _geometry_factors(width, r_max)
    numerator = rho_exp_angs * r_max**3 / 3 - factor * B
    if numerator <= 0:
        raise ValueError(
            f"core_new = {numerator/A:.6f} ≤ 0: factor ({factor:.6f}) × B ({B:.2f}) "
            f"exceeds ρ_exp × R³/3 ({rho_exp_angs * r_max**3 / 3:.2f}). "
            "Reduce factor or increase r_max."
        )
    return numerator / A


def main():
    parser = argparse.ArgumentParser(
        description="Analytical atom counts and density-normalised core values"
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default="full_data/res-2025-01-13/fit_results.json",
        help="Path to fit_results.json (default: full_data/res-2025-01-13/fit_results.json)"
    )
    parser.add_argument(
        "--exp-density",
        default="experimental_density.json",
        help="Path to experimental_density.json (default: experimental_density.json)"
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

    exp_path = Path(args.exp_density)
    if not exp_path.exists():
        print(f"Error: '{exp_path}' not found", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as f:
        fit = json.load(f)

    with open(exp_path) as f:
        exp_data = json.load(f)

    data_dir = Path(args.data_dir) if args.data_dir else json_path.parent

    has_combined = fit.get("combined_factor") is not None

    # ------------------------------------------------------------------ #
    # Section 1: Analytical Atom Count vs Actual                          #
    # ------------------------------------------------------------------ #
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

    actuals = {}
    for ds in fit["datasets"]:
        stem = Path(ds["filename"]).stem
        npy_path = data_dir / ds["filename"]
        if npy_path.exists():
            atoms = np.load(npy_path)
            actual = len(atoms)
        else:
            actual = ds["num_atoms"]
            print(f"  Warning: {npy_path.name} not found, using stored num_atoms={actual}",
                  file=sys.stderr)
        actuals[stem] = actual

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

    # ------------------------------------------------------------------ #
    # Section 2: Density-Normalised Core Values                           #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 90)
    print("Density-Normalised Core Values  (core adjusted to match experimental ρ)")
    print("=" * 90)
    print(f"Exp density file : {exp_path}")
    print()
    print("Derivation:")
    print("  ρ_avg = N / V_sphere = N / (4π/3·R³) = ρ_exp")
    print("  N = 4π·(core·A + factor·B)  where A, B are geometry integrals")
    print("  → core_new = (ρ_exp·R³/3 − factor·B) / A")
    print()

    exp_densities = exp_data["data"]  # stem → {density, nominal_radius_angs}

    # Table for individual fit normalisation
    print("  Individual fit (factor, width per particle):")
    hdr2 = (f"  {'Dataset':<30} {'R_eff':>6} {'R_nom':>6}  "
            f"{'ρ_exp(Å⁻³)':>12}  "
            f"{'core_fit':>10} {'core_new':>10}  "
            f"{'N_new':>10} {'N_exp_nom':>10} {'err_new%':>9}")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))

    for ds in fit["datasets"]:
        stem = Path(ds["filename"]).stem
        if stem not in exp_densities:
            print(f"  Warning: no experimental density for {stem}", file=sys.stderr)
            continue
        entry = exp_densities[stem]
        rho_exp_angs = entry["density"] * EXP_DENSITY_SCALE
        r_eff = ds["r_max"]
        r_nom = entry["nominal_radius_angs"]
        core_new = core_from_experimental_density(
            rho_exp_angs, ds["indiv_factor"], ds["indiv_width"], r_eff
        )
        n_new = analytical_atom_count(
            ds["indiv_factor"], core_new, ds["indiv_width"], r_eff
        )
        n_exp_nom = rho_exp_angs * (4 * np.pi / 3) * r_nom ** 3
        actual = actuals[stem]
        err_new = (n_new - actual) / actual * 100
        print(f"  {ds['filename']:<30} {r_eff:>6.2f} {r_nom:>6.1f}  "
              f"{rho_exp_angs:>12.8f}  "
              f"{ds['indiv_core']:>10.6f} {core_new:>10.6f}  "
              f"{n_new:>10.1f} {n_exp_nom:>10.1f} {err_new:>+9.2f}%")

    print()

    if has_combined:
        print(f"  Combined fit (shared factor={fit['combined_factor']:.6f}, "
              f"width={fit['combined_width']:.4f}):")
        hdr3 = (f"  {'Dataset':<30} {'R_eff':>6} {'R_nom':>6}  "
                f"{'ρ_exp(Å⁻³)':>12}  "
                f"{'core_comb':>10} {'core_new':>10}  "
                f"{'N_new':>10} {'N_exp_nom':>10} {'err_new%':>9}")
        print(hdr3)
        print("  " + "-" * (len(hdr3) - 2))

        for ds in fit["datasets"]:
            stem = Path(ds["filename"]).stem
            if stem not in exp_densities:
                continue
            entry = exp_densities[stem]
            rho_exp_angs = entry["density"] * EXP_DENSITY_SCALE
            r_eff = ds["r_max"]
            r_nom = entry["nominal_radius_angs"]
            core_new = core_from_experimental_density(
                rho_exp_angs, fit["combined_factor"], fit["combined_width"], r_eff
            )
            n_new = analytical_atom_count(
                fit["combined_factor"], core_new, fit["combined_width"], r_eff
            )
            n_exp_nom = rho_exp_angs * (4 * np.pi / 3) * r_nom ** 3
            actual = actuals[stem]
            err_new = (n_new - actual) / actual * 100
            comb_core = ds.get("comb_core", float("nan"))
            print(f"  {ds['filename']:<30} {r_eff:>6.2f} {r_nom:>6.1f}  "
                  f"{rho_exp_angs:>12.8f}  "
                  f"{comb_core:>10.6f} {core_new:>10.6f}  "
                  f"{n_new:>10.1f} {n_exp_nom:>10.1f} {err_new:>+9.2f}%")

    print()
    print("  R_eff                : effective simulated radius = r_max from MC sampling")
    print("  R_nom                : nominal radius = D_nominal / 2 (from filename)")
    print("  Note                 : experimental particle radii are close to R_nom but not identical")
    print("  core_fit / core_comb : original fit values")
    print("  core_new             : analytically solved to satisfy ρ_avg = ρ_exp  (uses R_eff)")
    print("  N_new                : piecewise integral over R_eff  ≡  ρ_exp × (4π/3) × R_eff³")
    print("  N_exp_nom            : ρ_exp × (4π/3) × R_nom³  (nominal-radius uniform sphere)")
    print("  err_new%             : (N_new - N_actual) / N_actual × 100")


if __name__ == "__main__":
    main()

