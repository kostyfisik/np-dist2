#!/usr/bin/env python3
"""Script to analyze lattice parameter distribution and density profiles for nanoparticles.

This script processes .npy files containing atomic coordinates and calculates:
1. Local lattice parameters as a function of radial distance
2. Local ionic densities as a function of radial distance
3. Plots showing density vs radial coordinate
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from src.np_dist2.analysis import (
    calculate_lattice_parameter_distribution,
    convert_lattice_to_density
)


def process_file(
    filepath: Path,
    output_dir: Path,
    num_directions: int = 100,
    cylinder_radius: float = 5.0,
    create_plot: bool = True,
    dislocation_aware: bool = False
) -> dict:
    """
    Process a single .npy file and calculate lattice parameter distribution.

    Args:
        filepath: Path to the .npy file
        output_dir: Directory to save output files
        num_directions: Number of random directions to sample
        cylinder_radius: Radius of the cylinder for sampling
        create_plot: Whether to create plots
        dislocation_aware: Whether to use dislocation-aware lattice parameter calculation

    Returns:
        Dictionary with processing results and statistics
    """
    try:
        atoms = np.load(filepath)

        if atoms.ndim != 2 or atoms.shape[1] != 3:
            print(f"Warning: {filepath.name} has invalid shape {atoms.shape}, skipping", file=sys.stderr)
            return {'status': 'error', 'message': 'invalid shape'}

        if len(atoms) == 0:
            print(f"Warning: {filepath.name} is empty, skipping", file=sys.stderr)
            return {'status': 'error', 'message': 'empty file'}

        # Calculate lattice parameter distribution
        print(f"  Calculating lattice parameter distribution (aware={dislocation_aware})...", end=" ", flush=True)
        radial_distances, lattice_params = calculate_lattice_parameter_distribution(
            atoms, num_directions, cylinder_radius, dislocation_aware=dislocation_aware
        )

        if len(radial_distances) == 0:
            print("No data collected")
            return {'status': 'error', 'message': 'no data collected'}

        print(f"{len(radial_distances)} data points")

        # Convert to density
        print(f"  Converting to density...", end=" ", flush=True)
        densities = convert_lattice_to_density(lattice_params, structure='fcc')
        print("Done")

        # Prepare output filename base (without extension)
        base_name = filepath.stem

        # Save results as .npy files
        output_r_file = output_dir / f"{base_name}_radial_distances.npy"
        output_lattice_file = output_dir / f"{base_name}_lattice_params.npy"
        output_density_file = output_dir / f"{base_name}_densities.npy"

        np.save(output_r_file, radial_distances)
        np.save(output_lattice_file, lattice_params)
        np.save(output_density_file, densities)

        print(f"  Saved data to {output_dir}")

        # Optimize fit parameters against raw data (sorted)
        sort_idx = np.argsort(radial_distances)
        r_sorted = radial_distances[sort_idx]
        density_sorted = densities[sort_idx]

        left_mask = r_sorted <= (r_sorted[0] + r_sorted[-1]) / 2
        core_init = float(np.mean(density_sorted[left_mask])) if np.any(left_mask) else 0.057
        x0 = [1.0, core_init / CORE_REF, 1.0]
        bounds = [(1e-6, None), (1e-6, None), (1.0 / WIDTH_REF, 7.0 / WIDTH_REF)]
        result = minimize(density_fitness, x0=x0, args=(r_sorted, density_sorted),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-15})
        factor_opt = result.x[0] * FACTOR_REF
        core_opt   = result.x[1] * CORE_REF
        width_opt  = result.x[2] * WIDTH_REF
        print(f"  Optimized fit: factor={factor_opt:.6f}, core={core_opt:.6f}, width={width_opt:.4f} "
              f"(fitness={result.fun:.6f})")
        print(f"  Convergence: success={result.success}, nit={result.nit}, nfev={result.nfev}")
        print(f"  Message: {result.message}")

        # Calculate statistics
        stats = {
            'status': 'success',
            'num_atoms': len(atoms),
            'num_data_points': len(radial_distances),
            'r_min': np.min(radial_distances),
            'r_max': np.max(radial_distances),
            'lattice_mean': np.mean(lattice_params),
            'lattice_std': np.std(lattice_params),
            'density_mean': np.mean(densities),
            'density_std': np.std(densities),
            'fit_factor': factor_opt,
            'fit_core': core_opt,
            'fit_width': width_opt,
            'fit_fitness': result.fun,
            'fit_converged': result.success,
        }

        # Create plot
        if create_plot:
            print(f"  Creating plot...", end=" ", flush=True)
            plot_file = output_dir / f"fit_{base_name}_density_profile.png"
            create_density_plot(
                radial_distances,
                densities,
                lattice_params,
                plot_file,
                base_name,
                factor_opt,
                core_opt,
                width_opt,
            )
            print(f"Saved to {plot_file.name}")
            stats['plot_file'] = str(plot_file)

        return stats

    except Exception as e:
        print(f"Error processing {filepath.name}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': str(e)}


# Reference scale values for parameter normalisation (optimisation works around 1.0)
FACTOR_REF = 0.000186
CORE_REF   = 0.057636
WIDTH_REF  = 4.2107


def density_fitness(params: np.ndarray, vector_x: np.ndarray, density_data: np.ndarray) -> float:
    """
    Fitness function: RMSE between density_fit and raw density data.

    params are normalised by (FACTOR_REF, CORE_REF, WIDTH_REF) so the optimizer
    operates around unit values.  Evaluated only on the outer 15 Å.

    Args:
        params: [factor/FACTOR_REF, core/CORE_REF, width/WIDTH_REF]
        vector_x: Sorted radial positions
        density_data: Raw density values corresponding to vector_x

    Returns:
        Root mean squared error of (density_fit - density_data)
    """
    factor = params[0] * FACTOR_REF
    core   = params[1] * CORE_REF
    width  = params[2] * WIDTH_REF
    R = vector_x[-1]
    outer_mask = vector_x >= R - 15.0
    vx = vector_x[outer_mask]
    dd = density_data[outer_mask]
    fit_vals = density_fit(vx, factor=factor, core=core, width=width)
    return np.sqrt(np.mean((fit_vals - dd) ** 2))


def density_fit(vector_x: np.ndarray, factor: float = 0.001, core: float = 0.057, width: float = 3) -> np.ndarray:
    """
    Piecewise fit function for density profile.

    Constant at `core` from 0 to a = vector_x[-1] - width, then transitions
    into a parabola centered at (a, core) with slope factor*(x-a)**2.

    Args:
        vector_x: Array of radial positions (must be sorted)
        factor: Parabolic slope coefficient
        core: Constant core density level
        width: Width of the parabolic transition region at the surface

    Returns:
        Array of fitted density values
    """
    a = vector_x[-1] - width
    y = np.where(vector_x <= a, core, core + factor * (vector_x - a) ** 2)
    return y


def create_density_plot(
    radial_distances: np.ndarray,
    densities: np.ndarray,
    lattice_params: np.ndarray,
    output_file: Path,
    title: str,
    factor: float,
    core: float,
    width: float,
):
    """
    Create a plot showing density and the optimized fit vs radial distance.

    Args:
        radial_distances: Array of radial positions
        densities: Array of local densities
        lattice_params: Array of local lattice parameters
        output_file: Path to save the plot
        title: Title for the plot
        factor: Optimized parabolic slope coefficient
        core: Optimized core density level
        width: Optimized parabolic transition width
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    # Sort data by radial distance for better visualization
    sort_idx = np.argsort(radial_distances)
    r_sorted = radial_distances[sort_idx]
    density_sorted = densities[sort_idx]

    # Plot: Density vs radius
    ax1.scatter(r_sorted, density_sorted, alpha=0.5, s=10, c='blue')

    fit_y = density_fit(r_sorted, factor=factor, core=core, width=width)
    ax1.plot(r_sorted, fit_y, 'r-', linewidth=2,
             label=f'fit (factor={factor:.4f}, core={core:.4f}, width={width:.2f})')

    # Set y-axis limits based on fit range + 5% margin
    fit_min, fit_max = fit_y.min(), fit_y.max()
    margin = fit_max * 0.05
    ax1.set_ylim(fit_min - margin, fit_max + margin)

    ax1.set_xlabel('Radial Distance (Å)', fontsize=12)
    ax1.set_ylabel('Local Ionic Density (atoms/Ų)', fontsize=12)
    ax1.set_title(f'Density Profile: {title}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to process all .npy files in a directory."""
    parser = argparse.ArgumentParser(
        description="Analyze lattice parameter distribution and density profiles for nanoparticles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default directory
  python analysis_lattice_distribution.py

  # Specify custom directory
  python analysis_lattice_distribution.py /path/to/data

  # Custom parameters with dislocation-aware mode
  python analysis_lattice_distribution.py data/ --dislocation-aware
        """
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="sample_data/res-2025-01-13",
        help="Directory containing .npy files (default: sample_data/res-2025-01-13)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory for results (default: same as data_dir)"
    )
    parser.add_argument(
        "-n", "--num-directions",
        type=int,
        default=100,
        help="Number of random directions to sample (default: 100)"
    )
    parser.add_argument(
        "-r", "--cylinder-radius",
        type=float,
        default=5.0,
        help="Radius of sampling cylinder in Angstroms (default: 5.0)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip creating plots"
    )
    parser.add_argument(
        "--dislocation-aware",
        action="store_true",
        help="Enable dislocation-aware mode (compares 9 vs 12 neighbors)"
    )

    args = parser.parse_args()

    # Validate input directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Directory '{data_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    if not data_dir.is_dir():
        print(f"Error: '{data_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = data_dir

    # Find all .npy files
    npy_files = sorted(data_dir.glob("*.npy"))

    # Filter out previously generated result files
    npy_files = [
        f for f in npy_files
        if not any(suffix in f.stem for suffix in ['_radial_distances', '_lattice_params', '_densities'])
    ]

    # Only run the target data sets
    target_sets = ['Ag30angs_avg101', 'Ag53angs_avg101', 'Ag78angs_avg101', 'Ag100angs_avg101']
    npy_files = [f for f in npy_files if f.stem in target_sets]

    if not npy_files:
        print(f"Error: No .npy files found in '{data_dir}'", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print(f"Lattice Distribution Analysis")
    print("=" * 80)
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(npy_files)} file(s) to process")
    print(f"Parameters: {args.num_directions} directions, {args.cylinder_radius} Å cylinder radius")
    print(f"Mode: {'Dislocation Aware' if args.dislocation_aware else 'Standard'}")
    print(f"Create plots: {not args.no_plot}")
    print("=" * 80)
    print()

    # Process all files
    results = []
    for i, filepath in enumerate(npy_files, 1):
        print(f"[{i}/{len(npy_files)}] Processing {filepath.name}")
        stats = process_file(
            filepath,
            output_dir,
            args.num_directions,
            args.cylinder_radius,
            create_plot=not args.no_plot,
            dislocation_aware=args.dislocation_aware
        )
        stats['filename'] = filepath.name
        results.append(stats)
        print()

    # Print summary
    print("=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    successful.sort(key=lambda r: r['r_max'])

    print(f"Total files: {len(results)}")
    print(f"Successfully processed: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\n" + "-" * 80)
        print("STATISTICS FOR SUCCESSFULLY PROCESSED FILES")
        print("-" * 80)
        print(f"{'Filename':<40} {'Data Pts':<10} {'Density (mean±std)':<25}")
        print("-" * 80)

        for r in successful:
            density_str = f"{r['density_mean']:.4f} ± {r['density_std']:.4f}"
            print(f"{r['filename']:<40} {r['num_data_points']:<10} {density_str:<25}")

        # Overall statistics
        all_densities_mean = np.mean([r['density_mean'] for r in successful])
        all_lattice_mean = np.mean([r['lattice_mean'] for r in successful])

        print("-" * 80)
        print(f"\nOverall average density: {all_densities_mean:.4f} atoms/Ų")
        print(f"Overall average lattice parameter: {all_lattice_mean:.4f} Å")

        print("\n" + "-" * 80)
        print("FIT PARAMETERS SUMMARY")
        print("-" * 80)
        print(f"{'Filename':<40} {'R_max':>7}  {'factor':<12} {'core':<12} {'width':<10} {'fitness':<12} {'conv'}")
        print("-" * 80)
        for r in successful:
            print(f"{r['filename']:<40} {r['r_max']:>7.2f}  {r['fit_factor']:<12.6f} {r['fit_core']:<12.6f} "
                  f"{r['fit_width']:<10.4f} {r['fit_fitness']:<12.6f} {str(r['fit_converged'])}")
        print("-" * 80)

    if failed:
        print("\n" + "-" * 80)
        print("FAILED FILES")
        print("-" * 80)
        for r in failed:
            print(f"  {r['filename']}: {r.get('message', 'unknown error')}")

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()