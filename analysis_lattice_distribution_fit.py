#!/usr/bin/env python3
"""Script to analyze lattice parameter distribution and density profiles for nanoparticles.

This script processes .npy files containing atomic coordinates and calculates:
1. Local lattice parameters as a function of radial distance
2. Local ionic densities as a function of radial distance
3. Plots showing density vs radial coordinate
"""

import argparse
import json
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
            'num_atoms': int(len(atoms)),
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
            # raw sorted data kept for combined optimisation and plotting
            'r_sorted': r_sorted,
            'density_sorted': density_sorted,
            'base_name': base_name,
            'radial_distances': radial_distances,
            'densities': densities,
            'lattice_params': lattice_params,
        }

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


def combined_density_fitness(params: np.ndarray, datasets: list) -> float:
    """
    Combined fitness across multiple datasets with shared factor and width,
    but independent core per dataset.

    params are normalised by reference values so the optimizer operates around 1.0.

    Args:
        params: [factor/FACTOR_REF, width/WIDTH_REF, core0/CORE_REF, core1/CORE_REF, ...]
        datasets: list of (r_sorted, density_sorted) tuples, one per dataset

    Returns:
        Mean RMSE across all datasets (outer 15 Å window)
    """
    factor = params[0] * FACTOR_REF
    width  = params[1] * WIDTH_REF
    total = 0.0
    for i, (r_sorted, density_sorted) in enumerate(datasets):
        core = params[2 + i] * CORE_REF
        R = r_sorted[-1]
        outer_mask = r_sorted >= R - 15.0
        vx = r_sorted[outer_mask]
        dd = density_sorted[outer_mask]
        fit_vals = density_fit(vx, factor=factor, core=core, width=width)
        total += np.sqrt(np.mean((fit_vals - dd) ** 2))
    return total / len(datasets)


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
    combined_factor: float = None,
    combined_core: float = None,
    combined_width: float = None,
):
    """
    Create a plot showing density, the individual optimized fit (red) and
    optionally the combined fit (green) vs radial distance.
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    sort_idx = np.argsort(radial_distances)
    r_sorted = radial_distances[sort_idx]
    density_sorted = densities[sort_idx]

    ax1.scatter(r_sorted, density_sorted, alpha=0.5, s=10, c='blue')

    fit_y = density_fit(r_sorted, factor=factor, core=core, width=width)
    ax1.plot(r_sorted, fit_y, 'r-', linewidth=2,
             label=f'individual (factor={factor:.4f}, core={core:.4f}, width={width:.2f})')

    if combined_factor is not None:
        cfit_y = density_fit(r_sorted, factor=combined_factor, core=combined_core, width=combined_width)
        ax1.plot(r_sorted, cfit_y, 'g-', linewidth=2,
                 label=f'combined (factor={combined_factor:.4f}, core={combined_core:.4f}, width={combined_width:.2f})')
        all_fit = np.concatenate([fit_y, cfit_y])
    else:
        all_fit = fit_y

    fit_min, fit_max = all_fit.min(), all_fit.max()
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


def create_summary_plot(successful: list, output_file: Path):
    """
    Create a single summary plot with all particles' data points and both fit curves.

    Each particle gets a distinct color. Individual fits are solid lines,
    combined fits are dashed lines. Y limits are set from the global min/max
    of all fit curves with 5% of max as top/bottom margin.
    """
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(successful)))

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    all_fit_vals = []

    for color, r in zip(colors, successful):
        r_sorted       = r['r_sorted']
        density_sorted = r['density_sorted']
        label          = r['base_name']

        ax.scatter(r_sorted, density_sorted, alpha=0.3, s=5, color=color)

        ind_y = density_fit(r_sorted, factor=r['fit_factor'],
                            core=r['fit_core'], width=r['fit_width'])
        ax.plot(r_sorted, ind_y, '-', color=color, linewidth=2,
                label=f'{label} individual')
        all_fit_vals.append(ind_y)

        if 'comb_factor' in r:
            comb_y = density_fit(r_sorted, factor=r['comb_factor'],
                                 core=r['comb_core'], width=r['comb_width'])
            ax.plot(r_sorted, comb_y, '--', color=color, linewidth=2,
                    label=f'{label} combined')
            all_fit_vals.append(comb_y)

    # Y limits: global min/max of all fit curves ± 5% of max
    all_vals = np.concatenate(all_fit_vals)
    y_min, y_max = all_vals.min(), all_vals.max()
    margin = y_max * 0.05
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_xlabel('Radial Distance (Å)', fontsize=12)
    ax.set_ylabel('Local Ionic Density (atoms/Ų)', fontsize=12)
    ax.set_title('Density Profile Summary — all particles\n'
                 '(solid = individual fit, dashed = combined fit)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Summary plot saved to {output_file.name}")


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

    # Combined optimisation: shared factor + width, individual core per dataset
    combined_params = {}
    if successful:
        print("\n" + "=" * 80)
        print("COMBINED OPTIMISATION (shared factor & width, per-dataset core)")
        print("=" * 80)
        datasets = [(r['r_sorted'], r['density_sorted']) for r in successful]
        n = len(datasets)
        # Initial: mean individual factor/width, individual cores
        factor_init = np.mean([r['fit_factor'] for r in successful]) / FACTOR_REF
        width_init  = np.mean([r['fit_width']  for r in successful]) / WIDTH_REF
        cores_init  = [r['fit_core'] / CORE_REF for r in successful]
        x0_c = [factor_init, width_init] + cores_init
        bounds_c = ([(1e-6, None), (1.0 / WIDTH_REF, 7.0 / WIDTH_REF)]
                    + [(1e-6, None)] * n)
        res_c = minimize(combined_density_fitness, x0=x0_c, args=(datasets,),
                         method='L-BFGS-B', bounds=bounds_c,
                         options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-15})
        combined_factor = res_c.x[0] * FACTOR_REF
        combined_width  = res_c.x[1] * WIDTH_REF
        combined_cores  = [res_c.x[2 + i] * CORE_REF for i in range(n)]
        print(f"  factor={combined_factor:.6f}, width={combined_width:.4f} "
              f"(fitness={res_c.fun:.6f})")
        print(f"  Convergence: success={res_c.success}, nit={res_c.nit}, nfev={res_c.nfev}")
        print(f"  Message: {res_c.message}")
        for i, r in enumerate(successful):
            combined_params[r['filename']] = {
                'factor': combined_factor,
                'core':   combined_cores[i],
                'width':  combined_width,
            }
            r['comb_factor']  = combined_factor
            r['comb_core']    = combined_cores[i]
            r['comb_width']   = combined_width
            r['comb_fitness'] = res_c.fun

    # Create plots (all datasets, individual + combined curves)
    if not args.no_plot:
        print()
        for r in successful:
            base_name = r['base_name']
            plot_file = output_dir / f"fit_{base_name}_density_profile.png"
            cp = combined_params.get(r['filename'])
            print(f"  Plotting {base_name}...", end=" ", flush=True)
            create_density_plot(
                r['radial_distances'],
                r['densities'],
                r['lattice_params'],
                plot_file,
                base_name,
                r['fit_factor'], r['fit_core'], r['fit_width'],
                combined_factor=cp['factor'] if cp else None,
                combined_core=cp['core']   if cp else None,
                combined_width=cp['width'] if cp else None,
            )
            print(f"Saved to {plot_file.name}")
        print()

        # Summary plot across all particles
        summary_file = output_dir / "fit_summary_all_particles.png"
        print(f"  Creating summary plot...", end=" ", flush=True)
        create_summary_plot(successful, summary_file)

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

        if combined_params:
            print("\n" + "-" * 80)
            print("COMBINED FIT SUMMARY  "
                  f"(shared factor={list(combined_params.values())[0]['factor']:.6f}, "
                  f"width={list(combined_params.values())[0]['width']:.4f})")
            print("-" * 80)
            print(f"{'Filename':<40} {'R_max':>7}  {'core (indiv)':<14} {'core (comb)':<14} {'comb fitness'}")
            print("-" * 80)
            for r in successful:
                print(f"{r['filename']:<40} {r['r_max']:>7.2f}  "
                      f"{r['fit_core']:<14.6f} {r['comb_core']:<14.6f} {r['comb_fitness']:.6f}")
            print("-" * 80)

        # Save fit results as JSON
        fit_json = {
            "combined_factor":  float(successful[0]['comb_factor']) if combined_params else None,
            "combined_width":   float(successful[0]['comb_width'])  if combined_params else None,
            "combined_fitness": float(successful[0]['comb_fitness']) if combined_params else None,
            "datasets": [
                {
                    "filename":     r['filename'],
                    "r_max":        float(r['r_max']),
                    "num_atoms":    int(r['num_atoms']),
                    "indiv_factor": float(r['fit_factor']),
                    "indiv_core":   float(r['fit_core']),
                    "indiv_width":  float(r['fit_width']),
                    "indiv_fitness":float(r['fit_fitness']),
                    "comb_core":    float(r['comb_core']) if combined_params else None,
                }
                for r in successful
            ],
        }
        json_file = output_dir / "fit_results.json"
        with open(json_file, "w") as f:
            json.dump(fit_json, f, indent=2)
        print(f"\nFit results saved to: {json_file}")

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