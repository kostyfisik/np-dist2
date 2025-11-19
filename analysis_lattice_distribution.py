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
from src.np_dist2.analysis import (
    calculate_lattice_parameter_distribution,
    convert_lattice_to_density
)


def process_file(
    filepath: Path,
    output_dir: Path,
    num_directions: int = 100,
    cylinder_radius: float = 5.0,
    create_plot: bool = True
) -> dict:
    """
    Process a single .npy file and calculate lattice parameter distribution.

    Args:
        filepath: Path to the .npy file
        output_dir: Directory to save output files
        num_directions: Number of random directions to sample
        cylinder_radius: Radius of the cylinder for sampling
        create_plot: Whether to create plots

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
        print(f"  Calculating lattice parameter distribution...", end=" ", flush=True)
        radial_distances, lattice_params = calculate_lattice_parameter_distribution(
            atoms, num_directions, cylinder_radius
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
        }

        # Create plot
        if create_plot:
            print(f"  Creating plot...", end=" ", flush=True)
            plot_file = output_dir / f"{base_name}_density_profile.png"
            create_density_plot(
                radial_distances,
                densities,
                lattice_params,
                plot_file,
                base_name
            )
            print(f"Saved to {plot_file.name}")
            stats['plot_file'] = str(plot_file)

        return stats

    except Exception as e:
        print(f"Error processing {filepath.name}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': str(e)}


def create_density_plot(
    radial_distances: np.ndarray,
    densities: np.ndarray,
    lattice_params: np.ndarray,
    output_file: Path,
    title: str
):
    """
    Create a plot showing density and lattice parameter vs radial distance.

    Args:
        radial_distances: Array of radial positions
        densities: Array of local densities
        lattice_params: Array of local lattice parameters
        output_file: Path to save the plot
        title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Sort data by radial distance for better visualization
    sort_idx = np.argsort(radial_distances)
    r_sorted = radial_distances[sort_idx]
    density_sorted = densities[sort_idx]
    lattice_sorted = lattice_params[sort_idx]

    # Plot 1: Density vs radius
    ax1.scatter(r_sorted, density_sorted, alpha=0.5, s=10, c='blue')

    # Add binned average line with ±1 std
    if len(r_sorted) > 10:
        num_bins = min(50, len(r_sorted) // 10)
        bin_edges = np.linspace(r_sorted.min(), r_sorted.max(), num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_means = []
        bin_stds = []

        for i in range(num_bins):
            mask = (r_sorted >= bin_edges[i]) & (r_sorted < bin_edges[i + 1])
            if np.any(mask):
                bin_means.append(np.mean(density_sorted[mask]))
                bin_stds.append(np.std(density_sorted[mask]))
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)

        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)

        # Plot binned average and ±1 std band
        ax1.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Binned average', zorder=10)
        ax1.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                         alpha=0.3, color='red', label='±1 std', zorder=5)

        # Scale y-axis to match -1std to +1std range
        valid_mask = ~np.isnan(bin_means) & ~np.isnan(bin_stds)
        if np.any(valid_mask):
            y_min = np.nanmin(bin_means[valid_mask] - bin_stds[valid_mask])
            y_max = np.nanmax(bin_means[valid_mask] + bin_stds[valid_mask])
            y_margin = (y_max - y_min) * 0.05  # 5% margin
            ax1.set_ylim(y_min - y_margin, y_max + y_margin)

        ax1.legend()

    ax1.set_xlabel('Radial Distance (Å)', fontsize=12)
    ax1.set_ylabel('Local Ionic Density (atoms/Ų)', fontsize=12)
    ax1.set_title(f'Density Profile: {title}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Lattice parameter vs radius
    ax2.scatter(r_sorted, lattice_sorted, alpha=0.5, s=10, c='green')

    # Add binned average line with ±1 std
    if len(r_sorted) > 10:
        bin_means_lattice = []
        bin_stds_lattice = []
        for i in range(num_bins):
            mask = (r_sorted >= bin_edges[i]) & (r_sorted < bin_edges[i + 1])
            if np.any(mask):
                bin_means_lattice.append(np.mean(lattice_sorted[mask]))
                bin_stds_lattice.append(np.std(lattice_sorted[mask]))
            else:
                bin_means_lattice.append(np.nan)
                bin_stds_lattice.append(np.nan)

        bin_means_lattice = np.array(bin_means_lattice)
        bin_stds_lattice = np.array(bin_stds_lattice)

        # Plot binned average and ±1 std band
        ax2.plot(bin_centers, bin_means_lattice, 'orange', linewidth=2, label='Binned average', zorder=10)
        ax2.fill_between(bin_centers, bin_means_lattice - bin_stds_lattice,
                         bin_means_lattice + bin_stds_lattice,
                         alpha=0.3, color='orange', label='±1 std', zorder=5)

        # Scale y-axis to match -1std to +1std range
        valid_mask = ~np.isnan(bin_means_lattice) & ~np.isnan(bin_stds_lattice)
        if np.any(valid_mask):
            y_min = np.nanmin(bin_means_lattice[valid_mask] - bin_stds_lattice[valid_mask])
            y_max = np.nanmax(bin_means_lattice[valid_mask] + bin_stds_lattice[valid_mask])
            y_margin = (y_max - y_min) * 0.05  # 5% margin
            ax2.set_ylim(y_min - y_margin, y_max + y_margin)

        ax2.legend()

    ax2.set_xlabel('Radial Distance (Å)', fontsize=12)
    ax2.set_ylabel('Local Lattice Parameter (Å)', fontsize=12)
    ax2.set_title('Lattice Parameter Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

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

  # Custom parameters
  python analysis_lattice_distribution.py data/ -n 50 -r 3.0 --no-plot
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
            create_plot=not args.no_plot
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
