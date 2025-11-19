#!/usr/bin/env python3
"""Script to calculate effective radius for nanoparticle data files.

This script processes .npy files containing atomic coordinates and calculates
the effective radius using cylindrical sampling across multiple random directions.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from src.np_dist2.analysis import calculate_effective_radius


def process_file(filepath: Path, num_directions: int = 100, cylinder_radius: float = 5.0) -> tuple[str, float, float]:
    """
    Process a single .npy file and calculate its effective radius.

    Args:
        filepath: Path to the .npy file
        num_directions: Number of random directions to sample
        cylinder_radius: Radius of the cylinder for sampling

    Returns:
        Tuple of (filename, effective_radius, std_dev)
    """
    try:
        atoms = np.load(filepath)

        if atoms.ndim != 2 or atoms.shape[1] != 3:
            print(f"Warning: {filepath.name} has invalid shape {atoms.shape}, skipping", file=sys.stderr)
            return filepath.name, np.nan, np.nan

        if len(atoms) == 0:
            print(f"Warning: {filepath.name} is empty, skipping", file=sys.stderr)
            return filepath.name, 0.0, 0.0

        eff_radius, all_radii = calculate_effective_radius(atoms, num_directions, cylinder_radius)
        std_dev = np.std(all_radii) if len(all_radii) > 0 else 0.0

        return filepath.name, eff_radius, std_dev

    except Exception as e:
        print(f"Error processing {filepath.name}: {e}", file=sys.stderr)
        return filepath.name, np.nan, np.nan


def main():
    """Main function to process all .npy files in a directory."""
    parser = argparse.ArgumentParser(
        description="Calculate effective radius for nanoparticle atomic coordinates"
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="sample_data/res-2025-01-13",
        help="Directory containing .npy files (default: sample_data/res-2025-01-13)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: <input_dir>/effective_radius_results.txt)"
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

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Directory '{input_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find all .npy files
    npy_files = sorted(input_dir.glob("*.npy"))

    if not npy_files:
        print(f"Error: No .npy files found in '{input_dir}'", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(npy_files)} .npy file(s) in {input_dir}")
    print(f"Processing with {args.num_directions} directions and cylinder radius {args.cylinder_radius} Å")
    print()

    # Process all files
    results = []
    for filepath in npy_files:
        print(f"Processing {filepath.name}...", end=" ", flush=True)
        filename, eff_radius, std_dev = process_file(
            filepath,
            args.num_directions,
            args.cylinder_radius
        )
        results.append((filename, eff_radius, std_dev))
        if not np.isnan(eff_radius):
            print(f"R_eff = {eff_radius:.3f} Å (σ = {std_dev:.3f} Å)")
        else:
            print("Failed")

    # Determine output file path
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_dir / "effective_radius_results.txt"

    # Write results to file
    with open(output_file, 'w') as f:
        f.write("Effective Radius Analysis Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Number of directions: {args.num_directions}\n")
        f.write(f"Cylinder radius: {args.cylinder_radius} Å\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Filename':<40} {'Eff. Radius (Å)':<20} {'Std. Dev (Å)':<20}\n")
        f.write("-" * 80 + "\n")

        for filename, eff_radius, std_dev in results:
            if np.isnan(eff_radius):
                f.write(f"{filename:<40} {'N/A':<20} {'N/A':<20}\n")
            else:
                f.write(f"{filename:<40} {eff_radius:<20.3f} {std_dev:<20.3f}\n")

        f.write("-" * 80 + "\n")

        # Calculate statistics
        valid_radii = [r for _, r, _ in results if not np.isnan(r)]
        if valid_radii:
            f.write(f"\nSummary Statistics:\n")
            f.write(f"  Files processed successfully: {len(valid_radii)}\n")
            f.write(f"  Mean effective radius: {np.mean(valid_radii):.3f} Å\n")
            f.write(f"  Median effective radius: {np.median(valid_radii):.3f} Å\n")
            f.write(f"  Min effective radius: {np.min(valid_radii):.3f} Å\n")
            f.write(f"  Max effective radius: {np.max(valid_radii):.3f} Å\n")

    print()
    print(f"Results written to: {output_file}")
    print(f"Successfully processed {len(valid_radii)}/{len(results)} files")


if __name__ == "__main__":
    main()
