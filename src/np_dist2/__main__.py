import click
import numpy as np
from matplotlib import pyplot as plt
from np_dist2.io import read_timestep, read_raw_data
from np_dist2.analysis import get_lat_vs_r, get_lat_vs_r_extended, calculate_ion_density, radial_distribution, plot_radial_distribution, average_timesteps

# FCC lattice constant for Ag (Angstroms)
FCC_AG_LATTICE_CONSTANT = 4.0896
# Ideal FCC density: 4 atoms per unit cell volume
FCC_IDEAL_DENSITY = 4 / (FCC_AG_LATTICE_CONSTANT**3)


@click.group()
@click.version_option()
def main() -> None:
    """Np Dist2 - A toolkit for nanoparticle simulation analysis."""


@main.command(name="time-avg")
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--num', default=101, help='Number of timesteps to average.')
@click.option('--output', default='atoms_avg.npy', help='Output numpy file.')
def time_avg(directory: str, num: int, output: str) -> None:
    """Averages atomic positions over simulation timesteps."""
    try:
        click.echo(f"Averaging up to {num} timesteps in directory: {directory}")
        avg_pos = average_timesteps(directory, num)
        np.save(output, avg_pos)
        click.echo(f"Averaged positions of {avg_pos.shape[0]} atoms saved to: {output}")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.ClickException(str(e))
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        raise click.ClickException(str(e))


@main.command(name="plot-lat")
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', default='lat_plot.png', help='Output plot file.')
def plot_lat(input_file: str, output: str) -> None:
    """Plot lattice parameters vs radial distance."""
    # Load the data
    atoms_pos = np.load(input_file)
    
    # Calculate lattice parameters
    lats, rs = get_lat_vs_r(np.copy(atoms_pos))
    
    # Create and save the plot
    plt.figure()
    plt.scatter(rs, lats, s=1, alpha=0.3)
    plt.title(f"Lattice parameters for {input_file}")
    plt.xlabel("Radial distance")
    plt.ylabel("Lattice parameter")
    plt.savefig(output, dpi=600)
    plt.close()
    
    click.echo(f"Lattice plot saved to: {output}")


@main.command(name="plot-dist")
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--r-cyl', default=3.0, help='Cylinder radius for filtering atoms.')
@click.option('--output', default='dist_plot.png', help='Output plot file.')
def plot_dist(input_file: str, r_cyl: float, output: str) -> None:
    """Plot distance distribution with cylindrical filtering."""
    # Load the data
    atoms_pos = np.load(input_file)
    
    # Filter atoms within cylinder
    X, Y, Z = atoms_pos[:, 0], atoms_pos[:, 1], atoms_pos[:, 2]
    cond = (np.sqrt(X**2 + Y**2) < r_cyl) & (Z > 0)
    X2 = X[cond]
    Y2 = Y[cond]
    Z2 = Z[cond]
    atoms_pos3 = np.array([X2, Y2, Z2]).T
    
    # Calculate lattice parameters
    lats, rs, ns, lat2s = get_lat_vs_r_extended(np.copy(atoms_pos))
    _, rs3, ns, lat3s = get_lat_vs_r_extended(np.copy(atoms_pos3))
    
    # Create and save the plot
    plt.figure()
    plt.scatter(rs, lat2s, s=1, alpha=0.3)
    plt.scatter(rs3, lat3s, s=8, alpha=0.8)
    plt.title(f"Distance distribution for {input_file} (r_cyl={r_cyl})")
    plt.ylim((2.75, 2.95))
    plt.xlabel("Radial distance")
    plt.ylabel("Lattice parameter")
    plt.savefig(output, dpi=600)
    plt.close()
    
    click.echo(f"Distance distribution plot saved to: {output}")


@main.command(name="plot-ion")
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--thickness', default=1.5, help='Measuring sphere thickness.')
@click.option('--output', default='ion_plot.png', help='Output plot file.')
def plot_ion(input_file: str, thickness: float, output: str) -> None:
    """Plot ion density distribution."""
    # Load the data
    atoms_pos = np.load(input_file)
    
    # Calculate ion density
    atoms_r = np.linalg.norm(atoms_pos, axis=1)
    h2 = thickness / 2.0  # Half thickness
    r_min = h2
    dr = h2 / 300.0  # Step size
    
    rad = []
    den = []
    for i in range(int(np.max(atoms_r) / dr) + 10):
        r_current = r_min + dr * i
        density = calculate_ion_density(atoms_r, r_current, h2)
        rad.append(r_current)
        den.append(density)
    
    # Create and save the plot
    plt.figure()
    plt.plot(rad, den)
    plt.title(f"Ion density for {input_file}, thickness h={thickness}")
    plt.xlabel("R, angs")
    plt.ylabel("n_i, ions/(angs^3)")
    plt.axhline(
        y=FCC_IDEAL_DENSITY,  # FCC, 4 atoms per unit cell volume
        color="g",
        linestyle="-",
        alpha=0.4,
    )
    plt.savefig(output, dpi=600)
    plt.close()
    
    click.echo(f"Ion density plot saved to: {output}")


@main.command(name="plot-rdf")
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--r-min', default=0.0, help='Minimum radius for RDF calculation.')
@click.option('--r-max', default=10.0, help='Maximum radius for RDF calculation.')
@click.option('--cutoff', default=3.5, help='Cutoff distance for RDF calculation.')
@click.option('--resolution', default=300, help='Resolution for RDF calculation.')
@click.option('--output', default='rdf_plot.png', help='Output plot file.')
def plot_rdf(input_file: str, r_min: float, r_max: float, cutoff: float, resolution: int, output: str) -> None:
    """Plot radial distribution function."""
    # Load the data
    atoms_pos = np.load(input_file)
    
    # Calculate radial distribution
    x, g = radial_distribution(atoms_pos, r_min, r_max, cutoff, resolution)
    
    # Create and save the plot
    plot_radial_distribution(x, g, output)
    
    click.echo(f"Radial distribution plot saved to: {output}")


if __name__ == "__main__":
    main()