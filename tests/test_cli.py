"""Test cases for the CLI commands."""

import os
import pytest
import numpy as np
from click.testing import CliRunner
from pathlib import Path

from np_dist2.__main__ import main


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_cli_help(runner: CliRunner) -> None:
    """Test that the main CLI help works."""
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert "Np Dist2 - A toolkit for nanoparticle simulation analysis" in result.output


def test_cli_short_help(runner: CliRunner) -> None:
    """Test that the main CLI short help works."""
    result = runner.invoke(main, ['-h'])
    assert result.exit_code == 0
    assert "Np Dist2 - A toolkit for nanoparticle simulation analysis" in result.output


def test_time_avg_help(runner: CliRunner) -> None:
    """Test that the time-avg command help works."""
    result = runner.invoke(main, ['time-avg', '--help'])
    assert result.exit_code == 0
    assert "Averages atomic positions over simulation timesteps" in result.output


def test_time_avg_short_help(runner: CliRunner) -> None:
    """Test that the time-avg command short help works."""
    result = runner.invoke(main, ['time-avg', '-h'])
    assert result.exit_code == 0
    assert "Averages atomic positions over simulation timesteps" in result.output


def test_plot_lat_help(runner: CliRunner) -> None:
    """Test that the plot-lat command help works."""
    result = runner.invoke(main, ['plot-lat', '--help'])
    assert result.exit_code == 0
    assert "Plot lattice parameters vs radial distance" in result.output


def test_plot_lat_short_help(runner: CliRunner) -> None:
    """Test that the plot-lat command short help works."""
    result = runner.invoke(main, ['plot-lat', '-h'])
    assert result.exit_code == 0
    assert "Plot lattice parameters vs radial distance" in result.output


def test_plot_dist_help(runner: CliRunner) -> None:
    """Test that the plot-dist command help works."""
    result = runner.invoke(main, ['plot-dist', '--help'])
    assert result.exit_code == 0
    assert "Plot distance distribution with cylindrical filtering" in result.output


def test_plot_ion_help(runner: CliRunner) -> None:
    """Test that the plot-ion command help works."""
    result = runner.invoke(main, ['plot-ion', '--help'])
    assert result.exit_code == 0
    assert "Plot ion density distribution" in result.output


def test_plot_rdf_help(runner: CliRunner) -> None:
    """Test that the plot-rdf command help works."""
    result = runner.invoke(main, ['plot-rdf', '--help'])
    assert result.exit_code == 0
    assert "Plot radial distribution function" in result.output


def test_plot_lat_success(runner: CliRunner, tmp_path: Path) -> None:
    """Test that plot-lat command runs successfully with valid input."""
    # Create a dummy numpy file in the temp path
    input_file = tmp_path / "test_atoms.npy"
    np.save(input_file, np.random.rand(10, 3))
    output_file = tmp_path / "lat_plot.png"

    result = runner.invoke(main, ['plot-lat', str(input_file), '--output', str(output_file)])

    assert result.exit_code == 0
    assert "Lattice plot saved to:" in result.output


def test_plot_lat_creates_output(runner: CliRunner, tmp_path: Path) -> None:
    """Test that plot-lat command creates the specified output file."""
    # Create a dummy numpy file in the temp path
    input_file = tmp_path / "test_atoms.npy"
    np.save(input_file, np.random.rand(10, 3))
    output_file = tmp_path / "lat_plot.png"

    result = runner.invoke(main, ['plot-lat', str(input_file), '--output', str(output_file)])

    assert result.exit_code == 0
    assert output_file.exists()


def test_plot_lat_file_not_found(runner: CliRunner) -> None:
    """Test that plot-lat command handles non-existent input file."""
    result = runner.invoke(main, ['plot-lat', 'non_existent_file.npy'])
    
    # Click should handle this with a non-zero exit code
    assert result.exit_code != 0
    # Should contain an error message
    assert "Error" in result.output or "not found" in result.output


def test_plot_dist_success(runner: CliRunner, tmp_path: Path) -> None:
    """Test that plot-dist command runs successfully with valid input."""
    # Create a dummy numpy file in the temp path
    input_file = tmp_path / "test_atoms.npy"
    np.save(input_file, np.random.rand(10, 3))
    output_file = tmp_path / "dist_plot.png"

    result = runner.invoke(main, ['plot-dist', str(input_file), '--output', str(output_file)])

    assert result.exit_code == 0
    assert "Distance distribution plot saved to:" in result.output


def test_plot_dist_with_options(runner: CliRunner, tmp_path: Path) -> None:
    """Test that plot-dist command works with custom options."""
    # Create a dummy numpy file in the temp path
    input_file = tmp_path / "test_atoms.npy"
    np.save(input_file, np.random.rand(10, 3))
    output_file = tmp_path / "dist_plot.png"

    result = runner.invoke(main, ['plot-dist', str(input_file), '--r-cyl', '2.5', '--output', str(output_file)])

    assert result.exit_code == 0
    assert "Distance distribution plot saved to:" in result.output


def test_plot_ion_success(runner: CliRunner, tmp_path: Path) -> None:
    """Test that plot-ion command runs successfully with valid input."""
    # Create a dummy numpy file in the temp path
    input_file = tmp_path / "test_atoms.npy"
    np.save(input_file, np.random.rand(10, 3))
    output_file = tmp_path / "ion_plot.png"

    result = runner.invoke(main, ['plot-ion', str(input_file), '--output', str(output_file)])

    assert result.exit_code == 0
    assert "Ion density plot saved to:" in result.output


def test_plot_rdf_success(runner: CliRunner, tmp_path: Path) -> None:
    """Test that plot-rdf command runs successfully with valid input."""
    # Create a dummy numpy file in the temp path
    input_file = tmp_path / "test_atoms.npy"
    np.save(input_file, np.random.rand(10, 3))
    output_file = tmp_path / "rdf_plot.png"

    result = runner.invoke(main, ['plot-rdf', str(input_file), '--output', str(output_file)])

    assert result.exit_code == 0
    assert "Radial distribution plot saved to:" in result.output


def test_time_avg_file_not_found(runner: CliRunner) -> None:
    """Test that time-avg command handles non-existent directory."""
    result = runner.invoke(main, ['time-avg', 'non_existent_directory'])
    
    # Click should handle this with a non-zero exit code
    assert result.exit_code != 0
    # Should contain an error message
    assert "Error" in result.output or "not found" in result.output