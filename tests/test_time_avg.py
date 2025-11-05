"""Tests for the time-avg functionality."""

import numpy as np
import pytest
from click.testing import CliRunner
from pathlib import Path
from np_dist2.__main__ import main
from np_dist2.analysis import average_timesteps


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_average_timesteps(tmp_path: Path) -> None:
    """Unit test for the averaging logic."""
    # Create mock dump files
    for i in range(3):
        content = f"""ITEM: TIMESTEP
{i*1000}
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 {i+1}.0 {i+1}.0 {i+1}.0
2 1 {i+2}.0 {i+2}.0 {i+2}.0
"""
        (tmp_path / f"dump_superficie.{i*1000}").write_text(content)

    avg_pos = average_timesteps(str(tmp_path), 3)
    
    expected = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    np.testing.assert_allclose(avg_pos, expected)


def test_time_avg_cli(runner: CliRunner, tmp_path: Path) -> None:
    """End-to-end test for the time-avg CLI command."""
    # Create mock dump files
    for i in range(3):
        content = f"""ITEM: TIMESTEP
{i*1000}
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 {i+1}.0 {i+1}.0 {i+1}.0
2 1 {i+2}.0 {i+2}.0 {i+2}.0
"""
        (tmp_path / f"dump_superficie.{i*1000}").write_text(content)

    output_file = tmp_path / "avg.npy"
    result = runner.invoke(main, ['time-avg', str(tmp_path), '--num', '3', '--output', str(output_file)])

    assert result.exit_code == 0
    assert output_file.exists()
    
    # Verify the content of the output file
    avg_data = np.load(output_file)
    expected = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    np.testing.assert_allclose(avg_data, expected)


def test_time_avg_cli_no_files(runner: CliRunner, tmp_path: Path) -> None:
    """Test that time-avg CLI handles directory with no dump files."""
    output_file = tmp_path / "avg.npy"
    result = runner.invoke(main, ['time-avg', str(tmp_path), '--num', '3', '--output', str(output_file)])
    
    # Print the result for debugging
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    
    # The command should fail when there are no files
    assert result.exit_code != 0
    assert "Error" in result.output or "No dump files found" in result.output
    assert not output_file.exists()