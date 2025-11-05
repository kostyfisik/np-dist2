"""Integration test to verify that all components work together."""

import subprocess
import sys
from pathlib import Path


def test_library_imports():
    """Test that all library modules can be imported."""
    try:
        from np_dist2 import grid_generator, lattice, rotate3d
        from np_dist2.grid_generator import gen_unit_grid
        from np_dist2.lattice import get_lat_by_id
        print("✓ All library modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Library import failed: {e}")
        return False


def test_library_functionality():
    """Test basic library functionality."""
    try:
        from np_dist2.grid_generator import gen_unit_grid
        from np_dist2.lattice import get_lat_by_id
        
        # Generate a simple grid
        grid = gen_unit_grid(1)
        assert len(grid) == 8, f"Expected 8 points, got {len(grid)}"
        
        # Analyze lattice properties
        r, lat, num = get_lat_by_id(0, grid)
        assert r == 0.0, f"Expected r=0.0, got {r}"
        assert lat > 0, f"Expected lat>0, got {lat}"
        assert num > 0, f"Expected num>0, got {num}"
        
        print("✓ Library functionality test passed")
        return True
    except Exception as e:
        print(f"✗ Library functionality test failed: {e}")
        return False


def test_cli_help():
    """Test that CLI help works."""
    try:
        # Use the Python from our virtual environment
        venv_python = Path(".venv/bin/python")
        if not venv_python.exists():
            venv_python = "python"
            
        result = subprocess.run(
            [str(venv_python), "-m", "np_dist2", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✓ CLI help command works")
            return True
        else:
            print(f"✗ CLI help command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ CLI help test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("Running np-dist2 integration tests...")
    print("=" * 50)
    
    tests = [
        test_library_imports,
        test_library_functionality,
        test_cli_help,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 50)
    print(f"Integration tests: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("❌ Some integration tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())