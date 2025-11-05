import pytest
import numpy as np
from np_dist2.rotate3d import (
    rotate_vector,
    rotate_vector_euler,
    rotate_array,
)  # Import functions from previous block


def test_rotate_vector():
    """Test basic vector rotation"""
    vector = np.array([1, 0, 0])
    result = rotate_vector(vector, np.pi / 2, 0)
    expected = np.array([1, 0, 0])
    assert np.allclose(result, expected)

    # Test 90 degree rotation around Y axis
    result = rotate_vector(vector, 0, -np.pi / 2)
    expected = np.array([0, 0, 1])
    assert np.allclose(result, expected)

    vector = np.array([0, 1, 0])
    result = rotate_vector(vector, np.pi / 2, 0)
    expected = np.array([0, 0, 1])

    assert np.allclose(result, expected)


def test_rotate_vector_euler():
    """Test Euler angle rotation"""
    vector = np.array([1, 0, 0])
    angles = (np.pi / 2, -np.pi / 2, 0)
    result = rotate_vector_euler(vector, angles)
    expected = np.array([0, 0, 1])
    assert np.allclose(result, expected)

    vector = np.array([0, 1, 0])
    angles = (np.pi / 2, np.pi / 2, 0)
    result = rotate_vector_euler(vector, angles)
    expected = np.array([1, 0, 0])
    assert np.allclose(result, expected)

    vector = np.array([1, 0, 0])
    angles = (np.pi / 2, np.pi / 2, 0)
    result = rotate_vector_euler(vector, angles)
    expected = np.array([0, 0, -1])
    assert np.allclose(result, expected)


def test_rotate_array():
    angles = (np.pi / 2, np.pi / 2, 0)
    arr = np.array([[1, 0, 0], [0, 1, 0]])
    expected = np.array([[0, 0, -1], [1, 0, 0]])
    result = rotate_array(arr, angles)
    assert np.allclose(result, expected)


def test_vector_magnitude():
    """Test that rotation preserves vector magnitude"""
    vector = np.array([1, 2, 3])
    rotated = rotate_vector(vector, np.pi / 4, np.pi / 4)
    assert np.allclose(np.linalg.norm(vector), np.linalg.norm(rotated))


def test_invalid_input():
    """Test handling of invalid input"""
    with pytest.raises(ValueError):
        rotate_vector(None, np.pi / 2, np.pi / 2)
    with pytest.raises(ValueError):
        rotate_vector([1, 2], np.pi / 2, np.pi / 2)  # Wrong dimension


def test_rotate_zero_vector():
    """Test rotating a zero vector - should always result in zero vector"""
    zero_vector = np.array([0, 0, 0])
    result = rotate_vector(zero_vector, np.pi / 2, np.pi / 4)
    expected = np.array([0, 0, 0])
    assert np.allclose(result, expected)


def test_rotate_zero_angle():
    """Test rotating any vector by zero angles - should return original vector"""
    vector = np.array([1, 2, 3])
    result = rotate_vector(vector, 0, 0)
    assert np.allclose(result, vector)


def test_rotate_360_degrees():
    """Test rotating by 2π (360 degrees) - should return original vector"""
    vector = np.array([1, 2, 3])
    result = rotate_vector(vector, 2 * np.pi, 2 * np.pi)
    # Due to floating point precision, we allow a small tolerance
    assert np.allclose(result, vector, atol=1e-10)


def test_rotate_array_complex():
    """Test rotate_array with more complex cases"""
    # Test with multiple vectors and different angles
    angles = (np.pi / 4, np.pi / 6, np.pi / 3)
    arr = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    result = rotate_array(arr, angles)
    
    # Check that we still have the same number of vectors
    assert result.shape == arr.shape
    
    # Check that magnitudes are preserved
    original_magnitudes = np.linalg.norm(arr, axis=1)
    result_magnitudes = np.linalg.norm(result, axis=1)
    assert np.allclose(original_magnitudes, result_magnitudes)


# Example usage
if __name__ == "__main__":
    # Create a sample vector
    vector = np.array([1, 2, 3])

    # Rotate by 45 degrees around X and Y axes
    rotated = rotate_vector(vector, np.pi / 4, np.pi / 4)
    print(f"Original vector: {vector}")
    print(f"Rotated vector: {rotated}")

    # Verify magnitude preservation
    print(f"\nOriginal magnitude: {np.linalg.norm(vector):.2f}")
    print(f"Rotated magnitude: {np.linalg.norm(rotated):.2f}")