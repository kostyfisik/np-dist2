import numpy as np


def rotate_vector(vector, angle_x, angle_y):
    """
    Rotate a 3D vector by two angles:
    - angle_x: rotation around x-axis in radians
    - angle_y: rotation around y-axis in radians

    Args:
        vector (list/array): Initial 3D vector [x, y, z]
        angle_x (float): Rotation angle around x-axis in radians
        angle_y (float): Rotation angle around y-axis in radians

    Returns:
        array: Rotated vector
    """
    vector = np.array(vector)

    # Rotation matrix around x-axis
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )

    # Rotation matrix around y-axis
    Ry = np.array(
        [
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ]
    )

    # Apply rotations sequentially
    rotated_vector = Rx @ vector
    rotated_vector = Ry @ rotated_vector

    return rotated_vector


def rotate_vector_euler(vector, angles):
    """
    Rotate a 3D vector by Euler angles.

    Args:
        vector (list/array): Initial 3D vector [x, y, z]
        angles (tuple/list): Three Euler angles in radians

    Returns:
        array: Rotated vector
    """
    alpha, beta, _ = angles
    return rotate_vector(vector, alpha, beta)


def rotate_array(arr, angles):

    return np.apply_along_axis(rotate_vector_euler, 1, arr, angles)
