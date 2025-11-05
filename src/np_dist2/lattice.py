import numpy as np


def get_lat_by_id(id: int, grid: np.ndarray):
    ref = grid[id]
    ref_arr = np.ones(grid.shape) * ref
    lat_arr = np.sort(np.linalg.norm(np.abs(grid - ref_arr), axis=1))
    assert np.allclose(lat_arr[0], [0, 0, 0])
    lat_arr = lat_arr[1:20]
    lat_min = np.min(lat_arr)
    lat2 = lat_arr[lat_arr < lat_min * 1.2]
    r = np.linalg.norm(ref)
    return r, np.mean(lat2), len(lat2)
