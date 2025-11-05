from np_dist2.lattice import get_lat_by_id
from np_dist2.grid_generator import gen_unit_grid, shift_grid, shift_grid_to_origin
import numpy as np


def test_get_lat_by_id_simple():
    grid = gen_unit_grid(1)
    r, lat, num = get_lat_by_id(0, grid)
    assert np.isclose(0, r)
    assert np.isclose(lat, 1)
    assert np.isclose(num, 3)


def find_vector_indices(arr, vec, rtol=1e-09, atol=1e-08):
    match = np.all(np.isclose(arr, vec, rtol=rtol, atol=atol), axis=1)
    print(match)
    return np.where(match)[0]


# def find_vector_indices(arr, vec):
#     return np.where(np.all(np.isclose(arr, vec), axis=1))[0][0]


def test_get_lat_by_id_simple2():
    grid = gen_unit_grid(2)
    id = find_vector_indices(grid, [1, 1, 1])
    r, lat, num = get_lat_by_id(id, grid)
    assert np.isclose(np.sqrt(3), r)
    assert np.isclose(lat, 1)
    assert np.isclose(num, 6)


def test_get_lat_by_id_simple3():
    grid = gen_unit_grid(2)
    id = find_vector_indices(grid, [1, 1, 1])
    grid2 = shift_grid_to_origin(grid)
    r, lat, num = get_lat_by_id(id, grid2)
    assert np.isclose(0, r)
    assert np.isclose(lat, 1)
    assert np.isclose(num, 6)


def test_get_lat_by_id_fcc():
    grid = gen_unit_grid(2)
    fc = gen_unit_grid(1)
    sub1 = shift_grid(fc, [0.5, 0.5, 0])
    sub2 = shift_grid(fc, [0.5, 0, 0.5])
    sub3 = shift_grid(fc, [0, 0.5, 0.5])
    fcc = np.vstack((grid, sub1, sub2, sub3))

    id = find_vector_indices(fcc, [1, 1, 1])
    r, lat, num = get_lat_by_id(id, fcc)
    assert np.isclose(np.sqrt(3), r)
    assert np.isclose(lat, 1 / np.sqrt(2))
    assert np.isclose(num, 12)


def test_get_lat_by_id_fcc2():
    grid = gen_unit_grid(2)
    fc = gen_unit_grid(1)
    sub1 = shift_grid(fc, [0.5, 0.5, 0])
    sub2 = shift_grid(fc, [0.5, 0, 0.5])
    sub3 = shift_grid(fc, [0, 0.5, 0.5])
    fcc = np.vstack((grid, sub1, sub2, sub3))
    id = find_vector_indices(fcc, [1, 1, 1])

    fcc = shift_grid_to_origin(fcc)

    _, lat, num = get_lat_by_id(id, fcc)
    assert np.isclose(lat, 1 / np.sqrt(2))
    assert np.isclose(num, 12)
