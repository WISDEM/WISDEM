import numpy as np
import matplotlib.pyplot as plt

import wisdem.optimization_drivers.nsga2.fast_nondom_sort as fns


def test_fast_nondom_sort_unit_sphere():
    """
    unit sphere front domination test

    generate points sampled within a unit sphere, compute the Pareto fronts,
    then verify that points in a given front are not dominated by any point in
    the current or any subsequent front by a brute force calculation

    Returns
    -------
    None
    """

    # configuration
    dim = 3
    N_samples = 100
    seed_rng = 1234

    # get a bunch of random points on the surface
    np.random.seed(seed_rng)
    vec = np.random.normal(0, 1, (N_samples, dim))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    vec *= np.random.random((N_samples, 1))  # very distance from center

    # convert for working
    vec2list = [tuple(v) for v in vec]
    fronts = fns.fast_nondom_sort(vec2list)

    # a function to make sure a point vtx is not dominated by any vertex in a front in or after idx_front
    def is_dominated(idx_vtx, idx_front):
        for idx_f in range(idx_front, len(fronts)):
            for idx_vtx2 in fronts[idx_f]:
                if np.allclose(idx_vtx, idx_vtx2):
                    continue
                # check if vtx is dominated by vtx2
                if all(np.array(vec2list[idx_vtx]) >= np.array(vec2list[idx_vtx2])):
                    return True
        return False

    # loop over front, vertices within the front
    for idx_front, front in enumerate(fronts):
        for idx_vtx in front:
            # make sure that this point is not dominated by any point in this
            # or subsequent fronts
            assert not is_dominated(idx_vtx, idx_front)


def test_fast_nondom_sort_cube():
    """
    make sure that some fronts generated from a cube match gold values

    test that non-domination fronts match a set of points computed by hand
    using geometric patterns

    Returns
    -------
    None
    """

    # configuration
    dim = 3
    N_side = 5
    x0 = 2.0
    x1 = 4.0
    y0 = 1.0
    y1 = 3.0
    z0 = -2.0
    z1 = 0.0

    # get a cube
    XYZ_grid = np.vstack(
        [
            v.flat
            for v in np.meshgrid(
                *[
                    np.linspace(x0, x1, N_side),
                    np.linspace(y0, y1, N_side),
                    np.linspace(z0, z1, N_side),
                ]
            )
        ]
    ).T
    print(XYZ_grid.shape)

    # get the first front
    fronts = fns.fast_nondom_sort([tuple(v) for v in XYZ_grid])

    # reference values for comparison, based on patterns
    assert dim == 3, "reference values only work for 3D"
    fronts_ref = [
        [0],  # first front: only total minimizer dominates all
        [1, N_side, N_side**2],  # three points, based on patterns in meshgrid
        [
            2,
            N_side + 1,
            2 * N_side,
            (N_side**2 + 1),
            N_side * (N_side + 1),
            2 * N_side**2,
        ],  # six points, based on patterns in meshgrid
    ]

    # verify that the computed fronts match reference values
    for idx_front, front_ref in enumerate(fronts_ref):
        assert np.allclose(fronts[idx_front], front_ref)
