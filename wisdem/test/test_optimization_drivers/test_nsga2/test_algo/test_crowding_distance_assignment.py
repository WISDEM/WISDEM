from random import seed
import numpy as np
import matplotlib.pyplot as plt

import wisdem.optimization_drivers.nsga2.crowding_distance_assignment as cda


def test_crowding_distance_assignment_line_1D_equidistant():
    """
    test crowding distance assignment on a line in 1D space

    generate points sampled along a line in 1D, compute the crowding distance,
    then verify that points at the ends of the line have infinite crowding
    distance, and points in the middle have a finite, equivalent value
    """

    # configuration
    N_samples = 100
    x0 = -1.0
    x1 = 1.0
    dx = (x1 - x0) / (N_samples - 1)

    # generate equidistant points
    x_pts = np.linspace(x0, x1, N_samples)

    # stack vector
    vec = np.vstack([x_pts]).T

    # compute crowding distances
    crowding_distances = cda.crowding_distance_assignment(vec)

    # check the values
    assert np.isinf(crowding_distances[0])  # first entry should be inf
    assert np.isinf(crowding_distances[-1])  # last entry should be inf

    # everything else should be equal to each other
    # ... and the standard spacing
    assert np.allclose(crowding_distances[1:-1], dx)


def test_crowding_distance_assignment_line_2D_equidistant():
    """
    test crowding distance assignment on a line in 2D space

    generate points sampled along a line in 2D, compute the crowding distance,
    then verify that points at the ends of the line have infinite crowding
    distance, and points in the middle have a finite, equivalent value
    """

    # configuration
    N_samples = 100
    x0 = -1.0
    x1 = 1.0
    y0 = 0.0
    y1 = 2.0
    dx = (x1 - x0) / (N_samples - 1)
    dy = (y1 - y0) / (N_samples - 1)

    # generate equidistant points
    x_pts = np.linspace(x0, x1, N_samples)
    y_pts = np.linspace(y0, y1, N_samples)

    # stack vector
    vec = np.vstack([x_pts, y_pts]).T

    # compute crowding distances
    crowding_distances = cda.crowding_distance_assignment(vec)

    # check the values
    assert np.isinf(crowding_distances[0])  # first entry should be inf
    assert np.isinf(crowding_distances[-1])  # last entry should be inf

    # everything else should be equal to each other
    # ... and the sum of the two dimensions standard spacings
    assert np.allclose(crowding_distances[1:-1], dx + dy)


def test_crowding_distance_assignment_line_2D_exponential():

    # configuration
    N_samples = 100
    x0 = -1.0
    x1 = 1.0
    y0 = 0.0
    y1 = 2.0

    # generate points with exponential spacing
    xi_spacing = np.linspace(0.0, 1.0, N_samples)
    x_pts = xi_spacing**2
    y_pts = xi_spacing**3

    # stack vector
    vec = np.vstack([x_pts, y_pts]).T

    # compute crowding distances
    crowding_distances = cda.crowding_distance_assignment(vec)

    # check the values
    assert np.isinf(crowding_distances[0])  # first entry should be inf
    assert np.isinf(crowding_distances[-1])  # last entry should be inf

    # reference crowding distances
    CD_ref = (x_pts[2:] - x_pts[:-2]) / (x1 - x0) + (y_pts[2:] - y_pts[:-2]) / (y1 - y0)

    # assert that these are correct
    assert np.allclose(crowding_distances[1:-1], 2 * CD_ref)
