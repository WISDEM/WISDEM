_pBEAM.Beam.axialStrain
=======================

.. method:: Beam.axialStrain(n, x, y, z)

    Computes the axial strain along the structure at given locations.

    Parameters
    ----------
    n : int
        number of locations to evaluate strain at
    x : ndarray (m)
        x coordinates of points to evaluate strain at (distance from elastic center)
    y : ndarray (m)
        y coordinates of points to evaluate strain at (distance from elastic center)
    z : ndarray (m)
        z coordinate of point to evaluate strain at (distance from elastic center)

    Returns
    -------
    epsilon : ndarray (N/m**2)
        axial strain at given points due to the axial loads and bi-directional bending

