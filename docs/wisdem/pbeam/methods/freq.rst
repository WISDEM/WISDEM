_pBEAM.Beam.naturalFrequencies
==============================

.. method:: Beam.naturalFrequencies(n)

    Computes first n natural frequncies of beam

    Parameters
    ----------
    n : int
        number of natural frequencies to return.  if n exceeds the total
        DOF of the structure then all natural frequencies are returned

    Returns
    -------
    freq : ndarray (Hz)
        natural frequncies from lowest to highest.  Size n except for
        case noted above.