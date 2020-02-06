_pBEAM.Beam.naturalFrequenciesAndEigenvectors
=============================================

.. method:: Beam.naturalFrequenciesAndEigenvectors(n)

    Computes first n natural frequncies and associated eigenvectors of beam

    Parameters
    ----------
    n : int
        number of natural frequencies/eigenvectors to return.  if n exceeds the total
        DOF of the structure then all natural frequencies are returned

    Returns
    -------
    freq : ndarray (Hz)
        natural frequncies from lowest to highest.  Size n except for
        case noted above.
    vector : ndarray(ndarray)
        vector[i] corresponds to freq[i]