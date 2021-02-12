from __future__ import division

import numpy as np
from scipy.optimize import root, fsolve


def distfunc(dinp, ival=1):
    """
    function for smoothly distributing points along a curve
    according to a set of control points

    Adapted from NN Soerensen's distfunc.f

    parameters:
    -----------
    dinp: list
        list of control points with the form
        [[s0, ds0, n0], [s1, ds1, n1], ... [s<n>, ds<n>, n<n>]]
        where
            s<n> is the curve fraction at each control point,
            ds<n> is the cell size at each control point,
            n<n> is the cell count at each control point.

    returns:
    --------
    s: array
        distribed points
    """

    dinp = np.asarray(dinp)

    ndist = int(dinp[-1, 2])
    nn = len(dinp)
    fdist = np.zeros(ndist)
    dy = np.zeros(ndist)

    fdist[0] = 0.0
    s0 = dinp[0, 0]
    d0 = dinp[0, 1]
    n0 = np.int_(dinp[0, 2]) - 1

    for i in range(1, nn):
        s1 = dinp[i, 0]
        if i > 0 and d0 < 0.0:
            d0 = fdist[n0] - fdist[n0 - 1]
        d1 = dinp[i, 1]
        n1 = np.int_(dinp[i, 2]) - 1
        _len = s1 - s0
        delta1 = d0
        delta2 = d1
        if ival == 1:
            dy[n0 : n1 + 1] = tanhdist(delta1, delta2, _len, n0, n1)
        if ival == 2:
            dy[n0 : n1 + 1] = sinhdist(delta1, delta2, _len, n0, n1)

        for j in range(n0 + 1, n1 + 1):
            fdist[j] = fdist[n0] + dy[j]

        s0 = s1
        d0 = d1
        n0 = n1

    return fdist


def tanhdist(delta1=None, delta2=None, _len=None, i1=None, i2=None):

    if i2 == i1:
        return fdist
    delta1 = delta1 / _len
    delta2 = delta2 / _len
    ni = i2 - i1
    fdist = np.zeros(ni + 1)
    if delta1 <= 0.0 and 1.0 / delta2 < ni:
        delta1 = 1 / (ni ** 2 * delta2 * 1.02)
    else:
        if delta2 <= 0.0 and 1.0 / delta1 < ni:
            delta2 = 1 / (ni ** 2 * delta1 * 1.02)
    if delta1 > 0.0 and delta2 > 0.0:
        a = np.sqrt(delta2) / np.sqrt(delta1)
        b = 1.0 / (ni * np.sqrt(delta1 * delta2))
        if b >= 1.0:
            delta = transsinh(b)
            for i in range(ni + 1):
                ftmp = 0.5 * (1 + np.tanh(delta * (i / ni - 0.5)) / np.tanh(0.5 * delta))
                fdist[i] = ftmp / (a + (1 - a) * ftmp)
        else:
            delta = transtanh(b)
            for i in range(ni + 1):
                ftmp = 0.5 * (1 + np.sinh(delta * (i / ni - 0.5)) / np.sinh(0.5 * delta))
                fdist[i] = ftmp / (a + (1 - a) * ftmp)
    else:
        if delta1 > 0.0:
            b = 1.0 / (ni * delta1)
            delta = transsinh(b)
            for i in range(ni + 1):
                fdist[i] = 1.0 + np.tanh(0.5 * delta * (i / ni - 1.0)) / np.tanh(0.5 * delta)
        else:
            if delta2 > 0.0:
                b = 1.0 / (ni * delta2)
                delta = transsinh(b)
                for i in range(ni + 1):
                    fdist[i] = np.tanh(0.5 * delta * i / ni) / np.tanh(0.5 * delta)
            else:
                print("Error from tandist, no cell hight is given")
    for i in range(ni + 1):
        fdist[i] = fdist[i] * _len
    return fdist


def sinhdist(delta1=None, delta2=None, _len=None, i1=None, i2=None):

    if i2 == i1:
        return fdist
    delta1 = delta1 / _len
    delta2 = delta2 / _len
    ni = i2 - i1
    if (delta1 <= 0.0) and ((1 / delta2) < ni):
        delta1 = 1 / (ni ** 2 * delta2 * 1.02)
    else:
        if (delta2 <= 0.0) and (1 / delta1 < ni):
            delta2 = 1 / (ni ** 2 * delta1 * 1.02)
    if (delta1 > 0.0) and (delta2 > 0.0):
        a = np.sqrt(delta2) / np.sqrt(delta1)
        b = 1.0 / (ni * np.sqrt(delta1 * delta2))
        if b >= 1.0:
            delta = transsinh(b)
            for i in range(ni + 1):
                ftmp = 0.5 * (1 + np.tanh(delta * (i / ni - 0.5)) / np.tanh(0.5 * delta))
                fdist[i] = ftmp / (a + (1 - a) * ftmp)
        else:
            delta = transtanh(b)
            for i in range(ni + 1):
                ftmp = 0.5 * (1 + np.sinh(delta * (i / ni - 0.5)) / np.sinh(0.5 * delta))
                fdist[i] = ftmp / (a + (1 - a) * ftmp)
    else:
        if delta1 > 0.0:
            b = 1.0 / (ni * delta1)
            delta = transsinh(b)
            for i in range(ni + 1):
                fdist[i] = np.sinh(delta * i / ni) / np.sinh(delta)
        else:
            if delta2 > 0.0:
                b = 1.0 / (ni * delta2)
                delta = transsinh(b)
                for i in range(ni + 1):
                    fdist[i] = 1 - np.sinh(delta * (1 - i / ni)) / np.sinh(delta)
            else:
                disp_([char(" Error from tandist, no cell hight is given ")])
    for i in range(ni):
        fdist[i] = fdist[i] * _len
    return fdist


def transsinh(b):
    """
    solve the transcendental equation

    b=sinh(delta)/delta

    using scipy.optimize.root
    """
    res = root(_transsinh, 1.0, args=(b))

    return res["x"]


def _transsinh(delta, b):

    return delta / (np.sinh(delta) + 1.0 * 10 ** (-60)) - 1.0 / b


def transtanh(b):
    """
    solve the transcendental equation

    b=sinh(delta)/delta

    using scipy.optimize.root
    """
    res = root(_transtanh, 1.0, args=(b))

    return res["x"]


def _transtanh(delta, b):

    return delta / (np.tanh(0.5 * delta) + 1.0 * 10 ** (-60)) - 2.0 / b


def transsinh_(b=None):
    """
    unused fortran translation of Newton-Raphson solver for
    solving the transcendental equation

    b=sinh(delta)/delta
    """
    rlim = 1.0e-6
    nmax = 300
    delta = 1.0
    delta_old = delta
    for n in range(nmax):
        f = delta / (np.sinh(delta) + 1.0 * 10 ** (-60)) - 1.0 / b
        df = (np.sinh(delta) - delta * np.cosh(delta)) / np.sinh(delta) ** 2
        delta = delta - 0.5 * f / df
        if (n == nmax) and (np.abs((delta - delta_old) / delta_old) > rlim):
            disp_([char(" Convergence problem in transsinh dist. function !!! ")])
            disp_([char(" residual "), num2str_((delta - delta_old) / delta_old)])
        delta_old = delta
    return delta


def transtanh_(b=None):
    """
    unused fortran translation of Newton-Raphson solver for
    solving the transcendental equation

    b=tanh(delta)/delta
    """
    if b > 1.0:
        print(" Error form transtanh: Can not be gridded ")
        return delta
    rlim = 1.0e-6
    nmax = 100000
    delta = 1.0
    delta_old = delta
    for n in range(nmax):
        f = delta / (np.tanh(0.5 * delta) + 1.0 * 10 ** (-60)) - 2.0 / b
        df = (np.tanh(0.5 * delta) - delta * (1 - np.tanh(0.5 * delta) ** 2)) / (
            np.tanh(0.5 * delta) ** 2 + 1.0 * 10 ** (-60)
        )
        delta = delta - 0.5 * f / df
        res = np.abs((delta - delta_old) / delta_old)
        if res < rlim:
            return delta
        if (n == nmax) and (res > rlim):
            print(" Convergence problem in transtanh dist. function !!! ")
            print(" residual ", ((delta - delta_old) / delta_old))
            print("n= ", n, "  nmax=", nmax, "   delta=", delta, "   delta_old=", delta_old, "  rlim=", rlim)
        delta_old = delta
    return delta


if __name__ == "__main__":

    d = distfunc([[0, 0.0001, 1], [0.5, 0.001, 30], [1, 0.0001, 100]])
    import matplotlib.pylab as plt

    plt.plot(range(100), d, "-o")
    plt.show()
