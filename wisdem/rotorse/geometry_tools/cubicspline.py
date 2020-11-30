import numpy as np
from scipy.linalg import solve_banded
from wisdem.commonse.utilities import _checkIfFloat


class NaturalCubicSpline(object):
    """
    class implementation of utilities.cubic_with_deriv
    """

    def __init__(self, xp, yp):

        if np.any(np.diff(xp) < 0):
            raise TypeError("xp must be in ascending order")

        # n = len(x)
        self.m = len(xp)

        xk = xp[1:-1]
        yk = yp[1:-1]
        xkp = xp[2:]
        ykp = yp[2:]
        xkm = xp[:-2]
        ykm = yp[:-2]

        b = (ykp - yk) / (xkp - xk) - (yk - ykm) / (xk - xkm)
        l = (xk - xkm) / 6.0
        d = (xkp - xkm) / 3.0
        u = (xkp - xk) / 6.0
        # u[0] = 0.0  # non-existent entries
        # l[-1] = 0.0

        # solve for second derivatives
        fpp = solve_banded((1, 1), np.array([u, d, l]), b)
        self.fpp = np.concatenate([[0.0], fpp, [0.0]])  # natural spline
        self.xp = xp
        self.yp = yp

    def __call__(self, x, deriv=False):

        x, n = _checkIfFloat(x)
        y = np.zeros(n)
        dydx = np.zeros(n)
        dydxp = np.zeros((n, self.m))
        dydyp = np.zeros((n, self.m))

        # find location in vector
        for i in range(n):
            if x[i] < self.xp[0]:
                j = 0
            elif x[i] > self.xp[-1]:
                j = self.m - 2
            else:
                for j in range(self.m - 1):
                    if self.xp[j + 1] > x[i]:
                        break
            x1 = self.xp[j]
            y1 = self.yp[j]
            x2 = self.xp[j + 1]
            y2 = self.yp[j + 1]

            A = (x2 - x[i]) / (x2 - x1)
            B = 1 - A
            C = 1.0 / 6 * (A ** 3 - A) * (x2 - x1) ** 2
            D = 1.0 / 6 * (B ** 3 - B) * (x2 - x1) ** 2

            y[i] = A * y1 + B * y2 + C * self.fpp[j] + D * self.fpp[j + 1]
            dAdx = -1.0 / (x2 - x1)
            dBdx = -dAdx
            dCdx = 1.0 / 6 * (3 * A ** 2 - 1) * dAdx * (x2 - x1) ** 2
            dDdx = 1.0 / 6 * (3 * B ** 2 - 1) * dBdx * (x2 - x1) ** 2
            dydx[i] = dAdx * y1 + dBdx * y2 + dCdx * self.fpp[j] + dDdx * self.fpp[j + 1]

        if n == 1:
            y = y[0]
            dydx = dydx[0]

        if deriv:
            return y, dydx

        else:
            return y
