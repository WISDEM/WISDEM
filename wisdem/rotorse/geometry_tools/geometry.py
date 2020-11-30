from __future__ import print_function
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import pchip, Akima1DInterpolator, PchipInterpolator

from wisdem.rotorse.geometry_tools.geom_tools import curvature
from wisdem.rotorse.geometry_tools.cubicspline import NaturalCubicSpline
from wisdem.rotorse.geometry_tools.distfunc import distfunc
from wisdem.commonse.utilities import arc_length


class Curve(object):
    def __init__(self, points=None):
        super(Curve, self).__init__()

        self.length = 0.0  # Total curve length
        self.s = np.array([])  # Curve accumulated curve length
        self.points = np.array([])  # coordinates of curve
        self.ni = 0  # Number of points

        if points is not None:
            self.initialize(points)

    def initialize(self, points):

        self.points = points
        self.ni = points.shape[0]

        self._compute_s()
        self._compute_dp()
        self._build_splines()

    def _compute_s(self):
        """
        compute normalized curve length
        """
        s = arc_length(self.points)
        self.length = s[-1]
        self.ds = np.diff(s)
        self.s = s / s[-1]

    def _compute_dp(self):
        """compute the unit direction vectors along the curve"""

        t1 = np.gradient(self.points[:, :])[0]
        self.dp = np.array([t1[i, :] / np.linalg.norm(t1[i, :]) for i in range(np.shape(t1)[0])])

    def _build_splines(self):

        self._splines = []

        for j in range(np.shape(self.points)[1]):
            self._splines.append(PchipInterpolator(self.s, self.points[:, j]))

    def redistribute(self, dist=None, s=None):

        if dist is not None:
            self.s = distfunc(dist)
        else:
            self.s = s

        self.ni = np.shape(self.s)[0]
        points = np.zeros((self.ni, np.shape(self.points)[1]))
        for i in range(points.shape[1]):
            points[:, i] = self._splines[i](self.s)

        self.initialize(points)

    def interp_s(self, s):
        """
        interpolate (x,y) at some curve fraction s
        """

        p = np.zeros(np.shape(self.points)[1])
        for i in range(np.shape(self.points)[1]):
            p[i] = self._splines[i](s)

        return p


class AirfoilShape(Curve):
    """
    Base class for airfoil shapes.

    The class automatically computes the LE and TE
    and can redistribute the points smoothly along the surface.
    Points along the surface need to be defined starting at the
    TE pressure side ending at the TE suction side.
    """

    def initialize(self, points):

        self.LE = np.array([])  # Leading edge coordinates
        self.TE = np.array([])  # Trailing edge coordinates
        self.sLE = 0.0  # Leading edge curve fraction
        self.chord = 0.0  # chord length

        super(AirfoilShape, self).initialize(points)
        self.computeLETE()

    def computeLETE(self):
        """
        computes the leading and trailing edge of the airfoil.

        TE is computed as the mid-point between lower and upper TE points
        LE is computed as the point with maximum distance from the TE.
        """

        self.TE = np.array([np.average(self.points[[0, -1], 0]), np.average(self.points[[0, -1], 1])])

        res = minimize(self._sdist, (0.5), method="SLSQP", bounds=[(0, 1)])
        self.sLE = res["x"][0]
        xLE = self._splines[0](self.sLE)
        yLE = self._splines[1](self.sLE)
        self.LE = np.array([xLE, yLE])
        self.curvLE = NaturalCubicSpline(self.s, curvature(self.points))(self.sLE)
        self.chord = np.linalg.norm(self.LE - self.TE)

    def _sdist(self, s):

        x = self._splines[0](s)
        y = self._splines[1](s)
        return -(((x - self.TE[0]) ** 2 + (y - self.TE[1]) ** 2) ** 0.5)

    def leading_edge_dist(self, ni):
        """ function that returns a suitable cell size based on airfoil LE curvature """

        min_ds1 = 1.0 / ni * 0.1
        max_ds1 = 1.0 / ni * 0.5

        ds1 = max((min_ds1 - max_ds1) / 30.0 * abs(self.curvLE) + max_ds1, min_ds1)

        return ds1

    def redistribute(self, ni, even=False, dist=None, dLE=False, dTE=-1.0, s=None):
        """
        redistribute the points on the airfoil using fusedwind.lib.distfunc

        Parameters
        ----------
        ni : int
            total number of points on airfoil
        even : bool
            flag for getting an even distribution of points
        dist : list
            optional list of control points with the form
            [[s0, ds0, n0], [s1, ds1, n1], ... [s<n>, ds<n>, n<n>]]
            where\n
            s<n> is the normalized curve fraction at each control point,\n
            ds<n> is the normalized cell size at each control point,\n
            n<n> is the cell count at each control point.
        dLE : bool
            optional flag for automatically calculating a suitable leading edge cell
            size based on the local curvature
        dTE : float
            optional trailing edge cell size. If set to -1 the cell size will increase
            from the LE to TE according to the tanh distribution function used
            in distfunc
        """

        if even:
            dist = [
                [0, 1.0 / np.float(ni - 1), 1],
                [self.sLE, 1.0 / np.float(ni - 1), int(ni * self.sLE)],
                [1, 1.0 / np.float(ni - 1), ni],
            ]
        elif dLE:
            dist = [[0.0, dTE, 1], [self.sLE, self.leading_edge_dist(ni), ni / 2], [1.0, dTE, ni]]

        super(AirfoilShape, self).redistribute(dist, s=s)

        return self

    def redistribute_chordwise(self, dist):
        """
        redistribute the airfoil according to a chordwise distribution
        """

        # self.redistribute(self.ni, even=True)
        iLE = np.argmin(self.points[:, 0])
        ni = np.shape(dist)[0]
        dist = np.asarray(dist)
        points = np.zeros((np.shape(dist)[0] * 2 - 1, np.shape(self.points)[1]))

        # interpolate pressure side coordinates
        yps = NaturalCubicSpline(self.points[: iLE + 1, 0][::-1], self.points[: iLE + 1, 1][::-1])
        ps = yps(dist)
        # interpolate suction side coordinates
        yss = NaturalCubicSpline(self.points[iLE:, 0], self.points[iLE:, 1])
        ss = yss(dist)
        points[: ni - 1, 0] = dist[::-1][:-1]
        points[ni - 1 :, 0] = dist
        points[:, 1] = np.append(ps[::-1][:-1], ss, axis=0)
        return AirfoilShape(points)

    def s_to_11(self, s):
        """
        Transform the s coordinates from AirfoilShape format:

        * s=0 at TE pressure side (lower surface)
        * s=1 at TE suction side (upper surface)

        to the s coordinates from the input definition:

        * s=0 at LE
        * s=1 at TE suction side (upper surface)
        * s=-1 at TE pressure side (lower surface)
        """

        if s > self.sLE:
            return (s - self.sLE) / (1.0 - self.sLE)
        else:
            return -1.0 + s / self.sLE

    def s_to_01(self, s):
        """
        Transform the s coordinates from the input definition:

        * s=0 at LE
        * s=1 at TE suction side (upper surface)
        * s=-1 at TE pressure side (lower surface)

        to the backend defintion compatible with AirfoilShape():

        * s=0 at TE pressure side (lower surface)
        * s=1 at TE suction side (upper surface)
        """
        if s >= 0.0:
            return s * (1.0 - self.sLE) + self.sLE
        else:
            return (1.0 + s) * self.sLE

    def gurneyflap(self, gf_height, gf_length_factor):
        """add a Gurney flap shaped using a tanh function"""

        if gf_height == 0.0:
            return
        # if the length is not specified it is set to 3 x gf_height
        gf_length = gf_length_factor * gf_height

        # identify starting point of the gf along the chord
        x_gf = 1.0 - gf_length
        id1 = (np.abs(x_gf - self.points[0 : self.ni / 2, 0])).argmin() + 1
        s = np.linspace(x_gf, self.points[0, 0], 100)
        smax = s[-1] - s[0]
        h = np.zeros(100)
        for i in range(100):
            h[i] = (min(0.90 * gf_height, gf_height * (-np.tanh((s[i] - s[0]) / smax * 3) + 1.0))) / 0.90
        h = h[::-1]
        self.gfs = s
        self.gfh = h

        # add the gf shape to the airfoil
        points = self.points.copy()
        for i in range(0, id1):
            points[i, 1] = points[i, 1] - np.interp(points[i, 0], s, h)

        return AirfoilShape(points)

    def open_trailing_edge(self, t):
        """
        add thickness to airfoil
        """

        t0 = np.abs(self.points[-1, 1] - self.points[0, 1])
        dt = (t - t0) / 2.0
        print("dt", dt)
        # linearly add thickness from LE to TE
        iLE = np.argmin(self.points[:, 0])
        xLE = self.points[iLE, 0]
        tlin = np.array([np.linspace(xLE, self.TE[0], 100), np.linspace(0.0, dt, 100)]).T

        tspline = NaturalCubicSpline(tlin[:, 0], tlin[:, 1])

        ys = tspline(self.points[iLE:, 0]) + self.points[iLE:, 1]
        yp = -tspline(self.points[:iLE, 0][::-1])[::-1] + self.points[:iLE, 1]

        self.points[iLE:, 1] = ys
        self.points[:iLE, 1] = yp
        self.initialize(self.points)


def remap2grid(x_ref, y_ref, x, spline=PchipInterpolator, axis=-1):

    try:
        if axis != -1:
            spline_y = spline(x_ref, y_ref, axis=axis)
        else:
            spline_y = spline(x_ref, y_ref)
    except:
        x_ref = np.flip(x_ref, axis=0)
        y_ref = np.flip(y_ref, axis=0)
        if axis != -1:
            spline_y = spline(x_ref, y_ref, axis=axis)
        else:
            spline_y = spline(x_ref, y_ref)

    # error handling for x[-1] - x_ref[-1] > 0 and x[-1]~x_ref[-1]
    try:
        _ = iter(x)
        if x[-1] > max(x_ref) and np.isclose(x[-1], x_ref[-1]):
            x[-1] = x_ref[-1]
    except:
        if np.isclose(x, 0.0):
            x = 0.0
        if x > max(x_ref) and np.isclose(x, x_ref[-1]):
            x = x_ref[-1]

    y_out = spline_y(x)

    np.place(y_out, y_out < np.min(y_ref), np.min(y_ref))
    np.place(y_out, y_out > np.max(y_ref), np.max(y_ref))

    return y_out


def trailing_edge_smoothing(data):
    # correction to trailing edge shape for interpolated airfoils that smooths out unrealistic geometric errors
    # often brought about when transitioning between round, flatback, or sharp trailing edges

    # correct for self cross of TE (rare interpolation error)
    if data[-1, 1] < data[0, 1]:
        temp = data[0, 1]
        data[0, 1] = data[-1, 1]
        data[-1, 1] = temp

    # Find indices on Suction and Pressure side for last 85-95% and 95-100% chordwise
    idx_85_95 = [i_x for i_x, xi in enumerate(data[:, 0]) if xi > 0.85 and xi < 0.95]
    idx_95_100 = [i_x for i_x, xi in enumerate(data[:, 0]) if xi > 0.95 and xi < 1.0]

    idx_85_95_break = [i_idx for i_idx, d_idx in enumerate(np.diff(idx_85_95)) if d_idx > 1][0] + 1
    idx_85_95_SS = idx_85_95[:idx_85_95_break]
    idx_85_95_PS = idx_85_95[idx_85_95_break:]

    idx_95_100_break = [i_idx for i_idx, d_idx in enumerate(np.diff(idx_95_100)) if d_idx > 1][0] + 1
    idx_95_100_SS = idx_95_100[:idx_95_100_break]
    idx_95_100_PS = idx_95_100[idx_95_100_break:]

    # Interpolate the last 5% to the trailing edge
    idx_in_PS = idx_85_95_PS + [-1]
    x_corrected_PS = data[idx_95_100_PS, 0]
    y_corrected_PS = remap2grid(data[idx_in_PS, 0], data[idx_in_PS, 1], x_corrected_PS)

    idx_in_SS = [0] + idx_85_95_SS
    x_corrected_SS = data[idx_95_100_SS, 0]
    y_corrected_SS = remap2grid(data[idx_in_SS, 0], data[idx_in_SS, 1], x_corrected_SS)

    # Overwrite profile with corrected TE
    data[idx_95_100_SS, 1] = y_corrected_SS
    data[idx_95_100_PS, 1] = y_corrected_PS

    return data
