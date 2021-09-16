#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Andrew Ning on 2013-11-04.
Copyright (c) NREL. All rights reserved.
"""

import unittest
from io import StringIO

import numpy as np
from wisdem.pyframe3dd import Frame, Options, NodeData, ElementData, ReactionData, StaticLoadCase


class FrameTestEXA(unittest.TestCase):
    def setUp(self):

        # nodes
        node = np.arange(1, 13)
        x = np.array([0.0, 120.0, 240.0, 360.0, 480.0, 600.0, 720.0, 120.0, 240.0, 360.0, 480.0, 600.0])
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 120.0, 120.0, 120.0, 120.0, 120.0])
        z = np.zeros(12)
        r = np.zeros(12)
        nodes = NodeData(node, x, y, z, r)

        # reactions
        node = np.arange(1, 13)
        Kx = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        Ky = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        Kz = np.ones(12)
        Ktx = np.ones(12)
        Kty = np.ones(12)
        Ktz = np.zeros(12)
        rigid = 1
        reactions = ReactionData(node, Kx, Ky, Kz, Ktx, Kty, Ktz, rigid)

        # elements
        EL = np.arange(1, 22)
        N1 = np.array([1, 2, 3, 4, 5, 6, 1, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11])
        N2 = np.array([2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 9, 10, 11, 11, 11, 12, 12, 9, 10, 11, 12])
        Ax = 10.0 * np.ones(21)
        Asy = 1.0 * np.ones(21)
        Asz = 1.0 * np.ones(21)
        Jx = 1.0 * np.ones(21)
        Iy = 1.0 * np.ones(21)
        Iz = 0.01 * np.ones(21)
        E = 29000 * np.ones(21)
        G = 11500 * np.ones(21)
        roll = np.zeros(21)
        density = 7.33e-7 * np.ones(21)
        elements = ElementData(EL, N1, N2, Ax, Asy, Asz, Jx, Iy, Iz, E, G, roll, density)

        # parameters
        shear = False  # 1: include shear deformation
        geom = False  # 1: include geometric stiffness
        dx = 10.0  # x-axis increment for internal forces
        options = Options(shear, geom, dx)

        frame = Frame(nodes, reactions, elements, options)

        # load cases 1
        gx = 0.0
        gy = -386.4
        gz = 0.0

        load = StaticLoadCase(gx, gy, gz)

        nF = np.array([2, 3, 4, 5, 6])
        Fx = np.zeros(5)
        Fy = np.array([-10.0, -20.0, -20.0, -10.0, -20.0])
        Fz = np.zeros(5)
        Mxx = np.zeros(5)
        Myy = np.zeros(5)
        Mzz = np.zeros(5)

        load.changePointLoads(nF, Fx, Fy, Fz, Mxx, Myy, Mzz)

        nD = np.array([8])
        Dx = np.array([0.1])
        Dy = np.array([0.0])
        Dz = np.array([0.0])
        Dxx = np.array([0.0])
        Dyy = np.array([0.0])
        Dzz = np.array([0.0])

        load.changePrescribedDisplacements(nD, Dx, Dy, Dz, Dxx, Dyy, Dzz)

        frame.addLoadCase(load)

        # load cases
        gx = 0.0
        gy = -386.4
        gz = 0.0

        load = StaticLoadCase(gx, gy, gz)

        nF = np.array([3, 4, 5])
        Fx = np.array([20.0, 10.0, 20.0])
        Fy = np.zeros(3)
        Fz = np.zeros(3)
        Mxx = np.zeros(3)
        Myy = np.zeros(3)
        Mzz = np.zeros(3)

        load.changePointLoads(nF, Fx, Fy, Fz, Mxx, Myy, Mzz)

        EL = np.array([10, 13, 15])
        a = 6e-12 * np.ones(3)
        hy = 5.0 * np.ones(3)
        hz = 5.0 * np.ones(3)
        Typ = np.array([10.0, 15.0, 17.0])
        Tym = np.array([10.0, 15.0, 17.0])
        Tzp = np.array([10.0, 15.0, 17.0])
        Tzm = np.array([10.0, 15.0, 17.0])

        load.changeTemperatureLoads(EL, a, hy, hz, Typ, Tym, Tzp, Tzm)

        nD = np.array([1, 8])
        Dx = np.array([0.0, 0.1])
        Dy = np.array([-1.0, 0.0])
        Dz = np.array([0.0, 0.0])
        Dxx = np.array([0.0, 0.0])
        Dyy = np.array([0.0, 0.0])
        Dzz = np.array([0.0, 0.0])

        load.changePrescribedDisplacements(nD, Dx, Dy, Dz, Dxx, Dyy, Dzz)

        frame.addLoadCase(load)

        self.displacements, self.forces, self.reactions, self.internalForces, self.mass, self.modal = frame.run()

    def test_disp1(self):

        disp = self.displacements
        iCase = 0

        node = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int_)
        dx = np.array(
            [
                0.0,
                0.010776,
                0.035528,
                0.060279,
                0.086295,
                0.112311,
                0.129754,
                0.100000,
                0.089226,
                0.059394,
                0.029563,
                0.012122,
            ]
        )
        dy = np.array(
            [
                0.0,
                -0.171344,
                -0.297816,
                -0.332643,
                -0.295487,
                -0.184135,
                0.0,
                -0.152896,
                -0.289325,
                -0.332855,
                -0.291133,
                -0.166951,
            ]
        )
        dz = np.zeros(12)
        dxrot = np.zeros(12)
        dyrot = np.zeros(12)
        dzrot = np.array(
            [
                -0.501728,
                -0.087383,
                0.015516,
                0.000017,
                -0.015556,
                0.087387,
                0.501896,
                0.136060,
                -0.011102,
                -0.000000,
                0.011062,
                -0.136029,
            ]
        )

        np.testing.assert_array_equal(disp.node[iCase, :], node)
        np.testing.assert_array_almost_equal(disp.dx[iCase, :], dx, decimal=6)
        np.testing.assert_array_almost_equal(disp.dy[iCase, :], dy, decimal=6)
        np.testing.assert_array_almost_equal(disp.dz[iCase, :], dz, decimal=6)
        np.testing.assert_array_almost_equal(disp.dxrot[iCase, :], dxrot, decimal=6)
        np.testing.assert_array_almost_equal(disp.dyrot[iCase, :], dyrot, decimal=6)
        np.testing.assert_array_almost_equal(disp.dzrot[iCase, :], dzrot, decimal=6)

    def test_disp2(self):

        disp = self.displacements
        iCase = 1

        node = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int_)
        dx = np.array(
            [
                0.0,
                0.071965,
                0.134909,
                0.189577,
                0.220207,
                0.242560,
                0.254035,
                0.100000,
                0.048727,
                0.016113,
                -0.016502,
                -0.027973,
            ]
        )
        dy = np.array(
            [
                -1.000000,
                -1.067463,
                -1.018927,
                -0.850595,
                -0.615710,
                -0.325659,
                0.0,
                -1.076148,
                -1.018711,
                -0.850807,
                -0.615495,
                -0.314444,
            ]
        )
        dz = np.zeros(12)
        dxrot = np.zeros(12)
        dyrot = np.zeros(12)
        dzrot = np.array(
            [
                -0.501206,
                -0.086438,
                0.016991,
                0.001616,
                -0.013988,
                0.088765,
                0.503041,
                0.136834,
                -0.009551,
                0.001627,
                0.012588,
                -0.134603,
            ]
        )

        np.testing.assert_array_equal(disp.node[iCase, :], node)
        np.testing.assert_array_almost_equal(disp.dx[iCase, :], dx, decimal=6)
        np.testing.assert_array_almost_equal(disp.dy[iCase, :], dy, decimal=6)
        np.testing.assert_array_almost_equal(disp.dz[iCase, :], dz, decimal=6)
        np.testing.assert_array_almost_equal(disp.dxrot[iCase, :], dxrot, decimal=6)
        np.testing.assert_array_almost_equal(disp.dyrot[iCase, :], dyrot, decimal=6)
        np.testing.assert_array_almost_equal(disp.dzrot[iCase, :], dzrot, decimal=6)

    def test_force1(self):

        forces = self.forces
        iCase = 0

        output = StringIO(
            """
         1      1    -26.042      0.099      0.0        0.0        0.0       -1.853
         1      2     26.042      0.241      0.0        0.0        0.0       -6.648
         2      2    -59.816      0.162      0.0        0.0        0.0        2.644
         2      3     59.816      0.178      0.0        0.0        0.0       -3.656
         3      3    -59.815      0.172      0.0        0.0        0.0        3.553
         3      4     59.815      0.168      0.0        0.0        0.0       -3.319
         4      4    -62.872      0.168      0.0        0.0        0.0        3.319
         4      5     62.872      0.172      0.0        0.0        0.0       -3.554
         5      5    -62.872      0.178      0.0        0.0        0.0        3.657
         5      6     62.872      0.161      0.0        0.0        0.0       -2.643
         6      6    -42.155      0.241      0.0        0.0        0.0        6.647
         6      7     42.155      0.099      0.0        0.0        0.0        1.853
         7      1     64.086      0.148      0.0        0.0        0.0        1.853
         7      8    -63.746      0.192      0.0        0.0        0.0       -5.581
         8      2    -44.414      0.006      0.0        0.0        0.0       -0.176
         8      8     44.754     -0.006      0.0        0.0        0.0        0.904
         9      2     47.936      0.164      0.0        0.0        0.0        4.180
         9      9    -47.596      0.176      0.0        0.0        0.0       -5.173
        10      3    -20.350      0.001      0.0        0.0        0.0        0.103
        10      9     20.690     -0.001      0.0        0.0        0.0       -0.026
        11      4    -17.194     -0.171      0.0        0.0        0.0       -4.841
        11      9     17.534     -0.169      0.0        0.0        0.0        4.734
        12      4      0.682      0.0        0.0        0.0        0.0        0.0
        12     10     -0.342      0.0        0.0        0.0        0.0        0.0
        13      4    -12.872      0.171      0.0        0.0        0.0        4.841
        13     11     13.212      0.169      0.0        0.0        0.0       -4.734
        14      5    -10.350     -0.001      0.0        0.0        0.0       -0.104
        14     11     10.690      0.001      0.0        0.0        0.0        0.025
        15      6     29.472     -0.164      0.0        0.0        0.0       -4.180
        15     11    -29.132     -0.176      0.0        0.0        0.0        5.173
        16      6    -41.358     -0.006      0.0        0.0        0.0        0.175
        16     12     41.698      0.006      0.0        0.0        0.0       -0.905
        17      7     59.764     -0.148      0.0        0.0        0.0       -1.853
        17     12    -59.424     -0.192      0.0        0.0        0.0        5.580
        18      8     26.036      0.185      0.0        0.0        0.0        4.677
        18      9    -26.036      0.155      0.0        0.0        0.0       -2.832
        19      9     72.094      0.169      0.0        0.0        0.0        3.297
        19     10    -72.094      0.171      0.0        0.0        0.0       -3.447
        20     10     72.094      0.171      0.0        0.0        0.0        3.447
        20     11    -72.094      0.169      0.0        0.0        0.0       -3.297
        21     11     42.149      0.155      0.0        0.0        0.0        2.833
        21     12    -42.149      0.185      0.0        0.0        0.0       -4.675
        """
        )

        out = np.loadtxt(output)

        np.testing.assert_array_equal(forces.element[iCase, :], out[:, 0])
        np.testing.assert_array_equal(forces.node[iCase, :], out[:, 1])
        np.testing.assert_array_almost_equal(forces.Nx[iCase, :], out[:, 2], decimal=3)
        np.testing.assert_array_almost_equal(forces.Vy[iCase, :], out[:, 3], decimal=3)
        np.testing.assert_array_almost_equal(forces.Vz[iCase, :], out[:, 4], decimal=3)
        np.testing.assert_array_almost_equal(forces.Txx[iCase, :], out[:, 5], decimal=3)
        np.testing.assert_array_almost_equal(forces.Myy[iCase, :], out[:, 6], decimal=3)
        np.testing.assert_array_almost_equal(forces.Mzz[iCase, :], out[:, 7], decimal=3)

    def test_force2(self):

        forces = self.forces
        iCase = 1

        output = StringIO(
            """
         1      1   -173.916      0.099      0.0        0.0        0.0       -1.856
         1      2    173.916      0.241      0.0        0.0        0.0       -6.649
         2      2   -152.115      0.161      0.0        0.0        0.0        2.639
         2      3    152.115      0.178      0.0        0.0        0.0       -3.658
         3      3   -132.114      0.172      0.0        0.0        0.0        3.550
         3      4    132.114      0.168      0.0        0.0        0.0       -3.321
         4      4    -74.021      0.168      0.0        0.0        0.0        3.318
         4      5     74.021      0.172      0.0        0.0        0.0       -3.555
         5      5    -54.022      0.178      0.0        0.0        0.0        3.658
         5      6     54.022      0.161      0.0        0.0        0.0       -2.643
         6      6    -27.729      0.241      0.0        0.0        0.0        6.649
         6      7     27.729      0.099      0.0        0.0        0.0        1.854
         7      1    -28.651      0.148      0.0        0.0        0.0        1.856
         7      8     28.991      0.192      0.0        0.0        0.0       -5.577
         8      2     21.160      0.006      0.0        0.0        0.0       -0.171
         8      8    -20.820     -0.006      0.0        0.0        0.0        0.908
         9      2    -30.658      0.164      0.0        0.0        0.0        4.180
         9      9     30.998      0.176      0.0        0.0        0.0       -5.170
        10      3     -0.350      0.001      0.0        0.0        0.0        0.108
        10      9      0.690     -0.001      0.0        0.0        0.0       -0.021
        11      4     33.116     -0.171      0.0        0.0        0.0       -4.841
        11      9    -32.776     -0.169      0.0        0.0        0.0        4.734
        12      4      0.682      0.0        0.0        0.0        0.0        0.003
        12     10     -0.342      0.0        0.0        0.0        0.0        0.003
        13      4    -34.898      0.171      0.0        0.0        0.0        4.842
        13     11     35.237      0.169      0.0        0.0        0.0       -4.734
        14      5     -0.350     -0.001      0.0        0.0        0.0       -0.103
        14     11      0.690      0.001      0.0        0.0        0.0        0.025
        15      6     37.356     -0.164      0.0        0.0        0.0       -4.180
        15     11    -37.016     -0.176      0.0        0.0        0.0        5.173
        16      6    -26.933     -0.006      0.0        0.0        0.0        0.175
        16     12     27.273      0.006      0.0        0.0        0.0       -0.905
        17      7     39.363     -0.148      0.0        0.0        0.0       -1.854
        17     12    -39.023     -0.192      0.0        0.0        0.0        5.580
        18      8    123.910      0.185      0.0        0.0        0.0        4.668
        18      9   -123.910      0.155      0.0        0.0        0.0       -2.837
        19      9     78.818      0.169      0.0        0.0        0.0        3.294
        19     10    -78.818      0.171      0.0        0.0        0.0       -3.449
        20     10     78.818      0.171      0.0        0.0        0.0        3.447
        20     11    -78.818      0.169      0.0        0.0        0.0       -3.298
        21     11     27.723      0.155      0.0        0.0        0.0        2.834
        21     12    -27.723      0.185      0.0        0.0        0.0       -4.675
        """
        )

        out = np.loadtxt(output)

        np.testing.assert_array_equal(forces.element[iCase, :], out[:, 0])
        np.testing.assert_array_equal(forces.node[iCase, :], out[:, 1])
        np.testing.assert_array_almost_equal(forces.Nx[iCase, :], out[:, 2], decimal=3)
        np.testing.assert_array_almost_equal(forces.Vy[iCase, :], out[:, 3], decimal=3)
        np.testing.assert_array_almost_equal(forces.Vz[iCase, :], out[:, 4], decimal=3)
        np.testing.assert_array_almost_equal(forces.Txx[iCase, :], out[:, 5], decimal=3)
        np.testing.assert_array_almost_equal(forces.Myy[iCase, :], out[:, 6], decimal=3)
        np.testing.assert_array_almost_equal(forces.Mzz[iCase, :], out[:, 7], decimal=3)

    def test_reactions1(self):

        reactions = self.reactions
        iCase = 0

        output = StringIO(
            """
             1      19.168      45.519       0.0         0.0         0.0         0.0
             2       0.0         0.0         0.0         0.0         0.0         0.0
             3       0.0         0.0         0.0         0.0         0.0         0.0
             4       0.0         0.0         0.0         0.0         0.0         0.0
             5       0.0         0.0         0.0         0.0         0.0         0.0
             6       0.0         0.0         0.0         0.0         0.0         0.0
             7       0.0        42.463       0.0         0.0         0.0         0.0
             8     -19.168       0.0         0.0         0.0         0.0         0.0
             9       0.0         0.0         0.0         0.0         0.0         0.0
            10       0.0         0.0         0.0         0.0         0.0         0.0
            11       0.0         0.0         0.0         0.0         0.0         0.0
            12       0.0         0.0         0.0         0.0         0.0         0.0
        """
        )

        out = np.loadtxt(output)

        np.testing.assert_array_equal(reactions.node[iCase, :], out[:, 0])
        np.testing.assert_array_almost_equal(reactions.Fx[iCase, :], out[:, 1], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Fy[iCase, :], out[:, 2], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Fz[iCase, :], out[:, 3], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Mxx[iCase, :], out[:, 4], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Myy[iCase, :], out[:, 5], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Mzz[iCase, :], out[:, 6], decimal=3)

    def test_reactions2(self):

        reactions = self.reactions
        iCase = 1

        output = StringIO(
            """
         1    -194.280     -20.056       0.0         0.0         0.0         0.0
         2       0.0         0.0         0.0         0.0         0.0         0.0
         3       0.0         0.0         0.0         0.0         0.0         0.0
         4       0.0         0.0         0.0         0.0         0.0         0.0
         5       0.0         0.0         0.0         0.0         0.0         0.0
         6       0.0         0.0         0.0         0.0         0.0         0.0
         7       0.0        28.038       0.0         0.0         0.0         0.0
         8     144.280       0.0         0.0         0.0         0.0         0.0
         9       0.0         0.0         0.0         0.0         0.0         0.0
        10       0.0         0.0         0.0         0.0         0.0         0.0
        11       0.0         0.0         0.0         0.0         0.0         0.0
        12       0.0         0.0         0.0         0.0         0.0         0.0
        """
        )

        out = np.loadtxt(output)

        np.testing.assert_array_equal(reactions.node[iCase, :], out[:, 0])
        np.testing.assert_array_almost_equal(reactions.Fx[iCase, :], out[:, 1], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Fy[iCase, :], out[:, 2], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Fz[iCase, :], out[:, 3], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Mxx[iCase, :], out[:, 4], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Myy[iCase, :], out[:, 5], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Mzz[iCase, :], out[:, 6], decimal=3)

    def test_if1(self):

        intF = self.internalForces
        iE = 3
        iCase = 0

        output = StringIO(
            """
            0.000000e+00    6.287160e+01   -1.679863e-01    0.000000e+00    0.000000e+00    0.000000e+00   -3.319261e+00    6.027882e-02   -3.326427e-01    0.000000e+00    0.000000e+00
            1.000000e+01    6.287160e+01   -1.396631e-01    0.000000e+00    0.000000e+00    0.000000e+00   -1.781014e+00    6.244680e-02   -7.675230e-01    0.000000e+00    0.000000e+00
            2.000000e+01    6.287160e+01   -1.113400e-01    0.000000e+00    0.000000e+00    0.000000e+00   -5.259979e-01    6.461479e-02   -1.832824e+00    0.000000e+00    0.000000e+00
            3.000000e+01    6.287160e+01   -8.301690e-02    0.000000e+00    0.000000e+00    0.000000e+00    4.457867e-01    6.678278e-02   -3.095780e+00    0.000000e+00    0.000000e+00
            4.000000e+01    6.287160e+01   -5.469378e-02    0.000000e+00    0.000000e+00    0.000000e+00    1.134340e+00    6.895076e-02   -4.221295e+00    0.000000e+00    0.000000e+00
            5.000000e+01    6.287160e+01   -2.637066e-02    0.000000e+00    0.000000e+00    0.000000e+00    1.539662e+00    7.111875e-02   -4.971936e+00    0.000000e+00    0.000000e+00
            6.000000e+01    6.287160e+01    1.952455e-03    0.000000e+00    0.000000e+00    0.000000e+00    1.661753e+00    7.328673e-02   -5.207937e+00    0.000000e+00    0.000000e+00
            7.000000e+01    6.287160e+01    3.027557e-02    0.000000e+00    0.000000e+00    0.000000e+00    1.500613e+00    7.545472e-02   -4.887197e+00    0.000000e+00    0.000000e+00
            8.000000e+01    6.287160e+01    5.859869e-02    0.000000e+00    0.000000e+00    0.000000e+00    1.056242e+00    7.762271e-02   -4.065281e+00    0.000000e+00    0.000000e+00
            9.000000e+01    6.287160e+01    8.692181e-02    0.000000e+00    0.000000e+00    0.000000e+00    3.286393e-01    7.979069e-02   -2.895422e+00    0.000000e+00    0.000000e+00
            1.000000e+02    6.287160e+01    1.152449e-01    0.000000e+00    0.000000e+00    0.000000e+00   -6.821944e-01    8.195868e-02   -1.628517e+00    0.000000e+00    0.000000e+00
            1.100000e+02    6.287160e+01    1.435680e-01    0.000000e+00    0.000000e+00    0.000000e+00   -1.976259e+00    8.412667e-02   -6.131284e-01    0.000000e+00    0.000000e+00
            1.200000e+02    6.287160e+01    1.718912e-01    0.000000e+00    0.000000e+00    0.000000e+00   -3.553555e+00    8.629465e-02   -2.954865e-01    0.000000e+00    0.000000e+00
        """
        )

        out = np.loadtxt(output)

        np.testing.assert_array_almost_equal(intF[iE].x[iCase, :], out[:, 0], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Nx[iCase, :], out[:, 1], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Vy[iCase, :], out[:, 2], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Vz[iCase, :], out[:, 3], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Tx[iCase, :], out[:, 4], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].My[iCase, :], out[:, 5], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Mz[iCase, :], out[:, 6], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Dx[iCase, :], out[:, 7], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Dy[iCase, :], out[:, 8], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Dz[iCase, :], out[:, 9], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Rx[iCase, :], out[:, 10], decimal=3)

    def test_if2(self):

        intF = self.internalForces
        iE = 7
        iCase = 1

        output = StringIO(
            """
          0.000000e+00   -2.116037e+01   -6.145897e-03    0.000000e+00    0.000000e+00    0.000000e+00    1.708194e-01   -1.067463e+00   -7.196515e-02    0.000000e+00    0.000000e+00
          1.000000e+01   -2.113205e+01   -6.145897e-03    0.000000e+00    0.000000e+00    0.000000e+00    2.322784e-01   -1.068192e+00   -9.033631e-01    0.000000e+00    0.000000e+00
          2.000000e+01   -2.110372e+01   -6.145897e-03    0.000000e+00    0.000000e+00    0.000000e+00    2.937373e-01   -1.068920e+00   -1.654665e+00    0.000000e+00    0.000000e+00
          3.000000e+01   -2.107540e+01   -6.145897e-03    0.000000e+00    0.000000e+00    0.000000e+00    3.551963e-01   -1.069647e+00   -2.304678e+00    0.000000e+00    0.000000e+00
          4.000000e+01   -2.104708e+01   -6.145897e-03    0.000000e+00    0.000000e+00    0.000000e+00    4.166553e-01   -1.070373e+00   -2.832210e+00    0.000000e+00    0.000000e+00
          5.000000e+01   -2.101876e+01   -6.145897e-03    0.000000e+00    0.000000e+00    0.000000e+00    4.781142e-01   -1.071099e+00   -3.216067e+00    0.000000e+00    0.000000e+00
          6.000000e+01   -2.099043e+01   -6.145897e-03    0.000000e+00    0.000000e+00    0.000000e+00    5.395732e-01   -1.071823e+00   -3.435058e+00    0.000000e+00    0.000000e+00
          7.000000e+01   -2.096211e+01   -6.145897e-03    0.000000e+00    0.000000e+00    0.000000e+00    6.010322e-01   -1.072546e+00   -3.467988e+00    0.000000e+00    0.000000e+00
          8.000000e+01   -2.093379e+01   -6.145897e-03    0.000000e+00    0.000000e+00    0.000000e+00    6.624911e-01   -1.073269e+00   -3.293667e+00    0.000000e+00    0.000000e+00
          9.000000e+01   -2.090546e+01   -6.145897e-03    0.000000e+00    0.000000e+00    0.000000e+00    7.239501e-01   -1.073990e+00   -2.890900e+00    0.000000e+00    0.000000e+00
          1.000000e+02   -2.087714e+01   -6.145897e-03    0.000000e+00    0.000000e+00    0.000000e+00    7.854091e-01   -1.074710e+00   -2.238495e+00    0.000000e+00    0.000000e+00
          1.100000e+02   -2.084882e+01   -6.145897e-03    0.000000e+00    0.000000e+00    0.000000e+00    8.468680e-01   -1.075430e+00   -1.315259e+00    0.000000e+00    0.000000e+00
          1.200000e+02   -2.082049e+01   -6.145897e-03    0.000000e+00    0.000000e+00    0.000000e+00    9.083270e-01   -1.076148e+00   -1.000000e-01    0.000000e+00    0.000000e+00
        """
        )

        out = np.loadtxt(output)

        np.testing.assert_array_almost_equal(intF[iE].x[iCase, :], out[:, 0], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Nx[iCase, :], out[:, 1], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Vy[iCase, :], out[:, 2], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Vz[iCase, :], out[:, 3], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Tx[iCase, :], out[:, 4], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].My[iCase, :], out[:, 5], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Mz[iCase, :], out[:, 6], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Dx[iCase, :], out[:, 7], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Dy[iCase, :], out[:, 8], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Dz[iCase, :], out[:, 9], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Rx[iCase, :], out[:, 10], decimal=3)


class FrameTestEXB(unittest.TestCase):
    def setUp(self):

        # nodes
        string = StringIO(
            """
        1   0.0 0.0 1000    0.0
        2   -1200   -900    0.0 0.0
        3    1200   -900    0.0 0.0
        4    1200    900    0.0 0.0
        5   -1200    900    0.0 0.0
        """
        )
        out = np.loadtxt(string)

        nodes = NodeData(out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4])

        # reactions
        string = StringIO(
            """
          2 1  1  1  1  1  1
          3 1  1  1  1  1  1
          4 1  1  1  1  1  1
          5 1  1  1  1  1  1
        """
        )
        out = np.loadtxt(string, dtype=np.int_)
        rigid = 1

        reactions = ReactionData(out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4], out[:, 5], out[:, 6], rigid)

        # elements

        string = StringIO(
            """
        1 2 1   36.0    20.0    20.0    1000    492     492 200000  79300  0 7.85e-9
        2 1 3   36.0    20.0    20.0    1000    492     492 200000  79300  0 7.85e-9
        3 1 4   36.0    20.0    20.0    1000    492     492 200000  79300  0 7.85e-9
        4 5 1   36.0    20.0    20.0    1000    492     492 200000  79300  0 7.85e-9
        """
        )
        out = np.loadtxt(string)

        elements = ElementData(
            out[:, 0],
            out[:, 1],
            out[:, 2],
            out[:, 3],
            out[:, 4],
            out[:, 5],
            out[:, 6],
            out[:, 7],
            out[:, 8],
            out[:, 9],
            out[:, 10],
            out[:, 11],
            out[:, 12],
        )

        # parameters
        shear = True  # 1: include shear deformation
        geom = True  # 1: include geometric stiffness
        dx = 20.0  # x-axis increment for internal forces
        options = Options(shear, geom, dx)

        frame = Frame(nodes, reactions, elements, options)

        # dynamics
        nM = 6  # number of desired dynamic modes of vibration
        Mmethod = 1  # 1: subspace Jacobi     2: Stodola
        lump = 0  # 0: consistent mass ... 1: lumped mass matrix
        tol = 1e-9  # mode shape tolerance
        shift = 0.0  # shift value ... for unrestrained structures
        frame.enableDynamics(nM, Mmethod, lump, tol, shift)

        # load cases 1
        gx = 0.0
        gy = 0.0
        gz = -9806.33

        load = StaticLoadCase(gx, gy, gz)

        nF = np.array([1])
        Fx = np.array([100.0])
        Fy = np.array([-200.0])
        Fz = np.array([-100.0])
        Mxx = np.array([0.0])
        Myy = np.array([0.0])
        Mzz = np.array([0.0])

        load.changePointLoads(nF, Fx, Fy, Fz, Mxx, Myy, Mzz)

        frame.addLoadCase(load)

        gx = 0.0
        gy = 0.0
        gz = -9806.33

        load = StaticLoadCase(gx, gy, gz)

        EL = np.array([1, 2])
        Px = np.array([0.0, 0.0])
        Py = np.array([100.0, -200.0])
        Pz = np.array([-900.0, 200.0])
        x = np.array([600.0, 800.0])
        load.changeElementLoads(EL, Px, Py, Pz, x)

        frame.addLoadCase(load)

        N = np.array([1])
        EMs = np.array([0.1])
        EMx = np.array([0.0])
        EMy = np.array([0.0])
        EMz = np.array([0.0])
        EMxy = np.array([0.0])
        EMxz = np.array([0.0])
        EMyz = np.array([0.0])
        rhox = np.array([0.0])
        rhoy = np.array([0.0])
        rhoz = np.array([0.0])
        addGravityLoad = False
        frame.changeExtraNodeMass(N, EMs, EMx, EMy, EMz, EMxy, EMxz, EMyz, rhox, rhoy, rhoz, addGravityLoad)

        self.displacements, self.forces, self.reactions, self.internalForces, self.mass, self.modal = frame.run()

    def test_disp1(self):

        disp = self.displacements
        iCase = 0

        node = np.array([1, 2, 3, 4, 5])
        dx = np.array([0.014127, 0.0, 0.0, 0.0, 0.0])
        dy = np.array([-0.050229, 0.0, 0.0, 0.0, 0.0])
        dz = np.array([-0.022374, 0.0, 0.0, 0.0, 0.0])
        dxrot = np.array([0.000037, 0.0, 0.0, 0.0, 0.0])
        dyrot = np.array([0.000009, 0.0, 0.0, 0.0, 0.0])
        dzrot = np.array([0.000000, 0.0, 0.0, 0.0, 0.0])

        np.testing.assert_equal(disp.node[iCase, :], node)
        np.testing.assert_almost_equal(disp.dx[iCase, :], dx, decimal=6)
        np.testing.assert_almost_equal(disp.dy[iCase, :], dy, decimal=6)
        np.testing.assert_almost_equal(disp.dz[iCase, :], dz, decimal=6)
        np.testing.assert_almost_equal(disp.dxrot[iCase, :], dxrot, decimal=6)
        np.testing.assert_almost_equal(disp.dyrot[iCase, :], dyrot, decimal=6)
        np.testing.assert_almost_equal(disp.dzrot[iCase, :], dzrot, decimal=6)

    def test_force1(self):

        forces = self.forces
        iCase = 0

        output = StringIO(
            """
     1      2    113.543      0.003      2.082     -1.289   -627.689      6.040
     1      1   -110.772     -0.003      2.075      1.289    620.132      4.573
     2      1    185.886     -0.000      2.074      0.904   -620.114     -2.774
     2      3   -188.657      0.000      2.083     -0.904    627.325     -3.504
     3      1    -14.410     -0.007      2.075      1.285   -622.621     -4.568
     3      4     11.639      0.007      2.082     -1.285    628.130     -6.781
     4      5    -86.753      0.006      2.084     -0.908   -629.366      4.619
     4      1     89.524     -0.006      2.073      0.908    623.616      2.764
        """
        )

        out = np.loadtxt(output)

        np.testing.assert_array_equal(forces.element[iCase, :], out[:, 0])
        np.testing.assert_array_equal(forces.node[iCase, :], out[:, 1])
        np.testing.assert_array_almost_equal(forces.Nx[iCase, :], out[:, 2], decimal=3)
        np.testing.assert_array_almost_equal(forces.Vy[iCase, :], out[:, 3], decimal=3)
        np.testing.assert_array_almost_equal(forces.Vz[iCase, :], out[:, 4], decimal=3)
        np.testing.assert_array_almost_equal(forces.Txx[iCase, :], out[:, 5], decimal=3)
        np.testing.assert_array_almost_equal(forces.Myy[iCase, :], out[:, 6], decimal=3)
        np.testing.assert_array_almost_equal(forces.Mzz[iCase, :], out[:, 7], decimal=3)

    def test_reactions1(self):

        reactions = self.reactions
        iCase = 0

        output = StringIO(
            """
     2      74.653      55.994      64.715     373.079    -504.802       4.303
     3    -124.653      93.490     106.381     374.234     503.477      -2.418
     4       8.667       6.509      -4.724    -380.749     499.607      -4.936
     5     -58.667      44.008     -46.388    -380.267    -501.498       3.335
        """
        )

        out = np.loadtxt(output)

        np.testing.assert_array_equal(reactions.node[iCase, :], out[:, 0])
        np.testing.assert_array_almost_equal(reactions.Fx[iCase, :], out[:, 1], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Fy[iCase, :], out[:, 2], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Fz[iCase, :], out[:, 3], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Mxx[iCase, :], out[:, 4], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Myy[iCase, :], out[:, 5], decimal=3)
        np.testing.assert_array_almost_equal(reactions.Mzz[iCase, :], out[:, 6], decimal=3)

    def test_if1(self):

        intF = self.internalForces
        iE = 1
        iCase = 0

        output = StringIO(
            """
  0.000000e+00	 -1.858856e+02	  1.885306e-04	 -2.073998e+00	 -9.043267e-01	 -6.201138e+02	  2.773828e+00	  4.689002e-02	 -3.170647e-02	  4.370102e-03	  2.055861e-05
  2.000000e+01	 -1.859164e+02	  1.885306e-04	 -2.027881e+00	 -9.043267e-01	 -5.790859e+02	  2.704083e+00	  4.637362e-02	 -3.140374e-02	  2.564479e-03	  2.033054e-05
  4.000000e+01	 -1.859471e+02	  1.885306e-04	 -1.981764e+00	 -9.043267e-01	 -5.389804e+02	  2.634338e+00	  4.585715e-02	 -3.109157e-02	 -1.591829e-03	  2.010246e-05
  6.000000e+01	 -1.859779e+02	  1.885306e-04	 -1.935648e+00	 -9.043267e-01	 -4.997972e+02	  2.564593e+00	  4.534058e-02	 -3.077025e-02	 -7.935789e-03	  1.987438e-05
  8.000000e+01	 -1.860086e+02	  1.885306e-04	 -1.889531e+00	 -9.043267e-01	 -4.615363e+02	  2.494847e+00	  4.482394e-02	 -3.044007e-02	 -1.630812e-02	  1.964631e-05
  1.000000e+02	 -1.860394e+02	  1.885306e-04	 -1.843414e+00	 -9.043267e-01	 -4.241977e+02	  2.425102e+00	  4.430720e-02	 -3.010131e-02	 -2.655330e-02	  1.941823e-05
  1.200000e+02	 -1.860701e+02	  1.885306e-04	 -1.797297e+00	 -9.043267e-01	 -3.877815e+02	  2.355357e+00	  4.379038e-02	 -2.975426e-02	 -3.851953e-02	  1.919015e-05
  1.400000e+02	 -1.861009e+02	  1.885306e-04	 -1.751181e+00	 -9.043267e-01	 -3.522877e+02	  2.285612e+00	  4.327348e-02	 -2.939919e-02	 -5.205878e-02	  1.896207e-05
  1.600000e+02	 -1.861316e+02	  1.885306e-04	 -1.705064e+00	 -9.043267e-01	 -3.177161e+02	  2.215867e+00	  4.275649e-02	 -2.903639e-02	 -6.702678e-02	  1.873400e-05
  1.800000e+02	 -1.861624e+02	  1.885306e-04	 -1.658947e+00	 -9.043267e-01	 -2.840669e+02	  2.146121e+00	  4.223942e-02	 -2.866615e-02	 -8.328298e-02	  1.850592e-05
  2.000000e+02	 -1.861931e+02	  1.885306e-04	 -1.612831e+00	 -9.043267e-01	 -2.513400e+02	  2.076376e+00	  4.172225e-02	 -2.828875e-02	 -1.006906e-01	  1.827784e-05
  2.200000e+02	 -1.862238e+02	  1.885306e-04	 -1.566714e+00	 -9.043267e-01	 -2.195355e+02	  2.006631e+00	  4.120501e-02	 -2.790447e-02	 -1.191166e-01	  1.804976e-05
  2.400000e+02	 -1.862546e+02	  1.885306e-04	 -1.520597e+00	 -9.043267e-01	 -1.886533e+02	  1.936886e+00	  4.068768e-02	 -2.751359e-02	 -1.384317e-01	  1.782169e-05
  2.600000e+02	 -1.862853e+02	  1.885306e-04	 -1.474481e+00	 -9.043267e-01	 -1.586934e+02	  1.867141e+00	  4.017026e-02	 -2.711640e-02	 -1.585104e-01	  1.759361e-05
  2.800000e+02	 -1.863161e+02	  1.885306e-04	 -1.428364e+00	 -9.043267e-01	 -1.296559e+02	  1.797395e+00	  3.965276e-02	 -2.671319e-02	 -1.792308e-01	  1.736553e-05
  3.000000e+02	 -1.863468e+02	  1.885306e-04	 -1.382247e+00	 -9.043267e-01	 -1.015407e+02	  1.727650e+00	  3.913517e-02	 -2.630423e-02	 -2.004750e-01	  1.713745e-05
  3.200000e+02	 -1.863776e+02	  1.885306e-04	 -1.336130e+00	 -9.043267e-01	 -7.434782e+01	  1.657905e+00	  3.861750e-02	 -2.588981e-02	 -2.221286e-01	  1.690938e-05
  3.400000e+02	 -1.864083e+02	  1.885306e-04	 -1.290014e+00	 -9.043267e-01	 -4.807728e+01	  1.588160e+00	  3.809974e-02	 -2.547021e-02	 -2.440811e-01	  1.668130e-05
  3.600000e+02	 -1.864391e+02	  1.885306e-04	 -1.243897e+00	 -9.043267e-01	 -2.272908e+01	  1.518415e+00	  3.758190e-02	 -2.504572e-02	 -2.662257e-01	  1.645322e-05
  3.800000e+02	 -1.864698e+02	  1.885306e-04	 -1.197780e+00	 -9.043267e-01	  1.696785e+00	  1.448669e+00	  3.706397e-02	 -2.461661e-02	 -2.884594e-01	  1.622514e-05
  4.000000e+02	 -1.865005e+02	  1.885306e-04	 -1.151664e+00	 -9.043267e-01	  2.520032e+01	  1.378924e+00	  3.654595e-02	 -2.418319e-02	 -3.106829e-01	  1.599707e-05
  4.200000e+02	 -1.865313e+02	  1.885306e-04	 -1.105547e+00	 -9.043267e-01	  4.778152e+01	  1.309179e+00	  3.602785e-02	 -2.374571e-02	 -3.328006e-01	  1.576899e-05
  4.400000e+02	 -1.865620e+02	  1.885306e-04	 -1.059430e+00	 -9.043267e-01	  6.944038e+01	  1.239434e+00	  3.550967e-02	 -2.330448e-02	 -3.547207e-01	  1.554091e-05
  4.600000e+02	 -1.865928e+02	  1.885306e-04	 -1.013314e+00	 -9.043267e-01	  9.017691e+01	  1.169689e+00	  3.499140e-02	 -2.285977e-02	 -3.763553e-01	  1.531284e-05
  4.800000e+02	 -1.866235e+02	  1.885306e-04	 -9.671968e-01	 -9.043267e-01	  1.099911e+02	  1.099943e+00	  3.447304e-02	 -2.241187e-02	 -3.976199e-01	  1.508476e-05
  5.000000e+02	 -1.866543e+02	  1.885306e-04	 -9.210801e-01	 -9.043267e-01	  1.288830e+02	  1.030198e+00	  3.395460e-02	 -2.196106e-02	 -4.184342e-01	  1.485668e-05
  5.200000e+02	 -1.866850e+02	  1.885306e-04	 -8.749634e-01	 -9.043267e-01	  1.468525e+02	  9.604531e-01	  3.343607e-02	 -2.150763e-02	 -4.387211e-01	  1.462860e-05
  5.400000e+02	 -1.867158e+02	  1.885306e-04	 -8.288467e-01	 -9.043267e-01	  1.638997e+02	  8.907079e-01	  3.291746e-02	 -2.105185e-02	 -4.584078e-01	  1.440053e-05
  5.600000e+02	 -1.867465e+02	  1.885306e-04	 -7.827300e-01	 -9.043267e-01	  1.800246e+02	  8.209627e-01	  3.239877e-02	 -2.059402e-02	 -4.774249e-01	  1.417245e-05
  5.800000e+02	 -1.867772e+02	  1.885306e-04	 -7.366133e-01	 -9.043267e-01	  1.952271e+02	  7.512175e-01	  3.187998e-02	 -2.013441e-02	 -4.957069e-01	  1.394437e-05
  6.000000e+02	 -1.868080e+02	  1.885306e-04	 -6.904966e-01	 -9.043267e-01	  2.095073e+02	  6.814723e-01	  3.136111e-02	 -1.967330e-02	 -5.131920e-01	  1.371629e-05
  6.200000e+02	 -1.868387e+02	  1.885306e-04	 -6.443800e-01	 -9.043267e-01	  2.228651e+02	  6.117271e-01	  3.084216e-02	 -1.921099e-02	 -5.298220e-01	  1.348822e-05
  6.400000e+02	 -1.868695e+02	  1.885306e-04	 -5.982633e-01	 -9.043267e-01	  2.353007e+02	  5.419819e-01	  3.032312e-02	 -1.874776e-02	 -5.455428e-01	  1.326014e-05
  6.600000e+02	 -1.869002e+02	  1.885306e-04	 -5.521466e-01	 -9.043267e-01	  2.468139e+02	  4.722367e-01	  2.980400e-02	 -1.828388e-02	 -5.603038e-01	  1.303206e-05
  6.800000e+02	 -1.869310e+02	  1.885306e-04	 -5.060299e-01	 -9.043267e-01	  2.574047e+02	  4.024915e-01	  2.928479e-02	 -1.781965e-02	 -5.740581e-01	  1.280398e-05
  7.000000e+02	 -1.869617e+02	  1.885306e-04	 -4.599132e-01	 -9.043267e-01	  2.670732e+02	  3.327464e-01	  2.876549e-02	 -1.735534e-02	 -5.867627e-01	  1.257591e-05
  7.200000e+02	 -1.869925e+02	  1.885306e-04	 -4.137965e-01	 -9.043267e-01	  2.758194e+02	  2.630012e-01	  2.824611e-02	 -1.689124e-02	 -5.983784e-01	  1.234783e-05
  7.400000e+02	 -1.870232e+02	  1.885306e-04	 -3.676798e-01	 -9.043267e-01	  2.836433e+02	  1.932560e-01	  2.772664e-02	 -1.642764e-02	 -6.088695e-01	  1.211975e-05
  7.600000e+02	 -1.870539e+02	  1.885306e-04	 -3.215631e-01	 -9.043267e-01	  2.905448e+02	  1.235108e-01	  2.720709e-02	 -1.596481e-02	 -6.182043e-01	  1.189168e-05
  7.800000e+02	 -1.870847e+02	  1.885306e-04	 -2.754464e-01	 -9.043267e-01	  2.965240e+02	  5.376558e-02	  2.668746e-02	 -1.550304e-02	 -6.263546e-01	  1.166360e-05
  8.000000e+02	 -1.871154e+02	  1.885306e-04	 -2.293297e-01	 -9.043267e-01	  3.015808e+02	 -1.597961e-02	  2.616773e-02	 -1.504262e-02	 -6.332963e-01	  1.143552e-05
  8.200000e+02	 -1.871462e+02	  1.885306e-04	 -1.832130e-01	 -9.043267e-01	  3.057154e+02	 -8.572480e-02	  2.564793e-02	 -1.458382e-02	 -6.390087e-01	  1.120744e-05
  8.400000e+02	 -1.871769e+02	  1.885306e-04	 -1.370963e-01	 -9.043267e-01	  3.089275e+02	 -1.554700e-01	  2.512803e-02	 -1.412693e-02	 -6.434750e-01	  1.097937e-05
  8.600000e+02	 -1.872077e+02	  1.885306e-04	 -9.097956e-02	 -9.043267e-01	  3.112174e+02	 -2.252152e-01	  2.460805e-02	 -1.367224e-02	 -6.466822e-01	  1.075129e-05
  8.800000e+02	 -1.872384e+02	  1.885306e-04	 -4.486286e-02	 -9.043267e-01	  3.125849e+02	 -2.949604e-01	  2.408799e-02	 -1.322003e-02	 -6.486210e-01	  1.052321e-05
  9.000000e+02	 -1.872692e+02	  1.885306e-04	  1.253838e-03	 -9.043267e-01	  3.130301e+02	 -3.647056e-01	  2.356784e-02	 -1.277057e-02	 -6.492858e-01	  1.029513e-05
  9.200000e+02	 -1.872999e+02	  1.885306e-04	  4.737054e-02	 -9.043267e-01	  3.125530e+02	 -4.344508e-01	  2.304761e-02	 -1.232416e-02	 -6.486748e-01	  1.006706e-05
  9.400000e+02	 -1.873306e+02	  1.885306e-04	  9.348724e-02	 -9.043267e-01	  3.111535e+02	 -5.041960e-01	  2.252729e-02	 -1.188108e-02	 -6.467899e-01	  9.838979e-06
  9.600000e+02	 -1.873614e+02	  1.885306e-04	  1.396039e-01	 -9.043267e-01	  3.088316e+02	 -5.739412e-01	  2.200688e-02	 -1.144161e-02	 -6.436368e-01	  9.610902e-06
  9.800000e+02	 -1.873921e+02	  1.885306e-04	  1.857206e-01	 -9.043267e-01	  3.055875e+02	 -6.436864e-01	  2.148639e-02	 -1.100604e-02	 -6.392250e-01	  9.382824e-06
  1.000000e+03	 -1.874229e+02	  1.885306e-04	  2.318373e-01	 -9.043267e-01	  3.014210e+02	 -7.134315e-01	  2.096581e-02	 -1.057465e-02	 -6.335676e-01	  9.154747e-06
  1.020000e+03	 -1.874536e+02	  1.885306e-04	  2.779540e-01	 -9.043267e-01	  2.963322e+02	 -7.831767e-01	  2.044515e-02	 -1.014771e-02	 -6.266817e-01	  8.926670e-06
  1.040000e+03	 -1.874844e+02	  1.885306e-04	  3.240707e-01	 -9.043267e-01	  2.903210e+02	 -8.529219e-01	  1.992440e-02	 -9.725527e-03	 -6.185878e-01	  8.698592e-06
  1.060000e+03	 -1.875151e+02	  1.885306e-04	  3.701874e-01	 -9.043267e-01	  2.833875e+02	 -9.226671e-01	  1.940357e-02	 -9.308370e-03	 -6.093104e-01	  8.470515e-06
  1.080000e+03	 -1.875459e+02	  1.885306e-04	  4.163041e-01	 -9.043267e-01	  2.755317e+02	 -9.924123e-01	  1.888265e-02	 -8.896525e-03	 -5.988777e-01	  8.242438e-06
  1.100000e+03	 -1.875766e+02	  1.885306e-04	  4.624208e-01	 -9.043267e-01	  2.667536e+02	 -1.062158e+00	  1.836165e-02	 -8.490277e-03	 -5.873217e-01	  8.014360e-06
  1.120000e+03	 -1.876073e+02	  1.885306e-04	  5.085375e-01	 -9.043267e-01	  2.570531e+02	 -1.131903e+00	  1.784056e-02	 -8.089909e-03	 -5.746779e-01	  7.786283e-06
  1.140000e+03	 -1.876381e+02	  1.885306e-04	  5.546542e-01	 -9.043267e-01	  2.464303e+02	 -1.201648e+00	  1.731939e-02	 -7.695704e-03	 -5.609859e-01	  7.558206e-06
  1.160000e+03	 -1.876688e+02	  1.885306e-04	  6.007709e-01	 -9.043267e-01	  2.348851e+02	 -1.271393e+00	  1.679813e-02	 -7.307947e-03	 -5.462888e-01	  7.330128e-06
  1.180000e+03	 -1.876996e+02	  1.885306e-04	  6.468876e-01	 -9.043267e-01	  2.224176e+02	 -1.341138e+00	  1.627678e-02	 -6.926919e-03	 -5.306336e-01	  7.102051e-06
  1.200000e+03	 -1.877303e+02	  1.885306e-04	  6.930043e-01	 -9.043267e-01	  2.090278e+02	 -1.410883e+00	  1.575535e-02	 -6.552906e-03	 -5.140709e-01	  6.873974e-06
  1.220000e+03	 -1.877611e+02	  1.885306e-04	  7.391210e-01	 -9.043267e-01	  1.947156e+02	 -1.480629e+00	  1.523384e-02	 -6.186190e-03	 -4.966552e-01	  6.645896e-06
  1.240000e+03	 -1.877918e+02	  1.885306e-04	  7.852377e-01	 -9.043267e-01	  1.794811e+02	 -1.550374e+00	  1.471223e-02	 -5.827055e-03	 -4.784447e-01	  6.417819e-06
  1.260000e+03	 -1.878226e+02	  1.885306e-04	  8.313544e-01	 -9.043267e-01	  1.633243e+02	 -1.620119e+00	  1.419055e-02	 -5.475785e-03	 -4.595012e-01	  6.189741e-06
  1.280000e+03	 -1.878533e+02	  1.885306e-04	  8.774711e-01	 -9.043267e-01	  1.462451e+02	 -1.689864e+00	  1.366878e-02	 -5.132662e-03	 -4.398905e-01	  5.961664e-06
  1.300000e+03	 -1.878840e+02	  1.885306e-04	  9.235878e-01	 -9.043267e-01	  1.282436e+02	 -1.759609e+00	  1.314692e-02	 -4.797972e-03	 -4.196819e-01	  5.733587e-06
  1.320000e+03	 -1.879148e+02	  1.885306e-04	  9.697045e-01	 -9.043267e-01	  1.093198e+02	 -1.829355e+00	  1.262498e-02	 -4.471996e-03	 -3.989488e-01	  5.505509e-06
  1.340000e+03	 -1.879455e+02	  1.885306e-04	  1.015821e+00	 -9.043267e-01	  8.947364e+01	 -1.899100e+00	  1.210295e-02	 -4.155019e-03	 -3.777679e-01	  5.277432e-06
  1.360000e+03	 -1.879763e+02	  1.885306e-04	  1.061938e+00	 -9.043267e-01	  6.870514e+01	 -1.968845e+00	  1.158083e-02	 -3.847324e-03	 -3.562200e-01	  5.049355e-06
  1.380000e+03	 -1.880070e+02	  1.885306e-04	  1.108055e+00	 -9.043267e-01	  4.701431e+01	 -2.038590e+00	  1.105863e-02	 -3.549194e-03	 -3.343894e-01	  4.821277e-06
  1.400000e+03	 -1.880378e+02	  1.885306e-04	  1.154171e+00	 -9.043267e-01	  2.440114e+01	 -2.108335e+00	  1.053635e-02	 -3.260914e-03	 -3.123644e-01	  4.593200e-06
  1.420000e+03	 -1.880685e+02	  1.885306e-04	  1.200288e+00	 -9.043267e-01	  8.656435e-01	 -2.178081e+00	  1.001398e-02	 -2.982767e-03	 -2.902369e-01	  4.365123e-06
  1.440000e+03	 -1.880993e+02	  1.885306e-04	  1.246405e+00	 -9.043267e-01	 -2.359219e+01	 -2.247826e+00	  9.491525e-03	 -2.715035e-03	 -2.681026e-01	  4.137045e-06
  1.460000e+03	 -1.881300e+02	  1.885306e-04	  1.292521e+00	 -9.043267e-01	 -4.897236e+01	 -2.317571e+00	  8.968984e-03	 -2.458003e-03	 -2.460609e-01	  3.908968e-06
  1.480000e+03	 -1.881607e+02	  1.885306e-04	  1.338638e+00	 -9.043267e-01	 -7.527486e+01	 -2.387316e+00	  8.446358e-03	 -2.211955e-03	 -2.242148e-01	  3.680891e-06
  1.500000e+03	 -1.881915e+02	  1.885306e-04	  1.384755e+00	 -9.043267e-01	 -1.024997e+02	 -2.457061e+00	  7.923647e-03	 -1.977173e-03	 -2.026715e-01	  3.452813e-06
  1.520000e+03	 -1.882222e+02	  1.885306e-04	  1.430872e+00	 -9.043267e-01	 -1.306469e+02	 -2.526807e+00	  7.400850e-03	 -1.753941e-03	 -1.815415e-01	  3.224736e-06
  1.540000e+03	 -1.882530e+02	  1.885306e-04	  1.476988e+00	 -9.043267e-01	 -1.597164e+02	 -2.596552e+00	  6.877968e-03	 -1.542544e-03	 -1.609393e-01	  2.996659e-06
  1.560000e+03	 -1.882837e+02	  1.885306e-04	  1.523105e+00	 -9.043267e-01	 -1.897082e+02	 -2.666297e+00	  6.355000e-03	 -1.343263e-03	 -1.409830e-01	  2.768581e-06
  1.580000e+03	 -1.883145e+02	  1.885306e-04	  1.569222e+00	 -9.043267e-01	 -2.206224e+02	 -2.736042e+00	  5.831947e-03	 -1.156383e-03	 -1.217945e-01	  2.540504e-06
  1.600000e+03	 -1.883452e+02	  1.885306e-04	  1.615338e+00	 -9.043267e-01	 -2.524589e+02	 -2.805787e+00	  5.308809e-03	 -9.821879e-04	 -1.034996e-01	  2.312427e-06
  1.620000e+03	 -1.883760e+02	  1.885306e-04	  1.661455e+00	 -9.043267e-01	 -2.852177e+02	 -2.875533e+00	  4.785585e-03	 -8.209602e-04	 -8.622760e-02	  2.084349e-06
  1.640000e+03	 -1.884067e+02	  1.885306e-04	  1.707572e+00	 -9.043267e-01	 -3.188989e+02	 -2.945278e+00	  4.262276e-03	 -6.729839e-04	 -7.011169e-02	  1.856272e-06
  1.660000e+03	 -1.884374e+02	  1.885306e-04	  1.753688e+00	 -9.043267e-01	 -3.535024e+02	 -3.015023e+00	  3.738881e-03	 -5.385425e-04	 -5.528880e-02	  1.628194e-06
  1.680000e+03	 -1.884682e+02	  1.885306e-04	  1.799805e+00	 -9.043267e-01	 -3.890282e+02	 -3.084768e+00	  3.215401e-03	 -4.179195e-04	 -4.189958e-02	  1.400117e-06
  1.700000e+03	 -1.884989e+02	  1.885306e-04	  1.845922e+00	 -9.043267e-01	 -4.254764e+02	 -3.154513e+00	  2.691836e-03	 -3.113983e-04	 -3.008846e-02	  1.172040e-06
  1.720000e+03	 -1.885297e+02	  1.885306e-04	  1.892039e+00	 -9.043267e-01	 -4.628469e+02	 -3.224259e+00	  2.168185e-03	 -2.192625e-04	 -2.000359e-02	  9.439624e-07
  1.740000e+03	 -1.885604e+02	  1.885306e-04	  1.938155e+00	 -9.043267e-01	 -5.011398e+02	 -3.294004e+00	  1.644449e-03	 -1.417957e-04	 -1.179689e-02	  7.158851e-07
  1.760000e+03	 -1.885912e+02	  1.885306e-04	  1.984272e+00	 -9.043267e-01	 -5.403550e+02	 -3.363749e+00	  1.120627e-03	 -7.928135e-05	 -5.624016e-03	  4.878077e-07
  1.780000e+03	 -1.886219e+02	  1.885306e-04	  2.030389e+00	 -9.043267e-01	 -5.804925e+02	 -3.433494e+00	  5.967199e-04	 -3.200293e-05	 -1.644387e-03	  2.597304e-07
  1.802776e+03	 -1.886569e+02	  1.885306e-04	  2.082905e+00	 -9.043267e-01	 -6.273248e+02	 -3.503763e+00	  0.000000e+00	  0.000000e+00	  0.000000e+00	  0.000000e+00
        """
        )

        out = np.loadtxt(output)

        np.testing.assert_array_almost_equal(intF[iE].x[iCase, :], out[:, 0], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Nx[iCase, :], out[:, 1], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Vy[iCase, :], out[:, 2], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Vz[iCase, :], out[:, 3], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Tx[iCase, :], out[:, 4], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].My[iCase, :], out[:, 5], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Mz[iCase, :], out[:, 6], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Dx[iCase, :], out[:, 7], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Dy[iCase, :], out[:, 8], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Dz[iCase, :], out[:, 9], decimal=3)
        np.testing.assert_array_almost_equal(intF[iE].Rx[iCase, :], out[:, 10], decimal=3)

    def test_mass(self):

        mass = self.mass

        string = StringIO(
            """
         1 1.00723e-01 1.00738e-01 1.00733e-01 3.51392e+01 4.73634e+01 4.36768e+01
         2 1.26482e+02 1.26482e+02 1.26482e+02 1.26482e+02 1.26482e+02 1.26482e+02
         3 1.26482e+02 1.26482e+02 1.26482e+02 1.26482e+02 1.26482e+02 1.26482e+02
         4 1.26482e+02 1.26482e+02 1.26482e+02 1.26482e+02 1.26482e+02 1.26482e+02
         5 1.26482e+02 1.26482e+02 1.26482e+02 1.26482e+02 1.26482e+02 1.26482e+02
        """
        )
        out = np.loadtxt(string)

        np.testing.assert_almost_equal(mass.total_mass, 1.020379e-01, decimal=6)
        np.testing.assert_almost_equal(mass.struct_mass, 2.037858e-03, decimal=6)
        np.testing.assert_array_equal(mass.node, out[:, 0])
        np.testing.assert_array_almost_equal(mass.xmass, out[:, 1], decimal=3)
        np.testing.assert_array_almost_equal(mass.ymass, out[:, 2], decimal=3)
        np.testing.assert_array_almost_equal(mass.zmass, out[:, 3], decimal=3)
        np.testing.assert_array_almost_equal(mass.xinrta, out[:, 4], decimal=3)
        np.testing.assert_array_almost_equal(mass.yinrta, out[:, 5], decimal=3)
        np.testing.assert_array_almost_equal(mass.zinrta, out[:, 6], decimal=3)

    def test_modal(self):

        modal = self.modal
        iM = 0

        string = StringIO(
            """
     1  -2.609e-02   1.369e-04   6.904e-06   3.108e-04   1.344e-01  -5.727e-02
     2  -5.684e-09  -6.833e-10   6.882e-10   4.093e-07  -1.721e-06   1.211e-06
     3  -5.706e-09   7.092e-10  -6.812e-10  -4.137e-07  -1.724e-06   1.206e-06
     4  -3.930e-09  -3.054e-09  -7.138e-10   1.137e-06  -1.151e-06  -9.706e-08
     5  -3.948e-09   3.079e-09   7.073e-10  -1.142e-06  -1.155e-06  -9.307e-08
        """
        )
        out = np.loadtxt(string)

        np.testing.assert_almost_equal(modal.freq[iM], 18.807942, decimal=5)
        np.testing.assert_almost_equal(modal.xmpf[iM], -2.5467e-02, decimal=6)
        np.testing.assert_almost_equal(modal.ympf[iM], 6.6618e-05, decimal=9)
        np.testing.assert_almost_equal(modal.zmpf[iM], 6.9752e-07, decimal=11)
        np.testing.assert_array_equal(modal.node[iM, :], out[:, 0])
        np.testing.assert_array_almost_equal(modal.xdsp[iM, :], out[:, 1], decimal=3)
        np.testing.assert_array_almost_equal(modal.ydsp[iM, :], out[:, 2], decimal=3)
        np.testing.assert_array_almost_equal(modal.zdsp[iM, :], out[:, 3], decimal=3)
        np.testing.assert_array_almost_equal(modal.xrot[iM, :], out[:, 4], decimal=3)
        np.testing.assert_array_almost_equal(modal.yrot[iM, :], out[:, 5], decimal=3)
        np.testing.assert_array_almost_equal(modal.zrot[iM, :], out[:, 6], decimal=3)


class GravityAdd(unittest.TestCase):
    def test_addgrav_working(self):

        # nodes
        nnode = 3
        node = np.arange(1, 1 + nnode)
        x = np.zeros(nnode)
        y = np.zeros(nnode)
        z = 10.0 * np.arange(nnode)
        r = np.zeros(nnode)
        nodes = NodeData(node, x, y, z, r)

        # reactions
        rigid = 1e16
        node = np.array([1])
        Kx = Ky = Kz = Ktx = Kty = Ktz = np.array([rigid])
        reactions = ReactionData(node, Kx, Ky, Kz, Ktx, Kty, Ktz, rigid)

        # elements
        EL = np.arange(1, nnode)
        N1 = np.arange(1, nnode)
        N2 = np.arange(2, nnode + 1)
        Ax = 5.0 * np.ones(nnode - 1)
        Asy = 1.0 * np.ones(nnode - 1)
        Asz = 1.0 * np.ones(nnode - 1)
        Jx = 1.0 * np.ones(nnode - 1)
        Iy = 1.0 * np.ones(nnode - 1)
        Iz = 0.5 * np.ones(nnode - 1)
        E = 1e5 * np.ones(nnode - 1)
        G = 1e4 * np.ones(nnode - 1)
        roll = np.zeros(nnode - 1)
        density = 0.25 * np.ones(nnode - 1)
        elements = ElementData(EL, N1, N2, Ax, Asy, Asz, Jx, Iy, Iz, E, G, roll, density)
        mymass = np.sum(density * Ax * np.diff(z))

        # parameters
        shear = False  # 1: include shear deformation
        geom = False  # 1: include geometric stiffness
        dx = 1.0  # x-axis increment for internal forces
        options = Options(shear, geom, dx)

        frame = Frame(nodes, reactions, elements, options)

        N = np.array([nnode])
        EMs = np.array([1e3])
        EMxx = EMyy = EMzz = EMxy = EMxz = EMyz = np.array([0.0])
        rhox = rhoy = rhoz = np.array([0.0])
        addGravityLoad = True
        frame.changeExtraNodeMass(N, EMs, EMxx, EMyy, EMzz, EMxy, EMxz, EMyz, rhox, rhoy, rhoz, addGravityLoad)

        # dynamics
        nM = 1  # number of desired dynamic modes of vibration
        Mmethod = 1  # 1: subspace Jacobi     2: Stodola
        lump = 0  # 0: consistent mass ... 1: lumped mass matrix
        tol = 1e-9  # mode shape tolerance
        shift = 0.0  # shift value ... for unrestrained structures
        frame.enableDynamics(nM, Mmethod, lump, tol, shift)

        # Load case 1: Added mass with    gravity field and NO point loading
        # Load case 2: Added mass with NO gravity field and    point loading
        # Load case 3: Added mass with    gravity field and    point loading
        gx = gy = 0.0
        gz = -10.0
        load1 = StaticLoadCase(gx, gy, gz)
        load2 = StaticLoadCase(gx, gy, 0.0)
        load3 = StaticLoadCase(gx, gy, gz)

        nF = np.array([nnode])
        Fx = Fy = Mxx = Myy = Mzz = np.array([0.0])
        Fz = np.array([gz * (mymass + EMs)])
        load2.changePointLoads(nF, Fx, Fy, Fz, Mxx, Myy, Mzz)
        load3.changePointLoads(nF, Fx, Fy, Fz, Mxx, Myy, Mzz)

        frame.addLoadCase(load1)
        frame.addLoadCase(load2)
        frame.addLoadCase(load3)

        displacements, forces, reactions, internalForces, mass, modal = frame.run()

        # Check that the mass was added
        self.assertEqual(mass.struct_mass, mymass)
        self.assertEqual(mass.total_mass, mymass + EMs)

        # Check that the load cases are equivalent
        self.assertEqual(reactions.Fz[0, 0], reactions.Fz[1, 0])
        self.assertAlmostEqual(2 * reactions.Fz[0, 0], reactions.Fz[2, 0])


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(FrameTestEXA))
    suite.addTest(unittest.makeSuite(FrameTestEXB))
    suite.addTest(unittest.makeSuite(GravityAdd))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
