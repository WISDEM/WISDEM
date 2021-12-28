#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Andrew Ning on 2013-11-04.
Copyright (c) NREL. All rights reserved.
"""

import unittest

import numpy as np
from wisdem.pyframe3dd import Frame, Options, NodeData, ElementData, ReactionData, StaticLoadCase


class FrameTestEXA(unittest.TestCase):
    def testFixedFree_FreeFree(self):

        # nodes
        nn = 11
        node = np.arange(1, nn + 1, dtype=np.int_)
        x = y = r = np.zeros(nn)
        z = np.linspace(0, 100, nn)
        nodes = NodeData(node, x, y, z, r)

        # reactions
        rnode = np.array([1], dtype=np.int_)
        Kx = Ky = Kz = Ktx = Kty = Ktz = np.ones(1)
        rigid = 1
        reactions = ReactionData(rnode, Kx, Ky, Kz, Ktx, Kty, Ktz, rigid)

        # reactions
        rnode = np.array([], dtype=np.int_)
        Kx = Ky = Kz = Ktx = Kty = Ktz = rnode  # np.zeros(1)
        reactions0 = ReactionData(rnode, Kx, Ky, Kz, Ktx, Kty, Ktz, rigid)

        # elements
        ne = nn - 1
        EL = np.arange(1, ne + 1, dtype=np.int_)
        N1 = np.arange(1, nn, dtype=np.int_)
        N2 = np.arange(2, nn + 1, dtype=np.int_)
        Ax = Jx = Iy = Iz = 10 * np.ones(ne)
        Asy = Asz = 8 * np.ones(ne)
        E = 2e6 * np.ones(ne)
        G = 1e6 * np.ones(ne)
        roll = np.zeros(ne)
        rho = 1e-5 * np.ones(ne)
        elements = ElementData(EL, N1, N2, Ax, Asy, Asz, Jx, Iy, Iz, E, G, roll, rho)

        # parameters
        shear = False  # 1: include shear deformation
        geom = False  # 1: include geometric stiffness
        dx = -1.0  # x-axis increment for internal forces
        options = Options(shear, geom, dx)

        #### Fixed-free
        frame = Frame(nodes, reactions, elements, options)

        # dynamics
        nM = 15  # number of desired dynamic modes of vibration
        Mmethod = 1  # 1: subspace Jacobi     2: Stodola
        lump = 0  # 0: consistent mass ... 1: lumped mass matrix
        tol = 1e-9  # mode shape tolerance
        shift = -1e3  # shift value ... for unrestrained structures
        frame.enableDynamics(nM, Mmethod, lump, tol, shift)

        # load cases 1
        gx = 0.0
        gy = 0.0
        gz = -980.6

        load = StaticLoadCase(gx, gy, gz)

        frame.addLoadCase(load)

        displacements, forces, rxns, internalForces, mass, modal = frame.run()

        L = z.max() - z.min()
        beta = np.array([1.875194, 4.69361268, 7.85429819])
        anal = beta ** 2 / L ** 2 * np.sqrt(E[0] * Iy[0] / (rho[0] * Ax[0])) / np.pi / 2

        self.assertAlmostEqual(modal.freq[0], anal[0], 1)
        self.assertAlmostEqual(modal.freq[1], anal[0], 1)
        self.assertAlmostEqual(modal.freq[2], anal[1], 0)
        self.assertAlmostEqual(modal.freq[3], anal[1], 0)
        self.assertAlmostEqual(modal.freq[4], anal[2], -1)
        self.assertAlmostEqual(modal.freq[5], anal[2], -1)

        #### Free-free
        frame0 = Frame(nodes, reactions0, elements, options)
        frame0.enableDynamics(nM, Mmethod, lump, tol, shift)
        frame0.addLoadCase(load)
        displacements, forces, rxns, internalForces, mass, modal = frame0.run(nanokay=True)

        beta = np.array([4.72969344, 7.85302489, 10.99545361])
        anal = beta ** 2 / L ** 2 * np.sqrt(E[0] * Iy[0] / (rho[0] * Ax[0])) / np.pi / 2
        freq = modal.freq
        freq = freq[freq > 1e-1]
        self.assertAlmostEqual(freq[0], anal[0], -1)
        self.assertAlmostEqual(freq[1], anal[0], -1)
        self.assertAlmostEqual(freq[2], anal[1], -1)
        self.assertAlmostEqual(freq[3], anal[1], -1)
        self.assertAlmostEqual(freq[4], anal[2], -2)
        self.assertAlmostEqual(freq[5], anal[2], -2)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(FrameTestEXA))
    return suite


if __name__ == "__main__":
    unittest.TextTestRunner().run(suite())
