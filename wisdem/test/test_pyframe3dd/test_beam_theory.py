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
import numpy.testing as npt

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
        anal = beta**2 / L**2 * np.sqrt(E[0] * Iy[0] / (rho[0] * Ax[0])) / np.pi / 2

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
        anal = beta**2 / L**2 * np.sqrt(E[0] * Iy[0] / (rho[0] * Ax[0])) / np.pi / 2
        freq = modal.freq
        freq = freq[freq > 1e-1]
        self.assertAlmostEqual(freq[0], anal[0], -1)
        self.assertAlmostEqual(freq[1], anal[0], -1)
        self.assertAlmostEqual(freq[2], anal[1], -1)
        self.assertAlmostEqual(freq[3], anal[1], -1)
        self.assertAlmostEqual(freq[4], anal[2], -2)
        self.assertAlmostEqual(freq[5], anal[2], -2)

    def test_mpfs(self):
        node_raw = StringIO(
            """
            1	0.0	0.0	15.0	0.0
            2	0.0	0.0	24.485630720316095	0.0
            3	0.0	0.0	33.97126144063219	0.0
            4	0.0	0.0	43.456892160948286	0.0
            5	0.0	0.0	52.942522881264374	0.0
            6	0.0	0.0	62.42815360158047	0.0
            7	0.0	0.0	71.91378432189657	0.0
            8	0.0	0.0	81.39941504221267	0.0
            9	0.0	0.0	90.88504576252875	0.0
            10	0.0	0.0	100.37067648284484	0.0
            11	0.0	0.0	109.85630720316094	0.0
            12	0.0	0.0	113.30927648284484	0.0
            13	0.0	0.0	116.76224576252875	0.0
            14	0.0	0.0	120.21521504221265	0.0
            15	0.0	0.0	123.66818432189656	0.0
            16	0.0	0.0	127.12115360158046	0.0
            17	0.0	0.0	130.57412288126437	0.0
            18	0.0	0.0	134.02709216094826	0.0
            19	0.0	0.0	137.4800614406322	0.0
            20	0.0	0.0	140.93303072031608	0.0
            21	0.0	0.0	144.386	0.0
            """
        )
        node_data = np.loadtxt(node_raw)

        element_raw = StringIO(
            """
            1	1	2	1.8877546659578743	1.073959738935436	1.073959738935436	46.66380368737871	23.331901843689355	23.331901843689355	200000000000.0	79300000000.0	0.0	7800.0
            2	2	3	1.8877546659578743	1.073959738935436	1.073959738935436	46.66380368737871	23.331901843689355	23.331901843689355	200000000000.0	79300000000.0	0.0	7800.0
            3	3	4	1.8877546659578743	1.073959738935436	1.073959738935436	46.66380368737871	23.331901843689355	23.331901843689355	200000000000.0	79300000000.0	0.0	7800.0
            4	4	5	1.8877546659578743	1.073959738935436	1.073959738935436	46.66380368737871	23.331901843689355	23.331901843689355	200000000000.0	79300000000.0	0.0	7800.0
            5	5	6	1.8877546659578743	1.073959738935436	1.073959738935436	46.66380368737871	23.331901843689355	23.331901843689355	200000000000.0	79300000000.0	0.0	7800.0
            6	6	7	1.8877546659578743	1.073959738935436	1.073959738935436	46.66380368737871	23.331901843689355	23.331901843689355	200000000000.0	79300000000.0	0.0	7800.0
            7	7	8	1.8877546659578743	1.073959738935436	1.073959738935436	46.66380368737871	23.331901843689355	23.331901843689355	200000000000.0	79300000000.0	0.0	7800.0
            8	8	9	1.8877546659578743	1.073959738935436	1.073959738935436	46.66380368737871	23.331901843689355	23.331901843689355	200000000000.0	79300000000.0	0.0	7800.0
            9	9	10	1.8877546659578743	1.073959738935436	1.073959738935436	46.66380368737871	23.331901843689355	23.331901843689355	200000000000.0	79300000000.0	0.0	7800.0
            10	10	11	1.8877546659578743	1.073959738935436	1.073959738935436	46.66380368737871	23.331901843689355	23.331901843689355	200000000000.0	79300000000.0	0.0	7800.0
            11	11	12	0.8434632263529995	0.4791921604013138	0.4791921604013138	20.249187881226895	10.124593940613448	10.124593940613448	200000000000.0	79300000000.0	0.0	7800.0
            12	12	13	0.8133376783690309	0.4620970046088699	0.4620970046088699	18.156078445765353	9.078039222882676	9.078039222882676	200000000000.0	79300000000.0	0.0	7800.0
            13	13	14	0.7832121303850624	0.4450018640800275	0.4450018640800275	16.212419808704094	8.106209904352047	8.106209904352047	200000000000.0	79300000000.0	0.0	7800.0
            14	14	15	0.7530865824010938	0.4279067406303197	0.4279067406303197	14.412676400601093	7.2063382003005465	7.2063382003005465	200000000000.0	79300000000.0	0.0	7800.0
            15	15	16	0.7229610344171373	0.41081163637509993	0.41081163637509993	12.751312652014308	6.375656326007154	6.375656326007154	200000000000.0	79300000000.0	0.0	7800.0
            16	16	17	0.6928354864331688	0.3937165537940666	0.3937165537940666	11.222792993501606	5.611396496750803	5.611396496750803	200000000000.0	79300000000.0	0.0	7800.0
            17	17	18	0.6627099384492063	0.37662149581328724	0.37662149581328724	9.821581855621094	4.910790927810547	4.910790927810547	200000000000.0	79300000000.0	0.0	7800.0
            18	18	19	0.6325843904652436	0.3595264659103174	0.3595264659103174	8.542143668930589	4.271071834465294	4.271071834465294	200000000000.0	79300000000.0	0.0	7800.0
            19	19	20	0.6024588424812811	0.3424314682505728	0.3424314682505728	7.378942863988056	3.689471431994028	3.689471431994028	200000000000.0	79300000000.0	0.0	7800.0
            20	20	21	0.5723332944973126	0.3253365078662345	0.3253365078662345	6.326443871351455	3.1632219356757276	3.1632219356757276	200000000000.0	79300000000.0	0.0	7800.0
            """
        )
        element_data = np.loadtxt(element_raw)

        mass_raw = StringIO(
            """
            1	1556180638.0051677	1826303224093.325	1826303072282.6616	3209340708097.124
            20	957941.2855308752	384186621.1071053	259960402.33780164	241361567.37433502
            """
        )
        mass_data = np.loadtxt(mass_raw)

        # nodes
        nn = node_data.shape[0]
        node = np.arange(1, nn + 1, dtype=np.int_)
        x = y = r = np.zeros(nn)
        z = node_data[:, 3]
        nodes = NodeData(node, x, y, z, r)

        # reactions
        rnode = np.array([], dtype=np.int_)
        Kx = Ky = Kz = Ktx = Kty = Ktz = rnode
        rigid = 1
        reactions = ReactionData(rnode, Kx, Ky, Kz, Ktx, Kty, Ktz, rigid)

        # elements
        ne = element_data.shape[0]
        EL = np.arange(1, ne + 1, dtype=np.int_)
        N1 = np.arange(1, nn, dtype=np.int_)
        N2 = np.arange(2, nn + 1, dtype=np.int_)
        Ax = element_data[:, 3]
        Asy = element_data[:, 4]
        Asz = element_data[:, 5]
        Jx = element_data[:, 6]
        Iy = element_data[:, 7]
        Iz = element_data[:, 8]
        E = element_data[:, 9]
        G = element_data[:, 10]
        roll = element_data[:, 11]
        rho = element_data[:, 12]
        elements = ElementData(EL, N1, N2, Ax, Asy, Asz, Jx, Iy, Iz, E, G, roll, rho)

        # parameters
        shear = False  # 1: include shear deformation
        geom = False  # 1: include geometric stiffness
        dx = -1.0  # x-axis increment for internal forces
        options = Options(shear, geom, dx)

        #### Fixed-free
        frame = Frame(nodes, reactions, elements, options)

        # Extra mass
        nm = np.array(mass_data[:, 0], dtype=np.int_)
        EMs = mass_data[:, 1]
        EMxx = mass_data[:, 2]
        EMyy = mass_data[:, 3]
        EMzz = mass_data[:, 4]
        EMxy = EMxz = EMyz = np.array([0.0])
        rhox = rhoy = rhoz = np.array([0.0])
        addGravityLoad = False
        frame.changeExtraNodeMass(nm, EMs, EMxx, EMyy, EMzz, EMxy, EMxz, EMyz, rhox, rhoy, rhoz, addGravityLoad)

        # dynamics
        nM = 18  # number of desired dynamic modes of vibration
        Mmethod = 2  # 1: subspace Jacobi     2: Stodola
        lump = 0  # 0: consistent mass ... 1: lumped mass matrix
        tol = 1e-9  # mode shape tolerance
        shift = 10  # shift value ... for unrestrained structures
        frame.enableDynamics(nM, Mmethod, lump, tol, shift)

        # load cases 1
        gx = 0.0
        gy = 0.0
        gz = -9.80633

        load = StaticLoadCase(gx, gy, gz)

        frame.addLoadCase(load)

        _, _, _, _, _, modal = frame.run()

        mpfs = np.abs(np.c_[modal.xmpf, modal.ympf, modal.zmpf])
        ratios = mpfs.max(axis=1) / (mpfs.min(axis=1) + 1e-15)

        npt.assert_array_less(modal.freq[:6], 0.05)
        npt.assert_array_less(1e3, ratios[6:11])


if __name__ == "__main__":
    unittest.main()

