"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import unittest

import numpy as np

from wisdem.ccblade.ccblade import CCBlade, CCAirfoil

basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../examples/_airfoil_files/")


class TestGradients(unittest.TestCase):
    def setUp(self):

        # geometry
        self.Rhub = 1.5
        self.Rtip = 63.0

        self.r = np.array(
            [
                2.8667,
                5.6000,
                8.3333,
                11.7500,
                15.8500,
                19.9500,
                24.0500,
                28.1500,
                32.2500,
                36.3500,
                40.4500,
                44.5500,
                48.6500,
                52.7500,
                56.1667,
                58.9000,
                61.6333,
            ]
        )
        self.chord = np.array(
            [
                3.542,
                3.854,
                4.167,
                4.557,
                4.652,
                4.458,
                4.249,
                4.007,
                3.748,
                3.502,
                3.256,
                3.010,
                2.764,
                2.518,
                2.313,
                2.086,
                1.419,
            ]
        )
        self.theta = np.array(
            [
                13.308,
                13.308,
                13.308,
                13.308,
                11.480,
                10.162,
                9.011,
                7.795,
                6.544,
                5.361,
                4.188,
                3.125,
                2.319,
                1.526,
                0.863,
                0.370,
                0.106,
            ]
        )
        self.B = 3  # number of blades

        # atmosphere
        self.rho = 1.225
        self.mu = 1.81206e-5

        afinit = CCAirfoil.initFromAerodynFile  # just for shorthand

        # load all airfoils
        airfoil_types = [0] * 8
        airfoil_types[0] = afinit(basepath + os.sep + "Cylinder1.dat")
        airfoil_types[1] = afinit(basepath + os.sep + "Cylinder2.dat")
        airfoil_types[2] = afinit(basepath + os.sep + "DU40_A17.dat")
        airfoil_types[3] = afinit(basepath + os.sep + "DU35_A17.dat")
        airfoil_types[4] = afinit(basepath + os.sep + "DU30_A17.dat")
        airfoil_types[5] = afinit(basepath + os.sep + "DU25_A17.dat")
        airfoil_types[6] = afinit(basepath + os.sep + "DU21_A17.dat")
        airfoil_types[7] = afinit(basepath + os.sep + "NACA64_A17.dat")

        # place at appropriate radial stations
        af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

        self.af = [0] * len(self.r)
        for i in range(len(self.r)):
            self.af[i] = airfoil_types[af_idx[i]]

        self.tilt = -5.0
        self.precone = 2.5
        self.yaw = 0.0
        self.shearExp = 0.2
        self.hubHt = 80.0
        self.nSector = 8

        # create CCBlade object
        self.rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
        )

        # set conditions
        self.Uinf = 10.0
        tsr = 7.55
        self.pitch = 0.0
        self.Omega = self.Uinf * tsr / self.Rtip * 30.0 / np.pi  # convert to RPM
        self.azimuth = 90

        loads, derivs = self.rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        self.Np = loads["Np"]
        self.Tp = loads["Tp"]
        self.dNp = derivs["dNp"]
        self.dTp = derivs["dTp"]

        outputs, derivs = self.rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        self.P, self.T, self.Y, self.Z, self.Q, self.My, self.Mz, self.Mb = [
            outputs[k] for k in ("P", "T", "Y", "Z", "Q", "My", "Mz", "Mb")
        ]
        self.dP, self.dT, self.dY, self.dZ, self.dQ, self.dMy, self.dMz, self.dMb = [
            derivs[k] for k in ("dP", "dT", "dY", "dZ", "dQ", "dMy", "dMz", "dMb")
        ]
        self.CP, self.CT, self.CY, self.CZ, self.CQ, self.CMy, self.CMz, self.CMb = [
            outputs[k] for k in ("CP", "CT", "CY", "CZ", "CQ", "CMy", "CMz", "CMb")
        ]
        self.dCP, self.dCT, self.dCY, self.dCZ, self.dCQ, self.dCMy, self.dCMz, self.dCMb = [
            derivs[k] for k in ("dCP", "dCT", "dCY", "dCZ", "dCQ", "dCMy", "dCMz", "dCMb")
        ]

        self.rotor.derivatives = False
        self.n = len(self.r)
        self.npts = 1  # len(Uinf)

    def test_dr1(self):

        dNp_dr = self.dNp["dr"]
        dTp_dr = self.dTp["dr"]
        dNp_dr_fd = np.zeros((self.n, self.n))
        dTp_dr_fd = np.zeros((self.n, self.n))

        for i in range(self.n):
            r = np.array(self.r)
            delta = 1e-6 * r[i]
            r[i] += delta

            rotor = CCBlade(
                r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dr_fd[:, i] = (Npd - self.Np) / delta
            dTp_dr_fd[:, i] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dr_fd, dNp_dr, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dr_fd, dTp_dr, rtol=1e-4, atol=1e-8)

    def test_dr2(self):

        dT_dr = self.dT["dr"]
        dY_dr = self.dY["dr"]
        dZ_dr = self.dZ["dr"]
        dQ_dr = self.dQ["dr"]
        dMy_dr = self.dMy["dr"]
        dMz_dr = self.dMz["dr"]
        dMb_dr = self.dMb["dr"]
        dP_dr = self.dP["dr"]
        dT_dr_fd = np.zeros((self.npts, self.n))
        dY_dr_fd = np.zeros((self.npts, self.n))
        dZ_dr_fd = np.zeros((self.npts, self.n))
        dQ_dr_fd = np.zeros((self.npts, self.n))
        dMy_dr_fd = np.zeros((self.npts, self.n))
        dMz_dr_fd = np.zeros((self.npts, self.n))
        dMb_dr_fd = np.zeros((self.npts, self.n))
        dP_dr_fd = np.zeros((self.npts, self.n))

        for i in range(self.n):
            r = np.array(self.r)
            delta = 1e-6 * r[i]
            r[i] += delta

            rotor = CCBlade(
                r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
            Pd = outputs["P"]
            Td = outputs["T"]
            Yd = outputs["Y"]
            Zd = outputs["Z"]
            Qd = outputs["Q"]
            Myd = outputs["My"]
            Mzd = outputs["Mz"]
            Mbd = outputs["Mb"]

            dT_dr_fd[:, i] = (Td - self.T) / delta
            dY_dr_fd[:, i] = (Yd - self.Y) / delta
            dZ_dr_fd[:, i] = (Zd - self.Z) / delta
            dQ_dr_fd[:, i] = (Qd - self.Q) / delta
            dMy_dr_fd[:, i] = (Myd - self.My) / delta
            dMz_dr_fd[:, i] = (Mzd - self.Mz) / delta
            dMb_dr_fd[:, i] = (Mbd - self.Mb) / delta
            dP_dr_fd[:, i] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dr_fd, dT_dr, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dr_fd, dY_dr, rtol=4e-3, atol=1e-8)
        np.testing.assert_allclose(dZ_dr_fd, dZ_dr, rtol=4e-3, atol=1e-8)
        np.testing.assert_allclose(dQ_dr_fd, dQ_dr, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dMy_dr_fd, dMy_dr, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dMz_dr_fd, dMz_dr, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dMb_dr_fd, dMb_dr, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dP_dr_fd, dP_dr, rtol=3e-4, atol=1e-8)

    def test_dr3(self):

        dCT_dr = self.dCT["dr"]
        dCY_dr = self.dCY["dr"]
        dCZ_dr = self.dCZ["dr"]
        dCQ_dr = self.dCQ["dr"]
        dCMy_dr = self.dCMy["dr"]
        dCMz_dr = self.dCMz["dr"]
        dCMb_dr = self.dCMb["dr"]
        dCP_dr = self.dCP["dr"]
        dCT_dr_fd = np.zeros((self.npts, self.n))
        dCY_dr_fd = np.zeros((self.npts, self.n))
        dCZ_dr_fd = np.zeros((self.npts, self.n))
        dCQ_dr_fd = np.zeros((self.npts, self.n))
        dCMy_dr_fd = np.zeros((self.npts, self.n))
        dCMz_dr_fd = np.zeros((self.npts, self.n))
        dCMb_dr_fd = np.zeros((self.npts, self.n))
        dCP_dr_fd = np.zeros((self.npts, self.n))

        for i in range(self.n):
            r = np.array(self.r)
            delta = 1e-6 * r[i]
            r[i] += delta

            rotor = CCBlade(
                r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
            CPd = outputs["CP"]
            CTd = outputs["CT"]
            CYd = outputs["CY"]
            CZd = outputs["CZ"]
            CQd = outputs["CQ"]
            CMyd = outputs["CMy"]
            CMzd = outputs["CMz"]
            CMbd = outputs["CMb"]

            dCT_dr_fd[:, i] = (CTd - self.CT) / delta
            dCY_dr_fd[:, i] = (CYd - self.CY) / delta
            dCZ_dr_fd[:, i] = (CZd - self.CZ) / delta
            dCQ_dr_fd[:, i] = (CQd - self.CQ) / delta
            dCMy_dr_fd[:, i] = (CMyd - self.CMy) / delta
            dCMz_dr_fd[:, i] = (CMzd - self.CMz) / delta
            dCMb_dr_fd[:, i] = (CMbd - self.CMb) / delta
            dCP_dr_fd[:, i] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dr_fd, dCT_dr, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dr_fd, dCY_dr, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dr_fd, dCZ_dr, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dr_fd, dCQ_dr, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMy_dr_fd, dCMy_dr, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMz_dr_fd, dCMz_dr, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMb_dr_fd, dCMb_dr, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dr_fd, dCP_dr, rtol=3e-4, atol=1e-8)

    def test_dchord1(self):

        dNp_dchord = self.dNp["dchord"]
        dTp_dchord = self.dTp["dchord"]
        dNp_dchord_fd = np.zeros((self.n, self.n))
        dTp_dchord_fd = np.zeros((self.n, self.n))

        for i in range(self.n):
            chord = np.array(self.chord)
            delta = 1e-6 * chord[i]
            chord[i] += delta

            rotor = CCBlade(
                self.r,
                chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dchord_fd[:, i] = (Npd - self.Np) / delta
            dTp_dchord_fd[:, i] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dchord_fd, dNp_dchord, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dTp_dchord_fd, dTp_dchord, rtol=5e-5, atol=1e-8)

    def test_dchord2(self):

        dT_dchord = self.dT["dchord"]
        dY_dchord = self.dY["dchord"]
        dZ_dchord = self.dZ["dchord"]
        dQ_dchord = self.dQ["dchord"]
        dMy_dchord = self.dMy["dchord"]
        dMz_dchord = self.dMz["dchord"]
        dMb_dchord = self.dMb["dchord"]
        dP_dchord = self.dP["dchord"]
        dT_dchord_fd = np.zeros((self.npts, self.n))
        dY_dchord_fd = np.zeros((self.npts, self.n))
        dZ_dchord_fd = np.zeros((self.npts, self.n))
        dQ_dchord_fd = np.zeros((self.npts, self.n))
        dMy_dchord_fd = np.zeros((self.npts, self.n))
        dMz_dchord_fd = np.zeros((self.npts, self.n))
        dMb_dchord_fd = np.zeros((self.npts, self.n))
        dP_dchord_fd = np.zeros((self.npts, self.n))

        for i in range(self.n):
            chord = np.array(self.chord)
            delta = 1e-6 * chord[i]
            chord[i] += delta

            rotor = CCBlade(
                self.r,
                chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
            Pd = outputs["P"]
            Td = outputs["T"]
            Yd = outputs["Y"]
            Zd = outputs["Z"]
            Qd = outputs["Q"]
            Myd = outputs["My"]
            Mzd = outputs["Mz"]
            Mbd = outputs["Mb"]

            dT_dchord_fd[:, i] = (Td - self.T) / delta
            dY_dchord_fd[:, i] = (Yd - self.Y) / delta
            dZ_dchord_fd[:, i] = (Zd - self.Z) / delta
            dQ_dchord_fd[:, i] = (Qd - self.Q) / delta
            dMy_dchord_fd[:, i] = (Myd - self.My) / delta
            dMz_dchord_fd[:, i] = (Mzd - self.Mz) / delta
            dMb_dchord_fd[:, i] = (Mbd - self.Mb) / delta
            dP_dchord_fd[:, i] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dchord_fd, dT_dchord, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dY_dchord_fd, dY_dchord, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dZ_dchord_fd, dZ_dchord, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dchord_fd, dQ_dchord, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dchord_fd, dMy_dchord, rtol=4e-4, atol=1e-8)
        np.testing.assert_allclose(dMz_dchord_fd, dMz_dchord, rtol=2e-4, atol=1e-8)
        np.testing.assert_allclose(dMb_dchord_fd, dMb_dchord, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dchord_fd, dP_dchord, rtol=7e-5, atol=1e-8)

    def test_dchord3(self):

        dCT_dchord = self.dCT["dchord"]
        dCY_dchord = self.dCY["dchord"]
        dCZ_dchord = self.dCZ["dchord"]
        dCQ_dchord = self.dCQ["dchord"]
        dCMy_dchord = self.dCMy["dchord"]
        dCMz_dchord = self.dCMz["dchord"]
        dCMb_dchord = self.dCMb["dchord"]
        dCP_dchord = self.dCP["dchord"]
        dCT_dchord_fd = np.zeros((self.npts, self.n))
        dCY_dchord_fd = np.zeros((self.npts, self.n))
        dCZ_dchord_fd = np.zeros((self.npts, self.n))
        dCQ_dchord_fd = np.zeros((self.npts, self.n))
        dCMy_dchord_fd = np.zeros((self.npts, self.n))
        dCMz_dchord_fd = np.zeros((self.npts, self.n))
        dCMb_dchord_fd = np.zeros((self.npts, self.n))
        dCP_dchord_fd = np.zeros((self.npts, self.n))

        for i in range(self.n):
            chord = np.array(self.chord)
            delta = 1e-6 * chord[i]
            chord[i] += delta

            rotor = CCBlade(
                self.r,
                chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
            CPd = outputs["CP"]
            CTd = outputs["CT"]
            CYd = outputs["CY"]
            CZd = outputs["CZ"]
            CQd = outputs["CQ"]
            CMyd = outputs["CMy"]
            CMzd = outputs["CMz"]
            CMbd = outputs["CMb"]

            dCT_dchord_fd[:, i] = (CTd - self.CT) / delta
            dCY_dchord_fd[:, i] = (CYd - self.CY) / delta
            dCZ_dchord_fd[:, i] = (CZd - self.CZ) / delta
            dCQ_dchord_fd[:, i] = (CQd - self.CQ) / delta
            dCMy_dchord_fd[:, i] = (CMyd - self.CMy) / delta
            dCMz_dchord_fd[:, i] = (CMzd - self.CMz) / delta
            dCMb_dchord_fd[:, i] = (CMbd - self.CMb) / delta
            dCP_dchord_fd[:, i] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dchord_fd, dCT_dchord, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dCY_dchord_fd, dCY_dchord, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dCZ_dchord_fd, dCZ_dchord, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dCQ_dchord_fd, dCQ_dchord, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dchord_fd, dCMy_dchord, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dchord_fd, dCMz_dchord, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dchord_fd, dCMb_dchord, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dchord_fd, dCP_dchord, rtol=7e-5, atol=1e-8)

    def test_dtheta1(self):

        dNp_dtheta = self.dNp["dtheta"]
        dTp_dtheta = self.dTp["dtheta"]
        dNp_dtheta_fd = np.zeros((self.n, self.n))
        dTp_dtheta_fd = np.zeros((self.n, self.n))

        for i in range(self.n):
            theta = np.array(self.theta)
            delta = 1e-6 * theta[i]
            theta[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dtheta_fd[:, i] = (Npd - self.Np) / delta
            dTp_dtheta_fd[:, i] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dtheta_fd, dNp_dtheta, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dTp_dtheta_fd, dTp_dtheta, rtol=1e-4, atol=1e-8)

    def test_dtheta2(self):

        dT_dtheta = self.dT["dtheta"]
        dY_dtheta = self.dY["dtheta"]
        dZ_dtheta = self.dZ["dtheta"]
        dQ_dtheta = self.dQ["dtheta"]
        dMy_dtheta = self.dMy["dtheta"]
        dMz_dtheta = self.dMz["dtheta"]
        dMb_dtheta = self.dMb["dtheta"]
        dP_dtheta = self.dP["dtheta"]
        dT_dtheta_fd = np.zeros((self.npts, self.n))
        dY_dtheta_fd = np.zeros((self.npts, self.n))
        dZ_dtheta_fd = np.zeros((self.npts, self.n))
        dQ_dtheta_fd = np.zeros((self.npts, self.n))
        dMy_dtheta_fd = np.zeros((self.npts, self.n))
        dMz_dtheta_fd = np.zeros((self.npts, self.n))
        dMb_dtheta_fd = np.zeros((self.npts, self.n))
        dP_dtheta_fd = np.zeros((self.npts, self.n))

        for i in range(self.n):
            theta = np.array(self.theta)
            delta = 1e-6 * theta[i]
            theta[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
            Pd = outputs["P"]
            Td = outputs["T"]
            Yd = outputs["Y"]
            Zd = outputs["Z"]
            Qd = outputs["Q"]
            Myd = outputs["My"]
            Mzd = outputs["Mz"]
            Mbd = outputs["Mb"]

            dT_dtheta_fd[:, i] = (Td - self.T) / delta
            dY_dtheta_fd[:, i] = (Yd - self.Y) / delta
            dZ_dtheta_fd[:, i] = (Zd - self.Z) / delta
            dQ_dtheta_fd[:, i] = (Qd - self.Q) / delta
            dMy_dtheta_fd[:, i] = (Myd - self.My) / delta
            dMz_dtheta_fd[:, i] = (Mzd - self.Mz) / delta
            dMb_dtheta_fd[:, i] = (Mbd - self.Mb) / delta
            dP_dtheta_fd[:, i] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dtheta_fd, dT_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dtheta_fd, dY_dtheta, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dZ_dtheta_fd, dZ_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dtheta_fd, dQ_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dtheta_fd, dMy_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dtheta_fd, dMz_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dtheta_fd, dMb_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dtheta_fd, dP_dtheta, rtol=7e-5, atol=1e-8)

    def test_dtheta3(self):

        dCT_dtheta = self.dCT["dtheta"]
        dCY_dtheta = self.dCY["dtheta"]
        dCZ_dtheta = self.dCZ["dtheta"]
        dCQ_dtheta = self.dCQ["dtheta"]
        dCMy_dtheta = self.dCMy["dtheta"]
        dCMz_dtheta = self.dCMz["dtheta"]
        dCMb_dtheta = self.dCMb["dtheta"]
        dCP_dtheta = self.dCP["dtheta"]
        dCT_dtheta_fd = np.zeros((self.npts, self.n))
        dCY_dtheta_fd = np.zeros((self.npts, self.n))
        dCZ_dtheta_fd = np.zeros((self.npts, self.n))
        dCQ_dtheta_fd = np.zeros((self.npts, self.n))
        dCMy_dtheta_fd = np.zeros((self.npts, self.n))
        dCMz_dtheta_fd = np.zeros((self.npts, self.n))
        dCMb_dtheta_fd = np.zeros((self.npts, self.n))
        dCP_dtheta_fd = np.zeros((self.npts, self.n))

        for i in range(self.n):
            theta = np.array(self.theta)
            delta = 1e-6 * theta[i]
            theta[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
            CPd = outputs["CP"]
            CTd = outputs["CT"]
            CYd = outputs["CY"]
            CZd = outputs["CZ"]
            CQd = outputs["CQ"]
            CMyd = outputs["CMy"]
            CMzd = outputs["CMz"]
            CMbd = outputs["CMb"]

            dCT_dtheta_fd[:, i] = (CTd - self.CT) / delta
            dCY_dtheta_fd[:, i] = (CYd - self.CY) / delta
            dCZ_dtheta_fd[:, i] = (CZd - self.CZ) / delta
            dCQ_dtheta_fd[:, i] = (CQd - self.CQ) / delta
            dCMy_dtheta_fd[:, i] = (CMyd - self.CMy) / delta
            dCMz_dtheta_fd[:, i] = (CMzd - self.CMz) / delta
            dCMb_dtheta_fd[:, i] = (CMbd - self.CMb) / delta
            dCP_dtheta_fd[:, i] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dtheta_fd, dCT_dtheta, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dCY_dtheta_fd, dCY_dtheta, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dCZ_dtheta_fd, dCZ_dtheta, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dCQ_dtheta_fd, dCQ_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dtheta_fd, dCMy_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dtheta_fd, dCMz_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dtheta_fd, dCMb_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dtheta_fd, dCP_dtheta, rtol=7e-5, atol=1e-8)

    def test_dRhub1(self):

        dNp_dRhub = self.dNp["dRhub"]
        dTp_dRhub = self.dTp["dRhub"]

        dNp_dRhub_fd = np.zeros((self.n, 1))
        dTp_dRhub_fd = np.zeros((self.n, 1))

        Rhub = float(self.Rhub)
        delta = 1e-6 * Rhub
        Rhub += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dRhub_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dRhub_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dRhub_fd, dNp_dRhub, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dRhub_fd, dTp_dRhub, rtol=1e-4, atol=1e-6)

    def test_dRhub2(self):

        dT_dRhub = self.dT["dRhub"]
        dY_dRhub = self.dY["dRhub"]
        dZ_dRhub = self.dZ["dRhub"]
        dQ_dRhub = self.dQ["dRhub"]
        dMy_dRhub = self.dMy["dRhub"]
        dMz_dRhub = self.dMz["dRhub"]
        dMb_dRhub = self.dMb["dRhub"]
        dP_dRhub = self.dP["dRhub"]

        dT_dRhub_fd = np.zeros((self.npts, 1))
        dY_dRhub_fd = np.zeros((self.npts, 1))
        dZ_dRhub_fd = np.zeros((self.npts, 1))
        dQ_dRhub_fd = np.zeros((self.npts, 1))
        dMy_dRhub_fd = np.zeros((self.npts, 1))
        dMz_dRhub_fd = np.zeros((self.npts, 1))
        dMb_dRhub_fd = np.zeros((self.npts, 1))
        dP_dRhub_fd = np.zeros((self.npts, 1))

        Rhub = float(self.Rhub)
        delta = 1e-6 * Rhub
        Rhub += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dRhub_fd[:, 0] = (Td - self.T) / delta
        dY_dRhub_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dRhub_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dRhub_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dRhub_fd[:, 0] = (Myd - self.My) / delta
        dMz_dRhub_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dRhub_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dRhub_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dRhub_fd, dT_dRhub, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dRhub_fd, dY_dRhub, rtol=1e-3, atol=1e-8)
        np.testing.assert_allclose(dZ_dRhub_fd, dZ_dRhub, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dRhub_fd, dQ_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dRhub_fd, dMy_dRhub, rtol=5e-4, atol=1e-8)
        np.testing.assert_allclose(dMz_dRhub_fd, dMz_dRhub, rtol=5e-4, atol=1e-8)
        np.testing.assert_allclose(dMb_dRhub_fd, dMb_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dRhub_fd, dP_dRhub, rtol=5e-5, atol=1e-8)

    def test_dRhub3(self):

        dCT_dRhub = self.dCT["dRhub"]
        dCY_dRhub = self.dCY["dRhub"]
        dCZ_dRhub = self.dCZ["dRhub"]
        dCQ_dRhub = self.dCQ["dRhub"]
        dCMy_dRhub = self.dCMy["dRhub"]
        dCMz_dRhub = self.dCMz["dRhub"]
        dCMb_dRhub = self.dCMb["dRhub"]
        dCP_dRhub = self.dCP["dRhub"]

        dCT_dRhub_fd = np.zeros((self.npts, 1))
        dCY_dRhub_fd = np.zeros((self.npts, 1))
        dCZ_dRhub_fd = np.zeros((self.npts, 1))
        dCQ_dRhub_fd = np.zeros((self.npts, 1))
        dCMy_dRhub_fd = np.zeros((self.npts, 1))
        dCMz_dRhub_fd = np.zeros((self.npts, 1))
        dCMb_dRhub_fd = np.zeros((self.npts, 1))
        dCP_dRhub_fd = np.zeros((self.npts, 1))

        Rhub = float(self.Rhub)
        delta = 1e-6 * Rhub
        Rhub += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dRhub_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dRhub_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dRhub_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dRhub_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dRhub_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dRhub_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dRhub_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dRhub_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dRhub_fd, dCT_dRhub, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dRhub_fd, dCY_dRhub, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dRhub_fd, dCZ_dRhub, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dRhub_fd, dCQ_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dRhub_fd, dCMy_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dRhub_fd, dCMz_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dRhub_fd, dCMb_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dRhub_fd, dCP_dRhub, rtol=5e-5, atol=1e-8)

    def test_dRtip1(self):

        dNp_dRtip = self.dNp["dRtip"]
        dTp_dRtip = self.dTp["dRtip"]

        dNp_dRtip_fd = np.zeros((self.n, 1))
        dTp_dRtip_fd = np.zeros((self.n, 1))

        Rtip = float(self.Rtip)
        delta = 1e-6 * Rtip
        Rtip += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dRtip_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dRtip_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dRtip_fd, dNp_dRtip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dRtip_fd, dTp_dRtip, rtol=1e-4, atol=1e-8)

    def test_dRtip2(self):

        dT_dRtip = self.dT["dRtip"]
        dY_dRtip = self.dY["dRtip"]
        dZ_dRtip = self.dZ["dRtip"]
        dQ_dRtip = self.dQ["dRtip"]
        dMy_dRtip = self.dMy["dRtip"]
        dMz_dRtip = self.dMz["dRtip"]
        dMb_dRtip = self.dMb["dRtip"]
        dP_dRtip = self.dP["dRtip"]

        dT_dRtip_fd = np.zeros((self.npts, 1))
        dY_dRtip_fd = np.zeros((self.npts, 1))
        dZ_dRtip_fd = np.zeros((self.npts, 1))
        dQ_dRtip_fd = np.zeros((self.npts, 1))
        dMy_dRtip_fd = np.zeros((self.npts, 1))
        dMz_dRtip_fd = np.zeros((self.npts, 1))
        dMb_dRtip_fd = np.zeros((self.npts, 1))
        dP_dRtip_fd = np.zeros((self.npts, 1))

        Rtip = float(self.Rtip)
        delta = 1e-6 * Rtip
        Rtip += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dRtip_fd[:, 0] = (Td - self.T) / delta
        dY_dRtip_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dRtip_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dRtip_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dRtip_fd[:, 0] = (Myd - self.My) / delta
        dMz_dRtip_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dRtip_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dRtip_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dRtip_fd, dT_dRtip, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dRtip_fd, dY_dRtip, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dRtip_fd, dZ_dRtip, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dRtip_fd, dQ_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dRtip_fd, dMy_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dRtip_fd, dMz_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dRtip_fd, dMb_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dRtip_fd, dP_dRtip, rtol=5e-5, atol=1e-8)

    def test_dRtip3(self):

        dCT_dRtip = self.dCT["dRtip"]
        dCY_dRtip = self.dCY["dRtip"]
        dCZ_dRtip = self.dCZ["dRtip"]
        dCQ_dRtip = self.dCQ["dRtip"]
        dCMy_dRtip = self.dCMy["dRtip"]
        dCMz_dRtip = self.dCMz["dRtip"]
        dCMb_dRtip = self.dCMb["dRtip"]
        dCP_dRtip = self.dCP["dRtip"]

        dCT_dRtip_fd = np.zeros((self.npts, 1))
        dCY_dRtip_fd = np.zeros((self.npts, 1))
        dCZ_dRtip_fd = np.zeros((self.npts, 1))
        dCQ_dRtip_fd = np.zeros((self.npts, 1))
        dCMy_dRtip_fd = np.zeros((self.npts, 1))
        dCMz_dRtip_fd = np.zeros((self.npts, 1))
        dCMb_dRtip_fd = np.zeros((self.npts, 1))
        dCP_dRtip_fd = np.zeros((self.npts, 1))

        Rtip = float(self.Rtip)
        delta = 1e-6 * Rtip
        Rtip += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dRtip_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dRtip_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dRtip_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dRtip_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dRtip_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dRtip_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dRtip_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dRtip_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dRtip_fd, dCT_dRtip, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dRtip_fd, dCY_dRtip, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dRtip_fd, dCZ_dRtip, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dRtip_fd, dCQ_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dRtip_fd, dCMy_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dRtip_fd, dCMz_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dRtip_fd, dCMb_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dRtip_fd, dCP_dRtip, rtol=5e-5, atol=1e-8)

    def test_dprecone1(self):

        dNp_dprecone = self.dNp["dprecone"]
        dTp_dprecone = self.dTp["dprecone"]

        dNp_dprecone_fd = np.zeros((self.n, 1))
        dTp_dprecone_fd = np.zeros((self.n, 1))

        precone = float(self.precone)
        delta = 1e-6 * precone
        precone += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dprecone_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dprecone_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dprecone_fd, dNp_dprecone, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(dTp_dprecone_fd, dTp_dprecone, rtol=1e-5, atol=1e-7)

    def test_dprecone2(self):

        dT_dprecone = self.dT["dprecone"]
        dY_dprecone = self.dY["dprecone"]
        dZ_dprecone = self.dZ["dprecone"]
        dQ_dprecone = self.dQ["dprecone"]
        dMy_dprecone = self.dMy["dprecone"]
        dMz_dprecone = self.dMz["dprecone"]
        dMb_dprecone = self.dMb["dprecone"]
        dP_dprecone = self.dP["dprecone"]

        dT_dprecone_fd = np.zeros((self.npts, 1))
        dY_dprecone_fd = np.zeros((self.npts, 1))
        dZ_dprecone_fd = np.zeros((self.npts, 1))
        dQ_dprecone_fd = np.zeros((self.npts, 1))
        dMy_dprecone_fd = np.zeros((self.npts, 1))
        dMz_dprecone_fd = np.zeros((self.npts, 1))
        dMb_dprecone_fd = np.zeros((self.npts, 1))
        dP_dprecone_fd = np.zeros((self.npts, 1))

        precone = float(self.precone)
        delta = 1e-6 * precone
        precone += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dprecone_fd[:, 0] = (Td - self.T) / delta
        dY_dprecone_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dprecone_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dprecone_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dprecone_fd[:, 0] = (Myd - self.My) / delta
        dMz_dprecone_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dprecone_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dprecone_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dprecone_fd, dT_dprecone, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dprecone_fd, dY_dprecone, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dprecone_fd, dZ_dprecone, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dprecone_fd, dQ_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dprecone_fd, dMy_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dprecone_fd, dMz_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dprecone_fd, dMb_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dprecone_fd, dP_dprecone, rtol=5e-5, atol=1e-8)

    def test_dprecone3(self):

        dCT_dprecone = self.dCT["dprecone"]
        dCY_dprecone = self.dCY["dprecone"]
        dCZ_dprecone = self.dCZ["dprecone"]
        dCQ_dprecone = self.dCQ["dprecone"]
        dCMy_dprecone = self.dCMy["dprecone"]
        dCMz_dprecone = self.dCMz["dprecone"]
        dCMb_dprecone = self.dCMb["dprecone"]
        dCP_dprecone = self.dCP["dprecone"]

        dCT_dprecone_fd = np.zeros((self.npts, 1))
        dCY_dprecone_fd = np.zeros((self.npts, 1))
        dCZ_dprecone_fd = np.zeros((self.npts, 1))
        dCQ_dprecone_fd = np.zeros((self.npts, 1))
        dCMy_dprecone_fd = np.zeros((self.npts, 1))
        dCMz_dprecone_fd = np.zeros((self.npts, 1))
        dCMb_dprecone_fd = np.zeros((self.npts, 1))
        dCP_dprecone_fd = np.zeros((self.npts, 1))

        precone = float(self.precone)
        delta = 1e-6 * precone
        precone += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dprecone_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dprecone_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dprecone_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dprecone_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dprecone_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dprecone_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dprecone_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dprecone_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dprecone_fd, dCT_dprecone, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dprecone_fd, dCY_dprecone, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dprecone_fd, dCZ_dprecone, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dprecone_fd, dCQ_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dprecone_fd, dCMy_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dprecone_fd, dCMz_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dprecone_fd, dCMb_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dprecone_fd, dCP_dprecone, rtol=5e-5, atol=1e-8)

    def test_dtilt1(self):

        dNp_dtilt = self.dNp["dtilt"]
        dTp_dtilt = self.dTp["dtilt"]

        dNp_dtilt_fd = np.zeros((self.n, 1))
        dTp_dtilt_fd = np.zeros((self.n, 1))

        tilt = float(self.tilt)
        delta = 1e-6 * tilt
        tilt += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dtilt_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dtilt_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dtilt_fd, dNp_dtilt, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dTp_dtilt_fd, dTp_dtilt, rtol=1e-5, atol=1e-8)

    def test_dtilt2(self):

        dT_dtilt = self.dT["dtilt"]
        dY_dtilt = self.dY["dtilt"]
        dZ_dtilt = self.dZ["dtilt"]
        dQ_dtilt = self.dQ["dtilt"]
        dMy_dtilt = self.dMy["dtilt"]
        dMz_dtilt = self.dMz["dtilt"]
        dMb_dtilt = self.dMb["dtilt"]
        dP_dtilt = self.dP["dtilt"]

        dT_dtilt_fd = np.zeros((self.npts, 1))
        dY_dtilt_fd = np.zeros((self.npts, 1))
        dZ_dtilt_fd = np.zeros((self.npts, 1))
        dQ_dtilt_fd = np.zeros((self.npts, 1))
        dMy_dtilt_fd = np.zeros((self.npts, 1))
        dMz_dtilt_fd = np.zeros((self.npts, 1))
        dMb_dtilt_fd = np.zeros((self.npts, 1))
        dP_dtilt_fd = np.zeros((self.npts, 1))

        tilt = float(self.tilt)
        delta = 1e-6 * tilt
        tilt += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dtilt_fd[:, 0] = (Td - self.T) / delta
        dY_dtilt_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dtilt_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dtilt_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dtilt_fd[:, 0] = (Myd - self.My) / delta
        dMz_dtilt_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dtilt_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dtilt_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dtilt_fd, dT_dtilt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dtilt_fd, dY_dtilt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dtilt_fd, dZ_dtilt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dtilt_fd, dQ_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dtilt_fd, dMy_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dtilt_fd, dMz_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dtilt_fd, dMb_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dtilt_fd, dP_dtilt, rtol=5e-5, atol=1e-8)

    def test_dtilt3(self):

        dCT_dtilt = self.dCT["dtilt"]
        dCY_dtilt = self.dCY["dtilt"]
        dCZ_dtilt = self.dCZ["dtilt"]
        dCQ_dtilt = self.dCQ["dtilt"]
        dCMy_dtilt = self.dCMy["dtilt"]
        dCMz_dtilt = self.dCMz["dtilt"]
        dCMb_dtilt = self.dCMb["dtilt"]
        dCP_dtilt = self.dCP["dtilt"]

        dCT_dtilt_fd = np.zeros((self.npts, 1))
        dCY_dtilt_fd = np.zeros((self.npts, 1))
        dCZ_dtilt_fd = np.zeros((self.npts, 1))
        dCQ_dtilt_fd = np.zeros((self.npts, 1))
        dCMy_dtilt_fd = np.zeros((self.npts, 1))
        dCMz_dtilt_fd = np.zeros((self.npts, 1))
        dCMb_dtilt_fd = np.zeros((self.npts, 1))
        dCP_dtilt_fd = np.zeros((self.npts, 1))

        tilt = float(self.tilt)
        delta = 1e-6 * tilt
        tilt += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dtilt_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dtilt_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dtilt_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dtilt_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dtilt_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dtilt_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dtilt_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dtilt_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dtilt_fd, dCT_dtilt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dtilt_fd, dCY_dtilt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dtilt_fd, dCZ_dtilt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dtilt_fd, dCQ_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dtilt_fd, dCMy_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dtilt_fd, dCMz_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dtilt_fd, dCMb_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dtilt_fd, dCP_dtilt, rtol=5e-5, atol=1e-8)

    def test_dhubht1(self):

        dNp_dhubht = self.dNp["dhubHt"]
        dTp_dhubht = self.dTp["dhubHt"]

        dNp_dhubht_fd = np.zeros((self.n, 1))
        dTp_dhubht_fd = np.zeros((self.n, 1))

        hubht = float(self.hubHt)
        delta = 1e-6 * hubht
        hubht += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            hubht,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dhubht_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dhubht_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dhubht_fd, dNp_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dhubht_fd, dTp_dhubht, rtol=1e-5, atol=1e-8)

    def test_dhubht2(self):

        dT_dhubht = self.dT["dhubHt"]
        dY_dhubht = self.dY["dhubHt"]
        dZ_dhubht = self.dZ["dhubHt"]
        dQ_dhubht = self.dQ["dhubHt"]
        dMy_dhubht = self.dMy["dhubHt"]
        dMz_dhubht = self.dMz["dhubHt"]
        dMb_dhubht = self.dMb["dhubHt"]
        dP_dhubht = self.dP["dhubHt"]

        dT_dhubht_fd = np.zeros((self.npts, 1))
        dY_dhubht_fd = np.zeros((self.npts, 1))
        dZ_dhubht_fd = np.zeros((self.npts, 1))
        dQ_dhubht_fd = np.zeros((self.npts, 1))
        dMy_dhubht_fd = np.zeros((self.npts, 1))
        dMz_dhubht_fd = np.zeros((self.npts, 1))
        dMb_dhubht_fd = np.zeros((self.npts, 1))
        dP_dhubht_fd = np.zeros((self.npts, 1))

        hubht = float(self.hubHt)
        delta = 1e-6 * hubht
        hubht += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            hubht,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dhubht_fd[:, 0] = (Td - self.T) / delta
        dY_dhubht_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dhubht_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dhubht_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dhubht_fd[:, 0] = (Myd - self.My) / delta
        dMz_dhubht_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dhubht_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dhubht_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dhubht_fd, dT_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dhubht_fd, dY_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dhubht_fd, dZ_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dhubht_fd, dQ_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dhubht_fd, dMy_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dhubht_fd, dMz_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dhubht_fd, dMb_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dhubht_fd, dP_dhubht, rtol=5e-5, atol=1e-8)

    def test_dhubht3(self):

        dCT_dhubht = self.dCT["dhubHt"]
        dCY_dhubht = self.dCY["dhubHt"]
        dCZ_dhubht = self.dCZ["dhubHt"]
        dCQ_dhubht = self.dCQ["dhubHt"]
        dCMy_dhubht = self.dCMy["dhubHt"]
        dCMz_dhubht = self.dCMz["dhubHt"]
        dCMb_dhubht = self.dCMb["dhubHt"]
        dCP_dhubht = self.dCP["dhubHt"]

        dCT_dhubht_fd = np.zeros((self.npts, 1))
        dCY_dhubht_fd = np.zeros((self.npts, 1))
        dCZ_dhubht_fd = np.zeros((self.npts, 1))
        dCQ_dhubht_fd = np.zeros((self.npts, 1))
        dCMy_dhubht_fd = np.zeros((self.npts, 1))
        dCMz_dhubht_fd = np.zeros((self.npts, 1))
        dCMb_dhubht_fd = np.zeros((self.npts, 1))
        dCP_dhubht_fd = np.zeros((self.npts, 1))

        hubht = float(self.hubHt)
        delta = 1e-6 * hubht
        hubht += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            hubht,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dhubht_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dhubht_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dhubht_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dhubht_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dhubht_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dhubht_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dhubht_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dhubht_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dhubht_fd, dCT_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dhubht_fd, dCY_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dhubht_fd, dCZ_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dhubht_fd, dCQ_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dhubht_fd, dCMy_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dhubht_fd, dCMz_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dhubht_fd, dCMb_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dhubht_fd, dCP_dhubht, rtol=5e-5, atol=1e-8)

    def test_dyaw1(self):

        dNp_dyaw = self.dNp["dyaw"]
        dTp_dyaw = self.dTp["dyaw"]

        dNp_dyaw_fd = np.zeros((self.n, 1))
        dTp_dyaw_fd = np.zeros((self.n, 1))

        yaw = float(self.yaw)
        delta = 1e-6
        yaw += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dyaw_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dyaw_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dyaw_fd, dNp_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dyaw_fd, dTp_dyaw, rtol=1e-5, atol=1e-8)

    def test_dyaw2(self):

        dT_dyaw = self.dT["dyaw"]
        dY_dyaw = self.dY["dyaw"]
        dZ_dyaw = self.dZ["dyaw"]
        dQ_dyaw = self.dQ["dyaw"]
        dMy_dyaw = self.dMy["dyaw"]
        dMz_dyaw = self.dMz["dyaw"]
        dMb_dyaw = self.dMb["dyaw"]
        dP_dyaw = self.dP["dyaw"]

        dT_dyaw_fd = np.zeros((self.npts, 1))
        dY_dyaw_fd = np.zeros((self.npts, 1))
        dZ_dyaw_fd = np.zeros((self.npts, 1))
        dQ_dyaw_fd = np.zeros((self.npts, 1))
        dMy_dyaw_fd = np.zeros((self.npts, 1))
        dMz_dyaw_fd = np.zeros((self.npts, 1))
        dMb_dyaw_fd = np.zeros((self.npts, 1))
        dP_dyaw_fd = np.zeros((self.npts, 1))

        yaw = float(self.yaw)
        delta = 1e-6
        yaw += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dyaw_fd[:, 0] = (Td - self.T) / delta
        dY_dyaw_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dyaw_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dyaw_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dyaw_fd[:, 0] = (Myd - self.My) / delta
        dMz_dyaw_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dyaw_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dyaw_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dyaw_fd, dT_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dyaw_fd, dY_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dyaw_fd, dZ_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dyaw_fd, dQ_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dyaw_fd, dMy_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dyaw_fd, dMz_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dyaw_fd, dMb_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dyaw_fd, dP_dyaw, rtol=5e-5, atol=1e-8)

    def test_dyaw3(self):

        dCT_dyaw = self.dCT["dyaw"]
        dCY_dyaw = self.dCY["dyaw"]
        dCZ_dyaw = self.dCZ["dyaw"]
        dCQ_dyaw = self.dCQ["dyaw"]
        dCMy_dyaw = self.dCMy["dyaw"]
        dCMz_dyaw = self.dCMz["dyaw"]
        dCMb_dyaw = self.dCMb["dyaw"]
        dCP_dyaw = self.dCP["dyaw"]

        dCT_dyaw_fd = np.zeros((self.npts, 1))
        dCY_dyaw_fd = np.zeros((self.npts, 1))
        dCZ_dyaw_fd = np.zeros((self.npts, 1))
        dCQ_dyaw_fd = np.zeros((self.npts, 1))
        dCMy_dyaw_fd = np.zeros((self.npts, 1))
        dCMz_dyaw_fd = np.zeros((self.npts, 1))
        dCMb_dyaw_fd = np.zeros((self.npts, 1))
        dCP_dyaw_fd = np.zeros((self.npts, 1))

        yaw = float(self.yaw)
        delta = 1e-6
        yaw += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dyaw_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dyaw_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dyaw_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dyaw_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dyaw_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dyaw_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dyaw_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dyaw_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dyaw_fd, dCT_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dyaw_fd, dCY_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dyaw_fd, dCZ_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dyaw_fd, dCQ_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dyaw_fd, dCMy_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dyaw_fd, dCMz_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dyaw_fd, dCMb_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dyaw_fd, dCP_dyaw, rtol=5e-5, atol=1e-8)

    def test_dshear1(self):

        dNp_dshear = self.dNp["dshear"]
        dTp_dshear = self.dTp["dshear"]

        dNp_dshear_fd = np.zeros((self.n, 1))
        dTp_dshear_fd = np.zeros((self.n, 1))

        shearExp = float(self.shearExp)
        delta = 1e-6
        shearExp += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dshear_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dshear_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dshear_fd, dNp_dshear, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dshear_fd, dTp_dshear, rtol=1e-5, atol=1e-8)

    def test_dshear2(self):

        dT_dshear = self.dT["dshear"]
        dY_dshear = self.dY["dshear"]
        dZ_dshear = self.dZ["dshear"]
        dQ_dshear = self.dQ["dshear"]
        dMy_dshear = self.dMy["dshear"]
        dMz_dshear = self.dMz["dshear"]
        dMb_dshear = self.dMb["dshear"]
        dP_dshear = self.dP["dshear"]

        dT_dshear_fd = np.zeros((self.npts, 1))
        dY_dshear_fd = np.zeros((self.npts, 1))
        dZ_dshear_fd = np.zeros((self.npts, 1))
        dQ_dshear_fd = np.zeros((self.npts, 1))
        dMy_dshear_fd = np.zeros((self.npts, 1))
        dMz_dshear_fd = np.zeros((self.npts, 1))
        dMb_dshear_fd = np.zeros((self.npts, 1))
        dP_dshear_fd = np.zeros((self.npts, 1))

        shearExp = float(self.shearExp)
        delta = 1e-6
        shearExp += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dshear_fd[:, 0] = (Td - self.T) / delta
        dY_dshear_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dshear_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dshear_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dshear_fd[:, 0] = (Myd - self.My) / delta
        dMz_dshear_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dshear_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dshear_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dshear_fd, dT_dshear, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dshear_fd, dY_dshear, rtol=5e-5)  # , atol=1e-8)
        np.testing.assert_allclose(dZ_dshear_fd, dZ_dshear, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dshear_fd, dQ_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dshear_fd, dMy_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dshear_fd, dMz_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dshear_fd, dMb_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dshear_fd, dP_dshear, rtol=5e-5, atol=1e-8)

    def test_dshear3(self):

        dCT_dshear = self.dCT["dshear"]
        dCY_dshear = self.dCY["dshear"]
        dCZ_dshear = self.dCZ["dshear"]
        dCQ_dshear = self.dCQ["dshear"]
        dCMy_dshear = self.dCMy["dshear"]
        dCMz_dshear = self.dCMz["dshear"]
        dCMb_dshear = self.dCMb["dshear"]
        dCP_dshear = self.dCP["dshear"]

        dCT_dshear_fd = np.zeros((self.npts, 1))
        dCY_dshear_fd = np.zeros((self.npts, 1))
        dCZ_dshear_fd = np.zeros((self.npts, 1))
        dCQ_dshear_fd = np.zeros((self.npts, 1))
        dCMy_dshear_fd = np.zeros((self.npts, 1))
        dCMz_dshear_fd = np.zeros((self.npts, 1))
        dCMb_dshear_fd = np.zeros((self.npts, 1))
        dCP_dshear_fd = np.zeros((self.npts, 1))

        shearExp = float(self.shearExp)
        delta = 1e-6
        shearExp += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dshear_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dshear_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dshear_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dshear_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dshear_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dshear_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dshear_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dshear_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dshear_fd, dCT_dshear, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dshear_fd, dCY_dshear, rtol=2e-5, atol=5e-8)
        np.testing.assert_allclose(dCZ_dshear_fd, dCZ_dshear, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dshear_fd, dCQ_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dshear_fd, dCMy_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dshear_fd, dCMz_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dshear_fd, dCMb_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dshear_fd, dCP_dshear, rtol=5e-5, atol=1e-8)

    def test_dazimuth1(self):

        dNp_dazimuth = self.dNp["dazimuth"]
        dTp_dazimuth = self.dTp["dazimuth"]

        dNp_dazimuth_fd = np.zeros((self.n, 1))
        dTp_dazimuth_fd = np.zeros((self.n, 1))

        azimuth = float(self.azimuth)
        delta = 1e-6 * azimuth
        azimuth += delta

        outputs, _ = self.rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, azimuth)
        Npd = outputs["Np"]
        Tpd = outputs["Tp"]

        dNp_dazimuth_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dazimuth_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dazimuth_fd, dNp_dazimuth, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dazimuth_fd, dTp_dazimuth, rtol=1e-5, atol=1e-6)

    def test_dUinf1(self):

        dNp_dUinf = self.dNp["dUinf"]
        dTp_dUinf = self.dTp["dUinf"]

        dNp_dUinf_fd = np.zeros((self.n, 1))
        dTp_dUinf_fd = np.zeros((self.n, 1))

        Uinf = float(self.Uinf)
        delta = 1e-6 * Uinf
        Uinf += delta

        outputs, _ = self.rotor.distributedAeroLoads(Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = outputs["Np"]
        Tpd = outputs["Tp"]

        dNp_dUinf_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dUinf_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dUinf_fd, dNp_dUinf, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dUinf_fd, dTp_dUinf, rtol=1e-5, atol=1e-6)

    def test_dUinf2(self):

        dT_dUinf = self.dT["dUinf"]
        dY_dUinf = self.dY["dUinf"]
        dZ_dUinf = self.dZ["dUinf"]
        dQ_dUinf = self.dQ["dUinf"]
        dMy_dUinf = self.dMy["dUinf"]
        dMz_dUinf = self.dMz["dUinf"]
        dMb_dUinf = self.dMb["dUinf"]
        dP_dUinf = self.dP["dUinf"]

        dT_dUinf_fd = np.zeros((self.npts, self.npts))
        dY_dUinf_fd = np.zeros((self.npts, self.npts))
        dZ_dUinf_fd = np.zeros((self.npts, self.npts))
        dQ_dUinf_fd = np.zeros((self.npts, self.npts))
        dMy_dUinf_fd = np.zeros((self.npts, self.npts))
        dMz_dUinf_fd = np.zeros((self.npts, self.npts))
        dMb_dUinf_fd = np.zeros((self.npts, self.npts))
        dP_dUinf_fd = np.zeros((self.npts, self.npts))

        Uinf = float(self.Uinf)
        delta = 1e-6 * Uinf
        Uinf += delta

        outputs, _ = self.rotor.evaluate([Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dUinf_fd[:, 0] = (Td - self.T) / delta
        dY_dUinf_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dUinf_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dUinf_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dUinf_fd[:, 0] = (Myd - self.My) / delta
        dMz_dUinf_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dUinf_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dUinf_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dUinf_fd, dT_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dUinf_fd, dY_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dUinf_fd, dZ_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dUinf_fd, dQ_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dUinf_fd, dMy_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dUinf_fd, dMz_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dUinf_fd, dMb_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dUinf_fd, dP_dUinf, rtol=5e-5, atol=1e-8)

    def test_dUinf3(self):

        dCT_dUinf = self.dCT["dUinf"]
        dCY_dUinf = self.dCY["dUinf"]
        dCZ_dUinf = self.dCZ["dUinf"]
        dCQ_dUinf = self.dCQ["dUinf"]
        dCMy_dUinf = self.dCMy["dUinf"]
        dCMz_dUinf = self.dCMz["dUinf"]
        dCMb_dUinf = self.dCMb["dUinf"]
        dCP_dUinf = self.dCP["dUinf"]

        dCT_dUinf_fd = np.zeros((self.npts, self.npts))
        dCY_dUinf_fd = np.zeros((self.npts, self.npts))
        dCZ_dUinf_fd = np.zeros((self.npts, self.npts))
        dCQ_dUinf_fd = np.zeros((self.npts, self.npts))
        dCMy_dUinf_fd = np.zeros((self.npts, self.npts))
        dCMz_dUinf_fd = np.zeros((self.npts, self.npts))
        dCMb_dUinf_fd = np.zeros((self.npts, self.npts))
        dCP_dUinf_fd = np.zeros((self.npts, self.npts))

        Uinf = float(self.Uinf)
        delta = 1e-6 * Uinf
        Uinf += delta

        outputs, _ = self.rotor.evaluate([Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dUinf_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dUinf_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dUinf_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dUinf_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dUinf_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dUinf_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dUinf_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dUinf_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dUinf_fd, dCT_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dUinf_fd, dCY_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dUinf_fd, dCZ_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dUinf_fd, dCQ_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dUinf_fd, dCMy_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dUinf_fd, dCMz_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dUinf_fd, dCMb_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dUinf_fd, dCP_dUinf, rtol=5e-5, atol=1e-8)

    def test_dOmega1(self):

        dNp_dOmega = self.dNp["dOmega"]
        dTp_dOmega = self.dTp["dOmega"]

        dNp_dOmega_fd = np.zeros((self.n, 1))
        dTp_dOmega_fd = np.zeros((self.n, 1))

        Omega = float(self.Omega)
        delta = 1e-6 * Omega
        Omega += delta

        loads, _ = self.rotor.distributedAeroLoads(self.Uinf, Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dOmega_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dOmega_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dOmega_fd, dNp_dOmega, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dOmega_fd, dTp_dOmega, rtol=1e-5, atol=1e-6)

    def test_dOmega2(self):

        dT_dOmega = self.dT["dOmega"]
        dY_dOmega = self.dY["dOmega"]
        dZ_dOmega = self.dZ["dOmega"]
        dQ_dOmega = self.dQ["dOmega"]
        dMy_dOmega = self.dMy["dOmega"]
        dMz_dOmega = self.dMz["dOmega"]
        dMb_dOmega = self.dMb["dOmega"]
        dP_dOmega = self.dP["dOmega"]

        dT_dOmega_fd = np.zeros((self.npts, self.npts))
        dY_dOmega_fd = np.zeros((self.npts, self.npts))
        dZ_dOmega_fd = np.zeros((self.npts, self.npts))
        dQ_dOmega_fd = np.zeros((self.npts, self.npts))
        dMy_dOmega_fd = np.zeros((self.npts, self.npts))
        dMz_dOmega_fd = np.zeros((self.npts, self.npts))
        dMb_dOmega_fd = np.zeros((self.npts, self.npts))
        dP_dOmega_fd = np.zeros((self.npts, self.npts))

        Omega = float(self.Omega)
        delta = 1e-6 * Omega
        Omega += delta

        outputs, _ = self.rotor.evaluate([self.Uinf], [Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dOmega_fd[:, 0] = (Td - self.T) / delta
        dY_dOmega_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dOmega_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dOmega_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dOmega_fd[:, 0] = (Myd - self.My) / delta
        dMz_dOmega_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dOmega_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dOmega_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dOmega_fd, dT_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dOmega_fd, dY_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dOmega_fd, dZ_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dOmega_fd, dQ_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dOmega_fd, dMy_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dOmega_fd, dMz_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dOmega_fd, dMb_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dOmega_fd, dP_dOmega, rtol=5e-5, atol=1e-8)

    def test_dOmega3(self):

        dCT_dOmega = self.dCT["dOmega"]
        dCY_dOmega = self.dCY["dOmega"]
        dCZ_dOmega = self.dCZ["dOmega"]
        dCQ_dOmega = self.dCQ["dOmega"]
        dCMy_dOmega = self.dCMy["dOmega"]
        dCMz_dOmega = self.dCMz["dOmega"]
        dCMb_dOmega = self.dCMb["dOmega"]
        dCP_dOmega = self.dCP["dOmega"]

        dCT_dOmega_fd = np.zeros((self.npts, self.npts))
        dCY_dOmega_fd = np.zeros((self.npts, self.npts))
        dCZ_dOmega_fd = np.zeros((self.npts, self.npts))
        dCQ_dOmega_fd = np.zeros((self.npts, self.npts))
        dCMy_dOmega_fd = np.zeros((self.npts, self.npts))
        dCMz_dOmega_fd = np.zeros((self.npts, self.npts))
        dCMb_dOmega_fd = np.zeros((self.npts, self.npts))
        dCP_dOmega_fd = np.zeros((self.npts, self.npts))

        Omega = float(self.Omega)
        delta = 1e-6 * Omega
        Omega += delta

        outputs, _ = self.rotor.evaluate([self.Uinf], [Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dOmega_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dOmega_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dOmega_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dOmega_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dOmega_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dOmega_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dOmega_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dOmega_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dOmega_fd, dCT_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dOmega_fd, dCY_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dOmega_fd, dCZ_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dOmega_fd, dCQ_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dOmega_fd, dCMy_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dOmega_fd, dCMz_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dOmega_fd, dCMb_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dOmega_fd, dCP_dOmega, rtol=5e-5, atol=1e-8)

    def test_dpitch1(self):

        dNp_dpitch = self.dNp["dpitch"]
        dTp_dpitch = self.dTp["dpitch"]

        dNp_dpitch_fd = np.zeros((self.n, 1))
        dTp_dpitch_fd = np.zeros((self.n, 1))

        pitch = float(self.pitch)
        delta = 1e-6
        pitch += delta

        loads, _ = self.rotor.distributedAeroLoads(self.Uinf, self.Omega, pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dpitch_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dpitch_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dpitch_fd, dNp_dpitch, rtol=5e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dpitch_fd, dTp_dpitch, rtol=5e-5, atol=1e-6)

    def test_dpitch2(self):

        dT_dpitch = self.dT["dpitch"]
        dY_dpitch = self.dY["dpitch"]
        dZ_dpitch = self.dZ["dpitch"]
        dQ_dpitch = self.dQ["dpitch"]
        dMy_dpitch = self.dMy["dpitch"]
        dMz_dpitch = self.dMz["dpitch"]
        dMb_dpitch = self.dMb["dpitch"]
        dP_dpitch = self.dP["dpitch"]

        dT_dpitch_fd = np.zeros((self.npts, 1))
        dY_dpitch_fd = np.zeros((self.npts, 1))
        dZ_dpitch_fd = np.zeros((self.npts, 1))
        dQ_dpitch_fd = np.zeros((self.npts, 1))
        dMy_dpitch_fd = np.zeros((self.npts, 1))
        dMz_dpitch_fd = np.zeros((self.npts, 1))
        dMb_dpitch_fd = np.zeros((self.npts, 1))
        dP_dpitch_fd = np.zeros((self.npts, 1))

        pitch = float(self.pitch)
        delta = 1e-6
        pitch += delta

        outputs, _ = self.rotor.evaluate([self.Uinf], [self.Omega], [pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dpitch_fd[:, 0] = (Td - self.T) / delta
        dY_dpitch_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dpitch_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dpitch_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dpitch_fd[:, 0] = (Myd - self.My) / delta
        dMz_dpitch_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dpitch_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dpitch_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dpitch_fd, dT_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dpitch_fd, dY_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dpitch_fd, dZ_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dpitch_fd, dQ_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dpitch_fd, dMy_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dpitch_fd, dMz_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dpitch_fd, dMb_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dpitch_fd, dP_dpitch, rtol=5e-5, atol=1e-8)

    def test_dpitch3(self):

        dCT_dpitch = self.dCT["dpitch"]
        dCY_dpitch = self.dCY["dpitch"]
        dCZ_dpitch = self.dCZ["dpitch"]
        dCQ_dpitch = self.dCQ["dpitch"]
        dCMy_dpitch = self.dCMy["dpitch"]
        dCMz_dpitch = self.dCMz["dpitch"]
        dCMb_dpitch = self.dCMb["dpitch"]
        dCP_dpitch = self.dCP["dpitch"]

        dCT_dpitch_fd = np.zeros((self.npts, 1))
        dCY_dpitch_fd = np.zeros((self.npts, 1))
        dCZ_dpitch_fd = np.zeros((self.npts, 1))
        dCQ_dpitch_fd = np.zeros((self.npts, 1))
        dCMy_dpitch_fd = np.zeros((self.npts, 1))
        dCMz_dpitch_fd = np.zeros((self.npts, 1))
        dCMb_dpitch_fd = np.zeros((self.npts, 1))
        dCP_dpitch_fd = np.zeros((self.npts, 1))

        pitch = float(self.pitch)
        delta = 1e-6
        pitch += delta

        outputs, _ = self.rotor.evaluate([self.Uinf], [self.Omega], [pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dpitch_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dpitch_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dpitch_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dpitch_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dpitch_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dpitch_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dpitch_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dpitch_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dpitch_fd, dCT_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dpitch_fd, dCY_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dpitch_fd, dCZ_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dpitch_fd, dCQ_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dpitch_fd, dCMy_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dpitch_fd, dCMz_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dpitch_fd, dCMb_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dpitch_fd, dCP_dpitch, rtol=5e-5, atol=1e-8)

    def test_dprecurve1(self):

        precurve = np.linspace(1, 10, self.n)
        precurveTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            precurve=precurve,
            precurveTip=precurveTip,
        )

        loads, derivs = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Np = loads["Np"]
        Tp = loads["Tp"]
        dNp = derivs["dNp"]
        dTp = derivs["dTp"]

        dNp_dprecurve = dNp["dprecurve"]
        dTp_dprecurve = dTp["dprecurve"]

        dNp_dprecurve_fd = np.zeros((self.n, self.n))
        dTp_dprecurve_fd = np.zeros((self.n, self.n))

        for i in range(self.n):
            pc = np.array(precurve)
            delta = 1e-6 * pc[i]
            pc[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
                precurve=pc,
                precurveTip=precurveTip,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dprecurve_fd[:, i] = (Npd - Np) / delta
            dTp_dprecurve_fd[:, i] = (Tpd - Tp) / delta

        np.testing.assert_allclose(dNp_dprecurve_fd, dNp_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dprecurve_fd, dTp_dprecurve, rtol=3e-4, atol=1e-8)

    def test_dprecurve2(self):

        precurve = np.linspace(1, 10, self.n)
        precurveTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            precurve=precurve,
            precurveTip=precurveTip,
        )

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        P = outputs["P"]
        T = outputs["T"]
        Y = outputs["Y"]
        Z = outputs["Z"]
        Q = outputs["Q"]
        My = outputs["My"]
        Mz = outputs["Mz"]
        Mb = outputs["Mb"]
        dP = derivs["dP"]
        dT = derivs["dT"]
        dY = derivs["dY"]
        dZ = derivs["dZ"]
        dQ = derivs["dQ"]
        dMy = derivs["dMy"]
        dMz = derivs["dMz"]
        dMb = derivs["dMb"]

        dT_dprecurve = dT["dprecurve"]
        dY_dprecurve = dY["dprecurve"]
        dZ_dprecurve = dZ["dprecurve"]
        dQ_dprecurve = dQ["dprecurve"]
        dMy_dprecurve = dMy["dprecurve"]
        dMz_dprecurve = dMz["dprecurve"]
        dMb_dprecurve = dMb["dprecurve"]
        dP_dprecurve = dP["dprecurve"]

        dT_dprecurve_fd = np.zeros((self.npts, self.n))
        dY_dprecurve_fd = np.zeros((self.npts, self.n))
        dZ_dprecurve_fd = np.zeros((self.npts, self.n))
        dQ_dprecurve_fd = np.zeros((self.npts, self.n))
        dMy_dprecurve_fd = np.zeros((self.npts, self.n))
        dMz_dprecurve_fd = np.zeros((self.npts, self.n))
        dMb_dprecurve_fd = np.zeros((self.npts, self.n))
        dP_dprecurve_fd = np.zeros((self.npts, self.n))
        for i in range(self.n):
            pc = np.array(precurve)
            delta = 1e-6 * pc[i]
            pc[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
                precurve=pc,
                precurveTip=precurveTip,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
            Pd = outputs["P"]
            Td = outputs["T"]
            Yd = outputs["Y"]
            Zd = outputs["Z"]
            Qd = outputs["Q"]
            Myd = outputs["My"]
            Mzd = outputs["Mz"]
            Mbd = outputs["Mb"]

            dT_dprecurve_fd[:, i] = (Td - T) / delta
            dY_dprecurve_fd[:, i] = (Yd - Y) / delta
            dZ_dprecurve_fd[:, i] = (Zd - Z) / delta
            dQ_dprecurve_fd[:, i] = (Qd - Q) / delta
            dMy_dprecurve_fd[:, i] = (Myd - My) / delta
            dMz_dprecurve_fd[:, i] = (Mzd - Mz) / delta
            dMb_dprecurve_fd[:, i] = (Mbd - Mb) / delta
            dP_dprecurve_fd[:, i] = (Pd - P) / delta

        np.testing.assert_allclose(dT_dprecurve_fd, dT_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dY_dprecurve_fd, dY_dprecurve, rtol=3e-3, atol=1e-8)
        np.testing.assert_allclose(dZ_dprecurve_fd, dZ_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dQ_dprecurve_fd, dQ_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dMy_dprecurve_fd, dMy_dprecurve, rtol=8e-4, atol=1e-8)
        np.testing.assert_allclose(dMz_dprecurve_fd, dMz_dprecurve, rtol=4e-3, atol=1e-8)
        np.testing.assert_allclose(dMb_dprecurve_fd, dMb_dprecurve, rtol=8e-4, atol=1e-8)
        np.testing.assert_allclose(dP_dprecurve_fd, dP_dprecurve, rtol=3e-4, atol=1e-8)

    def test_dprecurve3(self):

        precurve = np.linspace(1, 10, self.n)
        precurveTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            precurve=precurve,
            precurveTip=precurveTip,
        )

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CP = outputs["CP"]
        CT = outputs["CT"]
        CY = outputs["CY"]
        CZ = outputs["CZ"]
        CQ = outputs["CQ"]
        CMy = outputs["CMy"]
        CMz = outputs["CMz"]
        CMb = outputs["CMb"]
        dCP = derivs["dCP"]
        dCT = derivs["dCT"]
        dCY = derivs["dCY"]
        dCZ = derivs["dCZ"]
        dCQ = derivs["dCQ"]
        dCMy = derivs["dCMy"]
        dCMz = derivs["dCMz"]
        dCMb = derivs["dCMb"]

        dCT_dprecurve = dCT["dprecurve"]
        dCY_dprecurve = dCY["dprecurve"]
        dCZ_dprecurve = dCZ["dprecurve"]
        dCQ_dprecurve = dCQ["dprecurve"]
        dCMy_dprecurve = dCMy["dprecurve"]
        dCMz_dprecurve = dCMz["dprecurve"]
        dCMb_dprecurve = dCMb["dprecurve"]
        dCP_dprecurve = dCP["dprecurve"]

        dCT_dprecurve_fd = np.zeros((self.npts, self.n))
        dCY_dprecurve_fd = np.zeros((self.npts, self.n))
        dCZ_dprecurve_fd = np.zeros((self.npts, self.n))
        dCQ_dprecurve_fd = np.zeros((self.npts, self.n))
        dCMy_dprecurve_fd = np.zeros((self.npts, self.n))
        dCMz_dprecurve_fd = np.zeros((self.npts, self.n))
        dCMb_dprecurve_fd = np.zeros((self.npts, self.n))
        dCP_dprecurve_fd = np.zeros((self.npts, self.n))
        for i in range(self.n):
            pc = np.array(precurve)
            delta = 1e-6 * pc[i]
            pc[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
                precurve=pc,
                precurveTip=precurveTip,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
            CPd = outputs["CP"]
            CTd = outputs["CT"]
            CYd = outputs["CY"]
            CZd = outputs["CZ"]
            CQd = outputs["CQ"]
            CMyd = outputs["CMy"]
            CMzd = outputs["CMz"]
            CMbd = outputs["CMb"]

            dCT_dprecurve_fd[:, i] = (CTd - CT) / delta
            dCY_dprecurve_fd[:, i] = (CYd - CY) / delta
            dCZ_dprecurve_fd[:, i] = (CZd - CZ) / delta
            dCQ_dprecurve_fd[:, i] = (CQd - CQ) / delta
            dCMy_dprecurve_fd[:, i] = (CMyd - CMy) / delta
            dCMz_dprecurve_fd[:, i] = (CMzd - CMz) / delta
            dCMb_dprecurve_fd[:, i] = (CMbd - CMb) / delta
            dCP_dprecurve_fd[:, i] = (CPd - CP) / delta

        np.testing.assert_allclose(dCT_dprecurve_fd, dCT_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCY_dprecurve_fd, dCY_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCZ_dprecurve_fd, dCZ_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCQ_dprecurve_fd, dCQ_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMy_dprecurve_fd, dCMy_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMz_dprecurve_fd, dCMz_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMb_dprecurve_fd, dCMb_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dprecurve_fd, dCP_dprecurve, rtol=3e-4, atol=1e-8)

    def test_dpresweep1(self):

        presweep = np.linspace(1, 10, self.n)
        presweepTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            presweep=presweep,
            presweepTip=presweepTip,
        )

        loads, derivs = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Np = loads["Np"]
        Tp = loads["Tp"]
        dNp = derivs["dNp"]
        dTp = derivs["dTp"]

        dNp_dpresweep = dNp["dpresweep"]
        dTp_dpresweep = dTp["dpresweep"]

        dNp_dpresweep_fd = np.zeros((self.n, self.n))
        dTp_dpresweep_fd = np.zeros((self.n, self.n))

        for i in range(self.n):
            ps = np.array(presweep)
            delta = 1e-6 * ps[i]
            ps[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
                presweep=ps,
                presweepTip=presweepTip,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dpresweep_fd[:, i] = (Npd - Np) / delta
            dTp_dpresweep_fd[:, i] = (Tpd - Tp) / delta

        np.testing.assert_allclose(dNp_dpresweep_fd, dNp_dpresweep, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dpresweep_fd, dTp_dpresweep, rtol=1e-5, atol=1e-8)

    def test_dpresweep2(self):

        presweep = np.linspace(1, 10, self.n)
        presweepTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            presweep=presweep,
            presweepTip=presweepTip,
        )

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        P = outputs["P"]
        T = outputs["T"]
        Y = outputs["Y"]
        Z = outputs["Z"]
        Q = outputs["Q"]
        My = outputs["My"]
        Mz = outputs["Mz"]
        Mb = outputs["Mb"]
        dP = derivs["dP"]
        dT = derivs["dT"]
        dY = derivs["dY"]
        dZ = derivs["dZ"]
        dQ = derivs["dQ"]
        dMy = derivs["dMy"]
        dMz = derivs["dMz"]
        dMb = derivs["dMb"]

        dT_dpresweep = dT["dpresweep"]
        dY_dpresweep = dY["dpresweep"]
        dZ_dpresweep = dZ["dpresweep"]
        dQ_dpresweep = dQ["dpresweep"]
        dMy_dpresweep = dMy["dpresweep"]
        dMz_dpresweep = dMz["dpresweep"]
        dMb_dpresweep = dMb["dpresweep"]
        dP_dpresweep = dP["dpresweep"]

        dT_dpresweep_fd = np.zeros((self.npts, self.n))
        dY_dpresweep_fd = np.zeros((self.npts, self.n))
        dZ_dpresweep_fd = np.zeros((self.npts, self.n))
        dQ_dpresweep_fd = np.zeros((self.npts, self.n))
        dMy_dpresweep_fd = np.zeros((self.npts, self.n))
        dMz_dpresweep_fd = np.zeros((self.npts, self.n))
        dMb_dpresweep_fd = np.zeros((self.npts, self.n))
        dP_dpresweep_fd = np.zeros((self.npts, self.n))
        for i in range(self.n):
            ps = np.array(presweep)
            delta = 1e-6 * ps[i]
            ps[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
                presweep=ps,
                presweepTip=presweepTip,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
            Pd = outputs["P"]
            Td = outputs["T"]
            Yd = outputs["Y"]
            Zd = outputs["Z"]
            Qd = outputs["Q"]
            Myd = outputs["My"]
            Mzd = outputs["Mz"]
            Mbd = outputs["Mb"]

            dT_dpresweep_fd[:, i] = (Td - T) / delta
            dY_dpresweep_fd[:, i] = (Yd - Y) / delta
            dZ_dpresweep_fd[:, i] = (Zd - Z) / delta
            dQ_dpresweep_fd[:, i] = (Qd - Q) / delta
            dMy_dpresweep_fd[:, i] = (Myd - My) / delta
            dMz_dpresweep_fd[:, i] = (Mzd - Mz) / delta
            dMb_dpresweep_fd[:, i] = (Mbd - Mb) / delta
            dP_dpresweep_fd[:, i] = (Pd - P) / delta

        np.testing.assert_allclose(dT_dpresweep_fd, dT_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dY_dpresweep_fd, dY_dpresweep, rtol=2e-3, atol=1e-8)
        np.testing.assert_allclose(dZ_dpresweep_fd, dZ_dpresweep, rtol=4e-3, atol=1e-8)
        np.testing.assert_allclose(dQ_dpresweep_fd, dQ_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dMy_dpresweep_fd, dMy_dpresweep, rtol=1e-3, atol=1e-8)
        np.testing.assert_allclose(dMz_dpresweep_fd, dMz_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dMb_dpresweep_fd, dMb_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dP_dpresweep_fd, dP_dpresweep, rtol=3e-4, atol=1e-8)

    def test_dpresweep3(self):

        presweep = np.linspace(1, 10, self.n)
        presweepTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            presweep=presweep,
            presweepTip=presweepTip,
        )

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CP = outputs["CP"]
        CT = outputs["CT"]
        CY = outputs["CY"]
        CZ = outputs["CZ"]
        CQ = outputs["CQ"]
        CMy = outputs["CMy"]
        CMz = outputs["CMz"]
        CMb = outputs["CMb"]
        dCP = derivs["dCP"]
        dCT = derivs["dCT"]
        dCY = derivs["dCY"]
        dCZ = derivs["dCZ"]
        dCQ = derivs["dCQ"]
        dCMy = derivs["dCMy"]
        dCMz = derivs["dCMz"]
        dCMb = derivs["dCMb"]

        dCT_dpresweep = dCT["dpresweep"]
        dCY_dpresweep = dCY["dpresweep"]
        dCZ_dpresweep = dCZ["dpresweep"]
        dCQ_dpresweep = dCQ["dpresweep"]
        dCMy_dpresweep = dCMy["dpresweep"]
        dCMz_dpresweep = dCMz["dpresweep"]
        dCMb_dpresweep = dCMb["dpresweep"]
        dCP_dpresweep = dCP["dpresweep"]

        dCT_dpresweep_fd = np.zeros((self.npts, self.n))
        dCY_dpresweep_fd = np.zeros((self.npts, self.n))
        dCZ_dpresweep_fd = np.zeros((self.npts, self.n))
        dCQ_dpresweep_fd = np.zeros((self.npts, self.n))
        dCMy_dpresweep_fd = np.zeros((self.npts, self.n))
        dCMz_dpresweep_fd = np.zeros((self.npts, self.n))
        dCMb_dpresweep_fd = np.zeros((self.npts, self.n))
        dCP_dpresweep_fd = np.zeros((self.npts, self.n))
        for i in range(self.n):
            ps = np.array(presweep)
            delta = 1e-6 * ps[i]
            ps[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
                presweep=ps,
                presweepTip=presweepTip,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
            CPd = outputs["CP"]
            CTd = outputs["CT"]
            CYd = outputs["CY"]
            CZd = outputs["CZ"]
            CQd = outputs["CQ"]
            CMyd = outputs["CMy"]
            CMzd = outputs["CMz"]
            CMbd = outputs["CMb"]

            dCT_dpresweep_fd[:, i] = (CTd - CT) / delta
            dCY_dpresweep_fd[:, i] = (CYd - CY) / delta
            dCZ_dpresweep_fd[:, i] = (CZd - CZ) / delta
            dCQ_dpresweep_fd[:, i] = (CQd - CQ) / delta
            dCMy_dpresweep_fd[:, i] = (CMyd - CMy) / delta
            dCMz_dpresweep_fd[:, i] = (CMzd - CMz) / delta
            dCMb_dpresweep_fd[:, i] = (CMbd - CMb) / delta
            dCP_dpresweep_fd[:, i] = (CPd - CP) / delta

        np.testing.assert_allclose(dCT_dpresweep_fd, dCT_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCY_dpresweep_fd, dCY_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCZ_dpresweep_fd, dCZ_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCQ_dpresweep_fd, dCQ_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMy_dpresweep_fd, dCMy_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMz_dpresweep_fd, dCMz_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMb_dpresweep_fd, dCMb_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dpresweep_fd, dCP_dpresweep, rtol=3e-4, atol=1e-8)

    def test_dprecurveTip1(self):

        precurve = np.linspace(1, 10, self.n)
        precurveTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            precurve=precurve,
            precurveTip=precurveTip,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Np = loads["Np"]
        Tp = loads["Tp"]

        dNp_dprecurveTip_fd = np.zeros((self.n, 1))
        dTp_dprecurveTip_fd = np.zeros((self.n, 1))

        pct = float(precurveTip)
        delta = 1e-6 * pct
        pct += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
            precurve=precurve,
            precurveTip=pct,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]
        dNp_dprecurveTip_fd[:, 0] = (Npd - Np) / delta
        dTp_dprecurveTip_fd[:, 0] = (Tpd - Tp) / delta

        np.testing.assert_allclose(dNp_dprecurveTip_fd, 0.0, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dprecurveTip_fd, 0.0, rtol=1e-4, atol=1e-8)

    def test_dprecurveTip2(self):

        precurve = np.linspace(1, 10, self.n)
        precurveTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            precurve=precurve,
            precurveTip=precurveTip,
        )

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        P = outputs["P"]
        T = outputs["T"]
        Y = outputs["Y"]
        Z = outputs["Z"]
        Q = outputs["Q"]
        My = outputs["My"]
        Mz = outputs["Mz"]
        Mb = outputs["Mb"]
        dP = derivs["dP"]
        dT = derivs["dT"]
        dY = derivs["dY"]
        dZ = derivs["dZ"]
        dQ = derivs["dQ"]
        dMy = derivs["dMy"]
        dMz = derivs["dMz"]
        dMb = derivs["dMb"]

        dT_dprecurveTip = dT["dprecurveTip"]
        dY_dprecurveTip = dY["dprecurveTip"]
        dZ_dprecurveTip = dZ["dprecurveTip"]
        dQ_dprecurveTip = dQ["dprecurveTip"]
        dMy_dprecurveTip = dMy["dprecurveTip"]
        dMz_dprecurveTip = dMz["dprecurveTip"]
        dMb_dprecurveTip = dMb["dprecurveTip"]
        dP_dprecurveTip = dP["dprecurveTip"]

        dT_dprecurveTip_fd = np.zeros((self.npts, 1))
        dY_dprecurveTip_fd = np.zeros((self.npts, 1))
        dZ_dprecurveTip_fd = np.zeros((self.npts, 1))
        dQ_dprecurveTip_fd = np.zeros((self.npts, 1))
        dMy_dprecurveTip_fd = np.zeros((self.npts, 1))
        dMz_dprecurveTip_fd = np.zeros((self.npts, 1))
        dMb_dprecurveTip_fd = np.zeros((self.npts, 1))
        dP_dprecurveTip_fd = np.zeros((self.npts, 1))

        pct = float(precurveTip)
        delta = 1e-6 * pct
        pct += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
            precurve=precurve,
            precurveTip=pct,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dprecurveTip_fd[:, 0] = (Td - T) / delta
        dY_dprecurveTip_fd[:, 0] = (Yd - Y) / delta
        dZ_dprecurveTip_fd[:, 0] = (Zd - Z) / delta
        dQ_dprecurveTip_fd[:, 0] = (Qd - Q) / delta
        dMy_dprecurveTip_fd[:, 0] = (Myd - My) / delta
        dMz_dprecurveTip_fd[:, 0] = (Mzd - Mz) / delta
        dMb_dprecurveTip_fd[:, 0] = (Mbd - Mb) / delta
        dP_dprecurveTip_fd[:, 0] = (Pd - P) / delta

        np.testing.assert_allclose(dT_dprecurveTip_fd, dT_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dY_dprecurveTip_fd, dY_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dZ_dprecurveTip_fd, dZ_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dQ_dprecurveTip_fd, dQ_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dMy_dprecurveTip_fd, dMy_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dMz_dprecurveTip_fd, dMz_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dMb_dprecurveTip_fd, dMb_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dP_dprecurveTip_fd, dP_dprecurveTip, rtol=1e-4, atol=1e-8)

    def test_dprecurveTip3(self):

        precurve = np.linspace(1, 10, self.n)
        precurveTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            precurve=precurve,
            precurveTip=precurveTip,
        )

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CP = outputs["CP"]
        CT = outputs["CT"]
        CY = outputs["CY"]
        CZ = outputs["CZ"]
        CQ = outputs["CQ"]
        CMy = outputs["CMy"]
        CMz = outputs["CMz"]
        CMb = outputs["CMb"]
        dCP = derivs["dCP"]
        dCT = derivs["dCT"]
        dCY = derivs["dCY"]
        dCZ = derivs["dCZ"]
        dCQ = derivs["dCQ"]
        dCMy = derivs["dCMy"]
        dCMz = derivs["dCMz"]
        dCMb = derivs["dCMb"]

        dCT_dprecurveTip = dCT["dprecurveTip"]
        dCY_dprecurveTip = dCY["dprecurveTip"]
        dCZ_dprecurveTip = dCZ["dprecurveTip"]
        dCQ_dprecurveTip = dCQ["dprecurveTip"]
        dCMy_dprecurveTip = dCMy["dprecurveTip"]
        dCMz_dprecurveTip = dCMz["dprecurveTip"]
        dCMb_dprecurveTip = dCMb["dprecurveTip"]
        dCP_dprecurveTip = dCP["dprecurveTip"]

        dCT_dprecurveTip_fd = np.zeros((self.npts, 1))
        dCY_dprecurveTip_fd = np.zeros((self.npts, 1))
        dCZ_dprecurveTip_fd = np.zeros((self.npts, 1))
        dCQ_dprecurveTip_fd = np.zeros((self.npts, 1))
        dCMy_dprecurveTip_fd = np.zeros((self.npts, 1))
        dCMz_dprecurveTip_fd = np.zeros((self.npts, 1))
        dCMb_dprecurveTip_fd = np.zeros((self.npts, 1))
        dCP_dprecurveTip_fd = np.zeros((self.npts, 1))

        pct = float(precurveTip)
        delta = 1e-6 * pct
        pct += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
            precurve=precurve,
            precurveTip=pct,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dprecurveTip_fd[:, 0] = (CTd - CT) / delta
        dCY_dprecurveTip_fd[:, 0] = (CYd - CY) / delta
        dCZ_dprecurveTip_fd[:, 0] = (CZd - CZ) / delta
        dCQ_dprecurveTip_fd[:, 0] = (CQd - CQ) / delta
        dCMy_dprecurveTip_fd[:, 0] = (CMyd - CMy) / delta
        dCMz_dprecurveTip_fd[:, 0] = (CMzd - CMz) / delta
        dCMb_dprecurveTip_fd[:, 0] = (CMbd - CMb) / delta
        dCP_dprecurveTip_fd[:, 0] = (CPd - CP) / delta

        np.testing.assert_allclose(dCT_dprecurveTip_fd, dCT_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCY_dprecurveTip_fd, dCY_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCZ_dprecurveTip_fd, dCZ_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCQ_dprecurveTip_fd, dCQ_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCMy_dprecurveTip_fd, dCMy_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCMz_dprecurveTip_fd, dCMz_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCMb_dprecurveTip_fd, dCMb_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dprecurveTip_fd, dCP_dprecurveTip, rtol=1e-4, atol=1e-8)

    def test_dpresweepTip1(self):

        presweep = np.linspace(1, 10, self.n)
        presweepTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            presweep=presweep,
            presweepTip=presweepTip,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Np = loads["Np"]
        Tp = loads["Tp"]

        dNp_dpresweepTip_fd = np.zeros((self.n, 1))
        dTp_dpresweepTip_fd = np.zeros((self.n, 1))

        pst = float(presweepTip)
        delta = 1e-6 * pst
        pst += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
            presweep=presweep,
            presweepTip=pst,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]
        dNp_dpresweepTip_fd[:, 0] = (Npd - Np) / delta
        dTp_dpresweepTip_fd[:, 0] = (Tpd - Tp) / delta

        np.testing.assert_allclose(dNp_dpresweepTip_fd, 0.0, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dpresweepTip_fd, 0.0, rtol=1e-4, atol=1e-8)

    def test_dpresweepTip2(self):

        presweep = np.linspace(1, 10, self.n)
        presweepTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            presweep=presweep,
            presweepTip=presweepTip,
        )

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        P = outputs["P"]
        T = outputs["T"]
        Y = outputs["Y"]
        Z = outputs["Z"]
        Q = outputs["Q"]
        My = outputs["My"]
        Mz = outputs["Mz"]
        Mb = outputs["Mb"]
        dP = derivs["dP"]
        dT = derivs["dT"]
        dY = derivs["dY"]
        dZ = derivs["dZ"]
        dQ = derivs["dQ"]
        dMy = derivs["dMy"]
        dMz = derivs["dMz"]
        dMb = derivs["dMb"]

        dT_dpresweepTip = dT["dpresweepTip"]
        dY_dpresweepTip = dY["dpresweepTip"]
        dZ_dpresweepTip = dZ["dpresweepTip"]
        dQ_dpresweepTip = dQ["dpresweepTip"]
        dMy_dpresweepTip = dMy["dpresweepTip"]
        dMz_dpresweepTip = dMz["dpresweepTip"]
        dMb_dpresweepTip = dMb["dpresweepTip"]
        dP_dpresweepTip = dP["dpresweepTip"]

        dT_dpresweepTip_fd = np.zeros((self.npts, 1))
        dY_dpresweepTip_fd = np.zeros((self.npts, 1))
        dZ_dpresweepTip_fd = np.zeros((self.npts, 1))
        dQ_dpresweepTip_fd = np.zeros((self.npts, 1))
        dMy_dpresweepTip_fd = np.zeros((self.npts, 1))
        dMz_dpresweepTip_fd = np.zeros((self.npts, 1))
        dMb_dpresweepTip_fd = np.zeros((self.npts, 1))
        dP_dpresweepTip_fd = np.zeros((self.npts, 1))

        pst = float(presweepTip)
        delta = 1e-6 * pst
        pst += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
            presweep=presweep,
            presweepTip=pst,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dpresweepTip_fd[:, 0] = (Td - T) / delta
        dY_dpresweepTip_fd[:, 0] = (Yd - Y) / delta
        dZ_dpresweepTip_fd[:, 0] = (Zd - Z) / delta
        dQ_dpresweepTip_fd[:, 0] = (Qd - Q) / delta
        dMy_dpresweepTip_fd[:, 0] = (Myd - My) / delta
        dMz_dpresweepTip_fd[:, 0] = (Mzd - Mz) / delta
        dMb_dpresweepTip_fd[:, 0] = (Mbd - Mb) / delta
        dP_dpresweepTip_fd[:, 0] = (Pd - P) / delta

        np.testing.assert_allclose(dT_dpresweepTip_fd, dT_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dY_dpresweepTip_fd, dY_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dZ_dpresweepTip_fd, dZ_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dQ_dpresweepTip_fd, dQ_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dMy_dpresweepTip_fd, dMy_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dMz_dpresweepTip_fd, dMz_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dMb_dpresweepTip_fd, dMb_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dP_dpresweepTip_fd, dP_dpresweepTip, rtol=1e-4, atol=1e-8)

    def test_dpresweepTip3(self):

        presweep = np.linspace(1, 10, self.n)
        presweepTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            presweep=presweep,
            presweepTip=presweepTip,
        )

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CP = outputs["CP"]
        CT = outputs["CT"]
        CY = outputs["CY"]
        CZ = outputs["CZ"]
        CQ = outputs["CQ"]
        CMy = outputs["CMy"]
        CMz = outputs["CMz"]
        CMb = outputs["CMb"]
        dCP = derivs["dCP"]
        dCT = derivs["dCT"]
        dCY = derivs["dCY"]
        dCZ = derivs["dCZ"]
        dCQ = derivs["dCQ"]
        dCMy = derivs["dCMy"]
        dCMz = derivs["dCMz"]
        dCMb = derivs["dCMb"]

        dCT_dpresweepTip = dCT["dpresweepTip"]
        dCY_dpresweepTip = dCY["dpresweepTip"]
        dCZ_dpresweepTip = dCZ["dpresweepTip"]
        dCQ_dpresweepTip = dCQ["dpresweepTip"]
        dCMy_dpresweepTip = dCMy["dpresweepTip"]
        dCMz_dpresweepTip = dCMz["dpresweepTip"]
        dCMb_dpresweepTip = dCMb["dpresweepTip"]
        dCP_dpresweepTip = dCP["dpresweepTip"]

        dCT_dpresweepTip_fd = np.zeros((self.npts, 1))
        dCY_dpresweepTip_fd = np.zeros((self.npts, 1))
        dCZ_dpresweepTip_fd = np.zeros((self.npts, 1))
        dCQ_dpresweepTip_fd = np.zeros((self.npts, 1))
        dCMy_dpresweepTip_fd = np.zeros((self.npts, 1))
        dCMz_dpresweepTip_fd = np.zeros((self.npts, 1))
        dCMb_dpresweepTip_fd = np.zeros((self.npts, 1))
        dCP_dpresweepTip_fd = np.zeros((self.npts, 1))

        pst = float(presweepTip)
        delta = 1e-6 * pst
        pst += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
            presweep=presweep,
            presweepTip=pst,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dpresweepTip_fd[:, 0] = (CTd - CT) / delta
        dCY_dpresweepTip_fd[:, 0] = (CYd - CY) / delta
        dCZ_dpresweepTip_fd[:, 0] = (CZd - CZ) / delta
        dCQ_dpresweepTip_fd[:, 0] = (CQd - CQ) / delta
        dCMy_dpresweepTip_fd[:, 0] = (CMyd - CMy) / delta
        dCMz_dpresweepTip_fd[:, 0] = (CMzd - CMz) / delta
        dCMb_dpresweepTip_fd[:, 0] = (CMbd - CMb) / delta
        dCP_dpresweepTip_fd[:, 0] = (CPd - CP) / delta

        np.testing.assert_allclose(dCT_dpresweepTip_fd, dCT_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCY_dpresweepTip_fd, dCY_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCZ_dpresweepTip_fd, dCZ_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCQ_dpresweepTip_fd, dCQ_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCMy_dpresweepTip_fd, dCMy_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCMz_dpresweepTip_fd, dCMz_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCMb_dpresweepTip_fd, dCMb_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dpresweepTip_fd, dCP_dpresweepTip, rtol=1e-4, atol=1e-8)


class TestGradientsNotRotating(unittest.TestCase):
    def setUp(self):

        # geometry
        self.Rhub = 1.5
        self.Rtip = 63.0

        self.r = np.array(
            [
                2.8667,
                5.6000,
                8.3333,
                11.7500,
                15.8500,
                19.9500,
                24.0500,
                28.1500,
                32.2500,
                36.3500,
                40.4500,
                44.5500,
                48.6500,
                52.7500,
                56.1667,
                58.9000,
                61.6333,
            ]
        )
        self.chord = np.array(
            [
                3.542,
                3.854,
                4.167,
                4.557,
                4.652,
                4.458,
                4.249,
                4.007,
                3.748,
                3.502,
                3.256,
                3.010,
                2.764,
                2.518,
                2.313,
                2.086,
                1.419,
            ]
        )
        self.theta = np.array(
            [
                13.308,
                13.308,
                13.308,
                13.308,
                11.480,
                10.162,
                9.011,
                7.795,
                6.544,
                5.361,
                4.188,
                3.125,
                2.319,
                1.526,
                0.863,
                0.370,
                0.106,
            ]
        )
        self.B = 3  # number of blades

        # atmosphere
        self.rho = 1.225
        self.mu = 1.81206e-5

        afinit = CCAirfoil.initFromAerodynFile  # just for shorthand

        # load all airfoils
        airfoil_types = [0] * 8
        airfoil_types[0] = afinit(basepath + os.sep + "Cylinder1.dat")
        airfoil_types[1] = afinit(basepath + os.sep + "Cylinder2.dat")
        airfoil_types[2] = afinit(basepath + os.sep + "DU40_A17.dat")
        airfoil_types[3] = afinit(basepath + os.sep + "DU35_A17.dat")
        airfoil_types[4] = afinit(basepath + os.sep + "DU30_A17.dat")
        airfoil_types[5] = afinit(basepath + os.sep + "DU25_A17.dat")
        airfoil_types[6] = afinit(basepath + os.sep + "DU21_A17.dat")
        airfoil_types[7] = afinit(basepath + os.sep + "NACA64_A17.dat")

        # place at appropriate radial stations
        af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

        self.af = [0] * len(self.r)
        for i in range(len(self.r)):
            self.af[i] = airfoil_types[af_idx[i]]

        self.tilt = -5.0
        self.precone = 2.5
        self.yaw = 0.0
        self.shearExp = 0.2
        self.hubHt = 80.0
        self.nSector = 8

        # create CCBlade object
        self.rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
        )

        # set conditions
        self.Uinf = 10.0
        self.pitch = 0.0
        self.Omega = 0.0  # convert to RPM
        self.azimuth = 90

        loads, derivs = self.rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        self.Np = loads["Np"]
        self.Tp = loads["Tp"]
        self.dNp = derivs["dNp"]
        self.dTp = derivs["dTp"]

        self.rotor.derivatives = False
        self.n = len(self.r)
        self.npts = 1  # len(Uinf)

    def test_dr1(self):

        dNp_dr = self.dNp["dr"]
        dTp_dr = self.dTp["dr"]
        dNp_dr_fd = np.zeros((self.n, self.n))
        dTp_dr_fd = np.zeros((self.n, self.n))

        for i in range(self.n):
            r = np.array(self.r)
            delta = 1e-6 * r[i]
            r[i] += delta

            rotor = CCBlade(
                r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dr_fd[:, i] = (Npd - self.Np) / delta
            dTp_dr_fd[:, i] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dr_fd, dNp_dr, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dr_fd, dTp_dr, rtol=1e-4, atol=1e-8)

    def test_dchord1(self):

        dNp_dchord = self.dNp["dchord"]
        dTp_dchord = self.dTp["dchord"]
        dNp_dchord_fd = np.zeros((self.n, self.n))
        dTp_dchord_fd = np.zeros((self.n, self.n))

        for i in range(self.n):
            chord = np.array(self.chord)
            delta = 1e-6 * chord[i]
            chord[i] += delta

            rotor = CCBlade(
                self.r,
                chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dchord_fd[:, i] = (Npd - self.Np) / delta
            dTp_dchord_fd[:, i] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dchord_fd, dNp_dchord, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dTp_dchord_fd, dTp_dchord, rtol=5e-5, atol=1e-8)

    def test_dtheta1(self):

        dNp_dtheta = self.dNp["dtheta"]
        dTp_dtheta = self.dTp["dtheta"]
        dNp_dtheta_fd = np.zeros((self.n, self.n))
        dTp_dtheta_fd = np.zeros((self.n, self.n))

        for i in range(self.n):
            theta = np.array(self.theta)
            delta = 1e-6 * theta[i]
            theta[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dtheta_fd[:, i] = (Npd - self.Np) / delta
            dTp_dtheta_fd[:, i] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dtheta_fd, dNp_dtheta, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(dTp_dtheta_fd, dTp_dtheta, rtol=1e-4, atol=1e-6)

    def test_dRhub1(self):

        dNp_dRhub = self.dNp["dRhub"]
        dTp_dRhub = self.dTp["dRhub"]

        dNp_dRhub_fd = np.zeros((self.n, 1))
        dTp_dRhub_fd = np.zeros((self.n, 1))

        Rhub = float(self.Rhub)
        delta = 1e-6 * Rhub
        Rhub += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = self.rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dRhub_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dRhub_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dRhub_fd, dNp_dRhub, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(dTp_dRhub_fd, dTp_dRhub, rtol=1e-4, atol=1e-7)

    def test_dRtip1(self):

        dNp_dRtip = self.dNp["dRtip"]
        dTp_dRtip = self.dTp["dRtip"]

        dNp_dRtip_fd = np.zeros((self.n, 1))
        dTp_dRtip_fd = np.zeros((self.n, 1))

        Rtip = float(self.Rtip)
        delta = 1e-6 * Rtip
        Rtip += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dRtip_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dRtip_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dRtip_fd, dNp_dRtip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dRtip_fd, dTp_dRtip, rtol=1e-4, atol=1e-8)

    def test_dprecone1(self):

        dNp_dprecone = self.dNp["dprecone"]
        dTp_dprecone = self.dTp["dprecone"]

        dNp_dprecone_fd = np.zeros((self.n, 1))
        dTp_dprecone_fd = np.zeros((self.n, 1))

        precone = float(self.precone)
        delta = 1e-6 * precone
        precone += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dprecone_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dprecone_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dprecone_fd, dNp_dprecone, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dTp_dprecone_fd, dTp_dprecone, rtol=1e-6, atol=1e-8)

    def test_dtilt1(self):

        dNp_dtilt = self.dNp["dtilt"]
        dTp_dtilt = self.dTp["dtilt"]

        dNp_dtilt_fd = np.zeros((self.n, 1))
        dTp_dtilt_fd = np.zeros((self.n, 1))

        tilt = float(self.tilt)
        delta = 1e-6 * tilt
        tilt += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dtilt_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dtilt_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dtilt_fd, dNp_dtilt, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(dTp_dtilt_fd, dTp_dtilt, rtol=1e-5, atol=1e-6)

    def test_dhubht1(self):

        dNp_dhubht = self.dNp["dhubHt"]
        dTp_dhubht = self.dTp["dhubHt"]

        dNp_dhubht_fd = np.zeros((self.n, 1))
        dTp_dhubht_fd = np.zeros((self.n, 1))

        hubht = float(self.hubHt)
        delta = 1e-6 * hubht
        hubht += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            hubht,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dhubht_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dhubht_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dhubht_fd, dNp_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dhubht_fd, dTp_dhubht, rtol=1e-5, atol=1e-8)

    def test_dyaw1(self):

        dNp_dyaw = self.dNp["dyaw"]
        dTp_dyaw = self.dTp["dyaw"]

        dNp_dyaw_fd = np.zeros((self.n, 1))
        dTp_dyaw_fd = np.zeros((self.n, 1))

        yaw = float(self.yaw)
        delta = 1e-6
        yaw += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dyaw_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dyaw_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dyaw_fd, dNp_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dyaw_fd, dTp_dyaw, rtol=1e-5, atol=1e-8)

    def test_dshear1(self):

        dNp_dshear = self.dNp["dshear"]
        dTp_dshear = self.dTp["dshear"]

        dNp_dshear_fd = np.zeros((self.n, 1))
        dTp_dshear_fd = np.zeros((self.n, 1))

        shearExp = float(self.shearExp)
        delta = 1e-6
        shearExp += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dshear_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dshear_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dshear_fd, dNp_dshear, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dshear_fd, dTp_dshear, rtol=1e-5, atol=1e-8)

    def test_dazimuth1(self):

        dNp_dazimuth = self.dNp["dazimuth"]
        dTp_dazimuth = self.dTp["dazimuth"]

        dNp_dazimuth_fd = np.zeros((self.n, 1))
        dTp_dazimuth_fd = np.zeros((self.n, 1))

        azimuth = float(self.azimuth)
        delta = 1e-6 * azimuth
        azimuth += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = self.rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dazimuth_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dazimuth_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dazimuth_fd, dNp_dazimuth, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dazimuth_fd, dTp_dazimuth, rtol=1e-5, atol=1e-6)

    def test_dUinf1(self):

        dNp_dUinf = self.dNp["dUinf"]
        dTp_dUinf = self.dTp["dUinf"]

        dNp_dUinf_fd = np.zeros((self.n, 1))
        dTp_dUinf_fd = np.zeros((self.n, 1))

        Uinf = float(self.Uinf)
        delta = 1e-6 * Uinf
        Uinf += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = self.rotor.distributedAeroLoads(Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dUinf_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dUinf_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dUinf_fd, dNp_dUinf, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dUinf_fd, dTp_dUinf, rtol=1e-5, atol=1e-6)

    #
    # Omega is fixed at 0 so no need to run derivatives test
    #

    def test_dpitch1(self):

        dNp_dpitch = self.dNp["dpitch"]
        dTp_dpitch = self.dTp["dpitch"]

        dNp_dpitch_fd = np.zeros((self.n, 1))
        dTp_dpitch_fd = np.zeros((self.n, 1))

        pitch = float(self.pitch)
        delta = 1e-6
        pitch += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = self.rotor.distributedAeroLoads(self.Uinf, self.Omega, pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dpitch_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dpitch_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dpitch_fd, dNp_dpitch, rtol=5e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dpitch_fd, dTp_dpitch, rtol=5e-5, atol=1e-6)

    def test_dprecurve1(self):

        precurve = np.linspace(1, 10, self.n)
        precurveTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            precurve=precurve,
            precurveTip=precurveTip,
        )

        loads, derivs = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Np = loads["Np"]
        Tp = loads["Tp"]
        dNp = derivs["dNp"]
        dTp = derivs["dTp"]

        dNp_dprecurve = dNp["dprecurve"]
        dTp_dprecurve = dTp["dprecurve"]

        dNp_dprecurve_fd = np.zeros((self.n, self.n))
        dTp_dprecurve_fd = np.zeros((self.n, self.n))
        for i in range(self.n):
            pc = np.array(precurve)
            delta = 1e-6 * pc[i]
            pc[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
                precurve=pc,
                precurveTip=precurveTip,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dprecurve_fd[:, i] = (Npd - Np) / delta
            dTp_dprecurve_fd[:, i] = (Tpd - Tp) / delta

        np.testing.assert_allclose(dNp_dprecurve_fd, dNp_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dprecurve_fd, dTp_dprecurve, rtol=3e-4, atol=1e-8)

    def test_dpresweep1(self):

        presweep = np.linspace(1, 10, self.n)
        presweepTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            presweep=presweep,
            presweepTip=presweepTip,
        )

        loads, derivs = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Np = loads["Np"]
        Tp = loads["Tp"]
        dNp = derivs["dNp"]
        dTp = derivs["dTp"]

        dNp_dpresweep = dNp["dpresweep"]
        dTp_dpresweep = dTp["dpresweep"]

        dNp_dpresweep_fd = np.zeros((self.n, self.n))
        dTp_dpresweep_fd = np.zeros((self.n, self.n))
        for i in range(self.n):
            ps = np.array(presweep)
            delta = 1e-6 * ps[i]
            ps[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
                presweep=ps,
                presweepTip=presweepTip,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dpresweep_fd[:, i] = (Npd - Np) / delta
            dTp_dpresweep_fd[:, i] = (Tpd - Tp) / delta

        np.testing.assert_allclose(dNp_dpresweep_fd, dNp_dpresweep, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dpresweep_fd, dTp_dpresweep, rtol=1e-5, atol=1e-8)

    def test_dprecurveTip1(self):

        precurve = np.linspace(1, 10, self.n)
        precurveTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            precurve=precurve,
            precurveTip=precurveTip,
        )

        loads, derivs = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Np = loads["Np"]
        Tp = loads["Tp"]
        dNp = derivs["dNp"]
        dTp = derivs["dTp"]

        dNp_dprecurveTip_fd = np.zeros((self.n, 1))
        dTp_dprecurveTip_fd = np.zeros((self.n, 1))

        pct = float(precurveTip)
        delta = 1e-6 * pct
        pct += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
            precurve=precurve,
            precurveTip=pct,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]
        dNp_dprecurveTip_fd[:, 0] = (Npd - Np) / delta
        dTp_dprecurveTip_fd[:, 0] = (Tpd - Tp) / delta

        np.testing.assert_allclose(dNp_dprecurveTip_fd, 0.0, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dprecurveTip_fd, 0.0, rtol=1e-4, atol=1e-8)

    def test_dpresweepTip1(self):

        presweep = np.linspace(1, 10, self.n)
        presweepTip = 10.1
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            presweep=presweep,
            presweepTip=presweepTip,
        )

        loads, derivs = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Np = loads["Np"]
        Tp = loads["Tp"]
        dNp = derivs["dNp"]
        dTp = derivs["dTp"]

        dNp_dpresweepTip_fd = np.zeros((self.n, 1))
        dTp_dpresweepTip_fd = np.zeros((self.n, 1))

        pst = float(presweepTip)
        delta = 1e-6 * pst
        pst += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
            presweep=presweep,
            presweepTip=pst,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]
        dNp_dpresweepTip_fd[:, 0] = (Npd - Np) / delta
        dTp_dpresweepTip_fd[:, 0] = (Tpd - Tp) / delta

        np.testing.assert_allclose(dNp_dpresweepTip_fd, 0.0, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dpresweepTip_fd, 0.0, rtol=1e-4, atol=1e-8)


class TestGradientsFreestreamArray(unittest.TestCase):
    def setUp(self):

        # geometry
        self.Rhub = 1.5
        self.Rtip = 63.0

        self.r = np.array(
            [
                2.8667,
                5.6000,
                8.3333,
                11.7500,
                15.8500,
                19.9500,
                24.0500,
                28.1500,
                32.2500,
                36.3500,
                40.4500,
                44.5500,
                48.6500,
                52.7500,
                56.1667,
                58.9000,
                61.6333,
            ]
        )
        self.chord = np.array(
            [
                3.542,
                3.854,
                4.167,
                4.557,
                4.652,
                4.458,
                4.249,
                4.007,
                3.748,
                3.502,
                3.256,
                3.010,
                2.764,
                2.518,
                2.313,
                2.086,
                1.419,
            ]
        )
        self.theta = np.array(
            [
                13.308,
                13.308,
                13.308,
                13.308,
                11.480,
                10.162,
                9.011,
                7.795,
                6.544,
                5.361,
                4.188,
                3.125,
                2.319,
                1.526,
                0.863,
                0.370,
                0.106,
            ]
        )
        self.B = 3  # number of blades

        # atmosphere
        self.rho = 1.225
        self.mu = 1.81206e-5

        afinit = CCAirfoil.initFromAerodynFile  # just for shorthand

        # load all airfoils
        airfoil_types = [0] * 8
        airfoil_types[0] = afinit(basepath + os.sep + "Cylinder1.dat")
        airfoil_types[1] = afinit(basepath + os.sep + "Cylinder2.dat")
        airfoil_types[2] = afinit(basepath + os.sep + "DU40_A17.dat")
        airfoil_types[3] = afinit(basepath + os.sep + "DU35_A17.dat")
        airfoil_types[4] = afinit(basepath + os.sep + "DU30_A17.dat")
        airfoil_types[5] = afinit(basepath + os.sep + "DU25_A17.dat")
        airfoil_types[6] = afinit(basepath + os.sep + "DU21_A17.dat")
        airfoil_types[7] = afinit(basepath + os.sep + "NACA64_A17.dat")

        # place at appropriate radial stations
        af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

        self.af = [0] * len(self.r)
        for i in range(len(self.r)):
            self.af[i] = airfoil_types[af_idx[i]]

        self.tilt = -5.0
        self.precone = 2.5
        self.yaw = 0.0
        self.shearExp = 0.2
        self.hubHt = 80.0
        self.nSector = 8

        # create CCBlade object
        self.rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
        )

        # set conditions
        self.Uinf = np.array([10.0, 11.0, 12.0])
        tsr = 7.55
        self.pitch = np.zeros(3)
        self.Omega = self.Uinf * tsr / self.Rtip * 30.0 / np.pi  # convert to RPM

        outputs, derivs = self.rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        self.P, self.T, self.Y, self.Z, self.Q, self.My, self.Mz, self.Mb = [
            outputs[k] for k in ("P", "T", "Y", "Z", "Q", "My", "Mz", "Mb")
        ]
        self.dP, self.dT, self.dY, self.dZ, self.dQ, self.dMy, self.dMz, self.dMb = [
            derivs[k] for k in ("dP", "dT", "dY", "dZ", "dQ", "dMy", "dMz", "dMb")
        ]
        self.CP, self.CT, self.CY, self.CZ, self.CQ, self.CMy, self.CMz, self.CMb = [
            outputs[k] for k in ("CP", "CT", "CY", "CZ", "CQ", "CMy", "CMz", "CMb")
        ]
        self.dCP, self.dCT, self.dCY, self.dCZ, self.dCQ, self.dCMy, self.dCMz, self.dCMb = [
            derivs[k] for k in ("dCP", "dCT", "dCY", "dCZ", "dCQ", "dCMy", "dCMz", "dCMb")
        ]

        self.rotor.derivatives = False
        self.n = len(self.r)
        self.npts = len(self.Uinf)

    def test_dUinf2(self):

        dT_dUinf = self.dT["dUinf"]
        dY_dUinf = self.dY["dUinf"]
        dZ_dUinf = self.dZ["dUinf"]
        dQ_dUinf = self.dQ["dUinf"]
        dMy_dUinf = self.dMy["dUinf"]
        dMz_dUinf = self.dMz["dUinf"]
        dMb_dUinf = self.dMb["dUinf"]
        dP_dUinf = self.dP["dUinf"]

        dT_dUinf_fd = np.zeros((self.npts, self.npts))
        dY_dUinf_fd = np.zeros((self.npts, self.npts))
        dZ_dUinf_fd = np.zeros((self.npts, self.npts))
        dQ_dUinf_fd = np.zeros((self.npts, self.npts))
        dMy_dUinf_fd = np.zeros((self.npts, self.npts))
        dMz_dUinf_fd = np.zeros((self.npts, self.npts))
        dMb_dUinf_fd = np.zeros((self.npts, self.npts))
        dP_dUinf_fd = np.zeros((self.npts, self.npts))

        for i in range(self.npts):
            Uinf = np.copy(self.Uinf)
            delta = 1e-6 * Uinf[i]
            Uinf[i] += delta

            outputs, _ = self.rotor.evaluate(Uinf, self.Omega, self.pitch, coefficients=False)
            Pd = outputs["P"]
            Td = outputs["T"]
            Yd = outputs["Y"]
            Zd = outputs["Z"]
            Qd = outputs["Q"]
            Myd = outputs["My"]
            Mzd = outputs["Mz"]
            Mbd = outputs["Mb"]

            dT_dUinf_fd[:, i] = (Td - self.T) / delta
            dY_dUinf_fd[:, i] = (Yd - self.Y) / delta
            dZ_dUinf_fd[:, i] = (Zd - self.Z) / delta
            dQ_dUinf_fd[:, i] = (Qd - self.Q) / delta
            dMy_dUinf_fd[:, i] = (Myd - self.My) / delta
            dMz_dUinf_fd[:, i] = (Mzd - self.Mz) / delta
            dMb_dUinf_fd[:, i] = (Mbd - self.Mb) / delta
            dP_dUinf_fd[:, i] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dUinf_fd, dT_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dUinf_fd, dY_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dUinf_fd, dZ_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dUinf_fd, dQ_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dUinf_fd, dMy_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dUinf_fd, dMz_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dUinf_fd, dMb_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dUinf_fd, dP_dUinf, rtol=5e-5, atol=1e-8)

    def test_dUinf3(self):

        dCT_dUinf = self.dCT["dUinf"]
        dCY_dUinf = self.dCY["dUinf"]
        dCZ_dUinf = self.dCZ["dUinf"]
        dCQ_dUinf = self.dCQ["dUinf"]
        dCMy_dUinf = self.dCMy["dUinf"]
        dCMz_dUinf = self.dCMz["dUinf"]
        dCMb_dUinf = self.dCMb["dUinf"]
        dCP_dUinf = self.dCP["dUinf"]

        dCT_dUinf_fd = np.zeros((self.npts, self.npts))
        dCY_dUinf_fd = np.zeros((self.npts, self.npts))
        dCZ_dUinf_fd = np.zeros((self.npts, self.npts))
        dCQ_dUinf_fd = np.zeros((self.npts, self.npts))
        dCMy_dUinf_fd = np.zeros((self.npts, self.npts))
        dCMz_dUinf_fd = np.zeros((self.npts, self.npts))
        dCMb_dUinf_fd = np.zeros((self.npts, self.npts))
        dCP_dUinf_fd = np.zeros((self.npts, self.npts))

        for i in range(self.npts):
            Uinf = np.copy(self.Uinf)
            delta = 1e-6 * Uinf[i]
            Uinf[i] += delta

            outputs, _ = self.rotor.evaluate(Uinf, self.Omega, self.pitch, coefficients=True)
            CPd = outputs["CP"]
            CTd = outputs["CT"]
            CYd = outputs["CY"]
            CZd = outputs["CZ"]
            CQd = outputs["CQ"]
            CMyd = outputs["CMy"]
            CMzd = outputs["CMz"]
            CMbd = outputs["CMb"]

            dCT_dUinf_fd[:, i] = (CTd - self.CT) / delta
            dCY_dUinf_fd[:, i] = (CYd - self.CY) / delta
            dCZ_dUinf_fd[:, i] = (CZd - self.CZ) / delta
            dCQ_dUinf_fd[:, i] = (CQd - self.CQ) / delta
            dCMy_dUinf_fd[:, i] = (CMyd - self.CMy) / delta
            dCMz_dUinf_fd[:, i] = (CMzd - self.CMz) / delta
            dCMb_dUinf_fd[:, i] = (CMbd - self.CMb) / delta
            dCP_dUinf_fd[:, i] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dUinf_fd, dCT_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dUinf_fd, dCY_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dUinf_fd, dCZ_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dUinf_fd, dCQ_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dUinf_fd, dCMy_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dUinf_fd, dCMz_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dUinf_fd, dCMb_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dUinf_fd, dCP_dUinf, rtol=5e-5, atol=1e-8)

    def test_dOmega2(self):

        dT_dOmega = self.dT["dOmega"]
        dY_dOmega = self.dY["dOmega"]
        dZ_dOmega = self.dZ["dOmega"]
        dQ_dOmega = self.dQ["dOmega"]
        dMy_dOmega = self.dMy["dOmega"]
        dMz_dOmega = self.dMz["dOmega"]
        dMb_dOmega = self.dMb["dOmega"]
        dP_dOmega = self.dP["dOmega"]

        dT_dOmega_fd = np.zeros((self.npts, self.npts))
        dY_dOmega_fd = np.zeros((self.npts, self.npts))
        dZ_dOmega_fd = np.zeros((self.npts, self.npts))
        dQ_dOmega_fd = np.zeros((self.npts, self.npts))
        dMy_dOmega_fd = np.zeros((self.npts, self.npts))
        dMz_dOmega_fd = np.zeros((self.npts, self.npts))
        dMb_dOmega_fd = np.zeros((self.npts, self.npts))
        dP_dOmega_fd = np.zeros((self.npts, self.npts))

        for i in range(self.npts):
            Omega = np.copy(self.Omega)
            delta = 1e-6 * Omega[i]
            Omega[i] += delta

            outputs, _ = self.rotor.evaluate(self.Uinf, Omega, self.pitch, coefficients=False)
            Pd = outputs["P"]
            Td = outputs["T"]
            Yd = outputs["Y"]
            Zd = outputs["Z"]
            Qd = outputs["Q"]
            Myd = outputs["My"]
            Mzd = outputs["Mz"]
            Mbd = outputs["Mb"]

            dT_dOmega_fd[:, i] = (Td - self.T) / delta
            dY_dOmega_fd[:, i] = (Yd - self.Y) / delta
            dZ_dOmega_fd[:, i] = (Zd - self.Z) / delta
            dQ_dOmega_fd[:, i] = (Qd - self.Q) / delta
            dMy_dOmega_fd[:, i] = (Myd - self.My) / delta
            dMz_dOmega_fd[:, i] = (Mzd - self.Mz) / delta
            dMb_dOmega_fd[:, i] = (Mbd - self.Mb) / delta
            dP_dOmega_fd[:, i] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dOmega_fd, dT_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dOmega_fd, dY_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dOmega_fd, dZ_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dOmega_fd, dQ_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dOmega_fd, dMy_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dOmega_fd, dMz_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dOmega_fd, dMb_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dOmega_fd, dP_dOmega, rtol=5e-5, atol=1e-8)

    def test_dOmega3(self):

        dCT_dOmega = self.dCT["dOmega"]
        dCY_dOmega = self.dCY["dOmega"]
        dCZ_dOmega = self.dCZ["dOmega"]
        dCQ_dOmega = self.dCQ["dOmega"]
        dCMy_dOmega = self.dCMy["dOmega"]
        dCMz_dOmega = self.dCMz["dOmega"]
        dCMb_dOmega = self.dCMb["dOmega"]
        dCP_dOmega = self.dCP["dOmega"]

        dCT_dOmega_fd = np.zeros((self.npts, self.npts))
        dCY_dOmega_fd = np.zeros((self.npts, self.npts))
        dCZ_dOmega_fd = np.zeros((self.npts, self.npts))
        dCQ_dOmega_fd = np.zeros((self.npts, self.npts))
        dCMy_dOmega_fd = np.zeros((self.npts, self.npts))
        dCMz_dOmega_fd = np.zeros((self.npts, self.npts))
        dCMb_dOmega_fd = np.zeros((self.npts, self.npts))
        dCP_dOmega_fd = np.zeros((self.npts, self.npts))

        for i in range(self.npts):
            Omega = np.copy(self.Omega)
            delta = 1e-6 * Omega[i]
            Omega[i] += delta

            outputs, _ = self.rotor.evaluate(self.Uinf, Omega, self.pitch, coefficients=True)
            CPd = outputs["CP"]
            CTd = outputs["CT"]
            CYd = outputs["CY"]
            CZd = outputs["CZ"]
            CQd = outputs["CQ"]
            CMyd = outputs["CMy"]
            CMzd = outputs["CMz"]
            CMbd = outputs["CMb"]

            dCT_dOmega_fd[:, i] = (CTd - self.CT) / delta
            dCY_dOmega_fd[:, i] = (CYd - self.CY) / delta
            dCZ_dOmega_fd[:, i] = (CZd - self.CZ) / delta
            dCQ_dOmega_fd[:, i] = (CQd - self.CQ) / delta
            dCMy_dOmega_fd[:, i] = (CMyd - self.CMy) / delta
            dCMz_dOmega_fd[:, i] = (CMzd - self.CMz) / delta
            dCMb_dOmega_fd[:, i] = (CMbd - self.CMb) / delta
            dCP_dOmega_fd[:, i] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dOmega_fd, dCT_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dOmega_fd, dCY_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dOmega_fd, dCZ_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dOmega_fd, dCQ_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dOmega_fd, dCMy_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dOmega_fd, dCMz_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dOmega_fd, dCMb_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dOmega_fd, dCP_dOmega, rtol=5e-5, atol=1e-8)

    def test_dpitch2(self):

        dT_dpitch = self.dT["dpitch"]
        dY_dpitch = self.dY["dpitch"]
        dZ_dpitch = self.dZ["dpitch"]
        dQ_dpitch = self.dQ["dpitch"]
        dMy_dpitch = self.dMy["dpitch"]
        dMz_dpitch = self.dMz["dpitch"]
        dMb_dpitch = self.dMb["dpitch"]
        dP_dpitch = self.dP["dpitch"]

        dT_dpitch_fd = np.zeros((self.npts, self.npts))
        dY_dpitch_fd = np.zeros((self.npts, self.npts))
        dZ_dpitch_fd = np.zeros((self.npts, self.npts))
        dQ_dpitch_fd = np.zeros((self.npts, self.npts))
        dMy_dpitch_fd = np.zeros((self.npts, self.npts))
        dMz_dpitch_fd = np.zeros((self.npts, self.npts))
        dMb_dpitch_fd = np.zeros((self.npts, self.npts))
        dP_dpitch_fd = np.zeros((self.npts, self.npts))

        for i in range(self.npts):
            pitch = np.copy(self.pitch)
            delta = 1e-6
            pitch[i] += delta

            outputs, _ = self.rotor.evaluate(self.Uinf, self.Omega, pitch, coefficients=False)
            Pd = outputs["P"]
            Td = outputs["T"]
            Yd = outputs["Y"]
            Zd = outputs["Z"]
            Qd = outputs["Q"]
            Myd = outputs["My"]
            Mzd = outputs["Mz"]
            Mbd = outputs["Mb"]

            dT_dpitch_fd[:, i] = (Td - self.T) / delta
            dY_dpitch_fd[:, i] = (Yd - self.Y) / delta
            dZ_dpitch_fd[:, i] = (Zd - self.Z) / delta
            dQ_dpitch_fd[:, i] = (Qd - self.Q) / delta
            dMy_dpitch_fd[:, i] = (Myd - self.My) / delta
            dMz_dpitch_fd[:, i] = (Mzd - self.Mz) / delta
            dMb_dpitch_fd[:, i] = (Mbd - self.Mb) / delta
            dP_dpitch_fd[:, i] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dpitch_fd, dT_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dpitch_fd, dY_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dpitch_fd, dZ_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dpitch_fd, dQ_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dpitch_fd, dMy_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dpitch_fd, dMz_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dpitch_fd, dMb_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dpitch_fd, dP_dpitch, rtol=5e-5, atol=1e-8)

    def test_dpitch3(self):

        dCT_dpitch = self.dCT["dpitch"]
        dCY_dpitch = self.dCY["dpitch"]
        dCZ_dpitch = self.dCZ["dpitch"]
        dCQ_dpitch = self.dCQ["dpitch"]
        dCMy_dpitch = self.dCMy["dpitch"]
        dCMz_dpitch = self.dCMz["dpitch"]
        dCMb_dpitch = self.dCMb["dpitch"]
        dCP_dpitch = self.dCP["dpitch"]

        dCT_dpitch_fd = np.zeros((self.npts, self.npts))
        dCY_dpitch_fd = np.zeros((self.npts, self.npts))
        dCZ_dpitch_fd = np.zeros((self.npts, self.npts))
        dCQ_dpitch_fd = np.zeros((self.npts, self.npts))
        dCMy_dpitch_fd = np.zeros((self.npts, self.npts))
        dCMz_dpitch_fd = np.zeros((self.npts, self.npts))
        dCMb_dpitch_fd = np.zeros((self.npts, self.npts))
        dCP_dpitch_fd = np.zeros((self.npts, self.npts))

        for i in range(self.npts):
            pitch = np.copy(self.pitch)
            delta = 1e-6
            pitch[i] += delta

            outputs, _ = self.rotor.evaluate(self.Uinf, self.Omega, pitch, coefficients=True)
            CPd = outputs["CP"]
            CTd = outputs["CT"]
            CYd = outputs["CY"]
            CZd = outputs["CZ"]
            CQd = outputs["CQ"]
            CMyd = outputs["CMy"]
            CMzd = outputs["CMz"]
            CMbd = outputs["CMb"]

            dCT_dpitch_fd[:, i] = (CTd - self.CT) / delta
            dCY_dpitch_fd[:, i] = (CYd - self.CY) / delta
            dCZ_dpitch_fd[:, i] = (CZd - self.CZ) / delta
            dCQ_dpitch_fd[:, i] = (CQd - self.CQ) / delta
            dCMy_dpitch_fd[:, i] = (CMyd - self.CMy) / delta
            dCMz_dpitch_fd[:, i] = (CMzd - self.CMz) / delta
            dCMb_dpitch_fd[:, i] = (CMbd - self.CMb) / delta
            dCP_dpitch_fd[:, i] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dpitch_fd, dCT_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dpitch_fd, dCY_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dpitch_fd, dCZ_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dpitch_fd, dCQ_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dpitch_fd, dCMy_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dpitch_fd, dCMz_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dpitch_fd, dCMb_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dpitch_fd, dCP_dpitch, rtol=5e-5, atol=1e-8)


class TestGradients_RHub_Tip(unittest.TestCase):
    def setUp(self):

        # geometry
        self.Rhub = 1.5
        self.Rtip = 63.0

        self.r = np.array(
            [
                self.Rhub,
                5.6000,
                8.3333,
                11.7500,
                15.8500,
                19.9500,
                24.0500,
                28.1500,
                32.2500,
                36.3500,
                40.4500,
                44.5500,
                48.6500,
                52.7500,
                56.1667,
                58.9000,
                self.Rtip,
            ]
        )
        self.chord = np.array(
            [
                3.542,
                3.854,
                4.167,
                4.557,
                4.652,
                4.458,
                4.249,
                4.007,
                3.748,
                3.502,
                3.256,
                3.010,
                2.764,
                2.518,
                2.313,
                2.086,
                1.419,
            ]
        )
        self.theta = np.array(
            [
                13.308,
                13.308,
                13.308,
                13.308,
                11.480,
                10.162,
                9.011,
                7.795,
                6.544,
                5.361,
                4.188,
                3.125,
                2.319,
                1.526,
                0.863,
                0.370,
                0.106,
            ]
        )
        self.B = 3  # number of blades

        # atmosphere
        self.rho = 1.225
        self.mu = 1.81206e-5

        afinit = CCAirfoil.initFromAerodynFile  # just for shorthand

        # load all airfoils
        airfoil_types = [0] * 8
        airfoil_types[0] = afinit(basepath + os.sep + "Cylinder1.dat")
        airfoil_types[1] = afinit(basepath + os.sep + "Cylinder2.dat")
        airfoil_types[2] = afinit(basepath + os.sep + "DU40_A17.dat")
        airfoil_types[3] = afinit(basepath + os.sep + "DU35_A17.dat")
        airfoil_types[4] = afinit(basepath + os.sep + "DU30_A17.dat")
        airfoil_types[5] = afinit(basepath + os.sep + "DU25_A17.dat")
        airfoil_types[6] = afinit(basepath + os.sep + "DU21_A17.dat")
        airfoil_types[7] = afinit(basepath + os.sep + "NACA64_A17.dat")

        # place at appropriate radial stations
        af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

        self.af = [0] * len(self.r)
        for i in range(len(self.r)):
            self.af[i] = airfoil_types[af_idx[i]]

        self.tilt = -5.0
        self.precone = 2.5
        self.yaw = 0.0
        self.shearExp = 0.2
        self.hubHt = 80.0
        self.nSector = 8

        # create CCBlade object
        self.rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
        )

        # Update for FDs
        self.r = self.rotor.r.copy()

        # set conditions
        self.Uinf = 10.0
        tsr = 7.55
        self.pitch = 0.0
        self.Omega = self.Uinf * tsr / self.Rtip * 30.0 / np.pi  # convert to RPM
        self.azimuth = 90

        loads, derivs = self.rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        self.Np = loads["Np"]
        self.Tp = loads["Tp"]
        self.dNp = derivs["dNp"]
        self.dTp = derivs["dTp"]

        outputs, derivs = self.rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        self.P, self.T, self.Y, self.Z, self.Q, self.My, self.Mz, self.Mb = [
            outputs[k] for k in ("P", "T", "Y", "Z", "Q", "My", "Mz", "Mb")
        ]
        self.dP, self.dT, self.dY, self.dZ, self.dQ, self.dMy, self.dMz, self.dMb = [
            derivs[k] for k in ("dP", "dT", "dY", "dZ", "dQ", "dMy", "dMz", "dMb")
        ]
        self.CP, self.CT, self.CY, self.CZ, self.CQ, self.CMy, self.CMz, self.CMb = [
            outputs[k] for k in ("CP", "CT", "CY", "CZ", "CQ", "CMy", "CMz", "CMb")
        ]
        self.dCP, self.dCT, self.dCY, self.dCZ, self.dCQ, self.dCMy, self.dCMz, self.dCMb = [
            derivs[k] for k in ("dCP", "dCT", "dCY", "dCZ", "dCQ", "dCMy", "dCMz", "dCMb")
        ]

        self.rotor.derivatives = False
        self.n = len(self.r)
        self.npts = 1  # len(Uinf)

    def test_dr1(self):

        dNp_dr = self.dNp["dr"]
        dTp_dr = self.dTp["dr"]
        dNp_dr_fd = np.zeros((self.n, self.n))
        dTp_dr_fd = np.zeros((self.n, self.n))

        for i in range(self.n):
            r = np.array(self.r)
            delta = 1e-6 * r[i]
            r[i] += delta

            rotor = CCBlade(
                r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dr_fd[:, i] = (Npd - self.Np) / delta
            dTp_dr_fd[:, i] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dr_fd, dNp_dr, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dr_fd, dTp_dr, rtol=1e-4, atol=1e-8)

    def test_dr2(self):

        dT_dr = self.dT["dr"]
        dY_dr = self.dY["dr"]
        dZ_dr = self.dZ["dr"]
        dQ_dr = self.dQ["dr"]
        dMy_dr = self.dMy["dr"]
        dMz_dr = self.dMz["dr"]
        dMb_dr = self.dMb["dr"]
        dP_dr = self.dP["dr"]
        dT_dr_fd = np.zeros((self.npts, self.n))
        dY_dr_fd = np.zeros((self.npts, self.n))
        dZ_dr_fd = np.zeros((self.npts, self.n))
        dQ_dr_fd = np.zeros((self.npts, self.n))
        dMy_dr_fd = np.zeros((self.npts, self.n))
        dMz_dr_fd = np.zeros((self.npts, self.n))
        dMb_dr_fd = np.zeros((self.npts, self.n))
        dP_dr_fd = np.zeros((self.npts, self.n))

        for i in range(self.n):
            r = np.array(self.r)
            delta = 1e-6 * r[i]
            r[i] += delta

            rotor = CCBlade(
                r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
            Pd = outputs["P"]
            Td = outputs["T"]
            Yd = outputs["Y"]
            Zd = outputs["Z"]
            Qd = outputs["Q"]
            Myd = outputs["My"]
            Mzd = outputs["Mz"]
            Mbd = outputs["Mb"]

            dT_dr_fd[:, i] = (Td - self.T) / delta
            dY_dr_fd[:, i] = (Yd - self.Y) / delta
            dZ_dr_fd[:, i] = (Zd - self.Z) / delta
            dQ_dr_fd[:, i] = (Qd - self.Q) / delta
            dMy_dr_fd[:, i] = (Myd - self.My) / delta
            dMz_dr_fd[:, i] = (Mzd - self.Mz) / delta
            dMb_dr_fd[:, i] = (Mbd - self.Mb) / delta
            dP_dr_fd[:, i] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dr_fd, dT_dr, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dr_fd, dY_dr, rtol=5e-4, atol=1e-8)
        np.testing.assert_allclose(dZ_dr_fd, dZ_dr, rtol=5e-4, atol=1e-8)
        np.testing.assert_allclose(dQ_dr_fd, dQ_dr, rtol=1e-3)  # , atol=1e-8)
        np.testing.assert_allclose(dMy_dr_fd, dMy_dr, rtol=1e-3)  # , atol=1e-8)
        np.testing.assert_allclose(dMz_dr_fd, dMz_dr, rtol=1e-3)  # , atol=1e-8)
        np.testing.assert_allclose(dMb_dr_fd, dMb_dr, rtol=1e-3)  # , atol=1e-8)
        np.testing.assert_allclose(dP_dr_fd, dP_dr, rtol=1e-3)  # , atol=1e-8)

    def test_dr3(self):

        dCT_dr = self.dCT["dr"]
        dCY_dr = self.dCY["dr"]
        dCZ_dr = self.dCZ["dr"]
        dCQ_dr = self.dCQ["dr"]
        dCMy_dr = self.dCMy["dr"]
        dCMz_dr = self.dCMz["dr"]
        dCMb_dr = self.dCMb["dr"]
        dCP_dr = self.dCP["dr"]
        dCT_dr_fd = np.zeros((self.npts, self.n))
        dCY_dr_fd = np.zeros((self.npts, self.n))
        dCZ_dr_fd = np.zeros((self.npts, self.n))
        dCQ_dr_fd = np.zeros((self.npts, self.n))
        dCMy_dr_fd = np.zeros((self.npts, self.n))
        dCMz_dr_fd = np.zeros((self.npts, self.n))
        dCMb_dr_fd = np.zeros((self.npts, self.n))
        dCP_dr_fd = np.zeros((self.npts, self.n))

        for i in range(self.n):
            r = np.array(self.r)
            delta = 1e-6 * r[i]
            r[i] += delta

            rotor = CCBlade(
                r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
            CPd = outputs["CP"]
            CTd = outputs["CT"]
            CYd = outputs["CY"]
            CZd = outputs["CZ"]
            CQd = outputs["CQ"]
            CMyd = outputs["CMy"]
            CMzd = outputs["CMz"]
            CMbd = outputs["CMb"]

            dCT_dr_fd[:, i] = (CTd - self.CT) / delta
            dCY_dr_fd[:, i] = (CYd - self.CY) / delta
            dCZ_dr_fd[:, i] = (CZd - self.CZ) / delta
            dCQ_dr_fd[:, i] = (CQd - self.CQ) / delta
            dCMy_dr_fd[:, i] = (CMyd - self.CMy) / delta
            dCMz_dr_fd[:, i] = (CMzd - self.CMz) / delta
            dCMb_dr_fd[:, i] = (CMbd - self.CMb) / delta
            dCP_dr_fd[:, i] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dr_fd, dCT_dr, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dr_fd, dCY_dr, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dr_fd, dCZ_dr, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dr_fd, dCQ_dr, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMy_dr_fd, dCMy_dr, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMz_dr_fd, dCMz_dr, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMb_dr_fd, dCMb_dr, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dr_fd, dCP_dr, rtol=3e-4, atol=1e-7)

    def test_dchord1(self):

        dNp_dchord = self.dNp["dchord"]
        dTp_dchord = self.dTp["dchord"]
        dNp_dchord_fd = np.zeros((self.n, self.n))
        dTp_dchord_fd = np.zeros((self.n, self.n))

        for i in range(self.n):
            chord = np.array(self.chord)
            delta = 1e-6 * chord[i]
            chord[i] += delta

            rotor = CCBlade(
                self.r,
                chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dchord_fd[:, i] = (Npd - self.Np) / delta
            dTp_dchord_fd[:, i] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dchord_fd, dNp_dchord, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dTp_dchord_fd, dTp_dchord, rtol=5e-5, atol=1e-8)

    def test_dchord2(self):

        dT_dchord = self.dT["dchord"]
        dY_dchord = self.dY["dchord"]
        dZ_dchord = self.dZ["dchord"]
        dQ_dchord = self.dQ["dchord"]
        dMy_dchord = self.dMy["dchord"]
        dMz_dchord = self.dMz["dchord"]
        dMb_dchord = self.dMb["dchord"]
        dP_dchord = self.dP["dchord"]
        dT_dchord_fd = np.zeros((self.npts, self.n))
        dY_dchord_fd = np.zeros((self.npts, self.n))
        dZ_dchord_fd = np.zeros((self.npts, self.n))
        dQ_dchord_fd = np.zeros((self.npts, self.n))
        dMy_dchord_fd = np.zeros((self.npts, self.n))
        dMz_dchord_fd = np.zeros((self.npts, self.n))
        dMb_dchord_fd = np.zeros((self.npts, self.n))
        dP_dchord_fd = np.zeros((self.npts, self.n))

        for i in range(self.n):
            chord = np.array(self.chord)
            delta = 1e-6 * chord[i]
            chord[i] += delta

            rotor = CCBlade(
                self.r,
                chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
            Pd = outputs["P"]
            Td = outputs["T"]
            Yd = outputs["Y"]
            Zd = outputs["Z"]
            Qd = outputs["Q"]
            Myd = outputs["My"]
            Mzd = outputs["Mz"]
            Mbd = outputs["Mb"]

            dT_dchord_fd[:, i] = (Td - self.T) / delta
            dY_dchord_fd[:, i] = (Yd - self.Y) / delta
            dZ_dchord_fd[:, i] = (Zd - self.Z) / delta
            dQ_dchord_fd[:, i] = (Qd - self.Q) / delta
            dMy_dchord_fd[:, i] = (Myd - self.My) / delta
            dMz_dchord_fd[:, i] = (Mzd - self.Mz) / delta
            dMb_dchord_fd[:, i] = (Mbd - self.Mb) / delta
            dP_dchord_fd[:, i] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dchord_fd, dT_dchord, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dY_dchord_fd, dY_dchord, rtol=5e-3, atol=1e-8)
        np.testing.assert_allclose(dZ_dchord_fd, dZ_dchord, rtol=5e-3, atol=1e-8)
        np.testing.assert_allclose(dQ_dchord_fd, dQ_dchord, rtol=5e-3, atol=1e-8)
        np.testing.assert_allclose(dMy_dchord_fd, dMy_dchord, rtol=5e-3, atol=1e-8)
        np.testing.assert_allclose(dMz_dchord_fd, dMz_dchord, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dMb_dchord_fd, dMb_dchord, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dchord_fd, dP_dchord, rtol=7e-5, atol=1e-8)

    def test_dchord3(self):

        dCT_dchord = self.dCT["dchord"]
        dCY_dchord = self.dCY["dchord"]
        dCZ_dchord = self.dCZ["dchord"]
        dCQ_dchord = self.dCQ["dchord"]
        dCMy_dchord = self.dCMy["dchord"]
        dCMz_dchord = self.dCMz["dchord"]
        dCMb_dchord = self.dCMb["dchord"]
        dCP_dchord = self.dCP["dchord"]
        dCT_dchord_fd = np.zeros((self.npts, self.n))
        dCY_dchord_fd = np.zeros((self.npts, self.n))
        dCZ_dchord_fd = np.zeros((self.npts, self.n))
        dCQ_dchord_fd = np.zeros((self.npts, self.n))
        dCMy_dchord_fd = np.zeros((self.npts, self.n))
        dCMz_dchord_fd = np.zeros((self.npts, self.n))
        dCMb_dchord_fd = np.zeros((self.npts, self.n))
        dCP_dchord_fd = np.zeros((self.npts, self.n))

        for i in range(self.n):
            chord = np.array(self.chord)
            delta = 1e-6 * chord[i]
            chord[i] += delta

            rotor = CCBlade(
                self.r,
                chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
            CPd = outputs["CP"]
            CTd = outputs["CT"]
            CYd = outputs["CY"]
            CZd = outputs["CZ"]
            CQd = outputs["CQ"]
            CMyd = outputs["CMy"]
            CMzd = outputs["CMz"]
            CMbd = outputs["CMb"]

            dCT_dchord_fd[:, i] = (CTd - self.CT) / delta
            dCY_dchord_fd[:, i] = (CYd - self.CY) / delta
            dCZ_dchord_fd[:, i] = (CZd - self.CZ) / delta
            dCQ_dchord_fd[:, i] = (CQd - self.CQ) / delta
            dCMy_dchord_fd[:, i] = (CMyd - self.CMy) / delta
            dCMz_dchord_fd[:, i] = (CMzd - self.CMz) / delta
            dCMb_dchord_fd[:, i] = (CMbd - self.CMb) / delta
            dCP_dchord_fd[:, i] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dchord_fd, dCT_dchord, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dCY_dchord_fd, dCY_dchord, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dCZ_dchord_fd, dCZ_dchord, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dCQ_dchord_fd, dCQ_dchord, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dchord_fd, dCMy_dchord, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dchord_fd, dCMz_dchord, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dchord_fd, dCMb_dchord, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dchord_fd, dCP_dchord, rtol=7e-5, atol=1e-8)

    def test_dtheta1(self):

        dNp_dtheta = self.dNp["dtheta"]
        dTp_dtheta = self.dTp["dtheta"]
        dNp_dtheta_fd = np.zeros((self.n, self.n))
        dTp_dtheta_fd = np.zeros((self.n, self.n))

        for i in range(self.n):
            theta = np.array(self.theta)
            delta = 1e-6 * theta[i]
            theta[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dtheta_fd[:, i] = (Npd - self.Np) / delta
            dTp_dtheta_fd[:, i] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dtheta_fd, dNp_dtheta, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dTp_dtheta_fd, dTp_dtheta, rtol=1e-4, atol=1e-8)

    def test_dtheta2(self):

        dT_dtheta = self.dT["dtheta"]
        dY_dtheta = self.dY["dtheta"]
        dZ_dtheta = self.dZ["dtheta"]
        dQ_dtheta = self.dQ["dtheta"]
        dMy_dtheta = self.dMy["dtheta"]
        dMz_dtheta = self.dMz["dtheta"]
        dMb_dtheta = self.dMb["dtheta"]
        dP_dtheta = self.dP["dtheta"]
        dT_dtheta_fd = np.zeros((self.npts, self.n))
        dY_dtheta_fd = np.zeros((self.npts, self.n))
        dZ_dtheta_fd = np.zeros((self.npts, self.n))
        dQ_dtheta_fd = np.zeros((self.npts, self.n))
        dMy_dtheta_fd = np.zeros((self.npts, self.n))
        dMz_dtheta_fd = np.zeros((self.npts, self.n))
        dMb_dtheta_fd = np.zeros((self.npts, self.n))
        dP_dtheta_fd = np.zeros((self.npts, self.n))

        for i in range(self.n):
            theta = np.array(self.theta)
            delta = 1e-6 * theta[i]
            theta[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
            Pd = outputs["P"]
            Td = outputs["T"]
            Yd = outputs["Y"]
            Zd = outputs["Z"]
            Qd = outputs["Q"]
            Myd = outputs["My"]
            Mzd = outputs["Mz"]
            Mbd = outputs["Mb"]

            dT_dtheta_fd[:, i] = (Td - self.T) / delta
            dY_dtheta_fd[:, i] = (Yd - self.Y) / delta
            dZ_dtheta_fd[:, i] = (Zd - self.Z) / delta
            dQ_dtheta_fd[:, i] = (Qd - self.Q) / delta
            dMy_dtheta_fd[:, i] = (Myd - self.My) / delta
            dMz_dtheta_fd[:, i] = (Mzd - self.Mz) / delta
            dMb_dtheta_fd[:, i] = (Mbd - self.Mb) / delta
            dP_dtheta_fd[:, i] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dtheta_fd, dT_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dtheta_fd, dY_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dtheta_fd, dZ_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dtheta_fd, dQ_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dtheta_fd, dMy_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dtheta_fd, dMz_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dtheta_fd, dMb_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dtheta_fd, dP_dtheta, rtol=7e-5, atol=1e-8)

    def test_dtheta3(self):

        dCT_dtheta = self.dCT["dtheta"]
        dCY_dtheta = self.dCY["dtheta"]
        dCZ_dtheta = self.dCZ["dtheta"]
        dCQ_dtheta = self.dCQ["dtheta"]
        dCMy_dtheta = self.dCMy["dtheta"]
        dCMz_dtheta = self.dCMz["dtheta"]
        dCMb_dtheta = self.dCMb["dtheta"]
        dCP_dtheta = self.dCP["dtheta"]
        dCT_dtheta_fd = np.zeros((self.npts, self.n))
        dCY_dtheta_fd = np.zeros((self.npts, self.n))
        dCZ_dtheta_fd = np.zeros((self.npts, self.n))
        dCQ_dtheta_fd = np.zeros((self.npts, self.n))
        dCMy_dtheta_fd = np.zeros((self.npts, self.n))
        dCMz_dtheta_fd = np.zeros((self.npts, self.n))
        dCMb_dtheta_fd = np.zeros((self.npts, self.n))
        dCP_dtheta_fd = np.zeros((self.npts, self.n))

        for i in range(self.n):
            theta = np.array(self.theta)
            delta = 1e-6 * theta[i]
            theta[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                self.precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
            CPd = outputs["CP"]
            CTd = outputs["CT"]
            CYd = outputs["CY"]
            CZd = outputs["CZ"]
            CQd = outputs["CQ"]
            CMyd = outputs["CMy"]
            CMzd = outputs["CMz"]
            CMbd = outputs["CMb"]

            dCT_dtheta_fd[:, i] = (CTd - self.CT) / delta
            dCY_dtheta_fd[:, i] = (CYd - self.CY) / delta
            dCZ_dtheta_fd[:, i] = (CZd - self.CZ) / delta
            dCQ_dtheta_fd[:, i] = (CQd - self.CQ) / delta
            dCMy_dtheta_fd[:, i] = (CMyd - self.CMy) / delta
            dCMz_dtheta_fd[:, i] = (CMzd - self.CMz) / delta
            dCMb_dtheta_fd[:, i] = (CMbd - self.CMb) / delta
            dCP_dtheta_fd[:, i] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dtheta_fd, dCT_dtheta, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dCY_dtheta_fd, dCY_dtheta, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dCZ_dtheta_fd, dCZ_dtheta, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dCQ_dtheta_fd, dCQ_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dtheta_fd, dCMy_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dtheta_fd, dCMz_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dtheta_fd, dCMb_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dtheta_fd, dCP_dtheta, rtol=7e-5, atol=1e-8)

    def test_dRhub1(self):

        dNp_dRhub = self.dNp["dRhub"]
        dTp_dRhub = self.dTp["dRhub"]

        dNp_dRhub_fd = np.zeros((self.n, 1))
        dTp_dRhub_fd = np.zeros((self.n, 1))

        Rhub = float(self.Rhub)
        delta = 1e-6 * Rhub
        Rhub += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dRhub_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dRhub_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dRhub_fd, dNp_dRhub, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dRhub_fd, dTp_dRhub, rtol=1e-4, atol=1e-6)

    def test_dRhub2(self):

        dT_dRhub = self.dT["dRhub"]
        dY_dRhub = self.dY["dRhub"]
        dZ_dRhub = self.dZ["dRhub"]
        dQ_dRhub = self.dQ["dRhub"]
        dMy_dRhub = self.dMy["dRhub"]
        dMz_dRhub = self.dMz["dRhub"]
        dMb_dRhub = self.dMb["dRhub"]
        dP_dRhub = self.dP["dRhub"]

        dT_dRhub_fd = np.zeros((self.npts, 1))
        dY_dRhub_fd = np.zeros((self.npts, 1))
        dZ_dRhub_fd = np.zeros((self.npts, 1))
        dQ_dRhub_fd = np.zeros((self.npts, 1))
        dMy_dRhub_fd = np.zeros((self.npts, 1))
        dMz_dRhub_fd = np.zeros((self.npts, 1))
        dMb_dRhub_fd = np.zeros((self.npts, 1))
        dP_dRhub_fd = np.zeros((self.npts, 1))

        Rhub = float(self.Rhub)
        delta = 1e-6 * Rhub
        Rhub += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dRhub_fd[:, 0] = (Td - self.T) / delta
        dY_dRhub_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dRhub_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dRhub_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dRhub_fd[:, 0] = (Myd - self.My) / delta
        dMz_dRhub_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dRhub_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dRhub_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dRhub_fd, dT_dRhub, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dRhub_fd, dY_dRhub, rtol=1e-3, atol=1e-8)
        np.testing.assert_allclose(dZ_dRhub_fd, dZ_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dRhub_fd, dQ_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dRhub_fd, dMy_dRhub, rtol=7e-4, atol=1e-8)
        np.testing.assert_allclose(dMz_dRhub_fd, dMz_dRhub, rtol=7e-4, atol=1e-8)
        np.testing.assert_allclose(dMb_dRhub_fd, dMb_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dRhub_fd, dP_dRhub, rtol=5e-5, atol=1e-8)

    def test_dRhub3(self):

        dCT_dRhub = self.dCT["dRhub"]
        dCY_dRhub = self.dCY["dRhub"]
        dCZ_dRhub = self.dCZ["dRhub"]
        dCQ_dRhub = self.dCQ["dRhub"]
        dCMy_dRhub = self.dCMy["dRhub"]
        dCMz_dRhub = self.dCMz["dRhub"]
        dCMb_dRhub = self.dCMb["dRhub"]
        dCP_dRhub = self.dCP["dRhub"]

        dCT_dRhub_fd = np.zeros((self.npts, 1))
        dCY_dRhub_fd = np.zeros((self.npts, 1))
        dCZ_dRhub_fd = np.zeros((self.npts, 1))
        dCQ_dRhub_fd = np.zeros((self.npts, 1))
        dCMy_dRhub_fd = np.zeros((self.npts, 1))
        dCMz_dRhub_fd = np.zeros((self.npts, 1))
        dCMb_dRhub_fd = np.zeros((self.npts, 1))
        dCP_dRhub_fd = np.zeros((self.npts, 1))

        Rhub = float(self.Rhub)
        delta = 1e-6 * Rhub
        Rhub += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dRhub_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dRhub_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dRhub_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dRhub_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dRhub_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dRhub_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dRhub_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dRhub_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dRhub_fd, dCT_dRhub, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dRhub_fd, dCY_dRhub, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dRhub_fd, dCZ_dRhub, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dRhub_fd, dCQ_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dRhub_fd, dCMy_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dRhub_fd, dCMz_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dRhub_fd, dCMb_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dRhub_fd, dCP_dRhub, rtol=5e-5, atol=1e-8)

    def test_dRtip1(self):

        dNp_dRtip = self.dNp["dRtip"]
        dTp_dRtip = self.dTp["dRtip"]

        dNp_dRtip_fd = np.zeros((self.n, 1))
        dTp_dRtip_fd = np.zeros((self.n, 1))

        Rtip = float(self.Rtip)
        delta = 1e-6 * Rtip
        Rtip += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dRtip_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dRtip_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dRtip_fd, dNp_dRtip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dRtip_fd, dTp_dRtip, rtol=1e-4, atol=1e-8)

    def test_dRtip2(self):

        dT_dRtip = self.dT["dRtip"]
        dY_dRtip = self.dY["dRtip"]
        dZ_dRtip = self.dZ["dRtip"]
        dQ_dRtip = self.dQ["dRtip"]
        dMy_dRtip = self.dMy["dRtip"]
        dMz_dRtip = self.dMz["dRtip"]
        dMb_dRtip = self.dMb["dRtip"]
        dP_dRtip = self.dP["dRtip"]

        dT_dRtip_fd = np.zeros((self.npts, 1))
        dY_dRtip_fd = np.zeros((self.npts, 1))
        dZ_dRtip_fd = np.zeros((self.npts, 1))
        dQ_dRtip_fd = np.zeros((self.npts, 1))
        dMy_dRtip_fd = np.zeros((self.npts, 1))
        dMz_dRtip_fd = np.zeros((self.npts, 1))
        dMb_dRtip_fd = np.zeros((self.npts, 1))
        dP_dRtip_fd = np.zeros((self.npts, 1))

        Rtip = float(self.Rtip)
        delta = 1e-6 * Rtip
        Rtip += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dRtip_fd[:, 0] = (Td - self.T) / delta
        dY_dRtip_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dRtip_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dRtip_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dRtip_fd[:, 0] = (Myd - self.My) / delta
        dMz_dRtip_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dRtip_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dRtip_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dRtip_fd, dT_dRtip, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dRtip_fd, dY_dRtip, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dRtip_fd, dZ_dRtip, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dRtip_fd, dQ_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dRtip_fd, dMy_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dRtip_fd, dMz_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dRtip_fd, dMb_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dRtip_fd, dP_dRtip, rtol=5e-5, atol=1e-8)

    def test_dRtip3(self):

        dCT_dRtip = self.dCT["dRtip"]
        dCY_dRtip = self.dCY["dRtip"]
        dCZ_dRtip = self.dCZ["dRtip"]
        dCQ_dRtip = self.dCQ["dRtip"]
        dCMy_dRtip = self.dCMy["dRtip"]
        dCMz_dRtip = self.dCMz["dRtip"]
        dCMb_dRtip = self.dCMb["dRtip"]
        dCP_dRtip = self.dCP["dRtip"]

        dCT_dRtip_fd = np.zeros((self.npts, 1))
        dCY_dRtip_fd = np.zeros((self.npts, 1))
        dCZ_dRtip_fd = np.zeros((self.npts, 1))
        dCQ_dRtip_fd = np.zeros((self.npts, 1))
        dCMy_dRtip_fd = np.zeros((self.npts, 1))
        dCMz_dRtip_fd = np.zeros((self.npts, 1))
        dCMb_dRtip_fd = np.zeros((self.npts, 1))
        dCP_dRtip_fd = np.zeros((self.npts, 1))

        Rtip = float(self.Rtip)
        delta = 1e-6 * Rtip
        Rtip += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dRtip_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dRtip_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dRtip_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dRtip_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dRtip_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dRtip_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dRtip_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dRtip_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dRtip_fd, dCT_dRtip, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dRtip_fd, dCY_dRtip, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dRtip_fd, dCZ_dRtip, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dRtip_fd, dCQ_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dRtip_fd, dCMy_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dRtip_fd, dCMz_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dRtip_fd, dCMb_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dRtip_fd, dCP_dRtip, rtol=5e-5, atol=1e-8)

    def test_dprecone1(self):

        dNp_dprecone = self.dNp["dprecone"]
        dTp_dprecone = self.dTp["dprecone"]

        dNp_dprecone_fd = np.zeros((self.n, 1))
        dTp_dprecone_fd = np.zeros((self.n, 1))

        precone = float(self.precone)
        delta = 1e-6 * precone
        precone += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dprecone_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dprecone_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dprecone_fd, dNp_dprecone, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(dTp_dprecone_fd, dTp_dprecone, rtol=1e-5, atol=1e-7)

    def test_dprecone2(self):

        dT_dprecone = self.dT["dprecone"]
        dY_dprecone = self.dY["dprecone"]
        dZ_dprecone = self.dZ["dprecone"]
        dQ_dprecone = self.dQ["dprecone"]
        dMy_dprecone = self.dMy["dprecone"]
        dMz_dprecone = self.dMz["dprecone"]
        dMb_dprecone = self.dMb["dprecone"]
        dP_dprecone = self.dP["dprecone"]

        dT_dprecone_fd = np.zeros((self.npts, 1))
        dY_dprecone_fd = np.zeros((self.npts, 1))
        dZ_dprecone_fd = np.zeros((self.npts, 1))
        dQ_dprecone_fd = np.zeros((self.npts, 1))
        dMy_dprecone_fd = np.zeros((self.npts, 1))
        dMz_dprecone_fd = np.zeros((self.npts, 1))
        dMb_dprecone_fd = np.zeros((self.npts, 1))
        dP_dprecone_fd = np.zeros((self.npts, 1))

        precone = float(self.precone)
        delta = 1e-6 * precone
        precone += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dprecone_fd[:, 0] = (Td - self.T) / delta
        dY_dprecone_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dprecone_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dprecone_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dprecone_fd[:, 0] = (Myd - self.My) / delta
        dMz_dprecone_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dprecone_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dprecone_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dprecone_fd, dT_dprecone, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dprecone_fd, dY_dprecone, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dprecone_fd, dZ_dprecone, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dprecone_fd, dQ_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dprecone_fd, dMy_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dprecone_fd, dMz_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dprecone_fd, dMb_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dprecone_fd, dP_dprecone, rtol=5e-5, atol=1e-8)

    def test_dprecone3(self):

        dCT_dprecone = self.dCT["dprecone"]
        dCY_dprecone = self.dCY["dprecone"]
        dCZ_dprecone = self.dCZ["dprecone"]
        dCQ_dprecone = self.dCQ["dprecone"]
        dCMy_dprecone = self.dCMy["dprecone"]
        dCMz_dprecone = self.dCMz["dprecone"]
        dCMb_dprecone = self.dCMb["dprecone"]
        dCP_dprecone = self.dCP["dprecone"]

        dCT_dprecone_fd = np.zeros((self.npts, 1))
        dCY_dprecone_fd = np.zeros((self.npts, 1))
        dCZ_dprecone_fd = np.zeros((self.npts, 1))
        dCQ_dprecone_fd = np.zeros((self.npts, 1))
        dCMy_dprecone_fd = np.zeros((self.npts, 1))
        dCMz_dprecone_fd = np.zeros((self.npts, 1))
        dCMb_dprecone_fd = np.zeros((self.npts, 1))
        dCP_dprecone_fd = np.zeros((self.npts, 1))

        precone = float(self.precone)
        delta = 1e-6 * precone
        precone += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dprecone_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dprecone_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dprecone_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dprecone_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dprecone_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dprecone_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dprecone_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dprecone_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dprecone_fd, dCT_dprecone, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dprecone_fd, dCY_dprecone, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dprecone_fd, dCZ_dprecone, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dprecone_fd, dCQ_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dprecone_fd, dCMy_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dprecone_fd, dCMz_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dprecone_fd, dCMb_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dprecone_fd, dCP_dprecone, rtol=5e-5, atol=1e-8)

    def test_dtilt1(self):

        dNp_dtilt = self.dNp["dtilt"]
        dTp_dtilt = self.dTp["dtilt"]

        dNp_dtilt_fd = np.zeros((self.n, 1))
        dTp_dtilt_fd = np.zeros((self.n, 1))

        tilt = float(self.tilt)
        delta = 1e-6 * tilt
        tilt += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dtilt_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dtilt_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dtilt_fd, dNp_dtilt, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dTp_dtilt_fd, dTp_dtilt, rtol=1e-5, atol=1e-8)

    def test_dtilt2(self):

        dT_dtilt = self.dT["dtilt"]
        dY_dtilt = self.dY["dtilt"]
        dZ_dtilt = self.dZ["dtilt"]
        dQ_dtilt = self.dQ["dtilt"]
        dMy_dtilt = self.dMy["dtilt"]
        dMz_dtilt = self.dMz["dtilt"]
        dMb_dtilt = self.dMb["dtilt"]
        dP_dtilt = self.dP["dtilt"]

        dT_dtilt_fd = np.zeros((self.npts, 1))
        dY_dtilt_fd = np.zeros((self.npts, 1))
        dZ_dtilt_fd = np.zeros((self.npts, 1))
        dQ_dtilt_fd = np.zeros((self.npts, 1))
        dMy_dtilt_fd = np.zeros((self.npts, 1))
        dMz_dtilt_fd = np.zeros((self.npts, 1))
        dMb_dtilt_fd = np.zeros((self.npts, 1))
        dP_dtilt_fd = np.zeros((self.npts, 1))

        tilt = float(self.tilt)
        delta = 1e-6 * tilt
        tilt += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dtilt_fd[:, 0] = (Td - self.T) / delta
        dY_dtilt_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dtilt_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dtilt_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dtilt_fd[:, 0] = (Myd - self.My) / delta
        dMz_dtilt_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dtilt_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dtilt_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dtilt_fd, dT_dtilt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dtilt_fd, dY_dtilt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dtilt_fd, dZ_dtilt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dtilt_fd, dQ_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dtilt_fd, dMy_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dtilt_fd, dMz_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dtilt_fd, dMb_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dtilt_fd, dP_dtilt, rtol=5e-5, atol=1e-8)

    def test_dtilt3(self):

        dCT_dtilt = self.dCT["dtilt"]
        dCY_dtilt = self.dCY["dtilt"]
        dCZ_dtilt = self.dCZ["dtilt"]
        dCQ_dtilt = self.dCQ["dtilt"]
        dCMy_dtilt = self.dCMy["dtilt"]
        dCMz_dtilt = self.dCMz["dtilt"]
        dCMb_dtilt = self.dCMb["dtilt"]
        dCP_dtilt = self.dCP["dtilt"]

        dCT_dtilt_fd = np.zeros((self.npts, 1))
        dCY_dtilt_fd = np.zeros((self.npts, 1))
        dCZ_dtilt_fd = np.zeros((self.npts, 1))
        dCQ_dtilt_fd = np.zeros((self.npts, 1))
        dCMy_dtilt_fd = np.zeros((self.npts, 1))
        dCMz_dtilt_fd = np.zeros((self.npts, 1))
        dCMb_dtilt_fd = np.zeros((self.npts, 1))
        dCP_dtilt_fd = np.zeros((self.npts, 1))

        tilt = float(self.tilt)
        delta = 1e-6 * tilt
        tilt += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dtilt_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dtilt_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dtilt_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dtilt_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dtilt_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dtilt_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dtilt_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dtilt_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dtilt_fd, dCT_dtilt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dtilt_fd, dCY_dtilt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dtilt_fd, dCZ_dtilt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dtilt_fd, dCQ_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dtilt_fd, dCMy_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dtilt_fd, dCMz_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dtilt_fd, dCMb_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dtilt_fd, dCP_dtilt, rtol=5e-5, atol=1e-8)

    def test_dhubht1(self):

        dNp_dhubht = self.dNp["dhubHt"]
        dTp_dhubht = self.dTp["dhubHt"]

        dNp_dhubht_fd = np.zeros((self.n, 1))
        dTp_dhubht_fd = np.zeros((self.n, 1))

        hubht = float(self.hubHt)
        delta = 1e-6 * hubht
        hubht += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            hubht,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dhubht_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dhubht_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dhubht_fd, dNp_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dhubht_fd, dTp_dhubht, rtol=1e-5, atol=1e-8)

    def test_dhubht2(self):

        dT_dhubht = self.dT["dhubHt"]
        dY_dhubht = self.dY["dhubHt"]
        dZ_dhubht = self.dZ["dhubHt"]
        dQ_dhubht = self.dQ["dhubHt"]
        dMy_dhubht = self.dMy["dhubHt"]
        dMz_dhubht = self.dMz["dhubHt"]
        dMb_dhubht = self.dMb["dhubHt"]
        dP_dhubht = self.dP["dhubHt"]

        dT_dhubht_fd = np.zeros((self.npts, 1))
        dY_dhubht_fd = np.zeros((self.npts, 1))
        dZ_dhubht_fd = np.zeros((self.npts, 1))
        dQ_dhubht_fd = np.zeros((self.npts, 1))
        dMy_dhubht_fd = np.zeros((self.npts, 1))
        dMz_dhubht_fd = np.zeros((self.npts, 1))
        dMb_dhubht_fd = np.zeros((self.npts, 1))
        dP_dhubht_fd = np.zeros((self.npts, 1))

        hubht = float(self.hubHt)
        delta = 1e-6 * hubht
        hubht += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            hubht,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dhubht_fd[:, 0] = (Td - self.T) / delta
        dY_dhubht_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dhubht_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dhubht_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dhubht_fd[:, 0] = (Myd - self.My) / delta
        dMz_dhubht_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dhubht_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dhubht_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dhubht_fd, dT_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dhubht_fd, dY_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dhubht_fd, dZ_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dhubht_fd, dQ_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dhubht_fd, dMy_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dhubht_fd, dMz_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dhubht_fd, dMb_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dhubht_fd, dP_dhubht, rtol=5e-5, atol=1e-8)

    def test_dhubht3(self):

        dCT_dhubht = self.dCT["dhubHt"]
        dCY_dhubht = self.dCY["dhubHt"]
        dCZ_dhubht = self.dCZ["dhubHt"]
        dCQ_dhubht = self.dCQ["dhubHt"]
        dCMy_dhubht = self.dCMy["dhubHt"]
        dCMz_dhubht = self.dCMz["dhubHt"]
        dCMb_dhubht = self.dCMb["dhubHt"]
        dCP_dhubht = self.dCP["dhubHt"]

        dCT_dhubht_fd = np.zeros((self.npts, 1))
        dCY_dhubht_fd = np.zeros((self.npts, 1))
        dCZ_dhubht_fd = np.zeros((self.npts, 1))
        dCQ_dhubht_fd = np.zeros((self.npts, 1))
        dCMy_dhubht_fd = np.zeros((self.npts, 1))
        dCMz_dhubht_fd = np.zeros((self.npts, 1))
        dCMb_dhubht_fd = np.zeros((self.npts, 1))
        dCP_dhubht_fd = np.zeros((self.npts, 1))

        hubht = float(self.hubHt)
        delta = 1e-6 * hubht
        hubht += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            hubht,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dhubht_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dhubht_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dhubht_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dhubht_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dhubht_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dhubht_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dhubht_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dhubht_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dhubht_fd, dCT_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dhubht_fd, dCY_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dhubht_fd, dCZ_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dhubht_fd, dCQ_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dhubht_fd, dCMy_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dhubht_fd, dCMz_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dhubht_fd, dCMb_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dhubht_fd, dCP_dhubht, rtol=5e-5, atol=1e-8)

    def test_dyaw1(self):

        dNp_dyaw = self.dNp["dyaw"]
        dTp_dyaw = self.dTp["dyaw"]

        dNp_dyaw_fd = np.zeros((self.n, 1))
        dTp_dyaw_fd = np.zeros((self.n, 1))

        yaw = float(self.yaw)
        delta = 1e-6
        yaw += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dyaw_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dyaw_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dyaw_fd, dNp_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dyaw_fd, dTp_dyaw, rtol=1e-5, atol=1e-8)

    def test_dyaw2(self):

        dT_dyaw = self.dT["dyaw"]
        dY_dyaw = self.dY["dyaw"]
        dZ_dyaw = self.dZ["dyaw"]
        dQ_dyaw = self.dQ["dyaw"]
        dMy_dyaw = self.dMy["dyaw"]
        dMz_dyaw = self.dMz["dyaw"]
        dMb_dyaw = self.dMb["dyaw"]
        dP_dyaw = self.dP["dyaw"]

        dT_dyaw_fd = np.zeros((self.npts, 1))
        dY_dyaw_fd = np.zeros((self.npts, 1))
        dZ_dyaw_fd = np.zeros((self.npts, 1))
        dQ_dyaw_fd = np.zeros((self.npts, 1))
        dMy_dyaw_fd = np.zeros((self.npts, 1))
        dMz_dyaw_fd = np.zeros((self.npts, 1))
        dMb_dyaw_fd = np.zeros((self.npts, 1))
        dP_dyaw_fd = np.zeros((self.npts, 1))

        yaw = float(self.yaw)
        delta = 1e-6
        yaw += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dyaw_fd[:, 0] = (Td - self.T) / delta
        dY_dyaw_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dyaw_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dyaw_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dyaw_fd[:, 0] = (Myd - self.My) / delta
        dMz_dyaw_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dyaw_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dyaw_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dyaw_fd, dT_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dyaw_fd, dY_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dyaw_fd, dZ_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dyaw_fd, dQ_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dyaw_fd, dMy_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dyaw_fd, dMz_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dyaw_fd, dMb_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dyaw_fd, dP_dyaw, rtol=5e-5, atol=1e-8)

    def test_dyaw3(self):

        dCT_dyaw = self.dCT["dyaw"]
        dCY_dyaw = self.dCY["dyaw"]
        dCZ_dyaw = self.dCZ["dyaw"]
        dCQ_dyaw = self.dCQ["dyaw"]
        dCMy_dyaw = self.dCMy["dyaw"]
        dCMz_dyaw = self.dCMz["dyaw"]
        dCMb_dyaw = self.dCMb["dyaw"]
        dCP_dyaw = self.dCP["dyaw"]

        dCT_dyaw_fd = np.zeros((self.npts, 1))
        dCY_dyaw_fd = np.zeros((self.npts, 1))
        dCZ_dyaw_fd = np.zeros((self.npts, 1))
        dCQ_dyaw_fd = np.zeros((self.npts, 1))
        dCMy_dyaw_fd = np.zeros((self.npts, 1))
        dCMz_dyaw_fd = np.zeros((self.npts, 1))
        dCMb_dyaw_fd = np.zeros((self.npts, 1))
        dCP_dyaw_fd = np.zeros((self.npts, 1))

        yaw = float(self.yaw)
        delta = 1e-6
        yaw += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dyaw_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dyaw_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dyaw_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dyaw_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dyaw_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dyaw_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dyaw_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dyaw_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dyaw_fd, dCT_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dyaw_fd, dCY_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dyaw_fd, dCZ_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dyaw_fd, dCQ_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dyaw_fd, dCMy_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dyaw_fd, dCMz_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dyaw_fd, dCMb_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dyaw_fd, dCP_dyaw, rtol=5e-5, atol=1e-8)

    def test_dshear1(self):

        dNp_dshear = self.dNp["dshear"]
        dTp_dshear = self.dTp["dshear"]

        dNp_dshear_fd = np.zeros((self.n, 1))
        dTp_dshear_fd = np.zeros((self.n, 1))

        shearExp = float(self.shearExp)
        delta = 1e-6
        shearExp += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dshear_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dshear_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dshear_fd, dNp_dshear, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dshear_fd, dTp_dshear, rtol=1e-5, atol=1e-8)

    def test_dshear2(self):

        dT_dshear = self.dT["dshear"]
        dY_dshear = self.dY["dshear"]
        dZ_dshear = self.dZ["dshear"]
        dQ_dshear = self.dQ["dshear"]
        dMy_dshear = self.dMy["dshear"]
        dMz_dshear = self.dMz["dshear"]
        dMb_dshear = self.dMb["dshear"]
        dP_dshear = self.dP["dshear"]

        dT_dshear_fd = np.zeros((self.npts, 1))
        dY_dshear_fd = np.zeros((self.npts, 1))
        dZ_dshear_fd = np.zeros((self.npts, 1))
        dQ_dshear_fd = np.zeros((self.npts, 1))
        dMy_dshear_fd = np.zeros((self.npts, 1))
        dMz_dshear_fd = np.zeros((self.npts, 1))
        dMb_dshear_fd = np.zeros((self.npts, 1))
        dP_dshear_fd = np.zeros((self.npts, 1))

        shearExp = float(self.shearExp)
        delta = 1e-6
        shearExp += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dshear_fd[:, 0] = (Td - self.T) / delta
        dY_dshear_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dshear_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dshear_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dshear_fd[:, 0] = (Myd - self.My) / delta
        dMz_dshear_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dshear_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dshear_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dshear_fd, dT_dshear, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dshear_fd, dY_dshear, rtol=2e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dshear_fd, dZ_dshear, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dshear_fd, dQ_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dshear_fd, dMy_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dshear_fd, dMz_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dshear_fd, dMb_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dshear_fd, dP_dshear, rtol=5e-5, atol=1e-8)

    def test_dshear3(self):

        dCT_dshear = self.dCT["dshear"]
        dCY_dshear = self.dCY["dshear"]
        dCZ_dshear = self.dCZ["dshear"]
        dCQ_dshear = self.dCQ["dshear"]
        dCMy_dshear = self.dCMy["dshear"]
        dCMz_dshear = self.dCMz["dshear"]
        dCMb_dshear = self.dCMb["dshear"]
        dCP_dshear = self.dCP["dshear"]

        dCT_dshear_fd = np.zeros((self.npts, 1))
        dCY_dshear_fd = np.zeros((self.npts, 1))
        dCZ_dshear_fd = np.zeros((self.npts, 1))
        dCQ_dshear_fd = np.zeros((self.npts, 1))
        dCMy_dshear_fd = np.zeros((self.npts, 1))
        dCMz_dshear_fd = np.zeros((self.npts, 1))
        dCMb_dshear_fd = np.zeros((self.npts, 1))
        dCP_dshear_fd = np.zeros((self.npts, 1))

        shearExp = float(self.shearExp)
        delta = 1e-6
        shearExp += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            self.precone,
            self.tilt,
            self.yaw,
            shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dshear_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dshear_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dshear_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dshear_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dshear_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dshear_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dshear_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dshear_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dshear_fd, dCT_dshear, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dshear_fd, dCY_dshear, rtol=2e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dshear_fd, dCZ_dshear, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dshear_fd, dCQ_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dshear_fd, dCMy_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dshear_fd, dCMz_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dshear_fd, dCMb_dshear, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dshear_fd, dCP_dshear, rtol=5e-5, atol=1e-8)

    def test_dazimuth1(self):

        dNp_dazimuth = self.dNp["dazimuth"]
        dTp_dazimuth = self.dTp["dazimuth"]

        dNp_dazimuth_fd = np.zeros((self.n, 1))
        dTp_dazimuth_fd = np.zeros((self.n, 1))

        azimuth = float(self.azimuth)
        delta = 1e-6 * azimuth
        azimuth += delta

        outputs, _ = self.rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, azimuth)
        Npd = outputs["Np"]
        Tpd = outputs["Tp"]

        dNp_dazimuth_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dazimuth_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dazimuth_fd, dNp_dazimuth, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dazimuth_fd, dTp_dazimuth, rtol=1e-5, atol=1e-6)

    def test_dUinf1(self):

        dNp_dUinf = self.dNp["dUinf"]
        dTp_dUinf = self.dTp["dUinf"]

        dNp_dUinf_fd = np.zeros((self.n, 1))
        dTp_dUinf_fd = np.zeros((self.n, 1))

        Uinf = float(self.Uinf)
        delta = 1e-6 * Uinf
        Uinf += delta

        outputs, _ = self.rotor.distributedAeroLoads(Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = outputs["Np"]
        Tpd = outputs["Tp"]

        dNp_dUinf_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dUinf_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dUinf_fd, dNp_dUinf, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dUinf_fd, dTp_dUinf, rtol=1e-5, atol=1e-6)

    def test_dUinf2(self):

        dT_dUinf = self.dT["dUinf"]
        dY_dUinf = self.dY["dUinf"]
        dZ_dUinf = self.dZ["dUinf"]
        dQ_dUinf = self.dQ["dUinf"]
        dMy_dUinf = self.dMy["dUinf"]
        dMz_dUinf = self.dMz["dUinf"]
        dMb_dUinf = self.dMb["dUinf"]
        dP_dUinf = self.dP["dUinf"]

        dT_dUinf_fd = np.zeros((self.npts, self.npts))
        dY_dUinf_fd = np.zeros((self.npts, self.npts))
        dZ_dUinf_fd = np.zeros((self.npts, self.npts))
        dQ_dUinf_fd = np.zeros((self.npts, self.npts))
        dMy_dUinf_fd = np.zeros((self.npts, self.npts))
        dMz_dUinf_fd = np.zeros((self.npts, self.npts))
        dMb_dUinf_fd = np.zeros((self.npts, self.npts))
        dP_dUinf_fd = np.zeros((self.npts, self.npts))

        Uinf = float(self.Uinf)
        delta = 1e-6 * Uinf
        Uinf += delta

        outputs, _ = self.rotor.evaluate([Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dUinf_fd[:, 0] = (Td - self.T) / delta
        dY_dUinf_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dUinf_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dUinf_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dUinf_fd[:, 0] = (Myd - self.My) / delta
        dMz_dUinf_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dUinf_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dUinf_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dUinf_fd, dT_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dUinf_fd, dY_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dUinf_fd, dZ_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dUinf_fd, dQ_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dUinf_fd, dMy_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dUinf_fd, dMz_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dUinf_fd, dMb_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dUinf_fd, dP_dUinf, rtol=5e-5, atol=1e-8)

    def test_dUinf3(self):

        dCT_dUinf = self.dCT["dUinf"]
        dCY_dUinf = self.dCY["dUinf"]
        dCZ_dUinf = self.dCZ["dUinf"]
        dCQ_dUinf = self.dCQ["dUinf"]
        dCMy_dUinf = self.dCMy["dUinf"]
        dCMz_dUinf = self.dCMz["dUinf"]
        dCMb_dUinf = self.dCMb["dUinf"]
        dCP_dUinf = self.dCP["dUinf"]

        dCT_dUinf_fd = np.zeros((self.npts, self.npts))
        dCY_dUinf_fd = np.zeros((self.npts, self.npts))
        dCZ_dUinf_fd = np.zeros((self.npts, self.npts))
        dCQ_dUinf_fd = np.zeros((self.npts, self.npts))
        dCMy_dUinf_fd = np.zeros((self.npts, self.npts))
        dCMz_dUinf_fd = np.zeros((self.npts, self.npts))
        dCMb_dUinf_fd = np.zeros((self.npts, self.npts))
        dCP_dUinf_fd = np.zeros((self.npts, self.npts))

        Uinf = float(self.Uinf)
        delta = 1e-6 * Uinf
        Uinf += delta

        outputs, _ = self.rotor.evaluate([Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dUinf_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dUinf_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dUinf_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dUinf_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dUinf_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dUinf_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dUinf_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dUinf_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dUinf_fd, dCT_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dUinf_fd, dCY_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dUinf_fd, dCZ_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dUinf_fd, dCQ_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dUinf_fd, dCMy_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dUinf_fd, dCMz_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dUinf_fd, dCMb_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dUinf_fd, dCP_dUinf, rtol=5e-5, atol=1e-8)

    def test_dOmega1(self):

        dNp_dOmega = self.dNp["dOmega"]
        dTp_dOmega = self.dTp["dOmega"]

        dNp_dOmega_fd = np.zeros((self.n, 1))
        dTp_dOmega_fd = np.zeros((self.n, 1))

        Omega = float(self.Omega)
        delta = 1e-6 * Omega
        Omega += delta

        loads, _ = self.rotor.distributedAeroLoads(self.Uinf, Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dOmega_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dOmega_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dOmega_fd, dNp_dOmega, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dOmega_fd, dTp_dOmega, rtol=1e-5, atol=1e-6)

    def test_dOmega2(self):

        dT_dOmega = self.dT["dOmega"]
        dY_dOmega = self.dY["dOmega"]
        dZ_dOmega = self.dZ["dOmega"]
        dQ_dOmega = self.dQ["dOmega"]
        dMy_dOmega = self.dMy["dOmega"]
        dMz_dOmega = self.dMz["dOmega"]
        dMb_dOmega = self.dMb["dOmega"]
        dP_dOmega = self.dP["dOmega"]

        dT_dOmega_fd = np.zeros((self.npts, self.npts))
        dY_dOmega_fd = np.zeros((self.npts, self.npts))
        dZ_dOmega_fd = np.zeros((self.npts, self.npts))
        dQ_dOmega_fd = np.zeros((self.npts, self.npts))
        dMy_dOmega_fd = np.zeros((self.npts, self.npts))
        dMz_dOmega_fd = np.zeros((self.npts, self.npts))
        dMb_dOmega_fd = np.zeros((self.npts, self.npts))
        dP_dOmega_fd = np.zeros((self.npts, self.npts))

        Omega = float(self.Omega)
        delta = 1e-6 * Omega
        Omega += delta

        outputs, _ = self.rotor.evaluate([self.Uinf], [Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dOmega_fd[:, 0] = (Td - self.T) / delta
        dY_dOmega_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dOmega_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dOmega_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dOmega_fd[:, 0] = (Myd - self.My) / delta
        dMz_dOmega_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dOmega_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dOmega_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dOmega_fd, dT_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dOmega_fd, dY_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dOmega_fd, dZ_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dOmega_fd, dQ_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dOmega_fd, dMy_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dOmega_fd, dMz_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dOmega_fd, dMb_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dOmega_fd, dP_dOmega, rtol=5e-5, atol=1e-8)

    def test_dOmega3(self):

        dCT_dOmega = self.dCT["dOmega"]
        dCY_dOmega = self.dCY["dOmega"]
        dCZ_dOmega = self.dCZ["dOmega"]
        dCQ_dOmega = self.dCQ["dOmega"]
        dCMy_dOmega = self.dCMy["dOmega"]
        dCMz_dOmega = self.dCMz["dOmega"]
        dCMb_dOmega = self.dCMb["dOmega"]
        dCP_dOmega = self.dCP["dOmega"]

        dCT_dOmega_fd = np.zeros((self.npts, self.npts))
        dCY_dOmega_fd = np.zeros((self.npts, self.npts))
        dCZ_dOmega_fd = np.zeros((self.npts, self.npts))
        dCQ_dOmega_fd = np.zeros((self.npts, self.npts))
        dCMy_dOmega_fd = np.zeros((self.npts, self.npts))
        dCMz_dOmega_fd = np.zeros((self.npts, self.npts))
        dCMb_dOmega_fd = np.zeros((self.npts, self.npts))
        dCP_dOmega_fd = np.zeros((self.npts, self.npts))

        Omega = float(self.Omega)
        delta = 1e-6 * Omega
        Omega += delta

        outputs, _ = self.rotor.evaluate([self.Uinf], [Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dOmega_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dOmega_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dOmega_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dOmega_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dOmega_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dOmega_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dOmega_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dOmega_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dOmega_fd, dCT_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dOmega_fd, dCY_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dOmega_fd, dCZ_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dOmega_fd, dCQ_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dOmega_fd, dCMy_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dOmega_fd, dCMz_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dOmega_fd, dCMb_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dOmega_fd, dCP_dOmega, rtol=5e-5, atol=1e-8)

    def test_dpitch1(self):

        dNp_dpitch = self.dNp["dpitch"]
        dTp_dpitch = self.dTp["dpitch"]

        dNp_dpitch_fd = np.zeros((self.n, 1))
        dTp_dpitch_fd = np.zeros((self.n, 1))

        pitch = float(self.pitch)
        delta = 1e-6
        pitch += delta

        loads, _ = self.rotor.distributedAeroLoads(self.Uinf, self.Omega, pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]

        dNp_dpitch_fd[:, 0] = (Npd - self.Np) / delta
        dTp_dpitch_fd[:, 0] = (Tpd - self.Tp) / delta

        np.testing.assert_allclose(dNp_dpitch_fd, dNp_dpitch, rtol=5e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dpitch_fd, dTp_dpitch, rtol=5e-5, atol=1e-6)

    def test_dpitch2(self):

        dT_dpitch = self.dT["dpitch"]
        dY_dpitch = self.dY["dpitch"]
        dZ_dpitch = self.dZ["dpitch"]
        dQ_dpitch = self.dQ["dpitch"]
        dMy_dpitch = self.dMy["dpitch"]
        dMz_dpitch = self.dMz["dpitch"]
        dMb_dpitch = self.dMb["dpitch"]
        dP_dpitch = self.dP["dpitch"]

        dT_dpitch_fd = np.zeros((self.npts, 1))
        dY_dpitch_fd = np.zeros((self.npts, 1))
        dZ_dpitch_fd = np.zeros((self.npts, 1))
        dQ_dpitch_fd = np.zeros((self.npts, 1))
        dMy_dpitch_fd = np.zeros((self.npts, 1))
        dMz_dpitch_fd = np.zeros((self.npts, 1))
        dMb_dpitch_fd = np.zeros((self.npts, 1))
        dP_dpitch_fd = np.zeros((self.npts, 1))

        pitch = float(self.pitch)
        delta = 1e-6
        pitch += delta

        outputs, _ = self.rotor.evaluate([self.Uinf], [self.Omega], [pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dpitch_fd[:, 0] = (Td - self.T) / delta
        dY_dpitch_fd[:, 0] = (Yd - self.Y) / delta
        dZ_dpitch_fd[:, 0] = (Zd - self.Z) / delta
        dQ_dpitch_fd[:, 0] = (Qd - self.Q) / delta
        dMy_dpitch_fd[:, 0] = (Myd - self.My) / delta
        dMz_dpitch_fd[:, 0] = (Mzd - self.Mz) / delta
        dMb_dpitch_fd[:, 0] = (Mbd - self.Mb) / delta
        dP_dpitch_fd[:, 0] = (Pd - self.P) / delta

        np.testing.assert_allclose(dT_dpitch_fd, dT_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dY_dpitch_fd, dY_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dZ_dpitch_fd, dZ_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dpitch_fd, dQ_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMy_dpitch_fd, dMy_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMz_dpitch_fd, dMz_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dMb_dpitch_fd, dMb_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dpitch_fd, dP_dpitch, rtol=5e-5, atol=1e-8)

    def test_dpitch3(self):

        dCT_dpitch = self.dCT["dpitch"]
        dCY_dpitch = self.dCY["dpitch"]
        dCZ_dpitch = self.dCZ["dpitch"]
        dCQ_dpitch = self.dCQ["dpitch"]
        dCMy_dpitch = self.dCMy["dpitch"]
        dCMz_dpitch = self.dCMz["dpitch"]
        dCMb_dpitch = self.dCMb["dpitch"]
        dCP_dpitch = self.dCP["dpitch"]

        dCT_dpitch_fd = np.zeros((self.npts, 1))
        dCY_dpitch_fd = np.zeros((self.npts, 1))
        dCZ_dpitch_fd = np.zeros((self.npts, 1))
        dCQ_dpitch_fd = np.zeros((self.npts, 1))
        dCMy_dpitch_fd = np.zeros((self.npts, 1))
        dCMz_dpitch_fd = np.zeros((self.npts, 1))
        dCMb_dpitch_fd = np.zeros((self.npts, 1))
        dCP_dpitch_fd = np.zeros((self.npts, 1))

        pitch = float(self.pitch)
        delta = 1e-6
        pitch += delta

        outputs, _ = self.rotor.evaluate([self.Uinf], [self.Omega], [pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dpitch_fd[:, 0] = (CTd - self.CT) / delta
        dCY_dpitch_fd[:, 0] = (CYd - self.CY) / delta
        dCZ_dpitch_fd[:, 0] = (CZd - self.CZ) / delta
        dCQ_dpitch_fd[:, 0] = (CQd - self.CQ) / delta
        dCMy_dpitch_fd[:, 0] = (CMyd - self.CMy) / delta
        dCMz_dpitch_fd[:, 0] = (CMzd - self.CMz) / delta
        dCMb_dpitch_fd[:, 0] = (CMbd - self.CMb) / delta
        dCP_dpitch_fd[:, 0] = (CPd - self.CP) / delta

        np.testing.assert_allclose(dCT_dpitch_fd, dCT_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCY_dpitch_fd, dCY_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCZ_dpitch_fd, dCZ_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dpitch_fd, dCQ_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMy_dpitch_fd, dCMy_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMz_dpitch_fd, dCMz_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCMb_dpitch_fd, dCMb_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dpitch_fd, dCP_dpitch, rtol=5e-5, atol=1e-8)

    def test_dprecurve1(self):

        precurve = np.linspace(1, 10, self.n)
        precurveTip = precurve[-1]
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            precurve=precurve,
            precurveTip=precurveTip,
        )
        precurve = rotor.precurve.copy()

        loads, derivs = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Np = loads["Np"]
        Tp = loads["Tp"]
        dNp = derivs["dNp"]
        dTp = derivs["dTp"]

        dNp_dprecurve = dNp["dprecurve"]
        dTp_dprecurve = dTp["dprecurve"]

        dNp_dprecurve_fd = np.zeros((self.n, self.n))
        dTp_dprecurve_fd = np.zeros((self.n, self.n))

        for i in range(self.n):
            pc = np.array(precurve)
            delta = 1e-6 * pc[i]
            pc[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
                precurve=pc,
                precurveTip=precurveTip,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dprecurve_fd[:, i] = (Npd - Np) / delta
            dTp_dprecurve_fd[:, i] = (Tpd - Tp) / delta

        np.testing.assert_allclose(dNp_dprecurve_fd, dNp_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dprecurve_fd, dTp_dprecurve, rtol=3e-4, atol=1e-8)

    def test_dprecurve2(self):

        precurve = np.linspace(1, 10, self.n)
        precurveTip = precurve[-1]
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            precurve=precurve,
            precurveTip=precurveTip,
        )
        precurve = rotor.precurve.copy()

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        P = outputs["P"]
        T = outputs["T"]
        Y = outputs["Y"]
        Z = outputs["Z"]
        Q = outputs["Q"]
        My = outputs["My"]
        Mz = outputs["Mz"]
        Mb = outputs["Mb"]
        dP = derivs["dP"]
        dT = derivs["dT"]
        dY = derivs["dY"]
        dZ = derivs["dZ"]
        dQ = derivs["dQ"]
        dMy = derivs["dMy"]
        dMz = derivs["dMz"]
        dMb = derivs["dMb"]

        dT_dprecurve = dT["dprecurve"]
        dY_dprecurve = dY["dprecurve"]
        dZ_dprecurve = dZ["dprecurve"]
        dQ_dprecurve = dQ["dprecurve"]
        dMy_dprecurve = dMy["dprecurve"]
        dMz_dprecurve = dMz["dprecurve"]
        dMb_dprecurve = dMb["dprecurve"]
        dP_dprecurve = dP["dprecurve"]

        dT_dprecurve_fd = np.zeros((self.npts, self.n))
        dY_dprecurve_fd = np.zeros((self.npts, self.n))
        dZ_dprecurve_fd = np.zeros((self.npts, self.n))
        dQ_dprecurve_fd = np.zeros((self.npts, self.n))
        dMy_dprecurve_fd = np.zeros((self.npts, self.n))
        dMz_dprecurve_fd = np.zeros((self.npts, self.n))
        dMb_dprecurve_fd = np.zeros((self.npts, self.n))
        dP_dprecurve_fd = np.zeros((self.npts, self.n))
        for i in range(self.n):
            pc = np.array(precurve)
            delta = 1e-6 * pc[i]
            pc[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
                precurve=pc,
                precurveTip=precurveTip,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
            Pd = outputs["P"]
            Td = outputs["T"]
            Yd = outputs["Y"]
            Zd = outputs["Z"]
            Qd = outputs["Q"]
            Myd = outputs["My"]
            Mzd = outputs["Mz"]
            Mbd = outputs["Mb"]

            dT_dprecurve_fd[:, i] = (Td - T) / delta
            dY_dprecurve_fd[:, i] = (Yd - Y) / delta
            dZ_dprecurve_fd[:, i] = (Zd - Z) / delta
            dQ_dprecurve_fd[:, i] = (Qd - Q) / delta
            dMy_dprecurve_fd[:, i] = (Myd - My) / delta
            dMz_dprecurve_fd[:, i] = (Mzd - Mz) / delta
            dMb_dprecurve_fd[:, i] = (Mbd - Mb) / delta
            dP_dprecurve_fd[:, i] = (Pd - P) / delta

        np.testing.assert_allclose(dT_dprecurve_fd, dT_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dY_dprecurve_fd, dY_dprecurve, rtol=3e-3, atol=1e-8)
        np.testing.assert_allclose(dZ_dprecurve_fd, dZ_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dQ_dprecurve_fd, dQ_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dMy_dprecurve_fd, dMy_dprecurve, rtol=8e-4, atol=1e-8)
        np.testing.assert_allclose(dMz_dprecurve_fd, dMz_dprecurve, rtol=5e-3, atol=1e-8)
        np.testing.assert_allclose(dMb_dprecurve_fd, dMb_dprecurve, rtol=8e-4, atol=1e-8)
        np.testing.assert_allclose(dP_dprecurve_fd, dP_dprecurve, rtol=3e-4, atol=1e-8)

    def test_dprecurve3(self):

        precurve = np.linspace(1, 10, self.n)
        precurveTip = precurve[-1]
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            precurve=precurve,
            precurveTip=precurveTip,
        )
        precurve = rotor.precurve.copy()

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CP = outputs["CP"]
        CT = outputs["CT"]
        CY = outputs["CY"]
        CZ = outputs["CZ"]
        CQ = outputs["CQ"]
        CMy = outputs["CMy"]
        CMz = outputs["CMz"]
        CMb = outputs["CMb"]
        dCP = derivs["dCP"]
        dCT = derivs["dCT"]
        dCY = derivs["dCY"]
        dCZ = derivs["dCZ"]
        dCQ = derivs["dCQ"]
        dCMy = derivs["dCMy"]
        dCMz = derivs["dCMz"]
        dCMb = derivs["dCMb"]

        dCT_dprecurve = dCT["dprecurve"]
        dCY_dprecurve = dCY["dprecurve"]
        dCZ_dprecurve = dCZ["dprecurve"]
        dCQ_dprecurve = dCQ["dprecurve"]
        dCMy_dprecurve = dCMy["dprecurve"]
        dCMz_dprecurve = dCMz["dprecurve"]
        dCMb_dprecurve = dCMb["dprecurve"]
        dCP_dprecurve = dCP["dprecurve"]

        dCT_dprecurve_fd = np.zeros((self.npts, self.n))
        dCY_dprecurve_fd = np.zeros((self.npts, self.n))
        dCZ_dprecurve_fd = np.zeros((self.npts, self.n))
        dCQ_dprecurve_fd = np.zeros((self.npts, self.n))
        dCMy_dprecurve_fd = np.zeros((self.npts, self.n))
        dCMz_dprecurve_fd = np.zeros((self.npts, self.n))
        dCMb_dprecurve_fd = np.zeros((self.npts, self.n))
        dCP_dprecurve_fd = np.zeros((self.npts, self.n))
        for i in range(self.n):
            pc = np.array(precurve)
            delta = 1e-6 * pc[i]
            pc[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
                precurve=pc,
                precurveTip=precurveTip,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
            CPd = outputs["CP"]
            CTd = outputs["CT"]
            CYd = outputs["CY"]
            CZd = outputs["CZ"]
            CQd = outputs["CQ"]
            CMyd = outputs["CMy"]
            CMzd = outputs["CMz"]
            CMbd = outputs["CMb"]

            dCT_dprecurve_fd[:, i] = (CTd - CT) / delta
            dCY_dprecurve_fd[:, i] = (CYd - CY) / delta
            dCZ_dprecurve_fd[:, i] = (CZd - CZ) / delta
            dCQ_dprecurve_fd[:, i] = (CQd - CQ) / delta
            dCMy_dprecurve_fd[:, i] = (CMyd - CMy) / delta
            dCMz_dprecurve_fd[:, i] = (CMzd - CMz) / delta
            dCMb_dprecurve_fd[:, i] = (CMbd - CMb) / delta
            dCP_dprecurve_fd[:, i] = (CPd - CP) / delta

        np.testing.assert_allclose(dCT_dprecurve_fd, dCT_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCY_dprecurve_fd, dCY_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCZ_dprecurve_fd, dCZ_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCQ_dprecurve_fd, dCQ_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMy_dprecurve_fd, dCMy_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMz_dprecurve_fd, dCMz_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMb_dprecurve_fd, dCMb_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dprecurve_fd, dCP_dprecurve, rtol=3e-4, atol=1e-8)

    def test_dpresweep1(self):

        presweep = np.linspace(1, 10, self.n)
        presweepTip = presweep[-1]
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            presweep=presweep,
            presweepTip=presweepTip,
        )
        presweep = rotor.presweep.copy()

        loads, derivs = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Np = loads["Np"]
        Tp = loads["Tp"]
        dNp = derivs["dNp"]
        dTp = derivs["dTp"]

        dNp_dpresweep = dNp["dpresweep"]
        dTp_dpresweep = dTp["dpresweep"]

        dNp_dpresweep_fd = np.zeros((self.n, self.n))
        dTp_dpresweep_fd = np.zeros((self.n, self.n))

        for i in range(self.n):
            ps = np.array(presweep)
            delta = 1e-6 * ps[i]
            ps[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
                presweep=ps,
                presweepTip=presweepTip,
            )

            loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
            Npd = loads["Np"]
            Tpd = loads["Tp"]

            dNp_dpresweep_fd[:, i] = (Npd - Np) / delta
            dTp_dpresweep_fd[:, i] = (Tpd - Tp) / delta

        np.testing.assert_allclose(dNp_dpresweep_fd, dNp_dpresweep, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dpresweep_fd, dTp_dpresweep, rtol=1e-5, atol=1e-8)

    def test_dpresweep2(self):

        presweep = np.linspace(1, 10, self.n)
        presweepTip = presweep[-1]
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            presweep=presweep,
            presweepTip=presweepTip,
        )
        presweep = rotor.presweep.copy()

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        P = outputs["P"]
        T = outputs["T"]
        Y = outputs["Y"]
        Z = outputs["Z"]
        Q = outputs["Q"]
        My = outputs["My"]
        Mz = outputs["Mz"]
        Mb = outputs["Mb"]
        dP = derivs["dP"]
        dT = derivs["dT"]
        dY = derivs["dY"]
        dZ = derivs["dZ"]
        dQ = derivs["dQ"]
        dMy = derivs["dMy"]
        dMz = derivs["dMz"]
        dMb = derivs["dMb"]

        dT_dpresweep = dT["dpresweep"]
        dY_dpresweep = dY["dpresweep"]
        dZ_dpresweep = dZ["dpresweep"]
        dQ_dpresweep = dQ["dpresweep"]
        dMy_dpresweep = dMy["dpresweep"]
        dMz_dpresweep = dMz["dpresweep"]
        dMb_dpresweep = dMb["dpresweep"]
        dP_dpresweep = dP["dpresweep"]

        dT_dpresweep_fd = np.zeros((self.npts, self.n))
        dY_dpresweep_fd = np.zeros((self.npts, self.n))
        dZ_dpresweep_fd = np.zeros((self.npts, self.n))
        dQ_dpresweep_fd = np.zeros((self.npts, self.n))
        dMy_dpresweep_fd = np.zeros((self.npts, self.n))
        dMz_dpresweep_fd = np.zeros((self.npts, self.n))
        dMb_dpresweep_fd = np.zeros((self.npts, self.n))
        dP_dpresweep_fd = np.zeros((self.npts, self.n))
        for i in range(self.n):
            ps = np.array(presweep)
            delta = 1e-6 * ps[i]
            ps[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
                presweep=ps,
                presweepTip=presweepTip,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
            Pd = outputs["P"]
            Td = outputs["T"]
            Yd = outputs["Y"]
            Zd = outputs["Z"]
            Qd = outputs["Q"]
            Myd = outputs["My"]
            Mzd = outputs["Mz"]
            Mbd = outputs["Mb"]

            dT_dpresweep_fd[:, i] = (Td - T) / delta
            dY_dpresweep_fd[:, i] = (Yd - Y) / delta
            dZ_dpresweep_fd[:, i] = (Zd - Z) / delta
            dQ_dpresweep_fd[:, i] = (Qd - Q) / delta
            dMy_dpresweep_fd[:, i] = (Myd - My) / delta
            dMz_dpresweep_fd[:, i] = (Mzd - Mz) / delta
            dMb_dpresweep_fd[:, i] = (Mbd - Mb) / delta
            dP_dpresweep_fd[:, i] = (Pd - P) / delta

        np.testing.assert_allclose(dT_dpresweep_fd, dT_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dY_dpresweep_fd, dY_dpresweep, rtol=4e-3, atol=1e-8)
        np.testing.assert_allclose(dZ_dpresweep_fd, dZ_dpresweep, rtol=4e-3, atol=1e-8)
        np.testing.assert_allclose(dQ_dpresweep_fd, dQ_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dMy_dpresweep_fd, dMy_dpresweep, rtol=2e-3, atol=1e-8)
        np.testing.assert_allclose(dMz_dpresweep_fd, dMz_dpresweep, rtol=2e-3, atol=1e-8)
        np.testing.assert_allclose(dMb_dpresweep_fd, dMb_dpresweep, rtol=4e-4, atol=1e-8)
        np.testing.assert_allclose(dP_dpresweep_fd, dP_dpresweep, rtol=3e-4, atol=1e-8)

    def test_dpresweep3(self):

        presweep = np.linspace(1, 10, self.n)
        presweepTip = presweep[-1]
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            presweep=presweep,
            presweepTip=presweepTip,
        )
        presweep = rotor.presweep.copy()

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CP = outputs["CP"]
        CT = outputs["CT"]
        CY = outputs["CY"]
        CZ = outputs["CZ"]
        CQ = outputs["CQ"]
        CMy = outputs["CMy"]
        CMz = outputs["CMz"]
        CMb = outputs["CMb"]
        dCP = derivs["dCP"]
        dCT = derivs["dCT"]
        dCY = derivs["dCY"]
        dCZ = derivs["dCZ"]
        dCQ = derivs["dCQ"]
        dCMy = derivs["dCMy"]
        dCMz = derivs["dCMz"]
        dCMb = derivs["dCMb"]

        dCT_dpresweep = dCT["dpresweep"]
        dCY_dpresweep = dCY["dpresweep"]
        dCZ_dpresweep = dCZ["dpresweep"]
        dCQ_dpresweep = dCQ["dpresweep"]
        dCMy_dpresweep = dCMy["dpresweep"]
        dCMz_dpresweep = dCMz["dpresweep"]
        dCMb_dpresweep = dCMb["dpresweep"]
        dCP_dpresweep = dCP["dpresweep"]

        dCT_dpresweep_fd = np.zeros((self.npts, self.n))
        dCY_dpresweep_fd = np.zeros((self.npts, self.n))
        dCZ_dpresweep_fd = np.zeros((self.npts, self.n))
        dCQ_dpresweep_fd = np.zeros((self.npts, self.n))
        dCMy_dpresweep_fd = np.zeros((self.npts, self.n))
        dCMz_dpresweep_fd = np.zeros((self.npts, self.n))
        dCMb_dpresweep_fd = np.zeros((self.npts, self.n))
        dCP_dpresweep_fd = np.zeros((self.npts, self.n))
        for i in range(self.n):
            ps = np.array(presweep)
            delta = 1e-6 * ps[i]
            ps[i] += delta

            rotor = CCBlade(
                self.r,
                self.chord,
                self.theta,
                self.af,
                self.Rhub,
                self.Rtip,
                self.B,
                self.rho,
                self.mu,
                precone,
                self.tilt,
                self.yaw,
                self.shearExp,
                self.hubHt,
                self.nSector,
                derivatives=False,
                presweep=ps,
                presweepTip=presweepTip,
            )

            outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
            CPd = outputs["CP"]
            CTd = outputs["CT"]
            CYd = outputs["CY"]
            CZd = outputs["CZ"]
            CQd = outputs["CQ"]
            CMyd = outputs["CMy"]
            CMzd = outputs["CMz"]
            CMbd = outputs["CMb"]

            dCT_dpresweep_fd[:, i] = (CTd - CT) / delta
            dCY_dpresweep_fd[:, i] = (CYd - CY) / delta
            dCZ_dpresweep_fd[:, i] = (CZd - CZ) / delta
            dCQ_dpresweep_fd[:, i] = (CQd - CQ) / delta
            dCMy_dpresweep_fd[:, i] = (CMyd - CMy) / delta
            dCMz_dpresweep_fd[:, i] = (CMzd - CMz) / delta
            dCMb_dpresweep_fd[:, i] = (CMbd - CMb) / delta
            dCP_dpresweep_fd[:, i] = (CPd - CP) / delta

        np.testing.assert_allclose(dCT_dpresweep_fd, dCT_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCY_dpresweep_fd, dCY_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCZ_dpresweep_fd, dCZ_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCQ_dpresweep_fd, dCQ_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMy_dpresweep_fd, dCMy_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMz_dpresweep_fd, dCMz_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCMb_dpresweep_fd, dCMb_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dpresweep_fd, dCP_dpresweep, rtol=3e-4, atol=1e-8)

    def test_dprecurveTip1(self):

        precurve = np.linspace(1, 10, self.n)
        precurveTip = precurve[-1]
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            precurve=precurve,
            precurveTip=precurveTip,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Np = loads["Np"]
        Tp = loads["Tp"]

        dNp_dprecurveTip_fd = np.zeros((self.n, 1))
        dTp_dprecurveTip_fd = np.zeros((self.n, 1))

        pct = float(precurveTip)
        delta = 1e-6 * pct
        pct += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
            precurve=rotor.precurve,
            precurveTip=pct,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]
        dNp_dprecurveTip_fd[:, 0] = (Npd - Np) / delta
        dTp_dprecurveTip_fd[:, 0] = (Tpd - Tp) / delta

        np.testing.assert_allclose(dNp_dprecurveTip_fd, 0.0, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dprecurveTip_fd, 0.0, rtol=1e-4, atol=1e-8)

    def test_dprecurveTip2(self):

        precurve = np.linspace(1, 10, self.n)
        precurveTip = precurve[-1]
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            precurve=precurve,
            precurveTip=precurveTip,
        )

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        P = outputs["P"]
        T = outputs["T"]
        Y = outputs["Y"]
        Z = outputs["Z"]
        Q = outputs["Q"]
        My = outputs["My"]
        Mz = outputs["Mz"]
        Mb = outputs["Mb"]
        dP = derivs["dP"]
        dT = derivs["dT"]
        dY = derivs["dY"]
        dZ = derivs["dZ"]
        dQ = derivs["dQ"]
        dMy = derivs["dMy"]
        dMz = derivs["dMz"]
        dMb = derivs["dMb"]

        dT_dprecurveTip = dT["dprecurveTip"]
        dY_dprecurveTip = dY["dprecurveTip"]
        dZ_dprecurveTip = dZ["dprecurveTip"]
        dQ_dprecurveTip = dQ["dprecurveTip"]
        dMy_dprecurveTip = dMy["dprecurveTip"]
        dMz_dprecurveTip = dMz["dprecurveTip"]
        dMb_dprecurveTip = dMb["dprecurveTip"]
        dP_dprecurveTip = dP["dprecurveTip"]

        dT_dprecurveTip_fd = np.zeros((self.npts, 1))
        dY_dprecurveTip_fd = np.zeros((self.npts, 1))
        dZ_dprecurveTip_fd = np.zeros((self.npts, 1))
        dQ_dprecurveTip_fd = np.zeros((self.npts, 1))
        dMy_dprecurveTip_fd = np.zeros((self.npts, 1))
        dMz_dprecurveTip_fd = np.zeros((self.npts, 1))
        dMb_dprecurveTip_fd = np.zeros((self.npts, 1))
        dP_dprecurveTip_fd = np.zeros((self.npts, 1))

        pct = float(precurveTip)
        delta = 1e-6 * pct
        pct += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
            precurve=rotor.precurve,
            precurveTip=pct,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dprecurveTip_fd[:, 0] = (Td - T) / delta
        dY_dprecurveTip_fd[:, 0] = (Yd - Y) / delta
        dZ_dprecurveTip_fd[:, 0] = (Zd - Z) / delta
        dQ_dprecurveTip_fd[:, 0] = (Qd - Q) / delta
        dMy_dprecurveTip_fd[:, 0] = (Myd - My) / delta
        dMz_dprecurveTip_fd[:, 0] = (Mzd - Mz) / delta
        dMb_dprecurveTip_fd[:, 0] = (Mbd - Mb) / delta
        dP_dprecurveTip_fd[:, 0] = (Pd - P) / delta
        np.testing.assert_allclose(dT_dprecurveTip_fd, dT_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dY_dprecurveTip_fd, dY_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dZ_dprecurveTip_fd, dZ_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dQ_dprecurveTip_fd, dQ_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dMy_dprecurveTip_fd, dMy_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dMz_dprecurveTip_fd, dMz_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dMb_dprecurveTip_fd, dMb_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dP_dprecurveTip_fd, dP_dprecurveTip, rtol=1e-4, atol=1e-8)

    def test_dprecurveTip3(self):

        precurve = np.linspace(1, 10, self.n)
        precurveTip = precurve[-1]
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            precurve=precurve,
            precurveTip=precurveTip,
        )

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CP = outputs["CP"]
        CT = outputs["CT"]
        CY = outputs["CY"]
        CZ = outputs["CZ"]
        CQ = outputs["CQ"]
        CMy = outputs["CMy"]
        CMz = outputs["CMz"]
        CMb = outputs["CMb"]
        dCP = derivs["dCP"]
        dCT = derivs["dCT"]
        dCY = derivs["dCY"]
        dCZ = derivs["dCZ"]
        dCQ = derivs["dCQ"]
        dCMy = derivs["dCMy"]
        dCMz = derivs["dCMz"]
        dCMb = derivs["dCMb"]

        dCT_dprecurveTip = dCT["dprecurveTip"]
        dCY_dprecurveTip = dCY["dprecurveTip"]
        dCZ_dprecurveTip = dCZ["dprecurveTip"]
        dCQ_dprecurveTip = dCQ["dprecurveTip"]
        dCMy_dprecurveTip = dCMy["dprecurveTip"]
        dCMz_dprecurveTip = dCMz["dprecurveTip"]
        dCMb_dprecurveTip = dCMb["dprecurveTip"]
        dCP_dprecurveTip = dCP["dprecurveTip"]

        dCT_dprecurveTip_fd = np.zeros((self.npts, 1))
        dCY_dprecurveTip_fd = np.zeros((self.npts, 1))
        dCZ_dprecurveTip_fd = np.zeros((self.npts, 1))
        dCQ_dprecurveTip_fd = np.zeros((self.npts, 1))
        dCMy_dprecurveTip_fd = np.zeros((self.npts, 1))
        dCMz_dprecurveTip_fd = np.zeros((self.npts, 1))
        dCMb_dprecurveTip_fd = np.zeros((self.npts, 1))
        dCP_dprecurveTip_fd = np.zeros((self.npts, 1))

        pct = float(precurveTip)
        delta = 1e-6 * pct
        pct += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
            precurve=rotor.precurve,
            precurveTip=pct,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dprecurveTip_fd[:, 0] = (CTd - CT) / delta
        dCY_dprecurveTip_fd[:, 0] = (CYd - CY) / delta
        dCZ_dprecurveTip_fd[:, 0] = (CZd - CZ) / delta
        dCQ_dprecurveTip_fd[:, 0] = (CQd - CQ) / delta
        dCMy_dprecurveTip_fd[:, 0] = (CMyd - CMy) / delta
        dCMz_dprecurveTip_fd[:, 0] = (CMzd - CMz) / delta
        dCMb_dprecurveTip_fd[:, 0] = (CMbd - CMb) / delta
        dCP_dprecurveTip_fd[:, 0] = (CPd - CP) / delta

        np.testing.assert_allclose(dCT_dprecurveTip_fd, dCT_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCY_dprecurveTip_fd, dCY_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCZ_dprecurveTip_fd, dCZ_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCQ_dprecurveTip_fd, dCQ_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCMy_dprecurveTip_fd, dCMy_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCMz_dprecurveTip_fd, dCMz_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCMb_dprecurveTip_fd, dCMb_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dprecurveTip_fd, dCP_dprecurveTip, rtol=1e-4, atol=1e-8)

    def test_dpresweepTip1(self):

        presweep = np.linspace(1, 10, self.n)
        presweepTip = presweep[-1]
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            presweep=presweep,
            presweepTip=presweepTip,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Np = loads["Np"]
        Tp = loads["Tp"]

        dNp_dpresweepTip_fd = np.zeros((self.n, 1))
        dTp_dpresweepTip_fd = np.zeros((self.n, 1))

        pst = float(presweepTip)
        delta = 1e-6 * pst
        pst += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
            presweep=rotor.presweep,
            presweepTip=pst,
        )

        loads, _ = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
        Npd = loads["Np"]
        Tpd = loads["Tp"]
        dNp_dpresweepTip_fd[:, 0] = (Npd - Np) / delta
        dTp_dpresweepTip_fd[:, 0] = (Tpd - Tp) / delta

        np.testing.assert_allclose(dNp_dpresweepTip_fd, 0.0, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dpresweepTip_fd, 0.0, rtol=1e-4, atol=1e-8)

    def test_dpresweepTip2(self):

        presweep = np.linspace(1, 10, self.n)
        presweepTip = presweep[-1]
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            presweep=presweep,
            presweepTip=presweepTip,
        )

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        P = outputs["P"]
        T = outputs["T"]
        Y = outputs["Y"]
        Z = outputs["Z"]
        Q = outputs["Q"]
        My = outputs["My"]
        Mz = outputs["Mz"]
        Mb = outputs["Mb"]
        dP = derivs["dP"]
        dT = derivs["dT"]
        dY = derivs["dY"]
        dZ = derivs["dZ"]
        dQ = derivs["dQ"]
        dMy = derivs["dMy"]
        dMz = derivs["dMz"]
        dMb = derivs["dMb"]

        dT_dpresweepTip = dT["dpresweepTip"]
        dY_dpresweepTip = dY["dpresweepTip"]
        dZ_dpresweepTip = dZ["dpresweepTip"]
        dQ_dpresweepTip = dQ["dpresweepTip"]
        dMy_dpresweepTip = dMy["dpresweepTip"]
        dMz_dpresweepTip = dMz["dpresweepTip"]
        dMb_dpresweepTip = dMb["dpresweepTip"]
        dP_dpresweepTip = dP["dpresweepTip"]

        dT_dpresweepTip_fd = np.zeros((self.npts, 1))
        dY_dpresweepTip_fd = np.zeros((self.npts, 1))
        dZ_dpresweepTip_fd = np.zeros((self.npts, 1))
        dQ_dpresweepTip_fd = np.zeros((self.npts, 1))
        dMy_dpresweepTip_fd = np.zeros((self.npts, 1))
        dMz_dpresweepTip_fd = np.zeros((self.npts, 1))
        dMb_dpresweepTip_fd = np.zeros((self.npts, 1))
        dP_dpresweepTip_fd = np.zeros((self.npts, 1))

        pst = float(presweepTip)
        delta = 1e-6 * pst
        pst += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
            presweep=rotor.presweep,
            presweepTip=pst,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=False)
        Pd = outputs["P"]
        Td = outputs["T"]
        Yd = outputs["Y"]
        Zd = outputs["Z"]
        Qd = outputs["Q"]
        Myd = outputs["My"]
        Mzd = outputs["Mz"]
        Mbd = outputs["Mb"]

        dT_dpresweepTip_fd[:, 0] = (Td - T) / delta
        dY_dpresweepTip_fd[:, 0] = (Yd - Y) / delta
        dZ_dpresweepTip_fd[:, 0] = (Zd - Z) / delta
        dQ_dpresweepTip_fd[:, 0] = (Qd - Q) / delta
        dMy_dpresweepTip_fd[:, 0] = (Myd - My) / delta
        dMz_dpresweepTip_fd[:, 0] = (Mzd - Mz) / delta
        dMb_dpresweepTip_fd[:, 0] = (Mbd - Mb) / delta
        dP_dpresweepTip_fd[:, 0] = (Pd - P) / delta

        np.testing.assert_allclose(dT_dpresweepTip_fd, dT_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dY_dpresweepTip_fd, dY_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dZ_dpresweepTip_fd, dZ_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dQ_dpresweepTip_fd, dQ_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dMy_dpresweepTip_fd, dMy_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dMz_dpresweepTip_fd, dMz_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dMb_dpresweepTip_fd, dMb_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dP_dpresweepTip_fd, dP_dpresweepTip, rtol=1e-4, atol=1e-8)

    def test_dpresweepTip3(self):

        presweep = np.linspace(1, 10, self.n)
        presweepTip = presweep[-1]
        precone = 0.0
        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=True,
            presweep=presweep,
            presweepTip=presweepTip,
        )

        outputs, derivs = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CP = outputs["CP"]
        CT = outputs["CT"]
        CY = outputs["CY"]
        CZ = outputs["CZ"]
        CQ = outputs["CQ"]
        CMy = outputs["CMy"]
        CMz = outputs["CMz"]
        CMb = outputs["CMb"]
        dCP = derivs["dCP"]
        dCT = derivs["dCT"]
        dCY = derivs["dCY"]
        dCZ = derivs["dCZ"]
        dCQ = derivs["dCQ"]
        dCMy = derivs["dCMy"]
        dCMz = derivs["dCMz"]
        dCMb = derivs["dCMb"]

        dCT_dpresweepTip = dCT["dpresweepTip"]
        dCY_dpresweepTip = dCY["dpresweepTip"]
        dCZ_dpresweepTip = dCZ["dpresweepTip"]
        dCQ_dpresweepTip = dCQ["dpresweepTip"]
        dCMy_dpresweepTip = dCMy["dpresweepTip"]
        dCMz_dpresweepTip = dCMz["dpresweepTip"]
        dCMb_dpresweepTip = dCMb["dpresweepTip"]
        dCP_dpresweepTip = dCP["dpresweepTip"]

        dCT_dpresweepTip_fd = np.zeros((self.npts, 1))
        dCY_dpresweepTip_fd = np.zeros((self.npts, 1))
        dCZ_dpresweepTip_fd = np.zeros((self.npts, 1))
        dCQ_dpresweepTip_fd = np.zeros((self.npts, 1))
        dCMy_dpresweepTip_fd = np.zeros((self.npts, 1))
        dCMz_dpresweepTip_fd = np.zeros((self.npts, 1))
        dCMb_dpresweepTip_fd = np.zeros((self.npts, 1))
        dCP_dpresweepTip_fd = np.zeros((self.npts, 1))

        pst = float(presweepTip)
        delta = 1e-6 * pst
        pst += delta

        rotor = CCBlade(
            self.r,
            self.chord,
            self.theta,
            self.af,
            self.Rhub,
            self.Rtip,
            self.B,
            self.rho,
            self.mu,
            precone,
            self.tilt,
            self.yaw,
            self.shearExp,
            self.hubHt,
            self.nSector,
            derivatives=False,
            presweep=rotor.presweep,
            presweepTip=pst,
        )

        outputs, _ = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficients=True)
        CPd = outputs["CP"]
        CTd = outputs["CT"]
        CYd = outputs["CY"]
        CZd = outputs["CZ"]
        CQd = outputs["CQ"]
        CMyd = outputs["CMy"]
        CMzd = outputs["CMz"]
        CMbd = outputs["CMb"]

        dCT_dpresweepTip_fd[:, 0] = (CTd - CT) / delta
        dCY_dpresweepTip_fd[:, 0] = (CYd - CY) / delta
        dCZ_dpresweepTip_fd[:, 0] = (CZd - CZ) / delta
        dCQ_dpresweepTip_fd[:, 0] = (CQd - CQ) / delta
        dCMy_dpresweepTip_fd[:, 0] = (CMyd - CMy) / delta
        dCMz_dpresweepTip_fd[:, 0] = (CMzd - CMz) / delta
        dCMb_dpresweepTip_fd[:, 0] = (CMbd - CMb) / delta
        dCP_dpresweepTip_fd[:, 0] = (CPd - CP) / delta

        np.testing.assert_allclose(dCT_dpresweepTip_fd, dCT_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCY_dpresweepTip_fd, dCY_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCZ_dpresweepTip_fd, dCZ_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCQ_dpresweepTip_fd, dCQ_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCMy_dpresweepTip_fd, dCMy_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCMz_dpresweepTip_fd, dCMz_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCMb_dpresweepTip_fd, dCMb_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dpresweepTip_fd, dCP_dpresweepTip, rtol=1e-4, atol=1e-8)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestGradients))
    suite.addTest(unittest.makeSuite(TestGradientsNotRotating))
    suite.addTest(unittest.makeSuite(TestGradientsFreestreamArray))
    suite.addTest(unittest.makeSuite(TestGradients_RHub_Tip))
    return suite


if __name__ == "__main__":
    unittest.TextTestRunner().run(suite())
