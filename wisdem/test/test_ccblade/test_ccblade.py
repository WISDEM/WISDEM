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

import math
import unittest
from os import path

import numpy as np
from wisdem.ccblade.ccblade import CCBlade, CCAirfoil


class TestNREL5MW(unittest.TestCase):
    def setUp(self):

        # geometry
        Rhub = 1.5
        Rtip = 63.0

        r = np.array(
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
        chord = np.array(
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
        theta = np.array(
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
        B = 3  # number of blades

        # atmosphere
        rho = 1.225
        mu = 1.81206e-5

        afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
        basepath = path.join(path.dirname(path.realpath(__file__)), "../../../examples/_airfoil_files")

        # load all airfoils
        airfoil_types = [0] * 8
        airfoil_types[0] = afinit(path.join(basepath, "Cylinder1.dat"))
        airfoil_types[1] = afinit(path.join(basepath, "Cylinder2.dat"))
        airfoil_types[2] = afinit(path.join(basepath, "DU40_A17.dat"))
        airfoil_types[3] = afinit(path.join(basepath, "DU35_A17.dat"))
        airfoil_types[4] = afinit(path.join(basepath, "DU30_A17.dat"))
        airfoil_types[5] = afinit(path.join(basepath, "DU25_A17.dat"))
        airfoil_types[6] = afinit(path.join(basepath, "DU21_A17.dat"))
        airfoil_types[7] = afinit(path.join(basepath, "NACA64_A17.dat"))

        # place at appropriate radial stations
        af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

        af = [0] * len(r)
        for i in range(len(r)):
            af[i] = airfoil_types[af_idx[i]]

        tilt = -5.0
        precone = 2.5
        yaw = 0.0

        # create CCBlade object
        self.rotor = CCBlade(r, chord, theta, af, Rhub, Rtip, B, rho, mu, precone, tilt, yaw, shearExp=0.2, hubHt=90.0)

    def test_thrust_torque(self):

        Uinf = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        Omega = np.array(
            [
                6.972,
                7.183,
                7.506,
                7.942,
                8.469,
                9.156,
                10.296,
                11.431,
                11.890,
                12.100,
                12.100,
                12.100,
                12.100,
                12.100,
                12.100,
                12.100,
                12.100,
                12.100,
                12.100,
                12.100,
                12.100,
                12.100,
                12.100,
            ]
        )
        pitch = np.array(
            [
                0.000,
                0.000,
                0.000,
                0.000,
                0.000,
                0.000,
                0.000,
                0.000,
                0.000,
                3.823,
                6.602,
                8.668,
                10.450,
                12.055,
                13.536,
                14.920,
                16.226,
                17.473,
                18.699,
                19.941,
                21.177,
                22.347,
                23.469,
            ]
        )

        Pref = np.array(
            [
                42.9,
                188.2,
                427.9,
                781.3,
                1257.6,
                1876.2,
                2668.0,
                3653.0,
                4833.2,
                5296.6,
                5296.6,
                5296.6,
                5296.6,
                5296.6,
                5296.6,
                5296.6,
                5296.6,
                5296.7,
                5296.6,
                5296.7,
                5296.6,
                5296.6,
                5296.7,
            ]
        )
        Tref = np.array(
            [
                171.7,
                215.9,
                268.9,
                330.3,
                398.6,
                478.0,
                579.2,
                691.5,
                790.6,
                690.0,
                608.4,
                557.9,
                520.5,
                491.2,
                467.7,
                448.4,
                432.3,
                418.8,
                406.7,
                395.3,
                385.1,
                376.7,
                369.3,
            ]
        )
        Qref = np.array(
            [
                58.8,
                250.2,
                544.3,
                939.5,
                1418.1,
                1956.9,
                2474.5,
                3051.1,
                3881.3,
                4180.1,
                4180.1,
                4180.1,
                4180.1,
                4180.1,
                4180.1,
                4180.1,
                4180.1,
                4180.1,
                4180.1,
                4180.1,
                4180.1,
                4180.1,
                4180.1,
            ]
        )

        m_rotor = 110.0  # kg
        g = 9.81
        tilt = 5 * math.pi / 180.0
        Tref -= m_rotor * g * math.sin(tilt)  # remove weight of rotor that is included in reported results

        outputs, derivs = self.rotor.evaluate(Uinf, Omega, pitch)
        P, T, Q = [outputs[key] for key in ("P", "T", "Q")]

        # import matplotlib.pyplot as plt
        # plt.plot(Uinf, P/1e6)
        # plt.plot(Uinf, Pref/1e3)
        # plt.figure()
        # plt.plot(Uinf, T/1e6)
        # plt.plot(Uinf, Tref/1e3)
        # plt.show()

        idx = Uinf < 15
        np.testing.assert_allclose(Q[idx] / 1e6, Qref[idx] / 1e3, atol=0.15)
        np.testing.assert_allclose(P[idx] / 1e6, Pref[idx] / 1e3, atol=0.2)  # within 0.2 of 1MW
        np.testing.assert_allclose(T[idx] / 1e6, Tref[idx] / 1e3, atol=0.15)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestNREL5MW))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
