import unittest

import numpy as np
import numpy.testing as npt
from wisdem.pymap import pyMAP

epsilon = 1e-3  # finite difference epsilon

baseline2 = [
    "--------------- LINE DICTIONARY -------------------------",
    "LineType  Diam      MassDenInAir   EA        CB   CIntDamp  Ca    Cdn    Cdt",
    "(-)       (m)       (kg/m)        (N)        (-)   (Pa-s)   (-)   (-)    (-)",
    "steel     0.25       320.0     9800000000   1.0    -999.9 -999.9 -999.9 -999.9",
    "nylon     0.30       100.0     980000000    1.0    -999.9 -999.9 -999.9 -999.9",
    "--------------- NODE PROPERTIES -------------------------",
    "Node Type       X       Y       Z      M     B     FX    FY    FZ",
    "(-)  (-)       (m)     (m)     (m)    (kg)  (mˆ3)  (N)   (N)   (N)",
    "1    Fix     400        0     depth    0     0      #     #     #",
    "2    Connect #90       #0    #-80      0     0      0     0     0   ",
    "3    Vessel   20        20    -10      0     0      #     #     #",
    "4    Vessel   20       -20    -10      0     0      #     #     #",
    "--------------- LINE PROPERTIES -------------------------",
    "Line    LineType  UnstrLen  NodeAnch  NodeFair  Flags",
    "(-)      (-)       (m)       (-)       (-)       (-)",
    "1       steel      450        1         2    ",
    "2       nylon      90         2         3    ",
    "3       nylon      90         2         4    ",
    "--------------- SOLVER OPTIONS---------------------------",
    "Option",
    "(-)",
    " help",
    "outer_tol 1e-10",
    "repeat 120 ",
    "repeat 240",
]
nrel5mw_oc4 = [
    "---------------------- LINE DICTIONARY -----------------------------------------------------",
    "LineType  Diam     MassDenInAir    EA         CB   CIntDamp  Ca   Cdn    Cdt",
    "(-)       (m)      (kg/m)         (N)        (-)   (Pa-s)    (-)  (-)    (-)",
    "steel     0.0766  113.35         7.536e8     1.0    1.0E8    0.6 -1.0    0.05",
    "---------------------- NODE PROPERTIES -----------------------------------------------------",
    "Node      Type      X        Y         Z        M        V        FX       FY      FZ",
    "(-)       (-)      (m)      (m)       (m)      (kg)     (m^3)    (kN)     (kN)    (kN)",
    "1         Fix     418.8    725.383    -200     0        0        #        #       #",
    "2         Vessel   20.434   35.393     -14.0   0        0        #        #       #",
    "---------------------- LINE PROPERTIES -----------------------------------------------------",
    "Line     LineType  UnstrLen   NodeAnch  NodeFair  Flags",
    "(-)      (-)       (m)        (-)       (-)       (-)",
    "1        steel     835.35      1         2        plot ",
    "---------------------- SOLVER OPTIONS-----------------------------------------",
    "Option",
    "(-)",
    "repeat 240 120",
    "ref_position 0 0 0",
]


class NRELOC4(unittest.TestCase):
    def setUp(self):

        self.mymoor = pyMAP()

        self.mymoor.map_set_sea_depth(350)  # m
        self.mymoor.map_set_gravity(9.81)  # m/s^2
        self.mymoor.map_set_sea_density(1025.0)  # kg/m^3
        # self.mymoor.read_list_input(baseline2)
        self.mymoor.read_list_input(nrel5mw_oc4)
        self.mymoor.init()

    def tearDown(self):
        self.mymoor.end()

    def test_disp0(self):

        self.mymoor.displace_vessel(0, 0, 0, 0, 0, 0)
        self.mymoor.update_states(0.0, 0)

        K = self.mymoor.linear(epsilon)
        Ktruth0 = np.array(
            [
                [
                    70735.46783049824,
                    0.06861850852146745,
                    0.03164552617818117,
                    -1.8161400611243153,
                    -107993.21199813858,
                    -0.7199701001372887,
                ],
                [
                    0.3786282322835177,
                    70735.28922839311,
                    -0.029779504984617233,
                    107990.95722660399,
                    -3.232303075492382,
                    -2.2268171323667048,
                ],
                [
                    0.03450122312642634,
                    0.28001220198348165,
                    19074.7946856427,
                    0.22611861195036909,
                    -0.200674869120121,
                    4.706659653585632,
                ],
                [
                    -1.7506835283711553,
                    109682.20173611568,
                    -0.11558912228792906,
                    87420082.72956844,
                    -180.4521125741303,
                    -28.26136251166463,
                ],
                [
                    -109684.40142428153,
                    -2.8174332692287862,
                    -0.14091143384575844,
                    121.97450486943126,
                    87420104.18408969,
                    29.04599497653544,
                ],
                [
                    -0.8536792593076825,
                    -2.2268955498802825,
                    4.453814239241183,
                    -37.094427317380905,
                    34.32691376656294,
                    117254988.7859909,
                ],
            ]
        )
        npt.assert_almost_equal(K, Ktruth0, 4)

    def test_disp5(self):

        surge = 5.0  # 5 meter surge displacements
        self.mymoor.displace_vessel(surge, 0, 0, 0, 0, 0)
        self.mymoor.update_states(0.0, 0)

        K = self.mymoor.linear(epsilon)
        Ktruth5 = np.array(
            [
                [
                    86131.7058475397,
                    0.05506689194589853,
                    5360.3444378240965,
                    -2.209400015821757,
                    -203434.16918208823,
                    2.088261405191588,
                ],
                [
                    0.36263425135985017,
                    60871.7709575066,
                    -0.024427485186606646,
                    58904.85032479839,
                    -3.10470350086689,
                    -4306.699440307544,
                ],
                [
                    5355.589583865367,
                    0.24867214960977435,
                    19161.56132973265,
                    -0.07433195808698656,
                    -32426.0905967094,
                    4.806700007385643,
                ],
                [
                    -1.5074419206939638,
                    109682.19979056167,
                    -0.38653454976156354,
                    87420082.78226468,
                    -181.88006011769176,
                    -28.259312299080193,
                ],
                [
                    -109684.40141848987,
                    -2.817431522998959,
                    -0.1409068936482072,
                    121.97449927777052,
                    87420104.18408456,
                    29.045994982123375,
                ],
                [
                    -0.8536778041161597,
                    -2.226892755868903,
                    4.4538171496242285,
                    -37.09442823380232,
                    34.32691330090165,
                    117254988.78599086,
                ],
            ]
        )
        npt.assert_almost_equal(K, Ktruth5, 4)

        # We need to call update states after linearization to find the equilibrium
        self.mymoor.update_states(0.0, 0)

        line_number = 0
        H, V = self.mymoor.get_fairlead_force_2d(line_number)
        self.assertAlmostEqual(H, 804211.3987520998, 2)
        self.assertAlmostEqual(V, 598004.6901663609, 2)

        fx, fy, fz = self.mymoor.get_fairlead_force_3d(line_number)
        self.assertAlmostEqual(fx, -398302.72792638425, 2)
        self.assertAlmostEqual(fy, -698649.3475336606, 2)
        self.assertAlmostEqual(fz, 598004.6901663609, 2)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(NRELOC4))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
