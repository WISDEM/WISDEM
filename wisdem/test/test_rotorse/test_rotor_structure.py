import os
import copy
import time
import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt
import wisdem.rotorse.rotor_structure as rs
from wisdem.commonse import gravity

ARCHIVE1 = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "nrel5mw_test.npz"
ARCHIVE2 = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "nrel5mw_test2.npz"


class TestRS(unittest.TestCase):
    def testBladeCurvature(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        npts = 101
        myzero = np.zeros(npts)
        myone = np.ones(npts)
        options = {}
        options["WISDEM"] = {}
        options["WISDEM"]["RotorSE"] = {}
        options["WISDEM"]["RotorSE"]["n_span"] = npts

        myobj = rs.BladeCurvature(modeling_options=options)

        # Straight blade: Z is 'r'
        inputs["r"] = np.linspace(0, 100, npts)
        inputs["precurve"] = myzero
        inputs["presweep"] = myzero
        inputs["precone"] = 0.0
        myobj.compute(inputs, outputs)
        npt.assert_equal(outputs["3d_curv"], myzero)
        npt.assert_equal(outputs["x_az"], myzero)
        npt.assert_equal(outputs["y_az"], myzero)
        npt.assert_equal(outputs["z_az"], inputs["r"])

        # Some coning: Z is 'r'
        inputs["precone"] = 3.0
        myobj.compute(inputs, outputs)
        npt.assert_equal(outputs["3d_curv"], 3 * myone)
        npt.assert_equal(outputs["x_az"], myzero)
        npt.assert_equal(outputs["y_az"], myzero)
        npt.assert_equal(outputs["z_az"], inputs["r"])

        # Some curve: X is 'flap'
        inputs["precurve"] = np.linspace(0, 1, npts)
        inputs["precone"] = 0.0
        myobj.compute(inputs, outputs)
        cone = -np.rad2deg(np.arctan(inputs["precurve"] / (inputs["r"] + 1e-20)))
        cone[0] = cone[1]
        npt.assert_almost_equal(outputs["3d_curv"], cone)
        npt.assert_equal(outputs["x_az"], inputs["precurve"])
        npt.assert_equal(outputs["y_az"], myzero)
        npt.assert_equal(outputs["z_az"], inputs["r"])

        # Some curve: Y is 'edge'
        inputs["precurve"] = myzero
        inputs["presweep"] = np.linspace(0, 1, npts)
        myobj.compute(inputs, outputs)
        npt.assert_almost_equal(outputs["3d_curv"], myzero)
        npt.assert_equal(outputs["x_az"], myzero)
        npt.assert_equal(outputs["y_az"], inputs["presweep"])
        npt.assert_equal(outputs["z_az"], inputs["r"])

        # Some curve and sweep
        inputs["precurve"] = np.linspace(0, 2, npts)
        inputs["presweep"] = np.linspace(0, 1, npts)
        inputs["precone"] = 0.0
        myobj.compute(inputs, outputs)
        cone = -np.rad2deg(np.arctan(inputs["precurve"] / (inputs["r"] + 1e-20)))
        cone[0] = cone[1]
        npt.assert_almost_equal(outputs["3d_curv"], cone)
        npt.assert_equal(outputs["x_az"], inputs["precurve"])
        npt.assert_equal(outputs["y_az"], inputs["presweep"])
        npt.assert_equal(outputs["z_az"], inputs["r"])

    def testTotalLoads(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        npts = 101
        myzero = np.zeros(npts)
        myone = np.ones(npts)
        options = {}
        options["WISDEM"] = {}
        options["WISDEM"]["RotorSE"] = {}
        options["WISDEM"]["RotorSE"]["n_span"] = npts

        myobj = rs.TotalLoads(modeling_options=options)

        # Pass through
        inputs["r"] = np.linspace(0, 100, npts)
        inputs["aeroloads_Px"] = 10.0 * myone
        inputs["aeroloads_Py"] = 5.0 * myone
        inputs["aeroloads_Pz"] = 1.0 * myone
        inputs["aeroloads_Omega"] = 0.0
        inputs["aeroloads_pitch"] = 0.0
        inputs["aeroloads_azimuth"] = 0.0
        inputs["theta"] = myzero
        inputs["tilt"] = 0.0
        inputs["3d_curv"] = myzero
        inputs["z_az"] = inputs["r"]
        inputs["rhoA"] = myzero  # 20.0*myone
        inputs["dynamicFactor"] = 1.0
        myobj.compute(inputs, outputs)
        npt.assert_equal(outputs["Px_af"], inputs["aeroloads_Px"])
        npt.assert_equal(outputs["Py_af"], inputs["aeroloads_Py"])
        npt.assert_equal(outputs["Pz_af"], inputs["aeroloads_Pz"])

        # With theta and pitch
        inputs["aeroloads_pitch"] = 20.0
        inputs["theta"] = 70 * myone
        myobj.compute(inputs, outputs)
        npt.assert_almost_equal(outputs["Px_af"], -inputs["aeroloads_Py"])
        npt.assert_almost_equal(outputs["Py_af"], inputs["aeroloads_Px"])
        npt.assert_equal(outputs["Pz_af"], inputs["aeroloads_Pz"])

        # With gravity
        inputs["aeroloads_pitch"] = 0.0
        inputs["theta"] = myzero
        inputs["rhoA"] = 20.0 * myone
        myobj.compute(inputs, outputs)
        npt.assert_almost_equal(outputs["Px_af"], inputs["aeroloads_Px"])
        npt.assert_almost_equal(outputs["Py_af"], inputs["aeroloads_Py"])
        npt.assert_equal(outputs["Pz_af"], inputs["aeroloads_Pz"] - 20 * gravity)

        # With gravity, az 180
        inputs["aeroloads_azimuth"] = 180.0
        myobj.compute(inputs, outputs)
        npt.assert_almost_equal(outputs["Px_af"], inputs["aeroloads_Px"])
        npt.assert_almost_equal(outputs["Py_af"], inputs["aeroloads_Py"])
        npt.assert_equal(outputs["Pz_af"], inputs["aeroloads_Pz"] + 20 * gravity)

        # With gravity, az 90
        inputs["aeroloads_azimuth"] = 90.0
        myobj.compute(inputs, outputs)
        npt.assert_almost_equal(outputs["Px_af"], inputs["aeroloads_Px"])
        npt.assert_almost_equal(outputs["Py_af"], inputs["aeroloads_Py"] - 20 * gravity)
        npt.assert_almost_equal(outputs["Pz_af"], inputs["aeroloads_Pz"])

        # With centrifical
        inputs["aeroloads_Omega"] = 5.0
        myobj.compute(inputs, outputs)
        npt.assert_almost_equal(outputs["Px_af"], inputs["aeroloads_Px"])
        npt.assert_almost_equal(outputs["Py_af"], inputs["aeroloads_Py"] - 20 * gravity)
        npt.assert_almost_equal(outputs["Pz_af"], inputs["aeroloads_Pz"] + 20 * inputs["r"] * (5 * 2 * np.pi / 60) ** 2)

    def testRunFrame3DD(self):
        inputs = {}
        outputs = {}

        nrel5mw = np.load(ARCHIVE1)
        for k in nrel5mw.files:
            inputs[k] = nrel5mw[k]

        npts = len(inputs["r"])
        nfreq = 10
        options = {}
        options["WISDEM"] = {}
        options["WISDEM"]["RotorSE"] = {}
        options["WISDEM"]["RotorSE"]["n_span"] = npts
        options["WISDEM"]["RotorSE"]["n_freq"] = nfreq

        myobj = rs.RunFrame3DD(modeling_options=options)
        myobj.n_span = npts
        myobj.n_freq = nfreq
        myobj.compute(inputs, outputs)

        # Truth values
        dx = [-0.00000000e+00,  1.55573988e-03,  6.46103793e-03,  1.51693803e-02,
        2.86776698e-02,  4.91497876e-02,  7.88315735e-02,  1.18931921e-01,
        1.70044378e-01,  2.32768629e-01,  3.07699928e-01,  3.95422156e-01,
        4.96788836e-01,  6.13113403e-01,  7.45976232e-01,  8.96909866e-01,
        1.06730796e+00,  1.25845536e+00,  1.47158164e+00,  1.70803798e+00,
        1.96934465e+00,  2.25637035e+00,  2.56856312e+00,  2.90438964e+00,
        3.26181931e+00,  3.63829955e+00,  4.03072756e+00,  4.43554805e+00,
        4.84940322e+00,  5.26857338e+00 ]
        dy = [ 0.00000000e+00, -4.01080002e-04, -2.02108580e-03, -6.08940835e-03,
       -1.48330522e-02, -3.09115592e-02, -5.58845427e-02, -8.98665150e-02,
       -1.32496165e-01, -1.83537761e-01, -2.42794736e-01, -3.10036945e-01,
       -3.85096172e-01, -4.67890944e-01, -5.58322353e-01, -6.56184779e-01,
       -7.61171094e-01, -8.72892227e-01, -9.90884310e-01, -1.11464019e+00,
       -1.24364122e+00, -1.37731846e+00, -1.51501797e+00, -1.65603044e+00,
       -1.79963753e+00, -1.94514961e+00, -2.09193787e+00, -2.23947439e+00,
       -2.38739628e+00, -2.53547341e+00 ]
        dz = [-0.00000000e+00, -5.76635508e-05, -1.38122944e-04, -2.25226392e-04,
       -3.15407862e-04, -4.09010260e-04, -5.02353599e-04, -5.93159024e-04,
       -6.81185450e-04, -7.66286578e-04, -8.48065266e-04, -9.26577316e-04,
       -1.00265190e-03, -1.07674611e-03, -1.14855163e-03, -1.21772038e-03,
       -1.28385666e-03, -1.34655739e-03, -1.40542054e-03, -1.46013988e-03,
       -1.51151134e-03, -1.56061633e-03, -1.60752808e-03, -1.65179039e-03,
       -1.69289294e-03, -1.73017476e-03, -1.76270246e-03, -1.78909031e-03,
       -1.80823696e-03, -1.81810390e-03  ]

        freqs = [0.80747665,  1.21045433,  2.92756539,  4.09616024,  6.75168715,
        8.84500175,  9.41165308, 12.36777335, 14.73146625, 16.32802481 ]

        npt.assert_almost_equal(outputs["dx"], dx, decimal=2)
        npt.assert_almost_equal(outputs["dy"], dy, decimal=2)
        npt.assert_almost_equal(outputs['dz'], dz, decimal=3) # Very small numbers, so no precision
        npt.assert_almost_equal(outputs["freqs"], freqs, decimal=0)

    def testComputeStrains(self):
        inputs = {}
        outputs = {}

        nrel5mw = np.load(ARCHIVE2)
        for k in nrel5mw.files:
            inputs[k] = nrel5mw[k]

        npts = len(inputs["EA"])
        options = {}
        options["WISDEM"] = {}
        options["WISDEM"]["RotorSE"] = {}
        options["WISDEM"]["RotorSE"]["n_span"] = npts

        myobj = rs.ComputeStrains(modeling_options=options)
        myobj.n_span = npts
        myobj.compute(inputs, outputs)

        # Truth values
        strainU_spar = [-9.94653310e-04, -1.42178217e-03, -1.30153437e-03, -1.35180421e-03,
       -1.77060677e-03, -2.06609907e-03, -2.10919690e-03, -2.07825547e-03,
       -2.06864760e-03, -2.03184245e-03, -1.98342575e-03, -1.95081457e-03,
       -1.96332549e-03, -1.98052188e-03, -1.97609884e-03, -1.95515284e-03,
       -1.92032128e-03, -1.87140247e-03, -1.81491993e-03, -1.75429432e-03,
       -1.68330358e-03, -1.54346157e-03, -1.38325181e-03, -1.20752329e-03,
       -1.01544367e-03, -8.07250975e-04, -5.89182576e-04, -3.72428306e-04,
       -2.05881929e-04,  2.75098215e-16]
        strainL_spar = [ 9.49750186e-04,  1.34436442e-03,  1.22185577e-03,  1.18864382e-03,
        1.47303895e-03,  1.70124772e-03,  1.74787600e-03,  1.73045151e-03,
        1.72996703e-03,  1.71012964e-03,  1.67918224e-03,  1.66941270e-03,
        1.70984558e-03,  1.75390385e-03,  1.77220502e-03,  1.76577512e-03,
        1.73979307e-03,  1.69635625e-03,  1.64556934e-03,  1.59589641e-03,
        1.53374892e-03,  1.40509331e-03,  1.25655876e-03,  1.09256077e-03,
        9.13274257e-04,  7.19483906e-04,  5.17945122e-04,  3.20504412e-04,
        1.71423112e-04, -8.72183853e-17]
        strainU_te = [-8.60897040e-05, -1.07142218e-05, -6.29147673e-04, -1.61895232e-03,
       -2.18677395e-03, -2.37507760e-03, -2.34664096e-03, -2.16168417e-03,
       -2.05243040e-03, -1.94369084e-03, -1.81088622e-03, -1.64369183e-03,
       -1.43090003e-03, -1.20001883e-03, -1.01301046e-03, -8.36623715e-04,
       -6.62938230e-04, -4.92317707e-04, -2.93718533e-04, -1.19933975e-04,
        6.06752109e-06,  9.42173841e-06,  3.52819663e-05,  5.66530476e-05,
        6.76831140e-05,  7.75385459e-05,  6.58070132e-05,  4.60111024e-05,
        2.56786817e-05,  4.74181995e-16]
        strainL_te = [-8.60897040e-05, -1.07142218e-05, -6.29147673e-04, -1.61895232e-03,
       -2.18677395e-03, -2.37507760e-03, -2.34664096e-03, -2.16168417e-03,
       -2.05243040e-03, -1.94369084e-03, -1.81088622e-03, -1.64369183e-03,
       -1.43090003e-03, -1.20001883e-03, -1.01301046e-03, -8.36623715e-04,
       -6.62938230e-04, -4.92317707e-04, -2.93718533e-04, -1.19933975e-04,
        6.06752109e-06,  9.42173841e-06,  3.52819663e-05,  5.66530476e-05,
        6.76831140e-05,  7.75385459e-05,  6.58070132e-05,  4.60111024e-05,
        2.56786817e-05,  4.74181995e-16]

        npt.assert_almost_equal(outputs["strainU_spar"], strainU_spar, decimal=4)
        npt.assert_almost_equal(outputs["strainL_spar"], strainL_spar, decimal=4)
        npt.assert_almost_equal(outputs['strainU_te'], strainU_te, decimal=4)
        npt.assert_almost_equal(outputs["strainL_te"], strainL_te, decimal=4)


    def testConstraints(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}

        npts = 101
        myzero = np.zeros(npts)
        myone = np.ones(npts)
        options = {}
        options["WISDEM"] = {}
        options["WISDEM"]["RotorSE"] = {}
        options["WISDEM"]["RotorSE"]["n_span"] = npts
        options["WISDEM"]["RotorSE"]["n_freq"] = 6
        options["WISDEM"]["RotorSE"]["gamma_freq"] = 1.1

        myopt = {}
        myopt["design_variables"] = {}
        myopt["design_variables"]["blade"] = {}
        myopt["design_variables"]["blade"]["structure"] = {}
        myopt["design_variables"]["blade"]["spar_cap_ss"] = {}
        myopt["design_variables"]["blade"]["spar_cap_ps"] = {}
        myopt["design_variables"]["blade"]["spar_cap_ss"]["n_opt"] = 3
        myopt["design_variables"]["blade"]["spar_cap_ps"]["n_opt"] = 3

        myobj = rs.DesignConstraints(modeling_options=options, opt_options=myopt)

        # Straight blade: Z is 'r'
        inputs["strainU_spar"] = np.linspace(0.4, 0.6, npts)
        inputs["strainL_spar"] = 0.6 * myone
        inputs["min_strainU_spar"] = inputs["min_strainL_spar"] = 0.0
        inputs["max_strainU_spar"] = inputs["max_strainL_spar"] = 0.5
        inputs["s"] = np.linspace(0, 1, npts)
        inputs["s_opt_spar_cap_ss"] = inputs["s_opt_spar_cap_ps"] = np.array([0.0, 0.5, 1.0])
        inputs["rated_Omega"] = 10.0
        inputs["flap_mode_freqs"] = 0.6 * np.ones(3)
        inputs["edge_mode_freqs"] = 0.4 * np.ones(3)
        discrete_inputs["blade_number"] = 3

        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        npt.assert_equal(outputs["constr_max_strainU_spar"], np.array([0.8, 1.0, 1.2]))
        npt.assert_equal(outputs["constr_max_strainL_spar"], 1.2 * np.ones(3))
        npt.assert_almost_equal(outputs["constr_flap_f_margin"], 0.5 - 0.9 * 0.6)
        npt.assert_almost_equal(outputs["constr_edge_f_margin"], 1.1 * 0.4 - 0.5)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRS))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
