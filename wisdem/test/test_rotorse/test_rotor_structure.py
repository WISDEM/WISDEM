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
        dx = [
            -0.00000000e00,
            1.49215819e-03,
            6.18800345e-03,
            1.45003910e-02,
            2.73434922e-02,
            4.67234228e-02,
            7.47186854e-02,
            1.12432828e-01,
            1.60390656e-01,
            2.19118508e-01,
            2.89135305e-01,
            3.70946655e-01,
            4.65304000e-01,
            5.73377669e-01,
            6.96576803e-01,
            8.36259653e-01,
            9.93650895e-01,
            1.16987021e00,
            1.36598166e00,
            1.58315284e00,
            1.82269515e00,
            2.08532553e00,
            2.37048502e00,
            2.67674350e00,
            3.00223660e00,
            3.34464212e00,
            3.70115151e00,
            4.06855089e00,
            4.44377456e00,
            4.82335348e00,
        ]
        dy = [
            0.00000000e00,
            -3.60405636e-04,
            -1.82452740e-03,
            -5.52381033e-03,
            -1.35018161e-02,
            -2.81923557e-02,
            -5.10138481e-02,
            -8.20504020e-02,
            -1.20945435e-01,
            -1.67456836e-01,
            -2.21379110e-01,
            -2.82478136e-01,
            -3.50577674e-01,
            -4.25581185e-01,
            -5.07377170e-01,
            -5.95758732e-01,
            -6.90429602e-01,
            -7.91023050e-01,
            -8.97108761e-01,
            -1.00822123e00,
            -1.12388783e00,
            -1.24359430e00,
            -1.36675646e00,
            -1.49274889e00,
            -1.62094474e00,
            -1.75074652e00,
            -1.88161303e00,
            -2.01309315e00,
            -2.14487848e00,
            -2.27677580e00,
        ]
        dz = [
            -0.00000000e00,
            -5.76635508e-05,
            -1.38122944e-04,
            -2.25226392e-04,
            -3.15407862e-04,
            -4.09010260e-04,
            -5.02353599e-04,
            -5.93159024e-04,
            -6.81185450e-04,
            -7.66286578e-04,
            -8.48065266e-04,
            -9.26577316e-04,
            -1.00265190e-03,
            -1.07674611e-03,
            -1.14855163e-03,
            -1.21772038e-03,
            -1.28385666e-03,
            -1.34655739e-03,
            -1.40542054e-03,
            -1.46013988e-03,
            -1.51151134e-03,
            -1.56061633e-03,
            -1.60752808e-03,
            -1.65179039e-03,
            -1.69289294e-03,
            -1.73017476e-03,
            -1.76270246e-03,
            -1.78909031e-03,
            -1.80823696e-03,
            -1.81810390e-03,
        ]

        freqs = [
            0.84289055,
            1.23705125,
            2.96645445,
            4.12373582,
            6.78932171,
            8.85003324,
            9.44144293,
            12.40158703,
            14.73949144,
            16.36176423,
        ]

        npt.assert_almost_equal(outputs["dx"], dx, decimal=2)
        npt.assert_almost_equal(outputs["dy"], dy, decimal=2)
        npt.assert_almost_equal(outputs["dz"], dz, decimal=3)  # Very small numbers, so no precision
        npt.assert_almost_equal(outputs["freqs"], freqs, decimal=0)

    def testComputeStrains(self):
        inputs = {}
        outputs = {}

        nrel5mw = np.load(ARCHIVE1)
        for k in nrel5mw.files:
            inputs[k.replace("_strain_", "_")] = nrel5mw[k]

        nrel5mw = np.load(ARCHIVE2)
        for k in nrel5mw.files:
            inputs[k.replace("_strain_", "_")] = nrel5mw[k]

        npts = len(inputs["EA"])
        inputs["chord"] = np.ones(npts)

        options = {}
        options["WISDEM"] = {}
        options["WISDEM"]["RotorSE"] = {}
        options["WISDEM"]["RotorSE"]["n_span"] = npts

        myobj = rs.ComputeStrains(modeling_options=options)
        myobj.n_span = npts
        myobj.compute(inputs, outputs)

        # Truth values
        strainU_spar = [
            -9.60168580e-04,
            -1.36569964e-03,
            -1.24469019e-03,
            -1.28357551e-03,
            -1.66756907e-03,
            -1.93501003e-03,
            -1.96953848e-03,
            -1.93469820e-03,
            -1.91977182e-03,
            -1.87978271e-03,
            -1.82931324e-03,
            -1.79372052e-03,
            -1.79979393e-03,
            -1.81043159e-03,
            -1.80173183e-03,
            -1.77805206e-03,
            -1.74193967e-03,
            -1.69332271e-03,
            -1.63809273e-03,
            -1.57934205e-03,
            -1.51158159e-03,
            -1.38263396e-03,
            -1.23607178e-03,
            -1.07654944e-03,
            -9.03175151e-04,
            -7.15939331e-04,
            -5.20175274e-04,
            -3.25702417e-04,
            -1.74039663e-04,
            1.74416274e-08,
        ]
        strainL_spar = [
            9.15570014e-04,
            1.29000729e-03,
            1.16792215e-03,
            1.12951553e-03,
            1.39061424e-03,
            1.59834244e-03,
            1.63691918e-03,
            1.61549834e-03,
            1.60998340e-03,
            1.58649770e-03,
            1.55285394e-03,
            1.53887355e-03,
            1.57102695e-03,
            1.60654520e-03,
            1.61866703e-03,
            1.60814193e-03,
            1.57992027e-03,
            1.53608654e-03,
            1.48582402e-03,
            1.43671112e-03,
            1.37668380e-03,
            1.25754211e-03,
            1.12136865e-03,
            9.72322322e-04,
            8.10394743e-04,
            6.36106067e-04,
            4.55315076e-04,
            2.78529226e-04,
            1.43294401e-04,
            -1.50254249e-08,
        ]
        strainU_te = [
            -6.80721070e-05,
            1.29038018e-05,
            -5.63358330e-04,
            -1.47303023e-03,
            -1.97661798e-03,
            -2.13936720e-03,
            -2.10889476e-03,
            -1.93325463e-03,
            -1.82684737e-03,
            -1.72214778e-03,
            -1.59653435e-03,
            -1.43982158e-03,
            -1.24145856e-03,
            -1.02867861e-03,
            -8.57677465e-04,
            -6.97817496e-04,
            -5.41870474e-04,
            -3.90003040e-04,
            -2.14082599e-04,
            -6.15553419e-05,
            4.71495337e-05,
            4.51485998e-05,
            6.19187715e-05,
            7.44529189e-05,
            7.80493815e-05,
            8.10151027e-05,
            6.55113225e-05,
            4.38455715e-05,
            2.30186021e-05,
            -3.63739628e-09,
        ]
        strainL_te = [
            -6.80721070e-05,
            1.29038018e-05,
            -5.63358330e-04,
            -1.47303023e-03,
            -1.97661798e-03,
            -2.13936720e-03,
            -2.10889476e-03,
            -1.93325463e-03,
            -1.82684737e-03,
            -1.72214778e-03,
            -1.59653435e-03,
            -1.43982158e-03,
            -1.24145856e-03,
            -1.02867861e-03,
            -8.57677465e-04,
            -6.97817496e-04,
            -5.41870474e-04,
            -3.90003040e-04,
            -2.14082599e-04,
            -6.15553419e-05,
            4.71495337e-05,
            4.51485998e-05,
            6.19187715e-05,
            7.44529189e-05,
            7.80493815e-05,
            8.10151027e-05,
            6.55113225e-05,
            4.38455715e-05,
            2.30186021e-05,
            -3.63739628e-09,
        ]

        npt.assert_almost_equal(outputs["strainU_spar"], strainU_spar, decimal=3)
        npt.assert_almost_equal(outputs["strainL_spar"], strainL_spar, decimal=3)
        npt.assert_almost_equal(outputs["strainU_te"], strainU_te, decimal=3)
        npt.assert_almost_equal(outputs["strainL_te"], strainL_te, decimal=3)

        x_spar = inputs["xu_spar"][0]
        y_spar = inputs["yu_spar"][0]
        A = inputs["A"][0]
        E = inputs["EA"][0] / A
        npt.assert_almost_equal(outputs["axial_root_sparU_load2stress"][:2], 0.0)
        npt.assert_almost_equal(outputs["axial_root_sparU_load2stress"][5], 0.0)
        npt.assert_almost_equal(outputs["axial_root_sparU_load2stress"][2], -1.0 / A, decimal=3)
        npt.assert_almost_equal(outputs["axial_root_sparU_load2stress"][3], -E * y_spar / inputs["EI11"][0], decimal=3)
        npt.assert_almost_equal(outputs["axial_root_sparU_load2stress"][4], E * x_spar / inputs["EI22"][0], decimal=3)
        x_spar = inputs["xl_spar"][0]
        y_spar = inputs["yl_spar"][0]
        npt.assert_almost_equal(outputs["axial_root_sparL_load2stress"][:2], 0.0)
        npt.assert_almost_equal(outputs["axial_root_sparL_load2stress"][5], 0.0)
        npt.assert_almost_equal(outputs["axial_root_sparL_load2stress"][2], -1.0 / A, decimal=3)
        npt.assert_almost_equal(outputs["axial_root_sparL_load2stress"][3], -E * y_spar / inputs["EI11"][0], decimal=3)
        npt.assert_almost_equal(outputs["axial_root_sparL_load2stress"][4], E * x_spar / inputs["EI22"][0], decimal=3)

        k = np.argmax(inputs["chord"])
        x_te = inputs["xu_te"][k]
        y_te = inputs["yu_te"][k]
        A = inputs["A"][k]
        E = inputs["EA"][k] / A
        npt.assert_almost_equal(outputs["axial_maxc_teU_load2stress"][:2], 0.0)
        npt.assert_almost_equal(outputs["axial_maxc_teU_load2stress"][5], 0.0)
        npt.assert_almost_equal(outputs["axial_maxc_teU_load2stress"][2], -1.0 / A, decimal=3)
        npt.assert_almost_equal(outputs["axial_maxc_teU_load2stress"][3], -E * y_te / inputs["EI11"][k], decimal=3)
        npt.assert_almost_equal(outputs["axial_maxc_teU_load2stress"][4], E * x_te / inputs["EI22"][k], decimal=3)
        x_te = inputs["xl_te"][k]
        y_te = inputs["yl_te"][k]
        npt.assert_almost_equal(outputs["axial_maxc_teL_load2stress"][:2], 0.0)
        npt.assert_almost_equal(outputs["axial_maxc_teL_load2stress"][5], 0.0)
        npt.assert_almost_equal(outputs["axial_maxc_teL_load2stress"][2], -1.0 / A, decimal=3)
        npt.assert_almost_equal(outputs["axial_maxc_teL_load2stress"][3], -E * y_te / inputs["EI11"][k], decimal=3)
        npt.assert_almost_equal(outputs["axial_maxc_teL_load2stress"][4], E * x_te / inputs["EI22"][k], decimal=3)

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
        myopt["design_variables"]["blade"]["te_ss"] = {}
        myopt["design_variables"]["blade"]["te_ps"] = {}
        myopt["design_variables"]["blade"]["te_ss"]["n_opt"] = 3
        myopt["design_variables"]["blade"]["te_ps"]["n_opt"] = 3

        myobj = rs.DesignConstraints(modeling_options=options, opt_options=myopt)

        # Straight blade: Z is 'r'
        inputs["strainU_spar"] = np.linspace(0.4, 0.6, npts)
        inputs["strainL_spar"] = 0.6 * myone
        inputs["strainU_te"] = np.linspace(0.4, 0.6, npts)
        inputs["strainL_te"] = 0.6 * myone
        # inputs["min_strainU_spar"] = inputs["min_strainL_spar"] = 0.0
        inputs["max_strainU_spar"] = inputs["max_strainL_spar"] = 0.5
        inputs["max_strainU_te"] = inputs["max_strainL_te"] = 0.5
        inputs["s"] = np.linspace(0, 1, npts)
        inputs["s_opt_spar_cap_ss"] = inputs["s_opt_spar_cap_ps"] = np.array([0.0, 0.5, 1.0])
        inputs["s_opt_te_ss"] = inputs["s_opt_te_ps"] = np.array([0.0, 0.5, 1.0])
        inputs["rated_Omega"] = 10.0
        inputs["flap_mode_freqs"] = 0.6 * np.ones(3)
        inputs["edge_mode_freqs"] = 0.4 * np.ones(3)
        discrete_inputs["blade_number"] = 3

        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        npt.assert_equal(outputs["constr_max_strainU_spar"], np.array([0.8, 1.0, 1.2]))
        npt.assert_equal(outputs["constr_max_strainL_spar"], 1.2 * np.ones(3))
        npt.assert_equal(outputs["constr_max_strainU_te"], np.array([0.8, 1.0, 1.2]))
        npt.assert_equal(outputs["constr_max_strainL_te"], 1.2 * np.ones(3))
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
