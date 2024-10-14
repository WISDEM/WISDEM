import os
import unittest

import numpy as np
import numpy.testing as npt

import wisdem.rotorse.rotor_structure as rs
from wisdem.commonse import gravity

ARCHIVE1 = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "nrel5mw_test.npz"
ARCHIVE2 = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "nrel5mw_test2.npz"


class TestRS(unittest.TestCase):
    def testBladeCurvature(self):
        inputs = {}
        outputs = {}
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
        inputs["Rhub"] = 1.0
        inputs["blade_span_cg"] = 0.5 * inputs["r"].max()
        myobj.compute(inputs, outputs)
        npt.assert_equal(outputs["3d_curv"], myzero)
        npt.assert_equal(outputs["x_az"], myzero)
        npt.assert_equal(outputs["y_az"], myzero)
        npt.assert_equal(outputs["z_az"], inputs["r"])
        npt.assert_equal(outputs["blades_cg_hubcc"], 0.0)

        # Some coning: Z is 'r'
        inputs["precone"] = 3.0
        myobj.compute(inputs, outputs)
        npt.assert_equal(outputs["3d_curv"], 3 * myone)
        npt.assert_equal(outputs["x_az"], myzero)
        npt.assert_equal(outputs["y_az"], myzero)
        npt.assert_equal(outputs["z_az"], inputs["r"])
        npt.assert_equal(outputs["blades_cg_hubcc"], 51*np.sin(np.deg2rad(3)))

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
        npt.assert_almost_equal(outputs["blades_cg_hubcc"], 51*np.sin(np.deg2rad(cone[50])))

        # Some curve: Y is 'edge'
        inputs["precurve"] = myzero
        inputs["presweep"] = np.linspace(0, 1, npts)
        myobj.compute(inputs, outputs)
        npt.assert_almost_equal(outputs["3d_curv"], myzero)
        npt.assert_equal(outputs["x_az"], myzero)
        npt.assert_equal(outputs["y_az"], inputs["presweep"])
        npt.assert_equal(outputs["z_az"], inputs["r"])
        npt.assert_equal(outputs["blades_cg_hubcc"], np.zeros(3))

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
        npt.assert_almost_equal(outputs["blades_cg_hubcc"], 51*np.sin(np.deg2rad(cone[50])))

    def testTotalBladeLoads(self):
        inputs = {}
        outputs = {}
        npts = 101
        myzero = np.zeros(npts)
        myone = np.ones(npts)
        options = {}
        options["WISDEM"] = {}
        options["WISDEM"]["RotorSE"] = {}
        options["WISDEM"]["RotorSE"]["n_span"] = npts

        myobj = rs.TotalBladeLoads(modeling_options=options)

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
        dx = [-0.00000000e+00,  1.64638232e-03,  6.79572518e-03,  1.57057110e-02,
        2.91434108e-02,  4.91143389e-02,  7.77022110e-02,  1.15994210e-01,
        1.64492394e-01,  2.23728011e-01,  2.94236207e-01,  3.76468424e-01,
        4.70954179e-01,  5.78649328e-01,  7.00809412e-01,  8.38646276e-01,
        9.93321364e-01,  1.16597533e+00,  1.35778027e+00,  1.56975422e+00,
        1.80318407e+00,  2.05880282e+00,  2.33562566e+00,  2.63177288e+00,
        2.94510570e+00,  3.27320110e+00,  3.61333244e+00,  3.96258446e+00,
        4.31781570e+00,  4.67663896e+00]
        dy = [ 0.00000000e+00, -3.74477007e-04, -1.87858152e-03, -5.72303746e-03,
       -1.40296446e-02, -2.91760415e-02, -5.25470603e-02, -8.42038955e-02,
       -1.23721261e-01, -1.70836582e-01, -2.25359315e-01, -2.87027856e-01,
       -3.55549679e-01, -4.30734595e-01, -5.12432617e-01, -6.00416367e-01,
       -6.94413956e-01, -7.94115178e-01, -8.99164661e-01, -1.00909604e+00,
       -1.12345396e+00, -1.24175723e+00, -1.36337226e+00, -1.48764046e+00,
       -1.61394215e+00, -1.74170551e+00, -1.87042453e+00, -1.99968813e+00,
       -2.12920515e+00, -2.25881481e+00]
        dz = [-0.00000000e+00, -6.16448640e-05, -1.46019534e-04, -2.32287182e-04,
       -3.21439626e-04, -4.16434884e-04, -5.11204867e-04, -6.03366938e-04,
       -6.92726810e-04, -7.79183257e-04, -8.62353953e-04, -9.41712542e-04,
       -1.01708823e-03, -1.08989853e-03, -1.15986208e-03, -1.22711716e-03,
       -1.29165787e-03, -1.35337047e-03, -1.41204679e-03, -1.46642332e-03,
       -1.51832829e-03, -1.56716073e-03, -1.61256368e-03, -1.65439674e-03,
       -1.69245182e-03, -1.72635996e-03, -1.75550444e-03, -1.77888069e-03,
       -1.79390194e-03, -1.80321598e-03]

        freqs = [0.83325001,  1.22280141,  2.9647584 ,  4.09217506,  6.86529582,
        8.83012966,  9.47967717, 12.57008964, 14.66188868, 16.55302485]

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
        strainU_te = [-1.13026497e-04, -2.92410541e-04,  6.98603371e-05,  7.83552246e-04,
        1.03320930e-03,  1.11029244e-03,  1.07605717e-03,  1.00197550e-03,
        9.79127686e-04,  9.35549026e-04,  8.73496045e-04,  8.04623727e-04,
        7.46684889e-04,  7.13059487e-04,  6.92932872e-04,  6.66579608e-04,
        6.39927843e-04,  6.11292545e-04,  5.76948672e-04,  5.47664578e-04,
        5.09787645e-04,  4.35963414e-04,  3.45126223e-04,  2.70740910e-04,
        2.09983984e-04,  1.56888815e-04,  1.08873980e-04,  6.50476231e-05,
        3.44554780e-05, -3.57317878e-09]
        strainL_te = [-1.13026497e-04,  3.26449732e-04,  5.89457792e-04,  1.10079177e-03,
        1.24723249e-03,  1.26863500e-03,  1.21717638e-03,  1.13461525e-03,
        1.11141359e-03,  1.06753695e-03,  1.00345800e-03,  9.32406960e-04,
        8.77049612e-04,  8.49694704e-04,  8.40210719e-04,  8.34931478e-04,
        8.33043510e-04,  8.26570472e-04,  8.07511816e-04,  7.88325999e-04,
        7.48216769e-04,  6.60545168e-04,  5.48461897e-04,  4.40175017e-04,
        3.38218694e-04,  2.43921743e-04,  1.59467146e-04,  8.86770957e-05,
        4.62181259e-05, -3.57317878e-09]

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
        npt.assert_almost_equal(outputs["axial_root_sparU_load2stress"][3], -E * y_spar / inputs["EI11"][0], decimal=2)
        npt.assert_almost_equal(outputs["axial_root_sparU_load2stress"][4], E * x_spar / inputs["EI22"][0], decimal=3)
        x_spar = inputs["xl_spar"][0]
        y_spar = inputs["yl_spar"][0]
        npt.assert_almost_equal(outputs["axial_root_sparL_load2stress"][:2], 0.0)
        npt.assert_almost_equal(outputs["axial_root_sparL_load2stress"][5], 0.0)
        npt.assert_almost_equal(outputs["axial_root_sparL_load2stress"][2], -1.0 / A, decimal=3)
        npt.assert_almost_equal(outputs["axial_root_sparL_load2stress"][3], -E * y_spar / inputs["EI11"][0], decimal=2)
        npt.assert_almost_equal(outputs["axial_root_sparL_load2stress"][4], E * x_spar / inputs["EI22"][0], decimal=3)

        k = np.argmax(inputs["chord"])
        x_te = inputs["xu_te"][k]
        y_te = inputs["yu_te"][k]
        A = inputs["A"][k]
        E = inputs["EA"][k] / A
        npt.assert_almost_equal(outputs["axial_maxc_teU_load2stress"][:2], 0.0)
        npt.assert_almost_equal(outputs["axial_maxc_teU_load2stress"][5], 0.0)
        npt.assert_almost_equal(outputs["axial_maxc_teU_load2stress"][2], -1.0 / A, decimal=3)
        npt.assert_almost_equal(outputs["axial_maxc_teU_load2stress"][3], -E * y_te / inputs["EI11"][k], decimal=2)
        npt.assert_almost_equal(outputs["axial_maxc_teU_load2stress"][4], E * x_te / inputs["EI22"][k], decimal=2)
        x_te = inputs["xl_te"][k]
        y_te = inputs["yl_te"][k]
        npt.assert_almost_equal(outputs["axial_maxc_teL_load2stress"][:2], 0.0)
        npt.assert_almost_equal(outputs["axial_maxc_teL_load2stress"][5], 0.0)
        npt.assert_almost_equal(outputs["axial_maxc_teL_load2stress"][2], -1.0 / A, decimal=3)
        npt.assert_almost_equal(outputs["axial_maxc_teL_load2stress"][3], -E * y_te / inputs["EI11"][k], decimal=2)
        npt.assert_almost_equal(outputs["axial_maxc_teL_load2stress"][4], E * x_te / inputs["EI22"][k], decimal=2)

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
        inputs["tors_mode_freqs"] = 5. * np.ones(3)
        discrete_inputs["blade_number"] = 3

        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        npt.assert_equal(outputs["constr_max_strainU_spar"], np.array([0.8, 1.0, 1.2]))
        npt.assert_equal(outputs["constr_max_strainL_spar"], 1.2 * np.ones(3))
        npt.assert_equal(outputs["constr_max_strainU_te"], np.array([0.8, 1.0, 1.2]))
        npt.assert_equal(outputs["constr_max_strainL_te"], 1.2 * np.ones(3))
        npt.assert_almost_equal(outputs["constr_flap_f_margin"], 0.5 - 0.9 * 0.6)
        npt.assert_almost_equal(outputs["constr_edge_f_margin"], 1.1 * 0.4 - 0.5)


def suite():
    suite = [
        unittest.TestLoader().loadTestsFromTestCase(TestRS),
    ]
    return unittest.TestSuite(suite)


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
