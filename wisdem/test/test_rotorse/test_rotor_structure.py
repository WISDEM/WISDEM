import os
import copy
import time
import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt
import wisdem.rotorse.rotor_structure as rs
from wisdem.commonse import gravity

ARCHIVE = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "nrel5mw_test.npz"


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
        options["RotorSE"] = {}
        options["RotorSE"]["n_span"] = npts

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
        options["RotorSE"] = {}
        options["RotorSE"]["n_span"] = npts

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
        outputs0 = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}

        nrel5mw = np.load(ARCHIVE)
        for k in nrel5mw.files:
            inputs[k] = nrel5mw[k]

        npts = len(inputs["r"])
        nfreq = 10
        myzero = np.zeros(npts)
        myone = np.ones(npts)
        options = {}
        options["RotorSE"] = {}
        options["RotorSE"]["n_span"] = npts
        options["RotorSE"]["n_freq"] = nfreq

        # myold = rs.RunpBEAM(modeling_options=options)
        # myold.n_span = npts
        # myold.n_freq = nfreq
        # myold.compute(inputs, outputs0)

        myobj = rs.RunFrame3DD(modeling_options=options, pbeam=True)
        myobj.n_span = npts
        myobj.n_freq = nfreq
        myobj.compute(inputs, outputs)

        # Old 'truth' values from pBeam for this case
        dx = [
            0.0,
            0.00141456462281356,
            0.006046075473872762,
            0.013997773008441697,
            0.025763168594556164,
            0.042987893510700224,
            0.06752619574650477,
            0.10043620832740843,
            0.14231565940104443,
            0.19379537454330903,
            0.2554315913314681,
            0.3277528802376881,
            0.4115164047302365,
            0.507952293090317,
            0.6185339647972563,
            0.7446811924870854,
            0.887712673612395,
            1.0488517072539125,
            1.229227681502309,
            1.4300147298454138,
            1.6525460260148614,
            1.897539622364651,
            2.164181174421259,
            2.4508537578612914,
            2.7554711983877653,
            3.0754456110786856,
            3.407658922616137,
            3.7485638128203593,
            4.094535307014125,
            4.442486879344595,
        ]
        dy = [
            0.0,
            -0.0002393587576678093,
            -0.001114340042733925,
            -0.002917646466903267,
            -0.006061989175912945,
            -0.01080672068689353,
            -0.017120417324754375,
            -0.02489451580749532,
            -0.034042824970915285,
            -0.044532070939502,
            -0.05633714895844655,
            -0.06943193714258569,
            -0.08380352410478145,
            -0.0994591964664334,
            -0.11640689833297371,
            -0.13464979982803094,
            -0.154180951018928,
            -0.17498107397218537,
            -0.19701921587918705,
            -0.22025284579154356,
            -0.24462915090063914,
            -0.2700830496424555,
            -0.29652778353322085,
            -0.32383722075445665,
            -0.3518611798457019,
            -0.3804395912618648,
            -0.40941036223179295,
            -0.4386212624399022,
            -0.4679477954466567,
            -0.4973095360487988,
        ]
        dz = [
            0.0,
            0.0019250569819883384,
            0.004379695708376994,
            0.006810926744093996,
            0.009103406790346215,
            0.011309382969046286,
            0.013381674322741162,
            0.01528868596964843,
            0.01703649671747967,
            0.018630702645581414,
            0.020074676935679433,
            0.021380529818039173,
            0.022570754489427247,
            0.02365748316199807,
            0.02464093218364619,
            0.025522332981045,
            0.02630374650111282,
            0.026988213247013886,
            0.027579746880064458,
            0.02808353313582705,
            0.028511614125726035,
            0.0288765424646023,
            0.029182672563069247,
            0.029431542503638254,
            0.029625677824830148,
            0.029768366420322663,
            0.02986390466480914,
            0.029918291097336788,
            0.02994112218203409,
            0.02994112218203409,
        ]
        strainU_spar = [
            -0.0010475163153972262,
            -0.0013986769798977122,
            -0.001266659331922622,
            -0.001221231570964513,
            -0.0014745125814331888,
            -0.0016346647119261978,
            -0.0016571805295507711,
            -0.001639578025475371,
            -0.0016360799596948448,
            -0.0016132448702987478,
            -0.001581647215961993,
            -0.0015722767608206357,
            -0.0016015331964585391,
            -0.0016316630303846382,
            -0.0016459955985732276,
            -0.0016444299943688036,
            -0.0016280989965703663,
            -0.0015968153046302064,
            -0.0015564106223320568,
            -0.0015076112856690542,
            -0.0014438044330357454,
            -0.0013152616257443052,
            -0.001165803332789195,
            -0.0010008896729886833,
            -0.000820407210721399,
            -0.0006254471355903306,
            -0.00042301137199668237,
            -0.0002266185573570306,
            -7.787688110003286e-05,
            -8.452247736003957e-20,
        ]
        strainL_spar = [
            0.0010033649279611217,
            0.001320059088696804,
            0.0011765663115246973,
            0.0010682723915864819,
            0.001261698797218624,
            0.0014535285289408383,
            0.001488090662870691,
            0.0014789430994416692,
            0.0014847532640997491,
            0.001473097872803424,
            0.0014531118638867325,
            0.0014524709641388395,
            0.0014989689261914179,
            0.0015482837035659725,
            0.001571865822290736,
            0.0015703583458154327,
            0.0015484425166577144,
            0.0015087787269318386,
            0.0014601436099279605,
            0.0014095617493783926,
            0.001344883042658277,
            0.0012189904097461106,
            0.001074088026988576,
            0.000915079468936567,
            0.0007422789331969248,
            0.0005573606121688152,
            0.00036801558183681535,
            0.0001885774317501334,
            5.703223911709856e-05,
            6.847813798424832e-20,
        ]
        strainU_te = [
            -0.00018831017179029934,
            0.0001564808185105892,
            -4.007387320898343e-05,
            4.32017079004803e-05,
            -0.00013058365092129232,
            -0.0004825626405842731,
            -0.0006293129936039802,
            -0.0006702930969171158,
            -0.0006836797321727139,
            -0.0006690248086189902,
            -0.0006432746516370206,
            -0.0006105615924174821,
            -0.0005755069583164888,
            -0.0005256539839565068,
            -0.0004629290459191326,
            -0.00039514529764005216,
            -0.00032027143392179987,
            -0.00023667790826399187,
            -0.00014863806218502014,
            -6.513886999418668e-05,
            2.8472375497671037e-06,
            3.982831792984643e-05,
            7.283549675475475e-05,
            9.675012563446571e-05,
            0.00010647415147718156,
            0.00010047393010823578,
            7.840738470137019e-05,
            4.401096970503345e-05,
            9.990513939578569e-06,
            1.4317716195311674e-20,
        ]
        strainL_te = [
            -0.00018831017179029934,
            0.0006323870631140332,
            0.00038507470463629143,
            0.0002715050029485856,
            2.4980529283484406e-05,
            -0.0003625267278053038,
            -0.0005170310406773607,
            -0.000561276619318335,
            -0.0005728531675218926,
            -0.0005573687655653194,
            -0.0005314266147145265,
            -0.0004975707430414213,
            -0.00045581500849408763,
            -0.0003968068588466455,
            -0.0003214435061480015,
            -0.0002311093703021596,
            -0.00013047982274033598,
            -2.4719948884792993e-05,
            7.743738500586994e-05,
            0.00016873502073202595,
            0.00023835688024762374,
            0.00026724256122175353,
            0.00026128491833934806,
            0.00024236838359163482,
            0.00021130137092867206,
            0.00016802106204811244,
            0.00011565960405679938,
            5.9472164694258254e-05,
            1.4259492681900045e-05,
            1.4317716195311674e-20,
        ]
        freqs_pbeam = [
            0.9738420825300849,
            1.0721316487107038,
            3.004619628557025,
            4.126253265056667,
            6.782363480565717,
            8.87428666415306,
            9.941474123833059,
            11.990368213119352,
            14.77993982832396,
            18.35088365841697,
        ]

        npt.assert_almost_equal(outputs["dx"], dx, decimal=2)
        npt.assert_almost_equal(outputs["dy"], dy, decimal=2)
        # npt.assert_almost_equal(outputs['dz'], dz, decimal=2) # Very small numbers, so no precision
        npt.assert_almost_equal(outputs["strainU_spar"], strainU_spar, decimal=3)
        npt.assert_almost_equal(outputs["strainL_spar"], strainL_spar, decimal=3)
        npt.assert_almost_equal(outputs["strainU_te"], strainU_te, decimal=3)
        npt.assert_almost_equal(outputs["strainL_te"], strainL_te, decimal=3)
        npt.assert_almost_equal(outputs["freqs"], freqs_pbeam, decimal=0)

    def testRunFrame3DD_TipDeflection(self):
        inputs = {}
        outputs0 = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}

        nrel5mw = np.load(ARCHIVE)
        for k in nrel5mw.files:
            inputs[k] = nrel5mw[k]

        npts = len(inputs["r"])
        nfreq = 10
        myzero = np.zeros(npts)
        myone = np.ones(npts)
        options = {}
        options["RotorSE"] = {}
        options["RotorSE"]["n_span"] = npts
        options["RotorSE"]["n_freq"] = nfreq

        # Tip deflection the old pBeam way
        myobj0 = rs.RunFrame3DD(modeling_options=options, pbeam=True)
        myobj0.n_span = npts
        myobj0.n_freq = nfreq
        myobj0.compute(inputs, outputs0)

        mytip = rs.TipDeflection()
        inputs["dx_tip"] = outputs0["dx"][-1]
        inputs["dy_tip"] = outputs0["dy"][-1]
        inputs["dz_tip"] = outputs0["dz"][-1]
        inputs["theta_tip"] = inputs["theta"][-1]
        inputs["pitch_load"] = 0.0
        inputs["tilt"] = 0.0
        inputs["3d_curv_tip"] = 0.0
        inputs["dynamicFactor"] = 1.0
        mytip.compute(inputs, outputs0)

        # Tip deflection the new, enhanced way with geometric stiffening
        myobj = rs.RunFrame3DD(modeling_options=options, pbeam=False)
        myobj.n_span = npts
        myobj.n_freq = nfreq
        myobj.compute(inputs, outputs)

        inputs["dx_tip"] = outputs["dx"][-1]
        inputs["dy_tip"] = outputs["dy"][-1]
        inputs["dz_tip"] = outputs["dz"][-1]
        inputs["theta_tip"] = inputs["theta"][-1]
        mytip.compute(inputs, outputs)
        self.assertLess(outputs["tip_deflection"], outputs0["tip_deflection"])

    def testConstraints(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}

        npts = 101
        myzero = np.zeros(npts)
        myone = np.ones(npts)
        options = {}
        options["RotorSE"] = {}
        options["RotorSE"]["n_span"] = npts
        options["RotorSE"]["n_freq"] = 6
        options["RotorSE"]["gamma_freq"] = 1.1

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
