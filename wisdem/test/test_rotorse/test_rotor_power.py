import os
import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt
import wisdem.rotorse.rotor_power as rp

# Load in airfoil and blade shape inputs for NREL 5MW
ARCHIVE = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "regulation.npz"
NPZFILE = np.load(ARCHIVE)


def fillprob(prob, n_pc, n_span):
    prob.setup()
    for k in NPZFILE.files:
        prob[k] = NPZFILE[k]

    prob.set_val("v_min", 4.0, units="m/s")
    prob.set_val("v_max", 25.0, units="m/s")
    prob.set_val("rated_power", 5e6, units="W")
    prob.set_val("omega_min", 0.0, units="rpm")
    prob.set_val("omega_max", 100.0, units="rpm")
    prob.set_val("control_maxTS", 90.0, units="m/s")
    prob.set_val("tsr_operational", 10.0)
    prob.set_val("control_pitch", 0.0, units="deg")
    prob.set_val("gearbox_efficiency", 0.975)
    prob.set_val("generator_efficiency", 0.975 * np.ones(n_pc))
    prob.set_val("lss_rpm", np.linspace(0.1, 100, n_pc))
    prob.set_val("drivetrainType", "GEARED")

    prob.set_val("Rhub", 1.0, units="m")
    prob.set_val("Rtip", 70.0, units="m")
    prob.set_val("hub_height", 100.0, units="m")
    prob.set_val("precone", 0.0, units="deg")
    prob.set_val("tilt", 0.0, units="deg")
    prob.set_val("yaw", 0.0, units="deg")
    prob.set_val("precurve", np.zeros(n_span), units="m")
    prob.set_val("precurveTip", 0.0, units="m")
    prob.set_val("presweep", np.zeros(n_span), units="m")
    prob.set_val("presweepTip", 0.0, units="m")

    prob.set_val("shearExp", 0.25)
    prob.set_val("nSector", 4)
    prob.set_val("tiploss", True)
    prob.set_val("hubloss", True)
    prob.set_val("wakerotation", True)
    prob.set_val("usecd", True)

    return prob


class TestServo(unittest.TestCase):
    def testGust(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}

        myobj = rp.GustETM(std=2.5)

        inputs["V_mean"] = 10.0
        inputs["V_hub"] = 15.0
        discrete_inputs["turbulence_class"] = "A"
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        sigma = 0.32 * (0.072 * 8.0 * 3.5 + 10.0)
        expect = 15.0 + 2.5 * sigma
        self.assertEqual(outputs["V_gust"], expect)

        # Test lower case
        discrete_inputs["turbulence_class"] = "c"
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        sigma = 0.24 * (0.072 * 8.0 * 3.5 + 10.0)
        expect = 15.0 + 2.5 * sigma
        self.assertEqual(outputs["V_gust"], expect)

        # Test bad class
        discrete_inputs["turbulence_class"] = "d"
        try:
            myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        except ValueError:
            self.assertTrue(True)

    def testRegulationTrajectory(self):
        prob = om.Problem()

        (n_span, n_aoa, n_Re, n_tab) = NPZFILE["airfoils_cl"].shape
        n_pc = 22

        modeling_options = {}
        modeling_options["WISDEM"] = {}
        modeling_options["WISDEM"]["RotorSE"] = {}
        modeling_options["WISDEM"]["RotorSE"]["n_span"] = n_span
        modeling_options["WISDEM"]["RotorSE"]["n_aoa"] = n_aoa
        modeling_options["WISDEM"]["RotorSE"]["n_Re"] = n_Re
        modeling_options["WISDEM"]["RotorSE"]["n_tab"] = n_tab
        modeling_options["WISDEM"]["RotorSE"]["regulation_reg_III"] = True
        modeling_options["WISDEM"]["RotorSE"]["n_pc"] = n_pc
        modeling_options["WISDEM"]["RotorSE"]["n_pc_spline"] = n_pc
        modeling_options["WISDEM"]["RotorSE"]["peak_thrust_shaving"] = False
        modeling_options["WISDEM"]["RotorSE"]["fix_pitch_regI12"] = False

        prob.model.add_subsystem(
            "powercurve", rp.RegulatedPowerCurve(modeling_options=modeling_options), promotes=["*"]
        )
        prob = fillprob(prob, n_pc, n_span)

        # All reg 2: no maxTS, no max rpm, no power limit
        prob["omega_max"] = 1e3
        prob["control_maxTS"] = 1e5
        prob["rated_power"] = 1e16
        prob.run_model()

        grid0 = np.cumsum(np.abs(np.diff(np.cos(np.linspace(-np.pi / 4.0, np.pi / 2.0, n_pc)))))
        grid1 = (grid0 - grid0[0]) / (grid0[-1] - grid0[0])
        V_expect0 = grid1 * (prob["v_max"] - prob["v_min"]) + prob["v_min"]
        V_spline = np.linspace(prob["v_min"], prob["v_max"], n_pc)
        irated = 12

        V_expect1 = np.sort(np.r_[V_expect0, prob["rated_V"]])
        Omega_tsr = V_expect1 * 10 * 60 / 70.0 / 2.0 / np.pi
        npt.assert_equal(prob["V"], V_expect1)
        npt.assert_equal(prob["V_spline"], V_spline.flatten())
        npt.assert_allclose(prob["Omega"], Omega_tsr)
        npt.assert_equal(prob["pitch"], np.zeros(n_pc))
        npt.assert_array_almost_equal(prob["Cp"], prob["Cp_aero"] * 0.975 * 0.975)
        npt.assert_allclose(prob["Cp"], prob["Cp"][0])
        npt.assert_allclose(prob["Cp_aero"], prob["Cp_aero"][0])
        myCp = prob["P"] / (0.5 * 1.225 * V_expect1 ** 3.0 * np.pi * 70 ** 2)
        npt.assert_allclose(myCp, myCp[0], rtol=1e-6)
        self.assertGreater(myCp[0], 0.4)
        self.assertGreater(0.5, myCp[0])
        npt.assert_allclose(myCp, prob["Cp"], rtol=1e-6)
        npt.assert_array_less(prob["P"][:-2], prob["P"][1:-1])
        npt.assert_array_less(prob["Q"][:-2], prob["Q"][1:-1])
        npt.assert_array_less(prob["T"][:-2], prob["T"][1:-1])
        self.assertEqual(prob["rated_V"], V_expect1[-1])
        self.assertAlmostEqual(prob["rated_Omega"][0], Omega_tsr[-1], 5)
        self.assertEqual(prob["rated_pitch"], 0.0)

        # Test no maxTS, max rpm, no power limit
        prob["omega_max"] = 15.0
        prob["control_maxTS"] = 1e5
        prob["rated_power"] = 1e16
        prob.run_model()
        V_expect1 = np.sort(np.r_[V_expect0, prob["rated_V"]])
        Omega_tsr = V_expect1 * 10 * 60 / 70.0 / 2.0 / np.pi
        Omega_expect = np.minimum(Omega_tsr, 15.0)
        npt.assert_allclose(prob["V"], V_expect1)
        npt.assert_equal(prob["V_spline"], V_spline.flatten())
        npt.assert_allclose(prob["Omega"], Omega_expect)
        npt.assert_equal(prob["pitch"][:irated], 0.0)
        npt.assert_array_less(0.0, np.abs(prob["pitch"][(irated + 1) :]))
        npt.assert_array_almost_equal(prob["Cp"], prob["Cp_aero"] * 0.975 * 0.975)
        npt.assert_array_less(prob["P"][:-2], prob["P"][1:-1])
        npt.assert_array_less(prob["Q"][:-2], prob["Q"][1:-1])
        npt.assert_array_less(prob["T"][:-2], prob["T"][1:-1])
        self.assertAlmostEqual(prob["rated_V"], V_expect1[-1], 3)
        self.assertAlmostEqual(prob["rated_Omega"][0], 15.0)
        self.assertEqual(prob["rated_pitch"], 0.0)
        myCp = prob["P"] / (0.5 * 1.225 * V_expect1 ** 3.0 * np.pi * 70 ** 2)
        npt.assert_allclose(myCp[:irated], myCp[0])
        npt.assert_allclose(myCp[:irated], prob["Cp"][:irated])

        # Test maxTS, no max rpm, no power limit
        prob["omega_max"] = 1e3
        prob["control_maxTS"] = 105.0
        prob["rated_power"] = 1e16
        prob.run_model()
        V_expect1 = np.sort(np.r_[V_expect0, prob["rated_V"]])
        # V_expect1[irated] = 105./10.
        Omega_tsr = V_expect1 * 10 * 60 / 70.0 / 2.0 / np.pi
        Omega_expect = np.minimum(Omega_tsr, 105.0 / 70.0 / 2 / np.pi * 60)
        npt.assert_allclose(prob["V"], V_expect1)
        npt.assert_equal(prob["V_spline"], V_spline.flatten())
        npt.assert_allclose(prob["Omega"], Omega_expect)
        npt.assert_equal(prob["pitch"][:irated], 0.0)
        npt.assert_array_less(0.0, np.abs(prob["pitch"][irated:]))
        npt.assert_array_almost_equal(prob["Cp"], prob["Cp_aero"] * 0.975 * 0.975)
        npt.assert_array_less(prob["P"][:-2], prob["P"][1:-1])
        npt.assert_array_less(prob["Q"][:-2], prob["Q"][1:-1])
        npt.assert_array_less(prob["T"][:-2], prob["T"][1:-1])
        self.assertEqual(prob["rated_V"], V_expect1[-1])
        self.assertAlmostEqual(prob["rated_Omega"][0], Omega_expect[-1])
        self.assertEqual(prob["rated_pitch"], 0.0)
        myCp = prob["P"] / (0.5 * 1.225 * V_expect1 ** 3.0 * np.pi * 70 ** 2)
        npt.assert_allclose(myCp[:irated], myCp[0])
        npt.assert_allclose(myCp[:irated], prob["Cp"][:irated])

        # Test no maxTS, no max rpm, power limit
        prob["omega_max"] = 1e3
        prob["control_maxTS"] = 1e4
        prob["rated_power"] = 5e6
        prob.run_model()
        V_expect1 = np.sort(np.r_[V_expect0, prob["rated_V"]])
        Omega_tsr = V_expect1 * 10 * 60 / 70.0 / 2.0 / np.pi
        Omega_expect = np.minimum(Omega_tsr, prob["rated_Omega"])
        npt.assert_allclose(prob["V"], V_expect1)
        npt.assert_equal(prob["V_spline"], V_spline.flatten())
        npt.assert_allclose(prob["Omega"], Omega_expect)
        npt.assert_equal(prob["pitch"][:irated], 0.0)
        npt.assert_array_less(0.0, np.abs(prob["pitch"][(irated + 1) :]))
        npt.assert_array_almost_equal(prob["Cp"], prob["Cp_aero"] * 0.975 * 0.975)
        npt.assert_array_less(prob["P"][:irated], prob["P"][1 : (irated + 1)])
        npt.assert_allclose(prob["P"][irated:], 5e6, rtol=1e-4, atol=0)
        # npt.assert_array_less(prob['Q'], prob['Q'][1:])
        npt.assert_array_less(prob["T"], prob["T"][irated] + 1e-1)
        # print('RATED T',prob["T"][irated])
        # self.assertEqual(prob['rated_V'], V_expect1[-1])
        self.assertAlmostEqual(prob["rated_Omega"][0], Omega_expect[-1])
        self.assertEqual(prob["rated_pitch"], 0.0)
        myCp = prob["P"] / (0.5 * 1.225 * V_expect1 ** 3.0 * np.pi * 70 ** 2)
        npt.assert_allclose(myCp[:irated], myCp[0])
        npt.assert_allclose(myCp[:irated], prob["Cp"][:irated])

        # Test min & max rpm, no power limit
        prob["omega_min"] = 7.0
        prob["omega_max"] = 15.0
        prob["control_maxTS"] = 1e5
        prob["rated_power"] = 1e16
        prob.run_model()
        V_expect1 = np.sort(np.r_[V_expect0, prob["rated_V"]])
        # V_expect1[irated] = 15.*70*2*np.pi/(10.*60.)
        Omega_tsr = V_expect1 * 10 * 60 / 70.0 / 2.0 / np.pi
        Omega_expect = np.maximum(np.minimum(Omega_tsr, 15.0), 7.0)
        npt.assert_allclose(prob["V"], V_expect1)
        npt.assert_equal(prob["V_spline"], V_spline.flatten())
        npt.assert_allclose(prob["Omega"], Omega_expect)
        npt.assert_array_less(0.0, np.abs(prob["pitch"][Omega_expect != Omega_tsr]))
        npt.assert_array_almost_equal(prob["Cp"], prob["Cp_aero"] * 0.975 * 0.975)
        npt.assert_array_less(prob["P"][:-2], prob["P"][1:-1])
        npt.assert_array_less(prob["Q"][:-2], prob["Q"][1:-1])
        npt.assert_array_less(prob["T"][:-2], prob["T"][1:-1])
        self.assertEqual(prob["rated_V"], V_expect1[-1])
        self.assertAlmostEqual(prob["rated_Omega"][0], 15.0)
        self.assertEqual(prob["rated_pitch"], 0.0)
        myCp = prob["P"] / (0.5 * 1.225 * V_expect1 ** 3.0 * np.pi * 70 ** 2)
        npt.assert_allclose(myCp[Omega_expect == Omega_tsr], myCp[6])
        npt.assert_allclose(myCp[Omega_expect == Omega_tsr], prob["Cp"][Omega_expect == Omega_tsr])

        # Test fixed pitch
        prob["omega_min"] = 0.0
        prob["omega_max"] = 15.0
        prob["control_maxTS"] = 1e5
        prob["rated_power"] = 1e16
        prob["control_pitch"] = 5.0
        prob.run_model()
        V_expect1 = np.sort(np.r_[V_expect0, prob["rated_V"]])
        # V_expect1[irated] = 15.*70*2*np.pi/(10.*60.)
        Omega_tsr = V_expect1 * 10 * 60 / 70.0 / 2.0 / np.pi
        Omega_expect = np.minimum(Omega_tsr, 15.0)
        npt.assert_allclose(prob["V"], V_expect1)
        npt.assert_equal(prob["V_spline"], V_spline.flatten())
        npt.assert_allclose(prob["Omega"], Omega_expect)
        npt.assert_equal(prob["pitch"][:irated], 5.0)
        npt.assert_array_less(0.0, np.abs(prob["pitch"][irated:]))
        npt.assert_array_almost_equal(prob["Cp"], prob["Cp_aero"] * 0.975 * 0.975)
        npt.assert_array_less(prob["P"][:-2], prob["P"][1:-1])
        npt.assert_array_less(prob["Q"][:-2], prob["Q"][1:-1])
        npt.assert_array_less(prob["T"][:-2], prob["T"][1:-1])
        self.assertAlmostEqual(prob["rated_V"], V_expect1[-1], 3)
        self.assertAlmostEqual(prob["rated_Omega"][0], 15.0)
        self.assertEqual(prob["rated_pitch"], 5.0)
        myCp = prob["P"] / (0.5 * 1.225 * V_expect1 ** 3.0 * np.pi * 70 ** 2)
        npt.assert_allclose(myCp[:irated], myCp[0])
        npt.assert_allclose(myCp[:irated], prob["Cp"][:irated])

    def testRegulationTrajectoryNoRegion3(self):
        prob = om.Problem()

        # Load in airfoil and blade shape inputs for NREL 5MW
        (n_span, n_aoa, n_Re, n_tab) = NPZFILE["airfoils_cl"].shape
        n_pc = 22

        modeling_options = {}
        modeling_options["WISDEM"] = {}
        modeling_options["WISDEM"]["RotorSE"] = {}
        modeling_options["WISDEM"]["RotorSE"]["n_span"] = n_span
        modeling_options["WISDEM"]["RotorSE"]["n_aoa"] = n_aoa
        modeling_options["WISDEM"]["RotorSE"]["n_Re"] = n_Re
        modeling_options["WISDEM"]["RotorSE"]["n_tab"] = n_tab
        modeling_options["WISDEM"]["RotorSE"]["regulation_reg_III"] = False
        modeling_options["WISDEM"]["RotorSE"]["n_pc"] = n_pc
        modeling_options["WISDEM"]["RotorSE"]["n_pc_spline"] = n_pc
        modeling_options["WISDEM"]["RotorSE"]["peak_thrust_shaving"] = False
        modeling_options["WISDEM"]["RotorSE"]["fix_pitch_regI12"] = False

        prob.model.add_subsystem(
            "powercurve", rp.RegulatedPowerCurve(modeling_options=modeling_options), promotes=["*"]
        )
        prob = fillprob(prob, n_pc, n_span)

        # All reg 2: no maxTS, no max rpm, no power limit
        prob["omega_max"] = 1e3
        prob["control_maxTS"] = 1e5
        prob["rated_power"] = 1e16
        prob.run_model()

        grid0 = np.cumsum(np.abs(np.diff(np.cos(np.linspace(-np.pi / 4.0, np.pi / 2.0, n_pc)))))
        grid1 = (grid0 - grid0[0]) / (grid0[-1] - grid0[0])
        V_expect0 = grid1 * (prob["v_max"] - prob["v_min"]) + prob["v_min"]
        V_spline = np.linspace(prob["v_min"], prob["v_max"], n_pc)
        irated = 12

        V_expect1 = np.sort(np.r_[V_expect0, prob["rated_V"]])
        Omega_tsr = V_expect1 * 10 * 60 / 70.0 / 2.0 / np.pi
        npt.assert_equal(prob["V"], V_expect1)
        npt.assert_equal(prob["V_spline"], V_spline.flatten())
        npt.assert_allclose(prob["Omega"], Omega_tsr)
        npt.assert_equal(prob["pitch"], np.zeros(n_pc))
        npt.assert_array_almost_equal(prob["Cp"], prob["Cp_aero"] * 0.975 * 0.975)
        npt.assert_allclose(prob["Cp"], prob["Cp"][0], rtol=1e-6)
        npt.assert_allclose(prob["Cp_aero"], prob["Cp_aero"][0])
        myCp = prob["P"] / (0.5 * 1.225 * V_expect1 ** 3.0 * np.pi * 70 ** 2)
        npt.assert_allclose(myCp, myCp[0], rtol=1e-6)
        self.assertGreater(myCp[0], 0.4)
        self.assertGreater(0.5, myCp[0])
        npt.assert_allclose(myCp, prob["Cp"], rtol=1e-6)
        npt.assert_array_less(prob["P"][:-2], prob["P"][1:-1])
        npt.assert_array_less(prob["Q"][:-2], prob["Q"][1:-1])
        npt.assert_array_less(prob["T"][:-2], prob["T"][1:-1])
        self.assertEqual(prob["rated_V"], V_expect1[-1])
        self.assertAlmostEqual(prob["rated_Omega"][0], Omega_tsr[-1], 5)
        self.assertEqual(prob["rated_pitch"], 0.0)

        # Test no maxTS, no max rpm, power limit
        prob["omega_max"] = 1e3
        prob["control_maxTS"] = 1e4
        prob["rated_power"] = 5e6
        prob.run_model()
        V_expect1 = np.sort(np.r_[V_expect0, prob["rated_V"]])
        Omega_tsr = V_expect1 * 10 * 60 / 70.0 / 2.0 / np.pi
        Omega_expect = np.minimum(Omega_tsr, prob["rated_Omega"])
        npt.assert_allclose(prob["V"], V_expect1)
        npt.assert_equal(prob["V_spline"], V_spline.flatten())
        npt.assert_allclose(prob["Omega"], Omega_expect)
        npt.assert_equal(prob["pitch"], 0.0)
        # npt.assert_array_less(0.0, np.abs(prob['pitch'][(irated+1):]))
        npt.assert_allclose(prob["Cp"][: (irated + 1)], prob["Cp_aero"][: (irated + 1)] * 0.975 * 0.975)
        npt.assert_array_less(prob["P"][:irated], prob["P"][1 : (irated + 1)])
        npt.assert_allclose(prob["P"][irated:], 5e6, rtol=1e-6, atol=0)
        # npt.assert_equal(prob['Q'][(irated+1):], prob['Q'][irated])
        npt.assert_equal(prob["T"][(irated + 1) :], 0.0)
        npt.assert_array_less(prob["T"], prob["T"][irated] + 1e-1)
        # self.assertEqual(prob['rated_V'], V_expect1[-1])
        self.assertAlmostEqual(prob["rated_Omega"][0], Omega_expect[-1])
        self.assertEqual(prob["rated_pitch"], 0.0)
        myCp = prob["P"] / (0.5 * 1.225 * V_expect1 ** 3.0 * np.pi * 70 ** 2)
        npt.assert_allclose(myCp[:irated], myCp[0])
        npt.assert_allclose(myCp[:irated], prob["Cp"][:irated])

    def testRegulationTrajectory_PeakShaving(self):
        prob = om.Problem()

        (n_span, n_aoa, n_Re, n_tab) = NPZFILE["airfoils_cl"].shape
        n_pc = 22

        modeling_options = {}
        modeling_options["WISDEM"] = {}
        modeling_options["WISDEM"]["RotorSE"] = {}
        modeling_options["WISDEM"]["RotorSE"]["n_span"] = n_span
        modeling_options["WISDEM"]["RotorSE"]["n_aoa"] = n_aoa
        modeling_options["WISDEM"]["RotorSE"]["n_Re"] = n_Re
        modeling_options["WISDEM"]["RotorSE"]["n_tab"] = n_tab
        modeling_options["WISDEM"]["RotorSE"]["regulation_reg_III"] = True
        modeling_options["WISDEM"]["RotorSE"]["n_pc"] = n_pc
        modeling_options["WISDEM"]["RotorSE"]["n_pc_spline"] = n_pc
        modeling_options["WISDEM"]["RotorSE"]["peak_thrust_shaving"] = True
        modeling_options["WISDEM"]["RotorSE"]["fix_pitch_regI12"] = False
        modeling_options["WISDEM"]["RotorSE"]["thrust_shaving_coeff"] = 0.8

        prob.model.add_subsystem(
            "powercurve", rp.RegulatedPowerCurve(modeling_options=modeling_options), promotes=["*"]
        )
        prob = fillprob(prob, n_pc, n_span)

        # All reg 2: no maxTS, no max rpm, no power limit
        prob["omega_max"] = 1e3
        prob["control_maxTS"] = 1e5
        prob["rated_power"] = 1e16
        prob.run_model()

        grid0 = np.cumsum(np.abs(np.diff(np.cos(np.linspace(-np.pi / 4.0, np.pi / 2.0, n_pc)))))
        grid1 = (grid0 - grid0[0]) / (grid0[-1] - grid0[0])
        V_expect0 = grid1 * (prob["v_max"] - prob["v_min"]) + prob["v_min"]
        V_spline = np.linspace(prob["v_min"], prob["v_max"], n_pc)
        irated = 12

        V_expect1 = np.sort(np.r_[V_expect0, prob["rated_V"]])
        Omega_tsr = V_expect1 * 10 * 60 / 70.0 / 2.0 / np.pi
        npt.assert_equal(prob["V"], V_expect1)
        npt.assert_equal(prob["V_spline"], V_spline.flatten())
        npt.assert_allclose(prob["Omega"], Omega_tsr)
        npt.assert_equal(prob["pitch"], np.zeros(n_pc))
        npt.assert_array_almost_equal(prob["Cp"], prob["Cp_aero"] * 0.975 * 0.975)
        npt.assert_allclose(prob["Cp"], prob["Cp"][0], rtol=1e-6)
        npt.assert_allclose(prob["Cp_aero"], prob["Cp_aero"][0])
        myCp = prob["P"] / (0.5 * 1.225 * V_expect1 ** 3.0 * np.pi * 70 ** 2)
        npt.assert_allclose(myCp, myCp[0], rtol=1e-6)
        self.assertGreater(myCp[0], 0.4)
        self.assertGreater(0.5, myCp[0])
        npt.assert_allclose(myCp, prob["Cp"], rtol=1e-6)
        npt.assert_array_less(prob["P"][:-2], prob["P"][1:-1])
        npt.assert_array_less(prob["Q"][:-2], prob["Q"][1:-1])
        npt.assert_array_less(prob["T"][:-2], prob["T"][1:-1])
        self.assertEqual(prob["rated_V"], V_expect1[-1])
        self.assertAlmostEqual(prob["rated_Omega"][0], Omega_tsr[-1], 5)
        self.assertEqual(prob["rated_pitch"], 0.0)

        # Test no maxTS, max rpm, no power limit
        prob["omega_max"] = 15.0
        prob["control_maxTS"] = 1e5
        prob["rated_power"] = 1e16
        prob.run_model()
        V_expect1 = np.sort(np.r_[V_expect0, prob["rated_V"]])
        Omega_tsr = V_expect1 * 10 * 60 / 70.0 / 2.0 / np.pi
        Omega_expect = np.minimum(Omega_tsr, 15.0)
        npt.assert_allclose(prob["V"], V_expect1)
        npt.assert_equal(prob["V_spline"], V_spline.flatten())
        npt.assert_allclose(prob["Omega"], Omega_expect)
        npt.assert_equal(prob["pitch"][:irated], 0.0)
        npt.assert_array_less(0.0, np.abs(prob["pitch"][(irated + 1) :]))
        npt.assert_array_almost_equal(prob["Cp"], prob["Cp_aero"] * 0.975 * 0.975)
        npt.assert_array_less(prob["P"][:-2], prob["P"][1:-1])
        npt.assert_array_less(prob["Q"][:-2], prob["Q"][1:-1])
        npt.assert_array_less(prob["T"][:-2], prob["T"][1:-1])
        self.assertAlmostEqual(prob["rated_V"], V_expect1[-1], 3)
        self.assertAlmostEqual(prob["rated_Omega"][0], 15.0)
        self.assertEqual(prob["rated_pitch"], 0.0)
        myCp = prob["P"] / (0.5 * 1.225 * V_expect1 ** 3.0 * np.pi * 70 ** 2)
        npt.assert_allclose(myCp[:irated], myCp[0])
        npt.assert_allclose(myCp[:irated], prob["Cp"][:irated])

        # Test maxTS, no max rpm, no power limit
        prob["omega_max"] = 1e3
        prob["control_maxTS"] = 105.0
        prob["rated_power"] = 1e16
        prob.run_model()
        V_expect1 = np.sort(np.r_[V_expect0, prob["rated_V"]])
        # V_expect1[irated] = 105./10.
        Omega_tsr = V_expect1 * 10 * 60 / 70.0 / 2.0 / np.pi
        Omega_expect = np.minimum(Omega_tsr, 105.0 / 70.0 / 2 / np.pi * 60)
        npt.assert_allclose(prob["V"], V_expect1)
        npt.assert_equal(prob["V_spline"], V_spline.flatten())
        npt.assert_allclose(prob["Omega"], Omega_expect)
        npt.assert_equal(prob["pitch"][:irated], 0.0)
        npt.assert_array_less(0.0, np.abs(prob["pitch"][irated:]))
        npt.assert_array_almost_equal(prob["Cp"], prob["Cp_aero"] * 0.975 * 0.975)
        npt.assert_array_less(prob["P"][:-2], prob["P"][1:-1])
        npt.assert_array_less(prob["Q"][:-2], prob["Q"][1:-1])
        npt.assert_array_less(prob["T"][:-2], prob["T"][1:-1])
        self.assertEqual(prob["rated_V"], V_expect1[-1])
        self.assertAlmostEqual(prob["rated_Omega"][0], Omega_expect[-1])
        self.assertEqual(prob["rated_pitch"], 0.0)
        myCp = prob["P"] / (0.5 * 1.225 * V_expect1 ** 3.0 * np.pi * 70 ** 2)
        npt.assert_allclose(myCp[:irated], myCp[0])
        npt.assert_allclose(myCp[:irated], prob["Cp"][:irated])

        # Test no maxTS, no max rpm, power limit
        prob["omega_max"] = 1e3
        prob["control_maxTS"] = 1e4
        prob["rated_power"] = 5e6
        prob.run_model()
        V_expect1 = np.sort(np.r_[V_expect0, prob["rated_V"]])
        Omega_tsr = V_expect1 * 10 * 60 / 70.0 / 2.0 / np.pi
        Omega_expect = np.minimum(Omega_tsr, prob["rated_Omega"])
        npt.assert_allclose(prob["V"], V_expect1)
        npt.assert_equal(prob["V_spline"], V_spline.flatten())
        npt.assert_allclose(prob["Omega"][:irated], Omega_expect[:irated])
        npt.assert_equal(prob["pitch"][: (irated - 1)], 0.0)
        npt.assert_array_less(0.0, np.abs(prob["pitch"][irated:]))
        npt.assert_array_almost_equal(prob["Cp"], prob["Cp_aero"] * 0.975 * 0.975)
        npt.assert_array_less(prob["P"][:irated], prob["P"][1 : (irated + 1)])
        npt.assert_allclose(prob["P"][irated:], 5e6, rtol=1e-4, atol=0)
        npt.assert_array_less(prob["T"], 0.8 * 880899)  # From print out in first test
        self.assertAlmostEqual(prob["rated_Omega"][0], Omega_expect[-1])
        self.assertGreater(prob["rated_pitch"], 0.0)
        myCp = prob["P"] / (0.5 * 1.225 * V_expect1 ** 3.0 * np.pi * 70 ** 2)
        npt.assert_allclose(myCp[: (irated - 1)], myCp[0])
        npt.assert_allclose(myCp[: (irated - 1)], prob["Cp"][: (irated - 1)])

    def testRegulationTrajectory_reindex(self):
        prob = om.Problem()

        debug_archive = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "debug.npz"
        debug_npz = np.load(debug_archive)

        (n_span, n_aoa, n_Re, n_tab) = debug_npz["airfoils_cl"].shape
        n_pc = 50

        modeling_options = {}
        modeling_options["WISDEM"] = {}
        modeling_options["WISDEM"]["RotorSE"] = {}
        modeling_options["WISDEM"]["RotorSE"]["n_span"] = n_span
        modeling_options["WISDEM"]["RotorSE"]["n_aoa"] = n_aoa
        modeling_options["WISDEM"]["RotorSE"]["n_Re"] = n_Re
        modeling_options["WISDEM"]["RotorSE"]["n_tab"] = n_tab
        modeling_options["WISDEM"]["RotorSE"]["regulation_reg_III"] = True
        modeling_options["WISDEM"]["RotorSE"]["n_pc"] = n_pc
        modeling_options["WISDEM"]["RotorSE"]["n_pc_spline"] = n_pc
        modeling_options["WISDEM"]["RotorSE"]["peak_thrust_shaving"] = False
        modeling_options["WISDEM"]["RotorSE"]["fix_pitch_regI12"] = False
        modeling_options["WISDEM"]["RotorSE"]["thrust_shaving_coeff"] = 1.0

        prob.model.add_subsystem(
            "powercurve", rp.RegulatedPowerCurve(modeling_options=modeling_options), promotes=["*"]
        )
        prob.setup()
        for k in debug_npz.files:
            prob[k] = debug_npz[k]

        prob.run_model()
        grid0 = np.cumsum(np.abs(np.diff(np.cos(np.linspace(-np.pi / 4.0, np.pi / 2.0, n_pc)))))
        grid1 = (grid0 - grid0[0]) / (grid0[-1] - grid0[0])
        V_expect0 = grid1 * (prob["v_max"] - prob["v_min"]) + prob["v_min"]
        V_spline = np.linspace(prob["v_min"], prob["v_max"], n_pc)

        V_expect1 = np.sort(np.r_[V_expect0, prob["rated_V"]])
        npt.assert_equal(prob["V"], V_expect1)
        npt.assert_equal(prob["V_spline"], V_spline.flatten())


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestServo))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
