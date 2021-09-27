import unittest

import numpy as np
import numpy.testing as npt
import wisdem.drivetrainse.layout as lay
import wisdem.drivetrainse.drive_structure as ds
from wisdem.commonse import gravity

npts = 12


class TestDirectStructure(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}
        self.opt = {}

        self.discrete_inputs["upwind"] = True

        self.inputs["L_12"] = 2.0
        self.inputs["L_h1"] = 1.0
        self.inputs["L_generator"] = 3.25
        # self.inputs['L_2n'] = 1.5
        # self.inputs['L_grs'] = 1.1
        # self.inputs['L_gsn'] = 1.1
        self.inputs["L_hss"] = 0.75
        self.inputs["L_gearbox"] = 1.2
        self.inputs["overhang"] = 6.25
        self.inputs["drive_height"] = 4.875
        self.inputs["tilt"] = 4.0
        self.inputs["access_diameter"] = 0.9

        myones = np.ones(5)
        self.inputs["lss_diameter"] = 3.3 * myones
        self.inputs["lss_wall_thickness"] = 0.45 * myones
        self.inputs["hss_diameter"] = 1.6 * np.ones(3)
        self.inputs["hss_wall_thickness"] = 0.25 * np.ones(3)
        self.inputs["nose_diameter"] = 2.2 * myones
        self.inputs["nose_wall_thickness"] = 0.1 * myones
        self.inputs["bedplate_wall_thickness"] = 0.06 * np.ones(npts)

        self.inputs["bedplate_flange_width"] = 1.5
        self.inputs["bedplate_flange_thickness"] = 0.05
        # self.inputs['bedplate_web_height'] = 1.0
        self.inputs["bedplate_web_thickness"] = 0.05

        self.inputs["D_top"] = 6.5
        self.inputs["hub_diameter"] = 4.0

        self.inputs["other_mass"] = 200e3
        self.inputs["mb1_mass"] = 10e3
        self.inputs["mb1_I"] = 10e3 * 0.5 * 2 ** 2 * np.ones(3)
        self.inputs["mb2_mass"] = 10e3
        self.inputs["mb2_I"] = 10e3 * 0.5 * 1.5 ** 2 * np.ones(3)
        self.inputs["mb1_max_defl_ang"] = 0.008
        self.inputs["mb2_max_defl_ang"] = 0.008

        self.inputs["m_stator"] = 100e3
        self.inputs["cm_stator"] = -0.3
        self.inputs["I_stator"] = np.array([1e6, 5e5, 5e5, 0.0, 0.0, 0.0])

        self.inputs["generator_rotor_mass"] = 100e3
        self.inputs["cm_rotor"] = -0.3
        self.inputs["generator_rotor_I"] = np.array([1e6, 5e5, 5e5, 0.0, 0.0, 0.0])

        self.inputs["generator_stator_mass"] = 100e3
        self.inputs["cm_rotor"] = -0.3
        self.inputs["generator_stator_I"] = np.array([1e6, 5e5, 5e5, 0.0, 0.0, 0.0])

        self.inputs["generator_mass"] = 200e3
        self.inputs["generator_I"] = np.array([2e6, 1e6, 1e6, 0.0, 0.0, 0.0])

        self.inputs["gearbox_mass"] = 100e3
        self.inputs["gearbox_I"] = np.array([1e6, 5e5, 5e5])

        self.inputs["brake_mass"] = 10e3
        self.inputs["brake_I"] = np.array([1e4, 5e3, 5e3])

        self.inputs["carrier_mass"] = 10e3
        self.inputs["carrier_I"] = np.array([1e4, 5e3, 5e3])

        self.inputs["gear_ratio"] = 1.0

        self.inputs["F_mb1"] = np.array([2409.750e3, -1716.429e3, 74.3529e3]).reshape((3, 1))
        self.inputs["F_mb2"] = np.array([2409.750e3, -1716.429e3, 74.3529e3]).reshape((3, 1))
        self.inputs["M_mb1"] = np.array([-1.83291e7, 6171.7324e3, 5785.82946e3]).reshape((3, 1))
        self.inputs["M_mb2"] = np.array([-1.83291e7, 6171.7324e3, 5785.82946e3]).reshape((3, 1))

        self.inputs["hub_system_mass"] = 100e3
        self.inputs["hub_system_cm"] = 2.0
        self.inputs["hub_system_I"] = np.array([2409.750e3, -1716.429e3, 74.3529e3, 0.0, 0.0, 0.0])
        self.inputs["F_hub"] = np.array([2409.750e3, 0.0, 74.3529e2]).reshape((3, 1))
        self.inputs["M_hub"] = np.array([-1.83291e4, 6171.7324e2, 5785.82946e2]).reshape((3, 1))

        self.inputs["lss_E"] = self.inputs["hss_E"] = self.inputs["bedplate_E"] = 210e9
        self.inputs["lss_G"] = self.inputs["hss_G"] = self.inputs["bedplate_G"] = 80.8e9
        self.inputs["lss_rho"] = self.inputs["hss_rho"] = self.inputs["bedplate_rho"] = 7850.0
        self.inputs["lss_Xy"] = self.inputs["hss_Xy"] = self.inputs["bedplate_Xy"] = 250e6

        self.opt["gamma_f"] = 1.35
        self.opt["gamma_m"] = 1.3
        self.opt["gamma_n"] = 1.0

    def compute_layout(self, direct=True):
        myobj = lay.DirectLayout() if direct else lay.GearedLayout()
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        for k in self.outputs.keys():
            self.inputs[k] = self.outputs[k]

    def testBaseF_BaseM(self):
        self.inputs["tilt"] = 0.0
        self.inputs["F_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb2"] = np.zeros(3).reshape((3, 1))
        self.compute_layout()
        myobj = ds.Nose_Stator_Bedplate_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][-1], 0.0)
        F0 = self.outputs["base_F"]
        M0 = self.outputs["base_M"]

        self.inputs["other_mass"] += 500e3
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1])
        npt.assert_almost_equal(self.outputs["base_M"][2], 0.0)

        self.inputs["M_mb1"] = 10e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"], M0 + self.inputs["M_mb1"], decimal=0)

        self.inputs["M_mb2"] = 20e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"], M0 + self.inputs["M_mb1"] + self.inputs["M_mb2"], decimal=-1)

        self.inputs["F_mb1"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        self.inputs["F_mb2"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 2 * self.inputs["F_mb2"][:2])
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity + 2 * 50e2)

    def testBaseF_BaseM_withTilt(self):
        self.inputs["tilt"] = 5.0
        self.inputs["F_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb2"] = np.zeros(3).reshape((3, 1))
        self.compute_layout()
        myobj = ds.Nose_Stator_Bedplate_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][-1], 0.0)
        F0 = self.outputs["base_F"]
        M0 = self.outputs["base_M"]

        self.inputs["other_mass"] += 500e3
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1])
        npt.assert_almost_equal(self.outputs["base_M"][2], 0.0)

        self.inputs["M_mb1"] = 10e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1] + self.inputs["M_mb1"][1], decimal=0)

        self.inputs["M_mb2"] = 20e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(
            self.outputs["base_M"][1], M0[1] + self.inputs["M_mb1"][1] + self.inputs["M_mb2"][1], decimal=-1
        )

        self.inputs["F_mb1"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        self.inputs["F_mb2"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][1], 2 * self.inputs["F_mb2"][1])

    def testBaseF_BaseM_Downwind(self):
        self.inputs["tilt"] = 0.0
        self.discrete_inputs["upwind"] = False
        self.inputs["F_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb2"] = np.zeros(3).reshape((3, 1))
        self.compute_layout()
        myobj = ds.Nose_Stator_Bedplate_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][-1], 0.0)
        F0 = self.outputs["base_F"]
        M0 = self.outputs["base_M"]

        self.inputs["other_mass"] += 500e3
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1])
        npt.assert_almost_equal(self.outputs["base_M"][2], 0.0)

        self.inputs["M_mb1"] = 10e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"], M0 + self.inputs["M_mb1"], decimal=0)

        self.inputs["M_mb2"] = 20e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"], M0 + self.inputs["M_mb1"] + self.inputs["M_mb2"], decimal=-1)

        self.inputs["F_mb1"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        self.inputs["F_mb2"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 2 * self.inputs["F_mb2"][:2])
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity + 2 * 50e2)

    def testBaseF_BaseM_withTilt_Downwind(self):
        self.inputs["tilt"] = 5.0
        self.discrete_inputs["upwind"] = False
        self.inputs["F_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb2"] = np.zeros(3).reshape((3, 1))
        self.compute_layout()
        myobj = ds.Nose_Stator_Bedplate_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][-1], 0.0)
        F0 = self.outputs["base_F"]
        M0 = self.outputs["base_M"]

        self.inputs["other_mass"] += 500e3
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1])
        npt.assert_almost_equal(self.outputs["base_M"][2], 0.0)

        self.inputs["M_mb1"] = 10e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1] + self.inputs["M_mb1"][1], decimal=0)

        self.inputs["M_mb2"] = 20e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(
            self.outputs["base_M"][1], M0[1] + self.inputs["M_mb1"][1] + self.inputs["M_mb2"][1], decimal=-1
        )

        self.inputs["F_mb1"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        self.inputs["F_mb2"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][1], 2 * self.inputs["F_mb2"][1])

    def testBaseF_BaseM_Geared(self):
        self.inputs["tilt"] = 0.0
        self.inputs["F_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_torq"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_generator"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_torq"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_generator"] = np.zeros(3).reshape((3, 1))
        self.compute_layout(False)
        myobj = ds.Bedplate_IBeam_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2, 0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_M"][[0, 2], 0], 0.0, decimal=2)
        F0 = self.outputs["base_F"][:, 0]
        M0 = self.outputs["base_M"][:, 0]

        self.inputs["other_mass"] += 500e3
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2, 0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2, 0], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][[0, 2], 0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1])

        self.inputs["M_mb1"] = 10e3 * np.arange(1, 4).reshape((3, 1))
        self.inputs["M_mb2"] = 20e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2, 0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2, 0], F0[2] - 500e3 * gravity, decimal=0)
        # npt.assert_almost_equal(self.outputs['base_M'], M0+self.inputs['M_mb1']+self.inputs['M_mb2'], decimal=-1)

        self.inputs["F_mb1"] = self.inputs["F_mb2"] = self.inputs["F_generator"] = self.inputs["F_torq"] = np.array(
            [30e2, 40e2, 50e2]
        ).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2, 0], 4 * self.inputs["F_mb1"][:2, 0], decimal=1)
        npt.assert_almost_equal(self.outputs["base_F"][2, 0], F0[2] - 500e3 * gravity + 4 * 50e2, decimal=0)

    def testBaseF_BaseM_withTilt_Geared(self):
        self.inputs["tilt"] = 5.0
        self.inputs["F_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_torq"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_generator"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_torq"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_generator"] = np.zeros(3).reshape((3, 1))
        self.compute_layout(False)
        myobj = ds.Bedplate_IBeam_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2, 0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_M"][[0, 2], 0], 0.0, decimal=2)
        F0 = self.outputs["base_F"][:, 0]
        M0 = self.outputs["base_M"][:, 0]

        self.inputs["other_mass"] += 500e3
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2, 0], 0.0, decimal=1)
        npt.assert_almost_equal(self.outputs["base_F"][2, 0], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][[0, 2], 0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1])

        self.inputs["M_mb1"] = 10e3 * np.arange(1, 4).reshape((3, 1))
        self.inputs["M_mb2"] = 20e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2, 0], 0.0, decimal=1)
        npt.assert_almost_equal(self.outputs["base_F"][2, 0], F0[2] - 500e3 * gravity, decimal=0)
        # npt.assert_almost_equal(self.outputs['base_M'], M0+self.inputs['M_mb1']+self.inputs['M_mb2'], decimal=-1)

        self.inputs["F_mb1"] = self.inputs["F_mb2"] = self.inputs["F_generator"] = self.inputs["F_torq"] = np.array(
            [30e2, 40e2, 50e2]
        ).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][1, 0], 4 * self.inputs["F_mb1"][1, 0], decimal=1)

    def testRunRotatingDirect_noTilt(self):
        self.inputs["tilt"] = 0.0
        self.inputs["F_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_hub"] = np.zeros(3).reshape((3, 1))
        self.compute_layout()
        myobj = ds.Hub_Rotor_LSS_Frame(n_dlcs=1, modeling_options=self.opt, direct_drive=True)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        F0 = self.outputs["F_mb1"].flatten()
        M0 = self.outputs["M_mb2"].flatten()
        self.assertGreater(0.0, F0[-1])
        # self.assertGreater(0.0, M0[1])
        npt.assert_almost_equal(self.outputs["F_mb1"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"][[0, 2]], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_torq"], 0.0, decimal=2)
        self.assertAlmostEqual(
            self.outputs["lss_spring_constant"], 80.8e9 * np.pi * (3.3 ** 4 - 2.4 ** 4) / 32 / self.inputs["L_lss"], 4
        )

        g = np.array([30e2, 40e2, 50e2])
        self.inputs["F_hub"] = g.reshape((3, 1))
        self.inputs["M_hub"] = 2 * g.reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["F_mb1"].flatten(), g + F0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[1], g[-1] * 1 + 2 * g[1] + M0[1], decimal=1)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[2], -g[1] * 1 + 2 * g[2], decimal=1)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_torq"].flatten(), np.r_[2 * g[0], 0.0, 0.0], decimal=2)

        r = 0.5 * 3.3
        self.assertAlmostEqual(self.outputs["lss_axial_load2stress"][0], 1.0 / (np.pi * (r ** 2 - (r - 0.45) ** 2)))
        npt.assert_almost_equal(self.outputs["lss_axial_load2stress"][1:4], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_axial_load2stress"][4:], r / (0.25 * np.pi * (r ** 4 - (r - 0.45) ** 4))
        )
        self.assertAlmostEqual(self.outputs["lss_shear_load2stress"][0], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][1], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][2], 0.0)
        npt.assert_almost_equal(self.outputs["lss_shear_load2stress"][4:], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_shear_load2stress"][3], r / (0.5 * np.pi * (r ** 4 - (r - 0.45) ** 4))
        )

    def testRunRotatingDirect_withTilt(self):
        self.inputs["tilt"] = 5.0
        self.inputs["F_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_hub"] = np.zeros(3).reshape((3, 1))
        self.compute_layout()
        myobj = ds.Hub_Rotor_LSS_Frame(n_dlcs=1, modeling_options=self.opt, direct_drive=True)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        F0 = self.outputs["F_mb1"].flatten()
        M0 = self.outputs["M_mb2"].flatten()
        self.assertGreater(0.0, F0[0])
        self.assertGreater(0.0, F0[-1])
        # self.assertGreater(0.0, M0[1])
        npt.assert_almost_equal(self.outputs["F_mb1"][1], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"][[0, 2]], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_torq"], 0.0, decimal=2)
        self.assertAlmostEqual(
            self.outputs["lss_spring_constant"], 80.8e9 * np.pi * (3.3 ** 4 - 2.4 ** 4) / 32 / self.inputs["L_lss"], 4
        )

        g = np.array([30e2, 40e2, 50e2])
        self.inputs["F_hub"] = g.reshape((3, 1))
        self.inputs["M_hub"] = 2 * g.reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["F_mb1"].flatten(), g + F0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[1], g[-1] * 1 + 2 * g[1] + M0[1], decimal=1)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[2], -g[1] * 1 + 2 * g[2], decimal=1)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_torq"].flatten(), np.r_[2 * g[0], 0.0, 0.0], decimal=2)

        r = 0.5 * 3.3
        self.assertAlmostEqual(self.outputs["lss_axial_load2stress"][0], 1.0 / (np.pi * (r ** 2 - (r - 0.45) ** 2)))
        npt.assert_almost_equal(self.outputs["lss_axial_load2stress"][1:4], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_axial_load2stress"][4:], r / (0.25 * np.pi * (r ** 4 - (r - 0.45) ** 4))
        )
        self.assertAlmostEqual(self.outputs["lss_shear_load2stress"][0], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][1], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][2], 0.0)
        npt.assert_almost_equal(self.outputs["lss_shear_load2stress"][4:], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_shear_load2stress"][3], r / (0.5 * np.pi * (r ** 4 - (r - 0.45) ** 4))
        )

    def testRunRotatingGeared_noTilt(self):
        self.inputs["tilt"] = 0.0
        self.inputs["gear_ratio"] = 50.0
        self.inputs["F_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_hub"] = np.zeros(3).reshape((3, 1))
        self.compute_layout(False)
        myobj = ds.Hub_Rotor_LSS_Frame(n_dlcs=1, modeling_options=self.opt, direct_drive=False)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        F0 = self.outputs["F_mb1"].flatten()
        M0 = self.outputs["M_mb2"].flatten()
        self.assertGreater(0.0, F0[-1])
        # self.assertGreater(0.0, M0[1])
        npt.assert_almost_equal(self.outputs["F_mb1"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"][[0, 2]], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_torq"], 0.0, decimal=2)
        self.assertAlmostEqual(
            self.outputs["lss_spring_constant"], 80.8e9 * np.pi * (3.3 ** 4 - 2.4 ** 4) / 32 / self.inputs["L_lss"], 4
        )

        g = np.array([30e2, 40e2, 50e2])
        self.inputs["F_hub"] = g.reshape((3, 1))
        self.inputs["M_hub"] = 2 * g.reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["F_mb1"].flatten(), g + F0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[1], g[-1] * 1 + 2 * g[1] + M0[1], decimal=2)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[2], -g[1] * 1 + 2 * g[2], decimal=2)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_torq"].flatten(), np.r_[2 * g[0], 0.0, 0.0], decimal=2)

        r = 0.5 * 3.3
        self.assertAlmostEqual(self.outputs["lss_axial_load2stress"][0], 1.0 / (np.pi * (r ** 2 - (r - 0.45) ** 2)))
        npt.assert_almost_equal(self.outputs["lss_axial_load2stress"][1:4], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_axial_load2stress"][4:], r / (0.25 * np.pi * (r ** 4 - (r - 0.45) ** 4))
        )
        self.assertAlmostEqual(self.outputs["lss_shear_load2stress"][0], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][1], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][2], 0.0)
        npt.assert_almost_equal(self.outputs["lss_shear_load2stress"][4:], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_shear_load2stress"][3], r / (0.5 * np.pi * (r ** 4 - (r - 0.45) ** 4))
        )

    def testRunRotatingGeared_withTilt(self):
        self.inputs["tilt"] = 5.0
        self.inputs["gear_ratio"] = 50.0
        self.inputs["F_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_hub"] = np.zeros(3).reshape((3, 1))
        self.compute_layout(False)
        myobj = ds.Hub_Rotor_LSS_Frame(n_dlcs=1, modeling_options=self.opt, direct_drive=False)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        F0 = self.outputs["F_mb1"].flatten()
        M0 = self.outputs["M_mb2"].flatten()
        self.assertGreater(0.0, F0[0])
        self.assertGreater(0.0, F0[-1])
        # self.assertGreater(0.0, M0[1])
        npt.assert_almost_equal(self.outputs["F_mb1"][1], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"][[0, 2]], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_torq"], 0.0, decimal=2)
        self.assertAlmostEqual(
            self.outputs["lss_spring_constant"], 80.8e9 * np.pi * (3.3 ** 4 - 2.4 ** 4) / 32 / self.inputs["L_lss"], 4
        )

        g = np.array([30e2, 40e2, 50e2])
        self.inputs["F_hub"] = g.reshape((3, 1))
        self.inputs["M_hub"] = 2 * g.reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["F_mb1"].flatten(), g + F0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[1], g[-1] * 1 + 2 * g[1] + M0[1], decimal=2)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[2], -g[1] * 1 + 2 * g[2], decimal=2)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_torq"].flatten(), np.r_[2 * g[0], 0.0, 0.0], decimal=2)

        r = 0.5 * 3.3
        self.assertAlmostEqual(self.outputs["lss_axial_load2stress"][0], 1.0 / (np.pi * (r ** 2 - (r - 0.45) ** 2)))
        npt.assert_almost_equal(self.outputs["lss_axial_load2stress"][1:4], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_axial_load2stress"][4:], r / (0.25 * np.pi * (r ** 4 - (r - 0.45) ** 4))
        )
        self.assertAlmostEqual(self.outputs["lss_shear_load2stress"][0], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][1], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][2], 0.0)
        npt.assert_almost_equal(self.outputs["lss_shear_load2stress"][4:], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_shear_load2stress"][3], r / (0.5 * np.pi * (r ** 4 - (r - 0.45) ** 4))
        )

    def testHSS_noTilt(self):
        self.inputs["tilt"] = 0.0
        self.inputs["gear_ratio"] = 50.0
        self.inputs["F_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_hub"] = np.zeros(3).reshape((3, 1))
        self.compute_layout(False)
        myobj = ds.HSS_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs)
        F0 = self.outputs["F_generator"].flatten()
        M0 = self.outputs["M_generator"].flatten()
        self.assertGreater(0.0, F0[-1])
        self.assertGreater(0.0, M0[1])
        npt.assert_almost_equal(self.outputs["F_generator"].flatten()[:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_generator"].flatten()[[0, 2]], 0.0, decimal=2)

        g = np.array([30e2, 40e2, 50e2])
        self.inputs["F_hub"] = g.reshape((3, 1))
        self.inputs["M_hub"] = 2 * g.reshape((3, 1))
        self.compute_layout(False)
        myobj = ds.HSS_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs)
        npt.assert_almost_equal(self.outputs["F_generator"].flatten(), F0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_generator"].flatten(), np.r_[2 * g[0] / 50.0, M0[1], 0.0], decimal=2)

    def testHSS_withTilt(self):
        self.inputs["tilt"] = 5.0
        self.inputs["gear_ratio"] = 50.0
        self.inputs["F_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_hub"] = np.zeros(3).reshape((3, 1))
        self.compute_layout(False)
        myobj = ds.HSS_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs)
        F0 = self.outputs["F_generator"].flatten()
        M0 = self.outputs["M_generator"].flatten()
        self.assertGreater(0.0, F0[0])
        self.assertGreater(0.0, F0[-1])
        self.assertGreater(0.0, M0[1])
        npt.assert_almost_equal(self.outputs["F_generator"].flatten()[1], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_generator"].flatten()[[0, 2]], 0.0, decimal=2)

        g = np.array([30e2, 40e2, 50e2])
        self.inputs["F_hub"] = g.reshape((3, 1))
        self.inputs["M_hub"] = 2 * g.reshape((3, 1))
        self.compute_layout(False)
        myobj = ds.HSS_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs)
        npt.assert_almost_equal(self.outputs["F_generator"].flatten(), F0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_generator"].flatten(), np.r_[2 * g[0] / 50.0, M0[1], 0.0], decimal=2)

    def testShaftTheoryLSS(self):
        # https://www.engineersedge.com/calculators/torsional-stress-calculator.htm
        self.inputs["tilt"] = 0.0
        self.inputs["F_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_hub"] = np.array([1e5, 0.0, 0.0]).reshape((3, 1))
        self.inputs["brake_mass"] = 0.0
        self.inputs["brake_I"] = np.zeros(3)
        self.inputs["generator_rotor_mass"] = 0.0
        self.inputs["cm_rotor"] = 0.0
        self.inputs["generator_rotor_I"] = np.zeros(6)
        self.inputs["hub_system_mass"] = 0.0
        self.inputs["hub_system_cm"] = 0.0
        self.inputs["hub_system_I"] = np.zeros(6)
        myones = np.ones(5)
        self.inputs["lss_diameter"] = 5 * myones
        self.inputs["lss_wall_thickness"] = 0.5 * myones
        self.inputs["G"] = 100e9
        self.inputs["lss_rho"] = 1e-6
        self.compute_layout()
        myobj = ds.Hub_Rotor_LSS_Frame(n_dlcs=1, modeling_options=self.opt, direct_drive=True)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        J = 0.5 * np.pi * (2.5 ** 4 - 2 ** 4)
        sigma = 1e5 / J * 2.5
        npt.assert_almost_equal(self.outputs["lss_axial_stress"], 0.0, decimal=4)
        npt.assert_almost_equal(self.outputs["lss_shear_stress"].flatten(), np.r_[np.zeros(3), sigma], decimal=4)

    def testShaftTheoryHSS(self):
        # https://www.engineersedge.com/calculators/torsional-stress-calculator.htm
        self.inputs["tilt"] = 0.0
        self.inputs["gear_ratio"] = 50.0
        self.inputs["s_hss"] = np.array([0.0, 0.5, 1.0])
        self.inputs["M_hub"] = np.array([1e5, 0.0, 0.0]).reshape((3, 1))
        self.inputs["s_generator"] = 0.0
        self.inputs["generator_mass"] = 0.0
        self.inputs["generator_I"] = np.zeros(3)
        self.inputs["brake_mass"] = 0.0
        self.inputs["brake_I"] = np.zeros(3)
        self.inputs["hub_system_mass"] = 0.0
        self.inputs["hub_system_cm"] = 0.0
        self.inputs["hub_system_I"] = np.zeros(6)
        myones = np.ones(3)
        self.inputs["hss_diameter"] = 5 * myones
        self.inputs["hss_wall_thickness"] = 0.5 * myones
        self.inputs["G"] = 100e9
        self.inputs["hss_rho"] = 1e-6
        self.compute_layout()
        myobj = ds.HSS_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs)
        J = 0.5 * np.pi * (2.5 ** 4 - 2 ** 4)
        sigma = 1e5 / 50.0 / J * 2.5
        npt.assert_almost_equal(self.outputs["hss_axial_stress"], 0.0, decimal=4)
        npt.assert_almost_equal(self.outputs["hss_bending_stress"], 0.0, decimal=4)
        npt.assert_almost_equal(self.outputs["hss_shear_stress"].flatten(), sigma * np.ones(2), decimal=4)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDirectStructure))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
