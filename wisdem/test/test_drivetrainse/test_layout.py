import unittest

import numpy as np
import numpy.testing as npt
import wisdem.drivetrainse.layout as lay

npts = 12
ct = np.cos(np.deg2rad(5))
st = np.sin(np.deg2rad(5))


class TestDirectLayout(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        self.inputs["L_12"] = 2.0
        self.inputs["L_h1"] = 1.0
        self.inputs["L_generator"] = 3.25
        # self.inputs['L_2n'] = 1.5
        # self.inputs['L_grs'] = 1.1
        # self.inputs['L_gsn'] = 1.1
        self.inputs["overhang"] = 6.25 + 0.5 * 6.5 + 2
        self.inputs["drive_height"] = 4.875
        self.inputs["tilt"] = 5.0
        self.inputs["access_diameter"] = 0.9

        myones = np.ones(2)
        self.inputs["lss_diameter"] = 2.3 * myones
        self.inputs["nose_diameter"] = 1.33 * myones
        self.inputs["lss_wall_thickness"] = 0.05 * myones
        self.inputs["nose_wall_thickness"] = 0.04 * myones

        self.inputs["bedplate_wall_thickness"] = 0.06 * np.ones(4)
        self.inputs["D_top"] = 6.5
        self.inputs["hub_diameter"] = 4.0

        self.inputs["lss_rho"] = self.inputs["bedplate_rho"] = 7850.0

        self.discrete_inputs["upwind"] = True

    def testBedplateLengthHeight(self):
        self.inputs["tilt"] = 0.0
        myobj = lay.DirectLayout()
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertAlmostEqual(self.outputs["L_nose"], 3.5)
        self.assertAlmostEqual(self.outputs["L_lss"], 3.0)
        self.assertAlmostEqual(self.outputs["L_drive"], 4.5)
        self.assertAlmostEqual(self.outputs["L_bedplate"], 5.0)
        self.assertAlmostEqual(self.outputs["H_bedplate"], 4.875)
        self.assertAlmostEqual(self.outputs["constr_length"], 5 - 0.5 * 6.5)
        self.assertAlmostEqual(self.outputs["constr_height"], 4.875)

        self.inputs["overhang"] = 2.0 + 0.5 * 6.5 + 2
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertAlmostEqual(self.outputs["L_nose"], 3.5)
        self.assertAlmostEqual(self.outputs["L_lss"], 3.0)
        self.assertAlmostEqual(self.outputs["L_drive"], 4.5)
        self.assertAlmostEqual(self.outputs["L_bedplate"], 0.5 * 6.5)
        self.assertAlmostEqual(self.outputs["H_bedplate"], 4.875)
        self.assertAlmostEqual(self.outputs["constr_length"], -2.5)
        self.assertAlmostEqual(self.outputs["constr_height"], 4.875)

    def testNoTiltUpwind(self):
        self.inputs["tilt"] = 0.0
        myobj = lay.DirectLayout()
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        self.assertAlmostEqual(self.outputs["L_nose"], 3.5)
        self.assertAlmostEqual(self.outputs["L_lss"], 3.0)
        self.assertAlmostEqual(self.outputs["L_drive"], 4.5)
        self.assertAlmostEqual(self.outputs["L_bedplate"], self.inputs["overhang"] - self.outputs["L_drive"] - 2)
        self.assertAlmostEqual(self.outputs["H_bedplate"], self.inputs["drive_height"])
        self.assertAlmostEqual(self.outputs["D_bearing1"], 2.3 - 0.05 - 1.33)
        self.assertAlmostEqual(self.outputs["D_bearing2"], 2.3 - 0.05 - 1.33)

        npt.assert_equal(self.outputs["constr_access"][:, -1], 1.33 - 0.08 - 0.9)
        npt.assert_equal(self.outputs["constr_access"][:, 0], 2.3 - 0.1 - 1.33 - 0.25 * 0.9)
        self.assertAlmostEqual(self.outputs["constr_length"], 5 - 0.5 * 6.5)
        self.assertAlmostEqual(self.outputs["constr_height"], self.outputs["H_bedplate"])

        self.assertAlmostEqual(self.outputs["s_rotor"], 2 + 1.5 + 0.5)
        self.assertAlmostEqual(self.outputs["s_stator"], 0.75)
        self.assertAlmostEqual(self.outputs["s_mb1"], 1.5 + 2.0)
        self.assertAlmostEqual(self.outputs["s_mb2"], 1.5)

        self.assertAlmostEqual(self.outputs["x_bedplate"][-1], -5.0)
        self.assertAlmostEqual(self.outputs["x_bedplate_inner"][-1], -5.0)
        self.assertAlmostEqual(self.outputs["x_bedplate_outer"][-1], -5.0)

        self.assertAlmostEqual(self.outputs["x_bedplate"][0], 0.0)
        self.assertAlmostEqual(self.outputs["x_bedplate_inner"][0], -0.5 * 6.5)
        self.assertAlmostEqual(self.outputs["x_bedplate_outer"][0], 0.5 * 6.5)
        self.assertAlmostEqual(self.outputs["D_bedplate"][0], 6.5)

        self.assertAlmostEqual(self.outputs["z_bedplate"][0], 0.0)
        self.assertAlmostEqual(self.outputs["z_bedplate_inner"][0], 0.0)
        self.assertAlmostEqual(self.outputs["z_bedplate_outer"][0], 0.0)

        self.assertAlmostEqual(self.outputs["z_bedplate"][-1], 4.875)
        self.assertAlmostEqual(self.outputs["z_bedplate_inner"][-1], 4.875 - 0.5 * 1.33)
        self.assertAlmostEqual(self.outputs["z_bedplate_outer"][-1], 4.875 + 0.5 * 1.33)
        self.assertAlmostEqual(self.outputs["D_bedplate"][-1], 1.33)

    def testTiltUpwind(self):
        self.inputs["tilt"] = 5.0
        self.inputs["overhang"] = 5 + (2 + 4.5) * ct
        self.inputs["drive_height"] = 4.875 + (2 + 4.5) * st
        myobj = lay.DirectLayout()
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        self.assertAlmostEqual(self.outputs["L_nose"], 3.5)
        self.assertAlmostEqual(self.outputs["L_lss"], 3.0)
        self.assertAlmostEqual(self.outputs["L_drive"], 4.5)
        self.assertAlmostEqual(self.outputs["L_bedplate"], self.inputs["overhang"] - (2 + self.outputs["L_drive"]) * ct)
        self.assertAlmostEqual(
            self.outputs["H_bedplate"], self.inputs["drive_height"] - (2 + self.outputs["L_drive"]) * st
        )
        self.assertAlmostEqual(self.outputs["D_bearing1"], 2.3 - 0.05 - 1.33)
        self.assertAlmostEqual(self.outputs["D_bearing2"], 2.3 - 0.05 - 1.33)

        npt.assert_equal(self.outputs["constr_access"][:, -1], 1.33 - 0.08 - 0.9)
        npt.assert_equal(self.outputs["constr_access"][:, 0], 2.3 - 0.1 - 1.33 - 0.25 * 0.9)
        self.assertAlmostEqual(
            self.outputs["constr_length"],
            self.inputs["overhang"] - (2 + self.outputs["L_drive"]) * ct - 0.5 * self.inputs["D_top"],
        )
        self.assertAlmostEqual(self.outputs["constr_height"], self.outputs["H_bedplate"])

        self.assertAlmostEqual(self.outputs["s_rotor"], 2 + 1.5 + 0.5)
        self.assertAlmostEqual(self.outputs["s_stator"], 0.75)
        self.assertAlmostEqual(self.outputs["s_mb1"], 1.5 + 2.0)
        self.assertAlmostEqual(self.outputs["s_mb2"], 1.5)

        self.assertAlmostEqual(self.outputs["x_bedplate"][-1], -5.0)
        self.assertAlmostEqual(self.outputs["x_bedplate_inner"][-1], -5.0)
        self.assertAlmostEqual(self.outputs["x_bedplate_outer"][-1], -5.0)

        self.assertAlmostEqual(self.outputs["x_bedplate"][0], 0.0)
        self.assertAlmostEqual(self.outputs["x_bedplate_inner"][0], -0.5 * 6.5)
        self.assertAlmostEqual(self.outputs["x_bedplate_outer"][0], 0.5 * 6.5)
        self.assertAlmostEqual(self.outputs["D_bedplate"][0], 6.5)

        self.assertAlmostEqual(self.outputs["z_bedplate"][0], 0.0)
        self.assertAlmostEqual(self.outputs["z_bedplate_inner"][0], 0.0)
        self.assertAlmostEqual(self.outputs["z_bedplate_outer"][0], 0.0)

        self.assertAlmostEqual(self.outputs["z_bedplate"][-1], 4.875)
        self.assertAlmostEqual(self.outputs["z_bedplate_inner"][-1], 4.875 - 0.5 * 1.33)
        self.assertAlmostEqual(self.outputs["z_bedplate_outer"][-1], 4.875 + 0.5 * 1.33)
        self.assertAlmostEqual(self.outputs["D_bedplate"][-1], 1.33)

    def testNoTiltDownwind(self):
        self.discrete_inputs["upwind"] = False
        self.inputs["tilt"] = 0.0
        myobj = lay.DirectLayout()
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        self.assertAlmostEqual(self.outputs["L_nose"], 3.5)
        self.assertAlmostEqual(self.outputs["L_lss"], 3.0)
        self.assertAlmostEqual(self.outputs["L_drive"], 4.5)
        self.assertAlmostEqual(self.outputs["L_bedplate"], self.inputs["overhang"] - self.outputs["L_drive"] - 2)
        self.assertAlmostEqual(self.outputs["H_bedplate"], self.inputs["drive_height"])
        self.assertAlmostEqual(self.outputs["D_bearing1"], 2.3 - 0.05 - 1.33)
        self.assertAlmostEqual(self.outputs["D_bearing2"], 2.3 - 0.05 - 1.33)

        npt.assert_equal(self.outputs["constr_access"][:, -1], 1.33 - 0.08 - 0.9)
        npt.assert_equal(self.outputs["constr_access"][:, 0], 2.3 - 0.1 - 1.33 - 0.25 * 0.9)
        self.assertAlmostEqual(self.outputs["constr_length"], 5 - 0.5 * 6.5)
        self.assertAlmostEqual(self.outputs["constr_height"], self.outputs["H_bedplate"])

        self.assertAlmostEqual(self.outputs["s_rotor"], 2 + 1.5 + 0.5)
        self.assertAlmostEqual(self.outputs["s_stator"], 0.75)
        self.assertAlmostEqual(self.outputs["s_mb1"], 1.5 + 2.0)
        self.assertAlmostEqual(self.outputs["s_mb2"], 1.5)

        self.assertAlmostEqual(self.outputs["x_bedplate"][-1], 5.0)
        self.assertAlmostEqual(self.outputs["x_bedplate_inner"][-1], 5.0)
        self.assertAlmostEqual(self.outputs["x_bedplate_outer"][-1], 5.0)

        self.assertAlmostEqual(self.outputs["x_bedplate"][0], 0.0)
        self.assertAlmostEqual(self.outputs["x_bedplate_inner"][0], 0.5 * 6.5)
        self.assertAlmostEqual(self.outputs["x_bedplate_outer"][0], -0.5 * 6.5)
        self.assertAlmostEqual(self.outputs["D_bedplate"][0], 6.5)

        self.assertAlmostEqual(self.outputs["z_bedplate"][0], 0.0)
        self.assertAlmostEqual(self.outputs["z_bedplate_inner"][0], 0.0)
        self.assertAlmostEqual(self.outputs["z_bedplate_outer"][0], 0.0)

        self.assertAlmostEqual(self.outputs["z_bedplate"][-1], 4.875)
        self.assertAlmostEqual(self.outputs["z_bedplate_inner"][-1], 4.875 - 0.5 * 1.33)
        self.assertAlmostEqual(self.outputs["z_bedplate_outer"][-1], 4.875 + 0.5 * 1.33)
        self.assertAlmostEqual(self.outputs["D_bedplate"][-1], 1.33)

    def testTiltDownwind(self):
        self.discrete_inputs["upwind"] = False
        self.inputs["tilt"] = 5.0
        self.inputs["overhang"] = 5 + (2 + 4.5) * ct
        self.inputs["drive_height"] = 4.875 + (2 + 4.5) * st
        myobj = lay.DirectLayout()
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        self.assertAlmostEqual(self.outputs["L_nose"], 3.5)
        self.assertAlmostEqual(self.outputs["L_lss"], 3.0)
        self.assertAlmostEqual(self.outputs["L_drive"], 4.5)
        self.assertAlmostEqual(self.outputs["L_bedplate"], self.inputs["overhang"] - (2 + self.outputs["L_drive"]) * ct)
        self.assertAlmostEqual(
            self.outputs["H_bedplate"], self.inputs["drive_height"] - (2 + self.outputs["L_drive"]) * st
        )
        self.assertAlmostEqual(self.outputs["D_bearing1"], 2.3 - 0.05 - 1.33)
        self.assertAlmostEqual(self.outputs["D_bearing2"], 2.3 - 0.05 - 1.33)

        npt.assert_equal(self.outputs["constr_access"][:, -1], 1.33 - 0.08 - 0.9)
        npt.assert_equal(self.outputs["constr_access"][:, 0], 2.3 - 0.1 - 1.33 - 0.25 * 0.9)
        self.assertAlmostEqual(
            self.outputs["constr_length"],
            self.inputs["overhang"] - (2 + self.outputs["L_drive"]) * ct - 0.5 * self.inputs["D_top"],
        )
        self.assertAlmostEqual(self.outputs["constr_height"], self.outputs["H_bedplate"])

        self.assertAlmostEqual(self.outputs["s_rotor"], 2 + 1.5 + 0.5)
        self.assertAlmostEqual(self.outputs["s_stator"], 0.75)
        self.assertAlmostEqual(self.outputs["s_mb1"], 1.5 + 2.0)
        self.assertAlmostEqual(self.outputs["s_mb2"], 1.5)

        self.assertAlmostEqual(self.outputs["x_bedplate"][-1], 5.0)
        self.assertAlmostEqual(self.outputs["x_bedplate_inner"][-1], 5.0)
        self.assertAlmostEqual(self.outputs["x_bedplate_outer"][-1], 5.0)

        self.assertAlmostEqual(self.outputs["x_bedplate"][0], 0.0)
        self.assertAlmostEqual(self.outputs["x_bedplate_inner"][0], 0.5 * 6.5)
        self.assertAlmostEqual(self.outputs["x_bedplate_outer"][0], -0.5 * 6.5)
        self.assertAlmostEqual(self.outputs["D_bedplate"][0], 6.5)

        self.assertAlmostEqual(self.outputs["z_bedplate"][0], 0.0)
        self.assertAlmostEqual(self.outputs["z_bedplate_inner"][0], 0.0)
        self.assertAlmostEqual(self.outputs["z_bedplate_outer"][0], 0.0)

        self.assertAlmostEqual(self.outputs["z_bedplate"][-1], 4.875)
        self.assertAlmostEqual(self.outputs["z_bedplate_inner"][-1], 4.875 - 0.5 * 1.33)
        self.assertAlmostEqual(self.outputs["z_bedplate_outer"][-1], 4.875 + 0.5 * 1.33)
        self.assertAlmostEqual(self.outputs["D_bedplate"][-1], 1.33)

    def testMassValues(self):
        self.discrete_inputs["upwind"] = True
        self.inputs["tilt"] = 0.0
        self.inputs["drive_height"] = 5.0
        self.inputs["D_top"] = 3.0
        self.inputs["overhang"] = 4.5 + 3.5 + 0.5 * 3.0 + 2
        myones = np.ones(5)
        self.inputs["lss_diameter"] = 2.0 * myones
        self.inputs["nose_diameter"] = 3.0 * myones
        self.inputs["lss_wall_thickness"] = 0.05 * myones
        self.inputs["nose_wall_thickness"] = 0.05 * myones
        self.inputs["bedplate_wall_thickness"] = 0.05 * np.ones(npts)
        myobj = lay.DirectLayout()
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        rho = self.inputs["lss_rho"]
        m_bedplate = 5 * 0.5 * np.pi * np.pi * (1.5 ** 2 - (1.5 - 0.05) ** 2) * rho
        self.assertAlmostEqual(self.outputs["bedplate_mass"], m_bedplate)
        self.assertAlmostEqual(self.outputs["bedplate_cm"][0], np.mean(self.outputs["x_bedplate"]), 0)
        self.assertAlmostEqual(self.outputs["bedplate_cm"][1], 0.0)
        self.assertAlmostEqual(self.outputs["bedplate_cm"][2], np.mean(self.outputs["z_bedplate"]), 0)

        m_lss = rho * np.pi * (1 ** 2 - 0.95 ** 2) * self.outputs["L_lss"]
        self.assertAlmostEqual(self.outputs["lss_mass"], m_lss)
        self.assertAlmostEqual(self.outputs["lss_cm"], 0.5 * (self.outputs["s_lss"][0] + self.outputs["s_lss"][-1]))
        self.assertAlmostEqual(self.outputs["lss_I"][0], 0.5 * m_lss * (1 ** 2 + 0.95 ** 2))
        self.assertAlmostEqual(
            self.outputs["lss_I"][1], (1 / 12) * m_lss * (3 * (1 ** 2 + 0.95 ** 2) + self.outputs["L_lss"] ** 2)
        )

        m_nose = rho * np.pi * (1.5 ** 2 - 1.45 ** 2) * self.outputs["L_nose"]
        self.assertAlmostEqual(self.outputs["nose_mass"], m_nose)
        self.assertAlmostEqual(self.outputs["nose_cm"], 0.5 * (self.outputs["s_nose"][0] + self.outputs["s_nose"][-1]))
        self.assertAlmostEqual(self.outputs["nose_I"][0], 0.5 * m_nose * (1.5 ** 2 + 1.45 ** 2))
        self.assertAlmostEqual(
            self.outputs["nose_I"][1], (1 / 12) * m_nose * (3 * (1.5 ** 2 + 1.45 ** 2) + self.outputs["L_nose"] ** 2)
        )


class TestGearedLayout(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        self.inputs["L_12"] = 2.0
        self.inputs["L_h1"] = 1.0
        self.inputs["overhang"] = 2.0 + 2.0
        self.inputs["drive_height"] = 4.875
        self.inputs["L_hss"] = 1.5
        self.inputs["L_generator"] = 1.25
        self.inputs["L_gearbox"] = 1.1
        self.inputs["tilt"] = 5.0

        myones = np.ones(2)
        self.inputs["lss_diameter"] = 2.3 * myones
        self.inputs["lss_wall_thickness"] = 0.05 * myones
        self.inputs["hss_diameter"] = 2.0 * myones
        self.inputs["hss_wall_thickness"] = 0.05 * myones

        self.inputs["bedplate_flange_width"] = 1.5
        self.inputs["bedplate_flange_thickness"] = 0.05
        # self.inputs['bedplate_web_height'] = 1.0
        self.inputs["bedplate_web_thickness"] = 0.05

        self.inputs["D_top"] = 6.5
        self.inputs["hub_diameter"] = 4.0

        self.inputs["lss_rho"] = self.inputs["hss_rho"] = self.inputs["bedplate_rho"] = 7850.0

        self.discrete_inputs["upwind"] = True

    def testNoTilt(self):
        self.inputs["tilt"] = 0.0
        myobj = lay.GearedLayout()
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        ds = 6.95 - 2
        self.assertAlmostEqual(self.outputs["L_lss"], 3.1)
        self.assertAlmostEqual(self.outputs["L_drive"], 6.95)
        npt.assert_almost_equal(
            self.outputs["s_drive"],
            np.array([0.0, 0.625, 1.25, 2.0, 2.75, 3.3, 3.85, 3.95, 4.95, 5.95, 6.45, 6.95]) - ds,
        )
        self.assertAlmostEqual(self.outputs["s_generator"], 0.625 - ds)
        self.assertAlmostEqual(self.outputs["s_gearbox"], 3.3 - ds)
        self.assertAlmostEqual(self.outputs["s_mb1"], 5.95 - ds)
        self.assertAlmostEqual(self.outputs["s_mb2"], 3.95 - ds)
        self.assertAlmostEqual(self.outputs["L_bedplate"], 6.95)
        self.assertAlmostEqual(self.outputs["H_bedplate"], 4.875)
        self.assertAlmostEqual(self.outputs["bedplate_web_height"], 4.725)
        self.assertAlmostEqual(self.outputs["constr_length"], 6.95 - 2 - 0.5 * 6.5)
        self.assertAlmostEqual(self.outputs["constr_height"], 4.875)

    def testTilt(self):
        myobj = lay.GearedLayout()
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        ds = 6.95 + 2 - 4 / ct
        self.assertAlmostEqual(self.outputs["L_lss"], 3.1)
        self.assertAlmostEqual(self.outputs["L_drive"], 6.95)
        npt.assert_almost_equal(
            self.outputs["s_drive"],
            np.array([0.0, 0.625, 1.25, 2.0, 2.75, 3.3, 3.85, 3.95, 4.95, 5.95, 6.45, 6.95]) - ds,
        )
        self.assertAlmostEqual(self.outputs["s_generator"], 0.625 - ds)
        self.assertAlmostEqual(self.outputs["s_gearbox"], 3.3 - ds)
        self.assertAlmostEqual(self.outputs["s_mb1"], 5.95 - ds)
        self.assertAlmostEqual(self.outputs["s_mb2"], 3.95 - ds)
        self.assertAlmostEqual(self.outputs["L_bedplate"], 6.95 * ct)
        self.assertAlmostEqual(self.outputs["H_bedplate"], 4.875 - (2 + 6.95) * st)
        self.assertAlmostEqual(self.outputs["bedplate_web_height"], 4.725 - (2 + 6.95) * st)
        self.assertAlmostEqual(self.outputs["constr_length"], (2 + 6.95) * ct - 2 - 2 - 0.5 * 6.5)
        self.assertAlmostEqual(self.outputs["constr_height"], 4.875 - (2 + 6.95) * st)

    def testMassValues(self):
        self.inputs["tilt"] = 0.0
        self.discrete_inputs["upwind"] = True
        myones = np.ones(5)
        self.inputs["lss_diameter"] = 2.0 * myones
        self.inputs["lss_wall_thickness"] = 0.05 * myones
        myones = np.ones(3)
        self.inputs["hss_diameter"] = 1.5 * myones
        self.inputs["hss_wall_thickness"] = 0.04 * myones
        myobj = lay.GearedLayout()
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        rho = self.inputs["lss_rho"]
        m_bedplate = 2 * rho * (2 * 1.5 * 0.05 + 4.725 * 0.05) * 6.95
        self.assertAlmostEqual(self.outputs["bedplate_mass"], m_bedplate)
        npt.assert_almost_equal(self.outputs["bedplate_cm"], np.r_[0.5 * 6.95 - 2 - 2.0, 0.0, 0.5 * 4.725 + 0.05])

        m_lss = rho * np.pi * (1 ** 2 - 0.95 ** 2) * self.outputs["L_lss"]
        self.assertAlmostEqual(self.outputs["lss_mass"], m_lss)
        self.assertAlmostEqual(self.outputs["lss_cm"], 0.5 * (self.outputs["s_lss"][0] + self.outputs["s_lss"][-1]))
        self.assertAlmostEqual(self.outputs["lss_I"][0], 0.5 * m_lss * (1 ** 2 + 0.95 ** 2))
        self.assertAlmostEqual(
            self.outputs["lss_I"][1], (1 / 12) * m_lss * (3 * (1 ** 2 + 0.95 ** 2) + self.outputs["L_lss"] ** 2)
        )

        m_hss = rho * np.pi * (0.75 ** 2 - 0.71 ** 2) * self.inputs["L_hss"]
        self.assertAlmostEqual(self.outputs["hss_mass"], m_hss)
        self.assertAlmostEqual(self.outputs["hss_cm"], 0.5 * (self.outputs["s_hss"][0] + self.outputs["s_hss"][-1]))
        self.assertAlmostEqual(self.outputs["hss_I"][0], 0.5 * m_hss * (0.75 ** 2 + 0.71 ** 2))
        self.assertAlmostEqual(
            self.outputs["hss_I"][1], (1 / 12) * m_hss * (3 * (0.75 ** 2 + 0.71 ** 2) + self.inputs["L_hss"] ** 2)
        )

        self.discrete_inputs["upwind"] = False
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["bedplate_cm"], np.r_[(2 + 2 - 0.5 * 6.95), 0.0, 0.5 * 4.725 + 0.05])
        self.assertAlmostEqual(self.outputs["lss_cm"], 0.5 * (self.outputs["s_lss"][0] + self.outputs["s_lss"][-1]))
        self.assertAlmostEqual(self.outputs["hss_cm"], 0.5 * (self.outputs["s_hss"][0] + self.outputs["s_hss"][-1]))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDirectLayout))
    suite.addTest(unittest.makeSuite(TestGearedLayout))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
