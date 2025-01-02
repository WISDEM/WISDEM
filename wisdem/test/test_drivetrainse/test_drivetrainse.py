import sys
import unittest
import traceback

import numpy as np
import openmdao.api as om
import numpy.testing as npt

from wisdem.drivetrainse.drivetrain import DrivetrainSE

npts = 12


def set_common(prob, opt):
    prob["n_blades"] = 3
    prob["rotor_diameter"] = 120.0
    prob["machine_rating"] = 5e3
    prob["D_top"] = 6.5
    prob["rated_torque"] = 10.25e6  # rev 1 9.94718e6
    prob["damping_ratio"] = 0.01

    prob["F_aero_hub"] = np.array([2409.750e3, 0.0, 74.3529e2]).reshape((3, 1))
    prob["M_aero_hub"] = np.array([-1.83291e4, 6171.7324e2, 5785.82946e2]).reshape((3, 1))

    prob["E_mat"] = 210e9 * np.ones((1, 3))
    prob["G_mat"] = 80.8e9 * np.ones((1, 3))
    prob["Xt_mat"] = 1e7 * np.ones((1, 3))
    prob["wohler_exp_mat"] = 10.0 * np.ones(1)
    prob["wohler_A_mat"] = 10.0 * np.ones(1)
    prob["rho_mat"] = 7850.0 * np.ones(1)
    prob["Xy_mat"] = 250e6 * np.ones(1)
    prob["unit_cost_mat"] = 3.0 * np.ones(1)
    prob["lss_material"] = prob["hss_material"] = prob["hub_material"] = prob["spinner_material"] = prob[
        "bedplate_material"
    ] = "steel"
    prob["material_names"] = ["steel"]

    prob["blade_mass"] = 170.0
    prob["blades_mass"] = 3*prob["blade_mass"]
    prob["blades_cm"] = 2.0
    prob["pitch_system.BRFM"] = 1.0e6
    prob["pitch_system_scaling_factor"] = 0.54

    prob["blade_root_diameter"] = 4.0
    prob["flange_t2shell_t"] = 4.0
    prob["flange_OD2hub_D"] = 0.5
    prob["flange_ID2flange_OD"] = 0.8
    prob["hub_rho"] = 7200.0
    prob["hub_Xy"] = 200.0e6
    prob["hub_stress_concentration"] = 2.5

    prob["n_front_brackets"] = 3
    prob["n_rear_brackets"] = 3
    prob["clearance_hub_spinner"] = 0.5
    prob["spin_hole_incr"] = 1.2
    prob["spinner_gust_ws"] = 70.0

    prob["hub_diameter"] = 6.235382907247958
    prob["minimum_rpm"] = 2  # 8.68                # rpm 9.6
    prob["rated_rpm"] = 10  # 8.68                # rpm 9.6
    prob["blades_I"] = 1e3 * np.ones(6)
    prob["blades_I"][0] = 199200777.51 * 30.0 * 5 / 10.0 / np.pi

    prob["bear1.bearing_type"] = "CARB"
    prob["bear2.bearing_type"] = "SRB"

    return prob


class TestGroup(unittest.TestCase):
    def testDirectDrive_withGen(self):
        opt = {}
        opt["WISDEM"] = {}
        opt["WISDEM"]["n_dlc"] = 1
        opt["WISDEM"]["DriveSE"] = {}
        opt["WISDEM"]["DriveSE"]["direct"] = True
        opt["WISDEM"]["DriveSE"]["hub"] = {}
        opt["WISDEM"]["DriveSE"]["hub"]["hub_gamma"] = 2.0
        opt["WISDEM"]["DriveSE"]["hub"]["spinner_gamma"] = 1.5
        opt["WISDEM"]["DriveSE"]["gamma_f"] = 1.35
        opt["WISDEM"]["DriveSE"]["gamma_m"] = 1.3
        opt["WISDEM"]["DriveSE"]["gamma_n"] = 1.0
        opt["WISDEM"]["RotorSE"] = {}
        opt["WISDEM"]["RotorSE"]["n_pc"] = 20
        opt["materials"] = {}
        opt["materials"]["n_mat"] = 1
        opt["WISDEM"]["GeneratorSE"] = {}
        opt["WISDEM"]["GeneratorSE"]["type"] = "pmsg_outer"
        opt["flags"] = {}
        opt["flags"]["generator"] = True

        prob = om.Problem(reports=False)
        prob.model = DrivetrainSE(modeling_options=opt)
        prob.setup()
        prob = set_common(prob, opt)

        prob["upwind"] = True

        prob["L_12"] = 2.0
        prob["L_h1"] = 1.0
        prob["L_generator"] = 3.25
        prob["overhang"] = 6.25
        prob["drive_height"] = 4.875
        prob["tilt"] = 4.0
        prob["access_diameter"] = 0.9

        myones = np.ones(2)
        prob["lss_diameter"] = 3.3 * myones
        prob["nose_diameter"] = 2.2 * myones
        prob["lss_wall_thickness"] = 0.45 * myones
        prob["nose_wall_thickness"] = 0.1 * myones
        prob["bedplate_wall_thickness"] = 0.06 * np.ones(4)
        prob["bear1.D_shaft"] = 2.2
        prob["bear2.D_shaft"] = 2.2
        prob["generator.D_shaft"] = 3.3
        prob["generator.D_nose"] = 2.2

        prob["generator.P_mech"] = 10.71947704e6  # rev 1 9.94718e6
        prob["generator.rad_ag"] = 4.0  # rev 1  4.92
        prob["generator.len_s"] = 1.7  # rev 2.3
        prob["generator.h_s"] = 0.7  # rev 1 0.3
        prob["generator.p"] = 70  # 100.0    # rev 1 160
        prob["generator.h_m"] = 0.005  # rev 1 0.034
        prob["generator.h_ys"] = 0.04  # rev 1 0.045
        prob["generator.h_yr"] = 0.06  # rev 1 0.045
        prob["generator.b"] = 2.0
        prob["generator.c"] = 5.0
        prob["generator.B_tmax"] = 1.9
        prob["generator.E_p"] = 3300 / np.sqrt(3)
        prob["generator.t_r"] = 0.05  # Rotor disc thickness
        prob["generator.h_sr"] = 0.04  # Rotor cylinder thickness
        prob["generator.t_s"] = 0.053  # Stator disc thickness
        prob["generator.h_ss"] = 0.04  # Stator cylinder thickness
        prob["generator.u_allow_pcent"] = 8.5  # % radial deflection
        prob["generator.y_allow_pcent"] = 1.0  # % axial deflection
        prob["generator.z_allow_deg"] = 0.05  # torsional twist
        prob["generator.sigma"] = 60.0e3  # Shear stress
        prob["generator.B_r"] = 1.279
        prob["generator.ratio_mw2pp"] = 0.8
        prob["generator.h_0"] = 5e-3
        prob["generator.h_w"] = 4e-3
        prob["generator.k_fes"] = 0.8
        prob["generator.C_Cu"] = 4.786  # Unit cost of Copper $/kg
        prob["generator.C_Fe"] = 0.556  # Unit cost of Iron $/kg
        prob["generator.C_Fes"] = 0.50139  # specific cost of Structural_mass $/kg
        prob["generator.C_PM"] = 95.0
        prob["generator.rho_Fe"] = 7700.0  # Steel density Kg/m3
        prob["generator.rho_Fes"] = 7850  # structural Steel density Kg/m3
        prob["generator.rho_Copper"] = 8900.0  # copper density Kg/m3
        prob["generator.rho_PM"] = 7450.0  # typical density Kg/m3 of neodymium magnets

        try:
            prob.run_model()
            self.assertTrue(True)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.assertTrue(False)

        # Test that key outputs are filled
        self.assertGreater(prob["nacelle_mass"], 100e3)
        self.assertLess(prob["nacelle_cm"][0], 0.0)
        self.assertGreater(prob["nacelle_cm"][2], 0.0)
        self.assertGreater(prob["rna_mass"], 100e3)
        self.assertLess(prob["rna_cm"][0], 0.0)
        self.assertGreater(prob["rna_cm"][2], 0.0)
        self.assertGreater(prob["generator_mass"], 100e3)
        self.assertGreater(prob["generator_cost"], 100e3)
        npt.assert_array_less(100e3, prob["generator_I"])
        npt.assert_array_less(0.8, prob["generator_efficiency"])
        npt.assert_array_less(prob["generator_efficiency"], 1.0)
        self.assertGreater(prob["drivetrain_spring_constant"], 1e10)
        self.assertGreater(prob["drivetrain_damping_coefficient"], 1e7)

    def testDirectDrive_withSimpleGen(self):
        opt = {}
        opt["WISDEM"] = {}
        opt["WISDEM"]["n_dlc"] = 1
        opt["WISDEM"]["DriveSE"] = {}
        opt["WISDEM"]["DriveSE"]["direct"] = True
        opt["WISDEM"]["DriveSE"]["hub"] = {}
        opt["WISDEM"]["DriveSE"]["hub"]["hub_gamma"] = 2.0
        opt["WISDEM"]["DriveSE"]["hub"]["spinner_gamma"] = 1.5
        opt["WISDEM"]["DriveSE"]["gamma_f"] = 1.35
        opt["WISDEM"]["DriveSE"]["gamma_m"] = 1.3
        opt["WISDEM"]["DriveSE"]["gamma_n"] = 1.0
        opt["WISDEM"]["RotorSE"] = {}
        opt["WISDEM"]["RotorSE"]["n_pc"] = 20
        opt["materials"] = {}
        opt["materials"]["n_mat"] = 1
        opt["flags"] = {}
        opt["flags"]["generator"] = False

        prob = om.Problem(reports=False)
        prob.model = DrivetrainSE(modeling_options=opt)
        prob.setup()
        prob = set_common(prob, opt)

        prob["upwind"] = True

        prob["L_12"] = 2.0
        prob["L_h1"] = 1.0
        prob["L_generator"] = 3.25
        prob["overhang"] = 6.25
        prob["drive_height"] = 4.875
        prob["tilt"] = 4.0
        prob["access_diameter"] = 0.9

        myones = np.ones(2)
        prob["lss_diameter"] = 3.3 * myones
        prob["nose_diameter"] = 2.2 * myones
        prob["lss_wall_thickness"] = 0.45 * myones
        prob["nose_wall_thickness"] = 0.1 * myones
        prob["bedplate_wall_thickness"] = 0.06 * np.ones(4)
        prob["bear1.D_shaft"] = 2.2
        prob["bear2.D_shaft"] = 2.2

        prob["generator_efficiency_user"] = np.zeros((20, 2))

        try:
            prob.run_model()
            self.assertTrue(True)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.assertTrue(False)

        # Test that key outputs are filled
        self.assertGreater(prob["nacelle_mass"], 100e3)
        self.assertLess(prob["nacelle_cm"][0], 0.0)
        self.assertGreater(prob["nacelle_cm"][2], 0.0)
        self.assertGreater(prob["rna_mass"], 100e3)
        self.assertLess(prob["rna_cm"][0], 0.0)
        self.assertGreater(prob["rna_cm"][2], 0.0)
        self.assertGreater(prob["generator_mass"], 100e3)
        # self.assertGreater(prob["generator_cost"], 100e3)
        npt.assert_array_less(100e3, prob["generator_I"])
        npt.assert_array_less(0.8, prob["generator_efficiency"])
        npt.assert_array_less(prob["generator_efficiency"], 1.0)
        self.assertGreater(prob["drivetrain_spring_constant"], 1e10)
        self.assertGreater(prob["drivetrain_damping_coefficient"], 1e7)

    def testGeared_withGen(self):
        opt = {}
        opt["WISDEM"] = {}
        opt["WISDEM"]["n_dlc"] = 1
        opt["WISDEM"]["DriveSE"] = {}
        opt["WISDEM"]["DriveSE"]["direct"] = False
        opt["WISDEM"]["DriveSE"]["use_gb_torque_density"] = False
        opt["WISDEM"]["DriveSE"]["gearbox_torque_density"] = 0.
        opt["WISDEM"]["DriveSE"]["hub"] = {}
        opt["WISDEM"]["DriveSE"]["hub"]["hub_gamma"] = 2.0
        opt["WISDEM"]["DriveSE"]["hub"]["spinner_gamma"] = 1.5
        opt["WISDEM"]["DriveSE"]["gamma_f"] = 1.35
        opt["WISDEM"]["DriveSE"]["gamma_m"] = 1.3
        opt["WISDEM"]["DriveSE"]["gamma_n"] = 1.0
        opt["WISDEM"]["GeneratorSE"] = {}
        opt["WISDEM"]["GeneratorSE"]["type"] = "dfig"
        opt["WISDEM"]["RotorSE"] = {}
        opt["WISDEM"]["RotorSE"]["n_pc"] = 20
        opt["materials"] = {}
        opt["materials"]["n_mat"] = 1
        opt["flags"] = {}
        opt["flags"]["generator"] = True

        prob = om.Problem(reports=False)
        prob.model = DrivetrainSE(modeling_options=opt)
        prob.setup()
        prob = set_common(prob, opt)

        prob["upwind"] = True

        prob["L_12"] = 2.0
        prob["L_h1"] = 1.0
        prob["overhang"] = 2.0
        prob["drive_height"] = 4.875
        prob["L_hss"] = 1.5
        prob["L_generator"] = 1.25
        prob["L_gearbox"] = 1.1
        prob["tilt"] = 5.0

        myones = np.ones(2)
        prob["lss_diameter"] = 2.3 * myones
        prob["lss_wall_thickness"] = 0.05 * myones
        prob["hss_diameter"] = 2.0 * myones
        prob["hss_wall_thickness"] = 0.05 * myones

        prob["bedplate_flange_width"] = 1.5
        prob["bedplate_flange_thickness"] = 0.05
        # prob['bedplate_web_height'] = 1.0
        prob["bedplate_web_thickness"] = 0.05
        prob["bear1.D_shaft"] = 2.3
        prob["bear2.D_shaft"] = 2.3

        prob["planet_numbers"] = np.array([3, 3, 0])
        prob["gear_configuration"] = "eep"
        # prob['shaft_factor'] = 'normal'
        prob["gear_ratio"] = 90.0

        prob["generator.rho_Fe"] = 7700.0
        prob["generator.rho_Fes"] = 7850.0
        prob["generator.rho_Copper"] = 8900.0
        prob["generator.rho_PM"] = 7450.0
        prob["generator.B_r"] = 1.2
        prob["generator.E"] = 2e11
        prob["generator.G"] = 79.3e9
        prob["generator.P_Fe0e"] = 1.0
        prob["generator.P_Fe0h"] = 4.0
        prob["generator.S_N"] = -0.002
        prob["generator.alpha_p"] = 0.5 * np.pi * 0.7
        prob["generator.b_r_tau_r"] = 0.45
        prob["generator.b_ro"] = 0.004
        prob["generator.b_s_tau_s"] = 0.45
        prob["generator.b_so"] = 0.004
        prob["generator.cofi"] = 0.85
        prob["generator.freq"] = 60
        prob["generator.h_i"] = 0.001
        prob["generator.h_sy0"] = 0.0
        prob["generator.h_w"] = 0.005
        prob["generator.k_fes"] = 0.9
        prob["generator.k_s"] = 0.2
        prob["generator.m"] = 3
        prob["generator.mu_0"] = np.pi * 4e-7
        prob["generator.mu_r"] = 1.06
        prob["generator.p"] = 3.0
        prob["generator.phi"] = np.deg2rad(90)
        prob["generator.ratio_mw2pp"] = 0.7
        prob["generator.resist_Cu"] = 1.8e-8 * 1.4
        prob["generator.sigma"] = 40e3
        prob["generator.v"] = 0.3
        prob["generator.y_tau_p"] = 1.0
        prob["generator.y_tau_pr"] = 10.0 / 12
        prob["generator.cofi"] = 0.9
        prob["generator.y_tau_p"] = 12.0 / 15.0
        prob["generator.sigma"] = 21.5e3
        prob["generator.rad_ag"] = 0.61
        prob["generator.len_s"] = 0.49
        prob["generator.h_s"] = 0.08
        prob["generator.I_0"] = 40.0
        prob["generator.B_symax"] = 1.3
        prob["generator.S_Nmax"] = -0.2
        prob["generator.h_0"] = 0.01
        prob["generator.k_fillr"] = 0.55
        prob["generator.k_fills"] = 0.65
        prob["generator.q1"] = 5
        prob["generator.q2"] = 4

        # prob['generator.D_shaft'] = 2.3

        try:
            prob.run_model()
            self.assertTrue(True)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.assertTrue(False)

        # Test that key outputs are filled
        self.assertGreater(prob["nacelle_mass"], 100e3)
        self.assertGreater(prob["nacelle_cm"][0], 0.0)
        self.assertGreater(prob["nacelle_cm"][2], 0.0)
        self.assertGreater(prob["rna_mass"], 100e3)
        self.assertGreater(prob["rna_cm"][0], 0.0)
        self.assertGreater(prob["rna_cm"][2], 0.0)
        self.assertGreater(prob["generator_mass"], 10e3)
        self.assertGreater(prob["generator_cost"], 10e3)
        npt.assert_array_less(1e3, prob["generator_I"])
        npt.assert_array_less(0.2, prob["generator_efficiency"][1:])
        npt.assert_array_less(prob["generator_efficiency"], 1.0)
        self.assertGreater(prob["drivetrain_spring_constant"], 1e10)
        self.assertGreater(prob["drivetrain_damping_coefficient"], 1e7)

    def testGeared_withSimpleGen(self):
        opt = {}
        opt["WISDEM"] = {}
        opt["WISDEM"]["n_dlc"] = 1
        opt["WISDEM"]["DriveSE"] = {}
        opt["WISDEM"]["DriveSE"]["direct"] = False
        opt["WISDEM"]["DriveSE"]["use_gb_torque_density"] = False
        opt["WISDEM"]["DriveSE"]["hub"] = {}
        opt["WISDEM"]["DriveSE"]["hub"]["hub_gamma"] = 2.0
        opt["WISDEM"]["DriveSE"]["hub"]["spinner_gamma"] = 1.5
        opt["WISDEM"]["DriveSE"]["gamma_f"] = 1.35
        opt["WISDEM"]["DriveSE"]["gamma_m"] = 1.3
        opt["WISDEM"]["DriveSE"]["gamma_n"] = 1.0
        opt["flags"] = {}
        opt["flags"]["generator"] = False
        opt["WISDEM"]["RotorSE"] = {}
        opt["WISDEM"]["RotorSE"]["n_pc"] = 20
        opt["materials"] = {}
        opt["materials"]["n_mat"] = 1

        prob = om.Problem(reports=False)
        prob.model = DrivetrainSE(modeling_options=opt)
        prob.setup()
        prob = set_common(prob, opt)

        prob["upwind"] = True

        prob["L_12"] = 2.0
        prob["L_h1"] = 1.0
        prob["overhang"] = 2.0
        prob["drive_height"] = 4.875
        prob["L_hss"] = 1.5
        prob["L_generator"] = 1.25
        prob["L_gearbox"] = 1.1
        prob["tilt"] = 5.0

        myones = np.ones(2)
        prob["lss_diameter"] = 2.3 * myones
        prob["lss_wall_thickness"] = 0.05 * myones
        prob["hss_diameter"] = 2.0 * myones
        prob["hss_wall_thickness"] = 0.05 * myones

        prob["bedplate_flange_width"] = 1.5
        prob["bedplate_flange_thickness"] = 0.05
        # prob['bedplate_web_height'] = 1.0
        prob["bedplate_web_thickness"] = 0.05
        prob["bear1.D_shaft"] = 2.3
        prob["bear2.D_shaft"] = 2.3
        # prob['generator.D_shaft'] = 2.3

        prob["planet_numbers"] = np.array([3, 3, 0])
        prob["gear_configuration"] = "eep"
        # prob['shaft_factor'] = 'normal'
        prob["gear_ratio"] = 90.0

        prob["generator_efficiency_user"] = np.zeros((20, 2))

        try:
            prob.run_model()
            self.assertTrue(True)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.assertTrue(False)

        # Test that key outputs are filled
        self.assertGreater(prob["nacelle_mass"], 100e3)
        self.assertGreater(prob["nacelle_cm"][0], 0.0)
        self.assertGreater(prob["nacelle_cm"][2], 0.0)
        self.assertGreater(prob["rna_mass"], 100e3)
        self.assertGreater(prob["rna_cm"][0], 0.0)
        self.assertGreater(prob["rna_cm"][2], 0.0)
        self.assertGreater(prob["generator_mass"], 10e3)
        # self.assertGreater(prob["generator_cost"], 10e3)
        npt.assert_array_less(1e3, prob["generator_I"])
        npt.assert_array_less(0.8, prob["generator_efficiency"])
        npt.assert_array_less(prob["generator_efficiency"], 1.0)
        self.assertGreater(prob["drivetrain_spring_constant"], 1e10)
        self.assertGreater(prob["drivetrain_damping_coefficient"], 1e7)


if __name__ == "__main__":
    unittest.main()
