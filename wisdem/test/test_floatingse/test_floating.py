import unittest

import numpy as np
import numpy.testing as npt
from openmdao.api import Problem
from wisdem.floatingse.floating import FloatingSE

npts = 5


class TestOC3Mass(unittest.TestCase):
    def testMassPropertiesSpar(self):

        opt = {}
        opt["floating"] = {}
        opt["flags"] = {}
        opt["flags"]["floating"] = True
        opt["flags"]["tower"] = False
        opt["General"] = {}
        opt["WISDEM"] = {}
        opt["WISDEM"]["n_dlc"] = 1
        opt["WISDEM"]["FloatingSE"] = {}
        opt["floating"]["members"] = {}
        opt["floating"]["members"]["n_members"] = 1
        opt["floating"]["members"]["n_height"] = [npts]
        opt["floating"]["members"]["n_bulkheads"] = [4]
        opt["floating"]["members"]["n_layers"] = [1]
        opt["floating"]["members"]["n_ballasts"] = [0]
        opt["floating"]["members"]["n_axial_joints"] = [1]
        opt["WISDEM"]["FloatingSE"]["frame3dd"] = {}
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["shear"] = True
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["geom"] = True
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["modal"] = False
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["tol"] = 1e-6
        opt["WISDEM"]["FloatingSE"]["gamma_f"] = 1.35  # Safety factor on loads
        opt["WISDEM"]["FloatingSE"]["gamma_m"] = 1.3  # Safety factor on materials
        opt["WISDEM"]["FloatingSE"]["gamma_n"] = 1.0  # Safety factor on consequence of failure
        opt["WISDEM"]["FloatingSE"]["gamma_b"] = 1.1  # Safety factor on buckling
        opt["WISDEM"]["FloatingSE"]["gamma_fatigue"] = 1.755  # Not used
        opt["WISDEM"]["FloatingSE"]["run_modal"] = True  # Not used
        opt["mooring"] = {}
        opt["mooring"]["n_attach"] = 3
        opt["mooring"]["n_anchors"] = 3
        opt["mooring"]["line_anchor"] = ["custom"] * 3
        opt["mooring"]["line_material"] = ["custom"] * 3

        opt["materials"] = {}
        opt["materials"]["n_mat"] = 2

        prob = Problem()
        prob.model = FloatingSE(modeling_options=opt)
        prob.setup()

        # Material properties
        prob["rho_mat"] = np.array([7850.0, 5000.0])  # Steel, ballast slurry [kg/m^3]
        prob["E_mat"] = 200e9 * np.ones((2, 3))  # Young's modulus [N/m^2]
        prob["G_mat"] = 79.3e9 * np.ones((2, 3))  # Shear modulus [N/m^2]
        prob["sigma_y_mat"] = 3.45e8 * np.ones(2)  # Elastic yield stress [N/m^2]
        prob["sigma_ult_mat"] = 5e8 * np.ones((2, 3))
        prob["wohler_exp_mat"] = 4.0 * np.ones(2)
        prob["wohler_A_mat"] = 7.5e8 * np.ones(2)
        prob["unit_cost_mat"] = np.array([2.0, 1.0])
        prob["material_names"] = ["steel", "slurry"]

        # Mass and cost scaling factors
        prob["labor_cost_rate"] = 1.0  # Cost factor for labor time [$/min]
        prob["painting_cost_rate"] = 14.4  # Cost factor for column surface finishing [$/m^2]
        prob["member0.outfitting_factor_in"] = 1.0  # Fraction of additional outfitting mass for each column

        # Column geometry
        h = np.array([49.0, 59.0, 8.0, 14.0])  # Length of each section [m]
        prob["member0.grid_axial_joints"] = [0.384615]  # Fairlead at 70m
        # prob["member0.ballast_grid"] = np.empy((0,2))
        # prob["member0.ballast_volume"] = np.empty(0)
        prob["member0.s_in"] = np.cumsum(np.r_[0, h]) / h.sum()
        prob["member0.outer_diameter_in"] = np.array([9.4, 9.4, 9.4, 6.5, 6.5])
        prob["member0.layer_thickness"] = 0.05 * np.ones((1, npts))
        prob["member0.layer_materials"] = ["steel"]
        prob["member0.ballast_materials"] = ["slurry", "seawater"]
        prob["member0:joint1"] = np.array([0.0, 0.0, 10.0 - h.sum()])
        prob["member0:joint2"] = np.array([0.0, 0.0, 10.0])  # Freeboard=10
        prob["member0.s_ghost1"] = 0.0
        prob["member0.s_ghost2"] = 1.0
        prob["member0.bulkhead_thickness"] = 0.05 * np.ones(4)  # Locations of internal bulkheads
        prob["member0.bulkhead_grid"] = np.array([0.0, 0.37692308, 0.89230769, 1.0])  # Thickness of internal bulkheads
        prob["member0.ring_stiffener_web_height"] = 0.10
        prob["member0.ring_stiffener_web_thickness"] = 0.04
        prob["member0.ring_stiffener_flange_width"] = 0.10
        prob["member0.ring_stiffener_flange_thickness"] = 0.02
        prob["member0.ring_stiffener_spacing"] = 0.016538462  # non-dimensional ring stiffener spacing
        prob["transition_node"] = prob["member0:joint2"]
        prob["transition_piece_mass"] = 0.0
        prob["transition_piece_cost"] = 0.0

        # Mooring parameters
        prob["line_diameter"] = 0.09  # Diameter of mooring line/chain [m]
        prob["line_length"] = 300 + 902.2  # Unstretched mooring line length
        prob["line_mass_density_coeff"] = 19.9e3
        prob["line_stiffness_coeff"] = 8.54e10
        prob["line_breaking_load_coeff"] = 176972.7
        prob["line_cost_rate_coeff"] = 3.415e4
        prob["fairlead_radius"] = 10.0  # Offset from shell surface for mooring attachment [m]
        prob["fairlead"] = 70.0
        prob["anchor_radius"] = 853.87  # Distance from centerline to sea floor landing [m]
        prob["anchor_cost"] = 1e5
        prob["anchor_mass"] = 0.0

        # Mooring constraints
        prob["max_surge_fraction"] = 0.1  # Max surge/sway offset [m]
        prob["survival_heel"] = np.deg2rad(10.0)  # Max heel (pitching) angle [deg->rad]
        prob["operational_heel"] = np.deg2rad(5.0)  # Max heel (pitching) angle [deg->rad]

        # Set environment to that used in OC3 testing campaign
        prob["rho_air"] = 1.226  # Density of air [kg/m^3]
        prob["mu_air"] = 1.78e-5  # Viscosity of air [kg/m/s]
        prob["rho_water"] = 1025.0  # Density of water [kg/m^3]
        prob["mu_water"] = 1.08e-3  # Viscosity of water [kg/m/s]
        prob["water_depth"] = 320.0  # Distance to sea floor [m]
        prob["Hsig_wave"] = 1.0  # Significant wave height [m]
        prob["Tsig_wave"] = 1e3  # Wave period [s]
        prob["shearExp"] = 0.11  # Shear exponent in wind power law
        prob["cm"] = 2.0  # Added mass coefficient
        prob["Uc"] = 0.0  # Mean current speed
        prob["beta_wind"] = prob["beta_wave"] = 0.0
        prob["cd_usr"] = -1.0  # Compute drag coefficient
        prob["env.Uref"] = 10.0
        prob["wind_reference_height"] = 100.0

        # Properties of turbine
        prob["turbine_mass"] = 0.0
        prob["turbine_cg"] = np.zeros(3)
        prob["turbine_I"] = np.zeros(6)
        prob["turbine_F"] = np.zeros(3)
        prob["turbine_M"] = np.zeros(3)

        prob.run_model()

        m_top = np.pi * 3.2 ** 2.0 * 0.05 * 7850.0
        ansys_m_bulk = 13204.0 + 2.0 * 27239.0 + m_top
        ansys_m_shell = 80150.0 + 32060.0 + 79701.0 + 1251800.0
        ansys_m_stiff = 1390.9 * 52 + 1282.2 + 1121.2 + 951.44 * 3
        ansys_m_spar = ansys_m_bulk + ansys_m_shell + ansys_m_stiff
        ansys_cg = np.array([0.0, 0.0, -58.926])
        ansys_Ixx = 2178400000.0 + m_top * (0.25 * 3.2 ** 2.0 + (10 - ansys_cg[-1]) ** 2)
        ansys_Iyy = 2178400000.0 + m_top * (0.25 * 3.2 ** 2.0 + (10 - ansys_cg[-1]) ** 2)
        ansys_Izz = 32297000.0 + 0.5 * m_top * 3.2 ** 2.0
        ansys_I = np.array([ansys_Ixx, ansys_Iyy, ansys_Izz, 0.0, 0.0, 0.0])

        npt.assert_allclose(
            ansys_m_bulk, prob["member0.bulkhead_mass"].sum(), rtol=0.03
        )  # ANSYS uses R_od, we use R_id, top cover seems unaccounted for
        npt.assert_allclose(ansys_m_shell, prob["member0.shell_mass"].sum(), rtol=0.01)
        npt.assert_allclose(ansys_m_stiff, prob["member0.stiffener_mass"].sum(), rtol=0.01)
        npt.assert_allclose(ansys_m_spar, prob["member0.total_mass"].sum(), rtol=0.01)
        npt.assert_allclose(ansys_cg, prob["member0.center_of_mass"], rtol=0.02)
        npt.assert_allclose(ansys_I, prob["member0.I_total"], rtol=0.02)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestOC3Mass))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
