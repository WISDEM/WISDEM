import unittest
import os
import  wisdem.drivetrainse.direct_drivese as dd
import openmdao.api as om
import numpy as np

class TestGroup(unittest.TestCase):
    
    def testDirectDrive(self):
        
        prob = om.Problem()
        prob.model = dd.DirectDriveSE(topLevelFlag=True, n_points=10, n_dlcs=1, model_generator=True)
        prob.setup()

        prob['upwind'] = True
        prob['direct_drive'] = True
        prob['n_blades'] = 3
        prob['rotor_rpm'] = 10.0
        prob['machine_rating'] = 5e3

        prob['L_12'] = 2.0
        prob['L_h1'] = 1.0
        prob['L_2n'] = 1.5
        prob['L_grs'] = 1.1
        prob['L_gsn'] = 1.1
        prob['L_bedplate'] = 5.0
        prob['H_bedplate'] = 4.875
        prob['tilt'] = 4.0
        prob['access_diameter'] = 0.9

        npts = 10
        myones = np.ones(5)
        prob['lss_diameter'] = 3.3*myones
        prob['nose_diameter'] = 2.2*myones
        prob['lss_wall_thickness'] = 0.45*myones
        prob['nose_wall_thickness'] = 0.1*myones
        prob['bedplate_wall_thickness'] = 0.06*np.ones(npts)
        prob['D_top'] = 6.5

        prob['F_hub'] = np.array([2409.750e3, 0.0, 74.3529e2]).reshape((3,1))
        prob['M_hub'] = np.array([-1.83291e4, 6171.7324e2, 5785.82946e2]).reshape((3,1))

        prob['E'] = 210e9
        prob['G'] = 80.8e9
        prob['v'] = 0.3
        prob['rho'] = 7850.
        prob['sigma_y'] = 250e6
        prob['gamma_f'] = 1.35
        prob['gamma_m'] = 1.3
        prob['gamma_n'] = 1.0

        prob['pitch_system.blade_mass']       = 17000.
        prob['pitch_system.BRFM']             = 1.e+6
        prob['pitch_system.scaling_factor']   = 0.54
        prob['pitch_system.rho']              = 7850.
        prob['pitch_system.Xy']               = 371.e+6

        prob['hub_shell.blade_root_diameter'] = 4.
        prob['flange_t2shell_t']              = 4.
        prob['flange_OD2hub_D']               = 0.5
        prob['flange_ID2flange_OD']           = 0.8
        prob['hub_shell.rho']                 = 7200.
        prob['in2out_circ']                   = 1.2 
        prob['hub_shell.max_torque']          = 30.e+6
        prob['hub_shell.Xy']                  = 200.e+6
        prob['stress_concentration']          = 2.5
        prob['hub_shell.gamma']               = 2.0
        prob['hub_shell.metal_cost']          = 3.00

        prob['n_front_brackets']              = 3
        prob['n_rear_brackets']               = 3
        prob['spinner.blade_root_diameter']   = 4.
        prob['clearance_hub_spinner']         = 0.5
        prob['spin_hole_incr']                = 1.2
        prob['spinner.gust_ws']               = 70
        prob['spinner.gamma']                 = 1.5
        prob['spinner.composite_Xt']          = 60.e6
        prob['spinner.composite_SF']          = 1.5
        prob['spinner.composite_rho']         = 1600.
        prob['spinner.Xy']                    = 225.e+6
        prob['spinner.metal_SF']              = 1.5
        prob['spinner.metal_rho']             = 7850.
        prob['spinner.composite_cost']        = 7.00
        prob['spinner.metal_cost']            = 3.00

        prob['generator.T_rated']        = 10.25e6       #rev 1 9.94718e6
        prob['generator.P_mech']         = 10.71947704e6 #rev 1 9.94718e6
        prob['generator.n_nom']          = 10            #8.68                # rpm 9.6
        prob['generator.r_g']            = 4.0           # rev 1  4.92
        prob['generator.len_s']          = 1.7           # rev 2.3
        prob['generator.h_s']            = 0.7            # rev 1 0.3
        prob['generator.p']              = 70            #100.0    # rev 1 160
        prob['generator.h_m']            = 0.005         # rev 1 0.034
        prob['generator.h_ys']           = 0.04          # rev 1 0.045
        prob['generator.h_yr']           = 0.06          # rev 1 0.045
        prob['generator.b']              = 2.
        prob['generator.c']              = 5.0
        prob['generator.B_tmax']         = 1.9
        prob['generator.E_p']            = 3300/np.sqrt(3)
        prob['generator.D_nose']         = 2*1.1             # Nose outer radius
        prob['generator.D_shaft']        = 2*1.34            # Shaft outer radius =(2+0.25*2+0.3*2)*0.5
        prob['generator.t_r']            = 0.05          # Rotor disc thickness
        prob['generator.h_sr']           = 0.04          # Rotor cylinder thickness
        prob['generator.t_s']            = 0.053         # Stator disc thickness
        prob['generator.h_ss']           = 0.04          # Stator cylinder thickness
        prob['generator.u_allow_pcent']  = 8.5            # % radial deflection
        prob['generator.y_allow_pcent']  = 1.0            # % axial deflection
        prob['generator.z_allow_deg']    = 0.05           # torsional twist
        prob['generator.sigma']          = 60.0e3         # Shear stress
        prob['generator.B_r']            = 1.279
        prob['generator.ratio_mw2pp']    = 0.8
        prob['generator.h_0']            = 5e-3
        prob['generator.h_w']            = 4e-3
        prob['generator.k_fes']          = 0.8
        prob['generator.C_Cu']         = 4.786         # Unit cost of Copper $/kg
        prob['generator.C_Fe']         = 0.556         # Unit cost of Iron $/kg
        prob['generator.C_Fes']        = 0.50139       # specific cost of Structural_mass $/kg
        prob['generator.C_PM']         =   95.0
        prob['generator.rho_Fe']       = 7700.0        # Steel density Kg/m3
        prob['generator.rho_Fes']      = 7850          # structural Steel density Kg/m3
        prob['generator.rho_Copper']   = 8900.0        # copper density Kg/m3
        prob['generator.rho_PM']       = 7450.0        # typical density Kg/m3 of neodymium magnets

        prob.run_model()
        self.assertTrue(True)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestGroup))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
