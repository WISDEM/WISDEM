import os
import numpy as np
import numpy.testing as npt
import unittest
import openmdao.api as om
from wisdem.assemblies.fixed_bottom.monopile_assembly_turbine_nodrive import MonopileTurbine
from wisdem.rotorse.rotor import Init_RotorSE_wRefBlade
from wisdem.rotorse.rotor_geometry_yaml import ReferenceBlade

nSec = 4
nSecTow = 3
NPTS = 100

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

class TestRegression(unittest.TestCase):
    
    def testAssembly(self):
        # Global inputs and outputs
        fname_schema  = mydir + os.sep + 'IEAontology_schema.yaml'
        fname_input   = mydir + os.sep + 'IEA-15-240-RWT.yaml'

        # Initialize blade design
        refBlade = ReferenceBlade()
        refBlade.verbose  = True
        refBlade.NINPUT       = 8
        Nsection_Tow          = 19
        refBlade.NPTS         = 30
        refBlade.spar_var     = ['Spar_cap_ss', 'Spar_cap_ps'] # SS, then PS
        refBlade.te_var       = 'TE_reinforcement'
        refBlade.validate     = False
        refBlade.fname_schema = fname_schema
        blade = refBlade.initialize(fname_input)
        Analysis_Level = 0

        FASTpref                        = {}
        FASTpref['Analysis_Level']      = Analysis_Level
        fst_vt = {}

        prob = om.Problem()
        prob.model=MonopileTurbine(RefBlade=blade, Nsection_Tow=Nsection_Tow, VerbosityCosts=False, FASTpref=FASTpref)
        prob.model.nonlinear_solver = om.NonlinearRunOnce()
        prob.model.linear_solver    = om.DirectSolver()
        prob.setup()

        prob = Init_RotorSE_wRefBlade(prob, blade, Analysis_Level = Analysis_Level, fst_vt = fst_vt)

        # Environmental parameters for the tower
        prob['significant_wave_height'] = 4.52
        prob['significant_wave_period'] = 9.45
        prob['water_depth']             = 30.
        prob['wind_reference_height'] = prob['hub_height'] = 150.
        prob['shearExp']                       = 0.11
        prob['rho']                            = 1.225
        prob['mu']                             = 1.7934e-5
        prob['water_density']                  = 1025.0
        prob['water_viscosity']                = 1.3351e-3
        prob['wind_beta'] = prob['wave_beta'] = 0.0

        # Steel properties for the tower
        prob['material_density']               = 7850.0
        prob['E']                              = 210e9
        prob['G']                              = 79.3e9
        prob['yield_stress']                   = 345e6
        prob['soil_G']                         = 140e6
        prob['soil_nu']                        = 0.4

        # Design constraints
        prob['max_taper_ratio']                = 0.4
        prob['min_diameter_thickness_ratio']   = 120.0

        # Safety factors
        prob['gamma_fatigue']   = 1.755 # (Float): safety factor for fatigue
        prob['gamma_f']         = 1.35  # (Float): safety factor for loads/stresses
        prob['gamma_m']         = 1.3   # (Float): safety factor for materials
        prob['gamma_freq']      = 1.1   # (Float): safety factor for resonant frequencies
        prob['gamma_n']         = 1.0
        prob['gamma_b']         = 1.1

        # Tower
        prob['tower_buckling_length']          = 30.0
        prob['tower_outfitting_factor']        = 1.07
        prob['foundation_height']       = -30.
        prob['suctionpile_depth']       = 45.
        prob['tower_section_height']    = np.array([5., 5., 5., 5., 5., 5., 5., 5., 5., 13.,  13.,  13.,  13.,  13.,  13.,  13.,  13.,  13., 12.58244309])
        prob['tower_outer_diameter'] = np.array([10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 9.92647687, 9.44319282, 8.83283769, 8.15148167, 7.38976138, 6.90908962, 6.74803581, 6.57231775, 6.5])
        prob['tower_wall_thickness'] = np.array([0.05534138, 0.05344902, 0.05150928, 0.04952705, 0.04751736, 0.04551709, 0.0435267, 0.04224176, 0.04105759, 0.0394965, 0.03645589, 0.03377851, 0.03219233, 0.03070819, 0.02910109, 0.02721289, 0.02400931, 0.0208264, 0.02399756])
        prob['tower_buckling_length']   = 15.0
        prob['transition_piece_mass']   = 100e3
        prob['transition_piece_height'] = 15.0

        prob['DC']      = 80.0
        prob['shear']   = True
        prob['geom']    = True
        prob['tower_force_discretization'] = 5.0
        prob['nM']      = 2
        prob['Mmethod'] = 1
        prob['lump']    = 0
        prob['tol']     = 1e-9
        prob['shift']   = 0.0

        # Offshore BOS
        prob['wtiv'] = 'example_wtiv'
        prob['feeder'] = 'future_feeder'
        prob['num_feeders'] = 1
        prob['oss_install_vessel'] = 'example_heavy_lift_vessel'
        prob['site_distance'] = 40.0
        prob['site_distance_to_landfall'] = 40.0
        prob['site_distance_to_interconnection'] = 40.0
        prob['plant_turbine_spacing'] = 7
        prob['plant_row_spacing'] = 7
        prob['plant_substation_distance'] = 1
        prob['tower_deck_space'] = 0.
        prob['nacelle_deck_space'] = 0.
        prob['blade_deck_space'] = 0.
        prob['port_cost_per_month'] = 2e6
        prob['monopile_deck_space'] = 0.
        prob['transition_piece_deck_space'] = 0.
        prob['commissioning_pct'] = 0.01
        prob['decommissioning_pct'] = 0.15
        prob['project_lifetime'] = prob['lifetime'] = 20.0    
        prob['number_of_turbines']             = 40
        prob['annual_opex']                    = 43.56 # $/kW/yr
        #prob['bos_costs']                      = 1234.5 # $/kW

        prob['tower_add_gravity'] = True

        # For turbine costs
        prob['offshore']             = True
        prob['crane']                = False
        prob['bearing_number']       = 2
        prob['crane_cost']           = 0.0
        prob['labor_cost_rate']      = 3.0
        prob['material_cost_rate']   = 2.0
        prob['painting_cost_rate']   = 28.8

        # Drivetrain
        prob['tilt']                    = 6.0
        prob['overhang']                = 11.075
        prob['hub_cm']                  = np.array([-10.685, 0.0, 5.471])
        prob['nac_cm']                  = np.array([-5.718, 0.0, 4.048])
        prob['hub_I']                   = np.array([1382171.187, 2169261.099, 2160636.794, 0.0, 0.0, 0.0])
        prob['nac_I']                   = np.array([13442265.552, 21116729.439, 18382414.385, 0.0, 0.0, 0.0])
        prob['hub_mass']                = 190e3
        prob['nac_mass']                = 797.275e3-190e3
        prob['hss_mass']                = 0.0
        prob['lss_mass']                = 19.504e3
        prob['cover_mass']              = 0.0
        prob['pitch_system_mass']       = 50e3
        prob['platforms_mass']          = 0.0
        prob['spinner_mass']            = 0.0
        prob['transformer_mass']        = 0.0
        prob['vs_electronics_mass']     = 0.0
        prob['yaw_mass']                = 100e3
        prob['gearbox_mass']            = 0.0
        prob['generator_mass']          = 226.7e3+145.25e3
        prob['bedplate_mass']           = 39.434e3
        prob['main_bearing_mass']       = 4.699e3

        prob.run_model()
        # Make sure we get here
        self.assertTrue(True)
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRegression))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
