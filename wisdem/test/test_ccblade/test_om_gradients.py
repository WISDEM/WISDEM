import unittest
import os

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from wisdem.ccblade.ccblade_component import CCBladeGeometry, CCBladePower, \
    CCBladeLoads, AeroHubLoads
    

np.random.seed(314)

class Test(unittest.TestCase):

    # def test_ccblade_geometry(self):
    #     n_input = 10
    # 
    #     prob = om.Problem()
    # 
    #     ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    # 
    #     # Add some arbitrary inputs
    #     ivc.add_output('Rtip', val=80., units='m')
    #     ivc.add_output('precurve_in', val=np.random.rand(n_input), units='m')
    #     ivc.add_output('presweep_in', val=np.random.rand(n_input), units='m')
    #     ivc.add_output('precone', val=2.2, units='deg')
    # 
    #     comp = CCBladeGeometry(NINPUT=n_input)
    #     prob.model.add_subsystem('comp', comp, promotes=['*'])
    # 
    #     prob.setup(force_alloc_complex=True)
    # 
    #     prob.run_model()
    # 
    #     check = prob.check_partials(compact_print=True, method='fd')
    # 
    #     assert_check_partials(check)
        
    def test_ccblade_loads(self):
        prob = om.Problem()
        
        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        
        # Add some arbitrary inputs        
        # Load in airfoil and blade shape inputs for NREL 5MW
        npzfile = np.load(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'smaller_dataset.npz', allow_pickle=True)
        ivc.add_output('airfoils_aoa', npzfile['aoa'], units='deg')
        ivc.add_output('airfoils_Re', npzfile['Re'])
        ivc.add_output('airfoils_cl', np.moveaxis(npzfile['cl'][:,:,:,np.newaxis], 0, 1))
        ivc.add_output('airfoils_cd', np.moveaxis(npzfile['cd'][:,:,:,np.newaxis], 0, 1))
        ivc.add_output('airfoils_cm', np.moveaxis(npzfile['cm'][:,:,:,np.newaxis], 0, 1))
        ivc.add_output('r', npzfile['r'], units='m')
        ivc.add_output('chord', npzfile['chord'], units='m')
        ivc.add_output('theta', npzfile['theta'], units='deg')

        n_span = npzfile['r'].size
        n_aoa = npzfile['aoa'].size
        n_Re = npzfile['Re'].size
        
        # parameters
        ivc.add_output('V_load', 12., units='m/s')
        ivc.add_output('Omega_load', 7.0, units='rpm')
        ivc.add_output('pitch_load', val=2.0, units='deg', desc='blade pitch setting')
        ivc.add_output('azimuth_load', val=3.0, units='deg', desc='blade azimuthal location')
        
        ivc.add_output('Rhub', 1., units='m')
        ivc.add_output('Rtip', 70., units='m')
        ivc.add_output('hub_height', 100., units='m')
        ivc.add_output('precone', 0., units='deg')
        ivc.add_output('tilt', 0., units='deg')
        ivc.add_output('yaw', 0., units='deg')
        ivc.add_output('precurve', np.zeros(n_span), units='m')
        ivc.add_output('precurveTip', 0., units='m')
        ivc.add_output('presweep', np.zeros(n_span), units='m')
        ivc.add_output('presweepTip', 0., units='m')
        
        ivc.add_output('rho', 1.225, units='kg/m**3')
        ivc.add_output('mu', 1.81206e-5, units='kg/(m*s)')
        ivc.add_output('shearExp', 0.25)
        ivc.add_discrete_output('nBlades', 3)
        ivc.add_discrete_output('nSector', 4)
        ivc.add_discrete_output('tiploss', True)
        ivc.add_discrete_output('hubloss', True)
        ivc.add_discrete_output('wakerotation', True)
        ivc.add_discrete_output('usecd', True)
        
        analysis_options = {}
        analysis_options['blade'] = {}
        analysis_options['blade']['n_span'] = n_span
        analysis_options['blade']['n_aoa'] = n_aoa
        analysis_options['blade']['n_Re'] = n_Re
        analysis_options['blade']['n_tab'] = 1

        n_span, n_aoa, n_Re, n_tab = np.moveaxis(npzfile['cl'][:,:,:,np.newaxis], 0, 1).shape
        analysis_options['airfoils'] = {}
        analysis_options['airfoils']['n_aoa'] = n_aoa
        analysis_options['airfoils']['n_Re'] = n_Re
        analysis_options['airfoils']['n_tab'] = n_tab
        
        comp = CCBladeLoads(analysis_options=analysis_options)
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        
        prob.setup(force_alloc_complex=True)

        prob.run_model()

        check = prob.check_partials(compact_print=True)
        
        # Manually filter some entries out of the assert_check_partials call.
        # Will want to add this functionality to OpenMDAO itself at some point.
        new_check = {}
        for comp_name in check:
            new_check[comp_name] = {}
            for (output_name, input_name) in check[comp_name]:
                if 'airfoil' not in input_name and 'rho' not in input_name and 'mu' not in input_name and 'shearExp' not in input_name:
                    new_check[comp_name][(output_name, input_name)] = check[comp_name][(output_name, input_name)]
        
        assert_check_partials(new_check, rtol=5e-5, atol=1e-4)
                
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Test))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())