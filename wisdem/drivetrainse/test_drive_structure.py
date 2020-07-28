import numpy as np
import numpy.testing as npt
import unittest
import wisdem.drivetrainse.drive_structure as ds
import wisdem.drivetrainse.layout as lay
import openmdao.api as om
from wisdem.commonse import gravity

npts = 10

class TestRun(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        self.discrete_inputs['upwind'] = True

        self.inputs['L_12'] = 2.0
        self.inputs['L_h1'] = 1.0
        self.inputs['L_2n'] = 1.5
        self.inputs['L_grs'] = 1.1
        self.inputs['L_gsn'] = 1.1
        self.inputs['L_bedplate'] = 5.0
        self.inputs['H_bedplate'] = 4.875
        self.inputs['tilt'] = 4.0
        self.inputs['access_diameter'] = 0.9

        myones = np.ones(5)
        self.inputs['lss_diameter'] = 3.3*myones
        self.inputs['nose_diameter'] = 2.2*myones
        self.inputs['lss_wall_thickness'] = 0.45*myones
        self.inputs['nose_wall_thickness'] = 0.1*myones
        self.inputs['bedplate_wall_thickness'] = 0.06*np.ones(npts)
        self.inputs['D_top'] = 6.5

        self.inputs['other_mass'] = 200e3
        self.inputs['mb1_mass'] = 10e3
        self.inputs['mb1_I'] = 10e3*0.5*2**2*np.ones(3)
        self.inputs['mb2_mass'] = 10e3
        self.inputs['mb2_I'] = 10e3*0.5*1.5**2*np.ones(3)

        self.inputs['m_stator'] = 100e3
        self.inputs['cm_stator'] = -0.3
        self.inputs['I_stator'] = np.array([1e6, 5e5, 5e5, 0.0, 0.0, 0.0])
        
        self.inputs['m_rotor'] = 100e3
        self.inputs['cm_rotor'] = -0.3
        self.inputs['I_rotor'] = np.array([1e6, 5e5, 5e5, 0.0, 0.0, 0.0])
        
        self.inputs['F_mb1'] = np.array([2409.750e3, -1716.429e3, 74.3529e3]).reshape((3,1))
        self.inputs['F_mb2'] = np.array([2409.750e3, -1716.429e3, 74.3529e3]).reshape((3,1))
        self.inputs['M_mb1'] = np.array([-1.83291e7, 6171.7324e3, 5785.82946e3]).reshape((3,1))
        self.inputs['M_mb2'] = np.array([-1.83291e7, 6171.7324e3, 5785.82946e3]).reshape((3,1))

        self.inputs['hub_system_mass'] = 100e3
        self.inputs['hub_system_cm'] = 2.0
        self.inputs['hub_system_I'] = np.array([2409.750e3, -1716.429e3, 74.3529e3, 0.0, 0.0, 0.0])
        self.inputs['F_hub'] = np.array([2409.750e3, 0.0, 74.3529e2]).reshape((3,1))
        self.inputs['M_hub'] = np.array([-1.83291e4, 6171.7324e2, 5785.82946e2]).reshape((3,1))
        
        self.inputs['E'] = 210e9
        self.inputs['G'] = 80.8e9
        self.inputs['rho'] = 7850.
        self.inputs['sigma_y'] = 250e6
        self.inputs['gamma_f'] = 1.35
        self.inputs['gamma_m'] = 1.3
        self.inputs['gamma_n'] = 1.0

    def compute_layout(self):
        myobj = lay.Layout(n_points=npts)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        for k in self.outputs.keys():
            self.inputs[k] = self.outputs[k]
        
    def testBaseF_BaseM(self):
        self.inputs['tilt'] = 0.0
        self.inputs['F_mb1'] = np.zeros(3).reshape((3,1))
        self.inputs['F_mb2'] = np.zeros(3).reshape((3,1))
        self.inputs['M_mb1'] = np.zeros(3).reshape((3,1))
        self.inputs['M_mb2'] = np.zeros(3).reshape((3,1))
        self.compute_layout()
        myobj = ds.Nose_Stator_Bedplate_Frame(n_points=npts, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs['base_F'][:2], 0.0)
        npt.assert_almost_equal(self.outputs['base_M'][0], 0.0)
        npt.assert_almost_equal(self.outputs['base_M'][-1], 0.0)
        F0 = self.outputs['base_F']
        M0 = self.outputs['base_M']

        m = self.inputs['mb1_mass']+self.inputs['mb2_mass']+self.inputs['m_stator']+self.outputs['nose_mass']+self.outputs['bedplate_mass']+self.inputs['other_mass']
        
        self.inputs['other_mass'] += 500e3
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs['base_F'][:2], 0.0)
        npt.assert_almost_equal(self.outputs['base_F'][2], F0[2]-500e3*gravity)
        npt.assert_almost_equal(self.outputs['base_M'][0], 0.0)
        npt.assert_almost_equal(self.outputs['base_M'][1], M0[1])
        npt.assert_almost_equal(self.outputs['base_M'][2], 0.0)

        self.inputs['M_mb1'] = 10e3*np.arange(1,4).reshape((3,1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs['base_F'][:2], 0.0)
        npt.assert_almost_equal(self.outputs['base_F'][2], F0[2]-500e3*gravity)
        npt.assert_almost_equal(self.outputs['base_M'], M0+self.inputs['M_mb1'], decimal=0)

        self.inputs['M_mb2'] = 20e3*np.arange(1,4).reshape((3,1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs['base_F'][:2], 0.0)
        npt.assert_almost_equal(self.outputs['base_F'][2], F0[2]-500e3*gravity)
        npt.assert_almost_equal(self.outputs['base_M'], M0+self.inputs['M_mb1']+self.inputs['M_mb2'], decimal=-1)

        self.inputs['F_mb1'] = np.array([30e2, 40e2, 50e2]).reshape((3,1))
        self.inputs['F_mb2'] = np.array([30e2, 40e2, 50e2]).reshape((3,1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs['base_F'][:2], 2*self.inputs['F_mb2'][:2])
        npt.assert_almost_equal(self.outputs['base_F'][2], F0[2]-500e3*gravity+2*50e2)

        
    def testRunRotating(self):
        self.compute_layout()
        myobj = ds.Hub_Rotor_Shaft_Frame(n_dlcs=1)
        myobj.compute(self.inputs, self.outputs)
        
        #self.assertAlmostEqual(self.outputs['mass'], 71319.91743405)
        #npt.assert_almost_equal(self.outputs['base_F'], [ 2409749.99999992, -1716429.        ,  -426142.52407101])
        #npt.assert_almost_equal(self.outputs['base_M'], [-9981495.84990769, 17606406.25680009, 13770321.10702468])

                    
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRun))
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
