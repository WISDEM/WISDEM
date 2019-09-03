import numpy as np
import numpy.testing as npt
import unittest
import wisdem.floatingse.map_mooring as mapMooring
from wisdem.floatingse.column import ColumnGeometry
from wisdem.pymap import pyMAP
from wisdem.commonse import gravity as g

def myisnumber(instr):
    try:
        float(instr)
    except:
        return False
    return True

myones = np.ones((100,))
truth=['---------------------- LINE DICTIONARY ---------------------------------------',
'LineType  Diam      MassDenInAir   EA            CB   CIntDamp  Ca   Cdn    Cdt',
'(-)       (m)       (kg/m)        (N)           (-)   (Pa-s)    (-)  (-)    (-)',
'chain   0.05    49.75   213500000.0   0.65   1.0E8   0.6   -1.0   0.05',
'---------------------- NODE PROPERTIES ---------------------------------------',
'Node Type X     Y    Z   M     V FX FY FZ',
'(-)  (-) (m)   (m)  (m) (kg) (m^3) (kN) (kN) (kN)',
'1   VESSEL   11.0   0.0   -10.0   0.0   0.0   #   #   #',
'2   FIX   175.0   0.0   depth   0.0   0.0   #   #   #',
'---------------------- LINE PROPERTIES ---------------------------------------',
'Line    LineType  UnstrLen  NodeAnch  NodeFair  Flags',
'(-)      (-)       (m)       (-)       (-)       (-)',
'1   chain   235.8   2   1',
'---------------------- SOLVER OPTIONS-----------------------------------------',
'Option',
'(-)',
'help',
' integration_dt 0',
' kb_default 3.0e6',
' cb_default 3.0e5',
' wave_kinematics',
'inner_ftol 1e-5',
'inner_gtol 1e-5',
'inner_xtol 1e-5',
'outer_tol 1e-3',
' pg_cooked 10000 1',
' outer_fd',
' outer_bd',
' outer_cd',
' inner_max_its 200',
' outer_max_its 600',
'repeat 120 240',
' krylov_accelerator 3',
' ref_position 0.0 0.0 0.0']


class TestMapMooring(unittest.TestCase):
    
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}

        self.inputs['wall_thickness'] = np.array([0.5, 0.5, 0.5])
        self.inputs['outer_diameter'] = 2*np.array([10.0, 10.0, 10.0])
        self.inputs['section_height'] = np.array([20.0, 30.0])
        self.inputs['z_param_in'] = self.inputs['z_full_in'] = np.r_[0.0, np.cumsum(self.inputs['section_height'])]-15.0
        self.inputs['section_center_of_mass'] = np.array([10.0, 35.0])
        self.inputs['freeboard'] = 15.0
        self.inputs['fairlead'] = 10.0
        self.inputs['fairlead_radius'] = 11.0
        self.inputs['anchor_radius'] = 175.0

        self.inputs['water_density'] = 1025.0 #1e3
        self.inputs['water_depth'] = 218.0 #100.0

        self.inputs['number_of_mooring_connections'] = 3
        self.inputs['mooring_lines_per_connection'] = 1
        self.inputs['mooring_line_length'] = 0.6*(self.inputs['water_depth'] + self.inputs['anchor_radius'])
        self.inputs['mooring_diameter'] = 0.05
        self.discrete_inputs['mooring_type'] = 'chain'
        self.discrete_inputs['anchor_type'] = 'suctionpile'
        self.inputs['drag_embedment_extra_length'] = 300.0
        self.inputs['max_offset'] = 10.0
        self.inputs['max_survival_heel'] = 10.0
        self.inputs['operational_heel'] = 10.0
        self.inputs['gamma_f'] = 1.35
        self.inputs['max_draft'] = 200.0

        # Needed for geometry prep
        self.inputs['stiffener_web_thickness'] = np.array([0.5, 0.5])
        self.inputs['stiffener_flange_thickness'] = np.array([0.3, 0.3])
        self.inputs['stiffener_web_height']  = np.array([1.0, 1.0])
        self.inputs['stiffener_flange_width'] = np.array([2.0, 2.0])
        self.inputs['stiffener_spacing'] = np.array([0.1, 0.1])
        self.inputs['Hs'] = 5.0
        self.inputs['mooring_cost_factor'] = 1.1

        self.inputs['tower_base_radius'] = 4.0
        
        self.set_geometry()

        self.mymap = mapMooring.MapMooring()
        self.mymap.set_properties(self.inputs, self.discrete_inputs)
        self.mymap.set_geometry(self.inputs, self.outputs)
        #self.mymap.finput = open(mapMooring.FINPUTSTR, 'wb')
        
    #def tearDown(self):
        #self.mymap.finput.close()
        
    def set_geometry(self):
        geom = ColumnGeometry(nSection=2, nFull=3)
        tempUnknowns = {}
        geom.compute(self.inputs, tempUnknowns)
        for pairs in tempUnknowns.items():
            self.inputs[pairs[0]] = pairs[1]

    def testSetProperties(self):
        pass
    '''
    def testWriteLineDict(self):
        self.mymap.write_line_dictionary(self.inputs)
        self.mymap.finput.close()
        A = self.read_input()

    def testWriteNode(self):
        self.mymap.write_node_properties_header()
        self.mymap.write_node_properties(1, 'fix',0,0,0)
        self.mymap.write_node_properties(2, 'vessel',0,0,0)
        self.mymap.finput.close()
        A = self.read_input()

    def testWriteLine(self):
        self.mymap.write_line_properties(self.inputs)
        self.mymap.finput.close()
        A = self.read_input()

    def testWriteSolver(self):
        self.mymap.write_solver_options(self.inputs)
        self.mymap.finput.close()
        A = self.read_input()
    '''
    def testWriteInputAll(self):
        self.mymap.write_input_file(self.inputs, self.discrete_inputs)
        actual = self.mymap.finput[:]
        expect = truth[:]
        self.assertEqual(len(expect), len(actual))
        
        for n in range(len(actual)):
            actualTok = actual[n].split()
            expectTok = expect[n].split()
            self.assertEqual(len(expectTok), len(actualTok))
            
            for k in range(len(actualTok)):
                if myisnumber(actualTok[k]):
                    self.assertEqual( float(actualTok[k]), float(expectTok[k]) )
                else:
                    self.assertEqual( actualTok[k], expectTok[k] )
            
    def testRunMap(self):
        self.mymap.runMAP(self.inputs, self.discrete_inputs, self.outputs)

        self.assertEqual(np.count_nonzero(self.outputs['mooring_neutral_load']), 9)
        self.assertEqual(np.count_nonzero(self.outputs['mooring_stiffness']), 36)
        self.assertEqual(np.count_nonzero(self.outputs['operational_heel_restoring_force']), 9)
        self.assertGreater(np.count_nonzero(self.outputs['mooring_plot_matrix']), 9*20-3)

    def testCost(self):
        self.mymap.compute_cost(self.inputs, self.discrete_inputs, self.outputs)
    
    def testListEntry(self):
        # Initiate MAP++ for this design
        mymap = pyMAP( )
        #mymap.ierr = 0
        mymap.map_set_sea_depth(self.inputs['water_depth'])
        mymap.map_set_gravity(g)
        mymap.map_set_sea_density(self.inputs['water_density'])
        mymap.read_list_input(truth)
        mymap.init( )
        mymap.displace_vessel(0, 0, 0, 0, 10, 0)
        mymap.update_states(0.0, 0)
        mymap.end()
        
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMapMooring))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
