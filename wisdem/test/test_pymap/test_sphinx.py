import numpy as np
import unittest
import numpy.testing as npt

from wisdem.pymap import pyMAP

epsilon = 1e-3 # finite difference epsilon

baseline2 = ['--------------- LINE DICTIONARY -------------------------',
             'LineType  Diam      MassDenInAir   EA        CB   CIntDamp  Ca    Cdn    Cdt',
             '(-)       (m)       (kg/m)        (N)        (-)   (Pa-s)   (-)   (-)    (-)',
             'steel     0.25       320.0     9800000000   1.0    -999.9 -999.9 -999.9 -999.9',
             'nylon     0.30       100.0     980000000    1.0    -999.9 -999.9 -999.9 -999.9',
             '--------------- NODE PROPERTIES -------------------------',
             'Node Type       X       Y       Z      M     B     FX    FY    FZ',
             '(-)  (-)       (m)     (m)     (m)    (kg)  (mˆ3)  (N)   (N)   (N)',
             '1    Fix     400        0     depth    0     0      #     #     #',
             '2    Connect #90       #0    #-80      0     0      0     0     0   ',
             '3    Vessel   20        20    -10      0     0      #     #     #',
             '4    Vessel   20       -20    -10      0     0      #     #     #',
             '--------------- LINE PROPERTIES -------------------------',
             'Line    LineType  UnstrLen  NodeAnch  NodeFair  Flags',
             '(-)      (-)       (m)       (-)       (-)       (-)',
             '1       steel      450        1         2    ',
             '2       nylon      90         2         3    ',
             '3       nylon      90         2         4    ',
             '--------------- SOLVER OPTIONS---------------------------',
             'Option',
             '(-)',
             ' help',
             'outer_tol 1e-5',
             'repeat 120 ',
             'repeat 240']

class Sphinx(unittest.TestCase):

    def setUp(self):

        self.mymoor = pyMAP()

        self.mymoor.map_set_sea_depth(350)      # m
        self.mymoor.map_set_gravity(9.81)       # m/s^2
        self.mymoor.map_set_sea_density(1025.0) # kg/m^3
        self.mymoor.read_list_input(baseline2)
        self.mymoor.init( )
        
    def tearDown(self):
        self.mymoor.end( )

    def test_disp0(self):

        self.mymoor.displace_vessel(0,0,0,0,0,0)
        self.mymoor.update_states(0.0,0)
        
        K = self.mymoor.linear(epsilon)
        Ktruth0 = np.array([[19863.77725926286, 0.0031161762308329344, 0.006781541742384434, -0.08841813541948795, -199805.98371336237, 0.09469501674175262],
                            [-0.0005938290996709839, 19863.794985009008, 4.022149369120598e-05, 199806.2955311034, 0.006879214197397232, -0.5612121894955635],
                            [0.0012314703781157732, 0.000961925252340734, 22706.839430262335, 0.09660981595516205, 0.06568804383277893, 0.027408823370933533],
                            [0.0022764125023968518, 199806.6482993163, 0.00385817838832736, 216811654.7982537, 0.0554393045604229, -52.05222964286804],
                            [-199806.19033204496, 0.0005093897925689816, 0.48789329594001174, 0.09869393706321716, 216811617.59684512, -0.05382020026445389],
                            [-0.001719072344712913, -0.5595305119641125, 0.004395493306219578, -85.25095704942942, 0.04449952021241188, 141222256.10000083]])
        npt.assert_almost_equal(K, Ktruth0, 4)

    def test_disp5(self):

        surge = 5.0 # 5 meter surge displacements
        self.mymoor.displace_vessel(surge,0,0,0,0,0)
        self.mymoor.update_states(0.0,0)

        K = self.mymoor.linear(epsilon)    
        Ktruth5 = np.array([[19581.103663747854, -0.0023128523025661707, 1168.6803222401068, 0.0654004979878664, -215033.93723955378, 0.05570240318775177],
                            [0.001740507286740467, 20717.62133530865, 0.0022237072698771954, 181293.2660083752, 0.13111066073179245, 1722.4661344662309],
                            [1168.728809681852, -6.967457011342049e-05, 23180.921257066075, 0.05636201240122318, -11881.312280893326, 0.02440623939037323],
                            [-0.003891280386596918, 199806.64687338867, -0.004342757165431976, 216811654.6317821, 0.011529773473739624, -52.26798169314861],
                            [-199806.1865159399, 0.0016587000573053956, 0.4864789661951363, -0.03555452078580856, 216811617.5761563, -0.04166644066572189],
                            [0.0007879607146605849, -0.5603800091193989, -0.0038701691664755344, -85.07654152065516, -0.10624760016798973, 141222256.07110184]])
        npt.assert_almost_equal(K, Ktruth5, 4)

        # We need to call update states after linearization to find the equilibrium
        self.mymoor.update_states(0.0,0)
        
        line_number = 0
        H,V = self.mymoor.get_fairlead_force_2d(line_number)
        self.assertAlmostEqual(H, 597513.33, 2)
        self.assertAlmostEqual(V, 1143438.75, 2)

        fx,fy,fz = self.mymoor.get_fairlead_force_3d(line_number)    
        self.assertAlmostEqual(fx, -597513.33, 2)
        self.assertAlmostEqual(fy, 0.0, 2)
        self.assertAlmostEqual(fz, 1143438.75, 2)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Sphinx))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())

