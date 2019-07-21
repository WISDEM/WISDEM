#!/usr/bin/env python
# encoding: utf-8
"""
test_tower_gradients.py

Created by Andrew Ning on 2013-12-20.
Copyright (c) NREL. All rights reserved.
"""

import unittest
import numpy as np
from wisdem.commonse.rna import RNAMass, RotorLoads
# --- tower setup ------
from wisdem.commonse.environment import PowerWind, LogWind
from wisdem.towerse.tower import TowerSE, TowerWindDrag, TowerWaveDrag, TowerDiscretization, GeometricConstraints#, JacketPositioning
from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver


class TotalDerivTestsFlorisAEPOpt(unittest.TestCase):

    def setUp(self):

        # --- geometry ----
        # --- geometry ----
        z_param = np.array([0.0, 43.8, 87.6])
        d_param = np.array([6.0, 4.935, 3.87])
        t_param = np.array([0.027*1.3, 0.023*1.3, 0.019*1.3])
        n = 15
        z_full = np.linspace(0.0, 87.6, n)
        L_reinforced = 30.0*np.ones(n)  # [m] buckling length
        theta_stress = 0.0*np.ones(n)
        yaw = 0.0

        # --- material props ---
        E = 210e9*np.ones(n)
        G = 80.8e9*np.ones(n)
        rho = 8500.0*np.ones(n)
        sigma_y = 450.0e6*np.ones(n)

        # --- spring reaction data.  Use float('inf') for rigid constraints. ---
        kidx = np.array([0], dtype=int)  # applied at base
        kx = np.array([float('inf')])
        ky = np.array([float('inf')])
        kz = np.array([float('inf')])
        ktx = np.array([float('inf')])
        kty = np.array([float('inf')])
        ktz = np.array([float('inf')])
        nK = len(kidx)

        # --- extra mass ----
        midx = np.array([n-1], dtype=int)  # RNA mass at top
        m = np.array([285598.8])
        mIxx = np.array([1.14930678e+08])
        mIyy = np.array([2.20354030e+07])
        mIzz = np.array([1.87597425e+07])
        mIxy = np.array([0.00000000e+00])
        mIxz = np.array([5.03710467e+05])
        mIyz = np.array([0.00000000e+00])
        mrhox = np.array([-1.13197635])
        mrhoy = np.array([0.])
        mrhoz = np.array([0.50875268])
        nMass = len(midx)
        addGravityLoadForExtraMass = True
        # -----------

        # --- wind ---
        wind_zref = 90.0
        wind_z0 = 0.0
        shearExp = 0.2
        # ---------------

        # if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also
        # # --- loading case 1: max Thrust ---
        wind_Uref1 = 11.73732
        plidx1 = np.array([n-1], dtype=int)  # at  top
        Fx1 = np.array([1284744.19620519])
        Fy1 = np.array([0.])
        Fz1 = np.array([-2914124.84400512])
        Mxx1 = np.array([3963732.76208099])
        Myy1 = np.array([-2275104.79420872])
        Mzz1 = np.array([-346781.68192839])
        nPL = len(plidx1)
        # # ---------------

        # # --- loading case 2: max wind speed ---
        wind_Uref2 = 70.0
        plidx2 = np.array([n-1], dtype=int)  # at  top
        Fx2 = np.array([930198.60063279])
        Fy2 = np.array([0.])
        Fz2 = np.array([-2883106.12368949])
        Mxx2 = np.array([-1683669.22411597])
        Myy2 = np.array([-2522475.34625363])
        Mzz2 = np.array([147301.97023764])
        # # ---------------

        # --- safety factors ---
        gamma_f = 1.35
        gamma_m = 1.3
        gamma_n = 1.0
        gamma_b = 1.1
        # ---------------

        # --- fatigue ---
        z_DEL = np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
        M_DEL = 1e3*np.array([8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003, 7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003, 5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003, 3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003, 2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003, 1.2673E+003, 1.2761E+003])
        nDEL = len(z_DEL)
        gamma_fatigue = 1.35*1.3*1.0
        life = 20.0
        m_SN = 4
        # ---------------


        # --- constraints ---
        min_d_to_t = 120.0
        min_taper = 0.4
        # ---------------

        # # V_max = 80.0  # tip speed
        # # D = 126.0
        # # .freq1p = V_max / (D/2) / (2*pi)  # convert to Hz

        nPoints = len(z_param)
        nFull = len(z_full)
        wind = 'PowerWind'

        prob = Problem()
        root = prob.root = Group()

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.opt_settings['Major iterations limit'] = 1000

        root.add('z_param', IndepVarComp('z_param', z_param))
        root.add('d_param', IndepVarComp('d_param', d_param))
        root.add('t_param', IndepVarComp('t_param', t_param))
        root.add('TowerSE', TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind))


        prob.driver.add_objective('TowerSE.tower1.mass', scaler=1E-6)
        prob.driver.add_desvar('z_param.z_param', lower=np.zeros(nPoints), upper=np.ones(nPoints)*1000., scaler=1E-2)
        prob.driver.add_desvar('t_param.t_param', lower=np.ones(nPoints)*0.001, upper=np.ones(nPoints)*1000., scaler=1E-6)
        prob.driver.add_desvar('d_param.d_param', np.array([2,2.1,2.2]), upper=np.ones(nPoints)*1000., scaler=1E-6)

        prob.root.connect('z_param.z_param', 'TowerSE.z_param')
        prob.root.connect('d_param.d_param', 'TowerSE.d_param')
        prob.root.connect('t_param.t_param', 'TowerSE.t_param')

        prob.driver.add_constraint('TowerSE.tower1.stress', upper=np.ones(n))
        prob.driver.add_constraint('TowerSE.tower2.stress', upper=np.ones(n))
        prob.driver.add_constraint('TowerSE.tower1.global_buckling', upper=np.ones(n))
        prob.driver.add_constraint('TowerSE.tower2.global_buckling', upper=np.ones(n))
        prob.driver.add_constraint('TowerSE.tower1.shell_buckling', upper=np.ones(n))
        prob.driver.add_constraint('TowerSE.tower2.shell_buckling', upper=np.ones(n))
        prob.driver.add_constraint('TowerSE.tower1.damage', upper=np.ones(n)*0.8)
        prob.driver.add_constraint('TowerSE.gc.weldability', upper=np.zeros(n))
        prob.driver.add_constraint('TowerSE.gc.manufacturability', upper=np.zeros(n))
        freq1p = 0.2  # 1P freq in Hz
        prob.driver.add_constraint('TowerSE.tower1.f1', lower=1.1*freq1p)
        prob.driver.add_constraint('TowerSE.tower2.f1', lower=1.1*freq1p)

        prob.setup()

        if wind=='PowerWind':
            prob['TowerSE.wind1.shearExp'] = shearExp
            prob['TowerSE.wind2.shearExp'] = shearExp

        # assign values to params

        # --- geometry ----
        #prob['TowerSE.z_param'] = z_param
        #prob['TowerSE.d_param'] = d_param
        #prob['TowerSE.t_param'] = t_param
        prob['TowerSE.z_full'] = z_full
        prob['TowerSE.tower1.L_reinforced'] = L_reinforced
        prob['TowerSE.distLoads1.yaw'] = yaw

        # --- material props ---
        prob['TowerSE.tower1.E'] = E
        prob['TowerSE.tower1.G'] = G
        prob['TowerSE.tower1.rho'] = rho
        prob['TowerSE.tower1.sigma_y'] = sigma_y

        # --- spring reaction data.  Use float('inf') for rigid constraints. ---
        prob['TowerSE.tower1.kidx'] = kidx
        prob['TowerSE.tower1.kx'] = kx
        prob['TowerSE.tower1.ky'] = ky
        prob['TowerSE.tower1.kz'] = kz
        prob['TowerSE.tower1.ktx'] = ktx
        prob['TowerSE.tower1.kty'] = kty
        prob['TowerSE.tower1.ktz'] = ktz

        # --- extra mass ----
        prob['TowerSE.tower1.midx'] = midx
        prob['TowerSE.tower1.m'] = m
        prob['TowerSE.tower1.mIxx'] = mIxx
        prob['TowerSE.tower1.mIyy'] = mIyy
        prob['TowerSE.tower1.mIzz'] = mIzz
        prob['TowerSE.tower1.mIxy'] = mIxy
        prob['TowerSE.tower1.mIxz'] = mIxz
        prob['TowerSE.tower1.mIyz'] = mIyz
        prob['TowerSE.tower1.mrhox'] = mrhox
        prob['TowerSE.tower1.mrhoy'] = mrhoy
        prob['TowerSE.tower1.mrhoz'] = mrhoz
        prob['TowerSE.tower1.addGravityLoadForExtraMass'] = addGravityLoadForExtraMass
        # -----------

        # --- wind ---
        prob['TowerSE.wind1.zref'] = wind_zref
        prob['TowerSE.wind1.z0'] = wind_z0
        # ---------------

        # # --- loading case 1: max Thrust ---
        prob['TowerSE.wind1.Uref'] = wind_Uref1
        prob['TowerSE.tower1.plidx'] = plidx1
        prob['TowerSE.tower1.Fx'] = Fx1
        prob['TowerSE.tower1.Fy'] = Fy1
        prob['TowerSE.tower1.Fz'] = Fz1
        prob['TowerSE.tower1.Mxx'] = Mxx1
        prob['TowerSE.tower1.Myy'] = Myy1
        prob['TowerSE.tower1.Mzz'] = Mzz1
        # # ---------------

        # # --- loading case 2: max Wind Speed ---
        prob['TowerSE.wind2.Uref'] = wind_Uref2
        prob['TowerSE.tower2.plidx'] = plidx2
        prob['TowerSE.tower2.Fx'] = Fx2
        prob['TowerSE.tower2.Fy'] = Fy2
        prob['TowerSE.tower2.Fz'] = Fz2
        prob['TowerSE.tower2.Mxx'] = Mxx2
        prob['TowerSE.tower2.Myy'] = Myy2
        prob['TowerSE.tower2.Mzz'] = Mzz2
        # # ---------------

        # --- safety factors ---
        prob['TowerSE.tower1.gamma_f'] = gamma_f
        prob['TowerSE.tower1.gamma_m'] = gamma_m
        prob['TowerSE.tower1.gamma_n'] = gamma_n
        prob['TowerSE.tower1.gamma_b'] = gamma_b
        # ---------------

        # --- fatigue ---
        prob['TowerSE.tower1.z_DEL'] = z_DEL
        prob['TowerSE.tower1.M_DEL'] = M_DEL
        prob['TowerSE.tower1.gamma_fatigue'] = gamma_fatigue
        prob['TowerSE.tower1.life'] = life
        prob['TowerSE.tower1.m_SN'] = m_SN
        # ---------------

        # --- constraints ---
        prob['TowerSE.gc.min_d_to_t'] = min_d_to_t
        prob['TowerSE.gc.min_taper'] = min_taper
        # ---------------

        # # --- run ---
        prob.run()

        print prob['TowerSE.gc.weldability']
        print prob['TowerSE.gc.manufacturability']

        self.J = prob.check_total_derivatives(out_stream=None)
        """
        self.connect('tower1.mass', 'mass')
        self.connect('tower1.f1', 'f1')
        self.connect('tower1.f2', 'f2')
        self.connect('tower1.top_deflection', 'top_deflection1')
        self.connect('tower2.top_deflection', 'top_deflection2')
        self.connect('tower1.stress', 'stress1')
        self.connect('tower2.stress', 'stress2')
        self.connect('tower1.global_buckling', 'global_buckling1')
        self.connect('tower2.global_buckling', 'global_buckling2')
        self.connect('tower1.shell_buckling', 'shell_buckling1')
        self.connect('tower2.shell_buckling', 'shell_buckling2')
        self.connect('tower1.damage', 'damage')
        self.connect('gc.weldability', 'weldability')
        self.connect('gc.manufacturability', 'manufacturability')
        """
    def testMass(self):
        np.testing.assert_allclose(self.J[('TowerSE.tower1.mass', 'z_param.z_param')]['J_fwd'], self.J[('TowerSE.tower1.mass', 'z_param.z_param')]['J_fd'], 1e-6, 1e-6)
        np.testing.assert_allclose(self.J[('TowerSE.tower1.mass', 'd_param.d_param')]['J_fwd'], self.J[('TowerSE.tower1.mass', 'd_param.d_param')]['J_fd'], 1e-6, 1e-6)
        np.testing.assert_allclose(self.J[('TowerSE.tower1.mass', 't_param.t_param')]['J_fwd'], self.J[('TowerSE.tower1.mass', 't_param.t_param')]['J_fd'], 1e-6, 1e-6)


"""
class TestTowerWindDrag(unittest.TestCase):

    def test1(self):

        twd = TowerWindDrag()
        twd.U = [0., 8.80496275, 10.11424623, 10.96861453, 11.61821801, 12.14846828, 12.59962946, 12.99412772, 13.34582791, 13.66394248, 13.95492553, 14.22348635, 14.47317364, 14.70673252, 14.92633314, 15.13372281, 15.33033057, 15.51734112, 15.69574825, 15.86639432, 16.03]
        twd.z = [0., 4.38, 8.76, 13.14, 17.52, 21.9, 26.28, 30.66, 35.04, 39.42, 43.8, 48.18, 52.56, 56.94, 61.32, 65.7, 70.08, 74.46, 78.84, 83.22, 87.6]
        twd.d = [6., 5.8935, 5.787, 5.6805, 5.574, 5.4675, 5.361, 5.2545, 5.148, 5.0415, 4.935, 4.8285, 4.722, 4.6155, 4.509, 4.4025, 4.296, 4.1895, 4.083, 3.9765, 3.87]
        twd.beta = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        twd.rho = 1.225
        twd.mu = 1.7934e-05

        names, errors = check_gradient(twd)

        for name, err in zip(names, errors):

            if name == 'd_windLoads.Px[0] / d_U[0]':
                tol = 2e-5  # central difference not accurate right at Re=0
            else:
                tol = 1e-6

            try:
                self.assertLessEqual(err, tol)
            except AssertionError, e:
                print '*** error in:', name
                raise e



class TestTowerWaveDrag(unittest.TestCase):

    def test1(self):

        twd = TowerWaveDrag()
        twd.U = [0., 8.80496275, 10.11424623, 10.96861453, 11.61821801, 12.14846828, 12.59962946, 12.99412772, 13.34582791, 13.66394248, 13.95492553, 14.22348635, 14.47317364, 14.70673252, 14.92633314, 15.13372281, 15.33033057, 15.51734112, 15.69574825, 15.86639432, 16.03]
        twd.z = [0., 4.38, 8.76, 13.14, 17.52, 21.9, 26.28, 30.66, 35.04, 39.42, 43.8, 48.18, 52.56, 56.94, 61.32, 65.7, 70.08, 74.46, 78.84, 83.22, 87.6]
        twd.d = [6., 5.8935, 5.787, 5.6805, 5.574, 5.4675, 5.361, 5.2545, 5.148, 5.0415, 4.935, 4.8285, 4.722, 4.6155, 4.509, 4.4025, 4.296, 4.1895, 4.083, 3.9765, 3.87]
        twd.beta = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        twd.rho = 1.225
        twd.mu = 1.7934e-05

        twd.A = 1.1*twd.U
        twd.cm = 2.0

        names, errors = check_gradient(twd)

        for name, err in zip(names, errors):

            if name == 'd_waveLoads.Px[0] / d_U[0]':
                tol = 2e-5  # central difference not accurate right at Re=0
            else:
                tol = 1e-6

            try:
                self.assertLessEqual(err, tol)
            except AssertionError, e:
                print '*** error in:', name
                raise e



class TestTowerDiscretization(unittest.TestCase):

    def test1(self):

        td = TowerDiscretization()
        td.towerHeight = 87.6
        td.z = np.array([0.0, 0.5, 1.0])
        td.d = np.array([6.0, 4.935, 3.87])
        td.t = np.array([0.0351, 0.0299, 0.0247])
        td.L_reinforced = np.array([29.2, 29.2, 29.2])
        td.n = np.array([10, 7])
        # td.n_reinforced = 3

        check_gradient_unit_test(self, td)


    def test2(self):

        td = TowerDiscretization()
        td.towerHeight = np.random.rand(1)[0]
        td.z = np.array([0.0, 0.5, 1.0])
        td.d = np.random.rand(3)
        td.t = np.random.rand(3)
        td.L_reinforced = np.array([29.2, 29.2, 29.2])
        td.n = np.array([10, 7])
        # td.n_reinforced = 3

        check_gradient_unit_test(self, td)


    def test3(self):

        td = TowerDiscretization()
        td.towerHeight = 87.6
        td.z = np.array([0.0, 0.5, 1.0])
        td.d = np.array([6.0, 4.935, 3.87])
        td.t = np.array([0.0351, 0.0299, 0.0247])
        td.L_reinforced = np.array([29.2, 29.2, 29.2])
        td.n = np.array([10, 7])

        td.monopileHeight = 30.0
        td.d_monopile = 6.0
        td.t_monopile = 0.0351
        td.n_monopile = 5

        check_gradient_unit_test(self, td)

class TestJacketPositioning(unittest.TestCase):

    def test1(self):

        jp = JacketPositioning()
        # inputs
        jp.sea_depth = 20.0
        jp.tower_length = 87.6
        jp.tower_to_shaft = 2.0
        jp.monopile_extension = 5.0
        jp.deck_height = 15.0
        jp.d_monopile = 6.0
        jp.t_monopile = 0.06
        jp.t_jacket = 0.05
        jp.d_tower_base = 6.0
        jp.d_tower_top = 3.87
        jp.t_tower_base = 0.027
        jp.t_tower_top = 0.019

        check_gradient_unit_test(self, jp, display=True)

# TODO: move these to commonse tests - remove RNA content from tower
class TestRNAMass(unittest.TestCase):

    def test1(self):

        rna = RNAMass()
        rna.blades_mass = 15241.323 * 3
        rna.hub_mass = 50421.4
        rna.nac_mass = 221245.8
        rna.hub_cm = [-6.3, 0., 3.15]
        rna.nac_cm = [-0.32, 0., 2.4]
        rna.blades_I = [26375976., 13187988., 13187988., 0., 0., 0.]
        rna.hub_I = [127297.8, 127297.8, 127297.8, 0., 0., 0.]
        rna.nac_I = [9908302.58, 912488.28, 1160903.54, 0., 0., 0.]

        check_gradient_unit_test(self, rna, tol=1e-5)


    def test2(self):

        rna = RNAMass()
        rna.blades_mass = np.random.rand(1)[0]
        rna.hub_mass = np.random.rand(1)[0]
        rna.nac_mass = np.random.rand(1)[0]
        rna.hub_cm = np.random.rand(3)
        rna.nac_cm = np.random.rand(3)
        rna.blades_I = np.random.rand(6)
        rna.hub_I = np.random.rand(6)
        rna.nac_I = np.random.rand(6)

        check_gradient_unit_test(self, rna)



class TestRotorLoads(unittest.TestCase):

    def test1(self):

        loads = RotorLoads()
        loads.F = [123.0, 0.0, 0.0]
        loads.M = [4843.0, 0.0, 0.0]
        loads.r_hub = [2.0, -3.2, 4.5]
        loads.rna_cm = [-3.0, 1.6, -4.0]
        loads.m_RNA = 200.0
        loads.tilt = 13.2
        loads.g = 9.81

        check_gradient_unit_test(self, loads, tol=2e-6)


    def test2(self):

        loads = RotorLoads()
        loads.F = [123.0, 0.0, 0.0]
        loads.M = [4843.0, 0.0, 0.0]
        loads.r_hub = [2.0, -3.2, 4.5]
        loads.rna_cm = [-3.0, 1.6, -4.0]
        loads.m_RNA = 200.0
        loads.tilt = 13.2
        loads.g = 9.81
        loads.downwind = True

        check_gradient_unit_test(self, loads, tol=2e-6)


    def test3(self):

        loads = RotorLoads()
        loads.F = [123.0, 101.0, -50.0]
        loads.M = [4843.0, -2239.0, 1232.0]
        loads.r_hub = [2.0, -3.2, 4.5]
        loads.rna_cm = [-3.0, 1.6, -4.0]
        loads.m_RNA = 200.0
        loads.tilt = 13.2
        loads.g = 9.81

        check_gradient_unit_test(self, loads)


    def test4(self):

        loads = RotorLoads()
        loads.F = [123.0, 101.0, -50.0]
        loads.M = [4843.0, -2239.0, 1232.0]
        loads.r_hub = [2.0, -3.2, 4.5]
        loads.rna_cm = [-3.0, 1.6, -4.0]
        loads.m_RNA = 200.0
        loads.tilt = 13.2
        loads.g = 9.81
        loads.rna_weightM = False

        check_gradient_unit_test(self, loads)




class TestGeometricConstraints(unittest.TestCase):

    def test1(self):

        gc = GeometricConstraints()
        gc.d = [4.0, 3.0, 2.0]
        gc.t = [0.4, 0.23, 0.14]


        check_gradient_unit_test(self, gc)

"""

if __name__ == "__main__":

    # fast = unittest.TestSuite()
    # fast.addTest(TestTowerDiscretization('test3'))
    # unittest.TextTestRunner().run(fast)


    unittest.main()
