import numpy as np
import openmdao.api as om
from wisdem.commonse import gravity, eps
import wisdem.commonse.UtilizationSupplement as Util
from wisdem.towerse.tower import TowerSE


class Tower:
    
    def __init__(self, n_control_points=3, n_refine=3):
        self.n_control_points = n_control_points
        self.n_refine = n_refine

    def setup(self):
        np.random.seed(314)

        # --- geometry ----
        n_control_points = self.n_control_points
        n_refine = self.n_refine
        h_param = np.diff(np.linspace(0.0, 87.6, n_control_points))
        d_param = np.linspace(6.0, 3.87, n_control_points) + np.random.rand(n_control_points)
        t_param = 1.3*np.linspace(0.025, 0.021, n_control_points-1)
        z_foundation = 0.0
        theta_stress = 0.0
        yaw = 0.0
        Koutfitting = 1.07

        # --- material props ---
        E = 210e9
        G = 80.8e9
        rho = 8500.0
        sigma_y = 450.0e6

        # --- extra mass ----
        m = np.array([285598.8])
        mIxx = 1.14930678e+08
        mIyy = 2.20354030e+07
        mIzz = 1.87597425e+07
        mIxy = 0.0
        mIxz = 5.03710467e+05
        mIyz = 0.0
        mI = np.array([mIxx, mIyy, mIzz, mIxy, mIxz, mIyz])
        mrho = np.array([-1.13197635, 0.0, 0.50875268])
        # -----------

        # --- wind ---
        wind_zref = 90.0
        wind_z0 = 0.0
        shearExp = 0.2
        cd_usr = -1.
        # ---------------

        # --- wave ---
        hmax = 0.0
        T = 1.0
        cm = 1.0
        suction_depth = 0.0
        soilG = 140e6
        soilnu = 0.4
        # ---------------

        # --- costs ---
        material_cost = 5.0
        labor_cost    = 100.0/60.0
        painting_cost = 30.0
        # ---------------

        # two load cases.  TODO: use a case iterator

        # # --- loading case 1: max Thrust ---
        wind_Uref1 = 11.73732
        Fx1 = 1284744.19620519
        Fy1 = 0.
        Fz1 = -2914124.84400512 + m*gravity
        Mxx1 = 3963732.76208099
        Myy1 = -2275104.79420872
        Mzz1 = -346781.68192839
        # # ---------------

        # # --- loading case 2: max wind speed ---
        wind_Uref2 = 70.0
        Fx2 = 930198.60063279
        Fy2 = 0.
        Fz2 = -2883106.12368949 + m*gravity
        Mxx2 = -1683669.22411597
        Myy2 = -2522475.34625363
        Mzz2 = 147301.97023764
        # # ---------------

        # Store analysis options
        analysis_options = {}
        analysis_options['materials'] = {}
        analysis_options['monopile'] = {}
        analysis_options['tower'] = {}
        analysis_options['tower']['buckling_length'] = 30.0
        analysis_options['tower']['monopile'] = False

        # --- safety factors ---
        analysis_options['tower']['gamma_f'] = 1.35
        analysis_options['tower']['gamma_m'] = 1.3
        analysis_options['tower']['gamma_n'] = 1.0
        analysis_options['tower']['gamma_b'] = 1.1
        # ---------------

        analysis_options['tower']['gamma_fatigue'] = 1.35*1.3*1.0
        life = 20.0
        # ---------------

        # -----Frame3DD------
        analysis_options['tower']['frame3dd']            = {}
        analysis_options['tower']['frame3dd']['DC']      = 80.0
        analysis_options['tower']['frame3dd']['shear']   = True
        analysis_options['tower']['frame3dd']['geom']    = True
        analysis_options['tower']['frame3dd']['dx']      = 5.0
        analysis_options['tower']['frame3dd']['Mmethod'] = 1
        analysis_options['tower']['frame3dd']['lump']    = 0
        analysis_options['tower']['frame3dd']['tol']     = 1e-9
        analysis_options['tower']['frame3dd']['shift']   = 0.0
        analysis_options['tower']['frame3dd']['add_gravity'] = True
        # ---------------

        # --- constraints ---
        min_d_to_t   = 120.0
        max_taper    = 0.2
        # ---------------

        analysis_options['tower']['n_height'] = len(d_param)
        analysis_options['tower']['n_refine'] = n_refine
        analysis_options['tower']['n_layers'] = 1
        analysis_options['monopile']['n_height'] = 0
        analysis_options['monopile']['n_layers'] = 0
        analysis_options['tower']['wind'] = 'PowerWind'
        analysis_options['tower']['nLC'] = 2
        analysis_options['materials']['n_mat'] = 1

        prob = om.Problem()
        prob.model = TowerSE(analysis_options=analysis_options, topLevelFlag=True)

        prob.setup()

        if analysis_options['tower']['wind'] == 'PowerWind':
            prob['shearExp'] = shearExp

        # --- geometry ----
        prob['hub_height'] = h_param.sum()
        prob['foundation_height'] = 0.0
        #prob['tower_section_height'] = h_param
        prob['tower_s'] = np.cumsum(np.r_[0.0, h_param]) / h_param.sum()
        prob['tower_height'] = h_param.sum()
        prob['tower_outer_diameter_in'] = d_param
        #prob['tower_wall_thickness'] = t_param
        prob['tower_layer_thickness'] = t_param.reshape( (1,len(t_param)) )
        prob['tower_outfitting_factor'] = Koutfitting
        prob['yaw'] = yaw
        prob['suctionpile_depth'] = suction_depth
        prob['suctionpile_depth_diam_ratio'] = 3.25
        prob['G_soil'] = soilG
        prob['nu_soil'] = soilnu
        # --- material props ---
        prob['E_mat'] = E*np.ones((1,3))
        prob['G_mat'] = G*np.ones((1,3))
        prob['rho_mat'] = rho
        prob['sigma_y_mat'] = sigma_y

        # --- extra mass ----
        prob['rna_mass'] = m
        prob['rna_I'] = mI
        prob['rna_cg'] = mrho
        # -----------

        # --- costs ---
        prob['unit_cost_mat'] = material_cost
        prob['labor_cost_rate']    = labor_cost
        prob['painting_cost_rate'] = painting_cost
        # -----------

        # --- wind & wave ---
        prob['wind_reference_height'] = wind_zref
        prob['wind_z0'] = wind_z0
        prob['cd_usr'] = cd_usr
        prob['rho_air'] = 1.225
        prob['mu_air'] = 1.7934e-5
        prob['rho_water'] = 1025.0
        prob['mu_water'] = 1.3351e-3
        prob['beta_wind'] = prob['beta_wave'] = 0.0
        prob['hsig_wave'] = hmax
        prob['Tsig_wave'] = T
        #prob['waveLoads1.U0'] = prob['waveLoads1.A0'] = prob['waveLoads1.beta0'] = prob['waveLoads2.U0'] = prob['waveLoads2.A0'] = prob['waveLoads2.beta0'] = 0.0

        # --- fatigue ---
        #prob['tower_z_DEL'] = z_DEL
        #prob['tower_M_DEL'] = M_DEL
        prob['life'] = life
        #prob['m_SN'] = m_SN
        # ---------------

        # --- constraints ---
        prob['min_d_to_t'] = min_d_to_t
        prob['max_taper'] = max_taper
        # ---------------


        # # --- loading case 1: max Thrust ---
        prob['wind1.Uref'] = wind_Uref1

        prob['pre1.rna_F'] = np.array([Fx1, Fy1, Fz1])
        prob['pre1.rna_M'] = np.array([Mxx1, Myy1, Mzz1])
        # # ---------------


        # # --- loading case 2: max Wind Speed ---
        prob['wind2.Uref'] = wind_Uref2

        prob['pre2.rna_F'] = np.array([Fx2, Fy2, Fz2])
        prob['pre2.rna_M' ] = np.array([Mxx2, Myy2, Mzz2])
        
        self.prob = prob
        
    def run(self, diameters):
        num_cases = diameters.shape[1]
        costs = np.zeros((num_cases))
        
        for i in range(num_cases):
            self.prob['tower_outer_diameter_in'][:] = diameters[:, i]
            self.prob.run_model()
            costs[i] = self.prob['tower_raw_cost']
            
        return costs
        
        
if __name__ == "__main__":
    
    diameters = np.linspace(1.5, 2., 3)**2
    
    low_tower = Tower(n_control_points=3, n_refine=3)
    low_tower.setup()
    high_tower = Tower(n_control_points=3, n_refine=31)
    high_tower.setup()
    
    import numpy as np
    from scipy.interpolate import Rbf
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    from testbed_components import simple_2D_low, simple_2D_high
    from traceback import print_exc

    # Following Algo 2.1 from Andrew March's dissertation
    np.random.seed(3)
    func_low = low_tower.run
    func_high = high_tower.run

    x_init = np.random.rand(3, 12) * 2 + 2
    x = x_init.copy()

    for i in range(20):
        y_low = func_low(x)
        y_high = func_high(x)

        # Construct RBF interpolater for error function
        differences = y_high - y_low
        
        input_arrays = np.split(x, x.shape[0], axis=0)
        
        input_arrays = [x.flatten() for x in input_arrays]
        
        try:
            e = Rbf(*input_arrays, differences, epsilon=0.1)
        except:
            print_exc()
            print("Done!")
            break

        # Create m_k = lofi + RBF
        def m(x):
            input_arrays = np.split(x, x.shape[0], axis=0)
            input_arrays = [x.flatten() for x in input_arrays]
            val = func_low(np.atleast_2d(x).T) + e(*input_arrays)
            return val

        x0 = x[:, -1]
        res = minimize(m, x0, method='SLSQP', tol=1e-6, bounds=[(2., 4.), (2., 4.), (2., 4.)])
        x_new = np.atleast_2d(res.x).T
        x = np.hstack((x, x_new))
        
        print(x_new)
        print(m(x_new))
        print()
        
    print()
    print(f'Number of high-fidelity calls for MFM: {x.shape[1]}')
    print(f'Answer found: {np.squeeze(x_new)}')
    print()

    # res = minimize(func_high, x_init[:, 2], method='SLSQP', tol=1e-6, bounds=[(2., 4.), (2., 4.), (2., 4.)])
    # 
    # print(f'Number of high-fidelity calls for hifi only: {res.nfev}, jac calls: {res.njev}')
    # print(f'Answer found: {res.x}')
    # print()

    
    