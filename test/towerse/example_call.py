import numpy as np
from openmdao.api import Group, Problem, IndepVarComp
from wisdem.towerse.tower import *
from wisdem.commonse.environment import PowerWind, LogWind
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # --- tower setup -----

    # --- geometry ----

    z_param = np.array([0.0, 20, 43.8, 87.6])
    d_param = np.array([6.0, 5.0, 4.935, 3.87])
    t_param = np.array([0.027*1.3, .024*1.3, 0.023*1.3, 0.019*1.3])
    n = 50
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
    kidx = np.array([0])  # applied at base
    kx = np.array([float('inf')])
    ky = np.array([float('inf')])
    kz = np.array([float('inf')])
    ktx = np.array([float('inf')])
    kty = np.array([float('inf')])
    ktz = np.array([float('inf')])
    nK = len(kidx)

    # --- extra mass ----
    midx = np.array([n-1])  # RNA mass at top
    m = [285598.8]
    mIxx = [1.14930678e+08]
    mIyy = [2.20354030e+07]
    mIzz = [1.87597425e+07]
    mIxy = [0.00000000e+00]
    mIxz = [5.03710467e+05]
    mIyz = [0.00000000e+00]
    mrhox = [-1.13197635]
    mrhoy = [0.]
    mrhoz = [0.50875268]
    nMass = len(m)
    addGravityLoadForExtraMass = True
    # -----------

    # --- wind ---
    wind_zref = 90.0
    wind_z0 = 0.0
    # ---------------

    # if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also
    # # --- loading case 1: max Thrust ---
    wind_Uref1 = 11.73732
    plidx1 = [n-1]  # at  top
    Fx1 = [1284744.19620519]
    Fy1 = [0.]
    Fz1 = [-2914124.84400512]
    Mxx1 = [3963732.76208099]
    Myy1 = [-2275104.79420872]
    Mzz1 = [-346781.68192839]
    nPL = len(plidx1)
    # # ---------------

    # # --- loading case 2: max wind speed ---
    wind_Uref2 = 70.0
    plidx2 = [n-1]  # at  top
    Fx2 = [930198.60063279]
    Fy2 = [0.]
    Fz2 = [-2883106.12368949]
    Mxx2 = [-1683669.22411597]
    Myy2 = [-2522475.34625363]
    Mzz2 = [147301.97023764]
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

    """
    z_param = [0.0, 43.8, 150]
    d_param = [6.0, 4.935, 3.87]
    t_param = [0.027*1.3, 0.023*1.3, 0.019*1.3]
    n = 15
    z_full = np.linspace(0.0, 150, n)
    L_reinforced = 30.0*np.ones(n)  # [m] buckling length
    yaw = 0.0
    # --- wind ---
    wind_zref = 100
    wind_z0 = 0.0
    wind_Uref1 = 13
    wind_Uref2 = 150
    # ---------------
    """


    nPoints = len(z_param)
    nFull = len(z_full)
    wind = 'PowerWind'

    prob = Problem(root=TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind))

    prob.setup()

    if wind=='PowerWind':
        prob['wind1.shearExp'] = 0.2
        prob['wind2.shearExp'] = 0.2

    # assign values to params

    # --- geometry ----
    prob['z_param'] = z_param
    prob['d_param'] = d_param
    prob['t_param'] = t_param
    prob['z_full'] = z_full
    prob['tower1.L_reinforced'] = L_reinforced
    prob['distLoads1.yaw'] = yaw

    # --- material props ---
    prob['tower1.E'] = E
    prob['tower1.G'] = G
    prob['tower1.rho'] = rho
    prob['tower1.sigma_y'] = sigma_y

    # --- spring reaction data.  Use float('inf') for rigid constraints. ---
    prob['tower1.kidx'] = kidx
    prob['tower1.kx'] = kx
    prob['tower1.ky'] = ky
    prob['tower1.kz'] = kz
    prob['tower1.ktx'] = ktx
    prob['tower1.kty'] = kty
    prob['tower1.ktz'] = ktz

    # --- extra mass ----
    prob['tower1.midx'] = midx
    prob['tower1.m'] = m
    prob['tower1.mIxx'] = mIxx
    prob['tower1.mIyy'] = mIyy
    prob['tower1.mIzz'] = mIzz
    prob['tower1.mIxy'] = mIxy
    prob['tower1.mIxz'] = mIxz
    prob['tower1.mIyz'] = mIyz
    prob['tower1.mrhox'] = mrhox
    prob['tower1.mrhoy'] = mrhoy
    prob['tower1.mrhoz'] = mrhoz
    prob['tower1.addGravityLoadForExtraMass'] = addGravityLoadForExtraMass
    # -----------

    # --- wind ---
    prob['wind1.zref'] = wind_zref
    prob['wind1.z0'] = wind_z0
    # ---------------

    # # --- loading case 1: max Thrust ---
    prob['wind1.Uref'] = wind_Uref1
    prob['tower1.plidx'] = plidx1
    prob['tower1.Fx'] = Fx1
    prob['tower1.Fy'] = Fy1
    prob['tower1.Fz'] = Fz1
    prob['tower1.Mxx'] = Mxx1
    prob['tower1.Myy'] = Myy1
    prob['tower1.Mzz'] = Mzz1
    # # ---------------

    # # --- loading case 2: max Wind Speed ---
    prob['wind2.Uref'] = wind_Uref2
    prob['tower2.plidx'] = plidx2
    prob['tower2.Fx'] = Fx2
    prob['tower2.Fy'] = Fy2
    prob['tower2.Fz'] = Fz2
    prob['tower2.Mxx'] = Mxx2
    prob['tower2.Myy'] = Myy2
    prob['tower2.Mzz'] = Mzz2
    # # ---------------

    # --- safety factors ---
    prob['tower1.gamma_f'] = gamma_f
    prob['tower1.gamma_m'] = gamma_m
    prob['tower1.gamma_n'] = gamma_n
    prob['tower1.gamma_b'] = gamma_b
    # ---------------

    # --- fatigue ---
    prob['tower1.z_DEL'] = z_DEL
    prob['tower1.M_DEL'] = M_DEL
    prob['tower1.gamma_fatigue'] = gamma_fatigue
    prob['tower1.life'] = life
    prob['tower1.m_SN'] = m_SN
    # ---------------

    # --- constraints ---
    prob['gc.min_d_to_t'] = min_d_to_t
    prob['gc.min_taper'] = min_taper
    # ---------------

    """
    # ---- tower ------
    prob.replace('wind1', PowerWind())
    prob.replace('wind2', PowerWind())
    # onshore (no waves)
    """




    # # --- run ---
    prob.run()

    z = prob['z_full']

    print 'mass (kg) =', prob['tower1.mass']
    print 'f1 (Hz) =', prob['tower1.f1']
    print 'f2 (Hz) =', prob['tower1.f2']
    print 'top_deflection1 (m) =', prob['tower1.top_deflection']
    print 'top_deflection2 (m) =', prob['tower2.top_deflection']
    print 'weldability =', prob['gc.weldability']
    print 'manufacturability =', prob['gc.manufacturability']
    print 'stress1 =', prob['tower1.stress']
    print 'stress2 =', prob['tower2.stress']
    print 'zs=', z
    print 'ds=', prob['d_full']
    print 'ts=', prob['t_full']
    print 'GL buckling =', prob['tower1.global_buckling']
    print 'GL buckling =', prob['tower2.global_buckling']
    print 'Shell buckling =', prob['tower1.shell_buckling']
    print 'Shell buckling =', prob['tower2.shell_buckling']
    print 'damage =', prob['tower1.damage']


    plt.figure(figsize=(5.0, 3.5))
    plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    plt.plot(prob['tower1.stress'], z, label='stress1')
    plt.plot(prob['tower2.stress'], z, label='stress2')
    plt.plot(prob['tower1.shell_buckling'], z, label='shell buckling 1')
    plt.plot(prob['tower2.shell_buckling'], z, label='shell buckling 2')
    plt.plot(prob['tower1.global_buckling'], z, label='global buckling 1')
    plt.plot(prob['tower2.global_buckling'], z, label='global buckling 2')
    plt.plot(prob['tower1.damage'], z, label='damage')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2)
    plt.xlabel('utilization')
    plt.ylabel('height along tower (m)')

    #plt.figure(2)
    #plt.plot(prob['t_full']/2.+max(prob['t_full']), z, 'ok')
    #plt.plot(max(prob['t_full'])-prob['t_full']/2., z, 'ok')

    fig = plt.figure(3)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(prob['wind1.U'], z)
    ax2.plot(prob['wind2.U'], z)
    plt.tight_layout()

    plt.show()

    # ------------

    optimize = True

    if optimize:

        # --- optimizer imports ---
        from openmdao.api import pyOptSparseDriver
        # ----------------------

        # --- Setup Optimizer ---
        N = len(z_param)
        prob = Problem(root=TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind))
        prob.root.add('p1', IndepVarComp('z_param', z_param))
        prob.root.add('p2', IndepVarComp('d_param', d_param))
        prob.root.add('p3', IndepVarComp('t_param', t_param))


        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.opt_settings['Major iterations limit'] = 1000
        # ----------------------

        # --- Objective ---
        prob.driver.add_objective('tower1.mass')
        # ----------------------

        # --- Design Variables ---

        prob.driver.add_desvar('p1.z_param', lower=np.ones(N)*0.0, upper=np.ones(N)*max(z_param))
        prob.driver.add_desvar('p2.d_param', lower=np.ones(N)*3, upper=np.ones(N)*15.0)
        prob.driver.add_desvar('p3.t_param', lower=np.ones(N)*0.005, upper=np.ones(N)*0.2)

        prob.root.connect('p1.z_param', 'z_param')
        prob.root.connect('p2.d_param', 'd_param')
        prob.root.connect('p3.t_param', 't_param')
        # ----------------------


        # --- Constraints ---
        prob.driver.add_constraint('tower1.stress', upper=np.ones(n))
        prob.driver.add_constraint('tower2.stress', upper=np.ones(n))
        prob.driver.add_constraint('tower1.global_buckling', upper=np.ones(n))
        prob.driver.add_constraint('tower2.global_buckling', upper=np.ones(n))
        prob.driver.add_constraint('tower1.shell_buckling', upper=np.ones(n))
        prob.driver.add_constraint('tower2.shell_buckling', upper=np.ones(n))
        prob.driver.add_constraint('tower1.damage', upper=np.ones(n)*0.8)
        prob.driver.add_constraint('gc.weldability', upper=np.zeros(N))
        prob.driver.add_constraint('gc.manufacturability', upper=np.zeros(N))
        freq1p = 0.2  # 1P freq in Hz
        prob.driver.add_constraint('tower1.f1', lower=1.1*freq1p)
        prob.driver.add_constraint('tower2.f1', lower=1.1*freq1p)
        # ----------------------

        prob.setup()


        if wind=='PowerWind':
            prob['wind1.shearExp'] = 0.2
            prob['wind2.shearExp'] = 0.2

        # assign values to params

        # --- geometry ----
        #prob['z_param'] = z_param
        #prob['d_param'] = d_param
        #prob['t_param'] = t_param
        prob['z_full'] = z_full
        prob['tower1.L_reinforced'] = L_reinforced
        prob['distLoads1.yaw'] = yaw

        # --- material props ---
        prob['tower1.E'] = E
        prob['tower1.G'] = G
        prob['tower1.rho'] = rho
        prob['tower1.sigma_y'] = sigma_y

        # --- spring reaction data.  Use float('inf') for rigid constraints. ---
        prob['tower1.kidx'] = kidx
        prob['tower1.kx'] = kx
        prob['tower1.ky'] = ky
        prob['tower1.kz'] = kz
        prob['tower1.ktx'] = ktx
        prob['tower1.kty'] = kty
        prob['tower1.ktz'] = ktz

        # --- extra mass ----
        prob['tower1.midx'] = midx
        prob['tower1.m'] = m
        prob['tower1.mIxx'] = mIxx
        prob['tower1.mIyy'] = mIyy
        prob['tower1.mIzz'] = mIzz
        prob['tower1.mIxy'] = mIxy
        prob['tower1.mIxz'] = mIxz
        prob['tower1.mIyz'] = mIyz
        prob['tower1.mrhox'] = mrhox
        prob['tower1.mrhoy'] = mrhoy
        prob['tower1.mrhoz'] = mrhoz
        prob['tower1.addGravityLoadForExtraMass'] = addGravityLoadForExtraMass
        # -----------

        # --- wind ---
        prob['wind1.zref'] = wind_zref
        prob['wind1.z0'] = wind_z0
        # ---------------

        # # --- loading case 1: max Thrust ---
        prob['wind1.Uref'] = wind_Uref1
        prob['tower1.plidx'] = plidx1
        prob['tower1.Fx'] = Fx1
        prob['tower1.Fy'] = Fy1
        prob['tower1.Fz'] = Fz1
        prob['tower1.Mxx'] = Mxx1
        prob['tower1.Myy'] = Myy1
        prob['tower1.Mzz'] = Mzz1
        # # ---------------

        # # --- loading case 2: max Wind Speed ---
        prob['wind2.Uref'] = wind_Uref2
        prob['tower2.plidx'] = plidx2
        prob['tower2.Fx'] = Fx2
        prob['tower2.Fy'] = Fy2
        prob['tower2.Fz'] = Fz2
        prob['tower2.Mxx'] = Mxx2
        prob['tower2.Myy'] = Myy2
        prob['tower2.Mzz'] = Mzz2
        # # ---------------

        # --- safety factors ---
        prob['tower1.gamma_f'] = gamma_f
        prob['tower1.gamma_m'] = gamma_m
        prob['tower1.gamma_n'] = gamma_n
        prob['tower1.gamma_b'] = gamma_b
        # ---------------

        # --- fatigue ---
        prob['tower1.z_DEL'] = z_DEL
        prob['tower1.M_DEL'] = M_DEL
        prob['tower1.gamma_fatigue'] = gamma_fatigue
        prob['tower1.life'] = life
        prob['tower1.m_SN'] = m_SN
        # ---------------

        # --- constraints ---
        prob['gc.min_d_to_t'] = min_d_to_t
        prob['gc.min_taper'] = min_taper
        # ---------------

        # --- run opt ---
        prob.run()
        # ---------------

        print 'Mass: ', prob['tower1.mass']
        print 'Z: ', prob['p1.z_param']
        print 'D: ', prob['p2.d_param']
        print 'T: ', prob['p3.t_param']

        plt.figure(figsize=(5.0, 3.5))
        plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
        plt.plot(prob['tower1.stress'], z, label='stress1')
        plt.plot(prob['tower2.stress'], z, label='stress2')
        plt.plot(prob['tower1.shell_buckling'], z, label='shell buckling 1')
        plt.plot(prob['tower2.shell_buckling'], z, label='shell buckling 2')
        plt.plot(prob['tower1.global_buckling'], z, label='global buckling 1')
        plt.plot(prob['tower2.global_buckling'], z, label='global buckling 2')
        plt.plot(prob['tower1.damage'], z, label='damage')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2)
        plt.xlabel('utilization')
        plt.ylabel('height along tower (m)')

        plt.show()
