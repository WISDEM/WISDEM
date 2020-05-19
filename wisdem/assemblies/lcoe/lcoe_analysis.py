import numpy as np

# openmdao imports
from openmdao.api import IndepVarComp, ScipyOptimizer, DumpRecorder

# FUSED wrapper
from fusedwind.fused_openmdao import FUSED_Group, FUSED_print,  FUSED_Problem, FUSED_setup, FUSED_run

from lcoeassembly import example_task37_lcoe

# global tower variables
nPoints = 3
nFull   = 5*(nPoints-1) + 1
wind = 'PowerWind'
nLC = 2
nDEL = 35

# initialize variables for tower optimization
def init_variables(prob):

    # Rotor and load inputs
    prob['rotor_diameter']=130.0  # m
    prob['blade_root_diameter'] = 2.6
    prob['rotor_speed']=11.753  # rpm m/s
    prob['machine_rating']=3350.0
    prob['drivetrain_efficiency']=0.93
    prob['rotor_torque']=1.5 * (prob['machine_rating'] * 1000) / (prob['rotor_speed'] * (np.pi / 30))
    prob['rotor_mass']=0.0  # accounted for in F_z # kg
    prob['blade_mass'] = 16441.0  # kg # used by hub
    prob['rotor_bending_moment_x']=5710973.513 # Nm  # Nm # from Excel loads tables
    prob['rotor_bending_moment_y']=5783279.297 # Nm
    prob['rotor_bending_moment_z']=975102.6807 # Nm
    prob['rotor_thrust']= 1350398.142 # N
    prob['rotor_force_y'] = 17455.11903 # N
    prob['rotor_force_z'] = 1144048.254 # N

    # Drivetrain inputs
    # machine rating see above
    # drivetrain efficiency see above
    prob['gear_ratio']=97.0  # 97:1 as listed in the 5 MW reference document
    prob['shaft_angle']=5.0*np.pi / 180.0  # rad
    prob['shaft_ratio']=0.10
    prob['planet_numbers']=[3, 3, 1]
    prob['shrink_disc_mass']=333.3 * prob['machine_rating'] / 1000.0  # estimated
    prob['carrier_mass']=8000.0  # estimated
    prob['flange_length']=0.5
    prob['overhang']=5.0
    prob['distance_hub2mb']=0.0 # letting it be internally calculated
    prob['gearbox_input_cm'] = 0.1
    prob['hss_input_length'] = 2.0

    # Tower inputs for nacelle calculations
    prob['tower_top_diameter']=3.0  # m

    # Turbine capital cost inputs (check consistency with inputs above)
    prob['crane'] = True
    prob['offshore'] = False
    prob['bearing_number'] = 2

    # Cost of energy inputs
    prob['turbine_number'] = 100
    capacity_factor = 0.415
    prob['aep'] = capacity_factor * prob['turbine_number'] * 8760 * prob['machine_rating']
    prob['bos_cost'] =  516 * prob['machine_rating'] * prob['turbine_number'] #taken from COE Review 2017 (to be published)
    prob['opex'] = 41 * prob['machine_rating'] * prob['turbine_number'] #taken from COE Review 2017 (to be published)
    prob['fcr'] = 0.079 #taken from COE Review 2017 (to be published)

    # Detailed tower inputs
    # --- geometry ----
    h_param = np.diff(np.array([0.0, 55.0, 110.0]))
    d_param = np.array([6.0, 5.93, prob['tower_top_diameter']])
    t_param = np.array([0.056971, 0.026741])
    z_foundation = 0.0
    L_reinforced = 30.0  # [m] buckling length
    theta_stress = 0.0
    yaw = 0.0
    Koutfitting = 1.07

    # --- material props ---
    E = 210e9
    G = 80.8e9
    rho = 8500.0
    sigma_y = 450.0e6

    # extra mass coming from rna calculator

    # --- wind ---
    wind_zref = 110.0
    wind_z0 = 0.0
    shearExp = 0.2
    cd_usr = None
    # ---------------

    # loads coming from rotor load inputs

    # --- safety factors ---
    # check that these are not duplicative
    gamma_f = 1.35
    gamma_m = 1.3
    gamma_n = 1.0
    gamma_b = 1.1
    # ---------------

    ''' # no fatigue in analysis or scale
    # --- fatigue ---
    z_DEL = np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
    M_DEL = 1e3*np.array([8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003, 7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003, 5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003, 3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003, 2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003, 1.2673E+003, 1.2761E+003])
    gamma_fatigue = 1.35*1.3*1.0
    life = 20.0
    m_SN = 4
    # ---------------
    '''

    # --- constraints ---
    min_d_to_t = 120.0
    max_taper = 0.2
    # ---------------

    # # V_max = 80.0  # tip speed
    # # D = 126.0
    # # .freq1p = V_max / (D/2) / (2*pi)  # convert to Hz

    if wind=='PowerWind':
        prob['wind1.shearExp'] = shearExp # = prob['wind2.shearExp'] 

    # assign values to params

    # --- geometry ----
    prob['hub_height'] = h_param.sum()
    prob['foundation_height'] = 0.0
    prob['tower_section_height'] = h_param
    prob['tower_outer_diameter'] = d_param
    prob['tower_wall_thickness'] = t_param
    prob['tower_buckling_length'] = L_reinforced
    prob['tower_outfitting_factor'] = Koutfitting
    prob['distLoads1.yaw'] = prob['distLoads2.yaw'] = yaw

    # --- material props ---
    prob['E'] = E
    prob['G'] = G
    prob['material_density'] = rho
    prob['sigma_y'] = sigma_y

    # coming from rna
    # --- extra mass ----
    #prob['rna_mass'] = m
    #prob['rna_I'] = mI
    #prob['rna_cg'] = mrho
    # -----------

    # Wave inputs ignored for land-based machine
    # --- wind & wave ---
    prob['wind1.zref'] = prob['wind2.zref'] = wind_zref
    prob['z0'] = wind_z0
    prob['cd_usr'] = cd_usr
    prob['windLoads1.rho'] = prob['windLoads2.rho'] = 1.225
    prob['windLoads1.mu'] = prob['windLoads2.mu'] = 1.7934e-5
    prob['wave1.rho'] = prob['wave2.rho'] = prob['waveLoads1.rho'] = prob['waveLoads2.rho'] = 1025.0
    prob['waveLoads1.mu'] = prob['waveLoads2.mu'] = 1.3351e-3
    prob['windLoads1.beta'] = prob['windLoads2.beta'] = prob['waveLoads1.beta'] = prob['waveLoads2.beta'] = 0.0
    #prob['waveLoads1.U0'] = prob['waveLoads1.A0'] = prob['waveLoads1.beta0'] = prob['waveLoads2.U0'] = prob['waveLoads2.A0'] = prob['waveLoads2.beta0'] = 0.0
    # ---------------

    # --- safety factors ---
    prob['gamma_f'] = gamma_f
    prob['gamma_m'] = gamma_m
    prob['gamma_n'] = gamma_n
    prob['gamma_b'] = gamma_b
    #prob['gamma_fatigue'] = gamma_fatigue # not using fatigue
    # ---------------

    # analysis setup inputs
    prob['DC'] = 80.0
    prob['shear'] = True
    prob['geom'] = False
    prob['tower_force_discretization'] = 5.0
    prob['nM'] = 2
    prob['Mmethod'] = 1
    prob['lump'] = 0
    prob['tol'] = 1e-9
    prob['shift'] = 0.0

    # coming from rotor load inputs (need to fix connections in assembly - manually assigning for now)
    # # --- loading case 1: max Thrust ---
    prob['wind1.Uref'] = 13.1 # rated wind speed
    prob['pre1.rna_F'] = np.array([prob['rotor_thrust'], prob['rotor_force_y'], prob['rotor_force_z']])
    prob['pre1.rna_M'] = np.array([prob['rotor_bending_moment_x'], prob['rotor_bending_moment_y'], prob['rotor_bending_moment_z']])
    # # ---------------

    ''' # no fatigue or scale
    # --- fatigue ---
    prob['tower_z_DEL'] = z_DEL
    prob['tower_M_DEL'] = M_DEL
    prob['life'] = life
    prob['m_SN'] = m_SN
    # ---------------
    '''

    # --- constraints ---
    prob['min_d_to_t'] = min_d_to_t
    prob['max_taper'] = max_taper
    # ---------------

    # # --- input initialization complete ---
    return prob

if __name__ == "__main__":

    # Task 37 Drivetrain variables
    # geared 3-stage Gearbox with induction generator machine
    IEC_Class='A'
    blade_number = 3
    drivetrain_design='geared'
    gear_configuration='eep'  # epicyclic-epicyclic-parallel
    mb1Type='SRB'
    mb2Type='SRB'
    shaft_factor='normal'
    uptower_transformer=True
    crane=True  # onboard crane present
    yaw_motors_number = 0 # default value - will be internally calculated

    lcoegroup = FUSED_Group()

    example_task37_lcoe(lcoegroup, mb1Type, mb2Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, uptower_transformer, yaw_motors_number, crane, blade_number)

    prob=FUSED_Problem(lcoegroup)
    FUSED_setup(prob)
    prob = init_variables(prob)

    FUSED_run(prob)

    #print('----- NREL 5 MW Turbine - 4 Point Suspension -----')
    #FUSED_print(lcoegroup)

    f = open('output.txt','w')
    f.write('Inputs\n')
    for io in lcoegroup.params:
        f.write(io + ' ' + str(lcoegroup.params[io])+'\n') 
    f.write('\n')
    f.write('Outputs\n')
    for io in lcoegroup.unknowns:
        f.write(io + ' ' + str(lcoegroup.unknowns[io])+'\n')        
    
    f.close()


    optimize = True
    if optimize:
        # --- Setup Optimizer ---
        prob.driver  = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP' #'COBYLA'
        prob.driver.options['tol'] = 1e-6
        prob.driver.options['maxiter'] = 100
        # ----------------------
        
        # --- Objective ---
        prob.driver.add_objective('tower1.mass', scaler=1e-6)
        # ----------------------
        
        # --- Design Variables ---
        prob.driver.add_desvar('tower_section_height', lower=5.0, upper=80.0)
        prob.driver.add_desvar('tower_outer_diameter', lower=3.87, upper=6.0)
        prob.driver.add_desvar('tower_wall_thickness', lower=2e-2, upper=2e-1)
        # ----------------------
        
        # Recorder
        #recorder = DumpRecorder('optimization.dat')
        #recorder.options['record_params'] = True
        #recorder.options['record_metadata'] = False
        #recorder.options['record_derivs'] = False
        #prob.driver.add_recorder(recorder)
        # ----------------------
        
        # --- Constraints ---
        prob.driver.add_constraint('height_constraint', lower=1e-2, upper=1.e-2)
        prob.driver.add_constraint('post1.stress', upper=1.0)
        prob.driver.add_constraint('post1.global_buckling', upper=1.0)
        prob.driver.add_constraint('post1.shell_buckling', upper=1.0)
        #prob.driver.add_constraint('tower.damage', upper=1.0)
        prob.driver.add_constraint('weldability', upper=0.0)
        prob.driver.add_constraint('manufacturability', lower=0.0)
        freq1p = prob['rotor_speed'] / 60.0  # 1P freq in Hz
        prob.driver.add_constraint('tower1.f1', lower= 1.1*freq1p)
        # ----------------------

        # Derivatives
        prob.root.deriv_options['type'] = 'fd'
        prob.root.deriv_options['form'] = 'central'
        prob.root.deriv_options['step_calc'] = 'relative'
        prob.root.deriv_options['step_size'] = 1e-5
        
        # --- run opt ---
        FUSED_setup(prob)
        prob = init_variables(prob)
        FUSED_run(prob)
        # ---------------
        print("Optimization results")
        print(prob.driver.get_constraints())
        print(prob.driver.get_desvars())
        print(prob.driver.get_objectives())
        print()
        print()
        print( 'zs=', prob['z_full'] )
        print( 'ds=', prob['d_full'] )
        print( 'ts=', prob['t_full'] )
        print( 'mass (kg) =', prob['tower_mass'] )
        print( 'cg (m) =', prob['tower_center_of_mass'] )
        print( 'weldability =', prob['weldability'] )
        print( 'manufacturability =', prob['manufacturability'] )
        print( '\nwind: ', prob['wind1.Uref'] )
        print( 'f1 (Hz) =', prob['tower1.f1'] )
        print( 'top_deflection1 (m) =', prob['post1.top_deflection'] )
        print( 'stress1 =', prob['post1.stress'] )
        print( 'GL buckling =', prob['post1.global_buckling'] )
        print( 'Shell buckling =', prob['post1.shell_buckling'] )
        print( 'damage =', prob['post1.damage'] )
        print( '\nwind: ', prob['wind2.Uref'] )
        print( 'f1 (Hz) =', prob['tower2.f1'] )
        print( 'top_deflection2 (m) =', prob['post2.top_deflection'] )
        print( 'stress2 =', prob['post2.stress'] )
        print( 'GL buckling =', prob['post2.global_buckling'] )
        print( 'Shell buckling =', prob['post2.shell_buckling'] )
        print( 'damage =', prob['post2.damage'] )
        print()
        print('----- NREL 5 MW Turbine - 4 Point Suspension -----')
        FUSED_print(lcoegroup)
    
        f = open('output.txt','w')
        f.write('Inputs\n')
        for io in lcoegroup.params:
            f.write(io + ' ' + str(lcoegroup.params[io])+'\n') 
        f.write('\n')
        f.write('Outputs\n')
        for io in lcoegroup.unknowns:
            f.write(io + ' ' + str(lcoegroup.unknowns[io])+'\n')        
        
        f.close()
