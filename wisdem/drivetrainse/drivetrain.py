
import numpy as np
import openmdao.api as om
from wisdem.drivetrainse.hub import Hub_System
from wisdem.drivetrainse.generator import Generator
from wisdem.drivetrainse.gearbox import Gearbox
import wisdem.drivetrainse.layout as lay
import wisdem.drivetrainse.drive_structure as ds
import wisdem.drivetrainse.drive_components as dc

        
class DrivetrainSE(om.Group):
    ''' 
    DirectDriveSE defines an OpenMDAO group that represents a wind turbine drivetrain without a gearbox and two main bearings.
    '''
    
    def initialize(self):
        self.options.declare('n_points')
        self.options.declare('n_dlcs')
        self.options.declare('model_generator')
        self.options.declare('direct_drive', default=True)
        self.options.declare('topLevelFlag', default=True)
    
    def setup(self):
        n_points = self.options['n_points']
        n_dlcs   = self.options['n_dlcs']
        direct   = self.options['direct_drive']

        # Independent variables that are unique to the drivetrain (except for the generator ones)
        ivc = om.IndepVarComp()
        # Common direct and geared
        ivc.add_output('L_12', 0.0, units='m')
        ivc.add_output('L_h1', 0.0, units='m')
        ivc.add_output('L_generator', 0.0, units='m')
        ivc.add_output('gear_ratio', 1.0)
        ivc.add_output('overhang', 0.0, units='m')
        ivc.add_output('drive_height', 0.0, units='m')
        ivc.add_output('access_diameter',0.0, units='m')
        ivc.add_output('lss_diameter', np.zeros(5), units='m')
        ivc.add_output('lss_wall_thickness', np.zeros(5), units='m')
        ivc.add_discrete_output('mb1Type', 'CARB')
        ivc.add_discrete_output('mb2Type', 'SRB')
        ivc.add_discrete_output('uptower', True)

        # Direct only
        ivc.add_output('nose_diameter', np.zeros(5), units='m')
        ivc.add_output('nose_wall_thickness', np.zeros(5), units='m')
        ivc.add_output('bedplate_wall_thickness', np.zeros(n_points), units='m')
        
        # Geared only
        ivc.add_output('L_hss', 0.0, units='m')
        ivc.add_output('hss_diameter', np.zeros(3), units='m')
        ivc.add_output('hss_wall_thickness', np.zeros(3), units='m')
        ivc.add_output('bedplate_flange_width', 0.0, units='m')
        ivc.add_output('bedplate_flange_thickness', 0.0, units='m')
        ivc.add_output('bedplate_web_thickness', 0.0, units='m')
        ivc.add_discrete_output('planet_numbers', np.array([3, 3, 0]))
        ivc.add_discrete_output('gear_configuration', val='eep')
        #ivc.add_discrete_output('shaft_factor', val='normal')
        
        self.add_subsystem('ivc', ivc, promotes=['*'])

        # Independent variables that may be duplicated at higher levels of aggregation
        if self.options['topLevelFlag']:
            sivc = om.IndepVarComp()
            sivc.add_discrete_output('upwind', True)
            sivc.add_discrete_output('n_blades', 3)
            sivc.add_output('tilt', 0.0, units='deg')
            sivc.add_output('E', 0.0, units='Pa')
            sivc.add_output('G', 0.0, units='Pa')
            sivc.add_output('v', 0.0)
            sivc.add_output('sigma_y', 0.0, units='Pa')
            sivc.add_output('Xy', 0.0, units='Pa')
            sivc.add_output('rho', 0.0, units='kg/m**3')
            sivc.add_output('D_top',     0.0, units='m')
            sivc.add_output('rotor_diameter',         0.0, units='m')
            sivc.add_output('rotor_rpm',              0.0, units='rpm')
            sivc.add_output('rotor_torque',           0.0, units='N*m')
            sivc.add_output('blades_I',               np.zeros(6), units='kg*m**2')
            sivc.add_output('blades_mass',             0.0, units='kg')
            sivc.add_output('F_hub',             np.zeros(3), units='N')
            sivc.add_output('M_hub',             np.zeros(3), units='N*m')
            sivc.add_output('gearbox_efficiency',     0.0)
            #sivc.add_output('generator_efficiency',   0.0)
            sivc.add_output('machine_rating',         0.0, units='kW')
            self.add_subsystem('sivc', sivc, promotes=['*'])
        else:
            self.add_subsystem('mat', DriveMaterials(), promotes=['*'])

        self.add_subsystem('hub', Hub_System(), promotes=['*'])
        self.add_subsystem('gear', Gearbox(direct_drive=direct), promotes=['*'])
        
        if direct:
            self.add_subsystem('layout', lay.DirectLayout(n_points=n_points), promotes=['*'])
        else:
            self.add_subsystem('layout', lay.GearedLayout(), promotes=['*'])
            
        self.add_subsystem('bear1', dc.MainBearing())
        self.add_subsystem('bear2', dc.MainBearing())
        self.add_subsystem('brake', dc.Brake(direct_drive=direct), promotes=['*'])
        self.add_subsystem('elec', dc.Electronics(), promotes=['*'])
        self.add_subsystem('yaw', dc.YawSystem(), promotes=['*'])
        if self.options['model_generator']:
            gentype = 'pmsg_outer' if direct else 'dfig'
            self.add_subsystem('generator', Generator(topLevelFlag=False, design=gentype), promotes=['generator_mass','generator_I','E','G','v','machine_rating'])
        else:
            self.add_subsystem('gensimp', dc.GeneratorSimple(direct_drive=direct), promotes=['*'])
        self.add_subsystem('misc', dc.MiscNacelleComponents(), promotes=['*'])
        self.add_subsystem('nac', dc.NacelleSystemAdder(), promotes=['*'])
        self.add_subsystem('rna', dc.RNA_Adder(), promotes=['*'])
        self.add_subsystem('lss', ds.Hub_Rotor_LSS_Frame(n_dlcs=n_dlcs, direct_drive=direct), promotes=['*'])
        if direct:
            self.add_subsystem('nose', ds.Nose_Stator_Bedplate_Frame(n_points=n_points, n_dlcs=n_dlcs), promotes=['*'])
        else:
            self.add_subsystem('hss', ds.HSS_Frame(n_dlcs=n_dlcs), promotes=['*'])
            self.add_subsystem('bed', ds.Bedplate_IBeam_Frame(n_dlcs=n_dlcs), promotes=['*'])

        self.connect('lss_diameter','bear1.D_shaft', src_indices=[0])
        self.connect('lss_diameter','bear2.D_shaft', src_indices=[-1])
        
        if direct:
            self.connect('D_bearing1','bear1.D_bearing')
            self.connect('D_bearing2','bear2.D_bearing')
            
        self.connect('mb1Type', 'bear1.bearing_type')
        self.connect('mb2Type', 'bear2.bearing_type')
        self.connect('bear1.mb_mass','mb1_mass')
        self.connect('bear1.mb_I','mb1_I')
        self.connect('bear1.mb_max_defl_ang','mb1_max_defl_ang')
        self.connect('s_mb1','mb1_cm')
        self.connect('bear2.mb_mass','mb2_mass')
        self.connect('bear2.mb_I','mb2_I')
        self.connect('bear2.mb_max_defl_ang','mb2_max_defl_ang')
        self.connect('s_mb2','mb2_cm')
        self.connect('s_gearbox','gearbox_cm')
        self.connect('s_generator','generator_cm')
        
        if self.options['model_generator']:
            self.connect('lss_diameter','generator.D_shaft', src_indices=[0])
            self.connect('generator.R_out','R_generator')
            
            if self.options['topLevelFlag']:
                self.connect('rotor_torque','generator.T_rated')
                self.connect('rotor_rpm','generator.n_nom')
                
            if direct:
                self.connect('nose_diameter','generator.D_nose', src_indices=[-1])
                self.connect('torq_deflection', 'generator.y_sh')
                self.connect('torq_rotation', 'generator.theta_sh')
                self.connect('stator_deflection', 'generator.y_bd')
                self.connect('stator_rotation', 'generator.theta_bd')
                self.connect('generator.rotor_mass','m_rotor')
                self.connect('generator.rotor_I','I_rotor')
                self.connect('generator.stator_mass','m_stator')
                self.connect('generator.stator_I','I_stator')

                self.linear_solver = lbgs = om.LinearBlockGS()
                self.nonlinear_solver = nlbgs = om.NonlinearBlockGS()
                nlbgs.options['maxiter'] = 3
        

def direct_example():
    npts = 10
    prob = om.Problem()
    prob.model = DrivetrainSE(topLevelFlag=True, n_points=npts, n_dlcs=1, model_generator=True)
    prob.setup()

    prob['rotor_diameter'] = 206.
    prob['upwind'] = True
    prob['direct_drive'] = True
    prob['n_blades'] = 3
    prob['rotor_rpm'] = 10.0
    prob['machine_rating'] = 5e3

    prob['L_12'] = 2.0
    prob['L_h1'] = 1.0
    prob['L_generator'] = 3.25
    prob['overhang'] = 6.25
    prob['drive_height'] = 4.875
    prob['tilt'] = 4.0
    prob['access_diameter'] = 0.9

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
    
    prob['pitch_system.blade_mass']       = 17000.
    prob['pitch_system.BRFM']             = 1.e+6
    prob['pitch_system.scaling_factor']   = 0.54
    prob['pitch_system.rho']              = 7850.
    prob['pitch_system.Xy']               = 371.e+6
    prob['blade_root_diameter']           = 4.

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

    return prob


def geared_example():
        
    npts = 10
    prob = om.Problem()
    prob.model = DrivetrainSE(topLevelFlag=True, n_points=npts, n_dlcs=1, model_generator=True, direct_drive=False)
    prob.setup()

    prob['upwind'] = True

    prob['L_12'] = 2.0
    prob['L_h1'] = 1.0        
    prob['overhang'] = 2.0
    prob['drive_height'] = 4.875
    prob['L_hss'] = 1.5
    prob['L_generator'] = 1.25
    prob['L_gearbox'] = 1.1
    prob['tilt'] = 5.0
        
    prob['n_blades'] = 3
    prob['rotor_rpm'] = 10.0
    prob['rotor_diameter'] = 120.0
    prob['machine_rating'] = 5e3
    prob['D_top'] = 6.5

    prob['F_hub'] = np.array([2409.750e3, 0.0, 74.3529e2]).reshape((3,1))
    prob['M_hub'] = np.array([-1.83291e4, 6171.7324e2, 5785.82946e2]).reshape((3,1))

    prob['E'] = 210e9
    prob['G'] = 80.8e9
    prob['v'] = 0.3
    prob['rho'] = 7850.
    prob['sigma_y'] = 250e6
    
    myones = np.ones(5)
    prob['lss_diameter'] = 2.3*myones
    prob['lss_wall_thickness'] = 0.05*myones
    myones = np.ones(3)
    prob['hss_diameter'] = 2.0*myones
    prob['hss_wall_thickness'] = 0.05*myones

    prob['bedplate_flange_width'] = 1.5
    prob['bedplate_flange_thickness'] = 0.05
    #prob['bedplate_web_height'] = 1.0
    prob['bedplate_web_thickness'] = 0.05

    prob['planet_numbers'] = np.array([3, 3, 0])
    prob['gear_configuration'] = 'eep'
    #prob['shaft_factor'] = 'normal'
    prob['gear_ratio'] = 90.0

    prob['pitch_system.blade_mass']       = 17000.
    prob['pitch_system.BRFM']             = 1.e+6
    prob['pitch_system.scaling_factor']   = 0.54
    prob['pitch_system.rho']              = 7850.
    prob['pitch_system.Xy']               = 371.e+6

    prob['blade_root_diameter']           = 4.
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
    
    prob['generator.Overall_eff']    = 93
    prob['generator.rho_Fe']         = 7700.0
    prob['generator.rho_Fes']        = 7850.0
    prob['generator.rho_Copper']     = 8900.0
    prob['generator.rho_PM']         = 7450.0
    prob['generator.B_r']            = 1.2
    prob['generator.E']              = 2e11
    prob['generator.G']              = 79.3e9
    prob['generator.P_Fe0e']         = 1.0
    prob['generator.P_Fe0h']         = 4.0
    prob['generator.S_N']            = -0.002
    prob['generator.alpha_p']        = 0.5*np.pi*0.7
    prob['generator.b_r_tau_r']      = 0.45
    prob['generator.b_ro']           = 0.004
    prob['generator.b_s_tau_s']      = 0.45
    prob['generator.b_so']           = 0.004
    prob['generator.cofi']           = 0.85
    prob['generator.freq']           = 60
    prob['generator.h_i']            = 0.001
    prob['generator.h_sy0']          = 0.0
    prob['generator.h_w']            = 0.005
    prob['generator.k_fes']          = 0.9
    prob['generator.k_s']            = 0.2
    prob['generator.m']     = 3
    prob['generator.mu_0']           = np.pi*4e-7
    prob['generator.mu_r']           = 1.06
    prob['generator.p']              = 3.0
    prob['generator.phi']            = np.deg2rad(90)
    prob['generator.ratio_mw2pp']    = 0.7
    prob['generator.resist_Cu']      = 1.8e-8*1.4
    prob['generator.sigma']          = 40e3
    prob['generator.v']              = 0.3
    prob['generator.y_tau_p']        = 1.0
    prob['generator.y_tau_pr']       = 10. / 12
    prob['generator.Gearbox_efficiency'] = 0.955
    prob['generator.cofi']               = 0.9
    prob['generator.y_tau_p']            = 12./15.
    prob['generator.sigma']              = 21.5e3
    prob['generator.rad_ag']             = 0.61
    prob['generator.len_s']              = 0.49
    prob['generator.h_s']                = 0.08
    prob['generator.I_0']                = 40.0
    prob['generator.B_symax']            = 1.3
    prob['generator.S_Nmax']             = -0.2
    prob['generator.h_0']                = 0.01
    prob['generator.k_fillr']        = 0.55
    prob['generator.k_fills']        = 0.65
    prob['generator.q1']    = 5
    prob['generator.q2']    = 4
    
    return prob

if __name__ == '__main__':
    #prob = direct_example()
    prob = geared_example()
    prob.run_model()

    print('Pitch system mass: ' + str(prob['pitch_mass'][0]) + ' kg')
    print('Hub shell mass: ' + str(prob['hub_mass'][0]) + ' kg')
    print('Hub shell outer diameter: ' + str(prob['hub_diameter'][0]) + ' m')
    print('Hub shell cost: ' + str(prob['hub_cost'][0]) + ' USD')
    print('Distance btw flange and cm of hub shell: ' + str(prob['hub_cm'][0]) + ' m')
    print('Mass moment of inertia of hub shell: ' + str(prob['hub_I']) + 'kg * m2')
    print('Spinner mass: ' + str(prob['spinner_mass'][0]) + ' kg')
    print('Spinner outer diameter: ' + str(prob['spinner.spinner_diameter'][0]) + ' m')
    print('Spinner cost: ' + str(prob['spinner_cost'][0]) + ' USD')
    print('Distance btw flange and cm of spinner: ' + str(prob['spinner_cm'][0]) + ' m')
    print('Mass moment of inertia of spinner: ' + str(prob['spinner_I']) + 'kg * m2')
    print('Overall hub system mass: ' + str(prob['hub_system_mass'][0]) + ' kg')
    print('Overall hub system cost: ' + str(prob['hub_system_cost'][0]) + ' USD')
    print('Distance btw shaft flange and cm of overall hub system: ' + str(prob['hub_system_cm'][0]) + ' m')
    print('Mass moment of inertia of the overall hub system: ' + str(prob['hub_system_I']) + 'kg * m2')
    
