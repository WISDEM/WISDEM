
import numpy as np
import openmdao.api as om
from wisdem.drivetrainse.hub import Hub_System
from wisdem.drivetrainse.generator import Generator
from wisdem.drivetrainse.gearbox import Gearbox
import wisdem.drivetrainse.layout as lay
import wisdem.drivetrainse.drive_structure as ds
import wisdem.drivetrainse.drive_components as dc

#----------------------------------------------------------------------------------------------
class DriveEfficiency(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_pc', default=20)
    
    def setup(self):
        n_pc  = self.options['n_pc']
        self.add_input('gearbox_efficiency', val=0.0)
        self.add_input('generator_efficiency', val=np.zeros((n_pc,2)))
        self.add_output('drivetrain_efficiency', val=np.zeros((n_pc,2)))
            
    def compute(self, inputs, outputs):
        outputs['drivetrain_efficiency'] = inputs['gearbox_efficiency']*inputs['generator_efficiency']

#----------------------------------------------------------------------------------------------
        
class DriveMaterials(om.ExplicitComponent):
    '''
    This component sifts through the material database and sets the material property data structures needed in this module
    '''

    def initialize(self):
        self.options.declare('n_mat')
        self.options.declare('direct', default=False)
    
    def setup(self):
        n_mat    = self.options['n_mat']
        
        self.add_input('E_mat', val=np.zeros([n_mat, 3]), units='Pa')
        self.add_input('G_mat', val=np.zeros([n_mat, 3]), units='Pa')
        self.add_input('Xt_mat', val=np.zeros([n_mat, 3]), units='Pa')
        self.add_input('sigma_y_mat', val=np.zeros(n_mat), units='Pa')
        self.add_input('rho_mat', val=np.zeros(n_mat), units='kg/m**3')
        self.add_input('unit_cost_mat', val=np.zeros(n_mat), units='USD/kg')
        self.add_discrete_input('material_names', val=n_mat * [''])
        self.add_discrete_input('lss_material', 'steel')
        self.add_discrete_input('hss_material', 'steel')
        self.add_discrete_input('hub_material', 'iron')
        self.add_discrete_input('spinner_material', 'carbon')
        self.add_discrete_input('bedplate_material', 'steel')

        self.add_output('hub_E', val=0.0, units='Pa')
        self.add_output('hub_G', val=0.0, units='Pa')
        self.add_output('hub_rho', val=0.0, units='kg/m**3')
        self.add_output('hub_Xy', val=0.0, units='Pa')
        self.add_output('hub_mat_cost', val=0.0, units='USD/kg')
        self.add_output('spinner_rho', val=0.0, units='kg/m**3')
        self.add_output('spinner_Xt', val=0.0, units='Pa')
        self.add_output('spinner_mat_cost', val=0.0, units='USD/kg')
        self.add_output('lss_E', val=0.0, units='Pa')
        self.add_output('lss_G', val=0.0, units='Pa')
        self.add_output('lss_rho', val=0.0, units='kg/m**3')
        self.add_output('lss_Xy', val=0.0, units='Pa')
        self.add_output('lss_cost', val=0.0, units='USD/kg')
        self.add_output('hss_E', val=0.0, units='Pa')
        self.add_output('hss_G', val=0.0, units='Pa')
        self.add_output('hss_rho', val=0.0, units='kg/m**3')
        self.add_output('hss_Xy', val=0.0, units='Pa')
        self.add_output('hss_cost', val=0.0, units='USD/kg')
        self.add_output('bedplate_E', val=0.0, units='Pa')
        self.add_output('bedplate_G', val=0.0, units='Pa')
        self.add_output('bedplate_rho', val=0.0, units='kg/m**3')
        self.add_output('bedplate_Xy', val=0.0, units='Pa')
        self.add_output('bedplate_mat_cost', val=0.0, units='USD/kg')
    
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Convert to isotropic material
        E    = np.mean(inputs['E_mat'], axis=1)
        G    = np.mean(inputs['G_mat'], axis=1)
        Xt   = np.mean(inputs['Xt_mat'], axis=1)
        sigy = inputs['sigma_y_mat']
        rho  = inputs['rho_mat']
        cost = inputs['unit_cost_mat']
        
        hub_name  = discrete_inputs['hub_material']
        spin_name = discrete_inputs['spinner_material']
        lss_name  = discrete_inputs['lss_material']
        bed_name  = discrete_inputs['bedplate_material']
        mat_names = discrete_inputs['material_names']

        # Get the index into the material list
        spin_imat = mat_names.index( spin_name )
        hub_imat  = mat_names.index( hub_name )
        lss_imat  = mat_names.index( lss_name )
        bed_imat  = mat_names.index( bed_name )

        outputs['hub_E']        = E[hub_imat]
        outputs['hub_G']        = G[hub_imat]
        outputs['hub_rho']      = rho[hub_imat]
        outputs['hub_Xy']       = sigy[hub_imat]
        outputs['hub_mat_cost']     = cost[hub_imat]

        outputs['spinner_rho']  = rho[spin_imat]
        outputs['spinner_Xt']   = Xt[spin_imat]
        outputs['spinner_mat_cost'] = cost[spin_imat]

        outputs['lss_E']        = E[lss_imat]
        outputs['lss_G']        = G[lss_imat]
        outputs['lss_rho']      = rho[lss_imat]
        outputs['lss_Xy']       = sigy[lss_imat]
        outputs['lss_cost']     = cost[lss_imat]

        outputs['bedplate_E']    = E[bed_imat]
        outputs['bedplate_G']    = G[bed_imat]
        outputs['bedplate_rho']  = rho[bed_imat]
        outputs['bedplate_Xy']   = sigy[bed_imat]
        outputs['bedplate_mat_cost'] = cost[bed_imat]

        if not self.options['direct']:
            hss_name  = discrete_inputs['hss_material']
            hss_imat  = mat_names.index( hss_name )
            outputs['hss_E']        = E[hss_imat]
            outputs['hss_G']        = G[hss_imat]
            outputs['hss_rho']      = rho[hss_imat]
            outputs['hss_Xy']       = sigy[hss_imat]
            outputs['hss_cost']     = cost[hss_imat]
        
        
#----------------------------------------------------------------------------------------------
class DrivetrainSE(om.Group):
    ''' 
    DirectDriveSE defines an OpenMDAO group that represents a wind turbine drivetrain without a gearbox and two main bearings.
    '''
    
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('n_dlcs')
        self.options.declare('topLevelFlag', default=True)
    
    def setup(self):
        opt = self.options['modeling_options']['drivetrainse']
        n_dlcs   = self.options['n_dlcs']
        direct   = opt['direct']
        dogen    =  self.options['modeling_options']['flags']['generator']
        n_points = opt['n_height']
        n_pc     = self.options['modeling_options']['servose']['n_pc']
        
        # Independent variables that may be duplicated at higher levels of aggregation
        if self.options['topLevelFlag']:
            sivc = om.IndepVarComp()
            
            # Common direct and geared
            sivc.add_output('L_12', 0.0, units='m')
            sivc.add_output('L_h1', 0.0, units='m')
            sivc.add_output('L_generator', 0.0, units='m')
            sivc.add_output('overhang', 0.0, units='m')
            sivc.add_output('drive_height', 0.0, units='m')
            sivc.add_output('lss_diameter', np.zeros(5), units='m')
            sivc.add_output('lss_wall_thickness', np.zeros(5), units='m')
            sivc.add_output('gear_ratio', 1.0)
            sivc.add_output('gearbox_efficiency', 1.0)
            sivc.add_output('converter_mass_user', 0.0, units='kg')
            sivc.add_output('transformer_mass_user', 0.0, units='kg')
            sivc.add_output('brake_mass_user', 0.0, units='kg')
            sivc.add_output('hvac_mass_coeff', 0.08, units='kg/kW')
            sivc.add_output('generator_efficiency_user', np.zeros((n_pc,2)) )
            sivc.add_output('generator_mass_user', 0.0, units='kg')
            sivc.add_discrete_output('mb1Type', 'CARB')
            sivc.add_discrete_output('mb2Type', 'SRB')
            sivc.add_discrete_output('uptower', True)

            sivc.add_discrete_output('upwind', True)
            sivc.add_discrete_output('n_blades', 3)
            sivc.add_output('tilt', 0.0, units='deg')
            sivc.add_output('hub_E', 0.0, units='Pa')
            sivc.add_output('hub_G', 0.0, units='Pa')
            sivc.add_output('hub_Xy', 0.0, units='Pa')
            sivc.add_output('hub_rho', 0.0, units='kg/m**3')
            sivc.add_output('hub_mat_cost', 0.0, units='USD/kg')
            sivc.add_output('spinner_Xt', 0.0, units='Pa')
            sivc.add_output('spinner_rho', 0.0, units='kg/m**3')
            sivc.add_output('spinner_mat_cost', 0.0, units='USD/kg')
            sivc.add_output('lss_E', 0.0, units='Pa')
            sivc.add_output('lss_G', 0.0, units='Pa')
            sivc.add_output('lss_Xy', 0.0, units='Pa')
            sivc.add_output('lss_rho', 0.0, units='kg/m**3')
            sivc.add_output('bedplate_E', 0.0, units='Pa')
            sivc.add_output('bedplate_G', 0.0, units='Pa')
            sivc.add_output('bedplate_Xy', 0.0, units='Pa')
            sivc.add_output('bedplate_rho', 0.0, units='kg/m**3')
            sivc.add_output('bedplate_mat_cost', 0.0, units='USD/kg')

            sivc.add_output('Xy', 0.0, units='Pa')
            sivc.add_output('D_top',     0.0, units='m')
            sivc.add_output('rotor_diameter',         0.0, units='m')
            sivc.add_output('rated_torque',           0.0, units='N*m')
            sivc.add_output('hub_diameter',         0.0, units='m')
            sivc.add_output('blades_I',               np.zeros(6), units='kg*m**2')
            sivc.add_output('blade_mass',             0.0, units='kg', desc='One blade')
            sivc.add_output('blades_mass',             0.0, units='kg', desc='All blades')
            sivc.add_output('F_hub',             np.zeros(3), units='N')
            sivc.add_output('M_hub',             np.zeros(3), units='N*m')
            sivc.add_output('machine_rating',         0.0, units='kW')
            
            if direct:
                # Direct only
                sivc.add_output('nose_diameter', np.zeros(5), units='m')
                sivc.add_output('nose_wall_thickness', np.zeros(5), units='m')
                sivc.add_output('bedplate_wall_thickness', np.zeros(n_points), units='m')
                sivc.add_output('access_diameter',0.0, units='m')
            else:
                # Geared only
                sivc.add_output('L_hss', 0.0, units='m')
                sivc.add_output('hss_diameter', np.zeros(3), units='m')
                sivc.add_output('hss_wall_thickness', np.zeros(3), units='m')
                sivc.add_output('bedplate_flange_width', 0.0, units='m')
                sivc.add_output('bedplate_flange_thickness', 0.0, units='m')
                sivc.add_output('bedplate_web_thickness', 0.0, units='m')
                sivc.add_discrete_output('planet_numbers', [3, 3, 0])
                sivc.add_discrete_output('gear_configuration', val='eep')
                #sivc.add_discrete_output('shaft_factor', val='normal')

                sivc.add_output('hss_E', 0.0, units='Pa')
                sivc.add_output('hss_G', 0.0, units='Pa')
                sivc.add_output('hss_Xy', 0.0, units='Pa')
                sivc.add_output('hss_rho', 0.0, units='kg/m**3')

            self.add_subsystem('sivc', sivc, promotes=['*'])
        else:
            self.add_subsystem('mat', DriveMaterials(direct=direct, n_mat=self.options['modeling_options']['materials']['n_mat']), promotes=['*'])

            
        # Core drivetrain modules
        self.add_subsystem('hub', Hub_System(topLevelFlag=False, modeling_options=opt['hub']), promotes=['*'])
        self.add_subsystem('gear', Gearbox(direct_drive=direct), promotes=['*'])
        
        if direct:
            self.add_subsystem('layout', lay.DirectLayout(n_points=n_points), promotes=['*'])
        else:
            self.add_subsystem('layout', lay.GearedLayout(), promotes=['*'])
            
        self.add_subsystem('bear1', dc.MainBearing())
        self.add_subsystem('bear2', dc.MainBearing())
        self.add_subsystem('brake', dc.Brake(direct_drive=direct), promotes=['*'])
        self.add_subsystem('elec', dc.Electronics(), promotes=['*'])
        self.add_subsystem('yaw', dc.YawSystem(), promotes=['yaw_mass','yaw_I','yaw_cm','rotor_diameter','D_top'])
        
        if dogen:
            gentype = self.options['modeling_options']['GeneratorSE']['type']
            self.add_subsystem('generator', Generator(topLevelFlag=False, design=gentype), promotes=['generator_mass','generator_cost','generator_I','machine_rating','generator_efficiency','rated_rpm','rated_torque'])
        else:
            self.add_subsystem('gensimp', dc.GeneratorSimple(direct_drive=direct, n_pc=n_pc), promotes=['*'])
            
        self.add_subsystem('misc', dc.MiscNacelleComponents(), promotes=['*'])
        self.add_subsystem('nac', dc.NacelleSystemAdder(), promotes=['*'])
        self.add_subsystem('rna', dc.RNA_Adder(), promotes=['*'])
        self.add_subsystem('lss', ds.Hub_Rotor_LSS_Frame(n_dlcs=n_dlcs, modeling_options=opt, direct_drive=direct), promotes=['*'])
        
        if direct:
            self.add_subsystem('nose', ds.Nose_Stator_Bedplate_Frame(n_points=n_points, modeling_options=opt, n_dlcs=n_dlcs), promotes=['*'])
        else:
            self.add_subsystem('hss', ds.HSS_Frame(modeling_options=opt, n_dlcs=n_dlcs), promotes=['*'])
            self.add_subsystem('bed', ds.Bedplate_IBeam_Frame(modeling_options=opt, n_dlcs=n_dlcs), promotes=['*'])

        self.add_subsystem('eff', DriveEfficiency(n_pc=n_pc), promotes=['*'])

        # Output-to-input connections
        self.connect('bedplate_rho', ['pitch_system.rho', 'spinner.metal_rho'])
        self.connect('bedplate_Xy', ['pitch_system.Xy', 'spinner.Xy'])
        self.connect('bedplate_mat_cost', 'spinner.metal_cost')
        self.connect('hub_rho', 'hub_shell.rho')
        self.connect('hub_Xy', 'hub_shell.Xy')
        self.connect('hub_mat_cost', 'hub_shell.metal_cost')
        self.connect('spinner_rho', ['spinner.composite_rho','rho_fiberglass'])
        self.connect('spinner_Xt', 'spinner.composite_Xt')
        self.connect('spinner_mat_cost', 'spinner.composite_cost')
        
        if direct:
            self.connect('D_bearing1','bear1.D_bearing')
            self.connect('D_bearing2','bear2.D_bearing')
            
        if self.options['topLevelFlag']:
            if direct:
                self.connect('nose_diameter','bear1.D_shaft', src_indices=[0])
                self.connect('nose_diameter','bear2.D_shaft', src_indices=[-1])
            else:
                self.connect('lss_diameter','bear1.D_shaft', src_indices=[0])
                self.connect('lss_diameter','bear2.D_shaft', src_indices=[-1])
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
        self.connect('bedplate_rho','yaw.rho')
        self.connect('s_gearbox','gearbox_cm')
        self.connect('s_generator','generator_cm')
        
        if dogen:
            self.connect('generator.R_out','R_generator')
            self.connect('bedplate_E','generator.E')
            self.connect('bedplate_G','generator.G')
            
            if direct:
                if self.options['topLevelFlag']:
                    self.connect('nose_diameter','generator.D_nose', src_indices=[-1])
                    self.connect('lss_diameter','generator.D_shaft', src_indices=[0])
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
            else:
                if self.options['topLevelFlag']:
                    self.connect('hss_diameter','generator.D_shaft', src_indices=[-1])
        
    
