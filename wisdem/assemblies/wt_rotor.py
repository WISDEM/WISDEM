import numpy as np
from openmdao.api import ExplicitComponent, Group, IndepVarComp, Problem
from wisdem.assemblies.load_IEA_yaml import WT_Data, Wind_Turbine, yaml2openmdao
from wisdem.rotorse.rotor_geometry import TurbineClass
from wisdem.rotorse.rotor_aeropower import RegulatedPowerCurve, Cp_Ct_Cq_Tables, NoStallConstraint, AEP
from wisdem.commonse.environment import PowerWind
from wisdem.commonse.distribution import WeibullWithMeanCDF

class ParametrizeBlade(ExplicitComponent):
    # Openmdao component to parameterize distributed quantities for the aerodynamic only analysis of the wind turbine rotor
    def initialize(self):
        self.options.declare('wt_init_options')

class RotorAeroPower(Group):
    # Openmdao group for the aerodynamic only analysis of the wind turbine rotor
    def initialize(self):
        self.options.declare('wt_init_options')
    def setup(self):
        wt_init_options = self.options['wt_init_options']

        self.add_subsystem('powercurve',        RegulatedPowerCurve(wt_init_options   = wt_init_options), promotes = ['control_Vin', 'control_Vout','control_ratedPower','control_minOmega','control_maxOmega','control_maxTS','control_tsr','control_pitch','drivetrainType','drivetrainEff','r','chord', 'theta','Rhub', 'Rtip', 'hub_height','precone', 'tilt','yaw','precurve','precurveTip','presweep','presweepTip', 'airfoils_aoa','airfoils_Re','airfoils_cl','airfoils_cd','airfoils_cm', 'nBlades', 'rho', 'mu'])
        self.add_subsystem('aeroperf_tables',   Cp_Ct_Cq_Tables(wt_init_options   = wt_init_options), promotes = ['control_Vin', 'control_Vout','r','chord', 'theta','Rhub', 'Rtip', 'hub_height','precone', 'tilt','yaw','precurve','precurveTip','presweep','presweepTip', 'airfoils_aoa','airfoils_Re','airfoils_cl','airfoils_cd','airfoils_cm', 'nBlades', 'rho', 'mu'])
        self.add_subsystem('stall_check',       NoStallConstraint(wt_init_options   = wt_init_options), promotes = ['airfoils_aoa','airfoils_cl','airfoils_cd','airfoils_cm'])
        # self.add_subsystem('wind',              PowerWind(nPoints=1))
        self.add_subsystem('cdf',               WeibullWithMeanCDF(nspline=200))
        self.add_subsystem('aep',               AEP(), promotes=['AEP'])


        self.connect('powercurve.aoa_cutin','stall_check.aoa_along_span')
        # self.connect('hub_height', 'wind.zref')

        # connections to cdf
        self.connect('powercurve.V_spline', 'cdf.x')
        
        # self.connect('shape_parameter',     'cdf.k')

        # connections to aep
        self.connect('cdf.F',               'aep.CDF_V')
        self.connect('powercurve.P_spline', 'aep.P')

class WT_Rotor(Group):
    # Openmdao group to run the aerostructural analysis of the wind turbine rotor
    
    def initialize(self):
        self.options.declare('wt_init_options')
        
    def setup(self):
        wt_init_options = self.options['wt_init_options']
        
        self.add_subsystem('wt',        Wind_Turbine(wt_init_options     = wt_init_options), promotes = ['*'])
        self.add_subsystem('wt_class',  TurbineClass())
        self.add_subsystem('param',     ParametrizeBlade())
        self.add_subsystem('ra',        RotorAeroPower(wt_init_options   = wt_init_options))

        self.connect('configuration.ws_class' , 'wt_class.turbine_class')
        self.connect('wt_class.V_mean',         'ra.cdf.xbar')
        self.connect('control.V_in' ,           'ra.control_Vin')
        self.connect('control.V_out' ,          'ra.control_Vout')
        self.connect('control.rated_power' ,    'ra.control_ratedPower')
        self.connect('control.min_Omega' ,      'ra.control_minOmega')
        self.connect('control.max_Omega' ,      'ra.control_maxOmega')
        self.connect('control.max_TS' ,         'ra.control_maxTS')
        self.connect('control.rated_TSR' ,      'ra.control_tsr')
        self.connect('control.rated_pitch' ,        'ra.control_pitch')
        self.connect('configuration.gearbox_type' , 'ra.drivetrainType')
        self.connect('assembly.r_blade',            'ra.r')
        self.connect('assembly.rotor_radius',       'ra.Rtip')
        self.connect('blade.outer_shape_bem.chord', 'ra.chord')
        self.connect('blade.outer_shape_bem.twist', 'ra.theta')
        self.connect('hub.radius',                  'ra.Rhub')
        self.connect('assembly.hub_height',         'ra.hub_height')
        self.connect('hub.cone',                    'ra.precone')
        self.connect('nacelle.uptilt',              'ra.tilt')
        self.connect('airfoils.aoa',                    'ra.airfoils_aoa')
        self.connect('airfoils.Re',                     'ra.airfoils_Re')
        self.connect('blade.interp_airfoils.cl_interp', 'ra.airfoils_cl')
        self.connect('blade.interp_airfoils.cd_interp', 'ra.airfoils_cd')
        self.connect('blade.interp_airfoils.cm_interp', 'ra.airfoils_cm')
        self.connect('configuration.n_blades',          'ra.nBlades')
        self.connect('blade.outer_shape_bem.s',         'ra.stall_check.s')
        self.connect('env.rho_air',                     'ra.rho')
        self.connect('env.mu_air',                      'ra.mu')
        self.connect('env.weibull_k',                   'ra.cdf.k')

if __name__ == "__main__":

    ## File management
    fname_input        = "reference_turbines/nrel5mw/nrel5mw_mod_update.yaml"
    # fname_input        = "/mnt/c/Material/Projects/Hitachi_Design/Design/turbine_inputs/aerospan_formatted_v13.yaml"
    fname_output       = "reference_turbines/nrel5mw/nrel5mw_mod_update_output.yaml"
    
    # Load yaml data into a pure python data structure
    wt_initial               = WT_Data()
    wt_initial.validate      = False
    wt_initial.fname_schema  = "reference_turbines/IEAontology_schema.yaml"
    wt_init_options, wt_init = wt_initial.initialize(fname_input)
    
    # Initialize openmdao problem
    wt_opt          = Problem()
    wt_opt.model    = WT_Rotor(wt_init_options = wt_init_options)
    wt_opt.setup()
    # Load wind turbine data from wt_initial to the openmdao problem
    wt_opt = yaml2openmdao(wt_opt, wt_init_options, wt_init)
    wt_opt.run_driver()
    
    # Save data coming from openmdao to an output yaml file
    wt_initial.write_ontology(wt_opt, fname_output)

    print('AEP = ' + str(wt_opt['ra.AEP']*1.e-6) + ' GWh')
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(wt_opt['ra.powercurve.V'], wt_opt['ra.powercurve.P']/1e6)
    plt.xlabel('wind speed (m/s)')
    plt.xlabel('power (W)')
    plt.show()

    n_pitch = len(wt_opt['ra.aeroperf_tables.pitch_vector'])
    n_tsr   = len(wt_opt['ra.aeroperf_tables.tsr_vector'])
    n_U     = len(wt_opt['ra.aeroperf_tables.U_vector'])
    for i in range(n_U):
        fig0, ax0 = plt.subplots()
        CS0 = ax0.contour(wt_opt['ra.aeroperf_tables.pitch_vector'], wt_opt['ra.aeroperf_tables.tsr_vector'], wt_opt['ra.aeroperf_tables.Cp'][:, :, i], levels=[0.0, 0.3, 0.40, 0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50 ])
        ax0.clabel(CS0, inline=1, fontsize=12)
        plt.title('Power Coefficient', fontsize=14, fontweight='bold')
        plt.xlabel('Pitch Angle [deg]', fontsize=14, fontweight='bold')
        plt.ylabel('TSR [-]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)

        fig0, ax0 = plt.subplots()
        CS0 = ax0.contour(wt_opt['ra.aeroperf_tables.pitch_vector'], wt_opt['ra.aeroperf_tables.tsr_vector'], wt_opt['ra.aeroperf_tables.Ct'][:, :, i])
        ax0.clabel(CS0, inline=1, fontsize=12)
        plt.title('Thrust Coefficient', fontsize=14, fontweight='bold')
        plt.xlabel('Pitch Angle [deg]', fontsize=14, fontweight='bold')
        plt.ylabel('TSR [-]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)

        
        fig0, ax0 = plt.subplots()
        CS0 = ax0.contour(wt_opt['ra.aeroperf_tables.pitch_vector'], wt_opt['ra.aeroperf_tables.tsr_vector'], wt_opt['ra.aeroperf_tables.Cq'][:, :, i])
        ax0.clabel(CS0, inline=1, fontsize=12)
        plt.title('Torque Coefficient', fontsize=14, fontweight='bold')
        plt.xlabel('Pitch Angle [deg]', fontsize=14, fontweight='bold')
        plt.ylabel('TSR [-]', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        plt.subplots_adjust(bottom = 0.15, left = 0.15)
        
        plt.show()

    # Angle of attack and stall angle
    faoa, axaoa = plt.subplots(1,1,figsize=(5.3, 4))
    axaoa.plot(wt_opt['ra.stall_check.s'], wt_opt['ra.stall_check.aoa_along_span'], label='AoA')
    axaoa.plot(wt_opt['ra.stall_check.s'], wt_opt['ra.stall_check.stall_angle_along_span'], '.', label='Stall')
    axaoa.legend(fontsize=12)
    plt.xlabel('Blade Span [m]', fontsize=14, fontweight='bold')
    plt.ylabel('Angle [deg]', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(color=[0.8,0.8,0.8], linestyle='--')
    plt.subplots_adjust(bottom = 0.15, left = 0.15)
    # fig_name = 'aoa.png'
    # faoa.savefig(folder_output + fig_name)
    plt.show()