import openmdao.api as om
from wisdem.floatingse.column import Column
from wisdem.floatingse.substructure import Substructure, SubstructureGeometry
from wisdem.floatingse.loading import Loading
from wisdem.floatingse.map_mooring import MapMooring
from wisdem.towerse.tower import TowerLeanSE
import numpy as np


class FloatingSE(om.Group):

    def initialize(self):
        self.options.declare('modeling_options')

    def setup(self):
        opt = self.options['modeling_options']['platform']
        n_mat = self.options['modeling_options']['materials']['n_mat']
        n_height_main = opt['columns']['main']['n_height']
        n_height_off  = opt['columns']['offset']['n_height']
        n_height_tow  = self.options['modeling_options']['tower']['n_height']

        self.set_input_defaults('mooring_type', 'chain')
        self.set_input_defaults('anchor_type', 'SUCTIONPILE')
        self.set_input_defaults('loading', 'hydrostatic')
        self.set_input_defaults('wave_period_range_low', 2.0, units='s')
        self.set_input_defaults('wave_period_range_high', 20.0, units='s')
        self.set_input_defaults('cd_usr', -1.0)
        self.set_input_defaults('zref', 100.0)
        self.set_input_defaults('number_of_offset_columns', 0)
        self.set_input_defaults('material_names', ['steel'])
        
        self.add_subsystem('tow', TowerLeanSE(modeling_options=self.options['modeling_options']),
                           promotes=['tower_s','tower_height','tower_outer_diameter_in','tower_layer_thickness','tower_outfitting_factor',
                                     'rna_mass','rna_cg','rna_I',
                                     'tower_mass','tower_I_base','hub_height','material_names',
                                     'labor_cost_rate','painting_cost_rate','unit_cost_mat','rho_mat','E_mat','G_mat','sigma_y_mat',
                                     ('transition_piece_height', 'main_freeboard'), ('foundation_height', 'main_freeboard')])
        
        # Next do main and ballast columns
        # Ballast columns are replicated from same design in the components
        column_promotes = ['E_mat','G_mat','sigma_y_mat','rho_air','mu_air','rho_water','mu_water','rho_mat',
                           'shearExp','yaw','Uc','water_depth',
                           'hsig_wave','Tsig_wave','cd_usr','cm','loading','beta_wind','beta_wave',
                           'max_draft','material_names',
                           'permanent_ballast_density','outfitting_factor','ballast_cost_rate',
                           'unit_cost_mat','labor_cost_rate','painting_cost_rate','outfitting_cost_rate',
                           'Uref', 'zref', 'wind_z0']
        main_column_promotes = column_promotes.copy()
        main_column_promotes.append(('freeboard', 'main_freeboard'))
        
        self.add_subsystem('main', Column(modeling_options=opt, column_options=opt['columns']['main'], n_mat=n_mat),
                           promotes=main_column_promotes)
                           
        off_column_promotes = column_promotes.copy()
        off_column_promotes.append(('freeboard', 'off_freeboard'))

        self.add_subsystem('off', Column(modeling_options=opt, column_options=opt['columns']['offset'], n_mat=n_mat),
                           promotes=off_column_promotes)

        # Run Semi Geometry for interfaces
        self.add_subsystem('sg', SubstructureGeometry(n_height_main=n_height_main,
                                                      n_height_off=n_height_off), promotes=['*'])

        # Next run MapMooring
        self.add_subsystem('mm', MapMooring(modeling_options=opt), promotes=['*'])
        
        # Add in the connecting truss
        self.add_subsystem('load', Loading(n_height_main=n_height_main,
                                           n_height_off=n_height_off,
                                           n_height_tow=n_height_tow,
                                           modeling_options=opt), promotes=['*'])

        # Run main Semi analysis
        self.add_subsystem('subs', Substructure(n_height_main=n_height_main,
                                                n_height_off=n_height_off,
                                                n_height_tow=n_height_tow), promotes=['*'])
        
        # Connect all input variables from all models
        self.connect('tow.d_full', ['windLoads.d','tower_d_full'])
        self.connect('tow.d_full', 'tower_d_base', src_indices=[0])
        self.connect('tow.t_full', 'tower_t_full')
        self.connect('tow.z_full', ['loadingWind.z','windLoads.z','tower_z_full']) # includes tower_z_full
        self.connect('tow.E_full', 'tower_E_full')
        self.connect('tow.G_full', 'tower_G_full')
        self.connect('tow.rho_full', 'tower_rho_full')
        self.connect('tow.sigma_y_full', 'tower_sigma_y_full')
        self.connect('tow.cm.mass','tower_mass_section')
        self.connect('tow.turbine_mass','main.stack_mass_in')
        self.connect('tow.tower_center_of_mass','tower_center_of_mass')
        self.connect('tow.tower_cost','tower_shell_cost')
        
        self.connect('main.z_full', ['main_z_nodes', 'main_z_full'])
        self.connect('main.d_full', 'main_d_full')
        self.connect('main.t_full', 'main_t_full')
        self.connect('main.E_full', 'main_E_full')
        self.connect('main.G_full', 'main_G_full')
        self.connect('main.rho_full', 'main_rho_full')
        self.connect('main.sigma_y_full', 'main_sigma_y_full')

        self.connect('off.z_full', ['offset_z_nodes', 'offset_z_full'])
        self.connect('off.d_full', 'offset_d_full')
        self.connect('off.t_full', 'offset_t_full')
        self.connect('off.E_full', 'offset_E_full')
        self.connect('off.G_full', 'offset_G_full')
        self.connect('off.rho_full', 'offset_rho_full')
        self.connect('off.sigma_y_full', 'offset_sigma_y_full')

        self.connect('max_offset_restoring_force', 'mooring_surge_restoring_force')
        self.connect('operational_heel_restoring_force', 'mooring_pitch_restoring_force')
        
        self.connect('main.z_center_of_mass', 'main_center_of_mass')
        self.connect('main.z_center_of_buoyancy', 'main_center_of_buoyancy')
        self.connect('main.I_column', 'main_moments_of_inertia')
        self.connect('main.Iwater', 'main_Iwaterplane')
        self.connect('main.Awater', 'main_Awaterplane')
        self.connect('main.displaced_volume', 'main_displaced_volume')
        self.connect('main.hydrostatic_force', 'main_hydrostatic_force')
        self.connect('main.column_added_mass', 'main_added_mass')
        self.connect('main.column_total_mass', 'main_mass')
        self.connect('main.column_total_cost', 'main_cost')
        self.connect('main.variable_ballast_interp_zpts', 'water_ballast_zpts_vector')
        self.connect('main.variable_ballast_interp_radius', 'water_ballast_radius_vector')
        self.connect('main.Px', 'main_Px')
        self.connect('main.Py', 'main_Py')
        self.connect('main.Pz', 'main_Pz')
        self.connect('main.qdyn', 'main_qdyn')

        self.connect('off.z_center_of_mass', 'offset_center_of_mass')
        self.connect('off.z_center_of_buoyancy', 'offset_center_of_buoyancy')
        self.connect('off.I_column', 'offset_moments_of_inertia')
        self.connect('off.Iwater', 'offset_Iwaterplane')
        self.connect('off.Awater', 'offset_Awaterplane')
        self.connect('off.displaced_volume', 'offset_displaced_volume')
        self.connect('off.hydrostatic_force', 'offset_hydrostatic_force')
        self.connect('off.column_added_mass', 'offset_added_mass')
        self.connect('off.column_total_mass', 'offset_mass')
        self.connect('off.column_total_cost', 'offset_cost')
        self.connect('off.Px', 'offset_Px')
        self.connect('off.Py', 'offset_Py')
        self.connect('off.Pz', 'offset_Pz')
        self.connect('off.qdyn', 'offset_qdyn')
        self.connect('off.draft', 'offset_draft')

