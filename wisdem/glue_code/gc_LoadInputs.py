import numpy as np
import wisdem.yaml as sch


class WindTurbineOntologyPython(object):
    # Pure python class to load the input yaml file and break into few sub-dictionaries, namely:
    #   - modeling_options: dictionary with all the inputs that will be passed as options to the openmdao components, such as the length of the arrays
    #   - blade: dictionary representing the entry blade in the yaml file
    #   - tower: dictionary representing the entry tower in the yaml file
    #   - nacelle: dictionary representing the entry nacelle in the yaml file
    #   - materials: dictionary representing the entry materials in the yaml file
    #   - airfoils: dictionary representing the entry airfoils in the yaml file


    def __init__(self, fname_input_wt, fname_input_modeling, fname_input_analysis):

        self.modeling_options = sch.load_modeling_yaml(fname_input_modeling)
        self.wt_init          = sch.load_geometry_yaml(fname_input_wt)
        self.analysis_options = sch.load_analysis_yaml(fname_input_analysis)
        self.defaults         = sch.load_default_geometry_yaml()
        self.set_run_flags()
        self.set_openmdao_vectors()
        self.set_opt_flags()


    def get_input_data(self):
        return self.wt_init, self.modeling_options, self.analysis_options


    def set_run_flags(self):
        # Create components flag struct
        self.modeling_options['flags'] = {}

        for k in self.defaults['components']:
            self.modeling_options['flags'][k] = k in self.wt_init['components']

        for k in self.defaults.keys():
            self.modeling_options['flags'][k] = k in self.wt_init

        # Generator flag
        self.modeling_options['flags']['generator'] = False
        if self.modeling_options['flags']['nacelle'] and 'generator' in self.wt_init['components']['nacelle']:
            self.modeling_options['flags']['generator'] = True
            if not 'GeneratorSE' in self.modeling_options: self.modeling_options['GeneratorSE'] = {}
            self.modeling_options['GeneratorSE']['type'] = self.wt_init['components']['nacelle']['generator']['generator_type'].lower()

        # Offshore flag
        self.modeling_options['offshore'] = 'water_depth' in self.wt_init['environment'] and self.wt_init['environment']['water_depth'] > 0.0

        # Put in some logic about what needs to be in there
        flags = self.modeling_options['flags']

        # Even if the block is in the inputs, the user can turn off via modeling options
        if flags['bos']:   flags['bos']   = self.modeling_options['Analysis_Flags']['BOS']
        if flags['blade']: flags['blade'] = self.modeling_options['Analysis_Flags']['RotorSE'] or self.modeling_options['Analysis_Flags']['ServoSE']
        if flags['tower']: flags['tower'] = self.modeling_options['Analysis_Flags']['TowerSE']
        if flags['hub']:   flags['hub']   = self.modeling_options['Analysis_Flags']['DriveSE']
        if flags['nacelle']: flags['nacelle'] = self.modeling_options['Analysis_Flags']['DriveSE']
        if flags['generator']: flags['generator'] = self.modeling_options['Analysis_Flags']['DriveSE']
        flags['hub'] = flags['nacelle'] = (flags['hub'] or flags['nacelle']) # Hub and nacelle have to go together
        
        # Blades and airfoils
        if flags['blade'] and not flags['airfoils']:
            raise ValueError('Blades/rotor analysis is requested but no airfoils are found')
        if flags['airfoils'] and not flags['blade']:
            print('WARNING: Airfoils provided but no blades/rotor found or RotorSE deactivated')

        # Blades, tower, monopile and environment
        if flags['blade'] and not flags['environment']:
            raise ValueError('Blades/rotor analysis is requested but no environment input found')
        if flags['tower'] and not flags['environment']:
            raise ValueError('Tower analysis is requested but no environment input found')
        if flags['monopile'] and not flags['environment']:
            raise ValueError('Monopile analysis is requested but no environment input found')
        if flags['floating'] and not flags['environment']:
            raise ValueError('Floating analysis is requested but no environment input found')
        if flags['environment'] and not (flags['blade'] or flags['tower'] or flags['monopile'] or flags['floating']):
            print('WARNING: Environment provided but no related component found found')

        # Tower, monopile and foundation
        if flags['tower'] and not flags['foundation']:
            raise ValueError('Tower analysis is requested but no foundation is found')
        if flags['monopile'] and not flags['foundation']:
            raise ValueError('Monopile analysis is requested but no foundation is found')
        if flags['foundation'] and not (flags['tower'] or flags['monopile']):
            print('WARNING: Foundation provided but no tower/monipile found or TowerSE deactivated')

        # Foundation and floating/monopile
        if flags['floating'] and flags['foundation']:
            raise ValueError('Cannot have both floating and foundation components')
        if flags['floating'] and flags['monopile']:
            raise ValueError('Cannot have both floating and monopile components')

        # Offshore flag
        if not self.modeling_options['offshore'] and (flags['monopile'] or flags['floating']):
            raise ValueError('Water depth must be > 0 to do monopile or floating analysis')


    def set_openmdao_vectors(self):
        # Class instance to determine all the parameters used to initialize the openmdao arrays, i.e. number of airfoils, number of angles of attack, number of blade spanwise stations, etc
        # ==modeling_options = {}

        # Materials
        self.modeling_options['materials']          = {}
        self.modeling_options['materials']['n_mat'] = len(self.wt_init['materials'])

        # Airfoils
        self.modeling_options['airfoils']           = {}
        if self.modeling_options['flags']['airfoils']:
            self.modeling_options['airfoils']['n_af']   = len(self.wt_init['airfoils'])
            self.modeling_options['airfoils']['n_aoa']  = self.modeling_options['rotorse']['n_aoa']
            if self.modeling_options['airfoils']['n_aoa'] / 4. == int(self.modeling_options['airfoils']['n_aoa'] / 4.):
                # One fourth of the angles of attack from -pi to -pi/6, half between -pi/6 to pi/6, and one fourth from pi/6 to pi
                self.modeling_options['airfoils']['aoa']    = np.unique(np.hstack([np.linspace(-np.pi, -np.pi / 6., int(self.modeling_options['airfoils']['n_aoa'] / 4. + 1)), np.linspace(-np.pi / 6., np.pi / 6., int(self.modeling_options['airfoils']['n_aoa'] / 2.)), np.linspace(np.pi / 6., np.pi, int(self.modeling_options['airfoils']['n_aoa'] / 4. + 1))]))
            else:
                self.modeling_options['airfoils']['aoa']    = np.linspace(-np.pi, np.pi, self.modeling_options['airfoils']['n_aoa'])
                print('WARNING: If you like a grid of angles of attack more refined between +- 30 deg, please choose a n_aoa in the analysis option input file that is a multiple of 4. The current value of ' + str(self.modeling_options['airfoils']['n_aoa']) + ' is not a multiple of 4 and an equally spaced grid is adopted.')
            Re_all = []
            for i in range(self.modeling_options['airfoils']['n_af']):
                for j in range(len(self.wt_init['airfoils'][i]['polars'])):
                    Re_all.append(self.wt_init['airfoils'][i]['polars'][j]['re'])
            self.modeling_options['airfoils']['n_Re']   = len(np.unique(Re_all))
            self.modeling_options['airfoils']['n_tab']  = 1
            self.modeling_options['airfoils']['n_xy']   = self.modeling_options['rotorse']['n_xy']
            self.modeling_options['airfoils']['af_used']      = self.wt_init['components']['blade']['outer_shape_bem']['airfoil_position']['labels']

        # Blade
        self.modeling_options['blade']              = {}
        if self.modeling_options['flags']['blade']:
            self.modeling_options['blade']['n_span']    = self.modeling_options['rotorse']['n_span']
            self.modeling_options['blade']['nd_span']   = np.linspace(0., 1., self.modeling_options['blade']['n_span']) # Equally spaced non-dimensional spanwise grid
            self.modeling_options['blade']['n_af_span'] = len(self.wt_init['components']['blade']['outer_shape_bem']['airfoil_position']['labels']) # This is the number of airfoils defined along blade span and it is often different than n_af, which is the number of airfoils defined in the airfoil database
            self.modeling_options['blade']['n_webs']    = len(self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'])
            self.modeling_options['blade']['n_layers']  = len(self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'])
            self.modeling_options['blade']['lofted_output'] = False
            self.modeling_options['blade']['n_freq']    = 10 # Number of blade nat frequencies computed

            self.modeling_options['blade']['layer_name'] = self.modeling_options['blade']['n_layers'] * ['']
            self.modeling_options['blade']['layer_mat']  = self.modeling_options['blade']['n_layers'] * ['']
            for i in range(self.modeling_options['blade']['n_layers']):
                self.modeling_options['blade']['layer_name'][i]  = self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['name']
                self.modeling_options['blade']['layer_mat'][i]   = self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['material']


            self.modeling_options['blade']['web_name']  = self.modeling_options['blade']['n_webs'] * ['']
            for i in range(self.modeling_options['blade']['n_webs']):
                self.modeling_options['blade']['web_name'][i]  = self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'][i]['name']

            # Distributed aerodynamic control devices along blade
            self.modeling_options['blade']['n_te_flaps']      = 0
            if 'aerodynamic_control' in self.wt_init['components']['blade']:
                if 'te_flaps' in self.wt_init['components']['blade']['aerodynamic_control']:
                    self.modeling_options['blade']['n_te_flaps'] = len(self.wt_init['components']['blade']['aerodynamic_control']['te_flaps'])
                    self.modeling_options['airfoils']['n_tab']   = 3
                else:
                    raise RuntimeError('A distributed aerodynamic control device is provided in the yaml input file, but not supported by wisdem.')

        # Drivetrain
        if self.modeling_options['flags']['nacelle']:
            self.modeling_options['drivetrainse']['direct'] = self.wt_init['assembly']['drivetrain'].lower() in ['direct','direct_drive','pm_direct_drive']

        # Tower
        if self.modeling_options['flags']['tower']:
            self.modeling_options['tower']['n_height']  = len(self.wt_init['components']['tower']['outer_shape_bem']['outer_diameter']['grid'])
            self.modeling_options['tower']['n_layers']  = len(self.wt_init['components']['tower']['internal_structure_2d_fem']['layers'])

        # Monopile
        self.modeling_options['monopile'] = {}
        if self.modeling_options['flags']['monopile']:
            self.modeling_options['monopile']['n_height']  = len(self.wt_init['components']['monopile']['outer_shape_bem']['outer_diameter']['grid'])
            self.modeling_options['monopile']['n_layers']  = len(self.wt_init['components']['monopile']['internal_structure_2d_fem']['layers'])

        # FloatingSE
        self.modeling_options['floating'] = {}

        # Assembly
        self.modeling_options['assembly'] = {}
        self.modeling_options['assembly']['number_of_blades'] = int(self.wt_init['assembly']['number_of_blades'])


    def set_opt_flags(self):
        # Recursively look for flags to set global optimization flag
        def recursive_flag(d):
            opt_flag = False
            for k,v in d.items():
                if isinstance(v, dict):
                    opt_flag = opt_flag or recursive_flag(v)
                elif k == 'flag':
                    opt_flag = opt_flag or v
            return opt_flag

        # The user can provide `opt_flag` in analysis_options.yaml,
        # but if it's not provided, we check the individual opt flags
        # from analysis_options.yaml and set a global `opt_flag`
        if 'opt_flag' in self.analysis_options['driver']:
            self.analysis_options['opt_flag'] = self.analysis_options['driver']['opt_flag']
        else:
            self.analysis_options['opt_flag'] = recursive_flag( self.analysis_options['optimization_variables'] )

        # If not an optimization DV, then the number of points should be same as the discretization
        blade_opt_options = self.analysis_options['optimization_variables']['blade']
        if not blade_opt_options['aero_shape']['twist']['flag']:
            blade_opt_options['aero_shape']['twist']['n_opt'] = self.modeling_options['rotorse']['n_span']
        elif blade_opt_options['aero_shape']['twist']['n_opt'] < 4:
                raise ValueError('Cannot optimize twist with less than 4 control points along blade span')

        if not blade_opt_options['aero_shape']['chord']['flag']:
            blade_opt_options['aero_shape']['chord']['n_opt'] = self.modeling_options['rotorse']['n_span']
        elif blade_opt_options['aero_shape']['chord']['n_opt'] < 4:
                raise ValueError('Cannot optimize chord with less than 4 control points along blade span')

        if not blade_opt_options['structure']['spar_cap_ss']['flag']:
            blade_opt_options['structure']['spar_cap_ss']['n_opt'] = self.modeling_options['rotorse']['n_span']
        elif blade_opt_options['structure']['spar_cap_ss']['n_opt'] < 4:
                raise ValueError('Cannot optimize spar cap suction side with less than 4 control points along blade span')

        if not blade_opt_options['structure']['spar_cap_ps']['flag']:
            blade_opt_options['structure']['spar_cap_ps']['n_opt'] = self.modeling_options['rotorse']['n_span']
        elif blade_opt_options['structure']['spar_cap_ps']['n_opt'] < 4:
                raise ValueError('Cannot optimize spar cap pressure side with less than 4 control points along blade span')



    def write_ontology(self, wt_opt, fname_output):

        if self.modeling_options['flags']['blade']:
            # Update blade outer shape
            self.wt_init['components']['blade']['outer_shape_bem']['airfoil_position']['grid']      = wt_opt['blade.opt_var.af_position'].tolist()
            self.wt_init['components']['blade']['outer_shape_bem']['chord']['grid']                 = wt_opt['blade.outer_shape_bem.s'].tolist()
            self.wt_init['components']['blade']['outer_shape_bem']['chord']['values']               = wt_opt['blade.pa.chord_param'].tolist()
            self.wt_init['components']['blade']['outer_shape_bem']['twist']['grid']                 = wt_opt['blade.outer_shape_bem.s'].tolist()
            self.wt_init['components']['blade']['outer_shape_bem']['twist']['values']               = wt_opt['blade.pa.twist_param'].tolist()
            self.wt_init['components']['blade']['outer_shape_bem']['pitch_axis']['grid']            = wt_opt['blade.outer_shape_bem.s'].tolist()
            self.wt_init['components']['blade']['outer_shape_bem']['pitch_axis']['values']          = wt_opt['blade.outer_shape_bem.pitch_axis'].tolist()
            self.wt_init['components']['blade']['outer_shape_bem']['reference_axis']['x']['grid']   = wt_opt['blade.outer_shape_bem.s'].tolist()
            self.wt_init['components']['blade']['outer_shape_bem']['reference_axis']['y']['grid']   = wt_opt['blade.outer_shape_bem.s'].tolist()
            self.wt_init['components']['blade']['outer_shape_bem']['reference_axis']['z']['grid']   = wt_opt['blade.outer_shape_bem.s'].tolist()
            self.wt_init['components']['blade']['outer_shape_bem']['reference_axis']['x']['values'] = wt_opt['blade.outer_shape_bem.ref_axis'][:,0].tolist()
            self.wt_init['components']['blade']['outer_shape_bem']['reference_axis']['y']['values'] = wt_opt['blade.outer_shape_bem.ref_axis'][:,1].tolist()
            self.wt_init['components']['blade']['outer_shape_bem']['reference_axis']['z']['values'] = wt_opt['blade.outer_shape_bem.ref_axis'][:,2].tolist()

            # Update blade structure
            # Reference axis from blade outer shape
            self.wt_init['components']['blade']['internal_structure_2d_fem']['reference_axis'] = self.wt_init['components']['blade']['outer_shape_bem']['reference_axis']
            # Webs positions
            for i in range(self.modeling_options['blade']['n_webs']):
                if 'rotation' in self.wt_init['components']['blade']['internal_structure_2d_fem']['webs']:
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'][i]['rotation']['grid']   = wt_opt['blade.internal_structure_2d_fem.s'].tolist()
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'][i]['rotation']['values'] = wt_opt['blade.internal_structure_2d_fem.web_rotation'][i,:].tolist()
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'][i]['offset_y_pa']['grid']   = wt_opt['blade.internal_structure_2d_fem.s'].tolist()
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'][i]['offset_y_pa']['values'] = wt_opt['blade.internal_structure_2d_fem.web_offset_y_pa'][i,:].tolist()
                if 'start_nd_arc' not in self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'][i]:
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'][i]['start_nd_arc'] = {}
                if 'end_nd_arc' not in self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'][i]:
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'][i]['end_nd_arc'] = {}
                self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'][i]['start_nd_arc']['grid']   = wt_opt['blade.internal_structure_2d_fem.s'].tolist()
                self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'][i]['start_nd_arc']['values'] = wt_opt['blade.internal_structure_2d_fem.web_start_nd'][i,:].tolist()
                self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'][i]['end_nd_arc']['grid']     = wt_opt['blade.internal_structure_2d_fem.s'].tolist()
                self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'][i]['end_nd_arc']['values']   = wt_opt['blade.internal_structure_2d_fem.web_end_nd'][i,:].tolist()

            # Structural layers
            for i in range(self.modeling_options['blade']['n_layers']):
                self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['thickness']['grid']      = wt_opt['blade.internal_structure_2d_fem.s'].tolist()
                self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['thickness']['values']    = wt_opt['blade.ps.layer_thickness_param'][i,:].tolist()
                if wt_opt['blade.internal_structure_2d_fem.definition_layer'][i] < 7:
                    if 'start_nd_arc' not in self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]:
                        self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['start_nd_arc'] = {}
                    if 'end_nd_arc' not in self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]:
                        self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['end_nd_arc'] = {}
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['start_nd_arc']['grid']   = wt_opt['blade.internal_structure_2d_fem.s'].tolist()
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['start_nd_arc']['values'] = wt_opt['blade.internal_structure_2d_fem.layer_start_nd'][i,:].tolist()
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['end_nd_arc']['grid']     = wt_opt['blade.internal_structure_2d_fem.s'].tolist()
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['end_nd_arc']['values']   = wt_opt['blade.internal_structure_2d_fem.layer_end_nd'][i,:].tolist()
                if wt_opt['blade.internal_structure_2d_fem.definition_layer'][i] > 1 and wt_opt['blade.internal_structure_2d_fem.definition_layer'][i] < 6:
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['width']['grid']   = wt_opt['blade.internal_structure_2d_fem.s'].tolist()
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['width']['values'] = wt_opt['blade.internal_structure_2d_fem.layer_width'][i,:].tolist()
                if wt_opt['blade.internal_structure_2d_fem.definition_layer'][i] == 2 or wt_opt['blade.internal_structure_2d_fem.definition_layer'][i] == 3:
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['rotation']['grid']   = wt_opt['blade.internal_structure_2d_fem.s'].tolist()
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['rotation']['values'] = wt_opt['blade.internal_structure_2d_fem.layer_rotation'][i,:].tolist()
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['offset_y_pa']['grid']   = wt_opt['blade.internal_structure_2d_fem.s'].tolist()
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['offset_y_pa']['values'] = wt_opt['blade.internal_structure_2d_fem.layer_offset_y_pa'][i,:].tolist()
                if wt_opt['blade.internal_structure_2d_fem.definition_layer'][i] == 4 or wt_opt['blade.internal_structure_2d_fem.definition_layer'][i] == 5:
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['midpoint_nd_arc']['grid']   = wt_opt['blade.internal_structure_2d_fem.s'].tolist()
                    self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['midpoint_nd_arc']['values'] = wt_opt['blade.internal_structure_2d_fem.layer_midpoint_nd'][i,:].tolist()

                self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['fiber_orientation'] = {}

                self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['fiber_orientation']['grid'] = wt_opt['blade.internal_structure_2d_fem.s'].tolist()
                self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['fiber_orientation']['values'] = np.zeros(len(wt_opt['blade.internal_structure_2d_fem.s'])).tolist()

            # Elastic properties of the blade
            self.wt_init['components']['blade']['elastic_properties_mb'] = {}
            self.wt_init['components']['blade']['elastic_properties_mb']['six_x_six'] = {}
            self.wt_init['components']['blade']['elastic_properties_mb']['six_x_six']['reference_axis'] = self.wt_init['components']['blade']['internal_structure_2d_fem']['reference_axis']
            self.wt_init['components']['blade']['elastic_properties_mb']['six_x_six']['twist'] = self.wt_init['components']['blade']['outer_shape_bem']['twist']
            self.wt_init['components']['blade']['elastic_properties_mb']['six_x_six']['stiff_matrix'] = {}
            self.wt_init['components']['blade']['elastic_properties_mb']['six_x_six']['stiff_matrix']['grid'] = wt_opt['blade.outer_shape_bem.s'].tolist()
            K = []
            for i in range(self.modeling_options['blade']['n_span']):
                Ki = np.zeros(21)
                Ki[11] = wt_opt['elastic.EA'][i]
                Ki[15] = wt_opt['elastic.EIxx'][i]
                Ki[18] = wt_opt['elastic.EIyy'][i]
                Ki[20] = wt_opt['elastic.GJ'][i]
                K.append(Ki.tolist())
            self.wt_init['components']['blade']['elastic_properties_mb']['six_x_six']['stiff_matrix']['values'] = K
            self.wt_init['components']['blade']['elastic_properties_mb']['six_x_six']['inertia_matrix'] = {}
            self.wt_init['components']['blade']['elastic_properties_mb']['six_x_six']['inertia_matrix']['grid'] = wt_opt['blade.outer_shape_bem.s'].tolist()
            I = []
            for i in range(self.modeling_options['blade']['n_span']):
                Ii = np.zeros(21)
                Ii[0]  = wt_opt['elastic.rhoA'][i]
                Ii[5]  = - wt_opt['elastic.rhoA'][i] * wt_opt['elastic.precomp.y_cg'][i]
                Ii[6]  = wt_opt['elastic.rhoA'][i]
                Ii[10] = wt_opt['elastic.rhoA'][i] * wt_opt['elastic.precomp.x_cg'][i]
                Ii[11] = wt_opt['elastic.rhoA'][i]
                Ii[12] = wt_opt['elastic.rhoA'][i] * wt_opt['elastic.precomp.y_cg'][i]
                Ii[13] = - wt_opt['elastic.rhoA'][i] * wt_opt['elastic.precomp.x_cg'][i]
                Ii[15] = wt_opt['elastic.precomp.edge_iner'][i]
                Ii[16] = wt_opt['elastic.precomp.edge_iner'][i]
                # Ii[18] = wt_opt['elastic.precomp.edge_iner'][i]
                Ii[20] = wt_opt['elastic.rhoJ'][i]
                I.append(Ii.tolist())
            self.wt_init['components']['blade']['elastic_properties_mb']['six_x_six']['inertia_matrix']['values'] = I

        if self.modeling_options['flags']['hub']:
            # Update hub
            self.wt_init['components']['hub']['cone_angle']                  = float(wt_opt['hub.cone'])
            self.wt_init['components']['hub']['flange_t2shell_t']            = float(wt_opt['hub.flange_t2shell_t'])
            self.wt_init['components']['hub']['flange_OD2hub_D']             = float(wt_opt['hub.flange_OD2hub_D'])
            self.wt_init['components']['hub']['flange_ID2OD']                = float(wt_opt['hub.flange_ID2flange_OD'])
            self.wt_init['components']['hub']['hub_blade_spacing_margin']    = float(wt_opt['hub.hub_in2out_circ'])
            self.wt_init['components']['hub']['hub_stress_concentration']    = float(wt_opt['hub.hub_stress_concentration'])
            self.wt_init['components']['hub']['n_front_brackets']            = int(wt_opt['hub.n_front_brackets'])
            self.wt_init['components']['hub']['n_rear_brackets']             = int(wt_opt['hub.n_rear_brackets'])
            self.wt_init['components']['hub']['clearance_hub_spinner']       = float(wt_opt['hub.clearance_hub_spinner'])
            self.wt_init['components']['hub']['spin_hole_incr']              = float(wt_opt['hub.spin_hole_incr'])
            self.wt_init['components']['hub']['pitch_system_scaling_factor'] = float(wt_opt['hub.pitch_system_scaling_factor'])
            self.wt_init['components']['hub']['spinner_gust_ws']             = float(wt_opt['hub.spinner_gust_ws'])

        if self.modeling_options['flags']['nacelle']:
            # Common direct and geared
            self.wt_init['components']['nacelle']['drivetrain']['uptilt']                    = float(wt_opt['nacelle.uptilt'])
            self.wt_init['components']['nacelle']['drivetrain']['distance_tt_hub']           = float(wt_opt['nacelle.distance_tt_hub'])
            self.wt_init['components']['nacelle']['drivetrain']['overhang']                  = float(wt_opt['nacelle.overhang'])
            self.wt_init['components']['nacelle']['drivetrain']['distance_hub_mb']           = float(wt_opt['nacelle.distance_hub2mb'])
            self.wt_init['components']['nacelle']['drivetrain']['distance_mb_mb']            = float(wt_opt['nacelle.distance_mb2mb'])
            self.wt_init['components']['nacelle']['drivetrain']['generator_length']          = float(wt_opt['nacelle.L_generator'])
            s_lss  = np.linspace(0.0, 1.0, len(wt_opt['nacelle.lss_diameter'])).tolist()
            self.wt_init['components']['nacelle']['drivetrain']['lss_diameter']['grid']      = s_lss
            self.wt_init['components']['nacelle']['drivetrain']['lss_diameter']['values']    = wt_opt['nacelle.lss_diameter'].tolist()
            self.wt_init['components']['nacelle']['drivetrain']['lss_wall_thickness']['grid']= s_lss
            self.wt_init['components']['nacelle']['drivetrain']['lss_wall_thickness']['values']= wt_opt['nacelle.lss_wall_thickness'].tolist()
            self.wt_init['components']['nacelle']['drivetrain']['gear_ratio']                = float(wt_opt['nacelle.gear_ratio'])
            self.wt_init['components']['nacelle']['drivetrain']['gearbox_efficiency']        = float(wt_opt['nacelle.gearbox_efficiency'])
            self.wt_init['components']['nacelle']['drivetrain']['mb1Type']                   = wt_opt['nacelle.mb1Type']
            self.wt_init['components']['nacelle']['drivetrain']['mb2Type']                   = wt_opt['nacelle.mb2Type']
            self.wt_init['components']['nacelle']['drivetrain']['uptower']                   = wt_opt['nacelle.uptower']
            self.wt_init['components']['nacelle']['drivetrain']['lss_material']              = wt_opt['nacelle.lss_material']
            self.wt_init['components']['nacelle']['drivetrain']['bedplate_material']         = wt_opt['nacelle.bedplate_material']

            if self.modeling_options['drivetrainse']['direct']:
                # Direct only
                s_nose = np.linspace(0.0, 1.0, len(wt_opt['nacelle.nose_diameter'])).tolist()
                s_bed  = np.linspace(0.0, 1.0, len(wt_opt['nacelle.bedplate_wall_thickness'])).tolist()
                self.wt_init['components']['nacelle']['drivetrain']['access_diameter']           = float(wt_opt['nacelle.access_diameter'])
                self.wt_init['components']['nacelle']['drivetrain']['nose_diameter']['grid']     = s_nose
                self.wt_init['components']['nacelle']['drivetrain']['nose_diameter']['values']   = wt_opt['nacelle.nose_diameter'].tolist()
                self.wt_init['components']['nacelle']['drivetrain']['nose_wall_thickness']['grid'] = s_nose
                self.wt_init['components']['nacelle']['drivetrain']['nose_wall_thickness']['values'] = wt_opt['nacelle.nose_wall_thickness'].tolist()
                self.wt_init['components']['nacelle']['drivetrain']['bedplate_wall_thickness']['grid'] = s_bed
                self.wt_init['components']['nacelle']['drivetrain']['bedplate_wall_thickness']['values']   = wt_opt['nacelle.bedplate_wall_thickness'].tolist()
            else:
                # Geared only
                s_hss  = np.linspace(0.0, 1.0, len(wt_opt['nacelle.hss_diameter'])).tolist()
                self.wt_init['components']['nacelle']['drivetrain']['hss_length']                = float(wt_opt['nacelle.hss_length'])
                self.wt_init['components']['nacelle']['drivetrain']['hss_diameter']['grid']      = s_hss
                self.wt_init['components']['nacelle']['drivetrain']['hss_diameter']['values']    = wt_opt['nacelle.hss_diameter'].tolist()
                self.wt_init['components']['nacelle']['drivetrain']['hss_wall_thickness']['grid'] = s_hss
                self.wt_init['components']['nacelle']['drivetrain']['hss_wall_thickness']['values']= wt_opt['nacelle.hss_wall_thickness'].tolist()
                self.wt_init['components']['nacelle']['drivetrain']['bedplate_flange_width']     = float(wt_opt['nacelle.bedplate_flange_width'])
                self.wt_init['components']['nacelle']['drivetrain']['bedplate_flange_thickness'] = float(wt_opt['nacelle.bedplate_flange_thickness'])
                self.wt_init['components']['nacelle']['drivetrain']['bedplate_web_thickness']    = float(wt_opt['nacelle.bedplate_web_thickness'])
                self.wt_init['components']['nacelle']['drivetrain']['gear_configuration']        = wt_opt['nacelle.gear_configuration']
                self.wt_init['components']['nacelle']['drivetrain']['planet_numbers']            = wt_opt['nacelle.planet_numbers']
                self.wt_init['components']['nacelle']['drivetrain']['hss_material']              = wt_opt['nacelle.hss_material']


        if self.modeling_options['flags']['generator']:

            self.wt_init['components']['nacelle']['generator']['B_r']         = float(wt_opt['generator.B_r'])
            self.wt_init['components']['nacelle']['generator']['P_Fe0e']      = float(wt_opt['generator.P_Fe0e'])
            self.wt_init['components']['nacelle']['generator']['P_Fe0h']      = float(wt_opt['generator.P_Fe0h'])
            self.wt_init['components']['nacelle']['generator']['S_N']         = float(wt_opt['generator.S_N'])
            self.wt_init['components']['nacelle']['generator']['alpha_p']     = float(wt_opt['generator.alpha_p'])
            self.wt_init['components']['nacelle']['generator']['b_r_tau_r']   = float(wt_opt['generator.b_r_tau_r'])
            self.wt_init['components']['nacelle']['generator']['b_ro']        = float(wt_opt['generator.b_ro'])
            self.wt_init['components']['nacelle']['generator']['b_s_tau_s']   = float(wt_opt['generator.b_s_tau_s'])
            self.wt_init['components']['nacelle']['generator']['b_so']        = float(wt_opt['generator.b_so'])
            self.wt_init['components']['nacelle']['generator']['cofi']        = float(wt_opt['generator.cofi'])
            self.wt_init['components']['nacelle']['generator']['freq']        = float(wt_opt['generator.freq'])
            self.wt_init['components']['nacelle']['generator']['h_i']         = float(wt_opt['generator.h_i'])
            self.wt_init['components']['nacelle']['generator']['h_sy0']       = float(wt_opt['generator.h_sy0'])
            self.wt_init['components']['nacelle']['generator']['h_w']         = float(wt_opt['generator.h_w'])
            self.wt_init['components']['nacelle']['generator']['k_fes']       = float(wt_opt['generator.k_fes'])
            self.wt_init['components']['nacelle']['generator']['k_fillr']     = float(wt_opt['generator.k_fillr'])
            self.wt_init['components']['nacelle']['generator']['k_fills']     = float(wt_opt['generator.k_fills'])
            self.wt_init['components']['nacelle']['generator']['k_s']         = float(wt_opt['generator.k_s'])
            self.wt_init['components']['nacelle']['generator']['m']           = float(wt_opt['generator.m'] )
            self.wt_init['components']['nacelle']['generator']['mu_0']        = float(wt_opt['generator.mu_0'])
            self.wt_init['components']['nacelle']['generator']['mu_r']        = float(wt_opt['generator.mu_r'])
            self.wt_init['components']['nacelle']['generator']['p']           = float(wt_opt['generator.p'] )
            self.wt_init['components']['nacelle']['generator']['phi']         = float(wt_opt['generator.phi'])
            self.wt_init['components']['nacelle']['generator']['q1']          = float(wt_opt['generator.q1'])
            self.wt_init['components']['nacelle']['generator']['q2']          = float(wt_opt['generator.q2'])
            self.wt_init['components']['nacelle']['generator']['ratio_mw2pp'] = float(wt_opt['generator.ratio_mw2pp'])
            self.wt_init['components']['nacelle']['generator']['resist_Cu']   = float(wt_opt['generator.resist_Cu'])
            self.wt_init['components']['nacelle']['generator']['sigma']       = float(wt_opt['generator.sigma'])
            self.wt_init['components']['nacelle']['generator']['y_tau_p']     = float(wt_opt['generator.y_tau_p'])
            self.wt_init['components']['nacelle']['generator']['y_tau_pr']    = float(wt_opt['generator.y_tau_pr'])

            self.wt_init['components']['nacelle']['generator']['I_0']         = float(wt_opt['generator.I_0'])
            self.wt_init['components']['nacelle']['generator']['d_r']         = float(wt_opt['generator.d_r'])
            self.wt_init['components']['nacelle']['generator']['h_m']         = float(wt_opt['generator.h_m'])
            self.wt_init['components']['nacelle']['generator']['h_0']         = float(wt_opt['generator.h_0'])
            self.wt_init['components']['nacelle']['generator']['h_s']         = float(wt_opt['generator.h_s'])
            self.wt_init['components']['nacelle']['generator']['len_s']       = float(wt_opt['generator.len_s'])
            self.wt_init['components']['nacelle']['generator']['n_r']         = float(wt_opt['generator.n_r'])
            self.wt_init['components']['nacelle']['generator']['rad_ag']      = float(wt_opt['generator.rad_ag'])
            self.wt_init['components']['nacelle']['generator']['t_wr']        = float(wt_opt['generator.t_wr'])

            self.wt_init['components']['nacelle']['generator']['n_s']         = float(wt_opt['generator.n_s'])
            self.wt_init['components']['nacelle']['generator']['b_st']        = float(wt_opt['generator.b_st'])
            self.wt_init['components']['nacelle']['generator']['d_s']         = float(wt_opt['generator.d_s'])
            self.wt_init['components']['nacelle']['generator']['t_ws']        = float(wt_opt['generator.t_ws'])

            self.wt_init['components']['nacelle']['generator']['rho_Copper']  = float(wt_opt['generator.rho_Copper'])
            self.wt_init['components']['nacelle']['generator']['rho_Fe']      = float(wt_opt['generator.rho_Fe'])
            self.wt_init['components']['nacelle']['generator']['rho_Fes']     = float(wt_opt['generator.rho_Fes'])
            self.wt_init['components']['nacelle']['generator']['rho_PM']      = float(wt_opt['generator.rho_PM'])

            self.wt_init['components']['nacelle']['generator']['C_Cu']        = float(wt_opt['generator.C_Cu'])
            self.wt_init['components']['nacelle']['generator']['C_Fe']        = float(wt_opt['generator.C_Fe'])
            self.wt_init['components']['nacelle']['generator']['C_Fes']       = float(wt_opt['generator.C_Fes'])
            self.wt_init['components']['nacelle']['generator']['C_PM']        = float(wt_opt['generator.C_PM'])

            if self.modeling_options['GeneratorSE']['type'] in ['pmsg_outer']:
                self.wt_init['components']['nacelle']['generator']['N_c']           = float(wt_opt['generator.N_c'])
                self.wt_init['components']['nacelle']['generator']['b']             = float(wt_opt['generator.b'] )
                self.wt_init['components']['nacelle']['generator']['c']             = float(wt_opt['generator.c'] )
                self.wt_init['components']['nacelle']['generator']['E_p']           = float(wt_opt['generator.E_p'])
                self.wt_init['components']['nacelle']['generator']['h_yr']          = float(wt_opt['generator.h_yr'])
                self.wt_init['components']['nacelle']['generator']['h_ys']          = float(wt_opt['generator.h_ys'])
                self.wt_init['components']['nacelle']['generator']['h_sr']          = float(wt_opt['generator.h_sr'])
                self.wt_init['components']['nacelle']['generator']['h_ss']          = float(wt_opt['generator.h_ss'])
                self.wt_init['components']['nacelle']['generator']['t_r']           = float(wt_opt['generator.t_r'])
                self.wt_init['components']['nacelle']['generator']['t_s']           = float(wt_opt['generator.t_s'])

                self.wt_init['components']['nacelle']['generator']['u_allow_pcent'] = float(wt_opt['generator.u_allow_pcent'])
                self.wt_init['components']['nacelle']['generator']['y_allow_pcent'] = float(wt_opt['generator.y_allow_pcent'])
                self.wt_init['components']['nacelle']['generator']['z_allow_deg']   = float(wt_opt['generator.z_allow_deg'])
                self.wt_init['components']['nacelle']['generator']['B_tmax']        = float(wt_opt['generator.B_tmax'])

            if self.modeling_options['GeneratorSE']['type'] in ['eesg','pmsg_arms','pmsg_disc']:
                self.wt_init['components']['nacelle']['generator']['tau_p']         = float(wt_opt['generator.tau_p'])
                self.wt_init['components']['nacelle']['generator']['h_ys']          = float(wt_opt['generator.h_ys'])
                self.wt_init['components']['nacelle']['generator']['h_yr']          = float(wt_opt['generator.h_yr'])
                self.wt_init['components']['nacelle']['generator']['b_arm']         = float(wt_opt['generator.b_arm'])

            elif self.modeling_options['GeneratorSE']['type'] in ['scig','dfig']:
                self.wt_init['components']['nacelle']['generator']['B_symax']       = float(wt_opt['generator.B_symax'])
                self.wt_init['components']['nacelle']['generator']['S_Nmax']        = float(wt_opt['generator.S_Nmax'])

        # Update tower
        if self.modeling_options['flags']['tower']:
            self.wt_init['components']['tower']['outer_shape_bem']['outer_diameter']['grid']          = wt_opt['tower.s'].tolist()
            self.wt_init['components']['tower']['outer_shape_bem']['outer_diameter']['values']        = wt_opt['tower.diameter'].tolist()
            self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['x']['grid']     = wt_opt['tower.s'].tolist()
            self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['y']['grid']     = wt_opt['tower.s'].tolist()
            self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['z']['grid']     = wt_opt['tower.s'].tolist()
            self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['x']['values']   = wt_opt['tower.ref_axis'][:,0].tolist()
            self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['y']['values']   = wt_opt['tower.ref_axis'][:,1].tolist()
            self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['z']['values']   = wt_opt['tower.ref_axis'][:,2].tolist()
            self.wt_init['components']['tower']['internal_structure_2d_fem']['outfitting_factor']     = float( wt_opt['tower.outfitting_factor'] )
            for i in range(self.modeling_options['tower']['n_layers']):
                self.wt_init['components']['tower']['internal_structure_2d_fem']['layers'][i]['thickness']['grid']      = wt_opt['tower.s'].tolist()
                self.wt_init['components']['tower']['internal_structure_2d_fem']['layers'][i]['thickness']['values']    = np.hstack((wt_opt['tower.layer_thickness'][i,:], wt_opt['tower.layer_thickness'][i,-1])).tolist()

        # Update monopile
        if self.modeling_options['flags']['monopile']:
            self.wt_init['components']['monopile']['suctionpile_depth']            = float( wt_opt['monopile.suctionpile_depth'] )
            self.wt_init['components']['monopile']['outer_shape_bem']['outer_diameter']['grid']          = wt_opt['monopile.s'].tolist()
            self.wt_init['components']['monopile']['outer_shape_bem']['outer_diameter']['values']        = wt_opt['monopile.diameter'].tolist()
            self.wt_init['components']['monopile']['outer_shape_bem']['reference_axis']['x']['grid']     = wt_opt['monopile.s'].tolist()
            self.wt_init['components']['monopile']['outer_shape_bem']['reference_axis']['y']['grid']     = wt_opt['monopile.s'].tolist()
            self.wt_init['components']['monopile']['outer_shape_bem']['reference_axis']['z']['grid']     = wt_opt['monopile.s'].tolist()
            self.wt_init['components']['monopile']['outer_shape_bem']['reference_axis']['x']['values']   = wt_opt['monopile.ref_axis'][:,0].tolist()
            self.wt_init['components']['monopile']['outer_shape_bem']['reference_axis']['y']['values']   = wt_opt['monopile.ref_axis'][:,1].tolist()
            self.wt_init['components']['monopile']['outer_shape_bem']['reference_axis']['z']['values']   = wt_opt['monopile.ref_axis'][:,2].tolist()
            self.wt_init['components']['monopile']['internal_structure_2d_fem']['outfitting_factor']     = float( wt_opt['monopile.outfitting_factor'] )
            for i in range(self.modeling_options['monopile']['n_layers']):
                self.wt_init['components']['monopile']['internal_structure_2d_fem']['layers'][i]['thickness']['grid']      = wt_opt['monopile.s'].tolist()
                self.wt_init['components']['monopile']['internal_structure_2d_fem']['layers'][i]['thickness']['values']    = wt_opt['monopile.layer_thickness'][i,:].tolist()

        # Update rotor nacelle assembly
        if self.modeling_options['flags']['RNA']:
            self.wt_init['components']['RNA'] = {}
            self.wt_init['components']['RNA']['elastic_properties_mb'] = {}
            self.wt_init['components']['RNA']['elastic_properties_mb']['mass']        = float(wt_opt['drivese.rna_mass'])
            self.wt_init['components']['RNA']['elastic_properties_mb']['inertia']     = wt_opt['drivese.rna_I_TT'].tolist()
            self.wt_init['components']['RNA']['elastic_properties_mb']['center_mass'] = wt_opt['drivese.rna_cm'].tolist()

        # Write yaml with updated values
        sch.write_geometry_yaml(self.wt_init, fname_output)
