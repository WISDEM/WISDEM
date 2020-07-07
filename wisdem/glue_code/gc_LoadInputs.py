import numpy as np
import time
from wisdem.aeroelasticse.FAST_reader import InputReader_OpenFAST
from wisdem.aeroelasticse.CaseLibrary import RotorSE_rated, RotorSE_DLC_1_4_Rated, RotorSE_DLC_7_1_Steady, RotorSE_DLC_1_1_Turb, power_curve, RotorSE_DAC_rated, RotorSE_steady_wind

try:
    import ruamel_yaml as ry
except:
    try:
        import ruamel.yaml as ry
    except:
        raise ImportError('No module named ruamel.yaml or ruamel_yaml')
import jsonschema as json


class WindTurbineOntologyPython(object):
    # Pure python class to load the input yaml file and break into few sub-dictionaries, namely:
    #   - analysis_options: dictionary with all the inputs that will be passed as options to the openmdao components, such as the length of the arrays
    #   - blade: dictionary representing the entry blade in the yaml file
    #   - tower: dictionary representing the entry tower in the yaml file
    #   - nacelle: dictionary representing the entry nacelle in the yaml file
    #   - materials: dictionary representing the entry materials in the yaml file
    #   - airfoils: dictionary representing the entry airfoils in the yaml file

    def initialize(self, fname_input_wt, fname_input_analysis):
        # Class instance to break the yaml into sub dictionaries
        
        self.analysis_options = self.load_yaml(fname_input_analysis)

        # Load wind turbine yaml input
        self.fname_input = fname_input_wt
        self.wt_init     = self.load_ontology(self.fname_input, validate=self.analysis_options['yaml']['validate'], fname_schema=self.analysis_options['yaml']['path2schema'])
        self.openmdao_vectors()
        
        # Openfast
        FASTpref                        = {}
        FASTpref['Analysis_Level']      = self.analysis_options['openfast']['Analysis_Level']
        FASTpref['FAST_ver']            = self.analysis_options['openfast']['FAST_ver']
        FASTpref['dev_branch']          = self.analysis_options['openfast']['dev_branch']
        FASTpref['FAST_exe']            = self.analysis_options['openfast']['FAST_exe']
        FASTpref['FAST_directory']      = self.analysis_options['openfast']['FAST_directory']
        FASTpref['FAST_InputFile']      = self.analysis_options['openfast']['FAST_InputFile']
        FASTpref['Turbsim_exe']         = self.analysis_options['openfast']['Turbsim_exe']
        FASTpref['FAST_namingOut']      = self.analysis_options['openfast']['FAST_namingOut']
        FASTpref['FAST_runDirectory']   = self.analysis_options['openfast']['FAST_runDirectory']
        FASTpref['path2dll']            = self.analysis_options['openfast']['path2dll']
        FASTpref['debug_level']         = self.analysis_options['openfast']['debug_level']
        FASTpref['DLC_gust']            = None      # Max deflection
        FASTpref['DLC_extrm']           = None      # Max strain
        FASTpref['DLC_turbulent']       = RotorSE_steady_wind
        FASTpref['DLC_powercurve']      = None      # AEP
        if FASTpref['Analysis_Level'] > 0:
            fast = InputReader_OpenFAST(FAST_ver=FASTpref['FAST_ver'], dev_branch=FASTpref['dev_branch'])
            fast.FAST_InputFile = FASTpref['FAST_InputFile']
            fast.FAST_directory = FASTpref['FAST_directory']
            fast.path2dll = FASTpref['path2dll']
            fast.execute()
            self.analysis_options['openfast']['fst_vt']   = fast.fst_vt
        else:
            self.analysis_options['openfast']['fst_vt']   = {}
        self.analysis_options['openfast']['FASTpref'] = FASTpref

        return self.analysis_options, self.wt_init

    def openmdao_vectors(self):
        # Class instance to determine all the parameters used to initialize the openmdao arrays, i.e. number of airfoils, number of angles of attack, number of blade spanwise stations, etc
        # ==analysis_options = {}
        
        # Materials
        self.analysis_options['materials']          = {}
        self.analysis_options['materials']['n_mat'] = len(self.wt_init['materials'])
        
        # Airfoils
        self.analysis_options['airfoils']           = {}
        self.analysis_options['airfoils']['n_af']   = len(self.wt_init['airfoils'])
        self.analysis_options['airfoils']['n_aoa']  = self.analysis_options['rotorse']['n_aoa']
        if self.analysis_options['airfoils']['n_aoa'] / 4. == int(self.analysis_options['airfoils']['n_aoa'] / 4.):
            # One fourth of the angles of attack from -pi to -pi/6, half between -pi/6 to pi/6, and one fourth from pi/6 to pi 
            self.analysis_options['airfoils']['aoa']    = np.unique(np.hstack([np.linspace(-np.pi, -np.pi / 6., int(self.analysis_options['airfoils']['n_aoa'] / 4. + 1)), np.linspace(-np.pi / 6., np.pi / 6., int(self.analysis_options['airfoils']['n_aoa'] / 2.)), np.linspace(np.pi / 6., np.pi, int(self.analysis_options['airfoils']['n_aoa'] / 4. + 1))]))
        else:
            self.analysis_options['airfoils']['aoa']    = np.linspace(-np.pi, np.pi, self.analysis_options['airfoils']['n_aoa'])
            print('WARNING: If you like a grid of angles of attack more refined between +- 30 deg, please choose a n_aoa in the analysis option input file that is a multiple of 4. The current value of ' + str(self.analysis_options['airfoils']['n_aoa']) + ' is not a multiple of 4 and an equally spaced grid is adopted.')
        Re_all = []
        for i in range(self.analysis_options['airfoils']['n_af']):
            for j in range(len(self.wt_init['airfoils'][i]['polars'])):
                Re_all.append(self.wt_init['airfoils'][i]['polars'][j]['re'])
        self.analysis_options['airfoils']['n_Re']   = len(np.unique(Re_all))
        self.analysis_options['airfoils']['n_tab']  = 1
        self.analysis_options['airfoils']['n_xy']   = self.analysis_options['rotorse']['n_xy']
        self.analysis_options['airfoils']['af_used']      = self.wt_init['components']['blade']['outer_shape_bem']['airfoil_position']['labels']
        self.analysis_options['airfoils']['xfoil_path']   = self.analysis_options['xfoil']['path']
        
        # Blade
        self.analysis_options['blade']              = {}
        self.analysis_options['blade']['n_span']    = self.analysis_options['rotorse']['n_span']
        self.analysis_options['blade']['nd_span']   = np.linspace(0., 1., self.analysis_options['blade']['n_span']) # Equally spaced non-dimensional spanwise grid
        self.analysis_options['blade']['n_af_span'] = len(self.wt_init['components']['blade']['outer_shape_bem']['airfoil_position']['labels']) # This is the number of airfoils defined along blade span and it is often different than n_af, which is the number of airfoils defined in the airfoil database
        self.analysis_options['blade']['n_webs']    = len(self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'])
        self.analysis_options['blade']['n_layers']  = len(self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'])
        self.analysis_options['blade']['lofted_output'] = False
        self.analysis_options['blade']['n_freq']    = 10 # Number of blade nat frequencies computed
        
        self.analysis_options['blade']['layer_name'] = self.analysis_options['blade']['n_layers'] * ['']
        self.analysis_options['blade']['layer_mat']  = self.analysis_options['blade']['n_layers'] * ['']
        for i in range(self.analysis_options['blade']['n_layers']):
            self.analysis_options['blade']['layer_name'][i]  = self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['name']
            self.analysis_options['blade']['layer_mat'][i]   = self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['material']


        self.analysis_options['blade']['web_name']  = self.analysis_options['blade']['n_webs'] * ['']
        for i in range(self.analysis_options['blade']['n_webs']):
            self.analysis_options['blade']['web_name'][i]  = self.wt_init['components']['blade']['internal_structure_2d_fem']['webs'][i]['name']

        # Distributed aerodynamic control devices along blade
        self.analysis_options['blade']['n_te_flaps']      = 0
        if 'aerodynamic_control' in self.wt_init['components']['blade']:
            if 'te_flaps' in self.wt_init['components']['blade']['aerodynamic_control']:
                self.analysis_options['blade']['n_te_flaps'] = len(self.wt_init['components']['blade']['aerodynamic_control']['te_flaps'])
                self.analysis_options['airfoils']['n_tab']   = 3
            else:
                exit('A distributed aerodynamic control device is provided in the yaml input file, but not supported by wisdem.')

        # Tower 
        self.analysis_options['tower']['n_height']  = len(self.wt_init['components']['tower']['outer_shape_bem']['outer_diameter']['grid'])
        self.analysis_options['tower']['n_layers']  = len(self.wt_init['components']['tower']['internal_structure_2d_fem']['layers'])

        # Monopile 
        self.analysis_options['monopile']              = {}
        self.analysis_options['tower']['monopile']  = 'monopile' in self.wt_init['components']
        if self.analysis_options['tower']['monopile']:
            self.analysis_options['monopile']['n_height']  = len(self.wt_init['components']['monopile']['outer_shape_bem']['outer_diameter']['grid'])
            self.analysis_options['monopile']['n_layers']  = len(self.wt_init['components']['monopile']['internal_structure_2d_fem']['layers'])

        # FloatingSE
        self.analysis_options['floating']          = {}
        self.analysis_options['floating']['flag']  = 'floating' in self.wt_init['components']
        
        # Assembly
        self.analysis_options['assembly'] = {}
        self.analysis_options['assembly']['number_of_blades'] = int(self.wt_init['assembly']['number_of_blades'])
        
    def load_ontology(self, fname_input, validate=False, fname_schema=''):
        """ Load inputs IEA turbine ontology yaml inputs, optional validation """
        # Read IEA turbine ontology yaml input file
        t_load = time.time()
        with open(fname_input, 'r') as myfile:
            inputs = myfile.read()

        # Validate the turbine input with the IEA turbine ontology schema
        yaml = ry.YAML()
        if validate:
            t_validate = time.time()

            with open(fname_schema, 'r') as myfile:
                schema = myfile.read()
            json.validate(yaml.load(inputs), yaml.load(schema))

            t_validate = time.time()-t_validate
            if self.analysis_options['general']['verbosity']:
                print('Complete: Schema "%s" validation: \t%f s'%(fname_schema, t_validate))
        else:
            t_validate = 0.

        if self.analysis_options['general']['verbosity']:
            t_load = time.time() - t_load - t_validate
            print('Complete: Load Input File: \t%f s'%(t_load))
        
        return yaml.load(inputs)
    
    def load_yaml(self, fname_input):
        """ Load optimization options """
        with open(fname_input, 'r') as myfile:
            inputs = myfile.read()
        yaml = ry.YAML()
        
        return yaml.load(inputs)

    def write_ontology(self, wt_opt, fname_output):

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
        for i in range(self.analysis_options['blade']['n_webs']):
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
        for i in range(self.analysis_options['blade']['n_layers']):
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
        for i in range(self.analysis_options['blade']['n_span']):
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
        for i in range(self.analysis_options['blade']['n_span']):
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

        if self.analysis_options['openfast']['update_hub_nacelle']:
            # Update hub
            self.wt_init['components']['hub']['outer_shape_bem']['diameter']   = float(wt_opt['hub.diameter'])
            self.wt_init['components']['hub']['outer_shape_bem']['cone_angle'] = float(wt_opt['hub.cone'])
            self.wt_init['components']['hub']['elastic_properties_mb']['system_mass']      = float(wt_opt['drivese.hub_system_mass'])
            self.wt_init['components']['hub']['elastic_properties_mb']['system_inertia']   = wt_opt['drivese.hub_system_I'].tolist()
            # Update nacelle
            self.wt_init['components']['nacelle']['outer_shape_bem']['uptilt_angle']        = float(wt_opt['nacelle.uptilt'])
            self.wt_init['components']['nacelle']['outer_shape_bem']['distance_tt_hub']     = float(wt_opt['nacelle.distance_tt_hub'])
            self.wt_init['components']['nacelle']['elastic_properties_mb']['above_yaw_mass']= float(wt_opt['drivese.above_yaw_mass'])
            self.wt_init['components']['nacelle']['elastic_properties_mb']['yaw_mass']      = float(wt_opt['drivese.yaw_mass'])
            self.wt_init['components']['nacelle']['elastic_properties_mb']['center_mass']   = wt_opt['drivese.nacelle_cm'].tolist()
            self.wt_init['components']['nacelle']['elastic_properties_mb']['inertia']       = wt_opt['drivese.nacelle_I'].tolist()

        #if self.analysis_options['tower']['run_towerse']:
        # Update tower
        self.wt_init['components']['tower']['outer_shape_bem']['outer_diameter']['grid']          = wt_opt['tower.s'].tolist()
        self.wt_init['components']['tower']['outer_shape_bem']['outer_diameter']['values']        = wt_opt['tower.diameter'].tolist()
        self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['x']['grid']     = wt_opt['tower.s'].tolist()
        self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['y']['grid']     = wt_opt['tower.s'].tolist()
        self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['z']['grid']     = wt_opt['tower.s'].tolist()
        self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['x']['values']   = wt_opt['tower.ref_axis'][:,0].tolist()
        self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['y']['values']   = wt_opt['tower.ref_axis'][:,1].tolist()
        self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['z']['values']   = wt_opt['tower.ref_axis'][:,2].tolist()
        for i in range(self.analysis_options['tower']['n_layers']):
            self.wt_init['components']['tower']['internal_structure_2d_fem']['layers'][i]['thickness']['grid']      = wt_opt['tower.s'].tolist()
            self.wt_init['components']['tower']['internal_structure_2d_fem']['layers'][i]['thickness']['values']    = np.hstack((wt_opt['tower.layer_thickness'][i,:], wt_opt['tower.layer_thickness'][i,-1])).tolist()

        # Update monopile
        if self.analysis_options['tower']['monopile']:
            self.wt_init['components']['monopile']['outer_shape_bem']['outer_diameter']['grid']          = wt_opt['monopile.s'].tolist()
            self.wt_init['components']['monopile']['outer_shape_bem']['outer_diameter']['values']        = wt_opt['monopile.diameter'].tolist()
            self.wt_init['components']['monopile']['outer_shape_bem']['reference_axis']['x']['grid']     = wt_opt['monopile.s'].tolist()
            self.wt_init['components']['monopile']['outer_shape_bem']['reference_axis']['y']['grid']     = wt_opt['monopile.s'].tolist()
            self.wt_init['components']['monopile']['outer_shape_bem']['reference_axis']['z']['grid']     = wt_opt['monopile.s'].tolist()
            self.wt_init['components']['monopile']['outer_shape_bem']['reference_axis']['x']['values']   = wt_opt['monopile.ref_axis'][:,0].tolist()
            self.wt_init['components']['monopile']['outer_shape_bem']['reference_axis']['y']['values']   = wt_opt['monopile.ref_axis'][:,1].tolist()
            self.wt_init['components']['monopile']['outer_shape_bem']['reference_axis']['z']['values']   = wt_opt['monopile.ref_axis'][:,2].tolist()
            for i in range(self.analysis_options['monopile']['n_layers']):
                self.wt_init['components']['monopile']['internal_structure_2d_fem']['layers'][i]['thickness']['grid']      = wt_opt['monopile.s'].tolist()
                self.wt_init['components']['monopile']['internal_structure_2d_fem']['layers'][i]['thickness']['values']    = wt_opt['monopile.layer_thickness'][i,:].tolist()


        # Update controller
        self.wt_init['control']['tsr']      = float(wt_opt['pc.tsr_opt'])
        self.wt_init['control']['PC_omega'] = float(wt_opt['control.PC_omega'])
        self.wt_init['control']['PC_zeta']  = float(wt_opt['control.PC_zeta'])
        self.wt_init['control']['VS_omega'] = float(wt_opt['control.VS_omega'])
        self.wt_init['control']['VS_zeta']  = float(wt_opt['control.VS_zeta'])
        self.wt_init['control']['Flp_omega']= float(wt_opt['control.Flp_omega'])
        self.wt_init['control']['Flp_zeta'] = float(wt_opt['control.Flp_zeta'])

        # Write yaml with updated values
        f = open(fname_output, "w")
        yaml=ry.YAML()
        yaml.default_flow_style = None
        yaml.width = float("inf")
        yaml.indent(mapping=4, sequence=6, offset=3)
        yaml.dump(self.wt_init, f)
        f.close()

        return None

if __name__ == "__main__":
    pass
