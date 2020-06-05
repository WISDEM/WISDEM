
try:
    import ruamel_yaml as ry
except:
    try:
        import ruamel.yaml as ry
    except:
        raise ImportError('No module named ruamel.yaml or ruamel_yaml')

import numpy as np
import jsonschema as json
import time, copy
from scipy.interpolate import PchipInterpolator, interp1d
import openmdao.api as om
from wisdem.rotorse.geometry_tools.geometry import AirfoilShape, trailing_edge_smoothing, remap2grid
from wisdem.rotorse.parametrize_rotor import ParametrizeBladeAero, ParametrizeBladeStruct
from wisdem.commonse.utilities import arc_length, arc_length_deriv
from wisdem.aeroelasticse.FAST_reader import InputReader_OpenFAST
from wisdem.aeroelasticse.CaseLibrary import RotorSE_rated, RotorSE_DLC_1_4_Rated, RotorSE_DLC_7_1_Steady, RotorSE_DLC_1_1_Turb, power_curve, RotorSE_DAC_rated


def calc_axis_intersection(xy_coord, rotation, offset, p_le_d, side, thk=0.):
    # dimentional analysis that takes a rotation and offset from the pitch axis and calculates the airfoil intersection
    # rotation
    offset_x   = offset*np.cos(rotation) + p_le_d[0]
    offset_y   = offset*np.sin(rotation) + p_le_d[1]

    m_rot      = np.sin(rotation)/np.cos(rotation)       # slope of rotated axis
    plane_rot  = [m_rot, -1*m_rot*p_le_d[0]+ p_le_d[1]]  # coefficients for rotated axis line: a1*x + a0

    m_intersection     = np.sin(rotation+np.pi/2.)/np.cos(rotation+np.pi/2.)   # slope perpendicular to rotated axis
    plane_intersection = [m_intersection, -1*m_intersection*offset_x+offset_y] # coefficients for line perpendicular to rotated axis line at the offset: a1*x + a0
    
    # intersection between airfoil surface and the line perpendicular to the rotated/offset axis
    y_intersection = np.polyval(plane_intersection, xy_coord[:,0])
    
    idx_le = np.argmin(xy_coord[:,0])
    xy_coord_arc = arc_length(xy_coord)
    arc_L = xy_coord_arc[-1]
    xy_coord_arc /= arc_L
    
    idx_inter      = np.argwhere(np.diff(np.sign(xy_coord[:,1] - y_intersection))).flatten() # find closest airfoil surface points to intersection 
    
    midpoint_arc = []
    for sidei in side:
        if sidei.lower() == 'suction':
            tangent_line = np.polyfit(xy_coord[idx_inter[0]:idx_inter[0]+2, 0], xy_coord[idx_inter[0]:idx_inter[0]+2, 1], 1)
        elif sidei.lower() == 'pressure':
            tangent_line = np.polyfit(xy_coord[idx_inter[1]:idx_inter[1]+2, 0], xy_coord[idx_inter[1]:idx_inter[1]+2, 1], 1)

        midpoint_x = (tangent_line[1]-plane_intersection[1])/(plane_intersection[0]-tangent_line[0])
        midpoint_y = plane_intersection[0]*(tangent_line[1]-plane_intersection[1])/(plane_intersection[0]-tangent_line[0]) + plane_intersection[1]

        # convert to arc position
        if sidei.lower() == 'suction':
            x_half = xy_coord[:idx_le+1,0]
            arc_half = xy_coord_arc[:idx_le+1]

        elif sidei.lower() == 'pressure':
            x_half = xy_coord[idx_le:,0]
            arc_half = xy_coord_arc[idx_le:]
        
        midpoint_arc.append(remap2grid(x_half, arc_half, midpoint_x, spline=interp1d))

    return midpoint_arc

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
        FASTpref['DLC_turbulent']       = RotorSE_DAC_rated
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

class Blade(om.Group):
    # Openmdao group with components with the blade data coming from the input yaml file.
    def initialize(self):
        self.options.declare('blade_init_options')
        self.options.declare('af_init_options')
        self.options.declare('opt_options')
                
    def setup(self):
        # Options
        blade_init_options = self.options['blade_init_options']
        af_init_options    = self.options['af_init_options']
        opt_options        = self.options['opt_options']
        
        # Optimization parameters initialized as indipendent variable component
        opt_var = om.IndepVarComp()
        opt_var.add_output('s_opt_twist',      val = np.ones(opt_options['optimization_variables']['blade']['aero_shape']['twist']['n_opt']))
        opt_var.add_output('s_opt_chord',      val = np.ones(opt_options['optimization_variables']['blade']['aero_shape']['twist']['n_opt']))
        opt_var.add_output('twist_opt_gain',   val = np.ones(opt_options['optimization_variables']['blade']['aero_shape']['twist']['n_opt']))
        opt_var.add_output('chord_opt_gain',   val = np.ones(opt_options['optimization_variables']['blade']['aero_shape']['chord']['n_opt']))
        opt_var.add_output('af_position',      val = np.ones(blade_init_options['n_af_span']))
        opt_var.add_output('spar_cap_ss_opt_gain', val = np.ones(opt_options['optimization_variables']['blade']['structure']['spar_cap_ss']['n_opt']))
        opt_var.add_output('spar_cap_ps_opt_gain', val = np.ones(opt_options['optimization_variables']['blade']['structure']['spar_cap_ps']['n_opt']))
        opt_var.add_output('te_flap_end', val = np.ones(blade_init_options['n_te_flaps']))
        opt_var.add_output('te_flap_ext', val = np.ones(blade_init_options['n_te_flaps']))
        self.add_subsystem('opt_var',opt_var)

        # Import outer shape BEM
        self.add_subsystem('outer_shape_bem', Blade_Outer_Shape_BEM(blade_init_options = blade_init_options), promotes = ['length'])

        # Parametrize blade outer shape
        self.add_subsystem('pa',    ParametrizeBladeAero(blade_init_options = blade_init_options, opt_options = opt_options)) # Parameterize aero (chord and twist)

        # Interpolate airfoil profiles and coordinates
        self.add_subsystem('interp_airfoils', Blade_Interp_Airfoils(blade_init_options = blade_init_options, af_init_options = af_init_options))
        
        # Connections to outer_shape_bem
        self.connect('opt_var.te_flap_end', 'outer_shape_bem.te_flap_span_end')
        self.connect('opt_var.te_flap_ext', 'outer_shape_bem.te_flap_span_ext')

        # Connections to blade aero parametrization
        self.connect('opt_var.s_opt_twist',       'pa.s_opt_twist')
        self.connect('opt_var.s_opt_chord',       'pa.s_opt_chord')
        self.connect('opt_var.twist_opt_gain',    'pa.twist_opt_gain')
        self.connect('opt_var.chord_opt_gain',    'pa.chord_opt_gain')

        self.connect('outer_shape_bem.s',           'pa.s')
        self.connect('outer_shape_bem.twist',       'pa.twist_original')
        self.connect('outer_shape_bem.chord',       'pa.chord_original')

        # Connections from oute_shape_bem to interp_airfoils
        self.connect('outer_shape_bem.s',           'interp_airfoils.s')
        self.connect('pa.chord_param',              'interp_airfoils.chord')
        self.connect('outer_shape_bem.pitch_axis',  'interp_airfoils.pitch_axis')
        self.connect('opt_var.af_position',         'interp_airfoils.af_position')
        
        # If the flag is true, generate the 3D x,y,z points of the outer blade shape
        if blade_init_options['lofted_output'] == True:
            self.add_subsystem('blade_lofted',    Blade_Lofted_Shape(blade_init_options = blade_init_options, af_init_options = af_init_options))
            self.connect('interp_airfoils.coord_xy_dim',    'blade_lofted.coord_xy_dim')
            self.connect('pa.twist_param',                  'blade_lofted.twist')
            self.connect('outer_shape_bem.s',               'blade_lofted.s')
            self.connect('outer_shape_bem.ref_axis',        'blade_lofted.ref_axis')
        
        # Import blade internal structure data and remap composites on the outer blade shape
        self.add_subsystem('internal_structure_2d_fem', Blade_Internal_Structure_2D_FEM(blade_init_options = blade_init_options, af_init_options = af_init_options))
        self.connect('outer_shape_bem.s',               'internal_structure_2d_fem.s')
        self.connect('pa.twist_param',                  'internal_structure_2d_fem.twist')
        self.connect('pa.chord_param',                  'internal_structure_2d_fem.chord')
        self.connect('outer_shape_bem.pitch_axis',      'internal_structure_2d_fem.pitch_axis')
        self.connect('interp_airfoils.coord_xy_dim',    'internal_structure_2d_fem.coord_xy_dim')

        self.add_subsystem('ps',    ParametrizeBladeStruct(blade_init_options = blade_init_options, opt_options = opt_options)) # Parameterize struct (spar caps ss and ps)

        # Connections to blade struct parametrization
        self.connect('opt_var.spar_cap_ss_opt_gain','ps.spar_cap_ss_opt_gain')
        self.connect('opt_var.spar_cap_ps_opt_gain','ps.spar_cap_ps_opt_gain')
        self.connect('outer_shape_bem.s',           'ps.s')
        # self.connect('internal_structure_2d_fem.layer_name',      'ps.layer_name')
        self.connect('internal_structure_2d_fem.layer_thickness', 'ps.layer_thickness_original')

        # Import trailing-edge flaps data
        n_te_flaps = blade_init_options['n_te_flaps']
        ivc = self.add_subsystem('dac_te_flaps', om.IndepVarComp())
        ivc.add_output('te_flap_start', val=np.zeros(n_te_flaps),               desc='1D array of the start positions along blade span of the trailing edge flap(s). Only values between 0 and 1 are meaningful.')
        ivc.add_output('te_flap_end',   val=np.zeros(n_te_flaps),               desc='1D array of the end positions along blade span of the trailing edge flap(s). Only values between 0 and 1 are meaningful.')
        ivc.add_output('chord_start',   val=np.zeros(n_te_flaps),               desc='1D array of the positions along chord where the trailing edge flap(s) start. Only values between 0 and 1 are meaningful.')
        ivc.add_output('delta_max_pos', val=np.zeros(n_te_flaps), units='rad',  desc='1D array of the max angle of the trailing edge flaps.')
        ivc.add_output('delta_max_neg', val=np.zeros(n_te_flaps), units='rad',  desc='1D array of the min angle of the trailing edge flaps.')

class Blade_Outer_Shape_BEM(om.Group):
    # Openmdao group with the blade outer shape data coming from the input yaml file.
    def initialize(self):
        self.options.declare('blade_init_options')

    def setup(self):
        blade_init_options = self.options['blade_init_options']
        n_af_span          = blade_init_options['n_af_span']
        self.n_span        = n_span = blade_init_options['n_span']
        
        ivc = self.add_subsystem('blade_outer_shape_indep_vars', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('af_position',        val=np.zeros(n_af_span),              desc='1D array of the non dimensional positions of the airfoils af_used defined along blade span.')
        ivc.add_output('s_default',          val=np.zeros(n_span),                 desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        ivc.add_output('chord_yaml',         val=np.zeros(n_span),    units='m',   desc='1D array of the chord values defined along blade span.')
        ivc.add_output('twist_yaml',         val=np.zeros(n_span),    units='rad', desc='1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).')
        ivc.add_output('pitch_axis_yaml',    val=np.zeros(n_span),                 desc='1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.')
        ivc.add_output('ref_axis_yaml',      val=np.zeros((n_span,3)),units='m',   desc='2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.')

        self.add_subsystem('compute_blade_outer_shape_bem', Compute_Blade_Outer_Shape_BEM(blade_init_options = blade_init_options), promotes = ['*'])

class Compute_Blade_Outer_Shape_BEM(om.ExplicitComponent):
    # Openmdao group with the blade outer shape data coming from the input yaml file.
    def initialize(self):
        self.options.declare('blade_init_options')

    def setup(self):
        blade_init_options = self.options['blade_init_options']
        n_af_span          = blade_init_options['n_af_span']
        self.n_span        = n_span = blade_init_options['n_span']
        self.n_te_flaps    = n_te_flaps = blade_init_options['n_te_flaps']

        # Inputs flaps
        self.add_input('te_flap_span_end', val=np.zeros(n_te_flaps),desc='1D array of the positions along blade span where the trailing edge flap(s) end. Only values between 0 and 1 are meaningful.')
        self.add_input('te_flap_span_ext', val=np.zeros(n_te_flaps),desc='1D array of the extensions along blade span of the trailing edge flap(s). Only values between 0 and 1 are meaningful.')

        self.add_input('s_default',        val=np.zeros(n_span),                 desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        self.add_input('chord_yaml',       val=np.zeros(n_span),    units='m',   desc='1D array of the chord values defined along blade span.')
        self.add_input('twist_yaml',       val=np.zeros(n_span),    units='rad', desc='1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).')
        self.add_input('pitch_axis_yaml',  val=np.zeros(n_span),                 desc='1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.')
        self.add_input('ref_axis_yaml',    val=np.zeros((n_span,3)),units='m',   desc='2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.')

        self.add_output('s',             val=np.zeros(n_span),                 desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        self.add_output('chord',         val=np.zeros(n_span),    units='m',   desc='1D array of the chord values defined along blade span.')
        self.add_output('twist',         val=np.zeros(n_span),    units='rad', desc='1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).')
        self.add_output('pitch_axis',    val=np.zeros(n_span),                 desc='1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.')
        self.add_output('ref_axis',      val=np.zeros((n_span,3)),units='m',   desc='2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.')
        
        self.add_output('length',       val = 0.0,               units='m',    desc='Scalar of the 3D blade length computed along its axis.')
        self.add_output('length_z',     val = 0.0,               units='m',    desc='Scalar of the 1D blade length along z, i.e. the blade projection in the plane ignoring prebend and sweep. For a straight blade this is equal to length')
        
    def compute(self, inputs, outputs):
        
        # If DAC devices are defined along span, manipulate the grid s to always have a grid point where it is needed, and reinterpolate the blade quantities, namely chord, twist, pitch axis, and reference axis
        if self.n_te_flaps > 0:
            nd_span_orig = np.linspace(0., 1.,self.n_span)

            chord_orig      = np.interp(nd_span_orig, inputs['s_default'], inputs['chord_yaml'])
            twist_orig      = np.interp(nd_span_orig, inputs['s_default'], inputs['twist_yaml'])
            pitch_axis_orig = np.interp(nd_span_orig, inputs['s_default'], inputs['pitch_axis_yaml'])
            ref_axis_orig   = np.zeros((self.n_span, 3))
            ref_axis_orig[:, 0] = np.interp(nd_span_orig,inputs['s_default'],inputs['ref_axis_yaml'][:, 0])
            ref_axis_orig[:, 1] = np.interp(nd_span_orig,inputs['s_default'],inputs['ref_axis_yaml'][:, 1])
            ref_axis_orig[:, 2] = np.interp(nd_span_orig,inputs['s_default'],inputs['ref_axis_yaml'][:, 2])

            outputs['s'] = copy.copy(nd_span_orig)
            
            # Account for blade flap start and end positions
            if len(inputs['te_flap_span_end']) > 0 :
                if inputs['te_flap_span_end'] >= 0.98:
                    flap_start = 0.98 - inputs['span_ext']
                    flap_end = 0.98
                    print('WARNING: te_flap_span_end optimization variable reached limits and was set to r/R = 0.98 when running XFoil')
                else:
                    flap_start = inputs['te_flap_span_end'] - inputs['te_flap_span_ext']
                    flap_end = inputs['te_flap_span_end']
            else:
                flap_start = 0.0
                flap_end = 0.0

            idx_flap_start = np.where(np.abs(nd_span_orig - flap_start) == (np.abs(nd_span_orig - flap_start)).min())[0][0]
            idx_flap_end = np.where(np.abs(nd_span_orig - flap_end) == (np.abs(nd_span_orig - flap_end)).min())[0][0]
            if idx_flap_start == idx_flap_end:
                idx_flap_end += 1
            outputs['s'][idx_flap_start] = flap_start
            outputs['s'][idx_flap_end] = flap_end
            outputs['chord'] = np.interp(outputs['s'], nd_span_orig, chord_orig)
            outputs['twist'] = np.interp(outputs['s'], nd_span_orig, twist_orig)
            outputs['pitch_axis'] = np.interp(outputs['s'], nd_span_orig, pitch_axis_orig)

            outputs['ref_axis'][:, 0] = np.interp(outputs['s'],nd_span_orig, ref_axis_orig[:, 0])
            outputs['ref_axis'][:, 1] = np.interp(outputs['s'],nd_span_orig, ref_axis_orig[:, 1])
            outputs['ref_axis'][:, 2] = np.interp(outputs['s'],nd_span_orig, ref_axis_orig[:, 2])
        else:
            outputs['s']            = inputs['s_default']
            outputs['chord']        = inputs['chord_yaml']
            outputs['twist']        = inputs['twist_yaml']
            outputs['pitch_axis']   = inputs['pitch_axis_yaml']
            outputs['ref_axis']     = inputs['ref_axis_yaml']

        outputs['length']   = arc_length(outputs['ref_axis'])[-1]
        outputs['length_z'] = outputs['ref_axis'][:,2][-1]

class Blade_Interp_Airfoils(om.ExplicitComponent):
    # Openmdao component to interpolate airfoil coordinates and airfoil polars along the span of the blade for a predefined set of airfoils coming from component Airfoils.
    # JPJ: can split this up into multiple components to ease derivative computation
    def initialize(self):
        self.options.declare('blade_init_options')
        self.options.declare('af_init_options')
        
    def setup(self):
        blade_init_options = self.options['blade_init_options']
        self.n_af_span     = n_af_span = blade_init_options['n_af_span']
        self.n_span        = n_span    = blade_init_options['n_span']
        af_init_options    = self.options['af_init_options']
        self.n_af          = n_af      = af_init_options['n_af'] # Number of airfoils
        self.n_aoa         = n_aoa     = af_init_options['n_aoa']# Number of angle of attacks
        self.n_Re          = n_Re      = af_init_options['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab         = n_tab     = af_init_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        self.n_xy          = n_xy      = af_init_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
        self.af_used       = af_init_options['af_used'] # Names of the airfoils adopted along blade span
                
        self.add_input('af_position',   val=np.zeros(n_af_span),              desc='1D array of the non dimensional positions of the airfoils af_used defined along blade span.')
        self.add_input('s',             val=np.zeros(n_span),                 desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        self.add_input('pitch_axis',    val=np.zeros(n_span),                 desc='1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.')
        self.add_input('chord',         val=np.zeros(n_span),    units='m',   desc='1D array of the chord values defined along blade span.')
        
        # Airfoil properties
        self.add_discrete_input('name', val=n_af * [''],                        desc='1D array of names of airfoils.')
        self.add_input('ac',        val=np.zeros(n_af),                         desc='1D array of the aerodynamic centers of each airfoil.')
        self.add_input('r_thick',   val=np.zeros(n_af),                         desc='1D array of the relative thicknesses of each airfoil.')
        self.add_input('aoa',       val=np.zeros(n_aoa),        units='rad',    desc='1D array of the angles of attack used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.')
        self.add_input('cl',        val=np.zeros((n_af, n_aoa, n_Re, n_tab)),   desc='4D array with the lift coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_input('cd',        val=np.zeros((n_af, n_aoa, n_Re, n_tab)),   desc='4D array with the drag coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_input('cm',        val=np.zeros((n_af, n_aoa, n_Re, n_tab)),   desc='4D array with the moment coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        
        # Airfoil coordinates
        self.add_input('coord_xy',  val=np.zeros((n_af, n_xy, 2)),              desc='3D array of the x and y airfoil coordinates of the n_af airfoils.')
        
        # Polars and coordinates interpolated along span
        self.add_output('r_thick_interp',   val=np.zeros(n_span),                         desc='1D array of the relative thicknesses of the blade defined along span.')
        self.add_output('ac_interp',        val=np.zeros(n_span),                         desc='1D array of the aerodynamic center of the blade defined along span.')
        self.add_output('cl_interp',        val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the lift coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_output('cd_interp',        val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the drag coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_output('cm_interp',        val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the moment coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_output('coord_xy_interp',  val=np.zeros((n_span, n_xy, 2)),              desc='3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The leading edge is place at x=0 and y=0.')
        self.add_output('coord_xy_dim',     val=np.zeros((n_span, n_xy, 2)), units = 'm', desc='3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.')
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        # Reconstruct the blade relative thickness along span with a pchip
        r_thick_used    = np.zeros(self.n_af_span)
        coord_xy_used   = np.zeros((self.n_af_span, self.n_xy, 2))
        coord_xy_interp = np.zeros((self.n_span, self.n_xy, 2))
        coord_xy_dim    = np.zeros((self.n_span, self.n_xy, 2))
        cl_used         = np.zeros((self.n_af_span, self.n_aoa, self.n_Re, self.n_tab))
        cl_interp       = np.zeros((self.n_span, self.n_aoa, self.n_Re, self.n_tab))
        cd_used         = np.zeros((self.n_af_span, self.n_aoa, self.n_Re, self.n_tab))
        cd_interp       = np.zeros((self.n_span, self.n_aoa, self.n_Re, self.n_tab))
        cm_used         = np.zeros((self.n_af_span, self.n_aoa, self.n_Re, self.n_tab))
        cm_interp       = np.zeros((self.n_span, self.n_aoa, self.n_Re, self.n_tab))
        
        for i in range(self.n_af_span):
            for j in range(self.n_af):
                if self.af_used[i] == discrete_inputs['name'][j]:                    
                    r_thick_used[i]     = inputs['r_thick'][j]
                    coord_xy_used[i,:,:]= inputs['coord_xy'][j]
                    cl_used[i,:,:,:]    = inputs['cl'][j,:,:,:]
                    cd_used[i,:,:,:]    = inputs['cd'][j,:,:,:]
                    cm_used[i,:,:,:]    = inputs['cm'][j,:,:,:]
                    break
        
        # Pchip does have an associated derivative method built-in:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.derivative.html#scipy.interpolate.PchipInterpolator.derivative
        spline         = PchipInterpolator
        rthick_spline  = spline(inputs['af_position'], r_thick_used)
        outputs['r_thick_interp'] = rthick_spline(inputs['s'])
        
        # Spanwise interpolation of the profile coordinates with a pchip
        r_thick_unique, indices  = np.unique(r_thick_used, return_index = True)
        profile_spline  = spline(r_thick_unique, coord_xy_used[indices, :, :])        
        coord_xy_interp = np.flip(profile_spline(np.flip(outputs['r_thick_interp'])), axis=0)
        
        
        for i in range(self.n_span):
            # Correction to move the leading edge (min x point) to (0,0)
            af_le = coord_xy_interp[i, np.argmin(coord_xy_interp[i,:,0]),:]
            coord_xy_interp[i,:,0] -= af_le[0]
            coord_xy_interp[i,:,1] -= af_le[1]
            c = max(coord_xy_interp[i,:,0]) - min(coord_xy_interp[i,:,0])
            coord_xy_interp[i,:,:] /= c
            # If the rel thickness is smaller than 0.4 apply a trailing ege smoothing step
            if outputs['r_thick_interp'][i] < 0.4: 
                coord_xy_interp[i,:,:] = trailing_edge_smoothing(coord_xy_interp[i,:,:])
            
        pitch_axis = inputs['pitch_axis']
        chord      = inputs['chord']

        
        coord_xy_dim = copy.copy(coord_xy_interp)
        coord_xy_dim[:,:,0] -= pitch_axis[:, np.newaxis]
        coord_xy_dim = coord_xy_dim*chord[:, np.newaxis, np.newaxis]
                
        
        # Spanwise interpolation of the airfoil polars with a pchip
        cl_spline = spline(r_thick_unique, cl_used[indices, :, :, :])        
        cl_interp = np.flip(cl_spline(np.flip(outputs['r_thick_interp'])), axis=0)
        cd_spline = spline(r_thick_unique, cd_used[indices, :, :, :])        
        cd_interp = np.flip(cd_spline(np.flip(outputs['r_thick_interp'])), axis=0)
        cm_spline = spline(r_thick_unique, cm_used[indices, :, :, :])        
        cm_interp = np.flip(cm_spline(np.flip(outputs['r_thick_interp'])), axis=0)
        
        
        # Plot interpolated polars
        # for i in range(self.n_span):    
            # plt.plot(inputs['aoa'], cl_interp[i,:,0,0], 'b')
            # plt.plot(inputs['aoa'], cd_interp[i,:,0,0], 'r')
            # plt.plot(inputs['aoa'], cm_interp[i,:,0,0], 'k')
            # plt.title(i)
            # plt.show()  
            
        outputs['coord_xy_interp'] = coord_xy_interp
        outputs['coord_xy_dim']    = coord_xy_dim
        outputs['cl_interp']       = cl_interp
        outputs['cd_interp']       = cd_interp
        outputs['cm_interp']       = cm_interp

        # # Plot interpolated coordinates
        # import matplotlib.pyplot as plt
        # for i in range(self.n_span):    
        #     plt.plot(coord_xy_interp[i,:,0], coord_xy_interp[i,:,1], 'k', label = 'coord_xy_interp')
        #     plt.plot(coord_xy_dim[i,:,0], coord_xy_dim[i,:,1], 'b', label = 'coord_xy_dim')
        #     plt.axis('equal')
        #     plt.title(i)
        #     plt.legend()
        #     plt.show()


        # # Smoothing
        # import matplotlib.pyplot as plt
        # # plt.plot(inputs['s'], inputs['chord'] * outputs['r_thick_interp'])
        # # plt.show()

        # # Absolute Thickness
        # abs_thick_init  = outputs['r_thick_interp']*inputs['chord']
        # s_interp_at     = np.array([0.0, 0.02,  0.1, 0.2, 0.8,  1.0 ])
        # abs_thick_int1  = np.interp(s_interp_at, inputs['s'],abs_thick_init)
        # f_interp2       = PchipInterpolator(s_interp_at,abs_thick_int1)
        # abs_thick_int2  = f_interp2(inputs['s'])

        # # # Relative thickness
        # r_thick_interp   = abs_thick_int2 / inputs['chord']
        # r_thick_airfoils = np.array([0.18, 0.211, 0.241, 0.301, 0.36 , 0.50, 1.00])
        # s_interp_rt      = np.interp(r_thick_airfoils, np.flip(r_thick_interp),np.flip(inputs['s']))
        # f_interp2        = PchipInterpolator(np.flip(s_interp_rt, axis=0),np.flip(r_thick_airfoils, axis=0))
        # r_thick_int2     = f_interp2(inputs['s'])

        
        # frt, axrt  = plt.subplots(1,1,figsize=(5.3, 4))
        # axrt.plot(inputs['s'], outputs['r_thick_interp']*100., c='k', label='Initial')
        # # axrt.plot(inputs['s'], r_thick_interp * 100., c='b', label='Interp')
        # # axrt.plot(s_interp_rt, r_thick_airfoils * 100., 'og', label='Airfoils')
        # # axrt.plot(inputs['s'], r_thick_int2 * 100., c='g', label='Reconstructed')
        # axrt.set(xlabel='r/R' , ylabel='Relative Thickness (%)')
        # axrt.legend()
        
        # fat, axat  = plt.subplots(1,1,figsize=(5.3, 4))
        # axat.plot(inputs['s'], abs_thick_init, c='k', label='Initial')
        # # axat.plot(s_interp_at, abs_thick_int1, 'ko', label='Interp Points')
        # # axat.plot(inputs['s'], abs_thick_int2, c='b', label='PCHIP')
        # # axat.plot(inputs['s'], r_thick_int2 * inputs['chord'], c='g', label='Reconstructed')
        # axat.set(xlabel='r/R' , ylabel='Absolute Thickness (m)')
        # axat.legend()
        # plt.show()
        # print(np.flip(s_interp_rt))
        # exit()

class Blade_Lofted_Shape(om.ExplicitComponent):
    # Openmdao component to generate the x, y, z coordinates of the points describing the blade outer shape.
    def initialize(self):
        self.options.declare('blade_init_options')
        self.options.declare('af_init_options')
        
    def setup(self):
        blade_init_options = self.options['blade_init_options']
        af_init_options    = self.options['af_init_options']
        self.n_span        = n_span = blade_init_options['n_span']
        self.n_xy          = n_xy   = af_init_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
                
        self.add_input('s',             val=np.zeros(n_span),                 desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        self.add_input('twist',         val=np.zeros(n_span),    units='rad', desc='1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).')
        self.add_input('ref_axis',      val=np.zeros((n_span,3)),units='m',   desc='2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.')
        
        self.add_input('coord_xy_dim',  val=np.zeros((n_span, n_xy, 2)),     units = 'm', desc='3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.')
        
        self.add_output('coord_xy_dim_twisted',val=np.zeros((n_span, n_xy, 2)), units = 'm', desc='3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.')
        self.add_output('3D_shape',     val = np.zeros((n_span * n_xy, 4)),   units = 'm', desc='4D array of the s, and x, y, and z coordinates of the points describing the outer shape of the blade. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.')
        
    def compute(self, inputs, outputs):

        for i in range(self.n_span):
            x = inputs['coord_xy_dim'][i,:,0]
            y = inputs['coord_xy_dim'][i,:,1]
            outputs['coord_xy_dim_twisted'][i,:,0] = x * np.cos(inputs['twist'][i]) - y * np.sin(inputs['twist'][i])
            outputs['coord_xy_dim_twisted'][i,:,1] = y * np.cos(inputs['twist'][i]) + x * np.sin(inputs['twist'][i])
                
        k=0
        for i in range(self.n_span):
            for j in range(self.n_xy):
                outputs['3D_shape'][k,:] = np.array([k, outputs['coord_xy_dim_twisted'][i,j,1], outputs['coord_xy_dim_twisted'][i,j,0], 0.0]) + np.hstack([0, inputs['ref_axis'][i,:]])
                k=k+1
        
        np.savetxt('3d_xyz_nrel5mw.dat', outputs['3D_shape'], header='\t point number [-]\t\t\t\t x [m] \t\t\t\t\t y [m]  \t\t\t\t z [m] \t\t\t\t The coordinate system follows the BeamDyn one.')
        
        import matplotlib.pyplot as plt
        for i in range(self.n_span):    
            plt.plot(outputs['coord_xy_dim_twisted'][i,:,0], outputs['coord_xy_dim_twisted'][i,:,1], 'k')
            plt.axis('equal')
            plt.title(i)
            plt.show()
            
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(outputs['3D_shape'][:,1],outputs['3D_shape'][:,2],outputs['3D_shape'][:,3])
        plt.show()

class Blade_Internal_Structure_2D_FEM(om.Group):
    # Openmdao group with the blade internal structure data coming from the input yaml file.
    def initialize(self):
        self.options.declare('blade_init_options')
        self.options.declare('af_init_options')

    def setup(self):
        blade_init_options = self.options['blade_init_options']
        af_init_options    = self.options['af_init_options']
        self.n_span        = n_span    = blade_init_options['n_span']
        self.n_webs        = n_webs    = blade_init_options['n_webs']
        self.n_layers      = n_layers  = blade_init_options['n_layers']
        self.n_xy          = n_xy      = af_init_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
        
        ivc = self.add_subsystem('blade_2dfem_indep_vars', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('layer_web',                 val=np.zeros(n_layers),                        desc='1D array of the web id the layer is associated to. If the layer is on the outer profile, this entry can simply stay equal to zero.')
        ivc.add_output('layer_thickness',           val=np.zeros((n_layers, n_span)), units='m',   desc='2D array of the thickness of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        ivc.add_output('layer_midpoint_nd',         val=np.zeros((n_layers, n_span)),              desc='2D array of the non-dimensional midpoint defined along the outer profile of a layer. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        ivc.add_discrete_output('layer_side',       val=n_layers * [''],                           desc='1D array setting whether the layer is on the suction or pressure side. This entry is only used if definition_layer is equal to 1 or 2.')
        ivc.add_discrete_output('definition_web',   val=np.zeros(n_webs),                          desc='1D array of flags identifying how webs are specified in the yaml. 1) offset+rotation=twist 2) offset+rotation')
        ivc.add_discrete_output('definition_layer', val=np.zeros(n_layers),                        desc='1D array of flags identifying how layers are specified in the yaml. 1) all around (skin, paint, ) 2) offset+rotation twist+width (spar caps) 3) offset+user defined rotation+width 4) midpoint TE+width (TE reinf) 5) midpoint LE+width (LE reinf) 6) layer position fixed to other layer (core fillers) 7) start and width 8) end and width 9) start and end nd 10) web layer')
        ivc.add_discrete_output('index_layer_start',val=np.zeros(n_layers),                        desc='Index used to fix a layer to another')
        ivc.add_discrete_output('index_layer_end',  val=np.zeros(n_layers),                        desc='Index used to fix a layer to another')
        
        ivc.add_output('web_start_nd_yaml',         val=np.zeros((n_webs, n_span)),                desc='2D array of the non-dimensional start point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.')
        ivc.add_output('web_end_nd_yaml',           val=np.zeros((n_webs, n_span)),                desc='2D array of the non-dimensional end point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.')
        ivc.add_output('web_rotation_yaml',         val=np.zeros((n_webs, n_span)),   units='rad', desc='2D array of the rotation angle of the shear webs in respect to the chord line. The first dimension represents each shear web, the second dimension represents each entry along blade span. If the rotation is equal to negative twist +- a constant, then the web is built straight.')
        ivc.add_output('web_offset_y_pa_yaml',      val=np.zeros((n_webs, n_span)),   units='m',   desc='2D array of the offset along the y axis to set the position of the shear webs. Positive values move the web towards the trailing edge, negative values towards the leading edge. The first dimension represents each shear web, the second dimension represents each entry along blade span.')
        ivc.add_output('layer_rotation_yaml',       val=np.zeros((n_layers, n_span)), units='rad', desc='2D array of the rotation angle of a layer in respect to the chord line. The first dimension represents each layer, the second dimension represents each entry along blade span. If the rotation is equal to negative twist +- a constant, then the layer is built straight.')
        ivc.add_output('layer_start_nd_yaml',       val=np.zeros((n_layers, n_span)),              desc='2D array of the non-dimensional start point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        ivc.add_output('layer_end_nd_yaml',         val=np.zeros((n_layers, n_span)),              desc='2D array of the non-dimensional end point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        ivc.add_output('layer_offset_y_pa_yaml',    val=np.zeros((n_layers, n_span)), units='m',   desc='2D array of the offset along the y axis to set the position of a layer. Positive values move the layer towards the trailing edge, negative values towards the leading edge. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        ivc.add_output('layer_width_yaml',          val=np.zeros((n_layers, n_span)), units='m',   desc='2D array of the width along the outer profile of a layer. The first dimension represents each layer, the second dimension represents each entry along blade span.')

        self.add_subsystem('compute_internal_structure_2d_fem', Compute_Blade_Internal_Structure_2D_FEM(blade_init_options = blade_init_options, af_init_options = af_init_options), promotes = ['*'])

class Compute_Blade_Internal_Structure_2D_FEM(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('blade_init_options')
        self.options.declare('af_init_options')

    def setup(self):
        blade_init_options = self.options['blade_init_options']
        af_init_options    = self.options['af_init_options']
        self.n_span        = n_span    = blade_init_options['n_span']
        self.n_webs        = n_webs    = blade_init_options['n_webs']
        self.n_layers      = n_layers  = blade_init_options['n_layers']
        self.n_xy          = n_xy      = af_init_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
        
         # From user defined yaml
        self.add_input('s',                     val=np.zeros(n_span),                          desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        self.add_input('web_rotation_yaml',     val=np.zeros((n_webs, n_span)),  units='rad',  desc='2D array of the rotation angle of the shear webs in respect to the chord line. The first dimension represents each shear web, the second dimension represents each entry along blade span. If the rotation is equal to negative twist +- a constant, then the web is built straight.')
        self.add_input('web_offset_y_pa_yaml',  val=np.zeros((n_webs, n_span)),  units='m',    desc='2D array of the offset along the y axis to set the position of the shear webs. Positive values move the web towards the trailing edge, negative values towards the leading edge. The first dimension represents each shear web, the second dimension represents each entry along blade span.')
        self.add_input('web_start_nd_yaml',     val=np.zeros((n_webs, n_span)),               desc='2D array of the non-dimensional start point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.')
        self.add_input('web_end_nd_yaml',       val=np.zeros((n_webs, n_span)),               desc='2D array of the non-dimensional end point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.')

        self.add_input('layer_web',             val=np.zeros(n_layers),                        desc='1D array of the web id the layer is associated to. If the layer is on the outer profile, this entry can simply stay equal to zero.')
        self.add_input('layer_thickness',       val=np.zeros((n_layers, n_span)), units='m',   desc='2D array of the thickness of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_input('layer_rotation_yaml',   val=np.zeros((n_layers, n_span)), units='rad', desc='2D array of the rotation angle of a layer in respect to the chord line. The first dimension represents each layer, the second dimension represents each entry along blade span. If the rotation is equal to negative twist +- a constant, then the layer is built straight.')
        self.add_input('layer_offset_y_pa_yaml',val=np.zeros((n_layers, n_span)), units='m',   desc='2D array of the offset along the y axis to set the position of a layer. Positive values move the layer towards the trailing edge, negative values towards the leading edge. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_input('layer_width_yaml',      val=np.zeros((n_layers, n_span)), units='m',   desc='2D array of the width along the outer profile of a layer. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_input('layer_midpoint_nd',     val=np.zeros((n_layers, n_span)),   desc='2D array of the non-dimensional midpoint defined along the outer profile of a layer. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_discrete_input('layer_side',   val=n_layers * [''],                desc='1D array setting whether the layer is on the suction or pressure side. This entry is only used if definition_layer is equal to 1 or 2.')
        self.add_input('layer_start_nd_yaml',   val=np.zeros((n_layers, n_span)),   desc='2D array of the non-dimensional start point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_input('layer_end_nd_yaml',      val=np.zeros((n_layers, n_span)),  desc='2D array of the non-dimensional end point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_discrete_input('definition_web',   val=np.zeros(n_webs),           desc='1D array of flags identifying how webs are specified in the yaml. 1) offset+rotation=twist 2) offset+rotation')
        self.add_discrete_input('definition_layer', val=np.zeros(n_layers),         desc='1D array of flags identifying how layers are specified in the yaml. 1) all around (skin, paint, ) 2) offset+rotation twist+width (spar caps) 3) offset+user defined rotation+width 4) midpoint TE+width (TE reinf) 5) midpoint LE+width (LE reinf) 6) layer position fixed to other layer (core fillers) 7) start and width 8) end and width 9) start and end nd 10) web layer')
        self.add_discrete_input('index_layer_start',val=np.zeros(n_layers),         desc='Index used to fix a layer to another')
        self.add_discrete_input('index_layer_end',  val=np.zeros(n_layers),         desc='Index used to fix a layer to another')

        # From blade outer shape
        self.add_input('coord_xy_dim',    val=np.zeros((n_span, n_xy, 2)), units = 'm', desc='3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.')
        self.add_input('twist',           val=np.zeros(n_span),            units='rad', desc='1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).')
        self.add_input('chord',           val=np.zeros(n_span),            units='m',   desc='1D array of the chord values defined along blade span.')
        self.add_input('pitch_axis',      val=np.zeros(n_span),                         desc='1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.')

        self.add_output('web_rotation',   val=np.zeros((n_webs, n_span)),  units='rad', desc='2D array of the rotation angle of the shear webs in respect to the chord line. The first dimension represents each shear web, the second dimension represents each entry along blade span. If the rotation is equal to negative twist +- a constant, then the web is built straight.')
        self.add_output('web_start_nd',   val=np.zeros((n_webs, n_span)),               desc='2D array of the non-dimensional start point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.')
        self.add_output('web_end_nd',     val=np.zeros((n_webs, n_span)),               desc='2D array of the non-dimensional end point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.')
        self.add_output('web_offset_y_pa',val=np.zeros((n_webs, n_span)),  units='m',   desc='2D array of the offset along the y axis to set the position of the shear webs. Positive values move the web towards the trailing edge, negative values towards the leading edge. The first dimension represents each shear web, the second dimension represents each entry along blade span.')
        self.add_output('layer_rotation', val=np.zeros((n_layers, n_span)),units='rad', desc='2D array of the rotation angle of a layer in respect to the chord line. The first dimension represents each layer, the second dimension represents each entry along blade span. If the rotation is equal to negative twist +- a constant, then the layer is built straight.')
        self.add_output('layer_start_nd', val=np.zeros((n_layers, n_span)),             desc='2D array of the non-dimensional start point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_output('layer_end_nd',   val=np.zeros((n_layers, n_span)),             desc='2D array of the non-dimensional end point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_output('layer_offset_y_pa',val=np.zeros((n_layers, n_span)), units='m',desc='2D array of the offset along the y axis to set the position of a layer. Positive values move the layer towards the trailing edge, negative values towards the leading edge. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_output('layer_width',    val=np.zeros((n_layers, n_span)), units='m',  desc='2D array of the width along the outer profile of a layer. The first dimension represents each layer, the second dimension represents each entry along blade span.')

        # # These outputs don't depend on anything and should be refactored to be
        # # outputs that come from an om.IndepVarComp.
        # self.declare_partials('definition_layer', '*', dependent=False)
        # self.declare_partials('layer_offset_y_pa', '*', dependent=False)
        # self.declare_partials('layer_thickness', '*', dependent=False)
        # self.declare_partials('layer_web', '*', dependent=False)
        # self.declare_partials('layer_width', '*', dependent=False)
        # self.declare_partials('s', '*', dependent=False)
        # self.declare_partials('web_offset_y_pa', '*', dependent=False)
        
        # self.declare_partials('layer_end_nd', ['coord_xy_dim', 'twist'], method='fd')
        # self.declare_partials('layer_midpoint_nd', ['coord_xy_dim'], method='fd')
        # self.declare_partials('layer_rotation', ['twist'], method='fd')
        # self.declare_partials('layer_start_nd', ['coord_xy_dim', 'twist'], method='fd')
        # self.declare_partials('web_end_nd', ['coord_xy_dim', 'twist'], method='fd')
        # self.declare_partials('web_rotation', ['twist'], method='fd')
        # self.declare_partials('web_start_nd', ['coord_xy_dim', 'twist'], method='fd')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        # Initialize temporary arrays for the outputs
        web_rotation    = np.zeros((self.n_webs, self.n_span))
        layer_rotation  = np.zeros((self.n_layers, self.n_span))
        web_start_nd    = np.zeros((self.n_webs, self.n_span))
        web_end_nd      = np.zeros((self.n_webs, self.n_span))
        layer_start_nd  = np.zeros((self.n_layers, self.n_span))
        layer_end_nd    = np.zeros((self.n_layers, self.n_span))

        layer_name      = self.options['blade_init_options']['layer_name']
        layer_mat       = self.options['blade_init_options']['layer_mat']
        web_name        = self.options['blade_init_options']['web_name']

        # Loop through spanwise stations
        for i in range(self.n_span):
            # Compute the arc length (arc_L_i), the non-dimensional arc coordinates (xy_arc_i), and the non dimensional position of the leading edge of the profile at position i
            xy_coord_i  = inputs['coord_xy_dim'][i,:,:]
            xy_arc_i    = arc_length(xy_coord_i)
            arc_L_i     = xy_arc_i[-1]
            xy_arc_i    /= arc_L_i
            idx_le      = np.argmin(xy_coord_i[:,0])
            LE_loc      = xy_arc_i[idx_le]
            chord       = inputs['chord'][i]
            p_le_i      = inputs['pitch_axis'][i]
            ratio_SCmax = 0.8
            ratio_Websmax = 0.75

            # Loop through the webs and compute non-dimensional start and end positions along the profile
            for j in range(self.n_webs):

                offset = inputs['web_offset_y_pa_yaml'][j,i]
                # Geometry checks on webs                    
                if offset < ratio_Websmax * (- chord * p_le_i) or offset > ratio_Websmax * (chord * (1. - p_le_i)):
                    offset_old = copy.copy(offset)
                    if offset_old <= 0.:
                        offset = ratio_Websmax * (- chord * p_le_i)
                    else:
                        offset = ratio_Websmax * (chord * (1. - p_le_i))
                    
                    outputs['web_offset_y_pa'][j,i] = copy.copy(offset)
                    layer_resize_warning = 'WARNING: Web "%s" may be too large to fit within chord. "offset_x_pa" changed from %f to %f at R=%f (i=%d)'%(web_name[j], offset_old, offset, inputs['s'][i], i)
                    print(layer_resize_warning)
                else:
                    outputs['web_offset_y_pa'][j,i] = copy.copy(offset)

                if discrete_inputs['definition_web'][j] == 1:
                    web_rotation[j,i] = - inputs['twist'][i]
                    web_start_nd[j,i], web_end_nd[j,i] = calc_axis_intersection(inputs['coord_xy_dim'][i,:,:], - web_rotation[j,i], outputs['web_offset_y_pa'][j,i], [0.,0.], ['suction', 'pressure'])
                elif discrete_inputs['definition_web'][j] == 2:
                    web_rotation[j,i] = - inputs['web_rotation_yaml'][j,i]
                    web_start_nd[j,i], web_end_nd[j,i] = calc_axis_intersection(inputs['coord_xy_dim'][i,:,:], - web_rotation[j,i], outputs['web_offset_y_pa'][j,i], [0.,0.], ['suction', 'pressure'])
                    if i == 0:
                        print('WARNING: The web ' + web_name[j] + ' is defined with a user-defined rotation. If you are planning to run a twist optimization, you may want to rethink this definition.')
                    if web_start_nd[j,i] < 0. or web_start_nd[j,i] > 1.:
                        print('WARNING: Blade web ' + web_name[j] + ' at n.d. span position ' + str(inputs['s'][i]) + ' has the n.d. start point outside the TE. Please check the yaml input file.')
                    if web_end_nd[j,i] < 0. or web_end_nd[j,i] > 1.:
                        print('WARNING: Blade web ' + web_name[j] + ' at n.d. span position ' + str(inputs['s'][i]) + ' has the n.d. end point outside the TE. Please check the yaml input file.')
                elif discrete_inputs['definition_web'][j] == 3:
                    web_start_nd[j,i] = inputs['web_start_nd_yaml'][j,i]
                    web_end_nd[j,i]   = inputs['web_end_nd_yaml'][j,i]
                else:
                    exit('Blade web ' + web_name[j] + ' not described correctly. Please check the yaml input file.')
                    
            # Loop through the layers and compute non-dimensional start and end positions along the profile for the different layer definitions
            for j in range(self.n_layers):
                if discrete_inputs['definition_layer'][j] == 1: # All around
                    layer_start_nd[j,i] = 0.
                    layer_end_nd[j,i]   = 1.
                elif discrete_inputs['definition_layer'][j] == 2 or discrete_inputs['definition_layer'][j] == 3: # Midpoint and width
                    if discrete_inputs['definition_layer'][j] == 2:
                        layer_rotation[j,i] = - inputs['twist'][i]
                    else:
                        layer_rotation[j,i] = - inputs['layer_rotation_yaml'][j,i]
                    midpoint = calc_axis_intersection(inputs['coord_xy_dim'][i,:,:], - layer_rotation[j,i], inputs['layer_offset_y_pa_yaml'][j,i], [0.,0.], [discrete_inputs['layer_side'][j]])[0]

                    # Geometry check to make sure the spar caps does not exceed 80% of the chord
                    width    = inputs['layer_width_yaml'][j,i]
                    offset   = inputs['layer_offset_y_pa_yaml'][j,i]
                    if offset + 0.5 * width > ratio_SCmax * chord * (1. - p_le_i) or offset - 0.5 * width < - ratio_SCmax * chord * p_le_i: # hitting TE or LE
                        width_old = copy.copy(width)
                        width     = 2. * min([ratio_SCmax * (chord * p_le_i ) , ratio_SCmax * (chord * (1. - p_le_i))])
                        offset    = 0.0
                        outputs['layer_width'][j,i]         = copy.copy(width)
                        outputs['layer_offset_y_pa'][j,i]   = copy.copy(offset)
                        layer_resize_warning = 'WARNING: Layer "%s" may be too large to fit within chord. "offset_x_pa" changed from %f to 0.0 and "width" changed from %f to %f at s=%f (i=%d)'%(layer_name[j], offset, width_old, width, inputs['s'][i], i)
                        print(layer_resize_warning)
                    else:
                        outputs['layer_width'][j,i]         = copy.copy(width)
                        outputs['layer_offset_y_pa'][j,i]   = copy.copy(offset)

                    layer_start_nd[j,i] = midpoint-width/arc_L_i/2.
                    layer_end_nd[j,i]   = midpoint+width/arc_L_i/2.

                elif discrete_inputs['definition_layer'][j] == 4: # Midpoint and width
                    midpoint = 1. 
                    inputs['layer_midpoint_nd'][j,i] = midpoint
                    width    = inputs['layer_width_yaml'][j,i]
                    outputs['layer_width'][j,i] = copy.copy(width)
                    layer_start_nd[j,i] = midpoint-width/arc_L_i/2.
                    layer_end_nd[j,i]   = width/arc_L_i/2.

                    # Geometry check to prevent overlap between SC and TE reinf
                    for k in range(self.n_layers):
                        if discrete_inputs['definition_layer'][k] == 2 or discrete_inputs['definition_layer'][k] == 3:
                            if layer_end_nd[j,i] > layer_start_nd[k,i] or layer_start_nd[j,i] < layer_end_nd[k,i]:
                                print('WARNING: The trailing edge reinforcement extends above the spar caps at station ' + str(i) + '. Please reduce its width.')

                elif discrete_inputs['definition_layer'][j] == 5: # Midpoint and width
                    midpoint = LE_loc
                    inputs['layer_midpoint_nd'][j,i] = midpoint
                    width    = inputs['layer_width_yaml'][j,i]
                    outputs['layer_width'][j,i] = copy.copy(width)
                    layer_start_nd[j,i] = midpoint-width/arc_L_i/2.
                    layer_end_nd[j,i]   = midpoint+width/arc_L_i/2.
                    # Geometry check to prevent overlap between SC and LE reinf
                    for k in range(self.n_layers):
                        if discrete_inputs['definition_layer'][k] == 2 or discrete_inputs['definition_layer'][k] == 3:
                            if discrete_inputs['layer_side'][k] == 'suction' and layer_start_nd[j,i] < layer_end_nd[k,i]:
                                print('WARNING: The leading edge reinforcement extends above the spar caps at station ' + str(i) + '. Please reduce its width.')
                            elif discrete_inputs['layer_side'][k] == 'pressure' and layer_end_nd[j,i] > layer_start_nd[k,i]:
                                print('WARNING: The leading edge reinforcement extends above the spar caps at station ' + str(i) + '. Please reduce its width.')
                            else:
                                pass
                elif discrete_inputs['definition_layer'][j] == 6: # Start and end locked to other element
                    # if inputs['layer_start_nd'][j,i] > 1:
                    layer_start_nd[j,i] = layer_end_nd[int(discrete_inputs['index_layer_start'][j]),i]
                    # if inputs['layer_end_nd'][j,i] > 1:
                    layer_end_nd[j,i]   = layer_start_nd[int(discrete_inputs['index_layer_end'][j]),i]
                elif discrete_inputs['definition_layer'][j] == 7: # Start nd and width
                    width    = inputs['layer_width_yaml'][j,i]
                    outputs['layer_width'][j,i] = copy.copy(width)
                    layer_start_nd[j,i] = inputs['layer_start_nd_yaml'][j,i]
                    layer_end_nd[j,i]   = layer_start_nd[j,i] + width/arc_L_i
                elif discrete_inputs['definition_layer'][j] == 8: # End nd and width
                    width    = inputs['layer_width_yaml'][j,i]
                    outputs['layer_width'][j,i] = copy.copy(width)
                    layer_end_nd[j,i]   = inputs['layer_end_nd_yaml'][j,i]
                    layer_start_nd[j,i] = layer_end_nd[j,i] - width/arc_L_i
                elif discrete_inputs['definition_layer'][j] == 9: # Start and end nd positions
                    layer_start_nd[j,i] = inputs['layer_start_nd_yaml'][j,i]
                    layer_end_nd[j,i]   = inputs['layer_end_nd_yaml'][j,i]
                elif discrete_inputs['definition_layer'][j] == 10: # Web layer
                    pass
                elif discrete_inputs['definition_layer'][j] == 11: # Start nd arc locked to LE
                    layer_start_nd[j,i] = LE_loc + 1.e-6
                    layer_end_nd[j,i]   = layer_start_nd[int(discrete_inputs['index_layer_end'][j]),i]
                elif discrete_inputs['definition_layer'][j] == 12: # End nd arc locked to LE
                    layer_end_nd[j,i] = LE_loc - 1.e-6
                    layer_start_nd[j,i] = layer_end_nd[int(discrete_inputs['index_layer_start'][j]),i]
                else:
                    exit('Blade layer ' + str(layer_name[j]) + ' not described correctly. Please check the yaml input file.')
        
        # Assign openmdao outputs
        outputs['web_rotation']   = web_rotation
        outputs['web_start_nd']   = web_start_nd
        outputs['web_end_nd']     = web_end_nd
        outputs['layer_rotation'] = layer_rotation
        outputs['layer_start_nd'] = layer_start_nd
        outputs['layer_end_nd']   = layer_end_nd

class Hub(om.Group):
    # Openmdao group with the hub data coming from the input yaml file.
    def setup(self):
        ivc = self.add_subsystem('hub_indep_vars', om.IndepVarComp(), promotes=['*'])
        
        ivc.add_output('diameter',     val=0.0, units='m',     desc='Diameter of the hub. It is equal to two times the distance of the blade root from the rotor center along the coned line.')
        ivc.add_output('cone',         val=0.0, units='rad',   desc='Cone angle of the rotor. It defines the angle between the rotor plane and the blade pitch axis. A standard machine has positive values.')
        ivc.add_output('drag_coeff',   val=0.0,                desc='Drag coefficient to estimate the aerodynamic forces generated by the hub.')

        ivc.add_output('system_mass',  val=0.0,         units='kg',        desc='Mass of hub system')
        ivc.add_output('system_I',     val=np.zeros(6), units='kg*m**2',   desc='Mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')
        
        exec_comp = om.ExecComp('radius = 0.5 * diameter', units='m', radius={'desc' : 'Radius of the hub. It defines the distance of the blade root from the rotor center along the coned line.'})
        self.add_subsystem('compute_radius', exec_comp, promotes=['*'])
        
class Tower(om.Group):
    
    def initialize(self):
        self.options.declare('tower_init_options')
        
    def setup(self):
        tower_init_options = self.options['tower_init_options']
        n_height           = tower_init_options['n_height']
        n_layers           = tower_init_options['n_layers']
        
        ivc = self.add_subsystem('tower_indep_vars', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('ref_axis', val=np.zeros((n_height, 3)), units='m', desc='2D array of the coordinates (x,y,z) of the tower reference axis. The coordinate system is the global coordinate system of OpenFAST: it is placed at tower base with x pointing downwind, y pointing on the side and z pointing vertically upwards. A standard tower configuration will have zero x and y values and positive z values.')
        ivc.add_output('diameter', val=np.zeros(n_height),     units='m',  desc='1D array of the outer diameter values defined along the tower axis.')
        ivc.add_output('layer_thickness',     val=np.zeros((n_layers, n_height-1)), units='m',    desc='2D array of the thickness of the layers of the tower structure. The first dimension represents each layer, the second dimension represents each piecewise-constant entry of the tower sections.')
        ivc.add_output('outfitting_factor',       val = 0.0,             desc='Multiplier that accounts for secondary structure mass inside of tower')
        ivc.add_discrete_output('layer_name', val=[],         desc='1D array of the names of the layers modeled in the tower structure.')
        ivc.add_discrete_output('layer_mat',  val=[],         desc='1D array of the names of the materials of each layer modeled in the tower structure.')
        
        self.add_subsystem('compute_tower_grid',
            ComputeGrid(init_options=tower_init_options),
            promotes=['*'])

class ComputeGrid(om.ExplicitComponent):
    """
    Compute the non-dimensional grid or a tower or monopile.
    
    Using the dimensional `ref_axis` array, this component computes the
    non-dimensional grid, height (vertical distance) and length (curve distance)
    of a tower or monopile.
    """
    
    def initialize(self):
        self.options.declare('init_options')
        
    def setup(self):
        init_options = self.options['init_options']
        n_height = init_options['n_height']

        self.add_input('ref_axis', val=np.zeros((n_height, 3)), units='m', desc='2D array of the coordinates (x,y,z) of the tower reference axis. The coordinate system is the global coordinate system of OpenFAST: it is placed at tower base with x pointing downwind, y pointing on the side and z pointing vertically upwards. A standard tower configuration will have zero x and y values and positive z values.')
        
        self.add_output('s', val=np.zeros(n_height), desc='1D array of the non-dimensional grid defined along the tower axis (0-tower base, 1-tower top)')
        self.add_output('height', val=0.0, units='m', desc='Scalar of the tower height computed along the z axis.')
        self.add_output('length', val=0.0, units='m', desc='Scalar of the tower length computed along its curved axis. A standard straight tower will be as high as long.')
        
        # Declare all partial derivatives.
        # For height wrt ref_axis, the Jacobian is fixed and is only populated
        # in one entry, so we define that entry and the value here.
        self.declare_partials('height', 'ref_axis', rows=[0], cols=[n_height*3-1], val=1.)
        self.declare_partials('length', 'ref_axis')
        self.declare_partials('s', 'ref_axis')
        
    def compute(self, inputs, outputs):
        # Compute tower height and tower length (a straight tower will be high as long)
        outputs['height'] = inputs['ref_axis'][-1,2]
        myarc = arc_length(inputs['ref_axis'])
        outputs['length'] = myarc[-1]
        
        if myarc[-1] > 0.0:
            outputs['s'] = myarc / myarc[-1]
            
    def compute_partials(self, inputs, partials):
        arc_distances, d_arc_distances_d_points = arc_length_deriv(inputs['ref_axis'])
        
        # The length is based on only the final point in the arc,
        # but that final point has sensitivity to all ref_axis points
        partials['length', 'ref_axis'] = d_arc_distances_d_points[-1, :]
        
        # Do quotient rule to get the non-dimensional grid derivatives
        low_d_high = arc_distances[-1] * d_arc_distances_d_points
        high_d_low = np.outer(arc_distances, d_arc_distances_d_points[-1, :])
        partials['s', 'ref_axis'] = (low_d_high - high_d_low) / arc_distances[-1]**2
            
class Monopile(om.Group):
    
    def initialize(self):
        self.options.declare('monopile_init_options')
        
    def setup(self):
        monopile_init_options = self.options['monopile_init_options']
        n_height           = monopile_init_options['n_height']
        n_layers           = monopile_init_options['n_layers']
        
        ivc = self.add_subsystem('monopile_indep_vars', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('diameter', val=np.zeros(n_height),     units='m',  desc='1D array of the outer diameter values defined along the tower axis.')
        ivc.add_discrete_output('layer_name', val=n_layers * [''],         desc='1D array of the names of the layers modeled in the tower structure.')
        ivc.add_discrete_output('layer_mat',  val=n_layers * [''],         desc='1D array of the names of the materials of each layer modeled in the tower structure.')
        ivc.add_output('layer_thickness',     val=np.zeros((n_layers, n_height-1)), units='m',    desc='2D array of the thickness of the layers of the tower structure. The first dimension represents each layer, the second dimension represents each piecewise-constant entry of the tower sections.')
        ivc.add_output('outfitting_factor',       val = 0.0,             desc='Multiplier that accounts for secondary structure mass inside of tower')
        ivc.add_output('transition_piece_height', val = 0.0, units='m',  desc='point mass height of transition piece above water line')
        ivc.add_output('transition_piece_mass',   val = 0.0, units='kg', desc='point mass of transition piece')
        ivc.add_output('gravity_foundation_mass', val = 0.0, units='kg', desc='extra mass of gravity foundation')
        ivc.add_output('suctionpile_depth',       val = 0.0, units='m',  desc='depth of foundation in the soil')
        ivc.add_output('suctionpile_depth_diam_ratio', 0.0, desc='ratio of sunction pile depth to mudline monopile diameter')
        
        self.add_subsystem('compute_monopile_grid',
            ComputeGrid(init_options=monopile_init_options),
            promotes=['*'])

        
class Floating(om.Group):
    def initialize(self):
        self.options.declare('floating_init_options')

    def setup(self):
        floating_init_options = self.options['floating_init_options']

        ivc = self.add_subsystem('floating_indep_vars', om.IndepVarComp(), promotes=['*'])
        
        ivc.add_output('radius_to_offset_column', 0.0, units='m')
        ivc.add_discrete_output('number_of_offset_columns', 0)
        ivc.add_output('fairlead_location', 0.0)
        ivc.add_output('fairlead_offset_from_shell', 0.0, units='m')
        ivc.add_output('outfitting_cost_rate', 0.0, units='USD/kg')
        ivc.add_discrete_output('loading', 'hydrostatic')
        ivc.add_output('transition_piece_height', val = 0.0, units='m',  desc='point mass height of transition piece above water line')
        ivc.add_output('transition_piece_mass',   val = 0.0, units='kg', desc='point mass of transition piece')

        self.add_subsystem('main_column',   Column(options=floating_init_options['column']['main']))
        self.add_subsystem('offset_column', Column(options=floating_init_options['column']['offset']))
        self.add_subsystem('tower',         Tower(options=floating_init_options['tower']))
        self.add_subsystem('mooring',       Mooring(options=floating_init_options['mooring']))
        
    
        
class Column(om.Group):
    def initialize(self):
        self.options.declare('column_init_options')

    def setup(self):
        column_init_options = self.options['column_init_options']
                           
        ivc = self.add_subsystem('column_indep_vars', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('diameter', val=np.zeros(n_height),     units='m',  desc='1D array of the outer diameter values defined along the column axis.')
        ivc.add_discrete_output('layer_name', val=n_layers * [''],         desc='1D array of the names of the layers modeled in the columnn structure.')
        ivc.add_discrete_output('layer_mat',  val=n_layers * [''],         desc='1D array of the names of the materials of each layer modeled in the column structure.')
        ivc.add_output('layer_thickness',     val=np.zeros((n_layers, n_height-1)), units='m',    desc='2D array of the thickness of the layers of the column structure. The first dimension represents each layer, the second dimension represents each piecewise-constant entry of the column sections.')
        ivc.add_output('freeboard', 0.0, units='m') # Have to add here because cannot promote ivc from Column before needed by tower.  Grr
        ivc.add_output('outfitting_factor',       val = 0.0,             desc='Multiplier that accounts for secondary structure mass inside of column')

        ivc.add_output('stiffener_web_height', np.zeros(n_sect), units='m')
        ivc.add_output('stiffener_web_thickness', np.zeros(n_sect), units='m')
        ivc.add_output('stiffener_flange_width', np.zeros(n_sect), units='m')
        ivc.add_output('stiffener_flange_thickness', np.zeros(n_sect), units='m')
        ivc.add_output('stiffener_spacing', np.zeros(n_sect), units='m')
        ivc.add_output('bulkhead_thickness', np.zeros(n_height), units='m')
        ivc.add_output('permanent_ballast_height', 0.0, units='m')
        ivc.add_output('buoyancy_tank_diameter', 0.0, units='m')
        ivc.add_output('buoyancy_tank_height', 0.0, units='m')
        ivc.add_output('buoyancy_tank_location', 0.0, units='m')
        
        self.add_subsystem('compute_monopile_grid',
            ComputeGrid(init_options=column_init_options),
            promotes=['*'])
    
        
class Mooring(om.Group):
    def initialize(self):
        self.options.declare('mooring_init_options')

    def setup(self):
        mooring_init_options = self.options['mooring_init_options']
                           
        ivc = self.add_subsystem('mooring_indep_vars', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('mooring_line_length', 0.0, units='m')
        ivc.add_output('anchor_radius', 0.0, units='m')
        ivc.add_output('mooring_diameter', 0.0, units='m')
        ivc.add_output('number_of_mooring_connections', 0)
        ivc.add_output('mooring_lines_per_connection', 0)
        ivc.add_discrete_output('mooring_type', 'chain')
        ivc.add_discrete_output('anchor_type', 'SUCTIONPILE')
        ivc.add_output('max_offset', 0.0, units='m')
        ivc.add_output('operational_heel', 0.0, units='deg')
        ivc.add_output('mooring_cost_factor', 0.0)
        ivc.add_output('max_survival_heel', 0.0, units='deg')

        
                           
class ComputeMaterialsProperties(om.ExplicitComponent):
    # Openmdao component with the wind turbine materials coming from the input yaml file. The inputs and outputs are arrays where each entry represents a material
    
    def initialize(self):
        self.options.declare('mat_init_options')
    
    def setup(self):
        
        mat_init_options = self.options['mat_init_options']
        self.n_mat = n_mat = mat_init_options['n_mat']
        
        self.add_discrete_input('name', val=n_mat * [''],                         desc='1D array of names of materials.')
        self.add_discrete_input('component_id', val=-np.ones(n_mat),              desc='1D array of flags to set whether a material is used in a blade: 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.isotropic.')
        self.add_input('rho_fiber',     val=np.zeros(n_mat),      units='kg/m**3',desc='1D array of the density of the fibers of the materials.')
        self.add_input('rho',           val=np.zeros(n_mat),      units='kg/m**3',desc='1D array of the density of the materials. For composites, this is the density of the laminate.')
        self.add_input('rho_area_dry',  val=np.zeros(n_mat),      units='kg/m**2',desc='1D array of the dry aerial density of the composite fabrics. Non-composite materials are kept at 0.')
        self.add_input('ply_t_from_yaml',        val=np.zeros(n_mat),      units='m',      desc='1D array of the ply thicknesses of the materials. Non-composite materials are kept at 0.')
        self.add_input('fvf_from_yaml',          val=np.zeros(n_mat),                      desc='1D array of the non-dimensional fiber volume fraction of the composite materials. Non-composite materials are kept at 0.')
        self.add_input('fwf_from_yaml',          val=np.zeros(n_mat),                      desc='1D array of the non-dimensional fiber weight- fraction of the composite materials. Non-composite materials are kept at 0.')
        
        self.add_output('ply_t',        val=np.zeros(n_mat),      units='m',      desc='1D array of the ply thicknesses of the materials. Non-composite materials are kept at 0.')
        self.add_output('fvf',          val=np.zeros(n_mat),                      desc='1D array of the non-dimensional fiber volume fraction of the composite materials. Non-composite materials are kept at 0.')
        self.add_output('fwf',          val=np.zeros(n_mat),                      desc='1D array of the non-dimensional fiber weight- fraction of the composite materials. Non-composite materials are kept at 0.')
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        density_resin = 0.
        for i in range(self.n_mat):
            if discrete_inputs['name'][i] == 'resin':
                density_resin = inputs['rho'][i]
                id_resin = i
        if density_resin==0.:
            exit('Error: a material named resin must be defined in the input yaml')
        
        fvf   = np.zeros(self.n_mat)
        fwf   = np.zeros(self.n_mat)
        ply_t = np.zeros(self.n_mat)
        
        for i in range(self.n_mat):
            if discrete_inputs['component_id'][i] > 1: # It's a composite
            
                # Formula to estimate the fiber volume fraction fvf from the laminate and the fiber densities
                fvf[i]  = (inputs['rho'][i] - density_resin) / (inputs['rho_fiber'][i] - density_resin) 
                if inputs['fvf_from_yaml'][i] > 0.:
                    if abs(fvf[i] - inputs['fvf_from_yaml'][i]) > 1e-3:
                        exit('Error: the fvf of composite ' + discrete_inputs['name'][i] + ' specified in the yaml is equal to '+ str(inputs['fvf_from_yaml'][i] * 100) + '%, but this value is not compatible to the other values provided. Given the fiber, laminate and resin densities, it should instead be equal to ' + str(fvf[i]*100.) + '%.')
                    else:
                        outputs['fvf'] = inputs['fvf_from_yaml']
                else:
                    outputs['fvf'][i] = fvf[i]
                    
                # Formula to estimate the fiber weight fraction fwf from the fiber volume fraction and the fiber densities
                fwf[i]  = inputs['rho_fiber'][i] * outputs['fvf'][i] / (density_resin + ((inputs['rho_fiber'][i] - density_resin) * outputs['fvf'][i]))
                if inputs['fwf_from_yaml'][i] > 0.:
                    if abs(fwf[i] - inputs['fwf_from_yaml'][i]) > 1e-3:
                        exit('Error: the fwf of composite ' + discrete_inputs['name'][i] + ' specified in the yaml is equal to '+ str(inputs['fwf_from_yaml'][i] * 100) + '%, but this value is not compatible to the other values provided. It should instead be equal to ' + str(fwf[i]*100.) + '%')
                    else:
                        outputs['fwf'] = inputs['fwf_from_yaml']
                else:
                    outputs['fwf'][i] = fwf[i]
                    
                # Formula to estimate the plyt thickness ply_t of a laminate from the aerial density, the laminate density and the fiber weight fraction
                ply_t[i] = inputs['rho_area_dry'][i] / inputs['rho'][i] / outputs['fwf'][i]
                if inputs['ply_t_from_yaml'][i] > 0.:
                    if abs(ply_t[i] - inputs['ply_t_from_yaml'][i]) > 1.e-4:
                        exit('Error: the ply_t of composite ' + discrete_inputs['name'][i] + ' specified in the yaml is equal to '+ str(inputs['ply_t_from_yaml'][i]) + 'm, but this value is not compatible to the other values provided. It should instead be equal to ' + str(ply_t[i]) + 'm. Alternatively, adjust the aerial density to ' + str(outputs['ply_t'][i] * inputs['rho'][i] * outputs['fwf'][i]) + ' kg/m2.')
                    else:
                        outputs['ply_t'] = inputs['ply_t_from_yaml']
                else:
                    outputs['ply_t'][i] = ply_t[i]      
    
class Materials(om.Group):
    # Openmdao group with the wind turbine materials coming from the input yaml file.
    # The inputs and outputs are arrays where each entry represents a material
    
    def initialize(self):
        self.options.declare('mat_init_options')
    
    def setup(self):
        mat_init_options = self.options['mat_init_options']
        self.n_mat = n_mat = mat_init_options['n_mat']
        
        ivc = self.add_subsystem('materials_indep_vars', om.IndepVarComp(), promotes=['*'])
        
        ivc.add_discrete_output('orth', val=np.zeros(n_mat),                      desc='1D array of flags to set whether a material is isotropic (0) or orthtropic (1). Each entry represents a material.')
        ivc.add_output('E',             val=np.zeros([n_mat, 3]), units='Pa',     desc='2D array of the Youngs moduli of the materials. Each row represents a material, the three columns represent E11, E22 and E33.')
        ivc.add_output('G',             val=np.zeros([n_mat, 3]), units='Pa',     desc='2D array of the shear moduli of the materials. Each row represents a material, the three columns represent G12, G13 and G23.')
        ivc.add_output('nu',            val=np.zeros([n_mat, 3]),                 desc='2D array of the Poisson ratio of the materials. Each row represents a material, the three columns represent nu12, nu13 and nu23.')
        ivc.add_output('Xt',            val=np.zeros([n_mat, 3]), units='Pa',     desc='2D array of the Ultimate Tensile Strength (UTS) of the materials. Each row represents a material, the three columns represent Xt12, Xt13 and Xt23.')
        ivc.add_output('Xc',            val=np.zeros([n_mat, 3]), units='Pa',     desc='2D array of the Ultimate Compressive Strength (UCS) of the materials. Each row represents a material, the three columns represent Xc12, Xc13 and Xc23.')
        ivc.add_output('sigma_y',       val=np.zeros(n_mat),      units='Pa',     desc='Yield stress of the material (in the principle direction for composites).')
        ivc.add_output('unit_cost',     val=np.zeros(n_mat),      units='USD/kg', desc='1D array of the unit costs of the materials.')
        ivc.add_output('waste',         val=np.zeros(n_mat),                      desc='1D array of the non-dimensional waste fraction of the materials.')
        ivc.add_output('roll_mass',     val=np.zeros(n_mat),      units='kg',     desc='1D array of the roll mass of the composite fabrics. Non-composite materials are kept at 0.')
        
        ivc.add_discrete_output('name', val=n_mat * [''],                         desc='1D array of names of materials.')
        ivc.add_discrete_output('component_id', val=-np.ones(n_mat),              desc='1D array of flags to set whether a material is used in a blade: 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.isotropic.')
        ivc.add_output('rho_fiber',     val=np.zeros(n_mat),      units='kg/m**3',desc='1D array of the density of the fibers of the materials.')
        ivc.add_output('rho',           val=np.zeros(n_mat),      units='kg/m**3',desc='1D array of the density of the materials. For composites, this is the density of the laminate.')
        ivc.add_output('rho_area_dry',  val=np.zeros(n_mat),      units='kg/m**2',desc='1D array of the dry aerial density of the composite fabrics. Non-composite materials are kept at 0.')
        ivc.add_output('ply_t_from_yaml',        val=np.zeros(n_mat),      units='m',      desc='1D array of the ply thicknesses of the materials. Non-composite materials are kept at 0.')
        ivc.add_output('fvf_from_yaml',          val=np.zeros(n_mat),                      desc='1D array of the non-dimensional fiber volume fraction of the composite materials. Non-composite materials are kept at 0.')
        ivc.add_output('fwf_from_yaml',          val=np.zeros(n_mat),                      desc='1D array of the non-dimensional fiber weight- fraction of the composite materials. Non-composite materials are kept at 0.')
        
        self.add_subsystem('compute_materials_properties', ComputeMaterialsProperties(mat_init_options=mat_init_options), promotes=['*'])


    
class WT_Assembly(om.ExplicitComponent):
    # Openmdao component that computes assembly quantities, such as the rotor coordinate of the blade stations, the hub height, and the blade-tower clearance
    def initialize(self):
        self.options.declare('blade_init_options')

    def setup(self):
        n_span             = self.options['blade_init_options']['n_span']

        self.add_input('blade_ref_axis',        val=np.zeros((n_span,3)),units='m',   desc='2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.')
        self.add_input('hub_radius',            val=0.0, units='m',         desc='Radius of the hub. It defines the distance of the blade root from the rotor center along the coned line.')
        self.add_input('tower_height',          val=0.0,    units='m',      desc='Scalar of the tower height computed along its axis from tower base.')
        self.add_input('foundation_height',     val=0.0,    units='m',      desc='Scalar of the foundation height computed along its axis.')
        self.add_input('distance_tt_hub',       val=0.0,    units='m',      desc='Vertical distance from tower top to hub center.')

        self.add_output('r_blade',              val=np.zeros(n_span), units='m',      desc='1D array of the dimensional spanwise grid defined along the rotor (hub radius to blade tip projected on the plane)')
        self.add_output('rotor_radius',         val=0.0,    units='m',      desc='Scalar of the rotor radius, defined ignoring prebend and sweep curvatures, and cone and uptilt angles.')
        self.add_output('rotor_diameter',       val=0.0,    units='m',      desc='Scalar of the rotor diameter, defined ignoring prebend and sweep curvatures, and cone and uptilt angles.')
        self.add_output('hub_height',           val=0.0,    units='m',      desc='Height of the hub in the global reference system, i.e. distance rotor center to ground.')

    def compute(self, inputs, outputs):
        
        outputs['r_blade']        = inputs['blade_ref_axis'][:,2] + inputs['hub_radius']
        outputs['rotor_radius']   = outputs['r_blade'][-1]
        outputs['rotor_diameter'] = outputs['rotor_radius'] * 2.
        outputs['hub_height']     = inputs['tower_height'] + inputs['distance_tt_hub'] + inputs['foundation_height']

class WindTurbineOntologyOpenMDAO(om.Group):
    # Openmdao group with all wind turbine data
    
    def initialize(self):
        self.options.declare('analysis_options')
        self.options.declare('opt_options')
        
    def setup(self):
        analysis_options = self.options['analysis_options']
        opt_options      = self.options['opt_options']
        
        # Material dictionary inputs
        self.add_subsystem('materials', Materials(mat_init_options = analysis_options['materials']))
        
        # Airfoil dictionary inputs
        airfoils    = om.IndepVarComp()
        af_init_options = analysis_options['airfoils']
        n_af            = af_init_options['n_af'] # Number of airfoils
        n_aoa           = af_init_options['n_aoa']# Number of angle of attacks
        n_Re            = af_init_options['n_Re'] # Number of Reynolds, so far hard set at 1
        n_tab           = af_init_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        n_xy            = af_init_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
        airfoils.add_discrete_output('name', val=n_af * [''],                        desc='1D array of names of airfoils.')
        airfoils.add_output('ac',        val=np.zeros(n_af),                         desc='1D array of the aerodynamic centers of each airfoil.')
        airfoils.add_output('r_thick',   val=np.zeros(n_af),                         desc='1D array of the relative thicknesses of each airfoil.')
        airfoils.add_output('aoa',       val=np.zeros(n_aoa),        units='rad',    desc='1D array of the angles of attack used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.')
        airfoils.add_output('Re',        val=np.zeros(n_Re),                         desc='1D array of the Reynolds numbers used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.')
        airfoils.add_output('tab',       val=np.zeros(n_tab),                        desc='1D array of the values of the "tab" entity used to define the polars of the airfoils. All airfoils defined in openmdao share this grid. The tab could for example represent a flap deflection angle.')
        airfoils.add_output('cl',        val=np.zeros((n_af, n_aoa, n_Re, n_tab)),   desc='4D array with the lift coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        airfoils.add_output('cd',        val=np.zeros((n_af, n_aoa, n_Re, n_tab)),   desc='4D array with the drag coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        airfoils.add_output('cm',        val=np.zeros((n_af, n_aoa, n_Re, n_tab)),   desc='4D array with the moment coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        # Airfoil coordinates
        airfoils.add_output('coord_xy',  val=np.zeros((n_af, n_xy, 2)),              desc='3D array of the x and y airfoil coordinates of the n_af airfoils.')
        self.add_subsystem('airfoils', airfoils)
        
        # Blade inputs and connections from airfoils
        self.add_subsystem('blade',         Blade(blade_init_options   = analysis_options['blade'], af_init_options   = analysis_options['airfoils'], opt_options = opt_options))
        self.connect('airfoils.name',    'blade.interp_airfoils.name')
        self.connect('airfoils.r_thick', 'blade.interp_airfoils.r_thick')
        self.connect('airfoils.coord_xy','blade.interp_airfoils.coord_xy')
        self.connect('airfoils.aoa',     'blade.interp_airfoils.aoa')
        self.connect('airfoils.cl',      'blade.interp_airfoils.cl')
        self.connect('airfoils.cd',      'blade.interp_airfoils.cd')
        self.connect('airfoils.cm',      'blade.interp_airfoils.cm')
        
        # Hub inputs
        self.add_subsystem('hub',           Hub())
        
        # Nacelle inputs
        nacelle = om.IndepVarComp()
        # Outer shape bem
        nacelle.add_output('uptilt',           val=0.0, units='rad',   desc='Nacelle uptilt angle. A standard machine has positive values.')
        nacelle.add_output('distance_tt_hub',  val=0.0, units='m',     desc='Vertical distance from tower top to hub center.')
        nacelle.add_output('overhang',         val=0.0, units='m',     desc='Horizontal distance from tower top to hub center.')
        # Mulit-body properties
        nacelle.add_output('above_yaw_mass',   val=0.0, units='kg', desc='Mass of the nacelle above the yaw system')
        nacelle.add_output('yaw_mass',         val=0.0, units='kg', desc='Mass of yaw system')
        nacelle.add_output('nacelle_cm',       val=np.zeros(3), units='m', desc='Center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        nacelle.add_output('nacelle_I',        val=np.zeros(6), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        # Drivetrain parameters
        nacelle.add_output('gear_ratio',       val=0.0)
        nacelle.add_output('shaft_ratio',      val=0.0)
        nacelle.add_discrete_output('planet_numbers',   val=np.zeros(3))
        nacelle.add_output('shrink_disc_mass', val=0.0, units='kg')
        nacelle.add_output('carrier_mass',     val=0.0, units='kg')
        nacelle.add_output('flange_length',    val=0.0, units='m')
        nacelle.add_output('gearbox_input_xcm',val=0.0, units='m')
        nacelle.add_output('hss_input_length', val=0.0, units='m')
        nacelle.add_output('distance_hub2mb',  val=0.0, units='m')
        nacelle.add_discrete_output('yaw_motors_number', val = 0)
        nacelle.add_output('drivetrain_eff',   val=0.0)
        self.add_subsystem('nacelle', nacelle)
        
        # Tower inputs
        self.add_subsystem('tower',         Tower(tower_init_options   = analysis_options['tower']))
        if analysis_options['tower']['monopile']:
            self.add_subsystem('monopile',  Monopile(monopile_init_options   = analysis_options['monopile']))
        
        # Foundation inputs
        foundation_ivc = self.add_subsystem('foundation', om.IndepVarComp())
        foundation_ivc.add_output('height',     val=0.0, units='m',     desc='Foundation height in respect to the ground level.')

        # Control inputs
        ctrl_ivc = self.add_subsystem('control', om.IndepVarComp())
        ctrl_ivc.add_output('rated_power',      val=0.0, units='W',         desc='Electrical rated power of the generator.')
        ctrl_ivc.add_output('V_in',             val=0.0, units='m/s',       desc='Cut in wind speed. This is the wind speed where region II begins.')
        ctrl_ivc.add_output('V_out',            val=0.0, units='m/s',       desc='Cut out wind speed. This is the wind speed where region III ends.')
        ctrl_ivc.add_output('minOmega',         val=0.0, units='rad/s',     desc='Minimum allowed rotor speed.')
        ctrl_ivc.add_output('maxOmega',         val=0.0, units='rad/s',     desc='Maximum allowed rotor speed.')
        ctrl_ivc.add_output('max_TS',           val=0.0, units='m/s',       desc='Maximum allowed blade tip speed.')
        ctrl_ivc.add_output('max_pitch_rate',   val=0.0, units='rad/s',     desc='Maximum allowed blade pitch rate')
        ctrl_ivc.add_output('max_torque_rate',  val=0.0, units='N*m/s',     desc='Maximum allowed generator torque rate')
        ctrl_ivc.add_output('rated_TSR',        val=0.0,                    desc='Constant tip speed ratio in region II.')
        ctrl_ivc.add_output('rated_pitch',      val=0.0, units='rad',       desc='Constant pitch angle in region II.')
        ctrl_ivc.add_output('PC_omega',         val=0.0, units='rad/s',     desc='Pitch controller natural frequency')
        ctrl_ivc.add_output('PC_zeta',          val=0.0,                    desc='Pitch controller damping ratio')
        ctrl_ivc.add_output('VS_omega',         val=0.0, units='rad/s',     desc='Generator torque controller natural frequency')
        ctrl_ivc.add_output('VS_zeta',          val=0.0,                    desc='Generator torque controller damping ratio')
        ctrl_ivc.add_output('Flp_omega',        val=0.0, units='rad/s',     desc='Flap controller natural frequency')
        ctrl_ivc.add_output('Flp_zeta',         val=0.0,                    desc='Flap controller damping ratio')
        # optional inputs - not connected right now!!
        ctrl_ivc.add_output('max_pitch',        val=0.0, units='rad',       desc='Maximum pitch angle , {default = 90 degrees}')
        ctrl_ivc.add_output('min_pitch',        val=0.0, units='rad',       desc='Minimum pitch angle [rad], {default = 0 degrees}')
        ctrl_ivc.add_output('vs_minspd',        val=0.0, units='rad/s',     desc='Minimum rotor speed [rad/s], {default = 0 rad/s}')
        ctrl_ivc.add_output('ss_cornerfreq',    val=0.0, units='rad/s',     desc='First order low-pass filter cornering frequency for setpoint smoother [rad/s]')
        ctrl_ivc.add_output('ss_vsgain',        val=0.0,                    desc='Torque controller setpoint smoother gain bias percentage [%, <= 1 ], {default = 100%}')
        ctrl_ivc.add_output('ss_pcgain',        val=0.0,                    desc='Pitch controller setpoint smoother gain bias percentage  [%, <= 1 ], {default = 0.1%}')
        ctrl_ivc.add_output('ps_percent',       val=0.0,                    desc='Percent peak shaving  [%, <= 1 ], {default = 80%}')
        ctrl_ivc.add_output('sd_maxpit',        val=0.0, units='rad',       desc='Maximum blade pitch angle to initiate shutdown [rad], {default = bld pitch at v_max}')
        ctrl_ivc.add_output('sd_cornerfreq',    val=0.0, units='rad/s',     desc='Cutoff Frequency for first order low-pass filter for blade pitch angle [rad/s], {default = 0.41888 ~ time constant of 15s}')
        ctrl_ivc.add_output('Kp_flap',          val=0.0, units='s',         desc='Proportional term of the PI controller for the trailing-edge flaps')
        ctrl_ivc.add_output('Ki_flap',          val=0.0,                    desc='Integral term of the PI controller for the trailing-edge flaps')

        # Wind turbine configuration inputs
        conf_ivc = self.add_subsystem('configuration', om.IndepVarComp())
        conf_ivc.add_discrete_output('ws_class',            val='',         desc='IEC wind turbine class. I - offshore, II coastal, III - land-based, IV - low wind speed site.')
        conf_ivc.add_discrete_output('turb_class',          val='',         desc='IEC wind turbine category. A - high turbulence intensity (land-based), B - mid turbulence, C - low turbulence (offshore).')
        conf_ivc.add_discrete_output('gearbox_type',        val='geared',   desc='Gearbox configuration (geared, direct-drive, etc.).')
        conf_ivc.add_discrete_output('rotor_orientation',   val='upwind',   desc='Rotor orientation, either upwind or downwind.')
        conf_ivc.add_discrete_output('n_blades',            val=3,          desc='Number of blades of the rotor.')

        # Environment inputs
        env_ivc = self.add_subsystem('env', om.IndepVarComp())
        env_ivc.add_output('rho_air',      val=1.225,       units='kg/m**3',    desc='Density of air')
        env_ivc.add_output('mu_air',       val=1.81e-5,     units='kg/(m*s)',   desc='Dynamic viscosity of air')
        env_ivc.add_output('weibull_k',    val=2.0,                             desc='Shape parameter of the Weibull probability density function of the wind.')
        env_ivc.add_output('shear_exp',    val=0.2,                             desc='Shear exponent of the wind.')
        env_ivc.add_output('speed_sound_air',  val=340.,    units='m/s',        desc='Speed of sound in air.')
        env_ivc.add_output('rho_water',    val=1025.,       units='kg/m**3',    desc='Density of ocean water')
        env_ivc.add_output('mu_water',     val=1.3351e-3,   units='kg/(m*s)',   desc='Dynamic viscosity of ocean water')
        env_ivc.add_output('G_soil',       val=140e6,       units='N/m**2',     desc='Shear stress of soil')
        env_ivc.add_output('nu_soil',      val=0.4,                             desc='Poisson ratio of soil')

        # Cost analysis inputs
        costs_ivc = self.add_subsystem('costs', om.IndepVarComp())
        costs_ivc.add_discrete_output('turbine_number',    val=0,             desc='Number of turbines at plant')
        costs_ivc.add_output('bos_per_kW',        val=0.0, units='USD/kW',    desc='Balance of system costs of the turbine')
        costs_ivc.add_output('offset_tcc_per_kW' ,val=0.0, units='USD/kW',    desc='Offset to turbine capital cost')
        costs_ivc.add_output('opex_per_kW',       val=0.0, units='USD/kW/yr', desc='Average annual operational expenditures of the turbine')
        costs_ivc.add_output('wake_loss_factor',  val=0.0,                    desc='The losses in AEP due to waked conditions')
        costs_ivc.add_output('fixed_charge_rate', val=0.0,                    desc = 'Fixed charge rate for coe calculation')
        costs_ivc.add_output('labor_rate', 0.0, units='USD/hr')
        costs_ivc.add_output('painting_rate', 0.0, units='USD/m**2')
        
        # Assembly setup
        self.add_subsystem('assembly',      WT_Assembly(blade_init_options   = analysis_options['blade']))
        self.connect('blade.outer_shape_bem.ref_axis',  'assembly.blade_ref_axis')
        self.connect('hub.radius',                      'assembly.hub_radius')
        self.connect('tower.height',                    'assembly.tower_height')
        self.connect('foundation.height',               'assembly.foundation_height')
        self.connect('nacelle.distance_tt_hub',         'assembly.distance_tt_hub')

        # Setup TSR optimization
        opt_var = self.add_subsystem('opt_var', om.IndepVarComp())
        opt_var.add_output('tsr_opt_gain',   val = 1.0)
        # Multiply the initial tsr with the tsr gain
        exec_comp = om.ExecComp('tsr_opt = tsr_original * tsr_gain')
        self.add_subsystem('pc', exec_comp)
        self.connect('opt_var.tsr_opt_gain', 'pc.tsr_gain')
        self.connect('control.rated_TSR',    'pc.tsr_original')

def yaml2openmdao(wt_opt, analysis_options, wt_init):
    # Function to assign values to the openmdao group Wind_Turbine and all its components
    
    blade           = wt_init['components']['blade']
    hub             = wt_init['components']['hub']
    nacelle         = wt_init['components']['nacelle']
    tower           = wt_init['components']['tower']
    foundation      = wt_init['components']['foundation']
    control         = wt_init['control']
    assembly        = wt_init['assembly']
    environment     = wt_init['environment']
    costs           = wt_init['costs']
    airfoils        = wt_init['airfoils']
    materials       = wt_init['materials']
    
    wt_opt = assign_blade_values(wt_opt, analysis_options, blade)
    wt_opt = assign_hub_values(wt_opt, hub)
    wt_opt = assign_nacelle_values(wt_opt, assembly, nacelle)
    wt_opt = assign_tower_values(wt_opt, analysis_options, tower)
    
    if analysis_options['tower']['monopile']:
        monopile = wt_init['components']['monopile']
        wt_opt   = assign_monopile_values(wt_opt, analysis_options, monopile)
        
    if analysis_options['floating']['flag']:
        floating = wt_init['components']['floating']
        wt_opt   = assign_floating_values(wt_opt, analysis_options, floating)
    else:
        wt_opt = assign_foundation_values(wt_opt, foundation)
        
    wt_opt = assign_control_values(wt_opt, analysis_options, control)
    wt_opt = assign_configuration_values(wt_opt, assembly)
    wt_opt = assign_environment_values(wt_opt, environment)
    wt_opt = assign_costs_values(wt_opt, costs)
    wt_opt = assign_airfoil_values(wt_opt, analysis_options, airfoils)
    wt_opt = assign_material_values(wt_opt, analysis_options, materials)

    return wt_opt
    
def assign_blade_values(wt_opt, analysis_options, blade):
    # Function to assign values to the openmdao group Blade
    wt_opt = assign_outer_shape_bem_values(wt_opt, analysis_options, blade['outer_shape_bem'])
    wt_opt = assign_internal_structure_2d_fem_values(wt_opt, analysis_options, blade['internal_structure_2d_fem'])
    wt_opt = assign_te_flaps_values(wt_opt, analysis_options, blade)
    
    return wt_opt
    
def assign_outer_shape_bem_values(wt_opt, analysis_options, outer_shape_bem):
    # Function to assign values to the openmdao component Blade_Outer_Shape_BEM
    
    nd_span     = analysis_options['blade']['nd_span']
    
    wt_opt['blade.outer_shape_bem.af_position'] = outer_shape_bem['airfoil_position']['grid']
    wt_opt['blade.opt_var.af_position']         = outer_shape_bem['airfoil_position']['grid']
    
    wt_opt['blade.outer_shape_bem.s_default']        = nd_span
    wt_opt['blade.outer_shape_bem.chord_yaml']       = np.interp(nd_span, outer_shape_bem['chord']['grid'], outer_shape_bem['chord']['values'])
    wt_opt['blade.outer_shape_bem.twist_yaml']       = np.interp(nd_span, outer_shape_bem['twist']['grid'], outer_shape_bem['twist']['values'])
    wt_opt['blade.outer_shape_bem.pitch_axis_yaml']  = np.interp(nd_span, outer_shape_bem['pitch_axis']['grid'], outer_shape_bem['pitch_axis']['values'])
    
    wt_opt['blade.outer_shape_bem.ref_axis_yaml'][:,0]  = np.interp(nd_span, outer_shape_bem['reference_axis']['x']['grid'], outer_shape_bem['reference_axis']['x']['values'])
    wt_opt['blade.outer_shape_bem.ref_axis_yaml'][:,1]  = np.interp(nd_span, outer_shape_bem['reference_axis']['y']['grid'], outer_shape_bem['reference_axis']['y']['values'])
    wt_opt['blade.outer_shape_bem.ref_axis_yaml'][:,2]  = np.interp(nd_span, outer_shape_bem['reference_axis']['z']['grid'], outer_shape_bem['reference_axis']['z']['values'])

    # # Smoothing of the shapes
    # # Chord
    # chord_init      = wt_opt['blade.outer_shape_bem.chord']
    # s_interp_c      = np.array([0.0, 0.05, 0.2, 0.35, 0.65, 0.9, 1.0 ])
    # f_interp1       = interp1d(nd_span,chord_init)
    # chord_int1      = f_interp1(s_interp_c)
    # f_interp2       = PchipInterpolator(s_interp_c,chord_int1)
    # chord_int2      = f_interp2(nd_span)
    
    # import matplotlib.pyplot as plt
    # fc, axc  = plt.subplots(1,1,figsize=(5.3, 4))
    # axc.plot(nd_span, chord_init, c='k', label='Initial')
    # axc.plot(s_interp_c, chord_int1, 'ko', label='Interp Points')
    # axc.plot(nd_span, chord_int2, c='b', label='PCHIP')
    # axc.set(xlabel='r/R' , ylabel='Chord (m)')
    # fig_name = 'interp_chord.png'
    # axc.legend()
    # # Planform
    # le_init = wt_opt['blade.outer_shape_bem.pitch_axis']*wt_opt['blade.outer_shape_bem.chord']
    # te_init = (1. - wt_opt['blade.outer_shape_bem.pitch_axis'])*wt_opt['blade.outer_shape_bem.chord']
    
    # s_interp_le     = np.array([0.0, 0.5, 0.8, 1.0])
    # f_interp1       = interp1d(wt_opt['blade.outer_shape_bem.s_default'],le_init)
    # le_int1         = f_interp1(s_interp_le)
    # f_interp2       = PchipInterpolator(s_interp_le,le_int1)
    # le_int2         = f_interp2(wt_opt['blade.outer_shape_bem.s_default'])
    
    # fpl, axpl  = plt.subplots(1,1,figsize=(5.3, 4))
    # axpl.plot(wt_opt['blade.outer_shape_bem.s_default'], -le_init, c='k', label='LE init')
    # axpl.plot(wt_opt['blade.outer_shape_bem.s_default'], -le_int2, c='b', label='LE smooth old pa')
    # axpl.plot(wt_opt['blade.outer_shape_bem.s_default'], te_init, c='k', label='TE init')
    # axpl.plot(wt_opt['blade.outer_shape_bem.s_default'], wt_opt['blade.outer_shape_bem.chord'] - le_int2, c='b', label='TE smooth old pa')
    # axpl.set(xlabel='r/R' , ylabel='Planform (m)')
    # axpl.legend()
    # plt.show()
    # # np.savetxt('temp.txt', le_int2/wt_opt['blade.outer_shape_bem.chord'])
    # exit()

    # # # Twist
    # theta_init      = wt_opt['blade.outer_shape_bem.twist']
    # s_interp      = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0 ])
    # f_interp1       = interp1d(nd_span,theta_init)
    # theta_int1      = f_interp1(s_interp)
    # f_interp2       = PchipInterpolator(s_interp,theta_int1)
    # theta_int2      = f_interp2(nd_span)
    
    # import matplotlib.pyplot as plt
    # fc, axc  = plt.subplots(1,1,figsize=(5.3, 4))
    # axc.plot(nd_span, theta_init, c='k', label='Initial')
    # axc.plot(s_interp, theta_int1, 'ko', label='Interp Points')
    # axc.plot(nd_span, theta_int2, c='b', label='PCHIP')
    # axc.set(xlabel='r/R' , ylabel='Twist (deg)')
    # axc.legend()
    # plt.show()
    # exit()
    
    return wt_opt
    
def assign_internal_structure_2d_fem_values(wt_opt, analysis_options, internal_structure_2d_fem):
    # Function to assign values to the openmdao component Blade_Internal_Structure_2D_FEM
    
    n_span          = analysis_options['blade']['n_span']
    n_webs          = analysis_options['blade']['n_webs']
    
    web_rotation    = np.zeros((n_webs, n_span))
    web_offset_y_pa = np.zeros((n_webs, n_span))
    web_start_nd    = np.zeros((n_webs, n_span))
    web_end_nd      = np.zeros((n_webs, n_span))
    definition_web  = np.zeros(n_webs)
    nd_span         = wt_opt['blade.outer_shape_bem.s_default']
    
    # Loop through the webs and interpolate spanwise the values
    for i in range(n_webs):
        if 'rotation' in internal_structure_2d_fem['webs'][i] and 'offset_y_pa' in internal_structure_2d_fem['webs'][i]:
            if 'fixed' in internal_structure_2d_fem['webs'][i]['rotation'].keys():
                if internal_structure_2d_fem['webs'][i]['rotation']['fixed'] == 'twist':
                    definition_web[i] = 1
                else:
                    exit('Invalid rotation reference for web ' + self.analysis_options['blade']['web_name'][i] + '. Please check the yaml input file')
            else:
                web_rotation[i,:] = np.interp(nd_span, internal_structure_2d_fem['webs'][i]['rotation']['grid'], internal_structure_2d_fem['webs'][i]['rotation']['values'], left=0., right=0.)
                definition_web[i] = 2
            web_offset_y_pa[i,:] = np.interp(nd_span, internal_structure_2d_fem['webs'][i]['offset_y_pa']['grid'], internal_structure_2d_fem['webs'][i]['offset_y_pa']['values'], left=0., right=0.)
        elif 'start_nd_arc' in internal_structure_2d_fem['webs'][i] and 'end_nd_arc' in internal_structure_2d_fem['webs'][i]:
            definition_web[i] = 3
            web_start_nd[i,:] = np.interp(nd_span, internal_structure_2d_fem['webs'][i]['start_nd_arc']['grid'], internal_structure_2d_fem['webs'][i]['start_nd_arc']['values'], left=0., right=0.)
            web_end_nd[i,:]   = np.interp(nd_span, internal_structure_2d_fem['webs'][i]['end_nd_arc']['grid'], internal_structure_2d_fem['webs'][i]['end_nd_arc']['values'], left=0., right=0.)
        else:
            exit('Webs definition not supported. Please check the yaml input.')
    
    n_layers        = analysis_options['blade']['n_layers']
    layer_name      = n_layers * ['']
    layer_mat       = n_layers * ['']
    thickness       = np.zeros((n_layers, n_span))
    fiber_orient    = np.zeros((n_layers, n_span))
    layer_rotation  = np.zeros((n_layers, n_span))
    layer_offset_y_pa  = np.zeros((n_layers, n_span))
    layer_width     = np.zeros((n_layers, n_span))
    layer_midpoint_nd  = np.zeros((n_layers, n_span))
    layer_start_nd  = np.zeros((n_layers, n_span))
    layer_end_nd    = np.zeros((n_layers, n_span))
    layer_web       = np.zeros(n_layers)
    layer_side      = n_layers * ['']
    definition_layer= np.zeros(n_layers)
    index_layer_start= np.zeros(n_layers)
    index_layer_end = np.zeros(n_layers)

    
    # Loop through the layers, interpolate along blade span, assign the inputs, and the definition flag
    for i in range(n_layers):
        layer_name[i]  = analysis_options['blade']['layer_name'][i]
        layer_mat[i]   = analysis_options['blade']['layer_mat'][i]
        thickness[i]   = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['thickness']['grid'], internal_structure_2d_fem['layers'][i]['thickness']['values'], left=0., right=0.)
        if 'rotation' not in internal_structure_2d_fem['layers'][i] and 'offset_y_pa' not in internal_structure_2d_fem['layers'][i] and 'width' not in internal_structure_2d_fem['layers'][i] and 'start_nd_arc' not in internal_structure_2d_fem['layers'][i] and 'end_nd_arc' not in internal_structure_2d_fem['layers'][i] and 'web' not in internal_structure_2d_fem['layers'][i]:
            definition_layer[i] = 1
            
        if 'rotation' in internal_structure_2d_fem['layers'][i] and 'offset_y_pa' in internal_structure_2d_fem['layers'][i] and 'width' in internal_structure_2d_fem['layers'][i] and 'side' in internal_structure_2d_fem['layers'][i]:
            if 'fixed' in internal_structure_2d_fem['layers'][i]['rotation'].keys():
                if internal_structure_2d_fem['layers'][i]['rotation']['fixed'] == 'twist':
                    definition_layer[i] = 2
                else:
                    exit('Invalid rotation reference for layer ' + layer_name[i] + '. Please check the yaml input file.')
            else:
                layer_rotation[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['rotation']['grid'], internal_structure_2d_fem['layers'][i]['rotation']['values'], left=0., right=0.)
                definition_layer[i] = 3
            layer_offset_y_pa[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['offset_y_pa']['grid'], internal_structure_2d_fem['layers'][i]['offset_y_pa']['values'], left=0., right=0.)
            layer_width[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['width']['grid'], internal_structure_2d_fem['layers'][i]['width']['values'], left=0., right=0.)
            layer_side[i]    = internal_structure_2d_fem['layers'][i]['side']
        if 'midpoint_nd_arc' in internal_structure_2d_fem['layers'][i] and 'width' in internal_structure_2d_fem['layers'][i]:
            if 'fixed' in internal_structure_2d_fem['layers'][i]['midpoint_nd_arc'].keys():
                if internal_structure_2d_fem['layers'][i]['midpoint_nd_arc']['fixed'] == 'TE':
                    layer_midpoint_nd[i,:] = np.ones(n_span)
                    definition_layer[i] = 4
                elif internal_structure_2d_fem['layers'][i]['midpoint_nd_arc']['fixed'] == 'LE':
                    definition_layer[i] = 5
                    # layer_midpoint_nd[i,:] = -np.ones(n_span) # To be assigned later!
            else:
                layer_midpoint_nd[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['midpoint_nd_arc']['grid'], internal_structure_2d_fem['layers'][i]['midpoint_nd_arc']['values'], left=0., right=0.)
            layer_width[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['width']['grid'], internal_structure_2d_fem['layers'][i]['width']['values'], left=0., right=0.)
        if 'start_nd_arc' in internal_structure_2d_fem['layers'][i] and definition_layer[i] == 0:
            if 'fixed' in internal_structure_2d_fem['layers'][i]['start_nd_arc'].keys():
                if internal_structure_2d_fem['layers'][i]['start_nd_arc']['fixed'] == 'TE':
                    layer_start_nd[i,:] = np.zeros(n_span)
                    # exit('No need to fix element to TE, set it to 0.')
                elif internal_structure_2d_fem['layers'][i]['start_nd_arc']['fixed'] == 'LE':
                    definition_layer[i] = 11
                else:
                    definition_layer[i] = 6
                    flag = False
                    for k in range(n_layers):
                        if layer_name[k] == internal_structure_2d_fem['layers'][i]['start_nd_arc']['fixed']:
                            index_layer_start[i] = k
                            flag = True
                            break
                    if flag == False:
                        exit('Error with layer ' + internal_structure_2d_fem['layers'][i]['name'])
            else:
                layer_start_nd[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['start_nd_arc']['grid'], internal_structure_2d_fem['layers'][i]['start_nd_arc']['values'], left=0., right=0.)
            if 'end_nd_arc' in internal_structure_2d_fem['layers'][i]:
                if 'fixed' in internal_structure_2d_fem['layers'][i]['end_nd_arc'].keys():
                    if internal_structure_2d_fem['layers'][i]['end_nd_arc']['fixed'] == 'TE':
                        layer_end_nd[i,:] = np.ones(n_span)
                        # exit('No need to fix element to TE, set it to 0.')
                    elif internal_structure_2d_fem['layers'][i]['end_nd_arc']['fixed'] == 'LE':
                        definition_layer[i] = 12
                    else:
                        flag = False
                        for k in range(n_layers):
                            if layer_name[k] == internal_structure_2d_fem['layers'][i]['end_nd_arc']['fixed']:
                                index_layer_end[i] = k
                                flag = True
                                break
                        if flag == False:
                            exit('Error with layer ' + internal_structure_2d_fem['layers'][i]['name'])
            if 'width' in internal_structure_2d_fem['layers'][i]:
                definition_layer[i] = 7
                layer_width[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['width']['grid'], internal_structure_2d_fem['layers'][i]['width']['values'], left=0., right=0.)
            
        if 'end_nd_arc' in internal_structure_2d_fem['layers'][i] and definition_layer[i] == 0:
            if 'fixed' in internal_structure_2d_fem['layers'][i]['end_nd_arc'].keys():
                if internal_structure_2d_fem['layers'][i]['end_nd_arc']['fixed'] == 'TE':
                    layer_end_nd[i,:] = np.ones(n_span)
                    # exit('No need to fix element to TE, set it to 0.')
                elif internal_structure_2d_fem['layers'][i]['end_nd_arc']['fixed'] == 'LE':
                    definition_layer[i] = 12
                else:
                    definition_layer[i] = 6
                    flag = False
                    if layer_name[k] == internal_structure_2d_fem['layers'][i]['end_nd_arc']['fixed']:
                        index_layer_end[i] = k
                        flag = True
                        break
                    if flag == False:
                        exit('Error with layer ' + internal_structure_2d_fem['layers'][i]['name'])
            else:
                layer_end_nd[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['end_nd_arc']['grid'], internal_structure_2d_fem['layers'][i]['end_nd_arc']['values'], left=0., right=0.)
            if 'width' in internal_structure_2d_fem['layers'][i]:
                definition_layer[i] = 8
                layer_width[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['width']['grid'], internal_structure_2d_fem['layers'][i]['width']['values'], left=0., right=0.)
            if 'start_nd_arc' in internal_structure_2d_fem['layers'][i]:
                definition_layer[i] = 9

        if 'web' in internal_structure_2d_fem['layers'][i]:
            web_name_i = internal_structure_2d_fem['layers'][i]['web']
            for j in range(analysis_options['blade']['n_webs']):
                if web_name_i == analysis_options['blade']['web_name'][j]:
                    k = j+1
                    break
            layer_web[i] = k
            definition_layer[i] = 10
    
    
    # Assign the openmdao values
    wt_opt['blade.internal_structure_2d_fem.layer_side']            = layer_side
    wt_opt['blade.internal_structure_2d_fem.layer_thickness']       = thickness
    wt_opt['blade.internal_structure_2d_fem.layer_midpoint_nd']     = layer_midpoint_nd
    wt_opt['blade.internal_structure_2d_fem.layer_web']             = layer_web
    wt_opt['blade.internal_structure_2d_fem.definition_web']        = definition_web
    wt_opt['blade.internal_structure_2d_fem.definition_layer']      = definition_layer
    wt_opt['blade.internal_structure_2d_fem.index_layer_start']     = index_layer_start
    wt_opt['blade.internal_structure_2d_fem.index_layer_end']       = index_layer_end

    wt_opt['blade.internal_structure_2d_fem.web_offset_y_pa_yaml']  = web_offset_y_pa
    wt_opt['blade.internal_structure_2d_fem.web_rotation_yaml']     = web_rotation
    wt_opt['blade.internal_structure_2d_fem.web_start_nd_yaml']     = web_start_nd
    wt_opt['blade.internal_structure_2d_fem.web_end_nd_yaml']       = web_end_nd
    wt_opt['blade.internal_structure_2d_fem.layer_offset_y_pa_yaml']= layer_offset_y_pa
    wt_opt['blade.internal_structure_2d_fem.layer_width_yaml']      = layer_width
    wt_opt['blade.internal_structure_2d_fem.layer_start_nd_yaml']   = layer_start_nd
    wt_opt['blade.internal_structure_2d_fem.layer_end_nd_yaml']     = layer_end_nd
    wt_opt['blade.internal_structure_2d_fem.layer_rotation_yaml']   = layer_rotation
    
    return wt_opt

def assign_te_flaps_values(wt_opt, analysis_options, blade):
    # Function to assign the trailing edge flaps data to the openmdao data structure
    if analysis_options['blade']['n_te_flaps'] > 0:   
        n_te_flaps = analysis_options['blade']['n_te_flaps']
        for i in range(n_te_flaps):
            wt_opt['blade.dac_te_flaps.te_flap_start'][i]   = blade['aerodynamic_control']['te_flaps'][i]['span_start']
            wt_opt['blade.dac_te_flaps.te_flap_end'][i]     = blade['aerodynamic_control']['te_flaps'][i]['span_end']
            wt_opt['blade.dac_te_flaps.chord_start'][i]     = blade['aerodynamic_control']['te_flaps'][i]['chord_start']
            wt_opt['blade.dac_te_flaps.delta_max_pos'][i]   = blade['aerodynamic_control']['te_flaps'][i]['delta_max_pos']
            wt_opt['blade.dac_te_flaps.delta_max_neg'][i]   = blade['aerodynamic_control']['te_flaps'][i]['delta_max_neg']

            wt_opt['blade.opt_var.te_flap_ext'] = blade['aerodynamic_control']['te_flaps'][i]['span_end'] - blade['aerodynamic_control']['te_flaps'][i]['span_start']
            wt_opt['blade.opt_var.te_flap_end'] = blade['aerodynamic_control']['te_flaps'][i]['span_end']

            # Checks for consistency
            if blade['aerodynamic_control']['te_flaps'][i]['span_start'] < 0.:
                exit('Error: the start along blade span of the trailing edge flap number ' + str(i) + ' is defined smaller than 0, which corresponds to blade root. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['span_start'] > 1.:
                exit('Error: the start along blade span of the trailing edge flap number ' + str(i) + ' is defined bigger than 1, which corresponds to blade tip. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['span_end'] < 0.:
                exit('Error: the end along blade span of the trailing edge flap number ' + str(i) + ' is defined smaller than 0, which corresponds to blade root. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['span_end'] > 1.:
                exit('Error: the end along blade span of the trailing edge flap number ' + str(i) + ' is defined bigger than 1, which corresponds to blade tip. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['span_start'] == blade['aerodynamic_control']['te_flaps'][i]['span_end']:
                exit('Error: the start and end along blade span of the trailing edge flap number ' + str(i) + ' are defined equal. Please check the yaml input.')
            elif i > 0:
                 if blade['aerodynamic_control']['te_flaps'][i]['span_start'] < blade['aerodynamic_control']['te_flaps'][i-1]['span_end']:
                     exit('Error: the start along blade span of the trailing edge flap number ' + str(i) + ' is smaller than the end of the trailing edge flap number ' + str(i-1) + '. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['chord_start'] < 0.2:
                exit('Error: the start along the chord of the trailing edge flap number ' + str(i) + ' is smaller than 0.2, which is too close to the leading edge. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['chord_start'] > 1.:
                exit('Error: the end along the chord of the trailing edge flap number ' + str(i) + ' is larger than 1., which is beyond the trailing edge. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['delta_max_pos'] > 30. / 180. * np.pi:
                exit('Error: the max positive deflection of the trailing edge flap number ' + str(i) + ' is larger than 30 deg, which is beyond the limits of applicability of this tool. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['delta_max_neg'] < -30. / 180. * np.pi:
                exit('Error: the max negative deflection of the trailing edge flap number ' + str(i) + ' is smaller than -30 deg, which is beyond the limits of applicability of this tool. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['delta_max_pos'] < blade['aerodynamic_control']['te_flaps'][i]['delta_max_neg']:
                exit('Error: the max positive deflection of the trailing edge flap number ' + str(i) + ' is smaller than the max negative deflection. Please check the yaml input.')
            else:
                pass

    return wt_opt

def assign_hub_values(wt_opt, hub):

    wt_opt['hub.diameter']    = hub['outer_shape_bem']['diameter']
    wt_opt['hub.cone']        = hub['outer_shape_bem']['cone_angle']
    wt_opt['hub.drag_coeff']  = hub['outer_shape_bem']['drag_coefficient']

    wt_opt['hub.system_mass'] = hub['elastic_properties_mb']['system_mass']
    wt_opt['hub.system_I']    = hub['elastic_properties_mb']['system_inertia']

    return wt_opt

def assign_nacelle_values(wt_opt, assembly, nacelle):

    wt_opt['nacelle.uptilt']            = nacelle['outer_shape_bem']['uptilt_angle']
    wt_opt['nacelle.distance_tt_hub']   = nacelle['outer_shape_bem']['distance_tt_hub']
    wt_opt['nacelle.overhang']          = nacelle['outer_shape_bem']['overhang']


    wt_opt['nacelle.above_yaw_mass']    = nacelle['elastic_properties_mb']['above_yaw_mass']
    wt_opt['nacelle.yaw_mass']          = nacelle['elastic_properties_mb']['yaw_mass']
    wt_opt['nacelle.nacelle_cm']        = nacelle['elastic_properties_mb']['center_mass']
    wt_opt['nacelle.nacelle_I']         = nacelle['elastic_properties_mb']['inertia']
    
    wt_opt['nacelle.gear_ratio']        = nacelle['drivetrain']['gear_ratio']
    wt_opt['nacelle.drivetrain_eff']    = nacelle['drivetrain']['efficiency']
    if assembly['drivetrain'].upper() != 'DIRECT DRIVE':
        wt_opt['nacelle.shaft_ratio']       = nacelle['drivetrain']['shaft_ratio']
        wt_opt['nacelle.planet_numbers']    = nacelle['drivetrain']['planet_numbers']
        wt_opt['nacelle.shrink_disc_mass']  = nacelle['drivetrain']['shrink_disc_mass']
        wt_opt['nacelle.carrier_mass']      = nacelle['drivetrain']['carrier_mass']
        wt_opt['nacelle.flange_length']     = nacelle['drivetrain']['flange_length']
        wt_opt['nacelle.gearbox_input_xcm'] = nacelle['drivetrain']['gearbox_input_xcm']
        wt_opt['nacelle.hss_input_length']  = nacelle['drivetrain']['hss_input_length']
        wt_opt['nacelle.distance_hub2mb']   = nacelle['drivetrain']['distance_hub2mb']
        wt_opt['nacelle.yaw_motors_number'] = nacelle['drivetrain']['yaw_motors_number']
    else:
        print('DriveSE not yet supports direct drive configurations!!! Please do not trust the nacelle design and the lcoe analysis!')
        wt_opt['nacelle.shaft_ratio']       = 0.1
        wt_opt['nacelle.planet_numbers']    = [3, 3, 1]
        wt_opt['nacelle.shrink_disc_mass']  = 1600
        wt_opt['nacelle.carrier_mass']      = 8000
        wt_opt['nacelle.flange_length']     = 0.5
        wt_opt['nacelle.gearbox_input_xcm'] = 0.1
        wt_opt['nacelle.hss_input_length']  = 1.5
        wt_opt['nacelle.distance_hub2mb']   = 1.912
        wt_opt['nacelle.yaw_motors_number'] = 1.


    return wt_opt

def assign_tower_values(wt_opt, analysis_options, tower):
    # Function to assign values to the openmdao component Tower
    n_height        = analysis_options['tower']['n_height'] # Number of points along tower height
    n_layers        = analysis_options['tower']['n_layers']
    
    layer_name      = n_layers * ['']
    layer_mat       = n_layers * ['']
    thickness       = np.zeros((n_layers, n_height-1))

    svec = np.unique( np.r_[tower['outer_shape_bem']['outer_diameter']['grid'],
                            tower['outer_shape_bem']['reference_axis']['x']['grid'],
                            tower['outer_shape_bem']['reference_axis']['y']['grid'],
                            tower['outer_shape_bem']['reference_axis']['z']['grid']] )
    
    wt_opt['tower.s'] = svec
    wt_opt['tower.diameter']   = np.interp(svec, tower['outer_shape_bem']['outer_diameter']['grid'], tower['outer_shape_bem']['outer_diameter']['values'])
    
    wt_opt['tower.ref_axis'][:,0]  = np.interp(svec, tower['outer_shape_bem']['reference_axis']['x']['grid'], tower['outer_shape_bem']['reference_axis']['x']['values'])
    wt_opt['tower.ref_axis'][:,1]  = np.interp(svec, tower['outer_shape_bem']['reference_axis']['y']['grid'], tower['outer_shape_bem']['reference_axis']['y']['values'])
    wt_opt['tower.ref_axis'][:,2]  = np.interp(svec, tower['outer_shape_bem']['reference_axis']['z']['grid'], tower['outer_shape_bem']['reference_axis']['z']['values'])

    for i in range(n_layers):
        layer_name[i]  = tower['internal_structure_2d_fem']['layers'][i]['name']
        layer_mat[i]   = tower['internal_structure_2d_fem']['layers'][i]['material']
        thickness[i]   = np.interp(svec, tower['internal_structure_2d_fem']['layers'][i]['thickness']['grid'], tower['internal_structure_2d_fem']['layers'][i]['thickness']['values'])[0: -1]

    wt_opt['tower.layer_name']        = layer_name
    wt_opt['tower.layer_mat']         = layer_mat
    wt_opt['tower.layer_thickness']   = thickness
    
    wt_opt['tower.outfitting_factor'] = tower['internal_structure_2d_fem']['outfitting_factor']    
    
    return wt_opt

def assign_monopile_values(wt_opt, analysis_options, monopile):
    # Function to assign values to the openmdao component Monopile
    n_height        = analysis_options['monopile']['n_height'] # Number of points along monopile height
    n_layers        = analysis_options['monopile']['n_layers']
    
    layer_name      = n_layers * ['']
    layer_mat       = n_layers * ['']
    thickness       = np.zeros((n_layers, n_height-1))

    svec = np.unique( np.r_[monopile['outer_shape_bem']['outer_diameter']['grid'],
                            monopile['outer_shape_bem']['reference_axis']['x']['grid'],
                            monopile['outer_shape_bem']['reference_axis']['y']['grid'],
                            monopile['outer_shape_bem']['reference_axis']['z']['grid']] )
    
    wt_opt['monopile.s'] = svec
    wt_opt['monopile.diameter']   = np.interp(svec, monopile['outer_shape_bem']['outer_diameter']['grid'], monopile['outer_shape_bem']['outer_diameter']['values'])
    
    wt_opt['monopile.ref_axis'][:,0]  = np.interp(svec, monopile['outer_shape_bem']['reference_axis']['x']['grid'], monopile['outer_shape_bem']['reference_axis']['x']['values'])
    wt_opt['monopile.ref_axis'][:,1]  = np.interp(svec, monopile['outer_shape_bem']['reference_axis']['y']['grid'], monopile['outer_shape_bem']['reference_axis']['y']['values'])
    wt_opt['monopile.ref_axis'][:,2]  = np.interp(svec, monopile['outer_shape_bem']['reference_axis']['z']['grid'], monopile['outer_shape_bem']['reference_axis']['z']['values'])

    for i in range(n_layers):
        layer_name[i]  = monopile['internal_structure_2d_fem']['layers'][i]['name']
        layer_mat[i]   = monopile['internal_structure_2d_fem']['layers'][i]['material']
        thickness[i]   = monopile['internal_structure_2d_fem']['layers'][i]['thickness']['values']

    wt_opt['monopile.layer_name']        = layer_name
    wt_opt['monopile.layer_mat']         = layer_mat
    wt_opt['monopile.layer_thickness']   = thickness

    wt_opt['monopile.outfitting_factor']          = monopile['outfitting_factor']    
    wt_opt['monopile.transition_piece_height']    = monopile['transition_piece_height']
    wt_opt['monopile.transition_piece_mass']      = monopile['transition_piece_mass']
    wt_opt['monopile.gravity_foundation_mass']    = monopile['gravity_foundation_mass']
    wt_opt['monopile.suctionpile_depth']          = monopile['suctionpile_depth']
    wt_opt['monopile.suctionpile_depth_diam_ratio']          = monopile['suctionpile_depth_diam_ratio']
    
    return wt_opt


def assign_foundation_values(wt_opt, foundation):

    wt_opt['foundation.height']    = foundation['height']

    return wt_opt


def assign_floating_values(wt_opt, analysis_options, floating):

    svec = np.unique( np.r_[floating['column']['outer_shape_bem']['outer_diameter']['grid'],
                            floating['column']['outer_shape_bem']['reference_axis']['x']['grid'],
                            floating['column']['outer_shape_bem']['reference_axis']['y']['grid'],
                            floating['column']['outer_shape_bem']['reference_axis']['z']['grid']] )
    
    wt_opt['floating.radius_to_offset_column'] = floating['column']['offset']['reference_axis']['x']
        
        ivc.add_output('radius_to_offset_column', 0.0, units='m')
        ivc.add_discrete_output('number_of_offset_columns', 0)
        ivc.add_output('fairlead_location', 0.0)
        ivc.add_output('fairlead_offset_from_shell', 0.0, units='m')
        ivc.add_output('outfitting_cost_rate', 0.0, units='USD/kg')
        ivc.add_discrete_output('loading', 'hydrostatic')
        ivc.add_output('transition_piece_height', val = 0.0, units='m',  desc='point mass height of transition piece above water line')
        ivc.add_output('transition_piece_mass',   val = 0.0, units='kg', desc='point mass of transition piece')

        # Replicate this main/offset
        ivc.add_output('diameter', val=np.zeros(n_height),     units='m',  desc='1D array of the outer diameter values defined along the column axis.')
        ivc.add_discrete_output('layer_name', val=n_layers * [''],         desc='1D array of the names of the layers modeled in the columnn structure.')
        ivc.add_discrete_output('layer_mat',  val=n_layers * [''],         desc='1D array of the names of the materials of each layer modeled in the column structure.')
        ivc.add_output('layer_thickness',     val=np.zeros((n_layers, n_height-1)), units='m',    desc='2D array of the thickness of the layers of the column structure. The first dimension represents each layer, the second dimension represents each piecewise-constant entry of the column sections.')
        ivc.add_output('freeboard', 0.0, units='m') # Have to add here because cannot promote ivc from Column before needed by tower.  Grr
        ivc.add_output('outfitting_factor',       val = 0.0,             desc='Multiplier that accounts for secondary structure mass inside of column')
        ivc.add_output('stiffener_web_height', np.zeros(n_sect), units='m')
        ivc.add_output('stiffener_web_thickness', np.zeros(n_sect), units='m')
        ivc.add_output('stiffener_flange_width', np.zeros(n_sect), units='m')
        ivc.add_output('stiffener_flange_thickness', np.zeros(n_sect), units='m')
        ivc.add_output('stiffener_spacing', np.zeros(n_sect), units='m')
        ivc.add_output('bulkhead_thickness', np.zeros(n_height), units='m')
        ivc.add_output('permanent_ballast_height', 0.0, units='m')
        ivc.add_output('buoyancy_tank_diameter', 0.0, units='m')
        ivc.add_output('buoyancy_tank_height', 0.0, units='m')
        ivc.add_output('buoyancy_tank_location', 0.0, units='m')

        ivc.add_output('mooring_line_length', 0.0, units='m')
        ivc.add_output('anchor_radius', 0.0, units='m')
        ivc.add_output('mooring_diameter', 0.0, units='m')
        ivc.add_output('number_of_mooring_connections', 0)
        ivc.add_output('mooring_lines_per_connection', 0)
        ivc.add_discrete_output('mooring_type', 'chain')
        ivc.add_discrete_output('anchor_type', 'SUCTIONPILE')
        ivc.add_output('max_offset', 0.0, units='m')
        ivc.add_output('operational_heel', 0.0, units='deg')
        ivc.add_output('mooring_cost_factor', 0.0)
        ivc.add_output('max_survival_heel', 0.0, units='deg')
        
    return wt_opt


def assign_control_values(wt_opt, analysis_options, control):
    # Controller parameters
    wt_opt['control.rated_power']   = control['rated_power']
    wt_opt['control.V_in']          = control['Vin']
    wt_opt['control.V_out']         = control['Vout']
    wt_opt['control.minOmega']      = control['minOmega']
    wt_opt['control.maxOmega']      = control['maxOmega']
    wt_opt['control.rated_TSR']     = control['tsr']
    wt_opt['control.rated_pitch']   = control['pitch']
    wt_opt['control.max_TS']        = control['maxTS']
    wt_opt['control.max_pitch_rate']= control['max_pitch_rate']
    wt_opt['control.max_torque_rate']= control['max_torque_rate']
    # ROSCO tuning parameters
    wt_opt['control.PC_omega']      = control['PC_omega']
    wt_opt['control.PC_zeta']       = control['PC_zeta']
    wt_opt['control.VS_omega']      = control['VS_omega']
    wt_opt['control.VS_zeta']       = control['VS_zeta']
    if analysis_options['servose']['Flp_Mode'] > 0:
        wt_opt['control.Flp_omega']      = control['Flp_omega']
        wt_opt['control.Flp_zeta']       = control['Flp_zeta']
    # # other optional parameters
    wt_opt['control.max_pitch']     = control['max_pitch']
    wt_opt['control.min_pitch']     = control['min_pitch']
    wt_opt['control.vs_minspd']     = control['vs_minspd']
    wt_opt['control.ss_vsgain']     = control['ss_vsgain']
    wt_opt['control.ss_pcgain']     = control['ss_pcgain']
    wt_opt['control.ps_percent']    = control['ps_percent']
    # Check for proper Flp_Mode, print warning
    if analysis_options['airfoils']['n_tab'] > 1 and analysis_options['servose']['Flp_Mode'] == 0:
            print('WARNING: servose.Flp_Mode should be >= 1 for aerodynamic control.')
    if analysis_options['airfoils']['n_tab'] == 1 and analysis_options['servose']['Flp_Mode'] > 0:
            print('WARNING: servose.Flp_Mode should be = 0 for no aerodynamic control.')
            
    return wt_opt

def assign_configuration_values(wt_opt, assembly):

    wt_opt['configuration.ws_class']   = assembly['turbine_class']
    wt_opt['configuration.turb_class']          = assembly['turbulence_class']
    wt_opt['configuration.gearbox_type']         = assembly['drivetrain']
    wt_opt['configuration.rotor_orientation']     = assembly['rotor_orientation'].lower()
    wt_opt['configuration.n_blades']     = assembly['number_of_blades']

    return wt_opt

def assign_environment_values(wt_opt, environment):

    wt_opt['env.rho_air']   = environment['air_density']
    wt_opt['env.mu_air']    = environment['air_dyn_viscosity']
    wt_opt['env.rho_water']   = environment['water_density']
    wt_opt['env.mu_water']    = environment['water_dyn_viscosity']
    wt_opt['env.weibull_k'] = environment['weib_shape_parameter']
    wt_opt['env.speed_sound_air'] = environment['air_speed_sound']
    wt_opt['env.shear_exp'] = environment['shear_exp']
    wt_opt['env.G_soil'] = environment['soil_shear_modulus']
    wt_opt['env.nu_soil'] = environment['soil_poisson']

    return wt_opt

def assign_costs_values(wt_opt, costs):

    wt_opt['costs.turbine_number']      = costs['turbine_number']
    wt_opt['costs.bos_per_kW']          = costs['bos_per_kW']
    wt_opt['costs.opex_per_kW']         = costs['opex_per_kW']
    wt_opt['costs.wake_loss_factor']    = costs['wake_loss_factor']
    wt_opt['costs.fixed_charge_rate']   = costs['fixed_charge_rate']
    if 'offset_tcc_per_kW' in costs:
        wt_opt['costs.offset_tcc_per_kW']   = costs['offset_tcc_per_kW']

    return wt_opt 

def assign_airfoil_values(wt_opt, analysis_options, airfoils):
    # Function to assign values to the openmdao component Airfoils
    
    n_af  = analysis_options['airfoils']['n_af']
    n_aoa = analysis_options['airfoils']['n_aoa']
    aoa   = analysis_options['airfoils']['aoa']
    n_Re  = analysis_options['airfoils']['n_Re']
    n_tab = analysis_options['airfoils']['n_tab']
    n_xy  = analysis_options['airfoils']['n_xy']
    
    name    = n_af * ['']
    ac      = np.zeros(n_af)
    r_thick = np.zeros(n_af)
    Re_all  = []
    for i in range(n_af):
        name[i]     = airfoils[i]['name']
        ac[i]       = airfoils[i]['aerodynamic_center']
        r_thick[i]  = airfoils[i]['relative_thickness']
        for j in range(len(airfoils[i]['polars'])):
            Re_all.append(airfoils[i]['polars'][j]['re'])
    Re = np.array(sorted(np.unique(Re_all)))
    
    cl = np.zeros((n_af, n_aoa, n_Re, n_tab))
    cd = np.zeros((n_af, n_aoa, n_Re, n_tab))
    cm = np.zeros((n_af, n_aoa, n_Re, n_tab))
    
    coord_xy = np.zeros((n_af, n_xy, 2))
    
    # Interp cl-cd-cm along predefined grid of angle of attack
    for i in range(n_af):
        n_Re_i = len(airfoils[i]['polars'])
        Re_j = np.zeros(n_Re_i)
        j_Re = np.zeros(n_Re_i, dtype=int)
        for j in range(n_Re_i):
            Re_j[j] = airfoils[i]['polars'][j]['re']
            j_Re[j] = np.argmin(Re-Re_j)
            cl[i,:,j_Re[j],0] = np.interp(aoa, airfoils[i]['polars'][j]['c_l']['grid'], airfoils[i]['polars'][j]['c_l']['values'])
            cd[i,:,j_Re[j],0] = np.interp(aoa, airfoils[i]['polars'][j]['c_d']['grid'], airfoils[i]['polars'][j]['c_d']['values'])
            cm[i,:,j_Re[j],0] = np.interp(aoa, airfoils[i]['polars'][j]['c_m']['grid'], airfoils[i]['polars'][j]['c_m']['values'])
    
            if abs(cl[i,0,j,0] - cl[i,-1,j,0]) > 1.e-5:
                cl[i,0,j,0] = cl[i,-1,j,0]
                print("WARNING: Airfoil " + name[i] + ' has the lift coefficient at Re ' + str(Re_j) + ' different between + and - pi rad. This is fixed automatically, but please check the input data.')
            if abs(cd[i,0,j,0] - cd[i,-1,j,0]) > 1.e-5:
                cd[i,0,j,0] = cd[i,-1,j,0]
                print("WARNING: Airfoil " + name[i] + ' has the drag coefficient at Re ' + str(Re_j) + ' different between + and - pi rad. This is fixed automatically, but please check the input data.')
            if abs(cm[i,0,j,0] - cm[i,-1,j,0]) > 1.e-5:
                cm[i,0,j,0] = cm[i,-1,j,0]
                print("WARNING: Airfoil " + name[i] + ' has the moment coefficient at Re ' + str(Re_j) + ' different between + and - pi rad. This is fixed automatically, but please check the input data.')
        
        # Re-interpolate cl-cd-cm along the Re dimension if less than n_Re were provided in the input yaml (common condition)
        for k in range(n_aoa):
            cl[i,k,:,0] = np.interp(Re, Re_j, cl[i,k,j_Re,0])
            cd[i,k,:,0] = np.interp(Re, Re_j, cd[i,k,j_Re,0])
            cm[i,k,:,0] = np.interp(Re, Re_j, cm[i,k,j_Re,0])


        points = np.column_stack((airfoils[i]['coordinates']['x'], airfoils[i]['coordinates']['y']))
        # Check that airfoil points are declared from the TE suction side to TE pressure side
        idx_le = np.argmin(points[:,0])
        if np.mean(points[:idx_le,1]) > 0.:
            points = np.flip(points, axis=0)
        
        # Remap points using class AirfoilShape
        af = AirfoilShape(points=points)
        af.redistribute(n_xy, even=False, dLE=True)
        s = af.s
        af_points = af.points
        
        # Add trailing edge point if not defined
        if [1,0] not in af_points.tolist():
            af_points[:,0] -= af_points[np.argmin(af_points[:,0]), 0]
        c = max(af_points[:,0])-min(af_points[:,0])
        af_points[:,:] /= c
        
        coord_xy[i,:,:] = af_points
        
        # Plotting
        # import matplotlib.pyplot as plt
        # plt.plot(af_points[:,0], af_points[:,1], '.')
        # plt.plot(af_points[:,0], af_points[:,1])
        # plt.show()
        
    # Assign to openmdao structure    
    wt_opt['airfoils.aoa']       = aoa
    wt_opt['airfoils.name']      = name
    wt_opt['airfoils.ac']        = ac
    wt_opt['airfoils.r_thick']   = r_thick
    wt_opt['airfoils.Re']        = Re  # Not yet implemented!
    wt_opt['airfoils.tab']       = 0.  # Not yet implemented!
    wt_opt['airfoils.cl']        = cl
    wt_opt['airfoils.cd']        = cd
    wt_opt['airfoils.cm']        = cm
    
    wt_opt['airfoils.coord_xy']  = coord_xy
     
    return wt_opt
    
def assign_material_values(wt_opt, analysis_options, materials):
    # Function to assign values to the openmdao component Materials
    
    n_mat = analysis_options['materials']['n_mat']
    
    name        = n_mat * ['']
    orth        = np.zeros(n_mat)
    component_id= -np.ones(n_mat)
    rho         = np.zeros(n_mat)
    sigma_y     = np.zeros(n_mat)
    E           = np.zeros([n_mat, 3])
    G           = np.zeros([n_mat, 3])
    nu          = np.zeros([n_mat, 3])
    Xt          = np.zeros([n_mat, 3])
    Xc          = np.zeros([n_mat, 3])
    rho_fiber   = np.zeros(n_mat)
    rho_area_dry= np.zeros(n_mat)
    fvf         = np.zeros(n_mat)
    fwf         = np.zeros(n_mat)
    ply_t       = np.zeros(n_mat)
    roll_mass   = np.zeros(n_mat)
    unit_cost   = np.zeros(n_mat)
    waste       = np.zeros(n_mat)
    
    for i in range(n_mat):
        name[i]    =  materials[i]['name']
        orth[i]    =  materials[i]['orth']
        rho[i]     =  materials[i]['rho']
        if 'component_id' in materials[i]:
            component_id[i] = materials[i]['component_id']
        if orth[i] == 0:
            if 'E' in materials[i]:
                E[i,:]  = np.ones(3) * materials[i]['E']
            if 'nu' in materials[i]:
                nu[i,:] = np.ones(3) * materials[i]['nu']
            if 'G' in materials[i]:
                G[i,:]  = np.ones(3) * materials[i]['G']
            elif 'nu' in materials[i]:
                G[i,:]  = np.ones(3) * materials[i]['E']/(2*(1+materials[i]['nu'])) # If G is not provided but the material is isotropic and we have E and nu we can just estimate it
                warning_shear_modulus_isotropic = 'WARNING: NO shear modulus, G, was provided for material "%s". The code assumes 2G*(1 + nu) = E, which is only valid for isotropic materials.'%name[i]
                print(warning_shear_modulus_isotropic)
            if 'Xt' in materials[i]:
                Xt[i,:] = np.ones(3) * materials[i]['Xt']
            if 'Xc' in materials[i]:
                Xc[i,:] = np.ones(3) * materials[i]['Xc']
        elif orth[i] == 1:
            E[i,:]  = materials[i]['E']
            G[i,:]  = materials[i]['G']
            nu[i,:] = materials[i]['nu']
            Xt[i,:] = materials[i]['Xt']
            Xc[i,:] = materials[i]['Xc']

        else:
            exit('The flag orth must be set to either 0 or 1. Error in material ' + name[i])
        if 'fiber_density' in materials[i]:
            rho_fiber[i]    = materials[i]['fiber_density']
        if 'area_density_dry' in materials[i]:
            rho_area_dry[i] = materials[i]['area_density_dry']
        if 'fvf' in materials[i]:
            fvf[i] = materials[i]['fvf']
        if 'fwf' in materials[i]:
            fwf[i] = materials[i]['fwf']
        if 'ply_t' in materials[i]:
            ply_t[i] = materials[i]['ply_t']
        if 'roll_mass' in materials[i]:
            roll_mass[i] = materials[i]['roll_mass']
        if 'unit_cost' in materials[i]:
            unit_cost[i] = materials[i]['unit_cost']
        if 'waste' in materials[i]:
            waste[i] = materials[i]['waste']
        if 'Xy' in materials[i]:
            sigma_y[i] =  materials[i]['Xy']

            
    wt_opt['materials.name']     = name
    wt_opt['materials.orth']     = orth
    wt_opt['materials.rho']      = rho
    wt_opt['materials.sigma_y']  = sigma_y
    wt_opt['materials.component_id']= component_id
    wt_opt['materials.E']        = E
    wt_opt['materials.G']        = G
    wt_opt['materials.Xt']       = Xt
    wt_opt['materials.Xc']       = Xc
    wt_opt['materials.nu']       = nu
    wt_opt['materials.rho_fiber']      = rho_fiber
    wt_opt['materials.rho_area_dry']   = rho_area_dry
    wt_opt['materials.fvf_from_yaml']      = fvf
    wt_opt['materials.fwf_from_yaml']      = fwf
    wt_opt['materials.ply_t_from_yaml']    = ply_t
    wt_opt['materials.roll_mass']= roll_mass
    wt_opt['materials.unit_cost']= unit_cost
    wt_opt['materials.waste']    = waste

    return wt_opt

if __name__ == "__main__":

    ## File management
    fname_input        = "reference_turbines/nrel5mw/nrel5mw_mod_update.yaml"
    # fname_input        = "/mnt/c/Material/Projects/Hitachi_Design/Design/turbine_inputs/aerospan_formatted_v13.yaml"
    fname_output       = "reference_turbines/nrel5mw/nrel5mw_mod_update_output.yaml"
    
    # Load yaml data into a pure python data structure
    wt_initial               = WindTurbineOntologyPython()
    wt_initial.validate      = False
    wt_initial.fname_schema  = "reference_turbines/IEAontology_schema.yaml"
    analysis_options, wt_init = wt_initial.initialize(fname_input)
    
    # Initialize openmdao problem
    wt_opt          = om.Problem()
    wt_opt.model    = WindTurbineOntologyOpenMDAO(analysis_options = analysis_options)
    wt_opt.setup()
    # Load wind turbine data from wt_initial to the openmdao problem
    wt_opt = yaml2openmdao(wt_opt, analysis_options, wt_init)
    wt_opt.run_driver()
    
    # Save data coming from openmdao to an output yaml file
    wt_initial.write_ontology(wt_opt, fname_output)
