import ruamel_yaml as ry
import numpy as np
import jsonschema as json
import time, copy
from scipy.interpolate import PchipInterpolator, interp1d
from openmdao.api import ExplicitComponent, Group, IndepVarComp, Problem
from wisdem.rotorse.geometry_tools.geometry import AirfoilShape, trailing_edge_smoothing, remap2grid
from wisdem.commonse.utilities import arc_length
from wisdem.aeroelasticse.FAST_reader import InputReader_OpenFAST
from wisdem.aeroelasticse.CaseLibrary import RotorSE_rated, RotorSE_DLC_1_4_Rated, RotorSE_DLC_7_1_Steady, RotorSE_DLC_1_1_Turb, power_curve


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
    xy_coord_arc = arc_length(xy_coord[:,0], xy_coord[:,1])
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
        FASTpref['cores']               = self.analysis_options['openfast']['cores']
        FASTpref['debug_level']         = self.analysis_options['openfast']['debug_level']
        FASTpref['DLC_gust']            = None      # Max deflection
        FASTpref['DLC_extrm']           = None      # Max strain
        FASTpref['DLC_turbulent']       = RotorSE_DLC_1_1_Turb
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
            print('If you like a grid of angles of attack more refined between +- 30 deg, please choose a n_aoa in the analysis option input file that is a multiple of 4. The current value of ' + str(self.analysis_options['airfoils']['n_aoa']) + ' is not a multiple of 4 and an equally spaced grid is adopted.')
        Re_all = []
        for i in range(self.analysis_options['airfoils']['n_af']):
            for j in range(len(self.wt_init['airfoils'][i]['polars'])):
                Re_all.append(self.wt_init['airfoils'][i]['polars'][j]['re'])
        self.analysis_options['airfoils']['n_Re']   = len(np.unique(Re_all))
        self.analysis_options['airfoils']['n_tab']  = 1
        self.analysis_options['airfoils']['n_xy']   = self.analysis_options['rotorse']['n_xy']
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
        
        # Distributed aerodynamic control devices along blade
        self.analysis_options['blade']['n_te_flaps']      = 0
        if 'aerodynamic_control' in self.wt_init['components']['blade']:
            if 'te_flaps' in self.wt_init['components']['blade']['aerodynamic_control']:
                self.analysis_options['blade']['n_te_flaps'] = len(self.wt_init['components']['blade']['aerodynamic_control']['te_flaps'])
                self.analysis_options['airfoils']['n_tab']   = 3
            else:
                exit('A distributed aerodynamic control device is provided in the yaml input file, but not supported by wisdem.')

        # Tower 
        self.analysis_options['tower']              = {}
        self.analysis_options['tower']['n_height']  = len(self.wt_init['components']['tower']['outer_shape_bem']['outer_diameter']['grid'])
        self.analysis_options['tower']['nd_height'] = np.linspace(0., 1., self.analysis_options['tower']['n_height']) # Equally spaced non-dimensional grid along tower height
        self.analysis_options['tower']['n_layers']  = len(self.wt_init['components']['tower']['internal_structure_2d_fem']['layers'])


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
        self.wt_init['components']['blade']['outer_shape_bem']['chord']['grid']     = wt_opt['blade.outer_shape_bem.s'].tolist()
        # self.wt_init['components']['blade']['outer_shape_bem']['chord']['values']   = wt_opt['blade.outer_shape_bem.chord'].tolist()
        self.wt_init['components']['blade']['outer_shape_bem']['chord']['values']   = wt_opt['param.pa.chord_param'].tolist()
        self.wt_init['components']['blade']['outer_shape_bem']['twist']['grid']     = wt_opt['blade.outer_shape_bem.s'].tolist()
        # self.wt_init['components']['blade']['outer_shape_bem']['twist']['values']   = wt_opt['blade.outer_shape_bem.twist'].tolist()
        self.wt_init['components']['blade']['outer_shape_bem']['twist']['values']   = wt_opt['param.pa.twist_param'].tolist()
        self.wt_init['components']['blade']['outer_shape_bem']['pitch_axis']['grid']     = wt_opt['blade.outer_shape_bem.s'].tolist()
        self.wt_init['components']['blade']['outer_shape_bem']['pitch_axis']['values']   = wt_opt['blade.outer_shape_bem.pitch_axis'].tolist()
        self.wt_init['components']['blade']['outer_shape_bem']['reference_axis']['x']['grid']     = wt_opt['blade.outer_shape_bem.s'].tolist()
        self.wt_init['components']['blade']['outer_shape_bem']['reference_axis']['y']['grid']     = wt_opt['blade.outer_shape_bem.s'].tolist()
        self.wt_init['components']['blade']['outer_shape_bem']['reference_axis']['z']['grid']     = wt_opt['blade.outer_shape_bem.s'].tolist()
        self.wt_init['components']['blade']['outer_shape_bem']['reference_axis']['x']['values']   = wt_opt['blade.outer_shape_bem.ref_axis'][:,0].tolist()
        self.wt_init['components']['blade']['outer_shape_bem']['reference_axis']['y']['values']   = wt_opt['blade.outer_shape_bem.ref_axis'][:,1].tolist()
        self.wt_init['components']['blade']['outer_shape_bem']['reference_axis']['z']['values']   = wt_opt['blade.outer_shape_bem.ref_axis'][:,2].tolist()

        # Update blade structure
        # Reference axis from blade outer shape
        self.wt_init['components']['blade']['internal_structure_2d_fem']['reference_axis'] = self.wt_init['components']['blade']['outer_shape_bem']['reference_axis']
        # Webs positions
        for i in range(self.analysis_options['blade']['n_webs']):
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
            self.wt_init['components']['blade']['internal_structure_2d_fem']['layers'][i]['thickness']['values']    = wt_opt['param.ps.layer_thickness_param'][i,:].tolist()
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

        # Update hub
        self.wt_init['components']['hub']['outer_shape_bem']['diameter']   = float(wt_opt['hub.diameter'])
        self.wt_init['components']['hub']['outer_shape_bem']['cone_angle'] = float(wt_opt['hub.cone'])

        # Update nacelle
        self.wt_init['components']['nacelle']['outer_shape_bem']['uptilt_angle']    = float(wt_opt['nacelle.uptilt'])
        self.wt_init['components']['nacelle']['outer_shape_bem']['distance_tt_hub'] = float(wt_opt['nacelle.distance_tt_hub'])

        # Update tower
        self.wt_init['components']['tower']['outer_shape_bem']['outer_diameter']['grid']      = wt_opt['tower.s'].tolist()
        self.wt_init['components']['tower']['outer_shape_bem']['outer_diameter']['values']    = wt_opt['tower.diameter'].tolist()
        self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['x']['grid']     = wt_opt['tower.s'].tolist()
        self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['y']['grid']     = wt_opt['tower.s'].tolist()
        self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['z']['grid']     = wt_opt['tower.s'].tolist()
        self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['x']['values']   = wt_opt['tower.ref_axis'][:,0].tolist()
        self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['y']['values']   = wt_opt['tower.ref_axis'][:,1].tolist()
        self.wt_init['components']['tower']['outer_shape_bem']['reference_axis']['z']['values']   = wt_opt['tower.ref_axis'][:,2].tolist()
        for i in range(self.analysis_options['tower']['n_layers']):
            self.wt_init['components']['tower']['internal_structure_2d_fem']['layers'][i]['thickness']['grid']      = wt_opt['tower.s'].tolist()
            self.wt_init['components']['tower']['internal_structure_2d_fem']['layers'][i]['thickness']['values']    = wt_opt['tower.layer_thickness'][i,:].tolist()

        # Write yaml with updated values
        f = open(fname_output, "w")
        yaml=ry.YAML()
        yaml.default_flow_style = None
        yaml.width = float("inf")
        yaml.indent(mapping=4, sequence=6, offset=3)
        yaml.dump(self.wt_init, f)

        return None

class Blade(Group):
    # Openmdao group with components with the blade data coming from the input yaml file.
    def initialize(self):
        self.options.declare('blade_init_options')
        self.options.declare('af_init_options')
                
    def setup(self):
        # Options
        blade_init_options = self.options['blade_init_options']
        af_init_options    = self.options['af_init_options']
        
        # Import outer shape BEM
        self.add_subsystem('outer_shape_bem', Blade_Outer_Shape_BEM(blade_init_options = blade_init_options), promotes = ['length'])
        
        # Interpolate airfoil profiles and coordinates
        self.add_subsystem('interp_airfoils', Blade_Interp_Airfoils(blade_init_options = blade_init_options, af_init_options = af_init_options))
        
        # Connections from oute_shape_bem to interp_airfoils
        self.connect('outer_shape_bem.s',           'interp_airfoils.s')
        self.connect('outer_shape_bem.chord',       'interp_airfoils.chord')
        self.connect('outer_shape_bem.pitch_axis',  'interp_airfoils.pitch_axis')
        self.connect('outer_shape_bem.af_used',     'interp_airfoils.af_used')
        self.connect('outer_shape_bem.af_position', 'interp_airfoils.af_position')
        
        # If the flag is true, generate the 3D x,y,z points of the outer blade shape
        if blade_init_options['lofted_output'] == True:
            self.add_subsystem('blade_lofted',    Blade_Lofted_Shape(blade_init_options = blade_init_options, af_init_options = af_init_options))
            self.connect('interp_airfoils.coord_xy_dim',    'blade_lofted.coord_xy_dim')
            self.connect('outer_shape_bem.twist',           'blade_lofted.twist')
            self.connect('outer_shape_bem.s',               'blade_lofted.s')
            self.connect('outer_shape_bem.ref_axis',        'blade_lofted.ref_axis')
        
        # Import blade internal structure data and remap composites on the outer blade shape
        self.add_subsystem('internal_structure_2d_fem', Blade_Internal_Structure_2D_FEM(blade_init_options = blade_init_options, af_init_options = af_init_options))
        self.connect('outer_shape_bem.twist',           'internal_structure_2d_fem.twist')
        self.connect('interp_airfoils.coord_xy_dim',    'internal_structure_2d_fem.coord_xy_dim')

        # Import trailing-edge flaps data
        self.add_subsystem('dac_te_flaps', TE_Flaps(blade_init_options = blade_init_options))

class Blade_Outer_Shape_BEM(ExplicitComponent):
    # Openmdao component with the blade outer shape data coming from the input yaml file.
    def initialize(self):
        self.options.declare('blade_init_options')
        
    def setup(self):
        blade_init_options = self.options['blade_init_options']
        n_af_span          = blade_init_options['n_af_span']
        n_span             = blade_init_options['n_span']
        
        self.add_discrete_output('af_used', val=n_af_span * [''],              desc='1D array of names of the airfoils actually defined along blade span.')
        
        self.add_output('af_position',   val=np.zeros(n_af_span),              desc='1D array of the non dimensional positions of the airfoils af_used defined along blade span.')
        self.add_output('s',             val=np.zeros(n_span),                 desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        self.add_output('chord',         val=np.zeros(n_span),    units='m',   desc='1D array of the chord values defined along blade span.')
        self.add_output('twist',         val=np.zeros(n_span),    units='rad', desc='1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).')
        self.add_output('pitch_axis',    val=np.zeros(n_span),                 desc='1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.')
        self.add_output('ref_axis',      val=np.zeros((n_span,3)),units='m',   desc='2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.')

        self.add_output('length',       val = 0.0,               units='m',    desc='Scalar of the 3D blade length computed along its axis.')
        self.add_output('length_z',     val = 0.0,               units='m',    desc='Scalar of the 1D blade length along z, i.e. the blade projection in the plane ignoring prebend and sweep. For a straight blade this is equal to length')
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        outputs['length']   = arc_length(outputs['ref_axis'][:,0], outputs['ref_axis'][:,1], outputs['ref_axis'][:,2])[-1]
        outputs['length_z'] = outputs['ref_axis'][:,2][-1]

class Blade_Interp_Airfoils(ExplicitComponent):
    # Openmdao component to interpolate airfoil coordinates and airfoil polars along the span of the blade for a predefined set of airfoils coming from component Airfoils.
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
        
        self.add_discrete_input('af_used', val=n_af_span * [''],              desc='1D array of names of the airfoils defined along blade span.')
        
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
                if discrete_inputs['af_used'][i] == discrete_inputs['name'][j]:                    
                    r_thick_used[i]     = inputs['r_thick'][j]
                    coord_xy_used[i,:,:]= inputs['coord_xy'][j]
                    cl_used[i,:,:,:]    = inputs['cl'][j,:,:,:]
                    cd_used[i,:,:,:]    = inputs['cd'][j,:,:,:]
                    cm_used[i,:,:,:]    = inputs['cm'][j,:,:,:]
                    break
        
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
        
        # Plot interpolated coordinates
        # import matplotlib.pyplot as plt
        # for i in range(self.n_span):    
        #     plt.plot(coord_xy_interp[i,:,0], coord_xy_interp[i,:,1], 'k', label = 'coord_xy_interp')
        #     plt.plot(coord_xy_dim[i,:,0], coord_xy_dim[i,:,1], 'b', label = 'coord_xy_dim')
        #     plt.axis('equal')
        #     plt.title(i)
        #     plt.legend()
        #     plt.show()

        
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

class Blade_Lofted_Shape(ExplicitComponent):
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
     
class Blade_Internal_Structure_2D_FEM(ExplicitComponent):
    # Openmdao component with the blade internal structure data coming from the input yaml file.
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
        
        
        self.add_input('coord_xy_dim',     val=np.zeros((n_span, n_xy, 2)),units = 'm',  desc='3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.')
        self.add_input('twist',            val=np.zeros(n_span),           units='rad',  desc='1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).')
        
        self.add_discrete_output('web_name', val=n_webs * [''],                          desc='1D array of the names of the shear webs defined in the blade structure.')
        
        self.add_output('s',              val=np.zeros(n_span),                          desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        self.add_output('web_rotation',   val=np.zeros((n_webs, n_span)),  units='rad',  desc='2D array of the rotation angle of the shear webs in respect to the chord line. The first dimension represents each shear web, the second dimension represents each entry along blade span. If the rotation is equal to negative twist +- a constant, then the web is built straight.')
        self.add_output('web_offset_y_pa',val=np.zeros((n_webs, n_span)),  units='m',    desc='2D array of the offset along the y axis to set the position of the shear webs. Positive values move the web towards the trailing edge, negative values towards the leading edge. The first dimension represents each shear web, the second dimension represents each entry along blade span.')
        self.add_output('web_start_nd',   val=np.zeros((n_webs, n_span)),                desc='2D array of the non-dimensional start point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.')
        self.add_output('web_end_nd',     val=np.zeros((n_webs, n_span)),                desc='2D array of the non-dimensional end point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.')
        
        self.add_discrete_output('layer_name', val=n_layers * [''],                      desc='1D array of the names of the layers modeled in the blade structure.')
        self.add_discrete_output('layer_mat',  val=n_layers * [''],                      desc='1D array of the names of the materials of each layer modeled in the blade structure.')
        self.add_discrete_output('layer_web',  val=n_layers * [''],                      desc='1D array of the names of the webs the layer is associated to. If the layer is on the outer profile this entry can simply stay empty.')
        
        self.add_output('layer_thickness',   val=np.zeros((n_layers, n_span)), units='m',    desc='2D array of the thickness of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_output('layer_rotation',    val=np.zeros((n_layers, n_span)), units='rad',  desc='2D array of the rotation angle of a layer in respect to the chord line. The first dimension represents each layer, the second dimension represents each entry along blade span. If the rotation is equal to negative twist +- a constant, then the layer is built straight.')
        self.add_output('layer_offset_y_pa', val=np.zeros((n_layers, n_span)), units='m',    desc='2D array of the offset along the y axis to set the position of a layer. Positive values move the layer towards the trailing edge, negative values towards the leading edge. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_output('layer_width',       val=np.zeros((n_layers, n_span)), units='m',    desc='2D array of the width along the outer profile of a layer. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_output('layer_midpoint_nd', val=np.zeros((n_layers, n_span)),               desc='2D array of the non-dimensional midpoint defined along the outer profile of a layer. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_discrete_output('layer_side',val=n_layers * [''],                           desc='1D array setting whether the layer is on the suction or pressure side. This entry is only used if definition_layer is equal to 1 or 2.')
        self.add_output('layer_start_nd',    val=np.zeros((n_layers, n_span)),               desc='2D array of the non-dimensional start point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        self.add_output('layer_end_nd',      val=np.zeros((n_layers, n_span)),               desc='2D array of the non-dimensional end point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.')
        
        self.add_discrete_output('definition_web',   val=np.zeros(n_webs),                   desc='1D array of flags identifying how webs are specified in the yaml. 1) offset+rotation=twist 2) offset+rotation')
        self.add_discrete_output('definition_layer', val=np.zeros(n_layers),                 desc='1D array of flags identifying how layers are specified in the yaml. 1) all around (skin, paint, ) 2) offset+rotation twist+width (spar caps) 3) offset+user defined rotation+width 4) midpoint TE+width (TE reinf) 5) midpoint LE+width (LE reinf) 6) layer position fixed to other layer (core fillers) 7) start and width 8) end and width 9) start and end nd 10) web layer')
        self.add_discrete_output('index_layer_start',    val=np.zeros(n_layers),             desc='Index used to fix a layer to another')
        self.add_discrete_output('index_layer_end',      val=np.zeros(n_layers),             desc='Index used to fix a layer to another')
    
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        # Initialize temporary arrays for the outputs
        web_rotation    = np.zeros((self.n_webs, self.n_span))
        layer_rotation  = np.zeros((self.n_layers, self.n_span))
        web_start_nd    = np.zeros((self.n_webs, self.n_span))
        web_end_nd      = np.zeros((self.n_webs, self.n_span))
        layer_start_nd  = np.zeros((self.n_layers, self.n_span))
        layer_end_nd    = np.zeros((self.n_layers, self.n_span))

        # Loop through spanwise stations
        for i in range(self.n_span):
            # Compute the arc length (arc_L_i), the non-dimensional arc coordinates (xy_arc_i), and the non dimensional position of the leading edge of the profile at position i
            xy_coord_i  = inputs['coord_xy_dim'][i,:,:]
            xy_arc_i    = arc_length(xy_coord_i[:,0], xy_coord_i[:,1])
            arc_L_i     = xy_arc_i[-1]
            xy_arc_i    /= arc_L_i
            idx_le      = np.argmin(xy_coord_i[:,0])
            LE_loc      = xy_arc_i[idx_le]
            # Loop through the webs and compute non-dimensional start and end positions along the profile
            for j in range(self.n_webs):
                if discrete_outputs['definition_web'][j] == 1:
                    web_rotation[j,i] = - inputs['twist'][i]
                    web_start_nd[j,i], web_end_nd[j,i] = calc_axis_intersection(inputs['coord_xy_dim'][i,:,:], web_rotation[j,i], outputs['web_offset_y_pa'][j,i], [0.,0.], ['suction', 'pressure'])
                elif discrete_outputs['definition_web'][j] == 2:
                    web_rotation[j,i] = - outputs['web_rotation'][j,i]
                    web_start_nd[j,i], web_end_nd[j,i] = calc_axis_intersection(inputs['coord_xy_dim'][i,:,:], web_rotation[j,i], outputs['web_offset_y_pa'][j,i], [0.,0.], ['suction', 'pressure'])
                    if i == 0:
                        print('The web ' + discrete_outputs['web_name'][j] + ' is defined with a user-defined rotation. If you are planning to run a twist optimization, you may want to rethink this definition.')
                    if web_start_nd[j,i] < 0. or web_start_nd[j,i] > 1.:
                        exit('Blade web ' + discrete_outputs['web_name'][j] + ' at n.d. span position ' + str(outputs['s'][i]) + ' has the n.d. start point outside the TE. Please check the yaml input file.')
                    if web_end_nd[j,i] < 0. or web_end_nd[j,i] > 1.:
                        exit('Blade web ' + discrete_outputs['web_name'][j] + ' at n.d. span position ' + str(outputs['s'][i]) + ' has the n.d. end point outside the TE. Please check the yaml input file.')
                else:
                    exit('Blade web ' + discrete_outputs['web_name'][j] + ' not described correctly. Please check the yaml input file.')
                    
            # Loop through the layers and compute non-dimensional start and end positions along the profile for the different layer definitions
            for j in range(self.n_layers):
                if discrete_outputs['definition_layer'][j] == 1: # All around
                    layer_start_nd[j,i] = 0.
                    layer_end_nd[j,i]   = 1.
                elif discrete_outputs['definition_layer'][j] == 2 or discrete_outputs['definition_layer'][j] == 3: # Midpoint and width
                    if discrete_outputs['definition_layer'][j] == 2:
                        layer_rotation[j,i] = - inputs['twist'][i]
                    else:
                        layer_rotation[j,i] = - outputs['layer_rotation'][j,i]
                    midpoint = calc_axis_intersection(inputs['coord_xy_dim'][i,:,:], layer_rotation[j,i], outputs['layer_offset_y_pa'][j,i], [0.,0.], [discrete_outputs['layer_side'][j]])[0]
                    width    = outputs['layer_width'][j,i]
                    layer_start_nd[j,i] = midpoint-width/arc_L_i/2.
                    layer_end_nd[j,i]   = midpoint+width/arc_L_i/2.
                elif discrete_outputs['definition_layer'][j] == 4: # Midpoint and width
                    midpoint = 1. 
                    outputs['layer_midpoint_nd'][j,i] = midpoint
                    width    = outputs['layer_width'][j,i]
                    layer_start_nd[j,i] = midpoint-width/arc_L_i/2.
                    layer_end_nd[j,i]   = width/arc_L_i/2.
                elif discrete_outputs['definition_layer'][j] == 5: # Midpoint and width
                    midpoint = LE_loc
                    outputs['layer_midpoint_nd'][j,i] = midpoint
                    width    = outputs['layer_width'][j,i]
                    layer_start_nd[j,i] = midpoint-width/arc_L_i/2.
                    layer_end_nd[j,i]   = midpoint+width/arc_L_i/2.
                elif discrete_outputs['definition_layer'][j] == 6: # Start and end locked to other element
                    # if outputs['layer_start_nd'][j,i] > 1:
                    layer_start_nd[j,i] = layer_end_nd[int(discrete_outputs['index_layer_start'][j]),i]
                    # if outputs['layer_end_nd'][j,i] > 1:
                    layer_end_nd[j,i]   = layer_start_nd[int(discrete_outputs['index_layer_end'][j]),i]
                elif discrete_outputs['definition_layer'][j] == 7: # Start nd and width
                    width    = outputs['layer_width'][j,i]
                    layer_start_nd[j,i] = outputs['layer_start_nd'][j,i]
                    layer_end_nd[j,i]   = layer_start_nd[j,i] + width/arc_L_i
                elif discrete_outputs['definition_layer'][j] == 8: # End nd and width
                    width    = outputs['layer_width'][j,i]
                    layer_end_nd[j,i]   = outputs['layer_end_nd'][j,i]
                    layer_start_nd[j,i] = layer_end_nd[j,i] - width/arc_L_i
                elif discrete_outputs['definition_layer'][j] == 9: # Start and end nd positions
                    layer_start_nd[j,i] = outputs['layer_start_nd'][j,i]
                    layer_end_nd[j,i]   = outputs['layer_end_nd'][j,i]
                elif discrete_outputs['definition_layer'][j] == 10: # Web layer
                    pass
                elif discrete_outputs['definition_layer'][j] == 11: # Start nd arc locked to LE
                    layer_start_nd[j,i] = LE_loc + 1.e-6
                    layer_end_nd[j,i]   = layer_start_nd[int(discrete_outputs['index_layer_end'][j]),i]
                elif discrete_outputs['definition_layer'][j] == 12: # End nd arc locked to LE
                    layer_end_nd[j,i] = LE_loc - 1.e-6
                    layer_start_nd[j,i] = layer_end_nd[int(discrete_outputs['index_layer_start'][j]),i]
                else:
                    exit('Blade layer ' + str(discrete_outputs['layer_name'][j]) + ' not described correctly. Please check the yaml input file.')
        
        # Assign openmdao outputs
        outputs['web_rotation']   = web_rotation
        outputs['web_start_nd']   = web_start_nd
        outputs['web_end_nd']     = web_end_nd
        outputs['layer_rotation'] = layer_rotation
        outputs['layer_start_nd'] = layer_start_nd
        outputs['layer_end_nd']   = layer_end_nd

class TE_Flaps(ExplicitComponent):
    # Openmdao component with the trailing edge flaps data coming from the input yaml file.
    def initialize(self):
        self.options.declare('blade_init_options')
        
    def setup(self):
        blade_init_options = self.options['blade_init_options']
        n_te_flaps = blade_init_options['n_te_flaps']

        self.add_output('te_flap_start', val=np.zeros(n_te_flaps),               desc='1D array of the start positions along blade span of the trailing edge flap(s). Only values between 0 and 1 are meaningful.')
        self.add_output('te_flap_end',   val=np.zeros(n_te_flaps),               desc='1D array of the end positions along blade span of the trailing edge flap(s). Only values between 0 and 1 are meaningful.')
        self.add_output('chord_start',   val=np.zeros(n_te_flaps),               desc='1D array of the positions along chord where the trailing edge flap(s) start. Only values between 0 and 1 are meaningful.')
        self.add_output('delta_max_pos', val=np.zeros(n_te_flaps), units='rad',  desc='1D array of the max angle of the trailing edge flaps.')
        self.add_output('delta_max_neg', val=np.zeros(n_te_flaps), units='rad',  desc='1D array of the min angle of the trailing edge flaps.')

class Hub(ExplicitComponent):
    # Openmdao component with the hub data coming from the input yaml file.
    def setup(self):
        self.add_output('diameter',     val=0.0, units='m',     desc='Diameter of the hub. It is equal to two times the distance of the blade root from the rotor center along the coned line.')
        self.add_output('cone',         val=0.0, units='rad',   desc='Cone angle of the rotor. It defines the angle between the rotor plane and the blade pitch axis. A standard machine has positive values.')
        self.add_output('drag_coeff',   val=0.0,                desc='Drag coefficient to estimate the aerodynamic forces generated by the hub.')

        self.add_output('radius',       val=0.0, units='m',     desc='Radius of the hub. It defines the distance of the blade root from the rotor center along the coned line.')

    def compute(self, inputs, outputs):

        outputs['radius'] = 0.5 * outputs['diameter']

class Nacelle(ExplicitComponent):
    # Openmdao component with the nacelle data coming from the input yaml file.
    def setup(self):
        self.add_output('uptilt',           val=0.0, units='rad',   desc='Nacelle uptilt angle. A standard machine has positive values.')
        self.add_output('distance_tt_hub',  val=0.0, units='m',     desc='Vertical distance from tower top to hub center.')
        self.add_output('overhang',         val=0.0, units='m',     desc='Horizontal distance from tower top to hub center.')
        # Drivetrain parameters
        self.add_output('gear_ratio',       val=0.0)
        self.add_output('shaft_ratio',      val=0.0)
        self.add_discrete_output('planet_numbers',   val=np.zeros(3))
        self.add_output('shrink_disc_mass', val=0.0, units='kg')
        self.add_output('carrier_mass',     val=0.0, units='kg')
        self.add_output('flange_length',    val=0.0, units='m')
        self.add_output('gearbox_input_xcm',val=0.0, units='m')
        self.add_output('hss_input_length', val=0.0, units='m')
        self.add_output('distance_hub2mb',  val=0.0, units='m')
        self.add_discrete_output('yaw_motors_number', val = 0)
        self.add_output('drivetrain_eff',   val=0.0)

class Tower(ExplicitComponent):
    # Openmdao component with the tower data coming from the input yaml file.
    def initialize(self):
        self.options.declare('tower_init_options')
        
    def setup(self):
        tower_init_options = self.options['tower_init_options']
        n_height           = tower_init_options['n_height']
        n_layers           = tower_init_options['n_layers']
                
        self.add_output('s',        val=np.zeros(n_height),                 desc='1D array of the non-dimensional grid defined along the tower axis (0-tower base, 1-tower top)')
        self.add_output('diameter', val=np.zeros(n_height),     units='m',  desc='1D array of the outer diameter values defined along the tower axis.')
        self.add_output('drag',     val=np.zeros(n_height),                 desc='1D array of the drag coefficients defined along the tower axis.')
        self.add_output('ref_axis', val=np.zeros((n_height,3)), units='m',  desc='2D array of the coordinates (x,y,z) of the tower reference axis. The coordinate system is the global coordinate system of OpenFAST: it is placed at tower base with x pointing downwind, y pointing on the side and z pointing vertically upwards. A standard tower configuration will have zero x and y values and positive z values.')

        self.add_discrete_output('layer_name', val=n_layers * [''],         desc='1D array of the names of the layers modeled in the tower structure.')
        self.add_discrete_output('layer_mat',  val=n_layers * [''],         desc='1D array of the names of the materials of each layer modeled in the tower structure.')
        self.add_output('layer_thickness',     val=np.zeros((n_layers, n_height)), units='m',    desc='2D array of the thickness of the layers of the tower structure. The first dimension represents each layer, the second dimension represents each entry along the tower axis.')

        self.add_output('height',   val = 0.0,                  units='m',  desc='Scalar of the tower height computed along the z axis.')
        self.add_output('length',   val = 0.0,                  units='m',  desc='Scalar of the tower length computed along its curved axis. A standard straight tower will be as high as long.')
        
        self.add_output('mass',   val = 0.0,                  units='kg',  desc='Temporary tower mass')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Compute tower height and tower length (a straight tower will be high as long)
        outputs['height']   = outputs['ref_axis'][-1,2]
        outputs['length']   = arc_length(outputs['ref_axis'][:,0], outputs['ref_axis'][:,1], outputs['ref_axis'][:,2])[-1]

        rhoA = np.pi * outputs['diameter'] * outputs['layer_thickness'][0,:]
        outputs['mass'] = np.trapz(rhoA, outputs['ref_axis'][:,2]) * 8500.

class Foundation(ExplicitComponent):
    # Openmdao component with the foundation data coming from the input yaml file.
    def setup(self):
        self.add_output('height',           val=0.0, units='m',     desc='Foundation height in respect to the ground level.')

class Airfoils(ExplicitComponent):
    def initialize(self):
        self.options.declare('af_init_options')
    
    def setup(self):
        af_init_options = self.options['af_init_options']
        n_af            = af_init_options['n_af'] # Number of airfoils
        n_aoa           = af_init_options['n_aoa']# Number of angle of attacks
        n_Re            = af_init_options['n_Re'] # Number of Reynolds, so far hard set at 1
        n_tab           = af_init_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        n_xy            = af_init_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
        
        # Airfoil properties
        self.add_discrete_output('name', val=n_af * [''],                        desc='1D array of names of airfoils.')
        self.add_output('ac',        val=np.zeros(n_af),                         desc='1D array of the aerodynamic centers of each airfoil.')
        self.add_output('r_thick',   val=np.zeros(n_af),                         desc='1D array of the relative thicknesses of each airfoil.')
        self.add_output('aoa',       val=np.zeros(n_aoa),        units='rad',    desc='1D array of the angles of attack used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.')
        self.add_output('Re',        val=np.zeros(n_Re),                         desc='1D array of the Reynolds numbers used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.')
        self.add_output('tab',       val=np.zeros(n_tab),                        desc='1D array of the values of the "tab" entity used to define the polars of the airfoils. All airfoils defined in openmdao share this grid. The tab could for example represent a flap deflection angle.')
        self.add_output('cl',        val=np.zeros((n_af, n_aoa, n_Re, n_tab)),   desc='4D array with the lift coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_output('cd',        val=np.zeros((n_af, n_aoa, n_Re, n_tab)),   desc='4D array with the drag coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_output('cm',        val=np.zeros((n_af, n_aoa, n_Re, n_tab)),   desc='4D array with the moment coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        
        # Airfoil coordinates
        self.add_output('coord_xy',  val=np.zeros((n_af, n_xy, 2)),              desc='3D array of the x and y airfoil coordinates of the n_af airfoils.')

class Materials(ExplicitComponent):
    # Openmdao component with the wind turbine materials coming from the input yaml file. The inputs and outputs are arrays where each entry represents a material
    
    def initialize(self):
        self.options.declare('mat_init_options')
    
    def setup(self):
        
        mat_init_options = self.options['mat_init_options']
        self.n_mat = n_mat = mat_init_options['n_mat']
        
        self.add_discrete_output('name', val=n_mat * [''],                         desc='1D array of names of materials.')
        self.add_discrete_output('orth', val=np.zeros(n_mat),                      desc='1D array of flags to set whether a material is isotropic (0) or orthtropic (1). Each entry represents a material.')
        self.add_discrete_output('component_id', val=-np.ones(n_mat),              desc='1D array of flags to set whether a material is used in a blade: 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.isotropic.')
        
        self.add_output('E',             val=np.zeros([n_mat, 3]), units='Pa',     desc='2D array of the Youngs moduli of the materials. Each row represents a material, the three columns represent E11, E22 and E33.')
        self.add_output('G',             val=np.zeros([n_mat, 3]), units='Pa',     desc='2D array of the shear moduli of the materials. Each row represents a material, the three columns represent G12, G13 and G23.')
        self.add_output('nu',            val=np.zeros([n_mat, 3]),                 desc='2D array of the Poisson ratio of the materials. Each row represents a material, the three columns represent nu12, nu13 and nu23.')
        self.add_output('Xt',            val=np.zeros([n_mat, 3]),                 desc='2D array of the Ultimate Tensile Strength (UTS) of the materials. Each row represents a material, the three columns represent Xt12, Xt13 and Xt23.')
        self.add_output('Xc',            val=np.zeros([n_mat, 3]),                 desc='2D array of the Ultimate Compressive Strength (UCS) of the materials. Each row represents a material, the three columns represent Xc12, Xc13 and Xc23.')
        self.add_output('rho',           val=np.zeros(n_mat),      units='kg/m**3',desc='1D array of the density of the materials. For composites, this is the density of the laminate.')
        self.add_output('unit_cost',     val=np.zeros(n_mat),      units='USD/kg', desc='1D array of the unit costs of the materials.')
        self.add_output('waste',         val=np.zeros(n_mat),                      desc='1D array of the non-dimensional waste fraction of the materials.')
        self.add_output('rho_fiber',     val=np.zeros(n_mat),      units='kg/m**3',desc='1D array of the density of the fibers of the materials.')
        self.add_output('rho_area_dry',  val=np.zeros(n_mat),      units='kg/m**2',desc='1D array of the dry aerial density of the composite fabrics. Non-composite materials are kept at 0.')
        self.add_output('roll_mass',     val=np.zeros(n_mat),      units='kg',     desc='1D array of the roll mass of the composite fabrics. Non-composite materials are kept at 0.')
        
        self.add_output('ply_t',        val=np.zeros(n_mat),      units='m',      desc='1D array of the ply thicknesses of the materials. Non-composite materials are kept at 0.')
        self.add_output('fvf',          val=np.zeros(n_mat),                      desc='1D array of the non-dimensional fiber volume fraction of the composite materials. Non-composite materials are kept at 0.')
        self.add_output('fwf',          val=np.zeros(n_mat),                      desc='1D array of the non-dimensional fiber weight- fraction of the composite materials. Non-composite materials are kept at 0.')
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        density_resin = 0.
        for i in range(self.n_mat):
            if discrete_outputs['name'][i] == 'resin':
                density_resin = outputs['rho'][i]
                id_resin = i
        if density_resin==0.:
            exit('Error: a material named resin must be defined in the input yaml')
        
        fvf   = np.zeros(self.n_mat)
        fwf   = np.zeros(self.n_mat)
        ply_t = np.zeros(self.n_mat)
        
        for i in range(self.n_mat):
            if discrete_outputs['component_id'][i] > 1: # It's a composite
                # Formula to estimate the fiber volume fraction fvf from the laminate and the fiber densities
                fvf[i]  = (outputs['rho'][i] - density_resin) / (outputs['rho_fiber'][i] - density_resin) 
                if outputs['fvf'][i] > 0.:
                    if abs(fvf[i] - outputs['fvf'][i]) > 1e-3:
                        exit('Error: the fvf of composite ' + discrete_outputs['name'][i] + ' specified in the yaml is equal to '+ str(outputs['fvf'][i] * 100) + '%, but this value is not compatible to the other values provided. Given the fiber, laminate and resin densities, it should instead be equal to ' + str(fvf[i]*100.) + '%.')
                else:
                    outputs['fvf'][i] = fvf[i]
                # Formula to estimate the fiber weight fraction fwf from the fiber volume fraction and the fiber densities
                fwf[i]  = outputs['rho_fiber'][i] * outputs['fvf'][i] / (density_resin + ((outputs['rho_fiber'][i] - density_resin) * outputs['fvf'][i]))
                if outputs['fwf'][i] > 0.:
                    if abs(fwf[i] - outputs['fwf'][i]) > 1e-3:
                        exit('Error: the fwf of composite ' + discrete_outputs['name'][i] + ' specified in the yaml is equal to '+ str(outputs['fwf'][i] * 100) + '%, but this value is not compatible to the other values provided. It should instead be equal to ' + str(fwf[i]*100.) + '%')
                else:
                    outputs['fwf'][i] = fwf[i]
                # Formula to estimate the plyt thickness ply_t of a laminate from the aerial density, the laminate density and the fiber weight fraction
                ply_t[i] = outputs['rho_area_dry'][i] / outputs['rho'][i] / outputs['fwf'][i]
                if outputs['ply_t'][i] > 0.:
                    if abs(ply_t[i] - outputs['ply_t'][i]) > 1.e-4:
                        exit('Error: the ply_t of composite ' + discrete_outputs['name'][i] + ' specified in the yaml is equal to '+ str(outputs['ply_t'][i]) + 'm, but this value is not compatible to the other values provided. It should instead be equal to ' + str(ply_t[i]) + 'm')
                else:
                    outputs['ply_t'][i] = ply_t[i]      

class Control(ExplicitComponent):
    # Openmdao component with the wind turbine controller data coming from the input yaml file.
    def setup(self):

        self.add_output('rated_power',      val=0.0, units='W',           desc='Electrical rated power of the generator.')
        self.add_output('V_in',             val=0.0, units='m/s',         desc='Cut in wind speed. This is the wind speed where region II begins.')
        self.add_output('V_out',            val=0.0, units='m/s',         desc='Cut out wind speed. This is the wind speed where region III ends.')
        self.add_output('minOmega',         val=0.0, units='rad/s',        desc='Minimum allowed rotor speed.')
        self.add_output('maxOmega',         val=0.0, units='rad/s',        desc='Maximum allowed rotor speed.')
        self.add_output('max_TS',           val=0.0, units='m/s',         desc='Maximum allowed blade tip speed.')
        self.add_output('max_pitch_rate',   val=0.0, units='rad/s',        desc='Maximum allowed blade pitch rate')
        self.add_output('max_torque_rate',  val=0.0, units='N*m/s',       desc='Maximum allowed generator torque rate')
        self.add_output('rated_TSR',        val=0.0,                      desc='Constant tip speed ratio in region II.')
        self.add_output('rated_pitch',      val=0.0, units='rad',         desc='Constant pitch angle in region II.')
        self.add_output('PC_omega',         val=0.0, units='rad/s',       desc='Pitch controller natural frequency')
        self.add_output('PC_zeta',          val=0.0,                      desc='Pitch controller damping ratio')
        self.add_output('VS_omega',         val=0.0, units='rad/s',       desc='Generator torque controller natural frequency')
        self.add_output('VS_zeta',          val=0.0,                      desc='Generator torque controller damping ratio')
        # optional inputs - not connected right now!!
        self.add_output('max_pitch',        val=0.0, units='rad',       desc='Maximum pitch angle , {default = 90 degrees}')
        self.add_output('min_pitch',        val=0.0, units='rad',       desc='Minimum pitch angle [rad], {default = 0 degrees}')
        self.add_output('vs_minspd',        val=0.0, units='rad/s',     desc='Minimum rotor speed [rad/s], {default = 0 rad/s}')
        self.add_output('ss_cornerfreq',    val=0.0, units='rad/s',     desc='First order low-pass filter cornering frequency for setpoint smoother [rad/s]')
        self.add_output('ss_vsgain',        val=0.0,                    desc='Torque controller setpoint smoother gain bias percentage [%, <= 1 ], {default = 100%}')
        self.add_output('ss_pcgain',        val=0.0,                    desc='Pitch controller setpoint smoother gain bias percentage  [%, <= 1 ], {default = 0.1%}')
        self.add_output('ps_percent',       val=0.0,                    desc='Percent peak shaving  [%, <= 1 ], {default = 80%}')
        self.add_output('sd_maxpit',        val=0.0, units='rad',       desc='Maximum blade pitch angle to initiate shutdown [rad], {default = bld pitch at v_max}')
        self.add_output('sd_cornerfreq',    val=0.0, units='rad/s',     desc='Cutoff Frequency for first order low-pass filter for blade pitch angle [rad/s], {default = 0.41888 ~ time constant of 15s}')
        self.add_output('Kp_flap',          val=0.0, units='s',         desc='Proportional term of the PI controller for the trailing-edge flaps')
        self.add_output('Ki_flap',          val=0.0,                    desc='Integral term of the PI controller for the trailing-edge flaps')
        
class Configuration(ExplicitComponent):
    # Openmdao component with the wind turbine configuration data (class, number of blades, upwind vs downwind, ...) coming from the input yaml file.
    def setup(self):

        self.add_discrete_output('ws_class',            val='', desc='IEC wind turbine class. I - offshore, II coastal, III - land-based, IV - low wind speed site.')
        self.add_discrete_output('turb_class',          val='', desc='IEC wind turbine category. A - high turbulence intensity (land-based), B - mid turbulence, C - low turbulence (offshore).')
        self.add_discrete_output('gearbox_type',        val='geared', desc='Gearbox configuration (geared, direct-drive, etc.).')
        self.add_discrete_output('rotor_orientation',   val='upwind', desc='Rotor orientation, either upwind or downwind.')
        self.add_discrete_output('n_blades',            val=3,        desc='Number of blades of the rotor.')

class Environment(ExplicitComponent):
    # Openmdao component with the environmental parameters
    def setup(self):

        self.add_output('rho_air',      val=1.225,        units='kg/m**3',    desc='Density of air')
        self.add_output('mu_air',       val=1.81e-5,      units='kg/(m*s)',   desc='Dynamic viscosity of air')
        self.add_output('weibull_k',    val=2.0,          desc='Shape parameter of the Weibull probability density function of the wind.')
        self.add_output('shear_exp',    val=0.2,          desc='Shear exponent of the wind.')
        self.add_output('speed_sound_air',  val=340.,     units='m/s',        desc='Speed of sound in air.')

class Costs(ExplicitComponent):
    # Openmdao component with the cost parameters
    def setup(self):

        self.add_discrete_output('turbine_number',    val=0,             desc='Number of turbines at plant')
        self.add_output('bos_per_kW',        val=0.0, units='USD/kW',    desc='Balance of system costs of the turbine')
        self.add_output('opex_per_kW',       val=0.0, units='USD/kW/yr', desc='Average annual operational expenditures of the turbine')
        self.add_output('wake_loss_factor',  val=0.0,                    desc='The losses in AEP due to waked conditions')
        self.add_output('fixed_charge_rate', val=0.0,                    desc = 'Fixed charge rate for coe calculation')

class WT_Assembly(ExplicitComponent):
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

class WindTurbineOntologyOpenMDAO(Group):
    # Openmdao group with all wind turbine data
    
    def initialize(self):
        self.options.declare('analysis_options')
        
    def setup(self):
        analysis_options = self.options['analysis_options']
        self.add_subsystem('materials', Materials(mat_init_options = analysis_options['materials']))
        self.add_subsystem('airfoils',  Airfoils(af_init_options   = analysis_options['airfoils']))
        
        self.add_subsystem('blade',         Blade(blade_init_options   = analysis_options['blade'], af_init_options   = analysis_options['airfoils']))
        self.add_subsystem('hub',           Hub())
        self.add_subsystem('nacelle',       Nacelle())
        self.add_subsystem('tower',         Tower(tower_init_options   = analysis_options['tower']))
        self.add_subsystem('foundation',    Foundation())
        self.add_subsystem('control',       Control())
        self.add_subsystem('configuration', Configuration())
        self.add_subsystem('env',           Environment())
        self.add_subsystem('assembly',      WT_Assembly(blade_init_options   = analysis_options['blade']))
        self.add_subsystem('costs',         Costs())

        self.connect('airfoils.name',    'blade.interp_airfoils.name')
        self.connect('airfoils.r_thick', 'blade.interp_airfoils.r_thick')
        self.connect('airfoils.coord_xy','blade.interp_airfoils.coord_xy')
        self.connect('airfoils.aoa',     'blade.interp_airfoils.aoa')
        self.connect('airfoils.cl',      'blade.interp_airfoils.cl')
        self.connect('airfoils.cd',      'blade.interp_airfoils.cd')
        self.connect('airfoils.cm',      'blade.interp_airfoils.cm')

        self.connect('blade.outer_shape_bem.ref_axis',  'assembly.blade_ref_axis')
        self.connect('hub.radius',                      'assembly.hub_radius')
        self.connect('tower.height',                    'assembly.tower_height')
        self.connect('foundation.height',               'assembly.foundation_height')
        self.connect('nacelle.distance_tt_hub',         'assembly.distance_tt_hub')
        
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
    wt_opt = assign_nacelle_values(wt_opt, nacelle)
    wt_opt = assign_tower_values(wt_opt, analysis_options, tower)
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
    
    wt_opt['blade.outer_shape_bem.af_used']     = outer_shape_bem['airfoil_position']['labels']
    wt_opt['blade.outer_shape_bem.af_position'] = outer_shape_bem['airfoil_position']['grid']
    
    wt_opt['blade.outer_shape_bem.s']           = nd_span
    wt_opt['blade.outer_shape_bem.chord']       = np.interp(nd_span, outer_shape_bem['chord']['grid'], outer_shape_bem['chord']['values'])
    wt_opt['blade.outer_shape_bem.twist']       = np.interp(nd_span, outer_shape_bem['twist']['grid'], outer_shape_bem['twist']['values'])
    wt_opt['blade.outer_shape_bem.pitch_axis']  = np.interp(nd_span, outer_shape_bem['pitch_axis']['grid'], outer_shape_bem['pitch_axis']['values'])
    
    wt_opt['blade.outer_shape_bem.ref_axis'][:,0]  = np.interp(nd_span, outer_shape_bem['reference_axis']['x']['grid'], outer_shape_bem['reference_axis']['x']['values'])
    wt_opt['blade.outer_shape_bem.ref_axis'][:,1]  = np.interp(nd_span, outer_shape_bem['reference_axis']['y']['grid'], outer_shape_bem['reference_axis']['y']['values'])
    wt_opt['blade.outer_shape_bem.ref_axis'][:,2]  = np.interp(nd_span, outer_shape_bem['reference_axis']['z']['grid'], outer_shape_bem['reference_axis']['z']['values'])
    
    return wt_opt
    
def assign_internal_structure_2d_fem_values(wt_opt, analysis_options, internal_structure_2d_fem):
    # Function to assign values to the openmdao component Blade_Internal_Structure_2D_FEM
    
    n_span          = analysis_options['blade']['n_span']
    n_webs          = analysis_options['blade']['n_webs']
    
    web_name        = n_webs * ['']
    web_rotation    = np.zeros((n_webs, n_span))
    web_offset_y_pa = np.zeros((n_webs, n_span))
    definition_web  = np.zeros(n_webs)
    nd_span         = wt_opt['blade.outer_shape_bem.s']
    
    # Loop through the webs and interpolate spanwise the values
    for i in range(n_webs):
        web_name[i] = internal_structure_2d_fem['webs'][i]['name']
        if 'rotation' in internal_structure_2d_fem['webs'][i] and 'offset_y_pa' in internal_structure_2d_fem['webs'][i]:
            if 'fixed' in internal_structure_2d_fem['webs'][i]['rotation'].keys():
                if internal_structure_2d_fem['webs'][i]['rotation']['fixed'] == 'twist':
                    definition_web[i] = 1
                else:
                    exit('Invalid rotation reference for web ' + web_name[i] + '. Please check the yaml input file')
            else:
                web_rotation[i,:] = np.interp(nd_span, internal_structure_2d_fem['webs'][i]['rotation']['grid'], internal_structure_2d_fem['webs'][i]['rotation']['values'], left=0., right=0.)
                definition_web[i] = 2
            web_offset_y_pa[i,:] = np.interp(nd_span, internal_structure_2d_fem['webs'][i]['offset_y_pa']['grid'], internal_structure_2d_fem['webs'][i]['offset_y_pa']['values'], left=0., right=0.)
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
    layer_web       = n_layers * ['']
    layer_side      = n_layers * ['']
    definition_layer= np.zeros(n_layers)
    index_layer_start= np.zeros(n_layers)
    index_layer_end = np.zeros(n_layers)

    
    # Loop through the layers, interpolate along blade span, assign the inputs, and the definition flag
    for i in range(n_layers):
        layer_name[i]  = internal_structure_2d_fem['layers'][i]['name']
        layer_mat[i]   = internal_structure_2d_fem['layers'][i]['material']
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
                    layer_start_nd[i,:] = np.ones(n_span)
                    exit('No need to fix element to TE, set it to 0.')
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
                        exit('No need to fix element to TE, set it to 0.')
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
                    exit('No need to fix element to TE, set it to 0.')
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
            layer_web[i] = internal_structure_2d_fem['layers'][i]['web']
            definition_layer[i] = 10
    
    
    # Assign the openmdao values
    wt_opt['blade.internal_structure_2d_fem.web_name']          = web_name
    wt_opt['blade.internal_structure_2d_fem.s']                 = nd_span
    wt_opt['blade.internal_structure_2d_fem.web_rotation']      = web_rotation
    wt_opt['blade.internal_structure_2d_fem.web_offset_y_pa']   = web_offset_y_pa
    
    wt_opt['blade.internal_structure_2d_fem.layer_name']        = layer_name
    wt_opt['blade.internal_structure_2d_fem.layer_mat']         = layer_mat
    wt_opt['blade.internal_structure_2d_fem.layer_side']        = layer_side
    wt_opt['blade.internal_structure_2d_fem.layer_thickness']   = thickness
    wt_opt['blade.internal_structure_2d_fem.layer_rotation']    = layer_rotation
    wt_opt['blade.internal_structure_2d_fem.layer_offset_y_pa'] = layer_offset_y_pa
    wt_opt['blade.internal_structure_2d_fem.layer_width']       = layer_width
    wt_opt['blade.internal_structure_2d_fem.layer_midpoint_nd'] = layer_midpoint_nd
    wt_opt['blade.internal_structure_2d_fem.layer_start_nd']    = layer_start_nd
    wt_opt['blade.internal_structure_2d_fem.layer_end_nd']      = layer_end_nd
    wt_opt['blade.internal_structure_2d_fem.layer_web']         = layer_web
    wt_opt['blade.internal_structure_2d_fem.definition_web']    = definition_web
    wt_opt['blade.internal_structure_2d_fem.definition_layer']  = definition_layer
    wt_opt['blade.internal_structure_2d_fem.index_layer_start'] = index_layer_start
    wt_opt['blade.internal_structure_2d_fem.index_layer_end']   = index_layer_end
    
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

            wt_opt['param.opt_var.te_flap_ext'] = blade['aerodynamic_control']['te_flaps'][i]['span_end'] - blade['aerodynamic_control']['te_flaps'][i]['span_start']
            wt_opt['param.opt_var.te_flap_end'] = blade['aerodynamic_control']['te_flaps'][i]['span_end']

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

    return wt_opt

def assign_nacelle_values(wt_opt, nacelle):

    wt_opt['nacelle.uptilt']            = nacelle['outer_shape_bem']['uptilt_angle']
    wt_opt['nacelle.distance_tt_hub']   = nacelle['outer_shape_bem']['distance_tt_hub']
    wt_opt['nacelle.overhang']          = nacelle['outer_shape_bem']['overhang']
    
    wt_opt['nacelle.gear_ratio']        = nacelle['drivetrain']['gear_ratio']
    wt_opt['nacelle.shaft_ratio']       = nacelle['drivetrain']['shaft_ratio']
    wt_opt['nacelle.planet_numbers']    = nacelle['drivetrain']['planet_numbers']
    wt_opt['nacelle.shrink_disc_mass']  = nacelle['drivetrain']['shrink_disc_mass']
    wt_opt['nacelle.carrier_mass']      = nacelle['drivetrain']['carrier_mass']
    wt_opt['nacelle.flange_length']     = nacelle['drivetrain']['flange_length']
    wt_opt['nacelle.gearbox_input_xcm'] = nacelle['drivetrain']['gearbox_input_xcm']
    wt_opt['nacelle.hss_input_length']  = nacelle['drivetrain']['hss_input_length']
    wt_opt['nacelle.distance_hub2mb']   = nacelle['drivetrain']['distance_hub2mb']
    wt_opt['nacelle.yaw_motors_number'] = nacelle['drivetrain']['yaw_motors_number']
    wt_opt['nacelle.drivetrain_eff']    = nacelle['drivetrain']['efficiency']

    return wt_opt

def assign_tower_values(wt_opt, analysis_options, tower):
    # Function to assign values to the openmdao component Tower
    n_height        = analysis_options['tower']['n_height'] # Number of points along tower height
    nd_height       = analysis_options['tower']['nd_height']# Non-dimensional height coordinate
    n_layers        = analysis_options['tower']['n_layers']
    
    layer_name      = n_layers * ['']
    layer_mat       = n_layers * ['']
    thickness       = np.zeros((n_layers, n_height))

    wt_opt['tower.s']          = nd_height
    wt_opt['tower.diameter']   = np.interp(nd_height, tower['outer_shape_bem']['outer_diameter']['grid'], tower['outer_shape_bem']['outer_diameter']['values'])
    wt_opt['tower.drag']       = np.interp(nd_height, tower['outer_shape_bem']['drag_coefficient']['grid'], tower['outer_shape_bem']['drag_coefficient']['values'])
    
    wt_opt['tower.ref_axis'][:,0]  = np.interp(nd_height, tower['outer_shape_bem']['reference_axis']['x']['grid'], tower['outer_shape_bem']['reference_axis']['x']['values'])
    wt_opt['tower.ref_axis'][:,1]  = np.interp(nd_height, tower['outer_shape_bem']['reference_axis']['y']['grid'], tower['outer_shape_bem']['reference_axis']['y']['values'])
    wt_opt['tower.ref_axis'][:,2]  = np.interp(nd_height, tower['outer_shape_bem']['reference_axis']['z']['grid'], tower['outer_shape_bem']['reference_axis']['z']['values'])

    for i in range(n_layers):
        layer_name[i]  = tower['internal_structure_2d_fem']['layers'][i]['name']
        layer_mat[i]   = tower['internal_structure_2d_fem']['layers'][i]['material']
        thickness[i]   = np.interp(nd_height, tower['internal_structure_2d_fem']['layers'][i]['thickness']['grid'], tower['internal_structure_2d_fem']['layers'][i]['thickness']['values'], left=0., right=0.)

    wt_opt['tower.layer_name']        = layer_name
    wt_opt['tower.layer_mat']         = layer_mat
    wt_opt['tower.layer_thickness']   = thickness

    return wt_opt

def assign_foundation_values(wt_opt, foundation):

    wt_opt['foundation.height']    = foundation['height']

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
    # # other optional parameters
    wt_opt['control.max_pitch']     = control['max_pitch']
    wt_opt['control.min_pitch']     = control['min_pitch']
    wt_opt['control.vs_minspd']     = control['vs_minspd']
    wt_opt['control.ss_vsgain']     = control['ss_vsgain']
    wt_opt['control.ss_pcgain']     = control['ss_pcgain']
    wt_opt['control.ps_percent']    = control['ps_percent']
    if analysis_options['servose']['Flp_Mode'] >= 1:
        wt_opt['control.Kp_flap']       = control['Kp_flap']
        wt_opt['control.Ki_flap']       = control['Ki_flap']
    
    
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
    wt_opt['env.weibull_k'] = environment['weib_shape_parameter']
    wt_opt['env.speed_sound_air'] = environment['air_speed_sound']
    wt_opt['env.shear_exp'] = environment['shear_exp']

    return wt_opt

def assign_costs_values(wt_opt, costs):

    wt_opt['costs.turbine_number']      = costs['turbine_number']
    wt_opt['costs.bos_per_kW']          = costs['bos_per_kW']
    wt_opt['costs.opex_per_kW']         = costs['opex_per_kW']
    wt_opt['costs.wake_loss_factor']    = costs['wake_loss_factor']
    wt_opt['costs.fixed_charge_rate']   = costs['fixed_charge_rate']

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
                print("Airfoil " + name[i] + ' has the lift coefficient at Re ' + str(Re_j) + ' different between + and - pi rad. This is fixed automatically, but please check the input data.')
            if abs(cd[i,0,j,0] - cd[i,-1,j,0]) > 1.e-5:
                cd[i,0,j,0] = cd[i,-1,j,0]
                print("Airfoil " + name[i] + ' has the drag coefficient at Re ' + str(Re_j) + ' different between + and - pi rad. This is fixed automatically, but please check the input data.')
            if abs(cm[i,0,j,0] - cm[i,-1,j,0]) > 1.e-5:
                cm[i,0,j,0] = cm[i,-1,j,0]
                print("Airfoil " + name[i] + ' has the moment coefficient at Re ' + str(Re_j) + ' different between + and - pi rad. This is fixed automatically, but please check the input data.')
        
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
        name[i] =  materials[i]['name']
        orth[i] =  materials[i]['orth']
        rho[i]  =  materials[i]['rho']
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
                warning_shear_modulus_isotropic = 'Ontology input warning: No shear modulus, G, provided for material "%s".  Assuming 2G*(1 + nu) = E, which is only valid for isotropic materials.'%name[i]
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
            
            
    wt_opt['materials.name']     = name
    wt_opt['materials.orth']     = orth
    wt_opt['materials.rho']      = rho
    wt_opt['materials.component_id']= component_id
    wt_opt['materials.E']        = E
    wt_opt['materials.G']        = G
    wt_opt['materials.Xt']       = Xt
    wt_opt['materials.Xc']       = Xc
    wt_opt['materials.nu']       = nu
    wt_opt['materials.rho_fiber']      = rho_fiber
    wt_opt['materials.rho_area_dry']   = rho_area_dry
    wt_opt['materials.fvf']      = fvf
    wt_opt['materials.fwf']      = fwf
    wt_opt['materials.ply_t']    = ply_t
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
    wt_opt          = Problem()
    wt_opt.model    = WindTurbineOntologyOpenMDAO(analysis_options = analysis_options)
    wt_opt.setup()
    # Load wind turbine data from wt_initial to the openmdao problem
    wt_opt = yaml2openmdao(wt_opt, analysis_options, wt_init)
    wt_opt.run_driver()
    
    # Save data coming from openmdao to an output yaml file
    wt_initial.write_ontology(wt_opt, fname_output)
