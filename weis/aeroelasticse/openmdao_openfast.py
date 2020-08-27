import numpy as np
import os, copy, warnings, shutil, sys
from scipy.interpolate                      import PchipInterpolator
from openmdao.api                           import ExplicitComponent
from wisdem.commonse.mpi_tools              import MPI
from wisdem.commonse.vertical_cylinder      import NFREQ
from wisdem.towerse.tower                   import get_nfull
from wisdem.servose.servose                 import eval_unsteady
from wisdem.rotorse.geometry_tools.geometry import remap2grid
from weis.aeroelasticse.FAST_writer       import InputWriter_OpenFAST
from weis.aeroelasticse.runFAST_pywrapper import runFAST_pywrapper, runFAST_pywrapper_batch
from weis.aeroelasticse.FAST_post         import FAST_IO_timeseries
from weis.aeroelasticse.CaseGen_IEC       import CaseGen_General, CaseGen_IEC

from pCrunch import Analysis, pdTools
import fatpack

if MPI:
    from mpi4py   import MPI
    from petsc4py import PETSc

class FASTLoadCases(ExplicitComponent):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        blade_init_options   = self.options['modeling_options']['blade']
        servose_init_options = self.options['modeling_options']['servose']
        openfast_init_options = self.options['modeling_options']['openfast']
        mat_init_options     = self.options['modeling_options']['materials']

        self.n_span        = n_span    = blade_init_options['n_span']
        self.n_pc          = n_pc      = servose_init_options['n_pc']
        n_OF     = len(openfast_init_options['dlc_settings']['Power_Curve']['U'])
        self.n_pitch       = n_pitch   = servose_init_options['n_pitch_perf_surfaces']
        self.n_tsr         = n_tsr     = servose_init_options['n_tsr_perf_surfaces']
        self.n_U           = n_U       = servose_init_options['n_U_perf_surfaces']
        self.n_mat         = n_mat    = mat_init_options['n_mat']
        self.n_layers      = n_layers = blade_init_options['n_layers']

        af_init_options    = self.options['modeling_options']['airfoils']
        self.n_xy          = n_xy      = af_init_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
        self.n_aoa         = n_aoa     = af_init_options['n_aoa']# Number of angle of attacks
        self.n_Re          = n_Re      = af_init_options['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab         = n_tab     = af_init_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        
        self.te_ss_var       = self.options['opt_options']['optimization_variables']['blade']['structure']['te_ss']['name']
        self.te_ps_var       = self.options['opt_options']['optimization_variables']['blade']['structure']['te_ps']['name']
        self.spar_cap_ss_var = self.options['opt_options']['optimization_variables']['blade']['structure']['spar_cap_ss']['name']
        self.spar_cap_ps_var = self.options['opt_options']['optimization_variables']['blade']['structure']['spar_cap_ps']['name']

        monopile     = self.options['modeling_options']['flags']['monopile']
        n_height_tow = self.options['modeling_options']['tower']['n_height']
        n_height_mon = 0 if not monopile else self.options['modeling_options']['monopile']['n_height']
        n_height     = n_height_tow if n_height_mon==0 else n_height_tow + n_height_mon - 1 # Should have one overlapping point
        nFull        = get_nfull(n_height)
        N_beam       = (nFull-1)*2
        n_freq_tower = int(NFREQ/2)
        n_freq_blade = int(self.options['modeling_options']['blade']['n_freq']/2)

        FASTpref = self.options['modeling_options']['openfast']
        # self.FatigueFile   = self.options['modeling_options']['rotorse']['FatigueFile']
        
        # ElastoDyn Inputs
        # Assuming the blade modal damping to be unchanged. Cannot directly solve from the Rayleigh Damping without making assumptions. J.Jonkman recommends 2-3% https://wind.nrel.gov/forum/wind/viewtopic.php?t=522
        self.add_input('r',                     val=np.zeros(n_span), units='m', desc='radial positions. r[0] should be the hub location \
            while r[-1] should be the blade tip. Any number \
            of locations can be specified between these in ascending order.')
        self.add_input('le_location',           val=np.zeros(n_span), desc='Leading-edge positions from a reference blade axis (usually blade pitch axis). Locations are normalized by the local chord length. Positive in -x direction for airfoil-aligned coordinate system')
        self.add_input('beam:Tw_iner',          val=np.zeros(n_span), units='m', desc='y-distance to elastic center from point about which above structural properties are computed')
        self.add_input('beam:rhoA',             val=np.zeros(n_span), units='kg/m', desc='mass per unit length')
        self.add_input('beam:EIyy',             val=np.zeros(n_span), units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_input('beam:EIxx',             val=np.zeros(n_span), units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_input('x_tc',                  val=np.zeros(n_span), units='m',      desc='x-distance to the neutral axis (torsion center)')
        self.add_input('y_tc',                  val=np.zeros(n_span), units='m',      desc='y-distance to the neutral axis (torsion center)')
        self.add_input('flap_mode_shapes',      val=np.zeros((n_freq_blade,5)), desc='6-degree polynomial coefficients of mode shapes in the flap direction (x^2..x^6, no linear or constant term)')
        self.add_input('edge_mode_shapes',      val=np.zeros((n_freq_blade,5)), desc='6-degree polynomial coefficients of mode shapes in the edge direction (x^2..x^6, no linear or constant term)')
        self.add_input('gearbox_efficiency',    val=0.0,               desc='Gearbox efficiency')
        self.add_input('gearbox_ratio',         val=0.0,               desc='Gearbox ratio')

        # ServoDyn Inputs
        self.add_input('generator_efficiency',   val=0.0,              desc='Generator efficiency')

        # tower properties
        self.add_input('fore_aft_modes',   val=np.zeros((n_freq_tower,5)),               desc='6-degree polynomial coefficients of mode shapes in the flap direction (x^2..x^6, no linear or constant term)')
        self.add_input('side_side_modes',  val=np.zeros((n_freq_tower,5)),               desc='6-degree polynomial coefficients of mode shapes in the edge direction (x^2..x^6, no linear or constant term)')
        self.add_input('sec_loc',          val=np.zeros(N_beam),                         desc='normalized sectional location')
        self.add_input('mass_den',         val=np.zeros(N_beam),         units='kg/m',   desc='sectional mass per unit length')
        self.add_input('foreaft_stff',     val=np.zeros(N_beam),         units='N*m**2', desc='sectional fore-aft bending stiffness per unit length about the Y_E elastic axis')
        self.add_input('sideside_stff',    val=np.zeros(N_beam),         units='N*m**2', desc='sectional side-side bending stiffness per unit length about the Y_E elastic axis')
        self.add_input('tower_section_height', val=np.zeros(n_height-1), units='m',      desc='parameterized section heights along cylinder')
        self.add_input('tower_outer_diameter', val=np.zeros(n_height),   units='m',      desc='cylinder diameter at corresponding locations')

        # DriveSE quantities
        self.add_input('hub_system_cm',   val=np.zeros(3),             units='m',  desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')
        self.add_input('hub_system_I',    val=np.zeros(6),             units='kg*m**2', desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')
        self.add_input('hub_system_mass', val=0.0,                     units='kg', desc='mass of hub system')
        self.add_input('above_yaw_mass',  val=0.0, units='kg', desc='Mass of the nacelle above the yaw system')
        self.add_input('yaw_mass',        val=0.0, units='kg', desc='Mass of yaw system')
        self.add_input('nacelle_cm',      val=np.zeros(3), units='m', desc='Center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.add_input('nacelle_I',       val=np.zeros(6), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        # AeroDyn Inputs
        self.add_input('ref_axis_blade',    val=np.zeros((n_span,3)),units='m',   desc='2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.')
        self.add_input('chord',             val=np.zeros(n_span), units='m', desc='chord at airfoil locations')
        self.add_input('theta',             val=np.zeros(n_span), units='deg', desc='twist at airfoil locations')
        self.add_input('rthick',            val=np.zeros(n_span), desc='relative thickness of airfoil distribution')
        self.add_input('pitch_axis',        val=np.zeros(n_span), desc='1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.')
        self.add_input('Rhub',              val=0.0, units='m', desc='dimensional radius of hub')
        self.add_input('Rtip',              val=0.0, units='m', desc='dimensional radius of tip')
        self.add_input('airfoils_cl',       val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd',       val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm',       val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa',      val=np.zeros((n_aoa)), units='deg', desc='angle of attack grid for polars')
        self.add_input('airfoils_Re',       val=np.zeros((n_Re)), desc='Reynolds numbers of polars')
        self.add_input('airfoils_Re_loc',   val=np.zeros((n_span, n_Re, n_tab)), desc='temporary - matrix of Re numbers')
        self.add_input('airfoils_Ma_loc',   val=np.zeros((n_span, n_Re, n_tab)), desc='temporary - matrix of Ma numbers')
        self.add_input('airfoils_Ctrl',     val=np.zeros((n_span, n_Re, n_tab)), units='deg',desc='Airfoil control paremeter (i.e. flap angle)')
        
        # Airfoil coordinates
        self.add_input('coord_xy_interp',   val=np.zeros((n_span, n_xy, 2)),              desc='3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The leading edge is place at x=0 and y=0.')
        
        # Turbine level inputs
        self.add_discrete_input('rotor_orientation',val='upwind', desc='Rotor orientation, either upwind or downwind.')
        self.add_input('hub_height',                val=0.0, units='m', desc='hub height')
        self.add_input('tower_height',              val=0.0, units='m', desc='tower height from the tower base')
        self.add_input('tower_base_height',         val=0.0, units='m', desc='tower base height from the ground or mean sea level')
        self.add_discrete_input('turbulence_class', val='A', desc='IEC turbulence class')
        self.add_discrete_input('turbine_class',    val='I', desc='IEC turbulence class')
        self.add_input('control_ratedPower',        val=0.,  units='W',    desc='machine power rating')
        self.add_input('control_maxOmega',          val=0.0, units='rpm',  desc='maximum allowed rotor rotation speed')
        self.add_input('control_maxTS',             val=0.0, units='m/s',  desc='maximum allowed blade tip speed')
        self.add_input('cone',             val=0.0, units='deg',   desc='Cone angle of the rotor. It defines the angle between the rotor plane and the blade pitch axis. A standard machine has positive values.')
        self.add_input('tilt',             val=0.0, units='deg',   desc='Nacelle uptilt angle. A standard machine has positive values.')
        self.add_input('overhang',         val=0.0, units='m',     desc='Horizontal distance from tower top to hub center.')

        # Initial conditions
        self.add_input('U_init',        val=np.zeros(n_pc), units='m/s', desc='wind speeds')
        self.add_input('Omega_init',    val=np.zeros(n_pc), units='rpm', desc='rotation speeds to run')
        self.add_input('pitch_init',    val=np.zeros(n_pc), units='deg', desc='pitch angles to run')
        self.add_input('V',             val=np.zeros(n_pc), units='m/s',  desc='wind vector')

        # Cp-Ct-Cq surfaces
        self.add_input('Cp_aero_table', val=np.zeros((n_tsr, n_pitch, n_U)), desc='Table of aero power coefficient')
        self.add_input('Ct_aero_table', val=np.zeros((n_tsr, n_pitch, n_U)), desc='Table of aero thrust coefficient')
        self.add_input('Cq_aero_table', val=np.zeros((n_tsr, n_pitch, n_U)), desc='Table of aero torque coefficient')
        self.add_input('pitch_vector',  val=np.zeros(n_pitch), units='deg',  desc='Pitch vector used')
        self.add_input('tsr_vector',    val=np.zeros(n_tsr),                 desc='TSR vector used')
        self.add_input('U_vector',      val=np.zeros(n_U),     units='m/s',  desc='Wind speed vector used')

        # Environmental conditions 
        self.add_input('Vrated',      val=0.0, units='m/s',      desc='rated wind speed')
        self.add_input('V_R25',       val=0.0, units='m/s',      desc='region 2.5 transition wind speed')
        self.add_input('Vgust',       val=0.0, units='m/s',      desc='gust wind speed')
        self.add_input('V_extreme1',  val=0.0, units='m/s',      desc='IEC extreme wind speed at hub height for a 1-year retunr period')
        self.add_input('V_extreme50', val=0.0, units='m/s',      desc='IEC extreme wind speed at hub height for a 50-year retunr period')
        self.add_input('V_mean_iec',  val=0.0, units='m/s',      desc='IEC mean wind for turbulence class')
        self.add_input('V_cutout',    val=0.0, units='m/s',      desc='Maximum wind speed (cut-out)')
        self.add_input('rho',         val=0.0, units='kg/m**3',  desc='density of air')
        self.add_input('mu',          val=0.0, units='kg/(m*s)', desc='dynamic viscosity of air')
        self.add_input('shearExp',    val=0.0,                   desc='shear exponent')
        
        # Blade composite material properties (used for fatigue analysis)
        self.add_input('gamma_f',      val=1.35,                             desc='safety factor on loads')
        self.add_input('gamma_m',      val=1.1,                              desc='safety factor on materials')
        self.add_input('E',            val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Youngs moduli of the materials. Each row represents a material, the three columns represent E11, E22 and E33.')
        self.add_input('Xt',           val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Ultimate Tensile Strength (UTS) of the materials. Each row represents a material, the three columns represent Xt12, Xt13 and Xt23.')
        self.add_input('Xc',           val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Ultimate Compressive Strength (UCS) of the materials. Each row represents a material, the three columns represent Xc12, Xc13 and Xc23.')
        self.add_input('m',            val=np.zeros([n_mat]),                desc='2D array of the S-N fatigue slope exponent for the materials') 

        # Blade composit layup info (used for fatigue analysis)
        self.add_input('sc_ss_mats',   val=np.zeros((n_span, n_mat)),        desc="spar cap, suction side,  boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
        self.add_input('sc_ps_mats',   val=np.zeros((n_span, n_mat)),        desc="spar cap, pressure side, boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
        self.add_input('te_ss_mats',   val=np.zeros((n_span, n_mat)),        desc="trailing edge reinforcement, suction side,  boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
        self.add_input('te_ps_mats',   val=np.zeros((n_span, n_mat)),        desc="trailing edge reinforcement, pressure side, boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
        self.add_discrete_input('definition_layer', val=np.zeros(n_layers),  desc='1D array of flags identifying how layers are specified in the yaml. 1) all around (skin, paint, ) 2) offset+rotation twist+width (spar caps) 3) offset+user defined rotation+width 4) midpoint TE+width (TE reinf) 5) midpoint LE+width (LE reinf) 6) layer position fixed to other layer (core fillers) 7) start and width 8) end and width 9) start and end nd 10) web layer')
        # self.add_discrete_input('layer_name',       val=n_layers * [''],     desc='1D array of the names of the layers modeled in the blade structure.')
        # self.add_discrete_input('layer_web',        val=n_layers * [''],     desc='1D array of the names of the webs the layer is associated to. If the layer is on the outer profile this entry can simply stay empty.')
        # self.add_discrete_input('layer_mat',        val=n_layers * [''],     desc='1D array of the names of the materials of each layer modeled in the blade structure.')
        self.layer_name = blade_init_options['layer_name']

        # FAST run preferences
        self.FASTpref            = FASTpref 
        self.Analysis_Level      = FASTpref['analysis_settings']['Analysis_Level']
        self.debug_level         = FASTpref['analysis_settings']['debug_level']
        self.FAST_ver            = FASTpref['file_management']['FAST_ver']
        if os.path.isabs(FASTpref['file_management']['FAST_exe']):
            self.FAST_exe = FASTpref['file_management']['FAST_exe']
        else:
            self.FAST_exe = os.path.join(os.path.dirname(self.options['modeling_options']['fname_input_modeling']), FASTpref['file_management']['FAST_exe'])
        if os.path.isabs(FASTpref['file_management']['FAST_directory']):
            self.FAST_directory = FASTpref['file_management']['FAST_directory']
        else:
            self.FAST_directory = os.path.join(os.path.dirname(self.options['modeling_options']['fname_input_modeling']), FASTpref['file_management']['FAST_directory'])
        if os.path.isabs(FASTpref['file_management']['Turbsim_exe']):
            self.Turbsim_exe = FASTpref['file_management']['Turbsim_exe']
        else:
            self.Turbsim_exe = os.path.join(os.path.dirname(self.options['modeling_options']['fname_input_modeling']), FASTpref['file_management']['Turbsim_exe'])
        self.FAST_InputFile      = FASTpref['file_management']['FAST_InputFile']
        if MPI:
            rank    = MPI.COMM_WORLD.Get_rank()
            self.FAST_runDirectory = os.path.join(FASTpref['file_management']['FAST_runDirectory'],'rank_%000d'%int(rank))
            self.FAST_namingOut  = FASTpref['file_management']['FAST_namingOut']+'_%000d'%int(rank)
        else:
            self.FAST_runDirectory = FASTpref['file_management']['FAST_runDirectory']
            self.FAST_namingOut  = FASTpref['file_management']['FAST_namingOut']
        self.cores               = FASTpref['analysis_settings']['cores']
        self.case                = {}
        self.channels            = {}

        self.clean_FAST_directory = False
        if 'clean_FAST_directory' in FASTpref.keys():
            self.clean_FAST_directory = FASTpref['clean_FAST_directory']

        self.mpi_run             = False
        if 'mpi_run' in FASTpref['analysis_settings'].keys():
            self.mpi_run         = FASTpref['analysis_settings']['mpi_run']
            if self.mpi_run:
                self.mpi_comm_map_down   = FASTpref['analysis_settings']['mpi_comm_map_down']
        

        self.add_output('My_std',      val=0.0,            units='N*m',  desc='standard deviation of blade root flap bending moment in out-of-plane direction')
        self.add_output('DEL_RootMyb', val=0.0,            units='N*m',  desc='damage equivalent load of blade root flap bending moment in out-of-plane direction')
        self.add_output('flp1_std',    val=0.0,            units='deg',  desc='standard deviation of trailing-edge flap angle')

        self.add_output('V_out',       val=np.zeros(n_OF), units='m/s',  desc='wind vector')
        self.add_output('P_out',       val=np.zeros(n_OF), units='W',    desc='rotor electrical power')
        self.add_output('Cp_out',      val=np.zeros(n_OF),               desc='rotor aero power coefficient')
        self.add_output('Omega_out',   val=np.zeros(n_OF), units='rpm',  desc='rotation speeds to run')
        self.add_output('pitch_out',   val=np.zeros(n_OF), units='deg',  desc='pitch angles to run')

        self.add_output('rated_V',     val=0.0,            units='m/s',  desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0,            units='rpm',  desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0,            units='deg',  desc='pitch setting at rated')
        self.add_output('rated_T',     val=0.0,            units='N',    desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q',     val=0.0,            units='N*m',  desc='rotor aerodynamic torque at rated')
        self.add_output('AEP',         val=0.0,            units='kW*h', desc='annual energy production')

        self.add_output('loads_r',      val=np.zeros(n_span), units='m', desc='radial positions along blade going toward tip')
        self.add_output('loads_Px',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_output('loads_Py',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_output('loads_Pz',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned z-direction')
        self.add_output('loads_Omega',  val=0.0, units='rpm', desc='rotor rotation speed')
        self.add_output('loads_pitch',  val=0.0, units='deg', desc='pitch angle')
        self.add_output('loads_azimuth', val=0.0, units='deg', desc='azimuthal angle')
        
        self.add_output('Fxyz',        val=np.zeros(3),    units='N')
        self.add_output('Mxyz',        val=np.zeros(3),    units='N*m')

        self.add_output('C_miners_SC_SS',           val=np.zeros((n_span, n_mat, 2)),    desc="Miner's rule cummulative damage to Spar Cap, suction side")
        self.add_output('C_miners_SC_PS',           val=np.zeros((n_span, n_mat, 2)),    desc="Miner's rule cummulative damage to Spar Cap, pressure side")
        # self.add_output('C_miners_TE_SS',           val=np.zeros((n_span, n_mat, 2)),    desc="Miner's rule cummulative damage to Trailing-Edge reinforcement, suction side")
        # self.add_output('C_miners_TE_PS',           val=np.zeros((n_span, n_mat, 2)),    desc="Miner's rule cummulative damage to Trailing-Edge reinforcement, pressure side")

        self.add_discrete_output('fst_vt_out', val={})

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        #print(impl.world_comm().rank, 'Rotor_fast','start')
        sys.stdout.flush()

        fst_vt = self.update_FAST_model(inputs, discrete_inputs)
        
        if self.Analysis_Level == 2:
            # Run FAST with ElastoDyn

            FAST_Output, case_list, dlc_list  = self.run_FAST(inputs, discrete_inputs, fst_vt)
            self.post_process(FAST_Output, case_list, dlc_list, inputs, discrete_inputs, outputs, discrete_outputs)

            # list_cases, list_casenames, required_channels, case_keys = self.DLC_creation(inputs, discrete_inputs, fst_vt)
            # FAST_Output = self.run_FAST(fst_vt, list_cases, list_casenames, required_channels)

        elif self.Analysis_Level == 1:
            # Write FAST files, do not run
            self.write_FAST(fst_vt, discrete_outputs)

        discrete_outputs['fst_vt_out'] = fst_vt

        # delete run directory. not recommended for most cases, use for large parallelization problems where disk storage will otherwise fill up
        if self.clean_FAST_directory:
            try:
                shutil.rmtree(self.FAST_runDirectory)
            except:
                print('Failed to delete directory: %s'%self.FAST_runDirectory)


    def update_FAST_model(self, inputs, discrete_inputs):

        # Create instance of FAST reference model 

        fst_vt = self.options['modeling_options']['openfast']['fst_vt']

        fst_vt['Fst']['OutFileFmt'] = 2

        # Update ElastoDyn
        fst_vt['ElastoDyn']['TipRad'] = inputs['Rtip'][0]
        fst_vt['ElastoDyn']['HubRad'] = inputs['Rhub'][0]
        if discrete_inputs['rotor_orientation'] == 'upwind':
            k = -1.
        else:
            k = 1
        fst_vt['ElastoDyn']['PreCone(1)'] = k*inputs['cone'][0]
        fst_vt['ElastoDyn']['PreCone(2)'] = k*inputs['cone'][0]
        fst_vt['ElastoDyn']['PreCone(3)'] = k*inputs['cone'][0]
        fst_vt['ElastoDyn']['ShftTilt']   = k*inputs['tilt'][0]
        fst_vt['ElastoDyn']['OverHang']   = k*inputs['overhang'][0]
        fst_vt['ElastoDyn']['GBoxEff']    = inputs['gearbox_efficiency'][0] * 100.
        fst_vt['ElastoDyn']['GBRatio']    = inputs['gearbox_ratio'][0]

        # Update ServoDyn
        fst_vt['ServoDyn']['GenEff']      = inputs['generator_efficiency'][0] * 100.

        # Masses from DriveSE
        if self.options['modeling_options']['openfast']['analysis_settings']['update_hub_nacelle']:
            fst_vt['ElastoDyn']['HubMass']   = inputs['hub_system_mass'][0]
            fst_vt['ElastoDyn']['HubIner']   = inputs['hub_system_I'][0]
            fst_vt['ElastoDyn']['HubCM']     = inputs['hub_system_cm'][0] # k*inputs['overhang'][0] - inputs['hub_system_cm'][0], but we need to solve the circular dependency in DriveSE first
            fst_vt['ElastoDyn']['NacMass']   = inputs['above_yaw_mass'][0]
            fst_vt['ElastoDyn']['YawBrMass'] = inputs['yaw_mass'][0]
            fst_vt['ElastoDyn']['NacYIner']  = inputs['nacelle_I'][2]
            fst_vt['ElastoDyn']['NacCMxn']   = -k*inputs['nacelle_cm'][0]
            fst_vt['ElastoDyn']['NacCMyn']   = inputs['nacelle_cm'][1]
            fst_vt['ElastoDyn']['NacCMzn']   = inputs['nacelle_cm'][2]

        

        if self.options['modeling_options']['openfast']['analysis_settings']['update_tower']:
            # TODO: there are issues here
            #   - running the 15MW caused 120 tower points, some where nonunique heights
            #   - hub height is wrong, not adding the tower top to hub correctly

            tower_base_height = max(inputs['tower_base_height'][0], fst_vt['ElastoDyn']['PtfmCMzt'])
            fst_vt['ElastoDyn']['TowerBsHt'] = tower_base_height # Height of tower base above ground level [onshore] or MSL [offshore] (meters)
            fst_vt['ElastoDyn']['PtfmRefzt'] = tower_base_height # Vertical distance from the ground level [onshore] or MSL [offshore] to the platform reference point (meters)
            fst_vt['ElastoDyn']['TowerHt']   = inputs['tower_height'][-1] + tower_base_height # Height of tower above ground level [onshore] or MSL [offshore] (meters)

            # Update Inflowwind
            fst_vt['InflowWind']['RefHt'] = inputs['hub_height'][0]
            fst_vt['InflowWind']['PLexp'] = inputs['shearExp'][0]

            # Update ElastoDyn Tower Input File
            fst_vt['ElastoDynTower']['NTwInpSt'] = len(inputs['sec_loc'])
            fst_vt['ElastoDynTower']['HtFract']  = inputs['sec_loc']
            fst_vt['ElastoDynTower']['TMassDen'] = inputs['mass_den']
            fst_vt['ElastoDynTower']['TwFAStif'] = inputs['foreaft_stff']
            fst_vt['ElastoDynTower']['TwSSStif'] = inputs['sideside_stff']
            fst_vt['ElastoDynTower']['TwFAM1Sh'] = inputs['fore_aft_modes'][0, :]  / sum(inputs['fore_aft_modes'][0, :])
            fst_vt['ElastoDynTower']['TwFAM2Sh'] = inputs['fore_aft_modes'][1, :]  / sum(inputs['fore_aft_modes'][1, :])
            fst_vt['ElastoDynTower']['TwSSM1Sh'] = inputs['side_side_modes'][0, :] / sum(inputs['side_side_modes'][0, :])
            fst_vt['ElastoDynTower']['TwSSM2Sh'] = inputs['side_side_modes'][1, :] / sum(inputs['side_side_modes'][1, :])
            
            twr_elev  = np.r_[0.0, np.cumsum(inputs['tower_section_height'])] + fst_vt['ElastoDyn']['TowerBsHt']
            tip_height= twr_elev[-1]-inputs['Rtip']
            twr_index = np.argmin(abs(twr_elev - tip_height))
            if twr_elev[twr_index] > tip_height:
                twr_index -= 1

            fst_vt['AeroDyn15']['NumTwrNds'] = len(inputs['tower_outer_diameter'][twr_index:])
            fst_vt['AeroDyn15']['TwrElev']   = twr_elev[twr_index:]
            fst_vt['AeroDyn15']['TwrDiam']   = inputs['tower_outer_diameter'][twr_index:]
            fst_vt['AeroDyn15']['TwrCd']     = np.ones_like(fst_vt['AeroDyn15']['TwrDiam']) * np.mean(fst_vt['AeroDyn15']['TwrCd'][twr_index:])

        # Update ElastoDyn Blade Input File
        fst_vt['ElastoDynBlade']['NBlInpSt']   = len(inputs['r'])
        fst_vt['ElastoDynBlade']['BlFract']    = (inputs['r']-inputs['Rhub'])/(inputs['Rtip']-inputs['Rhub'])
        fst_vt['ElastoDynBlade']['BlFract'][0] = 0.
        fst_vt['ElastoDynBlade']['BlFract'][-1]= 1.
        fst_vt['ElastoDynBlade']['PitchAxis']  = inputs['le_location']
        fst_vt['ElastoDynBlade']['StrcTwst']   = inputs['theta'] # to do: structural twist is not nessessarily (nor likely to be) the same as aero twist
        fst_vt['ElastoDynBlade']['BMassDen']   = inputs['beam:rhoA']
        fst_vt['ElastoDynBlade']['FlpStff']    = inputs['beam:EIyy']
        fst_vt['ElastoDynBlade']['EdgStff']    = inputs['beam:EIxx']
        for i in range(5):
            fst_vt['ElastoDynBlade']['BldFl1Sh'][i] = inputs['flap_mode_shapes'][0,i] / sum(inputs['flap_mode_shapes'][0,:])
            fst_vt['ElastoDynBlade']['BldFl2Sh'][i] = inputs['flap_mode_shapes'][1,i] / sum(inputs['flap_mode_shapes'][1,:])
            fst_vt['ElastoDynBlade']['BldEdgSh'][i] = inputs['edge_mode_shapes'][0,i] / sum(inputs['edge_mode_shapes'][0,:])
        
        # Update AeroDyn15
        fst_vt['AeroDyn15']['AirDens']   = inputs['rho'][0]
        fst_vt['AeroDyn15']['KinVisc']   = inputs['mu'][0] / inputs['rho'][0]

        # Update AeroDyn15 Blade Input File
        r = (inputs['r']-inputs['Rhub'])
        r[0]  = 0.
        r[-1] = inputs['Rtip']-inputs['Rhub']
        fst_vt['AeroDynBlade']['NumBlNds'] = self.n_span
        fst_vt['AeroDynBlade']['BlSpn']    = r
        fst_vt['AeroDynBlade']['BlCrvAC']  = inputs['ref_axis_blade'][:,0]
        fst_vt['AeroDynBlade']['BlSwpAC']  = inputs['ref_axis_blade'][:,1]
        fst_vt['AeroDynBlade']['BlCrvAng'] = np.degrees(np.arcsin(np.gradient(inputs['ref_axis_blade'][:,0])/np.gradient(r)))
        fst_vt['AeroDynBlade']['BlTwist']  = inputs['theta']
        fst_vt['AeroDynBlade']['BlChord']  = inputs['chord']
        fst_vt['AeroDynBlade']['BlAFID']   = np.asarray(range(1,self.n_span+1))

        # Update AeroDyn15 Airfoile Input Files
        # airfoils = inputs['airfoils']
        fst_vt['AeroDyn15']['NumAFfiles'] = self.n_span
        # fst_vt['AeroDyn15']['af_data'] = [{}]*len(airfoils)
        fst_vt['AeroDyn15']['af_data'] = []

        if self.n_tab > 1:
            fst_vt['AeroDyn15']['AFTabMod'] = 3

        for i in range(self.n_span): # No of blade radial stations
        
            fst_vt['AeroDyn15']['af_data'].append([])
            

            for j in range(self.n_tab): # No of tabs; if there are no flaps at this blade station
                unsteady = eval_unsteady(inputs['airfoils_aoa'], inputs['airfoils_cl'][i,:,0,j], inputs['airfoils_cd'][i,:,0,j], inputs['airfoils_cm'][i,:,0,j])
                fst_vt['AeroDyn15']['af_data'][i].append({})


                fst_vt['AeroDyn15']['af_data'][i][j]['InterpOrd'] = "DEFAULT"
                fst_vt['AeroDyn15']['af_data'][i][j]['NonDimArea']= 1
                if self.options['modeling_options']['openfast']['analysis_settings']['generate_af_coords']:
                    fst_vt['AeroDyn15']['af_data'][i][j]['NumCoords'] = '@"AF{:02d}_Coords.txt"'.format(i)
                else:
                    fst_vt['AeroDyn15']['af_data'][i][j]['NumCoords'] = 0
                fst_vt['AeroDyn15']['af_data'][i][j]['NumTabs']   = self.n_tab
                if inputs['airfoils_Re_loc'][i][0][j] == 0:  # check if Re ws locally determined (e.g. for trailing edge flaps)
                    fst_vt['AeroDyn15']['af_data'][i][j]['Re']        =  0.75       # TODO: functionality for multiple Re tables
                else:
                    fst_vt['AeroDyn15']['af_data'][i][j]['Re'] = inputs['airfoils_Re_loc'][i,0,j]/1000000  # give in millions
                fst_vt['AeroDyn15']['af_data'][i][j]['Ctrl'] = inputs['airfoils_Ctrl'][i,0,j]  # unsteady['Ctrl'] # added to unsteady function for variable flap controls at airfoils

                fst_vt['AeroDyn15']['af_data'][i][j]['InclUAdata']= "True"
                fst_vt['AeroDyn15']['af_data'][i][j]['alpha0']    = unsteady['alpha0']
                fst_vt['AeroDyn15']['af_data'][i][j]['alpha1']    = unsteady['alpha1']
                fst_vt['AeroDyn15']['af_data'][i][j]['alpha2']    = unsteady['alpha2']
                fst_vt['AeroDyn15']['af_data'][i][j]['eta_e']     = unsteady['eta_e']
                fst_vt['AeroDyn15']['af_data'][i][j]['C_nalpha']  = unsteady['C_nalpha']
                fst_vt['AeroDyn15']['af_data'][i][j]['T_f0']      = unsteady['T_f0']
                fst_vt['AeroDyn15']['af_data'][i][j]['T_V0']      = unsteady['T_V0']
                fst_vt['AeroDyn15']['af_data'][i][j]['T_p']       = unsteady['T_p']
                fst_vt['AeroDyn15']['af_data'][i][j]['T_VL']      = unsteady['T_VL']
                fst_vt['AeroDyn15']['af_data'][i][j]['b1']        = unsteady['b1']
                fst_vt['AeroDyn15']['af_data'][i][j]['b2']        = unsteady['b2']
                fst_vt['AeroDyn15']['af_data'][i][j]['b5']        = unsteady['b5']
                fst_vt['AeroDyn15']['af_data'][i][j]['A1']        = unsteady['A1']
                fst_vt['AeroDyn15']['af_data'][i][j]['A2']        = unsteady['A2']
                fst_vt['AeroDyn15']['af_data'][i][j]['A5']        = unsteady['A5']
                fst_vt['AeroDyn15']['af_data'][i][j]['S1']        = unsteady['S1']
                fst_vt['AeroDyn15']['af_data'][i][j]['S2']        = unsteady['S2']
                fst_vt['AeroDyn15']['af_data'][i][j]['S3']        = unsteady['S3']
                fst_vt['AeroDyn15']['af_data'][i][j]['S4']        = unsteady['S4']
                fst_vt['AeroDyn15']['af_data'][i][j]['Cn1']       = unsteady['Cn1']
                fst_vt['AeroDyn15']['af_data'][i][j]['Cn2']       = unsteady['Cn2']
                fst_vt['AeroDyn15']['af_data'][i][j]['St_sh']     = unsteady['St_sh']
                fst_vt['AeroDyn15']['af_data'][i][j]['Cd0']       = unsteady['Cd0']
                fst_vt['AeroDyn15']['af_data'][i][j]['Cm0']       = unsteady['Cm0']
                fst_vt['AeroDyn15']['af_data'][i][j]['k0']        = unsteady['k0']
                fst_vt['AeroDyn15']['af_data'][i][j]['k1']        = unsteady['k1']
                fst_vt['AeroDyn15']['af_data'][i][j]['k2']        = unsteady['k2']
                fst_vt['AeroDyn15']['af_data'][i][j]['k3']        = unsteady['k3']
                fst_vt['AeroDyn15']['af_data'][i][j]['k1_hat']    = unsteady['k1_hat']
                fst_vt['AeroDyn15']['af_data'][i][j]['x_cp_bar']  = unsteady['x_cp_bar']
                fst_vt['AeroDyn15']['af_data'][i][j]['UACutout']  = unsteady['UACutout']
                fst_vt['AeroDyn15']['af_data'][i][j]['filtCutOff']= unsteady['filtCutOff']
                fst_vt['AeroDyn15']['af_data'][i][j]['NumAlf']    = len(unsteady['Alpha'])
                fst_vt['AeroDyn15']['af_data'][i][j]['Alpha']     = np.array(unsteady['Alpha'])
                fst_vt['AeroDyn15']['af_data'][i][j]['Cl']        = np.array(unsteady['Cl'])
                fst_vt['AeroDyn15']['af_data'][i][j]['Cd']        = np.array(unsteady['Cd'])
                fst_vt['AeroDyn15']['af_data'][i][j]['Cm']        = np.array(unsteady['Cm'])
                fst_vt['AeroDyn15']['af_data'][i][j]['Cpmin']     = np.zeros_like(unsteady['Cm'])        
        
        fst_vt['AeroDyn15']['af_coord'] = []
        fst_vt['AeroDyn15']['rthick']   = np.zeros(self.n_span)
        for i in range(self.n_span):
            fst_vt['AeroDyn15']['af_coord'].append({})
            fst_vt['AeroDyn15']['af_coord'][i]['x']  = inputs['coord_xy_interp'][i,:,0]
            fst_vt['AeroDyn15']['af_coord'][i]['y']  = inputs['coord_xy_interp'][i,:,1]
            fst_vt['AeroDyn15']['rthick'][i]         = inputs['rthick'][i]
                
        # AeroDyn spanwise output positions
        r = r/r[-1]
        r_out_target = [0.1, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        idx_out      = [np.argmin(abs(r-ri)) for ri in r_out_target]
        self.R_out   = [fst_vt['AeroDynBlade']['BlSpn'][i] for i in idx_out]

        if len(self.R_out) != len(np.unique(self.R_out)):
            exit('ERROR: the spanwise resolution is too coarse and does not support 9 channels along blade span. Please increase it in the modeling_options.yaml.')
        
        fst_vt['AeroDyn15']['BlOutNd']  = [str(idx+1) for idx in idx_out]
        fst_vt['AeroDyn15']['NBlOuts']  = len(idx_out)

        fst_vt['ElastoDyn']['BldGagNd'] = [idx+1 for idx in idx_out]
        fst_vt['ElastoDyn']['NBlGages'] = len(idx_out)

        return fst_vt


    def run_FAST(self, inputs, discrete_inputs, fst_vt):

        case_list      = []
        case_name_list = []
        dlc_list       = []

        if self.FASTpref['dlc_settings']['run_IEC'] or self.FASTpref['dlc_settings']['run_blade_fatigue']:
            case_list_IEC, case_name_list_IEC, dlc_list_IEC = self.DLC_creation_IEC(inputs, discrete_inputs, fst_vt)
            case_list      += case_list_IEC
            case_name_list += case_name_list_IEC
            dlc_list       += dlc_list_IEC

        if self.FASTpref['dlc_settings']['run_power_curve']:
            case_list_pc, case_name_list_pc, dlc_list_pc = self.DLC_creation_powercurve(inputs, discrete_inputs, fst_vt)
            case_list      += case_list_pc
            case_name_list += case_name_list_pc
            dlc_list       += dlc_list_pc

        # Mandatory output channels to include 
        # TODO: what else is needed here?
        channels_out  = ["TipDxc1", "TipDyc1", "TipDzc1", "TipDxc2", "TipDyc2", "TipDzc2", "TipDxc3", "TipDyc3", "TipDzc3"]
        channels_out += ["RootMxc1", "RootMyc1", "RootMzc1", "RootMxc2", "RootMyc2", "RootMzc2", "RootMxc3", "RootMyc3", "RootMzc3"]
        channels_out += ["TipDxb1", "TipDyb1", "TipDzb1", "TipDxb2", "TipDyb2", "TipDzb2", "TipDxb3", "TipDyb3", "TipDzb3"]
        channels_out += ["RootMxb1", "RootMyb1", "RootMzb1", "RootMxb2", "RootMyb2", "RootMzb2", "RootMxb3", "RootMyb3", "RootMzb3"]
        channels_out += ["RootFxc1", "RootFyc1", "RootFzc1", "RootFxc2", "RootFyc2", "RootFzc2", "RootFxc3", "RootFyc3", "RootFzc3"]
        channels_out += ["RootFxb1", "RootFyb1", "RootFzb1", "RootFxb2", "RootFyb2", "RootFzb2", "RootFxb3", "RootFyb3", "RootFzb3"]
        channels_out += ["RtAeroCp", "RtAeroCt", "RotSpeed", "NacYaw",  "GenPwr", "GenTq", "BldPitch1", "BldPitch2", "BldPitch3", "Azimuth"]
        channels_out += ["Wind1VelX", "Wind1VelY", "Wind1VelZ"]
        channels_out += ["TwrBsMxt",  "TwrBsMyt", "TwrBsMzt"]
        channels_out += ["B1N1Fx", "B1N2Fx", "B1N3Fx", "B1N4Fx", "B1N5Fx", "B1N6Fx", "B1N7Fx", "B1N8Fx", "B1N9Fx", "B1N1Fy", "B1N2Fy", "B1N3Fy", "B1N4Fy", "B1N5Fy", "B1N6Fy", "B1N7Fy", "B1N8Fy", "B1N9Fy"]
        channels_out += ["B2N1Fx", "B2N2Fx", "B2N3Fx", "B2N4Fx", "B2N5Fx", "B2N6Fx", "B2N7Fx", "B2N8Fx", "B2N9Fx", "B2N1Fy", "B2N2Fy", "B2N3Fy", "B2N4Fy", "B2N5Fy", "B2N6Fy", "B2N7Fy", "B2N8Fy", "B2N9Fy"]
        channels_out += ["B3N1Fx", "B3N2Fx", "B3N3Fx", "B3N4Fx", "B3N5Fx", "B3N6Fx", "B3N7Fx", "B3N8Fx", "B3N9Fx", "B3N1Fy", "B3N2Fy", "B3N3Fy", "B3N4Fy", "B3N5Fy", "B3N6Fy", "B3N7Fy", "B3N8Fy", "B3N9Fy"]
        channels_out += ["Spn1MLxb1", "Spn2MLxb1", "Spn3MLxb1", "Spn4MLxb1", "Spn5MLxb1", "Spn6MLxb1", "Spn7MLxb1", "Spn8MLxb1", "Spn9MLxb1"]
        channels_out += ["Spn1MLyb1", "Spn2MLyb1", "Spn3MLyb1", "Spn4MLyb1", "Spn5MLyb1", "Spn6MLyb1", "Spn7MLyb1", "Spn8MLyb1", "Spn9MLyb1"]
        channels_out += ["RtAeroFxh", "RtAeroFyh", "RtAeroFzh"]
        channels_out += ["RotThrust", "LSShftFys", "LSShftFzs", "RotTorq", "LSSTipMys", "LSSTipMzs"]
        # Add additional options
        if ('channels_out',) in self.options['modeling_options']['openfast']['fst_settings']:
            channels_out += self.options['modeling_options']['openfast']['fst_settings'][('channels_out',)]



        channels = {}
        for var in channels_out:
            channels[var] = True

        # FAST wrapper setup
        fastBatch = runFAST_pywrapper_batch(FAST_ver=self.FAST_ver)
        fastBatch.channels = channels

        fastBatch.FAST_exe          = self.FAST_exe
        fastBatch.FAST_runDirectory = self.FAST_runDirectory
        fastBatch.FAST_InputFile    = self.FAST_InputFile
        fastBatch.FAST_directory    = self.FAST_directory
        fastBatch.debug_level       = self.debug_level
        fastBatch.fst_vt            = fst_vt
        fastBatch.post              = FAST_IO_timeseries

        fastBatch.case_list         = case_list
        fastBatch.case_name_list    = case_name_list
        fastBatch.channels          = channels

        fastBatch.overwrite_outfiles = True  #<--- Debugging only, set to False to prevent OpenFAST from running if the .outb already exists

        # Run FAST
        if self.mpi_run:
            FAST_Output = fastBatch.run_mpi(self.mpi_comm_map_down)
        else:
            if self.cores == 1:
                FAST_Output = fastBatch.run_serial()
            else:
                FAST_Output = fastBatch.run_multi(self.cores)

        self.fst_vt = fst_vt

        sys.stdout.flush()
        return FAST_Output, case_list, dlc_list

    def DLC_creation_IEC(self, inputs, discrete_inputs, fst_vt, powercurve=False):

        iec = CaseGen_IEC()

        # Turbine Data
        iec.Turbine_Class    = discrete_inputs['turbine_class']
        iec.Turbulence_Class = discrete_inputs['turbulence_class']
        iec.D                = fst_vt['ElastoDyn']['TipRad']*2. #np.min([fst_vt['InflowWind']['RefHt']*1.9 , fst_vt['ElastoDyn']['TipRad']*2.5])
        iec.z_hub            = fst_vt['InflowWind']['RefHt']

        # Turbine initial conditions
        iec.init_cond = {} # can leave as {} if data not available
        iec.init_cond[("ElastoDyn","RotSpeed")]        = {'U':inputs['U_init']}
        iec.init_cond[("ElastoDyn","RotSpeed")]['val'] = inputs['Omega_init']
        iec.init_cond[("ElastoDyn","BlPitch1")]        = {'U':inputs['U_init']}
        iec.init_cond[("ElastoDyn","BlPitch1")]['val'] = inputs['pitch_init']
        iec.init_cond[("ElastoDyn","BlPitch2")]        = iec.init_cond[("ElastoDyn","BlPitch1")]
        iec.init_cond[("ElastoDyn","BlPitch3")]        = iec.init_cond[("ElastoDyn","BlPitch1")]

        # Todo: need a way to handle Metocean conditions for Offshore
        # if offshore:
        #     iec.init_cond[("HydroDyn","WaveHs")]        = {'U':[3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 40, 50]}
        #     iec.init_cond[("HydroDyn","WaveHs")]['val'] = [1.101917033, 1.101917033, 1.179052649, 1.315715154, 1.536867124, 1.835816514, 2.187994638, 2.598127096, 3.061304068, 3.617035443, 4.027470219, 4.51580671, 4.51580671, 6.98, 10.7]
        #     iec.init_cond[("HydroDyn","WaveTp")]        = {'U':[3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 40, 50]}
        #     iec.init_cond[("HydroDyn","WaveTp")]['val'] = [8.515382435, 8.515382435, 8.310063688, 8.006300889, 7.6514231, 7.440581338, 7.460834063, 7.643300307, 8.046899942, 8.521314105, 8.987021024, 9.451641026, 9.451641026, 11.7, 14.2]

        # Setup dlc settings
        iec.dlc_inputs = {}
        iec.dlc_inputs['DLC']   = []
        iec.dlc_inputs['U']     = []
        iec.dlc_inputs['Seeds'] = []
        iec.dlc_inputs['Yaw']   = []

        if powercurve:
            # running turbulent power curve
            iec.dlc_inputs['DLC'].append(1.1)
            iec.dlc_inputs['U'].append(self.FASTpref['dlc_settings']['Power_Curve']['U'])
            iec.dlc_inputs['Seeds'].append(self.FASTpref['dlc_settings']['Power_Curve']['Seeds'])
            iec.dlc_inputs['Yaw'].append([])

        else:

            for dlc in self.FASTpref['dlc_settings']['IEC']:

                if 'DLC' in dlc.keys():
                    iec.dlc_inputs['DLC'].append(dlc['DLC'])
                else:
                    iec.dlc_inputs['DLC'].append([])

                if 'U' in dlc.keys():
                    iec.dlc_inputs['U'].append(dlc['U'])
                else:
                    if dlc['DLC'] == 1.4:
                        iec.dlc_inputs['U'].append([float(inputs['Vrated'])-2., float(inputs['Vrated']), float(inputs['Vrated'])+2.])
                    elif dlc['DLC'] == 5.1:
                        iec.dlc_inputs['U'].append([float(inputs['Vrated'])-2., float(inputs['Vrated'])+2., float(inputs['V_cutout'])])
                    elif dlc['DLC'] == 6.1:
                        iec.dlc_inputs['U'].append([float(inputs['V_extreme50'])])
                    elif dlc['DLC'] == 6.3:
                        iec.dlc_inputs['U'].append([float(inputs['V_extreme1'])])
                    else:
                        iec.dlc_inputs['U'].append([])

                if 'Seeds' in dlc.keys():
                    iec.dlc_inputs['Seeds'].append(dlc['Seeds'])
                else:
                    iec.dlc_inputs['Seeds'].append([])

                if 'Yaw' in dlc.keys():
                    iec.dlc_inputs['Yaw'].append(dlc['Yaw'])
                else:
                    iec.dlc_inputs['Yaw'].append([])

        iec.transient_dir_change        = '-'
        iec.transient_shear_orientation = 'v'
        if ("Fst", "TStart") not in list(self.options['modeling_options']['openfast']['fst_settings'].keys()):
            self.options['modeling_options']['openfast']['fst_settings'][('Fst','TStart')] = 120.
        T0                          = self.options['modeling_options']['openfast']['fst_settings'][("Fst", "TStart")]

        if ("Fst", "TMax") not in list(self.options['modeling_options']['openfast']['fst_settings'].keys()):
            self.options['modeling_options']['openfast']['fst_settings'][("Fst", "TMax")] = 720.
        iec.TMax                    = self.options['modeling_options']['openfast']['fst_settings'][("Fst", "TMax")]

        iec.TStart                      = (iec.TMax-T0)/2. + T0
        self.simtime                    = iec.TMax - T0
        self.TMax                       = iec.TMax
        self.T0                         = T0

        # path management
        iec.wind_dir        = self.FAST_runDirectory
        iec.Turbsim_exe     = self.Turbsim_exe
        iec.debug_level     = self.debug_level
        iec.overwrite       = False # TODO: elevate these options to analysis input file
        iec.run_dir         = self.FAST_runDirectory

        if self.mpi_run:
            iec.parallel_windfile_gen = True
            iec.mpi_run               = self.FASTpref['analysis_settings']['mpi_run']
            iec.comm_map_down         = self.FASTpref['analysis_settings']['mpi_comm_map_down']
        else:
            iec.parallel_windfile_gen = False

        if powercurve:
            iec.case_name_base  = self.FAST_namingOut + '_powercurve'
        else:
            iec.case_name_base  = self.FAST_namingOut + '_IEC'

        # OpenFAST Settings
        # load user overwrite settings
        case_inputs = {}
        for var in list(self.options['modeling_options']['openfast']['fst_settings'].keys()):
            case_inputs[var] = {'vals':[self.options['modeling_options']['openfast']['fst_settings'][var]], 'group':0}

        # Run case setup, generate wind inputs
        case_list, case_name_list, dlc_list = iec.execute(case_inputs=case_inputs)


        return case_list, case_name_list, dlc_list

    def DLC_creation_powercurve(self, inputs, discrete_inputs, fst_vt):

        if len(self.FASTpref['dlc_settings']['Power_Curve']['U']) > 0: # todo: need a warning if no powercurve wind speeds are specified and DLC 1.1 is not set
        
            if self.FASTpref['dlc_settings']['Power_Curve']['turbulent_power_curve']:

                case_list, case_name, dlc_list_IEC = self.DLC_creation_IEC(inputs, discrete_inputs, fst_vt, powercurve=True)

            else:
                U     = self.FASTpref['dlc_settings']['Power_Curve']['U']
                omega = np.interp(U, inputs['U_init'], inputs['Omega_init'])
                pitch = np.interp(U, inputs['U_init'], inputs['pitch_init'])

                # wind speeds
                case_inputs = {}
                case_inputs[("InflowWind","WindType")]   = {'vals':[1], 'group':0}
                case_inputs[("InflowWind","HWindSpeed")] = {'vals':U, 'group':1}
                case_inputs[("ElastoDyn","RotSpeed")]    = {'vals':omega, 'group':1}
                case_inputs[("ElastoDyn","BlPitch1")]    = {'vals':pitch, 'group':1}
                case_inputs[("ElastoDyn","BlPitch2")]    = case_inputs[("ElastoDyn","BlPitch1")]
                case_inputs[("ElastoDyn","BlPitch3")]    = case_inputs[("ElastoDyn","BlPitch1")]

                # User defined simulation settings
                if ("InflowWind","WindType") in case_inputs:
                    print('WARNING: You have defined ("InflowWind","WindType"} in the openfast settings.'
                            'This will overwrite the default powercurve settings')
                if ("InflowWind","HWindSpeed") in case_inputs:
                    print('WARNING: You have defined ("InflowWind","HWindSpeed"} in the openfast settings.'
                            'This will overwrite the default powercurve settings')
                for var in list(self.options['modeling_options']['openfast']['fst_settings'].keys()):
                    case_inputs[var] = {'vals':[self.options['modeling_options']['openfast']['fst_settings'][var]], 'group':0}

                case_list, case_name = CaseGen_General(case_inputs, self.FAST_runDirectory, self.FAST_namingOut + '_powercurve')

            dlc_list = [0.]*len(case_name)

            return case_list, case_name, dlc_list

        else:
            return [], [], []

    def post_process(self, FAST_Output, case_list, dlc_list, inputs, discrete_inputs, outputs, discrete_outputs):

        # Load pCrunch Analysis
        loads_analysis         = Analysis.Loads_Analysis()
        loads_analysis.verbose = self.options['modeling_options']['general']['verbosity']

        # Initial time
        loads_analysis.t0 = self.options['modeling_options']['openfast']['fst_settings'][('Fst','TStart')]
        
        # Calc summary stats on the magnitude of a vector
        loads_analysis.channels_magnitude = {'LSShftF':["RotThrust", "LSShftFys", "LSShftFzs"], 
                                             'LSShftM':["RotTorq", "LSSTipMys", "LSSTipMzs"]}
                                             # 'RootMc1': ["RootMxc1", "RootMyc1", "RootMzc1"],
                                             # 'RootMc2': ["RootMxc2", "RootMyc2", "RootMzc2"],
                                             # 'RootMc3': ["RootMxc3", "RootMyc3", "RootMzc3"]}
                                             # 'TipDc1':['TipDxc1', 'TipDyc1', 'TipDzc1'],
                                             # 'TipDc2':['TipDxc2', 'TipDyc2', 'TipDzc2'],
                                             # 'TipDc3':['TipDxc3', 'TipDyc3', 'TipDzc3']}

        # extreme event tables, return the value of these channels where over variables are at a maximum
        loads_analysis.channels_extreme_table  = ["B1N1Fx", "B1N2Fx", "B1N3Fx", "B1N4Fx", "B1N5Fx", "B1N6Fx", "B1N7Fx", "B1N8Fx", "B1N9Fx", "B1N1Fy", "B1N2Fy", "B1N3Fy", "B1N4Fy", "B1N5Fy", "B1N6Fy", "B1N7Fy", "B1N8Fy", "B1N9Fy"]
        loads_analysis.channels_extreme_table += ["B2N1Fx", "B2N2Fx", "B2N3Fx", "B2N4Fx", "B2N5Fx", "B2N6Fx", "B2N7Fx", "B2N8Fx", "B2N9Fx", "B2N1Fy", "B2N2Fy", "B2N3Fy", "B2N4Fy", "B2N5Fy", "B2N6Fy", "B2N7Fy", "B2N8Fy", "B2N9Fy"]
        loads_analysis.channels_extreme_table += ["B3N1Fx", "B3N2Fx", "B3N3Fx", "B3N4Fx", "B3N5Fx", "B3N6Fx", "B3N7Fx", "B3N8Fx", "B3N9Fx", "B3N1Fy", "B3N2Fy", "B3N3Fy", "B3N4Fy", "B3N5Fy", "B3N6Fy", "B3N7Fy", "B3N8Fy", "B3N9Fy"]
        loads_analysis.channels_extreme_table += ['RotSpeed', 'BldPitch1', 'BldPitch2', 'BldPitch3', 'Azimuth']
        loads_analysis.channels_extreme_table += ["RootMxc1", "RootMyc1", "RootMzc1", "RootMxc2", "RootMyc2", "RootMzc2", "RootMxc3", "RootMyc3", "RootMzc3"]
        loads_analysis.channels_extreme_table += ["RotThrust", "LSShftFys", "LSShftFzs", "RotTorq", "LSSTipMys", "LSSTipMzs"]

        # DEL info 
        loads_analysis.DEL_info = [('RootMyb1', 10), ('RootMyb2', 10), ('RootMyb3', 10)]

        # get summary stats
        sum_stats, extreme_table = loads_analysis.summary_stats(FAST_Output)

        
        ## Post process loads
        if self.FASTpref['dlc_settings']['run_IEC']:
            # TODO: support for BeamDyn
            # TODO: support for 2 bladed

            # Determine blade with the maximum deflection magnitude
            defl_mag = [max(sum_stats['TipDxc1']['max']), max(sum_stats['TipDxc2']['max']), max(sum_stats['TipDxc3']['max'])]
            if np.argmax(defl_mag) == 0:
                blade_chans_Fx = ["B1N1Fx", "B1N2Fx", "B1N3Fx", "B1N4Fx", "B1N5Fx", "B1N6Fx", "B1N7Fx", "B1N8Fx", "B1N9Fx"]
                blade_chans_Fy = ["B1N1Fy", "B1N2Fy", "B1N3Fy", "B1N4Fy", "B1N5Fy", "B1N6Fy", "B1N7Fy", "B1N8Fy", "B1N9Fy"]
                tip_max_chan   = "TipDxc1"
                bld_pitch_chan = "BldPitch1"
            if np.argmax(defl_mag) == 1:
                blade_chans_Fx = ["B2N1Fx", "B2N2Fx", "B2N3Fx", "B2N4Fx", "B2N5Fx", "B2N6Fx", "B2N7Fx", "B2N8Fx", "B2N9Fx"]
                blade_chans_Fy = ["B2N1Fy", "B2N2Fy", "B2N3Fy", "B2N4Fy", "B2N5Fy", "B2N6Fy", "B2N7Fy", "B2N8Fy", "B2N9Fy"]
                tip_max_chan   = "TipDxc2"
                bld_pitch_chan = "BldPitch2"
            if np.argmax(defl_mag) == 2:            
                blade_chans_Fx = ["B3N1Fx", "B3N2Fx", "B3N3Fx", "B3N4Fx", "B3N5Fx", "B3N6Fx", "B3N7Fx", "B3N8Fx", "B3N9Fx"]
                blade_chans_Fy = ["B3N1Fy", "B3N2Fy", "B3N3Fy", "B3N4Fy", "B3N5Fy", "B3N6Fy", "B3N7Fy", "B3N8Fy", "B3N9Fy"]
                tip_max_chan   = "TipDxc3"
                bld_pitch_chan = "BldPitch3"

            # Return spanwise forces at instance of largest deflection
            Fx = [extreme_table[tip_max_chan][np.argmax(sum_stats[tip_max_chan]['max'])][var]['val'] for var in blade_chans_Fx]
            Fy = [extreme_table[tip_max_chan][np.argmax(sum_stats[tip_max_chan]['max'])][var]['val'] for var in blade_chans_Fy]
            spline_Fx = PchipInterpolator(self.R_out, Fx)
            spline_Fy = PchipInterpolator(self.R_out, Fy)

            r = inputs['r']-inputs['Rhub']
            Fx_out = spline_Fx(r).flatten()
            Fy_out = spline_Fy(r).flatten()
            Fz_out = np.zeros_like(Fx_out)

            outputs['loads_r']       = r
            outputs['loads_Px']      = Fx_out
            outputs['loads_Py']      = Fy_out*-1.
            outputs['loads_Pz']      = Fz_out
            outputs['loads_Omega']   = extreme_table[tip_max_chan][np.argmax(sum_stats[tip_max_chan]['max'])]['RotSpeed']['val']
            outputs['loads_pitch']   = extreme_table[tip_max_chan][np.argmax(sum_stats[tip_max_chan]['max'])]['BldPitch1']['val']
            outputs['loads_azimuth'] = extreme_table[tip_max_chan][np.argmax(sum_stats[tip_max_chan]['max'])]['Azimuth']['val']

            # # Determine blade with the maximum root moment
            # defl_mag = [max(sum_stats['RootMc1']['max']), max(sum_stats['RootMc2']['max']), max(sum_stats['RootMc3']['max'])]
            # if np.argmax(defl_mag) == 0:
            #     outputs['Mxyz'] = np.array(["RootMxc1", "RootMyc1", "RootMzc1"])*1.e3
            # if np.argmax(defl_mag) == 1:
            #     outputs['Mxyz'] = np.array(["RootMxc2", "RootMyc2", "RootMzc2"])*1.e3
            # if np.argmax(defl_mag) == 2:
            #     outputs['Mxyz'] = np.array(["RootMxc3", "RootMyc3", "RootMzc3"])*1.e3

            ## Get hub momements and forces in the non-rotating frame
            outputs['Fxyz'] = np.array([extreme_table['LSShftF'][np.argmax(sum_stats['LSShftF']['max'])]['RotThrust']['val'],
                                        extreme_table['LSShftF'][np.argmax(sum_stats['LSShftF']['max'])]['LSShftFys']['val'],
                                        extreme_table['LSShftF'][np.argmax(sum_stats['LSShftF']['max'])]['LSShftFzs']['val']])*1.e3
            outputs['Mxyz'] = np.array([extreme_table['LSShftM'][np.argmax(sum_stats['LSShftM']['max'])]['RotTorq']['val'],
                                        extreme_table['LSShftM'][np.argmax(sum_stats['LSShftM']['max'])]['LSSTipMys']['val'],
                                        extreme_table['LSShftM'][np.argmax(sum_stats['LSShftM']['max'])]['LSSTipMzs']['val']])*1.e3

        if self.FASTpref['dlc_settings']['run_blade_fatigue']:

            # determine which dlc will be used for fatigue calculations, checks for dlc 1.2, then dlc 1.1
            idx_fat_12 = [i for i, dlc in enumerate(dlc_list) if dlc==1.2]
            idx_fat_11 = [i for i, dlc in enumerate(dlc_list) if dlc==1.1]
            if len(idx_fat_12) > 0:
                idx_fat = idx_fat_12
            elif len(idx_fat_11) > 0:
                idx_fat = idx_fat_11
            else:
                print('Warning: User turned on "run_blade_fatigue", but IEC DLC 1.1 or 1.2 are not being run. Fatigue analysis will not be run.')
                sys.stdout.flush()

            if len(idx_fat) > 0:
                outputs, discrete_outputs = self.BladeFatigue(FAST_Output, case_list, dlc_list, inputs, outputs, discrete_inputs, discrete_outputs)


        ## Get AEP and power curve
        if self.FASTpref['dlc_settings']['run_power_curve']:

            # determine which dlc will be used for the powercurve calculations, allows using dlc 1.1 if specific power curve calculations were not run
            idx_pwrcrv    = [i for i, dlc in enumerate(dlc_list) if dlc==0.]
            idx_pwrcrv_11 = [i for i, dlc in enumerate(dlc_list) if dlc==1.1]
            if len(idx_pwrcrv) == 0 and len(idx_pwrcrv_11) > 0:
                idx_pwrcrv = idx_pwrcrv_11

            # sort out power curve stats
            stats_pwrcrv = {}
            for var in sum_stats.keys():
                if var != 'meta':
                    stats_pwrcrv[var] = {}
                    for stat in sum_stats[var].keys():
                        stats_pwrcrv[var][stat] = [x for i, x in enumerate(sum_stats[var][stat]) if i in idx_pwrcrv]

            stats_pwrcrv['meta'] = sum_stats['meta']

            # get wind speed 
            if self.FASTpref['dlc_settings']['Power_Curve']['turbulent_power_curve']:
                U = []
                for fname in [case[('InflowWind', 'Filename')] for i, case in enumerate(case_list) if i in idx_pwrcrv]:
                    fname = os.path.split(fname)[-1]
                    ntm      = fname.split('NTM')[-1].split('_')
                    ntm_U    = float(".".join(ntm[1].strip("U").split('.')[:-1]))
                    ntm_Seed = float(".".join(ntm[2].strip("Seed").split('.')[:-1]))
                    U.append(ntm_U)
            else:
                U = [float(case[('InflowWind', 'HWindSpeed')]) for i, case in enumerate(case_list) if i in idx_pwrcrv]

            # calc AEP
            
            if len(U) > 1 and self.fst_vt['Fst']['CompServo'] == 1:
                pp               = Analysis.Power_Production()
                pp.windspeeds    = U
                pp.turbine_class = discrete_inputs['turbine_class']
                pwr_curve_vars   = ["GenPwr", "RtAeroCp", "RotSpeed", "BldPitch1"]
                AEP, perf_data   = pp.AEP(stats_pwrcrv, U, pwr_curve_vars=pwr_curve_vars)
                outputs['P_out']       = perf_data['GenPwr']['mean']
                outputs['Cp_out']      = perf_data['RtAeroCp']['mean']
                outputs['Omega_out']   = perf_data['RotSpeed']['mean']
                outputs['pitch_out']   = perf_data['BldPitch1']['mean']
                outputs['AEP']         = AEP
            else:
                outputs['Cp_out']      = stats_pwrcrv['RtAeroCp']['mean']
                outputs['AEP']         = 0.0
                outputs['Omega_out']   = stats_pwrcrv['RotSpeed']['mean']
                outputs['pitch_out']   = stats_pwrcrv['BldPitch1']['mean']
                if self.fst_vt['Fst']['CompServo'] == 1:
                    outputs['P_out']       = stats_pwrcrv['GenPwr']['mean']
                print('WARNING: OpenFAST is run at a single wind speed. AEP cannot be estimated.')

            

            outputs['V_out']       = np.unique(U)

            ## TODO: solve for V rated with the power curve data
            ## Maybe fit a least squares linear line above input rated wind speed, least squares quadratic to below rate, find intersection
            # outputs['rated_V']     = 
            # outputs['rated_Omega'] = 
            # outputs['rated_pitch'] = 
            # outputs['rated_T']     = 
            # outputs['rated_Q']     = 



        ## Is Nikhar actively using this?
        # DELs
        # del_channels = [('RootMyb1',10), ('RootMyb2',10), ('RootMyb3',10)]
        # dels = loads_analysis.get_DEL(FAST_Output, del_channels, binNum=100, t=FAST_Output[0]['Time'][-1])
        
        # Output
        outputs['DEL_RootMyb'] = np.max([np.max(sum_stats['RootMyb1']['DEL']), np.max(sum_stats['RootMyb2']['DEL']), np.max(sum_stats['RootMyb3']['DEL'])])
        outputs['My_std'] = np.max([np.max(sum_stats['RootMyb1']['std']), np.max(sum_stats['RootMyb2']['std']), np.max(sum_stats['RootMyb3']['std'])])

    def write_FAST(self, fst_vt, discrete_outputs):
        writer                   = InputWriter_OpenFAST(FAST_ver=self.FAST_ver)
        writer.fst_vt            = fst_vt
        writer.FAST_runDirectory = self.FAST_runDirectory
        writer.FAST_namingOut    = self.FAST_namingOut
        writer.execute()

        if self.debug_level > 0:
            print('RAN UPDATE: ', self.FAST_runDirectory, self.FAST_namingOut)



    def writeCpsurfaces(self, inputs):
        
        FASTpref  = self.options['modeling_options']['openfast']['FASTpref']
        file_name = os.path.join(FASTpref['file_management']['FAST_runDirectory'], FASTpref['file_management']['FAST_namingOut'] + '_Cp_Ct_Cq.dat')
        
        # Write Cp-Ct-Cq-TSR tables file
        n_pitch = len(inputs['pitch_vector'])
        n_tsr   = len(inputs['tsr_vector'])
        n_U     = len(inputs['U_vector'])
        
        file = open(file_name,'w')
        file.write('# ------- Rotor performance tables ------- \n')
        file.write('# ------------ Written using AeroElasticSE with data from CCBlade ------------\n')
        file.write('\n')
        file.write('# Pitch angle vector - x axis (matrix columns) (deg)\n')
        for i in range(n_pitch):
            file.write('%.2f   ' % inputs['pitch_vector'][i])
        file.write('\n# TSR vector - y axis (matrix rows) (-)\n')
        for i in range(n_tsr):
            file.write('%.2f   ' % inputs['tsr_vector'][i])
        file.write('\n# Wind speed vector - z axis (m/s)\n')
        for i in range(n_U):
            file.write('%.2f   ' % inputs['U_vector'][i])
        file.write('\n')
        
        file.write('\n# Power coefficient\n\n')
        
        for i in range(n_U):
            for j in range(n_tsr):
                for k in range(n_pitch):
                    file.write('%.5f   ' % inputs['Cp_aero_table'][j,k,i])
                file.write('\n')
            file.write('\n')
        
        file.write('\n#  Thrust coefficient\n\n')
        for i in range(n_U):
            for j in range(n_tsr):
                for k in range(n_pitch):
                    file.write('%.5f   ' % inputs['Ct_aero_table'][j,k,i])
                file.write('\n')
            file.write('\n')
        
        file.write('\n# Torque coefficient\n\n')
        for i in range(n_U):
            for j in range(n_tsr):
                for k in range(n_pitch):
                    file.write('%.5f   ' % inputs['Cq_aero_table'][j,k,i])
                file.write('\n')
            file.write('\n')
            
        file.close()


        return file_name


    def BladeFatigue(self, FAST_Output, case_list, dlc_list, inputs, outputs, discrete_inputs, discrete_outputs):

        # Perform rainflow counting
        if self.options['modeling_options']['general']['verbosity']:
            print('Running Rainflow Counting')
            sys.stdout.flush()

        rainflow = {}
        var_rainflow = ["RootMxb1", "Spn1MLxb1", "Spn2MLxb1", "Spn3MLxb1", "Spn4MLxb1", "Spn5MLxb1", "Spn6MLxb1", "Spn7MLxb1", "Spn8MLxb1", "Spn9MLxb1", "RootMyb1", "Spn1MLyb1", "Spn2MLyb1", "Spn3MLyb1", "Spn4MLyb1", "Spn5MLyb1", "Spn6MLyb1", "Spn7MLyb1", "Spn8MLyb1", "Spn9MLyb1"]
        for i, (datai, casei, dlci) in enumerate(zip(FAST_Output, case_list, dlc_list)):
            if dlci in [1.1, 1.2]:
            
                # Get wind speed and seed of output file
                ntm  = casei[('InflowWind', 'Filename')].split('NTM')[-1].split('_')
                U    = float(".".join(ntm[1].strip("U").split('.')[:-1]))
                Seed = float(".".join(ntm[2].strip("Seed").split('.')[:-1]))

                if U not in list(rainflow.keys()):
                    rainflow[U]       = {}
                if Seed not in list(rainflow[U].keys()):
                    rainflow[U][Seed] = {}
                
                # Rainflow counting by var
                if len(var_rainflow) == 0:
                    var_rainflow = list(datai.keys())


                # index for start/end of time series
                idx_s = np.argmax(datai["Time"] >= self.T0)
                idx_e = np.argmax(datai["Time"] >= self.TMax) + 1

                for var in var_rainflow:
                    ranges, means = fatpack.find_rainflow_ranges(datai[var][idx_s:idx_e], return_means=True)

                    rainflow[U][Seed][var] = {}
                    rainflow[U][Seed][var]['rf_amp']  = ranges.tolist()
                    rainflow[U][Seed][var]['rf_mean'] = means.tolist()
                    rainflow[U][Seed][var]['mean']    = float(np.mean(datai[var]))

        # save_yaml(self.FAST_resultsDirectory, 'rainflow.yaml', rainflow)
        # rainflow = load_yaml(self.FatigueFile, package=1)

        # Setup fatigue calculations
        U       = list(rainflow.keys())
        Seeds   = list(rainflow[U[0]].keys())
        chans   = list(rainflow[U[0]][Seeds[0]].keys())
        r_gage  = np.r_[0., self.R_out]
        r_gage /= r_gage[-1]
        simtime = self.simtime
        n_seeds = float(len(Seeds))
        n_gage  = len(r_gage)

        r       = (inputs['r']-inputs['r'][0])/(inputs['r'][-1]-inputs['r'][0])
        m_default = 8. # assume default m=10  (8 or 12 also reasonable)
        m       = [mi if mi > 0. else m_default for mi in inputs['m']]  # Assumption: if no S-N slope is given for a material, use default value TODO: input['m'] is not connected, only using the default currently

        eps_uts = inputs['Xt'][:,0]/inputs['E'][:,0]
        eps_ucs = inputs['Xc'][:,0]/inputs['E'][:,0]
        gamma_m = 1.#inputs['gamma_m']
        gamma_f = 1.#inputs['gamma_f']
        yrs     = 20.  # TODO
        t_life  = 60.*60.*24*365.24*yrs
        U_bar   = inputs['V_mean_iec']

        # pdf of wind speeds
        binwidth = np.diff(U)
        U_bins   = np.r_[[U[0] - binwidth[0]/2.], [np.mean([U[i-1], U[i]]) for i in range(1,len(U))], [U[-1] + binwidth[-1]/2.]]
        pdf = np.diff(RayleighCDF(U_bins, xbar=U_bar))
        if sum(pdf) < 0.9:
            print('Warning: Cummulative probability of wind speeds in rotor_loads_defl_strains.BladeFatigue is low, sum of weights: %f' % sum(pdf))
            print('Mean winds speed: %f' % U_bar)
            print('Simulated wind speeds: ', U)
            sys.stdout.flush()

        # Materials of analysis layers
        te_ss_var_ok       = False
        te_ps_var_ok       = False
        spar_cap_ss_var_ok = False
        spar_cap_ps_var_ok = False
        for i_layer in range(self.n_layers):
            if self.te_ss_var in self.layer_name:
                te_ss_var_ok        = True
            if self.te_ps_var in self.layer_name:
                te_ps_var_ok        = True
            if self.spar_cap_ss_var in self.layer_name:
                spar_cap_ss_var_ok  = True
            if self.spar_cap_ps_var in self.layer_name:
                spar_cap_ps_var_ok  = True

        # if te_ss_var_ok == False:
        #     print('The layer at the trailing edge suction side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.te_ss_var)
        # if te_ps_var_ok == False:
        #     print('The layer at the trailing edge pressure side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.te_ps_var)
        if spar_cap_ss_var_ok == False:
            print('The layer at the spar cap suction side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.spar_cap_ss_var)
        if spar_cap_ps_var_ok == False:
            print('The layer at the spar cap pressure side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.spar_cap_ps_var)
        sys.stdout.flush()

        # Get blade properties at gage locations
        y_tc       = remap2grid(r, inputs['y_tc'], r_gage)
        x_tc       = remap2grid(r, inputs['x_tc'], r_gage)
        chord      = remap2grid(r, inputs['chord'], r_gage)
        rthick     = remap2grid(r, inputs['rthick'], r_gage)
        pitch_axis = remap2grid(r, inputs['pitch_axis'], r_gage)
        EIyy       = remap2grid(r, inputs['beam:EIyy'], r_gage)
        EIxx       = remap2grid(r, inputs['beam:EIxx'], r_gage)

        te_ss_mats = np.floor(remap2grid(r, inputs['te_ss_mats'], r_gage, axis=0)) # materials is section
        te_ps_mats = np.floor(remap2grid(r, inputs['te_ps_mats'], r_gage, axis=0))
        sc_ss_mats = np.floor(remap2grid(r, inputs['sc_ss_mats'], r_gage, axis=0))
        sc_ps_mats = np.floor(remap2grid(r, inputs['sc_ps_mats'], r_gage, axis=0))

        c_TE       = chord*(1.-pitch_axis) + y_tc
        c_SC       = chord*rthick/2. + x_tc #this is overly simplistic, using maximum thickness point, should use the actual profiles
        sys.stdout.flush()

        C_miners_SC_SS_gage = np.zeros((n_gage, self.n_mat, 2))
        C_miners_SC_PS_gage = np.zeros((n_gage, self.n_mat, 2))
        C_miners_TE_SS_gage = np.zeros((n_gage, self.n_mat, 2))
        C_miners_TE_PS_gage = np.zeros((n_gage, self.n_mat, 2))

        # Map channels to output matrix
        chan_map   = {}
        for i_var, var in enumerate(chans):
            # Determine spanwise position
            if 'Root' in var:
                i_span = 0
            elif 'Spn' in var and 'M' in var:
                i_span = int(var.strip('Spn').split('M')[0])
            else:
                # not a spanwise output channel, skip
                print('Fatigue Model: Skipping channel: %s, not a spanwise moment' % var)
                sys.stdout.flush()
                chans.remove(var)
                continue
            # Determine if edgewise of flapwise moment
            if 'M' in var and 'x' in var:
                # Flapwise
                axis = 1
            elif 'M' in var and 'y' in var:
                # Edgewise
                axis = 0
            else:
                # not an edgewise / flapwise moment, skip
                print('Fatigue Model: Skipping channel: "%s", not an edgewise/flapwise moment' % var)
                sys.stdout.flush()
                continue

            chan_map[var] = {}
            chan_map[var]['i_gage'] = i_span
            chan_map[var]['axis']   = axis

        # Map composite sections
        composite_map = [['TE', 'SS', te_ss_var_ok],
                         ['TE', 'PS', te_ps_var_ok],
                         ['SC', 'SS', spar_cap_ss_var_ok],
                         ['SC', 'PS', spar_cap_ps_var_ok]]

        if self.options['modeling_options']['general']['verbosity']:
            print("Running Miner's Rule calculations")
            sys.stdout.flush()

        ########
        # Loop through composite sections, materials, output channels, and simulations (wind speeds * seeds)
        for comp_i in composite_map:

            #skip this composite section?
            if not comp_i[2]:
                continue

            #
            C_miners = np.zeros((n_gage, self.n_mat, 2))
            if comp_i[0]       == 'TE':
                c = c_TE
                if comp_i[1]   == 'SS':
                    mats = te_ss_mats
                elif comp_i[1] == 'PS':
                    mats = te_ps_mats
            elif comp_i[0]     == 'SC':
                c = c_SC
                if comp_i[1]   == 'SS':
                    mats = sc_ss_mats
                elif comp_i[1] == 'PS':
                    mats = sc_ps_mats

            for i_mat in range(self.n_mat):

                for i_var, var in enumerate(chans):
                    i_gage = chan_map[var]['i_gage']
                    axis   = chan_map[var]['axis']

                    # skip if material at this spanwise location is not included in the composite section
                    if mats[i_gage, i_mat] == 0.:
                        continue

                    # Determine if edgewise of flapwise moment
                    pitch_axis_i = pitch_axis[i_gage]
                    chord_i      = chord[i_gage]
                    c_i          = c[i_gage]
                    if axis == 0:
                        EI_i     = EIxx[i_gage]
                    else:
                        EI_i     = EIyy[i_gage]

                    for i_u, u in enumerate(U):
                        for i_s, seed in enumerate(Seeds):
                            M_mean = np.array(rainflow[u][seed][var]['rf_mean']) * 1.e3
                            M_amp  = np.array(rainflow[u][seed][var]['rf_amp']) * 1.e3

                            for M_mean_i, M_amp_i in zip(M_mean, M_amp):
                                n_cycles = 1.
                                eps_mean = M_mean_i*c_i/EI_i 
                                eps_amp  = M_amp_i*c_i/EI_i

                                if eps_amp != 0.:
                                    Nf = ((eps_uts[i_mat] + np.abs(eps_ucs[i_mat]) - np.abs(2.*eps_mean*gamma_m*gamma_f - eps_uts[i_mat] + np.abs(eps_ucs[i_mat]))) / (2.*eps_amp*gamma_m*gamma_f))**m[i_mat]
                                    n  = n_cycles * t_life * pdf[i_u] / (simtime * n_seeds)
                                    C_miners[i_gage, i_mat, axis]  += n/Nf

            # Assign outputs
            if comp_i[0] == 'SC' and comp_i[1] == 'SS':
                outputs['C_miners_SC_SS'] = remap2grid(r_gage, C_miners, r, axis=0)
            elif comp_i[0] == 'SC' and comp_i[1] == 'PS':
                outputs['C_miners_SC_PS'] = remap2grid(r_gage, C_miners, r, axis=0)
            # elif comp_i[0] == 'TE' and comp_i[1] == 'SS':
            #     outputs['C_miners_TE_SS'] = remap2grid(r_gage, C_miners, r, axis=0)
            # elif comp_i[0] == 'TE' and comp_i[1] == 'PS':
            #     outputs['C_miners_TE_PS'] = remap2grid(r_gage, C_miners, r, axis=0)

        return outputs, discrete_outputs

def RayleighCDF(x, xbar=10.):
    return 1.0 - np.exp(-np.pi/4.0*(x/xbar)**2)

class ModesElastoDyn(ExplicitComponent):
    """
    Component that adds a multiplicative factor to axial, torsional, and flap-edge coupling stiffness to mimic ElastoDyn
    
    Parameters
    ----------
    EA : numpy array[n_span], [N]
        1D array of the actual axial stiffness
    EIxy : numpy array[n_span], [Nm2]
        1D array of the actual flap-edge coupling stiffness
    GJ : numpy array[n_span], [Nm2]
        1D array of the actual torsional stiffness
    G  : numpy array[n_mat], [N/m2]
        1D array of the actual shear stiffness of the materials
    
    Returns
    -------
    EA_stiff : numpy array[n_span], [N]
        1D array of the stiff axial stiffness
    EIxy_stiff : numpy array[n_span], [Nm2]
        1D array of the stiff flap-edge coupling stiffness
    GJ_stiff : numpy array[n_span], [Nm2]
        1D array of the stiff torsional stiffness
    G_stiff  : numpy array[n_mat], [N/m2]
        1D array of the stiff shear stiffness of the materials
    
    """    
    def initialize(self):
        self.options.declare('modeling_options')

    def setup(self):
        n_span          = self.options['modeling_options']['blade']['n_span']
        n_mat           = self.options['modeling_options']['materials']['n_mat']

        self.add_input('EA',    val=np.zeros(n_span), units='N',        desc='axial stiffness')
        self.add_input('EIxy',  val=np.zeros(n_span), units='N*m**2',   desc='coupled flap-edge stiffness')
        self.add_input('GJ',    val=np.zeros(n_span), units='N*m**2',   desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')

        self.add_input('G',     val=np.zeros([n_mat, 3]), units='Pa',   desc='2D array of the shear moduli of the materials. Each row represents a material, the three columns represent G12, G13 and G23.')


        self.add_output('EA_stiff',  val=np.zeros(n_span), units='N',        desc='artifically stiff axial stiffness')
        self.add_output('EIxy_zero', val=np.zeros(n_span), units='N*m**2',   desc='artifically stiff coupled flap-edge stiffness')
        self.add_output('GJ_stiff',  val=np.zeros(n_span), units='N*m**2',   desc='artifically stiff torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_output('G_stiff',   val=np.zeros([n_mat, 3]), units='Pa',   desc='artificially stif 2D array of the shear moduli of the materials. Each row represents a material, the three columns represent G12, G13 and G23.')

    def compute(self, inputs, outputs):

        k = 10.

        outputs['EA_stiff']   = inputs['EA']   * k
        outputs['EIxy_zero']  = inputs['EIxy'] * 0.
        outputs['GJ_stiff']   = inputs['GJ']   * k
        outputs['G_stiff']    = inputs['G']    * k