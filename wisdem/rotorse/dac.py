import numpy as np
import os, sys
import copy
from openmdao.api import ExplicitComponent
from wisdem.ccblade import CCAirfoil
from wisdem.ccblade.Polar import Polar
import csv  # for exporting airfoil polar tables
import matplotlib.pyplot as plt

import multiprocessing as mp
from functools import partial
from wisdem.commonse.mpi_tools import MPI

def runXfoil(xfoil_path, x, y, Re, AoA_min=-9, AoA_max=25, AoA_inc=0.5, Ma = 0.0, multi_run=False, MPI_run=False):
    #This function is used to create and run xfoil simulations for a given set of airfoil coordinates

    # Set initial parameters needed in xfoil
    numNodes   = 310 # number of panels to use (260...but increases if needed)
    #dist_param = 0.15 # TE/LE panel density ratio (0.15)
    dist_param = 0.12 #This is current value that i am trying to help with convergence (!bem)
    #IterLimit = 100 # Maximum number of iterations to try and get to convergence
    IterLimit = 10 #This decreased IterLimit will speed up analysis (!bem)
    #panelBunch = 1.5 # Panel bunching parameter to bunch near larger changes in profile gradients (1.5)
    panelBunch = 1.6 #This is the value I am currently using to try and improve convergence (!bem)
    #rBunch = 0.15 # Region to LE bunching parameter (used to put additional panels near flap hinge) (0.15)
    rBunch = 0.08 #This is the current value that I am using (!bem)
    XT1 = 0.55 # Defining left boundary of bunching region on top surface (should be before flap)
    # XT1 = 1.0
    #XT2 = 0.85 # Defining right boundary of bunching region on top surface (should be after flap)
    XT2 = 0.9 #This is the value I am currently using (!bem)
    # XT2 = 1.0
    XB1 = 0.55 # Defining left boundary of bunching region on bottom surface (should be before flap)
    # XB1 = 1.0
    #XB2 = 0.85 # Defining right boundary of bunching region on bottom surface (should be after flap)
    XB2 = 0.9 #This is the current value that I am using (!bem)
    # XB2 = 1.0
    runFlag = 1 # Flag used in error handling
    dfdn = -0.5 # Change in angle of attack during initialization runs down to AoA_min
    runNum = 0 # Initialized run number
    dfnFlag = -10 # This flag is used to determine if xfoil needs to be re-run if the simulation fails due to convergence issues at low angles of attack

    # Set filenames 
    if multi_run or MPI_run:
        pid = mp.current_process().pid
        LoadFlnmAF = 'airfoil_r{}.txt'.format(pid)
        saveFlnmPolar = 'Polar_r{}.txt'.format(pid)
        xfoilFlnm  = 'xfoil_input_r{}.txt'.format(pid)
    # if MPI_run:
    #     rank = MPI.COMM_WORLD.Get_rank()
    #     LoadFlnmAF = 'airfoil_r{}.txt'.format(rank) # This is a temporary file that will be deleted after it is no longer needed
    #     saveFlnmPolar = 'Polar_r{}.txt'.format(rank) # file name of outpur xfoil polar (can be useful to look at during debugging...can also delete at end if you don't want it stored)
    #     xfoilFlnm  = 'xfoil_input_r{}.txt'.format(rank) # Xfoil run script that will be deleted after it is no longer needed
    else:
        LoadFlnmAF = 'airfoil.txt' # This is a temporary file that will be deleted after it is no longer needed
        saveFlnmPolar = 'Polar.txt' # file name of outpur xfoil polar (can be useful to look at during debugging...can also delete at end if you don't want it stored)
        xfoilFlnm  = 'xfoil_input.txt' # Xfoil run script that will be deleted after it is no longer needed

    while numNodes < 480 and runFlag > 0:
        # Cleaning up old files to prevent replacement issues
        if os.path.exists(saveFlnmPolar):
            os.remove(saveFlnmPolar)
        if os.path.exists(xfoilFlnm):
            os.remove(xfoilFlnm)
        if os.path.exists(LoadFlnmAF):
            os.remove(LoadFlnmAF)

        # Writing temporary airfoil coordinate file for use in xfoil
        dat=np.array([x,y])
        np.savetxt(LoadFlnmAF, dat.T, fmt=['%f','%f'])

        # %% Writes the Xfoil run script to read in coordinates, create flap, re-pannel, and create polar
        # Create the airfoil with flap
        fid = open(xfoilFlnm,"w")
        fid.write("PLOP \n G \n\n") # turn off graphics
        fid.write("LOAD \n")
        fid.write( LoadFlnmAF + "\n" + "\n") # name of .txt file with airfoil coordinates
        # fid.write( self.AFName + "\n") # set name of airfoil (internal to xfoil)
        fid.write("GDES \n") # enter into geometry editing tools in xfoil
        fid.write("UNIT \n") # normalize profile to unit chord
        fid.write("EXEC \n \n") # move buffer airfoil to current airfoil

        # Re-panel with specified number of panes and LE/TE panel density ratio
        fid.write("PPAR\n")
        fid.write("N \n" )
        fid.write(str(numNodes) + "\n")
        fid.write("P \n") # set panel bunching parameter
        fid.write(str(panelBunch) + " \n")
        fid.write("T \n") # set TE/LE panel density ratio
        fid.write( str(dist_param) + "\n")
        fid.write("R \n") # set region panel bunching ratio
        fid.write(str(rBunch) + " \n")
        fid.write("XT \n") # set region panel bunching bounds on top surface
        fid.write(str(XT1) +" \n" + str(XT2) + " \n")
        fid.write("XB \n") # set region panel bunching bounds on bottom surface
        fid.write(str(XB1) +" \n" + str(XB2) + " \n")
        fid.write("\n\n")

        # Set Simulation parameters (Re and max number of iterations)
        fid.write("OPER\n")
        fid.write("VISC \n")
        fid.write( str(Re) + "\n") # this sets Re to value specified in yaml file as an input
        #fid.write( "5000000 \n") # bem: I was having trouble geting convergence for some of the thinner airfoils at the tip for the large Re specified in the yaml, so I am hard coding in Re (5e6 is the highest I was able to get to using these paneling parameters)
        fid.write("MACH\n")
        fid.write(str(Ma)+" \n")
        fid.write("ITER \n")
        fid.write( str(IterLimit) + "\n")

        # Run simulations for range of AoA

        if dfnFlag > 0: # bem: This if statement is for the case when there are issues getting convergence at AoA_min.  It runs a preliminary set of AoA's down to AoA_min (does not save them)
            for ii in range(int((0.0-AoA_min)/AoA_inc+1)):
                fid.write("ALFA "+ str(0.0-ii*float(AoA_inc)) +"\n")

        fid.write("PACC\n\n\n") #Toggle saving polar on
        #fid.write("ASEQ 0 " + str(AoA_min) + " " + str(dfdn) + "\n") # The preliminary runs are just to get an initialize airfoil solution at min AoA so that the actual runs will not become unstable

        for ii in range(int((AoA_max-AoA_min)/AoA_inc+1)): # bem: run each AoA seperately (makes polar generation more convergence error tolerant)
            fid.write("ALFA "+ str(AoA_min+ii*float(AoA_inc)) +"\n")

        #fid.write("ASEQ " + str(AoA_min) + " " + "16" + " " + str(AoA_inc) + "\n") #run simulations for desired range of AoA using a coarse step size in AoA up to 16 deg
        #fid.write("ASEQ " + "16.5" + " " + str(AoA_max) + " " + "0.1" + "\n") #run simulations for desired range of AoA using a fine AoA increment up to final AoA to help with convergence issues at high Re
        fid.write("PWRT\n") #Toggle saving polar off
        fid.write(saveFlnmPolar + " \n \n")
        fid.write("QUIT \n")
        fid.close()

        # Run the XFoil calling command
        os.system(xfoil_path + " < " + xfoilFlnm + " > NUL") # <<< runs XFoil !
        try:
            flap_polar = np.loadtxt(saveFlnmPolar,skiprows=12)
        except:
            flap_polar = []  # in case no convergence was achieved


        # Error handling (re-run simulations with more panels if there is not enough data in polars)
        if np.size(flap_polar) < 3: # This case is if there are convergence issues at the lowest angles of attack
            plen = 0
            a0 = 0
            a1 = 0
            dfdn = -0.25 # decrease AoA step size during initialization to try and get convergence in the next run
            dfnFlag = 1 # Set flag to run initialization AoA down to AoA_min
            print('XFOIL convergence issues')
        else:
            plen = len(flap_polar[:,0]) # Number of AoA's in polar
            a0 = flap_polar[-1,0] # Maximum AoA in Polar
            a1 = flap_polar[0,0] # Minimum AoA in Polar
            dfnFlag = -10 # Set flag so that you don't need to run initialization sequence

        if a0 > 19. and plen >= 40 and a1 < -12.5: # The a0 > 19 is to check to make sure polar entered into stall regiem plen >= 40 makes sure there are enough AoA's in polar for interpolation and a1 < -15 makes sure polar contains negative stall.
            runFlag = -10 # No need ro re-run polar
        else:
            numNodes += 50 # Re-run with additional panels
            runNum += 1 # Update run number
            if numNodes > 480:
                Warning('NO convergence in XFoil achieved!')
            print('Refining paneling to ' + str(numNodes) + ' nodes')

    # Load back in polar data to be saved in instance variables
    #flap_polar = np.loadtxt(saveFlnmPolar,skiprows=12) # (note, we are assuming raw Xfoil polars when skipping the first 12 lines)
    # self.af_flap_polar = flap_polar
    # self.flap_polar_flnm = saveFlnmPolar # Not really needed unless you keep the files and want to load them later

    # Delete Xfoil run script file
    if os.path.exists(xfoilFlnm):
        os.remove(xfoilFlnm)
    if os.path.exists(saveFlnmPolar): # bem: For now leave the files, but eventually we can get rid of them (remove # in front of commands) so that we don't have to store them
        os.remove(saveFlnmPolar)
    if os.path.exists(LoadFlnmAF):
        os.remove(LoadFlnmAF)


    return flap_polar

class RunXFOIL(ExplicitComponent):
    # Openmdao component to run XFOIL and re-compute polars
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')
        
    def setup(self):
        blade_init_options = self.options['modeling_options']['blade']
        self.n_span        = n_span     = blade_init_options['n_span']
        self.n_te_flaps    = n_te_flaps = blade_init_options['n_te_flaps']
        af_init_options    = self.options['modeling_options']['airfoils']
        self.n_tab         = af_init_options['n_tab']
        self.n_aoa         = n_aoa      = af_init_options['n_aoa'] # Number of angle of attacks
        self.n_Re          = n_Re      = af_init_options['n_Re'] # Number of Reynolds, so far hard set at 1
        self.n_tab         = n_tab     = af_init_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
        self.n_xy          = n_xy      = af_init_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
        self.xfoil_path    = af_init_options['xfoil_path']

        # Use openfast cores for parallelization of xfoil 
        # nja - Probably want to change this so XFOIL parallelization is a flag?
        FASTpref = self.options['modeling_options']['openfast']
        xfoilpref = self.options['modeling_options']['xfoil']

        try:
            if xfoilpref['run_parallel']:
                self.cores = mp.cpu_count()
            else:
                self.cores = 1
        except KeyError:
            self.cores = 1
        
        if MPI and self.options['modeling_options']['Analysis_Flags']['OpenFAST']:
            self.mpi_comm_map_down = FASTpref['analysis_settings']['mpi_comm_map_down']

        # Inputs blade outer shape
        self.add_input('s',          val=np.zeros(n_span),                      desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')
        self.add_input('r',             val=np.zeros(n_span), units='m',   desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('coord_xy_interp',  val=np.zeros((n_span, n_xy, 2)),     desc='3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations.')
        self.add_input('chord',         val=np.zeros(n_span), units='m',   desc='chord length at each section')

        # Inputs flaps
        self.add_input('span_end',   val=np.zeros(n_te_flaps),                  desc='1D array of the positions along blade span where the trailing edge flap(s) end. Only values between 0 and 1 are meaningful.')
        self.add_input('span_ext',   val=np.zeros(n_te_flaps),                  desc='1D array of the extensions along blade span of the trailing edge flap(s). Only values between 0 and 1 are meaningful.')
        self.add_input('chord_start',val=np.zeros(n_te_flaps),                  desc='1D array of the positions along chord where the trailing edge flap(s) start. Only values between 0 and 1 are meaningful.')
        self.add_input('delta_max_pos', val=np.zeros(n_te_flaps), units='rad',  desc='1D array of the max angle of the trailing edge flaps.')
        self.add_input('delta_max_neg', val=np.zeros(n_te_flaps), units='rad',  desc='1D array of the min angle of the trailing edge flaps.')

        # Inputs control
        self.add_input('max_TS',         val=0.0, units='m/s',     desc='Maximum allowed blade tip speed.')
        self.add_input('rated_TSR',      val=0.0,                  desc='Constant tip speed ratio in region II.')

        # Inputs environment
        self.add_input('rho_air',      val=1.225,        units='kg/m**3',    desc='Density of air')
        self.add_input('mu_air',       val=1.81e-5,      units='kg/(m*s)',   desc='Dynamic viscosity of air')
        self.add_input('speed_sound_air',  val=340.,     units='m/s',        desc='Speed of sound in air.')

        # Inputs polars
        self.add_input('aoa',       val=np.zeros(n_aoa),        units='rad',    desc='1D array of the angles of attack used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.')
        self.add_input('cl_interp',        val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the lift coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_input('cd_interp',        val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the drag coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_input('cm_interp',        val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the moment coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')

        # Outputs flap geometry
        self.add_output('span_start', val=np.zeros(n_te_flaps),                  desc='1D array of the positions along blade span where the trailing edge flap(s) start. Only values between 0 and 1 are meaningful.')
        
        # Output polars
        self.add_output('cl_interp_flaps',  val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the lift coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_output('cd_interp_flaps',  val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the drag coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_output('cm_interp_flaps',  val=np.zeros((n_span, n_aoa, n_Re, n_tab)),   desc='4D array with the moment coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.')
        self.add_output('flap_angles',      val=np.zeros((n_span, n_Re, n_tab)), units = 'deg',   desc='3D array with the flap angles of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the Reynolds number, dimension 2 is along the number of tabs, which may describe multiple sets at the same station.')
        self.add_output('Re_loc',           val=np.zeros((n_span, n_Re, n_tab)),   desc='3D array with the Re. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the Reynolds number, dimension 2 is along the number of tabs, which may describe multiple sets at the same station.')
        self.add_output('Ma_loc',           val=np.zeros((n_span, n_Re, n_tab)),   desc='3D array with the Mach number. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the Reynolds number, dimension 2 is along the number of tabs, which may describe multiple sets at the same station.')

        # initialize saved data polar data. 
        # - This is filled if we're not changing the flaps, so we don't need to re-run xfoil every time
        self.saved_polar_data = {}

    def compute(self, inputs, outputs):

        # If trailing edge flaps are present, compute the perturbed profiles with XFOIL
        self.flap_profiles = [{} for i in range(self.n_span)]
        outputs['span_start'] = inputs['span_end'] - inputs['span_ext']
        if self.n_te_flaps > 0:
            try:
                from scipy.ndimage import gaussian_filter
            except:
                print('Cannot import the library gaussian_filter from scipy. Please check the conda environment and potential conflicts between numpy and scipy')
            for i in range(self.n_span):
                # Loop through the flaps specified in yaml file
                for k in range(self.n_te_flaps):
                    # Only create flap geometries where the yaml file specifies there is a flap (Currently going to nearest blade station location)
                    if inputs['s'][i] >= outputs['span_start'][k] and inputs['s'][i] <= inputs['span_end'][k]: 
                        self.flap_profiles[i]['flap_angles']= []
                        # Initialize the profile coordinates to zeros
                        self.flap_profiles[i]['coords']     = np.zeros([self.n_xy,2,self.n_tab]) 
                            # Ben:I am not going to force it to include delta=0.  If this is needed, a more complicated way of getting flap deflections to calculate is needed.
                        flap_angles = np.linspace(inputs['delta_max_neg'][k],inputs['delta_max_pos'][k],self.n_tab) * 180. / np.pi
                        # Loop through the flap angles
                        for ind, fa in enumerate(flap_angles):
                            # NOTE: negative flap angles are deflected to the suction side, i.e. positively along the positive z- (radial) axis
                            af_flap = CCAirfoil(np.array([1,2,3]), np.array([100]), np.zeros(3), np.zeros(3), np.zeros(3), inputs['coord_xy_interp'][i,:,0], inputs['coord_xy_interp'][i,:,1], "Profile"+str(i)) # bem:I am creating an airfoil name based on index...this structure/naming convention is being assumed in CCAirfoil.runXfoil() via the naming convention used in CCAirfoil.af_flap_coords(). Note that all of the inputs besides profile coordinates and name are just dummy varaiables at this point.
                            af_flap.af_flap_coords(self.xfoil_path, fa,  inputs['chord_start'][k],0.5,200) #bem: the last number is the number of points in the profile.  It is currently being hard coded at 200 but should be changed to make sure it is the same number of points as the other profiles
                            # self.flap_profiles[i]['coords'][:,0,ind] = af_flap.af_flap_xcoords # x-coords from xfoil file with flaps
                            # self.flap_profiles[i]['coords'][:,1,ind] = af_flap.af_flap_ycoords # y-coords from xfoil file with flaps
                            # self.flap_profiles[i]['coords'][:,0,ind] = af_flap.af_flap_xcoords  # x-coords from xfoil file with flaps and NO gaussian filter for smoothing
                            # self.flap_profiles[i]['coords'][:,1,ind] = af_flap.af_flap_ycoords  # y-coords from xfoil file with flaps and NO gaussian filter for smoothing
                            try:
                                self.flap_profiles[i]['coords'][:,0,ind] = gaussian_filter(af_flap.af_flap_xcoords, sigma=1) # x-coords from xfoil file with flaps and gaussian filter for smoothing
                                self.flap_profiles[i]['coords'][:,1,ind] = gaussian_filter(af_flap.af_flap_ycoords, sigma=1) # y-coords from xfoil file with flaps and gaussian filter for smoothing
                            except:
                                self.flap_profiles[i]['coords'][:,0,ind] = af_flap.af_flap_xcoords
                                self.flap_profiles[i]['coords'][:,1,ind] = af_flap.af_flap_ycoords
                            self.flap_profiles[i]['flap_angles'].append([])
                            self.flap_profiles[i]['flap_angles'][ind] = fa # Putting in flap angles to blade for each profile (can be used for debugging later)

                        # # ** The code below will plot the first three flap deflection profiles (in the case where there are only 3 this will correspond to max negative, zero, and max positive deflection cases)
                        # font = {'family': 'Times New Roman',
                        #         'weight': 'normal',
                        #         'size': 18}
                        # plt.rc('font', **font)
                        # plt.figure
                        # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                        # # plt.plot(self.flap_profiles[i]['coords'][:,0,0], self.flap_profiles[i]['coords'][:,1,0], 'r',self.flap_profiles[i]['coords'][:,0,1], self.flap_profiles[i]['coords'][:,1,1], 'k',self.flap_profiles[i]['coords'][:,0,2], self.flap_profiles[i]['coords'][:,1,2], 'b')
                        # plt.plot(self.flap_profiles[i]['coords'][:, 0, 0],
                        #         self.flap_profiles[i]['coords'][:, 1, 0], '.r',
                        #         self.flap_profiles[i]['coords'][:, 0, 2],
                        #         self.flap_profiles[i]['coords'][:, 1, 2], '.b',
                        #         self.flap_profiles[i]['coords'][:, 0, 1],
                        #         self.flap_profiles[i]['coords'][:, 1, 1], '.k')
                        
                        # # plt.xlabel('x')
                        # # plt.ylabel('y')
                        # plt.axis('equal')
                        # plt.axis('off')
                        # plt.tight_layout()
                        # plt.show()
                        # # # plt.savefig('temp/airfoil_polars/NACA63-self.618_flap_profiles.png', dpi=300)
                        # # # plt.savefig('temp/airfoil_polars/FFA-W3-self.211_flap_profiles.png', dpi=300)
                        # # # plt.savefig('temp/airfoil_polars/FFA-W3-self.241_flap_profiles.png', dpi=300)
                        # # # plt.savefig('temp/airfoil_polars/FFA-W3-self.301_flap_profiles.png', dpi=300)


        # ----------------------------------------------------- #
        # Determine airfoil polar tables blade sections #

        #  ToDo: shape of blade['profile'] differs from self.flap_profiles <<< change to same shape
        # only execute when flag_airfoil_polars = True
        flag_airfoil_polars = False  # <<< ToDo get through Yaml in the future ?!?

        if flag_airfoil_polars == True:
            # OUTDATED!!! - NJA

            af_orig_grid = blade['outer_shape_bem']['airfoil_position']['grid']
            af_orig_labels = blade['outer_shape_bem']['airfoil_position']['labels']
            af_orig_chord_grid = blade['outer_shape_bem']['chord']['grid']  # note: different grid than airfoil labels
            af_orig_chord_value = blade['outer_shape_bem']['chord']['values']

            for i_af_orig in range(len(af_orig_grid)):
                if af_orig_labels[i_af_orig] != 'circular':
                    print('Determine airfoil polars:')

                    # check index of chord grid for given airfoil radial station
                    for i_chord_grid in range(len(af_orig_chord_grid)):
                        if af_orig_chord_grid[i_chord_grid] == af_orig_grid[i_af_orig]:
                            c = af_orig_chord_value[i_chord_grid]  # get chord length at current radial station of original airfoil
                            c_index = i_chord_grid


                    flag_coord = 3  # Define which blade airfoil outer shapes coordinates to use (watch out for consistency throughout the model/analysis !!!)
                    #  Get orig coordinates (too many for XFoil)
                    if flag_coord == 1:
                        x_af = self.wt_ref['airfoils'][1]['coordinates']['x']
                        y_af = self.wt_ref['airfoils'][1]['coordinates']['y']


                    #  Get interpolated coords
                    if flag_coord == 2:
                        x_af = blade['profile'][:,0,c_index]
                        y_af = blade['profile'][:,1,c_index]


                    # create coords using ccblade and calling XFoil in order to be consistent with the flap method
                    if flag_coord == 3:
                        flap_angle = 0  # no te-flaps !
                        af_temp = CCAirfoil(np.array([1,2,3]), np.array([100]), np.zeros(3), np.zeros(3), np.zeros(3), blade['profile'][:,0,c_index],blade['profile'][:,1,c_index], "Profile"+str(c_index)) # bem:I am creating an airfoil name based on index...this structure/naming convention is being assumed in CCAirfoil.runXfoil() via the naming convention used in CCAirfoil.af_flap_coords(). Note that all of the inputs besides profile coordinates and name are just dummy varaiables at this point.
                        af_temp.af_flap_coords(self.xfoil_path, flap_angle,  0.8, 0.5, 200) #bem: the last number is the number of points in the profile.  It is currently being hard coded at 200 but should be changed to make sure it is the same number of points as the other profiles
                        # x_af = af_temp.af_flap_xcoords
                        # y_af = af_temp.af_flap_ycoords

                        x_af = gaussian_filter(af_temp.af_flap_xcoords, sigma=1)  # gaussian filter for smoothing (in order to be consistent with flap capabilities)
                        y_af = gaussian_filter(af_temp.af_flap_ycoords, sigma=1)  # gaussian filter for smoothing (in order to be consistent with flap capabilities)


                    rR = af_orig_grid[i_af_orig]  # non-dimensional blade radial station at cross section
                    R = blade['pf']['r'][-1]  # blade (global) radial length
                    tsr = blade['config']['tsr']  # tip-speed ratio
                    maxTS = blade['assembly']['control']['maxTS']  # max blade-tip speed (m/s) from yaml file
                    KinVisc = blade['environment']['air_data']['KinVisc']  # Kinematic viscosity (m^2/s) from yaml file
                    SpdSound = blade['environment']['air_data']['SpdSound']  # speed of sound (m/s) from yaml file
                    Re_af_orig_loc = c * maxTS * rR / KinVisc
                    Ma_af_orig_loc = maxTS * rR / SpdSound

                    print('Run xfoil for airfoil ' + af_orig_labels[i_af_orig] + ' at span section r/R = ' + str(rR) + ' with Re equal to ' + str(Re_af_orig_loc) + ' and Ma equal to ' + str(Ma_af_orig_loc))
                    # if af_orig_labels[i_af_orig] == 'NACA63-618':  # reduce AoAmin for (thinner) airfoil at the blade tip due to convergence reasons in XFoil
                    #     data = self.runXfoil(x_af, y_af_orig, Re_af_orig_loc, -13.5, 25., 0.5, Ma_af_orig_loc)
                    # else:
                    data = self.runXfoil(x_af, y_af, Re_af_orig_loc, -20., 25., 0.5, Ma_af_orig_loc)

                    oldpolar = Polar(Re_af_orig_loc, data[:, 0], data[:, 1], data[:, 2], data[:, 4])  # p[:,0] is alpha, p[:,1] is Cl, p[:,2] is Cd, p[:,4] is Cm

                    polar3d = oldpolar.correction3D(rR, c/R, tsr)  # Apply 3D corrections (made sure to change the r/R, c/R, and tsr values appropriately when calling AFcorrections())
                    cdmax = 1.5
                    polar = polar3d.extrapolate(cdmax)  # Extrapolate polars for alpha between -180 deg and 180 deg

                    cl_interp = np.interp(np.degrees(alpha), polar.alpha, polar.cl)
                    cd_interp = np.interp(np.degrees(alpha), polar.alpha, polar.cd)
                    cm_interp = np.interp(np.degrees(alpha), polar.alpha, polar.cm)

                    # --- PROFILE ---#
                    # write profile (that was input to XFoil; although previously provided in the yaml file)
                    with open('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_profile.csv', 'w') as profile_csvfile:
                        profile_csvfile_writer = csv.writer(profile_csvfile, delimiter=',')
                        profile_csvfile_writer.writerow(['x', 'y'])
                        for i in range(len(x_af)):
                            profile_csvfile_writer.writerow([x_af[i], y_af[i]])

                    # plot profile
                    plt.figure(i_af_orig)
                    plt.plot(x_af, y_af, 'k')
                    plt.axis('equal')
                    # plt.show()
                    plt.savefig('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_profile.png')
                    plt.close(i_af_orig)

                    # --- CL --- #
                    # write cl
                    with open('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_cl.csv', 'w') as cl_csvfile:
                        cl_csvfile_writer = csv.writer(cl_csvfile, delimiter=',')
                        cl_csvfile_writer.writerow(['alpha, deg', 'alpha, rad', 'cl'])
                        for i in range(len(cl_interp)):
                            cl_csvfile_writer.writerow([np.degrees(alpha[i]), alpha[i], cl_interp[i]])

                    # plot cl
                    plt.figure(i_af_orig)
                    fig, ax = plt.subplots(1,1, figsize= (8,5))
                    plt.plot(np.degrees(alpha), cl_interp, 'b')
                    plt.xlim(xmin=-25, xmax=25)
                    plt.grid(True)
                    autoscale_y(ax)
                    plt.xlabel('Angles of attack, deg')
                    plt.ylabel('Lift coefficient')
                    # plt.show()
                    plt.savefig('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_cl.png')
                    plt.close(i_af_orig)

                    # write cd
                    with open('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_cd.csv', 'w') as cd_csvfile:
                        cd_csvfile_writer = csv.writer(cd_csvfile, delimiter=',')
                        cd_csvfile_writer.writerow(['alpha, deg', 'alpha, rad', 'cd'])
                        for i in range(len(cd_interp)):
                            cd_csvfile_writer.writerow([np.degrees(alpha[i]), alpha[i], cd_interp[i]])

                    # plot cd
                    plt.figure(i_af_orig)
                    fig, ax = plt.subplots(1,1, figsize= (8,5))
                    plt.plot(np.degrees(alpha), cd_interp, 'r')
                    plt.xlim(xmin=-25, xmax=25)
                    plt.grid(True)
                    autoscale_y(ax)
                    plt.xlabel('Angles of attack, deg')
                    plt.ylabel('Drag coefficient')
                    # plt.show()
                    plt.savefig('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_cd.png')
                    plt.close(i_af_orig)

                    # write cm
                    with open('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_cm.csv', 'w') as cm_csvfile:
                        cm_csvfile_writer = csv.writer(cm_csvfile, delimiter=',')
                        cm_csvfile_writer.writerow(['alpha, deg', 'alpha, rad', 'cm'])
                        for i in range(len(cm_interp)):
                            cm_csvfile_writer.writerow([np.degrees(alpha[i]), alpha[i], cm_interp[i]])

                    # plot cm
                    plt.figure(i_af_orig)
                    fig, ax = plt.subplots(1,1, figsize= (8,5))
                    plt.plot(np.degrees(alpha), cm_interp, 'g')
                    plt.xlim(xmin=-25, xmax=25)
                    plt.grid(True)
                    autoscale_y(ax)
                    plt.xlabel('Angles of attack, deg')
                    plt.ylabel('Torque coefficient')
                    # plt.show()
                    plt.savefig('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_cm.png')
                    plt.close(i_af_orig)

                    # write additional information (Re, Ma, r/R)
                    with open('temp/airfoil_polars/' + af_orig_labels[i_af_orig] + '_add_info.csv', 'w') as csvfile:
                        csvfile_writer = csv.writer(csvfile, delimiter=',')
                        csvfile_writer.writerow(['Re', 'Ma', 'r/R'])
                        csvfile_writer.writerow([Re_af_orig_loc, Ma_af_orig_loc, rR])

                    plt.close('all')
        # ------------------------------------------------------------ #
        # Determine airfoil polar tables for blade sections with flaps #

        self.R        = inputs['r'][-1]  # Rotor radius in meters
        self.tsr      = inputs['rated_TSR']  # tip-speed ratio
        self.maxTS    = inputs['max_TS']  # max blade-tip speed (m/s) from yaml file
        self.KinVisc  = inputs['mu_air'] / inputs['rho_air']  # Kinematic viscosity (m^2/s) from yaml file
        self.SpdSound = inputs['speed_sound_air'] # speed of sound (m/s) from yaml file
        
        # Initialize
        cl_interp_flaps = inputs['cl_interp']
        cd_interp_flaps = inputs['cd_interp']
        cm_interp_flaps = inputs['cm_interp']
        fa_control = np.zeros((self.n_span, self.n_Re, self.n_tab))
        Re_loc = np.zeros((self.n_span, self.n_Re, self.n_tab))
        Ma_loc = np.zeros((self.n_span, self.n_Re, self.n_tab))

        # Get polars for flap angles
        if self.n_te_flaps > 0:
            if 'cl_interp_flaps' not in self.saved_polar_data.keys():
                
                run_xfoil_params = {}
                # Self
                run_xfoil_params['xfoil_path'] = self.xfoil_path
                run_xfoil_params['cores'] = self.cores
                run_xfoil_params['n_span'] = self.n_span
                run_xfoil_params['n_Re'] = self.n_Re
                run_xfoil_params['n_tab'] = self.n_tab
                run_xfoil_params['flap_profiles'] = self.flap_profiles
                run_xfoil_params['R'] = self.R
                run_xfoil_params['tsr'] = self.tsr
                run_xfoil_params['maxTS'] = self.maxTS
                run_xfoil_params['KinVisc'] = self.KinVisc
                run_xfoil_params['SpdSound'] = self.SpdSound
                # inputs
                run_xfoil_params['cl_interp'] = inputs['cl_interp']
                run_xfoil_params['cd_interp'] = inputs['cd_interp']
                run_xfoil_params['cm_interp'] = inputs['cm_interp']
                run_xfoil_params['chord'] = inputs['chord']
                run_xfoil_params['s'] = inputs['s']
                run_xfoil_params['r'] = inputs['r']
                run_xfoil_params['aoa'] = inputs['aoa']


                # Run XFoil as multiple processors with MPI
                if MPI:
                    run_xfoil_params['run_MPI'] = True
                    # mpi comm management
                    comm = MPI.COMM_WORLD
                    rank = comm.Get_rank()
                    sub_ranks = self.mpi_comm_map_down[rank]
                    size = len(sub_ranks)
                    
                    N_cases = self.n_span # total number of airfoil sections
                    N_loops = int(np.ceil(float(N_cases)/float(size)))  # number of times function calls need to "loop"

                    # iterate loops, populate polar tables
                    for i in range(N_loops):
                        idx_s = i*size
                        idx_e = min((i+1)*size, N_cases)

                        for idx, afi in enumerate(np.arange(idx_s,idx_e)):
                            data = [partial(get_flap_polars, run_xfoil_params), afi]
                            rank_j = sub_ranks[idx]
                            comm.send(data, dest=rank_j, tag=0)

                        # for rank_j in sub_ranks:
                        for idx, afi in enumerate(np.arange(idx_s, idx_e)):
                            rank_j = sub_ranks[idx]
                            polars_separate_af = comm.recv(source=rank_j, tag=1)
                            cl_interp_flaps[afi,:,:,:] = polars_separate_af[0]
                            cd_interp_flaps[afi,:,:,:] = polars_separate_af[1]
                            cm_interp_flaps[afi,:,:,:] = polars_separate_af[2]
                            fa_control[afi,:,:] = polars_separate_af[3]
                            Re_loc[afi,:,:] = polars_separate_af[4]
                            Ma_loc[afi,:,:] = polars_separate_af[5]
                    
                    # for afi in range(self.n_span):
                    #     # re-structure outputs
                        
                # Multiple processors, but not MPI
                elif self.cores > 1:
                    run_xfoil_params['run_multi'] = True

                    # separate airfoil sections w/ and w/o flaps
                    af_with_flaps = []
                    af_without_flaps = []
                    for afi in range(len(run_xfoil_params['flap_profiles'])):
                        if 'coords' in run_xfoil_params['flap_profiles'][afi]:
                            af_with_flaps.append(afi)
                        else:
                            af_without_flaps.append(afi)

                    print('Parallelizing Xfoil on {} cores'.format(self.cores))
                    pool = mp.Pool(self.cores)
                    polars_separate_flaps = pool.map(
                        partial(get_flap_polars, run_xfoil_params), af_with_flaps)
                    # parallelize flap-specific calls for better efficiency
                    polars_separate_noflaps = pool.map(
                        partial(get_flap_polars, run_xfoil_params), af_without_flaps)
                    pool.close()
                    pool.join()

                    for i, afi in enumerate(af_with_flaps):
                        cl_interp_flaps[afi,:,:,:] = polars_separate_flaps[i][0]
                        cd_interp_flaps[afi,:,:,:] = polars_separate_flaps[i][1]
                        cm_interp_flaps[afi,:,:,:] = polars_separate_flaps[i][2]
                        fa_control[afi,:,:] = polars_separate_flaps[i][3]
                        Re_loc[afi,:,:] = polars_separate_flaps[i][4]
                        Ma_loc[afi,:,:] = polars_separate_flaps[i][5]

                    for i, afi in enumerate(af_without_flaps):
                        cl_interp_flaps[afi,:,:,:] = polars_separate_noflaps[i][0]
                        cd_interp_flaps[afi,:,:,:] = polars_separate_noflaps[i][1]
                        cm_interp_flaps[afi,:,:,:] = polars_separate_noflaps[i][2]
                        fa_control[afi,:,:] = polars_separate_noflaps[i][3]
                        Re_loc[afi,:,:] = polars_separate_noflaps[i][4]
                        Ma_loc[afi,:,:] = polars_separate_noflaps[i][5]
                                            
                else:
                    for afi in range(self.n_span): # iterate number of radial stations for various airfoil tables
                        cl_interp_flaps_af, cd_interp_flaps_af, cm_interp_flaps_af, fa_control_af, Re_loc_af, Ma_loc_af = get_flap_polars(run_xfoil_params, afi)

                        cl_interp_flaps[afi,:,:,:] = cl_interp_flaps_af
                        cd_interp_flaps[afi,:,:,:] = cd_interp_flaps_af
                        cm_interp_flaps[afi,:,:,:] = cm_interp_flaps_af
                        fa_control[afi,:,:] = fa_control_af
                        Re_loc[afi,:,:] = Re_loc_af
                        Ma_loc[afi,:,:] = Ma_loc_af

                if not any([self.options['opt_options']['optimization_variables']['blade']['dac']['te_flap_ext']['flag'],
                            self.options['opt_options']['optimization_variables']['blade']['dac']['te_flap_end']['flag']]):
                    self.saved_polar_data['cl_interp_flaps'] = copy.copy(cl_interp_flaps)
                    self.saved_polar_data['cd_interp_flaps'] = copy.copy(cd_interp_flaps)
                    self.saved_polar_data['cm_interp_flaps'] = copy.copy(cm_interp_flaps)
                    self.saved_polar_data['fa_control'] = copy.copy(fa_control)
                    self.saved_polar_data['Re_loc'] = copy.copy(Re_loc)
                    self.saved_polar_data['Ma_loc'] = copy.copy(Ma_loc)
                    
            else:
                # load xfoil data from previous runs
                print('Skipping XFOIL and loading blade polar data from previous iteration.')
                cl_interp_flaps = self.saved_polar_data['cl_interp_flaps']
                cd_interp_flaps = self.saved_polar_data['cd_interp_flaps']
                cm_interp_flaps = self.saved_polar_data['cm_interp_flaps']
                fa_control = self.saved_polar_data['fa_control']  
                Re_loc = self.saved_polar_data['Re_loc']       
                Ma_loc = self.saved_polar_data['Ma_loc']       



                    # else:  # no flap at specific radial location (but in general 'aerodynamic_control' is defined in blade from yaml)
                    #     # for j in range(n_Re): # ToDo incorporade variable Re capability
                    #     for ind in range(self.n_tab):  # fill all self.n_tab slots even though no flaps exist at current radial position
                    #         c = inputs['chord'][afi]  # blade chord length at cross section
                    #         rR = inputs['r'][afi] / inputs['r'][-1]  # non-dimensional blade radial station at cross section
                    #         Re_loc[afi, :, ind] = c * maxTS * rR / KinVisc
                    #         Ma_loc[afi, :, ind] = maxTS * rR / SpdSound
                    #         for j in range(self.n_Re):
                    #             cl_interp_flaps[afi, :, j, ind] = inputs['cl_interp'][afi, :, j, 0]
                    #             cd_interp_flaps[afi, :, j, ind] = inputs['cl_interp'][afi, :, j, 0]
                    #             cm_interp_flaps[afi, :, j, ind] = inputs['cl_interp'][afi, :, j, 0]

        else:
            for afi in range(self.n_span):
                # for j in range(n_Re):  # ToDo incorporade variable Re capability
                for ind in range(self.n_tab):  # fill all self.n_tab slots even though no flaps exist at current radial position
                    c = inputs['chord'][afi]  # blade chord length at cross section
                    rR = inputs['r'][afi] / inputs['r'][-1]  # non-dimensional blade radial station at cross section
                    Re_loc[afi, :, ind] = c * self.maxTS * rR / self.KinVisc
                    Ma_loc[afi, :, ind] = self.maxTS * rR / self.SpdSound
                    
        outputs['cl_interp_flaps']  = cl_interp_flaps
        outputs['cd_interp_flaps']  = cd_interp_flaps
        outputs['cm_interp_flaps']  = cm_interp_flaps
        outputs['flap_angles']      = fa_control # use vector of flap angle controls
        outputs['Re_loc'] = Re_loc
        outputs['Ma_loc'] = Ma_loc

def get_flap_polars(run_xfoil_params, afi):
    '''
    Sort of a wrapper script for runXfoil - makes parallelization possible

    Parameters:
    -----------
    run_xfoil_params: dict
        contains all necessary information to succesfully run xFoil
    afi: int
        airfoil section index

    Returns:
    --------
    cl_interp_flaps_af: 3D array
        lift coefficient tables
    cd_interp_flaps_af: 3D array
        drag coefficient  tables
    cm_interp_flaps_af: 3D array
        moment coefficient tables
    fa_control_af: 2D array
        flap angle tables
    Re_loc_af: 2D array
        Reynolds number table
    Ma_loc_af: 2D array
        Mach number table
    '''
    cl_interp_flaps_af = run_xfoil_params['cl_interp'][afi]
    cd_interp_flaps_af = run_xfoil_params['cd_interp'][afi]
    cm_interp_flaps_af = run_xfoil_params['cm_interp'][afi]
    fa_control_af = np.zeros((run_xfoil_params['n_Re'], run_xfoil_params['n_tab']))
    Re_loc_af = np.zeros((run_xfoil_params['n_Re'], run_xfoil_params['n_tab']))
    Ma_loc_af = np.zeros((run_xfoil_params['n_Re'], run_xfoil_params['n_tab']))

    if 'coords' in run_xfoil_params['flap_profiles'][afi]: # check if 'coords' is an element of 'run_xfoil_params['flap_profiles']', i.e. if we have various flap angles
        # for j in range(n_Re): # ToDo incorporade variable Re capability
        for ind in range(run_xfoil_params['n_tab']):
            #fa = run_xfoil_params['flap_profiles'][afi]['flap_angles'][ind] # value of respective flap angle
            fa_control_af[:,ind] = run_xfoil_params['flap_profiles'][afi]['flap_angles'][ind] # flap angle vector of distributed aerodynamics control
            # eta = (blade['pf']['r'][afi]/blade['pf']['r'][-1])
            # eta = blade['outer_shape_bem']['chord']['grid'][afi]
            c   = run_xfoil_params['chord'][afi]  # blade chord length at cross section
            s   = run_xfoil_params['s'][afi]
            rR  = run_xfoil_params['r'][afi] / run_xfoil_params['r'][-1]  # non-dimensional blade radial station at cross section in the rotor coordinate system
            Re_loc_af[:,ind] = c* run_xfoil_params['maxTS'] * rR / run_xfoil_params['KinVisc']
            Ma_loc_af[:,ind] = run_xfoil_params['maxTS'] * rR / run_xfoil_params['SpdSound']

            print('Run xfoil for nondimensional blade span section s = ' + str(s) + ' with ' + str(fa_control_af[0,ind]) + ' deg flap deflection angle; Re equal to ' + str(Re_loc_af[0,ind]) + '; Ma equal to ' + str(Ma_loc_af[0,ind]))
            # if  rR > 0.88:  # reduce AoAmin for (thinner) airfoil at the blade tip due to convergence reasons in XFoil
            #     data = run_xfoil_params['runXfoil'](run_xfoil_params['flap_profiles'][afi]['coords'][:, 0, ind],run_xfoil_params['flap_profiles'][afi]['coords'][:, 1, ind],Re_loc_af[afi, j, ind], -13.5, 25., 0.5, Ma_loc_af[afi, j, ind])
            # else:  # normal case

            xfoil_kw = {'AoA_min': -20,
                        'AoA_max': 25,
                        'AoA_inc': 0.5,
                        'Ma':  Ma_loc_af[0, ind],
                        }

            if MPI:
                xfoil_kw['MPI_run'] = True
            elif run_xfoil_params['cores'] > 1:
                xfoil_kw['multi_run'] = True

            data = runXfoil(run_xfoil_params['xfoil_path'], run_xfoil_params['flap_profiles'][afi]['coords'][:, 0, ind],run_xfoil_params['flap_profiles'][afi]['coords'][:, 1, ind],Re_loc_af[0, ind], **xfoil_kw)


            # data = run_xfoil_params['runXfoil'](run_xfoil_params['flap_profiles'][afi]['coords'][:,0,ind], run_xfoil_params['flap_profiles'][afi]['coords'][:,1,ind], Re[j])
            # data[data[:,0].argsort()] # To sort data by increasing aoa
            # Apply corrections to airfoil polars
            # oldpolar= Polar(Re[j], data[:,0],data[:,1],data[:,2],data[:,4]) # p[:,0] is alpha, p[:,1] is Cl, p[:,2] is Cd, p[:,4] is Cm
            oldpolar= Polar(Re_loc_af[0,ind], data[:,0],data[:,1],data[:,2],data[:,4]) # p[:,0] is alpha, p[:,1] is Cl, p[:,2] is Cd, p[:,4] is Cm

            polar3d = oldpolar.correction3D(rR,c/run_xfoil_params['R'],run_xfoil_params['tsr']) # Apply 3D corrections (made sure to change the r/R, c/R, and tsr values appropriately when calling AFcorrections())
            cdmax   = 1.5
            polar   = polar3d.extrapolate(cdmax) # Extrapolate polars for alpha between -180 deg and 180 deg

            for j in range(run_xfoil_params['n_Re']):
                cl_interp_flaps_af[:,j,ind] = np.interp(np.degrees(run_xfoil_params['aoa']), polar.alpha, polar.cl)
                cd_interp_flaps_af[:,j,ind] = np.interp(np.degrees(run_xfoil_params['aoa']), polar.alpha, polar.cd)
                cm_interp_flaps_af[:,j,ind] = np.interp(np.degrees(run_xfoil_params['aoa']), polar.alpha, polar.cm)

        # # ** The code below will plot the three cl polars
        # import matplotlib.pyplot as plt
        # font = {'family': 'Times New Roman',
        #         'weight': 'normal',
        #         'size': 18}
        # plt.rc('font', **font)
        # plt.figure
        # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        # plt.plot(np.degrees(run_xfoil_params['aoa']), cl_interp_flaps_af[afi,:,0,0],'r', label='$\\delta_{flap}$ = -10 deg')  # -10
        # plt.plot(np.degrees(run_xfoil_params['aoa']), cl_interp_flaps_af[afi,:,0,1],'k', label='$\\delta_{flap}$ = 0 deg')  # 0
        # plt.plot(np.degrees(run_xfoil_params['aoa']), cl_interp_flaps_af[afi,:,0,2],'b', label='$\\delta_{flap}$ = +10 deg')  # +10
        # # plt.plot(np.degrees(run_xfoil_params['aoa']), cl_interp_flaps_af[afi,:,0,0],'r')  # -10
        # # plt.plot(np.degrees(run_xfoil_params['aoa']), cl_interp_flaps_af[afi,:,0,1],'k')  # 0
        # # plt.plot(np.degrees(run_xfoil_params['aoa']), cl_interp_flaps_af[afi,:,0,2],'b')  # +10
        # plt.xlim(xmin=-15, xmax=15)
        # plt.ylim(ymin=-1.7, ymax=2.2)
        # plt.grid(True)
        # # autoscale_y(ax)
        # plt.xlabel('Angles of attack, deg')
        # plt.ylabel('Lift coefficient')
        # plt.legend(loc='lower right')
        # plt.tight_layout()
        # plt.show()
        # # # # plt.savefig('airfoil_polars_check/r_R_1_0_cl_flaps.png', dpi=300)
        # # # # plt.savefig('airfoil_polars_check/NACA63-618_cl_flaps.png', dpi=300)
        # # # # plt.savefig('airfoil_polars_check/FFA-W3-211_cl_flaps.png', dpi=300)
        # # # # plt.savefig('airfoil_polars_check/FFA-W3-241_cl_flaps.png', dpi=300)
        # # # # plt.savefig('airfoil_polars_check/FFA-W3-301_cl_flaps.png', dpi=300)



    else:  # no flap at specific radial location (but in general 'aerodynamic_control' is defined in blade from yaml)
        for ind in range(run_xfoil_params['n_tab']):  # fill all run_xfoil_params['n_tab'] slots even though no flaps exist at current radial position
            c = run_xfoil_params['chord'][afi]  # blade chord length at cross section
            rR = run_xfoil_params['r'][afi] / run_xfoil_params['r'][-1]  # non-dimensional blade radial station at cross section
            Re_loc_af[:, ind] = c * run_xfoil_params['maxTS'] * rR / run_xfoil_params['KinVisc']
            Ma_loc_af[:, ind] = run_xfoil_params['maxTS'] * rR / run_xfoil_params['SpdSound']
            for j in range(run_xfoil_params['n_Re']):
                cl_interp_flaps_af[:,j,ind] = run_xfoil_params['cl_interp'][afi,:,j,0]
                cd_interp_flaps_af[:,j,ind] = run_xfoil_params['cd_interp'][afi,:,j,0]
                cm_interp_flaps_af[:,j,ind] = run_xfoil_params['cm_interp'][afi,:,j,0]

            for j in range(run_xfoil_params['n_Re']):
                cl_interp_flaps_af[:, j, ind] = cl_interp_flaps_af[:, j, 0]
                cd_interp_flaps_af[:, j, ind] = cd_interp_flaps_af[:, j, 0]
                cm_interp_flaps_af[:, j, ind] = cm_interp_flaps_af[:, j, 0]
    
    return cl_interp_flaps_af, cd_interp_flaps_af, cm_interp_flaps_af, fa_control_af, Re_loc_af, Ma_loc_af

