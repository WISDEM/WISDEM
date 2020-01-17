import numpy as np
from openmdao.api import ExplicitComponent
from scipy.ndimage import gaussian_filter
from wisdem.ccblade import CCAirfoil

def runXfoil(self, x, y, Re, AoA_min=-9, AoA_max=25, AoA_inc=0.5, Ma = 0.0):
    #This function is used to create and run xfoil simulations for a given set of airfoil coordinates

    # Set initial parameters needed in xfoil
    LoadFlnmAF = "airfoil.txt" # This is a temporary file that will be deleted after it is no longer needed
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
    saveFlnmPolar = "Polar.txt" # file name of outpur xfoil polar (can be useful to look at during debugging...can also delete at end if you don't want it stored)
    xfoilFlnm  = 'xfoil_input.txt' # Xfoil run script that will be deleted after it is no longer needed
    runFlag = 1 # Flag used in error handling
    dfdn = -0.5 # Change in angle of attack during initialization runs down to AoA_min
    runNum = 0 # Initialized run number
    dfnFlag = -10 # This flag is used to determine if xfoil needs to be re-run if the simulation fails due to convergence issues at low angles of attack

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
        os.system(self.xfoil_path + " < xfoil_input.txt  > NUL") # <<< runs XFoil !
        flap_polar = np.loadtxt(saveFlnmPolar,skiprows=12)


        # Error handling (re-run simulations with more panels if there is not enough data in polars)
        if flap_polar.size < 3: # This case is if there are convergence issues at the lowest angles of attack
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
        self.options.declare('blade_init_options')
        
    def setup(self):
        
        blade_init_options = self.options['blade_init_options']
        self.n_span        = n_span     = blade_init_options['n_span']
        self.n_te_flaps    = n_te_flaps = blade_init_options['n_te_flaps']

        self.add_input('s',          val=np.zeros(n_span),                      desc='1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)')

        self.add_input('span_start', val=np.zeros(n_te_flaps),                  desc='1D array of the positions along blade span where the trailing edge flap(s) start. Only values between 0 and 1 are meaningful.')
        self.add_input('span_end',   val=np.zeros(n_te_flaps),                  desc='1D array of the positions along blade span where the trailing edge flap(s) end. Only values between 0 and 1 are meaningful.')
        self.add_input('chord_start',val=np.zeros(n_te_flaps),                  desc='1D array of the positions along chord where the trailing edge flap(s) start. Only values between 0 and 1 are meaningful.')
        self.add_input('delta_max_pos', val=np.zeros(n_te_flaps), units='rad',  desc='1D array of the max angle of the trailing edge flaps.')
        self.add_input('delta_max_neg', val=np.zeros(n_te_flaps), units='rad',  desc='1D array of the min angle of the trailing edge flaps.')
        self.add_discrete_input('num_delta',  val=np.zeros(n_te_flaps),         desc='1D array of the number of points to discretize the polars between delta_max_neg and delta_max_pos.')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        flap_profiles = self.n_span * []
        # Only if trailing edge flaps are present
        if n_te_flaps > 0: 
                for i in range(self.n_span):
                    flap_profiles[i] = {}
                    # Loop through the flaps specified in yaml file
                    for k in range(n_te_flaps):
                        # Only create flap geometries where the yaml file specifies there is a flap (Currently going to nearest blade station location)
                        if inputs['s'][i] >= inputs['span_start'][k] and inputs['s'][i] <= inputs['span_end'][k]: 
                            flap_profiles[i]['flap_angles']= []
                            # Initialize the profile coordinates to zeros
                            flap_profiles[i]['coords']     = np.zeros((len(blade['profile'][:,0,0]),len(blade['profile'][0,:,0]),discrete_inputs['num_delta'][k])) 
                             # Ben:I am not going to force it to include delta=0.  If this is needed, a more complicated way of getting flap deflections to calculate is needed.
                            flap_angles = np.linspace(discrete_inputs['delta_max_neg'][k],discrete_inputs['delta_max_pos'][k],discrete_inputs['num_delta'][k])
                            # Loop through the flap angles
                            for ind, fa in enumerate(flap_angles):
                                # NOTE: negative flap angles are deflected to the suction side, i.e. positively along the positive z- (radial) axis
                                af_flap = CCAirfoil(np.array([1,2,3]), np.array([100]), np.zeros(3), np.zeros(3), np.zeros(3), blade['profile'][:,0,i],blade['profile'][:,1,i], "Profile"+str(i)) # bem:I am creating an airfoil name based on index...this structure/naming convention is being assumed in CCAirfoil.runXfoil() via the naming convention used in CCAirfoil.af_flap_coords(). Note that all of the inputs besides profile coordinates and name are just dummy varaiables at this point.
                                af_flap.af_flap_coords(xfoil_path, fa,  blade['aerodynamic_control']['te_flaps'][k]['chord_start'],0.5,200) #bem: the last number is the number of points in the profile.  It is currently being hard coded at 200 but should be changed to make sure it is the same number of points as the other profiles
                                # blade['flap_profiles'][i]['coords'][:,0,ind] = af_flap.af_flap_xcoords # x-coords from xfoil file with flaps
                                # blade['flap_profiles'][i]['coords'][:,1,ind] = af_flap.af_flap_ycoords # y-coords from xfoil file with flaps
                                # blade['flap_profiles'][i]['coords'][:,0,ind] = af_flap.af_flap_xcoords  # x-coords from xfoil file with flaps and NO gaussian filter for smoothing
                                # blade['flap_profiles'][i]['coords'][:,1,ind] = af_flap.af_flap_ycoords  # y-coords from xfoil file with flaps and NO gaussian filter for smoothing
                                blade['flap_profiles'][i]['coords'][:,0,ind] = gaussian_filter(af_flap.af_flap_xcoords, sigma=1) # x-coords from xfoil file with flaps and gaussian filter for smoothing
                                blade['flap_profiles'][i]['coords'][:,1,ind] = gaussian_filter(af_flap.af_flap_ycoords, sigma=1) # y-coords from xfoil file with flaps and gaussian filter for smoothing

                                blade['flap_profiles'][i]['flap_angles'].append([])
                                blade['flap_profiles'][i]['flap_angles'][ind] = fa # Putting in flap angles to blade for each profile (can be used for debugging later)

                            # ** The code below will plot the first three flap deflection profiles (in the case where there are only 3 this will correspond to max negative, zero, and max positive deflection cases)
                            # import matplotlib.pyplot as plt
                            # font = {'family': 'Times New Roman',
                            #         'weight': 'normal',
                            #         'size': 18}
                            # plt.rc('font', **font)
                            # plt.figure
                            # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                            # # plt.plot(blade['flap_profiles'][i]['coords'][:,0,0], blade['flap_profiles'][i]['coords'][:,1,0], 'r',blade['flap_profiles'][i]['coords'][:,0,1], blade['flap_profiles'][i]['coords'][:,1,1], 'k',blade['flap_profiles'][i]['coords'][:,0,2], blade['flap_profiles'][i]['coords'][:,1,2], 'b')
                            # plt.plot(blade['flap_profiles'][i]['coords'][:, 0, 0],
                            #          blade['flap_profiles'][i]['coords'][:, 1, 0], 'r',
                            #          blade['flap_profiles'][i]['coords'][:, 0, 2],
                            #          blade['flap_profiles'][i]['coords'][:, 1, 2], 'b',
                            #          blade['flap_profiles'][i]['coords'][:, 0, 1],
                            #          blade['flap_profiles'][i]['coords'][:, 1, 1], 'k')
                            #
                            # # plt.xlabel('x')
                            # # plt.ylabel('y')
                            # plt.axis('equal')
                            # plt.axis('off')
                            # plt.tight_layout()
                            # plt.show()
                            # # plt.savefig('temp/airfoil_polars/NACA63-618_flap_profiles.png', dpi=300)
                            # # plt.savefig('temp/airfoil_polars/FFA-W3-211_flap_profiles.png', dpi=300)
                            # # plt.savefig('temp/airfoil_polars/FFA-W3-241_flap_profiles.png', dpi=300)
                            # # plt.savefig('temp/airfoil_polars/FFA-W3-301_flap_profiles.png', dpi=300)




        pass
