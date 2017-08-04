
import numpy as np
import os

from rotorse.precomp import Profile, Orthotropic2DMaterial, CompositeSection  # TODO: can just pass file names and do this initialization inside of rotor
#from towerse.tower import TowerWithpBEAM
from commonse.environment import PowerWind, TowerSoil, LinearWaves
from jacketse.jacket import JcktGeoInputs,SoilGeoInputs,WaterInputs,WindInputs,RNAprops,TPlumpMass,Frame3DDaux,\
                    MatInputs,LegGeoInputs,XBrcGeoInputs,MudBrcGeoInputs,HBrcGeoInputs,TPGeoInputs,PileGeoInputs,\
                    TwrGeoInputs
from commonse.utilities import cosd, sind
from rotorse.rotoraero import RS2RPM


def configure_nrel5mw_turbine_with_jacket(turbine,wind_class='I',sea_depth = 0.0):
    """
    Inputs:
        rotor = RotorSE()
        nacelle = DriveSE()
        jacket = JacketSE()
        wind_class : str ('I', 'III', 'Offshore' - selected wind class for project)
        sea_depth : float (sea depth if an offshore wind plant)
    """

    # =================


    # === Turbine ===
    turbine.rho = 1.225  # (Float, kg/m**3): density of air
    turbine.mu = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    turbine.shear_exponent = 0.2  # (Float): shear exponent
    turbine.hub_height = 90.0  # (Float, m): hub height
    turbine.turbine_class = 'I'  # (Enum): IEC turbine class
    turbine.turbulence_class = 'B'  # (Enum): IEC turbulence class class
    turbine.cdf_reference_height_wind_speed = 90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
    turbine.g = 9.81  # (Float, m/s**2): acceleration of gravity
    # ======================

    # === rotor ===
    # --- blade grid ---
    turbine.rotor.initial_aero_grid = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667,
        0.43333333, 0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943, 0.93333333,
        0.97777724])  # (Array): initial aerodynamic grid on unit radius
    turbine.rotor.initial_str_grid = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154,
        0.0114340970275, 0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
        0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 0.276686558545, 0.3, 0.333640766319,
        0.36666667, 0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333,
        0.667358391486, 0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724,
        1.0])  # (Array): initial structural grid on unit radius
    turbine.rotor.idx_cylinder_aero = 3  # (Int): first idx in r_aero_unit of non-cylindrical section, constant twist inboard of here
    turbine.rotor.idx_cylinder_str = 14  # (Int): first idx in r_str_unit of non-cylindrical section
    turbine.rotor.hubFraction = 0.025  # (Float): hub location as fraction of radius
    # ------------------

    # --- blade geometry ---
    turbine.rotor.r_aero = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333,
        0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333,
        0.97777724])  # (Array): new aerodynamic grid on unit radius
    turbine.rotor.r_max_chord = 0.23577  # (Float): location of max chord on unit radius
    turbine.rotor.chord_sub = [3.2612, 4.5709, 3.3178, 1.4621]  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    turbine.rotor.theta_sub = [13.2783, 7.46036, 2.89317, -0.0878099]  # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    turbine.rotor.precurve_sub = [0.0, 0.0, 0.0]  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    turbine.rotor.delta_precurve_sub = [0.0, 0.0, 0.0]  # (Array, m): adjustment to precurve to account for curvature from loading
    turbine.rotor.sparT = [0.05, 0.047754, 0.045376, 0.031085, 0.0061398]  # (Array, m): spar cap thickness parameters
    turbine.rotor.teT = [0.1, 0.09569, 0.06569, 0.02569, 0.00569]  # (Array, m): trailing-edge thickness parameters
    turbine.rotor.bladeLength = 61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    turbine.rotor.delta_bladeLength = 0.0  # (Float, m): adjustment to blade length to account for curvature from loading
    turbine.rotor.precone = 2.5  # (Float, deg): precone angle
    turbine.rotor.tilt = 5.0  # (Float, deg): shaft tilt
    turbine.rotor.yaw = 0.0  # (Float, deg): yaw error
    turbine.rotor.nBlades = 3  # (Int): number of blades
    # ------------------

    # --- airfoil files ---
    import rotorse
    #basepath = os.path.join('5MW_files', '5MW_AFFiles')
    basepath = os.path.join('..','reference_turbines','nrel5mw','airfoils')

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = os.path.join(basepath, 'Cylinder1.dat')
    airfoil_types[1] = os.path.join(basepath, 'Cylinder2.dat')
    airfoil_types[2] = os.path.join(basepath, 'DU40_A17.dat')
    airfoil_types[3] = os.path.join(basepath, 'DU35_A17.dat')
    airfoil_types[4] = os.path.join(basepath, 'DU30_A17.dat')
    airfoil_types[5] = os.path.join(basepath, 'DU25_A17.dat')
    airfoil_types[6] = os.path.join(basepath, 'DU21_A17.dat')
    airfoil_types[7] = os.path.join(basepath, 'NACA64_A17.dat')

    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    n = len(af_idx)
    af = [0]*n
    for i in range(n):
        af[i] = airfoil_types[af_idx[i]]
    turbine.rotor.airfoil_files = af  # (List): names of airfoil file
    # ----------------------

    # --- control ---
    turbine.rotor.control.Vin = 3.0  # (Float, m/s): cut-in wind speed
    turbine.rotor.control.Vout = 25.0  # (Float, m/s): cut-out wind speed
    turbine.rotor.control.ratedPower = 5e6  # (Float, W): rated power
    turbine.rotor.control.minOmega = 0.0  # (Float, rpm): minimum allowed rotor rotation speed
    turbine.rotor.control.maxOmega = 12.0  # (Float, rpm): maximum allowed rotor rotation speed
    turbine.rotor.control.tsr = 7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
    turbine.rotor.control.pitch = 0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    turbine.rotor.pitch_extreme = 0.0  # (Float, deg): worst-case pitch at survival wind condition
    turbine.rotor.azimuth_extreme = 0.0  # (Float, deg): worst-case azimuth at survival wind condition
    turbine.rotor.VfactorPC = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation
    # ----------------------

    # --- aero and structural analysis options ---
    turbine.rotor.nSector = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
    turbine.rotor.npts_coarse_power_curve = 20  # (Int): number of points to evaluate aero analysis at
    turbine.rotor.npts_spline_power_curve = 200  # (Int): number of points to use in fitting spline to power curve
    turbine.rotor.AEP_loss_factor = 1.0  # (Float): availability and other losses (soiling, array, etc.)
    turbine.rotor.drivetrainType = 'geared'  # (Enum)
    turbine.rotor.nF = 5  # (Int): number of natural frequencies to compute
    turbine.rotor.dynamic_amplication_tip_deflection = 1.35  # (Float): a dynamic amplification factor to adjust the static deflection calculation
    # ----------------------


    # --- materials and composite layup  ---
    #basepath = os.path.join('5MW_files', '5MW_PrecompFiles')
    basepath = os.path.join('..', 'reference_turbines','nrel5mw','blade')

    materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, 'materials.inp'))

    ncomp = len(turbine.rotor.initial_str_grid)
    upper = [0]*ncomp
    lower = [0]*ncomp
    webs = [0]*ncomp
    profile = [0]*ncomp

    turbine.rotor.leLoc = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411,
        0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
        0.4, 0.4, 0.4, 0.4])    # (Array): array of leading-edge positions from a reference blade axis (usually blade pitch axis). locations are normalized by the local chord length. e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.  positive in -x direction for airfoil-aligned coordinate system
    turbine.rotor.sector_idx_strain_spar = [2]*ncomp  # (Array): index of sector for spar (PreComp definition of sector)
    turbine.rotor.sector_idx_strain_te = [3]*ncomp  # (Array): index of sector for trailing-edge (PreComp definition of sector)
    web1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.4114, 0.4102, 0.4094, 0.3876, 0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104, 0.307, 0.3003, 0.2982, 0.2935, 0.2899, 0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731, 0.2664, 0.2607, 0.2562, 0.1886, -1.0])
    web2 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5886, 0.5868, 0.5854, 0.5508, 0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896, 0.493, 0.4997, 0.5018, 0.5065, 0.5101, 0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269, 0.5336, 0.5393, 0.5438, 0.6114, -1.0])
    web3 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    turbine.rotor.chord_str_ref = np.array([3.2612, 3.3100915356, 3.32587052924, 3.34159388653, 3.35823798667, 3.37384375335,
        3.38939112914, 3.4774055542, 3.49839685, 3.51343645709, 3.87017220335, 4.04645623801, 4.19408216643,
         4.47641008477, 4.55844487985, 4.57383098262, 4.57285771934, 4.51914315648, 4.47677655262, 4.40075650022,
         4.31069949379, 4.20483735936, 4.08985563932, 3.82931757126, 3.74220276467, 3.54415796922, 3.38732428502,
         3.24931446473, 3.23421422609, 3.22701537997, 3.21972125648, 3.08979310611, 2.95152261813, 2.330753331,
         2.05553464181, 1.82577817774, 1.5860853279, 1.4621])  # (Array, m): chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c for this case)

    for i in range(ncomp):

        webLoc = []
        if web1[i] != -1:
            webLoc.append(web1[i])
        if web2[i] != -1:
            webLoc.append(web2[i])
        if web3[i] != -1:
            webLoc.append(web3[i])

        upper[i], lower[i], webs[i] = CompositeSection.initFromPreCompLayupFile(os.path.join(basepath, 'layup_' + str(i+1) + '.inp'), webLoc, materials)
        profile[i] = Profile.initFromPreCompFile(os.path.join(basepath, 'shape_' + str(i+1) + '.inp'))

    turbine.rotor.materials = materials  # (List): list of all Orthotropic2DMaterial objects used in defining the geometry
    turbine.rotor.upperCS = upper  # (List): list of CompositeSection objections defining the properties for upper surface
    turbine.rotor.lowerCS = lower  # (List): list of CompositeSection objections defining the properties for lower surface
    turbine.rotor.websCS = webs  # (List): list of CompositeSection objections defining the properties for shear webs
    turbine.rotor.profile = profile  # (List): airfoil shape at each radial position
    # --------------------------------------

    strain_ult_spar = 1.0e-2
    strain_ult_te = 2500*1e-6

    # --- fatigue ---
    turbine.rotor.rstar_damage = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500,
        0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])  # (Array): nondimensional radial locations of damage equivalent moments
    turbine.rotor.Mxb_damage = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003,
        1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002,
        1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])  # (Array, N*m): damage equivalent moments about blade c.s. x-direction
    turbine.rotor.Myb_damage = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003,
        1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002,
        3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])  # (Array, N*m): damage equivalent moments about blade c.s. y-direction
    turbine.rotor.strain_ult_spar = 1.0e-2  # (Float): ultimate strain in spar cap
    turbine.rotor.strain_ult_te = 2500*1e-6 * 2   # (Float): uptimate strain in trailing-edge panels, note that I am putting a factor of two for the damage part only.
    turbine.rotor.eta_damage = 1.35*1.3*1.0  # (Float): safety factor for fatigue
    turbine.rotor.m_damage = 10.0  # (Float): slope of S-N curve for fatigue analysis
    turbine.rotor.N_damage = 365*24*3600*20.0  # (Float): number of cycles used in fatigue analysis  TODO: make function of rotation speed
    # ----------------
    # =================


    # === nacelle ======
    turbine.nacelle.L_ms = 1.0  # (Float, m): main shaft length downwind of main bearing in low-speed shaft
    turbine.nacelle.L_mb = 2.5  # (Float, m): main shaft length in low-speed shaft

    turbine.nacelle.h0_front = 1.7  # (Float, m): height of Ibeam in bedplate front
    turbine.nacelle.h0_rear = 1.35  # (Float, m): height of Ibeam in bedplate rear

    turbine.nacelle.drivetrain_design = 'geared'
    turbine.nacelle.crane = True  # (Bool): flag for presence of crane
    turbine.nacelle.bevel = 0  # (Int): Flag for the presence of a bevel stage - 1 if present, 0 if not
    turbine.nacelle.gear_configuration = 'eep'  # (Str): tring that represents the configuration of the gearbox (stage number and types)

    turbine.nacelle.Np = [3, 3, 1]  # (Array): number of planets in each stage
    turbine.nacelle.ratio_type = 'optimal'  # (Str): optimal or empirical stage ratios
    turbine.nacelle.shaft_type = 'normal'  # (Str): normal or short shaft length
    #turbine.nacelle.shaft_angle = 5.0  # (Float, deg): Angle of the LSS inclindation with respect to the horizontal
    turbine.nacelle.shaft_ratio = 0.10  # (Float): Ratio of inner diameter to outer diameter.  Leave zero for solid LSS
    turbine.nacelle.carrier_mass = 8000.0 # estimated for 5 MW
    turbine.nacelle.mb1Type = 'CARB'  # (Str): Main bearing type: CARB, TRB or SRB
    turbine.nacelle.mb2Type = 'SRB'  # (Str): Second bearing type: CARB, TRB or SRB
    turbine.nacelle.yaw_motors_number = 8.0  # (Float): number of yaw motors
    turbine.nacelle.uptower_transformer = True
    turbine.nacelle.flange_length = 0.5 #m
    turbine.nacelle.gearbox_cm = 0.1
    turbine.nacelle.hss_length = 1.5
    turbine.nacelle.overhang = 5.0 #TODO - should come from turbine configuration level

    turbine.nacelle.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs

    # TODO: should come from rotor (these are FAST outputs)
    turbine.nacelle.DrivetrainEfficiency = 0.95
    turbine.nacelle.rotor_bending_moment_x = 330770.0# Nm
    turbine.nacelle.rotor_bending_moment_y = -16665000.0 # Nm
    turbine.nacelle.rotor_bending_moment_z = 2896300.0 # Nm
    turbine.nacelle.rotor_force_x = 599610.0 # N
    turbine.nacelle.rotor_force_y = 186780.0 # N
    turbine.nacelle.rotor_force_z = -842710.0 # N

    #turbine.nacelle.h0_rear = 1.35 # only used in drive smooth
    #turbine.nacelle.h0_front = 1.7

    # =================

    if wind_class == 'I':
        turbine.rotor.turbine_class = 'I'

    elif wind_class == 'III':
        turbine.rotor.turbine_class = 'III'

        # for fatigue based analysis of class III wind turbine
        turbine.tower.M_DEL = 1.028713178 * 1e3*np.array([7.8792E+003, 7.7507E+003, 7.4918E+003, 7.2389E+003, 6.9815E+003, 6.7262E+003, 6.4730E+003, 6.2174E+003, 5.9615E+003, 5.7073E+003, 5.4591E+003, 5.2141E+003, 4.9741E+003, 4.7399E+003, 4.5117E+003, 4.2840E+003, 4.0606E+003, 3.8360E+003, 3.6118E+003, 3.3911E+003, 3.1723E+003, 2.9568E+003, 2.7391E+003, 2.5294E+003, 2.3229E+003, 2.1246E+003, 1.9321E+003, 1.7475E+003, 1.5790E+003, 1.4286E+003, 1.3101E+003, 1.2257E+003, 1.1787E+003, 1.1727E+003, 1.1821E+003])

        turbine.rotor.Mxb_damage = 1e3*np.array([2.3617E+003, 2.0751E+003, 1.8051E+003, 1.5631E+003, 1.2994E+003, 1.0388E+003, 8.1384E+002, 6.2492E+002, 4.6916E+002, 3.4078E+002, 2.3916E+002, 1.5916E+002, 9.9752E+001, 5.6139E+001, 2.6492E+001, 1.0886E+001, 3.7210E+000, 4.3206E-001])
        turbine.rotor.Myb_damage = 1e3*np.array([2.5492E+003, 2.6261E+003, 2.4265E+003, 2.2308E+003, 1.9882E+003, 1.7184E+003, 1.4438E+003, 1.1925E+003, 9.6251E+002, 7.5564E+002, 5.7332E+002, 4.1435E+002, 2.8036E+002, 1.7106E+002, 8.7732E+001, 3.8678E+001, 1.3942E+001, 1.6600E+000])

    elif wind_class == 'Offshore':
        turbine.rotor.turbine_class = 'I'

    # =================

    # === jacket ===

    #--- Set Jacket Input Parameters ---#
    Jcktins=JcktGeoInputs()
    Jcktins.nlegs =4
    Jcktins.nbays =5
    Jcktins.batter=12.
    Jcktins.dck_botz =16.
    Jcktins.weld2D   =0.5
    Jcktins.VPFlag = True    #vertical pile T/F;  to enable piles in frame3DD set pileinputs.ndiv>0
    Jcktins.clamped= False    #whether or not the bottom of the structure is rigidly connected. Use False when equivalent spring constants are being used.
    Jcktins.AFflag = False  #whether or not to use apparent fixity piles
    Jcktins.PreBuildTPLvl = 2  #if >0, the TP is prebuilt according to rules per PreBuildTP

    #Soil inputs
    Soilinputs=SoilGeoInputs()
    Soilinputs.zbots   =-np.array([3.,5.,7.,15.,30.,50.])
    Soilinputs.gammas  =np.array([10000.,10000.,10000.,10000.,10000.,10000.])
    Soilinputs.cus     =np.array([60000.,60000.,60000.,60000.,60000.,60000.])
    Soilinputs.phis    =np.array([26.,26.,26.,26.,26.,26])#np.array([36.,33.,26.,37.,35.,37.5])#np.array([36.,33.,26.,37.,35.,37.5])
    Soilinputs.delta   =25.
    Soilinputs.sndflg   =True
    Soilinputs.PenderSwtch   =False #True
    Soilinputs.SoilSF   =1.

    #Water and wind inputs
    Waterinputs=WaterInputs()
    Waterinputs.wdepth   =30.
    Waterinputs.wlevel   =30. #Distance from bottom of structure to surface  THIS, I believe is no longer needed as piles may be negative in z, to check and remove in case
    Waterinputs.T=12.  #Wave Period
    Waterinputs.HW=10. #Wave Height
    '''Windinputs=WindInputs()
    Windinputs.HH=100. #CHECK HOW THIS COMPLIES....
    Windinputs.U50HH=30. #assumed gust speed'''

    #RNA loads              Fx-z,         Mxx-zz
    #RNA_F=np.array([1000.e3,0.,0.,0.,0.,0.])

    #Pile data
    Pilematin=MatInputs()
    Pilematin.matname=np.array(['steel'])
    Pilematin.E=np.array([ 25.e9])
    Dpile=2.5#0.75 # 2.0
    tpile=0.01
    Lp=20. #45

    Pileinputs=PileGeoInputs()
    Pileinputs.Pilematins=Pilematin
    Pileinputs.ndiv=0 #3
    Pileinputs.Dpile=Dpile
    Pileinputs.tpile=tpile
    Pileinputs.Lp=Lp #[m] Embedment length

    #Legs data
    legmatin=MatInputs()
    legmatin.matname=(['steel','steel','steel','steel'])
    legmatin.E=np.array([2.0e11])
    Dleg=np.array([2.0,1.8,1.8,1.8,1.8,1.8])
    tleg=1.55*np.array([0.0254]).repeat(Dleg.size)
    leginputs=LegGeoInputs()
    leginputs.legZbot   = 1.0
    leginputs.ndiv=1
    leginputs.legmatins=legmatin
    leginputs.Dleg=Dleg
    leginputs.tleg=tleg

    legbot_stmphin =1.5  #Distance from bottom of leg to second joint along z; must be>0

    #Xbrc data
    Xbrcmatin=MatInputs()
    Xbrcmatin.matname=np.array(['steel']).repeat(Jcktins.nbays)
    Xbrcmatin.E=np.array([ 2.2e11, 2.0e11,2.0e11,2.0e11,2.0e11])
    Dbrc=np.array([1.,1.,1.0,1.0,1.0])
    tbrc=np.array([1.,1.,1.0,1.0,1.0])*0.0254

    Xbrcinputs=XBrcGeoInputs()
    Xbrcinputs.Dbrc=Dbrc
    Xbrcinputs.tbrc=tbrc
    Xbrcinputs.ndiv=2#2
    Xbrcinputs.Xbrcmatins=Xbrcmatin
    Xbrcinputs.precalc=True   #This can be set to true if we want Xbraces to be precalculated in D and t, in which case the above set Dbrc and tbrc would be overwritten

    #Mbrc data
    Mbrcmatin=MatInputs()
    Mbrcmatin.matname=np.array(['steel'])
    Mbrcmatin.E=np.array([ 2.5e11])
    Dbrc_mud=1.5

    Mbrcinputs=MudBrcGeoInputs()
    Mbrcinputs.Dbrc_mud=Dbrc_mud
    Mbrcinputs.ndiv=2
    Mbrcinputs.Mbrcmatins=Mbrcmatin
    Mbrcinputs.precalc=True   #This can be set to true if we want Mudbrace to be precalculated in D and t, in which case the above set Dbrc_mud and tbrc_mud would be overwritten
    #Hbrc data
    Hbrcmatin=MatInputs()
    Hbrcmatin.matname=np.array(['steel'])
    Hbrcmatin.E=np.array([ 2.5e11])
    Dbrc_hbrc=1.1

    Hbrcinputs=HBrcGeoInputs()
    Hbrcinputs.Dbrch=Dbrc_hbrc
    Hbrcinputs.ndiv=0#2
    Hbrcinputs.Hbrcmatins=Hbrcmatin
    Hbrcinputs.precalc=True   #This can be set to true if we want Hbrace to be set=Xbrace top D and t, in which case the above set Dbrch and tbrch would be overwritten

    #TP data
    TPlumpinputs=TPlumpMass()
    TPlumpinputs.mass=300.e3 #[kg]

    TPstmpsmatin=MatInputs()
    TPbrcmatin=MatInputs()
    TPstemmatin=MatInputs()
    TPbrcmatin.matname=np.array(['steel'])
    TPbrcmatin.E=np.array([ 2.5e11])
    TPstemmatin.matname=np.array(['steel']).repeat(2)
    TPstemmatin.E=np.array([ 2.1e11]).repeat(2)

    TPinputs=TPGeoInputs()
    TPinputs.TPbrcmatins=TPbrcmatin
    TPinputs.TPstemmatins=TPstemmatin
    TPinputs.TPstmpmatins=TPstmpsmatin
    TPinputs.Dstrut=1.6
    TPinputs.Dgir=Dbrc_hbrc
    TPinputs.Dbrc=1.1
    TPinputs.hstump=0.0#1.0
    TPinputs.stumpndiv=1#2
    TPinputs.brcndiv=1#2
    TPinputs.girndiv=1#2
    TPinputs.strutndiv=1#2
    TPinputs.stemndiv=1#2
    TPinputs.nstems=3
    TPinputs.Dstem=np.array([6.]).repeat(TPinputs.nstems)
    TPinputs.tstem=np.array([0.1,0.11,0.11])
    TPinputs.hstem=np.array([4.,3.,1.])

    #Tower data
    Twrmatin=MatInputs()
    Twrmatin.matname=np.array(['steel'])
    Twrmatin.E=np.array([ 2.77e11])
    Db=5.6
    tb=0.05
    Dt=Db*0.55

    '''Twrinputs=TwrGeoInputs()
    Twrinputs.Twrmatins=Twrmatin
    #Twrinputs.Htwr=70.  #Trumped by HH
    Twrinputs.Htwr2frac=0.2   #fraction of tower height with constant x-section
    Twrinputs.ndiv=np.array([6,6])  #ndiv for uniform and tapered section
    Twrinputs.Db=Db
    Twrinputs.DTRb=Db/tb
    #Twrinputs.Dt=Dt'''

    TwrRigidTop=True #False       #False=Account for RNA via math rather than a physical rigidmember

    #RNA data
    '''RNAins=RNAprops()
    RNAins.mass=3*350.e3
    RNAins.I[0]=86.579E+6
    RNAins.I[1]=53.530E+6
    RNAins.I[2]=58.112E+6
    RNAins.CMoff[2]=2.34
    RNAins.yawangle=45.  #angle with respect to global X, CCW looking from above, wind from left
    RNAins.rna_weightM=True'''

    #Frame3DD parameters
    FrameAuxIns=Frame3DDaux()
    FrameAuxIns.sh_fg=1               #shear flag-->Timoshenko
    FrameAuxIns.deltaz=5.
    FrameAuxIns.geo_fg=0
    FrameAuxIns.nModes = 6             # number of desired dynamic modes of vibration
    FrameAuxIns.Mmethod = 1            # 1: subspace Jacobi     2: Stodola
    FrameAuxIns.lump = 0               # 0: consistent mass ... 1: lumped mass matrix
    FrameAuxIns.tol = 1e-9             # mode shape tolerance
    FrameAuxIns.shift = 0.0            # shift value ... for unrestrained structures
    FrameAuxIns.gvector=np.array([0.,0.,-9.8065])    #GRAVITY

    #Pass all inputs to jacket assembly
    turbine.jacket.JcktGeoIn=Jcktins
    turbine.jacket.Soilinputs=Soilinputs
    turbine.jacket.Waterinputs=Waterinputs
    #turbine.jacket.Windinputs=Windinputs
    #turbine.jacket.RNA_F=RNA_F
    turbine.jacket.Pileinputs=Pileinputs
    turbine.jacket.leginputs=leginputs
    turbine.jacket.legbot_stmphin =legbot_stmphin
    turbine.jacket.Xbrcinputs=Xbrcinputs
    turbine.jacket.Mbrcinputs=Mbrcinputs
    turbine.jacket.Hbrcinputs=Hbrcinputs
    turbine.jacket.TPlumpinputs=TPlumpinputs
    turbine.jacket.TPinputs=TPinputs
    #turbine.jacket.RNAinputs=RNAins
    #turbine.jacket.Twrinputs=Twrinputs
    turbine.jacket.Twrinputs.Twrmatins=Twrmatin
    turbine.jacket.Twrinputs.Htwr2frac=0.2   #fraction of tower height with constant x-section
    turbine.jacket.Twrinputs.ndiv=np.array([6,6])  #ndiv for uniform and tapered section
    turbine.jacket.Twrinputs.Db=Db
    turbine.jacket.Twrinputs.Dt=Dt
    turbine.jacket.Twrinputs.DTRb=Db/tb
    turbine.jacket.Twrinputs.DTRt=Db/Dt # TODO double check
    turbine.jacket.TwrRigidTop=TwrRigidTop
    turbine.jacket.FrameAuxIns=FrameAuxIns

