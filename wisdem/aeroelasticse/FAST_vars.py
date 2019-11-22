from numpy import zeros, array
import numpy as np
from wisdem.aeroelasticse.FAST_vars_out import FstOutput, Fst7Output

# This variable tree contains all parameters required to create a FAST model
# for FAST versions 7 and 8.


# .fst Simulation Control
Fst = {}
Fst['Echo']            = False         
Fst['AbortLevel']      = ''            
Fst['TMax']            = 0.0           
Fst['DT']              = 0.0           
Fst['InterpOrder']     = 0             
Fst['NumCrctn']        = 0             
Fst['DT_UJac']         = 0.0           
Fst['UJacSclFact']     = 0.0           

# Feature Switches and Flags
Fst['CompElast']    = 0             
Fst['CompInflow']   = 0             
Fst['CompAero']     = 0             
Fst['CompServo']    = 0             
Fst['CompHydro']    = 0             
Fst['CompSub']      = 0             
Fst['CompMooring']  = 0             
Fst['CompIce']      = 0             
Fst['CompNoise']    = 0             #FAST7 only

# Input Files
Fst['EDFile']          = ''            
Fst['BDBldFile1']      = ''            
Fst['BDBldFile2']      = ''            
Fst['BDBldFile3']      = ''            
Fst['InflowFile']      = ''            
Fst['AeroFile']        = ''            
Fst['ServoFile']       = ''            
Fst['HydroFile']       = ''            
Fst['SubFile']         = ''            
Fst['MooringFile']     = ''            
Fst['IceFile']         = ''            

# FAST Output Parameters
Fst['SumPrint']   = False         
Fst['SttsTime']   = 0.0           
Fst['ChkptTime']  = 0.0           
Fst['DT_Out']     = 0.0           
Fst['TStart']     = 0.0           
Fst['OutFileFmt'] = 0             
Fst['TabDelim']   = False         
Fst['OutFmt']     = ''            

# Fst
Fst['Linearize']    = False         
Fst['NLinTimes']    = 2             
Fst['LinTimes']     = (30,          60)
Fst['LinInputs']    = 1             
Fst['LinOutputs']   = 1             
Fst['LinOutJac']    = False         
Fst['LinOutMod']    = False         

# Fst
Fst['WrVTK']        = 0             
Fst['VTK_type']     = 0             
Fst['VTK_fields']   = False         
Fst['VTK_fps']      = 0             

# ElastoDyn Simulation Control
ElastoDyn = {}
ElastoDyn['Echo']             = False         
ElastoDyn['Method']           = 0             
ElastoDyn['DT']               = 0.0           

# Environmental Condition
ElastoDyn['Gravity']          = 0.0           

# Degrees of Freedom
ElastoDyn['FlapElastoDyn1']               = False         
ElastoDyn['FlapElastoDyn2']               = False         
ElastoDyn['EdgeElastoDyn']                = False         
ElastoDyn['TeetElastoDyn']                = False         
ElastoDyn['DrTrElastoDyn']                = False         
ElastoDyn['GenElastoDyn']                 = False         
ElastoDyn['YawElastoDyn']                 = False         
ElastoDyn['TwFAElastoDyn1']               = False         
ElastoDyn['TwFAElastoDyn2']               = False         
ElastoDyn['TwSSElastoDyn1']               = False         
ElastoDyn['TwSSElastoDyn2']               = False         
ElastoDyn['PtfmSgElastoDyn']              = False         
ElastoDyn['PtfmSwElastoDyn']              = False         
ElastoDyn['PtfmHvElastoDyn']              = False         
ElastoDyn['PtfmRElastoDyn']               = False         
ElastoDyn['PtfmPElastoDyn']               = False         
ElastoDyn['PtfmYElastoDyn']               = False         

# Initial Conditions
ElastoDyn['OoPDefl']          = 0.0           
ElastoDyn['IPDefl']           = 0.0           
ElastoDyn['BlPitch1']         = 0.0           
ElastoDyn['BlPitch2']         = 0.0           
ElastoDyn['BlPitch3']         = 0.0           
ElastoDyn['TeetDefl']         = 0.0           
ElastoDyn['Azimuth']          = 0.0           
ElastoDyn['RotSpeed']         = 0.0           
ElastoDyn['NacYaw']           = 0.0           
ElastoDyn['TTDspFA']          = 0.0           
ElastoDyn['TTDspSS']          = 0.0           
ElastoDyn['PtfmSurge']        = 0.0           
ElastoDyn['PtfmSway']         = 0.0           
ElastoDyn['PtfmHeave']        = 0.0           
ElastoDyn['PtfmRoll']         = 0.0           
ElastoDyn['PtfmPitch']        = 0.0           
ElastoDyn['PtfmYaw']          = 0.0           

# Turbine Configuration
ElastoDyn['NumBl']           = 0             
ElastoDyn['TipRad']          = 0.0           
ElastoDyn['HubRad']          = 0.0           
ElastoDyn['PreCone(1)']        = 0.0           
ElastoDyn['PreCone(2)']        = 0.0           
ElastoDyn['PreCone(3)']        = 0.0           
ElastoDyn['HubCM']           = 0.0           
ElastoDyn['UndSling']        = 0.0           
ElastoDyn['Delta3']          = 0.0           
ElastoDyn['AzimB1Up']        = 0.0           
ElastoDyn['OverHang']        = 0.0           
ElastoDyn['ShftGagL']        = 0.0           
ElastoDyn['ShftTilt']        = 0.0           
ElastoDyn['NacCMxn']         = 0.0           
ElastoDyn['NacCMyn']         = 0.0           
ElastoDyn['NacCMzn']         = 0.0           
ElastoDyn['NcIMUxn']         = 0.0           
ElastoDyn['NcIMUyn']         = 0.0           
ElastoDyn['NcIMUzn']         = 0.0           
ElastoDyn['Twr2Shft']        = 0.0           
ElastoDyn['TowerHt']         = 0.0           
ElastoDyn['TowerBsHt']       = 0.0           
ElastoDyn['PtfmCMxt']        = 0.0           
ElastoDyn['PtfmCMyt']        = 0.0           
ElastoDyn['PtfmCMzt']        = 0.0           
ElastoDyn['PtfmRefzt']       = 0.0           

# Mass and Inertia
ElastoDyn['TipMass(1)']       = 0.0           
ElastoDyn['TipMass(2)']       = 0.0           
ElastoDyn['TipMass(3)']       = 0.0           
ElastoDyn['HubMass']        = 0.0           
ElastoDyn['HubIner']        = 0.0           
ElastoDyn['GenIner']        = 0.0           
ElastoDyn['NacMass']        = 0.0           
ElastoDyn['NacYIner']       = 0.0           
ElastoDyn['YawBrMass']      = 0.0           
ElastoDyn['PtfmMass']       = 0.0           
ElastoDyn['PtfmRIner']      = 0.0           
ElastoDyn['PtfmPIner']      = 0.0           
ElastoDyn['PtfmYIner']      = 0.0           

# ED Blade (Structure)
ElastoDyn['BldNodes']        = 0             
ElastoDyn['BldFile1']        = ''            
ElastoDyn['BldFile2']        = ''            
ElastoDyn['BldFile3']        = ''            

# Including the blade files and properties in the same object,
# as is done here, implies that the properties are done for all
# blades (assumed for now)

# General Model Inputs
ElastoDynBlade = {}
ElastoDynBlade['NBlInpSt']        = 0             #Number of blade input stations (-)
ElastoDynBlade['BldFlDmp1']       = 0.0           #Blade flap mode #1 structural damping in percent of critical (%)
ElastoDynBlade['BldFlDmp2']       = 0.0           #Blade flap mode #2 structural damping in percent of critical (%)
ElastoDynBlade['BldEdDmp1']       = 0.0           #Blade edge mode #1 structural damping in percent of critical (%)
ElastoDynBlade['FlStTunr1']       = 0.0           #Blade flapwise modal stiffness tuner, 1st mode (-)
ElastoDynBlade['FlStTunr2']       = 0.0           #Blade flapwise modal stiffness tuner, 2nd mode (-)
ElastoDynBlade['AdjBlMs']         = 0.0           #Factor to adjust blade mass density (-)
ElastoDynBlade['AdjFlSt']         = 0.0           #Factor to adjust blade flap stiffness (-)
ElastoDynBlade['AdjEdSt']         = 0.0           #Factor to adjust blade edge stiffness (-)
        
# Distributed Blade Properties
ElastoDynBlade['BlFract']         = zeros([1])    
ElastoDynBlade['AeroCent']        = zeros([1])    
ElastoDynBlade['PitchAxis']       = zeros([1])    
ElastoDynBlade['StrcTwst']        = zeros([1])    
ElastoDynBlade['BMassDen']        = zeros([1])    
ElastoDynBlade['FlpStff']         = zeros([1])    
ElastoDynBlade['EdgStff']         = zeros([1])    
ElastoDynBlade['GJStff']          = zeros([1])    
ElastoDynBlade['EAStff']          = zeros([1])    
ElastoDynBlade['Alpha']           = zeros([1])    
ElastoDynBlade['FlpIner']         = zeros([1])    
ElastoDynBlade['EdgIner']         = zeros([1])    
ElastoDynBlade['PrecrvRef']       = zeros([1])    
ElastoDynBlade['PreswpRef']       = zeros([1])    #[AH] Added during openmdao1 update
ElastoDynBlade['FlpcgOf']         = zeros([1])    
ElastoDynBlade['Edgcgof']         = zeros([1])    
ElastoDynBlade['FlpEAOf']         = zeros([1])    
ElastoDynBlade['EdgEAOf']         = zeros([1])    
        
# Blade Mode Shapes
ElastoDynBlade['BldFl1Sh']        = zeros([1])    
ElastoDynBlade['BldFl2Sh']        = zeros([1])    
ElastoDynBlade['BldEdgSh']        = zeros([1])    

# Rotor-Teeter
ElastoDyn['TeetMod']        = 0             
ElastoDyn['TeetDmpP']       = 0.0           
ElastoDyn['TeetDmp']        = 0.0           
ElastoDyn['TeetCDmp']       = 0.0           
ElastoDyn['TeetSStP']       = 0.0           
ElastoDyn['TeetHStP']       = 0.0           
ElastoDyn['TeetSSSp']       = 0.0           
ElastoDyn['TeetHSSp']       = 0.0           

ElastoDyn['GBoxEff']         = 0.0           
ElastoDyn['GBRatio']         = 0.0           
ElastoDyn['DTTorSpr']        = 0.0           
ElastoDyn['DTTorDmp']        = 0.0           

ElastoDyn['ElastoDyn']            = False         
ElastoDyn['FurlFile']           = ''            

ElastoDyn['TwrNodes']             = 0             
ElastoDyn['TwrFile']              = ''            

# General Tower Parameters
ElastoDynTower = {}
ElastoDynTower['NTwInptSt']            = 0             #Number of input stations to specify tower geometry
ElastoDynTower['CalcTMode']            = False         #calculate tower mode shapes internally {T: ignore mode shapes from below, F: use mode shapes from below} [CURRENTLY IGNORED] (flag)
ElastoDynTower['TwrFADmp1']            = 0.0           #Tower 1st fore-aft mode structural damping ratio (%)
ElastoDynTower['TwrFADmp2']            = 0.0           #Tower 2nd fore-aft mode structural damping ratio (%)
ElastoDynTower['TwrSSDmp1']            = 0.0           #Tower 1st side-to-side mode structural damping ratio (%)
ElastoDynTower['TwrSSDmp2']            = 0.0           #Tower 2nd side-to-side mode structural damping ratio (%)

# Tower Adjustment Factors
ElastoDynTower['FAStTunr1']            = 0.0           #Tower fore-aft modal stiffness tuner, 1st mode (-)
ElastoDynTower['FAStTunr2']            = 0.0           #Tower fore-aft modal stiffness tuner, 2nd mode (-)
ElastoDynTower['SSStTunr1']            = 0.0           #Tower side-to-side stiffness tuner, 1st mode (-)
ElastoDynTower['SSStTunr2']            = 0.0           #Tower side-to-side stiffness tuner, 2nd mode (-)
ElastoDynTower['AdjTwMa']              = 0.0           #Factor to adjust tower mass density (-)
ElastoDynTower['AdjFASt']              = 0.0           #Factor to adjust tower fore-aft stiffness (-)
ElastoDynTower['AdjSSSt']              = 0.0           #Factor to adjust tower side-to-side stiffness (-)
     
# Distributed Tower Properties
ElastoDynTower['HtFract']              = zeros([1])    
ElastoDynTower['TMassDen']             = zeros([1])    
ElastoDynTower['TwFAStif']             = zeros([1])    
ElastoDynTower['TwSSStif']             = zeros([1])    
ElastoDynTower['TwGJStif']             = zeros([1])    
ElastoDynTower['TwEAStif']             = zeros([1])    
ElastoDynTower['TwFAIner']             = zeros([1])    
ElastoDynTower['TwSSIner']             = zeros([1])    
ElastoDynTower['TwFAcgOf']             = zeros([1])    
ElastoDynTower['TwSScgOf']             = zeros([1])    
        
# Tower Mode Shapes
ElastoDynTower['TwFAM1Sh']             = zeros([1])    #Tower Fore-Aft Mode 1 Shape Coefficients x^2, x^3, x^4, x^5, x^6
ElastoDynTower['TwFAM2Sh']             = zeros([1])    #Tower Fore-Aft Mode 2 Shape Coefficients x^2, x^3, x^4, x^5, x^6
ElastoDynTower['TwSSM1Sh']             = zeros([1])    #Tower Side-to-Side Mode 1 Shape Coefficients x^2, x^3, x^4, x^5, x^6
ElastoDynTower['TwSSM2Sh']             = zeros([1])    #Tower Side-to-Side Mode 2 Shape Coefficients x^2, x^3, x^4, x^5, x^6

ElastoDyn = {}
ElastoDyn['SumPrint']       = False         
ElastoDyn['OutFile']        = 0             
ElastoDyn['TabDelim']       = False         
ElastoDyn['OutFmt']         = ''            
ElastoDyn['TStart']         = 0.0           
ElastoDyn['DecFact']        = 0.0           
ElastoDyn['NTwGages']       = 0             
ElastoDyn['TwrGagNd']       = []            
ElastoDyn['NBlGages']       = 0             
ElastoDyn['BldGagNd']       = []            

# Inflow Wind General Parameters
InflowWind = {}
InflowWind['Echo']            = False         
InflowWind['WindType']        = 0             
InflowWind['PropagationDir']  = 0.0           
InflowWind['NWindVel']        = 0             
InflowWind['WindVxiList']     = 0.0           
InflowWind['WindVyiList']     = 0.0           
InflowWind['WindVziList']     = 0.0           

# Parameters for Steady Wind Conditions [used only for WindType = 1]
InflowWind['HWindSpeed'] = 0.0           
InflowWind['RefHt']     = 0.0           
InflowWind['PLexp']     = 0.0           

# Parameters for Uniform wind file [used only for WindType = 2]
InflowWind['Filename'] = ''            
InflowWind['RefHt']    = 0.0           
InflowWind['RefLength'] = 0.0           

# Parameters for Binary TurbSim Full-Field files [used only for WindType = 3]
InflowWind['Filename'] = ''            

# Parameters for Binary Bladed-style Full-Field files [used only for WindType = 4]
InflowWind['FilenameRoot'] = ''            
InflowWind['TowerFile'] = False         

# Parameters for HAWC-format binary files [Only used with WindType = 5]
InflowWind['FileName_u']  = ''            
InflowWind['FileName_v']  = ''            
InflowWind['FileName_w']  = ''            
InflowWind['nx']          = 0             
InflowWind['ny']          = 0             
InflowWind['nz']          = 0             
InflowWind['dx']          = 0.0           
InflowWind['dy']          = 0.0           
InflowWind['dz']          = 0.0           
InflowWind['RefHt']       = 0.0           
InflowWind['ScaleMethod'] = 0             
InflowWind['SFx']         = 0.0           
InflowWind['SFy']         = 0.0           
InflowWind['SFz']         = 0.0           
InflowWind['SigmaFx']     = 0.0           
InflowWind['SigmaFy']     = 0.0           
InflowWind['SigmaFz']     = 0.0           
InflowWind['URef']        = 0.0           
InflowWind['WindProfile'] = 0             
InflowWind['PLExp']       = 0.0           
InflowWind['Z0']          = 0.0           

# Inflow Wind Output Parameters (actual OutList included in master OutList)
InflowWind['SumPrint']   = False         

# # Wnd Wind File Parameters
# WndWind = {}
# WndWind['TimeSteps']          = 0             #number of time steps
# WndWind['Time']               = zeros([1])    #time steps
# WndWind['HorSpd']             = zeros([1])    #horizontal wind speed
# WndWind['WindDir']            = zeros([1])    #wind direction
# WndWind['VerSpd']             = zeros([1])    #vertical wind speed
# WndWind['HorShr']             = zeros([1])    #horizontal shear
# WndWind['VerShr']             = zeros([1])    #vertical power-law shear
# WndWind['LnVShr']             = zeros([1])    #vertical linear shear
# WndWind['GstSpd']             = zeros([1])    #gust speed not sheared by Aerodyn

# AeroDyn Parameters
AeroDyn14 = {}
# General Model Inputs
AeroDyn14['StallMod']         = ""
AeroDyn14['UseCm']            = ""
AeroDyn14['InfModel']         = ""
AeroDyn14['IndModel']         = ""
AeroDyn14['AToler']           = 0.
AeroDyn14['TLModel']          = ""
AeroDyn14['HLModel']          = ""
AeroDyn14['TwrShad']          = ""
AeroDyn14['TwrPotent']        = False
AeroDyn14['TwrShadow']        = False
AeroDyn14['TwrFile']          = ""
AeroDyn14['CalcTwrAero']      = False
AeroDyn14['AirDens']          = 0.0
AeroDyn14['KinVisc']          = 0.0
AeroDyn14['DTAero']           = "default"
AeroDyn14['NumFoil']          = 0
AeroDyn14['FoilNm']           = [""]

AeroDynBlade = {}
AeroDynBlade['BldNodes']         = 0
AeroDynBlade['RNodes']           = np.asarray([]) 
AeroDynBlade['AeroTwst']         = np.asarray([]) 
AeroDynBlade['DRNodes']          = np.asarray([]) 
AeroDynBlade['Chord']            = np.asarray([]) 
AeroDynBlade['NFoil']            = np.asarray([]) 
AeroDynBlade['PrnElm']           = np.asarray([]) 


# AeroDyn Blade
AeroDynTower = {}
AeroDynTower['NTwrHt']        = 0             
AeroDynTower['NTwrRe']        = 0             
AeroDynTower['NTwrCD']        = 0             
AeroDynTower['Tower_Wake_Constant'] = 0.            
AeroDynTower['TwrHtFr']       = np.asarray([]) 
AeroDynTower['TwrWid']        = np.asarray([]) 
AeroDynTower['NTwrCDCol']     = np.asarray([]) 
AeroDynTower['TwrRe']         = np.asarray([]) 
AeroDynTower['TwrCD']         = np.asarray([]) 


# AeroDyn Airfoil Polar
AeroDynPolar = {}
AeroDynPolar['IDParam']     = 0.0           #Table ID Parameter (Typically Reynolds number)
AeroDynPolar['StallAngle']  = 0.0           #Stall angle (deg)
AeroDynPolar['ZeroCn']      = 0.0           #Zero lift angle of attack (deg)
AeroDynPolar['CnSlope']     = 0.0           #Cn slope for zero lift (dimensionless)
AeroDynPolar['CnPosStall']  = 0.0           #Cn at stall value for positive angle of attack
AeroDynPolar['CnNegStall']  = 0.0           #Cn at stall value for negative angle of attack
AeroDynPolar['alphaCdMin']  = 0.0           #Angle of attack for minimum CD (deg)
AeroDynPolar['CdMin']       = 0.0           #Minimum Cd Value

AeroDynPolar['alpha']       = zeros([1])    #angle of attack
AeroDynPolar['cl']          = zeros([1])    #coefficient of lift
AeroDynPolar['cd']          = zeros([1])    #coefficient of drag
AeroDynPolar['cm']          = zeros([1])    #coefficient of the pitching moment

AeroDynPolar['Re']          = 0.0
AeroDynPolar['Ctrl']        = 0.0
AeroDynPolar['InclUAdata']  = 0.0
AeroDynPolar['alpha0']      = 0.0
AeroDynPolar['alpha1']      = 0.0
AeroDynPolar['alpha2']      = 0.0
AeroDynPolar['eta_e']       = 0.0
AeroDynPolar['C_nalpha']    = 0.0
AeroDynPolar['T_f0']        = 0.0
AeroDynPolar['T_V0']        = 0.0
AeroDynPolar['T_p']         = 0.0
AeroDynPolar['T_VL']        = 0.0
AeroDynPolar['b1']          = 0.0
AeroDynPolar['b2']          = 0.0
AeroDynPolar['b5']          = 0.0
AeroDynPolar['A1']          = 0.0
AeroDynPolar['A2']          = 0.0
AeroDynPolar['A5']          = 0.0
AeroDynPolar['S1']          = 0.0
AeroDynPolar['S2']          = 0.0
AeroDynPolar['S3']          = 0.0
AeroDynPolar['S4']          = 0.0
AeroDynPolar['Cn1']         = 0.0
AeroDynPolar['Cn2']         = 0.0
AeroDynPolar['St_sh']       = 0.0
AeroDynPolar['Cd0']         = 0.0
AeroDynPolar['Cm0']         = 0.0
AeroDynPolar['k0']          = 0.0
AeroDynPolar['k1']          = 0.0
AeroDynPolar['k2']          = 0.0
AeroDynPolar['k3']          = 0.0
AeroDynPolar['k1_hat']      = 0.0
AeroDynPolar['x_cp_bar']    = 0.0
AeroDynPolar['UACutout']    = 0.0
AeroDynPolar['filtCutOff']  = 0.0

# AeroDyn15
AeroDyn15 = {}
AeroDyn15['Echo']           = False
AeroDyn15['DTAero']         = 0.0
AeroDyn15['WakeMod']        = 0
AeroDyn15['AFAeroMod']      = 0
AeroDyn15['TwrPotent']      = 0
AeroDyn15['TwrShadow']      = False
AeroDyn15['TwrAero']        = False
AeroDyn15['FrozenWake']     = False
AeroDyn15['CavitCheck']     = False

AeroDyn15['AirDens']        = 0.0
AeroDyn15['KinVisc']        = 0.0
AeroDyn15['SpdSound']       = 0.0
AeroDyn15['Patm']           = 0.0
AeroDyn15['Pvap']           = 0.0
AeroDyn15['FluidDepth']     = 0.0

AeroDyn15['SkewMod']        = 0
AeroDyn15['SkewModFactor']  = "default"
AeroDyn15['TipLoss']        = False
AeroDyn15['HubLoss']        = False
AeroDyn15['TanInd']         = False
AeroDyn15['AIDrag']         = False
AeroDyn15['TIDrag']         = False
AeroDyn15['IndToler']       = 0.0
AeroDyn15['MaxIter']        = 0

AeroDyn15['DBEMT_Mod']      = 2
AeroDyn15['tau1_const']     = 4

AeroDyn15['UAMod']          = 0
AeroDyn15['FLookup']        = False

AeroDyn15['InCol_Alfa']        = 0
AeroDyn15['InCol_Cl']          = 0
AeroDyn15['InCol_Cd']          = 0
AeroDyn15['InCol_Cm']          = 0
AeroDyn15['InCol_Cpmin']       = 0
AeroDyn15['NumAFfiles']        = 0
AeroDyn15['AFNames']           = []

AeroDyn15['UseBlCm']           = False
AeroDyn15['ADBlFile1']         = ''
AeroDyn15['ADBlFile2']         = ''
AeroDyn15['ADBlFile3']         = ''

AeroDyn15['NumTwrNds']         = 0
AeroDyn15['TwrElev']           = []
AeroDyn15['TwrDiam']           = []
AeroDyn15['TwrCd']             = []
AeroDyn15['TwrElev']           = [] 
AeroDyn15['TwrDiam']           = [] 
AeroDyn15['TwrCd']             = []

AeroDyn15['SumPrint']          = False
AeroDyn15['NBlOuts']           = 0
AeroDyn15['BlOutNd']           = []
AeroDyn15['NTwOuts']           = 0
AeroDyn15['TwOutNd']           = []


# ServoDyn Simulation Control
ServoDyn = {}
ServoDyn['Echo']             = False         
ServoDyn['DT']               = 0.0           

# Pitch Control
ServoDyn['PCMode']           = 0             
ServoDyn['TPCOn']            = 0.0           
ServoDyn['TPitManS1']        = 0.0           
ServoDyn['TPitManS2']        = 0.0           
ServoDyn['TPitManS3']        = 0.0           
ServoDyn['TPitManE1']        = 0.0           #FAST7 only
ServoDyn['TPitManE2']        = 0.0           #FAST7 only
ServoDyn['TPitManE3']        = 0.0           #FAST7 only
ServoDyn['PitManRat1']       = 0.0           
ServoDyn['PitManRat2']       = 0.0           
ServoDyn['PitManRat3']       = 0.0           
ServoDyn['BlPitchF1']        = 0.0           
ServoDyn['BlPitchF2']        = 0.0           
ServoDyn['BlPitchF3']        = 0.0           
ServoDyn['BlPitch1']         = 0.0           #FAST7 only
ServoDyn['BlPitch2']         = 0.0           #FAST7 only
ServoDyn['BlPitch3']         = 0.0           #FAST7 only

# Generator and Torque Control
ServoDyn['VSContrl']         = 0             
ServoDyn['GenModel']         = 0             
ServoDyn['GenEff']           = 0.0           
ServoDyn['GenTiStr']         = False         
ServoDyn['GenTiStp']         = False         
ServoDyn['SpdGenOn']         = 0.0           
ServoDyn['TimGenOn']         = 0.0           
ServoDyn['TimGenOf']         = 0.0           

# Simple Variable-Speed Torque Control
ServoDyn['VS_RtGnSp']        = 0.0           
ServoDyn['VS_RtTq']          = 0.0           
ServoDyn['VS_Rgn2K']         = 0.0           
ServoDyn['VS_SlPc']          = 0.0           

# Simple Induction Generator
ServoDyn['SIG_SlPc']         = 0.0           
ServoDyn['SIG_SySp']         = 0.0           
ServoDyn['SIG_RtTq']         = 0.0           
ServoDyn['SIG_PORt']         = 0.0           

# Thevenin-Equivalent Induction Generator
ServoDyn['TEC_Freq']         = 0.0           
ServoDyn['TEC_NPol']         = 0             
ServoDyn['TEC_SRes']         = 0.0           
ServoDyn['TEC_RRes']         = 0.0           
ServoDyn['TEC_VLL']          = 0.0           
ServoDyn['TEC_SLR']          = 0.0           
ServoDyn['TEC_RLR']          = 0.0           
ServoDyn['TEC_MR']           = 0.0           

# High-Speed Shaft Brake
ServoDyn['HSSBrMode']        = 0             
ServoDyn['THSSBrDp']         = 0.0           
ServoDyn['HSSBrDT']          = 0.0           
ServoDyn['HSSBrTqF']         = 0.0           

# Nacelle-Yaw Control
ServoDyn['YCMode']           = 0             
ServoDyn['TYCOn']            = 0.0           
ServoDyn['YawNeut']          = 0.0           
ServoDyn['YawSpr']           = 0.0           
ServoDyn['YawDamp']          = 0.0           
ServoDyn['TYawManS']         = 0.0           
ServoDyn['YawManRat']        = 0.0           
ServoDyn['NacYawF']          = 0.0           

# Tip Brake (used in FAST7 only)
ServoDyn['TiDynBrk']         = 0.0           
ServoDyn['TTpBrDp1']         = 0.0           
ServoDyn['TTpBrDp2']         = 0.0           
ServoDyn['TTpBrDp3']         = 0.0           
ServoDyn['TBDepISp1']        = 0.0           
ServoDyn['TBDepISp2']        = 0.0           
ServoDyn['TBDepISp3']        = 0.0           
ServoDyn['TBDrConN']         = 0.0           
ServoDyn['TBDrConD']         = 0.0           
ServoDyn['TpBrDT']           = 0.0           

# Tuned Mass Damper
ServoDyn = {}
ServoDyn['CompNTMD']         = False         
ServoDyn['NTMDfile']         = ''            
ServoDyn['CompTTMD']         = False         
ServoDyn['TTMDfile']         = ''            

# Bladed Interface
ServoDyn['DLL_FileName']     = ''            
ServoDyn['DLL_InFile']       = ''            
ServoDyn['DLL_ProcName']     = ''            
ServoDyn['DLL_DT']           = ''            
ServoDyn['DLL_Ramp']         = False         
ServoDyn['BPCutoff']         = 0.0           
ServoDyn['NacYaw_North']     = 0.0           
ServoDyn['Ptch_Cntrl']       = 0.0           
ServoDyn['Ptch_SetPnt']      = 0.0           
ServoDyn['Ptch_Min']         = 0.0           
ServoDyn['Ptch_Max']         = 0.0           
ServoDyn['PtchRate_Min']     = 0.0           
ServoDyn['PtchRate_Max']     = 0.0           
ServoDyn['Gain_OM']          = 0.0           
ServoDyn['GenSpd_MinOM']     = 0.0           
ServoDyn['GenSpd_MaxOM']     = 0.0           
ServoDyn['GenSpd_Dem']       = 0.0           
ServoDyn['GenTrq_Dem']       = 0.0           
ServoDyn['GenPwr_Dem']       = 0.0           
ServoDyn['DLL_NumTrq']       = 0.0           
ServoDyn['GenSpd_TLU']       = zeros([0])    
ServoDyn['GenTrq_TLU']       = zeros([0])    

# ServoDyn Output Params
ServoDyn['SumPrint']         = False         
ServoDyn['OutFile']          = 0             
ServoDyn['TabDelim']         = False         
ServoDyn['OutFmt']           = ''            
ServoDyn['TStart']           = 0.0           

# Bladed style Interface controller input file, intended for ROSCO https://github.com/NREL/ROSCO_toolbox
DISCON_in = {}
DISCON_in['LoggingLevel']      = 0
DISCON_in['F_LPFType']         = 0
DISCON_in['F_NotchType']       = 0
DISCON_in['IPC_ControlMode']   = 0
DISCON_in['VS_ControlMode']    = 0
DISCON_in['PC_ControlMode']    = 0
DISCON_in['Y_ControlMode']     = 0
DISCON_in['SS_Mode']           = 0
DISCON_in['WE_Mode']           = 0
DISCON_in['PS_Mode']           = 0

DISCON_in['F_LPFCornerFreq']   = 0.0
DISCON_in['F_LPFDamping']      = 0.0
DISCON_in['F_NotchCornerFreq'] = 0.0
DISCON_in['F_NotchBetaNumDen'] = []
DISCON_in['F_SSCornerFreq']    = 0.0

DISCON_in['PC_GS_n']           = 0
DISCON_in['PC_GS_angles']      = []
DISCON_in['PC_GS_KP']          = []
DISCON_in['PC_GS_KI']          = []
DISCON_in['PC_GS_KD']          = []
DISCON_in['PC_GS_TF']          = []
DISCON_in['PC_MaxPit']         = 0.0
DISCON_in['PC_MinPit']         = 0.0
DISCON_in['PC_MaxRat']         = 0.0
DISCON_in['PC_MinRat']         = 0.0
DISCON_in['PC_RefSpd']         = 0.0
DISCON_in['PC_FinePit']        = 0.0
DISCON_in['PC_Switch']         = 0.0
DISCON_in['Z_EnableSine']      = 0
DISCON_in['Z_PitchAmplitude']  = 0.0
DISCON_in['Z_PitchFrequency']  = 0.0

DISCON_in['IPC_IntSat']        = 0.0
DISCON_in['IPC_KI']            = []
DISCON_in['IPC_aziOffset']     = []
DISCON_in['IPC_CornerFreqAct'] = 0.0

DISCON_in['VS_GenEff']         = 0.0
DISCON_in['VS_ArSatTq']        = 0.0
DISCON_in['VS_MaxRat']         = 0.0
DISCON_in['VS_MaxTq']          = 0.0
DISCON_in['VS_MinTq']          = 0.0
DISCON_in['VS_MinOMSpd']       = 0.0
DISCON_in['VS_Rgn2K']          = 0.0
DISCON_in['VS_RtPwr']          = 0.0
DISCON_in['VS_RtTq']           = 0.0
DISCON_in['VS_RefSpd']         = 0.0
DISCON_in['VS_n']              = 0
DISCON_in['VS_KP']             = 0.0
DISCON_in['VS_KI']             = 0.0
DISCON_in['VS_TSRopt']         = 0.0

DISCON_in['SS_VSGain']         = 0.0
DISCON_in['SS_PCGain']         = 0.0

DISCON_in['WE_BladeRadius']    = 0.0
DISCON_in['WE_CP_n']           = 0
DISCON_in['WE_CP']             = []
DISCON_in['WE_Gamma']          = 0.0
DISCON_in['WE_GearboxRatio']   = 0.0
DISCON_in['WE_Jtot']           = 0.0
DISCON_in['WE_RhoAir']         = 0.0
DISCON_in['PerfFileName']      = ""
DISCON_in['PerfTableSize']     = []
DISCON_in['WE_FOPoles_N']      = 0
DISCON_in['WE_FOPoles_v']      = []
DISCON_in['WE_FOPoles']        = []

DISCON_in['Y_ErrThresh']       = 0.0
DISCON_in['Y_IPC_IntSat']      = 0.0
DISCON_in['Y_IPC_n']           = 0
DISCON_in['Y_IPC_KP']          = 0.0
DISCON_in['Y_IPC_KI']          = 0.0
DISCON_in['Y_IPC_omegaLP']     = 0.0
DISCON_in['Y_IPC_zetaLP']      = 0.0
DISCON_in['Y_MErrSet']         = 0.0
DISCON_in['Y_omegaLPFast']     = 0.0
DISCON_in['Y_omegaLPSlow']     = 0.0
DISCON_in['Y_Rate']            = 0.0

DISCON_in['FA_KI']             = 0.0
DISCON_in['FA_HPF_CornerFreq'] = 0.0
DISCON_in['FA_IntSat']         = 0.0
DISCON_in['PS_BldPitchMin_N']  = 0
DISCON_in['PS_WindSpeeds']     = []
DISCON_in['PS_BldPitchMin']    = []

# HydroDyn Input File
HydroDyn = {}
HydroDyn['Echo']             = False

# ENVIRONMENTAL CONDITIONS
HydroDyn['WtrDens']          = 0.
HydroDyn['WtrDpth']          = 0.
HydroDyn['MSL2SWL']          = 0.

# WAVES
HydroDyn['WaveMod']          = 0
HydroDyn['WaveStMod']        = 0
HydroDyn['WaveTMax']         = 0.
HydroDyn['WaveDT']           = 0.
HydroDyn['WaveHs']           = 0.
HydroDyn['WaveTp']           = 0.
HydroDyn['WavePkShp']        = "DEFAULT"
HydroDyn['WvLowCOff']        = 0.
HydroDyn['WvHiCOff']         = 0.
HydroDyn['WaveDir']          = 0.
HydroDyn['WaveDirMod']       = 0
HydroDyn['WaveDirSpread']    = 0.
HydroDyn['WaveNDir']         = 0
HydroDyn['WaveDirRange']     = 0.
HydroDyn['WaveSeed1']        = 0.
HydroDyn['WaveSeed2']        = 0.
HydroDyn['WaveNDAmp']        = False
HydroDyn['WvKinFile']        = ""
HydroDyn['NWaveElev']        = 0.
HydroDyn['WaveElevxi']       = 0.
HydroDyn['WaveElevyi']       = 0.

# 2ND-ORDER WAVES
HydroDyn['WvDiffQTF']        = False
HydroDyn['WvSumQTF']         = False
HydroDyn['WvLowCOffD']       = 0.
HydroDyn['WvHiCOffD']        = 0.
HydroDyn['WvLowCOffS']       = 0.
HydroDyn['WvHiCOffS']        = 0.

# CURRENT
HydroDyn['CurrMod']          = 0
HydroDyn['CurrSSV0']         = 0.
HydroDyn['CurrSSDir']        = "DEFAULT"
HydroDyn['CurrNSRef']        = 0.
HydroDyn['CurrNSV0']         = 0.
HydroDyn['CurrNSDir']        = 0.
HydroDyn['CurrDIV']          = 0.
HydroDyn['CurrDIDir']        = 0.

# FLOATING PLATFORM
HydroDyn['PotMod']           = 0
HydroDyn['PotFile']          = ""
HydroDyn['WAMITULEN']        = 0.
HydroDyn['PtfmVol0']         = 0.
HydroDyn['PtfmCOBxt']        = 0.
HydroDyn['PtfmCOByt']        = 0.
HydroDyn['RdtnMod']          = 0
HydroDyn['RdtnTMax']         = 0.
HydroDyn['RdtnDT']           = 0.

# 2ND-ORDER FLOATING PLATFORM FORCES
HydroDyn['MnDrift']          = 0.
HydroDyn['NewmanApp']        = 0.
HydroDyn['DiffQTF']          = 0.
HydroDyn['SumQTF']           = 0.

# FLOATING PLATFORM FORCE FLAGS
HydroDyn['PtfmSgF']          = True
HydroDyn['PtfmSwF']          = True
HydroDyn['PtfmHvF']          = True
HydroDyn['PtfmRF']           = True
HydroDyn['PtfmPF']           = True
HydroDyn['PtfmYF']           = True

# PLATFORM ADDITIONAL STIFFNESS AND DAMPING
HydroDyn['AddF0']            = np.zeros((1,6))
HydroDyn['AddCLin']          = np.zeros((6,6))
HydroDyn['AddBLin']          = np.zeros((6,6))
HydroDyn['AddBQuad']         = np.zeros((6,6))

# AXIAL COEFFICIENTS
HydroDyn['NAxCoef']          = 0
HydroDyn['AxCoefID']         = 0
HydroDyn['AxCd']             = 0.
HydroDyn['AxCa']             = 0.
HydroDyn['AxCp']             = 0.

# MEMBER JOINTS
HydroDyn['NJoints']          = 0
HydroDyn['JointID']          = []
HydroDyn['Jointxi']          = []
HydroDyn['Jointyi']          = []
HydroDyn['Jointzi']          = []
HydroDyn['JointAxID']        = []
HydroDyn['JointOvrlp']       = []

# MEMBER CROSS-SECTION PROPERTIES
HydroDyn['NPropSets']        = 0
HydroDyn['PropSetID']        = []
HydroDyn['PropD']            = []
HydroDyn['PropThck']         = []

# SIMPLE HYDRODYNAMIC COEFFICIENTS
HydroDyn['SimplCd']          = 0.
HydroDyn['SimplCdMG']        = 0.
HydroDyn['SimplCa']          = 0.
HydroDyn['SimplCaMG']        = 0.
HydroDyn['SimplCp']          = 0.
HydroDyn['SimplCpMG']        = 0.
HydroDyn['SimplAxCa']        = 0.
HydroDyn['SimplAxCaMG']      = 0.
HydroDyn['SimplAxCp']        = 0.
HydroDyn['SimplAxCpMG']      = 0.

# DEPTH-BASED HYDRODYNAMIC COEFFICIENTS
HydroDyn['NCoefDpth']        = 0
HydroDyn['Dpth']             = []
HydroDyn['DpthCd']           = []
HydroDyn['DpthCdMG']         = []
HydroDyn['DpthCa']           = []
HydroDyn['DpthCaMG']         = []
HydroDyn['DpthCp']           = []
HydroDyn['DpthCpMG']         = []
HydroDyn['DpthAxCa']         = []
HydroDyn['DpthAxCaMG']       = []
HydroDyn['DpthAxCp']         = []
HydroDyn['DpthAxCpMG']       = []

# MEMBER-BASED HYDRODYNAMIC COEFFICIENTS
HydroDyn['NCoefMembers']     = 0
HydroDyn['MemberID_HydC']    = []
HydroDyn['MemberCd1']        = []
HydroDyn['MemberCd2']        = []
HydroDyn['MemberCdMG1']      = []
HydroDyn['MemberCdMG2']      = []
HydroDyn['MemberCa1']        = []
HydroDyn['MemberCa2']        = []
HydroDyn['MemberCaMG1']      = []
HydroDyn['MemberCaMG2']      = []
HydroDyn['MemberCp1']        = []
HydroDyn['MemberCp2']        = []
HydroDyn['MemberCpMG1']      = []
HydroDyn['MemberCpMG2']      = []
HydroDyn['MemberAxCa1']      = []
HydroDyn['MemberAxCa2']      = []
HydroDyn['MemberAxCaMG1']    = []
HydroDyn['MemberAxCaMG2']    = []
HydroDyn['MemberAxCp1']      = []
HydroDyn['MemberAxCp2']      = []
HydroDyn['MemberAxCpMG1']    = []
HydroDyn['MemberAxCpMG2']    = []

# MEMBERS
HydroDyn['NMembers']         = 0
HydroDyn['MemberID']         = []
HydroDyn['MJointID1']        = []
HydroDyn['MJointID2']        = []
HydroDyn['MPropSetID1']      = []
HydroDyn['MPropSetID2']      = []
HydroDyn['MDivSize']         = []
HydroDyn['MCoefMod']         = []
HydroDyn['PropPot']          = []

# FILLED MEMBERS
HydroDyn['NFillGroups']      = 0
HydroDyn['FillNumM']         = []
HydroDyn['FillMList']        = []
HydroDyn['FillFSLoc']        = []
HydroDyn['FillDens']         = []

# MARINE GROWTH
HydroDyn['NMGDepths']        = 0
HydroDyn['MGDpth']           = []
HydroDyn['MGThck']           = []
HydroDyn['MGDens']           = []

# MEMBER OUTPUT LIST
HydroDyn['NMOutputs']        = 0
HydroDyn['MemberID_out']     = []
HydroDyn['NOutLoc']          = []
HydroDyn['NodeLocs']         = []

# JOINT OUTPUT LIST
HydroDyn['NJOutputs']        = 0
HydroDyn['JOutLst']          = 0

# OUTPUT
HydroDyn['HDSum']            = True
HydroDyn['OutAll']           = False
HydroDyn['OutSwtch']         = 2
HydroDyn['OutFmt']           = ""
HydroDyn['OutSFmt']          = ""

## MAP++ Input File
# LINE DICTIONARY
MAP = {}
MAP['LineType']              = ""
MAP['Diam']                  = 0.
MAP['MassDenInAir']          = 0.
MAP['EA']                    = 0.
MAP['CB']                    = 0.
MAP['CIntDamp']              = 0.
MAP['Ca']                    = 0.
MAP['Cdn']                   = 0.
MAP['Cdt']                   = 0.

# NODE PROPERTIES
MAP['Node']                  = []
MAP['Type']                  = []
MAP['X']                     = []
MAP['Y']                     = []
MAP['Z']                     = []
MAP['M']                     = []
MAP['B']                     = []
MAP['FX']                    = []
MAP['FY']                    = []
MAP['FZ']                    = []

# LINE PROPERTIES
MAP['Line']                  = 0
MAP['LineType']              = ""
MAP['UnstrLen']              = 0.
MAP['NodeAnch']              = 0
MAP['NodeFair']              = 0
MAP['Flags']                 = []

# SOLVER OPTIONS
MAP['Option']                = []

#######################
Fst7 = {}
Fst7['Echo']        = False
Fst7['ADAMSPrep']   = 0
Fst7['AnalMode']    = 0
Fst7['NumBl']       = 0
Fst7['TMax']        = 0.
Fst7['DT']          = 0.
Fst7['YCMode']      = 0
Fst7['TYCOn']       = 0.
Fst7['PCMode']      = 0
Fst7['TPCOn']       = 0.
Fst7['VSContrl']    = 0
Fst7['VS_RtGnSp']   = 0.
Fst7['VS_RtTq']     = 0.
Fst7['VS_Rgn2K']    = 0.
Fst7['VS_SlPc']     = 0.
Fst7['GenModel']    = 0
Fst7['GenTiStr']    = False
Fst7['GenTiStp']    = False
Fst7['SpdGenOn']    = 0.
Fst7['TimGenOn']    = 0.
Fst7['TimGenOf']    = 0.
Fst7['HSSBrMode']   = 0
Fst7['THSSBrDp']    = 0.
Fst7['TiDynBrk']    = 0.
Fst7['TTpBrDp1']    = 0.
Fst7['TTpBrDp2']    = 0.
Fst7['TTpBrDp3']    = 0.
Fst7['TBDepISp1']   = 0.
Fst7['TBDepISp2']   = 0.
Fst7['TBDepISp3']   = 0.
Fst7['TYawManS']    = 0.
Fst7['TYawManE']    = 0.
Fst7['NacYawF']     = 0.
Fst7['TPitManS1']   = 0.
Fst7['TPitManS2']   = 0.
Fst7['TPitManS3']   = 0.
Fst7['TPitManE1']   = 0.
Fst7['TPitManE2']   = 0.
Fst7['TPitManE3']   = 0.
Fst7['BlPitch1']    = 0.
Fst7['BlPitch2']    = 0.
Fst7['BlPitch3']    = 0.
Fst7['BlPitchF1']   = 0.
Fst7['BlPitchF2']   = 0.
Fst7['BlPitchF3']   = 0.
Fst7['Gravity']     = 0.
Fst7['FlapDOF1']    = False
Fst7['FlapDOF2']    = False
Fst7['EdgeDOF']     = False
Fst7['TeetDOF']     = False
Fst7['DrTrDOF']     = False
Fst7['GenDOF']      = False
Fst7['YawDOF']      = False
Fst7['TwFADOF1']    = False
Fst7['TwFADOF2']    = False
Fst7['TwSSDOF1']    = False
Fst7['TwSSDOF2']    = False
Fst7['CompAero']    = False
Fst7['CompNoise']   = False
Fst7['OoPDefl']     = 0.
Fst7['IPDefl']      = 0.
Fst7['TeetDefl']    = 0.
Fst7['Azimuth']     = 0.
Fst7['RotSpeed']    = 0.
Fst7['NacYaw']      = 0.
Fst7['TTDspFA']     = 0.
Fst7['TTDspSS']     = 0.
Fst7['TipRad']      = 0.
Fst7['HubRad']      = 0.
Fst7['PSpnElN']     = 0.
Fst7['UndSling']    = 0
Fst7['HubCM']       = 0.
Fst7['OverHang']    = 0.
Fst7['NacCMxn']     = 0.
Fst7['NacCMy']      = 0.
Fst7['NacCMz']      = 0.
Fst7['TowerH']      = 0.
Fst7['Twr2Shft']    = 0.
Fst7['TwrRBHt']     = 0.
Fst7['ShftTilt']    = 0.
Fst7['Delta3']      = 0.
Fst7['PreCone(1)']    = 0.
Fst7['PreCone(2)']    = 0.
Fst7['PreCone(3)']    = 0.
Fst7['AzimB1Up']    = 0.
Fst7['YawBrMass']   = 0.
Fst7['NacMas']      = 0.
Fst7['HubMas']      = 0.
Fst7['TipMass(1)']    = 0.
Fst7['TipMass(2)']    = 0.
Fst7['TipMass(3)']    = 0.
Fst7['NacYIner']    = 0.
Fst7['GenIner']     = 0.
Fst7['HubIner']     = 0.
Fst7['GBoxEff']     = 0.
Fst7['GenEff']      = 0.
Fst7['GBRatio']     = 0.
Fst7['GBRevers']    = False
Fst7['HSSBrTqF']    = 0.
Fst7['HSSBrDT']     = 0.
Fst7['DynBrkFi']    = ""
Fst7['DTTorSpr']    = 0.
Fst7['DTTorDmp']    = 0.
Fst7['SIG_SlPc']    = 0.
Fst7['SIG_SySp']    = 0.
Fst7['SIG_RtTq']    = 0.
Fst7['SIG_PORt']    = 0.
Fst7['TEC_Freq']    = 0.
Fst7['TEC_NPol']    = 0
Fst7['TEC_SRes']    = 0.
Fst7['TEC_RRes']    = 0.
Fst7['TEC_VLL']     = 0.
Fst7['TEC_SLR']     = 0.
Fst7['TEC_RLR']     = 0.
Fst7['TEC_MR']      = 0.
Fst7['PtfmModel']   = 0
Fst7['PtfmFile']    = ""
Fst7['TwrNodes']    = 0
Fst7['TwrFile']     = ""
Fst7['YawSpr']      = 0.0
Fst7['YawDamp']     = 0.0
Fst7['YawNeut']     = 0.0
Fst7['Furling']     = False
Fst7['FurlFile']    = ""
Fst7['TeetMod']     = 0
Fst7['TeetDmpP']    = 0.
Fst7['TeetDmp']     = 0.
Fst7['TeetCDmp']    = 0.
Fst7['TeetSStP']    = 0.
Fst7['TeetHStP']    = 0.
Fst7['TeetSSSp']    = 0.
Fst7['TeetHSSp']    = 0.
Fst7['TBDrConN']    = 0.
Fst7['TBDrConD']    = 0.
Fst7['TpBrDT']      = 0.
Fst7['BldFile1']    = ""
Fst7['BldFile2']    = ""
Fst7['BldFile3']    = ""
Fst7['ADFile']      = ""
Fst7['NoiseFile']   = ""
Fst7['ADAMSFile']   = ""
Fst7['LinFile']     = ""
Fst7['SumPrint']    = False
Fst7['OutFileFmt']  = 1
Fst7['TabDelim']    = False
Fst7['OutFmt']      = ""
Fst7['TStart']      = 0.0
Fst7['DecFact']     = 0.
Fst7['SttsTime']    = 0.
Fst7['NcIMUxn']     = 0.
Fst7['NcIMUyn']     = 0.
Fst7['NcIMUzn']     = 0.
Fst7['ShftGagL']    = 0.
Fst7['NTwGages']    = 0
Fst7['TwrGagNd']    = 0
Fst7['NBlGages']    = 0
Fst7['BldGagNd']    = 0

# ====== INITIALIZE FAST MODEL BY INITIALIZING ALL VARIABLE TREES ======

FstModel = {}

# Description
FstModel['description']       = ''            

FstModel['Fst']               = Fst
FstModel['ElastoDyn']         = ElastoDyn
FstModel['ElastoDynBlade']    = ElastoDynBlade
FstModel['ElastoDynTower']    = ElastoDynTower
FstModel['InflowWind']        = InflowWind
FstModel['AeroDyn14']         = AeroDyn14
FstModel['AeroDyn15']         = AeroDyn15
FstModel['AeroDynBlade']      = AeroDynBlade
FstModel['AeroDynTower']      = AeroDynTower
FstModel['AeroDynPolar']      = AeroDynPolar
FstModel['ServoDyn']          = ServoDyn
FstModel['DISCON_in']         = DISCON_in
FstModel['HydroDyn']          = HydroDyn
FstModel['MAP']               = MAP
FstModel['Fst7']              = Fst7
        
# List of Outputs (all input files -- FST, ED, SD)
# TODO: Update FstOutput for a few new outputs in FAST8
FstModel['outlist']           = FstOutput   #
FstModel['outlist7']          = Fst7Output


