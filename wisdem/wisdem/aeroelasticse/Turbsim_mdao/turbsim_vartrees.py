from numpy import zeros, array
import numpy as np

# This variable tree contains all parameters required to create a FAST model
# for FAST versions 7 and 8.


##########################
### Main input file ######
##########################
# Runtime Options
class runtime_options(object):
    def __init__(self):
         self.echo = False
         self.RandSeed1 = np.random.uniform(1, 1e8)
         self.RandSeed2 = '"RANLUX"'
         self.WrBHHTP = False
         self.WrFHHTP = False
         self.WrADHH = False
         self.WrADFF = True
         self.WrBLFF = False
         self.WrADTWR = False
         self.WrFMTFF = False
         self.WrACT = False
         self.Clockwise = True
         self.ScaleIEC = 0

# Turbine/Model Specifications
class tmspecs(object):
    def __init__(self):
         self.NumGrid_Z = 30
         self.NumGrid_Y = 30
         self.TimeStep = 0.05
         self.AnalysisTime = 630
         self.UsableTime = '"ALL"'
         self.HubHt = 90
         self.GridHeight = 138
         self.GridWidth = 138
         self.VFlowAng = 0
         self.HFlowAng = 0

# Meteorological Boundary Conditions
class metboundconds(object):
    def __init__(self):
         self.TurbModel = '"USRINP"'
         self.UserFile = 'turbulence_user.inp'
         self.IECstandard = '"1-ED3"'
         self.IECturbc = '"B"'
         self.IEC_WindType = '"NTM"'
         self.ETMc = '"default"'
         self.WindProfileType = '"USR"'
         self.ProfileFile = 'shear.profile'
         self.RefHt = 90
         self.URef = 12
         self.ZJetMax = '"default"'
         self.PLExp = '"default"'
         self.Z0 = '"default"'

# Non-IEC Meteorological Boundary Conditions
class noniecboundconds(object):
    def __init__(self):
         self.Latitude = '"default"'
         self.RICH_NO = 0
         self.UStar = '"default"'
         self.ZI = '"default"'
         self.PC_UW = -0.85
         self.PC_UV = 0.15
         self.PC_VW = -0.1

# Spatial Coherence Parameters
class spatialcoherance(object):
    def __init__(self):
         self.SCMod1 = '"GENERAL"'
         self.SCMod2 = '"GENERAL"'
         self.SCMod3 = '"GENERAL"'
         self.InCDec1 = [13.75, 0.04]
         self.InCDec2 = [9.85, 0.0015]
         self.InCDec3 = [9.5, 0.003]
         self.CohExp = 0.5

# Coherent Turbulence Scaling Parameters
class coherentTurbulence(object):
    def __init__(self):
         self.CTEventPath = '"Y:\Wind\Archive\Public\Projects\KH_Billow\EventData"'
         self.CTEventFile = '"Random"'
         self.Randomize = True
         self.DistScl = 1
         self.CTLy = 0.5
         self.CTLz = 0.5
         self.CTStartTime = 30

class turbsiminputs(object):
    def __init__(self):
         self.runtime_options = runtime_options()
         self.tmspecs = tmspecs()
         self.metboundconds = metboundconds()
         self.noniecboundconds = noniecboundconds()
         self.spatialcoherance = spatialcoherance()
         self.coherentTurbulence = coherentTurbulence()

##########################
### Turbulance input file ######
##########################

