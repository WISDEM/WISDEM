from turbsim_vartrees import turbsiminputs

class turbsimReader(object):
    def __init__(self):
        self.turbsim_vt = turbsiminputs()
    def read_input_file(self, input_file_name):
        inpf = open(input_file_name, 'r')

        # Runtime Options
        inpf.readline()
        inpf.readline()
        inpf.readline()
        self.turbsim_vt.runtime_options.echo = inpf.readline().split()[0]
        self.turbsim_vt.runtime_options.RandSeed1 = inpf.readline().split()[0]
        self.turbsim_vt.runtime_options.RandSeed2 = inpf.readline().split()[0]
        self.turbsim_vt.runtime_options.WrBHHTP = inpf.readline().split()[0]
        self.turbsim_vt.runtime_options.WrFHHTP = inpf.readline().split()[0]
        self.turbsim_vt.runtime_options.WrADHH = inpf.readline().split()[0]
        self.turbsim_vt.runtime_options.WrADFF = inpf.readline().split()[0]
        self.turbsim_vt.runtime_options.WrBLFF = inpf.readline().split()[0]
        self.turbsim_vt.runtime_options.WrADTWR = inpf.readline().split()[0]
        self.turbsim_vt.runtime_options.WrFMTFF = inpf.readline().split()[0]
        self.turbsim_vt.runtime_options.WrACT = inpf.readline().split()[0]
        self.turbsim_vt.runtime_options.Clockwise = inpf.readline().split()[0]
        self.turbsim_vt.runtime_options.ScaleIEC = inpf.readline().split()[0]

        # Turbine/Model Specifications
        inpf.readline()
        inpf.readline()
        self.turbsim_vt.tmspecs.NumGrid_Z = inpf.readline().split()[0]
        self.turbsim_vt.tmspecs.NumGrid_Y = inpf.readline().split()[0]
        self.turbsim_vt.tmspecs.TimeStep = inpf.readline().split()[0]
        self.turbsim_vt.tmspecs.AnalysisTime = inpf.readline().split()[0]
        self.turbsim_vt.tmspecs.UsableTime = inpf.readline().split()[0]
        self.turbsim_vt.tmspecs.HubHt = inpf.readline().split()[0]
        self.turbsim_vt.tmspecs.GridHeight = inpf.readline().split()[0]
        self.turbsim_vt.tmspecs.GridWidth = inpf.readline().split()[0]
        self.turbsim_vt.tmspecs.VFlowAng = inpf.readline().split()[0]
        self.turbsim_vt.tmspecs.HFlowAng = inpf.readline().split()[0]

        # Meteorological Boundary Conditions 
        inpf.readline()
        inpf.readline()
        self.turbsim_vt.metboundconds.TurbModel = inpf.readline().split()[0]
        self.turbsim_vt.metboundconds.UserFile = inpf.readline().split()[0].replace("'","").replace('"','')
        self.turbsim_vt.metboundconds.IECstandard = inpf.readline().split()[0]
        self.turbsim_vt.metboundconds.IECturbc = inpf.readline().split()[0]
        self.turbsim_vt.metboundconds.IEC_WindType = inpf.readline().split()[0]
        self.turbsim_vt.metboundconds.ETMc = inpf.readline().split()[0]
        self.turbsim_vt.metboundconds.WindProfileType = inpf.readline().split()[0]
        self.turbsim_vt.metboundconds.ProfileFile = inpf.readline().split()[0].replace("'","").replace('"','')
        self.turbsim_vt.metboundconds.RefHt = inpf.readline().split()[0]
        self.turbsim_vt.metboundconds.URef = inpf.readline().split()[0]
        self.turbsim_vt.metboundconds.ZJetMax = inpf.readline().split()[0]
        self.turbsim_vt.metboundconds.PLExp = inpf.readline().split()[0]
        self.turbsim_vt.metboundconds.Z0 = inpf.readline().split()[0]


        # Meteorological Boundary Conditions 
        inpf.readline()
        inpf.readline()
        self.turbsim_vt.noniecboundconds.Latitude = inpf.readline().split()[0]
        self.turbsim_vt.noniecboundconds.RICH_NO = inpf.readline().split()[0]
        self.turbsim_vt.noniecboundconds.UStar = inpf.readline().split()[0]
        self.turbsim_vt.noniecboundconds.ZI = inpf.readline().split()[0]
        self.turbsim_vt.noniecboundconds.PC_UW = inpf.readline().split()[0]
        self.turbsim_vt.noniecboundconds.PC_UV = inpf.readline().split()[0]
        self.turbsim_vt.noniecboundconds.PC_VW = inpf.readline().split()[0]

        # Spatial Coherence Parameters
        inpf.readline()
        inpf.readline()
        self.turbsim_vt.spatialcoherance.SCMod1 = inpf.readline().split()[0]
        self.turbsim_vt.spatialcoherance.SCMod2 = inpf.readline().split()[0]
        self.turbsim_vt.spatialcoherance.SCMod3 = inpf.readline().split()[0]
        self.turbsim_vt.spatialcoherance.InCDec1 = inpf.readline()[1:-2].split()
        self.turbsim_vt.spatialcoherance.InCDec2 = inpf.readline()[1:-2].split()
        self.turbsim_vt.spatialcoherance.InCDec3 = inpf.readline()[1:-2].split()
        self.turbsim_vt.spatialcoherance.CohExp = inpf.readline().split()[0]

        # Spatial Coherence Parameters
        inpf.readline()
        inpf.readline()
        self.turbsim_vt.coherentTurbulence.CTEventPath = inpf.readline().split()[0]
        self.turbsim_vt.coherentTurbulence.CTEventFile = inpf.readline().split()[0]
        self.turbsim_vt.coherentTurbulence.Randomize = inpf.readline().split()[0]
        self.turbsim_vt.coherentTurbulence.DistScl = inpf.readline().split()[0]
        self.turbsim_vt.coherentTurbulence.CTLy = inpf.readline().split()[0]
        self.turbsim_vt.coherentTurbulence.CTLz = inpf.readline().split()[0]
        self.turbsim_vt.coherentTurbulence.CTStartTime = inpf.readline().split()[0]

if __name__=='__main__':
    reader = turbsimReader()
    reader.read_input_file('turbsim_default.in')
