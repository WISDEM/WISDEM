import os, re, sys, copy
import yaml
import numpy as np
from functools import reduce
import operator

from wisdem.aeroelasticse.FAST_vars import FstModel
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import utilities as ROSCO_utilities


def fix_path(name):
    """ split a path, then reconstruct it using os.path.join """
    name = re.split("\\\|/", name)
    new = name[0]
    for i in range(1,len(name)):
        new = os.path.join(new, name[i])
    return new

def bool_read(text):
    # convert true/false strings to boolean
    if 'default' in text.lower():
        return str(text)
    else:
        if text.lower() == 'true':
            return True
        else:
            return False

def float_read(text):
    # return float with error handing for "default" values
    if 'default' in text.lower():
        return str(text)
    else:
        try:
            return float(text)
        except:
            return str(text)


def int_read(text):
    # return int with error handing for "default" values
    if 'default' in text.lower():
        return str(text)
    else:
        try:
            return int(text)
        except:
            return str(text)


class InputReader_Common(object):
    """ Methods for reading input files that are (relatively) unchanged across FAST versions."""

    def __init__(self, **kwargs):

        self.FAST_ver = 'OPENFAST'
        self.dev_branch = False      # branch: pullrequest/ganesh : 5b78391
        self.FAST_InputFile = None   # FAST input file (ext=.fst)
        self.FAST_directory = None   # Path to fst directory files
        self.path2dll       = None   # Path to dll file
        self.fst_vt = FstModel

        # Optional population class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        super(InputReader_Common, self).__init__()

    def read_yaml(self):
        f = open(self.FAST_yamlfile, 'r')
        self.fst_vt = yaml.load(f)

    def set_outlist(self, vartree_head, channel_list):
        """ Loop through a list of output channel names, recursively set them to True in the nested outlist dict """

        # given a list of nested dictionary keys, return the dict at that point
        def get_dict(vartree, branch):
            return reduce(operator.getitem, branch, vartree_head)
        # given a list of nested dictionary keys, set the value of the dict at that point
        def set_dict(vartree, branch, val):
            get_dict(vartree, branch[:-1])[branch[-1]] = val
        # recursively loop through outlist dictionaries to set output channels
        def loop_dict(vartree, search_var, branch):
            for var in vartree.keys():
                branch_i = copy.copy(branch)
                branch_i.append(var)
                if type(vartree[var]) is dict:
                    loop_dict(vartree[var], search_var, branch_i)
                else:
                    if var == search_var:
                        set_dict(vartree_head, branch_i, True)

        # loop through outchannels on this line, loop through outlist dicts to set to True
        for var in channel_list:
            var = var.replace(' ', '')
            loop_dict(vartree_head, var, [])

    def read_ElastoDynBlade(self):
        # ElastoDyn v1.00 Blade Input File
        # Currently no differences between FASTv8.16 and OpenFAST.
        if self.FAST_ver.lower() == 'fast7':
            blade_file = os.path.join(self.FAST_directory, self.fst_vt['Fst7']['BldFile1'])
        else:
            blade_file = os.path.join(self.FAST_directory, self.fst_vt['ElastoDyn']['BldFile1'])

        f = open(blade_file)
        # print blade_file
        f.readline()
        f.readline()
        f.readline()
        if self.FAST_ver.lower() == 'fast7':
            f.readline()
        
        # Blade Parameters
        self.fst_vt['ElastoDynBlade']['NBlInpSt'] = int(f.readline().split()[0])
        if self.FAST_ver.lower() == 'fast7':
            self.fst_vt['ElastoDynBlade']['CalcBMode'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDynBlade']['BldFlDmp1'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynBlade']['BldFlDmp2'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynBlade']['BldEdDmp1'] = float_read(f.readline().split()[0])
        
        # Blade Adjustment Factors
        f.readline()
        self.fst_vt['ElastoDynBlade']['FlStTunr1'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynBlade']['FlStTunr2'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynBlade']['AdjBlMs'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynBlade']['AdjFlSt'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynBlade']['AdjEdSt'] = float_read(f.readline().split()[0])
        
        # Distrilbuted Blade Properties
        f.readline()
        f.readline()
        f.readline()
        self.fst_vt['ElastoDynBlade']['BlFract'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
        self.fst_vt['ElastoDynBlade']['PitchAxis'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
        self.fst_vt['ElastoDynBlade']['StrcTwst'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
        self.fst_vt['ElastoDynBlade']['BMassDen'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
        self.fst_vt['ElastoDynBlade']['FlpStff'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
        self.fst_vt['ElastoDynBlade']['EdgStff'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
        if self.FAST_ver.lower() == 'fast7':
            self.fst_vt['ElastoDynBlade']['GJStff'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
            self.fst_vt['ElastoDynBlade']['EAStff'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
            self.fst_vt['ElastoDynBlade']['Alpha'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
            self.fst_vt['ElastoDynBlade']['FlpIner'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
            self.fst_vt['ElastoDynBlade']['EdgIner'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
            self.fst_vt['ElastoDynBlade']['PrecrvRef'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
            self.fst_vt['ElastoDynBlade']['PreswpRef'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
            self.fst_vt['ElastoDynBlade']['FlpcgOf'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
            self.fst_vt['ElastoDynBlade']['Edgcgof'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
            self.fst_vt['ElastoDynBlade']['FlpEAOf'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
            self.fst_vt['ElastoDynBlade']['EdgEAOf'] = [None] * self.fst_vt['ElastoDynBlade']['NBlInpSt']
        for i in range(self.fst_vt['ElastoDynBlade']['NBlInpSt']):
            data = f.readline().split()          
            self.fst_vt['ElastoDynBlade']['BlFract'][i]  = float_read(data[0])
            self.fst_vt['ElastoDynBlade']['PitchAxis'][i]  = float_read(data[1])
            self.fst_vt['ElastoDynBlade']['StrcTwst'][i]  = float_read(data[2])
            self.fst_vt['ElastoDynBlade']['BMassDen'][i]  = float_read(data[3])
            self.fst_vt['ElastoDynBlade']['FlpStff'][i]  = float_read(data[4])
            self.fst_vt['ElastoDynBlade']['EdgStff'][i]  = float_read(data[5])
            if self.FAST_ver.lower() == 'fast7':
                self.fst_vt['ElastoDynBlade']['GJStff'][i]  = float_read(data[6])
                self.fst_vt['ElastoDynBlade']['EAStff'][i]  = float_read(data[7])
                self.fst_vt['ElastoDynBlade']['Alpha'][i]  = float_read(data[8])
                self.fst_vt['ElastoDynBlade']['FlpIner'][i]  = float_read(data[9])
                self.fst_vt['ElastoDynBlade']['EdgIner'][i]  = float_read(data[10])
                self.fst_vt['ElastoDynBlade']['PrecrvRef'][i]  = float_read(data[11])
                self.fst_vt['ElastoDynBlade']['PreswpRef'][i]  = float_read(data[12])
                self.fst_vt['ElastoDynBlade']['FlpcgOf'][i]  = float_read(data[13])
                self.fst_vt['ElastoDynBlade']['Edgcgof'][i]  = float_read(data[14])
                self.fst_vt['ElastoDynBlade']['FlpEAOf'][i]  = float_read(data[15])
                self.fst_vt['ElastoDynBlade']['EdgEAOf'][i]  = float_read(data[16])

        f.readline()
        self.fst_vt['ElastoDynBlade']['BldFl1Sh'] = [None] * 5
        self.fst_vt['ElastoDynBlade']['BldFl2Sh'] = [None] * 5        
        self.fst_vt['ElastoDynBlade']['BldEdgSh'] = [None] * 5
        for i in range(5):
            self.fst_vt['ElastoDynBlade']['BldFl1Sh'][i]  = float_read(f.readline().split()[0])
        for i in range(5):
            self.fst_vt['ElastoDynBlade']['BldFl2Sh'][i]  = float_read(f.readline().split()[0])            
        for i in range(5):
            self.fst_vt['ElastoDynBlade']['BldEdgSh'][i]  = float_read(f.readline().split()[0])        

        f.close()

    def read_ElastoDynTower(self):
        # ElastoDyn v1.00 Tower Input Files
        # Currently no differences between FASTv8.16 and OpenFAST.

        if self.FAST_ver.lower() == 'fast7':
            tower_file = os.path.join(self.FAST_directory, self.fst_vt['Fst7']['TwrFile'])
        else:
            tower_file = os.path.join(self.FAST_directory, self.fst_vt['ElastoDyn']['TwrFile'])  
        
        f = open(tower_file)

        f.readline()
        f.readline()
        if self.FAST_ver.lower() == 'fast7':
            f.readline()

        # General Tower Paramters
        f.readline()
        self.fst_vt['ElastoDynTower']['NTwInpSt'] = int(f.readline().split()[0])
        if self.FAST_ver.lower() == 'fast7':
            self.fst_vt['ElastoDynTower']['CalcTMode'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDynTower']['TwrFADmp1'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynTower']['TwrFADmp2'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynTower']['TwrSSDmp1'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynTower']['TwrSSDmp2'] = float_read(f.readline().split()[0])
    
        # Tower Adjustment Factors
        f.readline()
        self.fst_vt['ElastoDynTower']['FAStTunr1'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynTower']['FAStTunr2'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynTower']['SSStTunr1'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynTower']['SSStTunr2'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynTower']['AdjTwMa'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynTower']['AdjFASt'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDynTower']['AdjSSSt'] = float_read(f.readline().split()[0])
     
        # Distributed Tower Properties   
        f.readline()
        f.readline()
        f.readline()
        self.fst_vt['ElastoDynTower']['HtFract'] = [None] * self.fst_vt['ElastoDynTower']['NTwInpSt']
        self.fst_vt['ElastoDynTower']['TMassDen'] = [None] * self.fst_vt['ElastoDynTower']['NTwInpSt']
        self.fst_vt['ElastoDynTower']['TwFAStif'] = [None] * self.fst_vt['ElastoDynTower']['NTwInpSt']
        self.fst_vt['ElastoDynTower']['TwSSStif'] = [None] * self.fst_vt['ElastoDynTower']['NTwInpSt']
        if self.FAST_ver.lower() == 'fast7':
            self.fst_vt['ElastoDynTower']['TwGJStif'] = [None] * self.fst_vt['ElastoDynTower']['NTwInpSt']
            self.fst_vt['ElastoDynTower']['TwEAStif'] = [None] * self.fst_vt['ElastoDynTower']['NTwInpSt']
            self.fst_vt['ElastoDynTower']['TwFAIner'] = [None] * self.fst_vt['ElastoDynTower']['NTwInpSt']
            self.fst_vt['ElastoDynTower']['TwSSIner'] = [None] * self.fst_vt['ElastoDynTower']['NTwInpSt']
            self.fst_vt['ElastoDynTower']['TwFAcgOf'] = [None] * self.fst_vt['ElastoDynTower']['NTwInpSt']
            self.fst_vt['ElastoDynTower']['TwSScgOf'] = [None] * self.fst_vt['ElastoDynTower']['NTwInpSt']

        for i in range(self.fst_vt['ElastoDynTower']['NTwInpSt']):
            data = f.readline().split()
            self.fst_vt['ElastoDynTower']['HtFract'][i]  = float_read(data[0])
            self.fst_vt['ElastoDynTower']['TMassDen'][i]  = float_read(data[1])
            self.fst_vt['ElastoDynTower']['TwFAStif'][i]  = float_read(data[2])
            self.fst_vt['ElastoDynTower']['TwSSStif'][i]  = float_read(data[3])
            if self.FAST_ver.lower() == 'fast7':
                self.fst_vt['ElastoDynTower']['TwGJStif'][i]  = float_read(data[4])
                self.fst_vt['ElastoDynTower']['TwEAStif'][i]  = float_read(data[5])
                self.fst_vt['ElastoDynTower']['TwFAIner'][i]  = float_read(data[6])
                self.fst_vt['ElastoDynTower']['TwSSIner'][i]  = float_read(data[7])
                self.fst_vt['ElastoDynTower']['TwFAcgOf'][i]  = float_read(data[8])
                self.fst_vt['ElastoDynTower']['TwSScgOf'][i]  = float_read(data[9])           
        
        # Tower Mode Shapes
        f.readline()
        self.fst_vt['ElastoDynTower']['TwFAM1Sh'] = [None] * 5
        self.fst_vt['ElastoDynTower']['TwFAM2Sh'] = [None] * 5
        for i in range(5):
            self.fst_vt['ElastoDynTower']['TwFAM1Sh'][i]  = float_read(f.readline().split()[0])
        for i in range(5):
            self.fst_vt['ElastoDynTower']['TwFAM2Sh'][i]  = float_read(f.readline().split()[0])        
        f.readline()
        self.fst_vt['ElastoDynTower']['TwSSM1Sh'] = [None] * 5
        self.fst_vt['ElastoDynTower']['TwSSM2Sh'] = [None] * 5          
        for i in range(5):
            self.fst_vt['ElastoDynTower']['TwSSM1Sh'][i]  = float_read(f.readline().split()[0])
        for i in range(5):
            self.fst_vt['ElastoDynTower']['TwSSM2Sh'][i]  = float_read(f.readline().split()[0]) 

        f.close()

    def read_AeroDyn14Polar(self, aerodynFile):
        # AeroDyn v14 Airfoil Polar Input File

        # open aerodyn file
        f = open(aerodynFile, 'r')
                
        airfoil = copy.copy(self.fst_vt['AeroDynPolar'])

        # skip through header
        airfoil['description'] = f.readline().rstrip()  # remove newline
        f.readline()
        airfoil['number_tables'] = int(f.readline().split()[0])

        IDParam = [float_read(val) for val in f.readline().split()[0:airfoil['number_tables']]]
        StallAngle = [float_read(val) for val in f.readline().split()[0:airfoil['number_tables']]]
        f.readline()
        f.readline()
        f.readline()
        ZeroCn = [float_read(val) for val in f.readline().split()[0:airfoil['number_tables']]]
        CnSlope = [float_read(val) for val in f.readline().split()[0:airfoil['number_tables']]]
        CnPosStall = [float_read(val) for val in f.readline().split()[0:airfoil['number_tables']]]
        CnNegStall = [float_read(val) for val in f.readline().split()[0:airfoil['number_tables']]]
        alphaCdMin = [float_read(val) for val in f.readline().split()[0:airfoil['number_tables']]]
        CdMin = [float_read(val) for val in f.readline().split()[0:airfoil['number_tables']]]

        data = []
        airfoil['af_tables'] = []
        while True:
            line = f.readline()
            if 'EOT' in line:
                break
            line = [float_read(s) for s in line.split()]
            if len(line) < 1:
                break
            data.append(line)

        # loop through tables
        for i in range(airfoil['number_tables']):
            polar = {}
            polar['IDParam'] = IDParam[i]
            polar['StallAngle'] = StallAngle[i]
            polar['ZeroCn'] = ZeroCn[i]
            polar['CnSlope'] = CnSlope[i]
            polar['CnPosStall'] = CnPosStall[i]
            polar['CnNegStall'] = CnNegStall[i]
            polar['alphaCdMin'] = alphaCdMin[i]
            polar['CdMin'] = CdMin[i]

            alpha = []
            cl = []
            cd = []
            cm = []
            # read polar information line by line
            for datai in data:
                if len(datai) == airfoil['number_tables']*3+1:
                    alpha.append(datai[0])
                    cl.append(datai[1 + 3*i])
                    cd.append(datai[2 + 3*i])
                    cm.append(datai[3 + 3*i])
                elif len(datai) == airfoil['number_tables']*2+1:
                    alpha.append(datai[0])
                    cl.append(datai[1 + 2*i])
                    cd.append(datai[2 + 2*i])

            polar['alpha'] = alpha
            polar['cl'] = cl
            polar['cd'] = cd
            polar['cm'] = cm
            airfoil['af_tables'].append(polar)

        f.close()

        return airfoil

    # def WndWindReader(self, wndfile):
    #     # .Wnd Wind Input File for Inflow
    #     wind_file = os.path.join(self.FAST_directory, wndfile)
    #     f = open(wind_file)

    #     data = []
    #     while 1:
    #         line = f.readline()
    #         if not line:
    #             break
    #         if line.strip().split()[0] != '!' and line[0] != '!':
    #             data.append(line.split())

    #     self.fst_vt['wnd_wind']['TimeSteps'] = len(data)
    #     self.fst_vt['wnd_wind']['Time'] = [None] * len(data)
    #     self.fst_vt['wnd_wind']['HorSpd'] = [None] * len(data)
    #     self.fst_vt['wnd_wind']['WindDir'] = [None] * len(data)
    #     self.fst_vt['wnd_wind']['VerSpd'] = [None] * len(data)
    #     self.fst_vt['wnd_wind']['HorShr'] = [None] * len(data)
    #     self.fst_vt['wnd_wind']['VerShr'] = [None] * len(data)
    #     self.fst_vt['wnd_wind']['LnVShr'] = [None] * len(data)
    #     self.fst_vt['wnd_wind']['GstSpd'] = [None] * len(data)        
    #     for i in range(len(data)):
    #         self.fst_vt['wnd_wind']['Time'][i]  = float_read(data[i][0])
    #         self.fst_vt['wnd_wind']['HorSpd'][i]  = float_read(data[i][1])
    #         self.fst_vt['wnd_wind']['WindDir'][i]  = float_read(data[i][2])
    #         self.fst_vt['wnd_wind']['VerSpd'][i]  = float_read(data[i][3])
    #         self.fst_vt['wnd_wind']['HorShr'][i]  = float_read(data[i][4])
    #         self.fst_vt['wnd_wind']['VerShr'][i]  = float_read(data[i][5])
    #         self.fst_vt['wnd_wind']['LnVShr'][i]  = float_read(data[i][6])
    #         self.fst_vt['wnd_wind']['GstSpd'][i]  = float_read(data[i][7])

    #     f.close()


class InputReader_OpenFAST(InputReader_Common):
    """ OpenFAST / FAST 8.16 input file reader """
    
    def execute(self):
          
        self.read_MainInput()
        self.read_ElastoDyn()
        self.read_ElastoDynBlade()
        self.read_ElastoDynTower()
        self.read_InflowWind()
        
        # if file_wind.split('.')[1] == 'wnd':
        #     self.WndWindReader(file_wind)
        # else:
        #     print 'Wind reader for file type .%s not implemented yet.' % file_wind.split('.')[1]
        # AeroDyn version selection
        if self.fst_vt['Fst']['CompAero'] == 1:
            self.read_AeroDyn14()
        elif self.fst_vt['Fst']['CompAero'] == 2:
            self.read_AeroDyn15()

        self.read_ServoDyn()
        self.read_DISCON_in()
        
        ROSCO_utilities.FileProcessing()
        pitch_vector, tsr_vector, Cp_table, Ct_table, Cq_table = ROSCO_utilities.FileProcessing.load_from_txt(self.fst_vt['DISCON_in']['PerfFileName'])

        RotorPerformance = ROSCO_turbine.RotorPerformance
        Cp = RotorPerformance(Cp_table,pitch_vector,tsr_vector)
        Ct = RotorPerformance(Ct_table,pitch_vector,tsr_vector)
        Cq = RotorPerformance(Cq_table,pitch_vector,tsr_vector)
        
        self.fst_vt['DISCON_in']['Cp'] = Cp
        self.fst_vt['DISCON_in']['Ct'] = Ct
        self.fst_vt['DISCON_in']['Cq'] = Cq
        self.fst_vt['DISCON_in']['Cp_pitch_initial_rad'] = pitch_vector
        self.fst_vt['DISCON_in']['Cp_TSR_initial'] = tsr_vector
        self.fst_vt['DISCON_in']['Cp_table'] = Cp_table
        self.fst_vt['DISCON_in']['Ct_table'] = Ct_table
        self.fst_vt['DISCON_in']['Cq_table'] = Cq_table
        
        if self.fst_vt['Fst']['CompHydro'] == 1: # SubDyn not yet implimented
            self.read_HydroDyn()
        if self.fst_vt['Fst']['CompSub'] == 1: # SubDyn not yet implimented
            self.read_SubDyn()
        if self.fst_vt['Fst']['CompMooring'] == 1: # only MAP++ implimented for mooring models
            self.read_MAP()

        if self.fst_vt['Fst']['CompElast'] == 2: # BeamDyn read assumes all 3 blades are the same
            self.read_BeamDyn()

    def read_MainInput(self):
        # Main FAST v8.16-v8.17 Input File
        # Currently no differences between FASTv8.16 and OpenFAST.
        fst_file = os.path.join(self.FAST_directory, self.FAST_InputFile)
        f = open(fst_file)

        # Header of .fst file
        f.readline()
        self.fst_vt['description'] = f.readline().rstrip()

        # Simulation Control (fst_sim_ctrl)
        f.readline()
        self.fst_vt['Fst']['Echo'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst']['AbortLevel'] = f.readline().split()[0][1:-1]
        self.fst_vt['Fst']['TMax'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst']['DT']  = float_read(f.readline().split()[0])
        self.fst_vt['Fst']['InterpOrder']  = int(f.readline().split()[0])
        self.fst_vt['Fst']['NumCrctn']  = int(f.readline().split()[0])
        self.fst_vt['Fst']['DT_UJac']  = float_read(f.readline().split()[0])
        self.fst_vt['Fst']['UJacSclFact']  = float_read(f.readline().split()[0])

        # Feature Switches and Flags (ftr_swtchs_flgs)
        f.readline()
        self.fst_vt['Fst']['CompElast'] = int(f.readline().split()[0])
        self.fst_vt['Fst']['CompInflow'] = int(f.readline().split()[0])
        self.fst_vt['Fst']['CompAero'] = int(f.readline().split()[0])
        self.fst_vt['Fst']['CompServo'] = int(f.readline().split()[0])
        self.fst_vt['Fst']['CompHydro'] = int(f.readline().split()[0])
        self.fst_vt['Fst']['CompSub'] = int(f.readline().split()[0])
        self.fst_vt['Fst']['CompMooring'] = int(f.readline().split()[0])
        self.fst_vt['Fst']['CompIce'] = int(f.readline().split()[0])

        # Input Files (input_files)
        f.readline()
        self.fst_vt['Fst']['EDFile'] = f.readline().split()[0][1:-1]
        self.fst_vt['Fst']['BDBldFile(1)'] = f.readline().split()[0][1:-1]
        self.fst_vt['Fst']['BDBldFile(2)'] = f.readline().split()[0][1:-1]
        self.fst_vt['Fst']['BDBldFile(3)'] = f.readline().split()[0][1:-1]
        self.fst_vt['Fst']['InflowFile'] = f.readline().split()[0][1:-1]
        self.fst_vt['Fst']['AeroFile'] = f.readline().split()[0][1:-1]
        self.fst_vt['Fst']['ServoFile'] = f.readline().split()[0][1:-1]
        self.fst_vt['Fst']['HydroFile'] = f.readline().split()[0][1:-1]
        self.fst_vt['Fst']['SubFile'] = f.readline().split()[0][1:-1]
        self.fst_vt['Fst']['MooringFile'] = f.readline().split()[0][1:-1]
        self.fst_vt['Fst']['IceFile'] = f.readline().split()[0][1:-1]

        # FAST Output Parameters (fst_output_params)
        f.readline()
        self.fst_vt['Fst']['SumPrint'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst']['SttsTime'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst']['ChkptTime'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst']['DT_Out'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst']['TStart'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst']['OutFileFmt'] = int(f.readline().split()[0])
        self.fst_vt['Fst']['TabDelim'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst']['OutFmt'] = f.readline().split()[0][1:-1]

        # Fst
        f.readline()
        self.fst_vt['Fst']['linearize'] = f.readline().split()[0]
        self.fst_vt['Fst']['NLinTimes'] = f.readline().split()[0]
        self.fst_vt['Fst']['LinTimes'] = re.findall(r'[^,\s]+', f.readline())[0:2]
        self.fst_vt['Fst']['LinInputs'] = f.readline().split()[0]
        self.fst_vt['Fst']['LinOutputs'] = f.readline().split()[0]
        self.fst_vt['Fst']['LinOutJac'] = f.readline().split()[0]
        self.fst_vt['Fst']['LinOutMod'] = f.readline().split()[0]

        # Visualization ()
        f.readline()
        self.fst_vt['Fst']['WrVTK'] = int(f.readline().split()[0])
        self.fst_vt['Fst']['VTK_type'] = int(f.readline().split()[0])
        self.fst_vt['Fst']['VTK_fields'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst']['VTK_fps'] = float_read(f.readline().split()[0])
        
        f.close()
        
    def read_ElastoDyn(self):
        # ElastoDyn v1.03 Input File
        # Currently no differences between FASTv8.16 and OpenFAST.

        ed_file = os.path.join(self.FAST_directory, self.fst_vt['Fst']['EDFile'])
        f = open(ed_file)

        f.readline()
        f.readline()

        # Simulation Control (ed_sim_ctrl)
        f.readline()
        self.fst_vt['ElastoDyn']['Echo'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['Method']  = int(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['DT'] = float_read(f.readline().split()[0])

        # Environmental Condition (envir_cond)
        f.readline()
        self.fst_vt['ElastoDyn']['Gravity'] = float_read(f.readline().split()[0])

        # Degrees of Freedom (dof)
        f.readline()
        self.fst_vt['ElastoDyn']['FlapDOF1'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['FlapDOF2'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['EdgeDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TeetDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['DrTrDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['GenDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['YawDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TwFADOF1'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TwFADOF2'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TwSSDOF1'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TwSSDOF2'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmSgDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmSwDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmHvDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmRDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmPDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmYDOF'] = bool_read(f.readline().split()[0])

        # Initial Conditions (init_conds)
        f.readline()
        self.fst_vt['ElastoDyn']['OoPDefl']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['IPDefl']     = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['BlPitch1']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['BlPitch2']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['BlPitch3']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TeetDefl']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['Azimuth']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['RotSpeed']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['NacYaw']     = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TTDspFA']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TTDspSS']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmSurge']  = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmSway']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmHeave']  = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmRoll']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmPitch']  = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmYaw']    = float_read(f.readline().split()[0])


        # Turbine Configuration (turb_config)
        f.readline()
        self.fst_vt['ElastoDyn']['NumBl']      = int(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TipRad']     = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['HubRad']     = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PreCone(1)']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PreCone(2)']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PreCone(3)']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['HubCM']      = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['UndSling']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['Delta3']     = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['AzimB1Up']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['OverHang']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['ShftGagL']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['ShftTilt']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['NacCMxn']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['NacCMyn']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['NacCMzn']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['NcIMUxn']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['NcIMUyn']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['NcIMUzn']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['Twr2Shft']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TowerHt']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TowerBsHt']  = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmCMxt']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmCMyt']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmCMzt']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmRefzt']  = float_read(f.readline().split()[0])

        # Mass and Inertia (mass_inertia)
        f.readline()
        self.fst_vt['ElastoDyn']['TipMass(1)']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TipMass(2)']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TipMass(3)']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['HubMass']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['HubIner']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['GenIner']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['NacMass']    = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['NacYIner']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['YawBrMass']  = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmMass']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmRIner']  = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmPIner']  = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['PtfmYIner']  = float_read(f.readline().split()[0])

        # ElastoDyn Blade (blade_struc)
        f.readline()
        self.fst_vt['ElastoDyn']['BldNodes'] = int(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['BldFile1'] = f.readline().split()[0][1:-1]
        self.fst_vt['ElastoDyn']['BldFile2'] = f.readline().split()[0][1:-1]
        self.fst_vt['ElastoDyn']['BldFile3'] = f.readline().split()[0][1:-1]

        # Rotor-Teeter (rotor_teeter)
        f.readline()
        self.fst_vt['ElastoDyn']['TeetMod']  = int(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TeetDmpP'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TeetDmp']  = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TeetCDmp'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TeetSStP'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TeetHStP'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TeetSSSp'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TeetHSSp'] = float_read(f.readline().split()[0])

        # Drivetrain (drivetrain)
        f.readline()
        self.fst_vt['ElastoDyn']['GBoxEff']  = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['GBRatio']  = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['DTTorSpr'] = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['DTTorDmp'] = float_read(f.readline().split()[0])

        # Furling (furling)
        f.readline()
        self.fst_vt['ElastoDyn']['Furling'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['FurlFile'] = f.readline().split()[0][1:-1]

        # Tower (tower)
        f.readline()
        self.fst_vt['ElastoDyn']['TwrNodes'] = int(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TwrFile'] = f.readline().split()[0][1:-1]

        # ED Output Parameters (ed_out_params)
        f.readline()
        self.fst_vt['ElastoDyn']['SumPrint'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['OutFile']  = int(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['TabDelim'] = bool_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['OutFmt']   = f.readline().split()[0][1:-1]
        self.fst_vt['ElastoDyn']['TStart']   = float_read(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['DecFact']  = int(f.readline().split()[0])
        self.fst_vt['ElastoDyn']['NTwGages'] = int(f.readline().split()[0])
        twrg = f.readline().split(',')
        if self.fst_vt['ElastoDyn']['NTwGages'] != 0: #loop over elements if there are gauges to be added, otherwise assign directly
            for i in range(self.fst_vt['ElastoDyn']['NTwGages']):
                self.fst_vt['ElastoDyn']['TwrGagNd'].append(twrg[i])
            self.fst_vt['ElastoDyn']['TwrGagNd'][-1]  = self.fst_vt['ElastoDyn']['TwrGagNd'][-1][:-1]   #remove last (newline) character
        else:
            self.fst_vt['ElastoDyn']['TwrGagNd'] = twrg
            self.fst_vt['ElastoDyn']['TwrGagNd'][-1]  = self.fst_vt['ElastoDyn']['TwrGagNd'][-1][:-1]
        self.fst_vt['ElastoDyn']['NBlGages'] = int(f.readline().split()[0])
        blg = f.readline().split(',')
        if self.fst_vt['ElastoDyn']['NBlGages'] != 0:
            for i in range(self.fst_vt['ElastoDyn']['NBlGages']):
                self.fst_vt['ElastoDyn']['BldGagNd'].append(blg[i])
            self.fst_vt['ElastoDyn']['BldGagNd'][-1]  = self.fst_vt['ElastoDyn']['BldGagNd'][-1][:-1]
        else:
            self.fst_vt['ElastoDyn']['BldGagNd'] = blg
            self.fst_vt['ElastoDyn']['BldGagNd'][-1]  = self.fst_vt['ElastoDyn']['BldGagNd'][-1][:-1]

        # Loop through output channel lines
        f.readline()
        data = f.readline()
        while data.split()[0] != 'END':
            channels = data.split('"')
            channel_list = channels[1].split(',')
            self.set_outlist(self.fst_vt['outlist']['ElastoDyn'], channel_list)

            data = f.readline()

        f.close()

    def read_BeamDyn(self):
        # BeamDyn Input File

        bd_file = os.path.join(self.FAST_directory, self.fst_vt['Fst']['BDBldFile(1)'])
        f = open(bd_file)

        f.readline()
        f.readline()
        f.readline()
        # ---------------------- SIMULATION CONTROL --------------------------------------
        self.fst_vt['BeamDyn']['Echo']             = bool_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['QuasiStaticInit']  = bool_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['rhoinf']           = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['quadrature']       = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['refine']           = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['n_fact']           = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['DTBeam']            = float_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['load_retries']     = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['NRMax']            = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['stop_tol']         = float_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['tngt_stf_fd']      = bool_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['tngt_stf_comp']    = bool_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['tngt_stf_pert']    = float_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['tngt_stf_difftol'] = float_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['RotStates']        = bool_read(f.readline().split()[0])
        f.readline()
        #---------------------- GEOMETRY PARAMETER --------------------------------------
        self.fst_vt['BeamDyn']['member_total']     = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['kp_total']         = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['members']          = []
        for i in range(self.fst_vt['BeamDyn']['member_total']):
            ln = f.readline().split()
            n_pts_i                   = int(ln[1])
            member_i                  = {}
            member_i['kp_xr']         = [None]*n_pts_i
            member_i['kp_yr']         = [None]*n_pts_i
            member_i['kp_zr']         = [None]*n_pts_i
            member_i['initial_twist'] = [None]*n_pts_i
            f.readline()
            f.readline()
            for j in range(n_pts_i):
                ln = f.readline().split()
                member_i['kp_xr'][j]          = float(ln[0])
                member_i['kp_yr'][j]          = float(ln[1])
                member_i['kp_zr'][j]          = float(ln[2])
                member_i['initial_twist'][j]  = float(ln[3])

            self.fst_vt['BeamDyn']['members'].append(member_i)
        #---------------------- MESH PARAMETER ------------------------------------------
        f.readline()
        self.fst_vt['BeamDyn']['order_elem']  = int_read(f.readline().split()[0])
        #---------------------- MATERIAL PARAMETER --------------------------------------
        f.readline()
        self.fst_vt['BeamDyn']['BldFile']     = f.readline().split()[0].replace('"','').replace("'",'')
        #---------------------- PITCH ACTUATOR PARAMETERS -------------------------------
        f.readline()
        self.fst_vt['BeamDyn']['UsePitchAct'] = bool_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['PitchJ']      = float_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['PitchK']      = float_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['PitchC']      = float_read(f.readline().split()[0])
        #---------------------- OUTPUTS -------------------------------------------------
        f.readline()
        self.fst_vt['BeamDyn']['SumPrint']    = bool_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['OutFmt']      = f.readline().split()[0]
        self.fst_vt['BeamDyn']['NNodeOuts']   = int_read(f.readline().split()[0])
        self.fst_vt['BeamDyn']['OutNd']       = [idx.strip() for idx in f.readline().split('NNodeOuts')[0].split(',')]
        # BeamDyn Outlist
        f.readline()
        data = f.readline()
        while data.split()[0] != 'END':
            channels = data.split('"')
            channel_list = channels[1].split(',')
            self.set_outlist(self.fst_vt['outlist']['BeamDyn'], channel_list)
            data = f.readline()

        f.close()

        self.read_BeamDynBlade()


    def read_BeamDynBlade(self):
        # BeamDyn Blade

        beamdyn_blade_file = os.path.join(self.FAST_directory, self.fst_vt['BeamDyn']['BldFile'])
        f = open(beamdyn_blade_file)
        
        f.readline()
        f.readline()
        f.readline()
        #---------------------- BLADE PARAMETERS --------------------------------------
        self.fst_vt['BeamDynBlade']['station_total'] = int_read(f.readline().split()[0])
        self.fst_vt['BeamDynBlade']['damp_type']     = int_read(f.readline().split()[0])
        f.readline()
        f.readline()
        f.readline()
        #---------------------- DAMPING COEFFICIENT------------------------------------
        ln = f.readline().split()
        self.fst_vt['BeamDynBlade']['mu1']           = float(ln[0])
        self.fst_vt['BeamDynBlade']['mu2']           = float(ln[1])
        self.fst_vt['BeamDynBlade']['mu3']           = float(ln[2])
        self.fst_vt['BeamDynBlade']['mu4']           = float(ln[3])
        self.fst_vt['BeamDynBlade']['mu5']           = float(ln[4])
        self.fst_vt['BeamDynBlade']['mu6']           = float(ln[5])
        f.readline()
        #---------------------- DISTRIBUTED PROPERTIES---------------------------------
        
        self.fst_vt['BeamDynBlade']['radial_stations'] = np.zeros((self.fst_vt['BeamDynBlade']['station_total']))
        self.fst_vt['BeamDynBlade']['beam_stiff']      = np.zeros((self.fst_vt['BeamDynBlade']['station_total'], 6, 6))
        self.fst_vt['BeamDynBlade']['beam_inertia']    = np.zeros((self.fst_vt['BeamDynBlade']['station_total'], 6, 6))
        for i in range(self.fst_vt['BeamDynBlade']['station_total']):
            self.fst_vt['BeamDynBlade']['radial_stations'][i]  = float_read(f.readline().split()[0])
            for j in range(6):
                self.fst_vt['BeamDynBlade']['beam_stiff'][i,j,:] = np.array([float(val) for val in f.readline().strip().split()])
            f.readline()
            for j in range(6):
                self.fst_vt['BeamDynBlade']['beam_inertia'][i,j,:] = np.array([float(val) for val in f.readline().strip().split()])
            f.readline()

        f.close()

    def read_InflowWind(self):
        # InflowWind v3.01
        # Currently no differences between FASTv8.16 and OpenFAST.
        inflow_file = os.path.normpath(os.path.join(self.FAST_directory, self.fst_vt['Fst']['InflowFile']))
        f = open(inflow_file)
        
        f.readline()
        f.readline()
        f.readline()

        # Inflow wind header parameters (inflow_wind)
        self.fst_vt['InflowWind']['Echo']           = bool_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['WindType']       = int(f.readline().split()[0])
        self.fst_vt['InflowWind']['PropogationDir'] = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['NWindVel']       = int(f.readline().split()[0])
        self.fst_vt['InflowWind']['WindVxiList']    = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['WindVyiList']    = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['WindVziList']    = float_read(f.readline().split()[0])

        # Parameters for Steady Wind Conditions [used only for WindType = 1] (steady_wind_params)
        f.readline()
        self.fst_vt['InflowWind']['HWindSpeed'] = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['RefHt'] = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['PLexp'] = float_read(f.readline().split()[0])

        # Parameters for Uniform wind file   [used only for WindType = 2] (uniform_wind_params)
        f.readline()
        self.fst_vt['InflowWind']['Filename'] = os.path.join(os.path.split(inflow_file)[0], f.readline().split()[0][1:-1])
        self.fst_vt['InflowWind']['RefHt'] = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['RefLength'] = float_read(f.readline().split()[0])

        # Parameters for Binary TurbSim Full-Field files   [used only for WindType = 3] (turbsim_wind_params)
        f.readline()
        self.fst_vt['InflowWind']['Filename'] = os.path.join(os.path.split(inflow_file)[0], f.readline().split()[0][1:-1])

        # Parameters for Binary Bladed-style Full-Field files   [used only for WindType = 4] (bladed_wind_params)
        f.readline()
        self.fst_vt['InflowWind']['FilenameRoot'] = f.readline().split()[0][1:-1]       
        self.fst_vt['InflowWind']['TowerFile'] = bool_read(f.readline().split()[0])

        # Parameters for HAWC-format binary files  [Only used with WindType = 5] (hawc_wind_params)
        f.readline()
        self.fst_vt['InflowWind']['FileName_u'] = os.path.normpath(os.path.join(os.path.split(inflow_file)[0], f.readline().split()[0][1:-1]))
        self.fst_vt['InflowWind']['FileName_v'] = os.path.normpath(os.path.join(os.path.split(inflow_file)[0], f.readline().split()[0][1:-1]))
        self.fst_vt['InflowWind']['FileName_w'] = os.path.normpath(os.path.join(os.path.split(inflow_file)[0], f.readline().split()[0][1:-1]))
        self.fst_vt['InflowWind']['nx']    = int(f.readline().split()[0])
        self.fst_vt['InflowWind']['ny']    = int(f.readline().split()[0])
        self.fst_vt['InflowWind']['nz']    = int(f.readline().split()[0])
        self.fst_vt['InflowWind']['dx']    = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['dy']    = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['dz']    = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['RefHt'] = float_read(f.readline().split()[0])

        # Scaling parameters for turbulence (still hawc_wind_params)
        f.readline()
        self.fst_vt['InflowWind']['ScaleMethod'] = int(f.readline().split()[0])
        self.fst_vt['InflowWind']['SFx']         = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['SFy']         = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['SFz']         = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['SigmaFx']     = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['SigmaFy']     = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['SigmaFz']     = float_read(f.readline().split()[0])

        # Mean wind profile parameters (added to HAWC-format files) (still hawc_wind_params)
        f.readline()
        self.fst_vt['InflowWind']['URef']        = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['WindProfile'] = int(f.readline().split()[0])
        self.fst_vt['InflowWind']['PLExp']       = float_read(f.readline().split()[0])
        self.fst_vt['InflowWind']['Z0']          = float_read(f.readline().split()[0])

        # Inflow Wind Output Parameters (inflow_out_params)
        f.readline()
        self.fst_vt['InflowWind']['SumPrint'] = bool_read(f.readline().split()[0])
        
        # NO INFLOW WIND OUTPUT PARAMETERS YET DEFINED IN FAST
        # f.readline()
        # data = f.readline()
        # while data.split()[0] != 'END':
        #     channels = data.split('"')
        #     channel_list = channels[1].split(',')
        #     for i in range(len(channel_list)):
        #         channel_list[i] = channel_list[i].replace(' ','')
        #         if channel_list[i] in self.fst_vt.outlist.inflow_wind_vt.__dict__.keys():
        #             self.fst_vt.outlist.inflow_wind_vt.__dict__[channel_list[i]] = True
        #     data = f.readline()

        f.close()

    def read_AeroDyn14(self):
        # AeroDyn v14.04

        ad_file = os.path.join(self.FAST_directory, self.fst_vt['Fst']['AeroFile'])
        f = open(ad_file)
        # AeroDyn file header (aerodyn)
        f.readline()
        f.readline()
        self.fst_vt['AeroDyn14']['StallMod'] = f.readline().split()[0]
        self.fst_vt['AeroDyn14']['UseCm'] = f.readline().split()[0]
        self.fst_vt['AeroDyn14']['InfModel'] = f.readline().split()[0]
        self.fst_vt['AeroDyn14']['IndModel'] = f.readline().split()[0]
        self.fst_vt['AeroDyn14']['AToler'] = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['TLModel'] = f.readline().split()[0]
        self.fst_vt['AeroDyn14']['HLModel'] = f.readline().split()[0]
        self.fst_vt['AeroDyn14']['TwrShad'] = f.readline().split()[0]
        self.fst_vt['AeroDyn14']['TwrPotent'] = bool_read(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['TwrShadow'] = bool_read(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['TwrFile'] = f.readline().split()[0].replace('"','').replace("'",'')
        self.fst_vt['AeroDyn14']['CalcTwrAero'] = bool_read(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['AirDens'] = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['KinVisc'] = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['DTAero'] = float_read(f.readline().split()[0])

        # AeroDyn Blade Properties (blade_aero)
        self.fst_vt['AeroDyn14']['NumFoil'] = int(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['FoilNm'] = [None] * self.fst_vt['AeroDyn14']['NumFoil']
        for i in range(self.fst_vt['AeroDyn14']['NumFoil']):
            af_filename = f.readline().split()[0]
            af_filename = fix_path(af_filename)
            self.fst_vt['AeroDyn14']['FoilNm'][i]  = af_filename[1:-1]
        
        self.fst_vt['AeroDynBlade']['BldNodes'] = int(f.readline().split()[0])
        f.readline()
        self.fst_vt['AeroDynBlade']['RNodes'] = [None] * self.fst_vt['AeroDynBlade']['BldNodes']
        self.fst_vt['AeroDynBlade']['AeroTwst'] = [None] * self.fst_vt['AeroDynBlade']['BldNodes']
        self.fst_vt['AeroDynBlade']['DRNodes'] = [None] * self.fst_vt['AeroDynBlade']['BldNodes']
        self.fst_vt['AeroDynBlade']['Chord'] = [None] * self.fst_vt['AeroDynBlade']['BldNodes']
        self.fst_vt['AeroDynBlade']['NFoil'] = [None] * self.fst_vt['AeroDynBlade']['BldNodes']
        self.fst_vt['AeroDynBlade']['PrnElm'] = [None] * self.fst_vt['AeroDynBlade']['BldNodes']       
        for i in range(self.fst_vt['AeroDynBlade']['BldNodes']):
            data = f.readline().split()
            self.fst_vt['AeroDynBlade']['RNodes'][i]  = float_read(data[0])
            self.fst_vt['AeroDynBlade']['AeroTwst'][i]  = float_read(data[1])
            self.fst_vt['AeroDynBlade']['DRNodes'][i]  = float_read(data[2])
            self.fst_vt['AeroDynBlade']['Chord'][i]  = float_read(data[3])
            self.fst_vt['AeroDynBlade']['NFoil'][i]  = int(data[4])
            self.fst_vt['AeroDynBlade']['PrnElm'][i]  = data[5]

        f.close()

        # create airfoil objects
        self.fst_vt['AeroDynBlade']['af_data'] = []
        for i in range(self.fst_vt['AeroDynBlade']['NumFoil']):
             self.fst_vt['AeroDynBlade']['af_data'].append(self.read_AeroDyn14Polar(os.path.join(self.FAST_directory,self.fst_vt['AeroDyn14']['FoilNm'][i])))

        # tower
        self.read_AeroDyn14Tower()

    def read_AeroDyn14Tower(self):
        # AeroDyn v14.04 Tower

        ad_tower_file = os.path.join(self.FAST_directory, self.fst_vt['aerodyn']['TwrFile'])
        f = open(ad_tower_file)

        f.readline()
        f.readline()
        self.fst_vt['AeroDynTower']['NTwrHt'] = int(f.readline().split()[0])
        self.fst_vt['AeroDynTower']['NTwrRe'] = int(f.readline().split()[0])
        self.fst_vt['AeroDynTower']['NTwrCD'] = int(f.readline().split()[0])
        self.fst_vt['AeroDynTower']['Tower_Wake_Constant'] = float_read(f.readline().split()[0])
        
        f.readline()
        f.readline()
        self.fst_vt['AeroDynTower']['TwrHtFr'] = [None]*self.fst_vt['AeroDynTower']['NTwrHt']
        self.fst_vt['AeroDynTower']['TwrWid'] = [None]*self.fst_vt['AeroDynTower']['NTwrHt']
        self.fst_vt['AeroDynTower']['NTwrCDCol'] = [None]*self.fst_vt['AeroDynTower']['NTwrHt']
        for i in range(self.fst_vt['AeroDynTower']['NTwrHt']):
            data = [float(val) for val in f.readline().split()]
            self.fst_vt['AeroDynTower']['TwrHtFr'][i]  = data[0] 
            self.fst_vt['AeroDynTower']['TwrWid'][i]  = data[1]
            self.fst_vt['AeroDynTower']['NTwrCDCol'][i]  = data[2]

        f.readline()
        f.readline()
        self.fst_vt['AeroDynTower']['TwrRe'] = [None]*self.fst_vt['AeroDynTower']['NTwrRe']
        self.fst_vt['AeroDynTower']['TwrCD'] = np.zeros((self.fst_vt['AeroDynTower']['NTwrRe'], self.fst_vt['AeroDynTower']['NTwrCD']))
        for i in range(self.fst_vt['AeroDynTower']['NTwrRe']):
            data = [float(val) for val in f.readline().split()]
            self.fst_vt['AeroDynTower']['TwrRe'][i]  = data[0]
            self.fst_vt['AeroDynTower']['TwrCD'][i,:]  = data[1:]


    def read_AeroDyn15(self):
        # AeroDyn v15.03

        ad_file = os.path.join(self.FAST_directory, self.fst_vt['Fst']['AeroFile'])
        f = open(ad_file)

        # General Option
        f.readline()
        f.readline()
        f.readline()
        self.fst_vt['AeroDyn15']['Echo']          = bool_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['DTAero']        = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['WakeMod']       = int(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['AFAeroMod']     = int(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['TwrPotent']     = int(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['TwrShadow']     = bool_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['TwrAero']       = bool_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['FrozenWake']    = bool_read(f.readline().split()[0])
        if self.FAST_ver.lower() != 'fast8':
                self.fst_vt['AeroDyn15']['CavitCheck']    = bool_read(f.readline().split()[0])

        # Environmental Conditions
        f.readline()
        self.fst_vt['AeroDyn15']['AirDens']        = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['KinVisc']        = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['SpdSound']       = float_read(f.readline().split()[0])
        if self.FAST_ver.lower() != 'fast8':
            self.fst_vt['AeroDyn15']['Patm']           = float_read(f.readline().split()[0])
            self.fst_vt['AeroDyn15']['Pvap']           = float_read(f.readline().split()[0])
            self.fst_vt['AeroDyn15']['FluidDepth']     = float_read(f.readline().split()[0])

        # Blade-Element/Momentum Theory Options
        f.readline()
        self.fst_vt['AeroDyn15']['SkewMod']               = int(f.readline().split()[0])
        if self.dev_branch:
            self.fst_vt['AeroDyn15']['SkewModFactor']     = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['TipLoss']               = bool_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['HubLoss']               = bool_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['TanInd']                = bool_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['AIDrag']                = bool_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['TIDrag']                = bool_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['IndToler']              = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['MaxIter']               = int(f.readline().split()[0])

        # Dynamic Blade-Element/Momentum Theory Options 
        if self.dev_branch:
            f.readline()
            self.fst_vt['AeroDyn15']['DBEMT_Mod']          = int(f.readline().split()[0])
            self.fst_vt['AeroDyn15']['tau1_const']         = int(f.readline().split()[0])

        # Beddoes-Leishman Unsteady Airfoil Aerodynamics Options
        f.readline()
        self.fst_vt['AeroDyn15']['UAMod']                  = int(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['FLookup']                = bool_read(f.readline().split()[0])

        # Airfoil Information
        f.readline()
        self.fst_vt['AeroDyn15']['AFTabMod']         = int(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['InCol_Alfa']       = int(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['InCol_Cl']         = int(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['InCol_Cd']         = int(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['InCol_Cm']         = int(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['InCol_Cpmin']      = int(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['NumAFfiles']       = int(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['AFNames']          = [None] * self.fst_vt['AeroDyn15']['NumAFfiles']
        for i in range(self.fst_vt['AeroDyn15']['NumAFfiles']):
            af_filename = fix_path(f.readline().split()[0])[1:-1]
            self.fst_vt['AeroDyn15']['AFNames'][i]   = os.path.abspath(os.path.join(self.FAST_directory, af_filename))

        # Rotor/Blade Properties
        f.readline()
        self.fst_vt['AeroDyn15']['UseBlCm']        = bool_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['ADBlFile1']      = f.readline().split()[0][1:-1]
        self.fst_vt['AeroDyn15']['ADBlFile2']      = f.readline().split()[0][1:-1]
        self.fst_vt['AeroDyn15']['ADBlFile3']      = f.readline().split()[0][1:-1]

        # Tower Influence and Aerodynamics
        f.readline()
        self.fst_vt['AeroDyn15']['NumTwrNds']      = int(f.readline().split()[0])
        f.readline()
        f.readline()
        self.fst_vt['AeroDyn15']['TwrElev']        = [None]*self.fst_vt['AeroDyn15']['NumTwrNds']
        self.fst_vt['AeroDyn15']['TwrDiam']        = [None]*self.fst_vt['AeroDyn15']['NumTwrNds']
        self.fst_vt['AeroDyn15']['TwrCd']          = [None]*self.fst_vt['AeroDyn15']['NumTwrNds']
        for i in range(self.fst_vt['AeroDyn15']['NumTwrNds']):
            data = [float(val) for val in f.readline().split()]
            self.fst_vt['AeroDyn15']['TwrElev'][i] = data[0] 
            self.fst_vt['AeroDyn15']['TwrDiam'][i] = data[1] 
            self.fst_vt['AeroDyn15']['TwrCd'][i]   = data[2]

        # Outputs
        f.readline()
        self.fst_vt['AeroDyn15']['SumPrint']    = bool_read(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['NBlOuts']     = int(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['BlOutNd']     = [idx.strip() for idx in f.readline().split('BlOutNd')[0].split(',')]
        self.fst_vt['AeroDyn15']['NTwOuts']     = int(f.readline().split()[0])
        self.fst_vt['AeroDyn15']['TwOutNd']     = [idx.strip() for idx in f.readline().split('TwOutNd')[0].split(',')]

        # AeroDyn15 Outlist
        f.readline()
        data = f.readline()
        while data.split()[0] != 'END':
            channels = data.split('"')
            channel_list = channels[1].split(',')
            self.set_outlist(self.fst_vt['outlist']['AeroDyn'], channel_list)
            data = f.readline()

        f.close()

        self.read_AeroDyn15Blade()
        self.read_AeroDyn15Polar()

    def read_AeroDyn15Blade(self):
        # AeroDyn v5.00 Blade Definition File

        ad_blade_file = os.path.join(self.FAST_directory, self.fst_vt['AeroDyn15']['ADBlFile1'])
        f = open(ad_blade_file)

        f.readline()
        f.readline()
        f.readline()
        # Blade Properties
        self.fst_vt['AeroDynBlade']['NumBlNds']       = int(f.readline().split()[0])
        f.readline()
        f.readline()
        self.fst_vt['AeroDynBlade']['BlSpn']          = [None]*self.fst_vt['AeroDynBlade']['NumBlNds']
        self.fst_vt['AeroDynBlade']['BlCrvAC']        = [None]*self.fst_vt['AeroDynBlade']['NumBlNds']
        self.fst_vt['AeroDynBlade']['BlSwpAC']        = [None]*self.fst_vt['AeroDynBlade']['NumBlNds']
        self.fst_vt['AeroDynBlade']['BlCrvAng']       = [None]*self.fst_vt['AeroDynBlade']['NumBlNds']
        self.fst_vt['AeroDynBlade']['BlTwist']        = [None]*self.fst_vt['AeroDynBlade']['NumBlNds']
        self.fst_vt['AeroDynBlade']['BlChord']        = [None]*self.fst_vt['AeroDynBlade']['NumBlNds']
        self.fst_vt['AeroDynBlade']['BlAFID']         = [None]*self.fst_vt['AeroDynBlade']['NumBlNds']
        for i in range(self.fst_vt['AeroDynBlade']['NumBlNds']):
            data = [float(val) for val in f.readline().split()]
            self.fst_vt['AeroDynBlade']['BlSpn'][i]   = data[0] 
            self.fst_vt['AeroDynBlade']['BlCrvAC'][i] = data[1] 
            self.fst_vt['AeroDynBlade']['BlSwpAC'][i] = data[2]
            self.fst_vt['AeroDynBlade']['BlCrvAng'][i]= data[3]
            self.fst_vt['AeroDynBlade']['BlTwist'][i] = data[4]
            self.fst_vt['AeroDynBlade']['BlChord'][i] = data[5]
            self.fst_vt['AeroDynBlade']['BlAFID'][i]  = data[6]
        
        f.close()

    def read_AeroDyn15Polar(self):
        # AirfoilInfo v1.01

        def readline_filterComments(f):
            read = True
            while read:
                line = f.readline().strip()
                if len(line)>0:
                    if line[0] != '!':
                        read = False
            return line


        self.fst_vt['AeroDyn15']['af_data'] = [None]*self.fst_vt['AeroDyn15']['NumAFfiles']

        for afi, af_filename in enumerate(self.fst_vt['AeroDyn15']['AFNames']):
            f = open(af_filename)
            # print af_filename

            polar = {}

            polar['InterpOrd']      = int_read(readline_filterComments(f).split()[0])
            polar['NonDimArea']     = int_read(readline_filterComments(f).split()[0])
            polar['NumCoords']      = readline_filterComments(f).split()[0]
            polar['NumTabs']        = int_read(readline_filterComments(f).split()[0])
            self.fst_vt['AeroDyn15']['af_data'][afi] = [None]*polar['NumTabs']

            for tab in range(polar['NumTabs']): # For multiple tables
                polar['Re']             = float_read(readline_filterComments(f).split()[0])
                polar['Ctrl']           = int_read(readline_filterComments(f).split()[0])
                polar['InclUAdata']     = bool_read(readline_filterComments(f).split()[0])

                # Unsteady Aero Data
                if polar['InclUAdata']:
                    polar['alpha0']     = float_read(readline_filterComments(f).split()[0])
                    polar['alpha1']     = float_read(readline_filterComments(f).split()[0])
                    polar['alpha2']     = float_read(readline_filterComments(f).split()[0])
                    polar['eta_e']      = float_read(readline_filterComments(f).split()[0])
                    polar['C_nalpha']   = float_read(readline_filterComments(f).split()[0])
                    polar['T_f0']       = float_read(readline_filterComments(f).split()[0])
                    polar['T_V0']       = float_read(readline_filterComments(f).split()[0])
                    polar['T_p']        = float_read(readline_filterComments(f).split()[0])
                    polar['T_VL']       = float_read(readline_filterComments(f).split()[0])
                    polar['b1']         = float_read(readline_filterComments(f).split()[0])
                    polar['b2']         = float_read(readline_filterComments(f).split()[0])
                    polar['b5']         = float_read(readline_filterComments(f).split()[0])
                    polar['A1']         = float_read(readline_filterComments(f).split()[0])
                    polar['A2']         = float_read(readline_filterComments(f).split()[0])
                    polar['A5']         = float_read(readline_filterComments(f).split()[0])
                    polar['S1']         = float_read(readline_filterComments(f).split()[0])
                    polar['S2']         = float_read(readline_filterComments(f).split()[0])
                    polar['S3']         = float_read(readline_filterComments(f).split()[0])
                    polar['S4']         = float_read(readline_filterComments(f).split()[0])
                    polar['Cn1']        = float_read(readline_filterComments(f).split()[0])
                    polar['Cn2']        = float_read(readline_filterComments(f).split()[0])
                    polar['St_sh']      = float_read(readline_filterComments(f).split()[0])
                    polar['Cd0']        = float_read(readline_filterComments(f).split()[0])
                    polar['Cm0']        = float_read(readline_filterComments(f).split()[0])
                    polar['k0']         = float_read(readline_filterComments(f).split()[0])
                    polar['k1']         = float_read(readline_filterComments(f).split()[0])
                    polar['k2']         = float_read(readline_filterComments(f).split()[0])
                    polar['k3']         = float_read(readline_filterComments(f).split()[0])
                    polar['k1_hat']     = float_read(readline_filterComments(f).split()[0])
                    polar['x_cp_bar']   = float_read(readline_filterComments(f).split()[0])
                    polar['UACutout']   = float_read(readline_filterComments(f).split()[0])
                    polar['filtCutOff'] = float_read(readline_filterComments(f).split()[0])

                # Polar Data
                polar['NumAlf']         = int_read(readline_filterComments(f).split()[0])
                polar['Alpha']          = [None]*polar['NumAlf']
                polar['Cl']             = [None]*polar['NumAlf']
                polar['Cd']             = [None]*polar['NumAlf']
                polar['Cm']             = [None]*polar['NumAlf']
                polar['Cpmin']          = [None]*polar['NumAlf']
                for i in range(polar['NumAlf']):
                    data = [float(val) for val in readline_filterComments(f).split()]
                    if self.fst_vt['AeroDyn15']['InCol_Alfa'] > 0:
                        polar['Alpha'][i] = data[self.fst_vt['AeroDyn15']['InCol_Alfa']-1]
                    if self.fst_vt['AeroDyn15']['InCol_Cl'] > 0:
                        polar['Cl'][i]    = data[self.fst_vt['AeroDyn15']['InCol_Cl']-1]
                    if self.fst_vt['AeroDyn15']['InCol_Cd'] > 0:
                        polar['Cd'][i]    = data[self.fst_vt['AeroDyn15']['InCol_Cd']-1]
                    if self.fst_vt['AeroDyn15']['InCol_Cm'] > 0:
                        polar['Cm'][i]    = data[self.fst_vt['AeroDyn15']['InCol_Cm']-1]
                    if self.fst_vt['AeroDyn15']['InCol_Cpmin'] > 0:
                        polar['Cpmin'][i] = data[self.fst_vt['AeroDyn15']['InCol_Cpmin']-1]

                self.fst_vt['AeroDyn15']['af_data'][afi][tab] = copy.copy(polar) # For multiple tables
            
            f.close()

    def read_ServoDyn(self):
        # ServoDyn v1.05 Input File
        # Currently no differences between FASTv8.16 and OpenFAST.


        sd_file = os.path.normpath(os.path.join(self.FAST_directory, self.fst_vt['Fst']['ServoFile']))
        f = open(sd_file)

        f.readline()
        f.readline()

        # Simulation Control (sd_sim_ctrl)
        f.readline()
        self.fst_vt['ServoDyn']['Echo'] = bool_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['DT'] = float_read(f.readline().split()[0])

        # Pitch Control (pitch_ctrl)
        f.readline()
        self.fst_vt['ServoDyn']['PCMode']       = int(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TPCOn']        = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TPitManS1']    = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TPitManS2']    = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TPitManS3']    = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['PitManRat1']   = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['PitManRat2']   = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['PitManRat3']   = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['BlPitchF1']    = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['BlPitchF2']    = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['BlPitchF3']    = float_read(f.readline().split()[0])

        # Geneartor and Torque Control (gen_torq_ctrl)
        f.readline()
        self.fst_vt['ServoDyn']['VSContrl'] = int(f.readline().split()[0])
        self.fst_vt['ServoDyn']['GenModel'] = int(f.readline().split()[0])
        self.fst_vt['ServoDyn']['GenEff']   = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['GenTiStr'] = bool_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['GenTiStp'] = bool_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['SpdGenOn'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TimGenOn'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TimGenOf'] = float_read(f.readline().split()[0])

        # Simple Variable-Speed Torque Control (var_speed_torq_ctrl)
        f.readline()
        self.fst_vt['ServoDyn']['VS_RtGnSp'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['VS_RtTq']   = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['VS_Rgn2K']  = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['VS_SlPc']   = float_read(f.readline().split()[0])

        # Simple Induction Generator (induct_gen)
        f.readline()
        self.fst_vt['ServoDyn']['SIG_SlPc'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['SIG_SySp'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['SIG_RtTq'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['SIG_PORt'] = float_read(f.readline().split()[0])

        # Thevenin-Equivalent Induction Generator (theveq_induct_gen)
        f.readline()
        self.fst_vt['ServoDyn']['TEC_Freq'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TEC_NPol'] = int(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TEC_SRes'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TEC_RRes'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TEC_VLL']  = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TEC_SLR']  = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TEC_RLR']  = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TEC_MR']   = float_read(f.readline().split()[0])

        # High-Speed Shaft Brake (shaft_brake)
        f.readline()
        self.fst_vt['ServoDyn']['HSSBrMode'] = int(f.readline().split()[0])
        self.fst_vt['ServoDyn']['THSSBrDp']  = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['HSSBrDT']   = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['HSSBrTqF']  = float_read(f.readline().split()[0])

        # Nacelle-Yaw Control (nac_yaw_ctrl)
        f.readline()
        self.fst_vt['ServoDyn']['YCMode']    = int(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TYCOn']     = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['YawNeut']   = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['YawSpr']    = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['YawDamp']   = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TYawManS']  = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['YawManRat'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['NacYawF']   = float_read(f.readline().split()[0])

        # Tuned Mass Damper (tuned_mass_damper)
        f.readline()
        self.fst_vt['ServoDyn']['CompNTMD'] = bool_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['NTMDfile'] = f.readline().split()[0][1:-1]
        self.fst_vt['ServoDyn']['CompTTMD'] = bool_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TTMDfile'] = f.readline().split()[0][1:-1]

        # Bladed Interface and Torque-Speed Look-Up Table (bladed_interface)
        f.readline()
        if self.path2dll == '' or self.path2dll == None:
            self.fst_vt['ServoDyn']['DLL_FileName'] = os.path.abspath(os.path.normpath(os.path.join(os.path.split(sd_file)[0], f.readline().split()[0][1:-1])))
        else:
            f.readline()
            self.fst_vt['ServoDyn']['DLL_FileName'] = self.path2dll
        self.fst_vt['ServoDyn']['DLL_InFile']   = f.readline().split()[0][1:-1]
        self.fst_vt['ServoDyn']['DLL_ProcName'] = f.readline().split()[0][1:-1]
        dll_dt_line = f.readline().split()[0]
        try:
            self.fst_vt['ServoDyn']['DLL_DT'] = float_read(dll_dt_line)
        except:
            self.fst_vt['ServoDyn']['DLL_DT'] = dll_dt_line[1:-1]
        self.fst_vt['ServoDyn']['DLL_Ramp']     = bool_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['BPCutoff']     = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['NacYaw_North'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['Ptch_Cntrl']   = int(f.readline().split()[0])
        self.fst_vt['ServoDyn']['Ptch_SetPnt']  = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['Ptch_Min']     = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['Ptch_Max']     = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['PtchRate_Min'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['PtchRate_Max'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['Gain_OM']      = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['GenSpd_MinOM'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['GenSpd_MaxOM'] = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['GenSpd_Dem']   = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['GenTrq_Dem']   = float_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['GenPwr_Dem']   = float_read(f.readline().split()[0])

        f.readline()

        self.fst_vt['ServoDyn']['DLL_NumTrq'] = int(f.readline().split()[0])
        f.readline()
        f.readline()
        self.fst_vt['ServoDyn']['GenSpd_TLU'] = [None] * self.fst_vt['ServoDyn']['DLL_NumTrq']
        self.fst_vt['ServoDyn']['GenTrq_TLU'] = [None] * self.fst_vt['ServoDyn']['DLL_NumTrq']
        for i in range(self.fst_vt['ServoDyn']['DLL_NumTrq']):
            data = f.readline().split()
            self.fst_vt['ServoDyn']['GenSpd_TLU'][i]  = float_read(data[0])
            self.fst_vt['ServoDyn']['GenTrq_TLU'][i]  = float_read(data[0])

        # ServoDyn Output Params (sd_out_params)
        f.readline()
        self.fst_vt['ServoDyn']['SumPrint'] = bool_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['OutFile']  = int(f.readline().split()[0])
        self.fst_vt['ServoDyn']['TabDelim'] = bool_read(f.readline().split()[0])
        self.fst_vt['ServoDyn']['OutFmt']   = f.readline().split()[0][1:-1]
        self.fst_vt['ServoDyn']['TStart']   = float_read(f.readline().split()[0])

        # ServoDyn Outlist
        f.readline()
        data = f.readline()
        while data.split()[0] != 'END':
            channels = data.split('"')
            channel_list = channels[1].split(',')
            self.set_outlist(self.fst_vt['outlist']['ServoDyn'], channel_list)
            data = f.readline()

        f.close()

    def read_DISCON_in(self):
        # Bladed style Interface controller input file, intended for ROSCO https://github.com/NREL/ROSCO_toolbox
        # file version for NREL Reference OpenSource Controller tuning logic on 11/01/19

        discon_in_file = os.path.normpath(os.path.join(self.FAST_directory, self.fst_vt['ServoDyn']['DLL_InFile']))

        if os.path.exists(discon_in_file):

            f = open(discon_in_file)

            f.readline()
            f.readline()
            f.readline()
            f.readline()
            # DEBUG
            self.fst_vt['DISCON_in']['LoggingLevel']      = int_read(f.readline().split()[0])
            f.readline()
            f.readline()

            # CONTROLLER FLAGS
            self.fst_vt['DISCON_in']['F_LPFType']         = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['F_NotchType']       = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['IPC_ControlMode']   = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['VS_ControlMode']    = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['PC_ControlMode']    = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['Y_ControlMode']     = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['SS_Mode']           = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['WE_Mode']           = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['PS_Mode']           = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['SD_Mode']           = int_read(f.readline().split()[0])

            # Error handling for different commits of Rosco that added features/lines.  This can probably be removed in the future
            ln1 = f.readline().split()
            Fl_Mode = False
            if len(ln1) >= 2:
                if ln1[2] == 'Fl_Mode':
                    Fl_Mode = True

            ln2 = f.readline().split()
            Flp_Mode = False
            if len(ln2) >= 2:
                if ln2[2] == 'Flp_Mode':
                    Flp_Mode = True

            if Fl_Mode:
                self.fst_vt['DISCON_in']['Fl_Mode']       = int_read(f.readline().split()[0])
                f.readline()
            else:
                self.fst_vt['DISCON_in']['Fl_Mode']       = 0
            if Flp_Mode:
                self.fst_vt['DISCON_in']['Flp_Mode']      = int_read(f.readline().split()[0])
                f.readline()
            else:
                self.fst_vt['DISCON_in']['Flp_Mode']      = 0

            # FILTERS
            self.fst_vt['DISCON_in']['F_LPFCornerFreq']   = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['F_LPFDamping']      = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['F_NotchCornerFreq'] = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['F_NotchBetaNumDen'] = [float(idx.strip()) for idx in f.readline().strip().split('F_NotchBetaNumDen')[0].split() if idx.strip() != '!']
            self.fst_vt['DISCON_in']['F_SSCornerFreq']    = float_read(f.readline().split()[0])
            if Fl_Mode:
                self.fst_vt['DISCON_in']['F_FlCornerFreq']  = [float(idx.strip()) for idx in f.readline().strip().split('F_FlCornerFreq')[0].split() if idx.strip() != '!']
            else:
                self.fst_vt['DISCON_in']['F_FlCornerFreq']  = 0.
            if Flp_Mode:
                self.fst_vt['DISCON_in']['F_FlpCornerFreq'] = [float(idx.strip()) for idx in f.readline().strip().split('F_FlpCornerFreq')[0].split() if idx.strip() != '!']
            else:
                self.fst_vt['DISCON_in']['F_FlpCornerFreq'] = 0.
            f.readline()
            f.readline()

            # BLADE PITCH CONTROL
            self.fst_vt['DISCON_in']['PC_GS_n']           = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['PC_GS_angles']      = [float(idx.strip()) for idx in f.readline().split('PC_GS_angles')[0].split() if idx.strip() != '!']
            self.fst_vt['DISCON_in']['PC_GS_KP']          = [float(idx.strip()) for idx in f.readline().split('PC_GS_KP')[0].split() if idx.strip() != '!']
            self.fst_vt['DISCON_in']['PC_GS_KI']          = [float(idx.strip()) for idx in f.readline().split('PC_GS_KI')[0].split() if idx.strip() != '!']
            self.fst_vt['DISCON_in']['PC_GS_KD']          = [float(idx.strip()) for idx in f.readline().split('PC_GS_KD')[0].split() if idx.strip() != '!']
            self.fst_vt['DISCON_in']['PC_GS_TF']          = [float(idx.strip()) for idx in f.readline().split('PC_GS_TF')[0].split() if idx.strip() != '!']
            self.fst_vt['DISCON_in']['PC_MaxPit']         = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['PC_MinPit']         = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['PC_MaxRat']         = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['PC_MinRat']         = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['PC_RefSpd']         = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['PC_FinePit']        = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['PC_Switch']         = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['Z_EnableSine']      = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['Z_PitchAmplitude']  = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['Z_PitchFrequency']  = float_read(f.readline().split()[0])
            f.readline()
            f.readline()

            # INDIVIDUAL PITCH CONTROL
            self.fst_vt['DISCON_in']['IPC_IntSat']        = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['IPC_KI']            = [float(idx.strip()) for idx in f.readline().split('IPC_KI')[0].split() if idx.strip() != '!']
            self.fst_vt['DISCON_in']['IPC_aziOffset']     = [float(idx.strip()) for idx in f.readline().split('IPC_aziOffset')[0].split() if idx.strip() != '!']
            self.fst_vt['DISCON_in']['IPC_CornerFreqAct'] = float_read(f.readline().split()[0])
            f.readline()
            f.readline()

            # VS TORQUE CONTROL
            self.fst_vt['DISCON_in']['VS_GenEff']         = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['VS_ArSatTq']        = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['VS_MaxRat']         = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['VS_MaxTq']          = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['VS_MinTq']          = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['VS_MinOMSpd']       = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['VS_Rgn2K']          = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['VS_RtPwr']          = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['VS_RtTq']           = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['VS_RefSpd']         = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['VS_n']              = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['VS_KP']             = [float_read(f.readline().split()[0])]
            self.fst_vt['DISCON_in']['VS_KI']             = [float_read(f.readline().split()[0])]
            self.fst_vt['DISCON_in']['VS_TSRopt']         = float_read(f.readline().split()[0])
            f.readline()
            f.readline()

            # SETPOINT SMOOTHER
            self.fst_vt['DISCON_in']['SS_VSGain']         = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['SS_PCGain']         = float_read(f.readline().split()[0])
            f.readline()
            f.readline()

            # WIND SPEED ESTIMATOR
            self.fst_vt['DISCON_in']['WE_BladeRadius']    = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['WE_CP_n']           = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['WE_CP']             = [float(idx.strip()) for idx in f.readline().split('WE_CP')[0].split() if idx.strip() != '!']
            self.fst_vt['DISCON_in']['WE_Gamma']          = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['WE_GearboxRatio']   = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['WE_Jtot']           = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['WE_RhoAir']         = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['PerfFileName']      = os.path.abspath(os.path.join(self.FAST_directory, f.readline().split()[0][1:-1]))
            self.fst_vt['DISCON_in']['PerfTableSize']     = [int(idx.strip()) for idx in f.readline().split('PerfTableSize')[0].split() if idx.strip() != '!']
            self.fst_vt['DISCON_in']['WE_FOPoles_N']      = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['WE_FOPoles_v']      = [float(idx.strip()) for idx in f.readline().split('WE_FOPoles_v')[0].split() if idx.strip() != '!']
            self.fst_vt['DISCON_in']['WE_FOPoles']        = [float(idx.strip()) for idx in f.readline().split('WE_FOPoles')[0].split() if idx.strip() != '!']
            f.readline()
            f.readline()

            # YAW CONTROL
            self.fst_vt['DISCON_in']['Y_ErrThresh']       = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['Y_IPC_IntSat']      = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['Y_IPC_n']           = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['Y_IPC_KP']          = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['Y_IPC_KI']          = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['Y_IPC_omegaLP']     = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['Y_IPC_zetaLP']      = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['Y_MErrSet']         = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['Y_omegaLPFast']     = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['Y_omegaLPSlow']     = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['Y_Rate']            = float_read(f.readline().split()[0])
            f.readline()
            f.readline()

            # TOWER FORE-AFT DAMPING
            self.fst_vt['DISCON_in']['FA_KI']             = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['FA_HPF_CornerFreq'] = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['FA_IntSat']         = float_read(f.readline().split()[0])
            f.readline()
            f.readline()

            # MINIMUM PITCH SATURATION
            self.fst_vt['DISCON_in']['PS_BldPitchMin_N']  = int_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['PS_WindSpeeds']     = [float(idx.strip()) for idx in f.readline().split('PS_WindSpeeds')[0].split() if idx.strip() != '!']
            self.fst_vt['DISCON_in']['PS_BldPitchMin']    = [float(idx.strip()) for idx in f.readline().split('PS_BldPitchMin')[0].split() if idx.strip() != '!']
            f.readline()
            f.readline()

            # SHUTDOWN
            self.fst_vt['DISCON_in']['SD_MaxPit']         = float_read(f.readline().split()[0])
            self.fst_vt['DISCON_in']['SD_CornerFreq']     = float_read(f.readline().split()[0])
            f.readline()
            f.readline()

            if Fl_Mode:
                # FLOATING
                self.fst_vt['DISCON_in']['Fl_Kp']         = float_read(f.readline().split()[0])
                f.readline()
                f.readline()
            else:
                self.fst_vt['DISCON_in']['Fl_Kp']         = 0.

            if Flp_Mode:
                # DISTRIBUTED AERODYNAMIC CONTROL
                self.fst_vt['DISCON_in']['Flp_Angle']     = float_read(f.readline().split()[0])
                self.fst_vt['DISCON_in']['Flp_Kp']        = float_read(f.readline().split()[0])
                self.fst_vt['DISCON_in']['Flp_Ki']        = float_read(f.readline().split()[0])
            else:
                self.fst_vt['DISCON_in']['Flp_Angle']     = 0.
                self.fst_vt['DISCON_in']['Flp_Kp']        = 0.
                self.fst_vt['DISCON_in']['Flp_Ki']        = 0.

            f.close()

            self.fst_vt['DISCON_in']['v_rated'] = 1.

        else:
            del self.fst_vt['DISCON_in']

    # def read_CpSurfaces(self):
        
    #     cp_surf__in_file = self.fst_vt['DISCON_in']['PerfFileName']

    #     if os.path.exists(cp_surf__in_file):

    #         f = open(cp_surf__in_file)

    #         f.readline()
    #         f.readline()
    #         f.readline()
    #         f.readline()
    #         pitch_vector = f.readline()
    #         n_pitch = len(pitch_vector)
    #         f.readline()
    #         tsr_vector = f.readline()
    #         n_tsr = len(tsr_vector)
    #         U_vector = f.readline()
    #         n_U = len(U_vector)

    #         Cp_table = np.zeros(n_U, n_tsr, n_pitch)
    #         Ct_table = np.zeros(n_U, n_tsr, n_pitch)
    #         Cq_table = np.zeros(n_U, n_tsr, n_pitch)

    #         f.readline()
    #         f.readline()
    #         f.readline()

    #         for i_U in range(n_U):
    #             for i_tsr in range(n_tsr):
    #                 Cp_table[i_U, i_tsr,:] = [float(idx.strip()) for idx in f.readline().split('WE_FOPoles_v')[0].split() if idx.strip() != '!']




    def read_HydroDyn(self):
        # AeroDyn v2.03

        hd_file = os.path.normpath(os.path.join(self.FAST_directory, self.fst_vt['Fst']['HydroFile']))
        f = open(hd_file)

        f.readline()
        f.readline()

        self.fst_vt['HydroDyn']['Echo'] = bool_read(f.readline().split()[0])
        # ENVIRONMENTAL CONDITIONS
        f.readline()
        self.fst_vt['HydroDyn']['WtrDens'] = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WtrDpth'] = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['MSL2SWL'] = float_read(f.readline().split()[0])

        # WAVES
        f.readline()
        self.fst_vt['HydroDyn']['WaveMod']       = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WaveStMod']     = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WaveTMax']      = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WaveDT']        = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WaveHs']        = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WaveTp']        = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WavePkShp']     = float_read(f.readline().split()[0]) # default
        self.fst_vt['HydroDyn']['WvLowCOff']     = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WvHiCOff']      = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WaveDir']       = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WaveDirMod']    = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WaveDirSpread'] = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WaveNDir']      = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WaveDirRange']  = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WaveSeed1']     = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WaveSeed2']     = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WaveNDAmp']     = bool_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WvKinFile']     = f.readline().split()[0][1:-1]
        self.fst_vt['HydroDyn']['NWaveElev']     = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WaveElevxi']    = [idx.strip() for idx in f.readline().split('WaveElevxi')[0].split(',')]
        self.fst_vt['HydroDyn']['WaveElevyi']    = [idx.strip() for idx in f.readline().split('WaveElevyi')[0].split(',')]

        # 2ND-ORDER WAVES
        f.readline()
        self.fst_vt['HydroDyn']['WvDiffQTF']     = bool_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WvSumQTF']      = bool_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WvLowCOffD']    = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WvHiCOffD']     = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WvLowCOffS']    = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['WvHiCOffS']     = float_read(f.readline().split()[0])

        # CURRENT
        f.readline()
        self.fst_vt['HydroDyn']['CurrMod']       = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['CurrSSV0']      = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['CurrSSDir']     = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['CurrNSRef']     = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['CurrNSV0']      = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['CurrNSDir']     = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['CurrDIV']       = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['CurrDIDir']     = float_read(f.readline().split()[0])

        # FLOATING PLATFORM
        f.readline()
        self.fst_vt['HydroDyn']['PotMod']        = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['PotFile']       = os.path.normpath(os.path.join(os.path.split(hd_file)[0], f.readline().split()[0][1:-1]))
        self.fst_vt['HydroDyn']['WAMITULEN']     = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['PtfmVol0']      = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['PtfmCOBxt']     = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['PtfmCOByt']     = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['RdtnMod']       = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['RdtnTMax']      = float_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['RdtnDT']        = float_read(f.readline().split()[0])

        # 2ND-ORDER FLOATING PLATFORM FORCES
        f.readline()
        self.fst_vt['HydroDyn']['MnDrift']       = int_read(f.readline().split()[0]) # ?
        self.fst_vt['HydroDyn']['NewmanApp']     = int_read(f.readline().split()[0]) # ?
        self.fst_vt['HydroDyn']['DiffQTF']       = int_read(f.readline().split()[0]) # ?
        self.fst_vt['HydroDyn']['SumQTF']        = int_read(f.readline().split()[0]) # ?

        # FLOATING PLATFORM FORCE FLAGS
        f.readline()
        self.fst_vt['HydroDyn']['PtfmSgF']       = bool_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['PtfmSwF']       = bool_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['PtfmHvF']       = bool_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['PtfmRF']        = bool_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['PtfmPF']        = bool_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['PtfmYF']        = bool_read(f.readline().split()[0])

        # PLATFORM ADDITIONAL STIFFNESS AND DAMPING
        f.readline()
        self.fst_vt['HydroDyn']['AddF0']         = [float(idx) for idx in f.readline().strip().split()[:6]]
        self.fst_vt['HydroDyn']['AddCLin']       = np.array([[float(idx) for idx in f.readline().strip().split()[:6]] for i in range(6)])
        self.fst_vt['HydroDyn']['AddBLin']       = np.array([[float(idx) for idx in f.readline().strip().split()[:6]] for i in range(6)])
        self.fst_vt['HydroDyn']['AddBQuad']      = np.array([[float(idx) for idx in f.readline().strip().split()[:6]] for i in range(6)])

        #AXIAL COEFFICIENTS
        f.readline()
        self.fst_vt['HydroDyn']['NAxCoef']       = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['AxCoefID']      = [None]*self.fst_vt['HydroDyn']['NAxCoef']
        self.fst_vt['HydroDyn']['AxCd']          = [None]*self.fst_vt['HydroDyn']['NAxCoef']
        self.fst_vt['HydroDyn']['AxCa']          = [None]*self.fst_vt['HydroDyn']['NAxCoef']
        self.fst_vt['HydroDyn']['AxCp']          = [None]*self.fst_vt['HydroDyn']['NAxCoef']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['HydroDyn']['NAxCoef']):
            ln = f.readline().split()
            self.fst_vt['HydroDyn']['AxCoefID'][i] = int(ln[0])
            self.fst_vt['HydroDyn']['AxCd'][i]     = float(ln[1])
            self.fst_vt['HydroDyn']['AxCa'][i]     = float(ln[2])
            self.fst_vt['HydroDyn']['AxCp'][i]     = float(ln[3])

        #MEMBER JOINTS
        f.readline()
        self.fst_vt['HydroDyn']['NJoints']    = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['JointID']    = [None]*self.fst_vt['HydroDyn']['NJoints']
        self.fst_vt['HydroDyn']['Jointxi']    = [None]*self.fst_vt['HydroDyn']['NJoints']
        self.fst_vt['HydroDyn']['Jointyi']    = [None]*self.fst_vt['HydroDyn']['NJoints']
        self.fst_vt['HydroDyn']['Jointzi']    = [None]*self.fst_vt['HydroDyn']['NJoints']
        self.fst_vt['HydroDyn']['JointAxID']  = [None]*self.fst_vt['HydroDyn']['NJoints']
        self.fst_vt['HydroDyn']['JointOvrlp'] = [None]*self.fst_vt['HydroDyn']['NJoints']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['HydroDyn']['NJoints']):
            ln = f.readline().split()
            self.fst_vt['HydroDyn']['JointID'][i]    = int(ln[0])
            self.fst_vt['HydroDyn']['Jointxi'][i]    = float(ln[1])
            self.fst_vt['HydroDyn']['Jointyi'][i]    = float(ln[2])
            self.fst_vt['HydroDyn']['Jointzi'][i]    = float(ln[3])
            self.fst_vt['HydroDyn']['JointAxID'][i]  = int(ln[4])
            self.fst_vt['HydroDyn']['JointOvrlp'][i] = int(ln[5])

        #MEMBER CROSS-SECTION PROPERTIES
        f.readline()
        self.fst_vt['HydroDyn']['NPropSets'] = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['PropSetID'] = [None]*self.fst_vt['HydroDyn']['NPropSets']
        self.fst_vt['HydroDyn']['PropD']     = [None]*self.fst_vt['HydroDyn']['NPropSets']
        self.fst_vt['HydroDyn']['PropThck']  = [None]*self.fst_vt['HydroDyn']['NPropSets']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['HydroDyn']['NPropSets']):
            ln = f.readline().split()
            self.fst_vt['HydroDyn']['PropSetID'][i] = int(ln[0])
            self.fst_vt['HydroDyn']['PropD'][i]     = float(ln[1])
            self.fst_vt['HydroDyn']['PropThck'][i]  = float(ln[2])

        #SIMPLE HYDRODYNAMIC COEFFICIENTS
        f.readline()
        f.readline()
        f.readline()
        ln = f.readline().split()
        self.fst_vt['HydroDyn']['SimplCd']     = float(ln[0])
        self.fst_vt['HydroDyn']['SimplCdMG']   = float(ln[1])
        self.fst_vt['HydroDyn']['SimplCa']     = float(ln[2])
        self.fst_vt['HydroDyn']['SimplCaMG']   = float(ln[3])
        self.fst_vt['HydroDyn']['SimplCp']     = float(ln[4])
        self.fst_vt['HydroDyn']['SimplCpMG']   = float(ln[5])
        self.fst_vt['HydroDyn']['SimplAxCa']   = float(ln[6])
        self.fst_vt['HydroDyn']['SimplAxCaMG'] = float(ln[7])
        self.fst_vt['HydroDyn']['SimplAxCp']   = float(ln[8])
        self.fst_vt['HydroDyn']['SimplAxCpMG'] = float(ln[9])

        #DEPTH-BASED HYDRODYNAMIC COEFFICIENTS
        f.readline()
        self.fst_vt['HydroDyn']['NCoefDpth']  = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['Dpth']       = [None]*self.fst_vt['HydroDyn']['NCoefDpth']
        self.fst_vt['HydroDyn']['DpthCd']     = [None]*self.fst_vt['HydroDyn']['NCoefDpth']
        self.fst_vt['HydroDyn']['DpthCdMG']   = [None]*self.fst_vt['HydroDyn']['NCoefDpth']
        self.fst_vt['HydroDyn']['DpthCa']     = [None]*self.fst_vt['HydroDyn']['NCoefDpth']
        self.fst_vt['HydroDyn']['DpthCaMG']   = [None]*self.fst_vt['HydroDyn']['NCoefDpth']
        self.fst_vt['HydroDyn']['DpthCp']     = [None]*self.fst_vt['HydroDyn']['NCoefDpth']
        self.fst_vt['HydroDyn']['DpthCpMG']   = [None]*self.fst_vt['HydroDyn']['NCoefDpth']
        self.fst_vt['HydroDyn']['DpthAxCa']   = [None]*self.fst_vt['HydroDyn']['NCoefDpth']
        self.fst_vt['HydroDyn']['DpthAxCaMG'] = [None]*self.fst_vt['HydroDyn']['NCoefDpth']
        self.fst_vt['HydroDyn']['DpthAxCp']   = [None]*self.fst_vt['HydroDyn']['NCoefDpth']
        self.fst_vt['HydroDyn']['DpthAxCpMG'] = [None]*self.fst_vt['HydroDyn']['NCoefDpth']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['HydroDyn']['NCoefDpth']):
            ln = f.readline().split()
            self.fst_vt['HydroDyn']['Dpth'][i]       = float(ln[0])
            self.fst_vt['HydroDyn']['DpthCd'][i]     = float(ln[1])
            self.fst_vt['HydroDyn']['DpthCdMG'][i]   = float(ln[2])
            self.fst_vt['HydroDyn']['DpthCa'][i]     = float(ln[3])
            self.fst_vt['HydroDyn']['DpthCaMG'][i]   = float(ln[4])
            self.fst_vt['HydroDyn']['DpthCp'][i]     = float(ln[5])
            self.fst_vt['HydroDyn']['DpthCpMG'][i]   = float(ln[6])
            self.fst_vt['HydroDyn']['DpthAxCa'][i]   = float(ln[7])
            self.fst_vt['HydroDyn']['DpthAxCaMG'][i] = float(ln[8])
            self.fst_vt['HydroDyn']['DpthAxCp'][i]   = float(ln[9])
            self.fst_vt['HydroDyn']['DpthAxCpMG'][i] = float(ln[10])

        #MEMBER-BASED HYDRODYNAMIC COEFFICIENTS
        f.readline()
        self.fst_vt['HydroDyn']['NCoefMembers']  = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['MemberID_HydC']      = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberCd1']     = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberCd2']     = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberCdMG1']   = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberCdMG2']   = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberCa1']     = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberCa2']     = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberCaMG1']   = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberCaMG2']   = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberCp1']     = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberCp2']     = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberCpMG1']   = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberCpMG2']   = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberAxCa1']   = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberAxCa2']   = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberAxCaMG1'] = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberAxCaMG2'] = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberAxCp1']   = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberAxCp2']   = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberAxCpMG1'] = [None]*self.fst_vt['HydroDyn']['NCoefMembers']
        self.fst_vt['HydroDyn']['MemberAxCpMG2'] = [None]*self.fst_vt['HydroDyn']['NCoefMembers']

        f.readline()
        f.readline()
        for i in range(self.fst_vt['HydroDyn']['NCoefMembers']):
            ln = f.readline().split()
            self.fst_vt['HydroDyn']['MemberID_HydC'][i]      = int(ln[0])
            self.fst_vt['HydroDyn']['MemberCd1'][i]     = float(ln[1])
            self.fst_vt['HydroDyn']['MemberCd2'][i]     = float(ln[2])
            self.fst_vt['HydroDyn']['MemberCdMG1'][i]   = float(ln[3])
            self.fst_vt['HydroDyn']['MemberCdMG2'][i]   = float(ln[4])
            self.fst_vt['HydroDyn']['MemberCa1'][i]     = float(ln[5])
            self.fst_vt['HydroDyn']['MemberCa2'][i]     = float(ln[6])
            self.fst_vt['HydroDyn']['MemberCaMG1'][i]   = float(ln[7])
            self.fst_vt['HydroDyn']['MemberCaMG2'][i]   = float(ln[8])
            self.fst_vt['HydroDyn']['MemberCp1'][i]     = float(ln[9])
            self.fst_vt['HydroDyn']['MemberCp2'][i]     = float(ln[10])
            self.fst_vt['HydroDyn']['MemberCpMG1'][i]   = float(ln[11])
            self.fst_vt['HydroDyn']['MemberCpMG2'][i]   = float(ln[12])
            self.fst_vt['HydroDyn']['MemberAxCa1'][i]   = float(ln[13])
            self.fst_vt['HydroDyn']['MemberAxCa2'][i]   = float(ln[14])
            self.fst_vt['HydroDyn']['MemberAxCaMG1'][i] = float(ln[15])
            self.fst_vt['HydroDyn']['MemberAxCaMG2'][i] = float(ln[16])
            self.fst_vt['HydroDyn']['MemberAxCp1'][i]   = float(ln[17])
            self.fst_vt['HydroDyn']['MemberAxCp2'][i]   = float(ln[18])
            self.fst_vt['HydroDyn']['MemberAxCpMG1'][i] = float(ln[19])
            self.fst_vt['HydroDyn']['MemberAxCpMG2'][i] = float(ln[20])

        #MEMBERS
        f.readline()
        self.fst_vt['HydroDyn']['NMembers']    = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['MemberID']    = [None]*self.fst_vt['HydroDyn']['NMembers']
        self.fst_vt['HydroDyn']['MJointID1']   = [None]*self.fst_vt['HydroDyn']['NMembers']
        self.fst_vt['HydroDyn']['MJointID2']   = [None]*self.fst_vt['HydroDyn']['NMembers']
        self.fst_vt['HydroDyn']['MPropSetID1'] = [None]*self.fst_vt['HydroDyn']['NMembers']
        self.fst_vt['HydroDyn']['MPropSetID2'] = [None]*self.fst_vt['HydroDyn']['NMembers']
        self.fst_vt['HydroDyn']['MDivSize']    = [None]*self.fst_vt['HydroDyn']['NMembers']
        self.fst_vt['HydroDyn']['MCoefMod']    = [None]*self.fst_vt['HydroDyn']['NMembers']
        self.fst_vt['HydroDyn']['PropPot']     = [None]*self.fst_vt['HydroDyn']['NMembers']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['HydroDyn']['NMembers']):
            ln = f.readline().split()
            self.fst_vt['HydroDyn']['MemberID'][i]    = int(ln[0])
            self.fst_vt['HydroDyn']['MJointID1'][i]   = int(ln[1])
            self.fst_vt['HydroDyn']['MJointID2'][i]   = int(ln[2])
            self.fst_vt['HydroDyn']['MPropSetID1'][i] = int(ln[3])
            self.fst_vt['HydroDyn']['MPropSetID2'][i] = int(ln[4])
            self.fst_vt['HydroDyn']['MDivSize'][i]    = float(ln[5])
            self.fst_vt['HydroDyn']['MCoefMod'][i]    = int(ln[6])
            self.fst_vt['HydroDyn']['PropPot'][i]     = bool_read(ln[7])

        #FILLED MEMBERS
        f.readline()
        self.fst_vt['HydroDyn']['NFillGroups'] = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['FillNumM']    = [None]*self.fst_vt['HydroDyn']['NFillGroups']
        self.fst_vt['HydroDyn']['FillMList']   = [None]*self.fst_vt['HydroDyn']['NFillGroups']
        self.fst_vt['HydroDyn']['FillFSLoc']   = [None]*self.fst_vt['HydroDyn']['NFillGroups']
        self.fst_vt['HydroDyn']['FillDens']    = [None]*self.fst_vt['HydroDyn']['NFillGroups']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['HydroDyn']['NFillGroups']):
            ln = f.readline().split()
            self.fst_vt['HydroDyn']['FillNumM'][i]  = int(ln[0])
            self.fst_vt['HydroDyn']['FillMList'][i] = [int(j) for j in ln[1:-2]]
            self.fst_vt['HydroDyn']['FillFSLoc'][i] = float(ln[-2])
            self.fst_vt['HydroDyn']['FillDens'][i]  = float(ln[-1])

        #MARINE GROWTH
        f.readline()
        self.fst_vt['HydroDyn']['NMGDepths'] = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['MGDpth']    = [None]*self.fst_vt['HydroDyn']['NMGDepths']
        self.fst_vt['HydroDyn']['MGThck']    = [None]*self.fst_vt['HydroDyn']['NMGDepths']
        self.fst_vt['HydroDyn']['MGDens']    = [None]*self.fst_vt['HydroDyn']['NMGDepths']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['HydroDyn']['NMGDepths']):
            ln = f.readline().split()
            self.fst_vt['HydroDyn']['MGDpth'][i] = float(ln[0])
            self.fst_vt['HydroDyn']['MGThck'][i] = float(ln[1])
            self.fst_vt['HydroDyn']['MGDens'][i] = float(ln[2])

        #MEMBER OUTPUT LIST
        f.readline()
        self.fst_vt['HydroDyn']['NMOutputs'] = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['MemberID_out']  = [None]*self.fst_vt['HydroDyn']['NMOutputs']
        self.fst_vt['HydroDyn']['NOutLoc']   = [None]*self.fst_vt['HydroDyn']['NMOutputs']
        self.fst_vt['HydroDyn']['NodeLocs']  = [None]*self.fst_vt['HydroDyn']['NMOutputs']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['HydroDyn']['NMOutputs']):
            ln = f.readline().split()
            self.fst_vt['HydroDyn']['MemberID_out'][i] = int(ln[0])
            self.fst_vt['HydroDyn']['NOutLoc'][i]  = int(ln[1])
            self.fst_vt['HydroDyn']['NodeLocs'][i] = float(ln[2])

        #JOINT OUTPUT LIST
        f.readline()
        self.fst_vt['HydroDyn']['NJOutputs'] = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['JOutLst']   = [int(idx.strip()) for idx in f.readline().split('JOutLst')[0].split(',')]

        #OUTPUT
        f.readline()
        self.fst_vt['HydroDyn']['HDSum']     = bool_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['OutAll']    = bool_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['OutSwtch']  = int_read(f.readline().split()[0])
        self.fst_vt['HydroDyn']['OutFmt']    = str(f.readline().split()[0])
        self.fst_vt['HydroDyn']['OutSFmt']   = str(f.readline().split()[0])

        self.fst_vt['HydroDyn']['HDSum']   
        self.fst_vt['HydroDyn']['OutAll']  
        self.fst_vt['HydroDyn']['OutSwtch']
        self.fst_vt['HydroDyn']['OutFmt']  
        self.fst_vt['HydroDyn']['OutSFmt'] 

        # HydroDyn Outlist
        f.readline()
        data = f.readline()
        while data.split()[0] != 'END':
            channels = data.split('"')
            channel_list = channels[1].split(',')
            self.set_outlist(self.fst_vt['outlist']['HydroDyn'], channel_list)
            data = f.readline()

        f.close()

    def read_SubDyn(self):
        # SubDyn v1.01

        sd_file = os.path.normpath(os.path.join(self.FAST_directory, self.fst_vt['Fst']['SubFile']))
        f = open(sd_file)
        f.readline()
        f.readline()
        f.readline()
        # SIMULATION CONTROL
        self.fst_vt['SubDyn']['Echo']      = bool_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['SDdeltaT']  = float_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['IntMethod'] = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['SttcSolve'] = bool_read(f.readline().split()[0])
        f.readline()
        # FEA and CRAIG-BAMPTON PARAMETERS
        self.fst_vt['SubDyn']['FEMMod']    = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['NDiv']      = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['CBMod']     = bool_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['Nmodes']    = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['JDampings'] = int_read(f.readline().split()[0])
        f.readline()
        # STRUCTURE JOINTS
        self.fst_vt['SubDyn']['NJoints']   = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['JointID']   = [None]*self.fst_vt['SubDyn']['NJoints']
        self.fst_vt['SubDyn']['JointXss']  = [None]*self.fst_vt['SubDyn']['NJoints']
        self.fst_vt['SubDyn']['JointYss']  = [None]*self.fst_vt['SubDyn']['NJoints']
        self.fst_vt['SubDyn']['JointZss']  = [None]*self.fst_vt['SubDyn']['NJoints']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['SubDyn']['NJoints']):
            ln = f.readline().split()
            self.fst_vt['SubDyn']['JointID'][i]    = int(ln[0])
            self.fst_vt['SubDyn']['JointXss'][i]   = float(ln[1])
            self.fst_vt['SubDyn']['JointYss'][i]   = float(ln[2])
            self.fst_vt['SubDyn']['JointZss'][i]   = float(ln[3])
        f.readline()
        # BASE REACTION JOINTS
        self.fst_vt['SubDyn']['NReact']   = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['RJointID'] = [None]*self.fst_vt['SubDyn']['NReact']
        self.fst_vt['SubDyn']['RctTDXss'] = [None]*self.fst_vt['SubDyn']['NReact']
        self.fst_vt['SubDyn']['RctTDYss'] = [None]*self.fst_vt['SubDyn']['NReact']
        self.fst_vt['SubDyn']['RctTDZss'] = [None]*self.fst_vt['SubDyn']['NReact']
        self.fst_vt['SubDyn']['RctRDXss'] = [None]*self.fst_vt['SubDyn']['NReact']
        self.fst_vt['SubDyn']['RctRDYss'] = [None]*self.fst_vt['SubDyn']['NReact']
        self.fst_vt['SubDyn']['RctRDZss'] = [None]*self.fst_vt['SubDyn']['NReact']
        self.fst_vt['SubDyn']['Rct_SoilFile'] = [None]*self.fst_vt['SubDyn']['NReact']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['SubDyn']['NReact']):
            ln = f.readline().split()
            self.fst_vt['SubDyn']['RJointID'][i] = int(ln[0])
            self.fst_vt['SubDyn']['RctTDXss'][i] = int(ln[1])
            self.fst_vt['SubDyn']['RctTDYss'][i] = int(ln[2])
            self.fst_vt['SubDyn']['RctTDZss'][i] = int(ln[3])
            self.fst_vt['SubDyn']['RctRDXss'][i] = int(ln[4])
            self.fst_vt['SubDyn']['RctRDYss'][i] = int(ln[5])
            self.fst_vt['SubDyn']['RctRDZss'][i] = int(ln[6])
            self.fst_vt['SubDyn']['Rct_SoilFile'][i] = ln[7]
        f.readline()
        # INTERFACE JOINTS
        self.fst_vt['SubDyn']['NInterf']   = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['IJointID'] = [None]*self.fst_vt['SubDyn']['NInterf']
        self.fst_vt['SubDyn']['ItfTDXss'] = [None]*self.fst_vt['SubDyn']['NInterf']
        self.fst_vt['SubDyn']['ItfTDYss'] = [None]*self.fst_vt['SubDyn']['NInterf']
        self.fst_vt['SubDyn']['ItfTDZss'] = [None]*self.fst_vt['SubDyn']['NInterf']
        self.fst_vt['SubDyn']['ItfRDXss'] = [None]*self.fst_vt['SubDyn']['NInterf']
        self.fst_vt['SubDyn']['ItfRDYss'] = [None]*self.fst_vt['SubDyn']['NInterf']
        self.fst_vt['SubDyn']['ItfRDZss'] = [None]*self.fst_vt['SubDyn']['NInterf']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['SubDyn']['NInterf']):
            ln = f.readline().split()
            self.fst_vt['SubDyn']['IJointID'][i] = int(ln[0])
            self.fst_vt['SubDyn']['ItfTDXss'][i] = int(ln[1])
            self.fst_vt['SubDyn']['ItfTDYss'][i] = int(ln[2])
            self.fst_vt['SubDyn']['ItfTDZss'][i] = int(ln[3])
            self.fst_vt['SubDyn']['ItfRDXss'][i] = int(ln[4])
            self.fst_vt['SubDyn']['ItfRDYss'][i] = int(ln[5])
            self.fst_vt['SubDyn']['ItfRDZss'][i] = int(ln[6])
        f.readline()
        # MEMBERS
        self.fst_vt['SubDyn']['NMembers']    = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['MemberID']    = [None]*self.fst_vt['SubDyn']['NMembers']
        self.fst_vt['SubDyn']['MJointID1']   = [None]*self.fst_vt['SubDyn']['NMembers']
        self.fst_vt['SubDyn']['MJointID2']   = [None]*self.fst_vt['SubDyn']['NMembers']
        self.fst_vt['SubDyn']['MPropSetID1'] = [None]*self.fst_vt['SubDyn']['NMembers']
        self.fst_vt['SubDyn']['MPropSetID2'] = [None]*self.fst_vt['SubDyn']['NMembers']
        self.fst_vt['SubDyn']['COSMID']      = [None]*self.fst_vt['SubDyn']['NMembers']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['SubDyn']['NMembers']):
            ln = f.readline().split()
            self.fst_vt['SubDyn']['MemberID'][i]    = int(ln[0])
            self.fst_vt['SubDyn']['MJointID1'][i]   = int(ln[1])
            self.fst_vt['SubDyn']['MJointID2'][i]   = int(ln[2])
            self.fst_vt['SubDyn']['MPropSetID1'][i] = int(ln[3])
            self.fst_vt['SubDyn']['MPropSetID2'][i] = int(ln[4])
            if len(ln) > 5:
                self.fst_vt['SubDyn']['COSMID'][i]  = int(ln[5])
        f.readline()
        # MEMBER X-SECTION PROPERTY data 1/2
        self.fst_vt['SubDyn']['NPropSets'] = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['PropSetID1'] = [None]*self.fst_vt['SubDyn']['NPropSets']
        self.fst_vt['SubDyn']['YoungE1']    = [None]*self.fst_vt['SubDyn']['NPropSets']
        self.fst_vt['SubDyn']['ShearG1']    = [None]*self.fst_vt['SubDyn']['NPropSets']
        self.fst_vt['SubDyn']['MatDens1']   = [None]*self.fst_vt['SubDyn']['NPropSets']
        self.fst_vt['SubDyn']['XsecD']     = [None]*self.fst_vt['SubDyn']['NPropSets']
        self.fst_vt['SubDyn']['XsecT']     = [None]*self.fst_vt['SubDyn']['NPropSets']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['SubDyn']['NPropSets']):
            ln = f.readline().split()
            self.fst_vt['SubDyn']['PropSetID1'][i] = int(ln[0])
            self.fst_vt['SubDyn']['YoungE1'][i]    = float(ln[1])
            self.fst_vt['SubDyn']['ShearG1'][i]    = float(ln[2])
            self.fst_vt['SubDyn']['MatDens1'][i]   = float(ln[3])
            self.fst_vt['SubDyn']['XsecD'][i]     = float(ln[4])
            self.fst_vt['SubDyn']['XsecT'][i]     = float(ln[5])
        f.readline()
        # MEMBER X-SECTION PROPERTY data 2/2
        self.fst_vt['SubDyn']['NXPropSets'] = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['PropSetID2']  = [None]*self.fst_vt['SubDyn']['NXPropSets']
        self.fst_vt['SubDyn']['YoungE2']     = [None]*self.fst_vt['SubDyn']['NXPropSets']
        self.fst_vt['SubDyn']['ShearG2']     = [None]*self.fst_vt['SubDyn']['NXPropSets']
        self.fst_vt['SubDyn']['MatDens2']    = [None]*self.fst_vt['SubDyn']['NXPropSets']
        self.fst_vt['SubDyn']['XsecA']      = [None]*self.fst_vt['SubDyn']['NXPropSets']
        self.fst_vt['SubDyn']['XsecAsx']    = [None]*self.fst_vt['SubDyn']['NXPropSets']
        self.fst_vt['SubDyn']['XsecAsy']    = [None]*self.fst_vt['SubDyn']['NXPropSets']
        self.fst_vt['SubDyn']['XsecJxx']    = [None]*self.fst_vt['SubDyn']['NXPropSets']
        self.fst_vt['SubDyn']['XsecJyy']    = [None]*self.fst_vt['SubDyn']['NXPropSets']
        self.fst_vt['SubDyn']['XsecJ0']     = [None]*self.fst_vt['SubDyn']['NXPropSets']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['SubDyn']['NXPropSets']):
            ln = f.readline().split()
            self.fst_vt['SubDyn']['PropSetID2'][i] = int(ln[0])
            self.fst_vt['SubDyn']['YoungE2'][i]    = float(ln[1])
            self.fst_vt['SubDyn']['ShearG2'][i]    = float(ln[2])
            self.fst_vt['SubDyn']['MatDens2'][i]   = float(ln[3])
            self.fst_vt['SubDyn']['XsecA'][i]     = float(ln[4])
            self.fst_vt['SubDyn']['XsecAsx'][i]   = float(ln[5])
            self.fst_vt['SubDyn']['XsecAsy'][i]   = float(ln[6])
            self.fst_vt['SubDyn']['XsecJxx'][i]   = float(ln[7])
            self.fst_vt['SubDyn']['XsecJyy'][i]   = float(ln[8])
            self.fst_vt['SubDyn']['XsecJ0'][i]    = float(ln[9])
        f.readline()
        # MEMBER COSINE MATRICES
        self.fst_vt['SubDyn']['NCOSMs'] = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['COSMID'] = [None]*self.fst_vt['SubDyn']['NCOSMs']
        self.fst_vt['SubDyn']['COSM11'] = [None]*self.fst_vt['SubDyn']['NCOSMs']
        self.fst_vt['SubDyn']['COSM12'] = [None]*self.fst_vt['SubDyn']['NCOSMs']
        self.fst_vt['SubDyn']['COSM13'] = [None]*self.fst_vt['SubDyn']['NCOSMs']
        self.fst_vt['SubDyn']['COSM21'] = [None]*self.fst_vt['SubDyn']['NCOSMs']
        self.fst_vt['SubDyn']['COSM22'] = [None]*self.fst_vt['SubDyn']['NCOSMs']
        self.fst_vt['SubDyn']['COSM23'] = [None]*self.fst_vt['SubDyn']['NCOSMs']
        self.fst_vt['SubDyn']['COSM31'] = [None]*self.fst_vt['SubDyn']['NCOSMs']
        self.fst_vt['SubDyn']['COSM32'] = [None]*self.fst_vt['SubDyn']['NCOSMs']
        self.fst_vt['SubDyn']['COSM33'] = [None]*self.fst_vt['SubDyn']['NCOSMs']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['SubDyn']['NCOSMs']):
            ln = f.readline().split()
            self.fst_vt['SubDyn']['COSMID'][i] = int(ln[0])
            self.fst_vt['SubDyn']['COSM11'][i] = float(ln[1])
            self.fst_vt['SubDyn']['COSM12'][i] = float(ln[2])
            self.fst_vt['SubDyn']['COSM13'][i] = float(ln[3])
            self.fst_vt['SubDyn']['COSM21'][i] = float(ln[4])
            self.fst_vt['SubDyn']['COSM22'][i] = float(ln[5])
            self.fst_vt['SubDyn']['COSM23'][i] = float(ln[6])
            self.fst_vt['SubDyn']['COSM31'][i] = float(ln[7])
            self.fst_vt['SubDyn']['COSM32'][i] = float(ln[8])
            self.fst_vt['SubDyn']['COSM33'][i] = float(ln[9])
        f.readline()
        # JOINT ADDITIONAL CONCENTRATED MASSES
        self.fst_vt['SubDyn']['NCmass']    = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['CMJointID'] = [None]*self.fst_vt['SubDyn']['NCmass']
        self.fst_vt['SubDyn']['JMass']     = [None]*self.fst_vt['SubDyn']['NCmass']
        self.fst_vt['SubDyn']['JMXX']      = [None]*self.fst_vt['SubDyn']['NCmass']
        self.fst_vt['SubDyn']['JMYY']      = [None]*self.fst_vt['SubDyn']['NCmass']
        self.fst_vt['SubDyn']['JMZZ']      = [None]*self.fst_vt['SubDyn']['NCmass']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['SubDyn']['NCmass']):
            ln = f.readline().split()
            self.fst_vt['SubDyn']['CMJointID'][i] = int(ln[0])
            self.fst_vt['SubDyn']['JMass'][i]     = float(ln[1])
            self.fst_vt['SubDyn']['JMXX'][i]      = float(ln[2])
            self.fst_vt['SubDyn']['JMYY'][i]      = float(ln[3])
            self.fst_vt['SubDyn']['JMZZ'][i]      = float(ln[4])
        f.readline()
        # OUTPUT
        self.fst_vt['SubDyn']['SSSum']    = bool_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['OutCOSM']  = bool_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['OutAll']   = bool_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['OutSwtch'] = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['TabDelim'] = bool_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['OutDec']   = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['OutFmt']   = f.readline().split()[0]
        self.fst_vt['SubDyn']['OutSFmt']  = f.readline().split()[0]
        f.readline()
        # MEMBER OUTPUT LIST
        self.fst_vt['SubDyn']['NMOutputs']     = int_read(f.readline().split()[0])
        self.fst_vt['SubDyn']['MemberID_out']  = [None]*self.fst_vt['SubDyn']['NMOutputs']
        self.fst_vt['SubDyn']['NOutCnt']       = [None]*self.fst_vt['SubDyn']['NMOutputs']
        self.fst_vt['SubDyn']['NodeCnt']       = [None]*self.fst_vt['SubDyn']['NMOutputs']
        ln = f.readline().split()
        ln = f.readline().split()
        for i in range(self.fst_vt['SubDyn']['NMOutputs']):
            ln = f.readline().split()
            self.fst_vt['SubDyn']['MemberID_out'][i] = int(ln[0])
            self.fst_vt['SubDyn']['NOutCnt'][i]      = int(ln[1])
            self.fst_vt['SubDyn']['NodeCnt'][i]      = int(ln[2])
        f.readline()
        # SSOutList
        data = f.readline()
        while data.split()[0] != 'END':
            channels = data.split('"')
            channel_list = channels[1].split(',')
            self.set_outlist(self.fst_vt['outlist']['SubDyn'], channel_list)
            data = f.readline()


    def read_MAP(self):
        # MAP++

        # TODO: this is likely not robust enough, only tested on the Hywind Spar
        # additional lines in these tables are likely

        map_file = os.path.normpath(os.path.join(self.FAST_directory, self.fst_vt['Fst']['MooringFile']))

        f = open(map_file)
        f.readline()
        f.readline()
        f.readline()
        data_line = f.readline().strip().split()
        self.fst_vt['MAP']['LineType']     = str(data_line[0])
        self.fst_vt['MAP']['Diam']         = float(data_line[1])
        self.fst_vt['MAP']['MassDenInAir'] = float(data_line[2])
        self.fst_vt['MAP']['EA']           = float(data_line[3])
        self.fst_vt['MAP']['CB']           = float(data_line[4])
        self.fst_vt['MAP']['CIntDamp']     = float(data_line[5])
        self.fst_vt['MAP']['Ca']           = float(data_line[6])
        self.fst_vt['MAP']['Cdn']          = float(data_line[7])
        self.fst_vt['MAP']['Cdt']          = float(data_line[8])
        f.readline()
        f.readline()
        f.readline()
        for i in range(2):
            data_node = f.readline().strip().split()
            self.fst_vt['MAP']['Node'].append(int(data_node[0]))
            self.fst_vt['MAP']['Type'].append(str(data_node[1]))
            self.fst_vt['MAP']['X'].append(float_read(data_node[2]))
            self.fst_vt['MAP']['Y'].append(float_read(data_node[3]))
            self.fst_vt['MAP']['Z'].append(float_read(data_node[4]))
            self.fst_vt['MAP']['M'].append(float_read(data_node[5]))
            self.fst_vt['MAP']['B'].append(float_read(data_node[6]))
            self.fst_vt['MAP']['FX'].append(float_read(data_node[7]))
            self.fst_vt['MAP']['FY'].append(float_read(data_node[8]))
            self.fst_vt['MAP']['FZ'].append(float_read(data_node[9]))
        f.readline()
        f.readline()
        f.readline()
        data_line_prop = f.readline().strip().split()
        self.fst_vt['MAP']['Line']     = int(data_line_prop[0])
        self.fst_vt['MAP']['LineType'] = str(data_line_prop[1])
        self.fst_vt['MAP']['UnstrLen'] = float(data_line_prop[2])
        self.fst_vt['MAP']['NodeAnch'] = int(data_line_prop[3])
        self.fst_vt['MAP']['NodeFair'] = int(data_line_prop[4])
        self.fst_vt['MAP']['Flags']    = [str(val) for val in data_line_prop[5:]]
        f.readline()
        f.readline()
        f.readline()
        self.fst_vt['MAP']['Option']   = [str(val) for val in f.readline().strip().split()]


class InputReader_FAST7(InputReader_Common):
    """ FASTv7.02 input file reader """
    
    def execute(self):
        self.read_MainInput()
        self.read_AeroDyn_FAST7()
        # if self.fst_vt['aerodyn']['wind_file_type'][1]  == 'wnd':
        #     self.WndWindReader(self.fst_vt['aerodyn']['WindFile'])
        # else:
        #     print 'Wind reader for file type .%s not implemented yet.' % self.fst_vt['aerodyn']['wind_file_type'][1]
        self.read_ElastoDynBlade()
        self.read_ElastoDynTower()

    def read_MainInput(self):

        fst_file = os.path.join(self.FAST_directory, self.FAST_InputFile)
        f = open(fst_file)

        # FAST Inputs
        f.readline()
        f.readline()
        self.fst_vt['description'] = f.readline().rstrip()
        f.readline()
        f.readline()
        self.fst_vt['Fst7']['Echo'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['ADAMSPrep'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['AnalMode'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['NumBl'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['TMax'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['DT']  = float_read(f.readline().split()[0])
        f.readline()
        self.fst_vt['Fst7']['YCMode'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['TYCOn'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['PCMode'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['TPCOn'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['VSContrl'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['VS_RtGnSp']  = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['VS_RtTq']  = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['VS_Rgn2K']  = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['VS_SlPc']  = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['GenModel'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['GenTiStr'] = bool(f.readline().split()[0])
        self.fst_vt['Fst7']['GenTiStp'] = bool(f.readline().split()[0])
        self.fst_vt['Fst7']['SpdGenOn'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TimGenOn'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TimGenOf'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['HSSBrMode'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['THSSBrDp'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TiDynBrk'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TTpBrDp1'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TTpBrDp2'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TTpBrDp3'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TBDepISp1'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TBDepISp2'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TBDepISp3'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TYawManS'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TYawManE'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['NacYawF'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TPitManS1'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TPitManS2'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TPitManS3'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TPitManE1'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TPitManE2'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TPitManE3'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['BlPitch1']  = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['BlPitch2']  = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['BlPitch3']  = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['B1PitchF1'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['B1PitchF2'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['B1PitchF3'] = float_read(f.readline().split()[0])
        f.readline()
        self.fst_vt['Fst7']['Gravity'] = float_read(f.readline().split()[0])
        f.readline()
        self.fst_vt['Fst7']['FlapDOF1'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['FlapDOF2'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['EdgeDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TeetDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['DrTrDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['GenDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['YawDOF'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TwFADOF1'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TwFADOF2'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TwSSDOF1'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TwSSDOF2'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['CompAero'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['CompNoise'] = bool_read(f.readline().split()[0])
        f.readline()
        self.fst_vt['Fst7']['OoPDefl'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['IPDefl'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TeetDefl'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['Azimuth'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['RotSpeed'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['NacYaw'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TTDspFA'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TTDspSS'] = float_read(f.readline().split()[0])
        f.readline()
        self.fst_vt['Fst7']['TipRad'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['HubRad'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['PSpnElN'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['UndSling'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['HubCM'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['OverHang'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['NacCMxn'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['NacCMyn'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['NacCMzn'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TowerHt'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['Twr2Shft'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TwrRBHt'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['ShftTilt'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['Delta3'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['PreCone(1)'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['PreCone(2)'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['PreCone(3)'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['AzimB1Up'] = float_read(f.readline().split()[0])
        f.readline()
        self.fst_vt['Fst7']['YawBrMass'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['NacMass'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['HubMass'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TipMass(1)'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TipMass(2)'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TipMass(3)'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['NacYIner'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['GenIner'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['HubIner'] = float_read(f.readline().split()[0])
        f.readline()
        self.fst_vt['Fst7']['GBoxEff'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['GenEff'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['GBRatio'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['GBRevers'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['HSSBrTqF'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['HSSBrDT'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['DynBrkFi'] = f.readline().split()[0]
        self.fst_vt['Fst7']['DTTorSpr'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['DTTorDmp'] = float_read(f.readline().split()[0])
        f.readline()
        self.fst_vt['Fst7']['SIG_SlPc'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['SIG_SySp'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['SIG_RtTq'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['SIG_PORt'] = float_read(f.readline().split()[0])
        f.readline()
        self.fst_vt['Fst7']['TEC_Freq'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TEC_NPol'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['TEC_SRes'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TEC_RRes'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TEC_VLL'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TEC_SLR'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TEC_RLR'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TEC_MR'] = float_read(f.readline().split()[0])
        f.readline()
        self.fst_vt['Fst7']['PtfmModel'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['PtfmFile'] = f.readline().split()[0][1:-1]
        f.readline()
        self.fst_vt['Fst7']['TwrNodes'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['TwrFile'] = f.readline().split()[0][1:-1]
        f.readline()
        self.fst_vt['Fst7']['YawSpr'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['YawDamp'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['YawNeut'] = float_read(f.readline().split()[0])
        f.readline()
        self.fst_vt['Fst7']['Furling'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['FurlFile'] = f.readline().split()[0]
        f.readline() 
        self.fst_vt['Fst7']['TeetMod'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['TeetDmpP'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TeetDmp'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TeetCDmp'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TeetSStP'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TeetHStP'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TeetSSSp'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TeetHSSp'] = float_read(f.readline().split()[0])
        f.readline()
        self.fst_vt['Fst7']['TBDrConN'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TBDrConD'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['TpBrDT'] = float_read(f.readline().split()[0])
        f.readline()
        self.fst_vt['Fst7']['BldFile1'] = f.readline().split()[0][1:-1] # TODO - different blade files
        self.fst_vt['Fst7']['BldFile2'] = f.readline().split()[0][1:-1]
        self.fst_vt['Fst7']['BldFile3'] = f.readline().split()[0][1:-1]
        f.readline() 
        self.fst_vt['Fst7']['ADFile'] = f.readline().split()[0][1:-1]
        f.readline()
        self.fst_vt['Fst7']['NoiseFile'] = f.readline().split()[0]
        f.readline()
        self.fst_vt['Fst7']['ADAMSFile'] = f.readline().split()[0]
        f.readline()
        self.fst_vt['Fst7']['LinFile'] = f.readline().split()[0]
        f.readline()
        self.fst_vt['Fst7']['SumPrint'] = bool_read(f.readline().split()[0])
        self.fst_vt['Fst7']['OutFileFmt'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['TabDelim'] = bool_read(f.readline().split()[0])

        self.fst_vt['Fst7']['OutFmt'] = f.readline().split()[0]
        self.fst_vt['Fst7']['TStart'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['DecFact'] = int(f.readline().split()[0])
        self.fst_vt['Fst7']['SttsTime'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['NcIMUxn'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['NcIMUyn'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['NcIMUzn'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['ShftGagL'] = float_read(f.readline().split()[0])
        self.fst_vt['Fst7']['NTwGages'] = int(f.readline().split()[0])
        twrg = f.readline().split(',')
        if self.fst_vt['Fst7']['NTwGages'] != 0: #loop over elements if there are gauges to be added, otherwise assign directly
            for i in range(self.fst_vt['Fst7']['NTwGages']):
                self.fst_vt['Fst7']['TwrGagNd'].append(twrg[i])
            self.fst_vt['Fst7']['TwrGagNd'][-1]  = self.fst_vt['Fst7']['TwrGagNd'][-1][0:2]
        else:
            self.fst_vt['Fst7']['TwrGagNd'] = twrg
            self.fst_vt['Fst7']['TwrGagNd'][-1]  = self.fst_vt['Fst7']['TwrGagNd'][-1][0:4]
        self.fst_vt['Fst7']['NBlGages'] = int(f.readline().split()[0])
        blg = f.readline().split(',')
        if self.fst_vt['Fst7']['NBlGages'] != 0:
            for i in range(self.fst_vt['Fst7']['NBlGages']):
                self.fst_vt['Fst7']['BldGagNd'].append(blg[i])
            self.fst_vt['Fst7']['BldGagNd'][-1]  = self.fst_vt['Fst7']['BldGagNd'][-1][0:2]
        else:
            self.fst_vt['Fst7']['BldGagNd'] = blg
            self.fst_vt['Fst7']['BldGagNd'][-1]  = self.fst_vt['Fst7']['BldGagNd'][-1][0:4]
    
        # Outlist (TODO - detailed categorization)
        f.readline()
        data = f.readline()
        while data.split()[0] != 'END':
            channels = data.split('"')
            channel_list = channels[1].split(',')
            self.set_outlist(self.fst_vt['outlist7'], channel_list)
            data = f.readline()

    def read_AeroDyn_FAST7(self):

        ad_file = os.path.join(self.FAST_directory, self.fst_vt['Fst7']['ADFile'])
        f = open(ad_file)

        # skip lines and check if nondimensional
        f.readline()
        self.fst_vt['AeroDyn14']['SysUnits'] = f.readline().split()[0]
        self.fst_vt['AeroDyn14']['StallMod'] = f.readline().split()[0]
        self.fst_vt['AeroDyn14']['UseCm'] = f.readline().split()[0]
        self.fst_vt['AeroDyn14']['InfModel'] = f.readline().split()[0]
        self.fst_vt['AeroDyn14']['IndModel'] = f.readline().split()[0]
        self.fst_vt['AeroDyn14']['AToler'] = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['TLModel'] = f.readline().split()[0]
        self.fst_vt['AeroDyn14']['HLModel'] = f.readline().split()[0]
        self.fst_vt['AeroDyn14']['WindFile'] = os.path.normpath(os.path.join(os.path.split(ad_file)[0], f.readline().split()[0][1:-1]))
        self.fst_vt['AeroDyn14']['wind_file_type'] = self.fst_vt['AeroDyn14']['WindFile'].split('.')
        self.fst_vt['AeroDyn14']['HH'] = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['TwrShad'] = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['ShadHWid'] = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['T_Shad_Refpt'] = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['AirDens'] = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['KinVisc'] = float_read(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['DTAero'] = float_read(f.readline().split()[0])

        self.fst_vt['AeroDyn14']['NumFoil'] = int(f.readline().split()[0])
        self.fst_vt['AeroDyn14']['FoilNm'] = [None] * self.fst_vt['AeroDyn14']['NumFoil']
        for i in range(self.fst_vt['AeroDyn14']['NumFoil']):
            af_filename = f.readline().split()[0]
            af_filename = fix_path(af_filename)
            self.fst_vt['AeroDyn14']['FoilNm'][i]  = af_filename[1:-1]
        
        self.fst_vt['AeroDynBlade']['BldNodes'] = int(f.readline().split()[0])
        f.readline()
        self.fst_vt['AeroDynBlade']['RNodes'] = [None] * self.fst_vt['AeroDynBlade']['BldNodes']
        self.fst_vt['AeroDynBlade']['AeroTwst'] = [None] * self.fst_vt['AeroDynBlade']['BldNodes']
        self.fst_vt['AeroDynBlade']['DRNodes'] = [None] * self.fst_vt['AeroDynBlade']['BldNodes']
        self.fst_vt['AeroDynBlade']['Chord'] = [None] * self.fst_vt['AeroDynBlade']['BldNodes']
        self.fst_vt['AeroDynBlade']['NFoil'] = [None] * self.fst_vt['AeroDynBlade']['BldNodes']
        self.fst_vt['AeroDynBlade']['PrnElm'] = [None] * self.fst_vt['AeroDynBlade']['BldNodes']       
        for i in range(self.fst_vt['AeroDynBlade']['BldNodes']):
            data = f.readline().split()
            self.fst_vt['AeroDynBlade']['RNodes'][i]  = float_read(data[0])
            self.fst_vt['AeroDynBlade']['AeroTwst'][i]  = float_read(data[1])
            self.fst_vt['AeroDynBlade']['DRNodes'][i]  = float_read(data[2])
            self.fst_vt['AeroDynBlade']['Chord'][i]  = float_read(data[3])
            self.fst_vt['AeroDynBlade']['NFoil'][i]  = int(data[4])
            self.fst_vt['AeroDynBlade']['PrnElm'][i]  = data[5]

        f.close()

        # create airfoil objects
        self.fst_vt['AeroDynBlade']['af_data'] = []
        for i in range(self.fst_vt['AeroDyn14']['NumFoil']):
             self.fst_vt['AeroDynBlade']['af_data'].append(self.read_AeroDyn14Polar(os.path.join(self.FAST_directory,self.fst_vt['AeroDyn14']['FoilNm'][i])))


if __name__=="__main__":
    
    FAST_ver = 'OpenFAST'
    read_yaml = False
    dev_branch = True

    if read_yaml:
        fast = InputReader_Common(FAST_ver=FAST_ver)
        fast.FAST_yamlfile = 'temp/OpenFAST/test.yaml'
        fast.read_yaml()

    else:
        if FAST_ver.lower() == 'fast7':
            fast = InputReader_FAST7(FAST_ver=FAST_ver)
            fast.FAST_InputFile = 'Test16.fst'   # FAST input file (ext=.fst)
            fast.FAST_directory = 'C:/Users/egaertne/WT_Codes/models/FAST_v7.02.00d-bjj/CertTest/'   # Path to fst directory files

        elif FAST_ver.lower() == 'fast8':
            fast = InputReader_OpenFAST(FAST_ver=FAST_ver)
            fast.FAST_InputFile = 'NREL5MW_onshore.fst'   # FAST input file (ext=.fst)
            fast.FAST_directory = 'C:/Users/egaertne/WT_Codes/models/FAST_v8.16.00a-bjj/ref/5mw_onshore/'   # Path to fst directory files

        elif FAST_ver.lower() == 'openfast':
            fast = InputReader_OpenFAST(FAST_ver=FAST_ver, dev_branch=dev_branch)
            fast.FAST_InputFile = '5MW_OC3Spar_DLL_WTurb_WavesIrr.fst'   # FAST input file (ext=.fst)
            fast.FAST_directory = 'C:/Users/egaertne/WT_Codes/models/openfast-dev/r-test/glue-codes/openfast/5MW_OC3Spar_DLL_WTurb_WavesIrr'   # Path to fst directory files

        fast.execute()

