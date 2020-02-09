import os, sys, copy, random, time
import operator
import yaml
import numpy as np
from functools import reduce

from wisdem.aeroelasticse.FAST_reader import InputReader_Common, InputReader_OpenFAST, InputReader_FAST7
from wisdem.aeroelasticse.FAST_vars import FstModel

from ROSCO_toolbox import utilities as ROSCO_utilities


# Builder


def auto_format(f, var):
    # Error handling for variables with 'Default' options
    if isinstance(var, str):
        f.write('{:}\n'.format(var))
    elif isinstance(var, int):
        f.write('{:3}\n'.format(var))
    elif isinstance(var, float):
        f.write('{: 2.15e}\n'.format(var))

def float_default_out(val):
    # formatted float output when 'default' is an option
    if type(val) is float:
        return '{: 22f}'.format(val)
    else:
        return '{:<22}'.format(val)

def int_default_out(val):
    # formatted int output when 'default' is an option
    if type(val) is float:
        return '{:<22d}'.format(val)
    else:
        return '{:<22}'.format(val)

# given a list of nested dictionary keys, return the dict at that point
def get_dict(vartree, branch):
    return reduce(operator.getitem, branch, vartree)

class InputWriter_Common(object):
    """ Methods for writing input files that are (relatively) unchanged across FAST versions."""

    def __init__(self, **kwargs):

        self.FAST_ver = 'OPENFAST'
        self.dev_branch = False       # branch: pullrequest/ganesh : 5b78391
        self.FAST_namingOut = None    #Master FAST file
        self.FAST_runDirectory = None #Output directory
        self.fst_vt = FstModel
        self.fst_update = {}

        # Optional population class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        super(InputWriter_Common, self).__init__()

    def write_yaml(self):
        self.FAST_yamlfile = os.path.join(self.FAST_runDirectory, self.FAST_namingOut+'.yaml')
        f = open(self.FAST_yamlfile, "w")
        yaml.dump(self.fst_vt, f)


    def update(self, fst_update={}):
        """ Change fast variables based on the user supplied values """
        if fst_update:
            self.fst_update = fst_update

        # recursively loop through fast variable levels and set them to their update values
        def loop_dict(vartree, branch):
            for var in vartree.keys():
                branch_i = copy.copy(branch)
                branch_i.append(var)
                if type(vartree[var]) is dict:
                    loop_dict(vartree[var], branch_i)
                else:
                    # try:
                    get_dict(self.fst_vt, branch_i[:-1])[branch_i[-1]] = get_dict(self.fst_update, branch_i[:-1])[branch_i[-1]]
                    # except:
                        # pass

        # make sure update dictionary is not empty
        if self.fst_update:
            # if update dictionary uses list keys, convert to nested dictionaries
            if type(list(self.fst_update.keys())[0]) in [list, tuple]:
                fst_update = copy.copy(self.fst_update)
                self.fst_update = {}
                for var_list in fst_update.keys():
                    branch = []
                    for i, var in enumerate(var_list[0:-1]):
                        if var not in get_dict(self.fst_update, branch).keys():
                            get_dict(self.fst_update, branch)[var] = {}
                        branch.append(var)

                    get_dict(self.fst_update, branch)[var_list[-1]] = fst_update[var_list]

            # set fast variables to update values
            loop_dict(self.fst_update, [])


    def write_ElastoDynBlade(self):

        self.fst_vt['ElastoDyn']['BldFile1'] = self.FAST_namingOut + '_ElastoDyn_blade.dat'
        self.fst_vt['ElastoDyn']['BldFile2'] = self.fst_vt['ElastoDyn']['BldFile1']
        self.fst_vt['ElastoDyn']['BldFile3'] = self.fst_vt['ElastoDyn']['BldFile1']
        blade_file = os.path.join(self.FAST_runDirectory,self.fst_vt['ElastoDyn']['BldFile1'])
        f = open(blade_file, 'w')

        f.write('------- ELASTODYN V1.00.* INDIVIDUAL BLADE INPUT FILE --------------------------\n')
        f.write('Generated with AeroElasticSE FAST driver\n')
        f.write('---------------------- BLADE PARAMETERS ----------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['NBlInpSt'], 'NBlInpSt', '- Number of blade input stations (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldFlDmp1'], 'BldFlDmp1', '- Blade flap mode #1 structural damping in percent of critical (%)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldFlDmp2'], 'BldFlDmp2', '- Blade flap mode #2 structural damping in percent of critical (%)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldEdDmp1'], 'BldEdDmp1', '- Blade edge mode #1 structural damping in percent of critical (%)\n'))
        f.write('---------------------- BLADE ADJUSTMENT FACTORS --------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['FlStTunr1'], 'FlStTunr1', '- Blade flapwise modal stiffness tuner, 1st mode (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['FlStTunr2'], 'FlStTunr2', '- Blade flapwise modal stiffness tuner, 2nd mode (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['AdjBlMs'], 'AdjBlMs', '- Factor to adjust blade mass density (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['AdjFlSt'], 'AdjFlSt', '- Factor to adjust blade flap stiffness (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['AdjEdSt'], 'AdjEdSt', '- Factor to adjust blade edge stiffness (-)\n'))
        f.write('---------------------- DISTRIBUTED BLADE PROPERTIES ----------------------------\n')
        f.write('    BlFract      PitchAxis      StrcTwst       BMassDen        FlpStff        EdgStff\n')
        f.write('      (-)           (-)          (deg)          (kg/m)         (Nm^2)         (Nm^2)\n')
        BlFract   = self.fst_vt['ElastoDynBlade']['BlFract']
        PitchAxis = self.fst_vt['ElastoDynBlade']['PitchAxis']
        StrcTwst  = self.fst_vt['ElastoDynBlade']['StrcTwst']
        BMassDen  = self.fst_vt['ElastoDynBlade']['BMassDen']
        FlpStff   = self.fst_vt['ElastoDynBlade']['FlpStff']
        EdgStff   = self.fst_vt['ElastoDynBlade']['EdgStff']
        for BlFracti, PitchAxisi, StrcTwsti, BMassDeni, FlpStffi, EdgStffi in zip(BlFract, PitchAxis, StrcTwst, BMassDen, FlpStff, EdgStff):
            f.write('{: 2.15e} {: 2.15e} {: 2.15e} {: 2.15e} {: 2.15e} {: 2.15e}\n'.format(BlFracti, PitchAxisi, StrcTwsti, BMassDeni, FlpStffi, EdgStffi))
        f.write('---------------------- BLADE MODE SHAPES ---------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldFl1Sh'][0], 'BldFl1Sh(2)', '- Flap mode 1, coeff of x^2\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldFl1Sh'][1], 'BldFl1Sh(3)', '-            , coeff of x^3\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldFl1Sh'][2], 'BldFl1Sh(4)', '-            , coeff of x^4\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldFl1Sh'][3], 'BldFl1Sh(5)', '-            , coeff of x^5\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldFl1Sh'][4], 'BldFl1Sh(6)', '-            , coeff of x^6\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldFl2Sh'][0], 'BldFl2Sh(2)', '- Flap mode 2, coeff of x^2\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldFl2Sh'][1], 'BldFl2Sh(3)', '-            , coeff of x^3\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldFl2Sh'][2], 'BldFl2Sh(4)', '-            , coeff of x^4\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldFl2Sh'][3], 'BldFl2Sh(5)', '-            , coeff of x^5\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldFl2Sh'][4], 'BldFl2Sh(6)', '-            , coeff of x^6\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldEdgSh'][0], 'BldEdgSh(2)', '- Edge mode 1, coeff of x^2\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldEdgSh'][1], 'BldEdgSh(3)', '-            , coeff of x^3\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldEdgSh'][2], 'BldEdgSh(4)', '-            , coeff of x^4\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldEdgSh'][3], 'BldEdgSh(5)', '-            , coeff of x^5\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynBlade']['BldEdgSh'][4], 'BldEdgSh(6)', '-            , coeff of x^6\n'))      
         
        f.close()


    def write_ElastoDynTower(self):

        self.fst_vt['ElastoDyn']['TwrFile'] = self.FAST_namingOut + '_ElastoDyn_tower.dat'
        tower_file = os.path.join(self.FAST_runDirectory,self.fst_vt['ElastoDyn']['TwrFile'])
        f = open(tower_file, 'w')

        f.write('------- ELASTODYN V1.00.* TOWER INPUT FILE -------------------------------------\n')
        f.write('Generated with AeroElasticSE FAST driver\n')
        f.write('---------------------- TOWER PARAMETERS ----------------------------------------\n')
        if self.FAST_ver.lower() == 'fast7':
            f.write('---\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['NTwInpSt'],  'NTwInpSt', '- Number of input stations to specify tower geometry\n'))
        if self.FAST_ver.lower() == 'fast7':
            f.write('{:}\n'.format(self.fst_vt['ElastoDynTower']['CalcTMode']))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwrFADmp1'], 'TwrFADmp(1)', '- Tower 1st fore-aft mode structural damping ratio (%)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwrFADmp2'], 'TwrFADmp(2)', '- Tower 2nd fore-aft mode structural damping ratio (%)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwrSSDmp1'], 'TwrSSDmp(1)', '- Tower 1st side-to-side mode structural damping ratio (%)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwrSSDmp2'], 'TwrSSDmp(2)', '- Tower 2nd side-to-side mode structural damping ratio (%)\n'))
        f.write('---------------------- TOWER ADJUSTMUNT FACTORS --------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['FAStTunr1'], 'FAStTunr(1)', '- Tower fore-aft modal stiffness tuner, 1st mode (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['FAStTunr2'], 'FAStTunr(2)', '- Tower fore-aft modal stiffness tuner, 2nd mode (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['SSStTunr1'], 'SSStTunr(1)', '- Tower side-to-side stiffness tuner, 1st mode (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['SSStTunr2'], 'SSStTunr(2)', '- Tower side-to-side stiffness tuner, 2nd mode (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['AdjTwMa'], 'AdjTwMa', '- Factor to adjust tower mass density (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['AdjFASt'], 'AdjFASt', '- Factor to adjust tower fore-aft stiffness (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['AdjSSSt'], 'AdjSSSt', '- Factor to adjust tower side-to-side stiffness (-)\n'))
        f.write('---------------------- DISTRIBUTED TOWER PROPERTIES ----------------------------\n')
        f.write('  HtFract       TMassDen         TwFAStif       TwSSStif\n')
        f.write('   (-)           (kg/m)           (Nm^2)         (Nm^2)\n')
        HtFract   = self.fst_vt['ElastoDynTower']['HtFract']
        TMassDen  = self.fst_vt['ElastoDynTower']['TMassDen']
        TwFAStif  = self.fst_vt['ElastoDynTower']['TwFAStif']
        TwSSStif  = self.fst_vt['ElastoDynTower']['TwSSStif']
        if self.FAST_ver.lower() == 'fast7':
            gs = self.fst_vt['ElastoDynTower']['TwGJStif']
            es = self.fst_vt['ElastoDynTower']['TwEAStif']
            fi = self.fst_vt['ElastoDynTower']['TwFAIner']
            si = self.fst_vt['ElastoDynTower']['TwSSIner']
            fo = self.fst_vt['ElastoDynTower']['TwFAcgOf']
            so = self.fst_vt['ElastoDynTower']['TwSScgOf']
            for a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 in zip(HtFract, TMassDen, TwFAStif, TwSSStif, gs, es, fi, si, fo, so):
                f.write('{:.9e}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9e}\t{:.9e}\n'.\
                format(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10))  
        else:
            for HtFracti, TMassDeni, TwFAStifi, TwSSStifi in zip(HtFract, TMassDen, TwFAStif, TwSSStif):
                f.write('{: 2.15e} {: 2.15e} {: 2.15e} {: 2.15e}\n'.format(HtFracti, TMassDeni, TwFAStifi, TwSSStifi))
        f.write('---------------------- TOWER FORE-AFT MODE SHAPES ------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwFAM1Sh'][0], 'TwFAM1Sh(2)', '- Mode 1, coefficient of x^2 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwFAM1Sh'][1], 'TwFAM1Sh(3)', '-       , coefficient of x^3 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwFAM1Sh'][2], 'TwFAM1Sh(4)', '-       , coefficient of x^4 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwFAM1Sh'][3], 'TwFAM1Sh(5)', '-       , coefficient of x^5 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwFAM1Sh'][4], 'TwFAM1Sh(6)', '-       , coefficient of x^6 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwFAM2Sh'][0], 'TwFAM2Sh(2)', '- Mode 2, coefficient of x^2 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwFAM2Sh'][1], 'TwFAM2Sh(3)', '-       , coefficient of x^3 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwFAM2Sh'][2], 'TwFAM2Sh(4)', '-       , coefficient of x^4 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwFAM2Sh'][3], 'TwFAM2Sh(5)', '-       , coefficient of x^5 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwFAM2Sh'][4], 'TwFAM2Sh(6)', '-       , coefficient of x^6 term\n'))
        f.write('---------------------- TOWER SIDE-TO-SIDE MODE SHAPES --------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwSSM1Sh'][0], 'TwSSM1Sh(2)', '- Mode 1, coefficient of x^2 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwSSM1Sh'][1], 'TwSSM1Sh(3)', '-       , coefficient of x^3 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwSSM1Sh'][2], 'TwSSM1Sh(4)', '-       , coefficient of x^4 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwSSM1Sh'][3], 'TwSSM1Sh(5)', '-       , coefficient of x^5 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwSSM1Sh'][4], 'TwSSM1Sh(6)', '-       , coefficient of x^6 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwSSM2Sh'][0], 'TwSSM2Sh(2)', '- Mode 2, coefficient of x^2 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwSSM2Sh'][1], 'TwSSM2Sh(3)', '-       , coefficient of x^3 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwSSM2Sh'][2], 'TwSSM2Sh(4)', '-       , coefficient of x^4 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwSSM2Sh'][3], 'TwSSM2Sh(5)', '-       , coefficient of x^5 term\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDynTower']['TwSSM2Sh'][4], 'TwSSM2Sh(6)', '-       , coefficient of x^6 term\n'))
        
        f.close()

    def write_AeroDyn14Polar(self, filename, a_i):
        # AeroDyn v14 Airfoil Polar Input File

        f = open(filename, 'w')
        f.write('AeroDyn airfoil file, Aerodyn v14.04 formatting\n')
        f.write('Generated with AeroElasticSE FAST driver\n')

        f.write('{:9d}\t{:}'.format(self.fst_vt['AeroDynBlade']['af_data'][a_i]['number_tables'], 'Number of airfoil tables in this file\n'))
        for i in range(self.fst_vt['AeroDynBlade']['af_data'][a_i]['number_tables']):
            param = self.fst_vt['AeroDynBlade']['af_data'][a_i]['af_tables'][i]
            f.write('{:9g}\t{:}'.format(i, 'Table ID parameter\n'))
            f.write('{: f}\t{:}'.format(param['StallAngle'], 'Stall angle (deg)\n'))
            f.write('{: f}\t{:}'.format(0, 'No longer used, enter zero\n'))
            f.write('{: f}\t{:}'.format(0, 'No longer used, enter zero\n'))
            f.write('{: f}\t{:}'.format(0, 'No longer used, enter zero\n'))
            f.write('{: f}\t{:}'.format(param['ZeroCn'], 'Angle of attack for zero Cn for linear Cn curve (deg)\n'))
            f.write('{: f}\t{:}'.format(param['CnSlope'], 'Cn slope for zero lift for linear Cn curve (1/rad)\n'))
            f.write('{: f}\t{:}'.format(param['CnPosStall'], 'Cn at stall value for positive angle of attack for linear Cn curve\n'))
            f.write('{: f}\t{:}'.format(param['CnNegStall'], 'Cn at stall value for negative angle of attack for linear Cn curve\n'))
            f.write('{: f}\t{:}'.format(param['alphaCdMin'], 'Angle of attack for minimum CD (deg)\n'))
            f.write('{: f}\t{:}'.format(param['CdMin'], 'Minimum CD value\n'))
            if param['cm']:
                for a, cl, cd, cm in zip(param['alpha'], param['cl'], param['cd'], param['cm']):
                    f.write('{: 6e}  {: 6e}  {: 6e}  {: 6e}\n'.format(a, cl, cd, cm))
            else:
                for a, cl, cd in zip(param['alpha'], param['cl'], param['cd']):
                    f.write('{: 6e}  {: 6e}  {: 6e}\n'.format(a, cl, cd))
        
        f.close()

    def get_outlist(self, vartree_head, channel_list=[]):
        """ Loop through a list of output channel names, recursively find values set to True in the nested outlist dict """

        # recursively search nested dictionaries
        def loop_dict(vartree, outlist_i):
            for var in vartree.keys():
                if type(vartree[var]) is dict:
                    loop_dict(vartree[var], outlist_i)
                else:
                    if vartree[var]:
                        outlist_i.append(var)
            return outlist_i

        # if specific outlist branches are not specified, get all
        if not channel_list:
            channel_list = vartree_head.keys()

        # loop through top level of dictionary
        outlist = []
        for var in channel_list:
            var = var.replace(' ', '')
            outlist_i = []
            outlist_i = loop_dict(vartree_head[var], outlist_i)
            if outlist_i:
                outlist.append(sorted(outlist_i))

        return outlist

    def update_outlist(self, channels):
        """ Loop through a list of output channel names, recursively search the nested outlist dict and set to specified value"""
        # 'channels' is a dict of channel names as keys with the boolean value they should be set to

        # given a list of nested dictionary keys, return the dict at that point
        def get_dict(vartree, branch):
            return reduce(operator.getitem, branch, self.fst_vt['outlist'])
        # given a list of nested dictionary keys, set the value of the dict at that point
        def set_dict(vartree, branch, val):
            get_dict(vartree, branch[:-1])[branch[-1]] = val
        # recursively loop through outlist dictionaries to set output channels
        def loop_dict(vartree, search_var, val, branch):
            for var in vartree.keys():
                branch_i = copy.copy(branch)
                branch_i.append(var)
                if type(vartree[var]) is dict:
                    loop_dict(vartree[var], search_var, val, branch_i)
                else:
                    if var == search_var:
                        set_dict(self.fst_vt['outlist'], branch_i, val)

        # loop through outchannels on this line, loop through outlist dicts to set to True
        channel_list = channels.keys()
        for var in channel_list:
            val = channels[var]
            var = var.replace(' ', '')
            loop_dict(self.fst_vt['outlist'], var, val, [])


class InputWriter_OpenFAST(InputWriter_Common):

    def execute(self):
        
        if not os.path.exists(self.FAST_runDirectory):
            os.makedirs(self.FAST_runDirectory)

        self.write_ElastoDynBlade()
        self.write_ElastoDynTower()
        self.write_ElastoDyn()
        # self.write_WindWnd()
        self.write_InflowWind()
        if self.fst_vt['Fst']['CompAero'] == 1:
            self.write_AeroDyn14()
        elif self.fst_vt['Fst']['CompAero'] == 2:
            self.write_AeroDyn15()
        
        if 'DISCON_in' in self.fst_vt:
            self.write_DISCON_in()
        self.write_ServoDyn()
        
        if self.fst_vt['Fst']['CompHydro'] == 1:
            self.write_HydroDyn()
        if self.fst_vt['Fst']['CompSub'] == 1:
            self.write_SubDyn()
        if self.fst_vt['Fst']['CompMooring'] == 1:
            self.write_MAP()

        if self.fst_vt['Fst']['CompElast'] == 2:
            self.write_BeamDyn()

        self.write_MainInput()


    def write_MainInput(self):
        # Main FAST v8.16-v8.17 Input File
        # Currently no differences between FASTv8.16 and OpenFAST.

        self.FAST_InputFileOut = os.path.join(self.FAST_runDirectory, self.FAST_namingOut+'.fst')

        # Keep simple for now:
        f = open(self.FAST_InputFileOut, 'w')

        # ===== .fst Input File =====

        f.write('------- OpenFAST INPUT FILE -------------------------------------------\n')
        f.write('Generated with AeroElasticSE FAST driver\n')
        f.write('---------------------- SIMULATION CONTROL --------------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['Fst']['Echo'], 'Echo', '- Echo input data to <RootName>.ech (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['Fst']['AbortLevel']+'"', 'AbortLevel', '- Error level when simulation should abort (string) {"WARNING", "SEVERE", "FATAL"}\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['TMax'], 'TMax', '- Total run time (s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['DT'], 'DT', '- Recommended module time step (s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['InterpOrder'], 'InterpOrder', '- Interpolation order for input/output time history (-) {1=linear, 2=quadratic}\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['NumCrctn'], 'NumCrctn', '- Number of correction iterations (-) {0=explicit calculation, i.e., no corrections}\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['DT_UJac'], 'DT_UJac', '- Time between calls to get Jacobians (s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['UJacSclFact'], 'UJacSclFact', '- Scaling factor used in Jacobians (-)\n'))
        f.write('---------------------- FEATURE SWITCHES AND FLAGS ------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['CompElast'], 'CompElast', '- Compute structural dynamics (switch) {1=ElastoDyn; 2=ElastoDyn + BeamDyn for blades}\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['CompInflow'], 'CompInflow', '- Compute inflow wind velocities (switch) {0=still air; 1=InflowWind; 2=external from OpenFOAM}\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['CompAero'], 'CompAero', '- Compute aerodynamic loads (switch) {0=None; 1=AeroDyn v14; 2=AeroDyn v15}\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['CompServo'], 'CompServo', '- Compute control and electrical-drive dynamics (switch) {0=None; 1=ServoDyn}\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['CompHydro'], 'CompHydro', '- Compute hydrodynamic loads (switch) {0=None; 1=HydroDyn}\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['CompSub'], 'CompSub', '- Compute sub-structural dynamics (switch) {0=None; 1=SubDyn; 2=External Platform MCKF}\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['CompMooring'], 'CompMooring', '- Compute mooring system (switch) {0=None; 1=MAP++; 2=FEAMooring; 3=MoorDyn; 4=OrcaFlex}\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['CompIce'], 'CompIce', '- Compute ice loads (switch) {0=None; 1=IceFloe; 2=IceDyn}\n'))
        f.write('---------------------- INPUT FILES ---------------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['Fst']['EDFile']+'"', 'EDFile', '- Name of file containing ElastoDyn input parameters (quoted string)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['Fst']['BDBldFile(1)']+'"', 'BDBldFile(1)', '- Name of file containing BeamDyn input parameters for blade 1 (quoted string)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['Fst']['BDBldFile(2)']+'"', 'BDBldFile(2)', '- Name of file containing BeamDyn input parameters for blade 2 (quoted string)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['Fst']['BDBldFile(3)']+'"', 'BDBldFile(3)', '- Name of file containing BeamDyn input parameters for blade 3 (quoted string)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['Fst']['InflowFile']+'"', 'InflowFile', '- Name of file containing inflow wind input parameters (quoted string)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['Fst']['AeroFile']+'"', 'AeroFile', '- Name of file containing aerodynamic input parameters (quoted string)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['Fst']['ServoFile']+'"', 'ServoFile', '- Name of file containing control and electrical-drive input parameters (quoted string)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['Fst']['HydroFile']+'"', 'HydroFile', '- Name of file containing hydrodynamic input parameters (quoted string)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['Fst']['SubFile']+'"', 'SubFile', '- Name of file containing sub-structural input parameters (quoted string)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['Fst']['MooringFile']+'"', 'MooringFile', '- Name of file containing mooring system input parameters (quoted string)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['Fst']['IceFile']+'"', 'IceFile', '- Name of file containing ice input parameters (quoted string)\n'))
        f.write('---------------------- OUTPUT --------------------------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['Fst']['SumPrint'], 'SumPrint', '- Print summary data to "<RootName>.sum" (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['SttsTime'], 'SttsTime', '- Amount of time between screen status messages (s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['ChkptTime'], 'ChkptTime', '- Amount of time between creating checkpoint files for potential restart (s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['DT_Out'], 'DT_Out', '- Time step for tabular output (s) (or "default")\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['TStart'], 'TStart', '- Time to begin tabular output (s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['OutFileFmt'], 'OutFileFmt', '- Format for tabular (time-marching) output file (switch) {1: text file [<RootName>.out], 2: binary file [<RootName>.outb], 3: both}\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['Fst']['TabDelim'], 'TabDelim', '- Use tab delimiters in text tabular output file? (flag) {uses spaces if false}\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['Fst']['OutFmt']+'"', 'OutFmt', '- Format used for text tabular output, excluding the time channel.  Resulting field should be 10 characters. (quoted string)\n'))
        f.write('---------------------- LINEARIZATION -------------------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['Fst']['Linearize'], 'Linearize', '- Linearization analysis (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['NLinTimes'], 'NLinTimes', '- Number of times to linearize (-) [>=1] [unused if Linearize=False]\n'))
        try:
            f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['LinTimes'], 'LinTimes', '- List of times at which to linearize (s) [1 to NLinTimes] [unused if Linearize=False]\n'))
        except:
            f.write('{:<22} {:<11} {:}'.format(', '.join(['{:}'.format(ti) for ti in self.fst_vt['Fst']['LinTimes']]), 'LinTimes', '- List of times at which to linearize (s) [1 to NLinTimes] [unused if Linearize=False]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['LinInputs'], 'LinInputs', '- Inputs included in linearization (switch) {0=none; 1=standard; 2=all module inputs (debug)} [unused if Linearize=False]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['LinOutputs'], 'LinOutputs', '- Outputs included in linearization (switch) {0=none; 1=from OutList(s); 2=all module outputs (debug)} [unused if Linearize=False]\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['Fst']['LinOutJac'], 'LinOutJac', '- Include full Jacobians in linearization output (for debug) (flag) [unused if Linearize=False; used only if LinInputs=LinOutputs=2]\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['Fst']['LinOutMod'], 'LinOutMod', '- Write module-level linearization output files in addition to output for full system? (flag) [unused if Linearize=False]\n'))
        f.write('---------------------- VISUALIZATION ------------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['WrVTK'], 'WrVTK', '- VTK visualization data output: (switch) {0=none; 1=initialization data only; 2=animation}\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['VTK_type'], 'VTK_type', '- Type of VTK visualization data: (switch) {1=surfaces; 2=basic meshes (lines/points); 3=all meshes (debug)} [unused if WrVTK=0]\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['Fst']['VTK_fields'], 'VTK_fields', '- Write mesh fields to VTK data files? (flag) {true/false} [unused if WrVTK=0]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['Fst']['VTK_fps'], 'VTK_fps', '- Frame rate for VTK output (frames per second){will use closest integer multiple of DT} [used only if WrVTK=2]\n'))

        f.close()


    def write_ElastoDyn(self):

        self.fst_vt['Fst']['EDFile'] = self.FAST_namingOut + '_ElastoDyn.dat'
        ed_file = os.path.join(self.FAST_runDirectory,self.fst_vt['Fst']['EDFile'])
        f = open(ed_file, 'w')

        f.write('------- ELASTODYN v1.03.* INPUT FILE -------------------------------------------\n')
        f.write('Generated with AeroElasticSE FAST driver\n')

        # ElastoDyn Simulation Control (ed_sim_ctrl)
        f.write('---------------------- SIMULATION CONTROL --------------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['Echo'], 'Echo', '- Echo input data to "<RootName>.ech" (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['Method'], 'Method', '- Integration method: {1: RK4, 2: AB4, or 3: ABM4} (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['DT'], 'DT', 'Integration time step (s)\n'))
        f.write('---------------------- ENVIRONMENTAL CONDITION ---------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['Gravity'], 'Gravity', '- Gravitational acceleration (m/s^2)\n'))
        f.write('---------------------- DEGREES OF FREEDOM --------------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['FlapDOF1'], 'FlapDOF1', '- First flapwise blade mode DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['FlapDOF2'], 'FlapDOF2', '- Second flapwise blade mode DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['EdgeDOF'], 'EdgeDOF', '- First edgewise blade mode DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TeetDOF'], 'TeetDOF', '- Rotor-teeter DOF (flag) [unused for 3 blades]\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['DrTrDOF'], 'DrTrDOF', '- Drivetrain rotational-flexibility DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['GenDOF'], 'GenDOF', '- Generator DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['YawDOF'], 'YawDOF', '- Yaw DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TwFADOF1'], 'TwFADOF1', '- First fore-aft tower bending-mode DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TwFADOF2'], 'TwFADOF2', '- Second fore-aft tower bending-mode DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TwSSDOF1'], 'TwSSDOF1', '- First side-to-side tower bending-mode DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TwSSDOF2'], 'TwSSDOF2', '- Second side-to-side tower bending-mode DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmSgDOF'], 'PtfmSgDOF', '- Platform horizontal surge translation DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmSwDOF'], 'PtfmSwDOF', '- Platform horizontal sway translation DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmHvDOF'], 'PtfmHvDOF', '- Platform vertical heave translation DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmRDOF'], 'PtfmRDOF', '- Platform roll tilt rotation DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmPDOF'], 'PtfmPDOF', '- Platform pitch tilt rotation DOF (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmYDOF'], 'PtfmYDOF', '- Platform yaw rotation DOF (flag)\n'))
        f.write('---------------------- INITIAL CONDITIONS --------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['OoPDefl'], 'OoPDefl', '- Initial out-of-plane blade-tip displacement (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['IPDefl'], 'IPDefl', '- Initial in-plane blade-tip deflection (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['BlPitch1'], 'BlPitch(1)', '- Blade 1 initial pitch (degrees)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['BlPitch2'], 'BlPitch(2)', '- Blade 2 initial pitch (degrees)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['BlPitch3'], 'BlPitch(3)', '- Blade 3 initial pitch (degrees) [unused for 2 blades]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TeetDefl'], 'TeetDefl', '- Initial or fixed teeter angle (degrees) [unused for 3 blades]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['Azimuth'], 'Azimuth', '- Initial azimuth angle for blade 1 (degrees)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['RotSpeed'], 'RotSpeed', '- Initial or fixed rotor speed (rpm)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['NacYaw'], 'NacYaw', '- Initial or fixed nacelle-yaw angle (degrees)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TTDspFA'], 'TTDspFA', '- Initial fore-aft tower-top displacement (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TTDspSS'], 'TTDspSS', '- Initial side-to-side tower-top displacement (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmSurge'], 'PtfmSurge', '- Initial or fixed horizontal surge translational displacement of platform (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmSway'], 'PtfmSway', '- Initial or fixed horizontal sway translational displacement of platform (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmHeave'], 'PtfmHeave', '- Initial or fixed vertical heave translational displacement of platform (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmRoll'], 'PtfmRoll', '- Initial or fixed roll tilt rotational displacement of platform (degrees)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmPitch'], 'PtfmPitch', '- Initial or fixed pitch tilt rotational displacement of platform (degrees)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmYaw'], 'PtfmYaw', '- Initial or fixed yaw rotational displacement of platform (degrees)\n'))
        f.write('---------------------- TURBINE CONFIGURATION -----------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['NumBl'], 'NumBl', '- Number of blades (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TipRad'], 'TipRad', '- The distance from the rotor apex to the blade tip (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['HubRad'], 'HubRad', '- The distance from the rotor apex to the blade root (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PreCone(1)'], 'PreCone(1)', '- Blade 1 cone angle (degrees)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PreCone(2)'], 'PreCone(2)', '- Blade 2 cone angle (degrees)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PreCone(3)'], 'PreCone(3)', '- Blade 3 cone angle (degrees) [unused for 2 blades]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['HubCM'], 'HubCM', '- Distance from rotor apex to hub mass [positive downwind] (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['UndSling'], 'UndSling', '- Undersling length [distance from teeter pin to the rotor apex] (meters) [unused for 3 blades]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['Delta3'], 'Delta3', '- Delta-3 angle for teetering rotors (degrees) [unused for 3 blades]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['AzimB1Up'], 'AzimB1Up', '- Azimuth value to use for I/O when blade 1 points up (degrees)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['OverHang'], 'OverHang', '- Distance from yaw axis to rotor apex [3 blades] or teeter pin [2 blades] (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['ShftGagL'], 'ShftGagL', '- Distance from rotor apex [3 blades] or teeter pin [2 blades] to shaft strain gages [positive for upwind rotors] (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['ShftTilt'], 'ShftTilt', '- Rotor shaft tilt angle (degrees)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['NacCMxn'], 'NacCMxn', '- Downwind distance from the tower-top to the nacelle CM (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['NacCMyn'], 'NacCMyn', '- Lateral  distance from the tower-top to the nacelle CM (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['NacCMzn'], 'NacCMzn', '- Vertical distance from the tower-top to the nacelle CM (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['NcIMUxn'], 'NcIMUxn', '- Downwind distance from the tower-top to the nacelle IMU (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['NcIMUyn'], 'NcIMUyn', '- Lateral  distance from the tower-top to the nacelle IMU (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['NcIMUzn'], 'NcIMUzn', '- Vertical distance from the tower-top to the nacelle IMU (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['Twr2Shft'], 'Twr2Shft', '- Vertical distance from the tower-top to the rotor shaft (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TowerHt'], 'TowerHt', '- Height of tower above ground level [onshore] or MSL [offshore] (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TowerBsHt'], 'TowerBsHt', '- Height of tower base above ground level [onshore] or MSL [offshore] (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmCMxt'], 'PtfmCMxt', '- Downwind distance from the ground level [onshore] or MSL [offshore] to the platform CM (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmCMyt'], 'PtfmCMyt', '- Lateral distance from the ground level [onshore] or MSL [offshore] to the platform CM (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmCMzt'], 'PtfmCMzt', '- Vertical distance from the ground level [onshore] or MSL [offshore] to the platform CM (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmRefzt'], 'PtfmRefzt', '- Vertical distance from the ground level [onshore] or MSL [offshore] to the platform reference point (meters)\n'))
        f.write('---------------------- MASS AND INERTIA ----------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TipMass(1)'], 'TipMass(1)', '- Tip-brake mass, blade 1 (kg)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TipMass(2)'], 'TipMass(2)', '- Tip-brake mass, blade 2 (kg)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TipMass(3)'], 'TipMass(3)', '- Tip-brake mass, blade 3 (kg) [unused for 2 blades]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['HubMass'], 'HubMass', '- Hub mass (kg)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['HubIner'], 'HubIner', '- Hub inertia about rotor axis [3 blades] or teeter axis [2 blades] (kg m^2)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['GenIner'], 'GenIner', '- Generator inertia about HSS (kg m^2)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['NacMass'], 'NacMass', '- Nacelle mass (kg)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['NacYIner'], 'NacYIner', '- Nacelle inertia about yaw axis (kg m^2)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['YawBrMass'], 'YawBrMass', '- Yaw bearing mass (kg)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmMass'], 'PtfmMass', '- Platform mass (kg)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmRIner'], 'PtfmRIner', '- Platform inertia for roll tilt rotation about the platform CM (kg m^2)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmPIner'], 'PtfmPIner', '- Platform inertia for pitch tilt rotation about the platform CM (kg m^2)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['PtfmYIner'], 'PtfmYIner', '- Platform inertia for yaw rotation about the platform CM (kg m^2)\n'))
        f.write('---------------------- BLADE ---------------------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['BldNodes'], 'BldNodes', '- Number of blade nodes (per blade) used for analysis (-)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['ElastoDyn']['BldFile1']+'"', 'BldFile1', '- Name of file containing properties for blade 1 (quoted string)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['ElastoDyn']['BldFile2']+'"', 'BldFile2', '- Name of file containing properties for blade 2 (quoted string)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['ElastoDyn']['BldFile3']+'"', 'BldFile3', '- Name of file containing properties for blade 3 (quoted string) [unused for 2 blades]\n'))
        f.write('---------------------- ROTOR-TEETER --------------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TeetMod'], 'TeetMod', '- Rotor-teeter spring/damper model {0: none, 1: standard, 2: user-defined from routine UserTeet} (switch) [unused for 3 blades]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TeetDmpP'], 'TeetDmpP', '- Rotor-teeter damper position (degrees) [used only for 2 blades and when TeetMod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TeetDmp'], 'TeetDmp', '- Rotor-teeter damping constant (N-m/(rad/s)) [used only for 2 blades and when TeetMod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TeetCDmp'], 'TeetCDmp', '- Rotor-teeter rate-independent Coulomb-damping moment (N-m) [used only for 2 blades and when TeetMod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TeetSStP'], 'TeetSStP', '- Rotor-teeter soft-stop position (degrees) [used only for 2 blades and when TeetMod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TeetHStP'], 'TeetHStP', '- Rotor-teeter hard-stop position (degrees) [used only for 2 blades and when TeetMod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TeetSSSp'], 'TeetSSSp', '- Rotor-teeter soft-stop linear-spring constant (N-m/rad) [used only for 2 blades and when TeetMod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TeetHSSp'], 'TeetHSSp', '- Rotor-teeter hard-stop linear-spring constant (N-m/rad) [used only for 2 blades and when TeetMod=1]\n'))
        f.write('---------------------- DRIVETRAIN ----------------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['GBoxEff'], 'GBoxEff', '- Gearbox efficiency (%)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['GBRatio'], 'GBRatio', '- Gearbox ratio (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['DTTorSpr'], 'DTTorSpr', '- Drivetrain torsional spring (N-m/rad)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['DTTorDmp'], 'DTTorDmp', '- Drivetrain torsional damper (N-m/(rad/s))\n'))
        f.write('---------------------- FURLING -------------------------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['Furling'], 'Furling', '- Read in additional model properties for furling turbine (flag) [must currently be FALSE)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['ElastoDyn']['FurlFile']+'"', 'FurlFile', '- Name of file containing furling properties (quoted string) [unused when Furling=False]\n'))
        f.write('---------------------- TOWER ---------------------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TwrNodes'], 'TwrNodes', '- Number of tower nodes used for analysis (-)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['ElastoDyn']['TwrFile']+'"', 'TwrFile', '- Name of file containing tower properties (quoted string)\n'))
        f.write('---------------------- OUTPUT --------------------------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['SumPrint'], 'SumPrint', '- Print summary data to "<RootName>.sum" (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['OutFile'], 'OutFile', '- Switch to determine where output will be placed: {1: in module output file only; 2: in glue code output file only; 3: both} (currently unused)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TabDelim'], 'TabDelim', '- Use tab delimiters in text tabular output file? (flag) (currently unused)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['ElastoDyn']['OutFmt']+'"', 'OutFmt', '- Format used for text tabular output (except time).  Resulting field should be 10 characters. (quoted string) (currently unused)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['TStart'], 'TStart', '- Time to begin tabular output (s) (currently unused)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['DecFact'], 'DecFact', '- Decimation factor for tabular output {1: output every time step} (-) (currently unused)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['NTwGages'], 'NTwGages', '- Number of tower nodes that have strain gages for output [0 to 9] (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(', '.join(self.fst_vt['ElastoDyn']['TwrGagNd']), 'TwrGagNd', '- List of tower nodes that have strain gages [1 to TwrNodes] (-) [unused if NTwGages=0]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ElastoDyn']['NBlGages'], 'NBlGages', '- Number of blade nodes that have strain gages for output [0 to 9] (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(', '.join(self.fst_vt['ElastoDyn']['BldGagNd']), 'BldGagNd', '- List of blade nodes that have strain gages [1 to BldNodes] (-) [unused if NBlGages=0]\n'))
        f.write('                   OutList             - The next line(s) contains a list of output parameters.  See OutListParameters.xlsx for a listing of available output channels, (-)\n')

        outlist = self.get_outlist(self.fst_vt['outlist'], ['ElastoDyn'])
        
        for channel_list in outlist:
            for i in range(len(channel_list)):
                f.write('"' + channel_list[i] + '"\n')
        
        f.write('END of input file (the word "END" must appear in the first 3 columns of this last OutList line)\n')
        f.write('---------------------------------------------------------------------------------------\n')
        f.close()


    def write_BeamDyn(self):
        self.fst_vt['Fst']['BDBldFile(1)'] = self.FAST_namingOut + '_BeamDyn.dat'
        self.fst_vt['Fst']['BDBldFile(2)'] = self.fst_vt['Fst']['BDBldFile(1)']
        self.fst_vt['Fst']['BDBldFile(3)'] = self.fst_vt['Fst']['BDBldFile(1)']

        self.write_BeamDynBlade()

        beamdyn_file = os.path.join(self.FAST_runDirectory,self.fst_vt['Fst']['BDBldFile(1)'])
        f            = open(beamdyn_file, 'w')

        f.write('--------- BEAMDYN with OpenFAST INPUT FILE -------------------------------------------\n')
        f.write('Generated with AeroElasticSE FAST driver\n')
        f.write('---------------------- SIMULATION CONTROL --------------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['BeamDyn']['Echo'], 'Echo', '- Echo input data to "<RootName>.ech" (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['BeamDyn']['QuasiStaticInit'], 'QuasiStaticInit', '- Use quasistatic pre-conditioning with centripetal accelerations in initialization (flag) [dynamic solve only]\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['BeamDyn']['rhoinf'], 'rhoinf', '- Numerical damping parameter for generalized-alpha integrator\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['BeamDyn']['quadrature'], 'quadrature', '- Quadrature method: 1=Gaussian; 2=Trapezoidal (switch)\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['BeamDyn']['refine'], 'refine', '- Refinement factor for trapezoidal quadrature (-). DEFAULT = 1 [used only when quadrature=2]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['BeamDyn']['n_fact'], 'n_fact', '- Factorization frequency (-). DEFAULT = 5\n'))
        f.write(float_default_out(self.fst_vt['BeamDyn']['DTBeam']) + '   {:<11} {:}'.format('DTBeam', '- Time step size (s).\n'))
        f.write(int_default_out(self.fst_vt['BeamDyn']['load_retries']) + '   {:<11} {:}'.format('load_retries', '- Number of factored load retries before quitting the aimulation\n'))
        f.write(int_default_out(self.fst_vt['BeamDyn']['NRMax']) + '   {:<11} {:}'.format('NRMax', '- Max number of iterations in Newton-Ralphson algorithm (-). DEFAULT = 10\n'))
        f.write(float_default_out(self.fst_vt['BeamDyn']['stop_tol']) + '   {:<11} {:}'.format('stop_tol', '- Tolerance for stopping criterion (-)\n'))
        print('----------')
        print(self.fst_vt['BeamDyn']['tngt_stf_fd'], type(self.fst_vt['BeamDyn']['tngt_stf_fd']))

        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['BeamDyn']['tngt_stf_fd'], 'tngt_stf_fd', '- Flag to use finite differenced tangent stiffness matrix (-)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['BeamDyn']['tngt_stf_comp'], 'tngt_stf_comp', '- Flag to compare analytical finite differenced tangent stiffness matrix  (-)\n'))
        f.write(float_default_out(self.fst_vt['BeamDyn']['tngt_stf_pert']) + '   {:<11} {:}'.format('tngt_stf_pert', '- perturbation size for finite differencing (-)\n'))
        f.write(float_default_out(self.fst_vt['BeamDyn']['tngt_stf_difftol']) + '   {:<11} {:}'.format('tngt_stf_difftol', '- Maximum allowable relative difference between analytical and fd tangent stiffness (-)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['BeamDyn']['RotStates'], 'RotStates', '- Orient states in the rotating frame during linearization? (flag) [used only when linearizing]\n'))
        f.write('---------------------- GEOMETRY PARAMETER --------------------------------------\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['BeamDyn']['member_total'], 'member_total', '- Total number of members (-)\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['BeamDyn']['kp_total'], 'kp_total', '- Total number of key points (-) [must be at least 3]\n'))
        for i in range(self.fst_vt['BeamDyn']['member_total']):
            mem = self.fst_vt['BeamDyn']['members'][i]
            f.write('{:<22} {:<11} {:}'.format(' '.join(['%d'%(i+1),'%d'%len(mem['kp_xr'])]), '', '- Member number; Number of key points in this member\n'))
            f.write(" ".join(['{:^21s}'.format(i) for i in ['kp_xr', 'kp_yr', 'kp_zr', 'initial_twist']])+'\n')
            f.write(" ".join(['{:^21s}'.format(i) for i in ['(m)', '(m)', '(m)', '(deg)']])+'\n')
            for j in range(len(mem['kp_xr'])):
                ln = []
                ln.append('{: 2.14e}'.format(mem['kp_xr'][j]))
                ln.append('{: 2.14e}'.format(mem['kp_yr'][j]))
                ln.append('{: 2.14e}'.format(mem['kp_zr'][j]))
                ln.append('{: 2.14e}'.format(mem['initial_twist'][j]))
                f.write(" ".join(ln) + '\n')
        f.write('---------------------- MESH PARAMETER ------------------------------------------\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['BeamDyn']['order_elem'], 'order_elem', '- Order of interpolation (basis) function (-)\n'))
        f.write('---------------------- MATERIAL PARAMETER --------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['BeamDyn']['BldFile']+'"', 'BldFile', '- Name of file containing properties for blade (quoted string)\n'))
        f.write('---------------------- PITCH ACTUATOR PARAMETERS -------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['BeamDyn']['UsePitchAct'], 'UsePitchAct', '- Whether a pitch actuator should be used (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['BeamDyn']['PitchJ'], 'PitchJ', '- Pitch actuator inertia (kg-m^2) [used only when UsePitchAct is true]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['BeamDyn']['PitchK'], 'PitchK', '- Pitch actuator stiffness (kg-m^2/s^2) [used only when UsePitchAct is true]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['BeamDyn']['PitchC'], 'PitchC', '- Pitch actuator damping (kg-m^2/s) [used only when UsePitchAct is true]\n'))
        f.write('---------------------- OUTPUTS -------------------------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['BeamDyn']['SumPrint'], 'SumPrint', '- Print summary data to "<RootName>.sum" (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['ServoDyn']['OutFmt']+'"', 'OutFmt', '- Format used for text tabular output, excluding the time channel.\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['BeamDyn']['NNodeOuts'], 'NNodeOuts', '- Number of nodes to output to file [0 - 9] (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(', '.join(self.fst_vt['BeamDyn']['OutNd']), 'OutNd', '- Nodes whose values will be output  (-)\n'))
        f.write('          OutList            - The next line(s) contains a list of output parameters. See OutListParameters.xlsx for a listing of available output channels, (-)\n')
        outlist = self.get_outlist(self.fst_vt['outlist'], ['BeamDyn'])
        for channel_list in outlist:
            for i in range(len(channel_list)):
                f.write('"' + channel_list[i] + '"\n')
        f.write('END of input file (the word "END" must appear in the first 3 columns of this last OutList line)')
        f.write('---------------------------------------------------------------------------------------')
        f.close()

        # f.write('{:<22} {:<11} {:}'.format(self.fst_vt['BeamDyn'][''], '', '\n'))
        # f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['BeamDyn'][''], '', '\n'))
        # f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['BeamDyn'][''], '', '\n'))
        # f.write('{: 2.15e} {:<11} {:}'.format(self.fst_vt['BeamDyn'][''], '', '\n'))
        # f.write(float_default_out(self.fst_vt['BeamDyn']['']) + '   {:<11} {:}'.format('', '\n'))
        # f.write(int_default_out(self.fst_vt['BeamDyn']['']) + '   {:<11} {:}'.format('', '\n'))

    def write_BeamDynBlade(self):

        # bd_blade_file = self.fst_vt['BeamDyn']['BldFile']
        bd_blade_file = os.path.abspath(os.path.join(self.FAST_runDirectory, self.FAST_namingOut + '_BeamDyn_Blade.dat'))
        self.fst_vt['BeamDyn']['BldFile'] = bd_blade_file
        f = open(bd_blade_file, 'w')

        f.write('------- BEAMDYN V1.00.* INDIVIDUAL BLADE INPUT FILE --------------------------\n')
        f.write('Generated with AeroElasticSE FAST driver\n')
        f.write('---------------------- BLADE PARAMETERS --------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['BeamDynBlade']['station_total'], 'station_total', '- Number of blade input stations (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['BeamDynBlade']['damp_type'], 'damp_type', '- Damping type: 0: no damping; 1: damped\n'))
        f.write('---------------------- DAMPING COEFFICIENT------------------------------------\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['mu1','mu2','mu3','mu4','mu5','mu6']])+'\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)','(-)','(-)','(-)','(-)','(-)']])+'\n')
        mu = [self.fst_vt['BeamDynBlade']['mu1'], self.fst_vt['BeamDynBlade']['mu2'], self.fst_vt['BeamDynBlade']['mu3'], self.fst_vt['BeamDynBlade']['mu4'], self.fst_vt['BeamDynBlade']['mu5'], self.fst_vt['BeamDynBlade']['mu6']]
        f.write(" ".join(['{:^11f}'.format(i) for i in mu])+'\n')
        f.write('---------------------- DISTRIBUTED PROPERTIES---------------------------------\n')

        for i in range(len(self.fst_vt['BeamDynBlade']['radial_stations'])):
            f.write('{: 2.15e}\n'.format(self.fst_vt['BeamDynBlade']['radial_stations'][i]))
            for j in range(6):
                f.write(" ".join(['{: 2.15e}'.format(i) for i in self.fst_vt['BeamDynBlade']['beam_stiff'][i,j,:]])+'\n')
            f.write('\n')
            for j in range(6):
                f.write(" ".join(['{: 2.15e}'.format(i) for i in self.fst_vt['BeamDynBlade']['beam_inertia'][i,j,:]])+'\n')
            f.write('\n')

        print(np.shape(self.fst_vt['BeamDynBlade']['radial_stations']))
        print(self.fst_vt['BeamDynBlade']['radial_stations'])


        f.write('\n')


    def write_InflowWind(self):
        self.fst_vt['Fst']['InflowFile'] = self.FAST_namingOut + '_InflowFile.dat'
        inflow_file = os.path.join(self.FAST_runDirectory,self.fst_vt['Fst']['InflowFile'])
        f = open(inflow_file, 'w')

        f.write('------- InflowWind v3.01.* INPUT FILE -------------------------------------------------------------------------\n')
        f.write('Generated with AeroElasticSE FAST driver\n')
        f.write('---------------------------------------------------------------------------------------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['Echo'], 'Echo', '- Echo input data to <RootName>.ech (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['WindType'], 'WindType', '- switch for wind file type (1=steady; 2=uniform; 3=binary TurbSim FF; 4=binary Bladed-style FF; 5=HAWC format; 6=User defined)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['PropagationDir'], 'PropagationDir', '- Direction of wind propagation (meteoroligical rotation from aligned with X (positive rotates towards -Y) -- degrees)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['NWindVel'], 'NWindVel', '- Number of points to output the wind velocity    (0 to 9)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['WindVxiList'], 'WindVxiList', '- List of coordinates in the inertial X direction (m)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['WindVyiList'], 'WindVyiList', '- List of coordinates in the inertial Y direction (m)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['WindVziList'], 'WindVziList', '- List of coordinates in the inertial Z direction (m)\n'))
        f.write('================== Parameters for Steady Wind Conditions [used only for WindType = 1] =========================\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['HWindSpeed'], 'HWindSpeed', '- Horizontal windspeed                            (m/s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['RefHt'], 'RefHtT1', '- Reference height for horizontal wind speed      (m)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['PLexp'], 'PLexp', '- Power law exponent                              (-)\n'))
        f.write('================== Parameters for Uniform wind file   [used only for WindType = 2] ============================\n')
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['InflowWind']['Filename']+'"', 'FilenameT2', '- Filename of time series data for uniform wind field.      (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['RefHt'], 'RefHtT2', '- Reference height for horizontal wind speed                (m)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['RefLength'], 'RefLength', '- Reference length for linear horizontal and vertical sheer (-)\n'))
        f.write('================== Parameters for Binary TurbSim Full-Field files   [used only for WindType = 3] ==============\n')
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['InflowWind']['Filename']+'"', 'FilenameT3', '- Name of the Full field wind file to use (.bts)\n'))
        f.write('================== Parameters for Binary Bladed-style Full-Field files   [used only for WindType = 4] =========\n')
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['InflowWind']['FilenameRoot']+'"', 'FilenameT4', '- Rootname of the full-field wind file to use (.wnd, .sum)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['TowerFile'], 'TowerFile', '- Have tower file (.twr) (flag)\n'))
        f.write('================== Parameters for HAWC-format binary files  [Only used with WindType = 5] =====================\n')
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['InflowWind']['FileName_u']+'"', 'FileName_u', '- name of the file containing the u-component fluctuating wind (.bin)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['InflowWind']['FileName_v']+'"', 'FileName_v', '- name of the file containing the v-component fluctuating wind (.bin)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['InflowWind']['FileName_w']+'"', 'FileName_w', '- name of the file containing the w-component fluctuating wind (.bin)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['nx'], 'nx', '- number of grids in the x direction (in the 3 files above) (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['ny'], 'ny', '- number of grids in the y direction (in the 3 files above) (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['nz'], 'nz', '- number of grids in the z direction (in the 3 files above) (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['dx'], 'dx', '- distance (in meters) between points in the x direction    (m)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['dy'], 'dy', '- distance (in meters) between points in the y direction    (m)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['dz'], 'dz', '- distance (in meters) between points in the z direction    (m)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['RefHt'], 'RefHtT5', '- reference height; the height (in meters) of the vertical center of the grid (m)\n'))
        f.write('-------------   Scaling parameters for turbulence   ---------------------------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['ScaleMethod'], 'ScaleMethod', '- Turbulence scaling method   [0 = none, 1 = direct scaling, 2 = calculate scaling factor based on a desired standard deviation]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['SFx'], 'SFx', '- Turbulence scaling factor for the x direction (-)   [ScaleMethod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['SFy'], 'SFy', '- Turbulence scaling factor for the y direction (-)   [ScaleMethod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['SFz'], 'SFz', '- Turbulence scaling factor for the z direction (-)   [ScaleMethod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['SigmaFx'], 'SigmaFx', '- Turbulence standard deviation to calculate scaling from in x direction (m/s)    [ScaleMethod=2]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['SigmaFy'], 'SigmaFy', '- Turbulence standard deviation to calculate scaling from in y direction (m/s)    [ScaleMethod=2]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['SigmaFz'], 'SigmaFz', '- Turbulence standard deviation to calculate scaling from in z direction (m/s)    [ScaleMethod=2]\n'))
        f.write('-------------   Mean wind profile parameters (added to HAWC-format files)   ---------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['URef'], 'URef', '- Mean u-component wind speed at the reference height (m/s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['WindProfile'], 'WindProfile', '- Wind profile type (0=constant;1=logarithmic,2=power law)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['PLExp'], 'PLExp', '- Power law exponent (-) (used for PL wind profile type only)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['Z0'], 'Z0', '- Surface roughness length (m) (used for LG wind profile type only)\n'))
        f.write('====================== OUTPUT ==================================================\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['InflowWind']['SumPrint'], 'SumPrint', '- Print summary data to <RootName>.IfW.sum (flag)\n'))
        f.write('OutList      - The next line(s) contains a list of output parameters.  See OutListParameters.xlsx for a listing of available output channels, (-)\n')
        
        outlist = self.get_outlist(self.fst_vt['outlist'], ['InflowWind'])
        for channel_list in outlist:
            for i in range(len(channel_list)):
                f.write('"' + channel_list[i] + '"\n')

        f.write('END of input file (the word "END" must appear in the first 3 columns of this last OutList line)\n')
        f.write('---------------------------------------------------------------------------------------\n')

        f.close()

    # def WndWindWriter(self, wndfile):

    #     wind_file = os.path.join(self.FAST_runDirectory,wndfile)
    #     f = open(wind_file, 'w')

    #     for i in range(self.fst_vt['wnd_wind']['TimeSteps']):
    #         f.write('{: 2.15e}\t{: 2.15e}\t{: 2.15e}\t{: 2.15e}\t{: 2.15e}\t{: 2.15e}\t{: 2.15e}\t{: 2.15e}\n'.format(\
    #                   self.fst_vt['wnd_wind']['Time'][i], self.fst_vt['wnd_wind']['HorSpd'][i], self.fst_vt['wnd_wind']['WindDir'][i],\
    #                   self.fst_vt['wnd_wind']['VerSpd'][i], self.fst_vt['wnd_wind']['HorShr'][i],\
    #                   self.fst_vt['wnd_wind']['VerShr'][i], self.fst_vt['wnd_wind']['LnVShr'][i], self.fst_vt['wnd_wind']['GstSpd'][i]))

    #     f.close()


    def write_AeroDyn14(self):

        # ======= Airfoil Files ========
        # make directory for airfoil files
        if not os.path.isdir(os.path.join(self.FAST_runDirectory,'AeroData')):
            try:
                os.mkdir(os.path.join(self.FAST_runDirectory,'AeroData'))
            except:
                try:
                    time.sleep(random.random())
                    if not os.path.isdir(os.path.join(self.FAST_runDirectory,'AeroData')):
                        os.mkdir(os.path.join(self.FAST_runDirectory,'AeroData'))
                except:
                    print("Error tring to make '%s'!"%os.path.join(self.FAST_runDirectory,'AeroData'))

        # create write airfoil objects to files
        for i in range(self.fst_vt['AeroDyn14']['NumFoil']):
             af_name = os.path.join(self.FAST_runDirectory, 'AeroData', 'Airfoil' + str(i) + '.dat')
             self.fst_vt['AeroDyn14']['FoilNm'][i]  = os.path.join('AeroData', 'Airfoil' + str(i) + '.dat')
             self.write_AeroDyn14Polar(af_name, i)

        self.fst_vt['Fst']['AeroFile'] = self.FAST_namingOut + '_AeroDyn14.dat'
        ad_file = os.path.join(self.FAST_runDirectory,self.fst_vt['Fst']['AeroFile'])
        f = open(ad_file,'w')

        # create Aerodyn Tower
        self.write_AeroDyn14Tower()

        # ======= Aerodyn Input File ========
        f.write('AeroDyn v14.04.* INPUT FILE\n\n')
        
        # f.write('{:}\n'.format(self.fst_vt['aerodyn']['SysUnits']))
        f.write('{:}\n'.format(self.fst_vt['AeroDyn14']['StallMod']))        
        
        f.write('{:}\n'.format(self.fst_vt['AeroDyn14']['UseCm']))
        f.write('{:}\n'.format(self.fst_vt['AeroDyn14']['InfModel']))
        f.write('{:}\n'.format(self.fst_vt['AeroDyn14']['IndModel']))
        f.write('{: 2.15e}\n'.format(self.fst_vt['AeroDyn14']['AToler']))
        f.write('{:}\n'.format(self.fst_vt['AeroDyn14']['TLModel']))
        f.write('{:}\n'.format(self.fst_vt['AeroDyn14']['HLModel']))
        f.write('{:}\n'.format(self.fst_vt['AeroDyn14']['TwrShad']))  
  
        f.write('{:}\n'.format(self.fst_vt['AeroDyn14']['TwrPotent']))  
  
        f.write('{:}\n'.format(self.fst_vt['AeroDyn14']['TwrShadow']))
        f.write('{:}\n'.format(self.fst_vt['AeroDyn14']['TwrFile']))
        f.write('{:}\n'.format(self.fst_vt['AeroDyn14']['CalcTwrAero']))  
  
        f.write('{: 2.15e}\n'.format(self.fst_vt['AeroDyn14']['AirDens']))  
  
        f.write('{: 2.15e}\n'.format(self.fst_vt['AeroDyn14']['KinVisc']))  
  
        f.write('{:2}\n'.format(self.fst_vt['AeroDyn14']['DTAero']))        
        

        f.write('{:2}\n'.format(self.fst_vt['AeroDynBlade']['NumFoil']))
        for i in range (self.fst_vt['AeroDynBlade']['NumFoil']):
            f.write('"{:}"\n'.format(self.fst_vt['AeroDynBlade']['FoilNm'][i]))

        f.write('{:2}\n'.format(self.fst_vt['AeroDynBlade']['BldNodes']))
        rnodes = self.fst_vt['AeroDynBlade']['RNodes']
        twist = self.fst_vt['AeroDynBlade']['AeroTwst']
        drnodes = self.fst_vt['AeroDynBlade']['DRNodes']
        chord = self.fst_vt['AeroDynBlade']['Chord']
        nfoil = self.fst_vt['AeroDynBlade']['NFoil']
        prnelm = self.fst_vt['AeroDynBlade']['PrnElm']
        f.write('Nodal properties\n')
        for r, t, dr, c, a, p in zip(rnodes, twist, drnodes, chord, nfoil, prnelm):
            f.write('{: 2.15e}\t{: 2.15e}\t{: 2.15e}\t{: 2.15e}\t{:5}\t{:}\n'.format(r, t, dr, c, a, p))

        f.close()        

    def write_AeroDyn14Tower(self):
        # AeroDyn v14.04 Tower
        self.fst_vt['AeroDyn14']['TwrFile'] = self.FAST_namingOut + '_AeroDyn14_tower.dat'
        filename = os.path.join(self.FAST_runDirectory, self.fst_vt['AeroDyn14']['TwrFile'])
        f = open(filename, 'w')

        f.write('AeroDyn tower file, Aerodyn v14.04 formatting\n')
        f.write('Generated with AeroElasticSE FAST driver\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDynTower']['NTwrHt'], 'NTwrHt', '- Number of tower input height stations listed (-)\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDynTower']['NTwrRe'], 'NTwrRe', '- Number of tower Re values (-)\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDynTower']['NTwrCD'], 'NTwrCD', '- Number of tower CD columns (-) Note: For current versions, this MUST be 1\n'))
        f.write('{: 2.15e} {:<11} {:}'.format(self.fst_vt['AeroDynTower']['Tower_Wake_Constant'], 'Tower_Wake_Constant', '- Tower wake constant (-) {0.0: full potential flow, 0.1: Bak model}\n'))
        f.write('---------------------- DISTRIBUTED TOWER PROPERTIES ----------------------------\n')
        f.write('TwrHtFr  TwrWid  NTwrCDCol\n')
        for HtFr, Wid, CDId in zip(self.fst_vt['AeroDynTower']['TwrHtFr'], self.fst_vt['AeroDynTower']['TwrWid'], self.fst_vt['AeroDynTower']['NTwrCDCol']):
            f.write('{: 2.15e}  {: 2.15e}   {:d}\n'.format(HtFr, Wid, int(CDId)))
        f.write('---------------------- Re v CD PROPERTIES --------------------------------------\n')
        f.write('TwrRe  '+ '  '.join(['TwrCD%d'%(i+1) for i in range(self.fst_vt['AeroDynTower']['NTwrCD'])]) +'\n')
        for Re, CD in zip(self.fst_vt['AeroDynTower']['TwrRe'], self.fst_vt['AeroDynTower']['TwrCD']):
            f.write('% 2.15e' %Re + '   '.join(['% 2.15e'%cdi for cdi in CD]) + '\n')
        
        f.close()
        
    def write_AeroDyn15(self):
        # AeroDyn v15.03

        # Generate AeroDyn v15 blade input file
        self.write_AeroDyn15Blade()

        # Generate AeroDyn v15 polars
        self.write_AeroDyn15Polar()
        
        # Generate AeroDyn v15 airfoil coordinates
        # self.write_AeroDyn15Coord()
        
        # Generate AeroDyn v15.03 input file
        self.fst_vt['Fst']['AeroFile'] = self.FAST_namingOut + '_AeroDyn15.dat'
        ad_file = os.path.join(self.FAST_runDirectory, self.fst_vt['Fst']['AeroFile'])
        f = open(ad_file, 'w')

        f.write('------- AERODYN v15.03.* INPUT FILE ------------------------------------------------\n')
        f.write('Generated with AeroElasticSE FAST driver\n')
        f.write('======  General Options  ============================================================================\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['Echo'], 'Echo', '- Echo the input to "<rootname>.AD.ech"?  (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['DTAero'], 'DTAero', '- Time interval for aerodynamic calculations {or "default"} (s)\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['WakeMod'], 'WakeMod', '- Type of wake/induction model (switch) {0=none, 1=BEMT}\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['AFAeroMod'], 'AFAeroMod', '- Type of blade airfoil aerodynamics model (switch) {1=steady model, 2=Beddoes-Leishman unsteady model} [must be 1 when linearizing]\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['TwrPotent'], 'TwrPotent', '- Type tower influence on wind based on potential flow around the tower (switch) {0=none, 1=baseline potential flow, 2=potential flow with Bak correction}\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['TwrShadow'], 'TwrShadow', '- Calculate tower influence on wind based on downstream tower shadow? (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['TwrAero'], 'TwrAero', '- Calculate tower aerodynamic loads? (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['FrozenWake'], 'FrozenWake', '- Assume frozen wake during linearization? (flag) [used only when WakeMod=1 and when linearizing]\n'))
        if self.FAST_ver.lower() != 'fast8':
            f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['CavitCheck'], 'CavitCheck', '- Perform cavitation check? (flag) TRUE will turn off unsteady aerodynamics\n'))
        f.write('======  Environmental Conditions  ===================================================================\n')
        f.write('{: 2.15e} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['AirDens'], 'AirDens', '- Air density (kg/m^3)\n'))
        f.write('{: 2.15e} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['KinVisc'], 'KinVisc', '- Kinematic air viscosity (m^2/s)\n'))
        f.write('{: 2.15e} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['SpdSound'], 'SpdSound', '- Speed of sound (m/s)\n'))
        if self.FAST_ver.lower() != 'fast8':
            f.write('{: 2.15e} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['Patm'], 'Patm', '- Atmospheric pressure (Pa) [used only when CavitCheck=True]\n'))
            f.write('{: 2.15e} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['Pvap'], 'Pvap', '- Vapour pressure of fluid (Pa) [used only when CavitCheck=True]\n'))
            f.write('{: 2.15e} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['FluidDepth'], 'FluidDepth', '- Water depth above mid-hub height (m) [used only when CavitCheck=True]\n'))
        f.write('======  Blade-Element/Momentum Theory Options  ====================================================== [used only when WakeMod=1]\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['SkewMod'], 'SkewMod', '- Type of skewed-wake correction model (switch) {1=uncoupled, 2=Pitt/Peters, 3=coupled} [used only when WakeMod=1]\n'))
        if (self.dev_branch):
            f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['SkewModFactor'], 'SkewModFactor', '- Constant used in Pitt/Peters skewed wake model {or "default" is 15/32*pi} (-) [used only when SkewMod=2; unused when WakeMod=0]\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['TipLoss'], 'TipLoss', '- Use the Prandtl tip-loss model? (flag) [used only when WakeMod=1]\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['HubLoss'], 'HubLoss', '- Use the Prandtl hub-loss model? (flag) [used only when WakeMod=1]\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['TanInd'], 'TanInd', '- Include tangential induction in BEMT calculations? (flag) [used only when WakeMod=1]\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['AIDrag'], 'AIDrag', '- Include the drag term in the axial-induction calculation? (flag) [used only when WakeMod=1]\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['TIDrag'], 'TIDrag', '- Include the drag term in the tangential-induction calculation? (flag) [used only when WakeMod=1 and TanInd=TRUE]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['IndToler'], 'IndToler', '- Convergence tolerance for BEMT nonlinear solve residual equation {or "default"} (-) [used only when WakeMod=1]\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['MaxIter'], 'MaxIter', '- Maximum number of iteration steps (-) [used only when WakeMod=1]\n'))
        if (self.dev_branch):
            f.write('======  Dynamic Blade-Element/Momentum Theory Options  ====================================================== [used only when WakeMod=1]\n')
            f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['SkewMod'], 'DBEMT_Mod', '- Type of dynamic BEMT (DBEMT) model {1=constant tau1, 2=time-dependent tau1} (-) [used only when WakeMod=2]\n'))
            f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['SkewMod'], 'tau1_const', '- Time constant for DBEMT (s) [used only when WakeMod=2 and DBEMT_Mod=1]\n'))
        f.write('======  Beddoes-Leishman Unsteady Airfoil Aerodynamics Options  ===================================== [used only when AFAeroMod=2]\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['UAMod'], 'UAMod', "Unsteady Aero Model Switch (switch) {1=Baseline model (Original), 2=Gonzalez's variant (changes in Cn,Cc,Cm), 3=Minemma/Pierce variant (changes in Cc and Cm)} [used only when AFAeroMod=2]\n"))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['FLookup'], 'FLookup', "Flag to indicate whether a lookup for f' will be calculated (TRUE) or whether best-fit exponential equations will be used (FALSE); if FALSE S1-S4 must be provided in airfoil input files (flag) [used only when AFAeroMod=2]\n"))
        f.write('======  Airfoil Information =========================================================================\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['AFTabMod'], 'AFTabMod', '- Interpolation method for multiple airfoil tables {1=1D interpolation on AoA (first table only); 2=2D interpolation on AoA and Re; 3=2D interpolation on AoA and UserProp} (-)\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['InCol_Alfa'], 'InCol_Alfa', '- The column in the airfoil tables that contains the angle of attack (-)\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['InCol_Cl'], 'InCol_Cl', '- The column in the airfoil tables that contains the lift coefficient (-)\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['InCol_Cd'], 'InCol_Cd', '- The column in the airfoil tables that contains the drag coefficient (-)\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['InCol_Cm'], 'InCol_Cm', '- The column in the airfoil tables that contains the pitching-moment coefficient; use zero if there is no Cm column (-)\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['InCol_Cpmin'], 'InCol_Cpmin', '- The column in the airfoil tables that contains the Cpmin coefficient; use zero if there is no Cpmin column (-)\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['NumAFfiles'], 'NumAFfiles', '- Number of airfoil files used (-)\n'))
        for i in range(self.fst_vt['AeroDyn15']['NumAFfiles']):
            if i == 0:
                f.write('"' + self.fst_vt['AeroDyn15']['AFNames'][i] + '"    AFNames            - Airfoil file names (NumAFfiles lines) (quoted strings)\n')
            else:
                f.write('"' + self.fst_vt['AeroDyn15']['AFNames'][i] + '"\n')
        f.write('======  Rotor/Blade Properties  =====================================================================\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['UseBlCm'], 'UseBlCm', '- Include aerodynamic pitching moment in calculations?  (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['AeroDyn15']['ADBlFile1']+'"', 'ADBlFile(1)', '- Name of file containing distributed aerodynamic properties for Blade #1 (-)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['AeroDyn15']['ADBlFile2']+'"', 'ADBlFile(2)', '- Name of file containing distributed aerodynamic properties for Blade #2 (-) [unused if NumBl < 2]\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['AeroDyn15']['ADBlFile3']+'"', 'ADBlFile(3)', '- Name of file containing distributed aerodynamic properties for Blade #3 (-) [unused if NumBl < 3]\n'))
        f.write('======  Tower Influence and Aerodynamics ============================================================= [used only when TwrPotent/=0, TwrShadow=True, or TwrAero=True]\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['NumTwrNds'], 'NumTwrNds', '- Number of tower nodes used in the analysis  (-) [used only when TwrPotent/=0, TwrShadow=True, or TwrAero=True]\n'))
        f.write('TwrElev        TwrDiam        TwrCd\n')
        f.write('(m)              (m)           (-)\n')
        for TwrElev, TwrDiam, TwrCd in zip(self.fst_vt['AeroDyn15']['TwrElev'], self.fst_vt['AeroDyn15']['TwrDiam'], self.fst_vt['AeroDyn15']['TwrCd']):
            f.write('{: 2.15e} {: 2.15e} {: 2.15e} \n'.format(TwrElev, TwrDiam, TwrCd))
        f.write('======  Tower Influence and Aerodynamics ============================================================= [used only when TwrPotent/=0, TwrShadow=True, or TwrAero=True]\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['SumPrint'], 'SumPrint', '- Generate a summary file listing input options and interpolated properties to "<rootname>.AD.sum"?  (flag)\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['NBlOuts'], 'NBlOuts', '- Number of blade node outputs [0 - 9] (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(', '.join(self.fst_vt['AeroDyn15']['BlOutNd']), 'BlOutNd', '- Blade nodes whose values will be output  (-)\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['AeroDyn15']['NTwOuts'], 'NTwOuts', '- Number of tower node outputs [0 - 9]  (-)\n'))
        f.write('{:<22} {:<11} {:}'.format(', '.join(self.fst_vt['AeroDyn15']['TwOutNd']), 'TwOutNd', '- Tower nodes whose values will be output  (-)\n'))
        f.write('                   OutList             - The next line(s) contains a list of output parameters.  See OutListParameters.xlsx for a listing of available output channels, (-)\n')

        outlist = self.get_outlist(self.fst_vt['outlist'], ['AeroDyn'])      
        for channel_list in outlist:
            for i in range(len(channel_list)):
                f.write('"' + channel_list[i] + '"\n')
        f.write('END of input file (the word "END" must appear in the first 3 columns of this last OutList line)\n')
        f.write('---------------------------------------------------------------------------------------\n')
        f.close()

    def write_AeroDyn15Blade(self):
        # AeroDyn v15.00 Blade
        self.fst_vt['AeroDyn15']['ADBlFile1'] = self.FAST_namingOut + '_AeroDyn15_blade.dat'
        self.fst_vt['AeroDyn15']['ADBlFile2'] = self.fst_vt['AeroDyn15']['ADBlFile1']
        self.fst_vt['AeroDyn15']['ADBlFile3'] = self.fst_vt['AeroDyn15']['ADBlFile1']
        filename = os.path.join(self.FAST_runDirectory, self.fst_vt['AeroDyn15']['ADBlFile1'])
        f = open(filename, 'w')

        f.write('------- AERODYN v15.00.* BLADE DEFINITION INPUT FILE -------------------------------------\n')
        f.write('Generated with AeroElasticSE FAST driver\n')
        f.write('======  Blade Properties =================================================================\n')
        f.write('{:<11d} {:<11} {:}'.format(self.fst_vt['AeroDynBlade']['NumBlNds'], 'NumBlNds', '- Number of blade nodes used in the analysis (-)\n'))
        f.write('    BlSpn        BlCrvAC        BlSwpAC        BlCrvAng       BlTwist        BlChord          BlAFID\n')
        f.write('     (m)           (m)            (m)            (deg)         (deg)           (m)              (-)\n')
        BlSpn    = self.fst_vt['AeroDynBlade']['BlSpn']
        BlCrvAC  = self.fst_vt['AeroDynBlade']['BlCrvAC']
        BlSwpAC  = self.fst_vt['AeroDynBlade']['BlSwpAC']
        BlCrvAng = self.fst_vt['AeroDynBlade']['BlCrvAng']
        BlTwist  = self.fst_vt['AeroDynBlade']['BlTwist']
        BlChord  = self.fst_vt['AeroDynBlade']['BlChord']
        BlAFID   = self.fst_vt['AeroDynBlade']['BlAFID']
        for Spn, CrvAC, SwpAC, CrvAng, Twist, Chord, AFID in zip(BlSpn, BlCrvAC, BlSwpAC, BlCrvAng, BlTwist, BlChord, BlAFID):
            f.write('{: 2.15e} {: 2.15e} {: 2.15e} {: 2.15e} {: 2.15e} {: 2.15e} {: 8d}\n'.format(Spn, CrvAC, SwpAC, CrvAng, Twist, Chord, int(AFID)))
        
        f.close()
        
    def write_AeroDyn15Polar(self):
        # Airfoil Info v1.01

        if not os.path.isdir(os.path.join(self.FAST_runDirectory,'Airfoils')):
            try:
                os.mkdir(os.path.join(self.FAST_runDirectory,'Airfoils'))
            except:
                try:
                    time.sleep(random.random())
                    if not os.path.isdir(os.path.join(self.FAST_runDirectory,'Airfoils')):
                        os.mkdir(os.path.join(self.FAST_runDirectory,'Airfoils'))
                except:
                    print("Error tring to make '%s'!"%os.path.join(self.FAST_runDirectory,'Airfoils'))


        self.fst_vt['AeroDyn15']['NumAFfiles'] = len(self.fst_vt['AeroDyn15']['af_data'])
        self.fst_vt['AeroDyn15']['AFNames'] = ['']*self.fst_vt['AeroDyn15']['NumAFfiles']

        for afi in range(int(self.fst_vt['AeroDyn15']['NumAFfiles'])):

            self.fst_vt['AeroDyn15']['AFNames'][afi] = os.path.join('Airfoils', self.FAST_namingOut + '_AeroDyn15_Polar_%02d.dat'%afi)
            af_file = os.path.join(self.FAST_runDirectory, self.fst_vt['AeroDyn15']['AFNames'][afi])
            f = open(af_file, 'w')

            f.write('! ------------ AirfoilInfo v1.01.x Input File ----------------------------------\n')
            f.write('! AeroElasticSE FAST driver\n')
            f.write('! line\n')
            f.write('! line\n')
            f.write('! ------------------------------------------------------------------------------\n')
            f.write('{:<22}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][0]['InterpOrd'], 'InterpOrd', '! Interpolation order to use for quasi-steady table lookup {1=linear; 3=cubic spline; "default"} [default=3]\n'))
            f.write('{:<22d}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][0]['NonDimArea'], 'NonDimArea', '! The non-dimensional area of the airfoil (area/chord^2) (set to 1.0 if unsure or unneeded)\n'))
            # f.write('@"AF{:02d}_Coords.txt"       {:<11} {:}'.format(afi, 'NumCoords', '! The number of coordinates in the airfoil shape file. Set to zero if coordinates not included.\n'))
            f.write('{:<22d}       {:<11} {:}'.format(0, 'NumCoords', '! The number of coordinates in the airfoil shape file. Set to zero if coordinates not included.\n'))
            # f.write('AF{:02d}_BL.txt              {:<11} {:}'.format(afi, 'BL_file', '! The file name including the boundary layer characteristics of the profile. Ignored if the aeroacoustic module is not called.\n'))
            # f.write('{:<22d}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][0]['NumTabs'], 'NumTabs', '! Number of airfoil tables in this file.  Each table must have lines for Re and Ctrl.\n'))


            # check if airfoils with multiple flaps exists.
            # if yes, allocate the number of airfoils to the respective radial stations
            if self.fst_vt['AeroDyn15']['af_data'][afi][0]['NumTabs'] > 1:
                # for tab_orig in range(self.fst_vt['AeroDyn15']['af_data'][afi][0]['NumTabs'] - 1):
                if self.fst_vt['AeroDyn15']['af_data'][afi][0]['Ctrl'] == self.fst_vt['AeroDyn15']['af_data'][afi][1]['Ctrl']:
                    num_tab = 1  # assume that all Ctrl angles of the flaps are identical if the first two are -> no flaps!
                else:
                    num_tab = self.fst_vt['AeroDyn15']['af_data'][afi][0]['NumTabs']
            else:
                num_tab = 1
            # f.write('{:<22d}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][0]['NumTabs'], 'NumTabs','! Number of airfoil tables in this file.  Each table must have lines for Re and Ctrl.\n'))
            f.write('{:<22d}   {:<11} {:}'.format(num_tab, 'NumTabs','! Number of airfoil tables in this file.  Each table must have lines for Re and Ctrl.\n'))

            # for tab in range(self.fst_vt['AeroDyn15']['af_data'][afi][0]['NumTabs']): # For writting multiple tables (different Re or Ctrl values)
            for tab in range(num_tab): # For writting multiple tables (different Re or Ctrl values)
                f.write('! ------------------------------------------------------------------------------\n')
                f.write("! data for table %i \n" % (tab + 1))
                f.write('! ------------------------------------------------------------------------------\n')
                f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['Re'], 'Re', '! Reynolds number in millions\n'))
                f.write('{:<22d}   {:<11} {:}'.format(int(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['Ctrl']), 'Ctrl', '! Control setting (must be 0 for current AirfoilInfo)\n'))
                f.write('{!s:<22}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['InclUAdata'], 'InclUAdata', '! Is unsteady aerodynamics data included in this table? If TRUE, then include 30 UA coefficients below this line\n'))
                f.write('!........................................\n')
                if self.fst_vt['AeroDyn15']['af_data'][afi][tab]['InclUAdata']:
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['alpha0'], 'alpha0', '! 0-lift angle of attack, depends on airfoil.\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['alpha1'], 'alpha1', '! Angle of attack at f=0.7, (approximately the stall angle) for AOA>alpha0. (deg)\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['alpha2'], 'alpha2', '! Angle of attack at f=0.7, (approximately the stall angle) for AOA<alpha0. (deg)\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['eta_e'], 'eta_e', '! Recovery factor in the range [0.85 - 0.95] used only for UAMOD=1, it is set to 1 in the code when flookup=True. (-)\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['C_nalpha'], 'C_nalpha', '! Slope of the 2D normal force coefficient curve. (1/rad)\n'))
                    f.write(float_default_out(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['T_f0']) + '   {:<11} {:}'.format('T_f0', '! Initial value of the time constant associated with Df in the expression of Df and f''. [default = 3]\n'))
                    f.write(float_default_out(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['T_V0']) + '   {:<11} {:}'.format('T_V0', '! Initial value of the time constant associated with the vortex lift decay process; it is used in the expression of Cvn. It depends on Re,M, and airfoil class. [default = 6]\n'))
                    f.write(float_default_out(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['T_p']) + '   {:<11} {:}'.format('T_p', '! Boundary-layer,leading edge pressure gradient time constant in the expression of Dp. It should be tuned based on airfoil experimental data. [default = 1.7]\n'))
                    f.write(float_default_out(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['T_VL']) + '   {:<11} {:}'.format('T_VL', '! Initial value of the time constant associated with the vortex advection process; it represents the non-dimensional time in semi-chords, needed for a vortex to travel from LE to trailing edge (TE); it is used in the expression of Cvn. It depends on Re, M (weakly), and airfoil. [valid range = 6 - 13, default = 11]\n'))
                    f.write(float_default_out(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['b1']) + '   {:<11} {:}'.format('b1', '! Constant in the expression of phi_alpha^c and phi_q^c.  This value is relatively insensitive for thin airfoils, but may be different for turbine airfoils. [from experimental results, defaults to 0.14]\n'))
                    f.write(float_default_out(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['b2']) + '   {:<11} {:}'.format('b2', '! Constant in the expression of phi_alpha^c and phi_q^c.  This value is relatively insensitive for thin airfoils, but may be different for turbine airfoils. [from experimental results, defaults to 0.53]\n'))
                    f.write(float_default_out(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['b5']) + '   {:<11} {:}'.format('b5', "! Constant in the expression of K'''_q,Cm_q^nc, and k_m,q.  [from  experimental results, defaults to 5]\n"))
                    f.write(float_default_out(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['A1']) + '   {:<11} {:}'.format('A1', '! Constant in the expression of phi_alpha^c and phi_q^c.  This value is relatively insensitive for thin airfoils, but may be different for turbine airfoils. [from experimental results, defaults to 0.3]\n'))
                    f.write(float_default_out(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['A2']) + '   {:<11} {:}'.format('A2', '! Constant in the expression of phi_alpha^c and phi_q^c.  This value is relatively insensitive for thin airfoils, but may be different for turbine airfoils. [from experimental results, defaults to 0.7]\n'))
                    f.write(float_default_out(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['A5']) + '   {:<11} {:}'.format('A5', "! Constant in the expression of K'''_q,Cm_q^nc, and k_m,q. [from experimental results, defaults to 1]\n"))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['S1'], 'S1', '! Constant in the f curve best-fit for alpha0<=AOA<=alpha1; by definition it depends on the airfoil. [ignored if UAMod<>1]\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['S2'], 'S2', '! Constant in the f curve best-fit for         AOA> alpha1; by definition it depends on the airfoil. [ignored if UAMod<>1]\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['S3'], 'S3', '! Constant in the f curve best-fit for alpha2<=AOA< alpha0; by definition it depends on the airfoil. [ignored if UAMod<>1]\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['S4'], 'S4', '! Constant in the f curve best-fit for         AOA< alpha2; by definition it depends on the airfoil. [ignored if UAMod<>1]\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['Cn1'], 'Cn1', '! Critical value of C0n at leading edge separation. It should be extracted from airfoil data at a given Mach and Reynolds number. It can be calculated from the static value of Cn at either the break in the pitching moment or the loss of chord force at the onset of stall. It is close to the condition of maximum lift of the airfoil at low Mach numbers.\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['Cn2'], 'Cn2', '! As Cn1 for negative AOAs.\n'))
                    # f.write('{: 22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi]['St_sh'], 'St_sh', "! Strouhal's shedding frequency constant.  [default = 0.19]\n"))
                    f.write(float_default_out(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['St_sh']) + '   {:<11} {:}'.format('St_sh', "! Strouhal's shedding frequency constant.  [default = 0.19]\n"))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['Cd0'], 'Cd0', '! 2D drag coefficient value at 0-lift.\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['Cm0'], 'Cm0', '! 2D pitching moment coefficient about 1/4-chord location, at 0-lift, positive if nose up. [If the aerodynamics coefficients table does not include a column for Cm, this needs to be set to 0.0]\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['k0'], 'k0', '! Constant in the \\hat(x)_cp curve best-fit; = (\\hat(x)_AC-0.25).  [ignored if UAMod<>1]\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['k1'], 'k1', '! Constant in the \\hat(x)_cp curve best-fit.  [ignored if UAMod<>1]\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['k2'], 'k2', '! Constant in the \\hat(x)_cp curve best-fit.  [ignored if UAMod<>1]\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['k3'], 'k3', '! Constant in the \\hat(x)_cp curve best-fit.  [ignored if UAMod<>1]\n'))
                    f.write('{:<22f}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['k1_hat'], 'k1_hat', '! Constant in the expression of Cc due to leading edge vortex effects.  [ignored if UAMod<>1]\n'))
                    f.write(float_default_out(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['x_cp_bar']) + '   {:<11} {:}'.format('x_cp_bar', '! Constant in the expression of \\hat(x)_cp^v. [ignored if UAMod<>1, default = 0.2]\n'))
                    f.write(float_default_out(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['UACutout']) + '   {:<11} {:}'.format('UACutout', '! Angle of attack above which unsteady aerodynamics are disabled (deg). [Specifying the string "Default" sets UACutout to 45 degrees]\n'))
                    f.write(float_default_out(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['filtCutOff']) + '   {:<11} {:}'.format('filtCutOff', '! Cut-off frequency (-3 dB corner frequency) for low-pass filtering the AoA input to UA, as well as the 1st and 2nd derivatives (Hz) [default = 20]\n'))

                f.write('!........................................\n')
                f.write('! Table of aerodynamics coefficients\n')
                f.write('{:<22d}   {:<11} {:}'.format(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['NumAlf'], 'NumAlf', '! Number of data lines in the following table\n'))
                f.write('!    Alpha      Cl      Cd        Cm\n')
                f.write('!    (deg)      (-)     (-)       (-)\n')

                polar_map = [self.fst_vt['AeroDyn15']['InCol_Alfa'], self.fst_vt['AeroDyn15']['InCol_Cl'], self.fst_vt['AeroDyn15']['InCol_Cd'], self.fst_vt['AeroDyn15']['InCol_Cm'], self.fst_vt['AeroDyn15']['InCol_Cpmin']]
                polar_map.remove(0)
                polar_map = [i-1 for i in polar_map]

                alpha = np.asarray(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['Alpha'])
                cl = np.asarray(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['Cl'])
                cd = np.asarray(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['Cd'])
                cm = np.asarray(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['Cm'])
                cpmin = np.asarray(self.fst_vt['AeroDyn15']['af_data'][afi][tab]['Cpmin'])
                polar = np.column_stack((alpha, cl, cd, cm, cpmin))
                polar = polar[:,polar_map]


                for row in polar:
                    f.write(' '.join(['{: 2.14e}'.format(val) for val in row])+'\n')
            
            f.close()
            
    def write_AeroDyn15Coord(self):

        self.fst_vt['AeroDyn15']['AFNames_coord'] = ['']*self.fst_vt['AeroDyn15']['NumAFfiles']
        
        for afi in range(int(self.fst_vt['AeroDyn15']['NumAFfiles'])):
            
            x     = self.fst_vt['AeroDyn15']['af_coord'][afi]['x']
            y     = self.fst_vt['AeroDyn15']['af_coord'][afi]['y']
            coord = np.vstack((x, y)).T

            self.fst_vt['AeroDyn15']['AFNames_coord'][afi] = os.path.join('Airfoils/AF%02d_Coords.txt'%afi)
            af_file = os.path.join(self.FAST_runDirectory, self.fst_vt['AeroDyn15']['AFNames_coord'][afi])
            f = open(af_file, 'w')
            
            f.write('{: 22d}   {:<11} {:}'.format(len(x)+1, 'NumCoords', '! The number of coordinates in the airfoil shape file (including an extra coordinate for airfoil reference).  Set to zero if coordinates not included.\n'))
            f.write('! ......... x-y coordinates are next if NumCoords > 0 .............\n')
            f.write('! x-y coordinate of airfoil reference\n')
            f.write('!  x/c        y/c\n')
            f.write('{: 5f}       0\n'.format(self.fst_vt['AeroDyn15']['rthick'][afi]))
            f.write('! coordinates of airfoil shape\n')
            f.write('! interpolation to 200 points\n')
            f.write('!  x/c        y/c\n')
            for row in coord:
                f.write(' '.join(['{: 2.14e}'.format(val) for val in row])+'\n')
            f.close()
    
    
    def write_ServoDyn(self):
        # ServoDyn v1.05 Input File

        self.fst_vt['Fst']['ServoFile'] = self.FAST_namingOut + '_ServoDyn.dat'
        sd_file = os.path.join(self.FAST_runDirectory,self.fst_vt['Fst']['ServoFile'])
        f = open(sd_file,'w')

        f.write('------- SERVODYN v1.05.* INPUT FILE --------------------------------------------\n')
        f.write('Generated with AeroElasticSE FAST driver\n')
        f.write('---------------------- SIMULATION CONTROL --------------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['Echo'], 'Echo', '- Echo input data to <RootName>.ech (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['DT'], 'DT', '- Communication interval for controllers (s) (or "default")\n'))
        f.write('---------------------- PITCH CONTROL -------------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['PCMode'], 'PCMode', '- Pitch control mode {0: none, 3: user-defined from routine PitchCntrl, 4: user-defined from Simulink/Labview, 5: user-defined from Bladed-style DLL} (switch)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TPCOn'], 'TPCOn', '- Time to enable active pitch control (s) [unused when PCMode=0]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TPitManS1'], 'TPitManS(1)', '- Time to start override pitch maneuver for blade 1 and end standard pitch control (s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TPitManS2'], 'TPitManS(2)', '- Time to start override pitch maneuver for blade 2 and end standard pitch control (s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TPitManS3'], 'TPitManS(3)', '- Time to start override pitch maneuver for blade 3 and end standard pitch control (s) [unused for 2 blades]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['PitManRat1'], 'PitManRat(1)', '- Pitch rate at which override pitch maneuver heads toward final pitch angle for blade 1 (deg/s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['PitManRat2'], 'PitManRat(2)', '- Pitch rate at which override pitch maneuver heads toward final pitch angle for blade 2 (deg/s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['PitManRat3'], 'PitManRat(3)', '- Pitch rate at which override pitch maneuver heads toward final pitch angle for blade 3 (deg/s) [unused for 2 blades]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['BlPitchF1'], 'BlPitchF(1)', '- Blade 1 final pitch for pitch maneuvers (degrees)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['BlPitchF2'], 'BlPitchF(2)', '- Blade 2 final pitch for pitch maneuvers (degrees)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['BlPitchF3'], 'BlPitchF(3)', '- Blade 3 final pitch for pitch maneuvers (degrees) [unused for 2 blades]\n'))
        f.write('---------------------- GENERATOR AND TORQUE CONTROL ----------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['VSContrl'], 'VSContrl', '- Variable-speed control mode {0: none, 1: simple VS, 3: user-defined from routine UserVSCont, 4: user-defined from Simulink/Labview, 5: user-defined from Bladed-style DLL} (switch)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['GenModel'], 'GenModel', '- Generator model {1: simple, 2: Thevenin, 3: user-defined from routine UserGen} (switch) [used only when VSContrl=0]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['GenEff'], 'GenEff', '- Generator efficiency [ignored by the Thevenin and user-defined generator models] (%)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['GenTiStr'], 'GenTiStr', '- Method to start the generator {T: timed using TimGenOn, F: generator speed using SpdGenOn} (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['GenTiStp'], 'GenTiStp', '- Method to stop the generator {T: timed using TimGenOf, F: when generator power = 0} (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['SpdGenOn'], 'SpdGenOn', '- Generator speed to turn on the generator for a startup (HSS speed) (rpm) [used only when GenTiStr=False]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TimGenOn'], 'TimGenOn', '- Time to turn on the generator for a startup (s) [used only when GenTiStr=True]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TimGenOf'], 'TimGenOf', '- Time to turn off the generator (s) [used only when GenTiStp=True]\n'))
        f.write('---------------------- SIMPLE VARIABLE-SPEED TORQUE CONTROL --------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['VS_RtGnSp'], 'VS_RtGnSp', '- Rated generator speed for simple variable-speed generator control (HSS side) (rpm) [used only when VSContrl=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['VS_RtTq'], 'VS_RtTq', '- Rated generator torque/constant generator torque in Region 3 for simple variable-speed generator control (HSS side) (N-m) [used only when VSContrl=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['VS_Rgn2K'], 'VS_Rgn2K', '- Generator torque constant in Region 2 for simple variable-speed generator control (HSS side) (N-m/rpm^2) [used only when VSContrl=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['VS_SlPc'], 'VS_SlPc', '- Rated generator slip percentage in Region 2 1/2 for simple variable-speed generator control (%) [used only when VSContrl=1]\n'))
        f.write('---------------------- SIMPLE INDUCTION GENERATOR ------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['SIG_SlPc'], 'SIG_SlPc', '- Rated generator slip percentage (%) [used only when VSContrl=0 and GenModel=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['SIG_SySp'], 'SIG_SySp', '- Synchronous (zero-torque) generator speed (rpm) [used only when VSContrl=0 and GenModel=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['SIG_RtTq'], 'SIG_RtTq', '- Rated torque (N-m) [used only when VSContrl=0 and GenModel=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['SIG_PORt'], 'SIG_PORt', '- Pull-out ratio (Tpullout/Trated) (-) [used only when VSContrl=0 and GenModel=1]\n'))
        f.write('---------------------- THEVENIN-EQUIVALENT INDUCTION GENERATOR -----------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TEC_Freq'], 'TEC_Freq', '- Line frequency [50 or 60] (Hz) [used only when VSContrl=0 and GenModel=2]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TEC_NPol'], 'TEC_NPol', '- Number of poles [even integer > 0] (-) [used only when VSContrl=0 and GenModel=2]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TEC_SRes'], 'TEC_SRes', '- Stator resistance (ohms) [used only when VSContrl=0 and GenModel=2]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TEC_RRes'], 'TEC_RRes', '- Rotor resistance (ohms) [used only when VSContrl=0 and GenModel=2]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TEC_VLL'], 'TEC_VLL', '- Line-to-line RMS voltage (volts) [used only when VSContrl=0 and GenModel=2]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TEC_SLR'], 'TEC_SLR', '- Stator leakage reactance (ohms) [used only when VSContrl=0 and GenModel=2]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TEC_RLR'], 'TEC_RLR', '- Rotor leakage reactance (ohms) [used only when VSContrl=0 and GenModel=2]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TEC_MR'], 'TEC_MR', '- Magnetizing reactance (ohms) [used only when VSContrl=0 and GenModel=2]\n'))
        f.write('---------------------- HIGH-SPEED SHAFT BRAKE ----------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['HSSBrMode'], 'HSSBrMode', '- HSS brake model {0: none, 1: simple, 3: user-defined from routine UserHSSBr, 4: user-defined from Simulink/Labview, 5: user-defined from Bladed-style DLL} (switch)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['THSSBrDp'], 'THSSBrDp', '- Time to initiate deployment of the HSS brake (s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['HSSBrDT'], 'HSSBrDT', '- Time for HSS-brake to reach full deployment once initiated (sec) [used only when HSSBrMode=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['HSSBrTqF'], 'HSSBrTqF', '- Fully deployed HSS-brake torque (N-m)\n'))
        f.write('---------------------- NACELLE-YAW CONTROL -------------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['YCMode'], 'YCMode', '- Yaw control mode {0: none, 3: user-defined from routine UserYawCont, 4: user-defined from Simulink/Labview, 5: user-defined from Bladed-style DLL} (switch)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TYCOn'], 'TYCOn', '- Time to enable active yaw control (s) [unused when YCMode=0]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['YawNeut'], 'YawNeut', '- Neutral yaw position--yaw spring force is zero at this yaw (degrees)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['YawSpr'], 'YawSpr', '- Nacelle-yaw spring constant (N-m/rad)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['YawDamp'], 'YawDamp', '- Nacelle-yaw damping constant (N-m/(rad/s))\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TYawManS'], 'TYawManS', '- Time to start override yaw maneuver and end standard yaw control (s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['YawManRat'], 'YawManRat', '- Yaw maneuver rate (in absolute value) (deg/s)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['NacYawF'], 'NacYawF', '- Final yaw angle for override yaw maneuvers (degrees)\n'))
        f.write('---------------------- TUNED MASS DAMPER ---------------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['CompNTMD'], 'CompNTMD', '- Compute nacelle tuned mass damper {true/false} (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['ServoDyn']['NTMDfile']+'"', 'NTMDfile', '- Name of the file for nacelle tuned mass damper (quoted string) [unused when CompNTMD is false]\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['CompTTMD'], 'CompTTMD', '- Compute tower tuned mass damper {true/false} (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['ServoDyn']['TTMDfile']+'"', 'TTMDfile', '- Name of the file for tower tuned mass damper (quoted string) [unused when CompTTMD is false]\n'))
        f.write('---------------------- BLADED INTERFACE ---------------------------------------- [used only with Bladed Interface]\n')
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['ServoDyn']['DLL_FileName']+'"', 'DLL_FileName', '- Name/location of the dynamic library {.dll [Windows] or .so [Linux]} in the Bladed-DLL format (-) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['ServoDyn']['DLL_InFile']+'"', 'DLL_InFile', '- Name of input file sent to the DLL (-) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['ServoDyn']['DLL_ProcName']+'"', 'DLL_ProcName', '- Name of procedure in DLL to be called (-) [case sensitive; used only with DLL Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['DLL_DT'], 'DLL_DT', '- Communication interval for dynamic library (s) (or "default") [used only with Bladed Interface]\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['DLL_Ramp'], 'DLL_Ramp', '- Whether a linear ramp should be used between DLL_DT time steps [introduces time shift when true] (flag) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['BPCutoff'], 'BPCutoff', '- Cuttoff frequency for low-pass filter on blade pitch from DLL (Hz) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['NacYaw_North'], 'NacYaw_North', '- Reference yaw angle of the nacelle when the upwind end points due North (deg) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['Ptch_Cntrl'], 'Ptch_Cntrl', '- Record 28: Use individual pitch control {0: collective pitch; 1: individual pitch control} (switch) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['Ptch_SetPnt'], 'Ptch_SetPnt', '- Record  5: Below-rated pitch angle set-point (deg) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['Ptch_Min'], 'Ptch_Min', '- Record  6: Minimum pitch angle (deg) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['Ptch_Max'], 'Ptch_Max', '- Record  7: Maximum pitch angle (deg) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['PtchRate_Min'], 'PtchRate_Min', '- Record  8: Minimum pitch rate (most negative value allowed) (deg/s) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['PtchRate_Max'], 'PtchRate_Max', '- Record  9: Maximum pitch rate  (deg/s) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['Gain_OM'], 'Gain_OM', '- Record 16: Optimal mode gain (Nm/(rad/s)^2) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['GenSpd_MinOM'], 'GenSpd_MinOM', '- Record 17: Minimum generator speed (rpm) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['GenSpd_MaxOM'], 'GenSpd_MaxOM', '- Record 18: Optimal mode maximum speed (rpm) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['GenSpd_Dem'], 'GenSpd_Dem', '- Record 19: Demanded generator speed above rated (rpm) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['GenTrq_Dem'], 'GenTrq_Dem', '- Record 22: Demanded generator torque above rated (Nm) [used only with Bladed Interface]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['GenPwr_Dem'], 'GenPwr_Dem', '- Record 13: Demanded power (W) [used only with Bladed Interface]\n'))
        f.write('---------------------- BLADED INTERFACE TORQUE-SPEED LOOK-UP TABLE -------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['DLL_NumTrq'], 'DLL_NumTrq', '- Record 26: No. of points in torque-speed look-up table {0 = none and use the optimal mode parameters; nonzero = ignore the optimal mode PARAMETERs by setting Record 16 to 0.0} (-) [used only with Bladed Interface]\n'))
        f.write('{:<22}\t{:<22}\n'.format("GenSpd_TLU", "GenTrq_TLU"))
        f.write('{:<22}\t{:<22}\n'.format("(rpm)", "(Nm)"))
        for i in range(self.fst_vt['ServoDyn']['DLL_NumTrq']):
            a1 = self.fst_vt['ServoDyn']['GenSpd_TLU'][i]
            a2 = self.fst_vt['ServoDyn']['GenTrq_TLU'][i]
            f.write('{:<22}\t{:<22}\n'.format(a1, a2))
        f.write('---------------------- OUTPUT --------------------------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['SumPrint'], 'SumPrint', '- Print summary data to <RootName>.sum (flag) (currently unused)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['OutFile'], 'OutFile', '- Switch to determine where output will be placed: {1: in module output file only; 2: in glue code output file only; 3: both} (currently unused)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TabDelim'], 'TabDelim', '- Use tab delimiters in text tabular output file? (flag) (currently unused)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['ServoDyn']['OutFmt']+'"', 'OutFmt', '- Format used for text tabular output (except time).  Resulting field should be 10 characters. (quoted string) (currently unused)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['ServoDyn']['TStart'], 'TStart', '- Time to begin tabular output (s) (currently unused)\n'))
        f.write('              OutList      - The next line(s) contains a list of output parameters.  See OutListParameters.xlsx for a listing of available output channels, (-)\n')
        
        outlist = self.get_outlist(self.fst_vt['outlist'], ['ServoDyn'])
        for channel_list in outlist:
            for i in range(len(channel_list)):
                f.write('"' + channel_list[i] + '"\n')

        f.write('END of input file (the word "END" must appear in the first 3 columns of this last OutList line)\n')
        f.write('---------------------------------------------------------------------------------------\n')

        f.close()

    def write_DISCON_in(self):

        # Generate Bladed style Interface controller input file, intended for ROSCO https://github.com/NREL/ROSCO_toolbox
        # file version for NREL Reference OpenSource Controller v1.0.0

        # Fill controller and turbine objects for ROSCO tuning
        # - controller
        controller = type('', (), {})()
        controller.pc_gain_schedule     = type('', (), {})()
        controller.vs_gain_schedule     = type('', (), {})()
        controller.LoggingLevel         = self.fst_vt['DISCON_in']['LoggingLevel']
        controller.F_LPFType            = self.fst_vt['DISCON_in']['F_LPFType']
        controller.F_NotchType          = self.fst_vt['DISCON_in']['F_NotchType']
        controller.IPC_ControlMode      = self.fst_vt['DISCON_in']['IPC_ControlMode']
        controller.VS_ControlMode       = self.fst_vt['DISCON_in']['VS_ControlMode']
        controller.PC_ControlMode       = self.fst_vt['DISCON_in']['PC_ControlMode']
        controller.Y_ControlMode        = self.fst_vt['DISCON_in']['Y_ControlMode']
        controller.SS_Mode              = self.fst_vt['DISCON_in']['SS_Mode']
        controller.WE_Mode              = self.fst_vt['DISCON_in']['WE_Mode']
        controller.PS_Mode              = self.fst_vt['DISCON_in']['PS_Mode']
        controller.SD_Mode              = self.fst_vt['DISCON_in']['SD_Mode']
        try:
            controller.Fl_Mode          = self.fst_vt['DISCON_in']['Fl_Mode']
        except:
            controller.Fl_Mode          = 0
        try:
            controller.Flp_Mode         = self.fst_vt['DISCON_in']['Flp_Mode']
        except:
            controller.Flp_Mode         = 0
        controller.F_LPFDamping         = self.fst_vt['DISCON_in']['F_LPFDamping']
        controller.ss_cornerfreq        = self.fst_vt['DISCON_in']['F_SSCornerFreq']
        controller.pitch_op_pc          = self.fst_vt['DISCON_in']['PC_GS_angles']
        controller.pc_gain_schedule.Kp  = self.fst_vt['DISCON_in']['PC_GS_KP']
        controller.pc_gain_schedule.Ki  = self.fst_vt['DISCON_in']['PC_GS_KI']
        controller.max_pitch            = self.fst_vt['DISCON_in']['PC_MaxPit']
        controller.min_pitch            = self.fst_vt['DISCON_in']['PC_MinPit']
        controller.vs_minspd            = self.fst_vt['DISCON_in']['VS_MinOMSpd']
        controller.vs_rgn2K             = self.fst_vt['DISCON_in']['VS_Rgn2K']
        controller.vs_refspd            = self.fst_vt['DISCON_in']['VS_RefSpd']
        controller.vs_gain_schedule.Kp  = self.fst_vt['DISCON_in']['VS_KP']
        controller.vs_gain_schedule.Ki  = self.fst_vt['DISCON_in']['VS_KI']
        controller.TSR_operational      = self.fst_vt['DISCON_in']['VS_TSRopt']
        controller.ss_vsgain            = self.fst_vt['DISCON_in']['SS_VSGain']
        controller.ss_pcgain            = self.fst_vt['DISCON_in']['SS_PCGain']
        controller.v                    = self.fst_vt['DISCON_in']['WE_FOPoles_v']
        controller.A                    = self.fst_vt['DISCON_in']['WE_FOPoles']
        # controller.ps_wind_speeds = self.fst_vt['DISCON_in']['ps_wind_speeds']
        controller.ps_min_bld_pitch     = self.fst_vt['DISCON_in']['PS_BldPitchMin']
        controller.sd_maxpit            = self.fst_vt['DISCON_in']['SD_MaxPit']
        controller.sd_cornerfreq        = self.fst_vt['DISCON_in']['SD_CornerFreq']
        try:
            controller.Kp_float         = self.fst_vt['DISCON_in']['Fl_Kp']
        except:
            controller.Kp_float         = 0.
        try:
            controller.Kp_flap          = self.fst_vt['DISCON_in']['Flp_Kp']
            controller.Ki_flap          = self.fst_vt['DISCON_in']['Flp_Ki']
            controller.flp_angle        = self.fst_vt['DISCON_in']['Flp_Angle']
        except:
            controller.Kp_flap          = 0.
            controller.Ki_flap          = 0.
            controller.flp_angle        = 0.
        turbine = type('', (), {})()
        turbine.Cp = type('', (), {})()
        turbine.Ct = type('', (), {})()
        turbine.Cq = type('', (), {})()
        turbine.rotor_radius            = self.fst_vt['DISCON_in']['WE_BladeRadius']
        turbine.v_rated                 = self.fst_vt['DISCON_in']['v_rated']
        try:
            turbine.bld_flapwise_freq   = self.fst_vt['DISCON_in']['F_FlpCornerFreq'][0]
        except:
            turbine.bld_flapwise_freq   = 0.
        turbine.bld_edgewise_freq       = self.fst_vt['DISCON_in']['F_LPFCornerFreq'] * 4.
        turbine.twr_freq                = self.fst_vt['DISCON_in']['F_NotchCornerFreq'] 
        try:
            turbine.ptfm_freq           = self.fst_vt['DISCON_in']['F_FlCornerFreq'][0]
        except:
            turbine.ptfm_freq           = 0.
        turbine.max_pitch_rate          = self.fst_vt['DISCON_in']['PC_MaxRat']
        turbine.min_pitch_rate          = self.fst_vt['DISCON_in']['PC_MinRat']
        turbine.max_torque_rate         = self.fst_vt['DISCON_in']['VS_MaxRat']
        turbine.rated_rotor_speed       = self.fst_vt['DISCON_in']['PC_RefSpd'] / self.fst_vt['DISCON_in']['WE_GearboxRatio']
        turbine.rated_power             = self.fst_vt['DISCON_in']['VS_RtPwr']
        turbine.rated_torque            = self.fst_vt['DISCON_in']['VS_RtTq']
        turbine.max_torque              = self.fst_vt['DISCON_in']['VS_MaxTq']
        turbine.TSR_operational         = self.fst_vt['DISCON_in']['VS_TSRopt']
        turbine.rho                     = self.fst_vt['DISCON_in']['WE_RhoAir']
        turbine.Ng                      = self.fst_vt['DISCON_in']['WE_GearboxRatio']
        turbine.GenEff                  = self.fst_vt['ServoDyn']['GenEff']
        turbine.J                       = self.fst_vt['DISCON_in']['WE_Jtot']
        turbine.Cp                      = self.fst_vt['DISCON_in']['Cp']
        turbine.Ct                      = self.fst_vt['DISCON_in']['Ct']
        turbine.Cq                      = self.fst_vt['DISCON_in']['Cq']
        turbine.Cp_table                = self.fst_vt['DISCON_in']['Cp_table']
        turbine.Ct_table                = self.fst_vt['DISCON_in']['Ct_table']
        turbine.Cq_table                = self.fst_vt['DISCON_in']['Cq_table']
        turbine.pitch_initial_rad       = self.fst_vt['DISCON_in']['Cp_pitch_initial_rad']
        turbine.TSR_initial             = self.fst_vt['DISCON_in']['Cp_TSR_initial']
        turbine.Cp.pitch_initial_rad    = self.fst_vt['DISCON_in']['Cp_pitch_initial_rad']
        turbine.Cp.TSR_initial          = self.fst_vt['DISCON_in']['Cp_TSR_initial']
        turbine.Ct.pitch_initial_rad    = self.fst_vt['DISCON_in']['Cp_pitch_initial_rad']
        turbine.Ct.TSR_initial          = self.fst_vt['DISCON_in']['Cp_TSR_initial']
        turbine.Cq.pitch_initial_rad    = self.fst_vt['DISCON_in']['Cp_pitch_initial_rad']
        turbine.Cq.TSR_initial          = self.fst_vt['DISCON_in']['Cp_TSR_initial']
        turbine.TurbineName             = self.fst_vt['description']
        
        # Define DISCON infile paths
        self.fst_vt['ServoDyn']['DLL_InFile'] = 'DISCON.IN'
        discon_in_file = os.path.join(self.FAST_runDirectory, self.fst_vt['ServoDyn']['DLL_InFile'])
        self.fst_vt['DISCON_in']['PerfFileName'] = self.FAST_namingOut + '_Cp_Ct_Cq.txt'
        
        # Write DISCON infiles
        file_processing = ROSCO_utilities.FileProcessing()
        file_processing.write_rotor_performance(turbine, txt_filename=os.path.join(self.FAST_runDirectory, self.fst_vt['DISCON_in']['PerfFileName']))
        file_processing.write_param_file(turbine,controller,param_file=discon_in_file, txt_filename=self.fst_vt['DISCON_in']['PerfFileName'])

    def write_HydroDyn(self):

        # Generate HydroDyn v2.03 input file
        self.fst_vt['Fst']['HydroFile'] = self.FAST_namingOut + '_HydroDyn.dat'
        hd_file = os.path.join(self.FAST_runDirectory, self.fst_vt['Fst']['HydroFile'])
        f = open(hd_file, 'w')

        f.write('------- HydroDyn v2.03.* Input File --------------------------------------------\n')
        f.write('Generated with AeroElasticSE FAST driver\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['Echo'], 'Echo', '- Echo the input file data (flag)\n'))
        f.write('---------------------- ENVIRONMENTAL CONDITIONS --------------------------------\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WtrDens'], 'WtrDens', '- Water density (kg/m^3)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WtrDpth'], 'WtrDpth', '- Water depth (meters)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['MSL2SWL'], 'MSL2SWL', '- Offset between still-water level and mean sea level (meters) [positive upward; unused when WaveMod = 6; must be zero if PotMod=1 or 2]\n'))
        f.write('---------------------- WAVES ---------------------------------------------------\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WaveMod'], 'WaveMod', '- Incident wave kinematics model {0: none=still water, 1: regular (periodic), 1P#: regular with user-specified phase, 2: JONSWAP/Pierson-Moskowitz spectrum (irregular), 3: White noise spectrum (irregular), 4: user-defined spectrum from routine UserWaveSpctrm (irregular), 5: Externally generated wave-elevation time series, 6: Externally generated full wave-kinematics time series [option 6 is invalid for PotMod/=0]} (switch)\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WaveStMod'], 'WaveStMod', '- Model for stretching incident wave kinematics to instantaneous free surface {0: none=no stretching, 1: vertical stretching, 2: extrapolation stretching, 3: Wheeler stretching} (switch) [unused when WaveMod=0 or when PotMod/=0]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WaveTMax'], 'WaveTMax', '- Analysis time for incident wave calculations (sec) [unused when WaveMod=0; determines WaveDOmega=2Pi/WaveTMax in the IFFT]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WaveDT'], 'WaveDT', '- Time step for incident wave calculations     (sec) [unused when WaveMod=0; 0.1<=WaveDT<=1.0 recommended; determines WaveOmegaMax=Pi/WaveDT in the IFFT]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WaveHs'], 'WaveHs', '- Significant wave height of incident waves (meters) [used only when WaveMod=1, 2, or 3]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WaveTp'], 'WaveTp', '- Peak-spectral period of incident waves       (sec) [used only when WaveMod=1 or 2]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WavePkShp'], 'WavePkShp', '- Peak-shape parameter of incident wave spectrum (-) or DEFAULT (string) [used only when WaveMod=2; use 1.0 for Pierson-Moskowitz]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WvLowCOff'], 'WvLowCOff', '- Low  cut-off frequency or lower frequency limit of the wave spectrum beyond which the wave spectrum is zeroed (rad/s) [unused when WaveMod=0, 1, or 6]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WvHiCOff'], 'WvHiCOff', '- High cut-off frequency or upper frequency limit of the wave spectrum beyond which the wave spectrum is zeroed (rad/s) [unused when WaveMod=0, 1, or 6]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WaveDir'], 'WaveDir', '- Incident wave propagation heading direction                         (degrees) [unused when WaveMod=0 or 6]\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WaveDirMod'], 'WaveDirMod', '- Directional spreading function {0: none, 1: COS2S}                  (-)       [only used when WaveMod=2,3, or 4]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WaveDirSpread'], 'WaveDirSpread', '- Wave direction spreading coefficient ( > 0 )                        (-)       [only used when WaveMod=2,3, or 4 and WaveDirMod=1]\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WaveNDir'], 'WaveNDir', '- Number of wave directions                                           (-)       [only used when WaveMod=2,3, or 4 and WaveDirMod=1; odd number only]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WaveDirRange'], 'WaveDirRange', '- Range of wave directions (full range: WaveDir +/- 1/2*WaveDirRange) (degrees) [only used when WaveMod=2,3,or 4 and WaveDirMod=1]\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WaveSeed1'], 'WaveSeed(1)', '- First  random seed of incident waves [-2147483648 to 2147483647]    (-)       [unused when WaveMod=0, 5, or 6]\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WaveSeed2'], 'WaveSeed(2)', '- Second random seed of incident waves [-2147483648 to 2147483647]    (-)       [unused when WaveMod=0, 5, or 6]\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WaveNDAmp'], 'WaveNDAmp', '- Flag for normally distributed amplitudes                            (flag)    [only used when WaveMod=2, 3, or 4]\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['HydroDyn']['WvKinFile']+'"', 'WvKinFile', '- Root name of externally generated wave data file(s)        (quoted string)    [used only when WaveMod=5 or 6]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['NWaveElev'], 'NWaveElev', '- Number of points where the incident wave elevations can be computed (-)       [maximum of 9 output locations]\n'))
        f.write('{:<22} {:<11} {:}'.format(", ".join(self.fst_vt['HydroDyn']['WaveElevxi']), 'WaveElevxi', '- List of xi-coordinates for points where the incident wave elevations can be output (meters) [NWaveElev points, separated by commas or white space; usused if NWaveElev = 0]\n'))
        f.write('{:<22} {:<11} {:}'.format(", ".join(self.fst_vt['HydroDyn']['WaveElevyi']), 'WaveElevyi', '- List of yi-coordinates for points where the incident wave elevations can be output (meters) [NWaveElev points, separated by commas or white space; usused if NWaveElev = 0]\n'))
        f.write('---------------------- 2ND-ORDER WAVES ----------------------------------------- [unused with WaveMod=0 or 6]\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WvDiffQTF'], 'WvDiffQTF', '- Full difference-frequency 2nd-order wave kinematics (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WvSumQTF'], 'WvSumQTF', '- Full summation-frequency  2nd-order wave kinematics (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WvLowCOffD'], 'WvLowCOffD', '- Low  frequency cutoff used in the difference-frequencies (rad/s) [Only used with a difference-frequency method]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WvHiCOffD'], 'WvHiCOffD', '- High frequency cutoff used in the difference-frequencies (rad/s) [Only used with a difference-frequency method]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WvLowCOffS'], 'WvLowCOffS', '- Low  frequency cutoff used in the summation-frequencies  (rad/s) [Only used with a summation-frequency  method]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WvHiCOffS'], 'WvHiCOffS', '- High frequency cutoff used in the summation-frequencies  (rad/s) [Only used with a summation-frequency  method]\n'))
        f.write('---------------------- CURRENT ------------------------------------------------- [unused with WaveMod=6]\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['CurrMod'], 'CurrMod', '- Current profile model {0: none=no current, 1: standard, 2: user-defined from routine UserCurrent} (switch)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['CurrSSV0'], 'CurrSSV0', '- Sub-surface current velocity at still water level  (m/s) [used only when CurrMod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['CurrSSDir'], 'CurrSSDir', '- Sub-surface current heading direction (degrees) or DEFAULT (string) [used only when CurrMod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['CurrNSRef'], 'CurrNSRef', '- Near-surface current reference depth            (meters) [used only when CurrMod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['CurrNSV0'], 'CurrNSV0', '- Near-surface current velocity at still water level (m/s) [used only when CurrMod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['CurrNSDir'], 'CurrNSDir', '- Near-surface current heading direction         (degrees) [used only when CurrMod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['CurrDIV'], 'CurrDIV', '- Depth-independent current velocity                 (m/s) [used only when CurrMod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['CurrDIDir'], 'CurrDIDir', '- Depth-independent current heading direction    (degrees) [used only when CurrMod=1]\n'))
        f.write('---------------------- FLOATING PLATFORM --------------------------------------- [unused with WaveMod=6]\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['PotMod'], 'PotMod', '- Potential-flow model {0: none=no potential flow, 1: frequency-to-time-domain transforms based on WAMIT output, 2: fluid-impulse theory (FIT)} (switch)\n'))
        f.write('{:<22} {:<11} {:}'.format('"'+self.fst_vt['HydroDyn']['PotFile']+'"', 'PotFile', '- Root name of potential-flow model data; WAMIT output files containing the linear, nondimensionalized, hydrostatic restoring matrix (.hst), frequency-dependent hydrodynamic added mass matrix and damping matrix (.1), and frequency- and direction-dependent wave excitation force vector per unit wave amplitude (.3) (quoted string) [MAKE SURE THE FREQUENCIES INHERENT IN THESE WAMIT FILES SPAN THE PHYSICALLY-SIGNIFICANT RANGE OF FREQUENCIES FOR THE GIVEN PLATFORM; THEY MUST CONTAIN THE ZERO- AND INFINITE-FREQUENCY LIMITS!]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['WAMITULEN'], 'WAMITULEN', '- Characteristic body length scale used to redimensionalize WAMIT output (meters) [only used when PotMod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['PtfmVol0'], 'PtfmVol0', '- Displaced volume of water when the platform is in its undisplaced position (m^3) [only used when PotMod=1; USE THE SAME VALUE COMPUTED BY WAMIT AS OUTPUT IN THE .OUT FILE!]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['PtfmCOBxt'], 'PtfmCOBxt', '- The xt offset of the center of buoyancy (COB) from the platform reference point (meters)  [only used when PotMod=1]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['PtfmCOByt'], 'PtfmCOByt', '- The yt offset of the center of buoyancy (COB) from the platform reference point (meters)  [only used when PotMod=1]\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['RdtnMod'], 'RdtnMod', '- Radiation memory-effect model {0: no memory-effect calculation, 1: convolution, 2: state-space} (switch) [only used when PotMod=1; STATE-SPACE REQUIRES *.ss INPUT FILE]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['RdtnTMax'], 'RdtnTMax', '- Analysis time for wave radiation kernel calculations (sec) [only used when PotMod=1; determines RdtnDOmega=Pi/RdtnTMax in the cosine transform; MAKE SURE THIS IS LONG ENOUGH FOR THE RADIATION IMPULSE RESPONSE FUNCTIONS TO DECAY TO NEAR-ZERO FOR THE GIVEN PLATFORM!]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['RdtnDT'], 'RdtnDT', '- Time step for wave radiation kernel calculations (sec) [only used when PotMod=1; DT<=RdtnDT<=0.1 recommended; determines RdtnOmegaMax=Pi/RdtnDT in the cosine transform]\n'))
        f.write('---------------------- 2ND-ORDER FLOATING PLATFORM FORCES ---------------------- [unused with WaveMod=0 or 6, or PotMod=0 or 2]\n')
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['MnDrift'], 'MnDrift', "- Mean-drift 2nd-order forces computed                                       {0: None; [7, 8, 9, 10, 11, or 12]: WAMIT file to use} [Only one of MnDrift, NewmanApp, or DiffQTF can be non-zero]\n"))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['NewmanApp'], 'NewmanApp', "- Mean- and slow-drift 2nd-order forces computed with Newman's approximation {0: None; [7, 8, 9, 10, 11, or 12]: WAMIT file to use} [Only one of MnDrift, NewmanApp, or DiffQTF can be non-zero. Used only when WaveDirMod=0]\n"))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['DiffQTF'], 'DiffQTF', "- Full difference-frequency 2nd-order forces computed with full QTF          {0: None; [10, 11, or 12]: WAMIT file to use}          [Only one of MnDrift, NewmanApp, or DiffQTF can be non-zero]\n"))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['SumQTF'], 'SumQTF', "- Full summation -frequency 2nd-order forces computed with full QTF          {0: None; [10, 11, or 12]: WAMIT file to use}\n"))
        f.write('---------------------- FLOATING PLATFORM FORCE FLAGS  -------------------------- [unused with WaveMod=6]\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['PtfmSgF'], 'PtfmSgF', '- Platform horizontal surge translation force (flag) or DEFAULT\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['PtfmSwF'], 'PtfmSwF', '- Platform horizontal sway translation force (flag) or DEFAULT\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['PtfmHvF'], 'PtfmHvF', '- Platform vertical heave translation force (flag) or DEFAULT\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['PtfmRF'], 'PtfmRF', '- Platform roll tilt rotation force (flag) or DEFAULT\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['PtfmPF'], 'PtfmPF', '- Platform pitch tilt rotation force (flag) or DEFAULT\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['PtfmYF'], 'PtfmYF', '- Platform yaw rotation force (flag) or DEFAULT\n'))
        f.write('---------------------- PLATFORM ADDITIONAL STIFFNESS AND DAMPING  --------------\n')
        f.write(" ".join(['{:14}'.format(i) for i in self.fst_vt['HydroDyn']['AddF0']])+"   AddF0    - Additional preload (N, N-m)\n")
        for j in range(6):
            ln = " ".join(['{:14}'.format(i) for i in self.fst_vt['HydroDyn']['AddCLin'][j,:]])
            if j == 0:
                ln = ln + "   AddCLin  - Additional linear stiffness (N/m, N/rad, N-m/m, N-m/rad)\n"
            else:
                ln = ln  + "\n"
            f.write(ln)
        for j in range(6):
            ln = " ".join(['{:14}'.format(i) for i in self.fst_vt['HydroDyn']['AddBLin'][j,:]])
            if j == 0:
                ln = ln + "   AddBLin  - Additional linear damping(N/(m/s), N/(rad/s), N-m/(m/s), N-m/(rad/s))\n"
            else:
                ln = ln  + "\n"
            f.write(ln)
        for j in range(6):
            ln = " ".join(['{:14}'.format(i) for i in self.fst_vt['HydroDyn']['AddBQuad'][j,:]])
            if j == 0:
                ln = ln + "   AddBQuad - Additional quadratic drag(N/(m/s)^2, N/(rad/s)^2, N-m(m/s)^2, N-m/(rad/s)^2)\n"
            else:
                ln = ln  + "\n"
            f.write(ln)
        f.write('---------------------- AXIAL COEFFICIENTS --------------------------------------\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['NAxCoef'], 'NAxCoef', '- Number of axial coefficients (-)\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['AxCoefID', 'AxCd', 'AxCa', 'AxCp']])+'\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)']*4])+'\n')
        for i in range(self.fst_vt['HydroDyn']['NAxCoef']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['AxCoefID'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['AxCd'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['AxCa'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['AxCp'][i]))
            f.write(" ".join(ln) + '\n')
        f.write('---------------------- MEMBER JOINTS -------------------------------------------\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['NJoints'], 'NJoints', '- Number of joints (-)   [must be exactly 0 or at least 2]\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['JointID', 'Jointxi', 'Jointyi', 'Jointzi', 'JointAxID', 'JointOvrlp']])+'   [JointOvrlp= 0: do nothing at joint, 1: eliminate overlaps by calculating super member]\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)', '(m)', '(m)', '(m)', '(-)', '(switch)']])+'\n')
        for i in range(self.fst_vt['HydroDyn']['NJoints']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['JointID'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['Jointxi'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['Jointyi'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['Jointzi'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['JointAxID'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['JointOvrlp'][i]))
            f.write(" ".join(ln) + '\n')
        f.write('---------------------- MEMBER CROSS-SECTION PROPERTIES -------------------------\n')
        f.write('{:<11d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['NPropSets'], 'NPropSets', '- Number of member property sets (-)\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['PropSetID', 'PropD', 'PropThck']])+'\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)', '(m)', '(m)']])+'\n')
        for i in range(self.fst_vt['HydroDyn']['NPropSets']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['PropSetID'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['PropD'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['PropThck'][i]))
            f.write(" ".join(ln) + '\n')
        f.write('---------------------- SIMPLE HYDRODYNAMIC COEFFICIENTS (model 1) --------------\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['SimplCd', 'SimplCdMG', 'SimplCa', 'SimplCaMG', 'SimplCp', 'SimplCpMG', 'SimplAxCa', 'SimplAxCaMG', 'SimplAxCp', 'SimplAxCpMG']])+'\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)']*10])+'\n')
        ln = []
        ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['SimplCd']))
        ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['SimplCdMG']))
        ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['SimplCa']))
        ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['SimplCaMG']))
        ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['SimplCp']))
        ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['SimplCpMG']))
        ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['SimplAxCa']))
        ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['SimplAxCaMG']))
        ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['SimplAxCp']))
        ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['SimplAxCpMG']))
        f.write(" ".join(ln) + '\n')
        f.write('---------------------- DEPTH-BASED HYDRODYNAMIC COEFFICIENTS (model 2) ---------\n')        
        f.write('{:<11d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['NCoefDpth'], 'NCoefDpth', '- Number of depth-dependent coefficients (-)\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['Dpth', 'DpthCd', 'DpthCdMG', 'DpthCa', 'DpthCaMG', 'DpthCp', 'DpthCpMG', 'DpthAxCa', 'DpthAxCaMG', 'DpthAxCp', 'DpthAxCpMG']])+'\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(m)', '(-)', '(-)', '(-)', '(-)', '(-)', '(-)', '(-)', '(-)', '(-)', '(-)']])+'\n')
        for i in range(self.fst_vt['HydroDyn']['NCoefDpth']):
            ln = []
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['Dpth'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['DpthCd'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['DpthCdMG'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['DpthCa'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['DpthCaMG'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['DpthCp'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['DpthCpMG'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['DpthAxCa'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['DpthAxCaMG'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['DpthAxCp'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['DpthAxCpMG'][i]))
            f.write(" ".join(ln) + '\n')
        f.write('---------------------- MEMBER-BASED HYDRODYNAMIC COEFFICIENTS (model 3) --------\n')
        f.write('{:<11d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['NCoefMembers'], 'NCoefMembers', '- Number of member-based coefficients (-)\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['MemberID_HydC', 'MemberCd1', 'MemberCd2', 'MemberCdMG1', 'MemberCdMG2', 'MemberCa1', 'MemberCa2', 'MemberCaMG1', 'MemberCaMG2', 'MemberCp1', 'MemberCp2', 'MemberCpMG1', 'MemberCpMG2', 'MemberAxCa1', 'MemberAxCa2', 'MemberAxCaMG1', 'MemberAxCaMG2', 'MemberAxCp1', 'MemberAxCp2', 'MemberAxCpMG1', 'MemberAxCpMG2']])+'\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)']*21])+'\n')
        for i in range(self.fst_vt['HydroDyn']['NCoefMembers']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['MemberID_HydC'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberCd1'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberCd2'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberCdMG1'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberCdMG2'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberCa1'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberCa2'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberCaMG1'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberCaMG2'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberCp1'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberCp2'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberCpMG1'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberCpMG2'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberAxCa1'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberAxCa2'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberAxCaMG1'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberAxCaMG2'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberAxCp1'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberAxCp2'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberAxCpMG1'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MemberAxCpMG2'][i]))
            f.write(" ".join(ln) + '\n')
        f.write('-------------------- MEMBERS -------------------------------------------------\n')
        f.write('{:<11d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['NMembers'], 'NMembers', '- Number of members (-)\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['MemberID', 'MJointID1', 'MJointID2', 'MPropSetID1', 'MPropSetID2', 'MDivSize', 'MCoefMod', 'PropPot']])+'   [MCoefMod=1: use simple coeff table, 2: use depth-based coeff table, 3: use member-based coeff table] [ PropPot/=0 if member is modeled with potential-flow theory]\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)', '(-)', '(-)', '(-)', '(-)', '(m)', '(switch)', '(flag)']])+'\n')
        for i in range(self.fst_vt['HydroDyn']['NMembers']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['MemberID'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['MJointID1'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['MJointID2'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['MPropSetID1'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['MPropSetID2'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MDivSize'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['MCoefMod'][i]))
            ln.append('{!s:^11}'.format(self.fst_vt['HydroDyn']['PropPot'][i]))
            f.write(" ".join(ln) + '\n')
        f.write("---------------------- FILLED MEMBERS ------------------------------------------\n")
        f.write('{:<11d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['NFillGroups'], 'NFillGroups', '- Number of filled member groups (-) [If FillDens = DEFAULT, then FillDens = WtrDens; FillFSLoc is related to MSL2SWL]\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['FillNumM', 'FillMList', 'FillFSLoc', 'FillDens']])+'\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)', '(-)', '(m)', '(kg/m^3)']])+'\n')
        for i in range(self.fst_vt['HydroDyn']['NFillGroups']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['FillNumM'][i]))
            ln.append(" ".join(['%d'%j for j in self.fst_vt['HydroDyn']['FillMList'][i]]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['FillFSLoc'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['FillDens'][i]))
            f.write(" ".join(ln) + '\n')
        f.write("---------------------- MARINE GROWTH -------------------------------------------\n")
        f.write('{:<11d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['NMGDepths'], 'NMGDepths', '- Number of marine-growth depths specified (-) [If FillDens = DEFAULT, then FillDens = WtrDens; FillFSLoc is related to MSL2SWL]\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['MGDpth', 'MGThck', 'MGDens']])+'\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(m)', '(m)', '(kg/m^3)']])+'\n')
        for i in range(self.fst_vt['HydroDyn']['NMGDepths']):
            ln = []
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MGDpth'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MGThck'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['MGDens'][i]))
            f.write(" ".join(ln) + '\n')
        f.write("---------------------- MEMBER OUTPUT LIST --------------------------------------\n")
        f.write('{:<11d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['NMOutputs'], 'NMOutputs', '- Number of member outputs (-) [must be < 10]\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['MemberID_out', 'NOutLoc', 'NodeLocs']])+'\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)']*3])+'\n')
        for i in range(self.fst_vt['HydroDyn']['NMOutputs']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['MemberID_out'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['HydroDyn']['NOutLoc'][i]))
            ln.append('{:^11}'.format(self.fst_vt['HydroDyn']['NodeLocs'][i]))
            f.write(" ".join(ln) + '\n')
        f.write("---------------------- JOINT OUTPUT LIST ---------------------------------------\n")
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['NJOutputs'], 'NJOutputs', '- Number of joint outputs [Must be < 10]\n'))
        f.write('{:<22} {:<11} {:}'.format(" ".join(["%d"%i for i in self.fst_vt['HydroDyn']['JOutLst']]), 'JOutLst', '- List of JointIDs which are to be output (-)[unused if NJOutputs=0]\n'))
        f.write("---------------------- OUTPUT --------------------------------------------------\n")
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['HDSum'], 'HDSum', '- Output a summary file [flag]\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['OutAll'], 'OutAll', '- Output all user-specified member and joint loads (only at each member end, not interior locations) [flag]\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['HydroDyn']['OutSwtch'], 'OutSwtch', '- Output requested channels to: [1=Hydrodyn.out, 2=GlueCode.out, 3=both files]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['OutFmt'], 'OutFmt', '- Output format for numerical results (quoted string) [not checked for validity!]\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['HydroDyn']['OutSFmt'], 'OutSFmt', '- Output format for header strings (quoted string) [not checked for validity!]\n'))
        f.write('---------------------- OUTPUT CHANNELS -----------------------------------------\n')
        outlist = self.get_outlist(self.fst_vt['outlist'], ['HydroDyn'])
        for channel_list in outlist:
            for i in range(len(channel_list)):
                f.write('"' + channel_list[i] + '"\n')
            
        f.write('END of output channels and end of file. (the word "END" must appear in the first 3 columns of this line)\n')
        
        f.close()

    def write_SubDyn(self):
        # Generate SubDyn v1.1 input file
        self.fst_vt['Fst']['SubFile'] = self.FAST_namingOut + '_SubDyn.dat'
        sd_file = os.path.join(self.FAST_runDirectory, self.fst_vt['Fst']['SubFile'])
        f = open(sd_file, 'w')

        f.write('----------- SubDyn v1.01.x MultiMember Support Structure Input File ------------\n')
        f.write('Generated with AeroElasticSE FAST driver\n')
        f.write('-------------------------- SIMULATION CONTROL  ---------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['SubDyn']['Echo'], 'Echo', '- Echo input data to "<rootname>.SD.ech" (flag)\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['SubDyn']['SDdeltaT'], 'SDdeltaT', '- Local Integration Step. If "default", the glue-code integration step will be used.\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['IntMethod'], 'IntMethod', '- Integration Method [1/2/3/4 = RK4/AB4/ABM4/AM2].\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['SubDyn']['SttcSolve'], 'SttcSolve', '- Solve dynamics about static equilibrium point\n'))
        f.write('-------------------- FEA and CRAIG-BAMPTON PARAMETERS---------------------------\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['FEMMod'], 'FEMMod', '- FEM switch: element model in the FEM. [1= Euler-Bernoulli(E-B);  2=Tapered E-B (unavailable);  3= 2-node Timoshenko;  4= 2-node tapered Timoshenko (unavailable)]\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['NDiv'], 'NDiv', '- Number of sub-elements per member\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['SubDyn']['CBMod'], 'CBMod', '- [T/F] If True perform C-B reduction, else full FEM dofs will be retained. If True, select Nmodes to retain in C-B reduced system.\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['Nmodes'], 'Nmodes', '- Number of internal modes to retain (ignored if CBMod=False). If Nmodes=0 --> Guyan Reduction.\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['JDampings'], 'JDampings', '- Damping Ratios for each retained mode (% of critical) If Nmodes>0, list Nmodes structural damping ratios for each retained mode (% of critical), or a single damping ratio to be applied to all retained modes. (last entered value will be used for all remaining modes).\n'))
        f.write('---- STRUCTURE JOINTS: joints connect structure members (~Hydrodyn Input File)---\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['NJoints'], 'NJoints', '- Number of joints (-)\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['JointID', 'JointXss', 'JointYss', 'JointZss']])+' [Coordinates of Member joints in SS-Coordinate System]\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)', '(m)', '(m)', '(m)']])+'\n')
        for i in range(self.fst_vt['SubDyn']['NJoints']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['JointID'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['JointXss'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['JointYss'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['JointZss'][i]))
            f.write(" ".join(ln) + '\n')        
        f.write('------------------- BASE REACTION JOINTS: 1/0 for Locked/Free DOF @ each Reaction Node ---------------------\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['NReact'], 'NReact', '- Number of Joints with reaction forces; be sure to remove all rigid motion DOFs of the structure  (else det([K])=[0])\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['RJointID', 'RctTDXss', 'RctTDYss', 'RctTDZss', 'RctRDXss', 'RctRDYss', 'RctRDZss']])+' [Global Coordinate System]\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)', '(flag)', '(flag)', '(flag)', '(flag)', '(flag)', '(flag)']])+'\n')
        for i in range(self.fst_vt['SubDyn']['NReact']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['RJointID'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['RctTDXss'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['RctTDYss'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['RctTDZss'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['RctRDXss'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['RctRDYss'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['RctRDZss'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['Rct_SoilFile'][i]))
            f.write(" ".join(ln) + '\n')
        f.write('------- INTERFACE JOINTS: 1/0 for Locked (to the TP)/Free DOF @each Interface Joint (only Locked-to-TP implemented thus far (=rigid TP)) ---------\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['NReact'], 'NReact', '- Number of Joints with reaction forces; be sure to remove all rigid motion DOFs of the structure  (else det([K])=[0])\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['IJointID', 'ItfTDXss', 'ItfTDYss', 'ItfTDZss', 'ItfRDXss', 'ItfRDYss', 'ItfRDZss']])+' [Global Coordinate System]\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)', '(flag)', '(flag)', '(flag)', '(flag)', '(flag)', '(flag)']])+'\n')
        for i in range(self.fst_vt['SubDyn']['NReact']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['IJointID'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['ItfTDXss'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['ItfTDYss'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['ItfTDZss'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['ItfRDXss'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['ItfRDYss'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['ItfRDZss'][i]))
            f.write(" ".join(ln) + '\n')
        f.write('----------------------------------- MEMBERS --------------------------------------\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['NMembers'], 'NMembers', '- Number of frame members\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['MemberID', 'MJointID1', 'MJointID2', 'MPropSetID1', 'MPropSetID2', 'COSMID']])+'\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)','(-)','(-)','(-)','(-)','(-)']])+'\n')
        for i in range(self.fst_vt['SubDyn']['NMembers']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['MemberID'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['MJointID1'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['MJointID2'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['MPropSetID1'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['MPropSetID2'][i]))
            if self.fst_vt['SubDyn']['NCOSMs'] > 0:
                ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['COSMID'][i]))
            f.write(" ".join(ln) + '\n')
        f.write('------------------ MEMBER X-SECTION PROPERTY data 1/2 [isotropic material for now: use this table for circular-tubular elements] ------------------------\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['NPropSets'], 'NPropSets', '- Number of structurally unique x-sections (i.e. how many groups of X-sectional properties are utilized throughout all of the members)\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['PropSetID', 'YoungE', 'ShearG1', 'MatDens', 'XsecD', 'XsecT']])+'\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)','(N/m2)','(N/m2)','(kg/m3)','(m)','(m)']])+'\n')
        for i in range(self.fst_vt['SubDyn']['NPropSets']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['PropSetID1'][i]))
            ln.append('{:^11e}'.format(self.fst_vt['SubDyn']['YoungE1'][i]))
            ln.append('{:^11e}'.format(self.fst_vt['SubDyn']['ShearG1'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['MatDens1'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['XsecD'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['XsecT'][i]))
            f.write(" ".join(ln) + '\n')
        f.write('------------------ MEMBER X-SECTION PROPERTY data 2/2 [isotropic material for now: use this table if any section other than circular, however provide COSM(i,j) below] ------------------------\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['NXPropSets'], 'NXPropSets', '- Number of structurally unique non-circular x-sections (if 0 the following table is ignored)\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['PropSetID', 'YoungE', 'ShearG2', 'MatDens', 'XsecA', 'XsecAsx', 'XsecAsy', 'XsecJxx', 'XsecJyy', 'XsecJ0']])+'\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)','(N/m2)','(N/m2)','(kg/m3)','(m2)','(m2)','(m2)','(m4)','(m4)','(m4)']])+'\n')
        for i in range(self.fst_vt['SubDyn']['NXPropSets']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['PropSetID2'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['YoungE2'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['ShearG2'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['MatDens2'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['XsecA'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['XsecAsx'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['XsecAsy'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['XsecJxx'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['XsecJyy'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['XsecJ0'][i]))
            f.write(" ".join(ln) + '\n')
        f.write('---------------------- MEMBER COSINE MATRICES COSM(i,j) ------------------------\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['NCOSMs'], 'NCOSMs', '- Number of unique cosine matrices (i.e., of unique member alignments including principal axis rotations); ignored if NXPropSets=0   or 9999 in any element below\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['COSMID', 'COSM11', 'COSM12', 'COSM13', 'COSM21', 'COSM22', 'COSM23', 'COSM31', 'COSM32', 'COSM33']])+'\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)','(-)','(-)','(-)','(-)','(-)','(-)','(-)','(-)','(-)']])+'\n')
        for i in range(self.fst_vt['SubDyn']['NCOSMs']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['COSMID'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['COSM11'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['COSM12'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['COSM13'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['COSM21'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['COSM22'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['COSM23'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['COSM31'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['COSM32'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['COSM33'][i]))
            f.write(" ".join(ln) + '\n')
        f.write('------------------------ JOINT ADDITIONAL CONCENTRATED MASSES--------------------------\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['NCmass'], 'NCmass', '- Number of joints with concentrated masses; Global Coordinate System\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['CMJointID', 'JMass', 'JMXX', 'JMYY', 'JMZZ']])+'\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)','(kg)','(kg*m^2)','(kg*m^2)','(kg*m^2)']])+'\n')
        for i in range(self.fst_vt['SubDyn']['NCmass']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['CMJointID'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['JMass'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['JMXX'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['JMYY'][i]))
            ln.append('{:^11}'.format(self.fst_vt['SubDyn']['JMZZ'][i]))
            f.write(" ".join(ln) + '\n')
        f.write('---------------------------- OUTPUT: SUMMARY & OUTFILE ------------------------------\n')
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['SubDyn']['SSSum'], 'SSSum', '- Output a Summary File (flag).It contains: matrices K,M  and C-B reduced M_BB, M-BM, K_BB, K_MM(OMG^2), PHI_R, PHI_L. It can also contain COSMs if requested.\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['SubDyn']['OutCOSM'], 'OutCOSM', '- Output cosine matrices with the selected output member forces (flag)\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['SubDyn']['OutAll'], 'OutAll', "- [T/F] Output all members' end forces\n"))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['OutSwtch'], 'OutSwtch', '- [1/2/3] Output requested channels to: 1=<rootname>.SD.out;  2=<rootname>.out (generated by FAST);  3=both files.\n'))
        f.write('{!s:<22} {:<11} {:}'.format(self.fst_vt['SubDyn']['TabDelim'], 'TabDelim', '- Generate a tab-delimited output in the <rootname>.SD.out file\n'))
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['OutDec'], 'OutDec', '- Decimation of output in the <rootname>.SD.out file\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['SubDyn']['OutFmt'], 'OutFmt', '- Output format for numerical results in the <rootname>.SD.out file\n'))
        f.write('{:<22} {:<11} {:}'.format(self.fst_vt['SubDyn']['OutSFmt'], 'OutSFmt', '- Output format for header strings in the <rootname>.SD.out file\n'))
        f.write('------------------------- MEMBER OUTPUT LIST ------------------------------------------\n')
        f.write('{:<22d} {:<11} {:}'.format(self.fst_vt['SubDyn']['NMOutputs'], 'NMOutputs', '- Number of members whose forces/displacements/velocities/accelerations will be output (-) [Must be <= 9].\n'))
        f.write(" ".join(['{:^11s}'.format(i) for i in ['MemberID', 'NOutCnt', 'NodeCnt']])+'[NOutCnt=how many nodes to get output for [< 10]; NodeCnt are local ordinal numbers from the start of the member, and must be >=1 and <= NDiv+1] If NMOutputs=0 leave blank as well.\n')
        f.write(" ".join(['{:^11s}'.format(i) for i in ['(-)','(-)','(-)']])+'\n')
        for i in range(self.fst_vt['SubDyn']['NMOutputs']):
            ln = []
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['MemberID_out'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['NOutCnt'][i]))
            ln.append('{:^11d}'.format(self.fst_vt['SubDyn']['NodeCnt'][i]))
            f.write(" ".join(ln) + '\n')
        f.write('------------------------- SSOutList: The next line(s) contains a list of output parameters that will be output in <rootname>.SD.out or <rootname>.out. ------\n')
        outlist = self.get_outlist(self.fst_vt['outlist'], ['SubDyn'])
        for channel_list in outlist:
            for i in range(len(channel_list)):
                f.write('"' + channel_list[i] + '"\n')
        f.write('END of output channels and end of file. (the word "END" must appear in the first 3 columns of this line)\n')
        f.close()

    def write_MAP(self):

        # Generate MAP++ input file
        self.fst_vt['Fst']['MooringFile'] = self.FAST_namingOut + '_MAP.dat'
        map_file = os.path.join(self.FAST_runDirectory, self.fst_vt['Fst']['MooringFile'])
        f = open(map_file, 'w')

        f.write('---------------------- LINE DICTIONARY ---------------------------------------\n')
        f.write(" ".join(['{:<11s}'.format(i) for i in ['LineType', 'Diam', 'MassDenInAir', 'EA', 'CB', 'CIntDamp', 'Ca', 'Cdn', 'Cdt']])+'\n')
        f.write(" ".join(['{:<11s}'.format(i) for i in ['(-)', '(m)', '(kg/m)', '(N)', '(-)', '(Pa-s)', '(-)', '(-)', '(-)']])+'\n')
        ln =[]
        ln.append('{:<11}'.format(self.fst_vt['MAP']['LineType']))
        ln.append('{:<11}'.format(self.fst_vt['MAP']['Diam']))
        ln.append('{:<11}'.format(self.fst_vt['MAP']['MassDenInAir']))
        ln.append('{:<11}'.format(self.fst_vt['MAP']['EA']))
        ln.append('{:<11}'.format(self.fst_vt['MAP']['CB']))
        ln.append('{:<11}'.format(self.fst_vt['MAP']['CIntDamp']))
        ln.append('{:<11}'.format(self.fst_vt['MAP']['Ca']))
        ln.append('{:<11}'.format(self.fst_vt['MAP']['Cdn']))
        ln.append('{:<11}'.format(self.fst_vt['MAP']['Cdt']))
        f.write(" ".join(ln) + '\n')
        f.write('---------------------- NODE PROPERTIES ---------------------------------------\n')
        f.write(" ".join(['{:<11s}'.format(i) for i in ['Node', 'Type', 'X', 'Y', 'Z', 'M', 'B', 'FX', 'FY', 'FZ']])+'\n')
        f.write(" ".join(['{:<11s}'.format(i) for i in ['(-)', '(-)', '(m)', '(m)', '(m)', '(kg)', '(m^3)', '(N)', '(N)', '(N)']])+'\n')
        for i in range(2):
            ln =[]
            ln.append('{:<11}'.format(self.fst_vt['MAP']['Node'][i]))
            ln.append('{:<11}'.format(self.fst_vt['MAP']['Type'][i]))
            ln.append('{:<11}'.format(self.fst_vt['MAP']['X'][i]))
            ln.append('{:<11}'.format(self.fst_vt['MAP']['Y'][i]))
            ln.append('{:<11}'.format(self.fst_vt['MAP']['Z'][i]))
            ln.append('{:<11}'.format(self.fst_vt['MAP']['M'][i]))
            ln.append('{:<11}'.format(self.fst_vt['MAP']['B'][i]))
            ln.append('{:<11}'.format(self.fst_vt['MAP']['FX'][i]))
            ln.append('{:<11}'.format(self.fst_vt['MAP']['FY'][i]))
            ln.append('{:<11}'.format(self.fst_vt['MAP']['FZ'][i]))
            f.write(" ".join(ln) + '\n')
        f.write('---------------------- LINE PROPERTIES ---------------------------------------\n')
        f.write(" ".join(['{:<11s}'.format(i) for i in ['Line', 'LineType', 'UnstrLen', 'NodeAnch', 'NodeFair', 'Flags']])+'\n')
        f.write(" ".join(['{:<11s}'.format(i) for i in ['(-)', '(-)', '(m)', '(-)', '(-)', '(-)']])+'\n')
        ln =[]
        ln.append('{:<11}'.format(self.fst_vt['MAP']['Line']))
        ln.append('{:<11}'.format(self.fst_vt['MAP']['LineType']))
        ln.append('{:<11}'.format(self.fst_vt['MAP']['UnstrLen']))
        ln.append('{:<11}'.format(self.fst_vt['MAP']['NodeAnch']))
        ln.append('{:<11}'.format(self.fst_vt['MAP']['NodeFair']))
        ln.append('{:<11}'.format(" ".join(self.fst_vt['MAP']['Flags'])))
        f.write(" ".join(ln) + '\n')
        f.write('---------------------- SOLVER OPTIONS-----------------------------------------\n')
        f.write('{:<11s}'.format('Option'+'\n'))
        f.write('{:<11s}'.format('(-)')+'\n')
        f.write(" ".join(self.fst_vt['MAP']['Option']).strip() + '\n')

        f.close()

        # f.write('{:<22} {:<11} {:}'.format(self.fst_vt['MAP'][''], '', '- \n'))
        # f.write('\n')
        

class InputWriter_FAST7(InputWriter_Common):

    def execute(self):
        
        if not os.path.exists(self.FAST_runDirectory):
            os.makedirs(self.FAST_runDirectory)

        # self.write_WindWnd()
        self.write_ElastoDynBlade()
        self.write_ElastoDynTower()
        self.write_AeroDyn_FAST7()

        self.write_MainInput()

    def write_MainInput(self):

        self.FAST_InputFileOut = os.path.join(self.FAST_runDirectory, self.FAST_namingOut+'.fst')
        ofh = open(self.FAST_InputFileOut, 'w')

        # FAST Inputs
        ofh.write('---\n')
        ofh.write('---\n')
        ofh.write('{:}\n'.format(self.fst_vt['description']))
        ofh.write('---\n')
        ofh.write('---\n')
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['Echo']))
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['ADAMSPrep']))
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['AnalMode']))
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['NumBl']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TMax']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['DT']))
        ofh.write('---\n')
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['YCMode']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TYCOn']))
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['PCMode']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TPCOn']))
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['VSContrl']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['VS_RtGnSp']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['VS_RtTq']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['VS_Rgn2K']))
        ofh.write('{:.5e}\n'.format(self.fst_vt['Fst7']['VS_SlPc']))
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['GenModel']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['GenTiStr']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['GenTiStp']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['SpdGenOn']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TimGenOn']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TimGenOf']))
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['HSSBrMode']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['THSSBrDp']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TiDynBrk']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TTpBrDp1']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TTpBrDp2']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TTpBrDp3']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TBDepISp1']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TBDepISp2']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TBDepISp3']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TYawManS']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TYawManE']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['NacYawF']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TPitManS1']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TPitManS2']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TPitManS3']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TPitManE1']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TPitManE2']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TPitManE3']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['BlPitch1']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['BlPitch2']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['BlPitch3']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['B1PitchF1']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['B1PitchF2']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['B1PitchF3']))
        ofh.write('---\n')
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['Gravity']))
        ofh.write('---\n')
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['FlapDOF1']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['FlapDOF2']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['EdgeDOF']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['TeetDOF']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['DrTrDOF']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['GenDOF']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['YawDOF']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['TwFADOF1']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['TwFADOF2']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['TwSSDOF1']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['TwSSDOF2']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['CompAero']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['CompNoise']))
        ofh.write('---\n')
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['OoPDefl']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['IPDefl']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TeetDefl']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['Azimuth']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['RotSpeed']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['NacYaw']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TTDspFA']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TTDspSS']))
        ofh.write('---\n')
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TipRad']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['HubRad']))
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['PSpnElN']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['UndSling']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['HubCM']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['OverHang']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['NacCMxn']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['NacCMyn']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['NacCMzn']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TowerHt']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['Twr2Shft']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TwrRBHt']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['ShftTilt']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['Delta3']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['PreCone(1)']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['PreCone(2)']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['PreCone(3)']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['AzimB1Up']))
        ofh.write('---\n')
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['YawBrMass']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['NacMass']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['HubMass']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TipMass(1)']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TipMass(2)']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TipMass(3)']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['NacYIner']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['GenIner']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['HubIner']))
        ofh.write('---\n')
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['GBoxEff']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['GenEff']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['GBRatio']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['GBRevers']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['HSSBrTqF']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['HSSBrDT']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['DynBrkFi']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['DTTorSpr']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['DTTorDmp']))
        ofh.write('---\n')
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['SIG_SlPc']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['SIG_SySp']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['SIG_RtTq']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['SIG_PORt']))
        ofh.write('---\n')
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TEC_Freq']))
        ofh.write('{:5}\n'.format(self.fst_vt['Fst7']['TEC_NPol']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TEC_SRes']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TEC_RRes']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TEC_VLL']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TEC_SLR']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TEC_RLR']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TEC_MR']))
        ofh.write('---\n')
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['PtfmModel']))
        ofh.write('"{:}"\n'.format(self.fst_vt['Fst7']['PtfmFile']))
        ofh.write('---\n')
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['TwrNodes']))
        ofh.write('"{:}"\n'.format(self.fst_vt['Fst7']['TwrFile']))
        ofh.write('---\n')
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['YawSpr']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['YawDamp']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['YawNeut']))
        ofh.write('---\n')
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['Furling']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['FurlFile']))
        ofh.write('---\n') 
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['TeetMod']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TeetDmpP']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TeetDmp']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TeetCDmp']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TeetSStP']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TeetHStP']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TeetSSSp']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TeetHSSp']))
        ofh.write('---\n')
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TBDrConN']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TBDrConD']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TpBrDT']))
        ofh.write('---\n')
        ofh.write('"{:}"\n'.format(self.fst_vt['Fst7']['BldFile1']))
        ofh.write('"{:}"\n'.format(self.fst_vt['Fst7']['BldFile2']))
        ofh.write('"{:}"\n'.format(self.fst_vt['Fst7']['BldFile3']))
        ofh.write('---\n') 
        ofh.write('"{:}"\n'.format(self.fst_vt['Fst7']['ADFile']))
        ofh.write('---\n')
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['NoiseFile']))
        ofh.write('---\n')
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['ADAMSFile']))
        ofh.write('---\n')
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['LinFile']))
        ofh.write('---\n')
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['SumPrint']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['OutFileFmt']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['TabDelim']))
        ofh.write('{:}\n'.format(self.fst_vt['Fst7']['OutFmt']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['TStart']))
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['DecFact']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['SttsTime']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['NcIMUxn']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['NcIMUyn']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['NcIMUzn']))
        ofh.write('{:.9f}\n'.format(self.fst_vt['Fst7']['ShftGagL']))
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['NTwGages']))
        for i in range(self.fst_vt['Fst7']['NTwGages']-1):
            ofh.write('{:3}, '.format(self.fst_vt['Fst7']['TwrGagNd'][i]))
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['TwrGagNd'][-1]))
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['NBlGages']))
        for i in range(self.fst_vt['Fst7']['NBlGages']-1):
            ofh.write('{:3}, '.format(self.fst_vt['Fst7']['BldGagNd'][i]))
        ofh.write('{:3}\n'.format(self.fst_vt['Fst7']['BldGagNd'][-1]))
    
        # Outlist
        ofh.write('Outlist\n')
        outlist = self.get_outlist(self.fst_vt['outlist7'], ['OutList'])
        for channel_list in outlist:
            for i in range(len(channel_list)):
                f.write('"' + channel_list[i] + '"\n')
        ofh.write('END\n')
        ofh.close()
        
        ofh.close()

    def write_AeroDyn_FAST7(self):
        if not os.path.isdir(os.path.join(self.FAST_runDirectory,'AeroData')):
            os.mkdir(os.path.join(self.FAST_runDirectory,'AeroData'))

        # create airfoil objects
        for i in range(self.fst_vt['AeroDyn14']['NumFoil']):
             af_name = os.path.join(self.FAST_runDirectory, 'AeroData', 'Airfoil' + str(i) + '.dat')
             self.fst_vt['AeroDyn14']['FoilNm'][i]  = os.path.join('AeroData', 'Airfoil' + str(i) + '.dat')
             self.write_AeroDyn14Polar(af_name, i)

        self.fst_vt['Fst7']['ADFile'] = self.FAST_namingOut + '_AeroDyn.dat'
        ad_file = os.path.join(self.FAST_runDirectory,self.fst_vt['Fst7']['ADFile'])
        ofh = open(ad_file,'w')
        
        ofh.write('Aerodyn input file for FAST\n')
        
        ofh.write('{:}\n'.format(self.fst_vt['AeroDyn14']['SysUnits']))
        ofh.write('{:}\n'.format(self.fst_vt['AeroDyn14']['StallMod']))        
        
        ofh.write('{:}\n'.format(self.fst_vt['AeroDyn14']['UseCm']))
        ofh.write('{:}\n'.format(self.fst_vt['AeroDyn14']['InfModel']))
        ofh.write('{:}\n'.format(self.fst_vt['AeroDyn14']['IndModel']))
        ofh.write('{:.3f}\n'.format(self.fst_vt['AeroDyn14']['AToler']))
        ofh.write('{:}\n'.format(self.fst_vt['AeroDyn14']['TLModel']))
        ofh.write('{:}\n'.format(self.fst_vt['AeroDyn14']['HLModel']))
        ofh.write('"{:}"\n'.format(self.fst_vt['AeroDyn14']['WindFile']))
        ofh.write('{:f}\n'.format(self.fst_vt['AeroDyn14']['HH']))  
  
        ofh.write('{:.1f}\n'.format(self.fst_vt['AeroDyn14']['TwrShad']))  
  
        ofh.write('{:.1f}\n'.format(self.fst_vt['AeroDyn14']['ShadHWid']))  
  
        ofh.write('{:.1f}\n'.format(self.fst_vt['AeroDyn14']['T_Shad_Refpt']))  
  
        ofh.write('{:.3f}\n'.format(self.fst_vt['AeroDyn14']['AirDens']))  
  
        ofh.write('{:.9f}\n'.format(self.fst_vt['AeroDyn14']['KinVisc']))  
  
        ofh.write('{:2}\n'.format(self.fst_vt['AeroDyn14']['DTAero']))        
        

        ofh.write('{:2}\n'.format(self.fst_vt['AeroDyn14']['NumFoil']))
        for i in range (self.fst_vt['AeroDyn14']['NumFoil']):
            ofh.write('"{:}"\n'.format(self.fst_vt['AeroDyn14']['FoilNm'][i]))

        ofh.write('{:2}\n'.format(self.fst_vt['AeroDynBlade']['BldNodes']))
        rnodes = self.fst_vt['AeroDynBlade']['RNodes']
        twist = self.fst_vt['AeroDynBlade']['AeroTwst']
        drnodes = self.fst_vt['AeroDynBlade']['DRNodes']
        chord = self.fst_vt['AeroDynBlade']['Chord']
        nfoil = self.fst_vt['AeroDynBlade']['NFoil']
        prnelm = self.fst_vt['AeroDynBlade']['PrnElm']
        ofh.write('Nodal properties\n')
        for r, t, dr, c, a, p in zip(rnodes, twist, drnodes, chord, nfoil, prnelm):
            ofh.write('{: 2.15e}\t{:.3f}\t{:.4f}\t{:.3f}\t{:5}\t{:}\n'.format(r, t, dr, c, a, p))

        ofh.close()



if __name__=="__main__":

    FAST_ver   = 'openfast'
    read_yaml  = False
    dev_branch = True

    fst_update = {}
    fst_update['Fst', 'TMax'] = 20.
    fst_update['AeroDyn15', 'TwrAero'] = False


    if read_yaml:
        fast = InputReader_Common(FAST_ver=FAST_ver)
        fast.FAST_yamlfile = 'temp/OpenFAST/test.yaml'
        fast.read_yaml()

    if FAST_ver.lower() == 'fast7':
        if not read_yaml:
            fast = InputReader_FAST7(FAST_ver=FAST_ver)
            fast.FAST_InputFile = 'Test16.fst'   # FAST input file (ext=.fst)
            fast.FAST_directory = 'C:/Users/egaertne/WT_Codes/models/FAST_v7.02.00d-bjj/CertTest/'   # Path to fst directory files
            fast.execute()
        
        fastout = InputWriter_FAST7(FAST_ver=FAST_ver)
        fastout.fst_vt = fast.fst_vt
        fastout.FAST_runDirectory = 'temp/FAST7'
        fastout.FAST_namingOut = 'test'
        fastout.execute()

    elif FAST_ver.lower() == 'fast8':
        if not read_yaml:
            fast = InputReader_OpenFAST(FAST_ver=FAST_ver)
            fast.FAST_InputFile = 'NREL5MW_onshore.fst'   # FAST input file (ext=.fst)
            fast.FAST_directory = 'C:/Users/egaertne/WT_Codes/models/FAST_v8.16.00a-bjj/ref/5mw_onshore/'   # Path to fst directory files
            fast.execute()
        
        fastout = InputWriter_OpenFAST(FAST_ver=FAST_ver)
        fastout.fst_vt = fast.fst_vt
        fastout.FAST_runDirectory = 'temp/FAST8'
        fastout.FAST_namingOut = 'test'
        fastout.execute()

    elif FAST_ver.lower() == 'openfast':
        if not read_yaml:
            fast = InputReader_OpenFAST(FAST_ver=FAST_ver, dev_branch=dev_branch)
            # fast.FAST_InputFile = '5MW_Land_DLL_WTurb.fst'   # FAST input file (ext=.fst)
            # fast.FAST_directory = 'C:/Users/egaertne/WT_Codes/models/openfast/glue-codes/fast/5MW_Land_DLL_WTurb'   # Path to fst directory files
            
            # fast.FAST_InputFile = "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.fst"
            # fast.FAST_directory = 'C:/Users/egaertne/WT_Codes/models/openfast-dev/r-test/glue-codes/openfast/5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth'

            fast.FAST_InputFile = '5MW_OC3Spar_DLL_WTurb_WavesIrr.fst'   # FAST input file (ext=.fst)
            fast.FAST_directory = 'C:/Users/egaertne/WT_Codes/models/openfast-dev/r-test/glue-codes/openfast/5MW_OC3Spar_DLL_WTurb_WavesIrr'   # Path to fst directory files

            fast.execute()
        
        fastout = InputWriter_OpenFAST(FAST_ver=FAST_ver, dev_branch=dev_branch)
        fastout.fst_vt = fast.fst_vt
        fastout.FAST_runDirectory = 'temp/OpenFAST'
        fastout.FAST_namingOut = 'test'
        fastout.update(fst_update=fst_update)
        fastout.execute()
    
    fastout.write_yaml()

    

