from __future__ import print_function
import os, sys, time, shutil
from functools import partial
import multiprocessing as mp
import numpy as np
import yaml
try:
    import ruamel_yaml as ry
except:
    try:
        import ruamel.yaml as ry
    except:
        raise ImportError('No module named ruamel.yaml or ruamel_yaml')

import matplotlib.pyplot as plt
import pandas as pd

from ROSCO_toolbox.utilities import FAST_IO

from pCrunch import Analysis, pdTools


class FAST_Processing(object):
    '''
    A class with tools to post process batch OpenFAST output data
    '''

    def __init__(self, **kwargs):
        # Optional population class attributes from key word arguments
        self.OpenFAST_outfile_list = [[]] # list of lists containing absolute path to fast output files. Each inner list corresponds to a dataset to compare, and should be of equal length
        self.dataset_names = []       # (Optional) N labels that identify each dataset
        # (Optional) AeroelasticSE case matrix text file. Used to generated descriptions of IEC DLCS
        self.fname_case_matrix = ''

        # Analysis Control
        self.parallel_analysis = False  # True/False; Perform post processing in parallel
        self.parallel_cores = None      # None/Int>0; (Only used if parallel_analysis==True) number of parallel cores, if None, multiprocessing will use the maximum available
        self.verbose = False            # True/False; Enable/Disable non-error message outputs to screen

        # Analysis Options
        self.t0 = None       # float>=0    ; start time to include in analysis
        self.tf = None  # float>=0,-1 ; end time to include in analysis

        # Load ranking
        self.ranking_vars = [   ["RotSpeed"], 
                                ["TipDxc1", "TipDxc2", "TipDxc3"], 
                                ["TipDyc1", "TipDyc2", "TipDyc3"], 
                                ['RootMyb1', 'RootMyb2', 'RootMyb3'], 
                                ['RootMxb1', 'RootMxb2', 'RootMxb3'],
                                ['TwrBsFyt'],
                                ]  # List of lists
        self.ranking_stats = [  'max',
                                'max',
                                'max',
                                'max',
                                'max',
                                    ] # should be same length as ranking_vars
        
        self.DEL_info = None  # [('RootMyb1', 10), ('RootMyb2', 10), ('RootMyb3', 10)]

        # Save settings
        self.results_dir       = 'temp_results'
        self.save_LoadRanking  = False  # NJA - does not exist yet
        self.save_SummaryStats = False

        # Load kwargs
        for k, w in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        # Setup multiprocessing:
        if self.parallel_analysis:
            # Make sure multi-processing cores are valid
            if not self.parallel_cores:
                self.parallel_cores = mp.cpu_count()
            elif self.parallel_cores == 1:
                self.parallel_analysis = False

        # Check for save directory
        if self.save_LoadRanking or self.save_SummaryStats:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)


        super(FAST_Processing, self).__init__()

        '''
        # Analysis Control
        self.plot_TimeSeries = False
        self.plot_FFT = False
        # self.plot_LoadRanking      = False
        
        # [str]       ; (Only used if plot_TimeSeries or plot_FFT = True) list of OpenFAST channels to plot
        self.plot_vars = []
        # [str]       ; (Optional)(Only used if plot_TimeSeries or plot_FFT = True) list axis labels for OpenFAST channels
        self.plot_vars_labels = []
        # float>0.    ; (Only used if plot_FFT = True) maximum frequency on x axis for FFT plots
        self.fft_x_max = 1.
        # True/False  ; (Only used if plot_FFT = True) Include 1P, 3P, 6P bands on FFT plots
        self.fft_show_RtSpeed = True
        # [floats]    ; (Optional)(Only used if plot_FFT = True) Additional frequencies to label on the FFT
        self.fft_include_f = []
        # [str]       ; (Optional)(Only used if plot_FFT = True) Legend names for additional frequencies to label on the FFT
        self.fft_include_f_labels = []
        # [str]       ; (Only used if save_LoadRanking = True) list of OpenFAST channels to perform load ranking on
        self.ranking_vars = []
        # str         ; (Only used if save_LoadRanking = True) statistic to rank loads by (min, max, mean, std, maxabs)
        self.ranking_statistic = 'maxabs'
        # int         ; (Only used if save_LoadRanking = True) number of loads to list per channel
        self.ranking_n_cases = 5
        '''

    def batch_processing(self):
        '''
        Run a full batch processing case!
        '''
        # ------------------ Input consistancy checks ------------------ #
        # Do we have a list of data?
        N = len(self.OpenFAST_outfile_list)
        if N == 0:
            raise ValueError('Output files not defined! Populate: "FastPost.OpenFAST_outfile_list". \n Quitting FAST_Processing.')
            

        # Do all the files exist?
        files_exist = True
        for i, flist in enumerate(self.OpenFAST_outfile_list):
            if isinstance(flist, str):
                if not os.path.exists(flist):
                    print('Warning! File "{}" does not exist.'.format(
                        flist))
                    self.OpenFAST_outfile_list.remove(flist)
            elif isinstance(flist, list):
                for fname in flist:
                    if not os.path.exists(fname):
                        files_exist = False
                        if len(self.dataset_names) > 0:
                            print('Warning! File "{}" from {} does not exist.'.format(
                                fname, self.dataset_names[i]))
                            flist.remove(fname)
                        else:
                            print('Warning! File "{}" from dataset {} of {} does not exist.'.format(
                                fname, i+1, N))
                            flist.remove(fname)

        # # load case matrix data to get descriptive case naming
        # if self.fname_case_matrix == '':
        #     print('Warning! No case matrix file provided, no case descriptions will be provided.')
        #     self.case_desc = ['Case ID %d' % i for i in range(M)]
        # else:
        #     cases = load_case_matrix(self.fname_case_matrix)
        #     self.case_desc = get_dlc_label(cases, include_seed=True)

        # get unique file namebase for datasets
        self.namebase = []
        if len(self.dataset_names) > 0:
            # use filename safe version of dataset names
            self.namebase = ["".join([c for c in name if c.isalpha() or c.isdigit() or c in [
                                     '_', '-']]).rstrip() for i, name in zip(range(N), self.dataset_names)]
        elif len(self.OpenFAST_outfile_list) > 0:
            # use out file naming
            if isinstance(self.OpenFAST_outfile_list[0], list):
                self.namebase = ['_'.join(os.path.split(flist[0])[1].split('_')[:-1])
                             for flist in self.OpenFAST_outfile_list]
            else:
                self.namebase = ['_'.join(os.path.split(flist)[1].split('_')[:-1])
                             for flist in self.OpenFAST_outfile_list]
        
        # check that names are unique
        if not len(self.namebase) == len(set(self.namebase)):
            self.namebase = []
        # as last resort, give generic name
        if not self.namebase:
            if isinstance(self.OpenFAST_outfile_list[0], str):
                # Just one dataset name for single dataset
                self.namebase = ['dataset1']
            else:
                self.namebase = ['dataset' + ('{}'.format(i)).zfill(len(str(N-1))) for i in range(N)]
        
        # Run design comparison if filenames list has multiple lists
        if (len(self.OpenFAST_outfile_list) > 1) and (isinstance(self.OpenFAST_outfile_list[0], list)): 
            # Load stats and load rankings for design comparisons
            stats, load_rankings = self.design_comparison(self.OpenFAST_outfile_list)
        
        else:
            # Initialize Analysis
            loads_analysis = Analysis.Loads_Analysis()
            loads_analysis.verbose = self.verbose
            loads_analysis.t0 = self.t0
            loads_analysis.tf = self.tf
            loads_analysis.DEL_info = self.DEL_info
            loads_analysis.ranking_stats = self.ranking_stats
            loads_analysis.ranking_vars = self.ranking_vars

            # run analysis in parallel
            if self.parallel_analysis:
                pool = mp.Pool(self.parallel_cores)
                try:
                    stats_separate = pool.map(
                        partial(loads_analysis.full_loads_analysis, get_load_ranking=False), self.OpenFAST_outfile_list)
                except:
                    stats_separate = pool.map(partial(loads_analysis.full_loads_analysis, get_load_ranking=False), self.OpenFAST_outfile_list[0])
                pool.close()
                pool.join()

                # Re-sort into the more "standard" dictionary/dataframe format we like
                stats = [pdTools.dict2df(ss).unstack() for ss in stats_separate]
                dft = pd.DataFrame(stats)
                dft = dft.reorder_levels([2, 0, 1], axis=1).sort_index(axis=1, level=0)
                stats = pdTools.df2dict(dft)

                # Get load rankings after stats are loaded
                load_rankings = loads_analysis.load_ranking(stats,
                                            names=self.dataset_names, get_df=False)
           
            # run analysis in serial
            else:
                # Initialize Analysis
                loads_analysis = Analysis.Loads_Analysis()
                loads_analysis.verbose = self.verbose
                loads_analysis.t0 = self.t0
                loads_analysis.tf = self.tf
                loads_analysis.ranking_stats = self.ranking_stats
                loads_analysis.ranking_vars = self.ranking_vars
                loads_analysis.DEL_info = self.DEL_info
                stats, load_rankings = loads_analysis.full_loads_analysis(self.OpenFAST_outfile_list, get_load_ranking=True)

        if self.save_SummaryStats:
            if isinstance(stats, dict):
                fname = self.namebase[0] + '_stats.yaml'
                if self.verbose:
                    print('Saving {}'.format(fname))
                save_yaml(self.results_dir, fname, stats)
            else:
                for namebase, st in zip(self.namebase, stats):
                    fname = namebase + '_stats.yaml'
                    if self.verbose:
                        print('Saving {}'.format(fname))
                    save_yaml(self.results_dir, fname, st)
        if self.save_LoadRanking:
            if isinstance(load_rankings, dict):
                fname = self.namebase[0] + '_LoadRanking.yaml'
                if self.verbose:
                    print('Saving {}'.format(fname))
                save_yaml(self.results_dir, fname, load_rankings)
            else:
                for namebase, lr in zip(self.namebase, load_rankings):
                    fname = namebase + '_LoadRanking.yaml'
                    if self.verbose:
                        print('Saving {}'.format(fname))
                    save_yaml(self.results_dir, fname, lr)


        return stats, load_rankings

    def design_comparison(self, filenames):
        '''
        Compare design runs

        Parameters:
        ----------
        filenames: list
            list of lists, where the inner lists are of equal length. 

        Returns:
        --------
        stats: dict
            dictionary of summary statistics data
        load_rankings: dict
            dictionary of load rankings
        '''


        # Make sure datasets are the same length
        ds_len = len(filenames[0])
        if any(len(dataset) != ds_len for dataset in filenames):
            raise ValueError('The datasets for filenames corresponding to the design comparison should all be the same size.')

        fnames = np.array(filenames).T.tolist()
        # Setup FAST_Analysis preferences
        loads_analysis = Analysis.Loads_Analysis()
        loads_analysis.verbose=self.verbose
        loads_analysis.t0 = self.t0
        loads_analysis.tf = self.tf
        loads_analysis.ranking_vars = self.ranking_vars
        loads_analysis.ranking_stats = self.ranking_stats
        loads_analysis.DEL_info = self.DEL_info
        
        if self.parallel_analysis: # run analysis in parallel
            # run analysis
            pool = mp.Pool(self.parallel_cores)
            stats_separate = pool.map(partial(loads_analysis.full_loads_analysis, get_load_ranking=False), fnames)
            pool.close()
            pool.join()
        
            # Re-sort into the more "standard" dictionary/dataframe format we like
            stats = [pdTools.dict2df(ss).unstack() for ss in stats_separate]
            dft = pd.DataFrame(stats)
            dft = dft.reorder_levels([2, 0, 1], axis=1).sort_index(axis=1, level=0)
            stats = pdTools.df2dict(dft)

            # Get load rankings after stats are loaded
            load_rankings = loads_analysis.load_ranking(stats) 

        else: # run analysis in serial
            stats = []
            load_rankings = []
            for file_sets in filenames:
                st, lr = loads_analysis.full_loads_analysis(file_sets, get_load_ranking=True)
                stats.append(st)
                load_rankings.append(lr)
            



        return stats, load_rankings

def get_windspeeds(case_matrix, return_df=False):
    '''
    Find windspeeds from case matrix

    Parameters:
    ----------
    case_matrix: dict
        case matrix data loaded from wisdem.aeroelasticse.Util.FileTools.load_yaml
    
    Returns:
    --------
    windspeed: list
        list of wind speeds
    seed: seed
        list of wind seeds
    IECtype: list
        list of IEC types 
    case_matrix: pd.DataFrame
        case matrix dataframe with appended wind info
    '''
    if isinstance(case_matrix, dict):
        cmatrix = case_matrix
    elif isinstance(case_matrix, pd.DataFrame):
        cmatrix = case_matrix.to_dict('list')
    else:
        raise TypeError('case_matrix must be a dict or pd.DataFrame.')


    windspeed = []
    seed = []
    IECtype = []
    # loop through and parse each inflow filename text entry to get wind and seed #
    for fname in  cmatrix[('InflowWind','Filename')]:
        if '.bts' in fname:
            obj = fname.split('U')[-1].split('_')
            obj2 = obj[1].split('Seed')[-1].split('.bts')
            windspeed.append(float(obj[0]))
            seed.append(float(obj2[0]))
            if 'NTM' in fname:
                IECtype.append('NTM')
            elif 'ETM' in fname:
                IECtype.append('NTM')
        elif 'ECD' in fname:
            obj = fname.split('U')[-1].split('.wnd')
            windspeed.append(float(obj[0]))
            seed.append([])
            IECtype.append('ECD')
        elif 'EWS' in fname:
            obj = fname.split('U')[-1].split('.wnd')
            windspeed.append(float(obj[0]))
            seed.append([])
            IECtype.append('EWS')
        
    if return_df:
        case_matrix = pd.DataFrame(case_matrix)
        case_matrix[('InflowWind','WindSpeed')] = windspeed
        case_matrix[('InflowWind','Seed')] = seed
        case_matrix[('InflowWind','IECtype')] = IECtype
        
        return windspeed, seed, IECtype, case_matrix
    
    else:
        return windspeed, seed, IECtype


def save_yaml(outdir, fname, data_out):
    ''' Save yaml file - ripped from WISDEM 
    
    Parameters:
    -----------
    outdir: str
        directory to save yaml
    fname: str
        filename for yaml
    data_out: dict
        data to dump to yaml
    '''
    fname = os.path.join(outdir, fname)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    f = open(fname, "w")
    yaml = ry.YAML()
    yaml.default_flow_style = None
    yaml.width = float("inf")
    yaml.indent(mapping=4, sequence=6, offset=3)
    yaml.dump(data_out, f)


def load_yaml(fname_input, package=0):
    ''' Import a .yaml file - ripped from WISDEM

    Parameters:
    -----------
    fname_input: str
        yaml file to load
    package: bool
        0 = yaml, 1 = ruamel
    '''
    if package == 0:
        with open(fname_input) as f:
            data = yaml.safe_load(f)
        return data

    elif package == 1:
        with open(fname_input, 'r') as myfile:
            text_input = myfile.read()
        myfile.close()
        ryaml = ry.YAML()
        return dict(ryaml.load(text_input))
