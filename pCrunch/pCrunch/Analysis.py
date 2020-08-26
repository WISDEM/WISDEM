from __future__ import print_function
import os, sys, time, shutil
from functools import partial
import multiprocessing as mp
import numpy as np
import ruamel_yaml as ry
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import fatpack # 3rd party module used for rainflow counting

from scipy.interpolate import PchipInterpolator
from ROSCO_toolbox.utilities import FAST_IO

from pCrunch import pdTools

class Loads_Analysis(object):
    '''
    Contains analysis tools to post-process OpenFAST output data. Most methods are written to support
    single instances of output data to ease parallelization. 

    Methods:
    --------
    full_loads_analysis
    summary_stats
    load_ranking
    fatigue
    '''
    def __init__(self, **kwargs):

        # Analysis time range
        self.t0 = None
        self.tf = None
        # Desired channels for analysis
        self.channel_list = []
        self.channels_magnitude = {}

        self.channels_extreme_table = []

        # Load Ranking 
        self.ranking_vars = [["RotSpeed"],
                             ["TipDxc1", "TipDxc2", "TipDxc3"],
                             ["TipDyc1", "TipDyc2", "TipDyc3"],
                             ['RootMyb1', 'RootMyb2', 'RootMyb3'],
                             ['RootMxb1', 'RootMxb2', 'RootMxb3'],
                             ['TwrBsFyt']
                             ]  # List of lists
        self.ranking_stats = ['max',
                              'max',
                              'max',
                              'max',
                              'max',
                              ]  # should be same length as ranking_vars

        self.DEL_info = None #[('RootMyb1', 10), ('RootMyb2', 10), ('RootMyb3', 10)]
        # verbose?
        self.verbose=False

        for k, w in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        super(Loads_Analysis, self).__init__()

    def full_loads_analysis(self, filenames, get_load_ranking=True, return_FastData=False):
        '''
        Load openfast data - get statistics - get load ranking - return data
        NOTE: Can be called to run in parallel if get_load_ranking=False (see Processing.batch_processing)

        Parameters:
        -------
        filenames: list
            List of filenames to load and analyse
        get_load_ranking: bool, optional
            Get the load ranking for all cases
        return_FastData, bool
            Return a dictionary or list constaining OpenFAST output data

        Outputs:
        --------
        sum_stats: dict
            dictionary of summary statistics
        load_rankings: dict
            dictionary of load rankings
        fast_data: list or dict
            list or dictionary containing OpenFAST output data
        '''
        # Load openfast data
        fast_io = FAST_IO()
        fast_data = fast_io.load_FAST_out(filenames, tmin=self.t0, tmax=self.tf, verbose=self.verbose)

        # Get summary statistics
        sum_stats = self.summary_stats(fast_data)

        # Get load rankings
        if get_load_ranking:
            load_rankings = self.load_ranking(sum_stats)


        if return_FastData:
            return sum_stats, fast_data
        if get_load_ranking: 
            return sum_stats, load_rankings
        if return_FastData and get_load_ranking:
            return sum_stats, load_rankings, fast_data
        else:
            return sum_stats

    def summary_stats(self, fast_data, channel_list=[]):
        '''
        Get summary statistics from openfast output data. 

        Parameters:
        ----------
        fast_data: list
            List of dictionaries containing openfast output data (returned from ROSCO_toolbox.FAST_IO.load_output)
        channel_list: list
            list of channels to collect data from. Defaults to all


        Returns:
        -------
        data_out: dict
            Dictionary containing summary statistics
        fast_outdata: dict, optional
            Dictionary of all OpenFAST output data. Only returned if return_data=true
        '''
        sum_stats     = {}
        extreme_table = {}

        for fd in fast_data:
            if self.verbose:
                print('Processing data for {}'.format(fd['meta']['name']))

            # Build channel list if it isn't input
            if channel_list == []:
                channel_list = list(fd.keys())

            # Process Data
            for channel in channel_list + list(self.channels_magnitude.keys()):
                if channel == 'meta':
                    if 'meta' not in sum_stats.keys():
                        sum_stats['meta'] = {}
                        sum_stats['meta']['name'] = []
                        sum_stats['meta']['filename'] = []
                    # save some meta data
                    sum_stats['meta']['name'] = fd['meta']['name']
                    sum_stats['meta']['filename'] = fd['meta']['filename']

                elif channel != 'Time' and channel in channel_list+list(self.channels_magnitude.keys()):
                    # try:

                    if channel in channel_list:
                        y_data = fd[channel]
                    elif channel in self.channels_magnitude.keys():
                        # calculate magnitude of a vector
                        vector = np.array([fd[var] for var in self.channels_magnitude[channel]])
                        n_dim, n_t = np.shape(vector)
                        y_data = np.array([np.sqrt(np.sum([vector[i,j]**2. for i in range(n_dim)])) for j in range(n_t)])

                    if channel not in sum_stats.keys():
                        sum_stats[channel] = {}
                        sum_stats[channel]['min'] = []
                        sum_stats[channel]['max'] = []
                        sum_stats[channel]['std'] = []
                        sum_stats[channel]['mean'] = []
                        sum_stats[channel]['abs'] = []
                        sum_stats[channel]['integrated'] = []

                    # calculate summary statistics
                    sum_stats[channel]['min'].append(float(min(y_data)))
                    sum_stats[channel]['max'].append(float(max(y_data)))
                    sum_stats[channel]['std'].append(float(np.std(y_data)))
                    sum_stats[channel]['mean'].append(float(np.mean(y_data)))
                    sum_stats[channel]['abs'].append(float(max(np.abs(y_data))))
                    sum_stats[channel]['integrated'].append(float(np.trapz(fd['Time'], y_data)))

                    if len(self.channels_extreme_table) > 0:
                        # outputting user specifed channels at the time where the maximum value occurs
                        if channel not in extreme_table:
                            extreme_table[channel] = []

                        extreme_table_i = {}
                        idx_max = np.argmax(y_data)
                        for var in self.channels_extreme_table:
                            extreme_table_i[var] = {}
                            extreme_table_i[var]['time'] = fd['Time'][idx_max]
                            extreme_table_i[var]['val']  = fd[var][idx_max]
                            
                        extreme_table[channel].append(extreme_table_i)

                    # except ValueError:
                    #     print('Error loading data from {}.'.format(channel))
                    # except:
                    #     print('{} is not in available OpenFAST output data.'.format(channel))

            # if self.channels_magnitude:
            #     for channel in self.channels_magnitude.keys():


            # Add DELS to summary stats
            if self.DEL_info:
                for channel, m in self.DEL_info:
                    if channel not in sum_stats.keys():
                        print('Cannot get DELs for {} because it does not exist in output data.'.format(channel))
                        break
                    if 'DEL' not in sum_stats[channel].keys():
                        sum_stats[channel]['DEL'] = []
                    
                    dfDEL = self.get_DEL([fd], [(channel, m)], t=fd['Time'][-1])
                    sum_stats[channel]['DEL'].append(float(dfDEL[channel][0]))


        if len(self.channels_extreme_table) > 0:
            return sum_stats, extreme_table
        else:
            return sum_stats


    def load_ranking(self, stats, names=[], get_df=False):
        '''
        Find load rankings for desired signals

        Parameters:
        -------
        stats: dict, list, pd.DataFrame
            summary statistic information
        ranking_stats: list
            desired statistics to rank for load ranking (e.g. ['max', 'std'])
        ranking_vars: list
            desired variables to for load ranking (e.g. ['GenTq', ['RootMyb1', 'RootMyb2', 'RootMyb3']]) 
        names: list of strings, optional
            names corresponding to each dataset
        get_df: bool, optional
            Return pd.DataFrame of data?
        
        Returns:
        -------
        load_ranking: dict
            dictionary containing load rankings
        load_ranking_df: pd.DataFrame
            pandas DataFrame containing load rankings
        '''
        
        # Make sure stats is in pandas df
        if isinstance(stats, dict):
            stats_df = pdTools.dict2df([stats], names=names)
        elif isinstance(stats, list):
            stats_df = pdTools.dict2df(stats, names=names)
        elif isinstance(stats, pd.DataFrame):
            stats_df = stats
        else:
            raise TypeError('Input stats is must be a dictionary, list, or pd.DataFrame containing OpenFAST output statistics.')


        # Ensure naming consitency
        if not names:
            names = list(stats_df.columns.levels[0])

        if self.verbose:
            print('Calculating load rankings.')
            
        # Column names to search in stats_df
        #  - [name, variable, stat],  i.e.['DLC1.1','TwrBsFxt','max']
        cnames = [pd.MultiIndex.from_product([names, var, [stat]])
                for var, stat in zip(self.ranking_vars, self.ranking_stats)]

        rank__ascending = False
        # Collect load rankings
        collected_rankings = []
        for col in cnames:
            # Set column names for dataframe
            mi_name = list(col.levels[0])
            mi_stat = col.levels[2]  # length = 1
            mi_idx = col.levels[2][0] + '_case_idx'
            if len(col.levels[1]) > 1:
                mi_var = [col.levels[1][0][:-1]]
            else:
                mi_var = list(col.levels[1])
            mi_colnames = pd.MultiIndex.from_product([mi_name, mi_var, [mi_idx, mi_stat[0]]])

            # Check for valid stats
            for c in col:
                if c not in list(stats_df.columns.values):
                    print('WARNING: {} does not exist in statistics.'.format(c))
                    col = col.drop(c)
                    # raise ValueError('{} does not exist in statistics'.format(c))
            # Go to next case if no [stat, var] exists in this set
            if len(col) == 0:
                continue
            # Extract desired variables from stats dataframe
            if mi_stat in ['max', 'abs']:
                var_df = stats_df[col].max(axis=1, level=0)
                rank__ascending = False
            elif mi_stat in ['min']:
                var_df = stats_df[col].min(axis=1, level=0)
                rank__ascending = True
            elif mi_stat in ['mean', 'std']:
                var_df = stats_df[col].mean(axis=1, level=0)
                rank__ascending = False

            # Combine ranking dataframes for each dataset
            var_df_list = [var_df[column].sort_values(
                ascending=rank__ascending).reset_index() for column in var_df.columns]
            single_lr = pd.concat(var_df_list, axis=1)
            single_lr.columns = mi_colnames
            collected_rankings.append(single_lr)

        # Combine dataframes for each case
        load_ranking_df = pd.concat(collected_rankings, axis=1).sort_index(axis=1)
        # Generate dict of info
        load_ranking = pdTools.df2dict(load_ranking_df)

        if get_df:
            return load_ranking, load_ranking_df
        else:
            return load_ranking

    def get_DEL(self, fast_data, chan_info, binNum=100, t=600):
        """ Calculates the short-term damage equivalent load of multiple variables
        
        Parameters: 
        -----------
        fast_data: list
            List of dictionaries containing openfast output data (returned from ROSCO_toolbox.FAST_IO.load_output)
        
        chan_info : list, tuple
            tuple/list containing channel names to be analyzed and corresponding fatigue slope factor "m"
            ie. ('TwrBsFxt',4)
        
        binNum : int
            number of bins for rainflow counting method (minimum=100)
        
        t : float/int
            Used to control DEL frequency. Default for 1Hz is 600 seconds for 10min data
        
        Outputs:
        -----------
        dfDEL : pd.DataFrame
            Damage equivalent load of each specified variable for one fast output file  
        """
        # check data types
        assert isinstance(fast_data, (list)), 'fast_data must be of type list'
        assert isinstance(chan_info, (list,tuple)), 'chan_info must be of type list or tuple'
        assert isinstance(binNum, (float,int)), 'binNum must be of type float or int'
        assert isinstance(t, (float,int)), 't must be of type float or int'

        # create dictionary from chan_dict
        dic = dict(chan_info)

        # pre-allocate list
        dflist = []
        names = []

        for fd in fast_data:
            dlist = [] # initiate blank list every loop
            # loop through channels and apply corresponding fatigue slope
            for var in dic.keys():
                if self.verbose:
                    print('Calculating DEL for {} in {}'.format(var, fd['meta']['name']))
                # find rainflow ranges
                ranges = fatpack.find_rainflow_ranges(fd[var])

                # find range count and bin
                Nrf, Srf = fatpack.find_range_count(ranges,binNum)

                # get DEL
                DELs = Srf**dic[var] * Nrf / t
                DEL = DELs.sum() ** (1/dic[var])
                dlist.append(DEL)
            # append DEL values for each channel to master list
            dflist.append(dlist)

            # create dataframe to return
            dfDEL = pd.DataFrame(np.transpose(dflist))
            dfDEL = dfDEL.T
            dfDEL.columns = dic.keys()
            
            #save simulation names
            names.append(fd['meta']['name'])
        dfDEL['Case_Name'] = names
        
        return dfDEL

class Power_Production(object):
    '''
    Class to generate power production stastics
    '''
    def __init__(self, **kwargs):
        # Turbine parameters
        self.turbine_class = 2

        for k, w in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        super(Power_Production, self).__init__()

    def prob_WindDist(self, windspeed, disttype='pdf'):
        ''' 
        Generates the probability of a windspeed given the cumulative distribution or probability
        density function of a Weibull distribution per IEC 61400.

        NOTE: This uses the range of wind speeds simulated over, so if the simulated wind speed range
        is not indicative of operation range, using this cdf to calculate AEP is invalid
        
        Parameters:
        -----------
        windspeed: float or list-like
            wind speed(s) to calculate probability of 
        disttype: str, optional      
            type of probability, currently supports CDF or PDF
        Outputs:
        ----------
        p_bin: list
            list containing probabilities per wind speed bin 
        '''
        if self.turbine_class in [1, 'I']:
            Vavg = 50 * 0.2
        elif self.turbine_class in [2, 'II']:
            Vavg = 42.5 * 0.2
        elif self.turbine_class in [3, 'III']:
            Vavg = 37.5 * 0.2

        # Define parameters
        k = 2 # Weibull shape parameter
        c = (2 * Vavg)/np.sqrt(np.pi) # Weibull scale parameter 

        if disttype.lower() == 'cdf':
            # Calculate probability of wind speed based on WeibulCDF
            wind_prob = 1 - np.exp(-(windspeed/c)**k)
        elif disttype.lower() == 'pdf':
            # Calculate probability of wind speed based on WeibulPDF
            wind_prob = (k/c) * (windspeed/c)**(k-1) * np.exp(-(windspeed/c)**k)
        else:
            raise ValueError('The {} probability distribution type is invalid'.format(disttype))
        
        return wind_prob

    def AEP(self, stats, windspeeds, U_pwr_curve=[], pwr_curve_vars=[]):
        '''
        Get AEPs for simulation cases

        TODO: Print/Save this someplace besides the console
    
        Parameters:
        ----------
        stats: dict, list, pd.DataFrame
            Dict (single case), list(multiple cases), df(single or multiple cases) containing
            summary statistics. 
        windspeeds: list-like
            List of wind speed values corresponding to each power output in the stats input 
            for a single dataset

        n_pwr_curve: list-like
            List of wind speed values to output power variables

        pwr_curve_vars: list of strings
            List of OpenFAST output channels to return the mean value as a function of wind speeds

        Returns:
        --------
        AEP: List
            Annual energy production corresponding to 
        '''
  
        # Make sure stats is in pandas df
        if isinstance(stats, dict):
            stats_df = pdTools.dict2df(stats)
        elif isinstance(stats, list):
            stats_df = pdTools.dict2df(stats)
        elif isinstance(stats, pd.DataFrame):
            stats_df = stats
        else: 
            raise TypeError('Input stats is must be a dictionary, list, or pd.DataFrame containing OpenFAST output statistics.')

        # Check windspeed length
        if len(windspeeds) == len(stats_df):
            ws = windspeeds
        elif int(len(windspeeds)/len(stats_df.columns.levels[0])) == len(stats_df):
            ws = windspeeds[0:len(stats_df)]
            print('WARNING: Assuming the input windspeed array is duplicated for each dataset.')
        else:
            raise ValueError(
                'Length of windspeeds is not the correct length for the input statistics.')

        # load power array
        if 'GenPwr' in stats_df.columns.levels[0]:
            pwr_array = stats_df.loc[:, ('GenPwr', 'mean')]
            pwr_array = pwr_array.to_frame()
        elif 'GenPwr' in stats_df.columns.levels[1]:
            pwr_array = stats_df.loc[:, (slice(None), 'GenPwr', 'mean')]
        else:
            raise ValueError("('GenPwr','Mean') does not exist in the input statistics.")
        
        # group and average powers by wind speeds
        pwr_array['windspeeds'] = ws 
        pwr_array = pwr_array.groupby('windspeeds').mean() 
        # find set of wind speeds
        ws_set = list(set(ws))
        # wind probability
        wind_prob = self.prob_WindDist(ws_set, disttype='pdf')
        # Calculate AEP
        AEP = np.trapz(pwr_array.T *  wind_prob, ws_set) * 8760

        # return power curves
        if len(pwr_curve_vars) > 0:
            performance_curves = {}
            if len(U_pwr_curve) > 0:
                performance_curves['U'] = U_pwr_curve
            else:
                performance_curves['U'] = ws_set

            for var in pwr_curve_vars:
                # get data
                if var in stats_df.columns.levels[0]:
                    perf_array = stats_df.loc[:, (var, 'mean')]
                    perf_array = perf_array.to_frame()
                elif var in stats_df.columns.levels[1]:
                    perf_array = stats_df.loc[:, (slice(None), var, 'mean')]
                else:
                    raise ValueError("(%s,'Mean') does not exist in the input statistics."%var)
                # average by wind speed
                perf_array['windspeeds'] = ws 
                perf_array = perf_array.groupby('windspeeds').mean()

                if len(U_pwr_curve) > 0:
                    spline = PchipInterpolator(ws_set, perf_array[var])
                    performance_curves[var] = spline(performance_curves['U']).flatten()
                else:
                    performance_curves[var] = perf_array[var]

        

        if len(pwr_curve_vars) > 0:
            return AEP, performance_curves
        else:
            return AEP

class wsPlotting(object):
    '''
    General plotting scripts.
    '''

    def __init__(self):
        pass

    def stat_curve(self, windspeeds, stats, plotvar, plottype, stat_idx=0, names=[]):
        '''
        Plot the turbulent power curve for a set of data. 
        Can be plotted as bar (good for comparing multiple cases) or line 

        Parameters:
        -------
        windspeeds: list-like
            List of wind speeds to plot
        stats: list, dict, or pd.DataFrame
            Dict (single case), list(multiple cases), df(single or multiple cases) containing
            summary statistics. 
        plotvar: str
            Type of variable to plot
        plottype: str
            bar or line 
        stat_idx: int, optional
            Index of datasets in stats to plot from
        
        Returns:
        --------
        fig: figure handle
        ax: axes handle
        '''

        # Check for valid inputs
        if isinstance(stats, dict):
            stats_df = pdTools.dict2df(stats)
            if any((stat_inds > 0) or (isinstance(stat_inds, list))):
                print('WARNING: stat_ind = {} is invalid for a single stats dictionary. Defaulting to stat_inds=0.')
                stat_inds = 0
        elif isinstance(stats, list):
            stats_df = pdTools.dict2df(stats)
        elif isinstance(stats, pd.DataFrame):
            stats_df = stats
        else:
            raise TypeError(
                'Input stats must be a dictionary, list, or pd.DataFrame containing OpenFAST output statistics.')

       
        # Check windspeed length
        if len(windspeeds) == len(stats_df):
            ws = windspeeds
        elif int(len(windspeeds)/len(stats_df.columns.levels[0])) == len(stats_df):
            ws = windspeeds[0:len(stats_df)]
        else:
            raise ValueError('Length of windspeeds is not the correct length for the input statistics')

        # Get statistical data for desired plot variable
        if plotvar in stats_df.columns.levels[0]:
            sdf = stats_df.loc[:, (plotvar, slice(None))].droplevel([0], axis=1)
        elif plotvar in stats_df.columns.levels[1]:
            sdf = stats_df.loc[:, (slice(None), plotvar, slice(None))].droplevel([1], axis=1)
        else:
            raise ValueError("{} does not exist in the input statistics.".format(plotvar))
        
        # Add windspeeds to data
        sdf['WindSpeeds']= ws
        # Group by windspeed and average each statistic (for multiple seeds)
        sdf = sdf.groupby('WindSpeeds').mean() 
        # Final wind speed values
        pl_windspeeds=sdf.index.values

        if plottype == 'bar':
            # Define mean and std dataframes
            means = sdf.loc[:, (slice(None), 'mean')].droplevel(1, axis=1)
            std = sdf.loc[:, (slice(None), 'std')].droplevel(1, axis=1)
            # Plot bar charts
            fig, ax = plt.subplots(constrained_layout=True)
            means.plot.bar(yerr=std, ax=ax, title=plotvar, capsize=2)
            ax.legend(names,loc='upper left')

        if plottype == 'line':
            # Define mean, min, max, and std dataframes
            means = sdf.loc[:, (sdf.columns.levels[0][stat_idx], 'mean')]
            smax = sdf.loc[:, (sdf.columns.levels[0][stat_idx], 'max')]
            smin = sdf.loc[:, (sdf.columns.levels[0][stat_idx], 'min')]
            std = sdf.loc[:, (sdf.columns.levels[0][stat_idx], 'std')]

            fig, ax = plt.subplots(constrained_layout=True)
            ax.errorbar(pl_windspeeds, means, [means - smin, smax - means],
                         fmt='k', ecolor='gray', lw=1, capsize=2)
            means.plot(yerr=std, ax=ax, 
                        capsize=2, lw=3, 
                        elinewidth=2, 
                        title=names[0] + ' - ' + plotvar)
            plt.grid(lw=0.5, linestyle='--')

        return fig, ax


    def distribution(self, fast_data, channels, caseid, names=None, kde=True):
        '''
        Distributions of data from desired fast runs and channels

        Parameters
        ----------
        fast_data: dict, list
            List or Dictionary containing OpenFAST output data from desired cases to compare
        channels: list
            List of strings of OpenFAST output channels e.g. ['RotSpeed','GenTq']
        caseid: list
            List of caseid's to compare
        names: list, optional
            Names of the runs to compare
        fignum: ind, (optional)
            Specified figure number. Useful to plot over previously made plot
        
        Returns:
        --------
        fig: figure handle
        ax: axes handle
        '''
        # Make sure input types allign
        if isinstance(fast_data, dict):
            fd = [fast_data]
        elif isinstance(fast_data, list):
            if len(caseid) == 1:
                fd = [fast_data[caseid[0]]]
            else:
                fd = [fast_data[case] for case in caseid]
        else:
            raise ValueError('fast_data is an improper data type')
            

        # if not names:
        #     names = [[]]*len(fd)

        for channel in channels:
            fig, ax = plt.subplots()
            for idx, data in enumerate(fd):
                # sns.kdeplot(data[channel], shade=True, label='case '+ str(idx))
                sns.distplot(data[channel], kde=kde, label='case ' + str(idx))  # For a histogram
                ax.set_title(channel + ' distribution')

                units = data['meta']['attribute_units'][data['meta']['channels'].index(channel)]
                ax.set_xlabel('{} [{}]'.format(channel, units))
                ax.grid(True)
            if names:
                ax.legend(names)
                
        return fig, ax

    def plot_load_ranking(self, load_rankings, case_matrix, classifier_type, 
                        classifier_names=[], n_rankings=10, caseidx_labels=False):
        '''
        Plot load rankings
        
        Parameters:
        -----------
        load_rankings: list, dict, or pd.DataFrame
            Dict (single case), list(multiple cases), df(single or multiple cases) containing
            load rankings. 
        case_matrix: dict or pdDataFrame
            Information mapping classifiers to load_rankings. 
            NOTE: the case matrix must have wind speeds in ('InflowWind','WindSpeeds') if you 
            wish to plot w.r.t. wind speed
        classifier_type: tuple or str
            classifier to denote load ranking cases. e.g. classifier_type=('IEC','DLC') will separate
            the load rankings by DLC type, assuming the case matrix is properly set up to map each
            DLC to the load ranking case
        classifier_names: list, optional
            Naming conventions for each classifier type for plotting purposes
        n_rankings: int, optional
            number of load rankings to plot
        caseidx_labels: bool, optional
            label x-axis with case index if True. If false, will try to plot with wind speed labels
            if they exist, then fall abck to case indeces.  

        TODO: Save figs
        '''

        # flag_DLC_name = False
        # n_rankings = 10
        # fig_ext = '.pdf'
        # font_size = 10
        # classifier_type = ('ServoDyn', 'DLL_FileName')
        # classifiers = list(set(cmw[classifier_type]))
        # classifier_names = ['ROSCO', 'legacy']
       
       # Check for valid inputs
        if isinstance(load_rankings, dict):
            load_ranking_df = pdTools.dict2df(load_rankings)
        elif isinstance(load_rankings, list):
            load_ranking_df = pdTools.dict2df(load_rankings)
        elif isinstance(load_rankings, pd.DataFrame):
            load_ranking_df = load_rankings
        else:
            raise TypeError(
                'Input stats must be a dictionary, list, or pd.DataFrame containing OpenFAST output statistics.')

        # Check multiindex size
        if len(load_ranking_df) == 2:
            load_ranking_df = pd.concat([load_ranking_df], keys=[dataset_0])

        # Check for classifier_names
        classifiers = list(set(case_matrix[classifier_type]))
        if not classifier_names:
            classifier_names = ['datatset_{}'.format(idx) for idx in range(len(classifiers))]

        # Check for wind speeds in case_matrix
        if not caseidx_labels:
            try:
                windspeeds = case_matrix[('InflowWind','WindSpeed')]
            except: 
                print('Unable to find wind speeds in case_matrix, plotting w.r.t case index')    
                caseidx_labels=True


        # Define a color map
        clrs = np.array([[127, 60, 141],
                        [17, 165, 121],
                        [57, 105, 172],
                        [242, 183, 1],
                        [231, 63, 116],
                        [128, 186, 90],
                        [230, 131, 16],
                        [256, 256, 256]]) / 256.

        # Get channel names
        channels = load_ranking_df.columns.levels[1]

        # initialize some variables
        colors = np.zeros((n_rankings, 3))
        labels = [''] * n_rankings
        labels_index = [''] * n_rankings
        fig_list = []
        ax_list = []
        # --- Generate plots ---
        for cidx, channel in enumerate(channels):
            # Pull out specific channel
            cdf = load_ranking_df.loc[:, (slice(None), channel, slice(None))].droplevel(1, axis=1)
            # put the load ranking from each dataset in a list so we can combine them
            cdf_list = [cdf[dataset] for dataset in cdf.columns.levels[0]] 
            chan_df = pd.concat(cdf_list) # combine all load rankings
            chan_stats = chan_df.columns.values # pull out the names of the columns
            chan_df.sort_values(by=chan_stats[0], ascending=False, inplace=True) # sort
            chan_df.reset_index(inplace=True, drop=True) # re-index

            # find colors and labels for plots
            for i in range(n_rankings):
                classifier = case_matrix[classifier_type][chan_df[chan_stats[1]][i]]
                colors[i, :] = clrs[min(len(clrs), classifiers.index(classifier))]

                if not caseidx_labels:
                    ws = windspeeds[chan_df[chan_stats[1]][i]]
                    labels[i] = classifier_names[classifiers.index(classifier)] + ' - ' + str(ws) + ' m/s'
                else:
                    labels[i] = classifier_names[classifiers.index(classifier)] + ' - Case ' + str(chan_df[chan_stats[1]][i])
        #         labels_index = ['case {}'.format(case) for case in chan_df[chan_stats[1]][0:n_rankings]]

            # make plot
            fig, ax = plt.subplots()
            chan_df[chan_stats[0]][0:n_rankings].plot.bar(color=colors)
            ax.set_ylabel(channel)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            plt.draw()

            fig_list.append(fig)
            ax_list.append(ax)

        #     if case_idx_labels:
        #         ax.set_xlabel('DLC [-]', fontsize=font_size+2, fontweight='bold')
        # #         ax.set_xticklabels(np.arange(n_rankings), labels=labels)
        #         ax.set_xticklabels(labels)
        #     else:
        #         #         ax.set_xticklabels(np.arange(n_rankings), labels=labels)

            


        return fig_list, ax_list
