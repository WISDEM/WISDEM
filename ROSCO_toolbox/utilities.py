# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from itertools import takewhile, product
import struct

try:
   from wisdem.aeroelasticse.Util import spectral
except:
    pass

# Some useful constants
now = datetime.datetime.now()
pi = np.pi
rad2deg = np.rad2deg(1)
deg2rad = np.deg2rad(1)
rpm2RadSec = 2.0*(np.pi)/60.0
RadSec2rpm = 60/(2.0 * np.pi)

class FAST_Plots():
    '''
    Some plotting utilities for OpenFAST data. 

    Methods:
    plot_fast_out
    plot_spectral
    '''

    def __init__(self):
        pass
    
    def plot_fast_out(self, cases, fast_dict, showplot=False, fignum=None, xlim=None):
        '''
        Plots OpenFAST outputs for desired channels

        Parameters:
        -----------
        cases : dict
            Dictionary of lists containing desired outputs
        fast_dict : dict
            Dictionary of OpenFAST output information, output from load_fast_out
        showplot: bool, optional
            Show the plot
        fignum: int, optional
            Define figure number. Note: Should only be used when plotting a singular case. 

        Returns:
        --------
        figlist: list
            list of figure handles
        axeslist: list
            list of axes handles
        '''
        figlist = []
        axeslist = []
        # Plot cases
        for case in cases.keys():
            # channels to plot
            channels = cases[case]
            # instantiate plot and legend
            fig, axes = plt.subplots(len(channels), 1, sharex=True, num=fignum)

            myleg = []
            for fast_out in fast_dict:
                # write legend
                Time = fast_out['Time']
                myleg.append(fast_out['meta']['name'])
                if len(channels) > 1:  # Multiple channels
                    for axj, channel in zip(axes, channels):
                        try:
                            # plot
                            axj.plot(Time, fast_out[channel])
                            # label
                            unit_idx = fast_out['meta']['channels'].index(channel)
                            axj.set(ylabel='{:^} \n ({:^})'.format(
                                channel,
                                fast_out['meta']['attribute_units'][unit_idx]))
                            axj.grid(True)
                        except:
                            print('{} is not available as an output channel.'.format(channel))
                    axes[0].set_title(case)
                else:                   # Single channel
                    try:
                        # plot
                        axes.plot(Time, fast_out[channel])
                        # label
                        axes.set(ylabel='{:^} \n ({:^})'.format(
                            channel,
                            fast_out['meta']['attribute_units'][unit_idx]))
                        axes.grid(True)
                        axes.set_title(case)
                    except:
                        print('{} is not available as an output channel.'.format(channel))
                plt.legend(myleg, loc='upper center', bbox_to_anchor=(
                    0.5, 0.0), borderaxespad=2, ncol=len(fast_dict))

            figlist.append(fig)
            axeslist.append(axes)

            if xlim:
                plt.xlim(xlim)

        if showplot:
            plt.show()

        return figlist, axeslist

    def plot_spectral(self, fast_dict, cases,
                      averaging='None', averaging_window='Hann', detrend=False, nExp=None,
                      show_RtSpeed=False, RtSpeed_idx=None,
                      add_freqs=None, add_freq_labels=None,
                      showplot=False, fignum=None):
        '''
        Plots OpenFAST outputs for desired channels

        Parameters:
        -----------
        fast_dict : dict
            Dictionary of OpenFAST output information, output from load_fast_out
        cases : list of tuples (str, int)
            Dictionary of lists containing desired outputs. 
            Of the format (channel, case), i.e. [('RotSpeed', 0)]
        averaging: str, optional
            PSD averaging method. None, Welch
        averaging_window: str, optional
            PSD averaging window method. Hamming, Hann, Rectangular
        detrend: bool, optional
            Detrend data?
        nExp: float, optional
            Exponent for hamming windowing
        show_RtSpeed: Bool, optional
            Plot 1p and 3p rotor speeds for simulation cases plotted
        RtSpeed_idx: ind, optional
            Specify the index for the simulation case that the rotor speed is plotted from. 
        add_freqs: list, optional
            List of floats containing additional frequencies to plot lines of
        add_freq_labels: list, optional
            List of strings to label add_freqs
        showplot: bool, optional
            Show the plot
        fignum: int, optional
            Define figure number. Note: Should only be used when plotting a singular case. 

        Returns:
        -------
        fig, ax - corresponds to generated figure
        '''
        if averaging.lower() not in ['none', 'welch']:
            raise ValueError('{} is not a supported averaging method.'.format(averaging))

        if averaging_window.lower() not in ['hamming', 'hann', 'rectangular']:
            raise ValueError('{} is not a supported averaging window.'.format(averaging_window))

        fig, ax = plt.subplots(num=fignum)

        leg = []
        for channel, run in cases:
            try:
                # Find time
                Time = fast_dict[run]['Time']
                # Load PSD
                fq, y, info = spectral.fft_wrap(
                    Time, fast_dict[run][channel], averaging=averaging, averaging_window=averaging_window, detrend=detrend, nExp=nExp)
                # Plot data
                plt.loglog(fq, y, label='{}, run: {}'.format(channel, str(run)))
            except:
                print('{} is not an available channel in run {}'.format(channel, str(run)))
        # Show rotor speed range (1P, 3P, 6P?)
        if show_RtSpeed:
            if not RtSpeed_idx:
                RtSpeed_idx = [0]
                print('No rotor speed run indices defined, plotting spectral range for the first run only.')
            for rt_idx in RtSpeed_idx:
                f_1P_min = min(fast_dict[rt_idx]['RotSpeed']) / 60.  # Hz
                f_1P_max = max(fast_dict[rt_idx]['RotSpeed']) / 60.
                f_3P_min = f_1P_min*3
                f_3P_max = f_1P_max*3
                f_6P_min = f_1P_min*6
                f_6P_max = f_1P_max*6
                plt.axvspan(f_1P_min, f_1P_max, alpha=0.5, color=[0.7, 0.7, 0.7], label='1P')
                plt.axvspan(f_3P_min, f_3P_max, alpha=0.5, color=[0.8, 0.8, 0.8], label='3P')
                # if f_6P_min<1.:
                #     plt.axvspan(f_6P_min, f_6P_max, alpha=0.5, color=[0.9,0.9,0.9], label='6P')

        # Add specific frequencies if desired
        if add_freqs:
            co = np.linspace(0.3, 0.0, len(add_freqs))
            if add_freq_labels is None:
                add_freq_labels = [None]*len(add_freqs)
            for freq, flabel, c in zip(add_freqs, add_freq_labels, co):
                plt.axvline(freq, color=[c, c, c])  # , label=flabel)
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                plt.text(freq+(10**np.floor(np.log10(freq))/10), 0.01, flabel, transform=trans)

        # Formatting
        plt.legend(loc='best')
        if len(list(cases)) > 1:
            plt.ylabel('PSD')
        else:
            unit_idx = fast_dict[run]['meta']['channels'].index(channel)
            plt.ylabel(
                'PSD ({}$^2$/Hz)'.format(fast_dict[run]['meta']['attribute_units'][unit_idx]))
        plt.xlabel('Frequency (Hz)')
        plt.grid(True)

        if showplot:
            plt.show()

        return fig, ax

class FAST_IO():
    ''' 
    A collection of utilities that may be useful for using the tools made accessbile in this toolbox with OpenFAST

    A number of the file processing tools used here were provided by or modified from Emanual Branlard's weio library: https://github.com/ebranlard/weio. 

    Methods:
    --------
    run_openfast
    load_fast_out
    load_ascii_output
    load_binary_output
    trim_output

    '''
    def __init__(self):
        pass

    def run_openfast(self,fast_dir,fastcall='OpenFAST',fastfile=None,chdir=False):
        '''
        Runs a openfast openfast simulation.
        
        ** Note ** 
        If running ROSCO, this function must be called from the same folder containing DISCON.IN,
            or the chdir flag must be turned on. This is an artifact of OpenFAST looking for DISCON.IN
            in the folder that it is called from. 

        Parameters:
        ------------
            fast_dir: string
                    Name of OpenFAST directory containing input files.
            fast_file: string
                    Name of OpenFAST directory containing input files.
            fastcall: string, optional
                    Line used to call openfast when executing from the terminal.
            fastfile: string, optional
                    Filename for *.fst input file. Function will find *.fst if not provided.
            chdir: bool, optional
                    Change directory to openfast model directory before running.
        '''

        # Define OpenFAST input filename
        if not fastfile:
            for file in os.listdir(fast_dir):
                if file.endswith('.fst'):
                    fastfile = file
        print('Using {} to run OpenFAST simulation'.format(fastfile))

        if chdir: # Change cwd before calling OpenFAST -- note: This is an artifact of needing to call OpenFAST from the same directory as DISCON.IN
            # save starting file path 
            original_path = os.getcwd()
            # change path, run OpenFAST
            os.chdir(fast_dir)
            print('Running OpenFAST simulation for {} through the ROSCO toolbox...'.format(fastfile))
            os.system('{} {}'.format(fastcall, os.path.join(fastfile)))
            print('OpenFAST simulation complete. ')
            # return to original path
            os.chdir(original_path)
        else:
            # Run OpenFAST
            print('Running OpenFAST simulation for {} through the ROSCO toolbox...'.format(fastfile))
            os.system('{} {}'.format(fastcall, os.path.join(fast_dir,fastfile)))
            print('OpenFAST simulation complete. ')


    def load_FAST_out(self, filenames, tmin=None, tmax=None, verbose=False):
        """Load a FAST binary or ascii output file
        
        Parameters
        ----------
        filenames : list
            list of filenames
        tmin : float, optional
            initial time to trim output data to 
        tmax : float, optional
            final data to trim output data to
        verbose : bool, optional
            Print updates
        
        Returns
        -------
        fastout: list
            List of dictionaries containing OpenFAST output data.
        """
        if type(filenames) is str:
            filenames = [filenames]
            
        # data = []
        # info = []
        fastout = []
        for i, filename in enumerate(filenames):
            assert os.path.isfile(filename), "File, %s, does not exists" % filename
            with open(filename, 'r') as f:
                if verbose:
                    print('Loading data from {}'.format(filename))
                try:
                    f.readline()
                except UnicodeDecodeError:
                    data, info = self.load_binary_output(filename)
                else:
                    data, info = self.load_ascii_output(filename)

            # Build dictionary
            fast_data = dict(zip(info['channels'],data.T))
            fast_data['meta'] = info
            fast_data['meta']['filename'] = filename
            fastout.append(fast_data)

        # Trim outputs
        if (tmin) or (tmax):
            self.trim_output(fastout, tmin=tmin, tmax=tmax, verbose=verbose)

        # return fastout
        return fastout


    def load_ascii_output(self, filename):
        '''
        Load FAST ascii output file 
        
        Parameters
        ----------
        filename : str
            filename
        
        Returns
        -------
        data : ndarray
            data values
        info : dict
            info containing:
                - name: filename
                - description: description of dataset
                - channels: list of attribute names
                - attribute_units: list of attribute units
        '''
        with open(filename) as f:
            info = {}
            info['name'] = os.path.splitext(os.path.basename(filename))[0]
            # Header is whatever is before the keyword `time`
            in_header = True
            header = []
            while in_header:
                l = f.readline()
                if not l:
                    raise Exception('Error finding the end of FAST out file header. Keyword Time missing.')
                in_header= (l+' dummy').lower().split()[0] != 'time'
                if in_header:
                    header.append(l)
                else:
                    info['description'] = header
                    info['channels'] = l.split()
                    info['attribute_units'] = [unit[1:-1] for unit in f.readline().split()]

            # Data, up to end of file or empty line (potential comment line at the end)
            data = np.array([l.strip().split() for l in takewhile(lambda x: len(x.strip())>0, f.readlines())]).astype(np.float)
            return data, info


    def load_binary_output(self, filename, use_buffer=True):
        """
                
        Info about ReadFASTbinary.m:
        
        Original Author: Bonnie Jonkman, National Renewable Energy Laboratory
        (c) 2012, National Renewable Energy Laboratory
        Edited for FAST v7.02.00b-bjj  22-Oct-2012
        03/09/15: Ported from ReadFASTbinary.m by Mads M Pedersen, DTU Wind
        10/24/18: Low memory/buffered version by E. Branlard, NREL
        18/01/19: New file format for exctended channels, by E. Branlard, NREL
        11/4/19: Implemented in ROSCO toolbox by N. Abbas, NREL
        8/6/20: Synced between rosco toolbox and weio by P Bortolotti, NREL

        Parameters
        ----------
        filename : str
            filename
        Returns
        -------
        data : ndarray
            data values
        info : dict
            info containing:
                - name: filename
                - description: description of dataset
                - channels: list of attribute names
                - attribute_units: list of attribute units
        """
        def fread(fid, n, type):
            fmt, nbytes = {'uint8': ('B', 1), 'int16':('h', 2), 'int32':('i', 4), 'float32':('f', 4), 'float64':('d', 8)}[type]
            return struct.unpack(fmt * n, fid.read(nbytes * n))

        def freadRowOrderTableBuffered(fid, n, type_in, nCols, nOff=0, type_out='float64'):
            """ 
            Reads of row-ordered table from a binary file.
            Read `n` data of type `type_in`, assumed to be a row ordered table of `nCols` columns.
            Memory usage is optimized by allocating the data only once.
            Buffered reading is done for improved performances (in particular for 32bit python)
            `nOff` allows for additional column space at the begining of the storage table.
            Typically, `nOff=1`, provides a column at the beginning to store the time vector.
            @author E.Branlard, NREL
            """
            fmt, nbytes = {'uint8': ('B', 1), 'int16':('h', 2), 'int32':('i', 4), 'float32':('f', 4), 'float64':('d', 8)}[type_in]
            nLines          = int(n/nCols)
            GoodBufferSize  = 4096*40
            nLinesPerBuffer = int(GoodBufferSize/nCols)
            BufferSize      = nCols * nLinesPerBuffer
            nBuffer         = int(n/BufferSize)
            # Allocation of data
            data = np.zeros((nLines,nCols+nOff), dtype = type_out)
            # Reading
            try:
                nIntRead   = 0
                nLinesRead = 0
                while nIntRead<n:
                    nIntToRead = min(n-nIntRead, BufferSize)
                    nLinesToRead = int(nIntToRead/nCols)
                    Buffer = np.array(struct.unpack(fmt * nIntToRead, fid.read(nbytes * nIntToRead)))
                    Buffer = Buffer.reshape(-1,nCols)
                    data[ nLinesRead:(nLinesRead+nLinesToRead),  nOff:(nOff+nCols)  ] = Buffer
                    nLinesRead = nLinesRead + nLinesToRead
                    nIntRead   = nIntRead   + nIntToRead
            except:
                raise Exception('Read only %d of %d values in file:' % (nIntRead, n, filename))
            return data


        FileFmtID_WithTime              = 1 # File identifiers used in FAST
        FileFmtID_WithoutTime           = 2
        FileFmtID_NoCompressWithoutTime = 3
        FileFmtID_ChanLen_In            = 4

        with open(filename, 'rb') as fid:
            #----------------------------        
            # get the header information
            #----------------------------

            FileID = fread(fid, 1, 'int16')[0]  #;             % FAST output file format, INT(2)

            if FileID not in [FileFmtID_WithTime, FileFmtID_WithoutTime, FileFmtID_NoCompressWithoutTime, FileFmtID_ChanLen_In]:
                raise Exception('FileID not supported {}. Is it a FAST binary file?'.format(FileID))

            if FileID == FileFmtID_ChanLen_In: 
                LenName = fread(fid, 1, 'int16')[0] # Number of characters in channel names and units
            else:
                LenName = 10                    # Default number of characters per channel name

            NumOutChans = fread(fid, 1, 'int32')[0]  #;             % The number of output channels, INT(4)
            NT = fread(fid, 1, 'int32')[0]  #;             % The number of time steps, INT(4)

            if FileID == FileFmtID_WithTime:
                TimeScl = fread(fid, 1, 'float64')  #;           % The time slopes for scaling, REAL(8)
                TimeOff = fread(fid, 1, 'float64')  #;           % The time offsets for scaling, REAL(8)
            else:
                TimeOut1 = fread(fid, 1, 'float64')  #;           % The first time in the time series, REAL(8)
                TimeIncr = fread(fid, 1, 'float64')  #;           % The time increment, REAL(8)

            if FileID == FileFmtID_NoCompressWithoutTime:
                ColScl = np.ones ((NumOutChans, 1)) # The channel slopes for scaling, REAL(4)
                ColOff = np.zeros((NumOutChans, 1)) # The channel offsets for scaling, REAL(4)
            else:
                ColScl = fread(fid, NumOutChans, 'float32')  # The channel slopes for scaling, REAL(4)
                ColOff = fread(fid, NumOutChans, 'float32')  # The channel offsets for scaling, REAL(4)

            LenDesc      = fread(fid, 1, 'int32')[0]  #;  % The number of characters in the description string, INT(4)
            DescStrASCII = fread(fid, LenDesc, 'uint8')  #;  % DescStr converted to ASCII
            DescStr      = "".join(map(chr, DescStrASCII)).strip()

            ChanName = []  # initialize the ChanName cell array
            for iChan in range(NumOutChans + 1):
                ChanNameASCII = fread(fid, LenName, 'uint8')  #; % ChanName converted to numeric ASCII
                ChanName.append("".join(map(chr, ChanNameASCII)).strip())

            ChanUnit = []  # initialize the ChanUnit cell array
            for iChan in range(NumOutChans + 1):
                ChanUnitASCII = fread(fid, LenName, 'uint8')  #; % ChanUnit converted to numeric ASCII
                ChanUnit.append("".join(map(chr, ChanUnitASCII)).strip()[1:-1])

            # -------------------------
            #  get the channel time series
            # -------------------------

            nPts = NT * NumOutChans  #;           % number of data points in the file

            if FileID == FileFmtID_WithTime:
                PackedTime = fread(fid, NT, 'int32')  #; % read the time data
                cnt = len(PackedTime)
                if cnt < NT:
                    raise Exception('Could not read entire %s file: read %d of %d time values' % (filename, cnt, NT))

            if use_buffer:
                # Reading data using buffers, and allowing an offset for time column (nOff=1)
                if FileID == FileFmtID_NoCompressWithoutTime:
                    data = freadRowOrderTableBuffered(fid, nPts, 'float64', NumOutChans, nOff=1, type_out='float64')
                else:
                    data = freadRowOrderTableBuffered(fid, nPts, 'int16', NumOutChans, nOff=1, type_out='float64')
            else:
                # NOTE: unpacking huge data not possible on 32bit machines
                if FileID == FileFmtID_NoCompressWithoutTime:
                    PackedData = fread(fid, nPts, 'float64')  #; % read the channel data
                else:
                    PackedData = fread(fid, nPts, 'int16')  #; % read the channel data

                cnt = len(PackedData)
                if cnt < nPts:
                    raise Exception('Could not read entire %s file: read %d of %d values' % (filename, cnt, nPts))
                data = np.array(PackedData).reshape(NT, NumOutChans)
                del PackedData

        if FileID == FileFmtID_WithTime:
            time = (np.array(PackedTime) - TimeOff) / TimeScl;
        else:
            time = TimeOut1 + TimeIncr * np.arange(NT)

        # -------------------------
        #  Scale the packed binary to real data
        # -------------------------
        if use_buffer:
            # Scaling Data
            for iCol in range(NumOutChans):
                if np.isnan(ColScl[iCol]) and np.isnan(ColOff[iCol]):
                    data[:,iCol+1] = 0 # probably due to a division by zero in Fortran
                else:
                    data[:,iCol+1] = (data[:,iCol+1] - ColOff[iCol]) / ColScl[iCol]
            # Adding time column
            data[:,0] = time
        else:
            # NOTE: memory expensive due to time conversion, and concatenation
            data = (data - ColOff) / ColScl
            data = np.concatenate([time.reshape(NT, 1), data], 1)

        info = {'name': os.path.splitext(os.path.basename(filename))[0],
                'description': DescStr,
                'channels': ChanName,
                'attribute_units': ChanUnit}
        return data, info

    def trim_output(self, fast_data, tmin=None, tmax=None, verbose=False):
        '''
        Trim loaded fast output data 
        Parameters
        ----------
        fast_data : list
            List of all output data from load_fast_out (list containing dictionaries)
        tmin : float, optional
            start time
        tmax : float, optional
            end time
        
        Returns
        -------
        fast_data : list
            list of dictionaries containing trimmed fast output data
        '''
        if isinstance(fast_data, dict):
            fast_data = [fast_data]



        # initial time array and associated index
        for fd in fast_data:
            if verbose:
                if tmin: 
                    tmin_v = str(tmin) + ' seconds'
                else:  
                    tmin_v = 'the beginning'
                if tmax: 
                    tmax_v = str(tmax) + ' seconds'
                else: 
                    tmax_v = 'the end'

                print('Trimming output data for {} from {} to {}.'.format(fd['meta']['name'], tmin_v, tmax_v))
            # Find time index range
            if tmin:
                T0ind = np.searchsorted(fd['Time'], tmin)
            else:
                T0ind = 0
            if tmax:
                Tfind = np.searchsorted(fd['Time'], tmax) + 1
            else: 
                Tfind = len(fd['Time'])

            if T0ind+1 > len(fd['Time']):
                raise ValueError('The initial time to trim {} to is after the end of the simulation.'.format(fd['meta']['name']))

            # # Modify time
            fd['Time'] = fd['Time'][T0ind:Tfind] - fd['Time'][T0ind]

            # Remove all vales in data where time is not in desired range
            for key in fd.keys():
                if key.lower() not in ['time', 'meta']:
                    fd[key] = fd[key][T0ind:Tfind]


        return fast_data


class FileProcessing():
    """
    Class FileProcessing used to write out controller 
        parameter files need to run ROSCO

    Methods:
    -----------
    write_DISCON
    read_DISCON
    write_rotor_performance
    load_from_txt
    """

    def __init__(self):
        pass

    def write_DISCON(self, turbine, controller, param_file='DISCON.IN', txt_filename='Cp_Ct_Cq.txt'):
        """
        Print the controller parameters to the DISCON.IN input file for the generic controller

        Parameters:
        -----------
        turbine: class
                 Turbine class containing turbine operation information (ref speeds, etc...)
        controller: class
                    Controller class containing controller operation information (gains, etc...)
        param_file: str, optional
            filename for parameter input file, should be DISCON.IN
        txt_filename: str, optional
                      filename of rotor performance file
        """
        print('Writing new controller parameter file parameter file: %s.' % param_file)
        # Should be obvious what's going on here...
        file = open(param_file,'w')
        file.write('! Controller parameter input file for the %s wind turbine\n' % turbine.TurbineName)
        file.write('!    - File written using ROSCO Controller tuning logic on %s\n' % now.strftime('%m/%d/%y'))
        file.write('\n')
        file.write('!------- DEBUG ------------------------------------------------------------\n')
        file.write('{0:<12d}        ! LoggingLevel		- {{0: write no debug files, 1: write standard output .dbg-file, 2: write standard output .dbg-file and complete avrSWAP-array .dbg2-file}}\n'.format(int(controller.LoggingLevel)))
        file.write('\n')
        file.write('!------- CONTROLLER FLAGS -------------------------------------------------\n')
        file.write('{0:<12d}        ! F_LPFType			- {{1: first-order low-pass filter, 2: second-order low-pass filter}}, [rad/s] (currently filters generator speed and pitch control signals\n'.format(int(controller.F_LPFType)))
        file.write('{0:<12d}        ! F_NotchType		- Notch on the measured generator speed and/or tower fore-aft motion (for floating) {{0: disable, 1: generator speed, 2: tower-top fore-aft motion, 3: generator speed and tower-top fore-aft motion}}\n'.format(int(controller.F_NotchType)))
        file.write('{0:<12d}        ! IPC_ControlMode	- Turn Individual Pitch Control (IPC) for fatigue load reductions (pitch contribution) {{0: off, 1: 1P reductions, 2: 1P+2P reductions}}\n'.format(int(controller.IPC_ControlMode)))
        file.write('{0:<12d}        ! VS_ControlMode	- Generator torque control mode in above rated conditions {{0: constant torque, 1: constant power, 2: TSR tracking PI control}}\n'.format(int(controller.VS_ControlMode)))
        file.write('{0:<12d}        ! PC_ControlMode    - Blade pitch control mode {{0: No pitch, fix to fine pitch, 1: active PI blade pitch control}}\n'.format(int(controller.PC_ControlMode)))
        file.write('{0:<12d}        ! Y_ControlMode		- Yaw control mode {{0: no yaw control, 1: yaw rate control, 2: yaw-by-IPC}}\n'.format(int(controller.Y_ControlMode)))
        file.write('{0:<12d}        ! SS_Mode           - Setpoint Smoother mode {{0: no setpoint smoothing, 1: introduce setpoint smoothing}}\n'.format(int(controller.SS_Mode)))
        file.write('{0:<12d}        ! WE_Mode           - Wind speed estimator mode {{0: One-second low pass filtered hub height wind speed, 1: Immersion and Invariance Estimator, 2: Extended Kalman Filter}}\n'.format(int(controller.WE_Mode)))
        file.write('{0:<12d}        ! PS_Mode           - Pitch saturation mode {{0: no pitch saturation, 1: implement pitch saturation}}\n'.format(int(controller.PS_Mode > 0)))
        file.write('{0:<12d}        ! SD_Mode           - Shutdown mode {{0: no shutdown procedure, 1: pitch to max pitch at shutdown}}\n'.format(int(controller.SD_Mode)))
        file.write('{0:<12d}        ! Fl_Mode           - Floating specific feedback mode {{0: no nacelle velocity feedback, 1: nacelle velocity feedback}}\n'.format(int(controller.Fl_Mode)))
        file.write('{0:<12d}        ! Flp_Mode          - Flap control mode {{0: no flap control, 1: steady state flap angle, 2: Proportional flap control}}\n'.format(int(controller.Flp_Mode)))
        file.write('\n')
        file.write('!------- FILTERS ----------------------------------------------------------\n') 
        file.write('{:<13.5f}       ! F_LPFCornerFreq	- Corner frequency (-3dB point) in the low-pass filters, [rad/s]\n'.format(turbine.bld_edgewise_freq * 1/4)) 
        file.write('{:<13.5f}       ! F_LPFDamping		- Damping coefficient [used only when F_FilterType = 2]\n'.format(controller.F_LPFDamping))
        file.write('{:<13.5f}       ! F_NotchCornerFreq	- Natural frequency of the notch filter, [rad/s]\n'.format(turbine.twr_freq))
        file.write('{:<10.5f}{:<9.5f} ! F_NotchBetaNumDen	- Two notch damping values (numerator and denominator, resp) - determines the width and depth of the notch, [-]\n'.format(0.0,0.25))
        file.write('{:<014.5f}      ! F_SSCornerFreq    - Corner frequency (-3dB point) in the first order low pass filter for the setpoint smoother, [rad/s].\n'.format(controller.ss_cornerfreq))
        file.write('{:<10.5f}{:<9.5f} ! F_FlCornerFreq    - Natural frequency and damping in the second order low pass filter of the tower-top fore-aft motion for floating feedback control [rad/s, -].\n'.format(turbine.ptfm_freq, 1.0))
        file.write('{:<10.5f}{:<9.5f} ! F_FlpCornerFreq   - Corner frequency and damping in the second order low pass filter of the blade root bending moment for flap control [rad/s, -].\n'.format(turbine.bld_flapwise_freq*1/3, 1.0))
        
        file.write('\n')
        file.write('!------- BLADE PITCH CONTROL ----------------------------------------------\n')
        file.write('{:<11d}         ! PC_GS_n			- Amount of gain-scheduling table entries\n'.format(len(controller.pitch_op_pc)))
        file.write('{}              ! PC_GS_angles	    - Gain-schedule table: pitch angles\n'.format(''.join('{:<4.6f}  '.format(controller.pitch_op_pc[i]) for i in range(len(controller.pitch_op_pc)))))            
        file.write('{}              ! PC_GS_KP		- Gain-schedule table: pitch controller kp gains\n'.format(''.join('{:<4.6f}  '.format(controller.pc_gain_schedule.Kp[i]) for i in range(len(controller.pc_gain_schedule.Kp)))))
        file.write('{}              ! PC_GS_KI		- Gain-schedule table: pitch controller ki gains\n'.format(''.join('{:<4.6f}  '.format(controller.pc_gain_schedule.Ki[i]) for i in range(len(controller.pc_gain_schedule.Ki)))))
        file.write('{}              ! PC_GS_KD			- Gain-schedule table: pitch controller kd gains\n'.format(''.join('{:<1.1f}  '.format(0.0) for i in range(len(controller.pc_gain_schedule.Ki)))))
        file.write('{}              ! PC_GS_TF			- Gain-schedule table: pitch controller tf gains (derivative filter)\n'.format(''.join('{:<1.1f}  '.format(0.0) for i in range(len(controller.pc_gain_schedule.Ki)))))
        file.write('{:<014.5f}      ! PC_MaxPit			- Maximum physical pitch limit, [rad].\n'.format(controller.max_pitch))
        file.write('{:<014.5f}      ! PC_MinPit			- Minimum physical pitch limit, [rad].\n'.format(controller.min_pitch))
        file.write('{:<014.5f}      ! PC_MaxRat			- Maximum pitch rate (in absolute value) in pitch controller, [rad/s].\n'.format(turbine.max_pitch_rate))
        file.write('{:<014.5f}      ! PC_MinRat			- Minimum pitch rate (in absolute value) in pitch controller, [rad/s].\n'.format(turbine.min_pitch_rate))
        file.write('{:<014.5f}      ! PC_RefSpd			- Desired (reference) HSS speed for pitch controller, [rad/s].\n'.format(turbine.rated_rotor_speed*turbine.Ng))
        file.write('{:<014.5f}      ! PC_FinePit		- Record 5: Below-rated pitch angle set-point, [rad]\n'.format(controller.min_pitch))
        file.write('{:<014.5f}      ! PC_Switch			- Angle above lowest minimum pitch angle for switch, [rad]\n'.format(1 * deg2rad))
        file.write('\n')
        file.write('!------- INDIVIDUAL PITCH CONTROL -----------------------------------------\n')
        file.write('{:<13.1f}       ! IPC_IntSat		- Integrator saturation (maximum signal amplitude contribution to pitch from IPC), [rad]\n'.format(0.0))
        file.write('{:<6.1f}{:<13.1f} ! IPC_KI			- Integral gain for the individual pitch controller: first parameter for 1P reductions, second for 2P reductions, [-]\n'.format(0.0,0.0))
        file.write('{:<6.1f}{:<13.1f} ! IPC_aziOffset		- Phase offset added to the azimuth angle for the individual pitch controller, [rad]. \n'.format(0.0,0.0))
        file.write('{:<13.1f}       ! IPC_CornerFreqAct - Corner frequency of the first-order actuators model, to induce a phase lag in the IPC signal {{0: Disable}}, [rad/s]\n'.format(0.0))
        file.write('\n')
        file.write('!------- VS TORQUE CONTROL ------------------------------------------------\n')
        file.write('{:<014.5f}      ! VS_GenEff			- Generator efficiency mechanical power -> electrical power, [should match the efficiency defined in the generator properties!], [%]\n'.format(turbine.GenEff))
        file.write('{:<014.5f}      ! VS_ArSatTq		- Above rated generator torque PI control saturation, [Nm]\n'.format(turbine.rated_torque))
        file.write('{:<014.5f}      ! VS_MaxRat			- Maximum torque rate (in absolute value) in torque controller, [Nm/s].\n'.format(turbine.max_torque_rate))
        file.write('{:<014.5f}      ! VS_MaxTq			- Maximum generator torque in Region 3 (HSS side), [Nm].\n'.format(turbine.max_torque))
        file.write('{:<014.5f}      ! VS_MinTq			- Minimum generator (HSS side), [Nm].\n'.format(0.0))
        file.write('{:<014.5f}      ! VS_MinOMSpd		- Optimal mode minimum speed, cut-in speed towards optimal mode gain path, [rad/s]\n'.format(controller.vs_minspd))
        file.write('{:<014.5f}      ! VS_Rgn2K			- Generator torque constant in Region 2 (HSS side), [N-m/(rad/s)^2]\n'.format(controller.vs_rgn2K))
        file.write('{:<014.5f}      ! VS_RtPwr			- Wind turbine rated power [W]\n'.format(turbine.rated_power))
        file.write('{:<014.5f}      ! VS_RtTq			- Rated torque, [Nm].\n'.format(turbine.rated_torque))
        file.write('{:<014.5f}      ! VS_RefSpd			- Rated generator speed [rad/s]\n'.format(controller.vs_refspd))
        file.write('{:<11d}         ! VS_n				- Number of generator PI torque controller gains\n'.format(1))
        file.write('{:<014.5f}      ! VS_KP				- Proportional gain for generator PI torque controller [1/(rad/s) Nm]. (Only used in the transitional 2.5 region if VS_ControlMode =/ 2)\n'.format(controller.vs_gain_schedule.Kp[-1]))
        file.write('{:<014.5f}      ! VS_KI				- Integral gain for generator PI torque controller [1/rad Nm]. (Only used in the transitional 2.5 region if VS_ControlMode =/ 2)\n'.format(controller.vs_gain_schedule.Ki[-1]))
        file.write('{:<13.2f}       ! VS_TSRopt			- Power-maximizing region 2 tip-speed-ratio [rad].\n'.format(turbine.TSR_operational))
        file.write('\n')
        file.write('!------- SETPOINT SMOOTHER ---------------------------------------------\n')
        file.write('{:<13.5f}       ! SS_VSGain         - Variable speed torque controller setpoint smoother gain, [-].\n'.format(controller.ss_vsgain))
        file.write('{:<13.5f}       ! SS_PCGain         - Collective pitch controller setpoint smoother gain, [-].\n'.format(controller.ss_pcgain))
        file.write('\n')
        file.write('!------- WIND SPEED ESTIMATOR ---------------------------------------------\n')
        file.write('{:<13.3f}       ! WE_BladeRadius	- Blade length (distance from hub center to blade tip), [m]\n'.format(turbine.rotor_radius))
        file.write('{:<11d}         ! WE_CP_n			- Amount of parameters in the Cp array\n'.format(1))
        file.write(          '{}    ! WE_CP - Parameters that define the parameterized CP(lambda) function\n'.format(''.join('{:<2.1f} '.format(0.0) for i in range(4))))
        file.write('{:<13.1f}		! WE_Gamma			- Adaption gain of the wind speed estimator algorithm [m/rad]\n'.format(0.0))
        file.write('{:<13.1f}       ! WE_GearboxRatio	- Gearbox ratio [>=1],  [-]\n'.format(turbine.Ng))
        file.write('{:<014.5f}      ! WE_Jtot			- Total drivetrain inertia, including blades, hub and casted generator inertia to LSS, [kg m^2]\n'.format(turbine.J))
        file.write('{:<13.3f}       ! WE_RhoAir			- Air density, [kg m^-3]\n'.format(turbine.rho))
        file.write(      '"{}"      ! PerfFileName      - File containing rotor performance tables (Cp,Ct,Cq)\n'.format(txt_filename))
        file.write('{:<7d} {:<10d}  ! PerfTableSize     - Size of rotor performance tables, first number refers to number of blade pitch angles, second number referse to number of tip-speed ratios\n'.format(len(turbine.Cp.pitch_initial_rad),len(turbine.Cp.TSR_initial)))
        file.write('{:<11d}         ! WE_FOPoles_N      - Number of first-order system poles used in EKF\n'.format(len(controller.A)))
        file.write('{}              ! WE_FOPoles_v      - Wind speeds corresponding to first-order system poles [m/s]\n'.format(''.join('{:<4.2f} '.format(controller.v[i]) for i in range(len(controller.v)))))
        file.write('{}              ! WE_FOPoles        - First order system poles\n'.format(''.join('{:<10.8f} '.format(controller.A[i]) for i in range(len(controller.A)))))
        file.write('\n')
        file.write('!------- YAW CONTROL ------------------------------------------------------\n')
        file.write('{:<13.1f}       ! Y_ErrThresh		- Yaw error threshold. Turbine begins to yaw when it passes this. [rad^2 s]\n'.format(0.0))
        file.write('{:<13.1f}       ! Y_IPC_IntSat		- Integrator saturation (maximum signal amplitude contribution to pitch from yaw-by-IPC), [rad]\n'.format(0.0))
        file.write('{:<11d}         ! Y_IPC_n			- Number of controller gains (yaw-by-IPC)\n'.format(1))
        file.write('{:<13.1f}       ! Y_IPC_KP			- Yaw-by-IPC proportional controller gain Kp\n'.format(0.0))
        file.write('{:<13.1f}       ! Y_IPC_KI			- Yaw-by-IPC integral controller gain Ki\n'.format(0.0))
        file.write('{:<13.1f}       ! Y_IPC_omegaLP		- Low-pass filter corner frequency for the Yaw-by-IPC controller to filtering the yaw alignment error, [rad/s].\n'.format(0.0))
        file.write('{:<13.1f}       ! Y_IPC_zetaLP		- Low-pass filter damping factor for the Yaw-by-IPC controller to filtering the yaw alignment error, [-].\n'.format(0.0))
        file.write('{:<13.1f}       ! Y_MErrSet			- Yaw alignment error, set point [rad]\n'.format(0.0))
        file.write('{:<13.1f}       ! Y_omegaLPFast		- Corner frequency fast low pass filter, 1.0 [Hz]\n'.format(0.0))
        file.write('{:<13.1f}       ! Y_omegaLPSlow		- Corner frequency slow low pass filter, 1/60 [Hz]\n'.format(0.0))
        file.write('{:<13.1f}       ! Y_Rate			- Yaw rate [rad/s]\n'.format(0.0))
        file.write('\n')
        file.write('!------- TOWER FORE-AFT DAMPING -------------------------------------------\n')
        file.write('{:<11d}         ! FA_KI				- Integral gain for the fore-aft tower damper controller, -1 = off / >0 = on [rad s/m] - !NJA - Make this a flag\n'.format(-1))
        file.write('{:<13.1f}       ! FA_HPF_CornerFreq	- Corner frequency (-3dB point) in the high-pass filter on the fore-aft acceleration signal [rad/s]\n'.format(0.0))
        file.write('{:<13.1f}       ! FA_IntSat			- Integrator saturation (maximum signal amplitude contribution to pitch from FA damper), [rad]\n'.format(0.0))
        file.write('\n')
        file.write('!------- MINIMUM PITCH SATURATION -------------------------------------------\n')
        file.write('{:<11d}         ! PS_BldPitchMin_N  - Number of values in minimum blade pitch lookup table (should equal number of values in PS_WindSpeeds and PS_BldPitchMin)\n'.format(len(controller.ps_min_bld_pitch)))
        file.write('{}              ! PS_WindSpeeds     - Wind speeds corresponding to minimum blade pitch angles [m/s]\n'.format(''.join('{:<4.2f} '.format(controller.v[i]) for i in range(len(controller.v)))))
        file.write('{}              ! PS_BldPitchMin    - Minimum blade pitch angles [rad]\n'.format(''.join('{:<10.8f} '.format(controller.ps_min_bld_pitch[i]) for i in range(len(controller.ps_min_bld_pitch)))))
        file.write('\n')
        file.write('!------- SHUTDOWN -----------------------------------------------------------\n')
        file.write('{:<014.5f}      ! SD_MaxPit         - Maximum blade pitch angle to initiate shutdown, [rad]\n'.format(controller.sd_maxpit))
        file.write('{:<014.5f}      ! SD_CornerFreq     - Cutoff Frequency for first order low-pass filter for blade pitch angle, [rad/s]\n'.format(controller.sd_cornerfreq))
        file.write('\n')
        file.write('!------- Floating -----------------------------------------------------------\n')
        file.write('{:<014.5f}      ! Fl_Kp             - Nacelle velocity proportional feedback gain [s]\n'.format(controller.Kp_float))
        file.write('\n')
        file.write('!------- FLAP ACTUATION -----------------------------------------------------\n')
        file.write('{:<014.5f}      ! Flp_Angle         - Initial or steady state flap angle [rad]\n'.format(controller.flp_angle))
        file.write('{:<014.8e}      ! Flp_Kp            - Blade root bending moment proportional gain for flap control [s]\n'.format(controller.Kp_flap[-1]))
        file.write('{:<014.8e}      ! Flp_Ki            - Flap displacement integral gain for flap control [s]\n'.format(controller.Ki_flap[-1]))
        file.write('{:<014.5f}      ! Flp_MaxPit        - Maximum (and minimum) flap pitch angle [rad]'.format(controller.flp_maxpit))
        file.close()

    def read_DISCON(self, DISCON_filename):
        '''
        Read the DISCON input file.

        Parameters:
        ----------
        DISCON_filename: string
            Name of DISCON input file to read
        
        Returns:
        --------
        DISCON_in: Dict
            Dictionary containing input parameters from DISCON_in, organized by parameter name
        '''
        
        DISCON_in = {}
        with open(DISCON_filename) as discon:
            for line in discon:

                # Skip whitespace and comment lines
                if (line[0] != '!') == (len(line.strip()) != 0):
                    
                    if (line.split()[1] != '!'):    # Array valued entries
                        array_length = line.split().index('!')
                        param = line.split()[array_length+1]
                        values = np.array( [float(x) for x in line.split()[:array_length]] )
                        DISCON_in[param] = values
                    else:                           # All other entries
                        param = line.split()[2]
                        value = line.split()[0]
                        # Remove printed quotations if string is in quotes
                        if (value[0] == '"') or (value[0] == "'"):
                            value = value[1:-1]
                        else:
                            value = float(value)
                            # Some checks for variables that are generally passed as lists
                            if param.lower() == 'vs_kp':
                                value = [value]
                            if param.lower() == 'vs_ki':
                                value = [value]
                            if param.lower() == 'flp_kp':
                                value = [value]
                            if param.lower() == 'flp_ki':
                                value = [value]
                        DISCON_in[param] = value

        return DISCON_in
    
    def write_rotor_performance(self,turbine,txt_filename='Cp_Ct_Cq.txt'):
        '''
        Write text file containing rotor performance data

        Parameters:
        ------------
            txt_filename: str, optional
                          Desired output filename to print rotor performance data. Default is Cp_Ct_Cq.txt
        '''
        print('Writing rotor performance text file: {}'.format(txt_filename))
        file = open(txt_filename,'w')
        # Headerlines
        file.write('# ----- Rotor performance tables for the {} wind turbine ----- \n'.format(turbine.TurbineName))
        file.write('# ------------ Written on {} using the ROSCO toolbox ------------ \n\n'.format(now.strftime('%b-%d-%y')))

        # Pitch angles, TSR, and wind speed
        file.write('# Pitch angle vector, {} entries - x axis (matrix columns) (deg)\n'.format(len(turbine.Cp.pitch_initial_rad)))
        for i in range(len(turbine.Cp.pitch_initial_rad)):
            file.write('{:0.4}   '.format(turbine.Cp.pitch_initial_rad[i] * rad2deg))
        file.write('\n# TSR vector, {} entries - y axis (matrix rows) (-)\n'.format(len(turbine.TSR_initial)))
        for i in range(len(turbine.TSR_initial)):
            file.write('{:0.4}    '.format(turbine.Cp.TSR_initial[i]))
        file.write('\n# Wind speed vector - z axis (m/s)\n')
        file.write('{:0.4}    '.format(turbine.v_rated))
        file.write('\n')
        
        # Cp
        file.write('\n# Power coefficient\n\n')
        for i in range(len(turbine.Cp.TSR_initial)):
            for j in range(len(turbine.Cp.pitch_initial_rad)):
                file.write('{0:.6f}   '.format(turbine.Cp_table[i,j]))
            file.write('\n')
        file.write('\n')
        
        # Ct
        file.write('\n#  Thrust coefficient\n\n')
        for i in range(len(turbine.Ct.TSR_initial)):
            for j in range(len(turbine.Ct.pitch_initial_rad)):
                file.write('{0:.6f}   '.format(turbine.Ct_table[i,j]))
            file.write('\n')
        file.write('\n')
        
        # Cq
        file.write('\n# Torque coefficient\n\n')
        for i in range(len(turbine.Cq.TSR_initial)):
            for j in range(len(turbine.Cq.pitch_initial_rad)):
                file.write('{0:.6f}   '.format(turbine.Cq_table[i,j]))
            file.write('\n')
        file.write('\n')
        file.close()

    def load_from_txt(self, txt_filename):
        '''
        Load rotor performance data from a *.txt file. 

        Parameters:
        -----------
            txt_filename: str
                            Filename of the text containing the Cp, Ct, and Cq data. This should be in the format printed by the write_rotorperformance function
        '''
        print('Loading rotor performace data from text file:', txt_filename)

        with open(txt_filename) as pfile:
            for line in pfile:
                # Read Blade Pitch Angles (degrees)
                if 'Pitch angle' in line:
                    pitch_initial = np.array([float(x) for x in pfile.readline().strip().split()])
                    pitch_initial_rad = pitch_initial * deg2rad             # degrees to rad            -- should this be conditional?

                # Read Tip Speed Ratios (rad)
                if 'TSR' in line:
                    TSR_initial = np.array([float(x) for x in pfile.readline().strip().split()])
                
                # Read Power Coefficients
                if 'Power' in line:
                    pfile.readline()
                    Cp = np.empty((len(TSR_initial),len(pitch_initial)))
                    for tsr_i in range(len(TSR_initial)):
                        Cp[tsr_i] = np.array([float(x) for x in pfile.readline().strip().split()])
                
                # Read Thrust Coefficients
                if 'Thrust' in line:
                    pfile.readline()
                    Ct = np.empty((len(TSR_initial),len(pitch_initial)))
                    for tsr_i in range(len(TSR_initial)):
                        Ct[tsr_i] = np.array([float(x) for x in pfile.readline().strip().split()])

                # Read Torque Coefficients
                if 'Torque' in line:
                    pfile.readline()
                    Cq = np.empty((len(TSR_initial),len(pitch_initial)))
                    for tsr_i in range(len(TSR_initial)):
                        Cq[tsr_i] = np.array([float(x) for x in pfile.readline().strip().split()])

            # return pitch_initial_rad TSR_initial Cp Ct Cq
            # Store necessary metrics for analysis and tuning
            # self.pitch_initial_rad = pitch_initial_rad
            # self.TSR_initial = TSR_initial
            # self.Cp_table = Cp
            # self.Ct_table = Ct 
            # self.Cq_table = Cq
            return pitch_initial_rad, TSR_initial, Cp, Ct, Cq

class DataProcessing():
    """
    Class DataProcessing used to process internal ROSCO toolbox data

    Methods:
    -----------
    DISCON_dict
    """
    def init(self):
        pass

    def DISCON_dict(self, turbine, controller, txt_filename=None):
        '''
        Convert the turbine and controller objects to a dictionary organized by the parameter names 
        that are defined in the DISCON.IN file.

        Parameters
        ----------
        turbine: obj
            Turbine object output from the turbine class
        controller: obj
            Controller object output from the controller class
        txt_filename: string, optional
            Name of rotor performance filename
        '''
        DISCON_dict = {}
        # ------- DEBUG -------
        DISCON_dict['LoggingLevel']	    = controller.LoggingLevel
        # ------- CONTROLLER FLAGS -------
        DISCON_dict['F_LPFType']	    = controller.F_LPFType
        DISCON_dict['F_NotchType']		= controller.F_NotchType
        DISCON_dict['IPC_ControlMode']	= controller.IPC_ControlMode
        DISCON_dict['VS_ControlMode']	= controller.VS_ControlMode
        DISCON_dict['PC_ControlMode']   = controller.PC_ControlMode
        DISCON_dict['Y_ControlMode']	= controller.Y_ControlMode
        DISCON_dict['SS_Mode']          = controller.SS_Mode
        DISCON_dict['WE_Mode']          = controller.WE_Mode
        DISCON_dict['PS_Mode']          = controller.PS_Mode
        DISCON_dict['SD_Mode']          = controller.SD_Mode
        DISCON_dict['Fl_Mode']          = controller.Fl_Mode
        DISCON_dict['Flp_Mode']         = controller.Flp_Mode
        # ------- FILTERS -------
        DISCON_dict['F_LPFCornerFreq']	    = turbine.bld_edgewise_freq * 1/4
        DISCON_dict['F_LPFDamping']		    = controller.F_LPFDamping
        DISCON_dict['F_NotchCornerFreq']    = turbine.twr_freq
        DISCON_dict['F_NotchBetaNumDen']    = [0.0, 0.25]
        DISCON_dict['F_SSCornerFreq']       = controller.ss_cornerfreq
        DISCON_dict['F_FlCornerFreq']       = [turbine.ptfm_freq, 1.0]
        DISCON_dict['F_FlpCornerFreq']      = [turbine.bld_flapwise_freq*1/3, 1.0]
        # ------- BLADE PITCH CONTROL -------
        DISCON_dict['PC_GS_n']			= len(controller.pitch_op_pc)
        DISCON_dict['PC_GS_angles']	    = controller.pitch_op_pc
        DISCON_dict['PC_GS_KP']		    = controller.pc_gain_schedule.Kp
        DISCON_dict['PC_GS_KI']		    = controller.pc_gain_schedule.Ki
        DISCON_dict['PC_GS_KD']			= [0.0 for i in range(len(controller.pc_gain_schedule.Ki))]
        DISCON_dict['PC_GS_TF']			= [0.0 for i in range(len(controller.pc_gain_schedule.Ki))]
        DISCON_dict['PC_MaxPit']		= controller.max_pitch
        DISCON_dict['PC_MinPit']		= controller.min_pitch
        DISCON_dict['PC_MaxRat']		= turbine.max_pitch_rate
        DISCON_dict['PC_MinRat']		= turbine.min_pitch_rate
        DISCON_dict['PC_RefSpd']		= turbine.rated_rotor_speed*turbine.Ng
        DISCON_dict['PC_FinePit']		= controller.min_pitch
        DISCON_dict['PC_Switch']		= 1 * deg2rad
        # ------- INDIVIDUAL PITCH CONTROL -------
        DISCON_dict['IPC_IntSat']		= 0.0
        DISCON_dict['IPC_KI']			= [0.0, 0.0]
        DISCON_dict['IPC_aziOffset']	= [0.0, 0.0]
        DISCON_dict['IPC_CornerFreqAct'] = 0.0
        # ------- VS TORQUE CONTROL -------
        DISCON_dict['VS_GenEff']		= turbine.GenEff
        DISCON_dict['VS_ArSatTq']		= turbine.rated_torque
        DISCON_dict['VS_MaxRat']		= turbine.max_torque_rate
        DISCON_dict['VS_MaxTq']			= turbine.max_torque
        DISCON_dict['VS_MinTq']			= 0.0
        DISCON_dict['VS_MinOMSpd']		= controller.vs_minspd
        DISCON_dict['VS_Rgn2K']			= controller.vs_rgn2K
        DISCON_dict['VS_RtPwr']			= turbine.rated_power
        DISCON_dict['VS_RtTq']			= turbine.rated_torque
        DISCON_dict['VS_RefSpd']		= controller.vs_refspd
        DISCON_dict['VS_n']				= 1
        DISCON_dict['VS_KP']			= [controller.vs_gain_schedule.Kp[-1]]
        DISCON_dict['VS_KI']			= [controller.vs_gain_schedule.Ki[-1]]
        DISCON_dict['VS_TSRopt']		= turbine.TSR_operational
        # ------- SETPOINT SMOOTHER -------
        DISCON_dict['SS_VSGain']         = controller.ss_vsgain
        DISCON_dict['SS_PCGain']         = controller.ss_pcgain
        # ------- WIND SPEED ESTIMATOR -------
        DISCON_dict['WE_BladeRadius']	= turbine.rotor_radius
        DISCON_dict['WE_CP_n']			= 1
        DISCON_dict['WE_CP']            = [0.0 for i in range(4)]
        DISCON_dict['WE_Gamma']			= 0.0
        DISCON_dict['WE_GearboxRatio']	= turbine.Ng
        DISCON_dict['WE_Jtot']			= turbine.J
        DISCON_dict['WE_RhoAir']		= turbine.rho
        DISCON_dict['PerfFileName']     = txt_filename
        DISCON_dict['PerfTableSize']    = [len(turbine.Cp.pitch_initial_rad),len(turbine.Cp.TSR_initial)]
        DISCON_dict['WE_FOPoles_N']     = len(controller.A)
        DISCON_dict['WE_FOPoles_v']     = controller.v
        DISCON_dict['WE_FOPoles']       = controller.A
        # ------- YAW CONTROL -------
        DISCON_dict['Y_ErrThresh']		= 0.0
        DISCON_dict['Y_IPC_IntSat']		= 0.0
        DISCON_dict['Y_IPC_n']			= 1
        DISCON_dict['Y_IPC_KP']			= 0.0
        DISCON_dict['Y_IPC_KI']			= 0.0
        DISCON_dict['Y_IPC_omegaLP']    = 0.0
        DISCON_dict['Y_IPC_zetaLP']		= 0.0
        DISCON_dict['Y_MErrSet']		= 0.0
        DISCON_dict['Y_omegaLPFast']	= 0.0
        DISCON_dict['Y_omegaLPSlow']	= 0.0
        DISCON_dict['Y_Rate']			= 0.0
        # ------- TOWER FORE-AFT DAMPING -------
        DISCON_dict['JA']                = -1
        DISCON_dict['FA_HPF_CornerFreq'] = 0.0
        DISCON_dict['FA_IntSat']		 = 0.0
        # ------- MINIMUM PITCH SATURATION -------
        DISCON_dict['PS_BldPitchMin_N'] = len(controller.ps_min_bld_pitch)
        DISCON_dict['PS_WindSpeeds']    = controller.v
        DISCON_dict['PS_BldPitchMin']   = controller.ps_min_bld_pitch
        # ------- SHUTDOWN -------
        DISCON_dict['SD_MaxPit']        = controller.sd_maxpit
        DISCON_dict['SD_CornerFreq']    = controller.sd_cornerfreq
        # ------- Floating -------
        DISCON_dict['Fl_Kp']            = controller.Kp_float
        # ------- FLAP ACTUATION -------
        DISCON_dict['Flp_Angle']        = controller.flp_angle
        DISCON_dict['Flp_Kp']           = [controller.Kp_flap[-1]]
        DISCON_dict['Flp_Ki']           = [controller.Ki_flap[-1]]
        DISCON_dict['Flp_MaxPit']       = controller.flp_maxpit

        return DISCON_dict
