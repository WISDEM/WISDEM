"""
Parsing of FAST output files, binary or text.
Binary read based on Matlab function ReadFASTbinary.m by Bonnie Jonkman.
Author: Evan Gaertner, National Renewable Energy Laboratory

Inputs:
    FileName      - string: contains file name to open
    OutFileFmt    - int: (optional) 1=textfile, 2=binary

Outputs:
    data          - dict: FAST output time series, output channel names as dict keys
    meta          - dict: additional meta data from output file, keys include: 'units', 'DescStr', 'FileID'

"""
import numpy as np
import struct
import os

def ReadFASToutFormat(FileName, OutFileFmt=0, Verbose=False):
    
    if OutFileFmt == 2:
        path,fname = os.path.split(FileName)
        FileName = os.path.join(path, '.'.join(fname.split('.')[:-1])+'.outb')
        Channels, ChanName, ChanUnit, FileID, DescStr = ReadFASTbinary(FileName)
    elif OutFileFmt == 1: 
        path,fname = os.path.split(FileName)
        FileName = os.path.join(path, '.'.join(fname.split('.')[:-1])+'.out')
        Channels, ChanName, ChanUnit, FileID, DescStr = ReadFASTtext(FileName)
    else:
        if Verbose:
            print('Attempting to read FAST output file: %s, format not specified'%FileName)
        error = False
        try:
            if Verbose:
                print('Attempting binary read')
            path,fname = os.path.split(FileName)
            FileName = os.path.join(path, '.'.join(fname.split('.')[:-1])+'.outb')
            Channels, ChanName, ChanUnit, FileID, DescStr = ReadFASTbinary(FileName)
            if Verbose:
                print('Success')
            error = False
        except:
            if Verbose:
                print('Failed')
            error = True
        if error:
            try:
                if Verbose:
                    print('Attempting text read')
                path,fname = os.path.split(FileName)
                FileName = os.path.join(path, '.'.join(fname.split('.')[:-1])+'.out')
                Channels, ChanName, ChanUnit, FileID, DescStr = ReadFASTtext(FileName)
                if Verbose:
                    print('Success')
                error = False
            except:
                if Verbose:
                    print('Failed')
                error = True
        if error:
            raise NameError('Unable read FAST output file: %s'%FileName)

    data = {}
    meta = {}
    meta['units'] = {}
    for i, (chan, unit) in enumerate(zip(ChanName, ChanUnit)):
        data[chan] = Channels[:,i]
        meta['units'][chan] = unit
    meta['FileID'] = FileID
    meta['DescStr'] = DescStr

    return data, meta

def ReadFASTbinary(FileName):
    LenName = 10    # number of characters per channel name
    LenUnit = 10    # number of characters per unit name

    #----------------------------        
    # load file binary data
    #----------------------------
    f = open(FileName, 'rb')
    data = f.read()
    f.close()
    i = 0           # data position pointer for binary unpacking

    #----------------------------        
    # get the header information
    #----------------------------

    FileID = struct.unpack('h',data[i:i+2])[0]              # FAST output file format, INT(2)
    i+=2
    NumOutChans = struct.unpack('i',data[i:i+4])[0]         # The number of output channels, INT(4)
    i+=4
    NT = struct.unpack('i',data[i:i+4])[0]                  # The number of time steps, INT(4)
    i+=4

    if FileID == 1: # with time
        TimeScl = struct.unpack('d',data[i:i+8])[0]         # The time slopes for scaling, REAL(8)
        i+=8
        TimeOff = struct.unpack('d',data[i:i+8])[0]         # The time offsets for scaling, REAL(8)
        i+=8
    else: # without time
        TimeOut1 = struct.unpack('d',data[i:i+8])[0]        # The first time in the time series, REAL(8)
        i+=8
        TimeIncr = struct.unpack('d',data[i:i+8])[0]        # The time increment, REAL(8)
        i+=8
    
    ColScl = [None]*NumOutChans
    for idx in range(0,NumOutChans):
        ColScl[idx] = struct.unpack('f',data[i:i+4])[0]     # The channel slopes for scaling, REAL(4)
        i+=4

    ColOff = [None]*NumOutChans
    for idx in range(0,NumOutChans):
        ColOff[idx] = struct.unpack('f',data[i:i+4])[0]     # The channel offsets for scaling, REAL(4)
        i+=4

    LenDesc = struct.unpack('i',data[i:i+4])[0]             # The number of characters in the description string, INT(4)
    i+=4
    try:
        DescStr = ''.join(struct.unpack(str('c'*LenDesc),data[i:i+LenDesc]))     # DescStr converted to ASCII
    except:
        DescStr = data[i:i+LenDesc].decode("utf-8").strip()
    i+=LenDesc

    ChanName = [None]*(NumOutChans+1)
    for idx in range(0,NumOutChans+1):
        try:
            ChanName[idx] = ''.join(''.join(struct.unpack('c'*LenName,data[i:i+LenName]))).strip()  # variable channel names
        except:
            ChanName[idx] = data[i:i+LenName].decode("utf-8").strip()
        i+=LenName

    ChanUnit = [None]*(NumOutChans+1)
    for idx in range(0,NumOutChans+1):
        try:
            ChanUnit[idx] = ''.join(''.join(struct.unpack('c'*LenUnit,data[i:i+LenUnit]))).strip()  # variable units
        except:
            ChanUnit[idx] = data[i:i+LenUnit].decode("utf-8").strip()
        i+=LenUnit

    #-------------------------        
    # get the channel time series
    #-------------------------
    nPts = NT*NumOutChans                                       # number of data points in the file   

    if FileID == 1:
        PackedTime = struct.unpack('i'*NT,data[i:i+4*NT])[0]    # read the time data
        i+=4*NT
    
    PackedData = [None]*nPts
    for idx in range(0,nPts):
        PackedData[idx] = struct.unpack('h',data[i:i+2])[0]     # read the channel data
        i+=2

    #-------------------------
    # Scale the packed binary to real data
    #-------------------------
    Channels = np.zeros((NT,NumOutChans+1))                     # output channels (including time in column 1)
    for it in range(0,NT):
        data_slice = PackedData[NumOutChans*(it):NumOutChans*(it+1)]
        for idx, (datai, ColOffi, ColScli) in enumerate(zip(data_slice, ColOff, ColScl)):
            Channels[it,idx+1] = (datai - ColOffi) / ColScli

    if FileID == 1:
        for idx, (PackedTimei, TimeOffi, TimeScli) in enumerate(zip(PackedTime, TimeOff, TimeScl)):
            Channels[idx,0] = (PackedTimei - TimeOffi) / TimeScli
    else:
        for idx in range(0,NT):
            Channels[idx,0] = TimeOut1 + TimeIncr*idx

    return Channels, ChanName, ChanUnit, FileID, DescStr

def ReadFASTtext(FileName):

    f = open(FileName, 'r')

    DescStr = []
    ln = f.readline()
    i = 0
    while ln != '':
        if 'Time' in ln:
            break
        DescStr.append(ln.strip())
        ln = f.readline()
        
    DescStr = ' '.join(DescStr).strip()
    ChanName = ln.split()
    ChanUnit = f.readline().split()
    Channels = np.loadtxt(f)
    f.close()

    return Channels, ChanName, ChanUnit, None, DescStr


if __name__ == "__main__":

    FileName = '0.outb'
    # FileName = 'temp/OpenFAST/09.out'
    
    data, meta = ReadFASToutFormat(FileName, OutFileFmt=2)

    import matplotlib.pyplot as plt
    plt.figure()
    xvar = 'Time'
    yvar = 'GenPwr'
    plt.plot(data[xvar], data['GenPwr'])
    plt.xlabel('%s, %s'%(xvar, meta['units'][xvar]))
    plt.ylabel('%s, %s'%(yvar, meta['units'][yvar]))
    plt.show()
