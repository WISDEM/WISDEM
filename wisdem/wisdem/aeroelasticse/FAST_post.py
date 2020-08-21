from __future__ import print_function
from wisdem.aeroelasticse.Util.ReadFASTout import ReadFASToutFormat
import numpy as np0

try:
    from ROSCO_toolbox.utilities import FAST_IO
    import ROSCO_toolbox
except:
    pass


def return_fname(fname):
    return fname

def return_timeseries(fname):
    data, meta = ReadFASToutFormat(fname, 2, Verbose=True)
    data['meta'] = meta
    return data

def FAST_IO_timeseries(fname):
    # interface to FAST_IO data load
    try:
        test = ROSCO_toolbox.__file__
    except:
        print('WARNING: ROSCO_toolbox required for wisdem.aeroelasticse.FAST_post.FAST_IO_timeseries')
    
    fast_io = FAST_IO()
    fast_data = fast_io.load_FAST_out(fname, verbose=True)[0]
    return fast_data
    

def return_stats(fname):
    data, meta = ReadFASToutFormat(fname, 2, Verbose=True)
    stats = {}
    for var in data.keys():
        stats[var] = {}
        stats[var]['mean']   = np.mean(data[var])
        stats[var]['min']    = min(data[var])
        stats[var]['max']    = max(data[var])
        stats[var]['std']    = np.std((data[var]))
        stats[var]['absmax'] = np.abs(max(data[var]))
    return stats