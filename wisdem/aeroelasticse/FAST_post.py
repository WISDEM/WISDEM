from __future__ import print_function
from wisdem.aeroelasticse.Util.ReadFASTout import ReadFASToutFormat
import numpy as np

def return_fname(fname):
    return fname

def return_timeseries(fname):
    data, meta = ReadFASToutFormat(fname, 2, Verbose=True)
    return data

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