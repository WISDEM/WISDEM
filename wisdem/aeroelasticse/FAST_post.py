from __future__ import print_function
from wisdem.aeroelasticse.Util.ReadFASTout import ReadFASToutFormat

def return_fname(fname):
    return fname

def return_timeseries(fname):
    data, meta = ReadFASToutFormat(fname, 2, Verbose=True)
    return data

