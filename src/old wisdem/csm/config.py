# config.py
# 2012 01 24
""" config file for global variables """

from csmPPI import *

# Initialize ref and current YYYYMM
# Calling program can override these
#   e.g., ppi.ref_yr = 2003, etc.

ref_yr  = 2002
ref_mon =    9
curr_yr = 2009
curr_mon =  13

ppi = PPI(ref_yr,ref_mon,curr_yr,curr_mon)

# print 'config.py created ppi'
