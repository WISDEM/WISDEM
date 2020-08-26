from __future__ import print_function
from ROSCO_toolbox.utilities import FAST_IO
import ROSCO_toolbox

def FAST_IO_timeseries(fname):
    # interface to FAST_IO data load
    try:
        test = ROSCO_toolbox.__file__
    except:
        print('WARNING: ROSCO_toolbox required for wisdem.aeroelasticse.FAST_post.FAST_IO_timeseries')
    
    fast_io = FAST_IO()
    fast_data = fast_io.load_FAST_out(fname, verbose=True)[0]
    return fast_data
