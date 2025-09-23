import os
import numpy as np
import multiprocessing as mp
import time
from wisdem import run_wisdem

parallel_flag = True

def parallel_runner(k, fgeom, fmodel, fanal, dover):
    iwt_opt, _, _ = run_wisdem(fgeom, fmodel, fanal, overridden_values=dover)
    return k, float(iwt_opt['financese.turbine_aep'][0])

def driver():
    start = time.time()

    # File management
    mydir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # get path to examples dir
    fname_wt_input = os.path.join(mydir, "02_reference_turbines", "IEA-15-240-RWT.yaml")
    fname_modeling_options = os.path.join(mydir, "02_reference_turbines", "modeling_options_iea15.yaml")
    fname_analysis_options = os.path.join(mydir, "02_reference_turbines", "analysis_options.yaml")

    # Initial run
    wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options,
                                                            fname_analysis_options)

    # Set parametric values
    blade_cones = np.deg2rad( np.arange(0, 5, 2) )
    shaft_tilts = np.deg2rad( np.arange(0, 5, 2) )

    # Set to grid of points and then sequence of run values
    Blades, Shafts = np.meshgrid(blade_cones, shaft_tilts)
    Blades = Blades.flatten()
    Shafts = Shafts.flatten()
    npts = Blades.size

    # Run parametric loop with overrides
    aep_output = np.zeros( npts ) # Initialize output container
    myargs = [] # Container for parallel run arguments
    for k in range(npts):
        overrides = {'hub.cone':Blades[k], 'drivetrain.uptilt': Shafts[k]}
        myargs.append([k, fname_wt_input, fname_modeling_options, fname_analysis_options, overrides])

        # Run cases in serial this way
        if not parallel_flag:
            iwt_opt, _, _ = run_wisdem(fname_wt_input, fname_modeling_options,
                                       fname_analysis_options, overridden_values=overrides)
            aep_output[k] = float(iwt_opt['financese.turbine_aep'][0])

    # Run cases in parallel this way
    if parallel_flag:
        ncore = max(1, mp.cpu_count() - 2)
        pool = mp.Pool(processes=ncore)
        results = pool.starmap(parallel_runner, myargs)
        for k in results:
            aep_output[ k[0] ] = k[1]
    
    print(aep_output)
    finish = time.time()
    print((finish-start)/60.0)

if __name__ == "__main__":
    driver()
    
