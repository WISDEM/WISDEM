import os
import numpy as np
import multiprocessing as mp
import time
from wisdem import run_wisdem

def runner(k, fgeom, fmodel, fanal, dover):
    iwt_opt, _, _ = run_wisdem(fgeom, fmodel, fanal, overridden_values=dover)
    return k, float(iwt_opt['financese.turbine_aep'][0])

def driver():
    start = time.time()

    # File management
    mydir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # get path to examples dir
    fname_wt_input = os.path.join(mydir, "02_reference_turbines", "IEA-15-240-RWT.yaml")
    fname_modeling_options = os.path.join(mydir, "02_reference_turbines", "modeling_options.yaml")
    fname_analysis_options = os.path.join(mydir, "02_reference_turbines", "analysis_options.yaml")

    # Initial run
    wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

    # Set parametric values
    blade_cones = np.deg2rad( np.arange(0, 5, 2) )
    shaft_tilts = np.deg2rad( np.arange(0, 5, 2) )
    Blades, Shafts = np.meshgrid(blade_cones, shaft_tilts)
    Blades = Blades.flatten()
    Shafts = Shafts.flatten()
    npts = Blades.size
    aep_output = np.zeros( npts ) #(blade_cones.size, shaft_tilts.size) )

    # Run parametric loop with overrides
    myargs = [] #np.c_[Blades, Shafts]
    for k in range(npts):
        overrides = {'hub.cone':Blades[k], 'nacelle.uptilt': Shafts[k]}
        myargs.append([k, fname_wt_input, fname_modeling_options, fname_analysis_options, overrides])
        
        # If want to do everything in serial, then:
        #iwt_opt, _, _ = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options, overridden_values=overrides)
        #aep_output[k] = float(iwt_opt['financese.turbine_aep'][0])

    # If you want to run this in parallel then:
    ncore = max(1, mp.cpu_count() - 2)
    pool = mp.Pool(processes=ncore)
    results = pool.starmap(runner, myargs)
    for iline in results:
        aep_output[iline[0]] = iline[1]
    
    print(aep_output)
    finish = time.time()
    print((finish-start)/60.0)

if __name__ == "__main__":
    driver()
    
