"""
Example file showing how to perform a finite-difference (FD) step size study
using WISDEM and then visualize the results.
"""


import os

import numpy as np
from wisdem import run_wisdem

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = os.path.join(os.path.dirname(mydir), "05_tower_monopile", "nrel5mw_tower.yaml")
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"

wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)


# Load the derivative information
all_derivs = np.load("total_derivs.npy", allow_pickle=True)[()]

# Select a key of interest; an (of, wrt) pair
key = ("wt.towerse.post.constr_stress", "wt.wt_init.tower.diameter")

# Collect data to plot later
step_sizes = []
deriv_values = []

for idx in all_derivs:
    derivs = all_derivs[idx]
    step_sizes.append(derivs["step_size"])
    deriv_values.append(derivs[key][0][0])

print("Step size | Deriv")
for step_size, deriv in zip(step_sizes, deriv_values):
    print(f"{step_size:06}     {deriv}")

try:
    # Plot the step size study results
    import matplotlib.pyplot as plt

    plt.plot(step_sizes, deriv_values)
    plt.xlabel("FD step size")
    plt.ylabel("Derivative value of stress wrt diameter")
    plt.gca().set_xscale("log")

    plt.gca().invert_xaxis()

    plt.show()
except:
    pass
