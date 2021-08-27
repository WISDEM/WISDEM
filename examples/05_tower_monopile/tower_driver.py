#!/usr/bin/env python3
import os

from wisdem import run_wisdem

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = mydir + os.sep + "nrel5mw_tower.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"

wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

# print results from the analysis or optimization
z = 0.5 * (wt_opt["towerse.z_full"][:-1] + wt_opt["towerse.z_full"][1:])
print("zs =", wt_opt["towerse.z_full"])
print("ds =", wt_opt["towerse.d_full"])
print("ts =", wt_opt["towerse.t_full"])
print("mass (kg) =", wt_opt["towerse.tower_mass"])
print("cg (m) =", wt_opt["towerse.tower_center_of_mass"])
print("d:t constraint =", wt_opt["towerse.constr_d_to_t"])
print("taper ratio constraint =", wt_opt["towerse.constr_taper"])
print("\nwind: ", wt_opt["towerse.env1.Uref"], wt_opt["towerse.env2.Uref"])
print("freq (Hz) =", wt_opt["towerse.tower.structural_frequencies"])
print("Fore-aft mode shapes =", wt_opt["towerse.tower.fore_aft_modes"])
print("Side-side mode shapes =", wt_opt["towerse.tower.side_side_modes"])
print("top_deflection1 (m) =", wt_opt["towerse.tower.top_deflection"])
print("Tower base forces1 (N) =", wt_opt["towerse.tower.turbine_F"])
print("Tower base moments1 (Nm) =", wt_opt["towerse.tower.turbine_M"])
print("stress1 =", wt_opt["towerse.post.constr_stress"])
print("GL buckling =", wt_opt["towerse.post.constr_global_buckling"])
print("Shell buckling =", wt_opt["towerse.post.constr_shell_buckling"])
