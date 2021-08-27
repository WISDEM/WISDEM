#!/usr/bin/env python3
import os

from wisdem import run_wisdem

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = mydir + os.sep + "nrel5mw_monopile.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options_monopile.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options_monopile.yaml"

wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

# print results from the analysis or optimization
z = 0.5 * (wt_opt["fixedse.z_full"][:-1] + wt_opt["fixedse.z_full"][1:])
print("zs =", wt_opt["fixedse.z_full"])
print("ds =", wt_opt["fixedse.d_full"])
print("ts =", wt_opt["fixedse.t_full"])
print("mass (kg) =", wt_opt["fixedse.monopile_mass"])
print("cg (m) =", wt_opt["fixedse.monopile_z_cg"])
print("d:t constraint =", wt_opt["fixedse.constr_d_to_t"])
print("taper ratio constraint =", wt_opt["fixedse.constr_taper"])
print("\nwind: ", wt_opt["fixedse.env1.Uref"], wt_opt["fixedse.env2.Uref"])
print("freq (Hz) =", wt_opt["fixedse.monopile.structural_frequencies"])
print("Fore-aft mode shapes =", wt_opt["fixedse.monopile.fore_aft_modes"])
print("Side-side mode shapes =", wt_opt["fixedse.monopile.side_side_modes"])
print("top_deflection1 (m) =", wt_opt["fixedse.monopile.top_deflection"])
print("Tower base forces1 (N) =", wt_opt["fixedse.monopile.mudline_F"])
print("Tower base moments1 (Nm) =", wt_opt["fixedse.monopile.mudline_M"])
print("stress1 =", wt_opt["fixedse.post.constr_stress"])
print("GL buckling =", wt_opt["fixedse.post.constr_global_buckling"])
print("Shell buckling =", wt_opt["fixedse.post.constr_shell_buckling"])
