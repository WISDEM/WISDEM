import os
from wisdem.glue_code.runWISDEM import run_wisdem


## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input         = mydir + os.sep + 'nrel5mw_tower.yaml'
fname_modeling_options = mydir + os.sep + 'modeling_options.yaml'
fname_analysis_options = mydir + os.sep + 'analysis_options.yaml'

wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

# print results from the analysis or optimization
z = 0.5 * (wt_opt['towerse.z_full'][:-1] + wt_opt['towerse.z_full'][1:])
print('zs =', wt_opt['towerse.z_full'])
print('ds =', wt_opt['towerse.d_full'])
print('ts =', wt_opt['towerse.t_full'])
print('mass (kg) =', wt_opt['towerse.tower_mass'])
print('cg (m) =', wt_opt['towerse.tower_center_of_mass'])
print('d:t constraint =', wt_opt['towerse.constr_d_to_t'])
print('taper ratio constraint =', wt_opt['towerse.constr_taper'])
print('\nwind: ', wt_opt['towerse.wind1.Uref'])
print('freq (Hz) =', wt_opt['towerse.post1.structural_frequencies'])
print('Fore-aft mode shapes =', wt_opt['towerse.post1.fore_aft_modes'])
print('Side-side mode shapes =', wt_opt['towerse.post1.side_side_modes'])
print('top_deflection1 (m) =', wt_opt['towerse.post1.top_deflection'])
print('Tower base forces1 (N) =', wt_opt['towerse.tower1.base_F'])
print('Tower base moments1 (Nm) =', wt_opt['towerse.tower1.base_M'])
print('stress1 =', wt_opt['towerse.post1.stress'])
print('GL buckling =', wt_opt['towerse.post1.global_buckling'])
print('Shell buckling =', wt_opt['towerse.post1.shell_buckling'])
print('\nwind: ', wt_opt['towerse.wind2.Uref'])
print('freq (Hz) =', wt_opt['towerse.post2.structural_frequencies'])
print('Fore-aft mode shapes =', wt_opt['towerse.post2.fore_aft_modes'])
print('Side-side mode shapes =', wt_opt['towerse.post2.side_side_modes'])
print('top_deflection2 (m) =', wt_opt['towerse.post2.top_deflection'])
print('Tower base forces2 (N) =', wt_opt['towerse.tower2.base_F'])
print('Tower base moments2 (Nm) =', wt_opt['towerse.tower2.base_M'])
print('stress2 =', wt_opt['towerse.post2.stress'])
print('GL buckling =', wt_opt['towerse.post2.global_buckling'])
print('Shell buckling =', wt_opt['towerse.post2.shell_buckling'])

