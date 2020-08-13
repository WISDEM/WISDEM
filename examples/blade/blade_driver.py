from wisdem.glue_code.runWISDEM import run_wisdem
import matplotlib.pyplot as plt
import numpy as np
import os

show_plots = False

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input         = mydir + os.sep + 'blade.yaml'
fname_modeling_options = mydir + os.sep + 'modeling_options.yaml'
fname_analysis_options = mydir + os.sep + 'analysis_options_blade.yaml'
fname_analysis_no_opt  = mydir + os.sep + 'analysis_options_blade_no_opt.yaml'

wt_opt1, analysis_options1, opt_options1 = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

# Printing and plotting results
print('AEP in GWh = ' + str(wt_opt1['sse.AEP']*1.e-6))
print('Cp = ' + str(wt_opt1['sse.powercurve.Cp_regII']))
print('Blade mass in kg = ' + str(wt_opt1['elastic.precomp.blade_mass']))
print('Nat frequencies blades in Hz = ' + str(wt_opt1['rlds.frame.flap_mode_freqs']))
print('Nat frequencies blades edge in Hz = ' + str(wt_opt1['rlds.frame.edge_mode_freqs']))
print('Rated omega in rpm = ' + str(wt_opt1['sse.powercurve.rated_Omega']))
print('1P in Hz = ' + str(wt_opt1['sse.powercurve.rated_Omega']/60.))
print('3P in Hz = ' + str(wt_opt1['sse.powercurve.rated_Omega']/60.*3.))
print('6P in Hz = ' + str(wt_opt1['sse.powercurve.rated_Omega']/60.*6.))
print('Tip tower clearance in m     = ' + str(wt_opt1['tcons.blade_tip_tower_clearance']))
print('Tip deflection constraint    = ' + str(wt_opt1['tcons.tip_deflection_ratio']))

folder_output = opt_options1['general']['folder_output']
fname_wt_output = os.path.join(folder_output, opt_options1['general']['fname_output']+'.yaml')
wt_opt2, analysis_options2, opt_options2 = run_wisdem(fname_wt_output, fname_modeling_options, fname_analysis_no_opt)

label1 = 'Orig'
label2 = 'Optimized'
fs=12
extension = '.png' # '.pdf'

# Twist
ftw, axtw = plt.subplots(1,1,figsize=(5.3, 4))
axtw.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['blade.outer_shape_bem.twist'] * 180. / np.pi,'--',color='tab:red', label=label1)
axtw.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['blade.outer_shape_bem.twist'] * 180. / np.pi,'-',color='tab:blue', label=label2)
axtw.legend(fontsize=fs)
axtw.set_ylim([-5, 20])
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Twist [deg]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'twist_opt' + extension
ftw.savefig(os.path.join(folder_output, fig_name))

# Chord
fc, axc = plt.subplots(1,1,figsize=(5.3, 4))
axc.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['blade.outer_shape_bem.chord'],'--', color='tab:red', label=label1)
axc.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['blade.outer_shape_bem.chord'],'-', color='tab:blue', label=label2)
axc.legend(fontsize=fs)
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Chord [m]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'chord' + extension
fc.savefig(os.path.join(folder_output, fig_name))

# rthick
fc, axc = plt.subplots(1,1,figsize=(5.3, 4))
axc.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['blade.interp_airfoils.r_thick_interp']*100.,'--', color='tab:red', label=label1)
axc.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['blade.interp_airfoils.r_thick_interp']*100.,'-', color='tab:blue', label=label2)
axc.legend(fontsize=fs)
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Real Thickness [%]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'rthick' + extension
fc.savefig(os.path.join(folder_output, fig_name))

# Edgewise stiffness
f, ax = plt.subplots(1,1,figsize=(5.3, 4))
ax.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['elastic.EIxx'],'--', color='tab:red', label=label1)
ax.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['elastic.EIxx'],'-',  color='tab:blue', label=label2)
ax.legend(fontsize=fs)
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Edgewise Stiffness [Nm2]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'edge' + extension
f.savefig(os.path.join(folder_output, fig_name))

# Stiffness
f, ax = plt.subplots(1,1,figsize=(5.3, 4))
ax.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['elastic.EIyy'],'--',color='tab:red', label=label1)
ax.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['elastic.EIyy'],'-',color='tab:blue', label=label2)
ax.legend(fontsize=fs)
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Flapwise Stiffness [Nm2]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'flap' + extension
f.savefig(os.path.join(folder_output, fig_name))

# Torsional stiffness
f, ax = plt.subplots(1,1,figsize=(5.3, 4))
ax.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['elastic.GJ'],'--',color='tab:red', label=label1)
ax.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['elastic.GJ'],'-',color='tab:blue', label=label2)
ax.legend(fontsize=fs)
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Torsional Stiffness [Nm2]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'gj' + extension
f.savefig(os.path.join(folder_output, fig_name))

# Mass
f, ax = plt.subplots(1,1,figsize=(5.3, 4))
ax.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['elastic.rhoA'],'--',color='tab:red', label=label1)
ax.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['elastic.rhoA'],'-',color='tab:blue', label=label2)
ax.legend(fontsize=fs)
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Unit Mass [kg/m]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'mass' + extension
f.savefig(os.path.join(folder_output, fig_name))

# Angle of attack and stall angle
faoa, axaoa = plt.subplots(1,1,figsize=(5.3, 4))
axaoa.plot(wt_opt1['stall_check.s'], wt_opt1['stall_check.aoa_along_span'],'--',color='tab:red', label=label1)
axaoa.plot(wt_opt2['stall_check.s'], wt_opt2['stall_check.aoa_along_span'],'-',color='tab:blue', label=label2)
axaoa.plot(wt_opt1['stall_check.s'], wt_opt1['stall_check.stall_angle_along_span'],':',color='tab:green', label='Stall')
axaoa.legend(fontsize=fs)
axaoa.set_ylim([0, 20])
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Angle of Attack [deg]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'aoa' + extension
faoa.savefig(os.path.join(folder_output, fig_name))

# Induction
fa, axa = plt.subplots(1,1,figsize=(5.3, 4))
axa.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['sse.powercurve.ax_induct_regII'],'--',color='tab:red', label=label1)
axa.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['sse.powercurve.ax_induct_regII'],'-',color='tab:blue', label=label2)
axa.legend(fontsize=fs)
axa.set_ylim([0, 0.5])
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Axial Induction [-]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'induction' + extension
fa.savefig(os.path.join(folder_output, fig_name))

# Lift coefficient
fcl, axcl = plt.subplots(1,1,figsize=(5.3, 4))
axcl.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['sse.powercurve.cl_regII'],'--',color='tab:red', label=label1)
axcl.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['sse.powercurve.cl_regII'],'-',color='tab:blue', label=label2)
axcl.legend(fontsize=fs)
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Lift Coefficient [-]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'lift_coeff' + extension
fcl.savefig(os.path.join(folder_output, fig_name))

# Airfoil efficiency
feff, axeff = plt.subplots(1,1,figsize=(5.3, 4))
axeff.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['sse.powercurve.cl_regII'] / wt_opt1['sse.powercurve.cd_regII'],'--',color='tab:red', label=label1)
axeff.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['sse.powercurve.cl_regII'] / wt_opt2['sse.powercurve.cd_regII'],'-',color='tab:blue', label=label2)
axeff.legend(fontsize=fs)
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Airfoil Efficiency [-]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'af_efficiency' + extension
feff.savefig(os.path.join(folder_output, fig_name))

# Spar caps
fsc, axsc = plt.subplots(1,1,figsize=(5.3, 4))
n_layers = len(wt_opt1['blade.internal_structure_2d_fem.layer_thickness'][:,0])
spar_ss_name = opt_options1['optimization_variables']['blade']['structure']['spar_cap_ss']['name']
spar_ps_name = opt_options2['optimization_variables']['blade']['structure']['spar_cap_ps']['name']
for i in range(n_layers):
    if analysis_options1['blade']['layer_name'][i] == spar_ss_name:
        axsc.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['blade.internal_structure_2d_fem.layer_thickness'][i,:] * 1.e+3,'--', color='tab:red', label=label1)
        axsc.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['blade.internal_structure_2d_fem.layer_thickness'][i,:] * 1.e+3,'-', color='tab:blue', label=label2)
axsc.legend(fontsize=fs)
plt.ylim([0., 120])
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Spar caps Thickness [mm]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'sc_opt' + extension
fsc.savefig(os.path.join(folder_output, fig_name))

# Strains spar caps
feps, axeps = plt.subplots(1,1,figsize=(5.3, 4))
axeps.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['rlds.frame.strainU_spar'] * 1.e+6,'--',color='tab:red', label=label1)
axeps.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['rlds.frame.strainU_spar'] * 1.e+6,'-',color='tab:blue', label=label2)
axeps.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['rlds.frame.strainL_spar'] * 1.e+6,'--',color='tab:red')
axeps.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['rlds.frame.strainL_spar'] * 1.e+6,'-',color='tab:blue')
# axeps.plot(np.array([0.,1.]), np.array([3000.,3000.]), ':',color='tab:green', label='Constraints')
# axeps.plot(np.array([0.,1.]), np.array([-3000.,-3000.]), ':', color='tab:green',)
plt.ylim([-5e+3, 5e+3])
axeps.legend(fontsize=fs)
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Spar Caps Strains [mu eps]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.2)
fig_name = 'strains_opt' + extension
feps.savefig(os.path.join(folder_output, fig_name))

if show_plots:
    plt.show()


# Printing and plotting results
print('AEP Initial = ' + str(wt_opt1['sse.AEP']*1.e-6) + ' GWh')
print('AEP Final   = ' + str(wt_opt2['sse.AEP']*1.e-6) + ' GWh')
print('Cp initial  = ' + str(wt_opt1['sse.powercurve.Cp_aero']))
print('Cp final    = ' + str(wt_opt2['sse.powercurve.Cp_aero']))
print('Design initial: LCOE in $/MWh: '  + str(wt_opt1['financese.lcoe']*1.e3))
print('Design final: LCOE in $/MWh: '    + str(wt_opt2['financese.lcoe']*1.e3))


print('Design initial: blade mass in kg: ' + str(wt_opt1['elastic.precomp.blade_mass']))
print('Design initial: blade cost in $: '  + str(wt_opt1['elastic.precomp.total_blade_cost']))
print('Design final: blade mass in kg: '   + str(wt_opt2['elastic.precomp.blade_mass']))
print('Design final: blade cost in $: '    + str(wt_opt2['elastic.precomp.total_blade_cost']))

print('Design initial: AEP in GWh: ' + str(wt_opt1['sse.AEP']))
print('Design final: AEP in GWh: ' + str(wt_opt2['sse.AEP']))

print('Design initial: tip deflection ratio: ' + str(wt_opt1['tcons.tip_deflection_ratio']))
print('Design final: tip deflection ratio: ' + str(wt_opt2['tcons.tip_deflection_ratio']))

print('Design initial: blade frequencies in Hz:')
print('Nat frequencies blades flap in Hz = ' + str(wt_opt1['rlds.frame.flap_mode_freqs']))
print('Nat frequencies blades edge in Hz = ' + str(wt_opt1['rlds.frame.edge_mode_freqs']))
print('Design initial: 3P frequency in Hz' + str(3.*wt_opt1['sse.powercurve.rated_Omega']/60.))
print('Design initial: 6P frequency in Hz' + str(6.*wt_opt1['sse.powercurve.rated_Omega']/60.))
print('Design final: blade frequencies in Hz:')
print('Nat frequencies blades flap in Hz = ' + str(wt_opt2['rlds.frame.flap_mode_freqs']))
print('Nat frequencies blades edge in Hz = ' + str(wt_opt2['rlds.frame.edge_mode_freqs']))
print('Design final: 3P frequency in Hz' + str(3.*wt_opt2['sse.powercurve.rated_Omega']/60.))
print('Design final: 6P frequency in Hz' + str(6.*wt_opt2['sse.powercurve.rated_Omega']/60.))

print('Design initial: forces at hub: ' + str(wt_opt1['rlds.aero_hub_loads.Fxyz_hub_aero']))
print('Design initial: moments at hub: ' + str(wt_opt1['rlds.aero_hub_loads.Mxyz_hub_aero']))
print('Design final: forces at hub: ' + str(wt_opt2['rlds.aero_hub_loads.Fxyz_hub_aero']))
print('Design final: moments at hub: ' + str(wt_opt2['rlds.aero_hub_loads.Mxyz_hub_aero']))

