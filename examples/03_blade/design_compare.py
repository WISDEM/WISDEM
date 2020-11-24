import numpy as np
import matplotlib.pyplot as plt
from wisdem import run_wisdem
import os

show_plots = True

## File management
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
fname_wt_input1        = run_dir + 'blade.yaml'
fname_wt_input2        = run_dir + os.sep + 'outputs_aero' + os.sep + 'blade_out.yaml'
# fname_wt_input2        = run_dir + os.sep + 'outputs_struct' + os.sep + 'blade_out.yaml'
# fname_wt_input2        = run_dir + os.sep + 'outputs_aerostruct' + os.sep + 'blade_out.yaml'
fname_modeling_options = run_dir + 'modeling_options.yaml'
fname_analysis_options = run_dir + 'analysis_options_no_opt.yaml'

wt_opt1, modeling_options1, analysis_options1 = run_wisdem(fname_wt_input1, fname_modeling_options, fname_analysis_options)
wt_opt2, modeling_options2, analysis_options2 = run_wisdem(fname_wt_input2, fname_modeling_options, fname_analysis_options)

label1    = 'Initial'
label2    = 'Optimized'
fs        = 12
extension = '.png' # '.pdf'
color1    = 'tab:blue'
color2    = 'tab:red'
color3    = 'tab:green'


folder_output = analysis_options1['general']['folder_output']

# Twist
ftw, axtw = plt.subplots(1,1,figsize=(5.3, 4))
axtw.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['blade.outer_shape_bem.twist'] * 180. / np.pi,'--',color = color1, label=label1)
axtw.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['blade.outer_shape_bem.twist'] * 180. / np.pi,'-',color = color2, label=label2)
s_opt_twist = np.linspace(0., 1., 8)
twist_opt  = np.interp(s_opt_twist, wt_opt2['blade.outer_shape_bem.s'], wt_opt2['ccblade.theta'])
axtw.plot(s_opt_twist, twist_opt * 180. / np.pi,'o',color = color2, markersize=3, label='Optimized - Control Points')
axtw.plot(s_opt_twist, np.array(analysis_options2['optimization_variables']['blade']['aero_shape']['twist']['lower_bound']) * 180. / np.pi, ':o', color=color3,markersize=3, label = 'Bounds')
axtw.plot(s_opt_twist, np.array(analysis_options2['optimization_variables']['blade']['aero_shape']['twist']['upper_bound']) * 180. / np.pi, ':o', color=color3, markersize=3)
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
axc.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['blade.outer_shape_bem.chord'],'--', color = color1, label=label1)
axc.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['blade.outer_shape_bem.chord'],'-', color = color2, label=label2)
s_opt_chord = np.linspace(0., 1., 8)
chord_opt  = np.interp(s_opt_chord, wt_opt2['blade.outer_shape_bem.s'], wt_opt2['ccblade.chord'])
chord_init = np.interp(s_opt_chord, wt_opt2['blade.outer_shape_bem.s'], wt_opt1['blade.outer_shape_bem.chord'])
axc.plot(s_opt_chord, chord_opt,'o',color = color2, markersize=3, label='Optimized - Control Points')
axc.plot(s_opt_chord, np.array(analysis_options2['optimization_variables']['blade']['aero_shape']['chord']['min_gain']) * chord_init, ':o', color=color3,markersize=3, label = 'Bounds')
axc.plot(s_opt_chord, np.array(analysis_options2['optimization_variables']['blade']['aero_shape']['chord']['max_gain']) * chord_init, ':o', color=color3, markersize=3)
axc.legend(fontsize=fs)
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Chord [m]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'chord' + extension
fc.savefig(os.path.join(folder_output, fig_name))

# Edgewise stiffness
f, ax = plt.subplots(1,1,figsize=(5.3, 4))
ax.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['elastic.EIxx'],'--', color = color1, label=label1)
ax.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['elastic.EIxx'],'-',  color = color2, label=label2)
ax.legend(fontsize=fs)
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Edgewise Stiffness [Nm2]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'edge' + extension
f.savefig(os.path.join(folder_output, fig_name))

# Edgewise stiffness
f, ax = plt.subplots(1,1,figsize=(5.3, 4))
ax.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['elastic.GJ'],'--', color = color1, label=label1)
ax.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['elastic.GJ'],'-',  color = color2, label=label2)
ax.legend(fontsize=fs)
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Torsional Stiffness [Nm2]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'torsion' + extension
f.savefig(os.path.join(folder_output, fig_name))

# Flapwise stiffness
f, ax = plt.subplots(1,1,figsize=(5.3, 4))
ax.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['elastic.EIyy'],'--',color = color1, label=label1)
ax.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['elastic.EIyy'],'-',color = color2, label=label2)
ax.legend(fontsize=fs)
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Flapwise Stiffness [Nm2]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'flap' + extension
f.savefig(os.path.join(folder_output, fig_name))

# Mass
f, ax = plt.subplots(1,1,figsize=(5.3, 4))
ax.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['elastic.rhoA'],'--',color = color1, label=label1)
ax.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['elastic.rhoA'],'-',color = color2, label=label2)
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
axaoa.plot(wt_opt1['stall_check.s'], wt_opt1['stall_check.aoa_along_span'],'--',color = color1, label=label1)
axaoa.plot(wt_opt2['stall_check.s'], wt_opt2['stall_check.aoa_along_span'],'-',color = color2, label=label2)
axaoa.plot(wt_opt1['stall_check.s'], wt_opt1['stall_check.stall_angle_along_span'],':',color=color3, label='Stall')
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
axa.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['sse.powercurve.ax_induct_regII'],'--',color = color1, label=label1)
axa.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['sse.powercurve.ax_induct_regII'],'-',color = color2, label=label2)
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
axcl.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['sse.powercurve.cl_regII'],'--',color = color1, label=label1)
axcl.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['sse.powercurve.cl_regII'],'-',color = color2, label=label2)
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
axeff.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['sse.powercurve.cl_regII'] / wt_opt1['sse.powercurve.cd_regII'],'--',color = color1, label=label1)
axeff.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['sse.powercurve.cl_regII'] / wt_opt2['sse.powercurve.cd_regII'],'-',color = color2, label=label2)
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
spar_ss_name = analysis_options1['optimization_variables']['blade']['structure']['spar_cap_ss']['name']
spar_ps_name = analysis_options2['optimization_variables']['blade']['structure']['spar_cap_ps']['name']
for i in range(n_layers):
    if modeling_options1['blade']['layer_name'][i] == spar_ss_name:
        axsc.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['blade.internal_structure_2d_fem.layer_thickness'][i,:] * 1.e+3,'--', color = color1, label=label1)
        axsc.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['blade.internal_structure_2d_fem.layer_thickness'][i,:] * 1.e+3,'-', color = color2, label=label2)

        s_opt_sc = np.linspace(0., 1., 8)
        sc_opt  = np.interp(s_opt_sc, wt_opt2['blade.outer_shape_bem.s'], wt_opt2['blade.internal_structure_2d_fem.layer_thickness'][i,:] * 1.e+3)
        sc_init = np.interp(s_opt_sc, wt_opt2['blade.outer_shape_bem.s'], wt_opt1['blade.internal_structure_2d_fem.layer_thickness'][i,:] * 1.e+3)
        axsc.plot(s_opt_sc, sc_opt,'o',color = color2, markersize=3, label='Optimized - Control Points')
        axsc.plot(s_opt_sc, np.array(analysis_options2['optimization_variables']['blade']['structure']['spar_cap_ss']['min_gain']) * sc_init, ':o', color=color3,markersize=3, label = 'Bounds')
        axsc.plot(s_opt_sc, np.array(analysis_options2['optimization_variables']['blade']['structure']['spar_cap_ss']['max_gain']) * sc_init, ':o', color=color3, markersize=3)


axsc.legend(fontsize=fs)
plt.ylim([0., 200])
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Spar Caps Thickness [mm]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'sc_opt' + extension
fsc.savefig(os.path.join(folder_output, fig_name))

# Skins
f, ax = plt.subplots(1,1,figsize=(5.3, 4))
ax.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['blade.internal_structure_2d_fem.layer_thickness'][1,:] * 1.e+3,'--', color = color1, label=label1)
ax.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['blade.internal_structure_2d_fem.layer_thickness'][1,:] * 1.e+3,'-', color = color2, label=label2)
ax.legend(fontsize=fs)
#plt.ylim([0., 120])
plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
plt.ylabel('Outer Shell Skin Thickness [mm]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.15)
fig_name = 'skin_opt' + extension
f.savefig(os.path.join(folder_output, fig_name))




# Strains spar caps
feps, axeps = plt.subplots(1,1,figsize=(5.3, 4))
axeps.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['rlds.frame.strainU_spar'] * 1.e+6,'--',color = color1, label=label1)
axeps.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['rlds.frame.strainU_spar'] * 1.e+6,'-',color = color2, label=label2)
axeps.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['rlds.frame.strainL_spar'] * 1.e+6,'--',color = color1)
axeps.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['rlds.frame.strainL_spar'] * 1.e+6,'-',color = color2)
# axeps.plot(np.array([0.,1.]), np.array([3000.,3000.]), ':',color=color3, label='Constraints')
# axeps.plot(np.array([0.,1.]), np.array([-3000.,-3000.]), ':', color=color3,)
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
print('AEP ' + label1 + ' = ' + str(wt_opt1['sse.AEP']*1.e-6) + ' GWh')
print('AEP ' + label2 + '   = ' + str(wt_opt2['sse.AEP']*1.e-6) + ' GWh')
print('Cp ' + label1 + '  = ' + str(wt_opt1['sse.powercurve.Cp_aero']))
print('Cp ' + label2 + '    = ' + str(wt_opt2['sse.powercurve.Cp_aero']))
print('Design ' + label1 + ': LCOE in $/MWh: '  + str(wt_opt1['financese.lcoe']*1.e3))
print('Design ' + label2 + ': LCOE in $/MWh: '    + str(wt_opt2['financese.lcoe']*1.e3))


print('Design ' + label1 + ': blade mass in kg: ' + str(wt_opt1['elastic.precomp.blade_mass']))
print('Design ' + label1 + ': blade cost in $: '  + str(wt_opt1['elastic.precomp.total_blade_cost']))
print('Design ' + label2 + ': blade mass in kg: '   + str(wt_opt2['elastic.precomp.blade_mass']))
print('Design ' + label2 + ': blade cost in $: '    + str(wt_opt2['elastic.precomp.total_blade_cost']))

print('Design ' + label1 + ': AEP in GWh: ' + str(wt_opt1['sse.AEP']))
print('Design ' + label2 + ': AEP in GWh: ' + str(wt_opt2['sse.AEP']))

print('Design ' + label1 + ': tip deflection ratio: ' + str(wt_opt1['tcons.tip_deflection_ratio']))
print('Design ' + label2 + ': tip deflection ratio: ' + str(wt_opt2['tcons.tip_deflection_ratio']))

print('Design ' + label2 + ': flap and edge blade frequencies in Hz:')
print(wt_opt1['rlds.frame.flap_mode_freqs'])
print(wt_opt1['rlds.frame.edge_mode_freqs'])
print('Design ' + label1 + ': 3P frequency in Hz' + str(3.*wt_opt1['sse.powercurve.rated_Omega']/60.))
print('Design ' + label1 + ': 6P frequency in Hz' + str(6.*wt_opt1['sse.powercurve.rated_Omega']/60.))
print('Design ' + label2 + ': flap and edge blade frequencies in Hz:')
print(wt_opt2['rlds.frame.flap_mode_freqs'])
print(wt_opt2['rlds.frame.edge_mode_freqs'])
print('Design ' + label2 + ': 3P frequency in Hz' + str(3.*wt_opt2['sse.powercurve.rated_Omega']/60.))
print('Design ' + label2 + ': 6P frequency in Hz' + str(6.*wt_opt2['sse.powercurve.rated_Omega']/60.))

print('Design ' + label1 + ': forces at hub: ' + str(wt_opt1['rlds.aero_hub_loads.Fxyz_hub_aero']))
print('Design ' + label1 + ': moments at hub: ' + str(wt_opt1['rlds.aero_hub_loads.Mxyz_hub_aero']))
print('Design ' + label2 + ': forces at hub: ' + str(wt_opt2['rlds.aero_hub_loads.Fxyz_hub_aero']))
print('Design ' + label2 + ': moments at hub: ' + str(wt_opt2['rlds.aero_hub_loads.Mxyz_hub_aero']))


lcoe_data = np.array([[wt_opt1['financese.turbine_number'], wt_opt2['financese.turbine_number']],
[wt_opt1['financese.machine_rating'][0], wt_opt2['financese.machine_rating'][0]],
[wt_opt1['financese.tcc_per_kW'][0], wt_opt2['financese.tcc_per_kW'][0]],
[wt_opt1['financese.bos_per_kW'][0], wt_opt2['financese.bos_per_kW'][0]],
[wt_opt1['financese.opex_per_kW'][0], wt_opt2['financese.opex_per_kW'][0]],
[wt_opt1['financese.turbine_aep'][0]*1.e-6, wt_opt2['financese.turbine_aep'][0]*1.e-6],
[wt_opt1['financese.fixed_charge_rate'][0]*100., wt_opt2['financese.fixed_charge_rate'][0]*100.],
[wt_opt1['financese.lcoe'][0]*1000., wt_opt2['financese.lcoe'][0]*1000.]])

np.savetxt(os.path.join(folder_output, 'lcoe.dat'), lcoe_data)


