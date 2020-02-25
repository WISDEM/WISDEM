from wisdem.assemblies.main import run_wisdem

import time

start_time = time.time()

## File management
fname_wt_input         = "/Users/rfeil/work/2_OpenFAST/00_analysis/00_wisdem_analysis/BAR2011n_noRe.yaml"
fname_analysis_options = "/Users/rfeil/work/2_OpenFAST/00_analysis/00_wisdem_analysis/analysis_options.yaml"
fname_opt_options      = "/Users/rfeil/work/2_OpenFAST/00_analysis/00_wisdem_analysis/optimization_options.yaml"
fname_wt_output        = "/Users/rfeil/work/2_OpenFAST/00_analysis/00_wisdem_analysis/BAR2011n_noRe_out.yaml"
folder_output          = '/Users/rfeil/work/2_OpenFAST/00_analysis/00_wisdem_analysis/it_0/'

wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_analysis_options, fname_opt_options, fname_wt_output, folder_output)

# Printing and plotting results
print('AEP in GWh = ' + str(float(wt_opt['sse.AEP']*1.e-6)))
print('Blade mass in kg = ' + str(float(wt_opt['elastic.precomp.blade_mass'])))
print('Natural blade frequencies in Hz = ' + str(wt_opt['elastic.curvefem.freq']))
print('1P in Hz = ' + str(float(wt_opt['sse.powercurve.rated_Omega']/60.)))
print('3P in Hz = ' + str(float(wt_opt['sse.powercurve.rated_Omega']/60.*3.)))
print('6P in Hz = ' + str(float(wt_opt['sse.powercurve.rated_Omega']/60.*6.)))
print('Tip tower clearance in m     = ' + str(float(wt_opt['tcons.blade_tip_tower_clearance'])))
print('Tip deflection constraint    = ' + str(float(wt_opt['tcons.tip_deflection_ratio'])))

end_time = time.time() - start_time
print('Elapsed computational time: ' + str(float(end_time)))




# import matplotlib.pyplot as plt
# feps, axeps = plt.subplots(1,1,figsize=(5.3, 4))
# axeps.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.pbeam.strainU_spar'], label='Spar ss')
# axeps.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.pbeam.strainL_spar'], label='Spar ps')
# axeps.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.pbeam.strainU_te'], label='TE ss')
# axeps.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.pbeam.strainL_te'], label='TE ps')
# axeps.legend(fontsize=fs)
# axeps.set_ylim([-5e-3, 5e-3])
# plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
# plt.ylabel('Strains [-]', fontsize=fs+2, fontweight='bold')
# plt.xticks(fontsize=fs)
# plt.yticks(fontsize=fs)
# plt.grid(color=[0.8,0.8,0.8], linestyle='--')
# plt.subplots_adjust(bottom = 0.15, left = 0.18)
# fig_name = 'eps.png'
# feps.savefig(folder_output + fig_name)
#
#
# # Angle of attack and stall angle
# faoa, axaoa = plt.subplots(1,1,figsize=(5.3, 4))
# axaoa.plot(wt_opt['sse.stall_check.s'], wt_opt['sse.stall_check.aoa_along_span'])
# axaoa.plot(wt_opt['sse.stall_check.s'], wt_opt['sse.stall_check.stall_angle_along_span'])
# axaoa.legend(fontsize=fs)
# axaoa.set_ylim([0, 20])
# plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
# plt.ylabel('Angle of Attack [deg]', fontsize=fs+2, fontweight='bold')
# plt.xticks(fontsize=fs)
# plt.yticks(fontsize=fs)
# plt.grid(color=[0.8,0.8,0.8], linestyle='--')
# plt.subplots_adjust(bottom = 0.15, left = 0.15)
# fig_name = 'aoa.png'
# faoa.savefig(folder_output + fig_name)
#
# # Induction
# fa, axa = plt.subplots(1,1,figsize=(5.3, 4))
# axa.plot(wt_opt['blade.outer_shape_bem.s'], wt_opt['sse.powercurve.ax_induct_cutin'])
# axa.legend(fontsize=fs)
# axa.set_ylim([0, 0.5])
# plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
# plt.ylabel('Axial Induction [-]', fontsize=fs+2, fontweight='bold')
# plt.xticks(fontsize=fs)
# plt.yticks(fontsize=fs)
# plt.grid(color=[0.8,0.8,0.8], linestyle='--')
# plt.subplots_adjust(bottom = 0.15, left = 0.15)
# fig_name = 'induction.png'
# fa.savefig(folder_output + fig_name)
#
# n_pitch = analysis_options['servose']['n_pitch_perf_surfaces']
# n_tsr   = analysis_options['servose']['n_tsr_perf_surfaces']
# n_U     = analysis_options['servose']['n_U_perf_surfaces']
#
#
# for i in range(n_U):
#     fig0, ax0 = plt.subplots(1,1,figsize=(5.3, 4))
#     CS0 = ax0.contour(wt_opt['sse.aeroperf_tables.pitch_vector'], wt_opt['sse.aeroperf_tables.tsr_vector'], wt_opt['sse.aeroperf_tables.Cp'][:, :, i], levels=[0.0, 0.3, 0.40, 0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50 ])
#     ax0.clabel(CS0, inline=1, fontsize=fs)
#     plt.title('Power Coefficient', fontsize=fs+2, fontweight='bold')
#     plt.xlabel('Pitch Angle [deg]', fontsize=fs+2, fontweight='bold')
#     plt.ylabel('TSR [-]', fontsize=fs+2, fontweight='bold')
#     plt.xticks(fontsize=fs)
#     plt.yticks(fontsize=fs)
#     plt.grid(color=[0.8,0.8,0.8], linestyle='--')
#     plt.subplots_adjust(bottom = 0.15, left = 0.15)
#
#
#     fig0, ax0 = plt.subplots(1,1,figsize=(5.3, 4))
#     CS0 = ax0.contour(wt_opt['sse.aeroperf_tables.pitch_vector'], wt_opt['sse.aeroperf_tables.tsr_vector'], wt_opt['sse.aeroperf_tables.Ct'][:, :, i])
#     ax0.clabel(CS0, inline=1, fontsize=fs)
#     plt.title('Thrust Coefficient', fontsize=fs+2, fontweight='bold')
#     plt.xlabel('Pitch Angle [deg]', fontsize=fs+2, fontweight='bold')
#     plt.ylabel('TSR [-]', fontsize=fs+2, fontweight='bold')
#     plt.xticks(fontsize=fs)
#     plt.yticks(fontsize=fs)
#     plt.grid(color=[0.8,0.8,0.8], linestyle='--')
#     plt.subplots_adjust(bottom = 0.15, left = 0.15)
#
#
#     fig0, ax0 = plt.subplots(1,1,figsize=(5.3, 4))
#     CS0 = ax0.contour(wt_opt['sse.aeroperf_tables.pitch_vector'], wt_opt['sse.aeroperf_tables.tsr_vector'], wt_opt['sse.aeroperf_tables.Cq'][:, :, i])
#     ax0.clabel(CS0, inline=1, fontsize=fs)
#     plt.title('Torque Coefficient', fontsize=fs+2, fontweight='bold')
#     plt.xlabel('Pitch Angle [deg]', fontsize=fs+2, fontweight='bold')
#     plt.ylabel('TSR [-]', fontsize=fs+2, fontweight='bold')
#     plt.xticks(fontsize=fs)
#     plt.yticks(fontsize=fs)
#     plt.grid(color=[0.8,0.8,0.8], linestyle='--')
#     plt.subplots_adjust(bottom = 0.15, left = 0.15)
#
#
# plt.show()