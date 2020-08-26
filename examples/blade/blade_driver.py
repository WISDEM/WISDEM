from wisdem.glue_code.runWISDEM import run_wisdem
import matplotlib.pyplot as plt
import numpy as np
import os
from wisdem.commonse.mpi_tools import MPI

show_plots = False

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input         = mydir + os.sep + 'blade.yaml'
fname_modeling_options = mydir + os.sep + 'modeling_options.yaml'
fname_analysis_options = mydir + os.sep + 'analysis_options_blade.yaml'

wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0

if rank == 0:
    folder_output = analysis_options['general']['folder_output']
    # Printing and plotting results
    print('Nominal AEP in GWh = ' + str(wt_opt['sse.AEP']*1.e-6))
    print('Blade mass in kg = ' + str(wt_opt['elastic.precomp.blade_mass']))
    print('Nat frequencies blades flap in Hz = ' + str(wt_opt['rlds.frame.flap_mode_freqs']))
    print('Nat frequencies blades edge in Hz = ' + str(wt_opt['rlds.frame.edge_mode_freqs']))
    print('1P in Hz = ' + str(wt_opt['sse.powercurve.rated_Omega']/60.))
    print('3P in Hz = ' + str(wt_opt['sse.powercurve.rated_Omega']/60.*3.))
    print('6P in Hz = ' + str(wt_opt['sse.powercurve.rated_Omega']/60.*6.))
    print('Tip tower clearance in m     = ' + str(wt_opt['tcons.blade_tip_tower_clearance']))
    print('Tip deflection constraint    = ' + str(wt_opt['tcons.tip_deflection_ratio']))

    print('Forces at hub: ' + str(wt_opt['rlds.aero_hub_loads.Fxyz_hub_aero']))
    print('Moments at hub: ' + str(wt_opt['rlds.aero_hub_loads.Mxyz_hub_aero']))

    print('Design initial: tsr: ' + str(wt_opt['control.rated_TSR']))
    print('Design final: tsr: ' + str(wt_opt['pc.tsr_opt']))

    print('Initial airfoil positions: ' + str(wt_opt['blade.outer_shape_bem.af_position']))
    print('Final airfoil positions: ' + str(wt_opt['blade.opt_var.af_position']))

    fs = 10

    import matplotlib.pyplot as plt
    feps, axeps = plt.subplots(1,1,figsize=(5.3, 4))
    axeps.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.frame.strainU_spar'], label='Spar ss')
    axeps.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.frame.strainL_spar'], label='Spar ps')
    axeps.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.frame.strainU_te'], label='TE ss')
    axeps.plot(wt_opt['assembly.r_blade'], wt_opt['rlds.frame.strainL_te'], label='TE ps')
    axeps.legend(fontsize=fs)
    axeps.set_ylim([-5e-3, 5e-3])
    plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
    plt.ylabel('Strains [-]', fontsize=fs+2, fontweight='bold')
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(color=[0.8,0.8,0.8], linestyle='--')
    plt.subplots_adjust(bottom = 0.15, left = 0.18)
    fig_name = 'eps.png'
    feps.savefig(os.path.join(folder_output , fig_name))


    # Angle of attack and stall angle
    faoa, axaoa = plt.subplots(1,1,figsize=(5.3, 4))
    axaoa.plot(wt_opt['stall_check.s'], wt_opt['stall_check.aoa_along_span'])
    axaoa.plot(wt_opt['stall_check.s'], wt_opt['stall_check.stall_angle_along_span'])
    axaoa.set_ylim([0, 20])
    plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
    plt.ylabel('Angle of Attack [deg]', fontsize=fs+2, fontweight='bold')
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(color=[0.8,0.8,0.8], linestyle='--')
    plt.subplots_adjust(bottom = 0.15, left = 0.15)
    fig_name = 'aoa.png'
    faoa.savefig(os.path.join(folder_output , fig_name))

    # Induction
    fa, axa = plt.subplots(1,1,figsize=(5.3, 4))
    axa.plot(wt_opt['blade.outer_shape_bem.s'], wt_opt['sse.powercurve.ax_induct_regII'])
    axa.set_ylim([0, 0.5])
    plt.xlabel('Blade Nondimensional Span [-]', fontsize=fs+2, fontweight='bold')
    plt.ylabel('Axial Induction [-]', fontsize=fs+2, fontweight='bold')
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(color=[0.8,0.8,0.8], linestyle='--')
    plt.subplots_adjust(bottom = 0.15, left = 0.15)
    fig_name = 'induction.png'
    fa.savefig(os.path.join(folder_output , fig_name))

    if show_plots: 
        plt.show()