import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator, interp1d
from wisdem.rotorse.rotor_geometry_yaml import remap2grid, remapAirfoil, arc_length, rotate, ReferenceBlade

def plot_spanwise_vars(blade, path, show_plots = True):
    
    # Chord
    fc, axc  = plt.subplots(1,1,figsize=(2.5, 2))
    fc.subplots_adjust(bottom=0.15, left=0.25)
    axc.plot(blade['pf']['s'], blade['pf']['chord'])
    axc.set(xlabel='r/R' , ylabel='Chord (m)')
    axc.grid(color=[0.8,0.8,0.8], linestyle='--')
    fig_name = 'design_planform_chord.png'
    fc.savefig(path + fig_name)
    
    # Theta
    ft, axt  = plt.subplots(1,1,figsize=(2.5, 2))
    ft.subplots_adjust(bottom=0.15, left=0.25)
    axt.plot(blade['pf']['s'], blade['pf']['theta'])
    axt.set(xlabel='r/R' , ylabel='Twist (deg)')
    axt.grid(color=[0.8,0.8,0.8], linestyle='--')
    fig_name = 'design_planform_theta.png'
    ft.savefig(path + fig_name)
    
    # Pitch axis
    fp, axp  = plt.subplots(1,1,figsize=(2.5, 2))
    fp.subplots_adjust(bottom=0.15, left=0.25)
    axp.plot(blade['pf']['s'], blade['pf']['p_le']*100.)
    axp.set(xlabel='r/R' , ylabel='Pitch Axis (%)')
    axp.grid(color=[0.8,0.8,0.8], linestyle='--')
    fig_name = 'design_planform_p_le.png'
    fp.savefig(path + fig_name)
    
    
    # Planform
    le = blade['pf']['p_le']*blade['pf']['chord']
    te = (1. - blade['pf']['p_le'])*blade['pf']['chord']

    fpl, axpl  = plt.subplots(1,1,figsize=(2.5, 2))
    fpl.subplots_adjust(bottom=0.15, left=0.25)
    axpl.plot(blade['pf']['s'], -le)
    axpl.plot(blade['pf']['s'], te)
    axpl.set(xlabel='r/R' , ylabel='Planform (m)')
    axpl.grid(color=[0.8,0.8,0.8], linestyle='--')
    axpl.legend()
    fig_name = 'design_planform_planform.png'
    fpl.savefig(path + fig_name)
    
    # Relative thickness
    frt, axrt  = plt.subplots(1,1,figsize=(2.5, 2))
    frt.subplots_adjust(bottom=0.15, left=0.25)
    axrt.plot(blade['pf']['s'], blade['pf']['rthick']*100.)
    axrt.set(xlabel='r/R' , ylabel='Relative Thickness (%)')
    axrt.grid(color=[0.8,0.8,0.8], linestyle='--')
    fig_name = 'design_planform_rthick.png'
    frt.savefig(path + fig_name)
    
    # Absolute thickness
    fat, axat  = plt.subplots(1,1,figsize=(2.5, 2))
    fat.subplots_adjust(bottom=0.15, left=0.25)
    axat.plot(blade['pf']['s'], blade['pf']['rthick']*blade['pf']['chord'])
    axat.set(xlabel='r/R' , ylabel='Absolute Thickness (m)')
    axat.grid(color=[0.8,0.8,0.8], linestyle='--')
    fig_name = 'design_planform_absthick.png'
    fat.savefig(path + fig_name)
    
    # Prebend
    fpb, axpb  = plt.subplots(1,1,figsize=(2.5, 2))
    fpb.subplots_adjust(bottom=0.15, left=0.25)
    axpb.plot(blade['pf']['s'], blade['pf']['precurve'])
    axpb.set(xlabel='r/R' , ylabel='Prebend (m)')
    axpb.grid(color=[0.8,0.8,0.8], linestyle='--')
    fig_name = 'design_planform_prebend.png'
    fpb.savefig(path + fig_name)
    
    # Sweep
    fsw, axsw  = plt.subplots(1,1,figsize=(2.5, 2))
    fsw.subplots_adjust(bottom=0.15, left=0.25)
    axsw.plot(blade['pf']['s'], blade['pf']['presweep'])
    axsw.set(xlabel='r/R' , ylabel='Presweep (m)')
    axsw.grid(color=[0.8,0.8,0.8], linestyle='--')
    fig_name = 'design_planform_presweep.png'
    plt.subplots_adjust(left = 0.14)
    fsw.savefig(path + fig_name)
    
    idx_spar  = [i for i, sec in enumerate(blade['st']['layers']) if sec['name'].lower()==self.spar_var[0].lower()][0]
    idx_te    = [i for i, sec in enumerate(blade['st']['layers']) if sec['name'].lower()==self.te_var.lower()][0]
    idx_skin  = [i for i, sec in enumerate(blade['st']['layers']) if sec['name'].lower()=='shell_skin'][0]
    
    # Spar caps thickness
    fsc, axsc  = plt.subplots(1,1,figsize=(2.5, 2))
    fsc.subplots_adjust(bottom=0.15, left=0.25)
    axsc.plot(blade['st']['layers'][idx_spar]['thickness']['grid'], blade['st']['layers'][idx_spar]['thickness']['values'])
    axsc.set(xlabel='r/R' , ylabel='Spar Caps Thickness (m)')
    axsc.grid(color=[0.8,0.8,0.8], linestyle='--')
    fig_name = 'design_planform_sc.png'
    plt.subplots_adjust(left = 0.14)
    fsc.savefig(path + fig_name)
    
    # TE reinf thickness
    fte, axte  = plt.subplots(1,1,figsize=(2.5, 2))
    fte.subplots_adjust(bottom=0.15, left=0.25)
    axte.plot(blade['st']['layers'][idx_te]['thickness']['grid'], blade['st']['layers'][idx_te]['thickness']['values'])
    axte.set(xlabel='r/R' , ylabel='TE Reinf. Thickness (m)')
    axte.grid(color=[0.8,0.8,0.8], linestyle='--')
    fig_name = 'design_planform_te.png'
    plt.subplots_adjust(left = 0.14)
    fte.savefig(path + fig_name)
    
    # Skin
    fsk, axsk  = plt.subplots(1,1,figsize=(2.5, 2))
    fsk.subplots_adjust(bottom=0.15, left=0.25)
    axsk.plot(blade['st']['layers'][idx_skin]['thickness']['grid'], blade['st']['layers'][idx_skin]['thickness']['values'])
    axsk.set(xlabel='r/R' , ylabel='Shell Skin Thickness (m)')
    axsk.grid(color=[0.8,0.8,0.8], linestyle='--')
    fig_name = 'design_planform_skin.png'
    fsk.savefig(path + fig_name)
    
    if show_plots:
        plt.show()
    return None

###################################

def find_half(arc_pos, arc, LE_idx, Naf):
    if arc_pos >= arc[LE_idx]:
        idx_half = range(LE_idx, Naf)
    else:
        idx_half = range(0, LE_idx)
    return idx_half

def arc2xy_section(arc, x, y, start_nd_arc, end_nd_arc, LE_idx, Naf, Nsec):

    # find SS or PS to interpolate over
    idx_half_start = find_half(start_nd_arc, arc, LE_idx, Naf)
    idx_half_end   = find_half(end_nd_arc, arc, LE_idx, Naf)

    # No LE or Te crossing of composite section
    # if idx_half_start == idx_half_end:
    section_x = remap2grid(arc[idx_half_start], x[idx_half_start], np.linspace(start_nd_arc, end_nd_arc, num = Nsec))#, spline=interp1d)
    section_y = remap2grid(x[idx_half_start], y[idx_half_start], section_x)#, spline=interp1d)

    # else:
        # print('LE / TE not yet implimented')

    return section_x, section_y

def arc2xy_web(arc, x, y, start_nd_arc, end_nd_arc, LE_idx, Naf, Nsec):

    # find SS or PS to interpolate over
    idx_half_start = find_half(start_nd_arc, arc, LE_idx, Naf)
    idx_half_end   = find_half(end_nd_arc, arc, LE_idx, Naf)

    # No LE or Te crossing of composite section
    x_start = remap2grid(arc[idx_half_start], x[idx_half_start], np.linspace(start_nd_arc, end_nd_arc, num = Nsec), spline=interp1d)
    y_start = remap2grid(x[idx_half_start], y[idx_half_start], x_start, spline=interp1d)
    x_end   = remap2grid(arc[idx_half_end], x[idx_half_end], np.linspace(start_nd_arc, end_nd_arc, num = Nsec), spline=interp1d)
    y_end   = remap2grid(x[idx_half_end], y[idx_half_end], x_end, spline=interp1d)

    return [x_start, x_end], [y_start, y_end]


def plot_lofted(blade, layer_vars, path, show_plots = True):

    NPTS = len(blade['pf']['s'])
    Naf  = len(blade['profile'][:,0,0])
    Nsec = 30
    

    
    
    # Get Profile Coordinates, xy and arc
    out = {}
    out['profile']           = {}
    out['profile']['xy']     = np.flip(copy.deepcopy(blade['profile']), axis=0)
    out['profile']['LE_idx'] = np.array([np.argmin(out['profile']['xy'][:,0,i]) for i in range(NPTS)])
    out['profile']['arc']    = []
    for i in range(NPTS):
        profile_i = out['profile']['xy'][:,:,i]
        # if profile_i[0,1] != profile_i[-1,1]:
        #     profile_i = np.row_stack((profile_i, profile_i[0,:])) 
        arc = arc_length(profile_i[:,0], profile_i[:,1])
        out['profile']['arc'].append(arc/arc[-1])

    # For each layer, get surface locations
    for var in layer_vars:
        type_sec = 'layers'
        idx_sec  = [i for i, sec in enumerate(blade['st'][type_sec]) if sec['name'].lower()==var.lower()][0]
        out[var] = {}
        out[var]['start_nd_arc'] = np.array(blade['st'][type_sec][idx_sec]['start_nd_arc']['values'])
        out[var]['end_nd_arc']   = np.array(blade['st'][type_sec][idx_sec]['end_nd_arc']['values'])

        out[var]['xy'] = np.zeros((Nsec,2,NPTS))
        for i in range(NPTS):
            profile_i = out['profile']['xy'][:,:,i]
            # if profile_i[0,1] != profile_i[-1,1]:
            #     profile_i = np.row_stack((profile_i, profile_i[0,:]))
            out[var]['xy'][:,:,i] = np.column_stack(arc2xy_section(out['profile']['arc'][i], profile_i[:,0], profile_i[:,1], out[var]['start_nd_arc'][i], out[var]['end_nd_arc'][i], out['profile']['LE_idx'][i], Naf, Nsec))

    # For each web, get locations

    # Other locations (LE, pitch axis, etc)

    # Loop through plot elements, span; translate
    plt_vars = ['profile']
    plt_vars.extend(layer_vars)
    
    
    for i in range(NPTS):
        for j, var in enumerate(plt_vars):
            # print(var,i)

            # center about pitch axis
            out[var]['xy'][:,0,i] = out[var]['xy'][:,0,i] - blade['pf']['p_le'][i]

            # scale by chord
            out[var]['xy'][:,:,i] = out[var]['xy'][:,:,i] * blade['pf']['chord'][i]

            # rotate
            out[var]['xy'][:,:,i] = np.column_stack(rotate(0., 0., out[var]['xy'][:,0,i], out[var]['xy'][:,1,i], np.radians(blade['pf']['theta'][i])))

            # prebend translation
            out[var]['xy'][:,1,i] = out[var]['xy'][:,1,i] - blade['pf']['precurve'][i]

            # sweep translation
            out[var]['xy'][:,0,i] = out[var]['xy'][:,0,i] - blade['pf']['presweep'][i]
            

        
        

    # Plotting

    ######### 2D
    color    = ['k','b', 'r']
    for i in range(NPTS):
        for j, var in enumerate(plt_vars):
            if var == 'profile':
                profile_i = out['profile']['xy'][:,:,i]
                if profile_i[0,1] != profile_i[-1,1]:
                    profile_i = np.row_stack((profile_i, profile_i[0,:]))
                plt.plot(profile_i[:,0], profile_i[:,1], color=color[j])
            else:
                # # if blade['st'][type_sec][20]['thickness']['values'][i] != None and blade['st'][type_sec][21]['thickness']['values'][i] != None:
                plt.plot(out[var]['xy'][:,0,i], out[var]['xy'][:,1,i], color=color[j])
                plt.plot([out[var]['xy'][0,0,i],out[var]['xy'][-1,0,i]], [out[var]['xy'][0,1,i],out[var]['xy'][-1,1,i]], '.', color=color[j])
    plt.show()
    plt.axis('equal')
        


    ######### 3D

    color    = ['k','b', 'r']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(NPTS):
        for j, var in enumerate(plt_vars):
            if var == 'profile':
                profile_i = out['profile']['xy'][:,:,i]
                if profile_i[0,1] != profile_i[-1,1]:
                    profile_i = np.row_stack((profile_i, profile_i[0,:]))
                Y = profile_i[:,0]
                Z = profile_i[:,1]
                X = [blade['pf']['r'][i]]*len(profile_i[:,0])
                ax.plot(X, Y, zs=Z, color=color[j])
            else:
                plt.plot(out[var]['xy'][:,0,i], out[var]['xy'][:,1,i], color=color[j])
                plt.plot([out[var]['xy'][0,0,i],out[var]['xy'][-1,0,i]], [out[var]['xy'][0,1,i],out[var]['xy'][-1,1,i]], '.', color=color[j])



    Y = out['profile']['xy'][:,0,0]
    Z = out['profile']['xy'][:,1,0]
    X = np.array([blade['pf']['r']]*len(Y))
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


    ax.view_init(azim=-90., elev=-180.)
    plt.show()

    return None


if __name__ == "__main__":

    fname_input       = "/mnt/c/Users/egaertne/WISDEM/nrel15mw/design/turbine_inputs/NREL15MW_opt_v03.yaml"
    dir_out           = "/mnt/c/Users/egaertne/WISDEM/nrel15mw/design/outputs/NREL15MW_opt_v03/post"

    ## Load and Format Blade
    refBlade = ReferenceBlade()
    refBlade.verbose  = True
    refBlade.spar_var = ['Spar_cap_ss', 'Spar_cap_ps']
    refBlade.te_var   = 'TE_reinforcement'
    refBlade.NINPUT   = 8
    refBlade.NPITS    = 50
    # refBlade.NPTS_AfProfile = 200
    refBlade.validate = False
    # refBlade.fname_schema = "turbine_inputs/IEAontology_schema.yaml"

    blade = refBlade.initialize(fname_input)

    plt_vars = ['Spar_cap_ss', 'Spar_cap_ps']
    plot_lofted(blade, plt_vars, dir_out)
