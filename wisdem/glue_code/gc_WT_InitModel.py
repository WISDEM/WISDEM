import numpy as np
from wisdem.rotorse.geometry_tools.geometry import AirfoilShape
from wisdem.commonse.utilities import arc_length
from wisdem.commonse.csystem import DirectionVector

def yaml2openmdao(wt_opt, modeling_options, wt_init):
    # Function to assign values to the openmdao group Wind_Turbine and all its components
    
    # These are the required components
    assembly        = wt_init['assembly']
    wt_opt = assign_configuration_values(wt_opt, assembly)

    materials       = wt_init['materials']
    wt_opt = assign_material_values(wt_opt, modeling_options, materials)

    # Now all of the optional components
    if modeling_options['flags']['environment']:
        environment     = wt_init['environment']
        wt_opt = assign_environment_values(wt_opt, environment, modeling_options['offshore'])
    else:
        environment = {}

    if modeling_options['flags']['blade']:
        blade           = wt_init['components']['blade']
        wt_opt = assign_blade_values(wt_opt, modeling_options, blade)
    else:
        blade = {}

    if modeling_options['flags']['airfoils']:
        airfoils        = wt_init['airfoils']
        wt_opt = assign_airfoil_values(wt_opt, modeling_options, airfoils)
    else:
        airfoils = {}
        
    if modeling_options['flags']['control']:
        control         = wt_init['control']
        wt_opt = assign_control_values(wt_opt, modeling_options, control)
    else:
        control = {}
        
    if modeling_options['flags']['hub']:
        hub    = wt_init['components']['hub']
        wt_opt = assign_hub_values(wt_opt, hub)
    else:
        hub = {}
        
    if modeling_options['flags']['nacelle']:
        nacelle         = wt_init['components']['nacelle']
        wt_opt = assign_nacelle_values(wt_opt, assembly, nacelle)
    else:
        nacelle = {}
        
    if modeling_options['flags']['RNA']:
        RNA = wt_init['components']['RNA']
    else:
        RNA = {}
        
    if modeling_options['flags']['tower']:
        tower           = wt_init['components']['tower']
        wt_opt = assign_tower_values(wt_opt, modeling_options, tower)
    else:
        tower = {}

    if modeling_options['flags']['monopile']:
        monopile = wt_init['components']['monopile']
        wt_opt   = assign_monopile_values(wt_opt, modeling_options, monopile)
    else:
        monopile = {}
        
    if modeling_options['flags']['floating']:
        floating = wt_init['components']['floating']
        wt_opt   = assign_floating_values(wt_opt, modeling_options, floating)
    else:
        floating = {}
        
    if modeling_options['flags']['foundation']:
        foundation      = wt_init['components']['foundation']
        wt_opt = assign_foundation_values(wt_opt, foundation)
    else:
        foundation = {}

    if modeling_options['flags']['bos']:
        bos           = wt_init['bos']
        wt_opt = assign_bos_values(wt_opt, bos, modeling_options['offshore'])
    else:
        costs = {}

    if modeling_options['flags']['costs']:
        costs           = wt_init['costs']
        wt_opt = assign_costs_values(wt_opt, costs)
    else:
        costs = {}
        
    if 'elastic_properties_mb' in blade.keys() and modeling_options['Analysis_Flags']['DriveSE']:
        wt_opt = assign_RNA_values(wt_opt, modeling_options, blade, RNA)

    return wt_opt
    
def assign_blade_values(wt_opt, modeling_options, blade):
    # Function to assign values to the openmdao group Blade
    wt_opt = assign_outer_shape_bem_values(wt_opt, modeling_options, blade['outer_shape_bem'])
    wt_opt = assign_internal_structure_2d_fem_values(wt_opt, modeling_options, blade['internal_structure_2d_fem'])
    wt_opt = assign_te_flaps_values(wt_opt, modeling_options, blade)
    
    return wt_opt
    
def assign_outer_shape_bem_values(wt_opt, modeling_options, outer_shape_bem):
    # Function to assign values to the openmdao component Blade_Outer_Shape_BEM
    
    nd_span     = modeling_options['blade']['nd_span']
    
    wt_opt['blade.outer_shape_bem.af_position'] = outer_shape_bem['airfoil_position']['grid']
    wt_opt['blade.opt_var.af_position']         = outer_shape_bem['airfoil_position']['grid']
    
    wt_opt['blade.outer_shape_bem.s_default']        = nd_span
    wt_opt['blade.outer_shape_bem.chord_yaml']       = np.interp(nd_span, outer_shape_bem['chord']['grid'], outer_shape_bem['chord']['values'])
    wt_opt['blade.outer_shape_bem.twist_yaml']       = np.interp(nd_span, outer_shape_bem['twist']['grid'], outer_shape_bem['twist']['values'])
    wt_opt['blade.outer_shape_bem.pitch_axis_yaml']  = np.interp(nd_span, outer_shape_bem['pitch_axis']['grid'], outer_shape_bem['pitch_axis']['values'])
    
    wt_opt['blade.outer_shape_bem.ref_axis_yaml'][:,0]  = np.interp(nd_span, outer_shape_bem['reference_axis']['x']['grid'], outer_shape_bem['reference_axis']['x']['values'])
    wt_opt['blade.outer_shape_bem.ref_axis_yaml'][:,1]  = np.interp(nd_span, outer_shape_bem['reference_axis']['y']['grid'], outer_shape_bem['reference_axis']['y']['values'])
    wt_opt['blade.outer_shape_bem.ref_axis_yaml'][:,2]  = np.interp(nd_span, outer_shape_bem['reference_axis']['z']['grid'], outer_shape_bem['reference_axis']['z']['values'])

    # # Smoothing of the shapes
    # # Chord
    # chord_init      = wt_opt['blade.outer_shape_bem.chord']
    # s_interp_c      = np.array([0.0, 0.05, 0.2, 0.35, 0.65, 0.9, 1.0 ])
    # f_interp1       = interp1d(nd_span,chord_init)
    # chord_int1      = f_interp1(s_interp_c)
    # f_interp2       = PchipInterpolator(s_interp_c,chord_int1)
    # chord_int2      = f_interp2(nd_span)
    
    # import matplotlib.pyplot as plt
    # fc, axc  = plt.subplots(1,1,figsize=(5.3, 4))
    # axc.plot(nd_span, chord_init, c='k', label='Initial')
    # axc.plot(s_interp_c, chord_int1, 'ko', label='Interp Points')
    # axc.plot(nd_span, chord_int2, c='b', label='PCHIP')
    # axc.set(xlabel='r/R' , ylabel='Chord (m)')
    # fig_name = 'interp_chord.png'
    # axc.legend()
    # # Planform
    # le_init = wt_opt['blade.outer_shape_bem.pitch_axis']*wt_opt['blade.outer_shape_bem.chord']
    # te_init = (1. - wt_opt['blade.outer_shape_bem.pitch_axis'])*wt_opt['blade.outer_shape_bem.chord']
    
    # s_interp_le     = np.array([0.0, 0.5, 0.8, 1.0])
    # f_interp1       = interp1d(wt_opt['blade.outer_shape_bem.s_default'],le_init)
    # le_int1         = f_interp1(s_interp_le)
    # f_interp2       = PchipInterpolator(s_interp_le,le_int1)
    # le_int2         = f_interp2(wt_opt['blade.outer_shape_bem.s_default'])
    
    # fpl, axpl  = plt.subplots(1,1,figsize=(5.3, 4))
    # axpl.plot(wt_opt['blade.outer_shape_bem.s_default'], -le_init, c='k', label='LE init')
    # axpl.plot(wt_opt['blade.outer_shape_bem.s_default'], -le_int2, c='b', label='LE smooth old pa')
    # axpl.plot(wt_opt['blade.outer_shape_bem.s_default'], te_init, c='k', label='TE init')
    # axpl.plot(wt_opt['blade.outer_shape_bem.s_default'], wt_opt['blade.outer_shape_bem.chord'] - le_int2, c='b', label='TE smooth old pa')
    # axpl.set(xlabel='r/R' , ylabel='Planform (m)')
    # axpl.legend()
    # plt.show()
    # # np.savetxt('temp.txt', le_int2/wt_opt['blade.outer_shape_bem.chord'])
    # exit()

    # # # Twist
    # theta_init      = wt_opt['blade.outer_shape_bem.twist']
    # s_interp      = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0 ])
    # f_interp1       = interp1d(nd_span,theta_init)
    # theta_int1      = f_interp1(s_interp)
    # f_interp2       = PchipInterpolator(s_interp,theta_int1)
    # theta_int2      = f_interp2(nd_span)
    
    # import matplotlib.pyplot as plt
    # fc, axc  = plt.subplots(1,1,figsize=(5.3, 4))
    # axc.plot(nd_span, theta_init, c='k', label='Initial')
    # axc.plot(s_interp, theta_int1, 'ko', label='Interp Points')
    # axc.plot(nd_span, theta_int2, c='b', label='PCHIP')
    # axc.set(xlabel='r/R' , ylabel='Twist (deg)')
    # axc.legend()
    # plt.show()
    # exit()
    
    return wt_opt
    
def assign_internal_structure_2d_fem_values(wt_opt, modeling_options, internal_structure_2d_fem):
    # Function to assign values to the openmdao component Blade_Internal_Structure_2D_FEM
    
    n_span          = modeling_options['blade']['n_span']
    n_webs          = modeling_options['blade']['n_webs']
    
    web_rotation    = np.zeros((n_webs, n_span))
    web_offset_y_pa = np.zeros((n_webs, n_span))
    web_start_nd    = np.zeros((n_webs, n_span))
    web_end_nd      = np.zeros((n_webs, n_span))
    definition_web  = np.zeros(n_webs)
    nd_span         = wt_opt['blade.outer_shape_bem.s_default']
    
    # Loop through the webs and interpolate spanwise the values
    for i in range(n_webs):
        if 'rotation' in internal_structure_2d_fem['webs'][i] and 'offset_y_pa' in internal_structure_2d_fem['webs'][i]:
            if 'fixed' in internal_structure_2d_fem['webs'][i]['rotation'].keys():
                if internal_structure_2d_fem['webs'][i]['rotation']['fixed'] == 'twist':
                    definition_web[i] = 1
                else:
                    exit('Invalid rotation reference for web ' + self.modeling_options['blade']['web_name'][i] + '. Please check the yaml input file')
            else:
                web_rotation[i,:] = np.interp(nd_span, internal_structure_2d_fem['webs'][i]['rotation']['grid'], internal_structure_2d_fem['webs'][i]['rotation']['values'], left=0., right=0.)
                definition_web[i] = 2
            web_offset_y_pa[i,:] = np.interp(nd_span, internal_structure_2d_fem['webs'][i]['offset_y_pa']['grid'], internal_structure_2d_fem['webs'][i]['offset_y_pa']['values'], left=0., right=0.)
        elif 'start_nd_arc' in internal_structure_2d_fem['webs'][i] and 'end_nd_arc' in internal_structure_2d_fem['webs'][i]:
            definition_web[i] = 3
            web_start_nd[i,:] = np.interp(nd_span, internal_structure_2d_fem['webs'][i]['start_nd_arc']['grid'], internal_structure_2d_fem['webs'][i]['start_nd_arc']['values'], left=0., right=0.)
            web_end_nd[i,:]   = np.interp(nd_span, internal_structure_2d_fem['webs'][i]['end_nd_arc']['grid'], internal_structure_2d_fem['webs'][i]['end_nd_arc']['values'], left=0., right=0.)
        else:
            exit('Webs definition not supported. Please check the yaml input.')
    
    n_layers        = modeling_options['blade']['n_layers']
    layer_name      = n_layers * ['']
    layer_mat       = n_layers * ['']
    thickness       = np.zeros((n_layers, n_span))
    fiber_orient    = np.zeros((n_layers, n_span))
    layer_rotation  = np.zeros((n_layers, n_span))
    layer_offset_y_pa  = np.zeros((n_layers, n_span))
    layer_width     = np.zeros((n_layers, n_span))
    layer_midpoint_nd  = np.zeros((n_layers, n_span))
    layer_start_nd  = np.zeros((n_layers, n_span))
    layer_end_nd    = np.zeros((n_layers, n_span))
    layer_web       = np.zeros(n_layers)
    layer_side      = n_layers * ['']
    definition_layer= np.zeros(n_layers)
    index_layer_start= np.zeros(n_layers)
    index_layer_end = np.zeros(n_layers)

    
    # Loop through the layers, interpolate along blade span, assign the inputs, and the definition flag
    for i in range(n_layers):
        layer_name[i]  = modeling_options['blade']['layer_name'][i]
        layer_mat[i]   = modeling_options['blade']['layer_mat'][i]
        thickness[i]   = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['thickness']['grid'], internal_structure_2d_fem['layers'][i]['thickness']['values'], left=0., right=0.)
        if 'rotation' not in internal_structure_2d_fem['layers'][i] and 'offset_y_pa' not in internal_structure_2d_fem['layers'][i] and 'width' not in internal_structure_2d_fem['layers'][i] and 'start_nd_arc' not in internal_structure_2d_fem['layers'][i] and 'end_nd_arc' not in internal_structure_2d_fem['layers'][i] and 'web' not in internal_structure_2d_fem['layers'][i]:
            definition_layer[i] = 1
            
        if 'rotation' in internal_structure_2d_fem['layers'][i] and 'offset_y_pa' in internal_structure_2d_fem['layers'][i] and 'width' in internal_structure_2d_fem['layers'][i] and 'side' in internal_structure_2d_fem['layers'][i]:
            if 'fixed' in internal_structure_2d_fem['layers'][i]['rotation'].keys():
                if internal_structure_2d_fem['layers'][i]['rotation']['fixed'] == 'twist':
                    definition_layer[i] = 2
                else:
                    exit('Invalid rotation reference for layer ' + layer_name[i] + '. Please check the yaml input file.')
            else:
                layer_rotation[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['rotation']['grid'], internal_structure_2d_fem['layers'][i]['rotation']['values'], left=0., right=0.)
                definition_layer[i] = 3
            layer_offset_y_pa[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['offset_y_pa']['grid'], internal_structure_2d_fem['layers'][i]['offset_y_pa']['values'], left=0., right=0.)
            layer_width[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['width']['grid'], internal_structure_2d_fem['layers'][i]['width']['values'], left=0., right=0.)
            layer_side[i]    = internal_structure_2d_fem['layers'][i]['side']
        if 'midpoint_nd_arc' in internal_structure_2d_fem['layers'][i] and 'width' in internal_structure_2d_fem['layers'][i]:
            if 'fixed' in internal_structure_2d_fem['layers'][i]['midpoint_nd_arc'].keys():
                if internal_structure_2d_fem['layers'][i]['midpoint_nd_arc']['fixed'] == 'TE':
                    layer_midpoint_nd[i,:] = np.ones(n_span)
                    definition_layer[i] = 4
                elif internal_structure_2d_fem['layers'][i]['midpoint_nd_arc']['fixed'] == 'LE':
                    definition_layer[i] = 5
                    # layer_midpoint_nd[i,:] = -np.ones(n_span) # To be assigned later!
            else:
                layer_midpoint_nd[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['midpoint_nd_arc']['grid'], internal_structure_2d_fem['layers'][i]['midpoint_nd_arc']['values'], left=0., right=0.)
            layer_width[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['width']['grid'], internal_structure_2d_fem['layers'][i]['width']['values'], left=0., right=0.)
        if 'start_nd_arc' in internal_structure_2d_fem['layers'][i] and definition_layer[i] == 0:
            if 'fixed' in internal_structure_2d_fem['layers'][i]['start_nd_arc'].keys():
                if internal_structure_2d_fem['layers'][i]['start_nd_arc']['fixed'] == 'TE':
                    layer_start_nd[i,:] = np.zeros(n_span)
                    # exit('No need to fix element to TE, set it to 0.')
                elif internal_structure_2d_fem['layers'][i]['start_nd_arc']['fixed'] == 'LE':
                    definition_layer[i] = 11
                else:
                    definition_layer[i] = 6
                    flag = False
                    for k in range(n_layers):
                        if layer_name[k] == internal_structure_2d_fem['layers'][i]['start_nd_arc']['fixed']:
                            index_layer_start[i] = k
                            flag = True
                            break
                    if flag == False:
                        exit('Error with layer ' + internal_structure_2d_fem['layers'][i]['name'])
            else:
                layer_start_nd[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['start_nd_arc']['grid'], internal_structure_2d_fem['layers'][i]['start_nd_arc']['values'], left=0., right=0.)
            if 'end_nd_arc' in internal_structure_2d_fem['layers'][i]:
                if 'fixed' in internal_structure_2d_fem['layers'][i]['end_nd_arc'].keys():
                    if internal_structure_2d_fem['layers'][i]['end_nd_arc']['fixed'] == 'TE':
                        layer_end_nd[i,:] = np.ones(n_span)
                        # exit('No need to fix element to TE, set it to 0.')
                    elif internal_structure_2d_fem['layers'][i]['end_nd_arc']['fixed'] == 'LE':
                        definition_layer[i] = 12
                    else:
                        flag = False
                        for k in range(n_layers):
                            if layer_name[k] == internal_structure_2d_fem['layers'][i]['end_nd_arc']['fixed']:
                                index_layer_end[i] = k
                                flag = True
                                break
                        if flag == False:
                            exit('Error with layer ' + internal_structure_2d_fem['layers'][i]['name'])
            if 'width' in internal_structure_2d_fem['layers'][i]:
                definition_layer[i] = 7
                layer_width[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['width']['grid'], internal_structure_2d_fem['layers'][i]['width']['values'], left=0., right=0.)
            
        if 'end_nd_arc' in internal_structure_2d_fem['layers'][i] and definition_layer[i] == 0:
            if 'fixed' in internal_structure_2d_fem['layers'][i]['end_nd_arc'].keys():
                if internal_structure_2d_fem['layers'][i]['end_nd_arc']['fixed'] == 'TE':
                    layer_end_nd[i,:] = np.ones(n_span)
                    # exit('No need to fix element to TE, set it to 0.')
                elif internal_structure_2d_fem['layers'][i]['end_nd_arc']['fixed'] == 'LE':
                    definition_layer[i] = 12
                else:
                    definition_layer[i] = 6
                    flag = False
                    if layer_name[k] == internal_structure_2d_fem['layers'][i]['end_nd_arc']['fixed']:
                        index_layer_end[i] = k
                        flag = True
                        break
                    if flag == False:
                        exit('Error with layer ' + internal_structure_2d_fem['layers'][i]['name'])
            else:
                layer_end_nd[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['end_nd_arc']['grid'], internal_structure_2d_fem['layers'][i]['end_nd_arc']['values'], left=0., right=0.)
            if 'width' in internal_structure_2d_fem['layers'][i]:
                definition_layer[i] = 8
                layer_width[i,:] = np.interp(nd_span, internal_structure_2d_fem['layers'][i]['width']['grid'], internal_structure_2d_fem['layers'][i]['width']['values'], left=0., right=0.)
            if 'start_nd_arc' in internal_structure_2d_fem['layers'][i]:
                definition_layer[i] = 9

        if 'web' in internal_structure_2d_fem['layers'][i]:
            web_name_i = internal_structure_2d_fem['layers'][i]['web']
            for j in range(modeling_options['blade']['n_webs']):
                if web_name_i == modeling_options['blade']['web_name'][j]:
                    k = j+1
                    break
            layer_web[i] = k
            definition_layer[i] = 10
    
    
    # Assign the openmdao values
    wt_opt['blade.internal_structure_2d_fem.layer_side']            = layer_side
    wt_opt['blade.internal_structure_2d_fem.layer_thickness']       = thickness
    wt_opt['blade.internal_structure_2d_fem.layer_midpoint_nd']     = layer_midpoint_nd
    wt_opt['blade.internal_structure_2d_fem.layer_web']             = layer_web
    wt_opt['blade.internal_structure_2d_fem.definition_web']        = definition_web
    wt_opt['blade.internal_structure_2d_fem.definition_layer']      = definition_layer
    wt_opt['blade.internal_structure_2d_fem.index_layer_start']     = index_layer_start
    wt_opt['blade.internal_structure_2d_fem.index_layer_end']       = index_layer_end

    wt_opt['blade.internal_structure_2d_fem.web_offset_y_pa_yaml']  = web_offset_y_pa
    wt_opt['blade.internal_structure_2d_fem.web_rotation_yaml']     = web_rotation
    wt_opt['blade.internal_structure_2d_fem.web_start_nd_yaml']     = web_start_nd
    wt_opt['blade.internal_structure_2d_fem.web_end_nd_yaml']       = web_end_nd
    wt_opt['blade.internal_structure_2d_fem.layer_offset_y_pa_yaml']= layer_offset_y_pa
    wt_opt['blade.internal_structure_2d_fem.layer_width_yaml']      = layer_width
    wt_opt['blade.internal_structure_2d_fem.layer_start_nd_yaml']   = layer_start_nd
    wt_opt['blade.internal_structure_2d_fem.layer_end_nd_yaml']     = layer_end_nd
    wt_opt['blade.internal_structure_2d_fem.layer_rotation_yaml']   = layer_rotation
    
    return wt_opt

def assign_te_flaps_values(wt_opt, modeling_options, blade):
    # Function to assign the trailing edge flaps data to the openmdao data structure
    if modeling_options['blade']['n_te_flaps'] > 0:   
        n_te_flaps = modeling_options['blade']['n_te_flaps']
        for i in range(n_te_flaps):
            wt_opt['blade.dac_te_flaps.te_flap_start'][i]   = blade['aerodynamic_control']['te_flaps'][i]['span_start']
            wt_opt['blade.dac_te_flaps.te_flap_end'][i]     = blade['aerodynamic_control']['te_flaps'][i]['span_end']
            wt_opt['blade.dac_te_flaps.chord_start'][i]     = blade['aerodynamic_control']['te_flaps'][i]['chord_start']
            wt_opt['blade.dac_te_flaps.delta_max_pos'][i]   = blade['aerodynamic_control']['te_flaps'][i]['delta_max_pos']
            wt_opt['blade.dac_te_flaps.delta_max_neg'][i]   = blade['aerodynamic_control']['te_flaps'][i]['delta_max_neg']

            wt_opt['blade.opt_var.te_flap_ext'] = blade['aerodynamic_control']['te_flaps'][i]['span_end'] - blade['aerodynamic_control']['te_flaps'][i]['span_start']
            wt_opt['blade.opt_var.te_flap_end'] = blade['aerodynamic_control']['te_flaps'][i]['span_end']

            # Checks for consistency
            if blade['aerodynamic_control']['te_flaps'][i]['span_start'] < 0.:
                exit('Error: the start along blade span of the trailing edge flap number ' + str(i) + ' is defined smaller than 0, which corresponds to blade root. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['span_start'] > 1.:
                exit('Error: the start along blade span of the trailing edge flap number ' + str(i) + ' is defined bigger than 1, which corresponds to blade tip. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['span_end'] < 0.:
                exit('Error: the end along blade span of the trailing edge flap number ' + str(i) + ' is defined smaller than 0, which corresponds to blade root. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['span_end'] > 1.:
                exit('Error: the end along blade span of the trailing edge flap number ' + str(i) + ' is defined bigger than 1, which corresponds to blade tip. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['span_start'] == blade['aerodynamic_control']['te_flaps'][i]['span_end']:
                exit('Error: the start and end along blade span of the trailing edge flap number ' + str(i) + ' are defined equal. Please check the yaml input.')
            elif i > 0:
                 if blade['aerodynamic_control']['te_flaps'][i]['span_start'] < blade['aerodynamic_control']['te_flaps'][i-1]['span_end']:
                     exit('Error: the start along blade span of the trailing edge flap number ' + str(i) + ' is smaller than the end of the trailing edge flap number ' + str(i-1) + '. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['chord_start'] < 0.2:
                exit('Error: the start along the chord of the trailing edge flap number ' + str(i) + ' is smaller than 0.2, which is too close to the leading edge. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['chord_start'] > 1.:
                exit('Error: the end along the chord of the trailing edge flap number ' + str(i) + ' is larger than 1., which is beyond the trailing edge. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['delta_max_pos'] > 30. / 180. * np.pi:
                exit('Error: the max positive deflection of the trailing edge flap number ' + str(i) + ' is larger than 30 deg, which is beyond the limits of applicability of this tool. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['delta_max_neg'] < -30. / 180. * np.pi:
                exit('Error: the max negative deflection of the trailing edge flap number ' + str(i) + ' is smaller than -30 deg, which is beyond the limits of applicability of this tool. Please check the yaml input.')
            elif blade['aerodynamic_control']['te_flaps'][i]['delta_max_pos'] < blade['aerodynamic_control']['te_flaps'][i]['delta_max_neg']:
                exit('Error: the max positive deflection of the trailing edge flap number ' + str(i) + ' is smaller than the max negative deflection. Please check the yaml input.')
            else:
                pass

    return wt_opt

def assign_hub_values(wt_opt, hub):

    wt_opt['hub.diameter']    = hub['outer_shape_bem']['diameter']
    wt_opt['hub.cone']        = hub['outer_shape_bem']['cone_angle']
    wt_opt['hub.drag_coeff']  = hub['outer_shape_bem']['drag_coefficient']

    wt_opt['hub.system_mass'] = hub['elastic_properties_mb']['system_mass']
    wt_opt['hub.system_I']    = hub['elastic_properties_mb']['system_inertia']
    wt_opt['hub.system_cm']   = hub['elastic_properties_mb']['system_center_mass']

    return wt_opt

def assign_nacelle_values(wt_opt, assembly, nacelle):

    wt_opt['nacelle.uptilt']            = nacelle['outer_shape_bem']['uptilt_angle']
    wt_opt['nacelle.distance_tt_hub']   = nacelle['outer_shape_bem']['distance_tt_hub']
    wt_opt['nacelle.overhang']          = nacelle['outer_shape_bem']['overhang']


    wt_opt['nacelle.above_yaw_mass']    = nacelle['elastic_properties_mb']['above_yaw_mass']
    wt_opt['nacelle.yaw_mass']          = nacelle['elastic_properties_mb']['yaw_mass']
    wt_opt['nacelle.nacelle_cm']        = nacelle['elastic_properties_mb']['center_mass']
    wt_opt['nacelle.nacelle_I']         = nacelle['elastic_properties_mb']['inertia']
    
    wt_opt['nacelle.gear_ratio']        = nacelle['drivetrain']['gear_ratio']
    wt_opt['nacelle.gearbox_efficiency']    = nacelle['drivetrain']['gearbox_efficiency']
    wt_opt['nacelle.generator_efficiency']  = nacelle['drivetrain']['generator_efficiency']
    if assembly['drivetrain'].upper() != 'DIRECT DRIVE':
        wt_opt['nacelle.shaft_ratio']       = nacelle['drivetrain']['shaft_ratio']
        wt_opt['nacelle.planet_numbers']    = nacelle['drivetrain']['planet_numbers']
        wt_opt['nacelle.shrink_disc_mass']  = nacelle['drivetrain']['shrink_disc_mass']
        wt_opt['nacelle.carrier_mass']      = nacelle['drivetrain']['carrier_mass']
        wt_opt['nacelle.flange_length']     = nacelle['drivetrain']['flange_length']
        wt_opt['nacelle.gearbox_input_xcm'] = nacelle['drivetrain']['gearbox_input_xcm']
        wt_opt['nacelle.hss_input_length']  = nacelle['drivetrain']['hss_input_length']
        wt_opt['nacelle.distance_hub2mb']   = nacelle['drivetrain']['distance_hub2mb']
        wt_opt['nacelle.yaw_motors_number'] = nacelle['drivetrain']['yaw_motors_number']
    else:
        print('DriveSE not yet supports direct drive configurations!!! Please do not trust the nacelle design and the lcoe analysis!')
        wt_opt['nacelle.shaft_ratio']       = 0.1
        wt_opt['nacelle.planet_numbers']    = [3, 3, 1]
        wt_opt['nacelle.shrink_disc_mass']  = 1600
        wt_opt['nacelle.carrier_mass']      = 8000
        wt_opt['nacelle.flange_length']     = 0.5
        wt_opt['nacelle.gearbox_input_xcm'] = 0.1
        wt_opt['nacelle.hss_input_length']  = 1.5
        wt_opt['nacelle.distance_hub2mb']   = 1.912
        wt_opt['nacelle.yaw_motors_number'] = 1.


    return wt_opt

def assign_tower_values(wt_opt, modeling_options, tower):
    # Function to assign values to the openmdao component Tower
    n_height        = modeling_options['tower']['n_height'] # Number of points along tower height
    n_layers        = modeling_options['tower']['n_layers']
    
    svec = np.unique( np.r_[tower['outer_shape_bem']['outer_diameter']['grid'],
                            tower['outer_shape_bem']['reference_axis']['x']['grid'],
                            tower['outer_shape_bem']['reference_axis']['y']['grid'],
                            tower['outer_shape_bem']['reference_axis']['z']['grid']] )
    
    wt_opt['tower.s'] = svec
    wt_opt['tower.diameter']   = np.interp(svec, tower['outer_shape_bem']['outer_diameter']['grid'], tower['outer_shape_bem']['outer_diameter']['values'])
    
    wt_opt['tower.ref_axis'][:,0]  = np.interp(svec, tower['outer_shape_bem']['reference_axis']['x']['grid'], tower['outer_shape_bem']['reference_axis']['x']['values'])
    wt_opt['tower.ref_axis'][:,1]  = np.interp(svec, tower['outer_shape_bem']['reference_axis']['y']['grid'], tower['outer_shape_bem']['reference_axis']['y']['values'])
    wt_opt['tower.ref_axis'][:,2]  = np.interp(svec, tower['outer_shape_bem']['reference_axis']['z']['grid'], tower['outer_shape_bem']['reference_axis']['z']['values'])

    layer_name      = n_layers * ['']
    layer_mat       = n_layers * ['']
    thickness       = np.zeros((n_layers, n_height))
    for i in range(n_layers):
        layer_name[i]  = tower['internal_structure_2d_fem']['layers'][i]['name']
        layer_mat[i]   = tower['internal_structure_2d_fem']['layers'][i]['material']
        thickness[i]   = np.interp(svec, tower['internal_structure_2d_fem']['layers'][i]['thickness']['grid'], tower['internal_structure_2d_fem']['layers'][i]['thickness']['values'])

    wt_opt['tower.layer_name']        = layer_name
    wt_opt['tower.layer_mat']         = layer_mat
    wt_opt['tower.layer_thickness']   = 0.5*(thickness[:,:-1]+thickness[:,1:])
    
    wt_opt['tower.outfitting_factor'] = tower['internal_structure_2d_fem']['outfitting_factor']    
    
    return wt_opt

def assign_monopile_values(wt_opt, modeling_options, monopile):
    # Function to assign values to the openmdao component Monopile
    n_height        = modeling_options['monopile']['n_height'] # Number of points along monopile height
    n_layers        = modeling_options['monopile']['n_layers']
    
    svec = np.unique( np.r_[monopile['outer_shape_bem']['outer_diameter']['grid'],
                            monopile['outer_shape_bem']['reference_axis']['x']['grid'],
                            monopile['outer_shape_bem']['reference_axis']['y']['grid'],
                            monopile['outer_shape_bem']['reference_axis']['z']['grid']] )
    
    wt_opt['monopile.s'] = svec
    wt_opt['monopile.diameter']   = np.interp(svec, monopile['outer_shape_bem']['outer_diameter']['grid'], monopile['outer_shape_bem']['outer_diameter']['values'])
    
    wt_opt['monopile.ref_axis'][:,0]  = np.interp(svec, monopile['outer_shape_bem']['reference_axis']['x']['grid'], monopile['outer_shape_bem']['reference_axis']['x']['values'])
    wt_opt['monopile.ref_axis'][:,1]  = np.interp(svec, monopile['outer_shape_bem']['reference_axis']['y']['grid'], monopile['outer_shape_bem']['reference_axis']['y']['values'])
    wt_opt['monopile.ref_axis'][:,2]  = np.interp(svec, monopile['outer_shape_bem']['reference_axis']['z']['grid'], monopile['outer_shape_bem']['reference_axis']['z']['values'])

    layer_name      = n_layers * ['']
    layer_mat       = n_layers * ['']
    thickness       = np.zeros((n_layers, n_height))
    for i in range(n_layers):
        layer_name[i]  = monopile['internal_structure_2d_fem']['layers'][i]['name']
        layer_mat[i]   = monopile['internal_structure_2d_fem']['layers'][i]['material']
        thickness[i]   = np.interp(svec, monopile['internal_structure_2d_fem']['layers'][i]['thickness']['grid'], monopile['internal_structure_2d_fem']['layers'][i]['thickness']['values'])

    wt_opt['monopile.layer_name']        = layer_name
    wt_opt['monopile.layer_mat']         = layer_mat
    wt_opt['monopile.layer_thickness']   = 0.5*(thickness[:,:-1]+thickness[:,1:])

    wt_opt['monopile.outfitting_factor']          = monopile['internal_structure_2d_fem']['outfitting_factor']    
    wt_opt['monopile.transition_piece_height']    = monopile['transition_piece_height']
    wt_opt['monopile.transition_piece_mass']      = monopile['transition_piece_mass']
    wt_opt['monopile.transition_piece_cost']      = monopile['transition_piece_cost']
    wt_opt['monopile.gravity_foundation_mass']    = monopile['gravity_foundation_mass']
    wt_opt['monopile.suctionpile_depth']          = monopile['suctionpile_depth']
    wt_opt['monopile.suctionpile_depth_diam_ratio']          = monopile['suctionpile_depth_diam_ratio']
    
    return wt_opt

def assign_foundation_values(wt_opt, foundation):

    wt_opt['foundation.height']    = foundation['height']

    return wt_opt

def assign_floating_values(wt_opt, modeling_options, floating):

    dx = (floating['column']['main']['reference_axis']['x']['values'].mean() -
          floating['column']['offset']['reference_axis']['x']['values'].mean() )
    dy = (floating['column']['main']['reference_axis']['y']['values'].mean() -
          floating['column']['offset']['reference_axis']['y']['values'].mean() )
    wt_opt['floating.radius_to_offset_column'] = np.sqrt(dx**2 + dy**2)
        
    wt_opt['floating.number_of_offset_columns'] = len(floating['column']['repeat'])
    for k in range(len(floating['mooring']['nodes'])):
        if floating['mooring']['nodes'][k]['node_type'] == 'vessel':
            wt_opt['fairlead_location'] = floating['mooring']['nodes'][k]['location']['z']
            wt_opt['fairlead_offset_from_shell'] = 1.0 # TODO
            break

    wt_opt['floating.outfitting_cost_rate'] = 20.0 # Lookup material here?
    wt_opt['floating.loading'] = 'hydrostatic' #if modeling_options['floating']['loading']['hydrostatic'] else 

    # Main column
    svec = np.unique( np.r_[floating['column']['main']['outer_shape_bem']['outer_diameter']['grid'],
                            floating['column']['main']['outer_shape_bem']['reference_axis']['x']['grid'],
                            floating['column']['main']['outer_shape_bem']['reference_axis']['y']['grid'],
                            floating['column']['main']['outer_shape_bem']['reference_axis']['z']['grid']] )
    
    wt_opt['floating.main.s'] = svec
    wt_opt['floating.main.diameter']   = np.interp(svec, floating['column']['main']['outer_shape_bem']['outer_diameter']['grid'], floating['column']['main']['outer_shape_bem']['outer_diameter']['values'])
    
    wt_opt['floating.main.ref_axis'][:,0]  = np.interp(svec, floating['column']['main']['outer_shape_bem']['reference_axis']['x']['grid'], floating['column']['main']['outer_shape_bem']['reference_axis']['x']['values'])
    wt_opt['floating.main.ref_axis'][:,1]  = np.interp(svec, floating['column']['main']['outer_shape_bem']['reference_axis']['y']['grid'], floating['column']['main']['outer_shape_bem']['reference_axis']['y']['values'])
    wt_opt['floating.main.ref_axis'][:,2]  = np.interp(svec, floating['column']['main']['outer_shape_bem']['reference_axis']['z']['grid'], floating['column']['main']['outer_shape_bem']['reference_axis']['z']['values'])

    wt_opt['floating.main.outfitting_factor']          = floating['columns']['main']['internal_structure']['outfitting_factor']    
    wt_opt['floating.main.stiffener_web_height']       = floating['columns']['main']['internal_structure']['stiffener_web_height']
    wt_opt['floating.main.stiffener_web_thickness']    = floating['columns']['main']['internal_structure']['stiffener_web_thickness']
    wt_opt['floating.main.stiffener_flange_thickness'] = floating['columns']['main']['internal_structure']['stiffener_flange_thickness']
    wt_opt['floating.main.stiffener_flange_width']     = floating['columns']['main']['internal_structure']['stiffener_flange_width']
    wt_opt['floating.main.stiffener_spacing']          = floating['columns']['main']['internal_structure']['stiffener_spacing']
    wt_opt['floating.main.permanent_ballast_volume']   = floating['columns']['main']['internal_structure']['ballast']['volume']

    wt_opt['floating.main.bulkhead_thickness']         = floating['columns']['main']['internal_structure']['bulkhead']['thickness']['values']
    wt_opt['floating.main.bulkhead_location']          = floating['columns']['main']['internal_structure']['bulkhead']['thickness']['grid']

    layer_name      = n_layers * ['']
    layer_mat       = n_layers * ['']
    thickness       = np.zeros((n_layers, n_height-1))
    for i in range(n_layers):
        layer_name[i]  = floating['columns']['main']['internal_structure']['layers'][i]['name']
        layer_mat[i]   = floating['columns']['main']['internal_structure']['layers'][i]['material']
        thickness[i]   = floating['columns']['main']['internal_structure']['layers'][i]['thickness']['values']

    wt_opt['floating.main.layer_name']     = layer_name
    wt_opt['floating.main.layer_mat']      = layer_mat
    wt_opt['floating.mainlayer_thickness'] = thickness


    # Offset column
    svec = np.unique( np.r_[floating['column']['offset']['outer_shape_bem']['outer_diameter']['grid'],
                            floating['column']['offset']['outer_shape_bem']['reference_axis']['x']['grid'],
                            floating['column']['offset']['outer_shape_bem']['reference_axis']['y']['grid'],
                            floating['column']['offset']['outer_shape_bem']['reference_axis']['z']['grid']] )
    
    wt_opt['floating.offset.s'] = svec
    wt_opt['floating.offset.diameter']   = np.interp(svec, floating['column']['offset']['outer_shape_bem']['outer_diameter']['grid'], floating['column']['offset']['outer_shape_bem']['outer_diameter']['values'])
    
    wt_opt['floating.offset.ref_axis'][:,0]  = np.interp(svec, floating['column']['offset']['outer_shape_bem']['reference_axis']['x']['grid'], floating['column']['offset']['outer_shape_bem']['reference_axis']['x']['values'])
    wt_opt['floating.offset.ref_axis'][:,1]  = np.interp(svec, floating['column']['offset']['outer_shape_bem']['reference_axis']['y']['grid'], floating['column']['offset']['outer_shape_bem']['reference_axis']['y']['values'])
    wt_opt['floating.offset.ref_axis'][:,2]  = np.interp(svec, floating['column']['offset']['outer_shape_bem']['reference_axis']['z']['grid'], floating['column']['offset']['outer_shape_bem']['reference_axis']['z']['values'])

    wt_opt['floating.offset.outfitting_factor']          = floating['columns']['offset']['internal_structure']['outfitting_factor']    
    wt_opt['floating.offset.stiffener_web_height']       = floating['columns']['offset']['internal_structure']['stiffener_web_height']
    wt_opt['floating.offset.stiffener_web_thickness']    = floating['columns']['offset']['internal_structure']['stiffener_web_thickness']
    wt_opt['floating.offset.stiffener_flange_thickness'] = floating['columns']['offset']['internal_structure']['stiffener_flange_thickness']
    wt_opt['floating.offset.stiffener_flange_width']     = floating['columns']['offset']['internal_structure']['stiffener_flange_width']
    wt_opt['floating.offset.stiffener_spacing']          = floating['columns']['offset']['internal_structure']['stiffener_spacing']
    wt_opt['floating.offset.permanent_ballast_volume']   = floating['columns']['offset']['internal_structure']['ballast']['volume']

    wt_opt['floating.offset.bulkhead_thickness']         = floating['columns']['offset']['internal_structure']['bulkhead']['thickness']['values']
    wt_opt['floating.offset.bulkhead_location']          = floating['columns']['offset']['internal_structure']['bulkhead']['thickness']['grid']

    layer_name      = n_layers * ['']
    layer_mat       = n_layers * ['']
    thickness       = np.zeros((n_layers, n_height-1))
    for i in range(n_layers):
        layer_name[i]  = floating['columns']['offset']['internal_structure']['layers'][i]['name']
        layer_mat[i]   = floating['columns']['offset']['internal_structure']['layers'][i]['material']
        thickness[i]   = floating['columns']['offset']['internal_structure']['layers'][i]['thickness']['values']

    wt_opt['floating.offset.layer_name']     = layer_name
    wt_opt['floating.offset.layer_mat']      = layer_mat
    wt_opt['floating.offsetlayer_thickness'] = thickness
    
    for k in range(len(floating['heave_plate'])):
        if floating['heave_plate'][k]['member_name'].lower().find('main') >= 0:
            # TODO: Might need grid/values on OD
            wt_opt['floating.main.buoyancy_tank_diameter'] = floating['heave_plate'][k]['outer_diameter']
            wt_opt['floating.main.buoyancy_tank_height']   = floating['heave_plate'][k]['wall_thickness']
            wt_opt['floating.main.buoyancy_tank_location'] = floating['heave_plate'][k]['grid_value']
            
        elif floating['heave_plate'][k]['member_name'].lower().find('off') >= 0:
            # TODO: Might need grid/values on OD
            wt_opt['floating.offset.buoyancy_tank_diameter'] = floating['heave_plate'][k]['outer_diameter']
            wt_opt['floating.offset.buoyancy_tank_height']   = floating['heave_plate'][k]['wall_thickness']
            wt_opt['floating.offset.buoyancy_tank_location'] = floating['heave_plate'][k]['grid_value']
            
    
    wt_opt['floating.transition_piece_height'] = wt_opt['floating.main.freeboard'] = floating['column']['main']['reference_axis']['z']['values'].max()
    wt_opt['floating.transition_piece_mass']   = floating['transition_piece_mass']

    line_length = np.zeros(len(floating['mooring']['lines']))
    for k in range(line_length.size):
        line_length[k] = floating['mooring']['lines'][k]['unstretched_length']
        line_type      = floating['mooring']['lines'][k]['line_type']
    wt_opt['floating.mooring.mooring_line_length'] = line_length.mean()

    nodes_xyz = np.zeros( (len(floating['mooring']['nodes']), 3) )
    for k in range(len(floating['mooring']['nodes'])):
        nodes_xyz[k,0] = floating['mooring']['nodes'][k]['location']['x']
        nodes_xyz[k,1] = floating['mooring']['nodes'][k]['location']['y']
        nodes_xyz[k,2] = floating['mooring']['nodes'][k]['location']['z']

    anchor_nodes = []
    vessel_nodes = []
    for k in range(len(floating['mooring']['nodes'])):
        if floating['mooring']['nodes'][k]['node_type'] == 'fixed':
            anchor_nodes.append( k )
        elif floating['mooring']['nodes'][k]['node_type'] == 'vessel':
            vessel_nodes.append( k )
    wt_opt['floating.mooring.number_of_mooring_connections'] = len(vessel_nodes)
    wt_opt['floating.mooring.mooring_lines_per_connection']  = len(floating['mooring']['lines']) / len(vessel_nodes)

    center_xyz = nodes_xyz[anchor_nodes, :].mean(axis=0)
    anchor_dist = nodes_xyz[anchor_nodes, :] - center_xyz[np.newaxis,:]
    wt_opt['floating.mooring.anchor_radius'] = np.sqrt( np.sum(anchor_dist**2, axis=1) ).mean()

    for k in range(len(floating['mooring']['line_type'])):
        if floating['mooring']['line_type'][k]['name'] == line_type:
            wt_opt['floating.mooring.mooring_diameter'] = floating['mooring']['line_type'][k]['diameter']
            
    wt_opt['floating.mooring.mooring_type'] = 'CHAIN'
    wt_opt['floating.mooring.anchor_type'] = 'SUCTIONPILE'

    # TODO- These should be set in constraints or analysis options
    wt_opt['floating.mooring.mooring_cost_factor'] = 1.0
    wt_opt['floating.mooring.max_offset']          = 50.0
    wt_opt['floating.mooring.operational_heel']    = 6.0
    wt_opt['floating.mooring.max_survival_heel']   = 10.0
        
    return wt_opt

def assign_control_values(wt_opt, modeling_options, control):
    # Controller parameters
    wt_opt['control.rated_power']   = control['rated_power']
    wt_opt['control.V_in']          = control['Vin']
    wt_opt['control.V_out']         = control['Vout']
    wt_opt['control.minOmega']      = control['minOmega']
    wt_opt['control.maxOmega']      = control['maxOmega']
    wt_opt['control.rated_TSR']     = control['tsr']
    wt_opt['control.rated_pitch']   = control['pitch']
    wt_opt['control.max_TS']        = control['maxTS']
    wt_opt['control.max_pitch_rate']= control['max_pitch_rate']
    wt_opt['control.max_torque_rate']= control['max_torque_rate']
            
    return wt_opt

def assign_configuration_values(wt_opt, assembly):

    wt_opt['configuration.ws_class']          = assembly['turbine_class']
    wt_opt['configuration.turb_class']        = assembly['turbulence_class']
    wt_opt['configuration.gearbox_type']      = assembly['drivetrain']
    wt_opt['configuration.rotor_orientation'] = assembly['rotor_orientation'].lower()
    wt_opt['configuration.n_blades']          = int(assembly['number_of_blades'])

    # Checks for errors
    if int(assembly['number_of_blades']) - assembly['number_of_blades'] != 0:
        print('ERROR: the number of blades must be an integer')

    return wt_opt

def assign_environment_values(wt_opt, environment, offshore):

    wt_opt['env.rho_air']         = environment['air_density']
    wt_opt['env.mu_air']          = environment['air_dyn_viscosity']
    if offshore:
        wt_opt['env.rho_water']       = environment['water_density']
        wt_opt['env.mu_water']        = environment['water_dyn_viscosity']
        wt_opt['env.water_depth']     = environment['water_depth']
        wt_opt['env.hsig_wave']       = environment['significant_wave_height']
        wt_opt['env.Tsig_wave']       = environment['significant_wave_period']
    wt_opt['env.weibull_k']       = environment['weib_shape_parameter']
    wt_opt['env.speed_sound_air'] = environment['air_speed_sound']
    wt_opt['env.shear_exp']       = environment['shear_exp']
    wt_opt['env.G_soil']          = environment['soil_shear_modulus']
    wt_opt['env.nu_soil']         = environment['soil_poisson']

    return wt_opt

def assign_bos_values(wt_opt, bos, offshore):

    wt_opt['bos.plant_turbine_spacing']       = bos['plant_turbine_spacing']
    wt_opt['bos.plant_row_spacing']           = bos['plant_row_spacing']
    wt_opt['bos.commissioning_pct']           = bos['commissioning_pct']
    wt_opt['bos.decommissioning_pct']         = bos['decommissioning_pct']
    wt_opt['bos.distance_to_substation']      = bos['distance_to_substation']
    wt_opt['bos.distance_to_interconnection'] = bos['distance_to_interconnection']
    if offshore:
        wt_opt['bos.site_distance']                     = bos['distance_to_site']
        wt_opt['bos.distance_to_landfall']              = bos['distance_to_landfall']
        wt_opt['bos.port_cost_per_month']               = bos['port_cost_per_month']
        wt_opt['bos.site_auction_price']                = bos['site_auction_price']
        wt_opt['bos.site_assessment_plan_cost']         = bos['site_assessment_plan_cost']
        wt_opt['bos.site_assessment_cost']              = bos['site_assessment_cost']
        wt_opt['bos.construction_operations_plan_cost'] = bos['construction_operations_plan_cost']
        wt_opt['bos.boem_review_cost']                  = bos['boem_review_cost']
        wt_opt['bos.design_install_plan_cost']          = bos['design_install_plan_cost']
    else:
        wt_opt['bos.interconnect_voltage']        = bos['interconnect_voltage']

    return wt_opt

def assign_costs_values(wt_opt, costs):

    wt_opt['costs.turbine_number']      = costs['turbine_number']
    wt_opt['costs.opex_per_kW']         = costs['opex_per_kW']
    wt_opt['costs.bos_per_kW']          = costs['bos_per_kW']
    wt_opt['costs.wake_loss_factor']    = costs['wake_loss_factor']
    wt_opt['costs.fixed_charge_rate']   = costs['fixed_charge_rate']
    if 'offset_tcc_per_kW' in costs:
        wt_opt['costs.offset_tcc_per_kW']   = costs['offset_tcc_per_kW']

    return wt_opt 

def assign_airfoil_values(wt_opt, modeling_options, airfoils):
    # Function to assign values to the openmdao component Airfoils
    
    n_af  = modeling_options['airfoils']['n_af']
    n_aoa = modeling_options['airfoils']['n_aoa']
    aoa   = modeling_options['airfoils']['aoa']
    n_Re  = modeling_options['airfoils']['n_Re']
    n_tab = modeling_options['airfoils']['n_tab']
    n_xy  = modeling_options['airfoils']['n_xy']
    
    name    = n_af * ['']
    ac      = np.zeros(n_af)
    r_thick = np.zeros(n_af)
    Re_all  = []
    for i in range(n_af):
        name[i]     = airfoils[i]['name']
        ac[i]       = airfoils[i]['aerodynamic_center']
        r_thick[i]  = airfoils[i]['relative_thickness']
        for j in range(len(airfoils[i]['polars'])):
            Re_all.append(airfoils[i]['polars'][j]['re'])
    Re = np.array(sorted(np.unique(Re_all)))
    
    cl = np.zeros((n_af, n_aoa, n_Re, n_tab))
    cd = np.zeros((n_af, n_aoa, n_Re, n_tab))
    cm = np.zeros((n_af, n_aoa, n_Re, n_tab))
    
    coord_xy = np.zeros((n_af, n_xy, 2))
    
    # Interp cl-cd-cm along predefined grid of angle of attack
    for i in range(n_af):
        n_Re_i = len(airfoils[i]['polars'])
        Re_j = np.zeros(n_Re_i)
        j_Re = np.zeros(n_Re_i, dtype=int)
        for j in range(n_Re_i):
            Re_j[j] = airfoils[i]['polars'][j]['re']
            j_Re[j] = np.argmin(Re-Re_j)
            cl[i,:,j_Re[j],0] = np.interp(aoa, airfoils[i]['polars'][j]['c_l']['grid'], airfoils[i]['polars'][j]['c_l']['values'])
            cd[i,:,j_Re[j],0] = np.interp(aoa, airfoils[i]['polars'][j]['c_d']['grid'], airfoils[i]['polars'][j]['c_d']['values'])
            cm[i,:,j_Re[j],0] = np.interp(aoa, airfoils[i]['polars'][j]['c_m']['grid'], airfoils[i]['polars'][j]['c_m']['values'])
    
            if abs(cl[i,0,j,0] - cl[i,-1,j,0]) > 1.e-5:
                cl[i,0,j,0] = cl[i,-1,j,0]
                print("WARNING: Airfoil " + name[i] + ' has the lift coefficient at Re ' + str(Re_j) + ' different between + and - pi rad. This is fixed automatically, but please check the input data.')
            if abs(cd[i,0,j,0] - cd[i,-1,j,0]) > 1.e-5:
                cd[i,0,j,0] = cd[i,-1,j,0]
                print("WARNING: Airfoil " + name[i] + ' has the drag coefficient at Re ' + str(Re_j) + ' different between + and - pi rad. This is fixed automatically, but please check the input data.')
            if abs(cm[i,0,j,0] - cm[i,-1,j,0]) > 1.e-5:
                cm[i,0,j,0] = cm[i,-1,j,0]
                print("WARNING: Airfoil " + name[i] + ' has the moment coefficient at Re ' + str(Re_j) + ' different between + and - pi rad. This is fixed automatically, but please check the input data.')
        
        # Re-interpolate cl-cd-cm along the Re dimension if less than n_Re were provided in the input yaml (common condition)
        for k in range(n_aoa):
            cl[i,k,:,0] = np.interp(Re, Re_j, cl[i,k,j_Re,0])
            cd[i,k,:,0] = np.interp(Re, Re_j, cd[i,k,j_Re,0])
            cm[i,k,:,0] = np.interp(Re, Re_j, cm[i,k,j_Re,0])


        points = np.column_stack((airfoils[i]['coordinates']['x'], airfoils[i]['coordinates']['y']))
        # Check that airfoil points are declared from the TE suction side to TE pressure side
        idx_le = np.argmin(points[:,0])
        if np.mean(points[:idx_le,1]) > 0.:
            points = np.flip(points, axis=0)
        
        # Remap points using class AirfoilShape
        af = AirfoilShape(points=points)
        af.redistribute(n_xy, even=False, dLE=True)
        s = af.s
        af_points = af.points
        
        # Add trailing edge point if not defined
        if [1,0] not in af_points.tolist():
            af_points[:,0] -= af_points[np.argmin(af_points[:,0]), 0]
        c = max(af_points[:,0])-min(af_points[:,0])
        af_points[:,:] /= c
        
        coord_xy[i,:,:] = af_points
        
        # Plotting
        # import matplotlib.pyplot as plt
        # plt.plot(af_points[:,0], af_points[:,1], '.')
        # plt.plot(af_points[:,0], af_points[:,1])
        # plt.show()
        
    # Assign to openmdao structure    
    wt_opt['airfoils.aoa']       = aoa
    wt_opt['airfoils.name']      = name
    wt_opt['airfoils.ac']        = ac
    wt_opt['airfoils.r_thick']   = r_thick
    wt_opt['airfoils.Re']        = Re  # Not yet implemented!
    wt_opt['airfoils.cl']        = cl
    wt_opt['airfoils.cd']        = cd
    wt_opt['airfoils.cm']        = cm
    
    wt_opt['airfoils.coord_xy']  = coord_xy
     
    return wt_opt
    
def assign_material_values(wt_opt, modeling_options, materials):
    # Function to assign values to the openmdao component Materials
    
    n_mat = modeling_options['materials']['n_mat']
    
    name        = n_mat * ['']
    orth        = np.zeros(n_mat)
    component_id= -np.ones(n_mat)
    rho         = np.zeros(n_mat)
    sigma_y     = np.zeros(n_mat)
    E           = np.zeros([n_mat, 3])
    G           = np.zeros([n_mat, 3])
    nu          = np.zeros([n_mat, 3])
    Xt          = np.zeros([n_mat, 3])
    Xc          = np.zeros([n_mat, 3])
    rho_fiber   = np.zeros(n_mat)
    rho_area_dry= np.zeros(n_mat)
    fvf         = np.zeros(n_mat)
    fwf         = np.zeros(n_mat)
    ply_t       = np.zeros(n_mat)
    roll_mass   = np.zeros(n_mat)
    unit_cost   = np.zeros(n_mat)
    waste       = np.zeros(n_mat)
    
    for i in range(n_mat):
        name[i]    =  materials[i]['name']
        orth[i]    =  materials[i]['orth']
        rho[i]     =  materials[i]['rho']
        if 'component_id' in materials[i]:
            component_id[i] = materials[i]['component_id']
        if orth[i] == 0:
            if 'E' in materials[i]:
                E[i,:]  = np.ones(3) * materials[i]['E']
            if 'nu' in materials[i]:
                nu[i,:] = np.ones(3) * materials[i]['nu']
            if 'G' in materials[i]:
                G[i,:]  = np.ones(3) * materials[i]['G']
            elif 'nu' in materials[i]:
                G[i,:]  = np.ones(3) * materials[i]['E']/(2*(1+materials[i]['nu'])) # If G is not provided but the material is isotropic and we have E and nu we can just estimate it
                warning_shear_modulus_isotropic = 'WARNING: NO shear modulus, G, was provided for material "%s". The code assumes 2G*(1 + nu) = E, which is only valid for isotropic materials.'%name[i]
                print(warning_shear_modulus_isotropic)
            if 'Xt' in materials[i]:
                Xt[i,:] = np.ones(3) * materials[i]['Xt']
            if 'Xc' in materials[i]:
                Xc[i,:] = np.ones(3) * materials[i]['Xc']
        elif orth[i] == 1:
            E[i,:]  = materials[i]['E']
            G[i,:]  = materials[i]['G']
            nu[i,:] = materials[i]['nu']
            Xt[i,:] = materials[i]['Xt']
            Xc[i,:] = materials[i]['Xc']

        else:
            exit('The flag orth must be set to either 0 or 1. Error in material ' + name[i])
        if 'fiber_density' in materials[i]:
            rho_fiber[i]    = materials[i]['fiber_density']
        if 'area_density_dry' in materials[i]:
            rho_area_dry[i] = materials[i]['area_density_dry']
        if 'fvf' in materials[i]:
            fvf[i] = materials[i]['fvf']
        if 'fwf' in materials[i]:
            fwf[i] = materials[i]['fwf']
        if 'ply_t' in materials[i]:
            ply_t[i] = materials[i]['ply_t']
        if 'roll_mass' in materials[i]:
            roll_mass[i] = materials[i]['roll_mass']
        if 'unit_cost' in materials[i]:
            unit_cost[i] = materials[i]['unit_cost']
        if 'waste' in materials[i]:
            waste[i] = materials[i]['waste']
        if 'Xy' in materials[i]:
            sigma_y[i] =  materials[i]['Xy']

            
    wt_opt['materials.name']     = name
    wt_opt['materials.orth']     = orth
    wt_opt['materials.rho']      = rho
    wt_opt['materials.sigma_y']  = sigma_y
    wt_opt['materials.component_id']= component_id
    wt_opt['materials.E']        = E
    wt_opt['materials.G']        = G
    wt_opt['materials.Xt']       = Xt
    wt_opt['materials.Xc']       = Xc
    wt_opt['materials.nu']       = nu
    wt_opt['materials.rho_fiber']      = rho_fiber
    wt_opt['materials.rho_area_dry']   = rho_area_dry
    wt_opt['materials.fvf_from_yaml']      = fvf
    wt_opt['materials.fwf_from_yaml']      = fwf
    wt_opt['materials.ply_t_from_yaml']    = ply_t
    wt_opt['materials.roll_mass']= roll_mass
    wt_opt['materials.unit_cost']= unit_cost
    wt_opt['materials.waste']    = waste

    return wt_opt

def assign_RNA_values(wt_opt, modeling_options, blade, RNA):

    def _assembleI(I):
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = I[0], I[1], I[2], I[3], I[4], I[5] 
        return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

    def _unassembleI(I):
        return np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])
    
    nd_span     = modeling_options['blade']['nd_span']
    n_span      = len(nd_span)
    ref_axis    = np.zeros((n_span,3))
    ref_axis[:,0]  = np.interp(nd_span, blade['outer_shape_bem']['reference_axis']['x']['grid'], blade['outer_shape_bem']['reference_axis']['x']['values'])
    ref_axis[:,1]  = np.interp(nd_span, blade['outer_shape_bem']['reference_axis']['y']['grid'], blade['outer_shape_bem']['reference_axis']['y']['values'])
    ref_axis[:,2]  = np.interp(nd_span, blade['outer_shape_bem']['reference_axis']['z']['grid'], blade['outer_shape_bem']['reference_axis']['z']['values'])

    blade_length = arc_length(ref_axis)[-1]
    
    rhoA = np.zeros(n_span)
    rhoA2interp = np.zeros(len(blade['elastic_properties_mb']['six_x_six']['inertia_matrix']['grid']))
    for i in range(len(rhoA2interp)):
        rhoA2interp[i] = blade['elastic_properties_mb']['six_x_six']['inertia_matrix']['values'][i][0]
    rhoA = np.interp(nd_span, blade['elastic_properties_mb']['six_x_six']['inertia_matrix']['grid'], rhoA2interp)

    rb   = nd_span * blade_length
    blade_mass = np.trapz(rhoA, rb)
    rR   = rb + wt_opt['hub.diameter'] * 0.5  
    blade_moment_of_inertia = np.trapz(rhoA * rR**2., rR)
    tilt = wt_opt['nacelle.uptilt'] * 180. / np.pi
    n_blades = wt_opt['configuration.n_blades']
    mass_all_blades = n_blades * blade_mass
    Ibeam = n_blades * blade_moment_of_inertia
    Ixx = Ibeam
    Iyy = Ibeam/2.0  # azimuthal average for 2 blades, exact for 3+
    Izz = Ibeam/2.0
    Ixy = 0.0
    Ixz = 0.0
    Iyz = 0.0  # azimuthal average for 2 blades, exact for 3+
    # rotate to yaw c.s.
    I = DirectionVector(Ixx, Iyy, Izz).hubToYaw(tilt)  # because off-diagonal components are all zero
    I_all_blades = np.array([I.x, I.y, I.z, Ixy, Ixz, Iyz])

    nac_mass = wt_opt['nacelle.above_yaw_mass'] + wt_opt['nacelle.yaw_mass']
    rotor_mass = mass_all_blades + wt_opt['hub.system_mass']


    # rna I
    blades_I = _assembleI(I_all_blades)
    hub_I = _assembleI(wt_opt['hub.system_I'])
    nac_I = _assembleI(wt_opt['nacelle.nacelle_I'])
    rotor_I = blades_I + hub_I

    R_hub = wt_opt['hub.system_cm']
    rotor_I_TT = rotor_I + rotor_mass*(np.dot(R_hub, R_hub)*np.eye(3) - np.outer(R_hub, R_hub))

    R_nac = wt_opt['nacelle.nacelle_cm']
    nac_I_TT = nac_I + nac_mass*(np.dot(R_nac, R_nac)*np.eye(3) - np.outer(R_nac, R_nac))


    wt_opt['drivese.rna_mass'] = rotor_mass + nac_mass
    wt_opt['drivese.rna_I_TT'] = _unassembleI(rotor_I_TT + nac_I_TT)
    wt_opt['drivese.rna_cm']   = (rotor_mass*np.array(R_hub) + nac_mass*np.array(R_nac))/(rotor_mass + nac_mass)

    if not RNA == {}:
        if abs(wt_opt['drivese.rna_mass'] - RNA['elastic_properties_mb']['mass']) > 1.e3:
            print('The mass of the RNA system does not match the quantities specified in blade, hub, and nacelle. Please check the input yaml.')
        if abs(sum(wt_opt['drivese.rna_I_TT']) - sum(RNA['elastic_properties_mb']['inertia'])) > 1.e5:
            print('The inertia of the RNA system does not match the quantities specified in blade, hub, and nacelle. Please check the input yaml.')
        if abs(sum(wt_opt['drivese.rna_cm']) - sum(RNA['elastic_properties_mb']['center_mass'])) > 1.e-1:
            print('The center of mass of the RNA system does not match the quantities specified in blade, hub, and nacelle. Please check the input yaml.')


    return wt_opt



if __name__ == "__main__":
    pass
