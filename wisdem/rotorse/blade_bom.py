import numpy as np
import os
import matplotlib.pyplot as plt


class blade_bom(object):

    def __init__(self):

        self.name           = ''
        self.bladeLength    = 0.0
        self.r              = np.asarray([0.0]) # Dimensional blade coordinate
        self.eta            = np.asarray([0.0]) # Non-dimensional blade coordinate
        self.chord          = np.asarray([0.0])
        self.le_location    = np.asarray([0.0])

        self.materials      = []
        self.upperCS        = []
        self.lowerCS        = []
        self.websCS         = []
        self.profile        = []
        
        # Material inputs
        self.density_GF     = 2600.        # [kg/m3]
        self.density_epoxy  = 1150.        # [kg/m3] Mixed density of resin Huntsman LY1564 and hardener Huntsman XP3416     
        material_dict       = {}
        
        # Coatings
        coatings_names = ['(Gelcoat)', 'Gelcoat', 'Paint', 'paint']
        for name in coatings_names:
            material_dict[name]              = {}
            material_dict[name]['component'] = [0]       # Flag to specify where the material is used. 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
            material_dict[name]['waste']     = 25.       # [%] Waste of the material during production
            material_dict[name]['ply_t']     = 0.00051   # [m] Gel coat thickness 0.51 mm
            material_dict[name]['unit_cost'] = 7.23      # [$/kg] 3.28 $/lbs
        
        
        # Sandwich fillers
        filler_names = ['(Balsa)', 'BalsaCore', 'FOAM', 'Foam','(Foam)', 'Balsa', 'medium_density_foam','balsa','foam']
        for name in filler_names:
            material_dict[name]                = {}
            material_dict[name]['component']   = [1]       # Flag to specify where the material is used. 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
            material_dict[name]['waste']       = 20.       # [%] Waste of the material during production
            material_dict[name]['unit_cost']   = 13.       # [$/m2]
        
        material_dict['(Resin)']                = {}
        material_dict['(Resin)']['component']   = [-1]      # Flag to specify where the material is used. -1 - nothing, 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
        material_dict['(Resin)']['waste']       = 20.       # [%] Waste of the material during production
        material_dict['(Resin)']['unit_cost']   = 13.       # [$/m2]
        material_dict['(Resin)']['ply_t']       = 0.0025 # [m] 
        
        # Triaxial fabrics
        triax_names = ['(TriaxFabric)', 'TriaxSkin', 'glassTri', 'glass_triax','(SNLTriax)','triax']
        for name in triax_names:
            material_dict[name]                      = {}
            material_dict[name]['component']         = [2]      # Flag to specify where the material is used. 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
            material_dict[name]['area_density_dry']  = 1.112    # [kg/m2] Unit mass dry fabric
            material_dict[name]['waste']             = 15.      # [%] Waste of the material during production
            material_dict[name]['unit_cost']         = 2.86     # [$/kg]
            material_dict[name]['roll_mass']         = 181.4368 # [kg] 400 lbs - mass of an individual roll
        
        
        # Uniaxial fabrics spar caps
        uniax_names = ['(SparCapMix)', 'UniaxSC', 'glassUD', 'glass_uni','(ELT5500EP3(Uni))','uniax','ud']
        for name in uniax_names:
            material_dict[name]                       = {}
            material_dict[name]['component']          = [4]# Flag to specify where the material is used. 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
            material_dict[name]['area_density_dry']   = 1.858    # [kg/m2] Unit mass dry fabric
            material_dict[name]['waste']              = 5.       # [%] Waste of the material during production
            material_dict[name]['unit_cost']          = 1.87     # [$/kg]
        
        
        # Uniaxial fabrics reinf
        uniax_names = ['UniaxTELEre']
        for name in uniax_names:
            material_dict[name]                       = {}
            material_dict[name]['component']          = [5]# Flag to specify where the material is used. 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
            material_dict[name]['area_density_dry']   = 1.858    # [kg/m2] Unit mass dry fabric
            material_dict[name]['waste']              = 5.       # [%] Waste of the material during production
            material_dict[name]['unit_cost']          = 1.87     # [$/kg]
        
        uniax_names_CF = ['(Newport307)', 'CarbonUD','carbon_uni'] # 50oz Carbon Uni for the spar caps
        for name in uniax_names_CF:
            material_dict[name]                       = {}
            material_dict[name]['component']          = [4]      # Flag to specify where the material is used. 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
            material_dict[name]['area_density_dry']   = 1.000    # [kg/m2] Unit mass dry fabric
            material_dict[name]['waste']              = 5.       # [%] Waste of the material during production
            material_dict[name]['unit_cost']          = 30.00    # [$/kg]
        
        
        
        # Biaxial fabrics
        biax_names = ['(RandomMat)', 'BiaxWebs', 'glassDB','glass_biax','(SaertexEP3(DB))','biax']
        for name in biax_names:
            material_dict[name]                        = {}
            material_dict[name]['component']           = [3]      # Flag to specify where the material is used. 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
            material_dict[name]['area_density_dry']    = 1.112    # [kg/m2] Unit mass dry fabric
            material_dict[name]['waste']               = 15.      # [%] Waste of the material during production
            material_dict[name]['unit_cost']           = 3.00     # [$/kg]
            material_dict[name]['roll_mass']           = 181.4368 # [kg] 400 lbs - mass of an individual roll
        
        
        
        
        
        
        self.material_dict  = material_dict
        
    def extract_specs(self):

        precomp_mat = {}
        
        core_mat_id      = np.zeros(len(self.materials))
        coating_mat_id   = -1
        le_reinf_mat_id  = -1
        te_reinf_mat_id  = -1

        
        for i, mat in enumerate(self.materials):
            precomp_mat[mat.name]             = {}
            precomp_mat[mat.name]['id']       = i + 1
            precomp_mat[mat.name]['name']     = mat.name
            precomp_mat[mat.name]['density']  = mat.rho
            try:
                precomp_mat[mat.name]['unit_cost']= self.material_dict[mat.name]['unit_cost']
                precomp_mat[mat.name]['waste']    = self.material_dict[mat.name]['waste']
                if min(self.material_dict[mat.name]['component']) > 1: # It's a composite
                    precomp_mat[mat.name]['fvf']  = (precomp_mat[mat.name]['density'] - self.density_epoxy) / (self.density_GF - self.density_epoxy) * 100. # [%] Fiber volume fraction
                    precomp_mat[mat.name]['fwf']  = self.density_GF * precomp_mat[mat.name]['fvf'] / 100. / (self.density_epoxy + ((self.density_GF - self.density_epoxy) * precomp_mat[mat.name]['fvf'] / 100.)) * 100.
                    precomp_mat[mat.name]['ply_t']= self.material_dict[mat.name]['area_density_dry'] / precomp_mat[mat.name]['density'] / (precomp_mat[mat.name]['fwf'] / 100.)
                    
                    if min(self.material_dict[mat.name]['component']) > 3: # The material does not need to be cut@station
                        precomp_mat[mat.name]['cut@station'] = 'N'
                    else:
                        precomp_mat[mat.name]['cut@station'] = 'Y'
                        precomp_mat[mat.name]['roll_mass']   = self.material_dict[mat.name]['roll_mass'] 
                else:
                    precomp_mat[mat.name]['fvf']  = 100.
                    precomp_mat[mat.name]['fwf']  = 100.
                    precomp_mat[mat.name]['cut@station'] = 'N'
                    if max(self.material_dict[mat.name]['component']) <= 0:
                        precomp_mat[mat.name]['ply_t']  = self.material_dict[mat.name]['ply_t']
                
                if 0 in self.material_dict[mat.name]['component']:
                    coating_mat_id = precomp_mat[mat.name]['id']        # Assigning the material to the coating
                elif 1 in self.material_dict[mat.name]['component']:    
                    core_mat_id[precomp_mat[mat.name]['id'] - 1]  = 1   # Assigning the material to the core
                elif 2 in self.material_dict[mat.name]['component']:    
                    skin_mat_id    = precomp_mat[mat.name]['id']        # Assigning the material to the shell skin
                elif 3 in self.material_dict[mat.name]['component']:    
                    skinwebs_mat_id= precomp_mat[mat.name]['id']        # Assigning the material to the webs skin 
                elif 4 in self.material_dict[mat.name]['component']:    
                    sc_mat_id      = precomp_mat[mat.name]['id']        # Assigning the material to the spar caps
                elif 5 in self.material_dict[mat.name]['component']:    
                    le_reinf_mat_id= precomp_mat[mat.name]['id']        # Assigning the material to the te reinf  
                    te_reinf_mat_id= precomp_mat[mat.name]['id']        # Assigning the material to the le reinf
            
            except:
                exit('ERROR: The material ' + mat.name + ' does not have its properties fully defined. Please set them in the first lines of blade_bom.py in RotorSE')
        
        
        
        # Width and thickness of the flanges
        blade_specs                                        = {}
        blade_specs['flange_width_inboard_LETE']           = 0.10         # [m] Width of the flanges of the outer surface until 70% of blade span
        blade_specs['flange_span_reduce_LETE']             = 70           # [%] Spanwise position after which flanges are reduced in width
        blade_specs['flange_width_tip_LETE']               = 0.01         # [m] Width of the flanges of the outer surface at blade tip
        blade_specs['flange_width_webs_SW']                = 0.05         # [m] Width of the flanges of the webs
        
        
        
        # ###############################################################################################################################
        # Code
        
        mat_names = precomp_mat.keys()

        
        if self.options['verbosity']:
            print('Number of composite laminates defined: %u' % (len(precomp_mat) - sum(core_mat_id)))
            print('Number of core fillers defined:        %u' % (sum(core_mat_id)))
            print('Total number of materials defined:     %u' % (len(precomp_mat)))
        
        t_layer = np.zeros(len(mat_names))
        density = np.zeros(len(mat_names))
        
        for i, name in enumerate(mat_names):
            if core_mat_id[precomp_mat[name]['id'] - 1] == 0:
                if self.options['verbosity']:
                    print('Composite :' + name)
                t_layer[precomp_mat[name]['id']-1] = precomp_mat[name]['ply_t']
                density[precomp_mat[name]['id']-1] = precomp_mat[name]['density']
            else:
                if self.options['verbosity']:
                    print('Filler    :' + name)
                density[precomp_mat[name]['id']-1] = precomp_mat[name]['density']
        
        # Reconstruct number of plies from laminate thickness and single ply thickness
        composite_rounding = False
        # Upper mold
        for i_section in range(len(self.upperCS)):
            for i_panel in range(len(self.upperCS[i_section].loc)-1):
                for i_mat in range(len(self.upperCS[i_section].n_plies[i_panel])):
                    mat_id      = int(self.upperCS[i_section].mat_idx[i_panel][i_mat])
                    if core_mat_id[mat_id] == 0:
                        n_ply_float = self.upperCS[i_section].t[i_panel][i_mat] / t_layer[mat_id]
                        
                        if self.options['discrete']:
                            self.upperCS[i_section].n_plies[i_panel][i_mat] = round(n_ply_float)                        
                            if composite_rounding == False and n_ply_float != round(n_ply_float):
                                composite_rounding = True
                        else:
                            self.upperCS[i_section].n_plies[i_panel][i_mat] = n_ply_float
                        
        # Lower mold
        for i_section in range(len(self.lowerCS)):
            for i_panel in range(len(self.lowerCS[i_section].loc)-1):
                for i_mat in range(len(self.lowerCS[i_section].n_plies[i_panel])):
                    mat_id      = int(self.lowerCS[i_section].mat_idx[i_panel][i_mat])
                    if core_mat_id[mat_id] == 0:
                        n_ply_float = self.lowerCS[i_section].t[i_panel][i_mat] / t_layer[mat_id]
                        if self.options['discrete']:
                            self.lowerCS[i_section].n_plies[i_panel][i_mat] = round(n_ply_float)
                            if composite_rounding == False and n_ply_float != round(n_ply_float):
                                composite_rounding = True
                        else:
                            self.lowerCS[i_section].n_plies[i_panel][i_mat] = n_ply_float
        
        # Webs
        n_webs = 0
        for i_section in range(len(self.lowerCS)):
            for i_web in range(len(self.websCS[i_section].n_plies)):
                n_webs          = max([n_webs , i_web + 1])
                for i_mat in range(len(self.websCS[i_section].n_plies[i_web])):
                    mat_id      = int(self.websCS[i_section].mat_idx[i_web][i_mat])
                    if core_mat_id[mat_id] == 0:
                        n_ply_float = self.websCS[i_section].t[i_web][i_mat] / t_layer[mat_id]
                        if self.options['discrete']:
                            self.websCS[i_section].n_plies[i_web][i_mat] = round(n_ply_float)
                            if composite_rounding == False and n_ply_float != round(n_ply_float):
                                composite_rounding = True
                        else:
                            self.websCS[i_section].n_plies[i_web][i_mat] = n_ply_float  
        
        if composite_rounding and self.options['show_warnings']:
            print('WARNING: number of composite plies not consistent with the thicknesses specified along the blade. Rounding is performed.')
        
        
        
        
        
        blade_specs['blade_length']             = self.bladeLength
        blade_specs['root_D']                   = self.chord[0]
        blade_specs['max_chord']                = max(self.chord)
        blade_specs['root_preform_length']      = 0.01 * blade_specs['blade_length'] # Currently assumed as 1% of BL
        blade_specs['n_webs']                   = n_webs
        
        
        # Reconstruct total area of the mold and total area per ply
        npts = len(self.r)
        unit_mass_tot   = np.zeros(npts)
        unit_mass_mat   = np.zeros([len(mat_names), npts])
        
        width_sc_lp     = np.zeros(npts)
        width_sc_hp     = np.zeros(npts)
        total_ply_edge  = np.zeros([len(mat_names), npts])
        n_plies_sc_lp   = np.zeros(npts)
        n_plies_sc_hp   = np.zeros(npts)
        
        n_plies_root_lp = np.zeros(len(self.upperCS[0].loc)-1)
        n_plies_root_hp = np.zeros(len(self.lowerCS[0].loc)-1)
        edge_fabric2lay_shell_lp        = np.zeros(npts)
        edge_fabric2lay_shell_hp        = np.zeros(npts)
        edge_fabric2lay_root_preform_lp = np.zeros(npts)
        edge_fabric2lay_root_preform_hp = np.zeros(npts)
        edge_fabric2lay_sc_lp           = np.zeros(npts)
        edge_fabric2lay_sc_hp           = np.zeros(npts)
        
        edge_lp_root    = np.zeros(npts)
        edge_hp_root    = np.zeros(npts)
        
        edge_lpskin_wo_flanges = np.zeros(npts)
        edge_hpskin_wo_flanges = np.zeros(npts)
        
        n_plies_le_lp   = np.zeros(npts)
        n_plies_le_hp   = np.zeros(npts)
        n_plies_te_lp   = np.zeros(npts)
        n_plies_te_hp   = np.zeros(npts)
        
        edge_le_lp      = np.zeros(npts)
        edge_le_hp      = np.zeros(npts)
        edge_te_lp      = np.zeros(npts)
        edge_te_hp      = np.zeros(npts)
        
        edgecore2lay_shell_lp   = np.zeros(npts)
        thick_core_shell_lp     = np.zeros(npts)
        unit_mass_core_shell_lp = np.zeros(npts)
        edgecore2lay_shell_hp   = np.zeros(npts)
        thick_core_shell_hp     = np.zeros(npts)
        unit_mass_core_shell_hp = np.zeros(npts)
        
        width_sc_start_lp       = 0.
        sc_start_section_lp     = 0.
        width_sc_end_lp         = 0.
        sc_end_section_lp       = 0.
        width_sc_start_hp       = 0.
        sc_start_section_hp     = 0.
        width_sc_end_hp         = 0.
        sc_end_section_hp       = 0.
        
        # Distinction between root preform and outer sheel skin
        root_preform_end        = np.argmin(abs(self.r - blade_specs['root_preform_length']))
        matrix_unit_mass        = 0.
        #############################################################################################################################        
        # Low pressure (upper) mold
        for i_section in range(npts):

            for i_panel in range(len(self.upperCS[i_section].loc)-1):
                core_check                      = 0
                edge1                           = np.argmin(abs(self.profile[i_section].x - self.upperCS[i_section].loc[i_panel]))
                edge2                           = np.argmin(abs(self.profile[i_section].x - self.upperCS[i_section].loc[i_panel+1]))
                arc_length                      = np.zeros(len(self.profile[i_section].x))
                
                for i_point in range(edge1 , edge2):                    
                    arc_length[i_point]         = self.chord[i_section] * ((self.profile[i_section].x[i_point+1]-self.profile[i_section].x[i_point])**2+(self.profile[i_section].yu[i_point+1]-self.profile[i_section].yu[i_point])**2)**0.5
                    
                width_panel                         = sum(arc_length)
                edge_lpskin_wo_flanges[i_section]   = edge_lpskin_wo_flanges[i_section] + width_panel
                
                if i_section <= root_preform_end:
                    edge_lp_root[i_section]         = edge_lp_root[i_section] + width_panel
                
                for i_mat in range(len(self.upperCS[i_section].n_plies[i_panel])):
                    mat_id                            = int(self.upperCS[i_section].mat_idx[i_panel][i_mat])
                    
                    
                    # total_ply_edge[mat_id, i_section] = total_ply_edge[mat_id,i_section] + width_panel * self.upperCS[i_section].n_plies[i_panel][i_mat]  # [m]                 
                    unit_mass_tot[i_section]          = unit_mass_tot[i_section] + width_panel * self.upperCS[i_section].t[i_panel][i_mat] * density[mat_id] # [kg/m]
                    unit_mass_mat[mat_id, i_section]  = unit_mass_mat[mat_id, i_section] + width_panel * self.upperCS[i_section].t[i_panel][i_mat] * density[mat_id] # [kg/m]
                    
                    if mat_id == sc_mat_id - 1 and 0 < i_panel < len(self.upperCS[i_section].loc) - 2: # Exclude LE and TE regions, as for some blades (e.g. DTU10MW) they have the same UD as in the spar caps
                        width_sc_lp[i_section]           = width_sc_lp[i_section] + width_panel
                        edge_fabric2lay_sc_lp[i_section] = edge_fabric2lay_sc_lp[i_section] + width_panel * self.upperCS[i_section].n_plies[i_panel][i_mat]   # [m]
                        n_plies_sc_lp[i_section]         = self.upperCS[i_section].n_plies[i_panel][i_mat] # [-]
                    
                    # Compute number of plies at blade root
                    if mat_id == skin_mat_id - 1:
                        if i_section == 0:
                            n_plies_root_lp[i_panel]        = n_plies_root_lp[i_panel] + self.upperCS[i_section].n_plies[i_panel][i_mat] # [-]
                        # Compute ply area    
                        if i_section <= root_preform_end:
                            edge_fabric2lay_shell_lp[i_section]         = edge_fabric2lay_shell_lp[i_section] + 0.5 * (width_panel * self.upperCS[i_section].n_plies[i_panel][i_mat]) # [m]
                            edge_fabric2lay_root_preform_lp[i_section]  = edge_fabric2lay_root_preform_lp[i_section] + 0.5 * (width_panel * self.upperCS[i_section].n_plies[i_panel][i_mat]) # [m]
                        else:
                            edge_fabric2lay_shell_lp[i_section] = edge_fabric2lay_shell_lp[i_section] + width_panel * self.upperCS[i_section].n_plies[i_panel][i_mat]   # [m]
                        
                    # Compute fabric length for the leading edge reinforcement
                    if mat_id == le_reinf_mat_id - 1 and i_panel == 0: # first panel
                        n_plies_le_lp[i_section]         = self.upperCS[i_section].n_plies[i_panel][i_mat]  # [-]
                        edge_le_lp[i_section]            = n_plies_le_lp[i_section] * width_panel           # [m]
                    if mat_id == le_reinf_mat_id - 1 and i_panel == 1 and self.options['show_warnings']:    # second panel    
                        print('WARNING: the leading edge reinforcement on the suction side of section ' + str(i_section) + ' is defined in more than the last panel along the chord. This may not be not realistic.')
                    
                    # Compute fabric length for the trailing edge reinforcement
                    if mat_id == te_reinf_mat_id - 1 and i_panel == len(self.upperCS[i_section].loc) - 2:   # last panel
                        n_plies_te_lp[i_section]         = self.upperCS[i_section].n_plies[i_panel][i_mat]  # [-]
                        edge_te_lp[i_section]            = n_plies_te_lp[i_section] * width_panel           # [m]
                    if mat_id == te_reinf_mat_id - 1 and i_panel == len(self.upperCS[i_section].loc) - 3 and self.options['show_warnings']: # one before last panel  
                        print('WARNING: the trailing edge reinforcement on the suction side of section ' + str(i_section) + ' is defined in more than the last panel along the chord. This may not be not realistic.')
                    
                    # Compute area with sandwich core
                    if core_mat_id[mat_id] == 1:
                        if core_check == 0:
                            edgecore2lay_shell_lp[i_section]    = edgecore2lay_shell_lp[i_section] + width_panel                                                              # [m]
                            thick_core_shell_lp[i_section]      = thick_core_shell_lp[i_section] + width_panel * self.upperCS[i_section].t[i_panel][i_mat]                   # [m2]
                            unit_mass_core_shell_lp[i_section]  = unit_mass_core_shell_lp[i_section] + width_panel * self.upperCS[i_section].t[i_panel][i_mat] * density[mat_id]   # [kg/m]
                            core_check               = 1
                        else:
                            if self.options['show_warnings']:
                                print('WARNING: the blade has multiple layers of sandwich core defined in each panel. This is not supported.')
                        
                if i_panel > 0 and i_section == 0:
                    if n_plies_root_lp[i_panel] != n_plies_root_lp[i_panel - 1] and self.options['show_warnings']:
                        print('WARNING: the blade shows ply drops at the root (eta = 0) on the suction side in the chordwise direction. This is not supported.')
            
                        
            
            if width_sc_start_lp == 0:
                width_sc_start_lp                   = width_sc_lp[i_section]
                sc_start_section_lp                 = i_section
            if width_sc_lp[i_section] != 0: 
                width_sc_end_lp                     = width_sc_lp[i_section]    
                sc_end_section_lp                   = i_section
        
        
        
        
        # Mold dimensions
        blade_specs['LE_length']                    = blade_specs['blade_length']
        blade_specs['TE_length']                    = blade_specs['blade_length']
        blade_specs['skin_perimeter_wo_root']       = 2. * blade_specs['blade_length']
        blade_specs['skin_perimeter_w_root']        = blade_specs['skin_perimeter_wo_root'] + np.pi * blade_specs['root_D']
         # Flanges extend to 70% of blade span at full width and they do taper down towards the tip
        blade_specs['flange_area_LETE']             = blade_specs['skin_perimeter_wo_root'] * blade_specs['flange_span_reduce_LETE'] / 100 * blade_specs['flange_width_inboard_LETE'] + blade_specs['skin_perimeter_wo_root'] * (1 - blade_specs['flange_span_reduce_LETE'] / 100) * (blade_specs['flange_width_inboard_LETE'] - blade_specs['flange_width_tip_LETE'])/2
        blade_specs['area_lpskin_wo_flanges']       = np.trapz(edge_lpskin_wo_flanges, self.r)
        blade_specs['area_lpskin_w_flanges']        = blade_specs['area_lpskin_wo_flanges'] + blade_specs['flange_area_LETE']
        
        # Shell
        blade_specs['fabric2lay_shell_lp']          = np.trapz(edge_fabric2lay_shell_lp, self.r)
        blade_specs['volume_shell_lp']              = blade_specs['fabric2lay_shell_lp'] * t_layer[skin_mat_id-1]
        blade_specs['mass_shell_lp']                = blade_specs['volume_shell_lp'] * density[skin_mat_id-1]
        
        # Root preform
        blade_specs['area_lp_root']                 = np.trapz(edge_lp_root, self.r)
        blade_specs['volume_root_preform_lp']       = np.trapz(edge_fabric2lay_root_preform_lp, self.r) * t_layer[skin_mat_id-1]
        blade_specs['mass_root_preform_lp']         = blade_specs['volume_root_preform_lp'] * density[skin_mat_id-1]
        blade_specs['n_plies_root_lp']              = n_plies_root_lp[0]
        
        # Spar cap
        blade_specs['length_sc_lp']                 = self.r[sc_end_section_lp] - self.r[sc_start_section_lp]           # [m]
        blade_specs['width_sc_start_lp']            = width_sc_start_lp                                                 # [m]
        blade_specs['width_sc_end_lp']              = width_sc_end_lp                                                   # [m]
        blade_specs['area_sc_lp']                   = blade_specs['width_sc_start_lp'] * blade_specs['length_sc_lp']    # [m2]
        blade_specs['fabric2lay_sc_lp']             = np.trapz(n_plies_sc_lp, self.r)                                   # [m]
        blade_specs['volume2lay_sc_lp']             = np.trapz(edge_fabric2lay_sc_lp,self.r) * t_layer[sc_mat_id-1]     # [m3]
        blade_specs['mass_sc_lp']                   = blade_specs['volume2lay_sc_lp'] * density[sc_mat_id-1]            # [kg]
        if width_sc_start_lp != width_sc_end_lp and self.options['show_warnings']:
            print('WARNING: the spar cap on the low pressure side is not straight. This is currently not supported by the code. Straight spar cap is assumed.')
        
        # LE reinforcement
        blade_specs['fabric2lay_le_reinf_lp']       = np.trapz(n_plies_le_lp, self.r)
        if le_reinf_mat_id > -1:
            blade_specs['volume_le_reinf_lp']       = np.trapz(edge_le_lp, self.r) * t_layer[le_reinf_mat_id-1]
        else:
            blade_specs['volume_le_reinf_lp']       = 0.
        blade_specs['mass_le_reinf_lp']             = blade_specs['volume_le_reinf_lp'] * density[le_reinf_mat_id-1]
        
        # TE reinforcement
        blade_specs['fabric2lay_te_reinf_lp']       = np.trapz(n_plies_te_lp, self.r)
        if te_reinf_mat_id > -1:
            blade_specs['volume_te_reinf_lp']       = np.trapz(edge_te_lp, self.r) * t_layer[te_reinf_mat_id-1]
        else:
            blade_specs['volume_te_reinf_lp']       = 0.
        blade_specs['mass_te_reinf_lp']             = blade_specs['volume_te_reinf_lp'] * density[te_reinf_mat_id-1]
        
        # Core        
        blade_specs['areacore2lay_shell_lp']        = np.trapz(edgecore2lay_shell_lp, self.r)
        blade_specs['volume_core_shell_lp']         = np.trapz(thick_core_shell_lp, self.r)
        blade_specs['mass_core_shell_lp']           = np.trapz(unit_mass_core_shell_lp, self.r)
        
        
        #############################################################################################################################
        # High pressure (lower) mold
        for i_section in range(npts):

            for i_panel in range(len(self.lowerCS[i_section].loc)-1):
                core_check                      = 0
                edge1                           = np.argmin(abs(self.profile[i_section].x - self.lowerCS[i_section].loc[i_panel]))
                edge2                           = np.argmin(abs(self.profile[i_section].x - self.lowerCS[i_section].loc[i_panel+1]))
                arc_length                      = np.zeros(len(self.profile[i_section].x))
                
                for i_point in range(edge1 , edge2):                    
                    arc_length[i_point]         = self.chord[i_section] * ((self.profile[i_section].x[i_point+1]-self.profile[i_section].x[i_point])**2+(self.profile[i_section].yl[i_point+1]-self.profile[i_section].yl[i_point])**2)**0.5
                    
                width_panel                     = sum(arc_length)
                edge_hpskin_wo_flanges[i_section] = edge_hpskin_wo_flanges[i_section] + width_panel
                
                if i_section <= root_preform_end:
                    edge_hp_root[i_section]     = edge_hp_root[i_section] + width_panel
                
                for i_mat in range(len(self.lowerCS[i_section].n_plies[i_panel])):
                    mat_id                            = int(self.lowerCS[i_section].mat_idx[i_panel][i_mat])
                    # total_ply_edge[mat_id, i_section] = total_ply_edge[mat_id, i_section] + width_panel * self.lowerCS[i_section].n_plies[i_panel][i_mat]            # [m]                 
                    unit_mass_tot[i_section]          = unit_mass_tot[i_section] + width_panel * self.lowerCS[i_section].t[i_panel][i_mat] * density[mat_id] # [kg/m]
                    unit_mass_mat[mat_id, i_section]  = unit_mass_mat[mat_id, i_section] + width_panel * self.lowerCS[i_section].t[i_panel][i_mat] * density[mat_id] # [kg/m]

                    if mat_id == sc_mat_id - 1 and 0 < i_panel < len(self.lowerCS[i_section].loc) - 2: # Exclude LE and TE regions, as for some blades (e.g. DTU10MW) they have the same UD as in the spar caps
                        width_sc_hp[i_section]           = width_sc_hp[i_section] + width_panel
                        edge_fabric2lay_sc_hp[i_section] = edge_fabric2lay_sc_hp[i_section] + width_panel * self.lowerCS[i_section].n_plies[i_panel][i_mat]   # [m]
                        n_plies_sc_hp[i_section]         = self.lowerCS[i_section].n_plies[i_panel][i_mat] # [-]
                        
                    
                    # Compute number of plies at blade root
                    if mat_id == skin_mat_id - 1:
                        if i_section == 0:
                            n_plies_root_hp[i_panel]        = n_plies_root_hp[i_panel] + self.lowerCS[i_section].n_plies[i_panel][i_mat] # [-]
                        # Compute ply area    
                        if i_section <= root_preform_end:
                            edge_fabric2lay_shell_hp[i_section]         = edge_fabric2lay_shell_hp[i_section] + 0.5 * (width_panel * self.lowerCS[i_section].n_plies[i_panel][i_mat]) # [m]
                            edge_fabric2lay_root_preform_hp[i_section]  = edge_fabric2lay_root_preform_hp[i_section] + 0.5 * (width_panel * self.lowerCS[i_section].n_plies[i_panel][i_mat]) # [m]
                        else:
                            edge_fabric2lay_shell_hp[i_section]         = edge_fabric2lay_shell_hp[i_section] + width_panel * self.lowerCS[i_section].n_plies[i_panel][i_mat]   # [m]
                        
                    # Compute fabric length for the leading edge reinforcement
                    if mat_id == le_reinf_mat_id - 1 and i_panel == 0: # first panel
                        n_plies_le_hp[i_section]         = self.lowerCS[i_section].n_plies[i_panel][i_mat]  # [-]
                        edge_le_hp[i_section]            = n_plies_le_hp[i_section] * width_panel           # [m]
                    if mat_id == le_reinf_mat_id - 1 and i_panel == 1 and self.options['show_warnings']:    # second panel    
                        print('WARNING: the leading edge reinforcement on the pressure side of section ' + str(i_section) + ' is defined in more than the last panel along the chord. This may not be not realistic.')
                    
                    # Compute fabric length for the trailing edge reinforcement
                    if mat_id == te_reinf_mat_id - 1 and i_panel == len(self.lowerCS[i_section].loc) - 2:   # last panel
                        n_plies_te_hp[i_section]         = self.lowerCS[i_section].n_plies[i_panel][i_mat]  # [-]
                        edge_te_hp[i_section]            = n_plies_te_hp[i_section] * width_panel           # [m]
                    if mat_id == te_reinf_mat_id - 1 and i_panel == len(self.lowerCS[i_section].loc) - 3 and self.options['show_warnings']: # one before last panel  
                        print('WARNING: the trailing edge reinforcement on the pressure side of section ' + str(i_section) + ' is defined in more than the last panel along the chord. This may not be not realistic.')
                    
                    # Compute area with sandwich core
                    if core_mat_id[mat_id] == 1:
                        if core_check == 0:
                            edgecore2lay_shell_hp[i_section]    = edgecore2lay_shell_hp[i_section] + width_panel                                                              # [m]
                            thick_core_shell_hp[i_section]      = thick_core_shell_hp[i_section] + width_panel * self.lowerCS[i_section].t[i_panel][i_mat]                   # [m2]
                            unit_mass_core_shell_hp[i_section]  = unit_mass_core_shell_hp[i_section] + width_panel * self.lowerCS[i_section].t[i_panel][i_mat] * density[mat_id]   # [kg/m]
                            core_check               = 1
                        else:
                            if self.options['show_warnings']:
                                print('WARNING: the blade has multiple layers of sandwich core defined in each panel. This is not supported.')
                        
                if i_panel > 0 and i_section == 0:
                    if n_plies_root_hp[i_panel] != n_plies_root_hp[i_panel - 1] and self.options['show_warnings']:
                        print('WARNING: the blade shows ply drops at the root (eta = 0) on the pressure side in the chordwise direction. This is not supported.')
            
                        
            
            if width_sc_start_hp == 0:
                width_sc_start_hp                   = width_sc_hp[i_section]
                sc_start_section_hp                 = i_section
            if width_sc_hp[i_section] != 0: 
                width_sc_end_hp                     = width_sc_hp[i_section]    
                sc_end_section_hp                   = i_section
        
        # Mold
        blade_specs['area_hpskin_wo_flanges']       = np.trapz(edge_hpskin_wo_flanges, self.r)
        blade_specs['area_hpskin_w_flanges']        = blade_specs['area_hpskin_wo_flanges'] + blade_specs['flange_area_LETE']
        
        # Shell
        blade_specs['fabric2lay_shell_hp']          = np.trapz(edge_fabric2lay_shell_hp, self.r)
        blade_specs['volume_shell_hp']              = blade_specs['fabric2lay_shell_hp'] * t_layer[skin_mat_id-1]
        blade_specs['mass_shell_hp']                = blade_specs['volume_shell_hp'] * density[skin_mat_id-1]
        
        # Root preform
        blade_specs['area_hp_root']                 = np.trapz(edge_hp_root, self.r)
        blade_specs['volume_root_preform_hp']       = np.trapz(edge_fabric2lay_root_preform_hp, self.r) * t_layer[skin_mat_id-1]
        blade_specs['mass_root_preform_hp']         = blade_specs['volume_root_preform_hp'] * density[skin_mat_id-1]
        blade_specs['n_plies_root_hp']              = n_plies_root_hp[0]
        
        # Spar cap
        blade_specs['length_sc_hp']                 = self.r[sc_end_section_hp] - self.r[sc_start_section_hp]           # [m]
        blade_specs['width_sc_start_hp']            = width_sc_start_hp                                                 # [m]
        blade_specs['width_sc_end_hp']              = width_sc_end_hp                                                   # [m]
        blade_specs['area_sc_hp']                   = blade_specs['width_sc_start_hp'] * blade_specs['length_sc_hp']    # [m2]
        blade_specs['fabric2lay_sc_hp']             = np.trapz(n_plies_sc_hp, self.r)                                   # [m]
        blade_specs['volume2lay_sc_hp']             = np.trapz(edge_fabric2lay_sc_hp,self.r) * t_layer[sc_mat_id-1]     # [m3]
        blade_specs['mass_sc_hp']                   = blade_specs['volume2lay_sc_hp'] * density[sc_mat_id-1]            # [kg]
        if width_sc_start_hp != width_sc_end_hp and self.options['show_warnings']:
            print('WARNING: the spar cap on the high pressure side is not straight. This is currently not supported by the code. Straight spar cap is assumed.')
        
        # LE reinforcement
        blade_specs['fabric2lay_le_reinf_hp']       = np.trapz(n_plies_le_hp, self.r)
        if le_reinf_mat_id > -1:
            blade_specs['volume_le_reinf_hp']       = np.trapz(edge_le_hp, self.r) * t_layer[le_reinf_mat_id-1]
        else:
            blade_specs['volume_le_reinf_hp']       = 0.
        blade_specs['mass_le_reinf_hp']             = blade_specs['volume_le_reinf_hp'] * density[le_reinf_mat_id-1]
        
        # TE reinforcement
        blade_specs['fabric2lay_te_reinf_hp']       = np.trapz(n_plies_te_hp, self.r)
        if te_reinf_mat_id > -1:
            blade_specs['volume_te_reinf_hp']       = np.trapz(edge_te_hp, self.r) * t_layer[te_reinf_mat_id-1]
        else:
            blade_specs['volume_te_reinf_hp']       = 0.
        blade_specs['mass_te_reinf_hp']             = blade_specs['volume_te_reinf_hp'] * density[te_reinf_mat_id-1]
        
        # Core        
        blade_specs['areacore2lay_shell_hp']        = np.trapz(edgecore2lay_shell_hp, self.r)
        blade_specs['volume_core_shell_hp']         = np.trapz(thick_core_shell_hp, self.r)
        blade_specs['mass_core_shell_hp']           = np.trapz(unit_mass_core_shell_hp, self.r)
        
        
        #############################################################################################################################        
        # Shear webs
        area_webs           = np.zeros(n_webs)
        height_webs_start   = np.zeros(n_webs)
        height_webs_end     = np.zeros(n_webs)
        webs_start_section  = np.zeros(n_webs)
        webs_end_section    = np.zeros(n_webs)
        web_height          = np.zeros([n_webs, npts])
        edgecore2lay_webs   = np.zeros([n_webs, npts])
        thick_core_webs     = np.zeros([n_webs, npts])
        unit_mass_core_webs = np.zeros([n_webs, npts])
        fabric2lay_webs     = np.zeros([n_webs, npts])
        volumeskin2lay_webs = np.zeros([n_webs, npts])
        
        for i_section in range(npts):
            for i_web in range(len(self.websCS[i_section].n_plies)):
                index_x                            = np.argmin(abs(self.profile[i_section].x - self.websCS[i_section].loc[i_web]))
                web_height[i_web, i_section]       = self.chord[i_section] * (self.profile[i_section].yu[index_x]-self.profile[i_section].yl[index_x])

                if height_webs_start[i_web] == 0:
                    height_webs_start[i_web]        = web_height[i_web, i_section]
                    webs_start_section[i_web]       = i_section
                if web_height[i_web, i_section] != 0:
                    height_webs_end[i_web]          = web_height[i_web, i_section]
                    webs_end_section[i_web]         = i_section
                
                for i_mat in range(len(self.websCS[i_section].n_plies[i_web])):
                    mat_id                              = int(self.websCS[i_section].mat_idx[i_web][i_mat])
                    
                    unit_mass_tot[i_section]          = unit_mass_tot[i_section] + web_height[i_web, i_section] * self.websCS[i_section].t[i_web][i_mat] * density[mat_id] # [kg/m]
                    unit_mass_mat[mat_id, i_section]  = unit_mass_mat[mat_id, i_section] + web_height[i_web, i_section] * self.websCS[i_section].t[i_web][i_mat] * density[mat_id] # [kg/m]
                    
                    # Compute area with sandwich core
                    if core_mat_id[mat_id] == 1:
                        edgecore2lay_webs[i_web, i_section]    = edgecore2lay_webs[i_web, i_section] + web_height[i_web, i_section]                                                              # [m]
                        thick_core_webs[i_web, i_section]      = thick_core_webs[i_web, i_section] + web_height[i_web, i_section] * self.lowerCS[i_section].t[i_panel][i_mat]                   # [m2]
                        unit_mass_core_webs[i_web, i_section]  = unit_mass_core_webs[i_web, i_section] + web_height[i_web, i_section] * self.lowerCS[i_section].t[i_panel][i_mat] * density[mat_id]   # [kg/m]
                    
                    # Compute area with sandwich skin
                    if mat_id == skinwebs_mat_id - 1:
                        fabric2lay_webs[i_web, i_section]      = fabric2lay_webs[i_web, i_section] + web_height[i_web, i_section] * self.websCS[i_section].n_plies[i_web][i_mat]                        # [m]
                        volumeskin2lay_webs[i_web, i_section]  = volumeskin2lay_webs[i_web, i_section] + web_height[i_web, i_section] * self.websCS[i_section].n_plies[i_web][i_mat] * t_layer[mat_id]  # [m2]
                   
                    
        
        #############################################################################################################################        
        # Summary
        blade_specs['height_webs_start']   = height_webs_start
        blade_specs['height_webs_end']     = height_webs_end
        blade_specs['length_webs']         = np.zeros(n_webs)          
        blade_specs['volume_core_webs']    = np.zeros(n_webs)     
        blade_specs['mass_core_webs']      = np.zeros(n_webs)       
        blade_specs['area_webs_wo_flanges']= np.zeros(n_webs) 
        blade_specs['area_webs_w_core']    = np.zeros(n_webs)     
        blade_specs['fabric2lay_webs']     = np.zeros(n_webs)      
        blade_specs['volumeskin2lay_webs'] = np.zeros(n_webs)  
        blade_specs['area_webs_w_flanges'] = np.zeros(n_webs)  
        blade_specs['mass_webs']           = np.zeros(n_webs)
        
        for i_web in range(n_webs):
            blade_specs['length_webs'][i_web]          = self.r[int(webs_end_section[i_web])] - self.r[int(webs_start_section[i_web])]
            blade_specs['volume_core_webs'][i_web]     = np.trapz(thick_core_webs[i_web,:],self.r)
            blade_specs['mass_core_webs'][i_web]       = np.trapz(unit_mass_core_webs[i_web,:],self.r)
            blade_specs['area_webs_wo_flanges'][i_web] = np.trapz(web_height[i_web,:], self.r)
            blade_specs['area_webs_w_core'][i_web]     = np.trapz(edgecore2lay_webs[i_web,:], self.r)
            blade_specs['fabric2lay_webs'][i_web]      = np.trapz(fabric2lay_webs[i_web,:], self.r)
            blade_specs['volumeskin2lay_webs'][i_web]  = np.trapz(volumeskin2lay_webs[i_web,:], self.r)
            blade_specs['mass_webs'][i_web]            = blade_specs['volumeskin2lay_webs'][i_web] * density[skinwebs_mat_id-1]
            blade_specs['area_webs_w_flanges'][i_web]  = blade_specs['area_webs_wo_flanges'][i_web] + 2 * blade_specs['length_webs'][i_web] * blade_specs['flange_width_webs_SW']

        
        
        mass_per_comp               = np.trapz(unit_mass_mat, self.r)
        
        blade_specs['blade_mass']   = np.trapz(unit_mass_tot, self.r)

        blade_specs['matrix_total_mass_wo_waste'] = 0.
        
        for name in mat_names:
            precomp_mat[name]['total_mass_wo_waste']     = mass_per_comp[precomp_mat[name]['id']-1] * precomp_mat[name]['fwf'] / 100.
            precomp_mat[name]['total_mass_w_waste']      = precomp_mat[name]['total_mass_wo_waste'] * (1 + precomp_mat[name]['waste']/100.)
            
            blade_specs['matrix_total_mass_wo_waste'] = blade_specs['matrix_total_mass_wo_waste'] + precomp_mat[name]['total_mass_wo_waste'] / (precomp_mat[name]['fwf']/100.) * (1 - precomp_mat[name]['fwf']/100.)
            
            precomp_mat[name]['total_cost_wo_waste']     = precomp_mat[name]['total_mass_wo_waste'] *  precomp_mat[name]['unit_cost']
            precomp_mat[name]['total_cost_w_waste']      = precomp_mat[name]['total_mass_w_waste']  * precomp_mat[name]['unit_cost']  
                        
            if core_mat_id[precomp_mat[name]['id'] -1] == 0:
            
                precomp_mat[name]['total_volume_wo_waste']   = mass_per_comp[precomp_mat[name]['id']-1] / precomp_mat[name]['density'] * precomp_mat[name]['fvf']
                precomp_mat[name]['total_volume_w_waste']    = precomp_mat[name]['total_volume_wo_waste'] * (1 + precomp_mat[name]['waste']/100.)
                
                precomp_mat[name]['total_ply_area_wo_waste'] = mass_per_comp[precomp_mat[name]['id']-1] / precomp_mat[name]['density'] / precomp_mat[name]['ply_t']
                precomp_mat[name]['total_ply_area_w_waste']  = precomp_mat[name]['total_ply_area_wo_waste'] * (1 + precomp_mat[name]['waste']/100.)
            else:
                precomp_mat[name]['total_ply_area_wo_waste'] = blade_specs['areacore2lay_shell_lp'] + blade_specs['areacore2lay_shell_hp'] + sum(blade_specs['area_webs_w_core'])
                precomp_mat[name]['total_ply_area_w_waste']  = precomp_mat[name]['total_ply_area_wo_waste'] * (1. + precomp_mat[name]['waste']/100.)
                
                precomp_mat[name]['total_volume_wo_waste']   = blade_specs['volume_core_shell_lp'] + blade_specs['volume_core_shell_hp'] + sum(blade_specs['volume_core_webs'])
                precomp_mat[name]['total_volume_w_waste']    = precomp_mat[name]['total_volume_wo_waste'] * (1. + precomp_mat[name]['waste']/100.)

            if self.options['verbosity']:
                print(name)
                print('surface no waste  %.2f m2  \t \t ---  \t \t volume no waste   %.2f m3' % (precomp_mat[name]['total_ply_area_wo_waste'] , precomp_mat[name]['total_volume_wo_waste']))
                print('mass no waste   %.2f kg  \t \t ---  \t \t mass with waste %.2f kg' % (precomp_mat[name]['total_mass_wo_waste'] , precomp_mat[name]['total_mass_w_waste']))
                print('cost no waste   %.2f $   \t \t ---  \t \t cost with waste   %.2f \n' % (precomp_mat[name]['total_cost_wo_waste'] , precomp_mat[name]['total_cost_w_waste']))
        
        
        return blade_specs, precomp_mat
        
        
    def compute_matrix_bonding(self, blade_specs, precomp_mat):
        
        # Resin and hardener
        matrix                                             = {}
        matrix['resin_unit_cost']                          = 3.63         # $/kg
        matrix['hardener_unit_cost']                       = 3.63         # $/kg
        matrix['mix_ratio']                                = 0.3          # Mix ratio by mass
        # Bonding   
        bonding                                            = {}
        bonding['flange_adhesive_squeezed']                = 20           # [%] Extra width of the adhesive once squeezed
        bonding['line_thick_LETE']                         = 0.006        # [m] Thickness of adhesive line along LE and TE
        bonding['line_thick_SW']                           = 0.008        # [m] Thickness of adhesive line along webs
        # Adhesive material properties
        bonding['adhesive_mix_ratio_mass']                 = 0.185        # Adhesive (Plexus) Mix ratio by mass
        bonding['adhesive_mix_ratio_volume']               = 0.1          # Adhesive (Plexus) Mix ratio by volume
        bonding['PartA_unit_cost']                         = 9.00         # [$/kg] Unit cost - 49.90 $/ga
        bonding['PartB_unit_cost']                         = 9.00         # [$/kg] Unit cost - 49.90 $/ga
        bonding['PartA_density']                           = 928.7        # [kg/m3] 7.75 lbs/gal
        bonding['PartB_density']                           = 1713.52      # [kg/m3] 14.30 lbs/gal
        
        if self.options['verbosity']:
            print('\n################################\nBOM - Resin, hardener and adhesive:')
        
        
        
        matrix['resin_total_mass']           = blade_specs['matrix_total_mass_wo_waste'] / (1 + matrix['mix_ratio'])
        matrix['hardener_total_mass']        = blade_specs['matrix_total_mass_wo_waste'] / (1 + (1/matrix['mix_ratio']))
        matrix['resin_total_cost']           = matrix['resin_total_mass'] * matrix['resin_unit_cost']
        matrix['hardener_total_cost']        = matrix['hardener_total_mass'] * matrix['hardener_unit_cost']
        
        if self.options['verbosity']:
            print('Resin    mass %.2f kg\t \t --- \t \t cost %.2f $' % (matrix['resin_total_mass'] , matrix['resin_total_cost']))
            print('Hardener mass %.2f kg\t \t --- \t \t cost %.2f $' % (matrix['hardener_total_mass'] , matrix['hardener_total_cost'])) 
        
        
        bonding_lines_vol                    = np.zeros(1 + blade_specs['n_webs'])
        # Volume of the leading and trailing edge bonding line
        bonding_lines_vol[0]                 = bonding['line_thick_LETE'] * blade_specs['flange_area_LETE'] *(1 + bonding['flange_adhesive_squeezed']/100.)
        
        for i in range(blade_specs['n_webs']):
            bonding_lines_vol[i + 1]         = 2 * bonding['line_thick_SW'] * blade_specs['length_webs'][i] * blade_specs['flange_width_webs_SW'] * (1 + bonding['flange_adhesive_squeezed']/100.)
        
        bonding['adhesive_total_volume']     = sum(bonding_lines_vol)
        bonding['adhesive_density']          = (bonding['PartA_density'] + (bonding['PartB_density'] + bonding['adhesive_mix_ratio_volume'])) / (1 + bonding['adhesive_mix_ratio_volume'])
        bonding['adhesive_total_mass']       = bonding['adhesive_total_volume'] * bonding['adhesive_density']
        bonding['PartA_total_mass']          = bonding['adhesive_total_mass'] / (1 + bonding['adhesive_mix_ratio_mass'])
        bonding['PartB_total_mass']          = bonding['adhesive_total_mass'] / (1 + (1 / bonding['adhesive_mix_ratio_mass']))
        bonding['PartA_total_volume']        = bonding['adhesive_total_volume'] / (1 + bonding['adhesive_mix_ratio_volume'])
        bonding['PartB_total_volume']        = bonding['adhesive_total_volume'] / (1 + (1 / bonding['adhesive_mix_ratio_volume']))
        bonding['PartA_total_cost']          = bonding['PartA_total_mass'] * bonding['PartA_unit_cost']
        bonding['PartB_total_cost']          = bonding['PartB_total_mass'] * bonding['PartB_unit_cost']
        
        return matrix, bonding
        
        
    def compute_metallic_parts(self, blade_specs):
        
        # Hub connection and lightning protection   
        metallic_parts                                     = {}
        metallic_parts['t_bolt_unit_cost']                 = 25.          # Cost of one t-bolt [$]
        metallic_parts['t_bolt_unit_mass']                 = 2.5          # Mass of one t-bolt [kg]
        metallic_parts['t_bolt_spacing']                   = 0.15         # Spacing of t-bolts [m]
        metallic_parts['barrel_nut_unit_cost']             = 12.          # Cost of one barrel nut [$]
        metallic_parts['barrel_nut_unit_mass']             = 1.9          # Mass of one barrel nut [kg]
        metallic_parts['LPS_unit_mass']                    = 1.00         # [kg/m] - Linear scaling based on the weight of 150 lbs for the 61.5 m NREL 5MW blade
        metallic_parts['LPS_unit_cost']                    = 40.00        # [$/m]  - Linear scaling based on the cost of 2500$ for the 61.5 m NREL 5MW blade
        
        # Number of root bolts is scaled linearly with the root circumference preserving spacing between bolts[-]
        if self.options['discrete']:
            metallic_parts['n_bolts']         = round(np.pi * blade_specs['root_D'] / metallic_parts['t_bolt_spacing']) 
        else:
            metallic_parts['n_bolts']         = np.pi * blade_specs['root_D'] / metallic_parts['t_bolt_spacing']
        
        # Hub connection and lightning protection system
        if self.options['verbosity']:
            print('\n###############################\nBOM - Hub connection and lightning protection system:')
        metallic_parts['bolts_cost']         = metallic_parts['n_bolts'] * metallic_parts['t_bolt_unit_cost']
        metallic_parts['nuts_cost']          = metallic_parts['n_bolts'] * metallic_parts['barrel_nut_unit_cost']
        if self.options['verbosity']:
            print('T-bolts     cost %.2f $ \t mass %.2f kg' % (metallic_parts['bolts_cost'], metallic_parts['n_bolts'] * metallic_parts['t_bolt_unit_mass']))
            print('Barrel nuts cost %.2f $ \t mass %.2f kg' % (metallic_parts['nuts_cost'], metallic_parts['n_bolts'] * metallic_parts['barrel_nut_unit_mass']))
        mid_span_station                     = np.argmin(abs(self.eta - 0.5))
        metallic_parts['LPS_mass']           = metallic_parts['LPS_unit_mass'] * (blade_specs['blade_length'] + self.chord[mid_span_station])

        metallic_parts['LPS_cost']           = metallic_parts['LPS_unit_cost'] * (blade_specs['blade_length'] + self.chord[mid_span_station]) 
        if self.options['verbosity']:
            print('Lightning PS mass %.2f kg\t \t --- \t \t cost %.2f $' % (metallic_parts['LPS_mass'] , metallic_parts['LPS_cost']))
        
        metallic_parts['tot_mass']           = metallic_parts['LPS_mass'] + metallic_parts['n_bolts'] * (metallic_parts['t_bolt_unit_mass'] + metallic_parts['barrel_nut_unit_mass'])
       
        
        
        return metallic_parts
        
        
    def compute_consumables(self, blade_specs):
        
        # Consumables
        if self.options['verbosity']:
            print('\n################################\nBOM - Consumables:')
        consumables                                          = {}
        # # LE Erosion Tape
        # consumables['LE_tape']                               = {}
        # consumables['LE_tape']['unit_length']                = 250. # [m] Roll length
        # consumables['LE_tape']['unit_cost']                  = 576. # [$/roll]
        # consumables['LE_tape']['waste']                      = 5.   # [%]
        # consumables['LE_tape']['units_per_blade']            = blade_specs['LE_length'] / consumables['LE_tape']['unit_length'] # Rolls per blade
        # consumables['LE_tape']['total_cost_wo_waste']        = consumables['LE_tape']['units_per_blade'] * consumables['LE_tape']['unit_cost']
        # consumables['LE_tape']['total_cost_w_waste']         = consumables['LE_tape']['total_cost_wo_waste'] * (1 + consumables['LE_tape']['waste']/100)
        # if self.options['verbosity']:
            # print('LE erosion tape cost %.2f $\t \t --- \t \t cost with waste %.2f $' % (consumables['LE_tape']['total_cost_wo_waste'] , consumables['LE_tape']['total_cost_w_waste']))
        # Peel Ply		
        consumables['peel_ply']                              = {}
        consumables['peel_ply']['unit_cost']                 = 1.94  # [$/m2] 0.18 $/sqft
        consumables['peel_ply']['waste']                     = 15.   # [%]
        consumables['peel_ply']['total_cost_wo_waste']       = (sum(blade_specs['area_webs_w_flanges']) + blade_specs['area_lpskin_w_flanges'] + blade_specs['area_hpskin_w_flanges'] + blade_specs['area_sc_lp'] + blade_specs['area_sc_hp'] + blade_specs['area_lp_root'] + blade_specs['area_hp_root']) * consumables['peel_ply']['unit_cost']
        consumables['peel_ply']['total_cost_w_waste']        = consumables['peel_ply']['total_cost_wo_waste'] * (1 + consumables['peel_ply']['waste']/100)
        if self.options['verbosity']:
            print('Peel ply cost %.2f $\t \t \t --- \t \t cost with waste %.2f $' % (consumables['peel_ply']['total_cost_wo_waste'] , consumables['peel_ply']['total_cost_w_waste']))
        # Non-Sanding Peel Ply	
        consumables['ns_peel_ply']                           = {}
        consumables['ns_peel_ply']['unit_cost']              = 1.67      # [$/m2] 0.15 $/sqft
        consumables['ns_peel_ply']['waste']                  = 10.       # [%]
        consumables['ns_peel_ply']['unit_width']             = 0.127     # [m] Roll width
        consumables['ns_peel_ply']['total_cost_wo_waste']    = consumables['ns_peel_ply']['unit_width'] * 2 * (blade_specs['TE_length'] + blade_specs['LE_length'] + sum(blade_specs['length_webs'])) * consumables['ns_peel_ply']['unit_cost']     
        consumables['ns_peel_ply']['total_cost_w_waste']     = consumables['ns_peel_ply']['total_cost_wo_waste'] * (1 + consumables['ns_peel_ply']['waste']/100)
        if self.options['verbosity']:
            print('Non-sand peel ply cost %.2f $\t \t --- \t \t cost with waste %.2f $' % (consumables['ns_peel_ply']['total_cost_wo_waste'] , consumables['ns_peel_ply']['total_cost_w_waste']))
        # Chopped Strand
        consumables['chopped_strand']                        = {}
        consumables['chopped_strand']['unit_cost']           = 2.16      # [$/kg] 0.98 $/lbs
        consumables['chopped_strand']['mass_length']         = 0.037     # [kg/m] 0.025 lb/ft
        consumables['chopped_strand']['waste']               = 5.        # [%]
        consumables['chopped_strand']['total_cost_wo_waste'] = consumables['chopped_strand']['mass_length'] * blade_specs['blade_length'] * consumables['chopped_strand']['unit_cost']
        consumables['chopped_strand']['total_cost_w_waste']  = consumables['chopped_strand']['total_cost_wo_waste'] * (1 + consumables['chopped_strand']['waste']/100)      
        if self.options['verbosity']:
            print('Chopped strand cost %.2f $\t \t --- \t \t cost with waste %.2f $' % (consumables['chopped_strand']['total_cost_wo_waste'] , consumables['chopped_strand']['total_cost_w_waste']))
        # 3M77 Adhesive, Bulk
        consumables['adhesive_bulk']                         = {}
        consumables['adhesive_bulk']['unit_cost']            = 10566.9   # [$/m3] 40 $/ga
        consumables['adhesive_bulk']['volume_area']          = 3.06e-5   # [m3/m2] 0.00075 ga/sf
        consumables['adhesive_bulk']['waste']                = 5.        # [%]
        consumables['adhesive_bulk']['total_cost_wo_waste']  = consumables['adhesive_bulk']['volume_area'] * (sum(blade_specs['area_webs_w_flanges']) + blade_specs['area_lpskin_w_flanges'] + blade_specs['area_hpskin_w_flanges'] + blade_specs['area_sc_lp'] + blade_specs['area_sc_hp'] + blade_specs['area_lp_root'] + blade_specs['area_hp_root']) * consumables['adhesive_bulk']['unit_cost']
        consumables['adhesive_bulk']['total_cost_w_waste']   = consumables['adhesive_bulk']['total_cost_wo_waste'] * (1 + consumables['adhesive_bulk']['waste']/100)      
        if self.options['verbosity']:
            print('Adhesive, bulk cost %.2f $\t \t --- \t \t cost with waste %.2f $' % (consumables['adhesive_bulk']['total_cost_wo_waste'] , consumables['adhesive_bulk']['total_cost_w_waste']))
        # 3M77 Adhesive, Cans
        consumables['adhesive_cans']                         = {}
        consumables['adhesive_cans']['unit_cost']            = 6.65      # [$]
        consumables['adhesive_cans']['waste']                = 5.        # [%]
        consumables['adhesive_cans']['units_area']           = 0.022     # [each/m2] 0.002 each/sf
        consumables['adhesive_cans']['units_blade']          = consumables['adhesive_cans']['units_area'] * (sum(blade_specs['area_webs_w_flanges']) + blade_specs['area_lpskin_w_flanges'] + blade_specs['area_hpskin_w_flanges'] + blade_specs['area_sc_lp'] + blade_specs['area_sc_hp'] + blade_specs['area_lp_root'] + blade_specs['area_hp_root'])
        consumables['adhesive_cans']['total_cost_wo_waste']  = consumables['adhesive_cans']['units_blade'] * consumables['adhesive_cans']['unit_cost']
        consumables['adhesive_cans']['total_cost_w_waste']   = consumables['adhesive_cans']['total_cost_wo_waste'] * (1 + consumables['adhesive_cans']['waste']/100)      
        if self.options['verbosity']:
            print('Adhesive, cans cost %.2f $\t \t --- \t \t cost with waste %.2f $' % (consumables['adhesive_cans']['total_cost_wo_waste'] , consumables['adhesive_cans']['total_cost_w_waste']))
        # Mold Release
        consumables['release_agent']                         = {}
        consumables['release_agent']['unit_cost']            = 15691.82  # [$/m3] - 59.40 $/gal
        consumables['release_agent']['waste']                = 5.        # [%]
        consumables['release_agent']['volume_area']          = 2.57e-5   # [m3/m2] 0.00063 ga/sf
        consumables['release_agent']['total_cost_wo_waste']  = consumables['release_agent']['volume_area'] * (sum(blade_specs['area_webs_w_flanges']) + blade_specs['area_lpskin_w_flanges'] + blade_specs['area_hpskin_w_flanges'] + blade_specs['area_sc_lp'] + blade_specs['area_sc_hp'] + blade_specs['area_lp_root'] + blade_specs['area_hp_root']) * consumables['release_agent']['unit_cost']
        consumables['release_agent']['total_cost_w_waste']   = consumables['release_agent']['total_cost_wo_waste'] * (1 + consumables['release_agent']['waste']/100)      
        if self.options['verbosity']:
            print('Mold release agent cost %.2f $ \t --- \t \t cost with waste %.2f $' % (consumables['release_agent']['total_cost_wo_waste'] , consumables['release_agent']['total_cost_w_waste']))
        # Flow Medium
        consumables['flow_medium']                           = {}
        consumables['flow_medium']['unit_cost']              = 0.646     # [$/m2] 0.06 $/sqft
        consumables['flow_medium']['waste']                  = 15.       # [%]
        consumables['flow_medium']['coverage']               = 70.       # [%]
        consumables['flow_medium']['total_cost_wo_waste']    = (sum(blade_specs['area_webs_w_flanges']) + blade_specs['area_lpskin_w_flanges'] + blade_specs['area_hpskin_w_flanges'] + blade_specs['area_sc_lp'] + blade_specs['area_sc_hp'] + blade_specs['area_lp_root'] + blade_specs['area_hp_root']) * consumables['flow_medium']['coverage'] / 100 * consumables['flow_medium']['unit_cost']
        consumables['flow_medium']['total_cost_w_waste']     = consumables['flow_medium']['total_cost_wo_waste'] * (1 + consumables['flow_medium']['waste']/100)
        if self.options['verbosity']:
            print('Flow medium cost %.2f $\t \t --- \t \t cost with waste %.2f $' % (consumables['flow_medium']['total_cost_wo_waste'] , consumables['flow_medium']['total_cost_w_waste']))
        # tubing - 3/8"
        consumables['tubing3/8']                             = {}
        consumables['tubing3/8']['unit_cost']                = 0.23      # [$/m] 0.07 $/ft
        consumables['tubing3/8']['waste']                    = 10.       # [%]
        consumables['tubing3/8']['length_per_length_blade']  = 5         # [m/m]
        consumables['tubing3/8']['length']                   = consumables['tubing3/8']['length_per_length_blade'] * blade_specs['blade_length']
        consumables['tubing3/8']['total_cost_wo_waste']      = consumables['tubing3/8']['length'] * consumables['tubing3/8']['unit_cost']
        consumables['tubing3/8']['total_cost_w_waste']       = consumables['tubing3/8']['total_cost_wo_waste'] * (1 + consumables['tubing3/8']['waste']/100)
        if self.options['verbosity']:
            print('Tubing 3/8" cost %.2f $\t \t --- \t \t cost with waste %.2f $' % (consumables['tubing3/8']['total_cost_wo_waste'] , consumables['tubing3/8']['total_cost_w_waste']))
        # tubing - 1/2"		
        consumables['tubing1/2']                             = {}
        consumables['tubing1/2']['unit_cost']                = 0.23      # [$/m] 0.07 $/ft
        consumables['tubing1/2']['waste']                    = 10.       # [%]
        consumables['tubing1/2']['length_per_length_blade']  = 5         # [m/m]
        consumables['tubing1/2']['length']                   = consumables['tubing1/2']['length_per_length_blade'] * blade_specs['blade_length']
        consumables['tubing1/2']['total_cost_wo_waste']      = consumables['tubing1/2']['length'] * consumables['tubing1/2']['unit_cost']
        consumables['tubing1/2']['total_cost_w_waste']       = consumables['tubing1/2']['total_cost_wo_waste'] * (1 + consumables['tubing1/2']['waste']/100)
        if self.options['verbosity']:
            print('Tubing 1/2" cost %.2f $\t \t --- \t \t cost with waste %.2f $' % (consumables['tubing1/2']['total_cost_wo_waste'] , consumables['tubing1/2']['total_cost_w_waste']))
        # tubing - 5/8"
        consumables['tubing5/8']                             = {}
        consumables['tubing5/8']['unit_cost']                = 0.49      # [$/m] 0.15 $/ft
        consumables['tubing5/8']['waste']                    = 10.       # [%]
        consumables['tubing5/8']['length_per_length_blade']  = 5         # [m/m]
        consumables['tubing5/8']['length']                   = consumables['tubing5/8']['length_per_length_blade'] * blade_specs['blade_length']
        consumables['tubing5/8']['total_cost_wo_waste']      = consumables['tubing5/8']['length'] * consumables['tubing5/8']['unit_cost']
        consumables['tubing5/8']['total_cost_w_waste']       = consumables['tubing5/8']['total_cost_wo_waste'] * (1 + consumables['tubing5/8']['waste']/100)
        if self.options['verbosity']:
            print('Tubing 5/8" cost %.2f $\t \t --- \t \t cost with waste %.2f $' % (consumables['tubing5/8']['total_cost_wo_waste'] , consumables['tubing5/8']['total_cost_w_waste']))
        # tubing - 3/4"		
        consumables['tubing3/4']                             = {}
        consumables['tubing3/4']['unit_cost']                = 0.62      # [$/m] 0.19 $/ft
        consumables['tubing3/4']['waste']                    = 10.       # [%]
        consumables['tubing3/4']['length_per_length_blade']  = 5         # [m/m]
        consumables['tubing3/4']['length']                   = consumables['tubing3/4']['length_per_length_blade'] * blade_specs['blade_length']
        consumables['tubing3/4']['total_cost_wo_waste']      = consumables['tubing3/4']['length'] * consumables['tubing3/4']['unit_cost']
        consumables['tubing3/4']['total_cost_w_waste']       = consumables['tubing3/4']['total_cost_wo_waste'] * (1 + consumables['tubing3/4']['waste']/100)
        if self.options['verbosity']:
            print('Tubing 3/4" cost %.2f $\t \t --- \t \t cost with waste %.2f $' % (consumables['tubing3/4']['total_cost_wo_waste'] , consumables['tubing3/4']['total_cost_w_waste']))
        # tubing - 7/8"	
        consumables['tubing7/8']                             = {}
        consumables['tubing7/8']['unit_cost']                = 0.49      # [$/m] 0.15 $/ft
        consumables['tubing7/8']['waste']                    = 10.       # [%]
        consumables['tubing7/8']['length_per_length_blade']  = 5         # [m/m]
        consumables['tubing7/8']['length']                   = consumables['tubing7/8']['length_per_length_blade'] * blade_specs['blade_length']
        consumables['tubing7/8']['total_cost_wo_waste']      = consumables['tubing7/8']['length'] * consumables['tubing7/8']['unit_cost']
        consumables['tubing7/8']['total_cost_w_waste']       = consumables['tubing7/8']['total_cost_wo_waste'] * (1 + consumables['tubing7/8']['waste']/100)
        if self.options['verbosity']:
            print('Tubing 7/8" cost %.2f $\t \t --- \t \t cost with waste %.2f $' % (consumables['tubing7/8']['total_cost_wo_waste'] , consumables['tubing7/8']['total_cost_w_waste']))        
        # Silicon flange tape
        consumables['tacky_tape']                            = {}
        consumables['tacky_tape']['unit_length']             = 3.5   # [m/roll]
        consumables['tacky_tape']['unit_cost']               = 21.11 # [$/roll]
        consumables['tacky_tape']['waste']                   = 5.    # [%]
        consumables['tacky_tape']['units_per_blade']         = (10. * blade_specs['blade_length']) / consumables['tacky_tape']['unit_length'] # [-]
        consumables['tacky_tape']['total_cost_wo_waste']     = consumables['tacky_tape']['units_per_blade'] * consumables['tacky_tape']['unit_cost']
        consumables['tacky_tape']['total_cost_w_waste']      = consumables['tacky_tape']['total_cost_wo_waste'] * (1 + consumables['tacky_tape']['waste']/100)
        if self.options['verbosity']:
            print('Tacky tape cost %.2f $\t \t --- \t \t cost with waste %.2f $' % (consumables['tacky_tape']['total_cost_wo_waste'] , consumables['tacky_tape']['total_cost_w_waste']))
        # 2" masking tape		
        consumables['masking_tape']                          = {}
        consumables['masking_tape']['unit_cost']             = 5.50    # [$/roll]
        consumables['masking_tape']['waste']                 = 10.     # [%]
        consumables['masking_tape']['roll_per_length']       = 0.328   # [roll/m]
        consumables['masking_tape']['units_per_blade']       = blade_specs['blade_length'] * consumables['masking_tape']['roll_per_length'] # [-]
        consumables['masking_tape']['total_cost_wo_waste']   = consumables['masking_tape']['units_per_blade'] * consumables['masking_tape']['unit_cost']
        consumables['masking_tape']['total_cost_w_waste']    = consumables['masking_tape']['total_cost_wo_waste'] * (1 + consumables['masking_tape']['waste']/100)
        if self.options['verbosity']:
            print('Masking tape cost %.2f $\t \t --- \t \t cost with waste %.2f $' % (consumables['masking_tape']['total_cost_wo_waste'] , consumables['masking_tape']['total_cost_w_waste']))
        # Chop Fiber
        consumables['chop_fiber']                            = {}
        consumables['chop_fiber']['unit_cost']               = 6.19      # [$/kg] 2.81 $/lbs
        consumables['chop_fiber']['mass_area']               = 9.76e-3   # [kg/m2] 0.002 lb/sf
        consumables['chop_fiber']['waste']                   = 10.       # [%]
        consumables['chop_fiber']['total_cost_wo_waste']     = consumables['chop_fiber']['mass_area'] * (blade_specs['area_lpskin_wo_flanges'] + blade_specs['area_hpskin_wo_flanges']) * consumables['chop_fiber']['unit_cost']
        consumables['chop_fiber']['total_cost_w_waste']      = consumables['chop_fiber']['total_cost_wo_waste'] * (1 + consumables['chop_fiber']['waste']/100)      
        if self.options['verbosity']:
            print('Chopped fiber cost %.2f $\t \t --- \t \t cost with waste %.2f $' % (consumables['chop_fiber']['total_cost_wo_waste'] , consumables['chop_fiber']['total_cost_w_waste']))
        # White Lightning
        consumables['white_lightning']                       = {}
        consumables['white_lightning']['unit_cost']          = 3006.278  # [$/m3] - 11.38 $/gal
        consumables['white_lightning']['waste']              = 10.       # [%]
        consumables['white_lightning']['volume_area']        = 2.04e-5   # [m3/m2] 0.0005 ga/sf
        consumables['white_lightning']['total_cost_wo_waste']= consumables['white_lightning']['volume_area'] * (blade_specs['area_lpskin_wo_flanges'] + blade_specs['area_hpskin_wo_flanges']) * consumables['white_lightning']['unit_cost']
        consumables['white_lightning']['total_cost_w_waste'] = consumables['white_lightning']['total_cost_wo_waste'] * (1 + consumables['white_lightning']['waste']/100)      
        if self.options['verbosity']:
            print('White lightning cost %.2f $ \t \t --- \t \t cost with waste %.2f $' % (consumables['white_lightning']['total_cost_wo_waste'] , consumables['white_lightning']['total_cost_w_waste'])) 
        # Hardener
        consumables['hardener']                              = {}
        consumables['hardener']['unit_cost']                 = 1.65      # [$/tube]
        consumables['hardener']['waste']                     = 10.       # [%]
        consumables['hardener']['units_area']                = 0.012     # [each/m2] 0.0011 tube/sf
        consumables['hardener']['units_blade']               = consumables['hardener']['units_area'] * (blade_specs['area_lpskin_wo_flanges'] + blade_specs['area_hpskin_wo_flanges'])
        consumables['hardener']['total_cost_wo_waste']       = consumables['hardener']['units_blade'] * consumables['hardener']['unit_cost']
        consumables['hardener']['total_cost_w_waste']        = consumables['hardener']['total_cost_wo_waste'] * (1 + consumables['hardener']['waste']/100)
        if self.options['verbosity']:
            print('Hardener tubes %.2f $\t \t \t --- \t \t cost with waste %.2f $' % (consumables['hardener']['total_cost_wo_waste'] , consumables['hardener']['total_cost_w_waste']))
        # Putty
        consumables['putty']                                 = {}
        consumables['putty']['unit_cost']                    = 6.00      # [$/kg] 
        consumables['putty']['mass_area']                    = 0.0244    # [kg/m2]
        consumables['putty']['waste']                        = 10.       # [%]
        consumables['putty']['total_cost_wo_waste']          = consumables['putty']['mass_area'] * (blade_specs['area_lpskin_wo_flanges'] + blade_specs['area_hpskin_wo_flanges']) * consumables['putty']['unit_cost']
        consumables['putty']['total_cost_w_waste']           = consumables['putty']['total_cost_wo_waste'] * (1 + consumables['putty']['waste']/100)      
        if self.options['verbosity']:
            print('Putty cost %.2f $\t \t \t --- \t \t cost with waste %.2f $' % (consumables['putty']['total_cost_wo_waste'] , consumables['putty']['total_cost_w_waste']))
        # Putty Catalyst		
        consumables['catalyst']                              = {}
        consumables['catalyst']['unit_cost']                 = 7.89      # [$/kg]  3.58 $/lbs
        consumables['catalyst']['mass_area']                 = 4.88e-3   # [kg/m2] 0.001 lb/sf
        consumables['catalyst']['waste']                     = 10.       # [%]
        consumables['catalyst']['total_cost_wo_waste']       = consumables['catalyst']['mass_area'] * (blade_specs['area_lpskin_wo_flanges'] + blade_specs['area_hpskin_wo_flanges']) * consumables['catalyst']['unit_cost']
        consumables['catalyst']['total_cost_w_waste']        = consumables['catalyst']['total_cost_wo_waste'] * (1 + consumables['catalyst']['waste']/100)      
        if self.options['verbosity']:
            print('Catalyst cost %.2f $\t \t \t --- \t \t cost with waste %.2f $' % (consumables['catalyst']['total_cost_wo_waste'] , consumables['catalyst']['total_cost_w_waste']))
        
        
        return consumables
    
    
    
    def compute_bom(self, blade_specs, precomp_mat, matrix, bonding, metallic_parts, consumables):
        
        # Composites, core and coating
        if self.options['verbosity']:
            print('#################################\nBOM - Composite fabrics, sandwich core and coating:')
        total_mat_cost_wo_waste = 0.
        total_mat_cost_w_waste  = 0.
        
        if self.options['tex_table']:
            tex_table_file = open('tex_tables.txt','w') 
            tex_table_file.write('\\begin{table}[htpb]\n')
            tex_table_file.write('\\caption{Total composite, core and coating costs of the %s blade.}\n' % self.name)
            tex_table_file.write('\\label{table:%s_1}\n' % self.name)
            tex_table_file.write('\\centering\n')
            tex_table_file.write('\\begin{tabular}{l c}\n')
            tex_table_file.write('\\toprule\n')
            tex_table_file.write('Material & Cost with waste [\\$] \\\\ \n')
            tex_table_file.write('\\midrule\n')              
        
        mat_names = precomp_mat.keys()
        for name in mat_names:
            total_mat_cost_wo_waste = total_mat_cost_wo_waste + precomp_mat[name]['total_cost_wo_waste']
            total_mat_cost_w_waste  = total_mat_cost_w_waste + precomp_mat[name]['total_cost_w_waste']
        
            if self.options['tex_table']:
                tex_table_file.write('%s & %.2f \\\\ \n' % (name , precomp_mat[name]['total_cost_w_waste']))

        if self.options['tex_table']:
            tex_table_file.write('\\textbf{Total} & \\textbf{%.2f} \\\\ \n' % total_mat_cost_w_waste)          
            tex_table_file.write('\\bottomrule\n')
            tex_table_file.write('\\end{tabular}\n')
            tex_table_file.write('\\end{table}\n')
            tex_table_file.close()
        if self.options['verbosity']:    
            print('\n TOTAL COMPOSITE, CORE AND COATING COSTS')
            print('Cost without waste %.2f $\t \t --- \t \t cost with waste %.2f $' % (total_mat_cost_wo_waste , total_mat_cost_w_waste))
        
        blade_specs['composite_cost'] = total_mat_cost_w_waste
        
        
        # Resin, hardener and bonding
        if self.options['verbosity']:
            print('Adhesive pA mass %.2f kg\t \t --- \t \t cost %.2f $' % (bonding['PartA_total_mass'] , bonding['PartA_total_cost']))
            print('Adhesive pB mass %.2f kg\t \t --- \t \t cost %.2f $' % (bonding['PartB_total_mass'] , bonding['PartB_total_cost'])) 
            print('\n TOTAL RESIN, HARDENER AND BONDING COSTS')
        total_resin_hard_bond_cost  = matrix['resin_total_cost'] + matrix['hardener_total_cost'] + bonding['PartA_total_cost'] + bonding['PartB_total_cost']
        if self.options['verbosity']:
            print('Cost %.2f $' % (total_resin_hard_bond_cost))
        
        if self.options['tex_table']:
            tex_table_file = open('tex_tables.txt','a') 
            tex_table_file.write('\n\n\n\\begin{table}[htpb]\n')
            tex_table_file.write('\\caption{Total resin, hardener and bonding costs of the %s blade.}\n' % self.name)
            tex_table_file.write('\\label{table:%s_2}\n' % self.name)
            tex_table_file.write('\\centering\n')
            tex_table_file.write('\\begin{tabular}{l c}\n')
            tex_table_file.write('\\toprule\n')
            tex_table_file.write('Material & Cost with waste [\\$] \\\\ \n')
            tex_table_file.write('\\midrule\n')
            tex_table_file.write('Resin & %.2f \\\\ \n' % matrix['resin_total_cost'])
            tex_table_file.write('Hardener & %.2f \\\\ \n' % matrix['hardener_total_cost'])
            tex_table_file.write('Adhesive Part A & %.2f \\\\ \n' % bonding['PartA_total_cost'])
            tex_table_file.write('Adhesive Part B & %.2f \\\\ \n' % bonding['PartB_total_cost'])
            tex_table_file.write('\\textbf{Total} & \\textbf{%.2f} \\\\ \n' % total_resin_hard_bond_cost)          
            tex_table_file.write('\\bottomrule\n')
            tex_table_file.write('\\end{tabular}\n')
            tex_table_file.write('\\end{table}\n')
            tex_table_file.close()
        
        
        # Metallic parts
        total_metallic_parts_cost = metallic_parts['bolts_cost'] + metallic_parts['nuts_cost'] + metallic_parts['LPS_cost']
        if self.options['verbosity']:
            print('\n TOTAL METALLIC PARTS COSTS')
            print('Cost %.2f $' % (total_metallic_parts_cost))
        
        
        if self.options['tex_table']:
            tex_table_file = open('tex_tables.txt','a') 
            tex_table_file.write('\n\n\n\\begin{table}[htpb]\n')
            tex_table_file.write('\\caption{Total metallic parts costs of the %s blade.}\n' % self.name)
            tex_table_file.write('\\label{table:%s_3}\n' % self.name)
            tex_table_file.write('\\centering\n')
            tex_table_file.write('\\begin{tabular}{l c}\n')
            tex_table_file.write('\\toprule\n')
            tex_table_file.write('Material & Cost [\\$] \\\\ \n')
            tex_table_file.write('\\midrule\n')
            tex_table_file.write('Bolts & %.2f \\\\ \n' % metallic_parts['bolts_cost'])
            tex_table_file.write('Barrel nuts & %.2f \\\\ \n' % metallic_parts['nuts_cost'])
            tex_table_file.write('LPS & %.2f \\\\ \n' % metallic_parts['LPS_cost'])
            tex_table_file.write('\\textbf{Total} & \\textbf{%.2f} \\\\ \n' % total_metallic_parts_cost)          
            tex_table_file.write('\\bottomrule\n')
            tex_table_file.write('\\end{tabular}\n')
            tex_table_file.write('\\end{table}\n')
            tex_table_file.close()
        
        
        # Consumables
        name_consumables                    = consumables.keys()
        total_consumable_cost_wo_waste      = 0.
        total_consumable_cost_w_waste       = 0.
        consumable_cost_w_waste             = []
        for name in name_consumables:
            total_consumable_cost_wo_waste  = total_consumable_cost_wo_waste + consumables[name]['total_cost_wo_waste']
            total_consumable_cost_w_waste   = total_consumable_cost_w_waste + consumables[name]['total_cost_w_waste']
            consumable_cost_w_waste.append(consumables[name]['total_cost_w_waste']) 
        
        if self.options['verbosity']:
            print('\n TOTAL CONSUMABLE COSTS')
            print('Cost without waste %.2f $\t \t --- \t \t cost with waste %.2f $' % (total_consumable_cost_wo_waste , total_consumable_cost_w_waste))
    
    
        total_blade_mat_cost_w_waste  =   total_mat_cost_w_waste + total_resin_hard_bond_cost + total_metallic_parts_cost + total_consumable_cost_w_waste
        blade_mass                    =   blade_specs['blade_mass']
        blade_mass_with_metallic      =   blade_mass + metallic_parts['tot_mass']
        
        if self.options['verbosity']:
            print('\n TOTAL BLADE MASS')
            print('Mass without waste %.2f kg\t \t --- \t \t mass with metallic parts %.2f kg' % (blade_mass , blade_mass_with_metallic))

               
        if self.options['tex_table']:
            tex_table_file = open('tex_tables.txt','a') 
            tex_table_file.write('\n\n\n\\begin{table}[htpb]\n')
            tex_table_file.write('\\caption{BoM of the %s blade.}\n' % self.name)
            tex_table_file.write('\\label{table:%s_4}\n' % self.name)
            tex_table_file.write('\\centering\n')
            tex_table_file.write('\\begin{tabular}{l c}\n')
            tex_table_file.write('\\toprule\n')
            tex_table_file.write('  & Cost with waste [\\$]\\\\ \n')
            tex_table_file.write('\\midrule\n')
            tex_table_file.write('Composite, core and coating & %.2f \\\\ \n' % (total_mat_cost_w_waste))
            tex_table_file.write('Resin, hardener and bonding & %.2f \\\\ \n' % (total_resin_hard_bond_cost))
            tex_table_file.write('Bolts and LPS & %.2f \\\\ \n' % (total_metallic_parts_cost))
            tex_table_file.write('Consumables & %.2f \\\\ \n' % (total_consumable_cost_w_waste))
            tex_table_file.write('\\textbf{Total} & \\textbf{%.2f} \\\\ \n' % (total_blade_mat_cost_w_waste))          
            tex_table_file.write('\\bottomrule\n')
            tex_table_file.write('\\end{tabular}\n')
            tex_table_file.write('\\end{table}\n')
            tex_table_file.close()
        
        
        # Plotting
        if self.options['generate_plots']:
            if 'dir_out' in self.options.keys():
                dir_out = os.path.abspath(self.options['dir_out'])
            else:
                dir_out = os.path.abspath('Plots')
            if not os.path.exists(dir_out):
                os.makedirs(dir_out)


            fig1, ax1 = plt.subplots()
            patches, texts,  = ax1.pie(consumable_cost_w_waste, explode=np.zeros(len(consumable_cost_w_waste)), labels=name_consumables,
                    shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            for i in range(len(texts)):
                texts[i].set_fontsize(6)
            fig1.savefig(os.path.join(dir_out, 'Consumables_' + self.name +'.png'))
            
            
            sizes  = total_mat_cost_w_waste , total_resin_hard_bond_cost , total_metallic_parts_cost , total_consumable_cost_w_waste
            labels = 'Fabrics, core and coating' , 'Resin, hardener and bonding', 'Bolts and LPS', 'Consumables'
            fig1, ax1 = plt.subplots()
            patches, texts = ax1.pie(sizes, explode=np.zeros(len(sizes)), labels=labels, shadow=True, startangle=0)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            for i in range(len(texts)):
                texts[i].set_fontsize(10)
            fig1.savefig(os.path.join(dir_out, 'BOM_' + self.name +'.png'))
            
        
        return total_blade_mat_cost_w_waste, blade_mass
        
        
        
        
        
        
        
        
    
    
