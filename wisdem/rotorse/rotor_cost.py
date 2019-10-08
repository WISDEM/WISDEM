import numpy as np
import time, os, warnings
import matplotlib.pyplot as plt
import math
from scipy.optimize import brentq
from openmdao.api import ExplicitComponent

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
            material_dict[name]['fiber_density']     = 2600.    # [kg/m3] Density of the fibers
            material_dict[name]['waste']             = 15.      # [%] Waste of the material during production
            material_dict[name]['unit_cost']         = 2.86     # [$/kg]
            material_dict[name]['roll_mass']         = 181.4368 # [kg] 400 lbs - mass of an individual roll
        
        
        # Uniaxial fabrics spar caps
        uniax_names = ['(SparCapMix)', 'UniaxSC', 'glassUD', 'glass_uni','(ELT5500EP3(Uni))','uniax','ud']
        for name in uniax_names:
            material_dict[name]                       = {}
            material_dict[name]['component']          = [4]# Flag to specify where the material is used. 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
            material_dict[name]['area_density_dry']   = 1.858    # [kg/m2] Unit mass dry fabric
            material_dict[name]['fiber_density']     = 2600.    # [kg/m3] Density of the fibers
            material_dict[name]['waste']              = 5.       # [%] Waste of the material during production
            material_dict[name]['unit_cost']          = 1.87     # [$/kg]
        
        
        # Uniaxial fabrics reinf
        uniax_names = ['UniaxTELEre']
        for name in uniax_names:
            material_dict[name]                       = {}
            material_dict[name]['component']          = [5]# Flag to specify where the material is used. 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
            material_dict[name]['area_density_dry']   = 1.858    # [kg/m2] Unit mass dry fabric
            material_dict[name]['fiber_density']     = 2600.    # [kg/m3] Density of the fibers
            material_dict[name]['waste']              = 5.       # [%] Waste of the material during production
            material_dict[name]['unit_cost']          = 1.87     # [$/kg]
        
        uniax_names_CF = ['(Newport307)', 'CarbonUD','carbon_uni'] # 50oz Carbon Uni for the spar caps
        for name in uniax_names_CF:
            material_dict[name]                       = {}
            material_dict[name]['component']          = [4]      # Flag to specify where the material is used. 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
            material_dict[name]['area_density_dry']   = 1.000    # [kg/m2] Unit mass dry fabric
            material_dict[name]['fiber_density']     = 1800.    # [kg/m3] Density of the fibers
            material_dict[name]['waste']              = 5.       # [%] Waste of the material during production
            material_dict[name]['unit_cost']          = 30.00    # [$/kg]
        
        
        
        # Biaxial fabrics
        biax_names = ['(RandomMat)', 'BiaxWebs', 'glassDB','glass_biax','(SaertexEP3(DB))','biax']
        for name in biax_names:
            material_dict[name]                        = {}
            material_dict[name]['component']           = [3]      # Flag to specify where the material is used. 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.
            material_dict[name]['area_density_dry']    = 1.112    # [kg/m2] Unit mass dry fabric
            material_dict[name]['fiber_density']     = 2600.    # [kg/m3] Density of the fibers
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
                    precomp_mat[mat.name]['fvf']  = (precomp_mat[mat.name]['density'] - self.density_epoxy) / (self.material_dict[mat.name]['fiber_density'] - self.density_epoxy) * 100. # [%] Fiber volume fraction
                    precomp_mat[mat.name]['fwf']  = self.material_dict[mat.name]['fiber_density'] * precomp_mat[mat.name]['fvf'] / 100. / (self.density_epoxy + ((self.material_dict[mat.name]['fiber_density'] - self.density_epoxy) * precomp_mat[mat.name]['fvf'] / 100.)) * 100.
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
                        thick_core_webs[i_web, i_section]      = thick_core_webs[i_web, i_section] + web_height[i_web, i_section] * self.websCS[i_section].t[i_web][i_mat]                   # [m2]
                        unit_mass_core_webs[i_web, i_section]  = unit_mass_core_webs[i_web, i_section] + web_height[i_web, i_section] * self.websCS[i_section].t[i_web][i_mat] * density[mat_id]   # [kg/m]
                    
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

class blade_labor_ct(object):


    def __init__(self , blade_specs , precomp_mat , metallic_parts):
        
        # # Blade input parameters
        # # Material inputs
        self.materials                               = precomp_mat
        # Root preform low pressure side
        self.root_parameters_lp = {}
        self.root_parameters_lp['blade_length']      = blade_specs['blade_length']              # Length of the blade [m]
        self.root_parameters_lp['length']            = blade_specs['root_preform_length']       # Root PF length [m]
        self.root_parameters_lp['root_D']            = blade_specs['root_D']                    # Root PF diameter [m]
        self.root_parameters_lp['half_circum']       = 0.5*np.pi*blade_specs['root_D']          # 1/2 root circumference [m]
        self.root_parameters_lp['area']              = self.root_parameters_lp['half_circum'] * self.root_parameters_lp['length']    # Root PF area [m2]
        self.root_parameters_lp['fabric2lay']        = round(blade_specs['n_plies_root_lp']/2)  # Number of root plies [-]
        self.root_parameters_lp['ply_volume']        = blade_specs['volume_root_preform_lp']    # Ply volume [m3]    
        # Root preform high pressure side
        self.root_parameters_hp = {}
        self.root_parameters_hp['blade_length']      = blade_specs['blade_length']              # Length of the blade [m]
        self.root_parameters_hp['length']            = self.root_parameters_lp['length']        # Root PF length [m], currently assumed as 1% of BL
        self.root_parameters_hp['root_D']            = self.root_parameters_lp['root_D']        # Root PF diameter [m]
        self.root_parameters_hp['half_circum']       = self.root_parameters_lp['half_circum']   # 1/2 root circumference [m]
        self.root_parameters_hp['area']              = self.root_parameters_lp['area']          # Root PF area [m2]
        self.root_parameters_hp['fabric2lay']        = round(blade_specs['n_plies_root_hp']/2)  # Number of root plies [-]
        self.root_parameters_hp['ply_volume']        = blade_specs['volume_root_preform_hp']    # Ply volume [m3]    
        # Shear webs
        self.n_webs                                  = blade_specs['n_webs']
        self.sw_parameters                           = {}
        self.sw_parameters['blade_length']           = blade_specs['blade_length']           # Length of the blade [m]
        self.sw_parameters['length']                 = blade_specs['length_webs']            # Length of the shear webs [m]
        self.sw_parameters['height1']                = blade_specs['height_webs_start']      # Heigth of the shear webs towards blade root [m]
        self.sw_parameters['height2']                = blade_specs['height_webs_end']        # Heigth of the shear webs towards blade tip [m]
        self.sw_parameters['core_area']              = blade_specs['area_webs_w_core']       # Area of the shear webs with sandwich core [m2]
        self.sw_parameters['area']                   = blade_specs['area_webs_w_flanges']    # Shear webs area [m2]
        self.sw_parameters['fabric2lay']             = blade_specs['fabric2lay_webs']        # Total ply area [m2]
        self.sw_parameters['ply_volume']             = blade_specs['volumeskin2lay_webs']    # Ply volume [m3]  
        # Spar cap low pressure side
        self.lp_sc_parameters                        = {}
        self.lp_sc_parameters['blade_length']        = blade_specs['blade_length']           # Length of the blade [m]
        self.lp_sc_parameters['length']              = blade_specs['length_sc_lp']           # Length of the spar caps [m]
        self.lp_sc_parameters['width']               = blade_specs['width_sc_start_lp']      # Width of the spar caps [m]
        self.lp_sc_parameters['area']                = blade_specs['length_sc_lp'] * blade_specs['width_sc_start_lp']     # Spar caps area [m2]
        self.lp_sc_parameters['area_wflanges']       = self.lp_sc_parameters['area'] * 1.5   # Spar caps area including flanges [m2] Assume the sides and the flanges of a spar cap mold equal about 1.5 times the area of the spar cap (for tool prep purposes)
        self.lp_sc_parameters['fabric2lay']          = blade_specs['fabric2lay_sc_lp']       # Total ply length [m]
        self.lp_sc_parameters['ply_volume']          = blade_specs['volume2lay_sc_lp']       # Ply volume [m3]
        # Spar caps high pressure side
        self.hp_sc_parameters                        = {}
        self.hp_sc_parameters['blade_length']        = blade_specs['blade_length']           # Length of the blade [m]
        self.hp_sc_parameters['length']              = blade_specs['length_sc_hp']           # Length of the spar caps [m]
        self.hp_sc_parameters['width']               = blade_specs['width_sc_start_hp']      # Width of the spar caps [m]
        self.hp_sc_parameters['area']                = blade_specs['length_sc_hp'] * blade_specs['width_sc_start_hp']     # Spar caps area [m2]
        self.hp_sc_parameters['area_wflanges']       = self.hp_sc_parameters['area'] * 1.5   # Spar caps area including flanges [m2] Assume the sides and the flanges of a spar cap mold equal about 1.5 times the area of the spar cap (for tool prep purposes)
        self.hp_sc_parameters['fabric2lay']          = blade_specs['fabric2lay_sc_hp']       # Total ply length [m]
        self.hp_sc_parameters['ply_volume']          = blade_specs['volume2lay_sc_hp']       # Ply volume [m3]
        # Low pressure skin
        self.lp_skin_parameters                      = {}
        self.lp_skin_parameters['blade_length']      = blade_specs['blade_length']           # Length of the blade [m]
        self.lp_skin_parameters['length']            = blade_specs['blade_length']           # Length of the skin mold [m]
        self.lp_skin_parameters['area']              = blade_specs['area_lpskin_wo_flanges'] # Skin area on the low pressure side [m2]
        self.lp_skin_parameters['area_wflanges']     = blade_specs['area_lpskin_w_flanges']  # Skin area including flanges [m2]
        self.lp_skin_parameters['fabric2lay']        = 0.5 * blade_specs['fabric2lay_shell_lp'] # Total ply area, outer layers [m2]. Assumed to be 50% of the total layers
        self.lp_skin_parameters['fabric2lay_inner']  = 0.5 * blade_specs['fabric2lay_shell_lp'] # Total ply area, inner layers [m2]. Assumed to be 50% of the total layers
        self.lp_skin_parameters['core_area']         = blade_specs['areacore2lay_shell_lp']  # Area of the outer shell with sandwich core [m2]
        self.lp_skin_parameters['n_root_plies']      = self.root_parameters_lp['fabric2lay'] # Number of root plies [-]
        self.lp_skin_parameters['total_TE']          = blade_specs['fabric2lay_te_reinf_lp'] # Total TE reinforcement layer [m]
        self.lp_skin_parameters['total_LE']          = blade_specs['fabric2lay_le_reinf_lp'] # Total LE reinforcement layer [m]
        self.lp_skin_parameters['perimeter_noroot']  = blade_specs['skin_perimeter_wo_root'] # Perimeter of the skin area excluding blade root [m]
        self.lp_skin_parameters['perimeter']         = blade_specs['skin_perimeter_w_root']  # Perimeter of the skin area [m]
        self.lp_skin_parameters['sc_length']         = blade_specs['length_sc_lp']           # Length of the spar cap [m]
        self.lp_skin_parameters['root_sect_length']  = blade_specs['root_preform_length']    # Root section length [m]
        self.lp_skin_parameters['root_half_circumf'] = self.root_parameters_lp['half_circum']# Root half circumference [m]
        # High pressure skin
        self.hp_skin_parameters                      = {}
        self.hp_skin_parameters['blade_length']      = blade_specs['blade_length']           # Length of the blade [m]
        self.hp_skin_parameters['length']            = blade_specs['blade_length']           # Length of the skin mold [m]
        self.hp_skin_parameters['area']              = blade_specs['area_hpskin_wo_flanges'] # Skin area [m2]
        self.hp_skin_parameters['area_wflanges']     = blade_specs['area_hpskin_w_flanges']  # Skin area including flanges [m2]
        self.hp_skin_parameters['fabric2lay']        = 0.5 * blade_specs['fabric2lay_shell_hp'] # Total ply area, outer layers [m2]. Assumed to be 50% of the total layers
        self.hp_skin_parameters['fabric2lay_inner']  = 0.5 * blade_specs['fabric2lay_shell_hp'] # Total ply area, inner layers [m2]. Assumed to be 50% of the total layers
        self.hp_skin_parameters['core_area']         = blade_specs['areacore2lay_shell_hp']  # Area of the outer shell with sandwich core [m2]
        self.hp_skin_parameters['n_root_plies']      = self.root_parameters_hp['fabric2lay'] # Number of root plies [-]
        self.hp_skin_parameters['total_TE']          = blade_specs['fabric2lay_te_reinf_hp'] # Total TE reinforcement layer [m]
        self.hp_skin_parameters['total_LE']          = blade_specs['fabric2lay_le_reinf_hp'] # Total LE reinforcement layer [m]
        self.hp_skin_parameters['perimeter_noroot']  = blade_specs['skin_perimeter_wo_root'] # Perimeter of the skin area excluding blade root [m]
        self.hp_skin_parameters['perimeter']         = blade_specs['skin_perimeter_w_root']  # Perimeter of the skin area [m]
        self.hp_skin_parameters['sc_length']         = blade_specs['length_sc_hp']           # Length of the spar cap [m]
        self.hp_skin_parameters['root_sect_length']  = blade_specs['root_preform_length']    # Root section length [m]
        self.hp_skin_parameters['root_half_circumf'] = self.root_parameters_hp['half_circum']# Root half circumference [m]
        # Assembly 
        self.assembly                                = {}
        self.assembly['sw_length']                   = self.sw_parameters['length']          # Length of the shear webs [m]
        self.assembly['perimeter_noroot']            = blade_specs['skin_perimeter_wo_root'] # Perimeter of the skin area without root [m]
        self.assembly['length']                      = blade_specs['blade_length']           # Length of the blade [m]
        self.assembly['n_webs']                      = blade_specs['n_webs']                 # Number of webs [-]
        # Demold
        self.demold                                  = {}
        self.demold['length']                        = blade_specs['blade_length']           # Length of the blade [m]
        # Trim
        self.trim                                    = {}
        self.trim['perimeter_noroot']                = blade_specs['skin_perimeter_wo_root'] # Length of the blade [m]
        # Overlay
        self.overlay                                 = {}
        self.overlay['length']                       = blade_specs['blade_length']           # Length of the blade [m]
        # Post curing
        self.post_cure                               = {}
        self.post_cure['length']                     = blade_specs['blade_length']           # Length of the blade [m]
        # Cut and drill
        self.cut_drill                               = {}
        self.cut_drill['length']                     = blade_specs['blade_length']           # Length of the blade [m]
        self.cut_drill['root_D']                     = blade_specs['root_D']                 # Diameter of blade root [m]
        self.cut_drill['root_perim']                 = self.cut_drill['root_D']*np.pi        # Perimeter of the root [m]
        self.cut_drill['n_bolts']                    = metallic_parts['n_bolts']             # Number of root bolts [-]
        # Root installation
        self.root_install                            = {}
        self.root_install['length']                  = blade_specs['blade_length']           # Length of the blade [m]
        self.root_install['root_perim']              = self.cut_drill['root_D']*np.pi        # Perimeter of the root [m]
        self.root_install['n_bolts']                 = self.cut_drill['n_bolts']             # Number of root bolts 
        # Surface preparation
        self.surface_prep                            = {}
        self.surface_prep['area']                    = blade_specs['area_lpskin_wo_flanges'] + blade_specs['area_hpskin_wo_flanges'] # Outer blade surface area [m2]
        # Paint
        self.paint                                   = {}
        self.paint['area']                           = blade_specs['area_lpskin_wo_flanges'] + blade_specs['area_hpskin_wo_flanges'] # Outer blade surface area [m2]
        # Surface finishing
        self.surface_finish                          = {}
        self.surface_finish['area']                  = blade_specs['area_lpskin_wo_flanges'] + blade_specs['area_hpskin_wo_flanges'] # Outer blade surface area [m2]
        # Weight and balance
        self.weight_balance                          = {}
        self.weight_balance['length']                = blade_specs['blade_length']           # Length of the blade [m]
        # Inspection
        self.inspection                              = {}
        self.inspection['length']                    = blade_specs['blade_length']           # Length of the blade [m]
        self.inspection['area']                      = blade_specs['area_lpskin_wo_flanges'] + blade_specs['area_hpskin_wo_flanges'] # Outer blade surface area [m2]
        # Shipping preparation
        self.shipping_prep                           = {}
        self.shipping_prep['length']                 = blade_specs['blade_length']           # Length of the blade [m]
        self.shipping_prep['n_bolts']                = self.cut_drill['n_bolts']             # Number of root bolts
        
    
    def execute_blade_labor_ct(self):
        
       
        # Run all manufacturing steps to estimate labor and cycle time
        if self.options['verbosity']:
            verbosity                                = 1
        else:
            verbosity                                = 0
        
        
        
        n_operations                                 = 20 + self.n_webs
        labor                                        = np.zeros(n_operations) # [hr]
        skin_mold_gating_ct                          = np.zeros(n_operations) # [hr]
        non_gating_ct                                = np.zeros(n_operations) # [hr]
        operation                                    = [[] for i in range(int(n_operations))]
        
        if verbosity:
            print('\n#################################\nLabor hours and cycle times')
        operation[0]                                                  = 'Material cutting'
        material_cutting                                              = material_cutting_labor(self.materials)
        material_cutting.material_cutting_steps()
        labor[0] , non_gating_ct[0]                                   = compute_total_labor_ct(material_cutting, operation[0] , verbosity)        
        
        operation[1]                                                  = 'Root preform lp'
        root_lp                                                       = root_preform_labor(self.root_parameters_lp)
        root_lp.manufacturing_steps()
        labor[1] , non_gating_ct[1]                                   = compute_total_labor_ct(root_lp, operation[1] , verbosity)
        
        operation[2]                                                  = 'Root preform hp'
        root_hp                                                       = root_preform_labor(self.root_parameters_hp)
        root_hp.manufacturing_steps()
        labor[2] , non_gating_ct[2]                                   = compute_total_labor_ct(root_hp, operation[2] , verbosity)
        
        for i_web in range(self.n_webs):
            operation[3 + i_web]                                      = 'Infusion shear web number ' + str(i_web+1)
            sw                                                        = shearweb_labor(self.sw_parameters, i_web)
            sw.manufacturing_steps(core=True)
            labor[3 + i_web] , non_gating_ct[3 + i_web]               = compute_total_labor_ct(sw, operation[3 + i_web], verbosity)
        
        operation[3 + self.n_webs]                                    = 'Infusion spar cap lp'
        sc_lp                                                         = sparcaps_labor(self.lp_sc_parameters)
        sc_lp.manufacturing_steps()
        labor[3 + self.n_webs] , non_gating_ct[3 + self.n_webs]       = compute_total_labor_ct(sc_lp, operation[3 + self.n_webs], verbosity)
        
        operation[4 + self.n_webs]                                    = 'Infusion spar cap hp'
        sc_hp                                                         = sparcaps_labor(self.hp_sc_parameters)
        sc_hp.manufacturing_steps()
        labor[4 + self.n_webs] , non_gating_ct[4 + self.n_webs]       = compute_total_labor_ct(sc_hp, operation[4 + self.n_webs], verbosity)
        
        
        # Gating operations
        operation[8 + self.n_webs]                                    = 'Demolding'
        demold                                                        = demold_labor(self.demold)
        demold.demold_steps()
        labor[8 + self.n_webs] , skin_mold_gating_ct[8 + self.n_webs] = compute_total_labor_ct(demold , operation[8 + self.n_webs], verbosity, no_contribution2ct = ['move2finishing'])
        
        
        # LP skin infusion
        operation[5 + self.n_webs]  = 'Lp skin'
        def labor_ct_lp_skin(team_size):
            lp_skin  = lphp_skin_labor(self.lp_skin_parameters, team_size)
            lp_skin.manufacturing_steps(core=True , Extra_Operations_Skin = True, trim_excess = False)
            labor, ct = compute_total_labor_ct(lp_skin, operation[5 + self.n_webs], verbosity, no_contribution2ct = ['layup_root_layers' , 'insert_TE_layers' , 'vacuum_line' , 'tack_tape'])
            
            return labor, ct
        
        def min_ct_lp_skin(team_size):
            _, ct = labor_ct_lp_skin(team_size)
            return ct - (23.9999 - skin_mold_gating_ct[8 + self.n_webs]) * 0.7    

        try:        
            team_size = brentq(lambda x: min_ct_lp_skin(x), 0.01, 100., xtol=1e-4)
        except:
            team_size = brentq(lambda x: min_ct_lp_skin(x), 0.01, 250., xtol=1e-4)

        if self.options['discrete']:
            team_size = round(team_size)
        labor[5 + self.n_webs] , skin_mold_gating_ct[5 + self.n_webs] = labor_ct_lp_skin(team_size)
        
        # HP skin infusion
        operation[6 + self.n_webs]  = 'Hp skin'
        def labor_ct_hp_skin(team_size):
            hp_skin  = lphp_skin_labor(self.hp_skin_parameters, team_size)
            hp_skin.manufacturing_steps(core=True , Extra_Operations_Skin = True, trim_excess = False)
            labor, ct = compute_total_labor_ct(hp_skin, operation[6 + self.n_webs], verbosity, no_contribution2ct = ['layup_root_layers' , 'insert_TE_layers' , 'vacuum_line' , 'tack_tape'])
            
            return labor, ct
        
        def min_ct_hp_skin(team_size):
            _, ct = labor_ct_hp_skin(team_size)
            
            return ct - (23.9999 - skin_mold_gating_ct[8 + self.n_webs]) * 0.7     
        
        try:        
            team_size = brentq(lambda x: min_ct_hp_skin(x), 0.01, 100., xtol=1e-4)
        except:
            team_size = brentq(lambda x: min_ct_hp_skin(x), 0.01, 250., xtol=1e-4)

        if self.options['discrete']:
            team_size = round(team_size)
        labor[6 + self.n_webs] , non_gating_ct[6 + self.n_webs] = labor_ct_hp_skin(team_size)
        
        # HP skin infusion
        operation[7 + self.n_webs]  = 'Assembly'
        def labor_ct_assembly(team_size):
            assembly  = assembly_labor(self.assembly, team_size)
            assembly.assembly_steps()
            labor, ct = compute_total_labor_ct(assembly , operation[7 + self.n_webs], verbosity, no_contribution2ct = ['remove_nonsand_prep_hp' , 'insert_sw' , 'fillet_sw_low', 'shear_clips'])
            
            return labor, ct
        
        def min_ct_assembly(team_size):
            _, ct = labor_ct_assembly(team_size)
                        
            return ct - (23.9999 - skin_mold_gating_ct[5 + self.n_webs] - skin_mold_gating_ct[8 + self.n_webs])      
        
        try:        
            team_size = brentq(lambda x: min_ct_assembly(x), 0.01, 100., xtol=1e-4)
        except:
            team_size = brentq(lambda x: min_ct_assembly(x), 0.01, 250., xtol=1e-4)
        if self.options['discrete']:
            team_size = round(team_size)
        labor[7 + self.n_webs] , skin_mold_gating_ct[7 + self.n_webs] = labor_ct_assembly(team_size)
        
        operation[9 + self.n_webs]                                    = 'Trim'
        trim                                                          = trim_labor(self.trim)
        trim.trim_steps()
        labor[9 + self.n_webs] , non_gating_ct[9 + self.n_webs]       = compute_total_labor_ct(trim , operation[9 + self.n_webs], verbosity)
        
        
        
        operation[10 + self.n_webs]                                   = 'Overlay'
        overlay                                                       = overlay_labor(self.overlay)
        overlay.overlay_steps()
        labor[10 + self.n_webs] , non_gating_ct[10 + self.n_webs]     = compute_total_labor_ct(overlay , operation[10 + self.n_webs], verbosity)
        
        operation[11 + self.n_webs]                                   = 'Post cure'
        post_cure                                                     = post_cure_labor(self.post_cure)
        post_cure.post_cure_steps()
        labor[11 + self.n_webs] , non_gating_ct[11 + self.n_webs]     = compute_total_labor_ct(post_cure , operation[11 + self.n_webs], verbosity)
               
        operation[12 + self.n_webs]                                   = 'Root cut and drill'
        cut_drill                                                     = cut_drill_labor(self.cut_drill)
        cut_drill.cut_drill_steps()
        labor[12 + self.n_webs] , non_gating_ct[12 + self.n_webs]     = compute_total_labor_ct(cut_drill , operation[12 + self.n_webs], verbosity)
        
        operation[13 + self.n_webs]                                   = 'Root hardware installation'
        root_install                                                  = root_install_labor(self.root_install)
        root_install.root_install_steps()
        labor[13 + self.n_webs] , non_gating_ct[13 + self.n_webs]     = compute_total_labor_ct(root_install , operation[13 + self.n_webs], verbosity)
        
        operation[14 + self.n_webs]                                   = 'Surface preparation'
        surface_prep                                                  = surface_prep_labor(self.surface_prep)
        surface_prep.surface_prep_steps()
        labor[14 + self.n_webs] , non_gating_ct[14 + self.n_webs]     = compute_total_labor_ct(surface_prep , operation[14 + self.n_webs], verbosity)
        
        operation[15 + self.n_webs]                                   = 'Painting'
        paint                                                         = paint_labor(self.paint)
        paint.paint_steps()
        labor[15 + self.n_webs] , non_gating_ct[15 + self.n_webs]     = compute_total_labor_ct(paint , operation[15 + self.n_webs], verbosity)

        operation[16 + self.n_webs]                                   = 'Surface finishing'
        surface_finish                                                = surface_finish_labor(self.surface_finish)
        surface_finish.surface_finish_steps()
        labor[16 + self.n_webs] , non_gating_ct[16 + self.n_webs]     = compute_total_labor_ct(surface_finish , operation[16 + self.n_webs], verbosity)

        operation[17 + self.n_webs]                                   = 'Weight balance'
        weight_balance                                                = weight_balance_labor(self.weight_balance)
        weight_balance.weight_balance_steps()
        labor[17 + self.n_webs] , non_gating_ct[17 + self.n_webs]     = compute_total_labor_ct(weight_balance , operation[17 + self.n_webs], verbosity)

        operation[18 + self.n_webs]                                   = 'Final inspection'
        inspection                                                    = inspection_labor(self.inspection)
        inspection.inspection_steps()
        labor[18 + self.n_webs] , non_gating_ct[18 + self.n_webs]     = compute_total_labor_ct(inspection , operation[18 + self.n_webs], verbosity)

        operation[19 + self.n_webs]                                   = 'Shipping preparation'
        shipping_prep                                                 = shipping_prep_labor(self.shipping_prep)
        shipping_prep.shipping_prep_steps()
        labor[19 + self.n_webs] , non_gating_ct[19 + self.n_webs]     = compute_total_labor_ct(shipping_prep , operation[19 + self.n_webs], verbosity)
        
        total_labor               = sum(labor)
        total_skin_mold_gating_ct = sum(skin_mold_gating_ct)
        total_non_gating_ct       = sum(non_gating_ct)
        
        
        if self.options['tex_table']:
            tex_table_file = open('tex_tables.txt','a') 
            tex_table_file.write('\n\n\n\\begin{table}[htpb]\n')
            tex_table_file.write('\\caption{Labor and CT of the %s blade.}\n' % self.name)
            tex_table_file.write('\\label{table:%s_5}\n' % self.name)
            tex_table_file.write('\\centering\n')
            tex_table_file.write('\\begin{tabular}{l c c c}\n')
            tex_table_file.write('\\toprule\n')
            tex_table_file.write('Operation & Labor [hr] & Skin Mold Gating CT [hr] & Non-Gating CT [hr]\\\\ \n')
            tex_table_file.write('\\midrule\n')
            for i_operation in range(len(operation)):
                tex_table_file.write('%s & %.2f & %.2f & %.2f \\\\ \n' % (operation[i_operation],labor[i_operation],skin_mold_gating_ct[i_operation],non_gating_ct[i_operation]))
            tex_table_file.write('\\textbf{Total} & \\textbf{%.2f} & \\textbf{%.2f} & \\textbf{%.2f} \\\\ \n' % (total_labor,total_skin_mold_gating_ct,total_non_gating_ct))
            tex_table_file.write('\\bottomrule\n')
            tex_table_file.write('\\end{tabular}\n')
            tex_table_file.write('\\end{table}\n')
            tex_table_file.close()
        
        if self.options['generate_plots']:
            if 'dir_out' in self.options.keys():
                dir_out = os.path.abspath(self.options['dir_out'])
            else:
                dir_out = os.path.abspath('Plots')
            if not os.path.exists(dir_out):
                os.makedirs(dir_out)


            # Plotting
            fig1, ax1 = plt.subplots()
            patches, texts,  = ax1.pie(labor, explode=np.zeros(len(labor)), labels=operation,
                    shadow=True, startangle=0)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            for i in range(len(texts)):
                texts[i].set_fontsize(8)
            fig1.savefig(os.path.join(dir_out, 'Labor_' + self.name +'.png'))
            
            fig2, ax2 = plt.subplots()
            patches, texts, autotexts = ax2.pie([skin_mold_gating_ct[5 + self.n_webs] , skin_mold_gating_ct[7 + self.n_webs] ,skin_mold_gating_ct[8 + self.n_webs]], explode=np.zeros(3), labels=[operation[5 + self.n_webs] , operation[7 + self.n_webs] ,operation[8 + self.n_webs]], autopct='%1.1f%%', shadow=True, startangle=0, textprops={'fontsize': 10})
            ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            for i in range(len(texts)):
                texts[i].set_fontsize(10)
            fig2.savefig(os.path.join(dir_out, 'Skin_mold_gating_ct_' + self.name +'.png'))
            
            fig3, ax3 = plt.subplots()
            patches, texts,  = ax3.pie(non_gating_ct, explode=np.zeros(len(non_gating_ct)), labels=operation,
                    shadow=True, startangle=0)
            ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            for i in range(len(texts)):
                texts[i].set_fontsize(8)
            fig3.savefig(os.path.join(dir_out, 'Non_gating_ct_' + self.name +'.png'))

        return operation, labor , skin_mold_gating_ct , non_gating_ct

class material_cutting_process(object):

    def material_cutting_steps(self):
        
        self.load_roll['labor_per_mat'] = []
        self.load_roll['ct_per_mat']    = []
        self.cutting['labor_per_mat']   = []
        self.cutting['ct_per_mat']      = []
        self.kitting['labor_per_mat']   = []
        self.kitting['ct_per_mat']      = []
        self.clean_up['labor_per_mat']  = []
        self.clean_up['ct_per_mat']     = []
        
        self.load_roll['labor']         = []
        self.load_roll['ct']            = []
        self.cutting['labor']           = []
        self.cutting['ct']              = []
        self.kitting['labor']           = []
        self.kitting['ct']              = []
        self.clean_up['labor']          = []
        self.clean_up['ct']             = []

        
        mat_names = self.materials.keys()
        
        
        for name in mat_names:
            if self.materials[name]['cut@station'] == 'Y':
                # Number of rolls
                self.materials[name]['n_rolls'] = self.materials[name]['total_mass_w_waste'] / self.materials[name]['roll_mass']
                # Loading and Machine Prep
                self.load_roll['labor_per_mat'].append(self.load_roll['unit_time'] * self.materials[name]['n_rolls'] * self.load_roll['n_pers'])
                self.load_roll['ct_per_mat'].append(self.load_roll['unit_time'] * self.materials[name]['n_rolls'])
                # Cutting
                cutting_labor = self.materials[name]['total_ply_area_w_waste'] / self.cutting['machine_rate'] * self.cutting['n_pers']
                cutting_ct    = self.materials[name]['total_ply_area_w_waste'] / self.cutting['machine_rate']
                self.cutting['labor_per_mat'].append(cutting_labor)
                self.cutting['ct_per_mat'   ].append(cutting_ct)
                # Kitting
                self.kitting['labor_per_mat'].append(cutting_ct * self.kitting['n_pers'])
                self.kitting['ct_per_mat'].append(0)
                # Clean-up
                cleaning_labor          = (self.materials[name]['total_ply_area_w_waste'] - self.materials[name]['total_ply_area_wo_waste']) / self.clean_up['clean_rate']
                self.clean_up['labor_per_mat'].append(cleaning_labor)
                self.clean_up['ct_per_mat'   ].append(cleaning_labor / self.clean_up['n_pers'])
                
            else:
                self.load_roll['labor_per_mat'].append(0)
                self.load_roll['ct_per_mat'].append(0)
                self.cutting['labor_per_mat'].append(0)
                self.cutting['ct_per_mat'].append(0)
                self.kitting['labor_per_mat'].append(0)
                self.kitting['ct_per_mat'].append(0)
                self.clean_up['labor_per_mat'].append(0)
                self.clean_up['ct_per_mat'].append(0)
                
        # Loading and Machine Prep
        self.load_roll['labor']          = sum(self.load_roll['labor_per_mat'])
        self.load_roll['ct']             = sum(self.load_roll['ct_per_mat'])
        # Cutting       
        self.cutting['labor']            = sum(self.cutting['labor_per_mat'])
        self.cutting['ct']               = sum(self.cutting['ct_per_mat'])
        # Kitting       
        self.kitting['labor']            = sum(self.kitting['labor_per_mat'])
        self.kitting['ct']               = sum(self.kitting['ct_per_mat'])
        # Clean-up      
        self.clean_up['labor']           = sum(self.clean_up['labor_per_mat'])
        self.clean_up['ct']              = sum(self.clean_up['ct_per_mat'])
        
        # Remove materials from self
        del self.materials

class material_cutting_labor(material_cutting_process):
    def __init__(self, material_parameters, process = {}):
        
        # # Material cutting - process parameters
        self.load_roll                 = {}
        self.cutting                   = {}
        self.kitting                   = {}
        self.clean_up                  = {}
        self.materials                 = {}
        
                
        # Load roll
        self.load_roll['n_pers']       = 2.                 # Number of personnel involved in the operation
        self.load_roll['unit_time']    = 15. / 60.          # Unit time - 15 minutes [hr]      
        # Cutting       
        self.cutting['n_pers']         = 2.                 # Number of personnel involved in the operation
        self.cutting['machine_rate']   = 833. * 0.9144**2   # Machine rate - 833 [yd2/hr]
        # Kitting       
        self.kitting['n_pers']         = 2.                 # Number of personnel involved in the operation 
        # Clean up      
        self.clean_up['n_pers']        = 2.                 # Number of personnel involved in the operation
        self.clean_up['clean_rate']    = 720. * 0.9144**2.  # Clean rate - 720 [yd2/hr]
        

        self.materials                 = material_parameters
        
        
        for var in process.keys():
            setattr(self, var, process[var])

class infusion_process(object):
    
    def manufacturing_steps(self, core=False, Extra_Operations_Skin=False, trim_excess = True):
        # Preparation of the tools
        self.tool_prep['labor'] , self.tool_prep['ct']  = compute_labor_ct(self.tool_prep['n_pers'], self.tool_prep['area'] , \
            self.tool_prep['ri_appl_rate']*self.tool_prep['n_pulls'], 0 , 0)
        
        # Lay-up of the composite fabric
        self.lay_up['labor'] , self.lay_up['ct']        = compute_labor_ct(self.lay_up['n_pers'], self.lay_up['fabric2lay'], self.lay_up['rate'], 0 , 0)
        
        # Extra operations
        if Extra_Operations_Skin:
            #  Insert the preformed root in the mold
            self.insert_root['labor'] , self.insert_root['ct']  = compute_labor_ct(self.insert_root['n_pers'], 0, 1, self.insert_root['time'], 1)
            
            #  Insert the spar caps in the mold
            if self.insert_sc['sc_length'] <= 30.:
                self.insert_sc['labor'] = self.insert_sc['time'] * self.insert_sc['n_pers']
            else:
                self.insert_sc['labor'] = (self.insert_sc['time'] + (self.insert_sc['sc_length'] - 30.) * self.insert_sc['rate']) * self.insert_sc['n_pers']
            self.insert_sc['ct'] = self.insert_sc['labor']/self.insert_sc['n_pers']
            
            #  Insert the root layers in the mold
            self.layup_root_layers['labor'] , self.layup_root_layers['ct']      = compute_labor_ct(self.layup_root_layers['n_pers'], self.layup_root_layers['n_plies'], self.layup_root_layers['rate'], 0, 0)
        
            #  Insert the trailing edge reinforcement layers in the mold
            self.insert_TE_layers['labor'] , self.insert_TE_layers['ct']        = compute_labor_ct(self.insert_TE_layers['n_pers'], self.insert_TE_layers['length'], self.insert_TE_layers['rate'], 0, 0)
            
            #  Insert the leading edge reinforcement layers in the mold
            self.insert_LE_layers['labor'] , self.insert_LE_layers['ct']        = compute_labor_ct(self.insert_LE_layers['n_pers'], self.insert_LE_layers['length'], self.insert_LE_layers['rate'], 0, 0)
            
            #  Insert the inner layers in the mold
            self.insert_inner_layers['labor'] , self.insert_inner_layers['ct']  = compute_labor_ct(self.insert_inner_layers['n_pers'], self.insert_inner_layers['fabric2lay'], self.insert_inner_layers['rate'], 0, 0)
        
        # Core placement
        if core:
            self.core_placement['labor'] , self.core_placement['ct'] = compute_labor_ct(self.core_placement['n_pers'], self.core_placement['area'], self.core_placement['rate'], 0, 0)
        else:
            self.core_placement['labor'] = 0.
            self.core_placement['ct'] = 0.
            
        # Application of the peel-ply
        self.peel_ply['labor'] , self.peel_ply['ct']        = compute_labor_ct(self.peel_ply['n_pers'], self.peel_ply['area'] , self.peel_ply['rate'] , 0, 0)

        # Application of the flow medium
        self.fm_app['labor'] , self.fm_app['ct']            = compute_labor_ct(self.fm_app['n_pers'], self.peel_ply['area'] * self.fm_app['coverage'] , self.fm_app['rate'] , 0, 0)
        
        # Application of the feed lines
        self.feed['labor'] , self.feed['ct']                = compute_labor_ct(self.feed['n_pers'], self.feed['length'] , self.feed['rate'] , 0, 0)
        
        # Application of vacuum lines
        self.vacuum_line['labor'] , self.vacuum_line['ct']  = compute_labor_ct(self.vacuum_line['n_pers'], self.vacuum_line['length'], self.vacuum_line['rate'] , 0, 0)
        
        # Application of the tack-tape
        self.tack_tape['labor'] , self.tack_tape['ct']      = compute_labor_ct(self.tack_tape['n_pers'], self.tack_tape['length'] , self.tack_tape['rate'] , 0, 0)
        
        # Application of the vacuum bag
        self.vacuum_bag['labor'] , self.vacuum_bag['ct']    = compute_labor_ct(self.vacuum_bag['n_pers'], self.peel_ply['area'] , self.vacuum_bag['rate'] , 0, 0)
        
        # Pull of full vacuuum
        self.vacuum_pull['labor'] , self.vacuum_pull['ct']  = compute_labor_ct(self.vacuum_pull['n_pers'], self.peel_ply['area'] , self.vacuum_pull['rate'] , 0, 1)
        
        # Check of vacuum leaks
        self.leak_chk['labor'] , self.leak_chk['ct']        = compute_labor_ct(self.leak_chk['n_pers'], self.tack_tape['length'] , self.leak_chk['rate'] , 0, 0)
        
        # Actual infusion
        self.infusion['labor'] , self.infusion['ct']        = compute_labor_ct(self.infusion['n_pers'], 0, 1, self.infusion['time'], 1)
        
        # Curing
        self.cure['labor'] , self.cure['ct']                = compute_labor_ct(self.cure['n_pers'], 0, 1, self.cure['time'], 1)
        
        # Debagging operations
        self.debag['labor'] , self.debag['ct']              = compute_labor_ct(self.debag['n_pers'], self.debag['area'], self.debag['rate'], 0, 0)
        
        # Demolding operations
        self.demold['labor'] , self.demold['ct']            = compute_labor_ct(self.demold['n_pers'], 0, 1, self.demold['time'], 0)
        
        if trim_excess:
            #  Trim (cut) of the excess fiberglass off of the root preform edges
            self.insert_prep_trim['labor'] , self.insert_prep_trim['ct'] = compute_labor_ct(self.insert_prep_trim['n_pers'], self.insert_prep_trim['length'], self.insert_prep_trim['rate'], 0, 0)
        else:
            self.insert_prep_trim['labor'] = 0.
            self.insert_prep_trim['ct']    = 0.

class root_preform_labor(infusion_process):

    def __init__(self, component_parameters, process = {} ):
        
        # Manufacturing process labor input data for a root preform
        self.tool_prep                     = {}
        self.lay_up                        = {}
        self.core_placement                = {}
        self.peel_ply                      = {}
        self.fm_app                        = {}
        self.feed                          = {}
        self.vacuum_line                   = {}
        self.tack_tape                     = {}
        self.vacuum_bag                    = {}
        self.vacuum_pull                   = {}
        self.leak_chk                      = {}
        self.infusion                      = {}
        self.cure                          = {}
        self.debag                         = {}
        self.demold                        = {}
        self.insert_prep_trim              = {}
        
        # Tool preparation
        self.tool_prep['n_pers']           = 2.         # Number of personnel involved in the operation
        self.tool_prep['n_pulls']          = 5.         # Number of pulls [-]
        self.tool_prep['ri_appl_rate']     = 12.        # "Rls appl rate per pn [m2/hr/man]
        self.tool_prep['area']             = component_parameters['area'] # Area to be prepared
        # Lay-up                           
        self.lay_up['n_pers']              = 2.         # Number of personnel involved in the operation
        self.lay_up['fabric2lay']          = component_parameters['fabric2lay']
        self.lay_up['rate']                = 8.       # Rate to lay composite [m2/hr/man]
        # Infusion preparation
        # Application of the peel ply
        self.peel_ply['n_pers']            = 2.         # Number of personnel involved in the operation
        self.peel_ply['area']              = component_parameters['area'] # Area where to apply peel-ply
        self.peel_ply['rate']              = 20.       # Peel-ply application rate [m2/hr/man]
        # Application of the flow medium          
        self.fm_app['n_pers']              = 2.         # Number of personnel involved in the operation
        self.fm_app['rate']                = 10.       # Flow-medium application rate [m2/hr/man]
        self.fm_app['coverage']            = 0.85       # Flow-medium coverage [% dec]
        # Installation feeding line         
        self.feed['n_pers']                = 2.         # Number of personnel involved in the operation
        self.feed['spacing']               = 0.5        # Spanwise spacing of the radial feed lines [m]
        self.feed['length']                = component_parameters['length'] + component_parameters['half_circum'] * component_parameters['length'] / self.feed['spacing']
        self.feed['rate']                  = 7.5        # Feed application rate [m/hr/man]
        # Vacuum line application           
        self.vacuum_line['n_pers']         = 1.         # Number of personnel involved in the operation
        self.vacuum_line['rate']           = 20.       # Vacuum line application rate [m/hr/man]
        self.vacuum_line['length']         = 2. * component_parameters['length']   # Vacuum line length [m]
        # Application tack tape             
        self.tack_tape['n_pers']           = 1.         # Number of personnel involved in the operation
        self.tack_tape['length']           = 2. * component_parameters['length'] + 2. * component_parameters['half_circum'] # Tack tape length [m]
        self.tack_tape['rate']             = 90.      # Tack tape application rate [m/hr/man]
        # Application vacuum bag            
        self.vacuum_bag['n_pers']          = 2.         # Number of personnel involved in the operation
        self.vacuum_bag['rate']            = 7.5        # Vacuum bag application rate [m2/hr/man]
        # Pull full vacuum                  
        self.vacuum_pull['n_pers']         = 2.         # Number of personnel involved in the operation
        self.vacuum_pull['rate']           = 45.        # Vacuum pull rate [m2/hr/man]
        # Check for leackages               
        self.leak_chk['n_pers']            = 2          # Number of personnel involved in the operation
        self.leak_chk['rate']              = 30.       # Leak_chk rate [m/hr/man]
        # Infusion                         
        self.infusion['n_pers']            = 1.         # Number of personnel involved in the operation
        p0                                 = 15.        # p0 of the polynomial fit 
        p1                                 = 1.         # p1 of the polynomial fit 
        p2                                 = 0.         # p2 of the polynomial fit 
        p3                                 = 0.         # p3 of the polynomial fit 
        self.infusion['time']              = (p0 + p1 * component_parameters['blade_length'] + p2 * component_parameters['blade_length']**2 + p3 * component_parameters['blade_length']**3) / 60. # Infusion time [hr]
        # Cure                             
        self.cure['n_pers']                = 1.         # Number of personnel involved in the operation
        self.cure['time']                  = 2.         # Curing time [hr]
        # Debag                            
        self.debag['n_pers']               = 2.         # Number of personnel involved in the operation
        self.debag['area']                 = component_parameters['area'] # Area to be debagged 
        self.debag['rate']                 = 20.       # Debag rate [m2/hr/man]
        # Demold                           
        self.demold['n_pers']              = 2.         # Number of personnel involved in the operation
        self.demold['time']                = 0.5        # Demold time [hr]
        # Insertion preparation and trim   
        self.insert_prep_trim['n_pers']    = 2.         # Number of personnel involved in the operation
        self.insert_prep_trim['length']    = 2. * component_parameters['length'] 
        self.insert_prep_trim['rate']      = 6.         # Trim rate [m/hr/man]
        
        
        
        for var in process.keys():
            setattr(self, var, process[var])        

class shearweb_labor(infusion_process):

    def __init__(self, component_parameters, i_web, process = {} ):
        
        # Manufacturing process labor input data for shear webs
        self.tool_prep                     = {}
        self.lay_up                        = {}
        self.core_placement                = {}
        self.peel_ply                      = {}
        self.fm_app                        = {}
        self.feed                          = {}
        self.vacuum_line                   = {}
        self.tack_tape                     = {}
        self.vacuum_bag                    = {}
        self.vacuum_pull                   = {}
        self.leak_chk                      = {}
        self.infusion                      = {}
        self.cure                          = {}
        self.debag                         = {}
        self.demold                        = {}
        self.insert_prep_trim              = {}
        
        # Tool preparation
        self.tool_prep['n_pers']           = 10.         # Number of personnel involved in the operation
        self.tool_prep['n_pulls']          = 5.         # Number of pulls [-]
        self.tool_prep['ri_appl_rate']     = 12.        # "Rls appl rate per pn [m2/hr/man]
        self.tool_prep['area']             = component_parameters['area'][i_web] # Area to be prepared
        # Lay-up                           
        self.lay_up['n_pers']              = 10.         # Number of personnel involved in the operation
        self.lay_up['fabric2lay']          = component_parameters['fabric2lay'][i_web]
        self.lay_up['rate']                = 24.       # Rate to lay composite [m2/hr/man]
        # Core                             
        self.core_placement['area']        = component_parameters['core_area'][i_web] # Area with sandwich core
        self.core_placement['n_pers']      = 10.         # Number of personnel involved in the operation - Ignored if the core_area is set to 0
        self.core_placement['rate']        = 12.        # Rate of core lay [m2/hr/man] - Ignored if the core_area is set to 0 
        # Infusion preparation
        # Application of the peel ply
        self.peel_ply['n_pers']            = 10.         # Number of personnel involved in the operation
        self.peel_ply['area']              = component_parameters['area'][i_web] # Area where to apply peel-ply
        self.peel_ply['area']              = component_parameters['area'][i_web]
        self.peel_ply['rate']              = 20.       # Peel-ply application rate [m2/hr/man]
        # Application of the flow medium          
        self.fm_app['n_pers']              = 10.         # Number of personnel involved in the operation
        self.fm_app['rate']                = 10.       # Flow-medium application rate [m2/hr/man]
        self.fm_app['coverage']            = 0.85       # Flow-medium coverage [% dec]
        # Installation feeding line         
        self.feed['n_pers']                = 10.         # Number of personnel involved in the operation
        self.feed['length']                = component_parameters['length'][i_web]
        self.feed['rate']                  = 7.5       # Feed application rate [m/hr/man]
        # Vacuum line application           
        self.vacuum_line['n_pers']         = 5.         # Number of personnel involved in the operation
        self.vacuum_line['rate']           = 20.         # Vacuum line application rate [m/hr/man]
        self.vacuum_line['length']         = 2. * component_parameters['length'][i_web]   # Vacuum line length [m]
        # Application tack tape             
        self.tack_tape['n_pers']           = 5.         # Number of personnel involved in the operation
        self.tack_tape['length']           = 2. * component_parameters['length'][i_web] + \
            component_parameters['height1'][i_web] + component_parameters['height2'][i_web] # Tack tape length [m]
        self.tack_tape['rate']             = 90.      # Tack tape application rate [m/hr/man]
        # Application vacuum bag            
        self.vacuum_bag['n_pers']          = 10.         # Number of personnel involved in the operation
        self.vacuum_bag['rate']            = 7.5       # Vacuum bag application rate [m2/hr/man]
        # Pull full vacuum                  
        self.vacuum_pull['n_pers']         = 10.         # Number of personnel involved in the operation
        self.vacuum_pull['rate']           = 45.       # Vacuum pull rate [m2/hr/man]
        # Check for leackages               
        self.leak_chk['n_pers']            = 10.         # Number of personnel involved in the operation
        self.leak_chk['rate']              = 30.       # Leak_chk rate [m/hr/man]
        # Infusion                         
        self.infusion['n_pers']            = 1.         # Number of personnel involved in the operation
        p0                                 = 11.983     # p0 of the polynomial fit 
        p1                                 = 0.3784     # p1 of the polynomial fit 
        p2                                 = 0.         # p2 of the polynomial fit 
        p3                                 = 0.         # p3 of the polynomial fit 
        self.infusion['time']              = (p0 + p1 * component_parameters['blade_length'] + p2 * component_parameters['blade_length']**2 + p3 * component_parameters['blade_length']**3) / 60. # Infusion time [hr]
        # Cure                             
        self.cure['n_pers']                = 1.         # Number of personnel involved in the operation
        self.cure['time']                  = 2.         # Curing time [hr]
        # Debag                            
        self.debag['n_pers']               = 10.         # Number of personnel involved in the operation
        self.debag['area']                 = component_parameters['area'][i_web] # Area to be debagged 
        self.debag['rate']                 = 20.        # Debag rate [m2/hr/man]
        # Demold                           
        self.demold['n_pers']              = 10.         # Number of personnel involved in the operation
        self.demold['time']                = 0.5        # Demold time [hr]
        # Insertion preparation and trim   
        self.insert_prep_trim['n_pers']    = 10.         # Number of personnel involved in the operation
        self.insert_prep_trim['length']    = component_parameters['length'][i_web] 
        self.insert_prep_trim['rate']      = 30.        # Trim rate [m/hr/man]
        

        
        for var in process.keys():
            setattr(self, var, process[var])

class sparcaps_labor(infusion_process):

    def __init__(self, component_parameters, process = {} ):
        
        # Manufacturing process labor input data for the spar caps
        self.tool_prep                     = {}
        self.lay_up                        = {}
        self.core_placement                = {}
        self.peel_ply                      = {}
        self.fm_app                        = {}
        self.feed                          = {}
        self.vacuum_line                   = {}
        self.tack_tape                     = {}
        self.vacuum_bag                    = {}
        self.vacuum_pull                   = {}
        self.leak_chk                      = {}
        self.infusion                      = {}
        self.cure                          = {}
        self.debag                         = {}
        self.demold                        = {}
        self.insert_prep_trim              = {}
        
        # Tool preparation
        self.tool_prep['n_pers']           = 10.         # Number of personnel involved in the operation
        self.tool_prep['n_pulls']          = 5.         # Number of pulls [-]
        self.tool_prep['ri_appl_rate']     = 12.        # "Rls appl rate per pn [m2/hr/man]
        self.tool_prep['area']             = component_parameters['area_wflanges'] # Area to be prepared
        # Lay-up                           
        self.lay_up['n_pers']              = 10.         # Number of personnel involved in the operation
        self.lay_up['fabric2lay']          = component_parameters['fabric2lay']
        self.lay_up['rate']                = 110.      # Rate to lay composite [m/hr/man]
        # Infusion preparation             
        # Application of the peel ply      
        self.peel_ply['n_pers']            = 10.         # Number of personnel involved in the operation
        self.peel_ply['area']              = component_parameters['area'] # Area where to apply peel-ply
        self.peel_ply['rate']              = 20.       # Peel-ply application rate [m2/hr/man]
        # Application of the flow medium          
        self.fm_app['n_pers']              = 10.         # Number of personnel involved in the operation
        self.fm_app['rate']                = 10.       # Flow-medium application rate [m2/hr/man]
        self.fm_app['coverage']            = 0.85       # Flow-medium coverage [% dec]
        # Installation feeding line         
        self.feed['n_pers']                = 10.         # Number of personnel involved in the operation
        self.feed['length']                = component_parameters['length']
        self.feed['rate']                  = 7.5        # Feed application rate [m/hr/man]
        # Vacuum line application         
        self.vacuum_line['n_pers']         = 5.         # Number of personnel involved in the operation
        self.vacuum_line['rate']           = 20.        # Vacuum line application rate [m/hr/man]
        self.vacuum_line['length']         = component_parameters['length']   # Vacuum line length [m]
        # Application tack tape             
        self.tack_tape['n_pers']           = 5.         # Number of personnel involved in the operation
        self.tack_tape['length']           = 2. * component_parameters['length'] + 2. * component_parameters['width'] # Tack tape length [m]
        self.tack_tape['rate']             = 90.      # Tack tape application rate [m/hr/man]
        # Application vacuum bag            
        self.vacuum_bag['n_pers']          = 10.       # Number of personnel involved in the operation
        self.vacuum_bag['rate']            = 7.5       # Vacuum bag application rate [m2/hr/man]
        # Pull full vacuum                  
        self.vacuum_pull['n_pers']         = 10.         # Number of personnel involved in the operation
        self.vacuum_pull['rate']           = 45.       # Vacuum pull rate [m2/hr/man]
        # Check for leackages               
        self.leak_chk['n_pers']            = 10.         # Number of personnel involved in the operation
        self.leak_chk['rate']              = 30.       # Leak_chk rate [m/hr/man]
        # Infusion                         
        self.infusion['n_pers']            = 1.         # Number of personnel involved in the operation
        p0                                 = 23.992     # p0 of the polynomial fit 
        p1                                 = 0.0037     # p1 of the polynomial fit 
        p2                                 = 0.0089     # p2 of the polynomial fit 
        p3                                 = 0.         # p3 of the polynomial fit 
        self.infusion['time']              = (p0 + p1 * component_parameters['blade_length'] + p2 * component_parameters['blade_length']**2 + p3 * component_parameters['blade_length']**3) / 60. # Infusion time [hr]
        # Cure                             
        self.cure['n_pers']                = 1.         # Number of personnel involved in the operation
        self.cure['time']                  = 2.         # Curing time [hr]
        # Debag                            
        self.debag['n_pers']               = 10.         # Number of personnel involved in the operation
        self.debag['area']                 = component_parameters['area'] # Area to be debagged 
        self.debag['rate']                 = 20.        # Debag rate [m2/hr/man]
        # Demold                           
        self.demold['n_pers']              = 10.         # Number of personnel involved in the operation
        self.demold['time']                = 0.5        # Demold time [hr]
        # Insertion preparation and trim   
        self.insert_prep_trim['n_pers']    = 10.         # Number of personnel involved in the operation
        self.insert_prep_trim['length']    = component_parameters['length'] 
        self.insert_prep_trim['rate']      = 30.        # Trim rate [m/hr/man]
        

        
        for var in process.keys():
            setattr(self, var, process[var])

class lphp_skin_labor(infusion_process):

    def __init__(self, component_parameters, team_size, process = {} ):
        
        # Manufacturing process labor input data for the low pressure and high pressure skins
        self.tool_prep                     = {}
        self.lay_up                        = {}
        self.insert_root                   = {}
        self.insert_sc                     = {}
        self.layup_root_layers             = {}
        self.core_placement                = {}
        self.insert_TE_layers              = {}
        self.insert_LE_layers              = {}
        self.insert_inner_layers           = {}
        self.peel_ply                      = {}
        self.fm_app                        = {}
        self.feed                          = {}
        self.vacuum_line                   = {}
        self.tack_tape                     = {}
        self.vacuum_bag                    = {}
        self.vacuum_pull                   = {}
        self.leak_chk                      = {}
        self.infusion                      = {}
        self.cure                          = {}
        self.debag                         = {}
        self.demold                        = {}
        self.insert_prep_trim              = {}
        
        
        # Tool preparation
        self.tool_prep['n_pers']           = team_size          # Number of personnel involved in the operation
        self.tool_prep['n_pulls']          = 5.                 # Number of pulls [-]
        self.tool_prep['ri_appl_rate']     = 12.                # "Rls appl rate per pn [m2/hr/man]
        self.tool_prep['area']             = component_parameters['area_wflanges'] # Area to be prepared
        # Lay-up                           
        self.lay_up['n_pers']              = team_size          # Number of personnel involved in the operation
        self.lay_up['fabric2lay']          = component_parameters['fabric2lay']
        self.lay_up['rate']                = 30.                # Rate to lay composite [m/hr/man]
        # Insert the preformed root        
        self.insert_root['n_pers']         = 0.25 * team_size   # Number of personnel involved in the operation
        self.insert_root['time']           = 0.25               # Root Preform Insertion Time [hr]
        # Insert the pre-fabricated spar caps
        self.insert_sc['n_pers']           = 0.25 * team_size   # Number of personnel involved in the operation
        self.insert_sc['sc_length']        = component_parameters['sc_length']
        self.insert_sc['time']             = 0.25               # Time to insert spar caps shorter than 30 meters [hr]
        self.insert_sc['rate']             = 0.00833            # Addsitional time per meter to insert spar caps longer than 30 meters [hr/m]
        # Layup of the root plies          
        self.layup_root_layers['n_pers']   = 0.25 * team_size   # Number of personnel involved in the operation
        self.layup_root_layers['n_plies']  = component_parameters['n_root_plies']
        self.layup_root_layers['rate']     = 16.                # Root layup rate
        # Core placement                   
        self.core_placement['n_pers']      = 0.75 * team_size   # Number of personnel involved in the operation - Ignored if the core_area is set to 0
        self.core_placement['area']        = component_parameters['core_area']        # Area with core [m2]
        self.core_placement['rate']        = 12.                # Rate of core lay [m2/hr/man] - Ignored if the core_area is set to 0
        # Layup of the trailing edge reinforcement
        self.insert_TE_layers['n_pers']    = 0.25 * team_size   # Number of personnel involved in the operation
        self.insert_TE_layers['length']    = component_parameters['total_TE'] # Length of the layers of trailing edge reinforcement
        self.insert_TE_layers['rate']      = 96.                # TE reinforcement layup rate
        # Layup of the leading edge reinforcement
        self.insert_LE_layers['n_pers']    = team_size          # Number of personnel involved in the operation
        self.insert_LE_layers['length']    = component_parameters['total_LE'] # Length of the layers of leading edge reinforcement
        self.insert_LE_layers['rate']      = 96.                # LE reinforcement layup rate
        # Layup of the inner layers
        self.insert_inner_layers['n_pers'] = team_size          # Number of personnel involved in the operation
        self.insert_inner_layers['fabric2lay'] = component_parameters['fabric2lay_inner']
        self.insert_inner_layers['rate']   = 30.                # Inner layers reinforcement layup rate
        # Infusion preparation
        # Application of the peel ply
        self.peel_ply['n_pers']            = team_size          # Number of personnel involved in the operation
        self.peel_ply['area']              = component_parameters['area'] # Area where to apply peel-ply
        self.peel_ply['rate']              = 22.5               # Peel-ply application rate [m2/hr/man]
        # Application of the flow medium          
        self.fm_app['n_pers']              = team_size          # Number of personnel involved in the operation
        self.fm_app['rate']                = 10.                # Flow-medium application rate [m2/hr/man]
        self.fm_app['coverage']            = 0.85               # Flow-medium coverage [% dec]
        # Installation feeding line         
        self.feed['n_pers']                = team_size          # Number of personnel involved in the operation
        self.feed['spacing']               = 0.5                # Spanwise spacing of the radial feed lines [m] 
        self.feed['length']                = 2 * component_parameters['root_sect_length'] + component_parameters['root_half_circumf'] * component_parameters['root_sect_length'] / self.feed['spacing'] + component_parameters['length'] - 2 + 4 * 0.65 * component_parameters['length']# Vacuum line length [m]
        self.feed['rate']                  = 10.                # Feed application rate [m/hr/man]
        # Vacuum line application           
        self.vacuum_line['n_pers']         = 0.5 * team_size    # Number of personnel involved in the operation
        self.vacuum_line['rate']           = 20.                # Vacuum line application rate [m/hr/man]
        self.vacuum_line['length']         = component_parameters['perimeter_noroot']   # Vacuum line length [m]
        # Application tack tape             
        self.tack_tape['n_pers']           = 0.5 * team_size    # Number of personnel involved in the operation
        self.tack_tape['length']           = component_parameters['perimeter'] # Tack tape length [m]
        self.tack_tape['rate']             = 360.               # Tack tape application rate [m/hr/man]
        # Application vacuum bag            
        self.vacuum_bag['n_pers']          = team_size          # Number of personnel involved in the operation
        self.vacuum_bag['rate']            = 7.5                # Vacuum bag application rate [m2/hr/man]
        # Pull full vacuum                  
        self.vacuum_pull['n_pers']         = team_size          # Number of personnel involved in the operation
        self.vacuum_pull['rate']           = 360.               # Vacuum pull rate [m2/hr/man]
        # Check for leackages               
        self.leak_chk['n_pers']            = team_size          # Number of personnel involved in the operation
        self.leak_chk['rate']              = 180.               # Leak_chk rate [m/hr/man]
        # Infusion                         
        self.infusion['n_pers']            = 3.                 # Number of personnel involved in the operation
        p0                                 = 15.972             # p0 of the polynomial fit 
        p1                                 = 3.1484             # p1 of the polynomial fit 
        p2                                 = -0.0568            # p2 of the polynomial fit 
        p3                                 = 0.0004             # p3 of the polynomial fit 
        self.infusion['time']              = (p0 + p1 * component_parameters['blade_length'] + p2 * component_parameters['blade_length']**2 + p3 * component_parameters['blade_length']**3) / 60. # Infusion time [hr]
        # Cure                             
        self.cure['n_pers']                = 1.                 # Number of personnel involved in the operation
        self.cure['time']                  = 3.                 # Curing time [hr]
        # Debag                            
        self.debag['n_pers']               = team_size          # Number of personnel involved in the operation
        self.debag['area']                 = component_parameters['area'] # Area to be debagged 
        self.debag['rate']                 = team_size          # Debag rate [m2/hr/man]
        # Demold                           
        self.demold['n_pers']              = team_size          # Number of personnel involved in the operation
        self.demold['time']                = 0.                 # Demold time [hr]
        

        
        
        for var in process.keys():
            setattr(self, var, process[var])

class assembly_process(object):
    
    def assembly_steps(self):
        # Remove non-sand / prep (LP)       
        self.remove_nonsand_prep_lp['labor'] , self.remove_nonsand_prep_lp['ct'] = compute_labor_ct(self.remove_nonsand_prep_lp['n_pers'], \
            self.remove_nonsand_prep_lp['length'] , self.remove_nonsand_prep_lp['rate'], 0 , 0)
        # Remove non-sand / prep (HP)       
        self.remove_nonsand_prep_hp['labor'] , self.remove_nonsand_prep_hp['ct'] = compute_labor_ct(self.remove_nonsand_prep_hp['n_pers'], \
            self.remove_nonsand_prep_hp['length'] , self.remove_nonsand_prep_hp['rate'], 0 , 0)
        # Insert SWs into fixture       
        self.insert_sw['labor'] , self.insert_sw['ct']                           = compute_labor_ct(self.insert_sw['n_pers'], self.insert_sw['length'] , 1, 0 , 1)
        # Dry fit - boundary mark
        self.dry_fit['labor'] , self.dry_fit['ct']                               = compute_labor_ct(self.dry_fit['n_pers'], self.dry_fit['length'] , self.dry_fit['rate'], 0 , 0)
        # Dispense lower adhesive
        self.low_adhesive['labor'] , self.low_adhesive['ct']                     = compute_labor_ct(self.low_adhesive['n_pers'], self.low_adhesive['length'] , self.low_adhesive['rate'], 0 , 0)
        # Bond SWs - lower
        self.bond_sw_low['labor'] , self.bond_sw_low['ct']                       = compute_labor_ct(self.bond_sw_low['n_pers'], 0 , 1, self.bond_sw_low['time'] , 1)
        # Fillet SW bonds
        self.fillet_sw_low['labor'] , self.fillet_sw_low['ct']                   = compute_labor_ct(self.fillet_sw_low['n_pers'], self.fillet_sw_low['length'], self.fillet_sw_low['rate'], 0 , 0)
        # Cure - lower adhesive
        self.cure_low['labor']                                                   = 0.
        self.cure_low['ct']    = self.cure_low['time']
        # Remove fixture
        self.remove_fixture['labor'] , self.remove_fixture['ct']                 = compute_labor_ct(self.remove_fixture['n_pers'], 0, 1 , self.remove_fixture['time'] , 1)
        # Dry fit - upper 
        self.dry_fit_up['labor'] , self.dry_fit_up['ct']                         = compute_labor_ct(self.dry_fit_up['n_pers'], self.dry_fit_up['length'], self.dry_fit_up['rate'], 0 , 0)
        self.dry_fit_up['ct']                                                    = self.dry_fit_up['ct'] + 2 * self.close_mold['time']
        # (SW height rework)
        self.sw_height_rework['labor']                                           = 0
        self.sw_height_rework['ct']                                              = 0
        # Dispense upper/perim adhesive
        self.up_adhesive['labor'] , self.up_adhesive['ct']                       = compute_labor_ct(self.up_adhesive['n_pers'], self.up_adhesive['length'], self.up_adhesive['rate'], 0. , 0)
        # Close mold
        self.close_mold['labor'] , self.close_mold['ct']                         = compute_labor_ct(self.close_mold['n_pers'], 0, 1, self.close_mold['time'] , 1)
        # Install shear clips
        self.shear_clips['labor'] , self.shear_clips['ct']                       = compute_labor_ct(self.shear_clips['n_pers'], self.shear_clips['length'], self.shear_clips['rate'], 0. , 0)
        # Cure - entire
        self.cure_entire['labor']                                                = 0. 
        self.cure_entire['ct']                                                   = self.cure_entire['time']
        # Open mold
        self.open_mold['labor'] , self.open_mold['ct']                           = compute_labor_ct(self.open_mold['n_pers'], 0, 1, self.open_mold['time'] , 1)

class assembly_labor(assembly_process):

    def __init__(self, blade_parameters, team_size, process = {} ):
        # # Assembly labor        
        self.remove_nonsand_prep_lp        = {}
        self.remove_nonsand_prep_hp        = {}
        self.insert_sw                     = {}
        self.dry_fit                       = {}
        self.low_adhesive                  = {}
        self.bond_sw_low                   = {}
        self.fillet_sw_low                 = {}
        self.cure_low                      = {}
        self.remove_fixture                = {}
        self.dry_fit_up                    = {}
        self.sw_height_rework              = {}
        self.up_adhesive                   = {}
        self.close_mold                    = {}
        self.shear_clips                   = {}
        self.cure_entire                   = {}
        self.open_mold                     = {}
        
        # Remove non-sand / prep (LP)       
        self.remove_nonsand_prep_lp['n_pers']   = team_size / 2.    # Number of personnel involved in the operation
        self.remove_nonsand_prep_lp['length']   = sum(blade_parameters['sw_length']) + blade_parameters['perimeter_noroot'] # Length where to remove sand [m]
        self.remove_nonsand_prep_lp['rate']     = 60.               # Rate of sand removal [m/hr/man]
        # Remove non-sand / prep (HP)       
        self.remove_nonsand_prep_hp['n_pers']   = team_size / 2.    # Number of personnel involved in the operation
        self.remove_nonsand_prep_hp['length']   = self.remove_nonsand_prep_lp['length'] # Length where to remove sand [m]
        self.remove_nonsand_prep_hp['rate']     = 60.               # Rate of sand removal [m/hr/man]
        # Insert SWs into fixture            
        self.insert_sw['n_pers']                = team_size              # Number of personnel involved in the operation
        self.insert_sw['time']                  = 0.25                   # Time to install the shear web in the mold for webs shorter than 50 meters [hr]
        self.insert_sw['rate']                  = 0.0167                 # Extra time per meter to install the shear web in the mold for webs longer than 50 meters [hr/m]
        insert_sw_len = np.zeros(len(blade_parameters['sw_length']))
        for i_web in range(len(blade_parameters['sw_length'])):     # Loop for all existing webs
            insert_sw_len[i_web]                = self.insert_sw['time'] - (self.insert_sw['rate']*(50. - blade_parameters['sw_length'][i_web]))
        self.insert_sw['length']                = sum(insert_sw_len)
        # Dry fit - boundary mark          
        self.dry_fit['n_pers']                  = team_size              # Number of personnel involved in the operation
        self.dry_fit['length']                  = sum(blade_parameters['sw_length']) # Length where to dry fit [m]
        self.dry_fit['rate']                    = 60.                    # Rate of dry fit [m/hr/man]
        # Dispense lower adhesive          
        self.low_adhesive['n_pers']        = team_size              # Number of personnel involved in the operation
        self.low_adhesive['length']        = sum(blade_parameters['sw_length']) # Length where to dispose adhesive [m]
        self.low_adhesive['rate']          = 60.                    # Rate to dispose adhesive [m/hr/man]
        # Bond SWs - lower           
        self.bond_sw_low['n_pers']         = team_size              # Number of personnel involved in the operation
        self.bond_sw_low['time']           = blade_parameters['n_webs'] * 0.25 # Time to bond the shear webs in the mold [hr]
        # Fillet SW bonds            
        self.fillet_sw_low['n_pers']       = team_size              # Number of personnel involved in the operation
        self.fillet_sw_low['length']       = 2. * sum(blade_parameters['sw_length'])
        self.fillet_sw_low['rate']         = 180.                   # Rate to fillett the webs [fillet/hr/man]
        # Cure - lower adhesive         
        self.cure_low['n_pers']            = 0.                     # Number of personnel involved in the operation
        self.cure_low['time']              = 2.                     # Time to cure the adhesive [hr]
        # Remove fixture             
        self.remove_fixture['n_pers']      = team_size              # Number of personnel involved in the operation
        self.remove_fixture['time']        = 0.25                   # Time to remove the fixture [hr]
        # Dry fit - upper            
        self.dry_fit_up['n_pers']          = team_size              # Number of personnel involved in the operation
        self.dry_fit_up['time']            = 0.0833                 # Time close or open the mold [hr]
        self.dry_fit_up['rate']            = 15.                    # Dry fit rate of the shear webs [m/hr/man]
        self.dry_fit_up['length']          = sum(blade_parameters['sw_length'])
        # (SW height rework)             
        self.sw_height_rework['n_pers']    = 0.                     # Number of personnel involved in the operation
        # Dispense upper/perim adhesive    
        self.up_adhesive['n_pers']         = team_size              # Number of personnel involved in the operation
        self.up_adhesive['length']         = sum(blade_parameters['sw_length']) + blade_parameters['perimeter_noroot']
        self.up_adhesive['rate']           = 60.                    # Rate to dispose adhesive [m/hr/man]
        # Close mold             
        self.close_mold['n_pers']          = team_size              # Number of personnel involved in the operation
        self.close_mold['time']            = self.dry_fit_up['time']# Time close or open the mold [hr]
        # Install shear clips            
        self.shear_clips['n_pers']         = team_size              # Number of personnel involved in the operation
        self.shear_clips['%vert']          = 50.                    # Vertical Fraction of the shear webs with shear clip coverage [%]
        self.shear_clips['%span']          = 10.                    # Spanwise fraction of the shear webs with shear clip coverage [%]
        self.shear_clips['rate']           = 4.                     # Shear clip lay rate [m/hr/man]
        self.shear_clips['length']         = sum(blade_parameters['sw_length']) * self.shear_clips['%vert'] * 4. / 100. + blade_parameters['length'] * self.shear_clips['%span'] * 2. / 100. # Length where to install shear clips [m]
        # Cure - entire          
        self.cure_entire['n_pers']         = 0.                     # Number of personnel involved in the operation
        self.cure_entire['time']           = 2.                     # Curing time
        # Open mold          
        self.open_mold['n_pers']           = 1.                     # Number of personnel involved in the operation
        self.open_mold['time']             = 0.0833                 # Time close or open the mold [hr]
        
        for var in process.keys():
            setattr(self, var, process[var])

class demold_process(object):
    
    def demold_steps(self):
        
        # Cool-down period
        self.cool_down['ct']               = self.cool_down['time']
        self.cool_down['labor']            = self.cool_down['ct'] * self.cool_down['n_pers']
        # Placement of lift straps
        if self.lift_straps['length'] <= 40.:
            self.lift_straps['ct']         = self.lift_straps['time']
        else:
            self.lift_straps['ct']         = self.lift_straps['time'] + (self.lift_straps['length'] - 40.) * self.lift_straps['rate']
        self.lift_straps['labor']          = self.lift_straps['ct'] * self.lift_straps['n_pers']
        # Transfer to blade cart
        if self.transfer2cart['length'] <= 60.:
            self.transfer2cart['ct']       = self.transfer2cart['time']
        else:                              
            self.transfer2cart['ct']       = self.transfer2cart['time'] + (self.transfer2cart['length'] - 60.) * self.transfer2cart['rate']
        self.transfer2cart['labor']        = self.transfer2cart['ct'] * self.transfer2cart['n_pers']
        # Move blade to finishing area
        if self.move2finishing['length'] <= 60.:
            self.move2finishing['ct']      = self.move2finishing['time']
        else:                              
            self.move2finishing['ct']      = self.move2finishing['time'] + (self.move2finishing['length'] - 60.) * self.move2finishing['rate']
        self.move2finishing['labor']       = self.move2finishing['ct'] * self.move2finishing['n_pers']

class demold_labor(demold_process):

    def __init__(self, blade_parameters, process = {} ):
        # # Demold labor        
        self.cool_down                     = {}
        self.lift_straps                   = {}
        self.transfer2cart                 = {}
        self.move2finishing                = {}
        
        # Cool-down period  
        self.cool_down['n_pers']           = 0.        # Number of personnel involved in the operation
        self.cool_down['time']             = 1.        # Time to cool down [hr]
        # Placement of lift straps   
        self.lift_straps['n_pers']         = 4.        # Number of personnel involved in the operation
        self.lift_straps['length']         = blade_parameters['length'] # Length of the blade [m]
        self.lift_straps['rate']           = 0.0083    # Premium lift straps for blades longer than 40 m [hr/m]
        self.lift_straps['time']           = 0.5       # Strap baseline [hr]
        # Transfer to blade cart           
        self.transfer2cart['n_pers']       = 4.        # Number of personnel involved in the operation
        self.transfer2cart['time']         = 0.25      # Time to move blade to the cart [hr]
        self.transfer2cart['length']       = blade_parameters['length'] # Length of the blade [m]
        self.transfer2cart['rate']         = 0.0167    # Extra time per meter length needed to move blades longer than 60 m [hr/m]
        # Remove non-sand / prep (LP)                
        self.move2finishing['n_pers']      = 4.        # Number of personnel involved in the operation
        self.move2finishing['time']        = 0.1667    # Time to move blade to the cart [hr]
        self.move2finishing['length']      = blade_parameters['length'] # Length of the blade [m]
        self.move2finishing['rate']        = 0.0056    # Extra time per meter length needed to move blades longer than 60 m [hr/m]
        

        for var in process.keys():
            setattr(self, var, process[var])

class trim_process(object):
    
    def trim_steps(self):
        
        # Move blade into trim booth
        self.move2trim['ct']               = self.move2trim['time']
        self.move2trim['labor']            = self.move2trim['ct'] * self.move2trim['n_pers']
        # Trim blade
        self.trim['labor'] , self.trim['ct']  = compute_labor_ct(self.trim['n_pers'], self.trim['length'], self.trim['rate'], 0 , 0)
        # Move blade out of trim booth
        self.move_out['ct']                = self.move_out['time']
        self.move_out['labor']             = self.move_out['ct'] * self.move_out['n_pers']

class trim_labor(trim_process):

    def __init__(self, blade_parameters, process = {} ):
        # # trim labor        
        self.move2trim                     = {}
        self.trim                          = {}
        self.move_out                      = {}
        
        # Move blade into trim booth  
        self.move2trim['n_pers']           = 3.       # Number of personnel involved in the operation
        self.move2trim['time']             = 0.5      # Time to move the blade to the trim booth [hr]
        # Trim blade                       
        self.trim['n_pers']                = 6.       # Number of personnel involved in the operation
        self.trim['length']                = blade_parameters['perimeter_noroot'] # Length of the blade [m]
        self.trim['rate']                  = 10.      # Trim rate [m/hr/man]
        # Move blade out of trim booth            
        self.move_out['n_pers']            = 3.       # Number of personnel involved in the operation
        self.move_out['time']              = 0.5      # Time to move blade to out of the trim booth [hr]
        

        for var in process.keys():
            setattr(self, var, process[var])

class overlay_process(object):
    
    def overlay_steps(self):
        
        # Move blade to station
        self.move2station['ct']            = self.move2station['time']
        self.move2station['labor']         = self.move2station['ct'] * self.move2station['n_pers']
        # Rotate blade 90
        if self.rotate90deg['length'] <= 40.:
            self.rotate90deg['ct']         = self.rotate90deg['time']
        else:                              
            self.rotate90deg['ct']         = self.rotate90deg['time'] + (self.rotate90deg['length'] - 40.) * self.rotate90deg['rate']
        self.rotate90deg['labor']          = self.rotate90deg['ct'] * self.rotate90deg['n_pers']
        # Place staging                    
        self.place_staging['ct']           = self.place_staging['time']
        self.place_staging['labor']        = self.place_staging['ct'] * self.place_staging['n_pers']
        # Repair over/under-bite           
        self.repair['ct']                  = 0.
        self.repair['labor']               = 0.
        # Install overlay                  
        self.inst_overlay['labor']         = self.inst_overlay['length'] / self.inst_overlay['rate']
        self.inst_overlay['ct']            = self.inst_overlay['labor'] / self.inst_overlay['n_pers']
        # Vacuum bag overlay               
        self.vacuum_bag['ct']              = self.vacuum_bag['length'] / self.vacuum_bag['rate']
        self.vacuum_bag['labor']           = self.vacuum_bag['ct'] * self.vacuum_bag['n_pers']
        # Cure of overlay                  
        self.cure['ct']                    = self.cure['time']
        self.cure['labor']                 = self.cure['ct'] * self.cure['n_pers']
        # Remove vacuum bag                
        self.remove_bag['ct']              = self.remove_bag['length'] / self.remove_bag['rate']
        self.remove_bag['labor']           = self.remove_bag['ct'] * self.remove_bag['n_pers']

class overlay_labor(overlay_process):

    def __init__(self, blade_parameters, process = {} ):
        # # Overlay labor        
        self.move2station                  = {}
        self.rotate90deg                   = {}
        self.place_staging                 = {}
        self.repair                        = {}
        self.inst_overlay                  = {}
        self.vacuum_bag                    = {}
        self.cure                          = {}
        self.remove_bag                    = {}
        
        coverage    = 20.     # [%] Percentage of overlay coverage
        OL_ply      = coverage / 100.*blade_parameters['length'] # Longest available overlay ply [m]
        avg_taper   = 0.5    # [m] Average taper on ends
        totalOL     = 2.*(OL_ply + (OL_ply - (2. * avg_taper)) + (OL_ply - (4. * avg_taper)) + (OL_ply - (6. * avg_taper)) + \
                        (OL_ply - (8. * avg_taper)) + (OL_ply - (10. * avg_taper)) + (OL_ply - (12. * avg_taper)) + (OL_ply - (14. * avg_taper)))
        
        # Move blade to station       
        self.move2station['n_pers']        = 3.       # Number of personnel involved in the operation
        self.move2station['time']          = 0.5      # Time to move the blade to the overlay station [hr]
        # Rotate blade 90               
        self.rotate90deg['n_pers']         = 3.       # Number of personnel involved in the operation
        self.rotate90deg['length']         = blade_parameters['length'] # Blade length [m]
        self.rotate90deg['time']           = 0.5      # Fixed time to rotate a blade shorter than 40 m [hr]
        self.rotate90deg['rate']           = 0.0083   # Extra time per meter length needed to rotate blades longer than 40 m [hr/m]
        # Place staging                    
        self.place_staging['n_pers']       = 6.       # Number of personnel involved in the operation
        self.place_staging['time']         = 0.25     # Time for place staging [hr]
        # Repair over/under-bite           
        self.repair['n_pers']              = 0.       # Number of personnel involved in the operation
        # Install overlay                  
        self.inst_overlay['n_pers']        = 12.       # Number of personnel involved in the operation
        self.inst_overlay['length']        = totalOL  # Length of overlay ply [m]
        self.inst_overlay['rate']          = 4.     # Rate to install overlay [m/hr/man]
        # Vacuum bag overlay               
        self.vacuum_bag['n_pers']          = 6.       # Number of personnel involved in the operation
        self.vacuum_bag['rate']            = 30.      # Rate to place vacuum bag [m/hr]
        self.vacuum_bag['length']          = 2 * OL_ply # 2x longest overlay ply [m]
        # Cure of overlay                  
        self.cure['n_pers']                = 6.       # Number of personnel involved in the operation
        self.cure['time']                  = 1.       # Curing time [hr]
        # Remove vacuum bag                
        self.remove_bag['n_pers']          = 3.       # Number of personnel involved in the operation
        self.remove_bag['rate']            = 60.      # Rate to remove vacuum bag [m/hr]
        self.remove_bag['length']          = OL_ply   # 2x longest overlay ply [m]
        

        for var in process.keys():
            setattr(self, var, process[var])

class post_cure_process(object):
    
    def post_cure_steps(self):
        
        
        # Place blade in oven carts
        if self.move2cart['length'] <= 40.:
            self.move2cart['ct']           = self.move2cart['time']
        else:                              
            self.move2cart['ct']           = self.move2cart['time'] + (self.move2cart['length'] - 40.) * self.move2cart['rate']
        self.move2cart['labor']            = self.move2cart['ct'] * self.move2cart['n_pers']
        # Move blade into oven             
        self.move2oven['ct']               = self.move2oven['time']
        self.move2oven['labor']            = self.move2oven['ct'] * self.move2oven['n_pers']
        # Post cure dwell                  
        self.post_cure['ct']               = self.post_cure['time']
        self.post_cure['labor']            = self.post_cure['ct'] * self.post_cure['n_pers']
        # Move blade out of oven           
        self.move_out['ct']                = self.move_out['time']
        self.move_out['labor']             = self.move_out['ct'] * self.move_out['n_pers']
        # Cool-down dwell                 
        self.cool_down['ct']               = self.cool_down['time']
        self.cool_down['labor']            = self.cool_down['ct'] * self.cool_down['n_pers']

class post_cure_labor(post_cure_process):

    def __init__(self, blade_parameters, process = {} ):
        # # Post_cure labor        
        self.move2cart                     = {}
        self.move2oven                     = {}
        self.post_cure                     = {}
        self.move_out                      = {}
        self.cool_down                     = {}


        # Place blade in oven carts     
        self.move2cart['n_pers']           = 3.       # Number of personnel involved in the operation
        self.move2cart['length']           = blade_parameters['length'] # Blade length [m]
        self.move2cart['time']             = 0.25     # Fixed time to move a blade shorter than 40 m [hr]
        self.move2cart['rate']             = 0.0042   # Extra time per meter length needed to move blades longer than 40 m [hr/m]
        # Move blade into oven             
        self.move2oven['n_pers']           = 2.       # Number of personnel involved in the operation
        self.move2oven['time']             = 0.1667   # Time to move blade into the oven [hr]
        # Post cure dwell                  
        self.post_cure['n_pers']           = 0.       # Number of personnel involved in the operation
        self.post_cure['time']             = 8.       # Time of post-curing [hr]
        # Move blade out of oven           
        self.move_out['n_pers']            = 2.       # Number of personnel involved in the operation
        self.move_out['time']              = 0.1667   # Time to move blade out of the oven [hr]
        # Cool-down dwell                  
        self.cool_down['n_pers']           = 0.       # Number of personnel involved in the operation
        self.cool_down['time']             = 1.       # Time needed to cool the blade down [hr]
        
        for var in process.keys():
            setattr(self, var, process[var])

class cut_drill_process(object):
    
    def cut_drill_steps(self):
        
        # Move blade / place in saddles
        if self.move2saddles['length'] <= 40.:
            self.move2saddles['ct']        = self.move2saddles['time']
        else:
            self.move2saddles['ct']        = self.move2saddles['time'] + (self.move2saddles['length'] - 40.) * self.move2saddles['rate']
        self.move2saddles['labor']         = self.move2saddles['ct'] * self.move2saddles['n_pers']
        # Check level / point of reference
        self.checklevel['ct']              = self.checklevel['time']
        self.checklevel['labor']           = self.checklevel['ct'] * self.checklevel['n_pers']
        # Machine trim blade root
        self.trim_root['ct']               = self.trim_root['root_perim'] / self.trim_root['rate']
        self.trim_root['labor']            = self.trim_root['ct'] * self.trim_root['n_pers']
        # Clear trim excess
        self.clear_excess['ct']            = self.clear_excess['time']
        self.clear_excess['labor']         = self.clear_excess['ct'] * self.clear_excess['n_pers']
        # Machine cut axial and radial holes
        self.cut_holes['ct']               = self.cut_holes['n_bolts'] * self.cut_holes['time_per_hole']
        self.cut_holes['labor']            = self.cut_holes['ct'] * self.cut_holes['n_pers']
        # Clear drill excess
        self.clear_excess2['ct']           = self.clear_excess2['time']
        self.clear_excess2['labor']        = self.clear_excess2['ct'] * self.clear_excess2['n_pers']

class cut_drill_labor(cut_drill_process):

    def __init__(self, blade_parameters, process = {} ):
        # # Cut_drill labor        
        self.move2saddles                  = {}
        self.checklevel                    = {}
        self.trim_root                     = {}
        self.clear_excess                  = {}
        self.cut_holes                     = {}
        self.clear_excess2                 = {}

        # Move blade / place in saddles     
        self.move2saddles['n_pers']        = 3.       # Number of personnel involved in the operation
        self.move2saddles['length']        = blade_parameters['length'] # Blade length [m]
        self.move2saddles['time']          = 0.1667   # Fixed time to move a blade shorter than 40 m [hr]
        self.move2saddles['rate']          = 0.0083   # Extra time per meter length needed to move blades longer than 40 m [hr/m]
        # Check level / point of reference
        self.checklevel['n_pers']          = 2.       # Number of personnel involved in the operation
        self.checklevel['time']            = 0.333    # Time to check the level and the point of reference [hr]
        # Machine trim blade root          
        self.trim_root['n_pers']           = 2.       # Number of personnel involved in the operation
        self.trim_root['root_perim']       = blade_parameters['root_perim']  # Blade root perimeter [m]
        self.trim_root['rate']             = 6.       # Root cutting rate [m/hr]
        # Clear trim excess       
        self.clear_excess['n_pers']        = 2.       # Number of personnel involved in the operation
        self.clear_excess['time']          = 0.25     # Time to clear trim excess [hr]
        # Machine cut axial and radial holes       
        self.cut_holes['n_pers']           = 2.       # Number of personnel involved in the operation
        self.cut_holes['n_bolts']          = blade_parameters['n_bolts']   # Number of root bolts [-]
        self.cut_holes['time_per_hole']    = 0.0333   # Time per hole [hr]
        # Clear drill excess
        self.clear_excess2['n_pers']       = 2.       # Number of personnel involved in the operation
        self.clear_excess2['time']         = 0.25     # Time needed to clear drill excess [hr]
        

        for var in process.keys():
            setattr(self, var, process[var])

class root_install_process(object):
    
    def root_install_steps(self):
        
        # Move blade and place it in carts
        if self.move2cart['length'] <= 40.:
            self.move2cart['ct']           = self.move2cart['time']
        else:                              
            self.move2cart['ct']           = self.move2cart['time'] + (self.move2cart['length'] - 40.) * self.move2cart['rate']
        self.move2cart['labor']            = self.move2cart['ct'] * self.move2cart['n_pers']
        # Install barrel nuts              
        self.barrel_nuts['labor']          = self.barrel_nuts['n_bolts'] / self.barrel_nuts['rate']
        self.barrel_nuts['ct']             = self.barrel_nuts['labor'] / self.barrel_nuts['n_pers']
        # Apply root band                  
        self.root_band['ct']               = self.root_band['root_perim'] / self.root_band['rate']
        self.root_band['labor']            = self.root_band['ct'] * self.root_band['n_pers']

class root_install_labor(root_install_process):

    def __init__(self, blade_parameters, process = {} ):
        # # Root_install labor        
        self.move2cart                     = {}
        self.barrel_nuts                   = {}
        self.root_band                     = {}

        # Move blade and place it in carts    
        self.move2cart['n_pers']           = 3.       # Number of personnel involved in the operation
        self.move2cart['length']           = blade_parameters['length'] # Blade length [m]
        self.move2cart['time']             = 0.1667   # Fixed time to move a blade shorter than 40 m [hr]
        self.move2cart['rate']             = 0.0083   # Extra time per meter length needed to move blades longer than 40 m [hr/m]
        # Install barrel nuts
        self.barrel_nuts['n_pers']         = 2.       # Number of personnel involved in the operation
        self.barrel_nuts['n_bolts']        = blade_parameters['n_bolts']   # Number of root bolts [-]
        self.barrel_nuts['rate']           = 24.      # Nut install rate [#/hr]
        # Apply root band          
        self.root_band['n_pers']           = 2.       # Number of personnel involved in the operation
        self.root_band['root_perim']       = blade_parameters['root_perim']  # Blade root perimeter [m]
        self.root_band['rate']             = 6.       # Apply root band rate [m/hr]
        

        
        for var in process.keys():
            setattr(self, var, process[var])

class surface_prep_process(object):
    
    def surface_prep_steps(self):
        
        # Move blade carts to surface preparation area  
        self.move2area['ct']               = self.move2area['time']
        self.move2area['labor']            = self.move2area['ct'] * self.move2area['n_pers']
        # Perform surface preparation      
        self.surface_prep['labor']         = self.surface_prep['area'] / self.surface_prep['rate']
        self.surface_prep['ct']            = self.surface_prep['labor'] / self.surface_prep['n_pers']

class surface_prep_labor(surface_prep_process):

    def __init__(self, blade_parameters, process = {} ):
        # # Surface preparation labor        
        self.move2area                     = {}
        self.surface_prep                  = {}

        # Move blade carts to surface preparation area    
        self.move2area['n_pers']           = 2.       # Number of personnel involved in the operation
        self.move2area['time']             = 0.1667   # Fixed time to move the blade [hr]
        # Perform surface preparation      
        self.surface_prep['n_pers']        = 8.       # Number of personnel involved in the operation
        self.surface_prep['area']          = blade_parameters['area']   # Total blade outer area [m2]
        self.surface_prep['rate']          = 6.       # Surface preparation rate [m2/hr]
        

        for var in process.keys():
            setattr(self, var, process[var])

class paint_process(object):
    
    def paint_steps(self):
    
        # Move blade carts to painting area  
        self.move2area['ct']               = self.move2area['time']
        self.move2area['labor']            = self.move2area['ct'] * self.move2area['n_pers']
        # Apply primer                     
        self.primer['ct']                  = self.primer['area'] / self.primer['rate']
        self.primer['labor']               = self.primer['ct'] * self.primer['n_pers']
        # Cure / tackify                   
        self.cure['ct']                    = self.cure['time']
        self.cure['labor']                 = self.cure['ct'] * self.cure['n_pers']
        # Apply top coat                   
        self.top_coat['ct']                = self.top_coat['area'] / self.top_coat['rate']
        self.top_coat['labor']             = self.top_coat['ct'] * self.top_coat['n_pers']
        # Cure                             
        self.cure2['ct']                   = self.cure2['time']
        self.cure2['labor']                = self.cure2['ct'] * self.cure2['n_pers']

class paint_labor(paint_process):

    def __init__(self, blade_parameters, process = {} ):
        # # Painting labor        
        self.move2area                     = {}
        self.primer                        = {}
        self.cure                          = {}
        self.top_coat                      = {}
        self.cure2                         = {}

        # Move blade carts to painting area    
        self.move2area['n_pers']           = 2.        # Number of personnel involved in the operation
        self.move2area['time']             = 0.1667    # Fixed time to move the blade [hr]
        # Apply primer                     
        self.primer['n_pers']              = 4.        # Number of personnel involved in the operation
        self.primer['area']                = blade_parameters['area']   # Total blade outer area [m2]
        self.primer['rate']                = 480.      # Rate of application  of the primer  - spray rate [m2/hr]
        # Cure / tackify                   
        self.cure['n_pers']                = 4.        # Number of personnel involved in the operation
        self.cure['time']                  = 1.        # Fixed time to cure / tackify the primer dwell
        # Apply top coat                   
        self.top_coat['n_pers']            = 4.        # Number of personnel involved in the operation
        self.top_coat['area']              = blade_parameters['area']   # Total blade outer area [m2]
        self.top_coat['rate']              = 480.      # Rate of application  of the top coat - spray rate [m2/hr]
        # Cure                             
        self.cure2['n_pers']               = 4.        # Number of personnel involved in the operation
        self.cure2['time']                 = 3.        # Fixed time for the paint to dwell
        

        
        for var in process.keys():
            setattr(self, var, process[var])

class surface_finish_process(object):
    
    def surface_finish_steps(self):
        
        # Move blade carts to surface finishing area  
        self.move2area['ct']               = self.move2area['time']
        self.move2area['labor']            = self.move2area['ct'] * self.move2area['n_pers']
        # Perform surface finishing
        self.surface_finish['labor']       = self.surface_finish['area'] / self.surface_finish['rate']
        self.surface_finish['ct']          = self.surface_finish['labor'] / self.surface_finish['n_pers']

class surface_finish_labor(surface_finish_process):

    def __init__(self, blade_parameters, process = {} ):
        # # Surface finishing labor        
        self.move2area                     = {}
        self.surface_finish                = {}

        # Move blade carts to surface finishing area    
        self.move2area['n_pers']           = 2.        # Number of personnel involved in the operation
        self.move2area['time']             = 0.1667    # Fixed time to move the blade [hr]
        # Perform surface finishing
        self.surface_finish['n_pers']      = 3.        # Number of personnel involved in the operation
        self.surface_finish['area']        = blade_parameters['area']   # Total blade outer area [m2]
        self.surface_finish['rate']        = 30.       # Surface finishing rate [m2/hr]
        
        for var in process.keys():
            setattr(self, var, process[var])

class weight_balance_process(object):
    
    def weight_balance_steps(self):

        # Move blade / place in saddles
        if self.move2saddles['length'] <= 40.:
            self.move2saddles['ct']        = self.move2saddles['time']
        else:
            self.move2saddles['ct']        = self.move2saddles['time'] + (self.move2saddles['length'] - 40.) * self.move2saddles['rate']
        self.move2saddles['labor']         = self.move2saddles['ct'] * self.move2saddles['n_pers']
         # Check balance
        self.check_balance['ct']           = self.check_balance['time']
        self.check_balance['labor']        = self.check_balance['ct'] * self.check_balance['n_pers']
        # Drill holes into balance boxes
        self.holes_boxes['ct']             = self.holes_boxes['time']
        self.holes_boxes['labor']          = self.holes_boxes['ct'] * self.holes_boxes['n_pers']
        # Mix balance box filler
        self.mix_filler['ct']              = self.mix_filler['time']
        self.mix_filler['labor']           = self.mix_filler['ct'] * self.mix_filler['n_pers']
        # Pump filler into balance boxes
        self.pump_filler['ct']             = self.pump_filler['time']
        self.pump_filler['labor']          = self.pump_filler['ct'] * self.pump_filler['n_pers']
        # Plug balance box holes
        self.plug_holes['ct']              = self.plug_holes['time']
        self.plug_holes['labor']           = self.plug_holes['ct'] * self.plug_holes['n_pers']

class weight_balance_labor(weight_balance_process):

    def __init__(self, blade_parameters, process = {} ):
        # # Weight and balance labor        
        self.move2saddles                  = {}
        self.check_balance                 = {}
        self.holes_boxes                   = {}
        self.mix_filler                    = {}
        self.pump_filler                   = {}
        self.plug_holes                    = {}
        
        # Move blade / place in saddles
        self.move2saddles['n_pers']        = 3.       # Number of personnel involved in the operation
        self.move2saddles['length']        = blade_parameters['length'] # Blade length [m]
        self.move2saddles['time']          = 0.1667   # Fixed time to move a blade shorter than 40 m [hr]
        self.move2saddles['rate']          = 0.0083   # Extra time per meter length needed to move blades longer than 40 m [hr/m]
        # Check balance
        self.check_balance['n_pers']       = 2.       # Number of personnel involved in the operation
        self.check_balance['time']         = 0.25     # Time needed [hr]
        # Drill holes into balance boxes
        self.holes_boxes['n_pers']         = 2.       # Number of personnel involved in the operation
        self.holes_boxes['time']           = 0.1667   # Time needed [hr]
        # Mix balance box filler
        self.mix_filler['n_pers']          = 2.       # Number of personnel involved in the operation
        self.mix_filler['time']            = 0.5      # Time needed [hr]
        # Pump filler into balance boxes
        self.pump_filler['n_pers']         = 2.       # Number of personnel involved in the operation
        self.pump_filler['time']           = 1.       # Time needed [hr]
        # Plug balance box holes
        self.plug_holes['n_pers']          = 2.       # Number of personnel involved in the operation
        self.plug_holes['time']            = 0.667    # Time needed [hr]

        
        for var in process.keys():
            setattr(self, var, process[var])

class inspection_process(object):
    
    def inspection_steps(self):
        
        # Move blade and place it in shipping saddles
        if self.move2saddles['length'] <= 40.:
            self.move2saddles['ct']        = self.move2saddles['time']
        else:
            self.move2saddles['ct']        = self.move2saddles['time'] + (self.move2saddles['length'] - 40.) * self.move2saddles['rate']
        self.move2saddles['labor']         = self.move2saddles['ct'] * self.move2saddles['n_pers']
        # Perform final inspection
        self.inspection['labor']           = self.inspection['area'] / self.inspection['rate']
        self.inspection['ct']              = self.inspection['labor'] / self.inspection['n_pers']

class inspection_labor(inspection_process):

    def __init__(self, blade_parameters, process = {} ):
        # # Final inspection labor        
        self.move2saddles                  = {}
        self.inspection                    = {}

        # move blade / place in shipping saddles  
        self.move2saddles['n_pers']        = 3.        # Number of personnel involved in the operation
        self.move2saddles['length']        = blade_parameters['length'] # Blade length [m]
        self.move2saddles['time']          = 0.333     # Fixed time to move a blade shorter than 40 m [hr]
        self.move2saddles['rate']          = 0.0083    # Extra time per meter length needed to move blades longer than 40 m [hr/m]
        # Perform final inspection
        self.inspection['n_pers']          = 2.        # Number of personnel involved in the operation
        self.inspection['area']            = blade_parameters['area']   # Total blade outer area [m2]
        self.inspection['rate']            = 360.      # Surface preparation rate [m2/hr]
        

        for var in process.keys():
            setattr(self, var, process[var])

class shipping_prep_process(object):
    
    def shipping_prep_steps(self):
        
        # Install final root bolts
        self.root_bolts['labor']           = self.root_bolts['n_bolts'] / self.root_bolts['rate']
        self.root_bolts['ct']              = self.root_bolts['labor'] / self.root_bolts['n_pers']
        # Install root plate
        self.root_plate['ct']              = self.root_plate['time']
        self.root_plate['labor']           = self.root_plate['ct'] * self.root_plate['n_pers']
        # Connect LPS
        self.connectLPS['ct']              = self.connectLPS['time']
        self.connectLPS['labor']           = self.connectLPS['ct'] * self.connectLPS['n_pers']
        # Install root cover
        self.root_cover['ct']              = self.root_cover['time']
        self.root_cover['labor']           = self.root_cover['ct'] * self.root_cover['n_pers']
        # Install 0 deg pitch plate
        self.pitch_plate['ct']             = self.pitch_plate['time']
        self.pitch_plate['labor']          = self.pitch_plate['ct'] * self.pitch_plate['n_pers']
        # Apply blade serial number
        self.serial_num['ct']              = self.serial_num['time']
        self.serial_num['labor']           = self.serial_num['ct'] * self.serial_num['n_pers']
        # Remove blade from factory
        if self.remove_blade['length'] <= 50.:
            self.remove_blade['ct']        = self.remove_blade['time']
        else:
            self.remove_blade['ct']        = self.remove_blade['time'] + (self.remove_blade['length'] - 50.) * self.remove_blade['rate']
        self.remove_blade['labor']         = self.remove_blade['ct'] * self.remove_blade['n_pers']

class shipping_prep_labor(shipping_prep_process):

    def __init__(self, blade_parameters, process = {} ):
        # # Shipping preparation labor        
        self.root_bolts                    = {}
        self.root_plate                    = {}
        self.connectLPS                    = {}
        self.root_cover                    = {}
        self.pitch_plate                   = {}
        self.serial_num                    = {}
        self.remove_blade                  = {}
        
        # Install final root bolts
        self.root_bolts['n_pers']          = 2.        # Number of personnel involved in the operation
        self.root_bolts['n_bolts']         = blade_parameters['n_bolts']   # Number of blade root bolts [#]
        self.root_bolts['rate']            = 120.      # Rate of bolts installation [#/hr]
        # Install root plate
        self.root_plate['n_pers']          = 2.        # Number of personnel involved in the operation
        self.root_plate['time']            = 0.4       # Fixed time to install the root plate [hr]
        # Connect LPS
        self.connectLPS['n_pers']          = 2.        # Number of personnel involved in the operation
        self.connectLPS['time']            = 0.5       # Fixed time to connect the LPS [hr]
        # Install root cover
        self.root_cover['n_pers']          = 2.        # Number of personnel involved in the operation
        self.root_cover['time']            = 0.25      # Fixed time to install the root cover [hr]
        # Install 0 pitch plate
        self.pitch_plate['n_pers']         = 2.        # Number of personnel involved in the operation
        self.pitch_plate['time']           = 0.1667    # Fixed time to install the 0 deg pitch plate [hr]
        # Apply blade serial number
        self.serial_num['n_pers']          = 2.        # Number of personnel involved in the operation
        self.serial_num['time']            = 0.333     # Fixed time to apply the blade serial number [hr]
        # Remove blade from factory
        self.remove_blade['n_pers']        = 3.        # Number of personnel involved in the operation
        self.remove_blade['length']        = blade_parameters['length'] # Blade length [m]
        self.remove_blade['time']          = 0.5       # Fixed time to move a blade shorter than 40 m [hr]
        self.remove_blade['rate']          = 0.0042    # Extra time per meter length needed to move blades longer than 40 m [hr/m]
        

        for var in process.keys():
            setattr(self, var, process[var])

def compute_labor_ct(n_workers, action, rate, time, flag):
    if flag:
        labor = n_workers*(action/rate + time)
    else:
        labor = action/rate + time
        
    ct  = labor/n_workers
    
    return labor, ct

def compute_total_labor_ct(data_struct, name, verbose, no_contribution2ct = []):
    
    process = data_struct.__dict__.keys()
    labor_total_per_process = 0.
    ct_total_per_process    = 0.
    if verbose:
        print('\n----------')
    for var in process:
        data = getattr(data_struct, var)
        labor_total_per_process += data['labor']
        if verbose:
            print('Activity: ' + var)
            print('labor: {:8.2f} hr \t \t --- \t \t ct: {:8.2f} hr'.format(float(data['labor']),float(data['ct'])))
        if  var not in no_contribution2ct:
            ct_total_per_process += data['ct']
    if verbose:
        print('\n' + name + ':')
        print('labor: {:8.2f} hr \t \t --- \t \t ct: {:8.2f} hr'.format(labor_total_per_process , float(ct_total_per_process)))
    return labor_total_per_process , ct_total_per_process

class virtual_factory(object):


    def __init__(self, blade_specs , operation, gating_ct, non_gating_ct, options):
        
        self.options = options
        
        # Blade inputs
        self.n_webs                           = blade_specs['n_webs']
                                              
        # Financial parameters                
        self.wage                             = 20.       # [$] Wage of an unskilled worker
        self.beni                             = 30.4      # [%] Benefits on wage and salary
        self.overhead                         = 30.       # [%] Labor overhead
        self.crr                              = 10.       # [%] Capital recovery rate
        self.wcp                              = 3.        # [month] Working capital period - amount of time it takes to turn the net current assets and current liabilities into cash
        self.p_life                           = 1         # [yr] Length of production run
        self.rejr                             = 0.25      # [%] Part reject rate per process
                                              
        # Productive lives                    
        self.building_life                    = 30.       # [yr] Building recovery life
        self.eq_life                          = 10.       # [yr] Equipment recovery life
        self.tool_life                        = 4.        # [yr] Productive tool life
                                              
        # Factory parameters                  
        self.n_blades                         = 1000      # [-] Number of blades that the factory aims at manufacturing
                                              
        self.install_cost                     = 10.       # [%] Installation costs
        self.price_space                      = 800.      # [$/m2] Price of building space
        self.maintenance_cost                 = 4.        # [%] Maintenance costs
        self.electr                           = 0.08 	    # [$/kWh] Price of electricity
        self.hours                            = 24.       # [hr] Working hours per day
        self.days                             = 250.      # [day] Working days per year
        self.avg_dt                           = 20.       # [%] Average downtime for workers and equipment
                                              
                                              
        # Compute cumulative rejection rate   
        self.cum_rejr                         = np.zeros(len(operation)) # [%]
        self.cum_rejr[-1]                     = 1. - (1. - (self.rejr / 100.))
        for i_op in range(1, len(operation)): 
            self.cum_rejr[-i_op-1]            = 1. - (1. - (self.rejr / 100)) * (1. - self.cum_rejr[-i_op])
        
        
        # Calculate the number of sets of lp and hp skin molds needed
        if self.options['discrete']:
            self.n_set_molds_skins            = np.ceil(self.n_blades * sum(gating_ct) / (1 - self.cum_rejr[5 + self.n_webs]) / (self.hours * self.days)) # [-] Number of skin mold sets (low and high pressure)
        else:
            self.n_set_molds_skins            = self.n_blades * sum(gating_ct) / (1 - self.cum_rejr[5 + self.n_webs]) / (self.hours * self.days) # [-] Number of skin mold sets (low and high pressure)
        
        # Number of parallel processes
        self.parallel_proc                    = np.ones(len(operation)) # [-]
        
        if self.options['discrete']:
            for i_op in range(0, len(operation)):
                self.parallel_proc[i_op]      = np.ceil(self.n_set_molds_skins * non_gating_ct[i_op] / sum(gating_ct) / (1 - self.cum_rejr[i_op]))
            n_molds_root                      = 2 * self.n_set_molds_skins * non_gating_ct[1] / sum(gating_ct) / (1 - self.cum_rejr[1])
            if n_molds_root < 1:
                self.parallel_proc[2]         = 0
            else:
                self.parallel_proc[1]         = np.ceil(self.n_set_molds_skins * non_gating_ct[ 1] / sum(gating_ct) / (1 - self.cum_rejr[1]))
                self.parallel_proc[2]         = np.ceil(self.n_set_molds_skins * non_gating_ct[ 2] / sum(gating_ct) / (1 - self.cum_rejr[2]))
            for i_web in range(self.n_webs):    
                self.parallel_proc[3 + i_web] = np.ceil(2 * self.n_set_molds_skins * non_gating_ct[3 + i_web] / sum(gating_ct)  / (1 - self.cum_rejr[3 + i_web]))
        else:
            for i_op in range(0, len(operation)):
                self.parallel_proc[i_op]      = self.n_set_molds_skins * non_gating_ct[i_op] / sum(gating_ct) / (1 - self.cum_rejr[i_op])
            n_molds_root                      = 2 * self.n_set_molds_skins * non_gating_ct[1] / sum(gating_ct) / (1 - self.cum_rejr[1])
            if n_molds_root < 1:
                self.parallel_proc[2]         = 0
            else:
                self.parallel_proc[1]         = self.n_set_molds_skins * non_gating_ct[ 1] / sum(gating_ct) / (1 - self.cum_rejr[1])
                self.parallel_proc[2]         = self.n_set_molds_skins * non_gating_ct[ 2] / sum(gating_ct) / (1 - self.cum_rejr[2])
            for i_web in range(self.n_webs):  
                self.parallel_proc[3 + i_web] = 2 * self.n_set_molds_skins * non_gating_ct[3 + i_web] / sum(gating_ct)  / (1 - self.cum_rejr[3 + i_web])            
        
        self.parallel_proc[5  + self.n_webs]  = self.n_set_molds_skins
        self.parallel_proc[6  + self.n_webs]  = self.n_set_molds_skins
        self.parallel_proc[7  + self.n_webs]  = self.n_set_molds_skins
        self.parallel_proc[8  + self.n_webs]  = self.n_set_molds_skins 
        
        
        
        # Building space per operation
        delta                                    = 2. #[m] Distance between blades
        self.floor_space                         = np.zeros(len(operation)) # [m2]
        self.floor_space[0]                      = 3. * blade_specs['blade_length'] # [m2] Material cutting        
        self.floor_space[1]                      = self.parallel_proc[ 1] * (delta + blade_specs['root_D']) * (delta + blade_specs['root_preform_length']) # [m2] Infusion root preform lp
        self.floor_space[2]                      = self.parallel_proc[ 2] * (delta + blade_specs['root_D']) * (delta + blade_specs['root_preform_length']) # [m2] Infusion root preform hp
        for i_web in range(self.n_webs):
            self.floor_space[3 + i_web]          = self.parallel_proc[ 3 + i_web] * (delta + blade_specs['length_webs'][i_web]) * (delta + blade_specs['height_webs_start'][i_web]) # [m2] Infusion webs
        self.floor_space[3 + self.n_webs]        = self.parallel_proc[ 3 + self.n_webs] * (delta + blade_specs['length_sc_lp']) * (delta + blade_specs['width_sc_start_lp'])        # [m2] Infusion spar caps
        self.floor_space[4 + self.n_webs]        = self.parallel_proc[ 4 + self.n_webs] * (delta + blade_specs['length_sc_hp']) * (delta + blade_specs['width_sc_start_hp'])        # [m2] Infusion spar caps
        self.floor_space[5 + self.n_webs]        = self.parallel_proc[ 5 + self.n_webs] * (blade_specs['max_chord'] + delta) * (blade_specs['blade_length'] + delta)                # [m2] Infusion skin shell lp
        self.floor_space[6 + self.n_webs]        = self.parallel_proc[ 6 + self.n_webs] * (blade_specs['max_chord'] + delta) * (blade_specs['blade_length'] + delta)                # [m2] Infusion skin shell hp
        self.floor_space[9 + self.n_webs]        = self.parallel_proc[ 9 + self.n_webs] * (blade_specs['max_chord'] + delta) * (blade_specs['blade_length'] + delta)                # [m2] Trim
        self.floor_space[10 + self.n_webs]       = self.parallel_proc[10 + self.n_webs] * (blade_specs['root_D']    + delta) * (blade_specs['blade_length'] + delta)                # [m2] Overlay
        self.floor_space[11 + self.n_webs]       = self.parallel_proc[11 + self.n_webs] * (blade_specs['root_D']    + delta) * (blade_specs['blade_length'] + delta)                # [m2] Post cure
        self.floor_space[12 + self.n_webs]       = self.parallel_proc[12 + self.n_webs] * (blade_specs['max_chord'] + delta) * (blade_specs['blade_length'] + delta)                # [m2] Root cut and drill
        self.floor_space[13 + self.n_webs]       = self.parallel_proc[13 + self.n_webs] * (blade_specs['root_D']    + delta) * (blade_specs['blade_length'] + delta)                # [m2] Root hardware install
        self.floor_space[14 + self.n_webs]       = self.parallel_proc[14 + self.n_webs] * (blade_specs['root_D']    + delta) * (blade_specs['blade_length'] + delta)                # [m2] Surface preparation
        self.floor_space[15 + self.n_webs]       = self.parallel_proc[15 + self.n_webs] * (blade_specs['root_D']    + delta) * (blade_specs['blade_length'] + delta)                # [m2] Paint
        self.floor_space[16 + self.n_webs]       = self.parallel_proc[16 + self.n_webs] * (blade_specs['root_D']    + delta) * (blade_specs['blade_length'] + delta)                # [m2] Surface inspection and finish
        self.floor_space[17 + self.n_webs]       = self.parallel_proc[17 + self.n_webs] * (blade_specs['root_D']    + delta) * (blade_specs['blade_length'] + delta)                # [m2] Weight and balance
        self.floor_space[18 + self.n_webs]       = self.parallel_proc[18 + self.n_webs] * (blade_specs['root_D']    + delta) * (blade_specs['blade_length'] + delta)                # [m2] Inspection
        self.floor_space[19 + self.n_webs]       = self.parallel_proc[19 + self.n_webs] * (blade_specs['root_D']    + delta) * (blade_specs['blade_length'] + delta)                # [m2] Shipping preparation

        
        # Average power consumption during each operation
        Cp          = 1.01812  # [kJ/kg/K] Kalogiannakis et. al 2003 
        Tcure       = 70       # [C]
        Tamb        = 22       # [C]
        OvenCycle   = 7        # [hr]
        EtaOven     = 0.5      # [-]
        
        kJ_per_kg   = Cp * (Tcure-Tamb) / (OvenCycle * 3600) / EtaOven
        
        self.power_consumpt                      = self.floor_space * 250 / self.hours / self.days # [kW] 80000 btu / sq ft
        self.power_consumpt[1]                   = self.power_consumpt[1] + self.parallel_proc[ 1] * blade_specs['mass_root_preform_lp'] * kJ_per_kg # [kW] Root preform lp
        self.power_consumpt[2]                   = self.power_consumpt[2] + self.parallel_proc[ 2] * blade_specs['mass_root_preform_hp'] * kJ_per_kg # [kW] Root preform hp
        for i_web in range(self.n_webs):
            self.power_consumpt[3 + i_web]       = self.power_consumpt[ 3 + i_web] + self.parallel_proc[3 + i_web] * blade_specs['mass_webs'][i_web] * kJ_per_kg # [kW] Root preform hp
        self.power_consumpt[3 + self.n_webs]     = self.power_consumpt[ 3 + self.n_webs] + self.parallel_proc[ 3 + self.n_webs] * blade_specs['mass_sc_lp'] * kJ_per_kg # [kW] Spar cap lp
        self.power_consumpt[4 + self.n_webs]     = self.power_consumpt[ 4 + self.n_webs] + self.parallel_proc[ 4 + self.n_webs] * blade_specs['mass_sc_hp'] * kJ_per_kg # [kW] Spar cap hp
        self.power_consumpt[5 + self.n_webs]     = self.power_consumpt[ 5 + self.n_webs] + self.parallel_proc[ 5 + self.n_webs] * (blade_specs['mass_shell_lp']) * kJ_per_kg # [kW] Shell lp
        self.power_consumpt[6 + self.n_webs]     = self.power_consumpt[ 6 + self.n_webs] + self.parallel_proc[ 6 + self.n_webs] * (blade_specs['mass_shell_hp']) * kJ_per_kg # [kW] Shell hp
        self.power_consumpt[11 + self.n_webs]    = self.power_consumpt[11 + self.n_webs] + self.parallel_proc[11 + self.n_webs] * blade_specs['blade_mass'] * kJ_per_kg # [kW] Post cure
        
        # Tooling investment per station per operation (molds)
        self.tooling_investment                  = np.zeros(len(operation)) # [$]
        price_mold_sqm                           = 5000.
        self.tooling_investment[1]               = price_mold_sqm * self.parallel_proc[1] * blade_specs['area_lp_root']  # [$] Mold of the root preform - lp, cost assumed equal to 50000 $ per meter square of surface
        self.tooling_investment[2]               = price_mold_sqm * self.parallel_proc[2] * blade_specs['area_hp_root']  # [$] Mold of the root preform - hp, cost assumed equal to 50000 $ per meter square of surface
        for i_web in range(self.n_webs):
            self.tooling_investment[3 + i_web]   = price_mold_sqm * self.parallel_proc[3 + i_web] * blade_specs['area_webs_w_flanges'][i_web] # [$] Mold of the webs, cost assumed equal to 10800 $ per meter square of surface
        self.tooling_investment[3 + self.n_webs] = price_mold_sqm * self.parallel_proc[3 + self.n_webs] * blade_specs['area_sc_lp'] # [$] Mold of the low pressure spar cap, cost assumed equal to 10800 $ per meter square of surface
        self.tooling_investment[4 + self.n_webs] = price_mold_sqm * self.parallel_proc[4 + self.n_webs] * blade_specs['area_sc_hp'] # [$] Mold of the high pressure spar cap, cost assumed equal to 10800 $ per meter square of surface
        self.tooling_investment[5 + self.n_webs] = price_mold_sqm * self.parallel_proc[5 + self.n_webs] * blade_specs['area_lpskin_w_flanges']  # [$] Mold of the low pressure skin shell, assumed equal to 9400 $ per meter square of surface
        self.tooling_investment[6 + self.n_webs] = price_mold_sqm * self.parallel_proc[6 + self.n_webs] * blade_specs['area_hpskin_w_flanges']  # [$] Mold of the low pressure skin shell, assumed equal to 9400 $ per meter square of surface        
        
        
        # Equipment investment per station per operation
        self.equipm_investment                   = np.zeros(len(operation)) # [$]     
        self.equipm_investment[0]                =   5000. * self.parallel_proc[ 0] * blade_specs['blade_length']  # [$] Equipment for material cutting is assumed at 5000 $ per meter of blade length
        self.equipm_investment[1]                =  15000. * self.parallel_proc[ 1] * blade_specs['root_D']        # [$] Equipment for root preform infusion is assumed at 15000 $ per meter of blade root diameter
        self.equipm_investment[2]                =  15000. * self.parallel_proc[ 2] * blade_specs['root_D']        # [$] Equipment for root preform infusion is assumed at 15000 $ per meter of blade root diameter
        for i_web in range(self.n_webs):    
            self.equipm_investment[3 + i_web]    =   1700. * self.parallel_proc[ 3 + i_web] * blade_specs['length_webs'][i_web]    # [$] Equipment for webs infusion is assumed at 1700 $ per meter of web length
        self.equipm_investment[3 + self.n_webs]  =   1700. * self.parallel_proc[ 3 + self.n_webs] * blade_specs['length_sc_lp']    # [$] Equipment for spar caps infusion is assumed at 1700 $ per meter of spar cap length
        self.equipm_investment[4 + self.n_webs]  =   1700. * self.parallel_proc[ 4 + self.n_webs] * blade_specs['length_sc_hp']    # [$] Equipment for spar caps infusion is assumed at 1700 $ per meter of spar cap length
        self.equipm_investment[5 + self.n_webs]  =   1600. * self.parallel_proc[ 5 + self.n_webs] * blade_specs['skin_perimeter_wo_root']# [$] Equipment for skins infusion is assumed at 1600 $ per meter of skin perimeter
        self.equipm_investment[6 + self.n_webs]  =   1600. * self.parallel_proc[ 6 + self.n_webs] * blade_specs['skin_perimeter_wo_root']# [$] Equipment for skins infusion is assumed at 1600 $ per meter of skin perimeter
        self.equipm_investment[7 + self.n_webs]  =   6600. * self.parallel_proc[ 7 + self.n_webs] * sum(blade_specs['length_webs'])# [$] Equipment for assembly is assumed equal to 6600 $ per meter of total webs length
        self.equipm_investment[9 + self.n_webs]  =  25000. * self.parallel_proc[ 9 + self.n_webs] * blade_specs['blade_length']    # [$] Equipment for trim booth is assumed at 25000 $ per meter of blade length
        self.equipm_investment[10 + self.n_webs] =    250. * self.parallel_proc[10 + self.n_webs] * blade_specs['blade_length']    # [$] Equipment for overlay is assumed at 250 $ per meter of blade length
        self.equipm_investment[11 + self.n_webs] =  28500. * self.parallel_proc[11 + self.n_webs] * blade_specs['blade_length']    # [$] Equipment for post-cure is assumed at 28500 $ per meter of blade length
        self.equipm_investment[12 + self.n_webs] = 390000. * self.parallel_proc[12 + self.n_webs] * blade_specs['root_D']          # [$] Equipment for root cut and drill is assumed at 390000 $ per meter of root diameter
        self.equipm_investment[13 + self.n_webs] =  15500. * self.parallel_proc[13 + self.n_webs] * blade_specs['root_D']          # [$] Equipment for root hardware install is assumed at 15500 $ per meter of root diameter
        self.equipm_investment[14 + self.n_webs] =    160. * self.parallel_proc[14 + self.n_webs] * (blade_specs['area_lpskin_wo_flanges'] + blade_specs['area_hpskin_wo_flanges']) # [$] Equipment for surface preparation is assumed at 160 $ per meter square of blade outer surface
        self.equipm_investment[15 + self.n_webs] =  57000. * self.parallel_proc[15 + self.n_webs] * blade_specs['blade_length']    # [$] Equipment for paint booth is assumed at 57000 $ per meter of blade length
        self.equipm_investment[16 + self.n_webs] =    800. * self.parallel_proc[16 + self.n_webs] * blade_specs['blade_length']    # [$] Equipment for surface inspection and finish is assumed at 800 $ per meter of blade length
        self.equipm_investment[17 + self.n_webs] = 200000. * self.parallel_proc[17 + self.n_webs]                                  # [$] Weight and Balance, assumed constant
        self.equipm_investment[18 + self.n_webs] =    400. * self.parallel_proc[18 + self.n_webs] * blade_specs['blade_length']    # [$] Equipment for final inspection is assumed at 400 $ per meter of blade length
        self.equipm_investment[19 + self.n_webs] =   8000. * self.parallel_proc[19 + self.n_webs] * blade_specs['root_D']          # [$] Equipment for shipping preparation is assumed at 8000 $ per meter of root diameter
        
        

        
        
        
       


        
    def execute_direct_labor_cost(self ,operation, labor_hours):
        
        if self.options['verbosity']:
            verbosity                                   = 1
        else:
            verbosity                                   = 0
        
        direct_labor_cost_per_blade             = np.zeros(len(operation)) # [$]
        direct_labor_cost_per_year              = np.zeros(len(operation)) # [$]
        
        if verbosity:
            print('\n#################################\nDirect labor cost')
        
        for i_op in range(0, len(operation)):
            direct_labor_cost_per_blade[i_op] , direct_labor_cost_per_year[i_op] = compute_direct_labor_cost(self, labor_hours[i_op], operation[i_op], self.cum_rejr[i_op], verbosity)
        
        total_direct_labor_cost_per_blade       = sum(direct_labor_cost_per_blade)
        total_direct_labor_cost_per_year        = sum(direct_labor_cost_per_year)
        
        total_labor_overhead_per_blade          = total_direct_labor_cost_per_blade * (self.overhead / 100.)
        
        return total_direct_labor_cost_per_blade , total_labor_overhead_per_blade
    
    
    def execute_utility_cost(self, operation, ct):
        
        if self.options['verbosity']:
            verbosity                                   = 1
        else:
            verbosity                                   = 0
        
        utility_cost_per_blade                  = np.zeros(len(operation)) # [$]
        utility_cost_per_year                   = np.zeros(len(operation)) # [$]
        
        if verbosity:
            print('\n#################################\nUtility cost')
        
        for i_op in range(0, len(operation)):
            utility_cost_per_blade[i_op] , utility_cost_per_year[i_op] = compute_utility_cost(self, ct[i_op], self.power_consumpt[i_op], operation[i_op], self.cum_rejr[i_op], verbosity)
        
        total_utility_cost_per_blade            = sum(utility_cost_per_blade)
        total_utility_labor_cost_per_year       = sum(utility_cost_per_year)
        
        return total_utility_cost_per_blade
    
    def execute_fixed_cost(self, operation, ct, blade_variable_cost_w_overhead):
        
        if self.options['verbosity']:
            verbosity                                   = 1
        else:
            verbosity                                   = 0
        
        building_cost_per_blade                     = np.zeros(len(operation)) # [$]
        building_cost_per_year                      = np.zeros(len(operation)) # [$]
        building_annuity                            = np.zeros(len(operation)) # [$]
        tooling_cost_per_blade                      = np.zeros(len(operation)) # [$]
        tooling_cost_per_year                       = np.zeros(len(operation)) # [$]
        tooling_annuity                             = np.zeros(len(operation)) # [$]
        equipment_cost_per_blade                    = np.zeros(len(operation)) # [$]
        equipment_cost_per_year                     = np.zeros(len(operation)) # [$]
        equipment_annuity                           = np.zeros(len(operation)) # [$]
        maintenance_cost_per_blade                  = np.zeros(len(operation)) # [$]
        maintenance_cost_per_year                   = np.zeros(len(operation)) # [$]
        
        
        if self.options['verbosity']:
            print('\n#################################\nFixed cost')
        
        for i_op in range(0, len(operation)):
            if verbosity:
                print('\nBuilding:')
            building_investment                 = self.floor_space[i_op] * self.price_space
            investment_bu                       = building_investment * self.parallel_proc[i_op]
            building_cost_per_blade[i_op], building_cost_per_year[i_op], building_annuity[i_op] = compute_cost_annuity(self, operation[i_op], investment_bu, self.building_life, verbosity)    
            
            if verbosity:
                print('\nTooling:')
            investment_to                       = self.tooling_investment[i_op] * self.parallel_proc[i_op]
            tooling_cost_per_blade[i_op], tooling_cost_per_year[i_op], tooling_annuity[i_op] = compute_cost_annuity(self, operation[i_op], investment_to, self.tool_life, verbosity)
            
            if verbosity:
                print('\nEquipment:')
            investment_eq                       = self.equipm_investment[i_op]  * self.parallel_proc[i_op]
            equipment_cost_per_blade[i_op], equipment_cost_per_year[i_op], equipment_annuity[i_op] = compute_cost_annuity(self, operation[i_op], investment_eq, self.eq_life, verbosity)
            
            if verbosity:
                print('\nMaintenance:')
            maintenance_cost_per_blade[i_op], maintenance_cost_per_year[i_op] = compute_maintenance_cost(self, operation[i_op], investment_eq, investment_to, investment_bu, verbosity)    
        
        # Sums across operations
        total_building_labor_cost_per_year      = sum(building_cost_per_year)
        total_building_cost_per_blade           = sum(building_cost_per_blade)
        
        total_tooling_labor_cost_per_year       = sum(tooling_cost_per_year)
        total_tooling_cost_per_blade            = sum(tooling_cost_per_blade)
        
        total_equipment_labor_cost_per_year     = sum(equipment_cost_per_year)
        total_equipment_cost_per_blade          = sum(equipment_cost_per_blade)

        total_maintenance_labor_cost_per_year   = sum(maintenance_cost_per_year)
        total_maintenance_cost_per_blade        = sum(maintenance_cost_per_blade)
        
        # Annuity
        equipment_annuity_tot                   = sum(equipment_annuity)
        tooling_annuity_tot                     = sum(tooling_annuity)
        building_annuity_tot                    = sum(building_annuity)
        
        working_annuity                         = np.pmt(self.crr /100. / 12. , self.wcp, -(self.wcp / 12. * (total_maintenance_labor_cost_per_year + blade_variable_cost_w_overhead * self.n_blades))) * 12.

        annuity_tot_per_year                    = equipment_annuity_tot + tooling_annuity_tot + building_annuity_tot + working_annuity
        
        
        cost_of_capital_per_year                = annuity_tot_per_year - (blade_variable_cost_w_overhead * self.n_blades + total_equipment_labor_cost_per_year + total_tooling_labor_cost_per_year + total_building_labor_cost_per_year + total_maintenance_labor_cost_per_year)
        cost_of_capital_per_blade               = cost_of_capital_per_year / self.n_blades

        
        return total_equipment_cost_per_blade, total_tooling_cost_per_blade, total_building_cost_per_blade, total_maintenance_cost_per_blade, cost_of_capital_per_blade

def compute_direct_labor_cost(self, labor_hours, operation, cum_rejr, verbosity):
        
        cost_per_blade = (self.wage * (1. + self.beni / 100.) * labor_hours) / (1. - self.avg_dt / 100.)/(1. - cum_rejr)
        cost_per_year  = cost_per_blade * self.n_blades
        if verbosity == 1:
            print('Activity: ' + operation)
            print('per blade: {:8.2f} $ \t \t --- \t \t per year: {:8.2f} $'.format(float(cost_per_blade),float(cost_per_year)))
        

        return cost_per_blade , cost_per_year

def compute_utility_cost(self, ct, power_consumpt, operation, cum_rejr, verbosity):
       
        cost_per_blade = (self.electr  * power_consumpt * ct) / (1. - self.avg_dt / 100.)/(1. - cum_rejr)
        cost_per_year  = cost_per_blade * self.n_blades
        
        if verbosity == 1:
            print('Activity: ' + operation)
            print('per blade: {:8.2f} $ \t \t --- \t \t per year: {:8.2f} $'.format(float(cost_per_blade),float(cost_per_year)))

        return cost_per_blade , cost_per_year

def compute_cost_annuity(self, operation, investment, life, verbosity):
       
        cost_per_year   = investment / life
        cost_per_blade  = cost_per_year / self.n_blades
        annuity         = np.pmt(self.crr / 100. / 12. , life * 12., -investment) * 12.
        
        if verbosity == 1:
            print('Activity: ' + operation)
            print('per blade: {:8.2f} $ \t \t --- \t \t per year: {:8.2f} $ \t \t --- \t \t annuity: {:8.2f} $'.format(float(cost_per_blade),float(cost_per_year),float(annuity)))

        return cost_per_blade , cost_per_year, annuity        

def compute_maintenance_cost(self, operation, investment_eq, investment_to, investment_bu, verbosity):
    cost_per_year   = self.maintenance_cost / 100. * (investment_eq + investment_to + investment_bu)
    cost_per_blade  = cost_per_year / self.n_blades
    
    if verbosity == 1:
        print('Activity: ' + operation)
        print('per blade: {:8.2f} $ \t \t --- \t \t per year: {:8.2f} $'.format(float(cost_per_blade),float(cost_per_year)))

    return cost_per_blade , cost_per_year
    
class blade_cost_model(object):

    def __init__(self, options={}):

        if options == {}:
            self.options['verbosity']        = False
            self.options['tex_table']        = False
            self.options['generate_plots']   = False
            self.options['show_plots']       = False
            self.options['show_warnings']    = False
            self.options['discrete']         = False
        else:
            self.options = options


    def init_from_refBlade(self, refBlade):
        # Code take from rotor_geometry.py (RotorSE). It computes layup properties, independent of which turbine it is
        # Setup paths
        strucpath       = refBlade.getStructPath()
        self.materials  = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(strucpath, 'materials.inp'))
        
        npts            = refBlade.npts
        self.upperCS    = [0]*npts
        self.lowerCS    = [0]*npts
        self.websCS     = [0]*npts
        self.profile    = [0]*npts

        for i in range(npts):
            webLoc      = []
            istr        = str(i) if refBlade.name == '3_35MW' or refBlade.name == '10MW' else str(i+1)
            self.upperCS[i], self.lowerCS[i], self.websCS[i] = CompositeSection.initFromPreCompLayupFile(os.path.join(strucpath, 'layup_' + istr + '.inp'), webLoc, self.materials , readLocW=True)
            self.profile[i]  = Profile.initFromPreCompFile(os.path.join(strucpath, 'shape_' + istr + '.inp'))

        self.name        = refBlade.name
        self.bladeLength = refBlade.bladeLength
        # self.eta         = refBlade.r
        self.r           = refBlade.r * refBlade.bladeLength
        self.chord       = refBlade.chord_ref
        self.le_location = refBlade.le_location

    def init_from_Ontology(self, refBlade):
        self.materials   = refBlade['precomp']['materials']
        self.upperCS     = refBlade['precomp']['upperCS']
        self.lowerCS     = refBlade['precomp']['lowerCS']
        self.websCS      = refBlade['precomp']['websCS']
        self.profile     = refBlade['precomp']['profile']

        self.name        = refBlade['config']['name']
        self.bladeLength = refBlade['pf']['r'][-1]
        self.r           = refBlade['pf']['r']
        self.chord       = refBlade['pf']['chord']
        self.le_location = refBlade['pf']['p_le']


    def execute_blade_cost_model(self):

        # print([self.materials, type(self.materials), len(self.materials)])
        # print([self.upperCS, type(self.upperCS), len(self.upperCS)])
        # print([self.lowerCS, type(self.lowerCS), len(self.lowerCS)])
        # print([self.websCS, type(self.websCS), len(self.websCS)])
        # print([self.profile, type(self.profile), len(self.profile)])
        # print([self.name, type(self.name), len(self.name)])
        # print([self.bladeLength, type(self.bladeLength)])
        # print([self.r, type(self.r), len(self.r)])
        # print([self.chord, type(self.chord), len(self.chord)])
        # print([self.le_location, type(self.le_location), len(self.le_location)])
        
        if self.options['verbosity'] == True:
            print('\n \n#####################################################\n')
            print('Blade Cost Model')
            print('National Renewable Energy Lab - Golden, CO')
            print('Bortolotti P, Dykes K, Murray R, Berry D')
            print('12th October 2018')
            print('\n#####################################################\n\n')
            print('BLADE OF THE TURBINE ' + self.name)
            print('\n\n#####################################################')
        
        t_init = time.time()
        
        # Bill of Materials
        bom                                                            = blade_bom()
        bom.options                                                    = self.options
        bom.name                                                       = self.name
        bom.bladeLength                                                = self.bladeLength
        # bom.eta                                                        = self.eta
        bom.r                                                          = self.r
        bom.chord                                                      = self.chord
        bom.le_location                                                = self.le_location
        bom.materials                                                  = self.materials
        bom.upperCS                                                    = self.upperCS
        bom.lowerCS                                                    = self.lowerCS
        bom.websCS                                                     = self.websCS
        bom.profile                                                    = self.profile
        
        
        blade_specs, precomp_mat                                       = bom.extract_specs()
        matrix, bonding                                                = bom.compute_matrix_bonding(blade_specs, precomp_mat)
        metallic_parts                                                 = bom.compute_metallic_parts(blade_specs)
        consumables                                                    = bom.compute_consumables(blade_specs)
        self.total_blade_mat_cost_w_waste, self.blade_mass             = bom.compute_bom(blade_specs, precomp_mat, matrix, bonding, metallic_parts, consumables)
        
        # Labor and cycle time
        labor_ct                                                       = blade_labor_ct(blade_specs, precomp_mat , metallic_parts)
        labor_ct.options                                               = self.options
        labor_ct.name                                                  = self.name
        operation , labor_hours , skin_mold_gating_ct , non_gating_ct  = labor_ct.execute_blade_labor_ct()
        total_labor_hours                                              = sum(labor_hours)
        total_skin_mold_gating_ct                                      = sum(skin_mold_gating_ct)
        total_non_gating_ct                                            = sum(non_gating_ct)
        
        # Virtual factory                     
        vf                                                             = virtual_factory(blade_specs , operation, skin_mold_gating_ct, non_gating_ct, self.options)
        vf.options                                                     = self.options
        self.total_cost_labor , self.total_labor_overhead              = vf.execute_direct_labor_cost(operation , labor_hours)
        self.total_cost_utility                                        = vf.execute_utility_cost(operation , skin_mold_gating_ct + non_gating_ct)
        self.blade_variable_cost                                       = self.total_blade_mat_cost_w_waste + self.total_cost_labor + self.total_cost_utility
        self.total_cost_equipment , self.total_cost_tooling, self.total_cost_building, self.total_maintenance_cost, self.cost_capital  = vf.execute_fixed_cost(operation , skin_mold_gating_ct + non_gating_ct, self.blade_variable_cost + self.total_labor_overhead)
        self.blade_fixed_cost                                          = self.total_cost_equipment + self.total_cost_tooling + self.total_cost_building + self.total_maintenance_cost + self.total_labor_overhead + self.cost_capital
        
        # Total
        self.total_blade_cost = self.blade_variable_cost + self.blade_fixed_cost
        
        if self.options['tex_table'] == True:
            tex_table_file = open('tex_tables.txt','a')
            tex_table_file.write('\n\n%s & %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \n' % (self.name, self.bladeLength, self.total_blade_mat_cost_w_waste, total_labor_hours, total_skin_mold_gating_ct, total_non_gating_ct))
            tex_table_file.close()
        
        if self.options['verbosity'] == True:
            print('\n#################################')
            print('TOTAL LABOR AND CYCLE TIME:')    
            print('Labor: %.2f hr' % (total_labor_hours))
            print('Skin Mold Gating Cycle Time: %.2f hr' % (total_skin_mold_gating_ct))
            print('Non-Gating Cycle Time: %.2f hr' % (total_non_gating_ct))
            print('################################')    
            
            print('\n################################')
            print('TOTAL BLADE COSTS')
            print('Material cost        %.2f $' % (self.total_blade_mat_cost_w_waste))
            print('Labor cost:          %.2f $' % (self.total_cost_labor))
            print('Overhead labor cost: %.2f $' % (self.total_labor_overhead))
            print('Utility cost:        %.2f $' % (self.total_cost_utility))
            print('Equipment cost:      %.2f $' % (self.total_cost_equipment))   
            print('Tooling cost:        %.2f $' % (self.total_cost_tooling))
            print('Building cost:       %.2f $' % (self.total_cost_building))
            print('Maintenance cost:    %.2f $' % (self.total_maintenance_cost))
            print('Cost of capital:     %.2f $' % (self.cost_capital))
            print('################################')
            print('Variable :           %.2f $' % (self.blade_variable_cost))
            print('Fixed :              %.2f $' % (self.blade_fixed_cost))
            print('################################')
            print('TOTAL :              %.2f $' % (self.total_blade_cost))
        
        
        if self.options['tex_table'] == True:
            tex_table_file = open('tex_tables.txt','a') 
            tex_table_file.write('\n\n\n\\begin{table}[htpb]\n')
            tex_table_file.write('\\caption{Total costs of the %s blade.}\n' % self.name)
            tex_table_file.write('\\label{table:%s_6}\n' % self.name)
            tex_table_file.write('\\centering\n')
            tex_table_file.write('\\begin{tabular}{l c}\n')
            tex_table_file.write('\\toprule\n')
            tex_table_file.write('       & Cost [\$]\\\\ \n')
            tex_table_file.write('\\midrule\n')
            tex_table_file.write('Material cost       & %.2f \\\\ \n' % (self.total_blade_mat_cost_w_waste))
            tex_table_file.write('Labor cost          & %.2f \\\\ \n' % (self.total_cost_labor))
            tex_table_file.write('Overhead labor cost & %.2f \\\\ \n' % (self.total_labor_overhead))
            tex_table_file.write('Utility cost        & %.2f \\\\ \n' % (self.total_cost_utility))
            tex_table_file.write('Equipment cost      & %.2f \\\\ \n' % (self.total_cost_equipment))   
            tex_table_file.write('Tooling cost        & %.2f \\\\ \n' % (self.total_cost_tooling))
            tex_table_file.write('Building cost       & %.2f \\\\ \n' % (self.total_cost_building))
            tex_table_file.write('Maintenance cost    & %.2f \\\\ \n' % (self.total_maintenance_cost))
            tex_table_file.write('Cost of capital     & %.2f \\\\ \n' % (self.cost_capital))
            tex_table_file.write('\\midrule\n')
            tex_table_file.write('Variable            & %.2f \\\\ \n' % (self.blade_variable_cost))
            tex_table_file.write('Fixed               & %.2f \\\\ \n' % (self.blade_fixed_cost))
            tex_table_file.write('\\midrule\n')
            tex_table_file.write('\\textbf{Total}     & \\textbf{%.2f} \\\\ \n' % (self.total_blade_cost))
            tex_table_file.write('\\bottomrule\n')
            tex_table_file.write('\\end{tabular}\n')
            tex_table_file.write('\\end{table}\n')
            tex_table_file.close()
        
        
        if self.options['generate_plots'] == True:
            costs       = [self.total_blade_mat_cost_w_waste, self.total_cost_labor, self.total_labor_overhead, self.total_cost_utility, self.total_cost_equipment, self.total_cost_tooling, self.total_cost_building, self.total_maintenance_cost, self.cost_capital]
            name_costs  = ['Materials', 'Labor', 'Overhead', 'Utility', 'Equipment', 'Tooling', 'Building', 'Maintenance' , 'Capital']    
            fig1, ax1   = plt.subplots()
            patches, texts = ax1.pie(costs, explode=np.zeros(len(costs)), labels=name_costs, shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            for i in range(len(texts)):
                texts[i].set_fontsize(10)
            fig1.savefig('Plots\Total_' + self.name +'.png')
            if self.options['show_plots'] == True:    
                plt.show()
                
        
        if self.options['verbosity'] == True:
            t_final = time.time()
            print('\n################################')
            print('Runtime %.2f seconds' % (t_final-t_init))
            print('################################')
    
        return self.total_blade_cost, self.blade_mass

# Class to initiate the blade cost model
class RotorCost(ExplicitComponent):
    def initialize(self):
        self.options.declare('RefBlade')
        # self.options.declare('NPTS')
        # self.options.declare('name',             default='')
        self.options.declare('verbosity',        default=False)
        self.options.declare('tex_table',        default=False)
        self.options.declare('generate_plots',   default=False)
        self.options.declare('show_plots',       default=False)
        self.options.declare('show_warnings',    default=False)
        self.options.declare('discrete',         default=False)
                             
    def setup(self):
                
        NPTS = len(self.options['RefBlade']['pf']['s'])
        
        # These parameters will come from outside
        self.add_discrete_input('materials',     val=np.zeros(NPTS), desc='material properties of composite materials')
        self.add_discrete_input('upperCS',       val=np.zeros(NPTS), desc='list of CompositeSection objections defining the properties for upper surface')
        self.add_discrete_input('lowerCS',       val=np.zeros(NPTS), desc='list of CompositeSection objections defining the properties for lower surface')
        self.add_discrete_input('websCS',        val=np.zeros(NPTS), desc='list of CompositeSection objections defining the properties for shear webs')
        self.add_discrete_input('profile',       val=np.zeros(NPTS), desc='list of CompositeSection profiles')
        
        self.add_input('Rtip',          val=0.0,            units='m', desc='rotor radius')
        self.add_input('Rhub',          val=0.0,            units='m', desc='hub radius')
        self.add_input('bladeLength',   val=0.0,            units='m', desc='blade length')
        self.add_input('r_pts',         val=np.zeros(NPTS), units='m', desc='blade radial locations, expressed in the rotor system')
        self.add_input('chord',         val=np.zeros(NPTS), units='m', desc='Chord distribution')
        self.add_input('le_location',   val=np.zeros(NPTS), desc='Leading-edge positions from a reference blade axis (usually blade pitch axis). Locations are normalized by the local chord length. Positive in -x direction for airfoil-aligned coordinate system')

        # outputs
        self.add_output('total_blade_cost', val=0.0, units='USD', desc='total blade cost')
        self.add_output('total_blade_mass', val=0.0, units='USD', desc='total blade cost')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        bcm             = blade_cost_model(options=self.options)
        bcm.name        = self.options['RefBlade']['config']['name']
        bcm.materials   = discrete_inputs['materials']
        bcm.upperCS     = discrete_inputs['upperCS']
        bcm.lowerCS     = discrete_inputs['lowerCS']
        bcm.websCS      = discrete_inputs['websCS']
        bcm.profile     = discrete_inputs['profile']
        bcm.chord       = inputs['chord']
                
        bcm.r           = (inputs['r_pts'] - inputs['Rhub'])/(inputs['Rtip'] - inputs['Rhub']) * float(inputs['bladeLength'])
        bcm.bladeLength = float(inputs['bladeLength'])
        
        bcm.le_location              = inputs['le_location']
        blade_cost, blade_mass       = bcm.execute_blade_cost_model()
        
        outputs['total_blade_cost'] = blade_cost
        outputs['total_blade_mass'] = blade_mass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
