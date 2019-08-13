import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import brentq

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

    
    
