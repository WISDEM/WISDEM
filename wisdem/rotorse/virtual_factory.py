
import numpy as np
import math

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
        
        