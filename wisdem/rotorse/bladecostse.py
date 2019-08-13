from wisdem.rotorse.blade_bom        import blade_bom
from wisdem.rotorse.blade_labor_ct   import blade_labor_ct
from wisdem.rotorse.virtual_factory  import virtual_factory
from wisdem.rotorse.precomp import Profile, CompositeSection, Orthotropic2DMaterial
import wisdem.rotorse
import matplotlib.pyplot    as plt
import numpy                as np
import time, os, warnings


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


        
        
    
    
