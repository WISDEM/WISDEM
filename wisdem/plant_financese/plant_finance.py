from openmdao.api import ExplicitComponent

import numpy as np

class PlantFinance(ExplicitComponent):
    def initialize(self):
        self.options.declare(verbosity, default=False)
        
    def setup(self):
        self.amortFactor = None
        self.outputSave = {}
        
        # Inputs
        self.add_input('turbine_cost' ,     val=0.0, units='USD',   desc='A wind turbine capital cost')
        self.add_discrete_input('turbine_number',    val=0,                  desc='Number of turbines at plant')
        self.add_input('turbine_bos_costs', val=0.0, units='USD',   desc='Balance of system costs of the turbine')
        self.add_input('turbine_avg_annual_opex',val=0.0, units='USD',desc='Average annual operational expenditures of the turbine')
        self.add_input('park_aep',          val=0.0, units='kW*h',  desc='Annual Energy Production of the wind plant')
        self.add_input('turbine_aep',       val=0.0, units='kW*h',  desc='Annual Energy Production of the wind turbine')
        self.add_input('wake_loss_factor',  val=0.0,                desc='The losses in AEP due to waked conditions')

        # parameters
        self.add_input('fixed_charge_rate', val=0.12,               desc = 'Fixed charge rate for coe calculation')
        self.add_input('tax_rate',          val=0.4,                desc = 'Tax rate applied to operations')
        self.add_input('discount_rate',     val=0.07,               desc = 'Applicable project discount rate')
        self.add_input('construction_time', val=1.0,  units='year', desc = 'Number of years to complete project construction')
        self.add_input('project_lifetime',  val=20.0, units='year', desc = 'Project lifetime for LCOE calculation')
        self.add_input('sea_depth',         val=20.0, units='m',    desc = 'Sea depth of project for offshore, (0 for onshore)')

        #Outputs
        self.add_output('lcoe',             val=0.0, units='USD/kW',desc='Levelized cost of energy for the wind plant')
        self.add_output('coe',              val=0.0, units='USD/kW',desc='Cost of energy for the wind plant - unlevelized')

        self.declare_partials('coe' , ['turbine_cost', 'bos_cost', 'avg_annual_opex', 'net_aep'])
        self.declare_partials('lcoe', ['turbine_cost', 'bos_cost', 'avg_annual_opex', 'net_aep'])
 
    
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack parameters
        depth       = inputs['sea_depth']
        n_turbine   = discrete_inputs['turbine_number']
        c_turbine   = inputs['turbine_cost']
        c_bos_turbine  = inputs['turbine_bos_costs']
        c_opex_turbine = inputs['turbine_avg_annual_opex']
        fcr         = inputs['fixed_charge_rate']
        tax         = inputs['tax_rate']
        r           = inputs['discount_rate']
        wlf         = inputs['wake_loss_factor']
        turb_aep    = inputs['turbine_aep']
        park_aep    = inputs['park_aep']
        t_construct = inputs['construction_time']
        t_project   = inputs['project_lifetime']
        
        # Handy offshore boolean flag
        offshore = (depth > 0.0)
        
        # Run a few checks on the inputs
        if n_turbine == 0:
            exit('ERROR: The number of the turbines in the plant is not initialized correctly and it is currently equal to 0. Check the connections to Plant_FinanceSE')
        
        if c_turbine == 0:
            exit('ERROR: The cost of the turbines in the plant is not initialized correctly and it is currently equal to 0 USD. Check the connections to Plant_FinanceSE')
            
        if c_bos_turbine == 0:
            print('WARNING: The BoS costs of the turbine are not initialized correctly and they are currently equal to 0 USD. Check the connections to Plant_FinanceSE')
        
        if c_opex_turbine == 0:
            print('WARNING: The Opex costs of the turbine are not initialized correctly and they are currently equal to 0 USD. Check the connections to Plant_FinanceSE')
        
        if park_aep == 0:
            if turb_aep != 0:
                park_aep = n_turbine * turb_aep * (1. - wlf)
            else:
                exit('ERROR: AEP is not connected properly. Both turbine_aep and park_aep are currently equal to 0 Wh. Check the connections to Plant_FinanceSE')
        
        
        icc     = n_turbine * (c_turbine + c_bos_turbine)
        c_opex  = n_turbine * c_opex_turbine
        
        if offshore:
           # warranty Premium 
           icc += (c_turbine * n_turbine / 1.10) * 0.15
        
        #compute COE and LCOE values
        outputs['coe'] = (icc*fcr + c_opex*(1-tax)) / park_aep

        self.amortFactor = (1. + 0.5 * ((1. + r)**t_construct - 1.)) * (r / (1. - (1. + r)**(-t_project)))
        outputs['lcoe'] = (icc * self.amortFactor + c_opex) / park_aep
        self.outputSave['coe'] = outputs['coe']
        self.outputSave['lcoe'] = outputs['lcoe']
        
        if self.options['verbosity']
            print('################################################')
            print('Computation of CoE and LCoE from Plant_FinanceSE')
            print('Inputs:')
            print('Water depth                      %.2f m'     % depth)
            print('Number of turbines in the park   %u'         % n_turbine)
            print('Cost of the single turbine       %.3f M USD' % (c_turbine * 1.e-006))  
            print('BoS costs of the single turbine  %.3f M USD' % (c_bos_turbine * 1.e-006))  
            print('Initial capital cost of the park %.3f M USD' % (icc * 1.e-006))  
            print('Opex costs of the single turbine %.3f M USD' % (c_opex_turbine * 1.e-006))
            print('Opex costs of the park           %.3f M USD' % (c_opex * 1.e-006))              
            print('Fixed charge rate                %.2f %%'    % (fcr * 100.))     
            print('Tax rate                         %.2f %%'    % (tax * 100.))        
            print('Discount rate                    %.2f %%'    % (r * 100.))        
            print('Wake loss factor                 %.2f %%'    % (wlf * 100.))         
            print('AEP of the single turbine        %.3f GWh'   % (turb_aep * 1.e-006))    
            print('AEP of the wind plant            %.3f GWh'   % (park_aep * 1.e-006))   
            print('Construction time                %.2f yr'    % t_construct) 
            print('Project lifetime                 %.2f yr'    % t_project)
            print('Outputs:')
            print('CoE                              %.3f USD/MW' % (outputs['coe']  * 1.e003))
            print('LCoE                             %.3f USD/MW' % (outputs['lcoe'] * 1.e003))
            print('################################################')
            
                    

    def compute_partials(self, inputs, J):
        # Unpack parameters
        depth       = inputs['sea_depth']
        n_turbine   = inputs['turbine_number']
        fcr         = inputs['fixed_charge_rate']
        tax         = inputs['tax_rate']
        r           = inputs['discount_rate']
        wlf         = inputs['wake_loss_factor']
        turb_aep    = inputs['turbine_aep']
        park_aep    = inputs['park_aep']
        t_construct = inputs['construction_time']
        t_project   = inputs['project_lifetime']

        # Handy offshore boolean flag
        offshore = (depth > 0.0)
        
        # Run a few checks on the inputs
        if n_turbine == 0:
            exit('ERROR: The number of the turbines in the plant is not initialized correctly and it is currently equal to 0. Check the connections to Plant_FinanceSE')
        
        if park_aep == 0:
            if turb_aep != 0:
                park_aep = n_turbine * turb_aep * (1. - wlf)
            else:
                exit('ERROR: AEP is not connected properly. Both turbine_aep and park_aep are currently equal to 0 Wh. Check the connections to Plant_FinanceSE')
        
        
        dicc_dcturb = n_turbine
        dicc_dcbos  = 1.0
        if offshore:
            dicc_dcturb = n_turbine * (1.0 + 0.15 / 1.10)
            
        dcoe_dcturb = dicc_dcturb * fcr / park_aep
        dcoe_dcbos  = dicc_dcbos  * fcr / park_aep
        dcoe_dopex  = (1. - tax)        / park_aep
        dcoe_daep   = -self.outputSave['coe']  / park_aep

        dlcoe_dcturb = dicc_dcturb * self.amortFactor / park_aep
        dlcoe_dcbos  = dicc_dcbos  * self.amortFactor / park_aep
        dlcoe_dopex  = 1.0                            / park_aep
        dlcoe_daep   = -self.outputSave['lcoe']              / park_aep

        
        J['coe' , 'turbine_cost'   ] = dcoe_dcturb
        J['coe' , 'bos_cost'       ] = dcoe_dcbos
        J['coe' , 'avg_annual_opex'] = dcoe_dopex
        J['coe' , 'net_aep'        ] = dcoe_daep
        J['lcoe', 'turbine_cost'   ] = dlcoe_dcturb
        J['lcoe', 'bos_cost'       ] = dlcoe_dcbos
        J['lcoe', 'avg_annual_opex'] = dlcoe_dopex
        J['lcoe', 'net_aep'        ] = dlcoe_daep
        
        
        

        
