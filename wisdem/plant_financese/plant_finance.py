from openmdao.api import ExplicitComponent, Group, Problem
import numpy as np

class PlantFinance(ExplicitComponent):
    def initialize(self):
        self.options.declare('verbosity',default=False)
        
    def setup(self):

        # Inputs
        self.add_input('turbine_cost' ,     val=0.0, units='USD',   desc='A wind turbine capital cost')
        self.add_discrete_input('turbine_number',    val=0,                  desc='Number of turbines at plant')
        self.add_input('turbine_bos_costs', val=0.0, units='USD',   desc='Balance of system costs of the turbine')
        self.add_input('turbine_avg_annual_opex',val=0.0, units='USD',desc='Average annual operational expenditures of the turbine')
        self.add_input('park_aep',          val=0.0, units='kW*h',  desc='Annual Energy Production of the wind plant')
        self.add_input('turbine_aep',       val=0.0, units='kW*h',  desc='Annual Energy Production of the wind turbine')
        self.add_input('wake_loss_factor',  val=0.0,                desc='The losses in AEP due to waked conditions')
        self.add_input('machine_rating',  val=0.0, units='MW',                desc='rating of the turbine')

        # parameters
        self.add_input('fixed_charge_rate', val=0.12,               desc = 'Fixed charge rate for coe calculation')

        #Outputs
        self.add_output('lcoe',             val=0.0, units='USD/kW',desc='Levelized cost of energy for the wind plant')

        self.declare_partials('*','*')
        
    
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack parameters
        n_turbine   = discrete_inputs['turbine_number']
        c_turbine   = inputs['turbine_cost'] 
        c_bos_turbine  = inputs['turbine_bos_costs'] 
        c_opex_turbine = inputs['turbine_avg_annual_opex'] 
        fcr         = inputs['fixed_charge_rate']
        wlf         = inputs['wake_loss_factor']
        turb_aep    = inputs['turbine_aep']
        t_rating    = inputs['machine_rating']
        
        # Run a few checks on the inputs
        if n_turbine == 0:
            exit('ERROR: The number of the turbines in the plant is not initialized correctly and it is currently equal to 0. Check the connections to Plant_FinanceSE')
        
        if c_turbine == 0:
            exit('ERROR: The cost of the turbines in the plant is not initialized correctly and it is currently equal to 0 USD. Check the connections to Plant_FinanceSE')
            
        if c_bos_turbine == 0:
            print('WARNING: The BoS costs of the turbine are not initialized correctly and they are currently equal to 0 USD. Check the connections to Plant_FinanceSE')
        
        if c_opex_turbine == 0:
            print('WARNING: The Opex costs of the turbine are not initialized correctly and they are currently equal to 0 USD. Check the connections to Plant_FinanceSE')
        
        if inputs['park_aep'] == 0:
            if turb_aep != 0:
                park_aep     =  n_turbine * turb_aep * (1. - wlf)
                dpark_dtaep  =  n_turbine            * (1. - wlf)
                dpark_dnturb =              turb_aep * (1. - wlf)
                dpark_dwlf   = -n_turbine * turb_aep
                dpark_dpaep  = 0.0
            else:
                exit('ERROR: AEP is not connected properly. Both turbine_aep and park_aep are currently equal to 0 Wh. Check the connections to Plant_FinanceSE')
        else:
            park_aep    = inputs['park_aep']
            dpark_dpaep = 1.0
            dpark_dtaep = dpark_dnturb = dpark_dwlf = 0.0
        
        npr           = n_turbine * t_rating # net park rating, used in net energy capture calculation below
        dnpr_dnturb   =             t_rating
        dnpr_dtrating = n_turbine
        
        nec           = park_aep     / (npr * 1.e003) # net energy rating, per COE report
        dnec_dwlf     = dpark_dwlf   / (npr * 1.e003)
        dnec_dtaep    = dpark_dtaep  / (npr * 1.e003)
        dnec_dpaep    = dpark_dpaep  / (npr * 1.e003)
        dnec_dnturb   = dpark_dnturb / (npr * 1.e003) - dnpr_dnturb   * nec / npr
        dnec_dtrating =                               - dnpr_dtrating * nec / npr
        
        icc     = (c_turbine + c_bos_turbine) / (t_rating * 1.e003) #$/kW, changed per COE report
        c_opex  = (c_opex_turbine) / (t_rating * 1.e003)  # $/kW, changed per COE report

        dicc_dtrating   = -icc / t_rating
        dcopex_dtrating = -c_opex / t_rating
        dicc_dcturb = dicc_dcbos = dcopex_dcopex = 1.0 / (t_rating * 1.e003)
           
        #compute COE and LCOE values
        lcoe = ((icc * fcr + c_opex) / nec) # changed per COE report
        outputs['lcoe'] = lcoe
        
        self.J = {}
        self.J['lcoe', 'turbine_cost'            ] = dicc_dcturb*fcr /nec
        self.J['lcoe', 'turbine_number'          ] = - dnec_dnturb*lcoe/nec
        self.J['lcoe', 'turbine_bos_costs'       ] = dicc_dcbos *fcr /nec
        self.J['lcoe', 'turbine_avg_annual_opex' ] = dcopex_dcopex   /nec
        self.J['lcoe', 'fixed_charge_rate'       ] = icc / nec
        self.J['lcoe', 'wake_loss_factor'        ] = -dnec_dwlf *lcoe/nec
        self.J['lcoe', 'turbine_aep'             ] = -dnec_dtaep*lcoe/nec
        self.J['lcoe', 'park_aep'                ] = -dnec_dpaep*lcoe/nec
        self.J['lcoe', 'machine_rating'          ] = (dicc_dtrating*fcr + dcopex_dtrating)/nec - dnec_dtrating*lcoe/nec
        
        if self.options['verbosity']:
            print('################################################')
            print('Computation of CoE and LCoE from Plant_FinanceSE')
            print('Inputs:')
            print('Number of turbines in the park   %u'              % n_turbine)
            print('Cost of the single turbine       %.3f M USD'      % (c_turbine * 1.e-006))  
            print('BoS costs of the single turbine  %.3f M USD'      % (c_bos_turbine * 1.e-006))  
            print('Initial capital cost of the park %.3f M USD'      % (icc * n_turbine * t_rating * 1.e-003))  
            print('Opex costs of the single turbine %.3f M USD'      % (c_opex_turbine * 1.e-006))
            print('Opex costs of the park           %.3f M USD'      % (c_opex_turbine * n_turbine * 1.e-006))              
            print('Fixed charge rate                %.2f %%'         % (fcr * 100.))     
            print('Wake loss factor                 %.2f %%'         % (wlf * 100.))         
            print('AEP of the single turbine        %.3f GWh'        % (turb_aep * 1.e-006))    
            print('AEP of the wind plant            %.3f GWh'        % (park_aep * 1.e-006))   
            print('Capital costs                    %.2f $/kW'       % icc) #added
            print('NEC                              %.2f MWh/MW/yr'  % nec) #added
            print('Outputs:')
            print('LCoE                             %.3f USD/MW'     % (lcoe  * 1.e003)) #removed "coe", best to have only one metric for cost
            print('################################################')
            
                    

    def compute_partials(self, inputs, J):
        J.update(self.J)



    
class Finance(Group):
    
    def setup(self):
        self.add_subsystem('plantfinancese', PlantFinance(verbosity = True), promotes=['*'])


if __name__ == "__main__":
    # Initialize OpenMDAO problem and FloatingSE Group
    prob = Problem()
    prob.model=Finance() # runs script
    prob.setup()

    rating = 2.32 #MW
    prob['machine_rating']          = rating
    prob['turbine_cost']            = 1093  * rating * 1.e+003  #USD
    prob['turbine_number']          = 87.
    prob['turbine_avg_annual_opex'] = 43.56 * rating * 1.e+003  # USD/yr Source: 70 $/kW/yr, updated from report, (70 is on the high side)
    prob['fixed_charge_rate']       = 0.079216644 # 7.9 % confirmed from report
    prob['turbine_bos_costs']       = 517. * rating * 1.e+003 # from apendix of report
    prob['wake_loss_factor']        = 0.15 # confirmed from report 
    prob['turbine_aep']             = 8428.56 * 1.e+003 # confirmed from report 
    
    prob.run_driver()

