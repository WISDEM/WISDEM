import numpy as np
from openmdao.api import ExplicitComponent, Group, IndepVarComp, Problem
from wisdem.assemblies.load_IEA_yaml import WT_Data, Wind_Turbine, yaml2openmdao
from wisdem.rotorse.rotor_aeropower import RegulatedPowerCurve

class RotorAeroPower(Group):
    def initialize(self):
        self.options.declare('wt_init_options')
    def setup(self):
        wt_init_options = self.options['wt_init_options']

        self.add_subsystem('powercurve', RegulatedPowerCurve(wt_init_options   = wt_init_options), promotes = ['control_Vin', 'control_Vout','airfoils_aoa','airfoils_cl','airfoils_cd','airfoils_cm'])


class WT_Rotor(Group):
    # Openmdao group to run the aerostructural analysis of the  wind turbine rotor
    
    def initialize(self):
        self.options.declare('wt_init_options')
        
    def setup(self):
        wt_init_options = self.options['wt_init_options']
        self.add_subsystem('wt',  Wind_Turbine(wt_init_options     = wt_init_options), promotes = ['*'])
        self.add_subsystem('ra',  RotorAeroPower(wt_init_options   = wt_init_options))

        self.connect('control.V_in' , 'ra.control_Vin')
        self.connect('control.V_out' , 'ra.control_Vout')
        self.connect('airfoils.aoa', 'ra.airfoils_aoa')
        self.connect('blade.interp_airfoils.cl_interp', 'ra.airfoils_cl')
        self.connect('blade.interp_airfoils.cd_interp', 'ra.airfoils_cd')
        self.connect('blade.interp_airfoils.cm_interp', 'ra.airfoils_cm')

if __name__ == "__main__":

    ## File management
    fname_input        = "reference_turbines/nrel5mw/nrel5mw_mod_update.yaml"
    # fname_input        = "/mnt/c/Material/Projects/Hitachi_Design/Design/turbine_inputs/aerospan_formatted_v13.yaml"
    fname_output       = "reference_turbines/nrel5mw/nrel5mw_mod_update_output.yaml"
    
    # Load yaml data into a pure python data structure
    wt_initial               = WT_Data()
    wt_initial.validate      = False
    wt_initial.fname_schema  = "reference_turbines/IEAontology_schema.yaml"
    wt_init_options, wt_init = wt_initial.initialize(fname_input)
    
    # Initialize openmdao problem
    wt_opt          = Problem()
    wt_opt.model    = WT_Rotor(wt_init_options = wt_init_options)
    wt_opt.setup()
    # Load wind turbine data from wt_initial to the openmdao problem
    wt_opt = yaml2openmdao(wt_opt, wt_init_options, wt_init)
    wt_opt.run_driver()
    
    # Save data coming from openmdao to an output yaml file
    wt_initial.write_ontology(wt_opt, fname_output)
