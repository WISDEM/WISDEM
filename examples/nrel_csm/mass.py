from __future__ import print_function
import numpy as np
from openmdao.api as om
from wisdem.turbine_costsse.nrel_csm_tcc_2015 import nrel_csm_mass_2015

# simple test of module
prob = om.Problem()
prob.model = nrel_csm_mass_2015()
prob.setup()

prob['rotor_diameter'] = 126.0
prob['turbine_class'] = 1
prob['blade_has_carbon'] = False
prob['blade_number'] = 3    
prob['machine_rating'] = 5000.0
prob['hub_height'] = 90.0
prob['main_bearing_number'] = 2
prob['crane'] = True
prob['max_tip_speed'] = 80.0
prob['max_efficiency'] = 0.90

# Evaluate the model
prob.run_model()

# Print all intermediate inputs and outputs to the screen
prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)
