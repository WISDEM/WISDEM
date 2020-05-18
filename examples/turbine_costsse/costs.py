from __future__ import print_function
import numpy as np
from openmdao.api as om
from wisdem.turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015

# simple test of module
prob = om.Problem()
prob.model = Turbine_CostsSE_2015(verbosity=True)
prob.setup()

prob['blade_mass']          = 17650.67  # inline with the windpact estimates
prob['hub_mass']            = 31644.5
prob['pitch_system_mass']   = 17004.0
prob['spinner_mass']        = 1810.5
prob['lss_mass']            = 31257.3
prob['main_bearing_mass']   = 0.5 * 9731.41
prob['gearbox_mass']        = 30237.60
prob['hss_mass']            = 1492.45
prob['generator_mass']      = 16699.85
prob['bedplate_mass']       = 93090.6
prob['yaw_mass']            = 11878.24
prob['tower_mass']          = 434559.0
prob['converter_mass']      = 1000.
prob['hvac_mass']           = 1000.
prob['cover_mass']          = 1000.
prob['platforms_mass']      = 1000.
prob['transformer_mass']    = 1000.

# other inputs
prob['machine_rating']      = 5000.0
prob['blade_number']        = 3
prob['crane']               = True
prob['main_bearing_number'] = 2

# Evaluate the model
prob.run_model()

# Print all intermediate inputs and outputs to the screen
prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)
