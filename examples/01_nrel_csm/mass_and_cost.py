# 0 ---------- (marker for docs)
import openmdao.api as om

from wisdem.nrelcsm.nrel_csm_mass_2015 import nrel_csm_2015

# 0 ---------- (marker for docs)

# 1 ---------- (marker for docs)
# OpenMDAO Problem instance
prob = om.Problem(reports=False)
prob.model = nrel_csm_2015()
prob.setup()
# 1 ---------- (marker for docs)

# 2 ---------- (marker for docs)
# Initialize variables for NREL CSM
prob["machine_rating"] = 5000.0
prob["rotor_diameter"] = 126.0
prob["turbine_class"] = 2
prob["tower_length"] = 90.0
prob["blade_number"] = 3
prob["blade_has_carbon"] = False
prob["max_tip_speed"] = 80.0
prob["max_efficiency"] = 0.90
prob["main_bearing_number"] = 2
prob["crane"] = True
# 2 ---------- (marker for docs)

# 3 ---------- (marker for docs)
# Evaluate the model
prob.run_model()
# 3 ---------- (marker for docs)

# 4 ---------- (marker for docs)
# Print all intermediate inputs and outputs to the screen
prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)
# 4 ---------- (marker for docs)
