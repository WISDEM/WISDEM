# 0 ---------- (marker for docs)
import openmdao.api as om

from wisdem.nrelcsm.nrel_csm_cost_2015 import Turbine_CostsSE_2015

# 0 ---------- (marker for docs)

# 1 ---------- (marker for docs)
# OpenMDAO Problem instance
prob = om.Problem(reports=False)
prob.model = Turbine_CostsSE_2015(verbosity=True)
prob.setup()
# 1 ---------- (marker for docs)

# 2 ---------- (marker for docs)
# Initialize variables for NREL CSM
prob["machine_rating"] = 5000.0
prob["blade_number"] = 3
prob["crane"] = True
prob["main_bearing_number"] = 2
# 2 ---------- (marker for docs)

# 3 ---------- (marker for docs)
# Component masses
prob["blade_mass"] = 15751.48043042
prob["hub_mass"] = 37548.40498997
prob["pitch_system_mass"] = 9334.08947551
prob["spinner_mass"] = 973.0
prob["lss_mass"] = 20568.96284886
prob["main_bearing_mass"] = 2245.41649102
prob["gearbox_mass"] = 21875.0
prob["hss_mass"] = 994.7
prob["generator_mass"] = 14900.0
prob["bedplate_mass"] = 41765.26095285
prob["yaw_mass"] = 12329.96247921
prob["hvac_mass"] = 400.0
prob["cover_mass"] = 6836.69
prob["tower_mass"] = 182336.48057717
prob["transformer_mass"] = 11485.0
prob["platforms_mass"] = 8220.65761911
# 3 ---------- (marker for docs)

# 4 ---------- (marker for docs)
# Evaluate the model
prob.run_model()
# 4 ---------- (marker for docs)

# 5 ---------- (marker for docs)
# Print all intermediate inputs and outputs to the screen
prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)
# 5 ---------- (marker for docs)
