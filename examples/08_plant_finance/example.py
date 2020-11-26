#!/usr/bin/env python3

# Import the libraries
import openmdao.api as om
from wisdem.plant_financese.plant_finance import PlantFinance

# Initialize the OpenMDAO instance
prob = om.Problem()
prob.model = PlantFinance()
prob.setup()

# Set variable inputs with intended units
prob.set_val("machine_rating", 2.32e3, units="kW")
prob.set_val("tcc_per_kW", 1093, units="USD/kW")
prob.set_val("turbine_number", 87)
prob.set_val("opex_per_kW", 43.56, units="USD/kW/yr")
prob.set_val("fixed_charge_rate", 0.079216644)
prob.set_val("bos_per_kW", 517.0, units="USD/kW")
prob.set_val("wake_loss_factor", 0.15)
prob.set_val("turbine_aep", 9915.95e3, units="kW*h")

# Run the model once
prob.run_model()

# Print all inputs and outputs to the screen
prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)
