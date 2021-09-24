import numpy as np
import openmdao.api as om

from wisdem.jacketse.components import JacketSE

modeling_options = {}
modeling_options["gamma_f"] = 1.35
modeling_options["gamma_m"] = 1.3
modeling_options["gamma_n"] = 1.0
modeling_options["gamma_b"] = 1.1

n_legs = 3
n_bays = 2
x_mb = True
n_dlc = 1

prob = om.Problem(
    model=JacketSE(
        n_legs=n_legs,
        n_bays=n_bays,
        x_mb=x_mb,
        n_dlc=n_dlc,
        modeling_options=modeling_options,
    )
)

# setup the optimization
prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = "SNOPT"

prob.model.add_design_var("r_foot", lower=8.0, upper=50.0)
prob.model.add_design_var("L", lower=50.0, upper=100.0)
prob.model.add_constraint("constr_jacket_stress", upper=1.0)
prob.model.add_objective("jacket_Fz", index=0, ref=1.0e3)

prob.model.approx_totals()

prob.setup()

# prob["turbine_F"] = [1.0e4, 0.0, 0.0]

# run the optimization
prob.run_driver()
