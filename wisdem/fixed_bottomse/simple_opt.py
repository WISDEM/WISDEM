import numpy as np
import openmdao.api as om

from wisdem.fixed_bottomse.jacket import JacketSE

modeling_options = {}
modeling_options["WISDEM"] = {}
modeling_options["WISDEM"]["n_dlc"] = 1
modeling_options["WISDEM"]["FixedBottomSE"] = {}
modeling_options["WISDEM"]["FixedBottomSE"]["mud_brace"] = True
modeling_options["WISDEM"]["FixedBottomSE"]["n_legs"] = 3
modeling_options["WISDEM"]["FixedBottomSE"]["n_bays"] = 2
modeling_options["WISDEM"]["FixedBottomSE"]["gamma_f"] = 1.0
modeling_options["WISDEM"]["FixedBottomSE"]["gamma_m"] = 1.0
modeling_options["WISDEM"]["FixedBottomSE"]["gamma_n"] = 1.0
modeling_options["WISDEM"]["FixedBottomSE"]["gamma_b"] = 1.0

prob = om.Problem(
    model=JacketSE(
        modeling_options=modeling_options,
    )
)

# setup the optimization
prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = "SNOPT"

prob.model.add_design_var("r_foot", lower=2.0, upper=20.0)
prob.model.add_design_var("r_head", lower=2.0, upper=10.0)
prob.model.add_design_var("L", lower=50.0, upper=100.0)
# prob.model.add_design_var("d_l", lower=0.5, upper=20.0)
prob.model.add_constraint("constr_jacket_stress", upper=1.0)
prob.model.add_objective("jacket_mass", index=0, ref=1.0e3)

prob.model.approx_totals(form="central")

prob.setup()

prob["turbine_F"] = [1.0e6, 0.0, 0.0]

# run the optimization
prob.run_driver()
