from __future__ import print_function

# OAS problem MFROBO
import numpy as np
from scipy.optimize import minimize
from collections import OrderedDict
from mfrobo_class import MFROBO
from testbed_components import simple_1D_low, simple_1D_medium, simple_1D_high


np.random.seed(314)

# Fidelity parameters
funcs = [simple_1D_high, simple_1D_medium, simple_1D_low]

Din = np.array([0.5])

Ex_stdx = OrderedDict()

Ex_stdx["dummy"] = (0.0, 0.02)

# Weight for variance in the objective function
eta = 3

# Target MSE for moment estimates
J_star = 1e-1

mfrobo_inst = MFROBO(funcs, Ex_stdx, eta, J_star, "mfrobo_out.pkl", nbXsamp=5)
mfrobo_inst.t_DinT = np.array([50., 0.1, 0.05])

bounds = [(0.0, 1.0)]

# construct the bounds in the form of constraints
cons = []
for factor in range(len(bounds)):
    lower, upper = bounds[factor]
    l = {"type": "ineq", "fun": lambda x, lb=lower, i=factor: x[i] - lb}
    u = {"type": "ineq", "fun": lambda x, ub=upper, i=factor: ub - x[i]}
    cons.append(l)
    cons.append(u)
    
# cons.append({'type': 'ineq', 'fun' : lambda x: mfrobo_inst.fake_con(x)})

res = minimize(
    mfrobo_inst.obj_func,
    Din,
    args=(),
    method="COBYLA",
    tol=1e-8,
    constraints=cons,
    options={"disp": True, "maxiter": 1000, "rhobeg": 0.1},
)
