import numpy as np
from time import time
import matplotlib.pyplot as plt
import openmdao.api as om
from testbed_components import simple_1D_low, simple_1D_high


np.random.seed(314)

mm = om.MultiFiMetaModelUnStructuredComp(nfi=2)
mm.add_input("x", np.zeros((1,)))
mm.add_output("y", np.zeros((1,)))

# Surrrogate model that implements the multifidelity cokriging method.
mm.options["default_surrogate"] = om.MultiFiCoKrigingSurrogate(normalize=False)

prob = om.Problem()
prob.model.add_subsystem("mm", mm, promotes=["*"])

prob.driver = om.pyOptSparseDriver()  # om.ScipyOptimizeDriver() #
prob.driver.options["optimizer"] = "SNOPT"
# prob.driver.opt_settings['Verify level'] = -1

# --- Objective ---
prob.model.add_objective("y")

prob.model.add_design_var("x", lower=0.0, upper=1.0)

prob.setup()

prob["x"] = 0.8

s = time()

num_high = 5
num_low = 11

x_high = np.linspace(0.0, 1.0, num_high)
y_high = simple_1D_high(x_high)

x_low = np.hstack((x_high, np.random.rand(num_low - num_high)))
y_low = simple_1D_low(x_low)


mm.options["train:x"] = x_high
mm.options["train:y"] = y_high
mm.options["train:x_fi2"] = x_low
mm.options["train:y_fi2"] = y_low

prob.run_driver()

print(f"Optimal input: {prob['x'][0]}, optimal output: {prob['y'][0]}")
print(f"{time() - s:.3f} secs")


x_full = np.linspace(0.0, 1.0, 101)
y_full = np.zeros(x_full.shape)

for i, x in enumerate(x_full):
    prob["x"] = x
    prob.run_model()
    y_full[i] = prob["y"]

y_full_low = simple_1D_low(x_full)
y_full_high = simple_1D_high(x_full)

plt.figure()

plt.plot(x_full, y_full_low, label="low-fidelity", c="tab:green")
plt.scatter(x_low, y_low, c="tab:green")

plt.plot(x_full, y_full_high, label="high-fidelity", c="tab:orange")
plt.scatter(x_high, y_high, c="tab:orange")

plt.plot(x_full, y_full, label="surrogate", c="tab:blue")

plt.legend()

plt.show()
