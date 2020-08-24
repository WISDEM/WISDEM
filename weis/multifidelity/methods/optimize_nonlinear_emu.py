import numpy as np
from time import time
import matplotlib.pyplot as plt
import openmdao.api as om
from testbed_components import simple_1D_low, simple_1D_high

import GPy
import emukit.multi_fidelity
import emukit.test_functions
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel


np.random.seed(314)

## Generate data for nonlinear example

high_fidelity = emukit.test_functions.non_linear_sin.nonlinear_sin_high
low_fidelity = emukit.test_functions.non_linear_sin.nonlinear_sin_low

x_plot = np.linspace(0, 1, 200)[:, None]
y_plot_l = low_fidelity(x_plot)
y_plot_h = high_fidelity(x_plot)

n_low_fidelity_points = 50

x_train_l = np.linspace(0, 1, n_low_fidelity_points)[:, None]
y_train_l = low_fidelity(x_train_l)

x_train_h = x_train_l[::4, :]
y_train_h = high_fidelity(x_train_h)

X_plot = convert_x_list_to_array([x_plot, x_plot])
X_plot_low = X_plot[:200]
X_plot_high = X_plot[200:]

### Convert lists of arrays to ND-arrays augmented with fidelity indicators

X_train, Y_train = convert_xy_lists_to_arrays([x_train_l, x_train_h], [y_train_l, y_train_h])
## Create nonlinear model

base_kernel = GPy.kern.RBF
kernels = make_non_linear_kernels(base_kernel, 2, X_train.shape[1] - 1)
nonlin_mf_model = NonLinearMultiFidelityModel(X_train, Y_train, n_fidelities=2, kernels=kernels, 
                                              verbose=True, optimization_restarts=1)
for m in nonlin_mf_model.models:
    m.Gaussian_noise.variance.fix(0)
    
nonlin_mf_model.optimize()

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
ei = ExpectedImprovement(nonlin_mf_model)

ei_locations = np.atleast_2d(np.array([[0.4232, 0.6761]]))

print(ei.evaluate(ei_locations))

# ## Compute mean and variance predictions
# 
# hf_mean_nonlin_mf_model, hf_var_nonlin_mf_model = nonlin_mf_model.predict(X_plot_high)
# hf_std_nonlin_mf_model = np.sqrt(hf_var_nonlin_mf_model)
# 
# lf_mean_nonlin_mf_model, lf_var_nonlin_mf_model = nonlin_mf_model.predict(X_plot_low)
# lf_std_nonlin_mf_model = np.sqrt(lf_var_nonlin_mf_model)
# 
# 
# ## Plot posterior mean and variance of nonlinear multi-fidelity model
# 
# plt.figure(figsize=(12,8))
# plt.fill_between(x_plot.flatten(), (lf_mean_nonlin_mf_model - 1.96*lf_std_nonlin_mf_model).flatten(), 
#                  (lf_mean_nonlin_mf_model + 1.96*lf_std_nonlin_mf_model).flatten(), color='g', alpha=0.3)
# plt.fill_between(x_plot.flatten(), (hf_mean_nonlin_mf_model - 1.96*hf_std_nonlin_mf_model).flatten(), 
#                  (hf_mean_nonlin_mf_model + 1.96*hf_std_nonlin_mf_model).flatten(), color='y', alpha=0.3)
# plt.plot(x_plot, y_plot_l, 'b')
# plt.plot(x_plot, y_plot_h, 'r')
# plt.plot(x_plot, lf_mean_nonlin_mf_model, '--', color='g')
# plt.plot(x_plot, hf_mean_nonlin_mf_model, '--', color='y')
# plt.scatter(x_train_h, y_train_h, color='r')
# plt.scatter(x_train_l, y_train_l, color='b')
# plt.xlabel('x')
# plt.ylabel('f (x)')
# plt.xlim(0, 1)
# plt.legend(['Low Fidelity', 'High Fidelity', 'Predicted Low Fidelity', 'Predicted High Fidelity'])
# plt.title('Nonlinear multi-fidelity model fit to low and high fidelity functions')
# plt.show()