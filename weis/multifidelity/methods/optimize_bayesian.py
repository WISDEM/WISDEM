import numpy as np
from emukit.test_functions.forrester import forrester, forrester_low
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
import GPy
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel


x_train_l = np.atleast_2d(np.random.rand(12)).T
x_train_h = np.atleast_2d(np.random.permutation(x_train_l)[:6])
y_train_l = forrester_low(x_train_l)
y_train_h = forrester(x_train_h)
X_train, Y_train = convert_xy_lists_to_arrays([x_train_l, x_train_h], [y_train_l, y_train_h])

num_fidelities = 2
kernels = [GPy.kern.RBF(1), GPy.kern.RBF(1)]
linear_mf_kernel = LinearMultiFidelityKernel(kernels)
gpy_linear_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, linear_mf_kernel, n_fidelities = 2)

gpy_linear_mf_model.mixed_noise.Gaussian_noise.fix(0)
gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.fix(0)

gpy_linear_mf_model.optimize()