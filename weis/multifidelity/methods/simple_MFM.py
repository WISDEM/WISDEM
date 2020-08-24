import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import OrderedDict
from traceback import print_exc
from weis.multifidelity.models.testbed_components import simple_2D_high_model, simple_2D_low_model


np.random.seed(13)

n_dims = 2
num_initial_points = 4

x_init = np.random.rand(num_initial_points, n_dims)
x = x_init.copy()
print()
print(x_init)

list_of_desvars = []
for i in range(num_initial_points):
    desvars = OrderedDict()
    desvars['x'] = x[i, :]
    list_of_desvars.append(desvars)

low = simple_2D_low_model(desvars, 'low_2D_results.pkl')
high = simple_2D_high_model(desvars, 'high_2D_results.pkl')

lofi_function = low.run_vec
hifi_function = high.run_vec

for i in range(21):
    y_low = lofi_function(list_of_desvars)
    y_high = hifi_function(list_of_desvars)

    # Construct RBF interpolater for error function
    differences = y_high - y_low
    
    input_arrays = np.split(x, x.shape[1], axis=1)
    
    input_arrays = [x.flatten() for x in input_arrays]
    
    try:
        e = Rbf(*input_arrays, differences, epsilon=0.1)
    except:
        print_exc()
        print("Done!")
        break

    # Create m_k = lofi + RBF
    def m(x):
        desvars = low.unflatten_desvars(x)
        return lofi_function([desvars]) + e(*x)
        
    n_plot = 11
    x_plot = np.linspace(0., 1., n_plot)
    X, Y = np.meshgrid(x_plot, x_plot)
    x_values = np.vstack((X.flatten(), Y.flatten())).T
    
    list_of_desvars_plot = []
    for j in range(n_plot*n_plot):
        desvars = OrderedDict()
        desvars['x'] = x_values[j]
        list_of_desvars_plot.append(desvars)
        
    y_plot_high = hifi_function(list_of_desvars_plot).reshape(n_plot, n_plot)

    plt.figure()
    plt.contourf(X, Y, y_plot_high)
    plt.scatter(x[:, 0], x[:, 1], color='white')
    
    plt.savefig(f'image_{i}.png')

    x0 = x[-1, :]
    res = minimize(m, x0, method='SLSQP', tol=1e-6, bounds=[(0., 1.), (0., 1.)])
    x_new = np.atleast_2d(res.x)
    
    print()
    x = np.vstack((x, x_new))
    print(x)
    
    print('========')
    desvars = OrderedDict()
    desvars['x'] = np.squeeze(x_new)
    list_of_desvars.append(desvars)
    

    
print()
print(f'Number of high-fidelity calls for MFM: {x.shape[0]}')
print(f'Answer found: {np.squeeze(x_new)}')
print()

def hifi(x):
    desvars = high.unflatten_desvars(x)
    return hifi_function([desvars])

res = minimize(hifi, x_init[2, :], method='SLSQP', tol=1e-6, bounds=[(0., 1.), (0., 1.)])

print(f'Number of high-fidelity calls for hifi only: {res.nfev}, jac calls: {res.njev}')
print(f'Answer found: {res.x}, {res.fun}')
print()
