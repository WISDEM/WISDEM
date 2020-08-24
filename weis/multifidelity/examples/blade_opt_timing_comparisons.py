import numpy as np
from weis.multifidelity.models.run_functions import CCBlade, OpenFAST
from scipy.optimize import minimize
from time import time
from weis.multifidelity.methods.trust_region import SimpleTrustRegion


bounds = np.array([[0.5, 1.5], [0.5, 1.5]])
desvars = {'blade.opt_var.chord_opt_gain' : np.array([1., 1.])}
model_low = CCBlade(desvars, 'CC_8_9.pkl')

s = time()

x0 = np.array([1., 1.])
scaled_function = lambda x: -model_low.run(x)['CP']
res = minimize(scaled_function, x0, method='SLSQP', tol=1e-10, bounds=bounds, options={'disp':False})
x_new = res.x

print('low fidelity')
print(res)
print(time() - s, 'seconds')
s = time()
print()

model_high = OpenFAST(desvars, 'OF_8_9.pkl')

scaled_function = lambda x: -model_high.run(x)['CP']
res = minimize(scaled_function, x0, method='SLSQP', tol=1e-10, bounds=bounds, options={'disp':False})
x_new = res.x

print('high fidelity')
print(res)
print(time() - s, 'seconds')
s = time()
print()

# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(13)

bounds = {'blade.opt_var.chord_opt_gain' : np.array([[0.5, 1.5], [0.5, 1.5]])}
desvars = {'blade.opt_var.chord_opt_gain' : np.array([1., 1.])}
model_low = CCBlade(desvars, 'CC_8_9_new.pkl')
model_high = OpenFAST(desvars, 'OF_8_9_new.pkl')
trust_region = SimpleTrustRegion(model_low, model_high, bounds, trust_radius=0.5, num_initial_points=30)

trust_region.add_objective('CP', scaler=-1.)

trust_region.optimize(plot=False)

print('multifidelity')
print(time() - s, 'seconds')
s = time()
print()