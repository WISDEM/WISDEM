import numpy as np
from nrel_openmdao_extensions.dakota_driver.run_dakota import do_full_optimization


from time import time
s = time()

bounds = {'blade.opt_var.chord_opt_gain' : np.array([[0.5, 1.5], [0.5, 1.5]])}
desvars = {'blade.opt_var.chord_opt_gain' : np.array([1., 1.])}
outputs = ['CP']
template_dir = 'template_dir/'
model_string = 'from weis.multifidelity.models.run_functions import FullCCBlade as model'
output_scalers = [-1.]

do_full_optimization(template_dir, desvars, outputs, bounds, model_string, output_scalers)

print(time() - s, 'seconds')


# bounds = {'x' : np.array([[0.0, 1.0], [0.0, 1.0]])}
# desvars = {'x' : np.array([0., 0.25])}
# outputs = ['y']
# template_dir = 'template_dir/'
# model_string = 'from weis.multifidelity.models.testbed_components import simple_2D_high_model as model'
# 
# do_full_optimization(template_dir, desvars, outputs, bounds, model_string)