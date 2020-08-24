import numpy as np
from weis.multifidelity.models.run_functions import CCBlade, OpenFAST
from weis.multifidelity.methods.trust_region import SimpleTrustRegion


# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(13)

bounds = {'blade.opt_var.chord_opt_gain' : np.array([[0.5, 1.5], [0.5, 1.5]])}
desvars = {'blade.opt_var.chord_opt_gain' : np.array([1., 1.])}
model_low = CCBlade(desvars, 'cc_results_nd.pkl')
model_high = OpenFAST(desvars, 'of_results_nd.pkl')
trust_region = SimpleTrustRegion(model_low, model_high, bounds, trust_radius=0.5)

trust_region.add_objective('CP', scaler=-1.)

trust_region.optimize(plot=True)
