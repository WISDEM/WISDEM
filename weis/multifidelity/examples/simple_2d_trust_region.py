import numpy as np
from weis.multifidelity.models.testbed_components import simple_2D_high_model, simple_2D_low_model
from weis.multifidelity.methods.trust_region import SimpleTrustRegion


# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(13)

bounds = {'x' : np.array([[0.0, 1.0], [0.0, 1.0]])}
desvars = {'x' : np.array([0., 0.25])}
model_low = simple_2D_low_model(desvars)
model_high = simple_2D_high_model(desvars)
trust_region = SimpleTrustRegion(model_low, model_high, bounds)

trust_region.add_objective('y')

trust_region.set_initial_point(desvars['x'])

trust_region.optimize()