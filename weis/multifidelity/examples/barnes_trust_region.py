import numpy as np
from weis.multifidelity.models.testbed_components import barnes_high_model, barnes_low_model
from weis.multifidelity.methods.trust_region import SimpleTrustRegion


# Following Algo 2.1 from Andrew March's dissertation
np.random.seed(13)

bounds = {'x' : np.array([[0.0, 80.0], [0.0, 80.0]])}
desvars = {'x' : np.array([40., 40.])}
model_low = barnes_low_model(desvars)
model_high = barnes_high_model(desvars)
trust_region = SimpleTrustRegion(model_low, model_high, bounds, trust_radius=10)

trust_region.add_objective('y')
# trust_region.add_constraint('c1', upper=0.)
# trust_region.add_constraint('c2', upper=0.)
# trust_region.add_constraint('c3', upper=0.)

trust_region.optimize(plot=False)
