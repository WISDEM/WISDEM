import unittest
import numpy as np
from weis.multifidelity.models.testbed_components import simple_2D_high_model, simple_2D_low_model
from weis.multifidelity.methods.trust_region import SimpleTrustRegion


class Test(unittest.TestCase):

    def test_optimization(self):
        np.random.seed(13)
        
        bounds = {'x' : np.array([[0.0, 1.0], [0.0, 1.0]])}
        desvars = {'x' : np.array([0., 0.25])}
        model_low = simple_2D_low_model(desvars)
        model_high = simple_2D_high_model(desvars)
        trust_region = SimpleTrustRegion(model_low, model_high, bounds, disp=False)
        
        trust_region.add_objective('y')
        
        results = trust_region.optimize()
        
        np.testing.assert_allclose(results['optimal_design'], [0., 0.333], atol=1e-3)
        

if __name__ == '__main__':
    unittest.main()