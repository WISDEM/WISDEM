import os
import unittest
from pathlib import Path

import numpy as np
#import numpy.testing as npt

import wisdem.inputs.validation as val

sample_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sample.yaml")

class TestValidation(unittest.TestCase):

    def testLoadStringPath(self):
        obj_str = val.load_yaml(val.fschema_opt)
        obj_path = val.load_yaml(Path(val.fschema_opt))

        self.assertEqual(obj_str, obj_path)

    def test_write_dict(self):
        obj1 = val.load_yaml(sample_yaml)
        obj2 = val.load_yaml(sample_yaml)

        obj1['components']['tower']['outer_shape_bem']['reference_axis']['z']['values'] = list(
            obj1['components']['tower']['outer_shape_bem']['reference_axis']['z']['values']
            )
        
        obj2['components']['tower']['outer_shape_bem']['reference_axis']['z']['values'] = np.array(
            obj2['components']['tower']['outer_shape_bem']['reference_axis']['z']['values']
            )

        ftemp1 = 'temp1.yaml'
        ftemp2 = 'temp2.yaml'
        val.write_yaml(obj1, ftemp1)
        val.write_yaml(obj2, ftemp2)
        
        obj1p = val.load_yaml(ftemp1)
        obj2p = val.load_yaml(ftemp2)

        self.assertEqual(obj1p, obj2p)
        
if __name__ == "__main__":
    unittest.main()
