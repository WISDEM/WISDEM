import unittest
import os
import importlib
from pathlib import Path

thisdir = os.path.dirname(os.path.realpath(__file__))
# Find examples directory- outside the module path

root_dir = os.path.dirname( os.path.dirname( os.path.dirname( thisdir )))
examples_dir = os.path.join(root_dir, 'examples')
all_examples = Path(examples_dir).rglob('*.py') if os.path.exists(examples_dir) else []

class TestExamples(unittest.TestCase):

    def testAllExamplesRun(self):
        for f in all_examples:
            try:
                if 'design_compare.py' not in str(f):
                    # Go to location due to relative path use for airfoil files
                    print(f)
                    basepath = os.path.dirname(os.path.realpath(f))
                    os.chdir(basepath)

                    # Get script/module name
                    froot = os.path.splitext(os.path.basename(f))[0]

                    # Use dynamic import capabilities
                    # https://www.blog.pythonlibrary.org/2016/05/27/python-201-an-intro-to-importlib/
                    spec = importlib.util.spec_from_file_location(froot, os.path.realpath(f))
                    mod  = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
            except:
                self.assertTrue(False)
 
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestExamples))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
