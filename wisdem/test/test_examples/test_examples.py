import os
import unittest
import importlib
from pathlib import Path

thisdir = os.path.dirname(os.path.realpath(__file__))
# Find examples directory- outside the module path

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(thisdir)))
examples_dir = os.path.join(root_dir, "examples")
all_examples = Path(examples_dir).rglob("*.py") if os.path.exists(examples_dir) else []

# 02_ref turbines are regression tested in test_gluecode, no need to duplicate runtime
all_scripts = [
    "01_nrel_csm/costs",
    "01_nrel_csm/mass",
    "01_nrel_csm/mass_and_cost",
    "01_nrel_csm/parametric",
    "03_blade/blade_driver",
    "04_openmdao/betz_limit",
    "04_openmdao/sellar",
    "05_tower_monopile/monopile_direct",
    "05_tower_monopile/monopile_driver",
    "05_tower_monopile/tower_direct",
    "05_tower_monopile/tower_driver",
    "06_drivetrain/drivetrain_direct",
    "06_drivetrain/drivetrain_geared",
    "07_generator/dfig",
    "07_generator/eesg",
    "07_generator/pmsg_arms",
    "07_generator/pmsg_disc",
    "07_generator/pmsg_outer",
    "07_generator/scig",
    "08_plant_finance/example",
    "09_floating/mooring_opt",
    "09_floating/semi_example",
    "09_floating/spar_example",
    "09_floating/spar_opt",
    "09_floating/tlp_example",
    "09_weis_floating/wisdem_driver_oc3",
    "09_weis_floating/wisdem_driver_oc4",
    "10_ccblade/example",
    "10_ccblade/gradients",
    "10_ccblade/precurve",
    "11_airfoilprep/example",
    "12_pyframe3dd/exB",
]


def execute_script(fscript):
    # Go to location due to relative path use for airfoil files
    print("\n\n")
    print("NOW RUNNING:", fscript)
    print()
    fullpath = os.path.join(examples_dir, fscript + ".py")
    basepath = os.path.join(examples_dir, fscript.split("/")[0])
    os.chdir(basepath)

    # Get script/module name
    froot = fscript.split("/")[-1]

    # Use dynamic import capabilities
    # https://www.blog.pythonlibrary.org/2016/05/27/python-201-an-intro-to-importlib/
    print(froot, os.path.realpath(fullpath))
    spec = importlib.util.spec_from_file_location(froot, os.path.realpath(fullpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


class TestExamples(unittest.TestCase):
    def test_nrel_csm(self):
        scripts = [m for m in all_scripts if m.find("csm") >= 0]
        for k in scripts:
            try:
                execute_script(k)
            except:
                print("Failed to run,", k)
                self.assertTrue(False)

    def test_reference_turbines(self):
        scripts = [m for m in all_scripts if m.find("reference") >= 0]
        for k in scripts:
            try:
                execute_script(k)
            except:
                print("Failed to run,", k)
                self.assertTrue(False)

    def test_blade(self):
        scripts = [m for m in all_scripts if m.find("blade") >= 0]
        for k in scripts:
            try:
                execute_script(k)
            except:
                print("Failed to run,", k)
                self.assertTrue(False)

    def test_openmdao(self):
        scripts = [m for m in all_scripts if m.find("openmdao") >= 0]
        for k in scripts:
            try:
                execute_script(k)
            except:
                print("Failed to run,", k)
                self.assertTrue(False)

    def test_tower_monopile(self):
        scripts = [m for m in all_scripts if m.find("monopile") >= 0]
        for k in scripts:
            try:
                execute_script(k)
            except:
                print("Failed to run,", k)
                self.assertTrue(False)

    def test_drivetrain(self):
        scripts = [m for m in all_scripts if m.find("drivetrain") >= 0]
        for k in scripts:
            try:
                execute_script(k)
            except:
                print("Failed to run,", k)
                self.assertTrue(False)

    def test_generator(self):
        scripts = [m for m in all_scripts if m.find("generator") >= 0]
        for k in scripts:
            try:
                execute_script(k)
            except:
                print("Failed to run,", k)
                self.assertTrue(False)

    def test_plant_finance(self):
        scripts = [m for m in all_scripts if m.find("finance") >= 0]
        for k in scripts:
            try:
                execute_script(k)
            except:
                print("Failed to run,", k)
                self.assertTrue(False)

    def test_floating(self):
        scripts = [m for m in all_scripts if m.find("floating") >= 0]
        for k in scripts:
            try:
                execute_script(k)
            except:
                print("Failed to run,", k)
                self.assertTrue(False)

    def test_ccblade(self):
        scripts = [m for m in all_scripts if m.find("ccblade") >= 0]
        for k in scripts:
            try:
                execute_script(k)
            except:
                print("Failed to run,", k)
                self.assertTrue(False)

    def test_airfoilprep(self):
        scripts = [m for m in all_scripts if m.find("airfoil") >= 0]
        for k in scripts:
            try:
                execute_script(k)
            except:
                print("Failed to run,", k)
                self.assertTrue(False)

    def test_pyframe3dd(self):
        scripts = [m for m in all_scripts if m.find("pyframe") >= 0]
        for k in scripts:
            try:
                execute_script(k)
            except:
                print("Failed to run,", k)
                self.assertTrue(False)

    """
    def testAllExamplesRun(self):
        for f in all_examples:
            try:
                if 'design_compare.py' not in str(f):
                    # Go to location due to relative path use for airfoil files
                    print('\n\n')
                    print('NOW RUNNING:',f)
                    print()
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
    """


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestExamples))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
