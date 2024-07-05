import os
import unittest
import importlib
from time import time
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
    "02_reference_turbines/iea10mw_driver",
    "02_reference_turbines/iea3p4mw_driver",
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
    "09_floating/semi_only_driver",
    "09_floating/spar_only_driver",
    "09_floating/spar_opt",
    "09_floating/tlp_example",
    "09_floating/nrel5mw-spar_oc3_driver",
    "09_floating/nrel5mw-semi_oc4_driver",
    "09_floating/iea15mw_driver",
    "10_ccblade/example",
    "10_ccblade/gradients",
    "10_ccblade/precurve",
    "11_user_custom/user_custom_opt",
    "12_pyframe3dd/exB",
    "13_design_of_experiments/doe_driver",
    "14_overridden_values/driver",
    "15_step_size_study/driver",
    "16_inverse_design/inverse_spar_design",
    "16_inverse_design/inverse_rotor_design",
    "17_jacket/jacket_driver",
    "18_rotor_tower_monopile/design_run",
    "19_rotor_drivetrain_tower/wisdem_driver"
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
    s = time()
    spec.loader.exec_module(mod)
    print(time() - s, "seconds to run")


class TestExamples(unittest.TestCase):
    def test_all_scripts(self):
        for ks, s in enumerate(all_scripts):
            with self.subTest(f"Running: {s}", i=ks):
                try:
                    execute_script(s)
                    self.assertTrue(True)
                except Exception:
                    self.assertEqual(s, "Success")


def suite():
    suite = [
        unittest.TestLoader().loadTestsFromTestCase(TestExamples),
    ]
    return unittest.TestSuite(suite)


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
