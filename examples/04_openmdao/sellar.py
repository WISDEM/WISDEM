# Import the libraries we need
import openmdao.api as om
import numpy as np

# -------

# Create the mock mathematical relationships associated with Discipline 1 using an OpenMDAO component
class SellarDis1(om.ExplicitComponent):
    """
    Component containing Discipline 1 -- no derivatives version.
    """

    def setup(self):

        # Global Design Variable
        self.add_input("z", val=np.zeros(2))

        # Local Design Variable
        self.add_input("x", val=0.0)

        # Coupling parameter
        self.add_input("y2", val=1.0)

        # Coupling output
        self.add_output("y1", val=1.0)

        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y1 = z1**2 + z2 + x1 - 0.2*y2
        """
        z1 = inputs["z"][0]
        z2 = inputs["z"][1]
        x1 = inputs["x"]
        y2 = inputs["y2"]

        outputs["y1"] = z1 ** 2 + z2 + x1 - 0.2 * y2
        # -- end Discipline 1


# Now create the mock mathematical relationships associated with Discipline 2 using an OpenMDAO component
class SellarDis2(om.ExplicitComponent):
    """
    Component containing Discipline 2 -- no derivatives version.
    """

    def setup(self):
        # Global Design Variable
        self.add_input("z", val=np.zeros(2))

        # Coupling parameter
        self.add_input("y1", val=1.0)

        # Coupling output
        self.add_output("y2", val=1.0)

        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y2 = y1**(.5) + z1 + z2
        """

        z1 = inputs["z"][0]
        z2 = inputs["z"][1]
        y1 = inputs["y1"]

        outputs["y2"] = y1 ** 0.5 + z1 + z2
        # -- end Discipline 2


# Assemble the system together in an OpenMDAO Group
class SellarMDA(om.Group):
    """
    Group containing the Sellar MDA.
    """

    def setup(self):
        indeps = self.add_subsystem("indeps", om.IndepVarComp(), promotes=["*"])
        indeps.add_output("x", 1.0)
        indeps.add_output("z", np.array([5.0, 2.0]))

        self.add_subsystem("d1", SellarDis1(), promotes=["y1", "y2"])
        self.add_subsystem("d2", SellarDis2(), promotes=["y1", "y2"])
        self.connect("x", "d1.x")
        self.connect("z", ["d1.z", "d2.z"])

        # Nonlinear Block Gauss Seidel is a gradient free solver to handle implicit loops
        self.nonlinear_solver = om.NonlinearBlockGS()

        self.add_subsystem(
            "obj_cmp",
            om.ExecComp("obj = x**2 + z[1] + y1 + exp(-y2)", z=np.array([0.0, 0.0]), x=0.0),
            promotes=["x", "z", "y1", "y2", "obj"],
        )

        self.add_subsystem("con_cmp1", om.ExecComp("con1 = 3.16 - y1"), promotes=["con1", "y1"])
        self.add_subsystem("con_cmp2", om.ExecComp("con2 = y2 - 24.0"), promotes=["con2", "y2"])
        # -- end Group


# Instantiate the model
prob = om.Problem()
prob.model = SellarMDA()
# --------

# Specify the optimization 'driver' and options
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.options["tol"] = 1e-8
# --------

# Assign objective and design variables
prob.model.add_design_var("x", lower=0, upper=10)
prob.model.add_design_var("z", lower=0, upper=10)
prob.model.add_objective("obj")
prob.model.add_constraint("con1", upper=0)
prob.model.add_constraint("con2", upper=0)

# Ask OpenMDAO to finite-difference across the whole model to compute the total gradients for the optimizer
# The other approach would be to finite-difference for the partials and build up the total derivative
prob.model.approx_totals()
# --------

# Execute! and display result
prob.setup()
prob.run_driver()

print("minimum found at")
print(float(prob["x"]))
print(prob["z"])
print("minimum objective")
print(float(prob["obj"]))
# --------
