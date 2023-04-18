# Import the OpenMDAO library
import openmdao.api as om

# --


# Specific the Actuator Disc theory into a derived OpenMDAO class
class ActuatorDisc(om.ExplicitComponent):
    # Inputs and Outputs
    def setup(self):
        # Inputs into the the model
        self.add_input("a", 0.5, desc="Indcued velocity factor")
        self.add_input("Area", 10.0, units="m**2", desc="Rotor disc area")
        self.add_input("rho", 1.225, units="kg/m**3", desc="Air density")
        self.add_input("Vu", 10.0, units="m/s", desc="Freestream air velocity, upstream of rotor")

        # Outputs
        self.add_output("Vr", 0.0, units="m/s", desc="Air velocity at rotor exit plane")
        self.add_output("Vd", 0.0, units="m/s", desc="Slipstream air velocity, downstream of rotor")
        self.add_output("Ct", 0.0, desc="Thrust coefficient")
        self.add_output("Cp", 0.0, desc="Power coefficient")
        self.add_output("power", 0.0, units="W", desc="Power produced by the rotor")
        self.add_output("thrust", 0.0, units="m/s")

        # Declare which outputs are dependent on which inputs
        self.declare_partials("Vr", ["a", "Vu"])
        self.declare_partials("Vd", "a")
        self.declare_partials("Ct", "a")
        self.declare_partials("thrust", ["a", "Area", "rho", "Vu"])
        self.declare_partials("Cp", "a")
        self.declare_partials("power", ["a", "Area", "rho", "Vu"])
        # --------

    # Core theory
    def compute(self, inputs, outputs):
        a = inputs["a"]
        Vu = inputs["Vu"]
        rho = inputs["rho"]
        Area = inputs["Area"]
        qA = 0.5 * rho * Area * Vu**2
        outputs["Vd"] = Vd = Vu * (1 - 2 * a)
        outputs["Vr"] = 0.5 * (Vu + Vd)
        outputs["Ct"] = Ct = 4 * a * (1 - a)
        outputs["thrust"] = Ct * qA
        outputs["Cp"] = Cp = Ct * (1 - a)
        outputs["power"] = Cp * qA * Vu
        # --------

    # Derivatives of outputs wrt inputs
    def compute_partials(self, inputs, J):
        a = inputs["a"]
        Vu = inputs["Vu"]
        Area = inputs["Area"]
        rho = inputs["rho"]

        a_times_area = a * Area
        one_minus_a = 1.0 - a
        a_area_rho_vu = a_times_area * rho * Vu

        J["Vr", "a"] = -Vu
        J["Vr", "Vu"] = one_minus_a
        J["Vd", "a"] = -2.0 * Vu
        J["Ct", "a"] = 4.0 - 8.0 * a
        J["thrust", "a"] = 0.5 * rho * Vu**2 * Area * J["Ct", "a"]
        J["thrust", "Area"] = 2.0 * Vu**2 * a * rho * one_minus_a
        J["thrust", "Vu"] = 4.0 * a_area_rho_vu * one_minus_a
        J["Cp", "a"] = 4.0 * a * (2.0 * a - 2.0) + 4.0 * one_minus_a**2
        J["power", "a"] = (
            2.0 * Area * Vu**3 * a * rho * (2.0 * a - 2.0) + 2.0 * Area * Vu**3 * rho * one_minus_a**2
        )
        J["power", "Area"] = 2.0 * Vu**3 * a * rho * one_minus_a**2
        J["power", "rho"] = 2.0 * a_times_area * Vu**3 * (one_minus_a) ** 2
        J["power", "Vu"] = 6.0 * Area * Vu**2 * a * rho * one_minus_a**2
        # -- end the class


# Optional: include underlying model in a group with Independent Variables
class Betz(om.Group):
    """
    Group containing the actuator disc equations for deriving the Betz limit.
    """

    def setup(self):
        indeps = self.add_subsystem("indeps", om.IndepVarComp(), promotes=["*"])
        indeps.add_output("a", 0.5)
        indeps.add_output("Area", 10.0, units="m**2")
        indeps.add_output("rho", 1.225, units="kg/m**3")
        indeps.add_output("Vu", 10.0, units="m/s")

        self.add_subsystem("a_disk", ActuatorDisc(), promotes=["a", "Area", "rho", "Vu"])
        # --------


# Instantiate the model
prob = om.Problem()
prob.model = Betz()
# -----

# Specify the optimization 'driver'
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
# -----

# Assign objective and design variables
prob.model.add_design_var("a", lower=0.0, upper=1.0)
prob.model.add_design_var("Area", lower=0.0, upper=1.0)
prob.model.add_objective("a_disk.Cp", scaler=-1.0)
# -----

# Execute!
prob.setup()
prob.run_driver()
# --------

# Display the output
print("Coefficient of power Cp = ", prob["a_disk.Cp"])
print("Induction factor a =", prob["a"])
print("Rotor disc Area =", prob["Area"], "m^2")
prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)
# --------
