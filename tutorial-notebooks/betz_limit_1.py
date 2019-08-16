import scipy
import openmdao.api as om

class ActuatorDisc(om.ExplicitComponent):
    def setup(self):
        # Inputs into the the model
        self.add_input('a', 0.5, desc='Indcued velocity factor')
        self.add_input('Area', 10.0, units='m**2', desc='Rotor disc area')
        self.add_input('rho', 1.225, units='kg/m**3', desc='Air density')
        self.add_input('Vu', 10.0, units='m/s', desc='Freestream air velocity, upstream of rotor')

        # Outputs
        self.add_output('Vr', 0.0, units='m/s', desc='Air velocity at rotor exit plane')
        self.add_output('Vd', 0.0, units='m/s', desc='Slipstream air velocity, downstream of rotor')
        self.add_output('Ct', 0.0, desc='Thrust coefficient')
        self.add_output('Cp', 0.0, desc='Power coefficient')
        self.add_output('power', 0.0, units='W', desc='Power produced by the rotor')
        self.add_output('thrust', 0.0, units='m/s')

        # Partial derivatives. You are telling OpenMDAO that all these will
        # be computed in compute_partials()
        self.declare_partials('Vr', ['a', 'Vu'])
        self.declare_partials('Vd', 'a')
        self.declare_partials('Ct', 'a')
        self.declare_partials('thrust', ['a', 'Area', 'rho', 'Vu'])
        self.declare_partials('Cp', 'a')
        self.declare_partials('power', ['a', 'Area', 'rho', 'Vu'])

    def compute(self, inputs, outputs):
        a = inputs['a']
        Vu = inputs['Vu']
        rho = inputs['rho']
        Area = inputs['Area']
        qA = 0.5 * rho * Area * Vu ** 2
        outputs['Vd'] = Vd = Vu * (1 - 2 * a)
        outputs['Vr'] = 0.5 * (Vu + Vd)
        outputs['Ct'] = Ct = 4 * a * (1 - a)
        outputs['thrust'] = Ct * qA
        outputs['Cp'] = Cp = Ct * (1 - a)
        outputs['power'] = Cp * qA * Vu
        
    def compute_partials(self, inputs, J):
        a = inputs['a']
        Vu = inputs['Vu']
        Area = inputs['Area']
        rho = inputs['rho']

        a_times_area = a * Area
        one_minus_a = 1.0 - a
        a_area_rho_vu = a_times_area * rho * Vu

        # We promised OpenMDAO that we would 
        J['Vr', 'a'] = -Vu
        J['Vr', 'Vu'] = one_minus_a
        J['Vd', 'a'] = -2.0 * Vu
        J['Ct', 'a'] = 4.0 - 8.0 * a
        J['thrust', 'a'] = 0.5 * rho * Vu**2 * Area * J['Ct', 'a']
        J['thrust', 'Area'] = 2.0 * Vu**2 * a * rho * one_minus_a
        J['thrust', 'Vu'] = 4.0 * a_area_rho_vu * one_minus_a
        J['Cp', 'a'] = 4.0 * a * (2.0 * a - 2.0) + 4.0 * one_minus_a**2
        J['power', 'a'] = 2.0 * Area * Vu**3 * a * rho * (2.0 * a - 2.0) + 2.0 * Area * Vu**3 * rho * one_minus_a**2
        J['power', 'Area'] = 2.0 * Vu**3 * a * rho * one_minus_a ** 2
        J['power', 'rho'] = 2.0 * a_times_area * Vu ** 3 * (one_minus_a)**2
        J['power', 'Vu'] = 6.0 * Area * Vu**2 * a * rho * one_minus_a**2

if __name__ == '__main__':
    prob = om.Problem()

    # Add a subsystem that promotes all independent variables as outputs
    # In essence, this is like defining a bunch of independent variables and
    # publishing them to the rest of the subsystems.
    indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('a', 0.5)
    indeps.add_output('Area', 10.0, units='m**2')
    indeps.add_output('rho', 1.225, units='kg/m**3')
    indeps.add_output('Vu', 10.0, units='m/s')

    # The ActuatorDisc model we created above needs the inputs specified
    # below. OpenMDAO will detect the promoted OUTPUTS of 'indeps' and connect
    # them to the promoted INPUTS of 'a_disk'. 
    prob.model.add_subsystem('a_disk', ActuatorDisc(), promotes_inputs=['a', 'Area', 'rho', 'Vu'])
    
    # Prepare to optimize
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.model.add_design_var('a', lower=0.0, upper=1.0)
    prob.model.add_design_var('Area', lower=0.0, upper=1.0)

    # We want to maximize the objective, but OpenMDAO will
    # want to minimize it. So we minimize the negative to find
    # the maximum. Note that a_disk.Cp is not promoted from 'a_disk',
    # therefore we must reference it with a_disk.Cp
    #
    # record (input) a_disk.a -> (output) a_disk.Cp
    prob.model.add_objective('a_disk.Cp', scaler=-1.0)

    prob.setup()
    prob.run_driver()

    # List the inputs and outputs to the model
    prob.model.list_inputs(values=False, hierarchical=False)
    prob.model.list_outputs(values=False, hierarchical=False)

    # Show the minimization result. Note that because Cp
    # wasn't promoted, we need to type a_disk.Cp
    print(f"Coefficient of power Cp = {prob['a_disk.Cp']}")
    print(f"Induction factor a={prob['a']}")
    print(f"Rotor disc Area={prob['Area']} m^2")
