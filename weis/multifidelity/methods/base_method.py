import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from collections import OrderedDict
import smt.surrogate_models as smt


class BaseMethod:
    """
    The base class that all multifidelity optimization methods inherit from.

    Attributes
    ----------
    model_low : BaseModel instance
        The low-fidelity model instance provided by the user.
    model_high : BaseModel instance
        The high-fidelity model instance provided by the user.
    bounds : dict
        A dictionary with keys for each design variable and the values of those
        keys correspond to the design variable bounds, e.g. [[0., 1.], ...].
    disp : bool, optional
        If True, the method will print out progress and results to the terminal.
    counter_plot : int
        Int for the number of plots that have been created so the filenames
        are saved correctly.
    objective : string
        Name of the objective function whose output is provided by the user-provided
        models.
    objective_scaler : float
        Multiplicative scaling factor applied to the objective function before
        passing it to the optimizer. Useful for mamximizing functions instead
        of minimizing them by providing a negative number.
    constraints : list of dicts
        Each dict contains the constraint output name, the constraint type (equality
        or inequality), and the constraint value. One dict for each set of constraints.
    list_of_constraints : list of dicts
        A list of converted constraints and callable functions ready to be used
        by Scipy optimize.
    n_dims : int
        Number of dimensions in the design space, i.e. number of design variables.
    design_vectors : array
        2-D array containing all of the queried design points. Rows (the 2nd dimension)
        contain the design vectors, whereas the number of columns (the 1st dimension)
        correspond to the number of design points.
    approximation_functions : dict of callables
        Dictionary whose keys are the string names of all functions of interest
        (objective and all constraints) and the corresponding values are callable
        funcs for an approximation of those values across the design space.    
    """

    def __init__(self, model_low, model_high, bounds, disp=True, num_initial_points=5):
        """
        Initialize the method by storing the models and populating the
        first points.
        
        Parameters
        ----------
        model_low : BaseModel instance
            The low-fidelity model instance provided by the user.
        model_high : BaseModel instance
            The high-fidelity model instance provided by the user.
        bounds : dict
            A dictionary with keys for each design variable and the values of those
            keys correspond to the design variable bounds, e.g. [[0., 1.], ...].
        disp : bool, optional
            If True, the method will print out progress and results to the terminal.
        num_initial_points : int
            The number of initial points to use to populate the surrogate-based
            approximations of the methods. In general, higher dimensional problems
            require more initial points to get a reasonable surrogate approximation.
        """

        self.model_low = model_low
        self.model_high = model_high
        
        self.bounds = self.flatten_bounds_dict(bounds)
        self.disp = disp

        self.initialize_points(num_initial_points)
        self.counter_plot = 0

        self.objective = None
        self.constraints = []
        
    def flatten_bounds_dict(self, bounds):
        """
        Given a dict of bounds, return an array of bound pairs.
        
        Parameters
        ----------
        bounds : dict
            Dict of bounds keys/values.
            
        Returns
        -------
        flattened_bounds : array
            Flattened array of bounds values.
        """
        flattened_bounds = []

        for key, value in bounds.items():
            if isinstance(value, (float, list)):
                value = np.array(value)
            flattened_value = np.squeeze(value.flatten()).reshape(-1, 2)
            flattened_bounds.extend(flattened_value)

        return np.array(flattened_bounds)

    def initialize_points(self, num_initial_points):
        """
        Populate the design_vectors array with a set of initial points that will be used
        to create the initial surrogate models.
        
        Modifies `design_vectors` in-place.
        
        Parameters
        ----------
        num_initial_points : int
            Number of points used to populate the initial surrogate model creation.
            These points are called using both the low- and high-fidelity models.
        
        """
        self.n_dims = self.model_high.total_size
        x_init_raw = np.random.rand(num_initial_points, self.n_dims)
        self.design_vectors = (
            x_init_raw * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        )

    def set_initial_point(self, initial_design):
        """
        Set the initial point for the optimization method.
        
        Modifies `design_vectors` in-place.
        
        Parameters
        ----------
        initial_design : array
            Initial design point for the optimization method.
        
        """
        if isinstance(initial_design, (float, list)):
            initial_design = np.array(initial_design)
        self.design_vectors = np.vstack((self.design_vectors, initial_design))

    def add_objective(self, objective_name, scaler=1.0):
        """
        Set the optimization objective string and scaler.
        
        Parameters
        ----------
        objective : string
            Name of the objective function whose output is provided by the user-provided
            models.
        scaler : float
            Multiplicative scaling factor applied to the objective function before
            passing it to the optimizer. Useful for mamximizing functions instead
            of minimizing them by providing a negative number.
        
        """
        self.objective = objective_name
        self.objective_scaler = scaler

    def add_constraint(self, constraint_name, equals=None, lower=None, upper=None):
        """
        Append user-defined constraints into a list of dicts with all constraint
        info to be used later.
        
        Modifies `constraints` in-place.
        
        Parameters
        ----------
        constraint_name : string
            Name of the output value to constrain.
        equals : float or None
            If a float, the value at which to constrain the output.
        lower : float or None
            If a float, the value of the lower bound for the constraint.
        upper : float or None
            If a float, the value of the upper bound for the constraint.
        
        """
        self.constraints.append(
            {"name": constraint_name, "equals": equals, "lower": lower, "upper": upper,}
        )

    def process_constraints(self):
        """
        Convert the list of user-defined constraint dicts into constraint functions
        compatible with Scipy optimize.
        
        Modifies `list_of_constraints` in-place.
        
        """
        list_of_constraints = []
        for constraint in self.constraints:
            scipy_constraint = {}

            func = self.approximation_functions[constraint["name"]]
            if constraint["equals"] is not None:
                scipy_constraint["type"] = "eq"
                scipy_constraint["fun"] = lambda x: np.squeeze(
                    func(x) - constraint["equals"]
                )

            if constraint["upper"] is not None:
                scipy_constraint["type"] = "ineq"
                scipy_constraint["fun"] = lambda x: np.squeeze(
                    constraint["upper"] - func(x)
                )

            if constraint["lower"] is not None:
                scipy_constraint["type"] = "ineq"
                scipy_constraint["fun"] = lambda x: np.squeeze(
                    func(x) - constraint["lower"]
                )

            list_of_constraints.append(scipy_constraint)

        self.list_of_constraints = list_of_constraints

    def construct_approximations(self, interp_method="smt"):
        """
        Create callable functions for each of the corrected low-fidelity
        models by constructing surrogate models for the error between
        low- and high-fidelity results.
        
        Follows the process laid out by multiple trust-region methods presented
        in the literature where we construct a corrected low-fidelity model.
        This correction comes from a surrogate model trained by the error between
        the low- and high-fidelity models. Each call to one of these callable
        funcs runs the low-fidelity model and queries the surrogate model at
        that design point to obtain the corrected output.
        
        This method create approximation functions for the objective function and
        all constraints in the same way.
        
        Depending on the smoothness of the underlying functions, certain surrogate
        models may be better suited to model the error between the models.
        Future studies could focus on when to use which surrogate model.
        
        Modifies `approximation_functions` in-place.
        
        Parameters
        ----------
        interp_method : string
            Set the type of surrogate model method to use, valid options are 'rbf'
            and 'smt' for now. 
        
        """
        outputs_low = self.model_low.run_vec(self.design_vectors)
        outputs_high = self.model_high.run_vec(self.design_vectors)

        approximation_functions = {}
        outputs_to_approximate = [self.objective]

        if len(self.constraints) > 0:
            for constraint in self.constraints:
                outputs_to_approximate.append(constraint["name"])

        for output_name in outputs_to_approximate:
            differences = outputs_high[output_name] - outputs_low[output_name]

            if interp_method == "rbf":
                input_arrays = np.split(self.design_vectors, self.design_vectors.shape[1], axis=1)
                input_arrays = [x.flatten() for x in input_arrays]

                # Construct RBF interpolater for error function
                e = Rbf(*input_arrays, differences)

                # Create m_k = lofi + RBF
                def approximation_function(x):
                    return self.model_low.run(x)[output_name] + e(*x)

            elif interp_method == "smt":
                # sm = smt.RMTB(
                #     xlimits=self.bounds,
                #     order=3,
                #     num_ctrl_pts=5,
                #     print_global=False,
                # )
                sm = smt.RBF(print_global=False,)

                sm.set_training_values(self.design_vectors, differences)
                sm.train()

                def approximation_function(x, output_name=output_name, sm=sm):
                    return self.model_low.run(x)[output_name] + sm.predict_values(
                        np.atleast_2d(x)
                    )

            # Create m_k = lofi + RBF
            approximation_functions[output_name] = approximation_function

        self.approximation_functions = approximation_functions
