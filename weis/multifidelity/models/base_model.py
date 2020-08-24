import os
import dill
from collections import OrderedDict
import numpy as np


class BaseModel:
    """
    Base model implementation of a model that all others inherit from.
    
    This class contains shared methods for any model implementation. These model
    implementations contain any functions to be called within a multifidelity
    optimization method. The models could be simple analytic functions, could
    wrap tools like WISDEM or OpenFAST, or could call external applications
    using file IO.
    """

    def __init__(self, desvars_init, warmstart_file=None):
        """
        Initialize the model using an initial set of design variables.
        
        If `warmstart_file` is given, load saved results from there and
        save off future function call results to that pickle file.
        
        Parameters
        ----------
        desvars_init : dict of desvars
            A dict containing keys with the names of desvars and the corresponding
            values are representative of the desvar size and shape. These values
            are not used; only their shape
        """

        self.saved_desvars = []
        self.saved_outputs = []

        self.warmstart_file = warmstart_file

        # If warmstart_file is provided, read in saved results from before
        if warmstart_file is not None:
            if os.path.exists(warmstart_file):
                with open(warmstart_file, "rb") as f:
                    saved_data = dill.load(f)

                # Save those results as lists attributed to this class
                self.saved_desvars = saved_data["desvars"]
                self.saved_outputs = saved_data["outputs"]

        # Save off design variable sizes so we can flatten and unflatten
        # the desvars vector
        self.set_desvar_size_dict(desvars_init)

    def set_desvar_size_dict(self, desvars):
        """
        Save desvar sizes so we can flatten and unflatten the desvar vector.
        
        This is needed because the Scipy optimizer and surrogate models
        need a flattened array of input values, not a dictionary of keys
        and desvars.
        
        Parameters
        ----------
        desvars : dict of desvars
            These desvars are simply used to record the size and shape of each
            desvar for use later when flattening/unflattening desvars.
        """
        self.desvar_shapes = OrderedDict()
        total_size = 0

        for key, value in desvars.items():
            if isinstance(value, (float, list)):
                value = np.array(value)

            self.desvar_shapes[key] = value.shape
            total_size += value.size

        self.total_size = total_size

    def save_results(self, desvars, outputs):
        """
        Save results to attribute lists and to the pickle file if provided.
        
        Parameters
        ----------
        desvars : dict of desvars
            Keys and values for desvars to save.
        outputs : dict of outputs
            Keys and values for output values from `compute()` to save.
        """
        self.saved_desvars.append(self.flatten_desvars(desvars))
        self.saved_outputs.append(outputs)

        # Only save to the pickle file if warmstart_file was provided
        if self.warmstart_file is not None:
            saved_data = {}
            saved_data["desvars"] = self.saved_desvars
            saved_data["outputs"] = self.saved_outputs
            with open(self.warmstart_file, "wb") as f:
                dill.dump(saved_data, f)

    def load_results(self, flattened_desvars):
        """
        Load results from the attribute lists for design points that have
        been previously evaluated.
        
        Parameters
        ----------
        flattened_desvars : array
            Flattened array of all desvar values at the desired design point.
        
        Returns
        -------
        outputs : dict or None
            If there are saved outputs, the dict contains those key/value
            output pairs. Otherweise, the return value is None. 
        """
        for i, saved_desvar in enumerate(self.saved_desvars):
            same_inputs = True

            # Check each of the saved input arrays against the newest
            # input array. If the function has been queried at that point
            # already, return the saved results.
            if not np.all(flattened_desvars == saved_desvar):
                same_inputs = False
                continue

            if same_inputs:
                return self.saved_outputs[i]

        # Else, return None, so the function needs to be evaluated at this point
        return None

    def compute(self, desvars):
        """
        Method to actually compute function outputs given desvars, needs to be
        implemented by the user in their model.
        
        Parameters
        ----------
        desvars : dict
            Dict of desvar keys/values.
            
        Returns
        -------
        outputs : dict
            Dict of output keys/values.
        """
        raise NotImplementedError(
            "This method needs to be provided by the inheriting model class."
        )

    def run(self, flattened_desvars):
        """
        Light wrapper for the user-defined `compute` method which checks
        if the design point has already been run.
        
        This method uses the flattened_desvars, or an array of only the desvar
        values, not the desvar dictionary. This makes it slightly easier
        to compare the input values and reduces the amount of translation needed
        for the optimization and surrgoate methods.
        
        Parameters
        ----------
        flattened_desvars : array
            Flattened array of all desvar values at the desired design point.
        
        Returns
        -------
        outputs : dict
            Returned dict of outputs from the `compute()` method provided by
            the user-defined model.
        """
        # Load results from previous run
        loaded_results = self.load_results(flattened_desvars)

        # If there are not results from before, actually run the model's
        # compute method and save the results
        if loaded_results is None:
            desvars = self.unflatten_desvars(flattened_desvars)
            outputs = self.compute(desvars)
            self.save_results(desvars, outputs)
            return outputs

        # Return the loaded results if applicable
        else:
            return loaded_results

    def run_vec(self, multiple_flattened_desvars):
        """
        Light wrapper to run the model at multiple design points.
        
        Parameters
        ----------
        multiple_flattened_desvars : array
            2D array of flattened desvars to query the model.
            
        Returns
        -------
        dict_of_results : dict
            Dict of key/array pairs where the arrays contain results
            for all of the design points listed in multiple_flattened_desvars.
        """
        dict_of_results = {}

        # Loop through each row of design variable arrays
        for i, flattened_desvars in enumerate(multiple_flattened_desvars):

            # Actually run the model (or grab saved results) at that design point
            outputs = self.run(flattened_desvars)

            # Pack the results into a dictionary
            for key in outputs:
                if key not in dict_of_results:
                    dict_of_results[key] = []
                dict_of_results[key].append(outputs[key])

        # Loop through each dict key and transform the lists into squeezed arrays
        for key in dict_of_results:
            dict_of_results[key] = np.squeeze(np.array(dict_of_results[key]))

        return dict_of_results

    def flatten_desvars(self, desvars):
        """
        Given a dict of desvars, return a flattened array of desvar values.
        
        Parameters
        ----------
        desvars : dict
            Dict of desvar keys/values.
            
        Returns
        -------
        flattened_desvars : array
            Flattened array of desvar values.
        """
        flattened_desvars = []

        for key, value in desvars.items():
            if isinstance(value, (float, list)):
                value = np.array(value)
            flattened_value = np.atleast_1d(np.squeeze(value.flatten()))
            flattened_desvars.extend(flattened_value)

        return np.array(flattened_desvars)

    def unflatten_desvars(self, flattened_desvars):
        """
        Given a flattened array of desvar values, return a dict of desvars.
        
        Parameters
        ----------
        flattened_desvars : array
            Flattened array of desvar values.
            
        Returns
        -------
        desvars : dict
            Dict of desvar keys/values.
        """
        size_counter = 0
        desvars = OrderedDict()
        for key, shape in self.desvar_shapes.items():
            size = int(np.prod(shape))
            desvars[key] = flattened_desvars[
                size_counter : size_counter + size
            ].reshape(shape)
            size_counter += size

        return desvars
