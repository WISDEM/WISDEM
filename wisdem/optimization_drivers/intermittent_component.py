import openmdao.api as om


class IntermittentComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_iterations_between_calls", 3)
        self.frozen_outputs = {}
        self.actual_compute_calls = 0

    def compute(self, inputs, outputs):
        """
        This is a wrapper for the actual compute call, `internal_compute()`,
        which calls the compute method every `num_iterations_between_calls`.
        This allows more expensive analyses to be run less often at the expense
        of accuracy in an optimization context.
        If you want the compute to be run only once at the beginning of the
        optimization, set `num_iterations_between_calls` to a very large number.
        """

        num_iterations_between_calls = self.options["num_iterations_between_calls"]

        # Determine if we are in a compute call in which we want to update the
        # frozen outputs
        regular_compute = (self.iter_count_without_approx % num_iterations_between_calls) == 0 and not self.under_approx
        approx_compute = (
            (self.iter_count_without_approx - 1) % num_iterations_between_calls
        ) == 0 and self.under_approx

        # If we're within one of those types of compute calls, call the actual
        # internal_compute() method to update the results
        if regular_compute or approx_compute:
            self.internal_compute(inputs, outputs)
            self.actual_compute_calls += 1

            # Save off the results to the frozen_outputs dict
            for key in outputs:
                self.frozen_outputs[key] = outputs[key]

        # If we're using the frozen results, simply set the outputs from those results
        else:
            for key in outputs:
                outputs[key] = self.frozen_outputs[key]

    def internal_compute(self, inputs, outputs):
        """
        This is the actual method where the computations should occur.
        """
        raise NotImplementedError("Please define an `internal_compute` method.")
