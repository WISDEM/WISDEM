# base method class
import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import OrderedDict
import smt.surrogate_models as smt
from weis.multifidelity.models.testbed_components import (
    simple_2D_high_model,
    simple_2D_low_model,
)
from weis.multifidelity.methods.base_method import BaseMethod


class SimpleTrustRegion(BaseMethod):
    """
    An implementation of a simple trust-region method, following
    the literature as presented in multiple places, but especially
    Andrew March's dissertation.

    Attributes
    ----------
    max_trust_radius : float
        Maximum size of the trust region radius. Not scaled, so this value
        is in real dimensioned values of the design space.
    eta : float
        Value to compare the ratio of actual reduction to predicted reduction
        of the objective value. A ratio higher than eta expands the trust region
        whereas a value lower than eta contracts the trust region.
    gtol : float
        Tolerance of the gradient for convergence criteria. If the norm
        of the gradient at the current design point is less than gtol,
        terminate the trust region method. Currently not implemented.
    trust_radius : float
        The current value of the trust region radius in dimensioned units.
    """

    def __init__(
        self,
        model_low,
        model_high,
        bounds,
        disp=True,
        num_initial_points=5,
        max_trust_radius=1000.0,
        eta=0.25,
        gtol=1e-4,
        trust_radius=0.2,
    ):
        """
        Initialize the trust region method and store the user-defined options.

        Parameters
        ----------
        max_trust_radius : float
            Maximum size of the trust region radius. Not scaled, so this value
            is in real dimensioned values of the design space.
        eta : float
            Value to compare the ratio of actual reduction to predicted reduction
            of the objective value. A ratio higher than eta expands the trust region
            whereas a value lower than eta contracts the trust region.
        gtol : float
            Tolerance of the gradient for convergence criteria. If the norm
            of the gradient at the current design point is less than gtol,
            terminate the trust region method. Currently not implemented.
        trust_radius : float
            The current value of the trust region radius in dimensioned units.


        """
        super().__init__(model_low, model_high, bounds, disp, num_initial_points)

        self.max_trust_radius = max_trust_radius
        self.eta = eta
        self.gtol = gtol
        self.trust_radius = trust_radius

    def find_next_point(self):
        """
        Find the design point corresponding to the minimum value of the
        corrected low-fidelity model within the trust region.

        This method uses the most recent design point in design_vectors as the initial
        point for the local optimization.

        Returns
        -------
        x_new : array
            The design point corresponding to the minimum value of the corrected
            low-fidelity model within the trust region.
        hits_boundary : boolean
            True is the new design point hits one of the boundaries of the
            trust region.
        """
        x0 = self.design_vectors[-1, :]

        # min (m_k(x_k + s_k)) st ||x_k|| <= del K
        trust_region_lower_bounds = x0 - self.trust_radius
        lower_bounds = np.maximum(trust_region_lower_bounds, self.bounds[:, 0])
        trust_region_upper_bounds = x0 + self.trust_radius
        upper_bounds = np.minimum(trust_region_upper_bounds, self.bounds[:, 1])

        bounds = list(zip(lower_bounds, upper_bounds))
        scaled_function = lambda x: self.objective_scaler * np.squeeze(
            self.approximation_functions[self.objective](x)
        )
        res = minimize(
            scaled_function,
            x0,
            method="SLSQP",
            tol=1e-10,
            bounds=bounds,
            constraints=self.list_of_constraints,
            options={"disp": False},
        )
        x_new = res.x

        tol = 1e-6
        if np.any(np.abs(trust_region_lower_bounds - x_new) < tol) or np.any(
            np.abs(trust_region_upper_bounds - x_new) < tol
        ):
            hits_boundary = True
        else:
            hits_boundary = False

        return x_new, hits_boundary

    def update_trust_region(self, x_new, hits_boundary):
        """
        Either expand or contract the trust region radius based on the
        value of the high-fidelity function at the proposed design point.

        Modifies `design_vectors` and `trust_radius` in-place.

        Parameters
        ----------
        x_new : array
            The design point corresponding to the minimum value of the corrected
            low-fidelity model within the trust region.
        hits_boundary : boolean
            True is the new design point hits one of the boundaries of the
            trust region.

        """
        # 3. Compute the ratio of actual improvement to predicted improvement
        prev_point_high = (
            self.objective_scaler
            * self.model_high.run(self.design_vectors[-1])[self.objective]
        )
        new_point_high = (
            self.objective_scaler * self.model_high.run(x_new)[self.objective]
        )
        new_point_approx = self.objective_scaler * self.approximation_functions[
            self.objective
        ](x_new)

        actual_reduction = prev_point_high - new_point_high
        predicted_reduction = prev_point_high - new_point_approx

        # 4. Accept or reject the trial point according to that ratio
        # Unclear if this logic is needed; it's better to update the surrogate model with a bad point, even
        if predicted_reduction <= 0:
            if self.disp:
                print("not enough reduction! rejecting point")
        else:
            self.design_vectors = np.vstack((self.design_vectors, np.atleast_2d(x_new)))

        if predicted_reduction == 0.0:
            rho = 0.0
        else:
            rho = actual_reduction / predicted_reduction

        # 5. Update trust region according to rho_k
        if rho >= self.eta and hits_boundary:
            self.trust_radius = min(2 * self.trust_radius, self.max_trust_radius)
        elif rho < self.eta:  # Unclear if this is the best check
            self.trust_radius *= 0.25

        if self.disp:
            print()
            print("Predicted reduction:", predicted_reduction[0][0])
            print("Actual reduction:", actual_reduction)
            print("Trust radius:", self.trust_radius)

    def optimize(self, plot=False):
        """
        Actually perform the trust-region optimization.

        Parameters
        ----------
        plot : boolean
            If True, plot a 2d representation of the optimization process.

        """
        self.construct_approximations()
        self.process_constraints()

        if plot:
            self.plot_functions()

        for i in range(30):
            self.process_constraints()
            x_new, hits_boundary = self.find_next_point()

            self.update_trust_region(x_new, hits_boundary)

            self.construct_approximations()

            if plot:
                self.plot_functions()

            x_test = self.design_vectors[-1, :]

            if self.trust_radius <= 1e-6:
                break

        results = {}
        results["optimal_design"] = self.design_vectors[-1, :]
        results["high_fidelity_func_value"] = self.model_high.run(
            self.design_vectors[-1, :]
        )[self.objective]
        results["number_high_fidelity_calls"] = len(self.design_vectors[:, 0])

        if self.disp:
            print()
            print(results)

        return results

    def plot_functions(self):
        """
        Plot a 2D contour plot of the design space and optimizer progress.

        Saves figures to .png files.

        """

        if self.n_dims == 2:
            n_plot = 9
            x_plot = np.linspace(self.bounds[0, 0], self.bounds[0, 1], n_plot)
            y_plot = np.linspace(self.bounds[1, 0], self.bounds[1, 1], n_plot)
            X, Y = np.meshgrid(x_plot, y_plot)
            x_values = np.vstack((X.flatten(), Y.flatten())).T

            y_plot_high = self.model_high.run_vec(x_values)[self.objective].reshape(
                n_plot, n_plot
            )

            # surrogate = []
            # for x_value in x_values:
            #     surrogate.append(np.squeeze(self.approximation_functions['con'](x_value)))
            # surrogate = np.array(surrogate)
            # y_plot_high = surrogate.reshape(n_plot, n_plot)

            fig = plt.figure(figsize=(7.05, 5))
            contour = plt.contourf(X, Y, y_plot_high, levels=201)
            plt.scatter(
                self.design_vectors[:, 0], self.design_vectors[:, 1], color="white"
            )
            ax = plt.gca()
            ax.set_aspect("equal", "box")

            cbar = fig.colorbar(contour)
            cbar.ax.set_ylabel("CP")
            ticks = np.round(np.linspace(0.305, 0.48286, 6), 3)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticks)

            x = self.design_vectors[-1, 0]
            y = self.design_vectors[-1, 1]
            points = np.array(
                [
                    [x + self.trust_radius, y + self.trust_radius],
                    [x + self.trust_radius, y - self.trust_radius],
                    [x - self.trust_radius, y - self.trust_radius],
                    [x - self.trust_radius, y + self.trust_radius],
                    [x + self.trust_radius, y + self.trust_radius],
                ]
            )
            plt.plot(points[:, 0], points[:, 1], "w--")

            plt.xlim(self.bounds[0])
            plt.ylim(self.bounds[1])

            plt.xlabel("Chord DV #1")
            plt.ylabel("Chord DV #2")

            plt.tight_layout()

            num_iter = self.design_vectors.shape[0]
            num_offset = 10

            if num_iter <= 5:
                for i in range(num_offset):
                    plt.savefig(
                        f"image_{self.counter_plot}.png", dpi=300, bbox_inches="tight"
                    )
                    self.counter_plot += 1
            else:
                plt.savefig(
                    f"image_{self.counter_plot}.png", dpi=300, bbox_inches="tight"
                )
                self.counter_plot += 1

        else:
            import niceplots

            x_full = np.atleast_2d(np.linspace(0.0, 1.0, 101)).T
            squeezed_x = np.squeeze(x_full)
            y_full = squeezed_x.copy()
            for i, x_val in enumerate(squeezed_x):
                y_full[i] = self.approximation_functions[self.objective](
                    np.atleast_2d(x_val.T)
                )

            y_full_high = self.model_high.run_vec(x_full)[self.objective]
            y_full_low = self.model_low.run_vec(x_full)[self.objective]

            y_low = self.model_low.run_vec(self.design_vectors)[self.objective]
            y_high = self.model_high.run_vec(self.design_vectors)[self.objective]

            plt.figure()

            plt.plot(squeezed_x, y_full_low, label="low-fidelity", c="tab:green")
            plt.scatter(self.design_vectors, y_low, c="tab:green")

            plt.plot(squeezed_x, y_full_high, label="high-fidelity", c="tab:orange")
            plt.scatter(self.design_vectors, y_high, c="tab:orange")

            plt.plot(squeezed_x, np.squeeze(y_full), label="surrogate", c="tab:blue")

            x = self.design_vectors[-1, 0]
            y_plot = y_high[-1]
            y_diff = 0.5
            x_lb = max(x - self.trust_radius, self.bounds[0, 0])
            x_ub = min(x + self.trust_radius, self.bounds[0, 1])
            points = np.array(
                [
                    [x_lb, y_plot],
                    [x_ub, y_plot],
                ]
            )
            plt.plot(points[:, 0], points[:, 1], "-", color="gray", clip_on=False)

            points = np.array(
                [
                    [x_lb, y_plot - y_diff],
                    [x_lb, y_plot + y_diff],
                ]
            )
            plt.plot(points[:, 0], points[:, 1], "-", color="gray", clip_on=False)

            points = np.array(
                [
                    [x_ub, y_plot - y_diff],
                    [x_ub, y_plot + y_diff],
                ]
            )
            plt.plot(points[:, 0], points[:, 1], "-", color="gray", clip_on=False)

            plt.xlim(self.bounds[0])
            plt.ylim([-10, 10])

            plt.xlabel("x")
            plt.ylabel("y")

            ax = plt.gca()
            ax.text(s="Low-fidelity", x=0.1, y=0.5, c="tab:green", fontsize=12)
            ax.text(s="High-fidelity", x=0.26, y=-8.5, c="tab:orange", fontsize=12)
            ax.text(
                s="Augmented low-fidelity", x=0.6, y=-10.0, c="tab:blue", fontsize=12
            )

            niceplots.adjust_spines(outward=True)

            plt.tight_layout()

            plt.savefig(f"1d_{self.counter_plot}.png", dpi=300)
            self.counter_plot += 1
