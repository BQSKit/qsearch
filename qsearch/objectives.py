import numpy as np
from . import utils, comparison


class Objective:
    # a minimal implementation of Objective involves implementing gen_error_func
    # implementing the gen_error_jac is required to take advantage of gradient-based optimizers
    # To use least-squares based optimizers, instead implement gen_error_residuals and gen_error_residuals_jac.  You can optionally implement gen_error_func and gen_error_jac as well
    # You can optionally customize gen_eval_func separately from the other functions
    def gen_eval_func(self, circuit, options):
        return self.gen_error_func(circuit, options)

    def gen_error_func(self, circuit, options):
        generated_error_residuals = self.gen_error_residuals(circuit, options)
        if generated_error_residuals is None:
            return None
        def generated_error_func(parameters):
            return np.sum(np.square(created_error_residuals(parameters)))
        return generated_error_func

    def gen_error_jac(self, circuit, options):
        generated_error_residuals_jac = self.gen_error_residuals_jac(circuit, options)
        if generated_error_residuals_jac is None:
            return None
        def generated_error_jac(parameters):
            return 2*np.sum(generated_error_residuals_jac(parameters, axis=1))

    def gen_error_residuals(self, circuit, options):
        return None

    def gen_error_residuals_jac(self, circuit, options):
        return None


class MatrixDistanceObjective(Objective):
    def gen_error_func(self, circuit, options):
        target = options.target
        def generated_error_func(parameters):
            return comparison.matrix_distance_squared(target, circuit.matrix(parameters))
        return generated_error_func

    def gen_error_jac(self, circuit, options):
        target = options.target
        def generated_error_jac(parameters):
            return comparison.matrix_distance_squared_jac(target, *circuit.mat_jac(parameters))
        return generated_error_jac

    def gen_error_residuals(self, circuit, options):
        target = options.target
        I = np.eye(options.target.shape[0])
        def generated_error_residuals(parameters):
            return comparison.matrix_residuals(target, circuit.matrix(parameters), I)
        return generated_error_residuals

    def gen_error_residuals_jac(self, circuit, options):
        target = options.target
        def generated_error_residuals_jac(parameters):
            return comparison.matrix_residuals_jac(target, *circuit.mat_jac(parameters))
        return generated_error_residuals_jac

class StateprepObjective(Objective):
    def gen_error_func(self, circuit, options):
        target = options.target
        initial_state = options.initial_state
        target_state = options.target_state
        def generated_error_func(parameters):
            return comparison.distance_with_initial_state(target_state, initial_state, target, circuit.matrix(parameters)) 

        return generated_error_func

    def gen_error_jac(self, circuit, options):
        target = options.target
        initial_state = options.initial_state
        target_state = options.target_state
        def generated_error_jac(parameters):
            return comparison.distance_with_initial_state_jac(target_state, initial_state, target, *circuit.mat_jac(parameters))

        return generated_error_jac

    def gen_error_residuals(self, circuit, options):
        target = options.target
        initial_state = options.initial_state
        target_state = options.target_state
        I = np.eye(target.shape[0], dtype='float64')
        return None
        def generated_error_residuals(parameters):
            return comparison.residuals_with_initial_state(target_state, initial_state, target, circuit.matrix(parameters), I)

        return generated_error_residuals

    def gen_error_residuals_jac(self, circuit, options):
        target = options.target
        initial_state = options.initial_state
        target_state = options.target_state
        return None
        def generated_error_residuals_jac(parameters):
            return comparison.residuals_with_initial_state_jac(target_state, initial_state, target, *circuit.mat_jac(parameters))

        return generated_error_residuals_jac

class BackwardsCompatibleObjective(Objective):
    def gen_error_func(self, circuit, options):
        target = options.target
        eval_func = options.eval_func
        def generated_eval_func(parameters):
            return eval_func(target, circuit.matrix(parameters))

        return generated_eval_func

    def gen_error_func(self, circuit, options):
        target = options.target
        error_func = options.error_func
        def generated_error_func(parameters):
            return error_func(target, circuit.matrix(parameters))

        return generated_error_func

    def gen_error_jac(self, circuit, options):
        target = options.target
        if not "error_jac" in options:
            return None
        error_jac = options.error_jac
        def generated_error_jac(parameters):
            return error_jac(target, *circuit.mat_jac(parameters))

        return generated_error_jac

    def gen_error_residuals(self, circuit, options):
        target = options.target
        if not "error_residuals" in options:
            return None
        error_residuals = options.error_residuals
        I = np.eye(target.shape[0], dtype='float64')
        def generated_error_residuals(parameters):
            return error_residuals(target, circuit.matrix(parameters), I)

        return generated_error_residuals

    def gen_error_residuals_jac(self, circuit, options):
        target = options.target
        if not "error_residuals_jac" in options:
            return None
        error_residuals_jac = options.error_residuals_jac
        def generated_error_residuals_jac(parameters):
            return error_residuals_jac(target, *circuit.mat_jac(parameters))

        return generated_error_residuals_jac
