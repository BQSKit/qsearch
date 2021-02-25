"""
This module contains functions for comparing and otherwise evaluating matrices, including distance functions, cost functions, and constraint functions.

The standarized format for these types of functions is as follows:

    def my_func(circuit, parameters, target, options):
        return <one or more real-valued numbers>

    def my_func_jac(circuit, parameters, target, jacs, options):
        return <one or more real-valued numbers>
"""

from . import utils
from . import comparison


# Error funcs based on matrix comparison.  These can also be used as cost funcs.

def error_distsq(circuit, parameters, target, options):
    return comparison.matrix_distance_squared(target, circuit.matrix(parameters))

def error_distsq_jac(circuit, parameters, target, jacs, options):
    return comparison.matrix_distance_squared_jac(target, circuit.matrix(parameters), jacs)


# Error funcs for stateprep

def error_stateprep_distsq(circuit, parameters, target, options):
    return 1-np.real(np.vdot(np.dot(target, options.initial_state), np.dot(circuit.matrix(parameters), options.initial_state)))

def error_stateprep_distsq_jac(circuit, parameters, target, options):
    return None # honestly I don't think this will be too complicated, but I do have to think about it


# Error residual funcs based on matrix comparison

def residuals_product(circuit, parameters, target, options):
    return comparison.matrix_residuals(target, circuit.matrix(parameters), options.fullsize_I)

def residuals_product_jac(circuit, parameters, target, options):
    return comparison.matrix_residuals_jac(target, *circuit.mat_jac(parameters))

def residuals_difference(circuit, parameters, target, options):
    return comparison.matrix_residuals_v2(target, circuit.matrix(parameters), options.fullsize_I)

def residuals_difference_jac(circuit, parameters, target, options):
    return comparison.matrix_residuals_v2_jac(target, *circuit.mat_jac(parameters))

def residuals_slice(circuit, parameters, target, options):
    return comparison.matrix_residuals_slice(options.slices, target, circuit.matrix(parameters), options.fullsize_I)

def residuals_slice_jac(circuit, parameters, target, options):
    return comparison.matrix_residuals_slice_jac(options.slices, target, circuit.mat_jac(parameters))

def residuals_blacklist(circuit, parameters, target, options):
    return comparison.matrix_residuals_blacklist(options.badrows, options.badcols, target, circuit.matrix(parameters), options.fullsize_)

def residuals_blacklist_jac(circuit, parameters, target, options):
    return comparison.matrix_residuals_blacklist_jac(options.badrows, options.badcols, target, circuit.matrix(parameters), jacs)


# cost funcs based on gate parameters

def cost_linear(circuit, parameters, target, options):
    return np.sum(np.abs(np.mod(parameters-0.25*np.pi, 0.5*np.pi) - 0.25*np.pi))

def cost_linear_jac(circuit, parameters, target, options):
    return -np.sign(np.mod(parameters, 0.5*np.pi)-0.25*np.pi)


# constraint funcs based on matrix comparison

def constraint_distsq(circuit, parameters, target, options):
    return options.constraint_threshold - comparison.matrix_distance_squared(circ.matrix(parameters), target)

def constraint_distsq_jac(circuit, parameters, target, jacs, options):
    return -comparison.matrix_distance_squared_jac(target, circuit.matrix(parameters), jacs)


# cost funcs based on combining two cost funcs

def cost_combo_linear(circuit, parameters, target, options):
    return options.cost1(circuit, parameters, target, options) + options.cost_coefficient * options.cost2(circuit, parameters, target, options)

def cost_combo_linear_jac(circuit, parameters, target, jacs, options):
    return options.cost1_jac(circuit, parameters, target, jacs, options) + options.cost_coefficient * options.cost2_jac(circuit, parameters, target, jacs, options)


# more cost combo functions will return in Thor: Love and Thunder
