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

# Error funcs based on matrix comparison
def err_distsq(circuit, parameters, target, options):
    return comparison.matrix_distance_squared(target, circuit.matrix(parameters))

def err_distq_jac(circuit, parameters, target, jacs, options):
    return comparison.matrix_distance_squared_jac(target, circuit.matrix(parameters), jacs)


# Error residual funcs based on matrix comparison
def res_product(circuit, parameters, target, options):
    return comparison.matrix_residuals(target, circuit.matrix(parameters), options.fullsize_I)

def res_product_jac(circuit, parameters, target, jacs, options):
    return comparison.matrix_distance_squared_jac(target, ciruit.matrix(parameters), jacs)

def res_difference(circuit, parameters, target, options):
    return comparison.matrix_residuals_v2(target, circuit.matrix(parameters), options.fullsize_I)

def res_difference_jac(circuit, parameters, target, jacs, options):
    return comparison.matrix_residuals_v2_jac(target, circuit.matrix(parameters), jacs)



