"""
This module provides defaults for Options objects.  This includes definitions of smart_default functions, and dictionaries to be used with set_defaults and set_smart_defaults.

Three default dictionaries are provided.

Attributes:
    standard_defaults : A dictionary containing defaults for standard gate synthesis.
    standard_smart_defaults : A dictionary containing smart_defaults functions for standard gate synthesis.
    stateprep_defaults : A dictionary containing defaults for stateprep synthesis.
"""

from . import utils, gatesets, solvers, backends, parallelizers, heuristics, logging, checkpoints, assemblers, comparison, objectives, compiler
from functools import partial
import numpy as np

def default_heuristic(options):
    if options.search_type == "astar":
        return heuristics.astar
    if options.search_type == "djikstra":
        return heuristics.djikstra
    elif options.search_type == "greedy":
        return heuristics.greedy
    raise KeyError("Unknown search_type {}, and no alternative heuristic provided.".format(options.search_type))

def default_logger(options):
    return logging.Logger(verbosity=options.verbosity, stdout_enabled=options.stdout_enabled, output_file=options.log_file)

def default_checkpoint(options):
    return checkpoints.FileCheckpoint(options=options)

def default_eval_func(options):
    if isinstance(options.objective, objectives.BackwardsCompatibleObjective):
        return options.error_func
    elif isinstance(options.objective, objectives.MatrixDistanceObjective):
        return comparison.matrix_distance_squared
    raise AttributeError("Could not find option 'eval_func'.  This option can be used in some cases for backwards compatability, but generally 'objective' should be used instead.")


def default_error_func(options):
    if isinstance(options.objective, objectives.MatrixDistanceObjective) or isinstance(options.objective, objectives.BackwardsCompatibleObjective):
        return comparison.matrix_distance_squared
    raise AttributeError("Could not find option 'error_func'.  This option can be used in some cases for backwards compatability, but generally 'objective' should be used instead.")

def default_error_residuals(options):
    if isinstance(options.objective, objectives.MatrixDistanceObjective) or isinstance(options.objective, objectives.BackwardsCompatibleObjective):
        return comparison.matrix_residuals
    raise AttributeError("No 'error_func' was found.  This option can be used in some cases for backwards compatability, but generally 'objective' should be used instead.")

def default_error_jac(options):
    if isinstance(options.objective, objectives.BackwardsCompatibleObjective):
        if options.error_func is comparison.matrix_distance_squared:
            return comparison.matrix_distance_squared_jac
        else:
            return None
    elif isinstance(options.objective, objectives.MatrixDistanceObjective):
        return comparison.matrix_distance_squared_jac
    raise AttributeError("Could not find option 'error_jac'.  This option can be used in some cases for backwards compatability, but generally 'objective' should be used instead.")

def default_error_residuals_jac(options):
    if isinstance(options.objective, objectives.BackwardsCompatibleObjective):
        if options.error_residuals is comparison.matrix_residuals:
            return comparison.matrix_residuals_jac
        else:
            return None
    elif isinstance(options.objective, objectives.MatrixDistanceObjective):
        return comparison.matrix_residuals_jac
    raise AttributeError("Could not find option 'error_residuals_jac'.  This option can be used in some cases for backwards compatability, but generally 'objective' should be used instead.")

def default_objective(options):
    if options.manually_entered("eval_func", "error_func", "error_jac", "error_residuals", "error_residuals_jac", operator="any"):
        return objectives.BackwardsCompatibleObjective()
    else:
        return objectives.MatrixDistanceObjective()

def stateprep_initial_state(options):
    v = np.zeros(options.target_state.shape,dtype='complex128')
    v[0] = 1
    return v

def stateprep_target(options):
    return np.eye(options.target_state.shape[0], dtype='complex128')

def default_compiler(options):
    return compiler.SearchCompiler # this gets around some pesky import loops

def identity(U):
    return U

standard_defaults = {
        "threshold":1e-10,
        "gateset":gatesets.Default(),
        "beams":-1,
        "delta": 0,
        "weight_limit":None,
        "search_type":"astar",
        "statefile":None,
        "objective":objectives.MatrixDistanceObjective(),
        "backend":backends.SmartDefaultBackend(),
        "parallelizer":parallelizers.MultiprocessingParallelizer,
        "log_file":None,
        "max_quality_optimization" : False,
        "assembler" : assemblers.ASSEMBLER_QISKIT,
        "write_location" : None,
        "unitary_preprocessor": utils.nearest_unitary,
        "timeout" : float('inf'),
        "blas_threads" : None,
        "verbosity" : 1,
        "stdout_enabled" : True,
        }
standard_smart_defaults = {
        "solver":solvers.default_solver,
        "heuristic":default_heuristic,
        "logger" :default_logger,
        "checkpoint":default_checkpoint,
        "compiler_class" : default_compiler,
        "objective" : default_objective,
        "eval_func" : default_eval_func,
        "error_func" : default_error_func,
        "error_jac" : default_error_jac,
        "error_residuals" : default_error_residuals,
        "error_residuals_jac" : default_error_residuals_jac,
        }

stateprep_defaults = {
        "objective" : objectives.StateprepObjective(),
}
stateprep_smart_defaults = {
        "target" : stateprep_target,
        "initial_state" : stateprep_initial_state,
}
