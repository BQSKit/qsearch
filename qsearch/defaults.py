"""
This module provides defaults for Options objects.  This includes definitions of smart_default functions, and dictionaries to be used with set_defaults and set_smart_defaults.

Three default dictionaries are provided.

Attributes:
    standard_defaults : A dictionary containing defaults for standard gate synthesis.
    standard_smart_defaults : A dictionary containing smart_defaults functions for standard gate synthesis.
    stateprep_defaults : A dictionary containing defaults for stateprep synthesis.
"""

from . import utils, gatesets, solvers, backends, parallelizers, heuristics, logging, checkpoints, assemblers, comparison
from functools import partial
import numpy as np



def default_eval_func(options):
    if options.error_func == comparison.matrix_residuals:
        return comparison.matrix_distance_squared
    else:
        return options.error_func

def default_heuristic(options):
    if options.search_type == "astar":
        return heuristics.astar
    if options.search_type == "djikstra":
        return heuristics.djikstra
    elif options.search_type == "greedy":
        return heuristics.greedy
    raise KeyError("Unknown search_type {}, and no alternative heuristic provided.".format(options.search_type))

def default_error_jac(options):
    if options.error_func == comparison.matrix_distance_squared:
        return comparison.matrix_distance_squared_jac
    else:
        return None

def default_error_residuals_jac(options):
    if options.error_residuals == comparison.matrix_residuals:
        return comparison.matrix_residuals_jac
    else:
        return None

def default_logger(options):
    return logging.Logger(verbosity=options.verbosity, stdout_enabled=options.stdout_enabled, output_file=options.log_file)

def default_checkpoint(options):
    return checkpoints.FileCheckpoint(options=options)

def default_gen_fullsize_I(options):
    return np.eye(options.target.shape[0], dtype='complex128')

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
        "error_func":comparison.matrix_distance_squared,
        "error_residuals":comparison.matrix_residuals,
        "backend":backends.SmartDefaultBackend(),
        "parallelizer":parallelizers.MultiprocessingParallelizer,
        "log_file":None,
        "max_quality_optimization" : False,
        "assembler" : assemblers.ASSEMBLER_QISKIT,
        "write_location" : None,
        "unitary_preprocessor": utils.nearest_unitary,
        "timeout" : float('inf'),
        }
standard_smart_defaults = {
        "eval_func":default_eval_func,
        "error_jac":default_error_jac,
        "error_residuals_jac":default_error_residuals_jac,
        "solver":solvers.default_solver,
        "heuristic":default_heuristic,
        "logger" :default_logger,
        "checkpoint":default_checkpoint,
        "fullsize_I":default_gen_fullsize_I,
        }

stateprep_defaults = {
        "error_residuals" : partial(comparison.matrix_residuals_slice, (0, slice(None))),
        "error_residuals_jac" : partial(comparison.matrix_residuals_slice_jac, (0, slice(None))),
        "eval_func" : partial(comparison.eval_func_from_residuals, partial(comparison.matrix_residuals_slice, (0, slice(None)))),
        "unitary_preprocessor": identity
        }

