"""
Defines Solver, a class used to wrap various numerical optimizers for finding parameters such that an ansatz circuit is a solution to a target unitary.
"""
import sys

import numpy as np
import scipy as sp
import scipy.optimize

from . import utils, objectives, comparison
from .gatesets import *
from .logging import Logger

try:
    from qsrs import LeastSquares_Jac_SolverNative, BFGS_Jac_SolverNative, native_from_object, matrix_residuals, matrix_residuals_jac
except ImportError:
    LeastSquares_Jac_SolverNative = BFGS_Jac_SolverNative = native_from_object = matrix_residuals= matrix_residuals_jac = None

def default_solver(options, x0=None):
    """Runs a complex list of tests to determine the best Solver for a specific situation."""
    objectives_opt = options.copy()
    options = options.copy()
    options.make_required("error_func", "error_residuals", "error_jac", "error_residuals_jac")

    # Choosse the best default solver for the given gateset
    ls_failed = False

    # check if Rust works on the layers
    gateset = options.gateset
    qudits = 0 if "target" not in options else int(np.log(options.target.shape[0]) // np.log(gateset.d))
    if "target" not in options:
        objectives_opt.target = np.eye(2, dtype='complex128')
    layers = [(gateset.initial_layer(qudits), 0)] + gateset.search_layers(qudits)
    circ = gateset.initial_layer(qudits)

    rs_failed = True
    if native_from_object is not None:
        for layer in layers:
            try:
                native_from_object(layer[0])
            except ValueError:
                break
        else:
            rs_failed = False

    # Check to see if the gateset and error func are explicitly supported by LeastSquares
    logger = options.logger

    if type(gateset).__module__ != QubitCNOTLinear.__module__:
        ls_failed = True

    if objectives_opt.objective.gen_error_residuals(circ, objectives_opt) is None:
        ls_failed = True

    if not ls_failed:
        # since all provided gatesets support jacobians, this is the only check we need
        if rs_failed or objectives_opt.error_residuals is not comparison.matrix_residuals: #could/should replace this test with one that calls gen_matrix_distance etc. and checks to see if the result is None
            logger.logprint("Smart default chose LeastSquares_Jac_Solver", verbosity=3)
            return LeastSquares_Jac_Solver()
        else:
            logger.logprint("Smart default chose LeastSquares_Jac_SolverNative", verbosity=3)
            return LeastSquares_Jac_SolverNative()

    if qudits < 1:
        logger.logprint("Smart default fell back to COBYLA_Solver.  Pass a different Solver to SearchCompiler for better results.", verbosity=1)
        return COBYLA_Solver() # handling this case for manually created SearchCompiler instances.  Better support for manual usage is unlikely to be implemented because Projects are generally recommended.

    # least squares won't work, so check for jacobian and rust success
    jac_failed = False
    for layer, _ in layers:
        try:
            layer.mat_jac(np.random.rand(layer.num_inputs))
        except:
            jac_failed = True
            break

    if objectives_opt.objective.gen_error_jac(circ, objectives_opt) is None: 
        jac_failed = True

    if jac_failed:
        logger.logprint("Smart default chose COBYLA_Solver", verbosity=2)
        return COBYLA_Solver()
    else:
        logger.logprint("Smart default chose BFGS_Jac_Solver", verbosity=2)
        return BFGS_Jac_Solver()
    # the default will have been chosen from LeastSquares, BFGS, or COBYLA

class Solver():
    """This class is used to wrap numerical optimizers for circuit solving."""
    def solve_for_unitary(self, circuit, options, x0=None):
        """Finds the best parameters that minimize error_func or error_residuals between the unitary from the circuit and options.target."""
        raise NotImplementedError

    def __eq__(self, other):
        if self is other:
            return True
        if self.__module__ == Solver.__module__:
            if type(self) == type(other):
                return True
        return False

    @property
    def distance_metric(self):
        return "Frobenius"

class CMA_Solver(Solver):
    """Uses cmaes gradient-free optimization from the cma package."""
    def solve_for_unitary(self, circuit, options, x0=None):
        try:
            import cma
        except ImportError:
            print("ERROR: Could not find cma, try running pip install quantum_synthesis[cma]", file=sys.stderr)
            sys.exit(1)
        error_func = options.gen_error_func(circuit, options)
        initial_guess = 'np.random.rand({})*2*np.pi'.format(circuit.num_inputs) if x0 is None else x0
        xopt, _ = cma.fmin2(error_func, initial_guess, 0.25, {'verb_disp':0, 'verb_log':0, 'bounds' : [0,2*np.pi]}, restarts=2)
        return (circuit.matrix(xopt), xopt)

class COBYLA_Solver(Solver):
    """Uses cobyla gradient-free optimization from scipy."""
    def solve_for_unitary(self, circuit, options, x0=None):
        error_func = options.objective.gen_error_func(circuit, options)
        initial_guess = np.array(np.random.rand(circuit.num_inputs))*2*np.pi if x0 is None else x0
        x = sp.optimize.fmin_cobyla(error_func, initial_guess, cons=[lambda x: np.all(np.less_equal(x,2*np.pi))], rhobeg=0.5, rhoend=1e-12, maxfun=1000*circuit.num_inputs)
        return (circuit.matrix(x), x)

class DIY_Solver(Solver):
    """An easier way to wrap a numerical optimizer than writing your own Solver class."""
    def __init__(self, f):
        """Uses the function f that takes in eval_func and initial_guess and returns the parameters that minimizes eval_func."""
        self.f = f

    def solve_for_unitary(self, circuit, options, x0=None):
        error_func = options.objective.gen_error_func(circuit, options)
        initial_guess = np.array(np.random.rand(circuit.num_inputs))*2*np.pi if x0 is None else x0
        x = f(error_func, initial_guess)

class BFGS_Jac_Solver(Solver):
    """A solver based on the BFGS implementation in scipy.  It requires gradients."""
    def solve_for_unitary(self, circuit, options, x0=None):
        error_jac = options.objective.gen_error_jac(circuit, options)
        result = sp.optimize.minimize(error_jac, np.random.rand(circuit.num_inputs)*2*np.pi if x0 is None else x0, method='BFGS', jac=True)
        xopt = result.x
        return (circuit.matrix(xopt), xopt)

class LeastSquares_Jac_Solver(Solver):
    """Uses the Leavenberg-Marquardt least-squares optimizer in scipy."""
    def solve_for_unitary(self, circuit, options, x0=None):
        # This solver is usually faster than BFGS, but has some caveats
        # 1. This solver relies on matrix residuals, and therefore ignores the specified error_func, making it currently not suitable for alternative synthesis goals like stateprep
        # 2. This solver (currently) does not correct for an overall phase, and so may not be able to find a solution for some gates with some gatesets.  It has been tested and works fine with QubitCNOTLinear, so any single-qubit and CNOT-based gateset is likely to work fine.
        error_residuals = options.objective.gen_error_residuals(circuit, options)
        error_residuals_jac = options.objective.gen_error_residuals_jac(circuit, options)
        if options.max_quality_optimization:
            result = sp.optimize.least_squares(error_residuals, np.random.rand(circuit.num_inputs)*2*np.pi if x0 is None else x0, error_residuals_jac, method="lm", ftol=5e-16, xtol=5e-16, gtol=1e-15)
        else:
            result = sp.optimize.least_squares(error_residuals, np.random.rand(circuit.num_inputs)*2*np.pi if x0 is None else x0, error_residuals_jac, method="lm")
        xopt = result.x
        return (circuit.matrix(xopt), xopt)

    @property
    def distance_metric(self):
        return "Residuals"

