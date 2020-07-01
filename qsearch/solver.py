import sys

import numpy as np
import scipy as sp
import scipy.optimize

from . import circuits
from . import utils
from .gatesets import *
from .logging import Logger


def default_solver(options):
    options = options.copy()
    # re-route the default behavior for error_func and error_jac because the default functions for those parameters often rely on the return valye from default_solver
    options.set_defaults(logger=Logger(), U=np.array([]), error_func=None, error_jac=None, target=None)
    options.remove_smart_defaults("error_func", "error_jac")

    # Choosse the best default solver for the given gateset
    ls_failed = False

    # Check to see if the gateset and error func are explicitly supported by LeastSquares
    gateset = options.gateset
    error_func = options.error_func
    error_jac = options.error_jac
    logger = options.logger
    dits = 0 if options.target is None else int(np.log(options.target.shape[0]) // np.log(gateset.d))

    if type(gateset).__module__ != QubitCNOTLinear.__module__:
        ls_failed = True
    elif error_func is not None and (error_func.__module__ != utils.matrix_distance_squared.__module__ or (error_func.__name__ != utils.matrix_distance_squared.__name__ and error_func.__name__ != utils.matrix_residuals.__name__)):
        ls_failed = True

    if not ls_failed:
        # since all provided gatesets support jacobians, this is the only check we need
        logger.logprint("Smart default chose LeastSquares_Jac_Solver", verbosity=2)
        return LeastSquares_Jac_Solver()

    if dits < 1:
        logger.logprint("Smart default fell back to COBYLA_Solver.  Pass a different Solver to SearchCompiler for better results.", verbosity=1)
        return COBYLA_Solver() # handling this case for manually created SearchCompiler instances.  Better support for manual usage is unlikely to be implemented because Projects are generally recommended.

    # least squares won't work, so check for jacobian and rust success
    jac_failed = False
    layers = [(gateset.initial_layer(dits), 0)] + gateset.search_layers(dits)
    for layer, _ in layers:
        try:
            layer.mat_jac(np.random.rand(layer.num_inputs))
        except:
            jac_failed = True

    if error_jac is None and error_func not in [None, utils.matrix_distance_squared, utils.matrix_residuals]:
        jac_failed = True

    if jac_failed:
        logger.logprint("Smart default chose COBYLA_Solver", verbosity=2)
        return COBYLA_Solver()
    else:
        logger.logprint("Smart default chose BFGS_Jac_Solver", verbosity=2)
        return BFGS_Jac_Solver()
    # the default will have been chosen from LeastSquares, BFGS, or COBYLA

class Solver():
    def solve_for_unitary(self, circuit, options):
        raise NotImplementedError

    def __eq__(self, other):
        if self is other:
            return True
        if self.__module__ == Solver.__module__:
            if type(self) == type(other):
                return True
        return False

class CMA_Solver(Solver):
    def solve_for_unitary(self, circuit, options):
        try:
            import cma
        except ImportError:
            print("ERROR: Could not find cma, try running pip install quantum_synthesis[cma]", file=sys.stderr)
            sys.exit(1)
        eval_func = lambda v: options.error_func(options.target, circuit.matrix(v))
        initial_guess = 'np.random.rand({})'.format(circuit.num_inputs)
        xopt, _ = cma.fmin2(eval_func, initial_guess, 0.25, {'verb_disp':0, 'verb_log':0, 'bounds' : [0,1]}, restarts=2)
        return (circuit.matrix(xopt), xopt)

class COBYLA_Solver(Solver):
    def solve_for_unitary(self, circuit, options):
        eval_func = lambda v: options.error_func(options.target, circuit.matrix(v))
        initial_guess = np.array(np.random.rand(circuit.num_inputs))
        x = sp.optimize.fmin_cobyla(eval_func, initial_guess, cons=[lambda x: np.all(np.less_equal(x,1))], rhobeg=0.5, rhoend=1e-12, maxfun=1000*circuit.num_inputs)
        return (circuit.matrix(x), x)

class DIY_Solver(Solver):
    def __init__(self, f):
        self.f = f # f is a function that takes in eval_func and initial_guess and returns the vector that minimizes eval_func.  The parameters may range between 0 and 1.

    def solve_for_unitary(self, circuit, options):
        eval_func = lambda v: options.error_func(options.target, circuit.matrix(v))
        initial_guess = np.array(np.random.rand(circuit.num_inputs))
        x = f(eval_func, initial_guess)

class NM_Solver(Solver):
    def solve_for_unitary(self, circuit, options):
        eval_func = lambda v: options.error_func(options.target, circuit.matrix(v))
        result = sp.optimize.minimize(eval_func, np.random.rand(circuit.num_inputs)*np.pi, method='Nelder-Mead', options={"ftol":1e-14})
        xopt = result.x
        return (circuit.matrix(xopt), xopt)

class CMA_Jac_Solver(Solver):
    def solve_for_unitary(self, circuit, options):
        try:
            import cma
        except ImportError:
            print("ERROR: Could not find cma, try running pip install quantum_synthesis[cma]", file=sys.stderr)
            sys.exit(1)
        eval_func = lambda v: options.error_func(options.target, circuit.matrix(v))
        jac_func  = lambda v: options.error_jac(options.target, circuit.mat_jac(v))
        initial_guess = 'np.random.rand({})'.format(circuit.num_inputs)
        xopt, es = cma.fmin2(eval_func, initial_guess, 0.25, {'verb_disp':0, 'verb_log':0, 'bounds' : [0,1]}, restarts=2, gradf=jac_func)
        if circuit.num_inputs > 18:
            raise Warning("Finished with {} evaluations".format(es.result[3]))
        return (circuit.matrix(xopt), xopt)

class BFGS_Jac_Solver(Solver):
    def solve_for_unitary(self, circuit, options):
        def eval_func(v):
            M, jacs = circuit.mat_jac(v)
            return options.error_jac(options.target, M, jacs)
        result = sp.optimize.minimize(eval_func, np.random.rand(circuit.num_inputs), method='BFGS', jac=True)
        xopt = result.x
        return (circuit.matrix(xopt), xopt)

class LeastSquares_Jac_Solver(Solver):
    def solve_for_unitary(self, circuit, options):
        # This solver is usually faster than BFGS, but has some caveats
        # 1. This solver relies on matrix residuals, and therefore ignores the specified error_func, making it currently not suitable for alternative synthesis goals like stateprep
        # 2. This solver (currently) does not correct for an overall phase, and so may not be able to find a solution for some gates with some gatesets.  It has been tested and works fine with QubitCNOTLinear, so any single-qubit and CNOT-based gateset is likely to work fine.
        I = np.eye(options.target.shape[0])
        eval_func = lambda v: options.error_residuals(options.target, circuit.matrix(v), I)
        jac_func = lambda v: options.error_residuals_jac(options.target, *circuit.mat_jac(v))
        result = sp.optimize.least_squares(eval_func, np.random.rand(circuit.num_inputs), jac_func, method="lm")
        xopt = result.x
        return (circuit.matrix(xopt), xopt)

