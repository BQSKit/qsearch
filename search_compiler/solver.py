import sys

import numpy as np
import scipy as sp
import scipy.optimize

from . import circuits
from . import utils
from .gatesets import *

try:
    from search_compiler_rs import native_from_object
    RUST_ENABLED = True
except ImportError:
    RUST_ENABLED = False
    def native_from_object(o):
        raise Exception("Native code not installed")

def default_solver(gateset, dits=0, error_func=None):
    # Choosse the best default solver for the given gateset
    ls_failed = False

    # Check to see if the gateset and error func are explicitly supported by LeastSquares
    if type(gateset).__module__ != QubitCNOTLinear.__module__:
        ls_failed = True
    elif type(gateset).__name__ not in [QubitCNOTLinear.__name__, QiskitU3Linear,__name__, QubitCNOTRing.__name__, QubitCNOTAdjacencyList.__name__, QubitCRZLinear.__name__, QubitCRZRing.__name__, ZXZXZCNOTLinear.__name__]:
        ls_failed = True
    elif error_func is None or error_func.__module__ != utils.matrix_distance_squared.__module__ or error_func.__name__ != utils.matrix_distance_squared.__name__:
        ls_failed = True

    if not ls_failed:
        # since all gatesets supported by LeastSquares are supported by rust, this is the only check we need
        if RUST_ENABLED:
            return LeastSquares_Jac_SolverNative()
        else:
            return LeastSquares_Jac_Solver()

    if dits < 1:
        return COBYLA_Solver() # handling this case for manually created SearchCompiler instances.  Better support for manual usage is unlikely to be implemented because Projects are generally recommended.

    # least squares won't work, so check for jacobian and rust success
    jac_failed = False
    rust_failed = False
    layers = [gateset.initial_layer(dits)] + gateset.search_layers(dits)
    for layer in layers:
        try:
            layer.mat_jac(np.random.rand(layer.num_inputs))
        except:
            jac_failed = True
        try:
            native_from_object(layer)
        except:
            rust_failed = True

    if jac_failed:
        if rust_failed:
            return COBYLA_Solver()
        else:
            return COBYLA_SolverNative()
    else:
        if rust_failed:
            return BFGS_Jac_Solver()
        else:
            return BFGS_Jac_SolverNative()
    # the default will have been chosen from LeastSquares, BFGS, or COBYLA, from either the python or "Native" rust variants


class Solver():
    def solve_for_unitary(self, circuit, U, error_func=utils.matrix_distance_squared):
        raise NotImplementedError

    def __eq__(self, other):
        if self is other:
            return True
        if self.__module__ == Solver.__module__:
            if type(self) == type(other):
                return True
        return False


class CMA_Solver(Solver):
    def solve_for_unitary(self, circuit, U, error_func=utils.matrix_distance_squared):
        try:
            import cma
        except ImportError:
            print("ERROR: Could not find cma, try running pip install quantum_synthesis[cma]", file=sys.stderr)
            sys.exit(1)
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        initial_guess = 'np.random.rand({})'.format(circuit.num_inputs)
        xopt, _ = cma.fmin2(eval_func, initial_guess, 0.25, {'verb_disp':0, 'verb_log':0, 'bounds' : [0,1]}, restarts=2)
        return (circuit.matrix(xopt), xopt)

class COBYLA_Solver(Solver):
    def solve_for_unitary(self, circuit, U, error_func=utils.matrix_distance_squared):
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        initial_guess = np.array(np.random.rand(circuit.num_inputs))
        x = sp.optimize.fmin_cobyla(eval_func, initial_guess, cons=[lambda x: np.all(np.less_equal(x,1))], rhobeg=0.5, rhoend=1e-12, maxfun=1000*circuit.num_inputs)
        return (circuit.matrix(x), x)

class DIY_Solver(Solver):
    def __init__(self, f):
        self.f = f # f is a function that takes in eval_func and initial_guess and returns the vector that minimizes eval_func.  The parameters may range between 0 and 1.

    def solve_for_unitary(self, circuit, U, error_func=utils.matrix_distance_squared):
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        initial_guess = np.array(np.random.rand(circuit.num_inputs))
        x = f(eval_func, initial_guess)

class COBYLA_SolverNative(COBYLA_Solver):
    def solve_for_unitary(self, circuit, U, error_func=utils.matrix_distance_squared):
        return super().solve_for_unitary(native_from_object(circuit), U, error_func=error_func)

class NM_Solver(Solver):
    def solve_for_unitary(self, circuit, U, error_func=utils.matrix_distance_squared):
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        result = sp.optimize.minimize(eval_func, np.random.rand(circuit.num_inputs)*np.pi, method='Nelder-Mead', options={"ftol":1e-14})
        xopt = result.x
        return (circuit.matrix(xopt), xopt)

class BFGS_Jac_Solver(Solver):
    def solve_for_unitary(self, circuit, U, error_func = None):
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        error_func_jac = utils.matrix_distance_squared_jac
        def eval_func(v):
            M, jacs = circuit.mat_jac(v)
            return error_func_jac(U, M, jacs)
        result = sp.optimize.minimize(eval_func, np.random.rand(circuit.num_inputs)*np.pi, method='BFGS', jac=True)
        xopt = result.x
        return (circuit.matrix(xopt), xopt)

class CMA_Jac_Solver(Solver):
    def solve_for_unitary(self, circuit, U, error_func=utils.matrix_distance_squared):
        try:
            import cma
        except ImportError:
            print("ERROR: Could not find cma, try running pip install quantum_synthesis[cma]", file=sys.stderr)
            sys.exit(1)
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        jac_func  = lambda v: utils.matrix_distance_squared_jac(U, circuit.matrix(v), circuit.jac(v))
        initial_guess = 'np.random.rand({})'.format(circuit.num_inputs)
        xopt, es = cma.fmin2(eval_func, initial_guess, 0.25, {'verb_disp':0, 'verb_log':0, 'bounds' : [0,1]}, restarts=2, gradf=jac_func)
        if circuit.num_inputs > 18:
            raise Warning("Finished with {} evaluations".format(es.result[3]))
        return (circuit.matrix(xopt), xopt)

class BFGS_Jac_SolverNative(BFGS_Jac_Solver):
    def solve_for_unitary(self, circuit, U, error_func=utils.matrix_distance_squared):
        return super().solve_for_unitary(native_from_object(circuit), U, error_func=error_func)

class LeastSquares_Jac_Solver(Solver):
    def solve_for_unitary(self, circuit, U, error_func=None):
        # This solver is usually faster than BFGS, but has some caveats
        # 1. This solver relies on matrix residuals, and therefore ignores the specified error_func, making it currently not suitable for alternative synthesis goals like stateprep
        # 2. This solver (currently) does not correct for an overall phase, and so may not be able to find a solution for some gates with some gatesets.  It has been tested and works fine with QubitCNOTLinear, so any single-qubit and CNOT-based gateset is likely to work fine.
        I = np.eye(U.shape[0])
        resi_func = utils.matrix_residuals
        eval_func = lambda v: resi_func(U, circuit.matrix(v), I)
        jac_func = lambda v: utils.matrix_residuals_jac(U, *circuit.mat_jac(v))
        result = sp.optimize.least_squares(eval_func, np.random.rand(circuit.num_inputs)*np.pi, jac_func, method="lm")
        xopt = result.x
        return (circuit.matrix(xopt), xopt)

class LeastSquares_Jac_SolverNative(LeastSquares_Jac_Solver):
    def solve_for_unitary(self, circuit, U, error_func=None):
        return super().solve_for_unitary(native_from_object(circuit), U, error_func=error_func)
