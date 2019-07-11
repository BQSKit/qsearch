import sys

import numpy as np

from . import circuits as circuits
from . import utils as util


def default_solver():
    try:
        import cma
    except ImportError:
        pass
    else:
        return CMA_Solver()
    try:
        import scipy
    except ImportError:
        pass
    else:
        return COBYLA_Solver()
    print(
        "ERROR: Could not find a solver library, please install one via pip install quantum_synthesis[<solver>] where <solver> is one of cma, cobyla, or bfgs.", file=sys.stderr)
    sys.exit(1)

class CMA_Solver():
    def solve_for_unitary(self, circuit, U, error_func=util.matrix_distance_squared, initial_guess=None):
        try:
            import cma
        except ImportError:
            print("ERROR: Could not find cma, try running pip install quantum_synthesis[cma]", file=sys.stderr)
            sys.exit(1)
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        if initial_guess is None:
            initial_guess = 'np.random.rand({})'.format(circuit.num_inputs)
        else:
            print("WARNING: Experimental inital guess configuration active")
            initial_guess = 'np.concatenate((np.array({}), np.array(np.random.rand({}))))'.format(repr(initial_guess), circuit.num_inputs - len(initial_guess))
        xopt, _ = cma.fmin2(eval_func, initial_guess, 0.25, {'verb_disp':0, 'verb_log':0, 'bounds' : [0,1]}, restarts=2)
        return (circuit.matrix(xopt), xopt)

class BFGS_Solver():
    def solve_for_unitary(self, circuit, U, error_func=util.matrix_distance_squared, initial_guess=None):
        try:
            import scipy as sp
            import scipy.optimize
        except ImportError:
            print("ERROR: Could not find scipy, try running pip install quantum_synthesis[bfgs]", file=sys.stderr)
            sys.exit(1)
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        result = sp.optimize.minimize(eval_func, np.random.rand(circuit.num_inputs)*np.pi, method='BFGS', bounds=[(0, 1) for _ in range(0, circuit.num_inputs)])
        xopt = result.x
        return (circuit.matrix(xopt), xopt)

class COBYLA_Solver():
    def solve_for_unitary(self, circuit, U, error_func=util.matrix_distance_squared, initial_guess=None):
        try:
            import scipy as sp
            import scipy.optimize
        except ImportError:
            print("ERROR: Could not find scipy, try running pip install quantum_synthesis[cobyla]", file=sys.stderr)
            sys.exit(1)
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        if initial_guess is None:
            initial_guess = []
        else:
            print("WARNING: Experimental inital guess configuration active")
        initial_guess = np.concatenate((np.array(initial_guess), np.array(np.random.rand(circuit.num_inputs - len(initial_guess)))))
        x = sp.optimize.fmin_cobyla(eval_func, initial_guess, cons=[lambda x: np.all(np.less_equal(x,1))], rhobeg=0.5, rhoend=1e-12, maxfun=1000*circuit.num_inputs)
        return (circuit.matrix(x), x)

class DIY_Solver():
    def __init__(self, f):
        self.f = f # f is a function that takes in eval_func and initial_guess and returns the vector that minimizes eval_func.  The parameters may range between 0 and 1.

    def solve_for_unitary(self, circuit, U, error_func=util.matrix_distance_squared, initial_guess=None):
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        if initial_guess is None:
            initial_guess = []
        else:
            print("WARNING: Experimental inital guess configuration active")
        initial_guess = np.concatenate((np.array(initial_guess), np.array(np.random.rand(circuit.num_inputs - len(initial_guess)))))
        x = f(eval_func, initial_guess)

