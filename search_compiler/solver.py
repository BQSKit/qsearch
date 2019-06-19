import numpy as np
import cma

from . import circuits as circuits
from . import utils as util


class CMA_Solver():
    def solve_for_unitary(self, circuit, U, error_func=util.matrix_distance_squared, initial_guess=None):
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        if initial_guess is None:
            initial_guess = 'np.random.rand({})'.format(circuit._num_inputs)
        else:
            print("WARNING: Experimental inital guess configuration active")
            initial_guess = 'np.concatenate((np.array({}), np.array(np.random.rand({}))))'.format(repr(initial_guess), circuit._num_inputs - len(initial_guess))
        xopt, _ = cma.fmin2(eval_func, initial_guess, 0.25, {'verb_disp':0, 'verb_log':0, 'bounds' : [0,1]}, restarts=2)
        return (circuit.matrix(xopt), xopt)


import scipy as sp
import scipy.optimize

class BFGS_Solver():
    def solve_for_unitary(self, circuit, U, error_func=util.matrix_distance_squared, initial_guess=None):
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        result = sp.optimize.minimize(eval_func, np.random.rand(circuit._num_inputs)*np.pi, method='BFGS', bounds=[(0, 1) for _ in range(0, circuit._num_inputs)])
        xopt = result.x
        return (circuit.matrix(xopt), xopt)

class COBYLA_Solver():
    def solve_for_unitary(self, circuit, U, error_func=util.matrix_distance_squared, initial_guess=None):
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        if initial_guess is None:
            initial_guess = []
        else:
            print("WARNING: Experimental inital guess configuration active")
        initial_guess = np.concatenate((np.array(initial_guess), np.array(np.random.rand(circuit._num_inputs - len(initial_guess)))))
        x = sp.optimize.fmin_cobyla(eval_func, initial_guess, cons=[lambda x: np.all(np.less_equal(x,1))], rhobeg=0.5, rhoend=1e-12, maxfun=1000*circuit._num_inputs)
        return (circuit.matrix(x), x)

