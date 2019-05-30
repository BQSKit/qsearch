import numpy as np
import cma

from . import circuits as circuits
from . import utils as util


class CMA_Solver():
    def solve_for_unitary(self, circuit, U, error_func=util.matrix_distance_squared, initial_guess=None):
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        if initial_guess is None:
            initial_guess = 'np.random.rand({})*4*np.pi'.format(circuit._num_inputs)
        else:
            print("WARNING: Experimental inital guess configuration active")
            initial_guess = 'np.concatenate((np.array({}), np.array(np.random.rand({})*2*np.pi)))'.format(repr(initial_guess), circuit._num_inputs - len(initial_guess))
        xopt, _ = cma.fmin2(eval_func, initial_guess, np.pi/4, {'verb_disp':0, 'verb_log':0, 'bounds' : [0,np.pi*2]}, restarts=2)
        return (circuit.matrix(xopt), xopt)


import scipy as sp
import scipy.optimize

class BFGS_Solver():
    def solve_for_unitary(self, circuit, U, error_func=util.matrix_distance_squared, initial_guess=None):
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        result = sp.optimize.minimize(eval_func, np.random.rand(circuit._num_inputs)*np.pi, method='BFGS', bounds=[(-2*np.pi, 2*np.pi) for _ in range(0, circuit._num_inputs)])
        xopt = result.x
        return (circuit.matrix(xopt), xopt)

class COBYLA_Solver():
    def solve_for_unitary(self, circuit, U, error_func=util.matrix_distance_squared, initial_guess=None):
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        if initial_guess is None:
            #initial_guess = 'np.random.rand({})*4*np.pi'.format(circuit._num_inputs)
            initial_guess = []
        #else:
            #print("WARNING: Experimental inital guess configuration active")
            #initial_guess = 'np.concatenate((np.array({}), np.array(np.random.rand({})*4*np.pi)))'.format(repr(initial_guess), circuit._num_inputs - len(initial_guess))
        initial_guess = np.concatenate((np.array(initial_guess), np.array(np.random.rand(circuit._num_inputs - len(initial_guess)) * 2 * np.pi)))
        #initial_guess = np.random.rand(circuit._num_inputs)*2*np.pi
        x = sp.optimize.fmin_cobyla(eval_func, initial_guess, cons=[lambda x: np.all(np.less_equal(x,2*np.pi))], rhobeg=np.pi, rhoend=1e-12, maxfun=1000*circuit._num_inputs)
        return (circuit.matrix(x), x)

