import numpy as np
import cma

import SC_Circuits as circuits
import SC_Utils as util


class CMA_Solver():
    def solve_for_unitary(self, circuit, U, error_func=util.matrix_distance_squared):
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        xopt, _ = cma.fmin2(eval_func, 'np.random.rand({})*np.pi'.format(circuit._num_inputs), np.pi/4, {'verb_disp':0, 'verb_log':0}, restarts=2)
        return (circuit.matrix(xopt), xopt)


import scipy as sp
import scipy.optimize

class BFGS_Solver():
    def solve_for_unitary(self, circuit, U, error_func=util.matrix_distance_squared):
        eval_func = lambda v: error_func(U, circuit.matrix(v))
        result = sp.optimize.minimize(eval_func, np.random.rand(circuit._num_inputs)*np.pi, method='Powell', bounds=[(-2*np.pi, 2*np.pi) for _ in range(0, circuit._num_inputs)])
        xopt = result.x
        return (circuit.matrix(xopt), xopt)

