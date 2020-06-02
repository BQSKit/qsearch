from . import utils
from .solver import Solver
import numpy as np
import scipy as sp
import scipy.optimize
from search_compiler_rs import native_from_object
import time

from math import gamma, pi, sqrt

from persistent_aposmm import aposmm

class MultiStart_Solver(Solver):

    def __init__(self, num_threads, optimizer_name):
        # add any other initialization or config you think is necessary
        # there is nothing our API requires about the initializer
        self.num_threads = num_threads
        self.optimizer_name = optimizer_name


    # this function call needs to keep this format to work with our existiing api
    def solve_for_unitary(self, circuit, U, error_func=utils.matrix_distance_squared, error_jac=utils.matrix_distance_squared_jac):

        circuit = native_from_object(circuit) # this converts a python circuit to a rust-implemented circuit which runs ~10x faster but conforms to the same API

        if self.optimizer_name == "BFGS":
            # feel free to re-format this eval_func as long as it uses circuit, U, and error_jac in the same way
            eval_func = lambda v: error_jac(U, *circuit.mat_jac(v))
            # eval_func returns (objective_value, [jacobian values]) (with the jacobian as a 1D numpy ndarray)

        if self.optimizer_name == "least_squares":
            I = np.eye(U.shape[0])
            # because scipy least squares takes the jacobian as a separate function, our least squares code is set up accordingly
            resid_func = lambda v: error_func(U, circuit.matrix(v), I)
            jac_func = lambda v: error_jac(U, *circuit.mat_jac(v))
            # resid_func returns [residuals] (as a 1D numpy ndarray)
            # jac_func returns the jacobian as a 2D numpy ndarray

            # note that in order to use least_squares, error_func should be set to utils.matrix_residuals, and error_jac should be set to utils.matrix_residuals_jac

        np.random.seed(4)
        n = circuit.num_inputs # the number of parameters to optimize (the length that v should be when passed to one of the lambdas created above)

        # start = time.time()
        # result = sp.optimize.minimize(eval_func, np.random.rand(circuit.num_inputs)*np.pi, method='BFGS', jac=True)
        # end = time.time()
        # xopt = result.x
        # print("BFGS found a point with function value {} after {} function evaluations ({} seconds)".format(result.fun, result.nfev, end-start),flush=True)

        start = time.time()
        num_par_runs = 5
        eval_max = num_par_runs*1000

        persis_info = {'rand_stream': np.random.RandomState(4), 'nworkers': 4}


        gen_out = [('x', float, n), ('x_on_cube', float, n), ('sim_id', int),
                   ('local_min', bool), ('local_pt', bool)]
        gen_specs = {'in': ['x', 'f', 'grad', 'local_pt', 'sim_id', 'returned', 'x_on_cube', 'local_min'],
                     'out': gen_out,
                     'user': {'initial_sample_size': 200,
                              # 'localopt_method': 'LD_MMA', # Needs gradients
                              'localopt_method': 'scipy_BFGS',
                              'opt_return_codes': [0],
                              'standalone': {'eval_max': eval_max,
                                             'obj_and_grad_func': eval_func,
                                             },
                              'rk_const': 0.5*((gamma(1+(n/2))*5)**(1/n))/sqrt(pi),
                              'xtol_abs': 1e-6,
                              'ftol_abs': 1e-6,
                              'dist_to_bound_multiple': 0.5,
                              'max_active_runs': num_par_runs,
                              'lb': np.zeros(n),
                              'ub': np.pi*np.ones(n)}
                     }
        H = []
        start = time.time()
        H, persis_info, exit_code = aposmm(H, persis_info, gen_specs, {})
        end = time.time()

        assert np.sum(H['returned']) >= eval_max, "Standalone persistent_aposmm, didn't evaluate enough points"
        assert persis_info.get('run_order'), "Standalone persistent_aposmm didn't do any localopt runs"

        best_found = np.argmin(H['f'][H['returned']])

        print("APOSMM found a point with function value {} after {} function evaluations ({} seconds)".format(np.min(H['f'][H['returned']]), len(H), end-start),flush=True)

        xopt = H['x'][best_found]

        return (circuit.matrix(xopt), xopt)

