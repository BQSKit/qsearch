from . import utils, logging
from .solver import Solver
import numpy as np
import scipy as sp
import scipy.optimize
from qsearch_rs import native_from_object
import time
from math import pi, gamma, sqrt
# from mpmath import gamma

from multiprocessing import Queue, Process
from .persistent_aposmm import initialize_APOSMM, decide_where_to_start_localopt, update_history_dist, add_to_local_H


class MultiStart_Solver(Solver):

    def __init__(self, num_threads, optimizer_name):
        # add any other initialization or config you think is necessary
        # there is nothing our API requires about the initializer
        self.num_threads = num_threads
        self.optimizer_name = optimizer_name


    # this function call needs to keep this format to work with our existiing api
    # def solve_for_unitary(self, circuit, U, error_func=utils.matrix_distance_squared, error_jac=utils.matrix_distance_squared_jac):
    def solve_for_unitary(self, circuit, options):
        circuit = native_from_object(circuit) # this converts a python circuit to a rust-implemented circuit which runs ~10x faster but conforms to the same API
        U = options.target
        logger = options.logger if "logger" in options else logging.Logger(verbosity=options.verbosity, stdout_enabled=options.stdout_enabled, output_file=options.log_file)
        if self.optimizer_name == "BFGS":
            # feel free to re-format this eval_func as long as it uses circuit, U, and error_jac in the same way
            eval_func = lambda v: error_jac(U, *circuit.mat_jac(v))
            # eval_func returns (objective_value, [jacobian values]) (with the jacobian as a 1D numpy ndarray)

        if self.optimizer_name == "least_squares":
            I = np.eye(U.shape[0])
            # because scipy least squares takes the jacobian as a separate function, our least squares code is set up accordingly
            resid_func = lambda v: options.error_residuals(U, circuit.matrix(v), I)
            jac_func = lambda v: options.error_residuals_jac(U, *circuit.mat_jac(v))

        #np.random.seed(4) # usually we do not want fixed seeds, but it can be useful for some debugging
        n = circuit.num_inputs # the number of parameters to optimize (the length that v should be when passed to one of the lambdas created above)
        initial_sample_size = 100  # How many points do you want to sample before deciding where to start runs.
        num_localopt_runs = self.num_threads  # How many localopt runs to start?

        def run_local_scipy_least_squares(x0, f, g, queue):
            '''worker function'''

            lb = np.zeros(len(x0))
            ub = np.ones(len(x0))

            res = sp.optimize.least_squares(f, x0, g, method="lm")
            queue.put(res)

        specs = {'lb': np.zeros(n),
                 'ub': np.ones(n),
                 'standalone': True,
                 'initial_sample_size':initial_sample_size}

        _, _, rk_const, ld, mu, nu, _, H = initialize_APOSMM([],specs,None)

        initial_sample = np.random.uniform(0, 1, (initial_sample_size, n))

        add_to_local_H(H, initial_sample, specs, on_cube=True)

        for i, x in enumerate(initial_sample):
            H['f'][i] = np.sum(resid_func(x)**2)

        H[['returned']] = True

        update_history_dist(H, n)
        starting_inds = decide_where_to_start_localopt(H, n, initial_sample_size, rk_const, ld, mu, nu)

        starting_points = H['x'][starting_inds[:num_localopt_runs]]

        start = time.time()
        q = Queue()
        processes = []
        rets = []
        for x0 in starting_points:
            p = Process(target=run_local_scipy_least_squares, args=(x0, resid_func, jac_func, q))
            processes.append(p)
            p.start()
        for p in processes:
            ret = q.get() # will block
            rets.append(ret)
        for p in processes:
            p.join()
        end = time.time()

        best_found = np.argmin([r['cost'] for r in rets])

        logger.logprint("Multistart with {} runs found a point with function value {} ({} seconds)".format(num_localopt_runs, rets[best_found]['cost'], end-start), verbosity=2)

        xopt = rets[best_found]['x']

        return (circuit.matrix(xopt), xopt)
