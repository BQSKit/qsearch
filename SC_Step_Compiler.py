from multiprocessing import Pool, cpu_count
from functools import partial
import heapq

from SC_Circuits import *

import SC_Gatesets as gatesets
from SC_CMA_Solver import CMA_Solver
import SC_Utils as utils
from SC_Logging import logprint

class Compiler():
    def compile(self, U, depth):
        return (U, None, None)

def evaluate_step(step, U, error_func, solver):
    return (step, solver.solve_for_unitary(step, U, error_func))

class Step_Compiler(Compiler):
    def __init__(self, threshold=0.01, d=2, error_func=util.matrix_distance_squared, gateset=gatesets.Default(), solver=CMA_Solver()):
        self.threshold = threshold
        self.error_func = error_func
        self.d = d
        self.gateset = gateset
        self.solver = solver

    def compile(self, U, depth):
        n = np.log(np.shape(U)[0])/np.log(self.d)

        if self.d**n != np.shape(U)[0]:
            raise ValueError("The target matrix of size {} is not compatible with qudits of size {}.".format(np.shape(U)[0], self.d))
        n = int(n)

        initial_layer = self.gateset.initial_layer(n, self.d)
        search_layers = self.gateset.search_layers(n, self.d)

        logprint("There are {} processors available to Pool.".format(cpu_count()))
        logprint("The branching factor is {}.".format(len(search_layers)))
        pool = Pool(min(len(search_layers),cpu_count()))
        logprint("Creating a pool of {} workers".format(pool._processes))

        root = ProductStep(initial_layer)
        result = self.solver.solve_for_unitary(root, U, self.error_func)
        best_value = self.error_func(U, result[0])
        best_pair = (result[0], root, result[1])
        logprint("New best! {} at depth 0".format(best_value))
        if depth == 0:
            return best_pair

        queue = [(best_value, 0, 0, root, result[0], result[1])]

        while len(queue) > 0:
            popped_value, current_depth, _, current_step, current_mat, current_vec  = heapq.heappop(queue)
            logprint("Popped a node with score: {} at depth: {}".format(popped_value, current_depth))

            tiebreaker=0
            for step, result in pool.imap_unordered(partial(evaluate_step, U=np.matmul(current_mat.H, U), error_func=self.error_func, solver=self.solver), search_layers):
                print(type(result[1]))
                new_vec = np.concatenate((current_vec, result[1]))
                new_mat = np.matmul(current_mat, result[0])
                new_step = current_step.appending(step)
                new_value = self.error_func(U, new_mat)
                if new_value < best_value:
                    best_value = new_value
                    best_pair = (new_mat, new_step, new_vec)
                    logprint("New best! score: {} at depth: {}".format(best_value, current_depth + 1))
                    if best_value < self.threshold:
                        pool.close()
                        pool.terminate()
                        queue = []
                        break
                if current_depth + 1 < depth:
                    heapq.heappush(queue, (new_value, current_depth+1, tiebreaker, new_step, new_mat, new_vec))
                    tiebreaker+=1

        pool.close()
        pool.terminate()
        pool.join()
        return best_pair

