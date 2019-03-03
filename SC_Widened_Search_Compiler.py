from multiprocessing import Pool, cpu_count
from functools import partial
import heapq
from scipy.misc import comb

from SC_Circuits import *

import SC_Gatesets as gatesets
from SC_CMA_Solver import CMA_Solver
import SC_Utils as utils
from SC_Logging import logprint

class Compiler():
    def compile(self, U, depth):
        return (U, None, None)

def evaluate_step(step, U, error_func, solver):
    return (step, solver.solve_for_unitary(step[0], U, error_func))

class Widened_Search_Compiler(Compiler):
    def __init__(self, threshold=0.01, d=2, error_func=util.matrix_distance_squared, gateset=gatesets.Default(), solver=CMA_Solver(), layer_width=2):
        self.threshold = threshold
        self.error_func = error_func
        self.d = d
        self.gateset = gateset
        self.solver = solver
        self.layer_width = layer_width

    def compile(self, U, depth):
        n = np.log(np.shape(U)[0])/np.log(self.d)

        if self.d**n != np.shape(U)[0]:
            raise ValueError("The target matrix of size {} is not compatible with qudits of size {}.".format(np.shape(U)[0], self.d))
        n = int(n)

        initial_layer = self.gateset.initial_layer(n, self.d)
        search_layers = self.gateset.search_layers(n, self.d)

        logprint("There are {} processors available to Pool.".format(cpu_count()))
        branch_factor = 1 if len(search_layers) == 1 else int((1-len(search_layers)**(self.layer_width+1)) / (1-len(search_layers)) - 1)
        logprint("The branching factor is {}.".format(branch_factor))
        pool = Pool(min(branch_factor,cpu_count()))
        logprint("Creating a pool of {} workers".format(pool._processes))

        root = (ProductStep(initial_layer), 0)
        result = self.solver.solve_for_unitary(root[0], U, self.error_func)
        best_value = self.error_func(U, result[0])
        best_pair = (result[0], root[0], result[1])
        logprint("New best! {} at depth 0".format(best_value))
        if depth == 0:
            return best_pair

        queue = [(best_value, 0, 0, root)]

        while len(queue) > 0:
            popped_value, current_depth, _, current_step = heapq.heappop(queue)
            logprint("Popped a node with score: {} at depth: {}".format(popped_value, current_depth))
            new_steps = [(current_step[0].appending(search_layer), current_step[1]+1) for search_layer in search_layers]
            gen_steps = new_steps
            for i in range(1, self.layer_width):
                gen_steps = [(gen_step[0].appending(search_layer), gen_step[1]+1) for gen_step in gen_steps for search_layer in search_layers]
                new_steps += gen_steps

            tiebreaker=0
            for step, result in pool.imap_unordered(partial(evaluate_step, U=U, error_func=self.error_func, solver=self.solver), new_steps):
                current_value = self.error_func(U, result[0])
                if current_value < best_value:
                    best_value = current_value
                    best_pair = (result[0], step[0], result[1])
                    logprint("New best! score: {} at depth: {}".format(best_value, step[1]))
                    if best_value < self.threshold:
                        pool.close()
                        pool.terminate()
                        queue = []
                        break
                if current_depth + 1 < depth:
                    heapq.heappush(queue, (current_value, step[1], tiebreaker, step))
                    tiebreaker+=1

        pool.close()
        pool.terminate()
        pool.join()
        return best_pair


