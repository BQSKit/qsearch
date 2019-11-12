from multiprocessing import Pool
from functools import partial
from timeit import default_timer as timer
import heapq

from . import heuristics, gatesets, utils, circuits
from solver import default_solver


class Compiler():
    def compile(self, U, depth):
        raise NotImplementedError("Subclasses of Compiler are expected to implement the compile method.")
        return (U, None, None)

class HeapqIter(list):
    def __iter__(self):
        return self

    def __next__(self):
        if len(self) == 0:
            raise StopIteration
        else:
            return heapq.heappop(self)

    def push(self, item):
        heapq.heappush(self, item)

def processTuple

class SearchCompiler(Compiler):
    def __init__(self, threshold=1e-10, error_func=utils.matrix_distance_squared, heuristic=heuristics.astar, gateset=gatesets.Default(), solver=default_solver())
        self.threshold = threshold
        self.error_func = error_func
        self.heuristic = heuristic
        self.gateset = gateset
        self.solver = solver
   
    def compile(self, U, depth=None, statefile=None):
        if self.gateset.d**dits != np.shape(U)[0]:
            raise ValueError("The target matrix of size {} is not compatible with qudits of size {}".format(np.shape(U)[0], self.gateset.d))

        I = circuits.IdentityStep(self.gateset.d)

        initial_layer = self.gateset.initial_layer(dits)
        search_layers = self.gateset.search_layers(dits)

        if len(search_layers) <= 0:
            print("This gateset has no branching factor so only an initial optimization will be run.")
            root = initial_layer
            result = self.solver.solve_for_unitary(root, U, self.error_func)
            return (result[0], root, result[1])


        pool = Pool()
        logprint("Creating a pool of {} workers.".format(pool._processes))

        recovered_state = checkpoint.recover(statefile)
        #TODO re-implement checkpointing for the new way things work
        if recovered_state == None or True:
            root = circuits.ProductStep(initial_layer)
            result = self.solver.solve_for_unitary(root, U, self.error_func)
            best_value = self.error_func(U, result[0])
            best_pair = (result[0], root._optimize(I), result[1])
            logprint("New best! {} at depth 0".format(self.heuristic(best_value, 0)))

            if depth == 0 or best_value < self.threshold:
                return best_pair

            queue = HeapIter([(h(best_value, 0), 0, best_value, -1, result[1], root)])

            root_successors = [root.appending(layer) for layer in search_layers]
            evaluated = {root : (0, best_value, root_successors, result[1])}

        while len(queue) > 0:
            pool.imap_unordered(
