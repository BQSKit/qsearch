from multiprocessing import Pool, cpu_count
from functools import partial
from timeit import default_timer as timer
import heapq

from .circuits import *

from . import gatesets as gatesets
from .solver import default_solver
from .logging import logprint
from . import checkpoint, utils, heuristics

class Compiler():
    def compile(self, U, depth):
        raise NotImplementedError("Subclasses of Compiler are expected to implement the compile method.")
        return (U, None, None)

def evaluate_step(tup, U, error_func, solver):
    step, depth, initial_guess = tup
    return (step, solver.solve_for_unitary(step, U, error_func, initial_guess), depth)

class SearchCompiler(Compiler):
    def __init__(self, threshold=0.01, d=2, error_func=utils.matrix_distance_squared, heuristic=heuristics.astar, gateset=gatesets.Default(), solver=default_solver(), beams=1):
        self.threshold = threshold
        self.error_func = error_func
        self.heuristic = heuristic
        self.d = d
        self.gateset = gateset
        self.solver = solver
        self.beams = int(beams)

    def compile(self, U, depth=None, statefile=None):
        h = self.heuristic
        n = int(np.round(np.log(np.shape(U)[0])/np.log(self.d)))

        if self.d**n != np.shape(U)[0]:
            raise ValueError("The target matrix of size {} is not compatible with qudits of size {}.".format(np.shape(U)[0], self.d))
        n = int(n)

        initial_layer = self.gateset.initial_layer(n, self.d)
        search_layers = self.gateset.search_layers(n, self.d)

        if len(search_layers) <= 0:
            print("This gateset has no branching factor so only an initial optimization will be run.")
            root = initial_layer
            result = self.solver.solve_for_unitary(root, U, self.error_func)
            return (result[0], root, result[1])


        logprint("There are {} processors available to Pool.".format(cpu_count()))
        logprint("The branching factor is {}.".format(len(search_layers)))
        beams = self.beams
        if self.beams < 1 and len(search_layers) > 0:
            beams = int(cpu_count() // len(search_layers))
        if beams < 1:
            beams = 1
        if beams > 1:
            logprint("The beam factor is {}.".format(beams))
        pool = Pool(min(len(search_layers)*beams,cpu_count()))
        logprint("Creating a pool of {} workers".format(pool._processes))

        recovered_state = checkpoint.recover(statefile)
        queue = []
        best_depth = 0
        best_value = 0
        best_pair  = 0
        tiebreaker = 0
        if recovered_state == None:
            root = ProductStep(initial_layer)
            root.index = 0
            result = self.solver.solve_for_unitary(root, U, self.error_func)
            best_value = self.error_func(U, result[0])
            best_pair = (result[0], root, result[1])
            logprint("New best! {} at depth 0".format(best_value/10))
            if depth == 0:
                return best_pair

            queue = [(h(best_value, 0), 0, best_value, -1, result[1], root)]
            #         heuristic      depth  distance tiebreaker vector structure
            #             0            1      2         3         4        5
            checkpoint.save((queue, best_depth, best_value, best_pair, tiebreaker), statefile)
        else:
            queue, best_depth, best_value, best_pair, tiebreaker = recovered_state
            logprint("Recovered state with best result {} at depth {}".format(best_value, best_depth))

        while len(queue) > 0:
            if best_value < self.threshold:
                pool.close()
                pool.terminate()
                queue = []
                break
            popped = []
            for _ in range(0, beams):
                if len(queue) == 0:
                    break
                tup = heapq.heappop(queue)
                popped.append(tup)
                logprint("Popped a node with score: {} at depth: {} with branch index: {}".format((tup[2]), tup[1], tup[5].index))

            then = timer()
            new_steps = [(current_tup[5].appending(search_layer), current_tup[1], current_tup[4]) for search_layer in search_layers for current_tup in popped]
            for i in range(0, len(new_steps)):
                new_steps[i][0].index = i

            for step, result, current_depth in pool.imap_unordered(partial(evaluate_step, U=U, error_func=self.error_func, solver=self.solver), new_steps):
                current_value = self.error_func(U, result[0])
                logprint("{}\t{}".format(current_value, current_depth+1), custom="heuristic-test")
                if (current_value < best_value and (best_value >= self.threshold or current_depth + 1 <= best_depth)) or (current_value < self.threshold and current_depth + 1 < best_depth):
                    best_value = current_value
                    best_pair = (result[0], step, result[1])
                    best_depth = current_depth + 1
                    logprint("New best! score: {} at depth: {} with branch index: {}".format(best_value, current_depth + 1, step.index))
                if depth is None or current_depth + 1 < depth:
                    heapq.heappush(queue, (h(current_value, current_depth+1), current_depth+1, current_value, tiebreaker, result[1], step))
                    tiebreaker+=1
            logprint("Layer completed after {} seconds".format(timer() - then))
            checkpoint.save((queue, best_depth, best_value, best_pair, tiebreaker), statefile)


        pool.close()
        pool.terminate()
        pool.join()
        logprint("Finished compilation at depth {} with score {}.".format(best_depth, best_value/10))
        logprint("final depth: {}".format(best_depth), custom="heuristic-depth")
        if statefile == None:
            checkpoint.delete()
        return best_pair

