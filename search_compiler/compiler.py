from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import chain
from timeit import default_timer as timer
import heapq

from .circuits import *

from . import solver as scsolver
from . import checkpoint, utils, heuristics, circuits, logging, gatesets

class Compiler():
    def compile(self, U, depth):
        raise NotImplementedError("Subclasses of Compiler are expected to implement the compile method.")
        return (U, None, None)

def evaluate_step(tup, U, error_func, error_jac, solver, I):
    step, depth = tup
    return (step, solver.solve_for_unitary(step, U, error_func), depth)

class SearchCompiler(Compiler):
    def __init__(self, threshold=1e-10, error_func=utils.matrix_distance_squared, error_jac=None, eval_func=None, heuristic=heuristics.astar, gateset=gatesets.Default(), solver=None, beams=-1, logger=None, verbosity=0):
        self.logger = logger
        self.verbosity = verbosity
        
        self.threshold = threshold
        self.error_func = error_func
        self.error_jac = error_jac

        if self.error_jac is None and error_func == utils.matrix_distance_squared:
            self.error_jac = utils.matrix_distance_squared_jac
        elif self.error_jac is None and error_func == utils.matrix_residuals:
            self.error_jac = utils.matrix_residuals_jac

        self.eval_func = eval_func

        if self.eval_func is None:
            if self.error_func == utils.matrix_residuals:
                self.eval_func = utils.matrix_distance_squared
            else:
                self.eval_func = self.error_func

        self.heuristic = heuristic
        self.gateset = gateset
        if solver is None:
            solver = scsolver.default_solver(gateset, 0, self.error_func, self.error_jac, self.logger)
            if type(solver) == scsolver.LeastSquares_Jac_Solver or type(solver) == scsolver.LeastSquares_Jac_SolverNative:
                # the default selector said we should switch to LeastSquares, so lets set the relevant values
                self.error_func = utils.matrix_residuals
                self.error_jac = utils.matrix_residuals_jac
                self.eval_func = utils.matrix_distance_squared

        self.solver = solver
        self.beams = int(beams)

    def compile(self, U, depth=None, statefile=None, logger=None):
        if logger is None:
            logger = self.logger
        if logger is None:
            logger = logging.Logger(stdout_enabled=True, verbosity=self.verbosity)

        startime = timer() # note, because all of this setup gets included in the total time, stopping and restarting the project may lead to time durations that are not representative of the runtime under normal conditions
        h = self.heuristic
        dits = int(np.round(np.log(np.shape(U)[0])/np.log(self.gateset.d)))

        if self.gateset.d**dits != np.shape(U)[0]:
            raise ValueError("The target matrix of size {} is not compatible with qudits of size {}.".format(np.shape(U)[0], self.d))

        I = circuits.IdentityStep(self.gateset.d)

        initial_layer = self.gateset.initial_layer(dits)
        search_layers = self.gateset.search_layers(dits)

        if len(search_layers) <= 0:
            logger.logprint("This gateset has no branching factor so only an initial optimization will be run.")
            root = initial_layer
            result = self.solver.solve_for_unitary(root, U, self.eval_func)
            return (result[0], root, result[1])


        logger.logprint("There are {} processors available to Pool.".format(cpu_count()))
        logger.logprint("The branching factor is {}.".format(len(search_layers)))
        beams = self.beams
        if self.beams < 1 and len(search_layers) > 0:
            beams = int(cpu_count() // len(search_layers))
        if beams < 1:
            beams = 1
        if beams > 1:
            logger.logprint("The beam factor is {}.".format(beams))
        pool = Pool(min(len(search_layers)*beams,cpu_count()))
        logger.logprint("Creating a pool of {} workers".format(pool._processes))

        recovered_state = checkpoint.recover(statefile)
        queue = []
        best_depth = 0
        best_value = 0
        best_pair  = 0
        tiebreaker = 0
        rectime = 0
        if recovered_state == None:
            root = ProductStep(initial_layer)
            result = self.solver.solve_for_unitary(root, U, self.error_func, self.error_jac)
            best_value = self.eval_func(U, result[0])
            best_pair = (root, result[1])
            logger.logprint("New best! {} at depth 0".format(best_value))
            if depth == 0:
                return best_pair

            queue = [(h(best_value, 0), 0, best_value, -1, result[1], root)]
            #         heuristic      depth  distance tiebreaker vector structure
            #             0            1      2         3         4        5
            checkpoint.save((queue, best_depth, best_value, best_pair, tiebreaker, timer()-startime), statefile)
        else:
            queue, best_depth, best_value, best_pair, tiebreaker, rectime = recovered_state
            logger.logprint("Recovered state with best result {} at depth {}".format(best_value, best_depth))

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
                logger.logprint("Popped a node with score: {} at depth: {}".format((tup[2]), tup[1]), verbosity=2)

            then = timer()
            new_steps = [(current_tup[5].appending(search_layer), current_tup[1]) for search_layer in search_layers for current_tup in popped]

            for step, result, current_depth in pool.imap_unordered(partial(evaluate_step, U=U, error_func=self.error_func, error_jac=self.error_jac, solver=self.solver, I=I), new_steps):
                current_value = self.eval_func(U, result[0])
                if (current_value < best_value and (best_value >= self.threshold or current_depth + 1 <= best_depth)) or (current_value < self.threshold and current_depth + 1 < best_depth):
                    best_value = current_value
                    best_pair = (step, result[1])
                    best_depth = current_depth + 1
                    logger.logprint("New best! score: {} at depth: {}".format(best_value, current_depth + 1))
                if depth is None or current_depth + 1 < depth:
                    heapq.heappush(queue, (h(current_value, current_depth+1), current_depth+1, current_value, tiebreaker, result[1], step))
                    tiebreaker+=1
            logger.logprint("Layer completed after {} seconds".format(timer() - then), verbosity=2)
            checkpoint.save((queue, best_depth, best_value, best_pair, tiebreaker, rectime+(timer()-startime)), statefile)


        pool.close()
        pool.terminate()
        pool.join()
        logger.logprint("Finished compilation at depth {} with score {} after {} seconds.".format(best_depth, best_value, rectime+(timer()-startime)))
        return best_pair

