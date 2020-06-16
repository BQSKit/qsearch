from functools import partial
from timeit import default_timer as timer
import heapq

from .circuits import *

from . import solver as scsolver
from .options import Options
from . import parallelizer, backend
from . import checkpoint, utils, heuristics, circuits, logging, gatesets

class Compiler():
    def __init__(self, *kwargs):
        raise NotImplementedError("Subclasses of Compiler are expected to implement their own initializers with relevant args")
    def compile(self, U, depth, statefile, logger):
        raise NotImplementedError("Subclasses of Compiler are expected to implement the compile method.")
        return (U, None)

def default_error_func(options):
    if type(options.solver) == scsolver.LeastSquares_Jac_Solver or type(options.solver) == scsolver.LeastSquares_Jac_SolverNative:
        return utils.matrix_residuals
    else:
        raise AttributeError("THIS: {}".format(options.solver))
        return utils.matrix_distance_squared

def default_eval_func(options):
    if options.error_func == utils.matrix_residuals:
        return utils.matrix_distance_squared
    else:
        return options.error_func

def default_heuristic(options):
    if "search_type" in options:
        if options.search_type == "breadth":
            return heuristics.breadth
        elif options.search_type == "greedy":
            return heuristics.greedy
    return heuristics.astar

def default_error_jac(options):
    if options.error_func == utils.matrix_distance_squared:
        return utils.matrix_distance_squared_jac
    elif options.error_func == utils.matrix_residuals:
        return utils.matrix_residuals_jac
    else:
        return None

class SearchCompiler(Compiler):
    def __init__(self, options=Options(), **xtraargs):
        self.options = options.copy()
        self.options.update(**xtraargs)
        defaults = {
                "threshold":1e-10,
                "gateset":gatesets.Default(),
                "beams":-1,
                "depth":None,
                "verbosity":1,
                "stdout_enabled":True,
                "log_file":None,
                "statefile":None
                }
        smart_defaults = {
                "error_func":default_error_func,
                "eval_func":default_eval_func,
                "error_jac":default_error_jac,
                "solver":scsolver.default_solver,
                "heuristic":default_heuristic
                }

        self.options.set_defaults(**defaults)
        self.options.set_smart_defaults(**smart_defaults)

    def compile(self, options=Options(), **xtraargs):
        options = self.options.updated(options)
        if "U" in xtraargs:
            # allowing the old name for legacy code purposes
            # maybe remove this at some point
            options.target = U
        options.make_required("target")
        options.update(**xtraargs)

        U = options.target
        depth = options.depth
        statefile = options.statefile
        logger = options.logger if "logger" in options else logging.Logger(verbosity=options.verbosity, stdout_enabled=options.stdout_enabled, output_file=options.log_file)
        solver = options.solver
        eval_func = options.eval_func
        error_func = options.error_func
        error_jac = options.error_jac

        startime = timer() # note, because all of this setup gets included in the total time, stopping and restarting the project may lead to time durations that are not representative of the runtime under normal conditions
        h = options.heuristic
        dits = int(np.round(np.log(np.shape(U)[0])/np.log(options.gateset.d)))

        if options.gateset.d**dits != np.shape(U)[0]:
            raise ValueError("The target matrix of size {} is not compatible with qudits of size {}.".format(np.shape(U)[0], self.d))

        I = circuits.IdentityStep(options.gateset.d)

        initial_layer = options.gateset.initial_layer(dits)
        search_layers = options.gateset.search_layers(dits)

        if len(search_layers) <= 0:
            logger.logprint("This gateset has no branching factor so only an initial optimization will be run.")
            root = initial_layer
            result = solver.solve_for_unitary(root, U, self.eval_func)
            return (result[0], root, result[1])

        #TODO: this is a placeholder
        parallel = parallelizer.MultiprocessingParallelizer(solver, U, error_func, error_jac, backend.NativeBackend())
        logger.logprint("There are {} processors available to Pool.".format(parallel.num_tasks()))
        logger.logprint("The branching factor is {}.".format(len(search_layers)))
        beams = int(options.beams)
        if beams < 1 and len(search_layers) > 0:
            beams = int(parallel.num_tasks() // len(search_layers))
        if beams < 1:
            beams = 1
        if beams > 1:
            logger.logprint("The beam factor is {}.".format(beams))

        recovered_state = checkpoint.recover(statefile)
        queue = []
        best_depth = 0
        best_value = 0
        best_pair  = 0
        tiebreaker = 0
        rectime = 0
        if recovered_state == None:
            root = ProductStep(initial_layer)
            result = solver.solve_for_unitary(root, U, error_func, error_jac)
            best_value = eval_func(U, result[0])
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
            if best_value < options.threshold:
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
            new_steps = [(current_tup[5].appending(search_layer[0]), current_tup[1], search_layer[1]) for search_layer in search_layers for current_tup in popped]
            for step, result, current_depth, weight in parallel.solve_circuits_parallel(new_steps):
            #for step, result, current_depth, weight in pool.imap_unordered(partial(evaluate_step, U=U, error_func=self.error_func, error_jac=self.error_jac, solver=self.solver, I=I), new_steps):
                current_value = eval_func(U, result[0])
                new_depth = current_depth + weight
                if (current_value < best_value and (best_value >= options.threshold or new_depth <= best_depth)) or (current_value < options.threshold and new_depth < best_depth):
                    best_value = current_value
                    best_pair = (step, result[1])
                    best_depth = new_depth
                    logger.logprint("New best! score: {} at depth: {}".format(best_value, new_depth))
                if depth is None or new_depth < depth:
                    heapq.heappush(queue, (h(current_value, new_depth), new_depth, current_value, tiebreaker, result[1], step))
                    tiebreaker+=1
            logger.logprint("Layer completed after {} seconds".format(timer() - then), verbosity=2)
            checkpoint.save((queue, best_depth, best_value, best_pair, tiebreaker, rectime+(timer()-startime)), statefile)


        logger.logprint("Finished compilation at depth {} with score {} after {} seconds.".format(best_depth, best_value, rectime+(timer()-startime)))
        parallel.done()
        return best_pair

