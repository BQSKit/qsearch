from functools import partial
from timeit import default_timer as timer
import heapq

from .circuits import *

from . import solver as scsolver
from .options import Options
from .defaults import standard_defaults, standard_smart_defaults
from . import parallelizer, backend
from . import checkpoint, utils, heuristics, circuits, logging, gatesets

class Compiler():
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses of Compiler are expected to implement their own initializers with relevant args")
    def compile(self, U, depth, statefile, logger):
        raise NotImplementedError("Subclasses of Compiler are expected to implement the compile method.")
        return (U, None)

class SearchCompiler(Compiler):
    def __init__(self, options=Options(), **xtraargs):
        self.options = options.copy()
        self.options.update(**xtraargs)
        self.options.set_defaults(verbosity=1, logfile=None, stdout_enabled=True, **standard_defaults)
        self.options.set_smart_defaults(**standard_smart_defaults)

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
        logger = options.logger

        starttime = timer() # note, because all of this setup gets included in the total time, stopping and restarting the project may lead to time durations that are not representative of the runtime under normal conditions
        h = options.heuristic
        dits = int(np.round(np.log(np.shape(U)[0])/np.log(options.gateset.d)))

        if options.gateset.d**dits != np.shape(U)[0]:
            raise ValueError("The target matrix of size {} is not compatible with qudits of size {}.".format(np.shape(U)[0], self.options.gateset.d))

        I = circuits.IdentityStep(options.gateset.d)

        initial_layer = options.gateset.initial_layer(dits)
        branching_factor = options.gateset.branching_factor(dits)
        if branching_factor <= 0:
            logger.logprint("This gateset has no branching factor so only an initial optimization will be run.")
            root = initial_layer
            result = options.solver.solve_for_unitary(options.backend.prepare_circuit(root, options), options)
            return (root, result[1])

        parallel = options.parallelizer(options)
        # TODO move these print statements somewhere else
        # this is good informati
        logger.logprint("There are {} processors available to Pool.".format(options.num_tasks))
        logger.logprint("The branching factor is {}.".format(branching_factor))
        beams = int(options.beams)
        if beams < 1 and branching_factor > 0:
            beams = int(options.num_tasks // branching_factor)
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
            if isinstance(initial_layer, ProductStep):
                root = initial_layer
            else:
                root = ProductStep(initial_layer)
            root = ProductStep(initial_layer)
            result = options.solver.solve_for_unitary(options.backend.prepare_circuit(root, options), options)
            best_value = options.eval_func(U, result[0])
            best_pair = (root, result[1])
            logger.logprint("New best! {} at depth 0".format(best_value))
            if depth == 0:
                return best_pair

            queue = [(h(best_value, 0), 0, best_value, -1, result[1], root)]
            #         heuristic      depth  distance tiebreaker vector structure
            #             0            1      2         3         4        5
            checkpoint.save((queue, best_depth, best_value, best_pair, tiebreaker, timer()-starttime), statefile)
        else:
            queue, best_depth, best_value, best_pair, tiebreaker, rectime = recovered_state
            logger.logprint("Recovered state with best result {} at depth {}".format(best_value, best_depth))

        options.generate_cache() # cache the results of smart_default settings, such as the default solver, before entering the main loop where the options will get pickled and the smart_default functions called many times because later caching won't persist cause of pickeling and multiple processes

        while len(queue) > 0:
            if 'timeout' in options and timer() - starttime > options.timeout:
                break
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
            new_steps = [(successor[0], current_tup[1], successor[1]) for current_tup in popped for successor in options.gateset.successors(current_tup[5])]
            for step, result, current_depth, weight in parallel.solve_circuits_parallel(new_steps):
                current_value = options.eval_func(U, result[0])
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
            checkpoint.save((queue, best_depth, best_value, best_pair, tiebreaker, rectime+(timer()-starttime)), statefile)


        logger.logprint("Finished compilation at depth {} with score {} after {} seconds.".format(best_depth, best_value, rectime+(timer()-starttime)))
        parallel.done()
        return {'structure': best_pair[0], 'vector': best_pair[1]}

