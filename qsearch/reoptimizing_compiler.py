from functools import partial
from timeit import default_timer as timer
import heapq
from scipy.stats import linregress
import numpy as np

from .circuits import *

from . import solver as scsolver
from .options import Options
from .defaults import standard_defaults as defaults, standard_smart_defaults as smart_defaults
from . import parallelizer, backend
from . import checkpoint, utils, heuristics, circuits, logging, gatesets
from .compiler import Compiler, SearchCompiler

class ReoptimizingCompiler(Compiler):
    def __init__(self, options=Options(), **xtraargs):
        self.options = options.copy()
        self.options.update(**xtraargs)
        self.options.set_defaults(verbosity=1, logfile=None, stdout_enabled=True, **defaults)
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

        overall_startime = timer() # note, because all of this setup gets included in the total time, stopping and restarting the project may lead to time durations that are not representative of the runtime under normal conditions
        dits = int(np.round(np.log(np.shape(U)[0])/np.log(options.gateset.d)))

        parallel = options.parallelizer(options)

        overall_best_pair = options.best_pair
        start_depth = len(overall_best_pair[0]._substeps) - 1
        if 'cut_depths' in options:
            # these are the "ideal" starting points, but we may need to modify them as we optimize
            midpoints = [1] + [pt + int((pt - prev)/2) for pt, prev in zip(options.cut_depths[1:],options.cut_depths)]
            print(f'midpoints initialized as {midpoints}')
        start_point = 1
        overall_best_value = options.eval_func(U, overall_best_pair[0].matrix(overall_best_pair[1]))
        while True:
            if 'timeout' in options and timer() - overall_startime > options.timeout:
                break
            best_circuit = overall_best_pair[0]
            best_circuit_depth = len(best_circuit._substeps) - 1
            if 'cut_depths' in options:
                insertion_points = midpoints
            else:
                insertion_points = range(start_point,best_circuit_depth)
            for point in insertion_points:
                if 'timeout' in options and timer() - overall_startime > options.timeout:
                    break
                startime = timer() # note, because all of this setup gets included in the total time, stopping and restarting the project may lead to time durations that are not representative of the runtime under normal conditions
                root = ProductStep(*best_circuit._substeps[:point], *best_circuit._substeps[point + depth:])
                h = options.heuristic
                dits = int(np.round(np.log(np.shape(U)[0])/np.log(options.gateset.d)))

                if options.gateset.d**dits != np.shape(U)[0]:
                    raise ValueError("The target matrix of size {} is not compatible with qudits of size {}.".format(np.shape(U)[0], self.options.gateset.d))

                I = circuits.IdentityStep(options.gateset.d)

                initial_layer = options.initial_layer if 'initial_layer' in options else options.gateset.initial_layer(dits)
                search_layers = options.gateset.search_layers(dits)

                if len(search_layers) <= 0:
                    logger.logprint("This gateset has no branching factor so only an initial optimization will be run.")
                    root = initial_layer
                    result = options.solver.solve_for_unitary(options.backend.prepare_circuit(root, options), options)
                    return (root, result[1])

                # TODO move these print statements somewhere else
                # this is good informati
                logger.logprint("There are {} processors available to Pool.".format(options.num_tasks))
                logger.logprint("The branching factor is {}.".format(len(search_layers)))
                beams = int(options.beams)
                if beams < 1 and len(search_layers) > 0:
                    beams = int(options.num_tasks // len(search_layers))
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
                    result = options.solver.solve_for_unitary(options.backend.prepare_circuit(root, options), options)
                    best_value = options.eval_func(U, result[0])
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

                options.generate_cache() # cache the results of smart_default settings, such as the default solver, before entering the main loop where the options will get pickled and the smart_default functions called many times because later caching won't persist cause of pickeling and multiple processes

                while len(queue) > 0:
                    if 'timeout' in options and timer() - overall_startime > options.timeout:
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
                    new_steps = [(current_tup[5].inserting(search_layer[0], depth=point), current_tup[1], search_layer[1]) for search_layer in search_layers for current_tup in popped]
                    for step, result, current_depth, weight in parallel.solve_circuits_parallel(new_steps):
                        current_value = options.eval_func(U, result[0])
                        new_depth = current_depth + weight
                        if (current_value < best_value and (best_value >= options.threshold or new_depth <= best_depth)) or (current_value < options.threshold and new_depth < best_depth):
                            best_value = current_value
                            best_pair = (step, result[1])
                            best_depth = new_depth
                            logger.logprint("New best! score: {} at depth: {}".format(best_value, new_depth))
                        if depth is None or new_depth < depth - 1:
                            heapq.heappush(queue, (h(current_value, new_depth), new_depth, current_value, tiebreaker, result[1], step))
                            tiebreaker+=1
                    logger.logprint("Layer completed after {} seconds".format(timer() - then), verbosity=2)
                    checkpoint.save((queue, best_depth, best_value, best_pair, tiebreaker, rectime+(timer()-startime)), statefile)


                logger.logprint("Finished compilation at depth {} with score {} after {} seconds.".format(best_depth, best_value, rectime+(timer()-startime)))
                new_circuit_depth = len(best_pair[0]._substeps) - 1
                if best_value < options.threshold and new_circuit_depth < best_circuit_depth:
                    logger.logprint(f"With starting point {point} re-optimized from depth {best_circuit_depth} to depth {new_circuit_depth}")
                    overall_best_pair = best_pair
                    overall_best_value = best_value
                    # select the points which are greater than the search window and adjust by new reoptimization
                    print(f'old midpoinst: {midpoints}')
                    midpoints = [i - (best_circuit_depth - new_circuit_depth) for i in midpoints if (i - (point + options.depth)) > 0]
                    print(f'new midpoints: {midpoints}')
                    break # break out so we can re-run optimization on the better circuit
                else:
                    logger.logprint(f"With starting point {point} no improvement was made to depth", verbosity=2)
                    print(f'old midpoinst: {midpoints}')
                    midpoints = [i for i in midpoints if (i - (point + options.depth)) > 0]
                    print(f'new midpoints: {midpoints}')
                    start_point = point
                    continue
            if new_circuit_depth >= best_circuit_depth:
                break
        parallel.done()
        logger.logprint("Finished all compilations at depth {} with score {} after {} seconds.".format(best_circuit_depth, overall_best_value, rectime+(timer()-overall_startime)))
        return {'structure': overall_best_pair[0], 'vector': overall_best_pair[1]}
        
