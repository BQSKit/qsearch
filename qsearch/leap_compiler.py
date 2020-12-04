"""
This module provides LeapCompiler, which is a more scalable variant of SearchCompiler, at the expense of producing somewhat longer circuits.  LeapReoptimizing_PostProcessor can be used to reduce circuit length back to levels that SearchCompiler might generate.
"""
from functools import partial
from timeit import default_timer as timer
import heapq
from scipy.stats import linregress
import numpy as np

from . import solvers as scsolver
from .options import Options
from .defaults import standard_defaults, standard_smart_defaults
from . import parallelizers, backends
from . import utils, heuristics, gates, logging, gatesets
from .compiler import Compiler, SearchCompiler
from .checkpoints import ChildCheckpoint


def cut_end(circ, depth):
    if isinstance(circ._substeps[0], gates.ProductGate):
        return cut_end(circ._substeps[0], depth)
    return gates.ProductGate(*circ._substeps[:-depth])


class LeapCompiler(Compiler):
    """LeapCompiler is a more scalable search based circuit compiler

    LeapCompiler uses fixed structure prefixes to greatly reduce the search space
    and speed up synthesis at the cost of optimiality. Thus it is recommended to use in conjunction
    with reoptimizing_compiler.LeapReoptimizing_PostProcessor() to obtain the best results.

    Options:
        target (required) : The unitary matrix to be synthesized, in the form of a numpy ndarray with dtype="complex128".
        gateset : The Gateset used for synthesis.
        weight_limit : A limit on the maximum weight for circuits to be expanded for further searching.  See gatesets.py for more information.  The default is None for unlimited.
        heuristic : A heuristic used to order the search tree.  See heuristics.py for more information.
        solver : A Solver used for optimizing the parameters in parameterized circuits generated by the search tree.
        parallelizer : A Parallelizer used for solving multiple parameterized circuits in parallel.
        beams : The number of nodes to pop from the search tree at a time.  The default value of -1 will create enough branches to maximize utilization of your CPU.
        error_func : The function that the Solver will attempt to minimize.
        eval_func : The function used by the heuristic in order to guide the search tree.  By default this is equal to error_func.
        error_jac : A function that returns a tuple of the value that error_func would generate and the jacobian of error_func
        error_residuals : A function that returns an array of real-valued residuals to be used by a least-squares-based Solver.
        error_residuals_jac : A function that returns the jacobian of error_residuals (note that it does NOT return a tuple of the residuals and the jacobian).
        timeout : An uper limit on the amount of time the compiler will spend trying to synthesize a circuit.  The default is float('inf'), for unlimited.
        checkpoint : The compiler will use this Checkpoint to save intermediate state, and will resume from this Checkpoint if there was an existing state.
        logger : A qsearch.logging.Logger that will be used for logging the synthesis process.
        min_depth : the minimum amount of searching 
    """
    def __init__(self, options=Options()):
        """Run LEAP on the compilation specified in options.
        
        Args:
            options: options for the compilations, see the class level documentation for details.
        """
        self.options = options.copy()
        self.options.set_defaults(verbosity=1, logfile=None, stdout_enabled=True, **standard_defaults)
        self.options.set_smart_defaults(**standard_smart_defaults)

    def compile(self, options=Options()):
        """Run LEAP on the compilation specified in options.
        
        Args:
            options: options for the compilations, see the class level documentation for details.
        """

        options = self.options.updated(options)
        options.make_required("target")

        U = options.unitary_preprocessor(options.target)
        depth = options.weight_limit

        child_checkpoint = ChildCheckpoint(Options(parent=options.checkpoint))

        logger = options.logger if "logger" in options else logging.Logger(verbosity=options.verbosity, stdout_enabled=options.stdout_enabled, output_file=options.log_file)

        starttime = timer() # note, because all of this setup gets included in the total time, stopping and restarting the project may lead to time durations that are not representative of the runtime under normal conditions
        rectime = 0
        qudits = int(np.round(np.log(np.shape(U)[0])/np.log(options.gateset.d)))

        sub_compiler = options.sub_compiler_class if 'sub_compiler_class' in options else SubCompiler
        sc = sub_compiler(options)
        recovered_state = child_checkpoint.recover_parent()
        if recovered_state is None:
            total_depth = 0
            best_value = 1.0
            depths = [total_depth]
            initial_layer = options.gateset.initial_layer(qudits)
        else:
            total_depth = recovered_state[0]
            best_value = recovered_state[1]
            depths = recovered_state[2]
            rectime = recovered_state[3]
            initial_layer = recovered_state[4]
        while True:
            if 'timeout' in options and timer() - starttime > options.timeout:
                break
            opts = options.updated(initial_layer=initial_layer, local_threshold=options.delta * best_value, overall_starttime=starttime, overall_best_value=best_value, checkpoint=child_checkpoint)
            best_pair, best_value, best_depth = sc.compile(opts)
            # clear child checkpoint for next run
            child_checkpoint.delete()
            total_depth += best_depth
            depths.append(total_depth)
            if best_value < options.threshold:
                break
            initial_layer = best_pair[0]
            child_checkpoint.save_parent((total_depth, best_value, depths, timer()-starttime, initial_layer))
        logger.logprint("Finished all sub-compilations at depth {} with score {} after {} seconds.".format(total_depth, best_value, (timer()-starttime)))
        return {'structure': best_pair[0], 'parameters': best_pair[1], 'cut_depths': depths}


class SubCompiler(Compiler):
    """A modified SearchCompiler for the LeapCompiler to use.
    """
    def __init__(self, options=Options()):
        self.options = options.copy()
        self.options.set_defaults(verbosity=1, logfile=None, stdout_enabled=True, **standard_defaults)
        self.options.set_smart_defaults(**standard_smart_defaults)

    def compile(self, options=Options()):
        options = self.options.updated(options)
        options.make_required("target")

        if "unitary_preprocessor" in options:
            U = options.unitary_preprocessor(options.target)
        depth = options.weight_limit
        checkpoint = options.checkpoint

        logger = options.logger if "logger" in options else logging.Logger(verbosity=options.verbosity, stdout_enabled=options.stdout_enabled, output_file=options.log_file)

        starttime = timer() # note, because all of this setup gets included in the total time, stopping and restarting the project may lead to time durations that are not representative of the runtime under normal conditions
        h = options.heuristic
        qudits = int(np.round(np.log(np.shape(U)[0])/np.log(options.gateset.d)))

        if options.gateset.d**qudits != np.shape(U)[0]:
            raise ValueError("The target matrix of size {} is not compatible with qudits of size {}.".format(np.shape(U)[0], self.options.gateset.d))

        I = gates.IdentityGate(d=options.gateset.d)

        initial_layer = options.initial_layer if 'initial_layer' in options else options.gateset.initial_layer(qudits)
        search_layers = options.gateset.search_layers(qudits)

        if len(search_layers) <= 0:
            logger.logprint("This gateset has no branching factor so only an initial optimization will be run.")
            root = initial_layer
            result = options.solver.solve_for_unitary(options.backend.prepare_circuit(root, options), options)
            return (root, result[1])

        parallel = options.parallelizer(options)
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

        recovered_state = checkpoint.recover()
        queue = []
        best_depth = 0
        best_value = 0
        best_pair  = 0
        tiebreaker = 0
        rectime = 0
        if recovered_state == None:
            if isinstance(initial_layer, gates.ProductGate):
                root = initial_layer
            else:
                root = gates.ProductGate(initial_layer)
            result = options.solver.solve_for_unitary(options.backend.prepare_circuit(root, options), options)
            best_value = options.eval_func(U, result[0])
            best_pair = (root, result[1])
            logger.logprint("New best! {} at depth 0".format(best_value))
            if depth == 0:
                return (best_pair, best_value, 0)

            queue = [(h(*best_pair, 0, options), 0, best_value, -1, result[1], root)]
            #         heuristic      depth  distance tiebreaker parameters structure
            #             0            1      2         3         4        5
            checkpoint.save((queue, best_depth, best_value, best_pair, tiebreaker, timer()-starttime))
        else:
            queue, best_depth, best_value, best_pair, tiebreaker, rectime = recovered_state
            logger.logprint("Recovered state with best result {} at depth {}".format(best_value, best_depth))

        options.generate_cache() # cache the results of smart_default settings, such as the default solver, before entering the main loop where the options will get pickled and the smart_default functions called many times because later caching won't persist cause of pickeling and multiple processes
        previous_bests_depths = []
        previous_bests_values = []
        try:
            while len(queue) > 0:
                if 'timeout' in options and timer() - options.overall_starttime > options.timeout:
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
                new_steps = [(current_tup[5].appending(search_layer[0]), current_tup[1], search_layer[1]) for search_layer in search_layers for current_tup in popped]
                for step, result, current_depth, weight in parallel.solve_circuits_parallel(new_steps):
                    current_value = options.eval_func(U, result[0])
                    new_depth = current_depth + weight
                    if (current_value < best_value and (best_value >= options.threshold or new_depth <= best_depth)) or (current_value < options.threshold and new_depth < best_depth):
                        best_value = current_value
                        best_pair = (step, result[1])
                        best_depth = new_depth
                        logger.logprint("New best! score: {} at depth: {}".format(best_value, new_depth))
                        if len(previous_bests_values) > 1:
                            with np.errstate(invalid='ignore', divide='ignore'):
                                slope, intercept, _rval, _pval, _stderr = linregress(previous_bests_depths, previous_bests_values)
                            predicted_best = slope * new_depth + intercept
                            delta = predicted_best - best_value
                            logger.logprint(f"Predicted best value {predicted_best} for new best with delta {delta}", verbosity=2)
                            if not np.isnan(predicted_best) and best_value < options.overall_best_value and delta < 0 and ('min_depth' not in options or new_depth >= options.min_depth):
                                parallel.done()
                                return (best_pair, best_value, best_depth)
                        previous_bests_depths.append(best_depth)
                        previous_bests_values.append(best_value)

                    if depth is None or new_depth < depth:
                        heapq.heappush(queue, (h(step, result[1], new_depth, options), new_depth, current_value, tiebreaker, result[1], step))
                        tiebreaker+=1
                logger.logprint("Layer completed after {} seconds".format(timer() - then), verbosity=2)
                checkpoint.save((queue, best_depth, best_value, best_pair, tiebreaker, rectime+(timer()-starttime)))
        finally:
            parallel.done()

        logger.logprint("Finished compilation at depth {} with score {} after {} seconds.".format(best_depth, best_value, rectime+(timer()-starttime)))
        return (best_pair, best_value, best_depth)
