"""
This module defines PostProcessor, a class used to modify circuits after they have been synthesized.

Several implementations are provided.

Attributes:
    BasicSingleQubitReduction_PostProcessor : Attempts to remove single-qubit gates without sacrificing the quality of the solution in terms of eval_func
    ParameterTuning_PostProcessor : Attempts to reduce eval_func simply by re-running the solver with stronger parameters.
    LEAPReoptimizing_PostProcessor : Reduces the length of circuits produced using LEAP by re-running segments of the circuit.
"""
from . import options as opt
from functools import partial
from timeit import default_timer as timer
import heapq
from scipy.stats import linregress
import numpy as np

from .gates import *

from . import solvers as scsolver
from .options import Options
from .defaults import standard_defaults, standard_smart_defaults
from . import parallelizers, backends
from . import utils, heuristics, gates, logging, gatesets
from .compiler import Compiler, SearchCompiler
from .checkpoints import ChildCheckpoint

class PostProcessor():
    """This class is used to modify circuits that have already been synthesized."""
    def __init__(self, options = opt.Options()):
        self.options=options

    def post_process_circuit(self, result, options=None):
        """
        Processes the circuit dictionary and returns a new one.

        Args:
            result : A dictionary containing a synthesized circuit.  Expect it to contain "structure" and "parameters", but it may contain more, depending on what previous PostProcessors were run and on the compiler.

        Returns:
            dict : A dictionary containing any updates that should be made to the circuit dictionary, such as new values for "structure" or "parameters" or arbitrary other data.    
        """
        return result


class BasicSingleQubitReduction_PostProcessor(PostProcessor):
    """Attempts to reduce the number of single-qubit gates in a circuit by sequentially removing a gate, attempting to use a Solver on it, and keeping that gate removed if successful."""
    def post_process_circuit(self, result, options=None):
        circuit = result["structure"]
        finalx = result["parameters"]
        options = self.options.updated(options)
        if "unitary_preprocessor" in options:
            target = options.unitary_preprocessor(options.target)
        else:
            target = options.target
        single_qubit_names = ["U3Gate()", "ZXZXZGate()", "XZXZGate()"]
        identitystr = "IdentityGate()"

        circstr = repr(circuit)
        initial_count = sum([circstr.count(sqn) for sqn in single_qubit_names])
        options.logger.logprint("Initial count: {}".format(initial_count), verbosity=2)
        finalcirc = circuit
        for gate in single_qubit_names:
            components = circstr.split(gate)
            while len(components) > 1:
                newstr = components[0] + identitystr + "".join([component + gate for component in components[1:-1]]) + components[-1]
                newcirc = eval(newstr)
                mat, xopt = options.solver.solve_for_unitary(newcirc, options)
                if options.objective.gen_eval_func(newcirc, options)(xopt) < options.threshold:
                    components = [components[0] + identitystr + components[1]] + components[2:]
                    finalx = xopt
                    finalcirc = newcirc
                else:
                    components = [components[0] + gate + components[1]] + components[2:]

            circstr = components[0]
        finalstr = repr(finalcirc)
        final_count = sum([finalstr.count(sqn) for sqn in single_qubit_names])
        options.logger.logprint("Final count: {}".format(final_count), verbosity=2)
        options.logger.logprint("Post-processing removed {}, or {}% of the single qubit gates".format(initial_count-final_count, 100*(initial_count-final_count)/initial_count))
        return {"structure":finalcirc, "parameters":finalx}

class ParameterTuning_PostProcessor(PostProcessor):
    """Attempts to reduce the eval_func value of the circuit simply by tuning the parameters better using stronger Solver parameters."""
    def post_process_circuit(self, result, options=None):
        circuit = result["structure"]
        initialx = result["parameters"]
        options = self.options.updated(options)
        options.max_quality_optimization = True
        eval_func = options.objective.gen_eval_func(circuit, options)
        if "unitary_preprocessor" in options:
            target = options.unitary_preprocessor(options.target)
        else:
            target = options.target
        initial_value = eval_func(initialx)
        options.logger.logprint("Initial Distance: {}".format(initial_value))

        U, x = options.solver.solve_for_unitary(circuit, options)

        final_value = eval_func(x)
        if np.abs(final_value) < np.abs(initial_value):
            options.logger.logprint("Improved Distance: {}".format(final_value))
            return {"parameters":x}
        else:
            options.logger.logprint("Rejected Distance: {}".format(final_value))
            return {}

class LEAPReoptimizing_PostProcessor(Compiler, PostProcessor):
    """A PostProcessor that re-optimizes LeapCompiler-compiled circuits via search.

    This PostProcessor puts "holes" in the circuit where LEAP fixed prefixes and runs
    qsearch on those holes to reduce the total number of gates.
    """
    def __init__(self, options=Options()):
        self.options = Options()
        self.options.set_defaults(**standard_defaults)
        self.options.set_smart_defaults(**standard_smart_defaults)
        self.options = self.options.updated(options)

    def post_process_circuit(self, result, options=None):
        """Re-optimize a LEAP circuit. Pass "depth" to indicate the size to re-synthesize.
        It is recommended to call like:
        `project.post_process(post_processing.LEAPReoptimizing_PostProcessor(), solver=multistart_solvers.MultiStart_Solver(8), parallelizer=parallelizers.ProcessPoolParallelizer, depth=7)`
        """
        if str(result['structure']).count('CNOT') <= (options.weight_limit if 'weight_limit' in options and options.weight_limit else options.reoptimize_size):
            return result
        if 'cut_depths' not in result:
            return result
        best_pair = (result['structure'], result['parameters'])
        opts = options.updated(best_pair=best_pair, cut_depths=result['cut_depths'])
        return self.compile(opts)


    def compile(self, options=Options()):
        """Backwards compatible interface since this is technically a Compiler.

        You should use LEAPReoptimizing_PostProcessor.post_process_circuit with the Project post_processing API.
        """
        options = self.options.updated(options)
        options.make_required("target")

        if "unitary_preprocessor" in options:
            U = options.unitary_preprocessor(options.target)
        depth = options.weight_limit if 'weight_limit' in options else options.reoptimize_size
        child_checkpoint = ChildCheckpoint(Options(parent=options.checkpoint))

        logger = options.logger if "logger" in options else logging.Logger(verbosity=options.verbosity, stdout_enabled=options.stdout_enabled, output_file=options.log_file)

        overall_startime = timer() # note, because all of this setup gets included in the total time, stopping and restarting the project may lead to time durations that are not representative of the runtime under normal conditions
        qudits = int(np.round(np.log(np.shape(U)[0])/np.log(options.gateset.d)))

        parallel = options.parallelizer(options)
        recovered_outer = child_checkpoint.recover_parent()
        if recovered_outer is None:
            overall_best_pair = options.best_pair
            start_depth = len(overall_best_pair[0]._subgates) - 1
            if 'cut_depths' in options:
                # these are the "ideal" starting points, but we may need to modify them as we optimize
                midpoints = [1] + [pt + int((pt - prev)/2) for pt, prev in zip(options.cut_depths[1:],options.cut_depths)]
                logger.logprint(f'Midpoints initialized as {midpoints}', verbosity=2)
            start_point = 1
            overall_best_value = options.objective.gen_eval_func(overall_best_pair[0], options)(overall_best_pair[1])
        else:
            overall_best_pair, start_depth, midpoints, start_point, overall_best_value = recovered_outer
        try:
            while True:
                if 'timeout' in options and timer() - overall_startime > options.timeout:
                    break
                best_circuit = overall_best_pair[0]
                best_circuit_depth = len(best_circuit._subgates) - 1
                if 'cut_depths' in options:
                    insertion_points = midpoints
                else:
                    insertion_points = range(start_point,best_circuit_depth)
                for point in insertion_points:
                    if 'timeout' in options and timer() - overall_startime > options.timeout:
                        break
                    startime = timer() # note, because all of this setup gets included in the total time, stopping and restarting the project may lead to time durations that are not representative of the runtime under normal conditions
                    window_size = depth or options.reoptimize_size
                    root = ProductGate(*best_circuit._subgates[:point], *best_circuit._subgates[point + window_size:])
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

                    recovered_state = child_checkpoint.recover()
                    queue = []
                    best_depth = 0
                    best_value = 0
                    best_pair  = 0
                    tiebreaker = 0
                    rectime = 0
                    if recovered_state == None:
                        result = options.solver.solve_for_unitary(options.backend.prepare_circuit(root, options), options)
                        best_value = options.objective.gen_eval_func(root, options)(result[1])
                        best_pair = (root, result[1])
                        logger.logprint("New best! {} at depth 0".format(best_value))
                        if depth == 0:
                            return best_pair

                        queue = [(h(*best_pair, 0, options), 0, best_value, -1, result[1], root)]
                        #         heuristic      depth  distance tiebreaker parameters structure
                        #             0            1      2         3         4        5
                        child_checkpoint.save((queue, best_depth, best_value, best_pair, tiebreaker, timer()-startime))
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
                            current_value = options.objective.gen_eval_func(step, options)(result[1])
                            new_depth = current_depth + weight
                            if (current_value < best_value and (best_value >= options.threshold or new_depth <= best_depth)) or (current_value < options.threshold and new_depth < best_depth):
                                best_value = current_value
                                best_pair = (step, result[1])
                                best_depth = new_depth
                                logger.logprint("New best! score: {} at depth: {}".format(best_value, new_depth))
                            if depth is None or new_depth < depth - 1:
                                heapq.heappush(queue, (h(step, result[1], new_depth, options), new_depth, current_value, tiebreaker, result[1], step))
                                tiebreaker+=1
                        logger.logprint("Layer completed after {} seconds".format(timer() - then), verbosity=2)
                        if (options.weight_limit is not None and best_depth >= options.weight_limit - 1) or ('reoptimize_size' in options and best_depth >= options.reoptimize_size - 1):
                            break
                        child_checkpoint.save((queue, best_depth, best_value, best_pair, tiebreaker, rectime+(timer()-startime)))


                    logger.logprint("Finished compilation at depth {} with score {} after {} seconds.".format(best_depth, best_value, rectime+(timer()-startime)))
                    new_circuit_depth = len(best_pair[0]._subgates) - 1
                    if best_value < options.threshold and new_circuit_depth < best_circuit_depth:
                        logger.logprint(f"With starting point {point} re-optimized from depth {best_circuit_depth} to depth {new_circuit_depth}")
                        overall_best_pair = best_pair
                        overall_best_value = best_value
                        # select the points which are greater than the search window and adjust by new reoptimization
                        logger.logprint(f'old midpoints: {midpoints}')
                        midpoints = [i - (best_circuit_depth - new_circuit_depth) for i in midpoints if (i - (point + window_size)) > 0]
                        logger.logprint(f'new midpoints: {midpoints}')
                        child_checkpoint.save(None)
                        child_checkpoint.save_parent((overall_best_pair, start_depth, midpoints, start_point, overall_best_value))
                        break # break out so we can re-run optimization on the better circuit
                    else:
                        logger.logprint(f"With starting point {point} no improvement was made to depth", verbosity=2)
                        logger.logprint(f'old midpoints: {midpoints}')
                        midpoints = [i for i in midpoints if (i - (point + window_size)) > 0]
                        logger.logprint(f'new midpoints: {midpoints}')
                        start_point = point
                        child_checkpoint.save(None)
                        child_checkpoint.save_parent((overall_best_pair, start_depth, midpoints, start_point, overall_best_value))
                        continue
                if new_circuit_depth >= best_circuit_depth:
                    break
        finally:
            parallel.done()
        logger.logprint("Finished all compilations at depth {} with score {} after {} seconds.".format(best_circuit_depth, overall_best_value, rectime+(timer()-overall_startime)))
        return {'structure': overall_best_pair[0], 'parameters': overall_best_pair[1]}
