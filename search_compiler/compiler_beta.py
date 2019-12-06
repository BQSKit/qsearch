from multiprocessing import Pool, Lock
from functools import partial
from timeit import default_timer as timer
import heapq
import numpy as np

from . import heuristics, gatesets, utils, circuits, checkpoint
from .solver import default_solver
from .logging import logprint, logstandard

class Compiler():
    def compile(self, U, depth):
        raise NotImplementedError("Subclasses of Compiler are expected to implement the compile method.")
        return (U, None, None)

class HeapIter(list):
    def __iter__(self):
        return self

    def __next__(self):
        if len(self) == 0:
            raise StopIteration
        else:
            i = 0
            while i < len(self.targets):
                target = self.targets[i]
                if hash(target[-1]) in self.completed or hash(target[-1]) in self._in_progress:
                    self.targets.pop(i)
                else:
                    i += 1
            if len(self.targets) > 0:
                node = self.targets.pop()
                logstandard("Prioritized", node[0], node[2], node[1], hash(node[-1]), len(self))
                self._in_progress[hash(node[-1])] = node
            else:
                node = heapq.heappop(self)
                while hash(node[-1]) in self.completed or hash(node[-1]) in self._in_progress:
                    if len(self) < 1:
                        raise StopIteration
                    node = heapq.heappop(self)
                self._in_progress[hash(node[-1])] = node
                logstandard("Popped", node[0], node[2], node[1], hash(node[-1]), len(self))
            return node

    def add_target(self, target):
        self.targets += [target]

    def add_helper(self, completed):
        self.targets = []
        self.completed = completed
        self._in_progress = dict()

    def finish(self, key):
        if key in self._in_progress:
            self._in_progress.pop(key)

    def pop(self):
        return heapq.heappop(self)

    def push(self, item):
        heapq.heappush(self, item)

def run_optimization(intup, U, error_func, solver, I, heuristic):
    step = intup[-1]
    depth = intup[1]
    ostep = step._optimize(I)
    result = solver.solve_for_unitary(ostep, U, error_func)
    value = error_func(U, result[0])
    return (step, result[1], depth + 1, value, heuristic(value, depth + 1), intup[0], intup[2])

class SearchCompiler(Compiler):
    def __init__(self, threshold=1e-10, error_func=utils.matrix_distance_squared, heuristic=heuristics.astar, gateset=gatesets.Default(), solver=default_solver()):
        self.threshold = threshold
        self.error_func = error_func
        self.heuristic = heuristic
        self.gateset = gateset
        self.solver = solver
   
    def compile(self, U, depth=None, statefile=None):
        start = timer()
        dits = int(np.round(np.log(np.shape(U)[0])/np.log(self.gateset.d)))
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
        if recovered_state == None:
            root = circuits.ProductStep(initial_layer)
            result = self.solver.solve_for_unitary(root, U, self.error_func)
            best_value = self.error_func(U, result[0])
            best_depth = 0
            best_pair = (result[0], root._optimize(I), result[1])
            logstandard("New best!", self.heuristic(best_value, best_depth), best_value, best_depth, hash(root), 0)

            if depth == 0 or best_value < self.threshold:
                return best_pair

            root_successors = [root.appending(layer) for layer in search_layers]
            search_queue = HeapIter([])
            results = {hash(root) : (self.heuristic(best_value, 0), 0, best_value, -1, result[1], root_successors)}
            search_head = (root, root_successors, result[1], self.heuristic(best_value, best_depth), best_depth, best_value)
            process_queue = HeapIter([])
            tiebreaker = 0
            for successor in root_successors:
                # items in the process queue have the sorting info from their parent
                process_queue.push((self.heuristic(best_value, 0), 0, best_value, tiebreaker, successor))
                tiebreaker += 1
        else:
            # time to recover the saved state
            search_head, results, search_queue, process_queue, best_depth, best_value, best_pair, tiebreaker, in_progress = recovered_state
            search_queue = HeapIter(search_queue)
            process_queue = HeapIter(process_queue)
            for key in in_progress:
                process_queue.push(in_progress[key])
            logprint("Recovered a state with best result H: {} V: {} D: {}".format(self.heuristic(best_value, best_depth), best_value, best_depth))

        process_queue.add_helper(results)
        for successor in search_head[1]:
            process_queue.add_target((search_head[3], search_head[4], search_head[5], tiebreaker, successor))
            tiebreaker += 1
        while len(process_queue) > 0:
            for step, vector, current_depth, value, h, old_h, old_v in pool.imap_unordered(partial(run_optimization, U=U, error_func=self.error_func, solver=self.solver, I=I, heuristic=self.heuristic), process_queue):
                logstandard("Processed", old_h, old_v, current_depth - 1, hash(step), len(process_queue))
                # generate the successors
                successors = [step.appending(layer) for layer in search_layers]
                # add the latest results
                results[hash(step)] = (h, current_depth, value, tiebreaker, vector, successors)
                tiebreaker += 1
                # only add successors to the process queue if the depth limit has not been exceeded and the threshold limit has not been met
                if (depth is None or current_depth < depth) and value > self.threshold:
                    for successor in successors:
                        process_queue.push((h, current_depth, value, tiebreaker, successor))
                        tiebreaker += 1

                # progress the search queue, if possible
                all_evaluated = True
                updated = False
                while all_evaluated:
                    waitingcount = 0
                    for waiter in search_head[1]:
                        if not hash(waiter) in results:
                            all_evaluated = False
                            waitingcount += 1
                            if updated and not hash(waiter) in process_queue._in_progress:
                                process_queue.add_target((search_head[3], search_head[4], search_head[5], tiebreaker, waiter))
                                tiebreaker += 1

                    if all_evaluated:
                        for waiter in search_head[1]:
                            if hash(waiter) == 3273027901:
                                waiter.draw()
                            rw = results[hash(waiter)]
                            search_queue.push((rw[0], rw[1], rw[2], rw[3], rw[4], rw[5], waiter))
                        new_results = search_queue.pop()
                        search_head = (new_results[-1], new_results[-2], new_results[-3], new_results[0], new_results[1], new_results[2])
                        updated = True
                        logstandard("Searched", new_results[0], new_results[2], new_results[1], hash(new_results[-1]), len(process_queue))
                        if new_results[2] < best_value:
                            best_value = new_results[2]
                            best_depth = new_results[1]
                            opt = new_results[-1]._optimize(I)
                            vector = new_results[-3]
                            best_pair = (opt.matrix(vector), opt, vector)
                            logstandard("New best!", self.heuristic(best_value, best_depth), best_value, best_depth, hash(new_results[-1]), len(process_queue))
                            if best_value <= self.threshold:
                                logprint("Solution that passes threshold found!")
                                search_queue = []
                                break
                    else:
                        cr = results[hash(search_head[0])]
                        logprint("Waiting for {}/{}\tH: {}".format(waitingcount, len(search_head[1]), np.around(cr[0], 5)))
                if best_value <= self.threshold:
                    process_queue = []
                    break
                process_queue.finish(hash(step))
                checkpoint.save((search_head, results.copy(), search_queue.copy(), process_queue.copy(), best_depth, best_value, best_pair, tiebreaker, process_queue._in_progress.copy()), statefile)

        pool.close()
        pool.terminate()
        pool.join()
        logprint("Finished compilation at depth {} with score {} and heuristic {} after {} seconds.".format(best_depth, best_value, self.heuristic(best_value, best_depth), timer() - start))
        return best_pair

