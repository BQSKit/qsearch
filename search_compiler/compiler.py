from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import chain
from timeit import default_timer as timer
import heapq

from .circuits import *

from . import gatesets as gatesets
from .solver import default_solver
from .logging import logprint
from . import checkpoint, utils, heuristics, circuits

class Compiler():
    def compile(self, U, depth):
        raise NotImplementedError("Subclasses of Compiler are expected to implement the compile method.")
        return (U, None, None)

def evaluate_step(tup, U, error_func, solver, I):
    step, depth = tup
    ostep = self.optimize_circuit(step, I)
    #ostep = step._optimize(I)
    return (step, solver.solve_for_unitary(ostep, U, error_func), depth)

class SearchCompiler(Compiler):
    def __init__(self, threshold=1e-10, error_func=utils.matrix_distance_squared, heuristic=heuristics.astar, gateset=gatesets.Default(), solver=default_solver(), beams=1):
        self.threshold = threshold
        self.error_func = error_func
        self.heuristic = heuristic
        self.gateset = gateset
        self.solver = solver
        self.beams = int(beams)

    def compile(self, U, depth=None, statefile=None):
        h = self.heuristic
        dits = int(np.round(np.log(np.shape(U)[0])/np.log(self.gateset.d)))

        if self.gateset.d**dits != np.shape(U)[0]:
            raise ValueError("The target matrix of size {} is not compatible with qudits of size {}.".format(np.shape(U)[0], self.d))

        I = circuits.IdentityStep(self.gateset.d)

        initial_layer = self.gateset.initial_layer(dits)
        search_layers = self.gateset.search_layers(dits)

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
            result = self.solver.solve_for_unitary(root, U, self.error_func)
            best_value = self.error_func(U, result[0])
            best_pair = (result[0], root._optimize(I), result[1])
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
                logprint("Popped a node with score: {} at depth: {}".format((tup[2]), tup[1]))

            then = timer()
            new_steps = [(current_tup[5].appending(search_layer), current_tup[1]) for search_layer in search_layers for current_tup in popped]

            for step, result, current_depth in pool.imap_unordered(partial(evaluate_step, U=U, error_func=self.error_func, solver=self.solver, I=I), new_steps):
                current_value = self.error_func(U, result[0])
                if (current_value < best_value and (best_value >= self.threshold or current_depth + 1 <= best_depth)) or (current_value < self.threshold and current_depth + 1 < best_depth):
                    best_value = current_value
                    best_pair = (result[0], step._optimize(I), result[1])
                    best_depth = current_depth + 1
                    logprint("New best! score: {} at depth: {}".format(best_value, current_depth + 1))
                if depth is None or current_depth + 1 < depth:
                    heapq.heappush(queue, (h(current_value, current_depth+1), current_depth+1, current_value, tiebreaker, result[1], step))
                    tiebreaker+=1
            logprint("Layer completed after {} seconds".format(timer() - then))
            checkpoint.save((queue, best_depth, best_value, best_pair, tiebreaker), statefile)


        pool.close()
        pool.terminate()
        pool.join()
        logprint("Finished compilation at depth {} with score {}.".format(best_depth, best_value/10))
        return best_pair


class OptimizeBlock():
    # a class used to represent blocks being moved around for reoptimization
    def __init__(self, circuit, index):
        self.circuit = circuit
        self.index = index
        self.dits = circuit.dits

    def generate(circuit, index=0):
        # split the circuit up into digestible chunks based on the three key QuantumStep types: Kronecker, Product, and Identity
        if isinstance(circuit, IdentityStep):
            return []
        elif isinstance(circuit, ProductStep):
            return list(chain.from_iterable((OptimizeBlock.generate(x, index) for x in circuit._substeps)))
        elif isinstance(circuit, KroneckerStep):
            output = []
            index = 0
            for substep in circuit._substeps:
                result = OptimizeBlock.generate(substep, index)
                if len(result) > 0:
                    index += result[0].dits
                    output += result
                else:
                    index += substep.dits # a placeholder IdentityStep was detected
        else:
            return [OptimizeBlock(circuit, index)]

class OptimizeBin():
    class SubBin():
        # Initial Setup Functions
        def __init__(self, index, I):
            self.prev_bin = None
            self.next_bin = None
            self.contents = []
            self.linked_prev = False
            self.linked_next = False
            self.index = index
            self.I = I

        def set_next(self, nex):
            self.next_bin = nex
            nex.prev_bin = self

        def set_prev(self, prev):
            self.prev_bin = prev
            prev.next_bin = self

        # Linked Collection Management Functions
        def linked_first(self):
            if not self.linked_prev:
                return self

        def linked_last(self):
            if not self.linked_next:
                return self

        def linked_length(self):
            node = self.linked_first()
            mlength = len(node.contents)
            while node.linked_next:
                node = node.next_bin
                mlength = max(mlength, len(node.contents))

        # Block Management Functions
        def add_block(self, block, position=0):
            self.contents.append(block)
            if position+1 < block.dits:
                self.next_bin.add_block(block, position+1)
                self.linked_next = True
                self.next_bin.linked_prev = True

        def potential_width(self, block):
            return max(self.linked_length, block.dits + block.index - self.linked_first.index)
        
        def long_bin_ready(self):
            # get to the first bin in the length
            if self.index > self.contents[0].index:
                if self.contents[0] is self.prev_bin.contents[0]:
                    return self.prev_bin.long_bin_ready()
                else:
                    return False
            # check the next few bins to see if they are all ready
            node = self
            target = self.contents[0]
            for _ in range(0, self.contents[0].dits):
                if not self.contents[0] is node.contents[0]:
                    return False
                node = node.next_bin
            return True

        def flush(self):
            # Recurse so we can assume that the main code is only executed on the first of a set of linked bins
            if self.linked_prev:
                return self.prev_bin.flush()
            steps = []
            while self.linked_length() > 0:
                node = self
                kronsteps = []
                while True:
                    if len(node.contents) < 1:
                        kronsteps.append(self.I) # placeholder identity due to finished list
                    elif node.contents[0].dits == 1:
                        kronsteps.append(node.contents.pop(0).circuit)
                    else:
                        # its a long bin
                        if not self.long_bin_ready():
                            kronsteps.append(self.I) # its not ready yet so apply placeholder identity
                        elif self.index == self.contents[0].index:
                            kronsteps.append(node.contents.pop(0).circuit)
                        else:
                            node.contents.pop(0) # the node was already added by a previous bin, so we just need to pop it
                    if node.linked_next:
                        node = node.next_bin
                    else:
                        break
                if len(kronsteps) == 1:
                    steps.append(kronsteps[0])
                elif len(kronsteps) > 1:
                    steps.append(KroneckerStep(*kronsteps))
            # unlink the bins before returning
            node = self
            while node.linked_next:
                node.linked_prev = False
                node.linked_next = False
                node = node.next_bin

            if len(steps) == 1:
                return OptimizeBin(steps[0], self.index)
            else:
                return OptimizeBin(ProductStep(*steps), self.index)

    # a class to hold and manage bins used for reoptimization
    def __init__(self, size):
        # create a list of sub-bins and link them
        self.bins = [SubBin() for _ in range(0, size)]
        self.bins[0].index = 0
        for i in range(1, size):
            self.bins[i].index = i
            self.bins[i].set_prev(self.bins[i-1])

    def check_width(self, block, maxwidth):
        target_bin = self.bins[block.index]
        return target_bin.potential_width(block) <= maxWidth

    def add_block(self, block):
        target_bin = self.bins[block.index]
        target_bin.add_block(block)
    
    def flush(self, index):
        return self.bins[index].flush()
 
def optimize_circuit(circuit, I):
    blocks = OptimizeBlock.generate(circuit)
    bins = OptimizeBin(circuit.dits)
    for opt_pass in range(1, circuit.dits):
        nextBlocks = []
        for block in blocks:
            if not bins.check_width(block, opt_pass):
                nextBlocks.append(bins.flush(block.index))
            bins.add_block(block)
        blocks = nextBlocks
    if len(blocks) == 1:
        return blocks[0].circuit
    else:
        return ProductStep(*[block.circuit for block in blocks])

