from multiprocessing import Pool, cpu_count
from functools import partial
import heapq

from CMA_Solver import *
import CMA_Utils as util
from CMA_Logging import logprint

class Compiler():
    def compile(self, U, depth):
        return (U, None)

def generate_double_steps(double_step, n, d):
    identity_step = IdentityStep(d)
    return [KroneckerStep(*[identity_step]*i, double_step, *[identity_step]*(n-i-2)) for i in range(0,n-1)]

def evaluate_step(step, U, error_func, error_target):
    return (step, step.solve_for_unitary(U, error_func, error_target))

class CMA_Search_Compiler(Compiler):
    def __init__(self, threshold=0.01, d=2, error_func=util.matrix_distance_squared):
        self.threshold = threshold
        self.error_func = error_func
        self.d = d

    def compile(self, U, depth):
        n = np.log(np.shape(U)[0])/np.log(self.d)

        if self.d**n != np.shape(U)[0]:
            raise ValueError("The target matrix of size {} is not compatible with qudits of size {}.".format(np.shape(U)[0], self.d))
        n = int(n)

        if self.d == 2:
            single_step = KroneckerStep(*[SingleQubitStep()]*n)
            double_steps = generate_double_steps(CQubitStep(), n, self.d)
        elif self.d == 3:
            single_step = KroneckerStep(*[SingleQutritStep()]*n)
            double_steps = generate_double_steps(CPIPhaseStep(), n, self.d)
        else:
            raise NotImplementedError("Qu-{}-its haven't been implemented yet.".format(self.d))

        pool = Pool(min(len(double_steps),cpu_count()))
        logprint("Creating a pool of {} workers".format(pool._processes))

        root = ProductStep(single_step)
        result = root.solve_for_unitary(U, self.error_func)
        best_value = self.error_func(U, result[0])
        best_pair = (result[0], root.path(result[1]))
        logprint("New best! {} at depth 0".format(best_value))
        if depth == 0:
            return best_pair

        queue = [(best_value, 0, 0, root)]

        while len(queue) > 0:
            _, current_depth, _, current_step = heapq.heappop(queue)
            logprint("popped a node of depth {}".format(current_depth))
            new_steps = [current_step.appending(double_step, single_step) for double_step in double_steps]

            tiebreaker=0
            for step, result in pool.imap_unordered(partial(evaluate_step, U=U, error_func=self.error_func, error_target=self.threshold), new_steps):
                current_value = self.error_func(U, result[0])
                if current_value < best_value:
                    best_value = current_value
                    best_pair = (result[0], step.path(result[1]))
                    logprint("New best! {} at depth {}".format(best_value, current_depth + 1))
                    if best_value < self.threshold:
                        pool.close()
                        pool.terminate()
                        queue = []
                        break
                if current_depth + 1 < depth:
                    heapq.heappush(queue, (current_value, current_depth+1, tiebreaker, step))
                    tiebreaker+=1

        pool.close()
        pool.terminate()
        pool.join()
        return best_pair


