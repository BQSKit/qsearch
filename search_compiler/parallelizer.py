from multiprocessing import get_context
from functools import partial

def evaluate_step(tup, U, error_func, error_jac, solver):
    step, depth, weight = tup
    return (step, solver.solve_for_unitary(step, U, error_func, error_jac), depth, weight)

class Parallelizer():
    def solve_circuits_parallel(self, tuples, ):
        return None


class MultiprocessingParallelizer(Parallelizer):
    def __init__(self):
        ctx = get_context("fork")
        self.pool = ctx.Pool()

    def solve_circuits_parallel(self, solver, tuples, U, error_func, error_jac):
        process_func = partial(evaluate_step, U=U, error_func=error_func, error_jac=error_jac, solver=solver)
        return self.pool.imap_unordered(process_func, tuples)

