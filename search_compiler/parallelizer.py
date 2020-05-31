from multiprocessing import get_context
from functools import partial

def evaluate_step(tup, U, error_func, error_jac, solver, backend):
    step, depth, weight = tup
    return (step, solver.solve_for_unitary(backend.prepare_circuit(step), U, error_func, error_jac), depth, weight)

class Parallelizer():
    def solve_circuits_parallel(self, tuples, ):
        return None


class MultiprocessingParallelizer(Parallelizer):
    def __init__(self):
        ctx = get_context("fork")
        self.pool = ctx.Pool()
        self.backend = None

    def solve_circuits_parallel(self, solver, tuples, U, error_func, error_jac):
        process_func = partial(evaluate_step, U=U, error_func=error_func, error_jac=error_jac, solver=solver, backend=self.backend)
        return self.pool.imap_unordered(process_func, tuples)

