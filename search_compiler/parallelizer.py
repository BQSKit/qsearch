from multiprocessing import get_context
from functools import partial
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def evaluate_step(tup, U, error_func, error_jac, solver, backend):
    step, depth, weight = tup
    return (step, solver.solve_for_unitary(backend.prepare_circuit(step), U, error_func, error_jac), depth, weight)

class Parallelizer():
    def solve_circuits_parallel(self, tuples, ):
        return None

    def done(self):
        pass


class MultiprocessingParallelizer(Parallelizer):
    def __init__(self, solver, U, error_func, error_jac, backend):
        ctx = get_context("fork")
        self.pool = ctx.Pool()
        self.backend = None
        self.process_func = partial(evaluate_step, U=U, error_func=error_func, error_jac=error_jac, solver=solver, backend=backend)

    def solve_circuits_parallel(self, tuples):
        yield from self.pool.imap_unordered(self.process_func, tuples)

class MPIParallelizer(Parallelizer):
    def __init__(self, solver, U, error_func, error_jac, backend):
        if MPI is not None:
            self.comm = MPI.COMM_WORLD
            self.comm.bcast(False, root=0)
        else:
            raise RuntimeError("mpi4py is not installed")
        eval = partial(evaluate_step, U=U, error_func=error_func, error_jac=error_jac, solver=solver, backend=backend)
        eval = self.comm.bcast(eval, root=0)

    def solve_circuits_parallel(self, tuples):
        # NOTE WELL: this should be kept in sync with the mpi_worker code in utils.py
        return self.map_steps(tuples)

    def map_steps(self, new_steps):
        base = 0
        done = False
        while len(new_steps) > 0:
            self.comm.bcast(False, root=0)
            for i in range(self.comm.size - 1):
                if i >= len(new_steps):
                    sendobj = None
                else:
                    sendobj = new_steps[i]
                self.comm.send(sendobj, dest=i+1, tag=i+1)
                res = self.comm.recv(source=i+1, tag=i+1)
                if res is not None:
                    yield res
            new_steps = new_steps[self.comm.size:]

    def done(self):
        self.comm.bcast(True, root=0)
