from multiprocessing import get_context, cpu_count
from concurrent.futures import ProcessPoolExecutor as Pool
from functools import partial
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def default_num_tasks(options):
    return cpu_count()

def evaluate_step(tup, options):
    step, depth, weight = tup
    return (step, options.solver.solve_for_unitary(options.backend.prepare_circuit(step, options), options), depth, weight)

class Parallelizer():
    def solve_circuits_parallel(self, tuples):
        return None

    def done(self):
        pass


class MultiprocessingParallelizer(Parallelizer):
    def __init__(self, options):
        ctx = get_context("fork")
        options.set_smart_defaults(num_tasks=default_num_tasks)

        self.pool = ctx.Pool(options.num_tasks)
        self.process_func = partial(evaluate_step, options=options)

    def solve_circuits_parallel(self, tuples):
        yield from self.pool.imap_unordered(self.process_func, tuples)

class ProcessPoolParallelizer(Parallelizer):
    def __init__(self, options):
        options.set_smart_defaults(num_tasks=default_num_tasks)

        self.pool = Pool(options.num_tasks)
        self.process_func = partial(evaluate_step, options=options)

    def solve_circuits_parallel(self, tuples):
        return self.pool.map(self.process_func, tuples)

class MPIParallelizer(Parallelizer):
    def __init__(self, options):
        if MPI is not None:
            self.comm = MPI.COMM_WORLD
            self.comm.bcast(False, root=0)
        else:
            raise RuntimeError("mpi4py is not installed")
        # HACK: We want to play nice with the multistart parallelizer
        #TODO this should probably be managed via the passed options going forward
        if hasattr(solver, 'num_threads'):
            self.tasks = self.comm.size - solver.num_threads
        else:
            self.tasks = self.comm.size
        eval = partial(evaluate_step, options)
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
