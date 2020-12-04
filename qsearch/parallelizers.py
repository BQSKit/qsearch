"""
This module defines Parallelizer, which is a class that defines how to perform multiple circuits in parallel.

Several implementations are provided.

Attributes:
    LokyParallelizer : A Parallelizer based on Loky, a "deadlock-free" ProcessPoolExecutor
    MultiprocessingParallelizer : A Parallelizer based on multiprocessing
    ProcessPoolParallelizer : A Parallelizer based on concurrent.futures.ProcessPoolExecutor
    MPIParallelizer : A distributed MPI based Parallelizer
    SequentialParallelizer : Mostly for debugging purposes, a Parallelizer that runs tasks one at a time.
"""

from multiprocessing import get_context, cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import signal
import sys

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    from loky import get_reusable_executor
except ImportError:
    get_reusable_executor = None


def default_num_tasks(options):
    return cpu_count()

def evaluate_step(tup, options):
    step, depth, weight = tup
    return (step, options.solver.solve_for_unitary(options.backend.prepare_circuit(step, options), options), depth, weight)

def single_task(opts):
    return 1

def process_initializer():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class Parallelizer():
    """Base class for all Parallelizers. Parallelizers calculate the value of multiple search nodes in parallel."""
    def solve_circuits_parallel(self, tuples):
        """Calculate the value of search tree nodes in parallel."""
        return None

    def done(self):
        """Finalize/Clean up any state needed to run the Parallelizer."""
        pass

class LokyParallelizer(Parallelizer):
    """A parallelizer based on Loky, a "deadlock-free" ProcessPoolExecutor.

    For more information on Loky see https://loky.readthedocs.io/en/stable/.
    """
    def __init__(self, options):
        options.set_smart_defaults(num_tasks=default_num_tasks)
        self.executor = get_reusable_executor(max_workers=options.num_tasks)
        self.process_func = partial(evaluate_step, options=options)

    def solve_circuits_parallel(self, tuples):
        return self.executor.map(self.process_func, tuples)

class MultiprocessingParallelizer(Parallelizer):
    """A Parallelizer based on muliprocessing. Note this cannot be used with the MultiStart_Solvers!"""
    def __init__(self, options):
        if sys.platform != 'win32':
            ctx = get_context('fork')
        else:
            ctx = get_context()
        options.set_smart_defaults(num_tasks=default_num_tasks)
        self.pool = ctx.Pool(options.num_tasks, initializer=process_initializer)
        self.process_func = partial(evaluate_step, options=options)

    def solve_circuits_parallel(self, tuples):
        yield from self.pool.imap_unordered(self.process_func, tuples)

    def done(self):
        self.pool.close()
        self.pool.terminate()
        self.pool.join()

class ProcessPoolParallelizer(Parallelizer):
    """A Parallelizer based on concurrent.futures.ProcessPoolExecutor."""
    def __init__(self, options):
        options.set_smart_defaults(num_tasks=default_num_tasks)
        if sys.version_info >= (3, 8, 0) and sys.platform != 'win32':
            ctx = get_context('fork')
            self.pool = ProcessPoolExecutor(options.num_tasks, mp_context=ctx, initializer=process_initializer)
        else:
            self.pool = ProcessPoolExecutor(options.num_tasks, initializer=process_initializer)

        self.process_func = partial(evaluate_step, options=options)

    def solve_circuits_parallel(self, tuples):
        return self.pool.map(self.process_func, tuples)

    def done(self):
        self.pool.shutdown()

class MPIParallelizer(Parallelizer):
    """A distributed MPI based Parallelizer.

    This implementation unfortunately requires some work on the part of the Project
    API or the user.
    """
    def __init__(self, options):
        options.set_smart_defaults(num_tasks=self.comm.size)
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

class SequentialParallelizer(Parallelizer):
    """A Paralleizer that isn't, it runs tasks one at a time (mostly for debugging).
    """
    def __init__(self, options):
        options.set_smart_defaults(num_tasks=single_task)
        self.process_func = partial(evaluate_step, options=options)

    def solve_circuits_parallel(self, tuples):
        return map(self.process_func, tuples)
