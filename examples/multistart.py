import qsearch
from qsearch import unitaries, advanced_unitaries, leap_compiler, solvers, multistart_solvers, parallelizers
import time

# create the project
with qsearch.Project("multistart") as project:
    # add some example circuits
    project.add_compilation("qft3", unitaries.qft(8))
    project.add_compilation("fredkin", unitaries.fredkin)
    project.add_compilation("toffoli", unitaries.toffoli)
    project.add_compilation("peres", unitaries.peres)
    # set the solver to MultiStart, passing 2 threads of parallelism
    project["solver"] = multistart_solvers.MultiStart_Solver(8)
    #project["inner_solver"] = solver.BFGS_Jac_Solver()  # optionally change the inner solver
    # Multistart requires nested processes, so we use ProcessPoolExecutor
    project["parallelizer"] = parallelizers.ProcessPoolParallelizer
    project.run()