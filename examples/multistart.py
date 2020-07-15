import qsearch
from qsearch import unitaries, advanced_unitaries, leap_compiler, solver, multistart_solver, parallelizer
import time

# create the project
with qsearch.Project("multistart") as project:
    # add some example circuits
    project.add_compilation("qft3", unitaries.qft(8))
    project.add_compilation("fredkin", unitaries.fredkin)
    project.add_compilation("toffoli", unitaries.toffoli)
    project.add_compilation("peres", unitaries.peres)
    # set the solver to MultiStart, passing 2 threads of parallelism
    project["solver"] = multistart_solver.MultiStart_Solver(2)
    #project["inner_solver"] = solver.BFGS_Jac_Solver()  # optionally change the inner solver
    # Multistart requires nested processes, so we use ProcessPoolExecutor
    project["parallelizer"] = parallelizer.ProcessPoolParallelizer
    project.run()