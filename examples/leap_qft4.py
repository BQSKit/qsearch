import qsearch
from qsearch import unitaries, advanced_unitaries, leap_compiler, multistart_solver, parallelizer
import time

# create the project
with qsearch.Project("leapex") as project:
    # Add 4 qubit qft
    project.add_compilation("qft4", unitaries.qft(16))
    # set a miniumum search depth (to reduce frequent chopping that gets nowhere)
    project["min_depth"] = 6
    # configure qsearch to use the leap compiler
    project["compiler_class"] = leap_compiler.LeapCompiler
    project.run()