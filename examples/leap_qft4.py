import qsearch
from qsearch import unitaries, advanced_unitaries, leap_compiler, multistart_solver, parallelizer, reoptimizing_compiler
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

with qsearch.Project("leapex-reoptimize") as reoptimize:
    for comp in project.compilations:
        target = project.get_target(comp)
        cdict = project.get_result(comp)
        best_pair = (cdict["structure"], cdict["vector"])
        # add LEAP compilations to reoptimize
        reoptimize.add_compilation(comp, target, cut_depths=cdict["cut_depths"], best_pair=best_pair, depth=7)
    # set to use the reoptimizing compiler
    reoptimize["compiler_class"] = reoptimizing_compiler.ReoptimizingCompiler
    reoptimize.run()
