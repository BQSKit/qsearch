
import qsearch
import numpy as np
from qsearch import unitaries, advanced_unitaries, utils, gatesets, leap_compiler, reoptimizing_compiler, multistart_solvers, parallelizers

def test_leap(project, check_project):
    project.add_compilation("qft4", unitaries.qft(16))
    project["min_depth"] = 6
    project["compiler_class"] = leap_compiler.LeapCompiler
    project.run()
    check_project(project)


def test_reoptimize(project, check_project):
    target = unitaries.qft(16)
    project.add_compilation("qft4", target)
    project["min_depth"] = 4
    project["compiler_class"] = leap_compiler.LeapCompiler
    project.run()
    project.post_process(reoptimizing_compiler.ReoptimizingCompiler(), solver=multistart_solvers.MultiStart_Solver(8), parallelizer=parallelizers.ProcessPoolParallelizer, depth=5)
    check_project(project)
