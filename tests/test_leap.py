
import qsearch
import numpy as np
from qsearch import unitaries, advanced_unitaries, utils, gatesets, leap_compiler, reoptimizing_compiler

def test_leap(project, check_project):
    project.add_compilation("qft4", unitaries.qft(16))
    project["min_depth"] = 6
    project["compiler_class"] = leap_compiler.LeapCompiler
    project.run()
    check_project(project)


def test_reoptimize(project, check_project):
    target = unitaries.qft(16)
    project.add_compilation("qft4", target)
    project["min_depth"] = 6
    project["compiler_class"] = leap_compiler.LeapCompiler
    project.run()
    cdict = project.get_result('qft4')
    best_pair = (cdict["structure"], cdict["vector"])
    cut_depths=cdict["cut_depths"]
    project.clear()
    project.add_compilation('qft4', target, cut_depths=cut_depths, best_pair=best_pair, depth=7)
    project.run()
    check_project(project)