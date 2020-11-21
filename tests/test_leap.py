
import qsearch
import numpy as np
from qsearch import unitaries, advanced_unitaries, utils, gatesets, leap_compiler, post_processing, solvers, parallelizers

import pytest
import sys


def test_leap_simple(project, check_project):
    project.add_compilation("qft3", unitaries.qft(8))
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
    project.post_process(post_processing.LEAPReoptimizing_PostProcessor(), reoptimize_size=5)
    check_project(project)

def test_reoptimize_short(project, check_project):
    target = unitaries.qft(8)
    project.add_compilation("qft3", target)
    project["min_depth"] = 6
    project["compiler_class"] = leap_compiler.LeapCompiler
    project.run()
    # delete some structure to check it doesn't crash if shorter than reoptimize size
    project._compilations['qft3']['structure']._subgates = project._compilations['qft3']['structure']._subgates[:3]
    project.post_process(post_processing.LEAPReoptimizing_PostProcessor(), reoptimize_size=7, timeout=30)
    # we don't check the project as it won't work of course, we just wanted to regression test against crashing in the re-optimizer
