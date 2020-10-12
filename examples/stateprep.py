import qsearch
from qsearch import leap_compiler
import numpy as np
from functools import partial


# create the project
p = qsearch.Project("stateprep-example")

# configure the project with the stateprep defaults instead of the standard synthesis defaults
p.configure(**qsearch.defaults.stateprep_defaults)
p["compiler_class"] = leap_compiler.LeapCompiler
p["solver"] = qsearch.solvers.LeastSquares_Jac_Solver()

# add states that are converted using generate_stateprep_target_matrix
p.add_compilation("basic_state_test", qsearch.utils.generate_stateprep_target_matrix([0.5,0,0,0.5j,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5j,0,0,0,0,0,0,0,-0.5,0,0,0,0,0]))

p.run()

