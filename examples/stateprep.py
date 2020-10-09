import qsearch
import numpy as np
from functools import partial


# create the project
p = qsearch.Project("stateprep-example")

# configure the project with the stateprep defaults instead of the standard synthesis defaults
p.configure(**qsearch.defaults.stateprep_defaults)
p["solver"] = qsearch.solver.LeastSquares_Jac_Solver()

# add states that are converted using generate_stateprep_target_matrix
p.add_compilation("basic_state_test", qsearch.utils.generate_stateprep_target_matrix([0.5,0,0,0.5j,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5j,0,0,0,0,0,0,0,-0.5,0,0,0,0,0]))

p.run()

