import qsearch
from qsearch import leap_compiler, post_processing, multistart_solvers, parallelizers
import numpy as np
from functools import partial


# create the project
p = qsearch.Project("stateprep-example")

# configure the project with the stateprep defaults instead of the standard synthesis defaults
p.configure(**qsearch.defaults.stateprep_defaults)
p["compiler_class"] = leap_compiler.LeapCompiler
p["solver"] = qsearch.solvers.LeastSquares_Jac_Solver()

# add states that are converted using generate_stateprep_target_matrix
p.add_compilation("basic_state_test", np.eye(32, dtype='complex128'), initial_state = np.array([1] + [0]*31, dtype='complex128'), target_state=[0.5,0,0,0.5j,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5j,0,0,0,0,0,0,0,-0.5,0,0,0,0,0])

p.run()

# run post-processing to improve circuits that were generated with LEAP
p.post_process(post_processing.LEAPReoptimizing_PostProcessor(), solver=multistart_solvers.MultiStart_Solver(16), parallelizer=parallelizers.ProcessPoolParallelizer, reoptimize_size=7)
