import qsearch
from qsearch import leap_compiler
import numpy as np
from functools import partial


# create the project
p = qsearch.Project("stateprep-example")

# configure the project with the stateprep defaults instead of the standard synthesis defaults

stateprep_options = qsearch.Options(smart_defaults=qsearch.defaults.stateprep_smart_defaults)


# add states that are converted using generate_stateprep_target_matrix

# It may seem strange that we have to pass an identity.  Qsearch is still doing unitary synthesis;
# stateprep is achieved by modifying the way we compare unitaries such that they are compared by
# the state produced by acting on an initial state.
# In this case, we are synthesizing a circuit designed to act on the |000> state to match the state
# resulting by performing the Identity on the desired state.
# Note that you can specify the intiial state for the circuit as initial_state (the default state is
# the zero state that matches target_state in number of qudits).
toffoli_magic_state = np.array([0.5,0,0.5,0,0.5,0,0,0.5], dtype='complex128')
p.add_compilation("toffoli_magic_state", np.eye(8,dtype='complex128'), target_state=toffoli_magic_state, options=stateprep_options)

p.run()

