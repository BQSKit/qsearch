from search_compiler.gatesets import QubitCNOTLinear
from search_compiler.circuits import *
from search_compiler.solver import CMA_Solver
import numpy as np


qubits = 5
qct = QubitCNOTLinear()
solv = CMA_Solver()
search = ProductStep(*qct.search_layers(qubits,1))
res = solv.solve_for_unitary(search, np.eye(4))
assert res[0].shape[0] == 4
