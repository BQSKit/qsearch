from search_compiler.sample_gates import qft
from search_compiler.solver import CMA_Solver
from search_compiler.gatesets import QubitCNOTLinear
from search_compiler.circuits import ProductStep
import numpy as np


qubits = 3
gateset = QubitCNOTLinear()
solv = CMA_Solver()
search = ProductStep(gateset.initial_layer(qubits, 2), *gateset.search_layers(qubits, 2))
target = qft(2**qubits)
sol = solv.solve_for_unitary(search, target)
print(sol[0])
