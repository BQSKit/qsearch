import qsearch
import numpy as np
from qsearch.gates import *

gate = SingleQutritStep()
totaldiff = [0] * gate.num_inputs
eps = 5e-11

for _ in range(100):
    v = np.random.rand(gate.num_inputs)
    M, Js = gate.mat_jac(v)
    for i in range(gate.num_inputs):
        v2 = np.copy(v)
        v2[i] = v[i] + eps
        U1 = gate.matrix(v2)
        v2[i] = v[i] - eps
        U2 = gate.matrix(v2)
        
        FD = (U1 - U2) / (2*eps)

        diffs = np.abs(np.sum(FD - Js[i]))
        totaldiff[i] += diffs

print(totaldiff)
