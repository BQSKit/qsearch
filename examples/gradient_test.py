import qsearch
import numpy as np
from qsearch.gates import *

gate = ProductGate(KroneckerGate(U3Gate(), U3Gate(), U3Gate()), KroneckerGate(IdentityGate(), ProductGate(CNOTGate(), KroneckerGate(XZXZGate(), U3Gate()))), KroneckerGate(ProductGate(CNOTGate(), KroneckerGate(XZXZGate(), U3Gate())), IdentityGate()), KroneckerGate(IdentityGate(), ProductGate(CNOTGate(), KroneckerGate(XZXZGate(), U3Gate()))))
totaldiff = [0] * gate.num_inputs
eps = 5e-5


def func(v):
    return qsearch.utils.distance_with_initial_state(np.array([1,0,0,0,0,0,0,0]),np.array([0.5,0,0.5,0,0.5,0,0,0.5]), np.eye(8),gate.matrix(v))

def jac_func(v):
    return qsearch.utils.distance_with_initial_state_jac(np.array([1,0,0,0,0,0,0,0]),np.array([0.5,0,0.5,0,0.5,0,0,0.5]), np.eye(8),*gate.mat_jac(v))[1]

for _ in range(100):
    v = np.random.rand(gate.num_inputs)
    M = func(v)
    Js = jac_func(v)
    for i in range(gate.num_inputs):
        v2 = np.copy(v)
        v2[i] = v[i] + eps
        U1 = func(v2)
        v2[i] = v[i] - eps
        U2 = func(v2)
        
        FD = (U1 - U2) / (2*eps)

        diffs = np.abs(np.sum(FD - Js[i]))
        totaldiff[i] += diffs

print(np.array(totaldiff) < 2*eps)
print(totaldiff)
