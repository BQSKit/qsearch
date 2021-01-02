import qsearch
import numpy as np
from qsearch.gates import *

import pytest

u = U3Gate()
xzxz = XZXZGate()
cnot = CNOTGate()
id = IdentityGate(1)
RUST_GATES = (
    XGate(),
    YGate(),
    ZGate(),
    XZXZGate(),
    ZXZXZGate(),
    U3Gate(),
    U2Gate(),
    IdentityGate(1),
    CNOTGate(),
    ProductGate(u, xzxz, u),
    KroneckerGate(u, u),
    KroneckerGate(u, cnot, xzxz),
    SingleQutritGate(),
    ProductGate(KroneckerGate(U3Gate(), U3Gate(), U3Gate()), KroneckerGate(ProductGate(CNOTGate(), KroneckerGate(XZXZGate(), U3Gate())), IdentityGate())),
)

@pytest.mark.parametrize("gate", RUST_GATES, ids=lambda gate: repr(gate))
def test_gradients(gate):
    totaldiff = [0] * gate.num_inputs
    eps = 1e-5
    repeats = 100
    for _ in range(repeats):
        v = np.random.rand(gate.num_inputs) * 2 * np.pi
        M, Js = gate.mat_jac(v)
        for i in range(gate.num_inputs):
            v2 = np.copy(v)
            v2[i] = v[i] + eps
            U1 = gate.matrix(v2)
            v2[i] = v[i] - eps
            U2 = gate.matrix(v2)
            
            FD = (U1 - U2) / (2*eps)

            diffs = np.sum(np.abs(FD - Js[i]))
            totaldiff[i] += diffs

    for i in range(gate.num_inputs):
        assert totaldiff[i] < eps
