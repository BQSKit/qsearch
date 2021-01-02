import numpy as np
from qsrs import native_from_object as native
import qsearch as qs
from qsearch.gates import *

import pytest

def check_circuit(circ):
    nc = native(circ)
    v = np.random.rand(circ.num_inputs)
    nmat = nc.matrix(v)
    mat = circ.matrix(v)
    assert np.allclose(mat, nmat), print(mat) or print(nmat)
    assert 1e-10 > qs.utils.matrix_distance_squared(mat, nmat), print(mat) or print(nmat)
    nmat, njs = nc.mat_jac(v)
    mat, js = circ.mat_jac(v)
    assert np.allclose(mat, nmat), print(mat) or print(nmat)
    for (i, (j, nj)) in enumerate(zip(js, njs)):
        assert np.allclose(j, nj), print(f'failed on {i}') or print(j) or print(nj)


u = U3Gate()
xzxz = XZXZGate()
RUST_GATES = (
    XGate(),
    YGate(),
    ZGate(),
    XZXZGate(),
    U3Gate(),
    U2Gate(),
    IdentityGate(1),
    CNOTGate(),
    CPIGate(),
    ProductGate(u, xzxz, u),
    KroneckerGate(u, xzxz, u),
    SingleQutritGate(),
)

@pytest.mark.parametrize("gate", RUST_GATES, ids=lambda gate: repr(gate))
def test_rust_circuits(gate):
    check_circuit(gate)

def test_CNOT_Linear():
    g = qs.gatesets.QubitCNOTLinear()
    check_circuit(g.initial_layer(4))
    for layer in g.search_layers(4):
        check_circuit(layer[0])

def test_CNOT_Ring():
    g = qs.gatesets.QubitCNOTRing()
    check_circuit(g.initial_layer(4))
    for layer in g.search_layers(4):
        check_circuit(layer[0])

def test_native_from_object_double():
    n = native(U3Gate())
    check_circuit(n)
