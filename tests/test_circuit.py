import numpy as np
from qsrs import native_from_object as native
import qsearch as qs

def check_circuit(c):
    nc = native(c)
    v = np.random.rand(c.num_inputs)
    nmat = nc.matrix(v)
    mat = c.matrix(v)
    assert np.allclose(mat, nmat), print(mat) or print(nmat)
    assert 1e-13 > qs.utils.matrix_distance_squared(mat, nmat), print(mat) or print(nmat)
    nmat, njs = nc.mat_jac(v)
    mat, js = c.mat_jac(v)
    assert np.allclose(mat, nmat), print(mat) or print(nmat)
    for (i, (j, nj)) in enumerate(zip(js, njs)):
        assert np.allclose(j, nj), print(f'failed on {i}') or print(j) or print(nj)

def test_XZXZ():
    check_circuit(qs.circuits.XZXZPartialQubitStep())

def test_U3():
    check_circuit(qs.circuits.QiskitU3QubitStep())

def test_Id():
    check_circuit(qs.circuits.IdentityStep(1))

def test_CNOT():
    check_circuit(qs.circuits.CNOTStep())

def test_Product():
    h = qs.circuits.QiskitU3QubitStep()
    g = qs.circuits.XZXZPartialQubitStep()
    check_circuit(qs.circuits.ProductStep(g,h,g))

def test_Kronecker():
    h = qs.circuits.QiskitU3QubitStep()
    g = qs.circuits.XZXZPartialQubitStep()
    check_circuit(qs.circuits.KroneckerStep(g,h, g))

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
    n = native(qs.circuits.QiskitU3QubitStep())
    check_circuit(n)

def test_constant_unitary():
    check_circuit(qs.circuits.CPIStep())

def test_singlequtrit():
    check_circuit(qs.circuits.SingleQutritStep())

if __name__ == '__main__':
    test_XZXZ()
    test_U3()
    test_Id()
    test_CNOT()
    test_Product()
    test_Kronecker()
    test_CNOT_Linear()
    test_CNOT_Ring()
    test_native_from_object_double()
    test_constant_unitary()
