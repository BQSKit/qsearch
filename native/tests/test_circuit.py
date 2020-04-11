import numpy as np
from search_compiler_rs import native_from_object as native, matrix_distance_squared
import search_compiler as sc

def test_circuit(c):
    nc = native(c)
    v = np.random.rand(c.num_inputs)
    nmat = nc.matrix(v)
    mat = c.matrix(v)
    assert np.allclose(mat, nmat), print(mat) or print(nmat)
    assert 1e-20 > matrix_distance_squared(mat, nmat) - sc.utils.matrix_distance_squared(mat, nmat)
    nmat, njs = nc.mat_jac(v)
    mat, js = c.mat_jac(v)
    assert np.allclose(mat, nmat), print(mat) or print(nmat)
    for (i, (j, nj)) in enumerate(zip(js, njs)):
        assert np.allclose(j, nj), print(f'failed on {i}') or print(j) or print(nj)

print('Testing XZXZ, U3, Identity, CNOT, Product, and Kronecker')
g = sc.circuits.XZXZPartialQubitStep()
test_circuit(g)
h = sc.circuits.QiskitU3QubitStep()
test_circuit(sc.circuits.IdentityStep(1))
test_circuit(sc.circuits.CNOTStep())
test_circuit(sc.circuits.QiskitU3QubitStep())
test_circuit(sc.circuits.ProductStep(g,h,g))
test_circuit(sc.circuits.KroneckerStep(g,h, g))
print('All passed!')