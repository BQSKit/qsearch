import qsearch
import numpy as np
from qsearch.gates import *

def compare_gradient(gate):
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
        assert totaldiff[i] < 1e-5


def test_gradients_U3():
    compare_gradient(U3Gate())

def test_gradients_X():
    compare_gradient(XGate())

def test_gradients_Y():
    compare_gradient(YGate())

def test_gradients_Z():
    compare_gradient(ZGate())

def test_gradients_ZXZXZ():
    compare_gradient(ZXZXZGate())

def test_gradients_XZXZ():
    compare_gradient(XZXZGate())

def test_gradients_SingleQutrit():
    compare_gradient(SingleQutritGate())

def test_gradients_Kronecker_Simple():
    compare_gradient(KroneckerGate(U3Gate(), U3Gate()))

def test_gradients_Kronecker_LessSimple():
    compare_gradient(KroneckerGate(U3Gate(), CNOTGate(), XZXZGate()))

def test_gradients_Product_Simple():
    compare_gradient(ProductGate(U3Gate(), U3Gate()))

def test_gradients_Product_LessSimple():
    compare_gradient(ProductGate(KroneckerGate(U3Gate(), U3Gate(), U3Gate()), KroneckerGate(ProductGate(CNOTGate(), KroneckerGate(XZXZGate(), U3Gate())), IdentityGate())))

